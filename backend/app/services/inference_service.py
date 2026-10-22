try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None

from functools import lru_cache
from fastapi import HTTPException
import logging
import pydicom
import numpy as np
from PIL import Image
import tempfile
import os
import uuid
import base64
import httpx
from typing import Dict, Any, List, Optional, Tuple
from ..core.config import get_settings
from ..models.detection import DicomMetadata, ImageInfo
from .preprocessing_service import PreprocessingService, get_preprocessing_service
from .verification_service import VerificationService, get_verification_service

logger = logging.getLogger(__name__)

# Minimum confidence threshold — slightly lower for recall; Gemini is the precision gate
CONFIDENCE_THRESHOLD = 0.25
# IoU threshold for Non-Maximum Suppression — tight suppression to eliminate duplicates
NMS_IOU_THRESHOLD = 0.30
# Containment (Intersection over Minimum) threshold — suppresses large duplicate boxes covering the same lesion
CONTAINMENT_IOM_THRESHOLD = 0.40

# ---------- Anatomical plausibility constants ----------
# Detections whose bounding-box area is smaller than this fraction of the
# full image are almost always sensor dust / noise.
MIN_DETECTION_AREA_FRACTION = 0.0008  # 0.08 %
# Detections larger than this fraction are sensor-edge artefacts
MAX_DETECTION_AREA_FRACTION = 0.45  # 45 %
# Aspect-ratio guard: dental lesions are never extremely elongated strips
MAX_ASPECT_RATIO = 5.0


@lru_cache
def get_roboflow_client(api_key: str):
    """Create and cache Roboflow client"""
    if InferenceHTTPClient is not None:
        return InferenceHTTPClient(
            api_url="https://detect.roboflow.com", api_key=api_key
        )
    return None


class InferenceService:
    """Service for handling dental condition detection inference"""

    def __init__(self):
        self.settings = get_settings()
        self.preprocessor: PreprocessingService = get_preprocessing_service()
        self.verifier: VerificationService = get_verification_service()
        from .multi_scanner_service import get_multi_scanning_service
        self.multi_scanner = get_multi_scanning_service()

    async def detect_dental_conditions(
        self, image_path: str, model_id: str = "adr/6"
    ) -> dict:
        """
        Run multi-stage scanning on dental image:
        Stage 1: Primary YOLO (Diolo) with segment-aware panoramic detection
        Stage 2: Second YOLO focused re-checking & clinical validation
        Stage 3: Segment-by-segment computer vision overlay & radiolucency analysis

        Args:
            image_path: Path to the image file
            model_id: Model ID to use for inference

        Returns:
            dict: Comprehensive Multi-Scanning results with predictions and CV overlay

        Raises:
            HTTPException: If inference fails
        """
        try:
            return await self.multi_scanner.scan_dental_image(image_path, model_id=model_id)
        except Exception as e:
            logger.error(f"Multi-scanning inference failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    def _filter_predictions(self, result: dict) -> dict:
        """
        Post-process inference results with high precision:
        1. Filter out low-confidence detections
        2. Remove outer border sensor/collimator edge artifacts
        3. Anatomical plausibility checks (size, aspect ratio)
        4. Apply Non-Maximum Suppression (NMS) + IoM Containment suppression to eliminate overlapping duplicates
        5. Standardize class labels ('Cavity', 'Periapical Lesion') and class IDs
        """
        predictions = result.get("predictions", [])
        if not predictions:
            return result

        img_data = result.get("image", {})
        img_w = img_data.get("width") or 2000
        img_h = img_data.get("height") or 2000
        img_area = img_w * img_h

        # Step 1: Filter by confidence threshold
        filtered = [p for p in predictions if p.get("confidence", 0) >= CONFIDENCE_THRESHOLD]
        logger.info(
            f"Confidence filter: {len(predictions)} -> {len(filtered)} "
            f"(threshold={CONFIDENCE_THRESHOLD})"
        )

        # Step 2: Remove sensor/collimator border artifacts
        non_edge = []
        for p in filtered:
            x, y, w, h = p.get("x", 0), p.get("y", 0), p.get("width", 0), p.get("height", 0)
            x1, y1 = x - w / 2, y - h / 2
            x2, y2 = x + w / 2, y + h / 2
            
            raw_cls = str(p.get("class", "")).lower()
            is_cavity_class = "cavity" in raw_cls or "caries" in raw_cls
            conf = p.get("confidence", 0)

            # Remove sensor/collimator edge noise and film border artifacts
            # Bounding boxes that touch the film perimeter are sensor/margin artifacts
            is_border_noise = (
                (y1 <= 10) or 
                (y2 >= img_h - 10) or
                (x1 <= 10) or
                (x2 >= img_w - 10)
            )
            if not is_border_noise:
                non_edge.append(p)
            else:
                logger.info(f"Pruned edge border artifact: {p.get('class')} at y1={y1:.1f}, x1={x1:.1f}")

        # Step 3: Anatomical plausibility checks
        plausible = []
        for p in non_edge:
            w, h = p.get("width", 0), p.get("height", 0)
            det_area = w * h

            # Size guard: cavities can be small initial enamel caries; allow smaller fraction
            raw_cls = str(p.get("class", "")).lower()
            is_cavity = "cavity" in raw_cls or "caries" in raw_cls
            min_frac = 0.00008 if is_cavity else MIN_DETECTION_AREA_FRACTION

            area_frac = det_area / img_area if img_area > 0 else 0
            if area_frac < min_frac:
                logger.debug(f"Rejected tiny detection ({area_frac:.4f})")
                continue
            if area_frac > MAX_DETECTION_AREA_FRACTION:
                logger.debug(f"Rejected huge detection ({area_frac:.4f})")
                continue

            # Aspect-ratio guard
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > MAX_ASPECT_RATIO:
                logger.debug(f"Rejected extreme aspect ratio ({aspect:.1f})")
                continue

            plausible.append(p)

        logger.info(
            f"Anatomical plausibility: {len(non_edge)} -> {len(plausible)}"
        )

        # Step 4: Sort by confidence descending and apply NMS + Containment suppression
        plausible.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        keep = []
        for pred in plausible:
            is_duplicate = False
            for kept in keep:
                iou, iom = self._compute_iou_and_iom(pred, kept)
                same_class = str(pred.get("class", "")).lower() == str(kept.get("class", "")).lower()

                # NMS suppression by IoU
                if iou > NMS_IOU_THRESHOLD:
                    is_duplicate = True
                    break

                # Containment suppression for detections of the same condition
                if same_class and iom > CONTAINMENT_IOM_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                self._standardize_class(pred)
                self._ensure_detection_id(pred)
                keep.append(pred)

        logger.info(f"NMS & precision filter: {len(filtered)} -> {len(keep)}")
        result["predictions"] = keep
        return result

    @staticmethod
    def _compute_iou_and_iom(a: dict, b: dict) -> Tuple[float, float]:
        """Compute IoU and Intersection over Minimum Area (containment)."""
        ax1 = a["x"] - a["width"] / 2
        ay1 = a["y"] - a["height"] / 2
        ax2 = a["x"] + a["width"] / 2
        ay2 = a["y"] + a["height"] / 2

        bx1 = b["x"] - b["width"] / 2
        by1 = b["y"] - b["height"] / 2
        bx2 = b["x"] + b["width"] / 2
        by2 = b["y"] + b["height"] / 2

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0, 0.0

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        iou = inter / union if union > 0 else 0.0
        min_area = min(area_a, area_b)
        iom = inter / min_area if min_area > 0 else 0.0
        return iou, iom

    def _remap_to_original(
        self,
        result: dict,
        preprocess_info: dict,
        original_w: Optional[int],
        original_h: Optional[int],
    ) -> dict:
        """Translate boxes from cropped/CLAHE pixels onto the original image."""
        crop_region = preprocess_info.get("crop_region") if preprocess_info else None
        offset_x, offset_y = 0.0, 0.0
        if crop_region and len(crop_region) >= 2:
            offset_x = float(crop_region[0])
            offset_y = float(crop_region[1])

        processed_size = preprocess_info.get("processed_size") if preprocess_info else None
        processed_h = processed_w = None
        if processed_size and len(processed_size) >= 2:
            # OpenCV-style [height, width]
            processed_h, processed_w = processed_size[0], processed_size[1]

        crop_w = crop_region[2] if crop_region and len(crop_region) >= 4 else processed_w
        crop_h = crop_region[3] if crop_region and len(crop_region) >= 4 else processed_h

        scale_x = 1.0
        scale_y = 1.0
        if crop_w and crop_h and processed_w and processed_h:
            if processed_w != crop_w or processed_h != crop_h:
                scale_x = crop_w / max(processed_w, 1)
                scale_y = crop_h / max(processed_h, 1)

        for pred in result.get("predictions", []):
            pred["x"] = pred.get("x", 0) * scale_x + offset_x
            pred["y"] = pred.get("y", 0) * scale_y + offset_y
            pred["width"] = pred.get("width", 0) * scale_x
            pred["height"] = pred.get("height", 0) * scale_y

        if original_w and original_h:
            result["image"] = {"width": original_w, "height": original_h}
        return result

    def _merge_predictions(
        self,
        primary: List[Dict[str, Any]],
        extra: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge YOLO/verified boxes with Gemini discoveries, suppressing duplicates."""
        merged: List[Dict[str, Any]] = []
        combined = list(primary or []) + list(extra or [])
        combined.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        for pred in combined:
            self._standardize_class(pred)
            self._ensure_detection_id(pred)
            is_duplicate = False
            for kept in merged:
                iou, iom = self._compute_iou_and_iom(pred, kept)
                same_class = str(pred.get("class", "")).lower() == str(
                    kept.get("class", "")
                ).lower()
                if iou > NMS_IOU_THRESHOLD:
                    is_duplicate = True
                    break
                if same_class and iom > CONTAINMENT_IOM_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged.append(pred)
        logger.info(
            f"Merged detections: {len(primary or [])} verified + "
            f"{len(extra or [])} discovered → {len(merged)}"
        )
        return merged

    @staticmethod
    def _ensure_detection_id(pred: Dict[str, Any]) -> None:
        if not pred.get("detection_id"):
            pred["detection_id"] = str(uuid.uuid4())

    @staticmethod
    def _standardize_class(pred: Dict[str, Any]) -> None:
        raw_cls = str(pred.get("class", "")).lower().strip()
        if "cavity" in raw_cls or "caries" in raw_cls:
            pred["class"] = "Cavity"
            pred["class_id"] = 0
        elif (
            "periapical" in raw_cls
            or raw_cls in ("pa", "periapical lesion", "periapical_lesion")
        ):
            pred["class"] = "Periapical Lesion"
            pred["class_id"] = 1
        elif not raw_cls:
            pred["class"] = "Other Condition"
            pred["class_id"] = 2
        else:
            # Preserve specific other-condition names (e.g. Restoration)
            display = pred.get("class") or "Other Condition"
            if raw_cls in ("other", "other condition"):
                pred["class"] = "Other Condition"
            else:
                pred["class"] = str(display).strip().title()
            pred["class_id"] = 2

    def parse_dicom_metadata(self, dicom_file_path: str) -> DicomMetadata:
        """
        Parse DICOM metadata from file

        Args:
            dicom_file_path: Path to the DICOM file

        Returns:
            DicomMetadata: Parsed metadata

        Raises:
            HTTPException: If DICOM parsing fails
        """
        try:
            dicom_data = pydicom.dcmread(dicom_file_path)

            # Extract pixel spacing if available
            pixel_spacing = None
            if hasattr(dicom_data, "PixelSpacing"):
                pixel_spacing = [float(x) for x in dicom_data.PixelSpacing]

            metadata = DicomMetadata(
                patient_id=getattr(dicom_data, "PatientID", None),
                patient_name=(
                    str(getattr(dicom_data, "PatientName", None))
                    if hasattr(dicom_data, "PatientName")
                    else None
                ),
                patient_birth_date=getattr(dicom_data, "PatientBirthDate", None),
                patient_sex=getattr(dicom_data, "PatientSex", None),
                study_date=getattr(dicom_data, "StudyDate", None),
                study_time=getattr(dicom_data, "StudyTime", None),
                study_description=getattr(dicom_data, "StudyDescription", None),
                series_description=getattr(dicom_data, "SeriesDescription", None),
                modality=getattr(dicom_data, "Modality", None),
                manufacturer=getattr(dicom_data, "Manufacturer", None),
                manufacturer_model_name=getattr(
                    dicom_data, "ManufacturerModelName", None
                ),
                rows=getattr(dicom_data, "Rows", None),
                columns=getattr(dicom_data, "Columns", None),
                pixel_spacing=pixel_spacing,
                bits_allocated=getattr(dicom_data, "BitsAllocated", None),
                bits_stored=getattr(dicom_data, "BitsStored", None),
                photometric_interpretation=getattr(
                    dicom_data, "PhotometricInterpretation", None
                ),
                acquisition_date=getattr(dicom_data, "AcquisitionDate", None),
                acquisition_time=getattr(dicom_data, "AcquisitionTime", None),
                institution_name=getattr(dicom_data, "InstitutionName", None),
                referring_physician_name=(
                    str(getattr(dicom_data, "ReferringPhysicianName", None))
                    if hasattr(dicom_data, "ReferringPhysicianName")
                    else None
                ),
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to parse DICOM metadata: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Failed to parse DICOM file: {str(e)}"
            )

    def convert_dicom_to_image(self, dicom_file_path: str) -> Tuple[str, ImageInfo]:
        """
        Convert DICOM file to a standard image format for inference

        Args:
            dicom_file_path: Path to the DICOM file

        Returns:
            Tuple of (converted_image_path, image_info)

        Raises:
            HTTPException: If conversion fails
        """
        try:
            dicom_data = pydicom.dcmread(dicom_file_path)

            # Apply modality LUT if present
            if (
                hasattr(dicom_data, "ModalityLUTSequence")
                and dicom_data.ModalityLUTSequence
            ):
                try:
                    from pydicom.pixel_data_handlers.util import apply_modality_lut

                    pixel_array = apply_modality_lut(dicom_data.pixel_array, dicom_data)
                except Exception as e:
                    logger.warning(f"Failed to apply modality LUT: {e}")
                    pixel_array = dicom_data.pixel_array
            else:
                pixel_array = dicom_data.pixel_array

            # Apply VOI LUT if present (windowing)
            if hasattr(dicom_data, "VOILUTSequence") and dicom_data.VOILUTSequence:
                try:
                    from pydicom.pixel_data_handlers.util import apply_voi_lut

                    pixel_array = apply_voi_lut(pixel_array, dicom_data)
                except Exception as e:
                    logger.warning(f"Failed to apply VOI LUT: {e}")
            elif hasattr(dicom_data, "WindowCenter") and hasattr(
                dicom_data, "WindowWidth"
            ):
                # Apply windowing manually
                try:
                    window_center = float(
                        dicom_data.WindowCenter[0]
                        if isinstance(dicom_data.WindowCenter, list)
                        else dicom_data.WindowCenter
                    )
                    window_width = float(
                        dicom_data.WindowWidth[0]
                        if isinstance(dicom_data.WindowWidth, list)
                        else dicom_data.WindowWidth
                    )

                    # Apply windowing
                    pixel_array = np.clip(
                        (pixel_array - (window_center - window_width / 2))
                        / window_width
                        * 255,
                        0,
                        255,
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply windowing: {e}")

            # Handle photometric interpretation
            if hasattr(dicom_data, "PhotometricInterpretation"):
                if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                    # Invert for MONOCHROME1 (0 = white, max = black)
                    pixel_array = pixel_array.max() - pixel_array

            # Normalize pixel values to 0-255 range
            if pixel_array.dtype != np.uint8:
                # Get the min and max values
                min_val = pixel_array.min()
                max_val = pixel_array.max()

                if max_val > min_val:
                    # Normalize to 0-255
                    pixel_array = (
                        (pixel_array - min_val) / (max_val - min_val) * 255
                    ).astype(np.uint8)
                else:
                    # Handle case where all pixels have the same value
                    pixel_array = np.full(pixel_array.shape, 128, dtype=np.uint8)

            # Handle different image dimensions
            if len(pixel_array.shape) == 2:
                # Grayscale - convert to RGB
                rgb_array = np.stack([pixel_array] * 3, axis=-1)
            elif len(pixel_array.shape) == 3:
                if pixel_array.shape[2] == 1:
                    # Single channel - convert to RGB
                    rgb_array = np.repeat(pixel_array, 3, axis=2)
                elif pixel_array.shape[2] == 3:
                    # Already RGB
                    rgb_array = pixel_array
                else:
                    # Multi-channel - use first 3 channels or convert first channel to RGB
                    if pixel_array.shape[2] >= 3:
                        rgb_array = pixel_array[:, :, :3]
                    else:
                        rgb_array = np.stack([pixel_array[:, :, 0]] * 3, axis=-1)
            else:
                raise ValueError(f"Unsupported pixel array shape: {pixel_array.shape}")

            # Create PIL Image
            image = Image.fromarray(rgb_array.astype(np.uint8))

            # Save as temporary JPEG file
            temp_image_fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
            os.close(temp_image_fd)

            image.save(temp_image_path, "JPEG", quality=95)

            # Create type-safe image info
            image_info = ImageInfo(
                original_shape=list(dicom_data.pixel_array.shape),
                converted_format="JPEG",
                converted_size=list(image.size),
                original_dtype=str(dicom_data.pixel_array.dtype),
                pixel_array_min=float(dicom_data.pixel_array.min()),
                pixel_array_max=float(dicom_data.pixel_array.max()),
                photometric_interpretation=getattr(
                    dicom_data, "PhotometricInterpretation", None
                ),
                transfer_syntax=(
                    str(dicom_data.file_meta.TransferSyntaxUID)
                    if hasattr(dicom_data, "file_meta")
                    and hasattr(dicom_data.file_meta, "TransferSyntaxUID")
                    else None
                ),
            )

            return temp_image_path, image_info

        except Exception as e:
            logger.error(f"Failed to convert DICOM to image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to convert DICOM file to image: {str(e)}",
            )

    async def detect_dental_conditions_from_dicom(
        self, dicom_file_path: str, model_id: str = "adr/6"
    ) -> Tuple[dict, DicomMetadata, ImageInfo]:
        """
        Process DICOM file: extract metadata, convert to image, and run inference

        Args:
            dicom_file_path: Path to the DICOM file
            model_id: Model ID to use for inference

        Returns:
            Tuple of (inference_results, metadata, image_info)

        Raises:
            HTTPException: If processing fails
        """
        converted_image_path = None
        try:
            # Parse DICOM metadata
            metadata = self.parse_dicom_metadata(dicom_file_path)

            # Convert DICOM to image
            converted_image_path, image_info = self.convert_dicom_to_image(
                dicom_file_path
            )

            # Run inference on converted image
            inference_results = await self.detect_dental_conditions(
                converted_image_path, model_id
            )

            return inference_results, metadata, image_info

        finally:
            # Clean up converted image file
            if converted_image_path and os.path.exists(converted_image_path):
                try:
                    os.unlink(converted_image_path)
                except OSError as e:
                    logger.warning(
                        f"Failed to delete converted image file {converted_image_path}: {e}"
                    )


# Dependency function to get inference service
def get_inference_service() -> InferenceService:
    return InferenceService()
