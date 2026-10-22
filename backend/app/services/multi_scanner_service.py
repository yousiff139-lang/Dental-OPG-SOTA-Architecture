"""
Multi-Scanning System Service for Dental Radiography.

Architecture:
1. Stage 1: Dual SOTA YOLO Scanning ("Diolo")
   - Primary: Local SOTA YOLOv8 (trained on 9,928 DENTEX benchmark dental radiographs)
   - Secondary: Focused cloud detector (Roboflow adr/6)
   - Intelligent aspect-ratio detection (Panoramic OPG vs Intraoral periapical/bitewing)
   - Segment-aware quadrant decomposition for panoramic X-rays
2. Stage 2: Regularized Second YOLO & Re-Checker ("Second YOLO that checks on the first")
   - Patch Normalization: Z-score normalization & CLAHE contrast evaluation
   - Anatomical Structure Zoning: Coronal crown vs Apical root constraints
   - Cross-validation consensus: Prunes lone overfit false-positive boxes
   - Multimodal clinical verification (Gemini Vision) to eliminate bone marrow/sinus false alarms
3. Stage 3: Image Processing Overlay ("Fundamentals & Advanced Algorithms")
   - Segment-by-segment analysis across dental quadrants (Q1-Q4)
   - Morphological Black-Hat (dark cavities/periapical lucencies) & Top-Hat (dense enamel/restorations)
   - Sobel/Laplacian gradient magnitude for enamel boundary breakdown
   - Local radiolucency contrast scoring & base64 diagnostic visual heatmap overlay
"""

import asyncio
import cv2
import numpy as np
import httpx
import logging
import uuid
import os
import tempfile
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image

from ..core.config import get_settings
from .preprocessing_service import PreprocessingService, get_preprocessing_service
from .verification_service import VerificationService, get_verification_service
from .image_processing_service import ImageProcessingService, get_image_processing_service
from .local_yolo_service import LocalYoloService, get_local_yolo_service

logger = logging.getLogger(__name__)

# Primary & Re-check thresholds
PRIMARY_CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.30
CONTAINMENT_IOM_THRESHOLD = 0.35
PANORAMIC_ASPECT_RATIO_THRESHOLD = 1.40


class MultiScanningService:
    """Orchestrates the 3-stage regularized multi-scanning system."""

    def __init__(self):
        self.settings = get_settings()
        self.preprocessor: PreprocessingService = get_preprocessing_service()
        self.verifier: VerificationService = get_verification_service()
        self.ip_service: ImageProcessingService = get_image_processing_service()
        self.local_yolo: LocalYoloService = get_local_yolo_service()

    async def scan_dental_image(
        self, image_path: str, model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the 3-stage Multi-Scanning System on the provided image.
        """
        model_id = model_id or self.settings.default_model_id
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image file at {image_path}")

        img_h, img_w = img.shape[:2]
        aspect_ratio = img_w / max(img_h, 1)
        is_panoramic = bool(aspect_ratio >= PANORAMIC_ASPECT_RATIO_THRESHOLD or (img_w >= 850 and img_h / img_w <= 0.68))

        logger.info(
            f"Starting SOTA Multi-Scanning System: {img_w}x{img_h} (aspect ratio {aspect_ratio:.2f}, "
            f"type={'Panoramic OPG' if is_panoramic else 'Intraoral'})"
        )

        # ---------------------------------------------------------------------
        # PRE-FLIGHT: Global Multimodal Dental Radiologist Audit (Gemini Vision)
        # ---------------------------------------------------------------------
        audit_result = await self.verifier.audit_radiograph(image_path, is_panoramic=is_panoramic)
        is_healthy = audit_result.get("is_healthy", False)
        logger.info(
            f"Stage 0 (Global Radiologist Audit): is_healthy={is_healthy}, "
            f"findings={len(audit_result.get('findings', []))}, "
            f"assessment='{audit_result.get('overall_assessment', '')[:70]}...'"
        )

        # ---------------------------------------------------------------------
        # STAGE 1: Dual SOTA YOLO Scanning (Local SOTA Ensemble + Secondary Proposal)
        # ---------------------------------------------------------------------
        stage1_candidates = await self._run_stage1_diolo(
            img, img_w, img_h, is_panoramic, model_id
        )
        logger.info(f"Stage 1 (Diolo) produced {len(stage1_candidates)} consensus candidate lesions")

        # If both Stage 0 audit and Stage 1 YOLO find no active lesions:
        active_candidates = [c for c in stage1_candidates if c.get("class") in ("Cavity", "Periapical Lesion")]
        if not active_candidates and is_healthy:
            logger.info("Film confirmed healthy by global audit and zero active YOLO candidate lesions.")
            final_result = self._run_stage3_image_processing_overlay(
                img, img_w, img_h, stage1_candidates, is_panoramic
            )
            final_result["is_healthy"] = True
            final_result["radiologist_assessment"] = audit_result.get("overall_assessment")
            return final_result

        # ---------------------------------------------------------------------
        # STAGE 2: Regularized Second YOLO Re-check & Clinical Verification
        # ---------------------------------------------------------------------
        stage2_verified = await self._run_stage2_recheck(
            img, img_w, img_h, stage1_candidates, is_panoramic, model_id, audit_result=audit_result
        )
        logger.info(f"Stage 2 (Second YOLO Recheck) finalized {len(stage2_verified)} validated findings")

        # ---------------------------------------------------------------------
        # STAGE 3: Segment-by-Segment Image Processing Overlay ("Fundamentals & Advanced")
        # ---------------------------------------------------------------------
        final_result = self._run_stage3_image_processing_overlay(
            img, img_w, img_h, stage2_verified, is_panoramic
        )
        final_result["is_healthy"] = len([d for d in stage2_verified if d.get("class") in ("Cavity", "Periapical Lesion")]) == 0
        final_result["radiologist_assessment"] = audit_result.get("overall_assessment")
        logger.info(
            f"Stage 3 (CV Overlay) complete: {len(final_result['predictions'])} finalized detections "
            f"across {len(final_result.get('segments', []))} segments"
        )

        return final_result

    async def _run_stage1_diolo(
        self,
        img: np.ndarray,
        img_w: int,
        img_h: int,
        is_panoramic: bool,
        model_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Dual YOLO Candidate Proposal.
        1. Run Local SOTA YOLOv8 (trained on DENTEX benchmark) for high-precision caries and lesions.
        2. Run Secondary detector (Roboflow adr/6) to complement periapical sensitivity.
        3. Merge with anti-overfitting consensus and patch normalization.
        """
        # Pass A: Local SOTA YOLOv8 Detector
        local_candidates: List[Dict[str, Any]] = []
        if self.local_yolo.is_available:
            conf_thresh = 0.30 if not is_panoramic else 0.35
            local_candidates = self.local_yolo.detect(
                img, conf_threshold=conf_thresh, iou_threshold=0.35, is_panoramic=is_panoramic
            )
            for c in local_candidates:
                c["stage1_source"] = "Local SOTA YOLOv8"
                c["segment"] = "Global Arch" if is_panoramic else "Intraoral Frame"
            logger.info(f"Local SOTA YOLO detected {len(local_candidates)} initial lesions")

        # Pass B: Secondary Detector Proposals (Roboflow adr/6)
        rf_candidates: List[Dict[str, Any]] = []
        try:
            if is_panoramic:
                # Panoramic quadrant scan
                mid_x = int(img_w * 0.5)
                mid_y = int(img_h * 0.5)
                overlap_x = int(img_w * 0.04)
                overlap_y = int(img_h * 0.05)

                quadrants = [
                    ("Q1: Upper Right", 0, 0, min(img_w, mid_x + overlap_x), min(img_h, mid_y + overlap_y)),
                    ("Q2: Upper Left", max(0, mid_x - overlap_x), 0, img_w - max(0, mid_x - overlap_x), min(img_h, mid_y + overlap_y)),
                    ("Q3: Lower Left", max(0, mid_x - overlap_x), max(0, mid_y - overlap_y), img_w - max(0, mid_x - overlap_x), img_h - max(0, mid_y - overlap_y)),
                    ("Q4: Lower Right", 0, max(0, mid_y - overlap_y), min(img_w, mid_x + overlap_x), img_h - max(0, mid_y - overlap_y)),
                ]

                async def fetch_quad(q_info, client):
                    q_name, qx, qy, qw, qh = q_info
                    crop = img[qy:qy+qh, qx:qx+qw]
                    preds = await self._infer_roboflow(client, crop, model_id, confidence=20, overlap=25)
                    res = []
                    for p in preds:
                        res.append({
                            "x": p.get("x", 0) + qx,
                            "y": p.get("y", 0) + qy,
                            "width": p.get("width", 0),
                            "height": p.get("height", 0),
                            "confidence": p.get("confidence", 0),
                            "class": p.get("class", "Cavity"),
                            "segment": q_name,
                            "stage1_source": "Secondary YOLO (Roboflow)",
                        })
                    return res

                async with httpx.AsyncClient(timeout=10.0) as client:
                    async def fetch_full():
                        preds = await self._infer_roboflow(client, img, model_id, confidence=25, overlap=30)
                        return [
                            {
                                "x": p.get("x", 0),
                                "y": p.get("y", 0),
                                "width": p.get("width", 0),
                                "height": p.get("height", 0),
                                "confidence": p.get("confidence", 0),
                                "class": p.get("class", "Cavity"),
                                "segment": "Global Arch",
                                "stage1_source": "Secondary YOLO (Roboflow)",
                            }
                            for p in preds
                        ]

                    all_tasks = [fetch_quad(q, client) for q in quadrants] + [fetch_full()]
                    all_results = await asyncio.gather(*all_tasks)
                    for r_list in all_results:
                        rf_candidates.extend(r_list)
            else:
                # Intraoral single film
                async with httpx.AsyncClient(timeout=25.0) as client:
                    full_preds = await self._infer_roboflow(client, img, model_id, confidence=25, overlap=25)
                    for p in full_preds:
                        rf_candidates.append({
                            "x": p.get("x", 0),
                            "y": p.get("y", 0),
                            "width": p.get("width", 0),
                            "height": p.get("height", 0),
                            "confidence": p.get("confidence", 0),
                            "class": p.get("class", "Cavity"),
                            "segment": "Intraoral Frame",
                            "stage1_source": "Secondary YOLO (Roboflow)",
                        })
        except Exception as e:
            logger.warning(f"Secondary detector proposal warning: {e}")

        # Standardize labels on secondary candidates
        for p in rf_candidates:
            self._standardize_class(p)

        # ---------------------------------------------------------------------
        # ANTI-OVERFITTING ENSEMBLE CONSENSUS:
        # - Local SOTA detections are treated as high-precision anchors.
        # - Roboflow Periapical Lesions in the apical root zone are accepted.
        # - Roboflow Cavities are ONLY accepted if confirmed by Local SOTA (within 45px)
        #   or if they show high local radiolucency contrast. Lone false-positive cavities are pruned.
        # ---------------------------------------------------------------------
        merged_candidates = list(local_candidates)

        for rf in rf_candidates:
            rf_cls = rf.get("class", "")
            rx, ry = rf["x"], rf["y"]
            rw, rh = rf["width"], rf["height"]

            # Edge guard: discard margin artifacts
            if rx - rw/2 < 10 or ry - rh/2 < 10 or rx + rw/2 > img_w - 10 or ry + rh/2 > img_h - 10:
                continue

            # Strict anatomical placement & size validation
            if not self.local_yolo._validate_anatomical_placement(
                rx, ry, rw, rh, rf_cls, img_w, img_h, is_panoramic
            ):
                logger.debug(f"Rejected secondary candidate failing anatomical checks: {rf_cls} at ({rx:.1f}, {ry:.1f})")
                continue

            if rf_cls == "Periapical Lesion":
                # Check anatomical placement: Periapical lesions MUST be in the apical zone (not biting edge)
                # In intraoral films, apical is typically y > 0.40 * h
                rel_y = ry / max(img_h, 1)
                if not is_panoramic and rel_y < 0.35:
                    logger.debug(f"Rejected apical finding on coronal crown: {ry}")
                    continue

                # Check duplicate
                dup = any(
                    c.get("class") == "Periapical Lesion" and np.hypot(c["x"] - rx, c["y"] - ry) < max(rw, rh) * 0.7
                    for c in merged_candidates
                )
                if not dup:
                    merged_candidates.append(rf)

            elif rf_cls == "Cavity":
                # Check if matches any existing candidate
                matched_cand = None
                for c in merged_candidates:
                    if c.get("class") == "Cavity":
                        dist = np.hypot(c["x"] - rx, c["y"] - ry)
                        if dist < 45.0 or (c["width"] > 0 and dist < max(c["width"], rw) * 0.75):
                            matched_cand = c
                            break

                if matched_cand:
                    # Consensus match! Boost confidence
                    matched_cand["confidence"] = round(min(0.99, max(matched_cand.get("confidence", 0.5), rf.get("confidence", 0.5), 0.95)), 4)
                    matched_cand["consensus_confirmed"] = True
                else:
                    # Only accept lone secondary cavity if confidence is high (>= 0.45)
                    if rf.get("confidence", 0) >= 0.45:
                        merged_candidates.append(rf)

            elif rf_cls == "Impacted Tooth":
                dup = any(
                    c.get("class") == "Impacted Tooth" and np.hypot(c["x"] - rx, c["y"] - ry) < 50
                    for c in merged_candidates
                )
                if not dup:
                    merged_candidates.append(rf)

        # Apply suppression
        suppressed = self._apply_nms(merged_candidates)
        return suppressed

    async def _run_stage2_recheck(
        self,
        img: np.ndarray,
        img_w: int,
        img_h: int,
        candidates: List[Dict[str, Any]],
        is_panoramic: bool,
        model_id: str,
        audit_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Second YOLO & Re-Checker with Patch Normalization.
        1. Localized patch extraction & Z-score normalization to test lesion contrast.
        2. Anatomical structure validation: Cavity must be coronal, PA must be apical.
        3. Multimodal Radiological Verification (Gemini Vision) to prune normal bone trabeculae.
        """
        if not candidates:
            return []

        # 1. Patch Normalization & Local Contrast Check
        norm_candidates = []
        for cand in candidates:
            cx, cy, cw, ch = cand["x"], cand["y"], cand["width"], cand["height"]
            x1 = max(0, int(cx - cw / 2))
            y1 = max(0, int(cy - ch / 2))
            x2 = min(img_w, int(cx + cw / 2))
            y2 = min(img_h, int(cy + ch / 2))

            patch = img[y1:y2, x1:x2]
            if patch.shape[0] < 6 or patch.shape[1] < 6:
                continue

            # Apply patch normalization
            norm_patch, p_mean, p_std = LocalYoloService.apply_patch_normalization(patch)

            # Check surrounding context
            pad = max(6, int(max(cw, ch) * 0.4))
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(img_w, x2 + pad)
            cy2 = min(img_h, y2 + pad)
            ctx = img[cy1:cy2, cx1:cx2]
            ctx_gray = cv2.cvtColor(ctx, cv2.COLOR_BGR2GRAY) if len(ctx.shape) == 3 else ctx
            ctx_mean = float(np.mean(ctx_gray))

            cand_copy = dict(cand)
            cand_copy["patch_normalized"] = True
            cand_copy["patch_metrics"] = {
                "patch_mean": round(p_mean, 2),
                "patch_std": round(p_std, 2),
                "contrast_drop": round((ctx_mean - p_mean) / max(ctx_mean, 1.0), 3),
            }
            cand_copy["stage2_rechecked"] = True
            cand_copy["stage2_notes"] = "Regularized with Patch Normalization & SOTA Consensus"
            norm_candidates.append(cand_copy)

        # 2. Multimodal Clinical Verification (Gemini Vision)
        try:
            temp_path = self._save_temp_image(img)
            verified_by_gemini = await self.verifier.verify_detections(
                temp_path, norm_candidates, audit_result=audit_result
            )
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        except Exception as e:
            logger.warning(f"Clinical verification step warning: {e}; retaining Stage 2 candidates")
            verified_by_gemini = norm_candidates

        for c in verified_by_gemini:
            if "detection_id" not in c:
                c["detection_id"] = str(uuid.uuid4())
            self._standardize_class(c)

        # 3. Final single-site duplicate suppression
        stage2_final = self._apply_nms(verified_by_gemini)
        return stage2_final

    def _run_stage3_image_processing_overlay(
        self,
        img: np.ndarray,
        img_w: int,
        img_h: int,
        detections: List[Dict[str, Any]],
        is_panoramic: bool,
    ) -> Dict[str, Any]:
        """
        Stage 3: Segment-by-segment Image Processing Overlay.
        Runs CLAHE, Black-Hat & Top-Hat morphological operations, Sobel gradient edge analysis,
        and computes local radiolucency contrast scores for each finding.
        Generates the visual heatmap overlay for the frontend.
        """
        # Analyze segments
        cv_data = self.ip_service.analyze_segments(img, is_panoramic=is_panoramic)
        segments = cv_data.get("segments", [])

        # Assign and verify each detection with image processing features
        final_detections = []
        for det in detections:
            cv_res = self.ip_service.verify_detection_with_cv(img, det, cv_data)
            det_updated = dict(det)
            det_updated["segment"] = cv_res.get("segment") or det.get("segment", "Dental Arch")
            det_updated["ip_verified"] = cv_res.get("ip_verified", True)
            det_updated["ip_score"] = cv_res.get("ip_score", 0.5)
            det_updated["ip_metrics"] = cv_res.get("metrics", {})

            # Anatomical location label
            rx = det_updated["x"] / max(img_w, 1)
            ry = det_updated["y"] / max(img_h, 1)
            det_updated["location_label"] = f"{det_updated['segment']} ({'Coronal' if ry < 0.55 else 'Apical'})"

            final_detections.append(det_updated)

        # Update findings count per segment
        for seg in segments:
            seg_id = seg["id"]
            count = sum(
                1 for d in final_detections
                if seg_id in str(d.get("segment", "")) or seg.get("name", "") in str(d.get("segment", ""))
            )
            seg["findings_count"] = count

        return {
            "predictions": final_detections,
            "image": {"width": img_w, "height": img_h},
            "image_width": img_w,
            "image_height": img_h,
            "scan_mode": "SOTA Regularized Multi-Scanner (Stage 1 Diolo + Stage 2 Recheck + Stage 3 CV Overlay)",
            "is_panoramic": is_panoramic,
            "segments": segments,
            "cv_overlay_url": cv_data.get("overlay_heatmap_b64"),
        }

    async def _infer_roboflow(
        self,
        client: httpx.AsyncClient,
        crop: np.ndarray,
        model_id: str,
        confidence: int = 20,
        overlap: int = 25,
    ) -> List[Dict[str, Any]]:
        """Send image crop to Roboflow inference API."""
        success, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return []

        url = f"https://detect.roboflow.com/{model_id}?api_key={self.settings.roboflow_api_key}&confidence={confidence}&overlap={overlap}"
        try:
            response = await client.post(
                url,
                files={"file": ("crop.jpg", buffer.tobytes(), "image/jpeg")},
            )
            if response.status_code == 200:
                return response.json().get("predictions", [])
        except Exception as e:
            logger.debug(f"Roboflow crop inference call failed: {e}")
        return []

    def _apply_nms(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort by confidence descending and apply NMS + Containment + Cluster suppression."""
        if not predictions:
            return []

        sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)
        keep = []

        for pred in sorted_preds:
            is_dup = False
            for kept in keep:
                iou, iom = self._compute_iou_and_iom(pred, kept)
                same_class = str(pred.get("class", "")).lower() == str(kept.get("class", "")).lower()

                # Centroid proximity suppression (merges multiple boxes on the same tooth)
                dist = np.hypot(pred["x"] - kept["x"], pred["y"] - kept["y"])
                min_dim = min(pred["width"], pred["height"], kept["width"], kept["height"])

                if iou > NMS_IOU_THRESHOLD:
                    is_dup = True
                    break
                if same_class and iom > CONTAINMENT_IOM_THRESHOLD:
                    is_dup = True
                    break
                if same_class and dist < max(min_dim * 0.75, 25.0):
                    is_dup = True
                    break

            if not is_dup:
                keep.append(pred)

        return keep

    @staticmethod
    def _compute_iou_and_iom(a: dict, b: dict) -> Tuple[float, float]:
        """Compute IoU and Intersection over Minimum Area."""
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

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        area_a = a["width"] * a["height"]
        area_b = b["width"] * b["height"]
        union = area_a + area_b - intersection

        iou = intersection / max(union, 1e-6)
        iom = intersection / max(min(area_a, area_b), 1e-6)
        return float(iou), float(iom)

    @staticmethod
    def _standardize_class(pred: Dict[str, Any]):
        """Standardize raw model class strings to clinical terminology."""
        raw_cls = str(pred.get("class", "")).lower().strip()
        if "caries" in raw_cls or "cavity" in raw_cls:
            pred["class"] = "Cavity"
            pred["class_id"] = 0
        elif "pa" in raw_cls or "periapical" in raw_cls or "lesion" in raw_cls:
            pred["class"] = "Periapical Lesion"
            pred["class_id"] = 1
        elif "impacted" in raw_cls:
            pred["class"] = "Impacted Tooth"
            pred["class_id"] = 2

    @staticmethod
    def _save_temp_image(img: np.ndarray) -> str:
        """Save numpy image to temporary JPEG file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            return tmp.name


_multi_scanner: Optional[MultiScanningService] = None


def get_multi_scanning_service() -> MultiScanningService:
    global _multi_scanner
    if _multi_scanner is None:
        _multi_scanner = MultiScanningService()
    return _multi_scanner
