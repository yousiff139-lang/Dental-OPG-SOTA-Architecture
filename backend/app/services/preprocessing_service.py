"""
Image preprocessing service for dental X-ray images.

Automatically detects and crops the X-ray film region from real-world
photographs (removing hands, gloves, backgrounds) and applies CLAHE
contrast enhancement to boost model accuracy.
"""

import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Preprocesses dental X-ray images before inference."""

    # Minimum fraction of the total image area that a valid X-ray region
    # must occupy.  If the largest dark contour is smaller than this we
    # assume the image is already a clean X-ray and skip cropping.
    MIN_XRAY_AREA_FRACTION = 0.04  # 4 %
    # Maximum fraction — reject contours that span the entire image
    MAX_XRAY_AREA_FRACTION = 0.92  # 92 %

    # CLAHE parameters
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_TILE_GRID = (8, 8)

    # Padding around the detected X-ray region (fraction of region size)
    CROP_PADDING = 0.02  # 2 % padding

    def preprocess_image(self, image_path: str) -> Tuple[str, dict]:
        """
        Full preprocessing pipeline:
        1. Auto-crop X-ray film region (if photo contains background)
        2. Apply CLAHE contrast enhancement
        3. Save processed image to a temp file

        Args:
            image_path: Path to the original image

        Returns:
            Tuple of (processed_image_path, preprocessing_info_dict)
        """
        info = {
            "cropped": False,
            "clahe_applied": False,
            "original_size": None,
            "crop_region": None,
            "processed_size": None,
        }

        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(
                    f"OpenCV could not read {image_path}; skipping preprocessing"
                )
                return image_path, info

            info["original_size"] = list(img.shape[:2])  # [height, width]

            # Step 1: Detect and crop X-ray region (phone photos of films only)
            cropped_img, crop_region = self._auto_crop_xray(img)
            if crop_region is not None:
                info["cropped"] = True
                info["crop_region"] = crop_region
                img = cropped_img
                logger.info(
                    f"Auto-cropped X-ray region: {crop_region}"
                )

            # Step 2: Apply CLAHE contrast enhancement
            enhanced_img = self._apply_clahe(img)
            info["clahe_applied"] = True

            info["processed_size"] = list(enhanced_img.shape[:2])

            # Step 3: Save to a temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(temp_fd)
            cv2.imwrite(temp_path, enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            logger.info(
                f"Preprocessing complete: "
                f"cropped={info['cropped']}, "
                f"original={info['original_size']}, "
                f"processed={info['processed_size']}"
            )
            return temp_path, info

        except Exception as e:
            logger.error(f"Preprocessing failed, using original: {e}")
            return image_path, info

    def _auto_crop_xray(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[list]]:
        """
        Detect the dark rectangular X-ray film within a real-world photo.

        Strategy:
        - Convert to grayscale
        - Adaptive threshold to isolate the dark X-ray region
        - Find the largest contour that meets area constraints
        - Crop to the bounding rect of that contour with padding

        Returns:
            (cropped_image, [x, y, w, h]) or (original_image, None) if
            no film region detected.
        """
        # Preserve full radiograph frame so tooth crowns, enamel, and cavities
        # are never clipped or lost.
        return img, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use Otsu's threshold to separate dark X-ray from lighter background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological close to fill small holes within the film
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Also try using the inverted approach for images where
        # X-ray is brighter than the background
        contours_inv, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Also check the non-inverted binary for edge cases
        binary_normal = cv2.bitwise_not(binary)
        contours_norm, _ = cv2.findContours(
            binary_normal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Combine and find the best candidate
        all_contours = list(contours_inv) + list(contours_norm)

        best_contour = None
        best_score = 0

        for contour in all_contours:
            area = cv2.contourArea(contour)
            area_frac = area / total_area

            # Must be a significant region but not the entire image
            if area_frac < self.MIN_XRAY_AREA_FRACTION:
                continue
            if area_frac > self.MAX_XRAY_AREA_FRACTION:
                continue

            # Prefer contours that look rectangular (X-ray films are rectangular)
            _, _, cw, ch = cv2.boundingRect(contour)
            bounding_area = cw * ch
            bounding_frac = bounding_area / total_area if total_area > 0 else 0
            # Skip "crops" that are basically the whole photo (hands, window, walls)
            if bounding_frac > self.MAX_XRAY_AREA_FRACTION:
                continue

            rectangularity = area / bounding_area if bounding_area > 0 else 0

            # Prefer compact film-like rectangles over tall full-frame blobs
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            aspect_penalty = 1.0 if aspect <= 2.2 else 0.4

            # Score: reward decent size and rectangular shape
            score = area_frac * rectangularity * aspect_penalty
            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is None:
            logger.info("No distinct X-ray film region detected; using full image")
            return img, None

        # Get bounding rectangle with padding
        bx, by, bw, bh = cv2.boundingRect(best_contour)

        pad_x = int(bw * self.CROP_PADDING)
        pad_y = int(bh * self.CROP_PADDING)

        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)

        cropped = img[y1:y2, x1:x2]

        # Sanity check: cropped region must be reasonably sized
        if cropped.shape[0] < 50 or cropped.shape[1] < 50:
            logger.warning("Cropped region too small; using full image")
            return img, None

        return cropped, [x1, y1, x2 - x1, y2 - y1]

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization)
        to improve X-ray contrast for better model detection.

        Works on the luminance channel so color information (if any) is
        preserved.
        """
        if len(img.shape) == 2:
            # Already grayscale
            clahe = cv2.createCLAHE(
                clipLimit=self.CLAHE_CLIP_LIMIT,
                tileGridSize=self.CLAHE_TILE_GRID,
            )
            return clahe.apply(img)

        # Convert to LAB color space, apply CLAHE on L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT,
            tileGridSize=self.CLAHE_TILE_GRID,
        )
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return result


def get_preprocessing_service() -> PreprocessingService:
    return PreprocessingService()
