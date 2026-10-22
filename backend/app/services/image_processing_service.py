"""
Fundamental and advanced image processing service for dental radiographs.

Provides:
1. Segment-by-segment anatomical decomposition (Q1-Q4 quadrants / arches)
2. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
3. Morphological filtering (Black-Hat for radiolucent cavities/lesions, Top-Hat for enamel/fillings)
4. Enamel gradient & edge continuity analysis (Sobel / Laplacian)
5. Local radiolucency contrast scoring for candidate lesion validation
6. Visual heatmap overlay generation for the frontend
"""

import cv2
import numpy as np
import base64
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ImageProcessingService:
    """Advanced computer vision and image processing algorithms for dental radiography."""

    def __init__(self):
        self.clahe_clip = 3.0
        self.clahe_grid = (8, 8)
        self.morph_kernel_size = (15, 15)

    def analyze_segments(
        self, img: np.ndarray, is_panoramic: bool = True
    ) -> Dict[str, Any]:
        """
        Decompose radiograph into dental segments/quadrants and compute
        fundamental and advanced image processing metrics on each segment.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # Step 1: CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        enhanced = clahe.apply(gray)

        # Step 2: Morphological filtering
        # Black-Hat isolates dark structures on bright backgrounds (caries pits, periapical radiolucencies)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

        # Top-Hat isolates bright structures (dense enamel crowns, metallic fillings)
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

        # Step 3: Edge and gradient analysis for enamel continuity
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient = cv2.magnitude(sobelx, sobely)
        gradient_norm = np.uint8(np.clip(gradient, 0, 255))

        # Define segments
        if is_panoramic:
            # 4 Anatomical Quadrants with slight overlap
            mid_x = int(w * 0.5)
            mid_y = int(h * 0.5)
            overlap_x = int(w * 0.03)
            overlap_y = int(h * 0.04)

            segments = [
                {
                    "id": "Q1",
                    "name": "Q1: Upper Right",
                    "dental_code": "UR",
                    "description": "Maxillary Right Quadrant (Teeth 18-11)",
                    "x": 0,
                    "y": 0,
                    "width": mid_x + overlap_x,
                    "height": mid_y + overlap_y,
                },
                {
                    "id": "Q2",
                    "name": "Q2: Upper Left",
                    "dental_code": "UL",
                    "description": "Maxillary Left Quadrant (Teeth 21-28)",
                    "x": max(0, mid_x - overlap_x),
                    "y": 0,
                    "width": w - max(0, mid_x - overlap_x),
                    "height": mid_y + overlap_y,
                },
                {
                    "id": "Q3",
                    "name": "Q3: Lower Left",
                    "dental_code": "LL",
                    "description": "Mandibular Left Quadrant (Teeth 31-38)",
                    "x": max(0, mid_x - overlap_x),
                    "y": max(0, mid_y - overlap_y),
                    "width": w - max(0, mid_x - overlap_x),
                    "height": h - max(0, mid_y - overlap_y),
                },
                {
                    "id": "Q4",
                    "name": "Q4: Lower Right",
                    "dental_code": "LR",
                    "description": "Mandibular Right Quadrant (Teeth 48-41)",
                    "x": 0,
                    "y": max(0, mid_y - overlap_y),
                    "width": mid_x + overlap_x,
                    "height": h - max(0, mid_y - overlap_y),
                },
            ]
        else:
            # Single-film intraoral radiograph: coronal (crowns) vs apical (roots)
            mid_y = int(h * 0.5)
            segments = [
                {
                    "id": "Coronal",
                    "name": "Coronal Segment",
                    "dental_code": "CRN",
                    "description": "Tooth crowns, enamel caps, and interproximal contact zones",
                    "x": 0,
                    "y": 0,
                    "width": w,
                    "height": int(h * 0.6),
                },
                {
                    "id": "Apical",
                    "name": "Apical / Periapical Segment",
                    "dental_code": "APC",
                    "description": "Root apices, periodontal ligament space, and alveolar bone",
                    "x": 0,
                    "y": int(h * 0.4),
                    "width": w,
                    "height": h - int(h * 0.4),
                },
            ]

        # Calculate statistics for each segment
        analyzed_segments = []
        for seg in segments:
            sx, sy, sw, sh = seg["x"], seg["y"], seg["width"], seg["height"]
            seg_gray = gray[sy:sy+sh, sx:sx+sw]
            seg_bh = blackhat[sy:sy+sh, sx:sx+sw]
            seg_th = tophat[sy:sy+sh, sx:sx+sw]
            seg_grad = gradient_norm[sy:sy+sh, sx:sx+sw]

            mean_density = float(np.mean(seg_gray))
            bh_activity = float(np.mean(seg_bh))
            th_activity = float(np.mean(seg_th))
            edge_sharpness = float(np.mean(seg_grad))

            analyzed_segments.append({
                **seg,
                "metrics": {
                    "mean_density": round(mean_density, 2),
                    "radiolucency_activity": round(bh_activity, 2),
                    "radiopaque_activity": round(th_activity, 2),
                    "edge_gradient": round(edge_sharpness, 2),
                }
            })

        # Generate base64 visual overlay heatmap (composite of blackhat radiolucency & edge boundaries)
        overlay_b64 = self._generate_overlay_heatmap(blackhat, gradient_norm)

        return {
            "segments": analyzed_segments,
            "overlay_heatmap_b64": overlay_b64,
            "image_dims": {"width": w, "height": h},
            "enhanced_image": enhanced,
            "blackhat": blackhat,
            "gradient": gradient_norm,
        }

    def verify_detection_with_cv(
        self,
        img: np.ndarray,
        pred: Dict[str, Any],
        cv_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a candidate bounding box using local pixel metrics:
        - Local radiolucency contrast ratio
        - Morphological black-hat signal
        - Enamel edge discontinuity score
        """
        h, w = img.shape[:2]
        cx = pred.get("x", 0)
        cy = pred.get("y", 0)
        bw = pred.get("width", 0)
        bh = pred.get("height", 0)

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(w, int(cx + bw / 2))
        y2 = min(h, int(cy + bh / 2))

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return {
                "ip_verified": True,
                "ip_score": 0.5,
                "metrics": {"reason": "Box too small for CV verification"},
            }

        # Crop patch and surrounding context
        pad = max(4, int(max(bw, bh) * 0.4))
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        patch = gray[y1:y2, x1:x2]
        context = gray[cy1:cy2, cx1:cx2]

        patch_mean = float(np.mean(patch))
        context_mean = float(np.mean(context))

        # Radiolucency ratio: positive if patch is darker than surrounding tooth/bone
        contrast_ratio = (context_mean - patch_mean) / max(context_mean, 1.0)

        # Morphological black-hat response inside the box
        if cv_data and "blackhat" in cv_data:
            bh_patch = cv_data["blackhat"][y1:y2, x1:x2]
            bh_score = float(np.mean(bh_patch)) / 255.0
        else:
            bh_score = 0.5

        # Gradient magnitude inside the box perimeter
        if cv_data and "gradient" in cv_data:
            grad_patch = cv_data["gradient"][y1:y2, x1:x2]
            grad_score = float(np.mean(grad_patch)) / 255.0
        else:
            grad_score = 0.5

        cls_lower = str(pred.get("class", "")).lower()
        is_cavity = "cavity" in cls_lower or "caries" in cls_lower

        # Lesions are radiolucent (darker than enamel/bone) or exhibit clear edge boundary contrast
        ip_score = float(np.clip(
            0.40 * max(0.0, contrast_ratio * 2.0) +
            0.35 * bh_score +
            0.25 * grad_score,
            0.0, 1.0
        ))

        # Determine segment
        segment_id = self._determine_segment(cx, cy, w, h)

        return {
            "ip_verified": bool(ip_score >= 0.15 or contrast_ratio >= -0.05),
            "ip_score": round(ip_score, 4),
            "segment": segment_id,
            "metrics": {
                "contrast_ratio": round(contrast_ratio, 3),
                "blackhat_signal": round(bh_score, 3),
                "gradient_edge": round(grad_score, 3),
            },
        }

    def _determine_segment(self, cx: float, cy: float, w: int, h: int) -> str:
        """Assign finding to its dental anatomical quadrant or zone."""
        if w > h * 1.3:
            # Panoramic radiograph
            if cy < h * 0.5:
                return "Q1: Upper Right" if cx < w * 0.5 else "Q2: Upper Left"
            else:
                return "Q4: Lower Right" if cx < w * 0.5 else "Q3: Lower Left"
        else:
            # Intraoral
            return "Coronal Zone" if cy < h * 0.5 else "Apical Zone"

    def _generate_overlay_heatmap(
        self, blackhat: np.ndarray, gradient: np.ndarray
    ) -> str:
        """
        Creates a colorized CV diagnostic overlay (cyan-amber radiolucency & edge breaches)
        and encodes it as a base64 PNG data URL.
        """
        try:
            # Combine blackhat radiolucency with gradient edge breaks
            combined = cv2.addWeighted(blackhat, 0.7, gradient, 0.3, 0)
            # Apply colormap: COLORMAP_INFERNO or COLORMAP_TURBO gives clear medical distinction
            colored = cv2.applyColorMap(combined, cv2.COLORMAP_MAGMA)

            # Mask out background/noise below threshold
            mask = combined > 35
            colored[~mask] = [0, 0, 0]

            # Compress for fast web transfer
            success, buffer = cv2.imencode(".png", colored)
            if success:
                b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
                return f"data:image/png;base64,{b64}"
        except Exception as e:
            logger.warning(f"Could not generate CV overlay heatmap: {e}")
        return ""


_ip_service: Optional[ImageProcessingService] = None


def get_image_processing_service() -> ImageProcessingService:
    global _ip_service
    if _ip_service is None:
        _ip_service = ImageProcessingService()
    return _ip_service
