"""
Gemini Vision verification and discovery for dental condition detection.

After the primary YOLO model (Roboflow) makes detections, this service:
1. Verifies each crop (prefer class correction over deletion).
2. Scans the full cropped film for missed cavities, periapical lesions,
   and other radiographic findings.
"""

import asyncio
import base64
import json
import logging
import uuid
from typing import List, Dict, Any, Optional

import cv2
import httpx
import numpy as np

from ..core.config import get_settings

logger = logging.getLogger(__name__)

# Gemini API key (from settings)
_FALLBACK_GEMINI_KEY = ""

# Only drop a YOLO box when Gemini is confident it is a false positive
FALSE_POSITIVE_DROP_CONFIDENCE = 0.75

VERIFICATION_PROMPT = """You are a dental radiologist verifying AI detections on a dental X-ray.

Examine the cropped X-ray region shown in this image. The AI model flagged this area as: "{detected_class}"

Your task is to determine if this detection is CORRECT, MISCLASSIFIED, or a FALSE POSITIVE.

CRITICAL DISTINCTION RULES:
- A **Cavity (Caries)** MUST be located strictly ON or WITHIN a TOOTH CROWN (enamel or dentin). If the cropped region shows alveolar bone, hard palate, maxillary sinus, marrow trabeculae, or background/margins without tooth crown structure, it is a FALSE POSITIVE: you MUST set "verified": false, "corrected_class": null.
- A **Periapical Lesion** is a well-defined radiolucent (dark) area AT or AROUND the ROOT APEX (tip) of a tooth.
- **Normal anatomy** includes: pulp chamber, root canal, periodontal ligament space, mental foramen, maxillary sinus, hard palate, trabecular bone patterns. These are NOT pathology.
- **Restorations/Fillings** appear as bright radiopaque structures on the tooth crown. They are NOT cavities.

Respond in STRICT JSON format:
{{
    "verified": true/false,
    "corrected_class": "Cavity" | "Periapical Lesion" | null,
    "actual_condition": "brief description of what you actually see",
    "confidence_adjustment": 0.0 to 1.0,
    "reasoning": "1-2 sentences explaining your determination"
}}

- "verified": true if the AI detection is an authentic lesion, false if it is bone, normal anatomy, margin artifact, or misclassified
- "corrected_class": the correct class if misclassified; null if it is normal anatomy/bone and should be removed
- "actual_condition": what this area actually shows (e.g. 'alveolar bone', 'hard palate', 'coronal caries')
- "confidence_adjustment": your confidence in this assessment (0.0 to 1.0)
- "reasoning": brief clinical reasoning"""

DISCOVERY_PROMPT = """You are a board-certified dental radiologist reading this dental radiograph.

Locate with HIGH PRECISION:
1. Cavities / Caries: dark radiolucent demineralization strictly on the tooth crown (enamel or dentin). Coordinates MUST strictly tightly enclose the lesion on the tooth crown itself. NEVER place a cavity on bone, palate, root, or background.
2. Periapical Lesions: well-defined dark radiolucency around the root apex (tip) of a non-vital or infected tooth.

Image dimensions are {image_width} pixels wide by {image_height} pixels tall.
Coordinates MUST be center_x, center_y, width, height in exact PIXELS of this image.

Respond in STRICT JSON:
{{
  "findings": [
    {{
      "class": "Cavity" | "Periapical Lesion",
      "x": float,
      "y": float,
      "width": float,
      "height": float,
      "confidence": float,
      "reasoning": "short clinical reason describing tooth location"
    }}
  ]
}}

If no lesions are visible, return {{"findings": []}}."""


class VerificationService:
    """Verifies YOLO detections and discovers missed findings with Gemini Vision."""

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.gemini_api_key or _FALLBACK_GEMINI_KEY
        self.model = self.settings.gemini_model or "gemini-flash-lite-latest"

    async def audit_radiograph(
        self,
        image_path: str,
        is_panoramic: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs a comprehensive global dental radiologist audit on the complete radiograph.
        Determines if the dentition is healthy / caries-free or has genuine pathology.
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"is_healthy": False, "overall_assessment": "Could not read image", "findings": []}

        img_h, img_w = img.shape[:2]
        # Optimize image size for faster network transfer
        scale = min(1.0, 1024.0 / max(img_w, img_h))
        if scale < 1.0:
            target_w, target_h = int(img_w * scale), int(img_h * scale)
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            resized = img

        _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(buffer.tobytes()).decode("ascii")

        prompt = f"""You are a board-certified dental radiologist reviewing this dental radiograph ({img_w}x{img_h} pixels, {'Panoramic OPG' if is_panoramic else 'Intraoral'}).

Examine the entire dentition, crown enamel/dentin, alveolar bone, periodontal ligament spaces, and root apices.
1. Determine whether this radiograph is 100% HEALTHY / CLEAN of active dental caries (cavities) and periapical lesions.
2. Note that normal developmental anatomy (e.g. unerupted/developing third molars / wisdom teeth, dental follicles, alveolar ridges, bone trabeculae) is NOT disease.
3. If there are NO authentic active cavities and NO periapical lesions, you MUST set "is_healthy": true and "findings": [].
4. If there ARE authentic active lesions, specify their class ('Cavity' or 'Periapical Lesion') and coordinates in pixels of the original image (0-{img_w} for x, 0-{img_h} for y).

Respond strictly in JSON:
{{
  "is_healthy": true/false,
  "overall_assessment": "concise clinical radiographic description",
  "has_third_molars": true/false,
  "findings": [
    {{
      "class": "Cavity" | "Periapical Lesion",
      "x": float_center_x,
      "y": float_center_y,
      "width": float_width,
      "height": float_height,
      "confidence": float_0_to_1,
      "reasoning": "brief justification"
    }}
  ]
}}"""
        payload = self._vision_payload(b64_image, prompt, max_tokens=1024)
        result_json = await self._call_gemini(payload)

        if not result_json:
            logger.warning("Global Gemini audit call failed; defaulting to unconfirmed")
            return {"is_healthy": False, "overall_assessment": "Audit call unavailable", "findings": []}

        return {
            "is_healthy": bool(result_json.get("is_healthy", False)),
            "overall_assessment": str(result_json.get("overall_assessment", "")),
            "has_third_molars": bool(result_json.get("has_third_molars", False)),
            "findings": result_json.get("findings", []),
        }

    async def verify_detections(
        self,
        image_path: str,
        predictions: List[Dict[str, Any]],
        audit_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Verify candidates against the global radiologist audit and localized verification.
        - If the global audit confirmed the film is 100% healthy/clean, prune all cavities & PA lesions!
        - Never fail open by retaining unverified low-confidence candidates when an API fails.
        """
        if not predictions:
            return []

        # If the radiologist audit suggests clean dentition, prune low-confidence noise,
        # but retain high-confidence or consensus candidates for crop-level verification
        candidates_to_verify = predictions
        if audit_result and audit_result.get("is_healthy"):
            candidates_to_verify = [
                p for p in predictions
                if p.get("class") == "Impacted Tooth" or p.get("confidence", 0) >= 0.40 or p.get("consensus_confirmed")
            ]
            if len(candidates_to_verify) < len(predictions):
                logger.info(
                    f"Global Health Filter: pruned {len(predictions) - len(candidates_to_verify)} low-confidence noise candidates."
                )
            if not candidates_to_verify:
                return []
            predictions = candidates_to_verify

        img = cv2.imread(image_path)
        if img is None:
            logger.warning("Cannot read image for verification; returning pruned candidates")
            return [p for p in predictions if p.get("confidence", 0) >= 0.75]

        img_h, img_w = img.shape[:2]
        import asyncio

        # Batch-verify or sequentially verify high-value candidates (cap at max 5 to prevent timeouts)
        top_candidates = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)[:6]
        async def verify_task(cand):
            if cand.get("class") == "Impacted Tooth":
                return cand
            try:
                return await self._verify_single_detection(img, img_w, img_h, cand)
            except Exception as e:
                logger.warning(f"Verification error for {cand.get('class')}: {e}")
                if cand.get("confidence", 0) >= 0.75:
                    return cand
                return None

        tasks = [verify_task(c) for c in top_candidates]
        results = await asyncio.gather(*tasks)
        verified = [r for r in results if r is not None]

        logger.info(
            f"Gemini verification: {len(predictions)} → {len(verified)} "
            f"(rejected {len(predictions) - len(verified)} false positives)"
        )
        return verified

    async def discover_findings(
        self,
        image_path: str,
        image_width: Optional[int],
        image_height: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Scan the cropped film for lesions YOLO missed."""
        img = cv2.imread(image_path)
        if img is None:
            logger.warning("Cannot read image for Gemini discovery")
            return []

        img_h, img_w = img.shape[:2]
        width = image_width or img_w
        height = image_height or img_h

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, buffer = cv2.imencode(".jpg", img, encode_params)
        if not success:
            logger.warning("Failed to encode image for Gemini discovery")
            return []

        b64_image = base64.b64encode(buffer.tobytes()).decode("ascii")
        prompt = DISCOVERY_PROMPT.format(image_width=width, image_height=height)
        payload = self._vision_payload(b64_image, prompt, max_tokens=2048)

        result_json = await self._call_gemini(payload)
        if not result_json:
            logger.warning("Gemini discovery returned no parseable JSON")
            return []

        findings = result_json.get("findings") or []
        discovered: List[Dict[str, Any]] = []
        for item in findings:
            pred = self._finding_to_prediction(item, width, height)
            if pred is not None:
                discovered.append(pred)

        logger.info(f"Gemini discovery found {len(discovered)} additional findings")
        return discovered

    async def _verify_single_detection(
        self,
        img: np.ndarray,
        img_w: int,
        img_h: int,
        pred: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Verify a single detection crop with Gemini Vision using contextual crop and target highlight."""
        cx = pred.get("x", 0)
        cy = pred.get("y", 0)
        bw = pred.get("width", 0)
        bh = pred.get("height", 0)

        # Context-aware crop: provide enough surrounding anatomy (tooth crowns and roots)
        min_dim = 160
        half_w = max(int(bw * 1.5), min_dim // 2)
        half_h = max(int(bh * 1.5), min_dim // 2)

        x1 = max(0, int(cx) - half_w)
        y1 = max(0, int(cy) - half_h)
        x2 = min(img_w, int(cx) + half_w)
        y2 = min(img_h, int(cy) + half_h)

        if (x2 - x1) < 25 or (y2 - y1) < 25:
            logger.debug("Detection crop too small for verification; keeping")
            return pred

        crop = img[y1:y2, x1:x2].copy()

        # Draw a visible red target rectangle on the candidate box to direct Gemini's attention
        rx1 = max(0, int(cx - bw / 2) - x1)
        ry1 = max(0, int(cy - bh / 2) - y1)
        rx2 = min(crop.shape[1], int(cx + bw / 2) - x1)
        ry2 = min(crop.shape[0], int(cy + bh / 2) - y1)
        cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

        _, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_crop = base64.b64encode(buffer.tobytes()).decode("ascii")

        detected_class = str(pred.get("class", "Unknown"))
        prompt = (
            f"You are a board-certified dental radiologist verifying an AI detection on a dental radiograph.\n"
            f"The RED bounding box in this image marks a candidate lesion flagged as: '{detected_class}'.\n"
            f"Your clinical task is to determine if this detection is an AUTHENTIC LESION or a FALSE POSITIVE.\n\n"
            f"DIAGNOSTIC CRITERIA:\n"
            f"1. A 'Cavity' (Dental Caries) MUST be located strictly ON or WITHIN a TOOTH CROWN (enamel or dentin). If the red box marks alveolar bone, an edentulous space (missing tooth site), bone trabeculae, hard palate, maxillary sinus, root canal, or empty margins, you MUST set 'verified': false, 'corrected_class': null.\n"
            f"2. A 'Periapical Lesion' MUST be a radiolucency surrounding the ROOT APEX of a tooth.\n"
            f"3. Normal anatomy (pulp chambers, alveolar ridges, bone resorption in edentulous spaces) is NOT pathology.\n\n"
            f"Respond strictly in JSON:\n"
            f'{{"verified": true/false, "corrected_class": "Cavity" | "Periapical Lesion" | null, "actual_condition": "description of structure in red box", "confidence_adjustment": 0.0-1.0, "reasoning": "1-2 sentence clinical assessment"}}'
        )
        payload = self._vision_payload(b64_crop, prompt, max_tokens=1024)
        result_json = await self._call_gemini(payload)

        if result_json is None:
            # Do NOT fail open on low-confidence or unverified candidate boxes!
            metrics = pred.get("patch_metrics", {})
            contrast_drop = metrics.get("contrast_drop", 0)
            if pred.get("confidence", 0) >= 0.80 and (contrast_drop > 0.08 or pred.get("consensus_confirmed")):
                logger.info("Keeping high-confidence consensus detection despite verification timeout")
                return pred
            logger.info(f"Pruned unverified candidate '{detected_class}' due to verification timeout")
            return None

        if isinstance(result_json, list):
            result_json = result_json[0] if (len(result_json) > 0 and isinstance(result_json[0], dict)) else {}

        verified = result_json.get("verified", True)
        corrected_class = result_json.get("corrected_class")
        confidence_adj = float(result_json.get("confidence_adjustment", 0.5) or 0.5)
        reasoning = result_json.get("reasoning", "")
        actual_condition = result_json.get("actual_condition", "")

        pred = dict(pred)

        if not verified and not corrected_class:
            logger.info(
                f"REJECTED false detection '{detected_class}' — "
                f"actual: {actual_condition}. Reason: {reasoning}"
            )
            return None

        if not verified and corrected_class:
            logger.info(
                f"CORRECTED detection '{detected_class}' → '{corrected_class}'. "
                f"Reason: {reasoning}"
            )
            pred["class"] = corrected_class
            pred["class_id"] = self._class_id_for(corrected_class)

        original_conf = pred.get("confidence", 0.5)
        blended_conf = (original_conf * 0.6) + (confidence_adj * 0.4)
        pred["confidence"] = round(blended_conf, 4)
        pred["gemini_verified"] = True
        pred["gemini_reasoning"] = reasoning
        return pred

    async def _call_gemini(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_models = [self.model, "gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]
        candidate_models = list(dict.fromkeys([m for m in raw_models if m]))
        async with httpx.AsyncClient(timeout=35.0) as client:
            for model_name in candidate_models:
                try:
                    url = (
                        f"https://generativelanguage.googleapis.com/v1beta/"
                        f"models/{model_name}:generateContent?key={self.api_key}"
                    )
                    response = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        text = (
                            data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        parsed = self._parse_json_response(text)
                        if parsed:
                            return parsed
                    else:
                        if response.status_code == 429:
                            await asyncio.sleep(2.0)
                        logger.warning(
                            f"Gemini {model_name} returned {response.status_code}: {response.text[:150]}"
                        )
                except Exception as req_err:
                    logger.warning(f"Gemini {model_name} request error: {req_err}")
        return None

    @staticmethod
    def _vision_payload(
        b64_image: str, prompt: str, max_tokens: int = 1024
    ) -> Dict[str, Any]:
        return {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_image,
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_tokens,
                "response_mime_type": "application/json",
            },
        }

    @staticmethod
    def _finding_to_prediction(
        item: Dict[str, Any], image_width: int, image_height: int
    ) -> Optional[Dict[str, Any]]:
        try:
            x = float(item.get("x", 0))
            y = float(item.get("y", 0))
            w = float(item.get("width", 0))
            h = float(item.get("height", 0))
        except (TypeError, ValueError):
            return None

        if w <= 1 and h <= 1 and x <= 1 and y <= 1:
            # Normalized coordinates from the vision model
            x *= image_width
            y *= image_height
            w *= image_width
            h *= image_height

        if w < 4 or h < 4:
            return None
        if x < 0 or y < 0 or x > image_width or y > image_height:
            return None

        cls = str(item.get("class") or "Other Condition").strip()
        class_id = VerificationService._class_id_for(cls)
        confidence = float(item.get("confidence") or 0.55)
        confidence = max(0.25, min(confidence, 0.95))

        return {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "confidence": round(confidence, 4),
            "class": cls,
            "class_id": class_id,
            "detection_id": str(uuid.uuid4()),
            "gemini_verified": True,
            "gemini_reasoning": str(item.get("reasoning") or "Discovered on full-film review"),
        }

    @staticmethod
    def _class_id_for(label: str) -> int:
        raw = (label or "").lower().strip()
        if raw in ("cavity", "caries"):
            return 0
        if raw in ("pa", "periapical", "periapical lesion", "periapical_lesion"):
            return 1
        return 2

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from Gemini response text and normalize lists to dicts."""
        try:
            clean = text.strip()
            if "```json" in clean:
                start = clean.find("```json") + 7
                end = clean.find("```", start)
                clean = clean[start:end].strip() if end != -1 else clean[start:].strip()
            elif "```" in clean:
                start = clean.find("```") + 3
                end = clean.find("```", start)
                clean = clean[start:end].strip() if end != -1 else clean[start:].strip()

            parsed = json.loads(clean)
            if isinstance(parsed, list):
                if len(parsed) > 0 and isinstance(parsed[0], dict):
                    # If this was discovery prompt expecting {"findings": [...]}
                    if any("class" in item or "x" in item for item in parsed):
                        return {"findings": parsed}
                    if "findings" in parsed[0]:
                        return parsed[0]
                    return parsed[0]
                return {}
            return parsed
        except (json.JSONDecodeError, IndexError) as e:
            logger.debug(f"Could not parse Gemini JSON response: {e}")
            return None


def get_verification_service() -> VerificationService:
    return VerificationService()
