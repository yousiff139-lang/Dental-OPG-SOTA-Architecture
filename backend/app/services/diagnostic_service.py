from typing import List, Dict, Any, Optional
import logging
import json
from functools import lru_cache
import httpx

from ..models.detection import Detection, DicomMetadata, DiagnosticReport
from ..core.config import get_settings

logger = logging.getLogger(__name__)


DENTAL_PROMPT = """You are a board-certified dental radiologist with 20+ years of clinical experience. You MUST provide EXTREMELY DETAILED diagnostic reports.

## ABSOLUTE REQUIREMENTS:
1. The "report" field MUST be AT LEAST 500 WORDS - this is NON-NEGOTIABLE
2. You MUST write multiple detailed paragraphs, not one-liners
3. Each detected condition needs its own detailed analysis section
4. Include specific anatomical locations, treatment procedures, and timelines
5. A dentist will use this report - make it clinically comprehensive

## DETECTED CONDITIONS:
{detections}

## PATIENT INFO:
{patient_info}

## EXAMPLE OF WHAT YOUR REPORT SHOULD LOOK LIKE:

For a single cavity detection, the report should be similar to this (adapt based on actual findings):

---
**RADIOGRAPHIC ANALYSIS:**

Upon careful examination of the submitted dental radiograph, AI-assisted detection has identified a carious lesion located in the posterior region. The lesion presents as a well-defined radiolucent area consistent with demineralization of tooth structure. Based on the radiographic appearance, the caries appears to extend beyond the enamel layer and has progressed into the dentin, suggesting a moderate-stage cavity that requires prompt intervention.

**CLINICAL SIGNIFICANCE:**

Dental caries at this stage represents active bacterial infection where Streptococcus mutans and other acidogenic bacteria have penetrated the enamel's protective barrier. The dentin, being more porous and containing dentinal tubules that communicate with the pulp, allows for faster progression of decay. If left untreated for 3-6 months, this lesion could progress to involve the pulp chamber, potentially leading to irreversible pulpitis, periapical abscess formation, or the need for root canal therapy or extraction. Early intervention at this stage offers an excellent prognosis with conservative restorative treatment.

**RECOMMENDED TREATMENT PROTOCOL:**

The primary treatment recommendation is a direct composite resin restoration. The procedure involves local anesthesia (2% lidocaine with 1:100,000 epinephrine or articaine for profound anesthesia), complete caries excavation using round carbide burs and spoon excavators, selective enamel etching with 37% phosphoric acid, application of a universal dental adhesive system, and incremental placement of a nanofilled or microhybrid composite resin in 2mm layers with light curing. Estimated chair time is 45-60 minutes. Alternative treatment options include glass ionomer cement if moisture control is challenging, or amalgam restoration if the cavity is extensive and in a high-stress area.

**PROGNOSIS AND FOLLOW-UP:**

With proper restoration, the prognosis is excellent with expected longevity of 7-15 years for composite restorations. The patient should return in 2 weeks for a post-operative check to evaluate adaptation and occlusion, then resume regular 6-month recall visits. Bitewing radiographs should be taken annually to monitor for recurrent decay at the restoration margins.

**HOME CARE INSTRUCTIONS:**

The patient should use fluoridated toothpaste (1450 ppm fluoride) twice daily with proper brushing technique, floss daily with special attention to the restored area, limit sugar intake especially between meals, and consider using a fluoride mouthwash before bed. Any sensitivity lasting more than 2 weeks or spontaneous pain should prompt an immediate return visit.
---

## YOUR TASK:
Write a similarly detailed report for the ACTUAL detected conditions. Adapt the content based on what was actually found (cavities, periapical lesions, etc.). DO NOT copy the example - create original clinical content.

## JSON RESPONSE FORMAT:
{{
    "report": "[Your detailed 500+ word report here with multiple paragraphs covering: Radiographic Analysis, Clinical Significance, Treatment Protocol, Prognosis, and Home Care. Be specific about each detected condition.]",
    
    "summary": "[3-4 sentences: What was found, how serious it is, what to do first, expected outcome]",
    
    "recommendations": [
        "IMMEDIATE: [Urgent action needed, e.g., 'Schedule restorative appointment within 2 weeks to prevent pulp involvement']",
        "TREATMENT: [Specific procedure, e.g., 'Class II MOD composite restoration with selective enamel etching technique']",
        "MEDICATION: [If needed, e.g., 'Prescribe Ibuprofen 400mg every 6 hours PRN for post-operative discomfort']",
        "PREVENTION: [e.g., 'Apply fluoride varnish to adjacent teeth, recommend prescription-strength fluoride toothpaste']",
        "HOME CARE: [e.g., 'Brush with soft-bristled toothbrush using modified Bass technique, floss daily']",
        "FOLLOW-UP: [e.g., 'Return in 2 weeks for post-op evaluation, then regular 6-month recalls with annual bitewings']"
    ],
    
    "severity_level": "low|moderate|high"
}}

CRITICAL: Your "report" field MUST be detailed and comprehensive like the example above. One-line responses are UNACCEPTABLE and will fail review."""

# Google Gemini API key
GEMINI_API_KEY = "AIzaSyC77FymOjTr-iKo3lV1rD1HiLixYPGEBT0"


class DiagnosticReportService:
    """Service for generating diagnostic reports using Google Gemini"""

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.gemini_api_key or GEMINI_API_KEY
        self.model = self.settings.gemini_model or "gemini-3.5-flash-lite"
        logger.info(f"DiagnosticReportService initialized with Gemini model: {self.model}")

    async def generate_diagnostic_report(
        self,
        detections: List[Detection],
        metadata: Optional[DicomMetadata] = None,
        image_info: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticReport:
        """Generate a diagnostic report from detection results"""

        detection_text = self._format_detections(detections)
        patient_info = self._format_patient_info(metadata)

        prompt = DENTAL_PROMPT.format(
            detections=detection_text,
            patient_info=patient_info,
        )

        try:
            logger.info(f"Calling Gemini API with {len(detections)} detections")
            
            # Combine system prompt and user prompt for Gemini
            full_prompt = f"""You are a board-certified dental radiologist. You MUST provide extremely detailed, comprehensive diagnostic reports. Your reports should be at least 500 words with multiple paragraphs covering radiographic analysis, clinical significance, treatment protocols, prognosis, and home care instructions. One-line responses are unacceptable.

{prompt}"""
            
            candidate_models = [self.model]
            for fallback in ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.8-flash", "gemini-2.5-flash-lite"]:
                if fallback not in candidate_models:
                    candidate_models.append(fallback)

            result = None
            last_error = None
            async with httpx.AsyncClient(timeout=120.0) as client:
                for model_candidate in candidate_models:
                    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_candidate}:generateContent"
                    try:
                        response = await client.post(
                            f"{api_url}?key={self.api_key}",
                            json={
                                "contents": [
                                    {
                                        "parts": [
                                            {"text": full_prompt}
                                        ]
                                    }
                                ],
                                "generationConfig": {
                                    "temperature": 0.3,
                                    "maxOutputTokens": 8192,
                                }
                            },
                            headers={
                                "Content-Type": "application/json"
                            },
                        )
                        
                        logger.info(f"Gemini API ({model_candidate}) response status: {response.status_code}")
                        
                        if response.status_code == 200:
                            result = response.json()
                            logger.info(f"Gemini API ({model_candidate}) response received successfully")
                            break
                        else:
                            last_error = f"{response.status_code} - {response.text}"
                            logger.warning(f"Gemini model {model_candidate} returned error: {last_error}. Trying fallback...")
                    except Exception as req_err:
                        last_error = str(req_err)
                        logger.warning(f"Request error with model {model_candidate}: {req_err}. Trying fallback...")

            if not result:
                raise Exception(f"All Gemini models failed. Last error: {last_error}")
                
            # Extract text from Gemini response format
            if "candidates" not in result or len(result["candidates"]) == 0:
                logger.error(f"No candidates in Gemini response: {result}")
                raise Exception("No candidates in Gemini response")
            
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            logger.info(f"Gemini response text length: {len(text)} chars")
            
            return self._parse_response(text, detections)

        except Exception as e:
            logger.error(f"Failed to generate diagnostic report: {e}", exc_info=True)
            return self._create_fallback_report(detections)

    def _parse_response(self, text: str, detections: List[Detection]) -> DiagnosticReport:
        """Parse the Gemini response into a DiagnosticReport"""
        try:
            clean_text = text.strip()
            
            # Remove markdown code blocks
            if "```json" in clean_text:
                start = clean_text.find("```json") + 7
                end = clean_text.find("```", start)
                clean_text = clean_text[start:end].strip() if end != -1 else clean_text[start:].strip()
            elif "```" in clean_text:
                start = clean_text.find("```") + 3
                end = clean_text.find("```", start)
                clean_text = clean_text[start:end].strip() if end != -1 else clean_text[start:].strip()
            
            logger.info(f"Parsing JSON response: {clean_text[:200]}...")
            
            data = json.loads(clean_text)
            
            # Validate required fields
            report = DiagnosticReport(
                report=data.get("report", "Analysis complete"),
                summary=data.get("summary", "See detailed analysis"),
                recommendations=data.get("recommendations", ["Consult dental professional"]),
                severity_level=data.get("severity_level", "moderate"),
            )
            
            logger.info(f"Successfully parsed diagnostic report with {len(report.recommendations)} recommendations")
            return report
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}, text: {text[:500]}")
            # If JSON parsing fails, try to use the text as-is
            return DiagnosticReport(
                report=text,
                summary="AI analysis completed - see detailed report",
                recommendations=["Review the detailed analysis above", "Consult dental professional"],
                severity_level="moderate",
            )
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return self._create_fallback_report(detections)

    def _create_fallback_report(self, detections: List[Detection]) -> DiagnosticReport:
        """Create a detailed fallback report when API fails"""
        if not detections:
            return DiagnosticReport(
                report="**RADIOGRAPHIC ANALYSIS:**\n\nNo significant pathological findings were detected in this dental radiograph.\n\nThe analyzed areas appear within normal radiographic limits. No evidence of carious lesions, periapical pathology, or other abnormalities was identified.\n\n**RECOMMENDATION:**\n\nContinue with routine preventive dental care and regular check-ups.",
                summary="No abnormalities detected.\n\nRoutine dental care recommended.",
                recommendations=[
                    "Continue regular dental checkups every 6 months",
                    "Maintain proper brushing technique twice daily",
                    "Use dental floss daily",
                    "Consider fluoride mouthwash for cavity prevention"
                ],
                severity_level="low",
            )
        
        # Build detailed fallback based on detected conditions
        conditions = [d.class_ for d in detections]
        condition_counts = {}
        for c in conditions:
            condition_counts[c] = condition_counts.get(c, 0) + 1
        
        report_parts = [f"**RADIOGRAPHIC ANALYSIS:**\n\nAI analysis has identified {len(detections)} area(s) of concern in this dental radiograph."]
        recommendations = []
        severity = "moderate"
        
        for condition, count in condition_counts.items():
            cond_lower = condition.lower()
            
            if "cavity" in cond_lower or "caries" in cond_lower:
                report_parts.append(f"**DENTAL CARIES ({count} detection{'s' if count > 1 else ''}):**\n\nDental caries represent demineralization of tooth structure caused by bacterial acid production. The detected lesion(s) indicate areas where enamel and potentially dentin have been compromised by the carious process.\n\nEarly intervention with restorative treatment is essential to prevent progression to pulp involvement, which could necessitate more extensive treatment such as root canal therapy.")
                recommendations.extend([
                    f"IMMEDIATE: Schedule dental appointment for {count} detected cavit{'y' if count == 1 else 'ies'}",
                    "TREATMENT: Composite or amalgam filling recommended",
                    "PREVENTION: Reduce sugar intake and acidic beverages",
                    "HOME CARE: Use fluoride toothpaste and consider prescription-strength fluoride gel"
                ])
            elif "periapical" in cond_lower or "lesion" in cond_lower or "pa" in cond_lower:
                severity = "high"
                report_parts.append(f"**PERIAPICAL PATHOLOGY ({count} detection{'s' if count > 1 else ''}):**\n\nPeriapical lesions indicate infection or inflammation at the tooth root apex, often resulting from pulp necrosis. This finding suggests bacterial invasion has extended beyond the tooth structure into the periapical tissues.\n\nThis requires prompt endodontic evaluation to prevent further spread of infection and potential complications.")
                recommendations.extend([
                    "URGENT: Schedule endodontic (root canal) evaluation within 1-2 weeks",
                    "WARNING: Monitor for increased pain, swelling, or fever",
                    "MEDICATION: Antibiotics may be prescribed if active infection is present",
                    "FOLLOW-UP: Seek immediate care if symptoms worsen"
                ])
            elif "other" in cond_lower:
                report_parts.append(f"**OTHER DENTAL CONDITION ({count} detection{'s' if count > 1 else ''}):**\n\nThe AI analysis has identified an area of interest that does not fit the typical pattern of a cavity or periapical lesion. This may include conditions such as calculus deposits, bone loss from periodontal disease, root resorption, hypercementosis, or other dental anomalies.\n\nProfessional clinical evaluation is recommended to provide a definitive diagnosis and determine if treatment is necessary.")
                recommendations.extend([
                    f"EVALUATION: Have your dentist examine the {count} flagged area{'s' if count > 1 else ''} at your next visit",
                    "NOTE: This finding may be a normal anatomical variant or a condition requiring monitoring"
                ])
            elif "impacted" in cond_lower or "wisdom" in cond_lower:
                if severity != "high":
                    severity = "low"
                report_parts.append(
                    f"**DEVELOPMENTAL / IMPACTED TOOTH ({count} finding{'s' if count > 1 else ''}):**\n\n"
                    "Radiographic examination reveals developing or unerupted tooth crypts (typically third molars / wisdom teeth). "
                    "In adolescent and young adult patients, third molar buds frequently exhibit normal radiolucent follicular borders. "
                    "This is an anatomical developmental finding rather than active disease.\n\n"
                    "Routine panoramic monitoring is recommended to evaluate eruption space and angulation over time."
                )
                recommendations.extend([
                    f"MONITORING: Track eruption progress of {count} third molar/wisdom tooth bud{'s' if count > 1 else ''} on periodic panoramic recalls",
                    "PREVENTION: Maintain meticulous oral hygiene in the posterior retromolar regions"
                ])
            else:
                report_parts.append(f"**{condition.upper()} ({count} detection{'s' if count > 1 else ''}):**\n\nThis finding requires professional dental evaluation to determine appropriate treatment. The AI has identified this area as requiring clinical attention.")
                recommendations.append(f"Consult dental professional regarding {condition} finding")
        
        report_parts.append("**PROFESSIONAL EVALUATION RECOMMENDED:**\n\nPlease consult with a qualified dental professional for a comprehensive clinical examination and definitive diagnosis. This AI-generated report is intended for informational purposes only.")
        
        recommendations.append("FOLLOW-UP: Schedule appointment as directed by your dental professional")
        
        # Build formatted summary with line breaks
        summary_parts = [
            f"Detected {len(detections)} condition{'s' if len(detections) > 1 else ''} requiring professional evaluation.",
            "",
            "Urgent attention recommended." if severity == "high" else "Schedule dental appointment soon."
        ]
        
        return DiagnosticReport(
            report="\n\n".join(report_parts),
            summary="\n".join(summary_parts),
            recommendations=recommendations[:6],  # Limit to 6 recommendations
            severity_level=severity,
        )

    def _format_detections(self, detections: List[Detection]) -> str:
        if not detections:
            return "No pathological findings detected in this X-ray."

        formatted = []
        for i, detection in enumerate(detections, 1):
            conf_pct = detection.confidence * 100
            conf_level = "HIGH" if conf_pct > 80 else "MODERATE" if conf_pct > 50 else "LOW"
            
            formatted.append(
                f"FINDING #{i}:\n"
                f"  - Condition: {detection.class_}\n"
                f"  - AI Confidence: {conf_pct:.1f}% ({conf_level})\n"
                f"  - Location: Position ({detection.x}, {detection.y}) in image\n"
                f"  - Approximate Size: {detection.width} x {detection.height} pixels"
            )
        return "\n\n".join(formatted)

    def _format_patient_info(self, metadata: Optional[DicomMetadata]) -> str:
        if not metadata:
            return "Patient demographics not available from image metadata."
        
        parts = []
        if metadata.patient_id:
            parts.append(f"Patient ID: {metadata.patient_id}")
        if metadata.patient_sex:
            parts.append(f"Sex: {metadata.patient_sex}")
        if metadata.study_date:
            parts.append(f"Study Date: {metadata.study_date}")
        if metadata.modality:
            parts.append(f"Modality: {metadata.modality}")
            
        return "\n".join(parts) if parts else "Limited patient information available."


@lru_cache()
def get_diagnostic_report_service() -> DiagnosticReportService:
    return DiagnosticReportService()
