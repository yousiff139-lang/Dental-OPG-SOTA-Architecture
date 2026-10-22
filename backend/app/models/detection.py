import uuid
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Detection(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_: str = Field(alias="class")
    class_id: int = 2
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    gemini_verified: Optional[bool] = None
    gemini_reasoning: Optional[str] = None
    location_label: Optional[str] = None
    segment: Optional[str] = None
    stage1_detected: Optional[bool] = True
    stage2_rechecked: Optional[bool] = None
    stage2_notes: Optional[str] = None
    ip_verified: Optional[bool] = None
    ip_score: Optional[float] = None
    ip_metrics: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow", "populate_by_name": True}


class DetectionResponse(BaseModel):
    predictions: List[Detection]
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    scan_mode: Optional[str] = "Multi-Scanner (YOLO 1 + YOLO 2 + CV Overlay)"
    is_panoramic: Optional[bool] = False
    is_healthy: Optional[bool] = None
    radiologist_assessment: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    cv_overlay_url: Optional[str] = None


class DicomMetadata(BaseModel):
    """DICOM metadata fields"""

    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    patient_birth_date: Optional[str] = None
    patient_sex: Optional[str] = None
    study_date: Optional[str] = None
    study_time: Optional[str] = None
    study_description: Optional[str] = None
    series_description: Optional[str] = None
    modality: Optional[str] = None
    manufacturer: Optional[str] = None
    manufacturer_model_name: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    pixel_spacing: Optional[List[float]] = None
    bits_allocated: Optional[int] = None
    bits_stored: Optional[int] = None
    photometric_interpretation: Optional[str] = None
    acquisition_date: Optional[str] = None
    acquisition_time: Optional[str] = None
    institution_name: Optional[str] = None
    referring_physician_name: Optional[str] = None


class ImageInfo(BaseModel):
    """Type-safe model for image conversion information"""

    original_shape: List[int] = Field(description="Original DICOM image dimensions")
    converted_format: str = Field(
        description="Format of the converted image (e.g., 'JPEG')"
    )
    converted_size: List[int] = Field(
        description="Size of converted image [width, height]"
    )
    original_dtype: str = Field(description="Original pixel data type")
    pixel_array_min: float = Field(description="Minimum pixel value in original image")
    pixel_array_max: float = Field(description="Maximum pixel value in original image")
    photometric_interpretation: Optional[str] = Field(
        None, description="DICOM photometric interpretation"
    )
    transfer_syntax: Optional[str] = Field(
        None, description="DICOM transfer syntax UID"
    )


class DicomDetectionResponse(BaseModel):
    """Response model for DICOM file detection"""

    predictions: List[Detection]
    metadata: DicomMetadata
    image_info: ImageInfo
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    scan_mode: Optional[str] = "Multi-Scanner (YOLO 1 + YOLO 2 + CV Overlay)"
    is_panoramic: Optional[bool] = False
    is_healthy: Optional[bool] = None
    radiologist_assessment: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    cv_overlay_url: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
    error_code: str = "VALIDATION_ERROR"


class DiagnosticReport(BaseModel):
    """Generated diagnostic report based on dental detections"""

    report: str = Field(description="The generated diagnostic report text")
    summary: str = Field(description="Brief summary of findings")
    recommendations: List[str] = Field(description="List of treatment recommendations")
    severity_level: str = Field(
        description="Overall severity assessment (low, moderate, high)"
    )
    generated_at: datetime = Field(default_factory=datetime.now)


class DiagnosticReportResponse(BaseModel):
    """Response model for diagnostic report generation"""

    diagnostic_report: DiagnosticReport
    detections_used: List[Detection] = Field(
        description="Detections that were analyzed"
    )
    metadata: Optional[DicomMetadata] = Field(
        None, description="DICOM metadata if available"
    )


class DiagnosticReportRequest(BaseModel):
    """Request model for diagnostic report generation"""

    predictions: List[Detection] = Field(description="Detection results to analyze")
    metadata: Optional[DicomMetadata] = Field(
        None, description="DICOM metadata if available"
    )
    image_info: Optional[ImageInfo] = Field(
        None, description="Image technical information"
    )
