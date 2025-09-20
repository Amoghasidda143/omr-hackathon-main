"""
Pydantic schemas for OMR Evaluation System API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """OMR processing status enum."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Student schemas
class StudentBase(BaseModel):
    """Base student schema."""
    student_id: str = Field(..., description="Unique student identifier")
    name: Optional[str] = Field(None, description="Student name")
    email: Optional[str] = Field(None, description="Student email")
    phone: Optional[str] = Field(None, description="Student phone number")


class StudentCreate(StudentBase):
    """Schema for creating a student."""
    pass


class StudentResponse(StudentBase):
    """Schema for student response."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Exam session schemas
class ExamSessionBase(BaseModel):
    """Base exam session schema."""
    session_name: str = Field(..., description="Name of the exam session")
    sheet_version: str = Field(..., description="OMR sheet version")
    total_questions: int = Field(100, description="Total number of questions")
    questions_per_subject: int = Field(20, description="Questions per subject")
    subjects: List[str] = Field(..., description="List of subjects")
    answer_key_version: str = Field(..., description="Answer key version")


class ExamSessionCreate(ExamSessionBase):
    """Schema for creating an exam session."""
    pass


class ExamSessionResponse(ExamSessionBase):
    """Schema for exam session response."""
    id: int
    created_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True


# OMR sheet schemas
class OMRSheetBase(BaseModel):
    """Base OMR sheet schema."""
    student_id: str
    exam_session_id: int
    sheet_version: str
    original_image_path: str


class OMRSheetResponse(OMRSheetBase):
    """Schema for OMR sheet response."""
    id: int
    processing_status: ProcessingStatus
    error_message: Optional[str] = None
    orientation_correction: int = 0
    bubbles_detected: int = 0
    questions_processed: int = 0
    processed_image_path: Optional[str] = None
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Subject score schemas
class SubjectScoreBase(BaseModel):
    """Base subject score schema."""
    subject_name: str
    correct_answers: int
    total_questions: int
    score: float
    percentage: float


class SubjectScoreResponse(SubjectScoreBase):
    """Schema for subject score response."""
    id: int
    exam_result_id: int
    student_id: str
    exam_session_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Exam result schemas
class ExamResultBase(BaseModel):
    """Base exam result schema."""
    student_id: str
    exam_session_id: int
    omr_sheet_id: int
    total_score: float
    total_percentage: float
    max_possible_score: float = 100.0
    subject_scores: List[SubjectScoreBase]
    student_answers: List[List[int]]
    correct_answers: List[List[int]]
    evaluation_method: str = "automated_omr"
    confidence_score: Optional[float] = None
    processing_time_seconds: Optional[float] = None


class ExamResultResponse(ExamResultBase):
    """Schema for exam result response."""
    id: int
    evaluated_at: datetime
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Answer key schemas
class AnswerKeyBase(BaseModel):
    """Base answer key schema."""
    version: str = Field(..., description="Answer key version")
    name: str = Field(..., description="Answer key name")
    description: Optional[str] = Field(None, description="Answer key description")
    answer_data: Dict[str, Any] = Field(..., description="Answer key data")
    total_questions: int = Field(..., description="Total number of questions")
    subjects: List[str] = Field(..., description="List of subjects")
    questions_per_subject: int = Field(20, description="Questions per subject")


class AnswerKeyCreate(AnswerKeyBase):
    """Schema for creating an answer key."""
    pass


class AnswerKeyResponse(AnswerKeyBase):
    """Schema for answer key response."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_active: bool = True

    class Config:
        from_attributes = True


# Batch processing schemas
class BatchProcessRequest(BaseModel):
    """Schema for batch processing request."""
    files: List[Any] = Field(..., description="List of uploaded files")
    student_ids: List[str] = Field(..., description="List of student IDs")
    exam_session_id: int = Field(..., description="Exam session ID")
    sheet_version: str = Field(..., description="OMR sheet version")

    @validator('student_ids')
    def validate_student_ids_length(cls, v, values):
        """Validate that student_ids length matches files length."""
        if 'files' in values and len(v) != len(values['files']):
            raise ValueError('Number of student IDs must match number of files')
        return v


class BatchProcessResponse(BaseModel):
    """Schema for batch processing response."""
    message: str
    omr_sheet_ids: List[int]
    status: ProcessingStatus


# Processing log schemas
class ProcessingLogBase(BaseModel):
    """Base processing log schema."""
    omr_sheet_id: int
    student_id: str
    log_level: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ProcessingLogResponse(ProcessingLogBase):
    """Schema for processing log response."""
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# System configuration schemas
class SystemConfigurationBase(BaseModel):
    """Base system configuration schema."""
    key: str = Field(..., description="Configuration key")
    value: str = Field(..., description="Configuration value")
    description: Optional[str] = Field(None, description="Configuration description")
    data_type: str = Field("string", description="Data type")


class SystemConfigurationResponse(SystemConfigurationBase):
    """Schema for system configuration response."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Statistics and reporting schemas
class ExamSessionStats(BaseModel):
    """Schema for exam session statistics."""
    exam_session_id: int
    total_students: int
    processed_sheets: int
    pending_sheets: int
    failed_sheets: int
    average_score: float
    highest_score: float
    lowest_score: float
    completion_rate: float


class SubjectStats(BaseModel):
    """Schema for subject statistics."""
    subject_name: str
    total_questions: int
    average_score: float
    highest_score: float
    lowest_score: float
    correct_answer_rate: float


class ProcessingStats(BaseModel):
    """Schema for processing statistics."""
    total_processed: int
    successful: int
    failed: int
    success_rate: float
    average_processing_time: float
    total_processing_time: float


# Error schemas
class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# File upload schemas
class FileUploadResponse(BaseModel):
    """Schema for file upload response."""
    filename: str
    file_path: str
    file_size: int
    content_type: str
    uploaded_at: datetime


# Validation schemas
class ValidationResult(BaseModel):
    """Schema for validation results."""
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None


# Export schemas
class ExportRequest(BaseModel):
    """Schema for export requests."""
    exam_session_id: int
    format: str = Field("csv", description="Export format (csv, excel)")
    include_details: bool = Field(False, description="Include detailed information")
    subject_filter: Optional[List[str]] = Field(None, description="Filter by subjects")


class ExportResponse(BaseModel):
    """Schema for export responses."""
    file_path: str
    filename: str
    file_size: int
    export_format: str
    record_count: int
    created_at: datetime