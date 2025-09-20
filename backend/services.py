"""
Service classes for OMR Evaluation System.
"""

import os
import uuid
import shutil
import json
import csv
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from .models import (
    Student, ExamSession, OMRSheet, ExamResult, SubjectScore, 
    AnswerKey, ProcessingLog, SystemConfiguration
)
from .schemas import (
    StudentCreate, StudentResponse, ExamSessionCreate, ExamSessionResponse,
    OMRSheetResponse, ExamResultResponse, AnswerKeyCreate, AnswerKeyResponse,
    ProcessingLogResponse, SystemConfigurationResponse, ExamSessionStats,
    SubjectStats, ProcessingStats, ValidationResult, ExportRequest
)
from ..omr_processor.omr_processor import OMRProcessor


class OMRProcessingService:
    """Service for OMR processing operations."""
    
    def __init__(self):
        self.processor = OMRProcessor()
    
    async def process_omr_sheet(self, 
                               file_path: str, 
                               sheet_version: str, 
                               student_id: str) -> Dict[str, Any]:
        """
        Process a single OMR sheet.
        
        Args:
            file_path: Path to the OMR sheet image
            sheet_version: Version of the OMR sheet
            student_id: Student identifier
            
        Returns:
            Processing result dictionary
        """
        try:
            result = self.processor.process_omr_sheet(file_path, sheet_version, student_id)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "student_id": student_id,
                "sheet_version": sheet_version
            }
    
    async def batch_process_omr_sheets(self, 
                                      file_paths: List[str], 
                                      sheet_version: str) -> List[Dict[str, Any]]:
        """
        Process multiple OMR sheets in batch.
        
        Args:
            file_paths: List of image file paths
            sheet_version: Version of the OMR sheets
            
        Returns:
            List of processing results
        """
        try:
            results = self.processor.batch_process(file_paths, sheet_version)
            return results
        except Exception as e:
            return [{"success": False, "error": str(e)} for _ in file_paths]
    
    async def validate_image(self, file_path: str) -> ValidationResult:
        """
        Validate if image is suitable for OMR processing.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Validation result
        """
        try:
            is_valid, message = self.processor.validate_image(file_path)
            return ValidationResult(
                is_valid=is_valid,
                message=message
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Validation error: {str(e)}"
            )
    
    def get_available_answer_versions(self) -> List[str]:
        """Get list of available answer key versions."""
        return self.processor.get_available_answer_versions()
    
    def add_answer_key(self, version: str, answer_key: Dict[str, Any]):
        """Add a new answer key."""
        self.processor.add_answer_key(version, answer_key)


class DatabaseService:
    """Service for database operations."""
    
    async def create_student(self, db: Session, student: StudentCreate) -> StudentResponse:
        """Create a new student record."""
        db_student = Student(**student.dict())
        db.add(db_student)
        db.commit()
        db.refresh(db_student)
        return StudentResponse.from_orm(db_student)
    
    async def get_students(self, db: Session, skip: int = 0, limit: int = 100) -> List[StudentResponse]:
        """Get list of students."""
        students = db.query(Student).offset(skip).limit(limit).all()
        return [StudentResponse.from_orm(student) for student in students]
    
    async def get_student(self, db: Session, student_id: str) -> Optional[StudentResponse]:
        """Get student by ID."""
        student = db.query(Student).filter(Student.student_id == student_id).first()
        return StudentResponse.from_orm(student) if student else None
    
    async def create_exam_session(self, db: Session, session: ExamSessionCreate) -> ExamSessionResponse:
        """Create a new exam session."""
        db_session = ExamSession(**session.dict())
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        return ExamSessionResponse.from_orm(db_session)
    
    async def get_exam_sessions(self, db: Session, active_only: bool = True) -> List[ExamSessionResponse]:
        """Get list of exam sessions."""
        query = db.query(ExamSession)
        if active_only:
            query = query.filter(ExamSession.is_active == True)
        sessions = query.all()
        return [ExamSessionResponse.from_orm(session) for session in sessions]
    
    async def create_omr_sheet(self, 
                              db: Session, 
                              student_id: str, 
                              exam_session_id: int, 
                              sheet_version: str, 
                              file_path: str) -> OMRSheetResponse:
        """Create a new OMR sheet record."""
        omr_sheet = OMRSheet(
            student_id=student_id,
            exam_session_id=exam_session_id,
            sheet_version=sheet_version,
            original_image_path=file_path
        )
        db.add(omr_sheet)
        db.commit()
        db.refresh(omr_sheet)
        return OMRSheetResponse.from_orm(omr_sheet)
    
    async def get_omr_sheet(self, db: Session, omr_sheet_id: int) -> Optional[OMRSheetResponse]:
        """Get OMR sheet by ID."""
        omr_sheet = db.query(OMRSheet).filter(OMRSheet.id == omr_sheet_id).first()
        return OMRSheetResponse.from_orm(omr_sheet) if omr_sheet else None
    
    async def update_omr_sheet_status(self, 
                                     db: Session, 
                                     omr_sheet_id: int, 
                                     status: str,
                                     error_message: Optional[str] = None,
                                     processed_image_path: Optional[str] = None) -> bool:
        """Update OMR sheet processing status."""
        omr_sheet = db.query(OMRSheet).filter(OMRSheet.id == omr_sheet_id).first()
        if not omr_sheet:
            return False
        
        omr_sheet.processing_status = status
        if error_message:
            omr_sheet.error_message = error_message
        if processed_image_path:
            omr_sheet.processed_image_path = processed_image_path
        if status == "completed":
            omr_sheet.processed_at = datetime.now()
        
        db.commit()
        return True
    
    async def save_exam_result(self, db: Session, omr_sheet_id: int, result_data: Dict[str, Any]) -> ExamResultResponse:
        """Save exam result to database."""
        # Get OMR sheet info
        omr_sheet = db.query(OMRSheet).filter(OMRSheet.id == omr_sheet_id).first()
        if not omr_sheet:
            raise ValueError("OMR sheet not found")
        
        # Create exam result
        exam_result = ExamResult(
            student_id=omr_sheet.student_id,
            exam_session_id=omr_sheet.exam_session_id,
            omr_sheet_id=omr_sheet_id,
            total_score=result_data["total_score"],
            total_percentage=result_data["total_percentage"],
            subject_scores=result_data["subject_scores"],
            student_answers=result_data["answers"],
            correct_answers=result_data["correct_answers"],
            evaluation_method=result_data.get("evaluation_metadata", {}).get("evaluation_method", "automated_omr"),
            processing_time_seconds=result_data.get("processing_metadata", {}).get("processing_time_seconds")
        )
        db.add(exam_result)
        db.commit()
        db.refresh(exam_result)
        
        # Create subject score records
        for subject_score_data in result_data["subject_scores"]:
            subject_score = SubjectScore(
                exam_result_id=exam_result.id,
                student_id=omr_sheet.student_id,
                exam_session_id=omr_sheet.exam_session_id,
                subject_name=subject_score_data["subject"],
                correct_answers=subject_score_data["correct"],
                total_questions=subject_score_data["total"],
                score=subject_score_data["score"],
                percentage=subject_score_data["percentage"]
            )
            db.add(subject_score)
        
        db.commit()
        return ExamResultResponse.from_orm(exam_result)
    
    async def get_exam_result_by_omr_sheet(self, db: Session, omr_sheet_id: int) -> Optional[ExamResultResponse]:
        """Get exam result by OMR sheet ID."""
        result = db.query(ExamResult).filter(ExamResult.omr_sheet_id == omr_sheet_id).first()
        return ExamResultResponse.from_orm(result) if result else None
    
    async def get_exam_session_results(self, 
                                      db: Session, 
                                      exam_session_id: int, 
                                      include_details: bool = False) -> List[Dict[str, Any]]:
        """Get all results for an exam session."""
        results = db.query(ExamResult).filter(ExamResult.exam_session_id == exam_session_id).all()
        
        if not include_details:
            return [
                {
                    "student_id": result.student_id,
                    "total_score": result.total_score,
                    "total_percentage": result.total_percentage,
                    "evaluated_at": result.evaluated_at.isoformat()
                }
                for result in results
            ]
        else:
            return [ExamResultResponse.from_orm(result).dict() for result in results]
    
    async def create_answer_key(self, db: Session, answer_key: AnswerKeyCreate) -> AnswerKeyResponse:
        """Create a new answer key."""
        db_answer_key = AnswerKey(**answer_key.dict())
        db.add(db_answer_key)
        db.commit()
        db.refresh(db_answer_key)
        return AnswerKeyResponse.from_orm(db_answer_key)
    
    async def get_answer_keys(self, db: Session, active_only: bool = True) -> List[AnswerKeyResponse]:
        """Get list of answer keys."""
        query = db.query(AnswerKey)
        if active_only:
            query = query.filter(AnswerKey.is_active == True)
        answer_keys = query.all()
        return [AnswerKeyResponse.from_orm(key) for key in answer_keys]
    
    async def get_system_configurations(self, db: Session) -> List[SystemConfigurationResponse]:
        """Get system configurations."""
        configs = db.query(SystemConfiguration).all()
        return [SystemConfigurationResponse.from_orm(config) for config in configs]
    
    async def update_system_configuration(self, db: Session, key: str, value: str) -> bool:
        """Update system configuration."""
        config = db.query(SystemConfiguration).filter(SystemConfiguration.key == key).first()
        if not config:
            return False
        
        config.value = value
        config.updated_at = datetime.now()
        db.commit()
        return True
    
    async def export_results_csv(self, db: Session, exam_session_id: int) -> str:
        """Export exam results as CSV."""
        results = db.query(ExamResult).filter(ExamResult.exam_session_id == exam_session_id).all()
        
        # Create results directory if it doesn't exist
        results_dir = "results/exports"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exam_results_{exam_session_id}_{timestamp}.csv"
        file_path = os.path.join(results_dir, filename)
        
        # Write CSV
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'student_id', 'total_score', 'total_percentage', 'evaluated_at'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'student_id': result.student_id,
                    'total_score': result.total_score,
                    'total_percentage': result.total_percentage,
                    'evaluated_at': result.evaluated_at.isoformat()
                })
        
        return file_path
    
    async def export_results_excel(self, db: Session, exam_session_id: int) -> str:
        """Export exam results as Excel."""
        results = db.query(ExamResult).filter(ExamResult.exam_session_id == exam_session_id).all()
        
        # Create results directory if it doesn't exist
        results_dir = "results/exports"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exam_results_{exam_session_id}_{timestamp}.xlsx"
        file_path = os.path.join(results_dir, filename)
        
        # Prepare data
        data = []
        for result in results:
            data.append({
                'student_id': result.student_id,
                'total_score': result.total_score,
                'total_percentage': result.total_percentage,
                'evaluated_at': result.evaluated_at.isoformat()
            })
        
        # Write Excel
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)
        
        return file_path


class FileService:
    """Service for file operations."""
    
    def __init__(self, uploads_dir: str = "uploads"):
        self.uploads_dir = uploads_dir
        os.makedirs(uploads_dir, exist_ok=True)
    
    def validate_file(self, file) -> bool:
        """Validate uploaded file."""
        # Check file size (50MB max)
        max_size = 50 * 1024 * 1024  # 50MB
        if hasattr(file, 'size') and file.size > max_size:
            return False
        
        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.pdf'}
        if hasattr(file, 'filename'):
            file_ext = os.path.splitext(file.filename.lower())[1]
            if file_ext not in allowed_extensions:
                return False
        
        return True
    
    async def save_uploaded_file(self, file) -> str:
        """Save uploaded file and return file path."""
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(self.uploads_dir, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return file_path
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from filesystem."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception:
            return False
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file information."""
        if not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        return {
            "file_path": file_path,
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }