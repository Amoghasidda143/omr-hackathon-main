"""


FastAPI backend for OMR Evaluation System.
Provides REST API endpoints for OMR processing, result management, and system administration.
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import os
import uuid
import shutil
from datetime import datetime
import json

from .database import get_db, create_tables
from .models import (
    Student, ExamSession, OMRSheet, ExamResult, SubjectScore, 
    AnswerKey, ProcessingLog, SystemConfiguration
)
from .schemas import (
    StudentCreate, StudentResponse, ExamSessionCreate, ExamSessionResponse,
    OMRSheetResponse, ExamResultResponse, AnswerKeyCreate, AnswerKeyResponse,
    ProcessingStatus, BatchProcessRequest, BatchProcessResponse
)
from .services import OMRProcessingService, DatabaseService, FileService
from .utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="OMR Evaluation System API",
    description="Automated OMR sheet evaluation and scoring system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
omr_service = OMRProcessingService()
db_service = DatabaseService()
file_service = FileService()

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup."""
    try:
        create_tables()
        logger.info("Database tables created successfully")
        
        # Initialize default configurations
        await initialize_default_configs()
        logger.info("Default configurations initialized")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

async def initialize_default_configs():
    """Initialize default system configurations."""
    db = next(get_db())
    try:
        # Check if configurations already exist
        existing_configs = db.query(SystemConfiguration).count()
        if existing_configs > 0:
            return
        
        # Add default configurations
        default_configs = [
            {
                "key": "max_file_size_mb",
                "value": "50",
                "description": "Maximum file size for uploads in MB",
                "data_type": "integer"
            },
            {
                "key": "supported_formats",
                "value": "jpg,jpeg,png,pdf",
                "description": "Supported file formats for OMR sheets",
                "data_type": "string"
            },
            {
                "key": "processing_timeout_seconds",
                "value": "300",
                "description": "Timeout for OMR processing in seconds",
                "data_type": "integer"
            },
            {
                "key": "bubble_detection_threshold",
                "value": "0.15",
                "description": "Threshold for bubble detection (0-1)",
                "data_type": "float"
            }
        ]
        
        for config in default_configs:
            config_obj = SystemConfiguration(**config)
            db.add(config_obj)
        
        db.commit()
        logger.info("Default configurations added")
        
    except Exception as e:
        logger.error(f"Error initializing default configs: {e}")
        db.rollback()
    finally:
        db.close()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Student management endpoints
@app.post("/api/students", response_model=StudentResponse)
async def create_student(student: StudentCreate, db: Session = Depends(get_db)):
    """Create a new student record."""
    try:
        return await db_service.create_student(db, student)
    except Exception as e:
        logger.error(f"Error creating student: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/students", response_model=List[StudentResponse])
async def get_students(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get list of students."""
    try:
        return await db_service.get_students(db, skip=skip, limit=limit)
    except Exception as e:
        logger.error(f"Error getting students: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/students/{student_id}", response_model=StudentResponse)
async def get_student(student_id: str, db: Session = Depends(get_db)):
    """Get student by ID."""
    try:
        student = await db_service.get_student(db, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        return student
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Exam session management endpoints
@app.post("/api/exam-sessions", response_model=ExamSessionResponse)
async def create_exam_session(session: ExamSessionCreate, db: Session = Depends(get_db)):
    """Create a new exam session."""
    try:
        return await db_service.create_exam_session(db, session)
    except Exception as e:
        logger.error(f"Error creating exam session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/exam-sessions", response_model=List[ExamSessionResponse])
async def get_exam_sessions(active_only: bool = True, db: Session = Depends(get_db)):
    """Get list of exam sessions."""
    try:
        return await db_service.get_exam_sessions(db, active_only=active_only)
    except Exception as e:
        logger.error(f"Error getting exam sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# OMR processing endpoints
@app.post("/api/omr/upload")
async def upload_omr_sheet(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    student_id: str = Form(...),
    exam_session_id: int = Form(...),
    sheet_version: str = Form(...),
    db: Session = Depends(get_db)
):
    """Upload and process an OMR sheet."""
    try:
        # Validate file
        if not file_service.validate_file(file):
            raise HTTPException(status_code=400, detail="Invalid file format or size")
        
        # Save uploaded file
        file_path = await file_service.save_uploaded_file(file)
        
        # Create OMR sheet record
        omr_sheet = await db_service.create_omr_sheet(
            db, student_id, exam_session_id, sheet_version, file_path
        )
        
        # Process OMR sheet in background
        background_tasks.add_task(
            process_omr_sheet_background,
            omr_sheet.id,
            file_path,
            sheet_version,
            student_id
        )
        
        return {
            "message": "OMR sheet uploaded successfully",
            "omr_sheet_id": omr_sheet.id,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading OMR sheet: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_omr_sheet_background(
    omr_sheet_id: int,
    file_path: str,
    sheet_version: str,
    student_id: str
):
    """Background task to process OMR sheet."""
    db = next(get_db())
    try:
        # Update status to processing
        await db_service.update_omr_sheet_status(db, omr_sheet_id, "processing")
        
        # Process the OMR sheet
        result = await omr_service.process_omr_sheet(file_path, sheet_version, student_id)
        
        if result["success"]:
            # Save exam result
            await db_service.save_exam_result(db, omr_sheet_id, result["result"])
            
            # Update OMR sheet status
            await db_service.update_omr_sheet_status(
                db, omr_sheet_id, "completed", 
                processed_image_path=result.get("processed_image_path")
            )
            
            logger.info(f"Successfully processed OMR sheet {omr_sheet_id}")
        else:
            # Update status to failed
            await db_service.update_omr_sheet_status(
                db, omr_sheet_id, "failed", 
                error_message=result["error"]
            )
            
            logger.error(f"Failed to process OMR sheet {omr_sheet_id}: {result['error']}")
            
    except Exception as e:
        logger.error(f"Error in background processing: {e}")
        await db_service.update_omr_sheet_status(
            db, omr_sheet_id, "failed", 
            error_message=str(e)
        )
    finally:
        db.close()

@app.get("/api/omr/{omr_sheet_id}/status")
async def get_omr_status(omr_sheet_id: int, db: Session = Depends(get_db)):
    """Get processing status of an OMR sheet."""
    try:
        omr_sheet = await db_service.get_omr_sheet(db, omr_sheet_id)
        if not omr_sheet:
            raise HTTPException(status_code=404, detail="OMR sheet not found")
        
        return {
            "omr_sheet_id": omr_sheet_id,
            "status": omr_sheet.processing_status,
            "error_message": omr_sheet.error_message,
            "uploaded_at": omr_sheet.uploaded_at.isoformat() if omr_sheet.uploaded_at else None,
            "processed_at": omr_sheet.processed_at.isoformat() if omr_sheet.processed_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting OMR status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/omr/{omr_sheet_id}/result")
async def get_omr_result(omr_sheet_id: int, db: Session = Depends(get_db)):
    """Get exam result for an OMR sheet."""
    try:
        result = await db_service.get_exam_result_by_omr_sheet(db, omr_sheet_id)
        if not result:
            raise HTTPException(status_code=404, detail="Exam result not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting OMR result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch processing endpoints
@app.post("/api/omr/batch-process")
async def batch_process_omr_sheets(
    background_tasks: BackgroundTasks,
    request: BatchProcessRequest,
    db: Session = Depends(get_db)
):
    """Process multiple OMR sheets in batch."""
    try:
        # Validate all files
        for file in request.files:
            if not file_service.validate_file(file):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file: {file.filename}"
                )
        
        # Save all files and create OMR sheet records
        omr_sheet_ids = []
        for i, file in enumerate(request.files):
            file_path = await file_service.save_uploaded_file(file)
            student_id = request.student_ids[i] if i < len(request.student_ids) else f"student_{uuid.uuid4().hex[:8]}"
            
            omr_sheet = await db_service.create_omr_sheet(
                db, student_id, request.exam_session_id, 
                request.sheet_version, file_path
            )
            omr_sheet_ids.append(omr_sheet.id)
        
        # Process all sheets in background
        background_tasks.add_task(
            batch_process_omr_sheets_background,
            omr_sheet_ids,
            request.sheet_version
        )
        
        return {
            "message": f"Batch processing started for {len(omr_sheet_ids)} sheets",
            "omr_sheet_ids": omr_sheet_ids,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def batch_process_omr_sheets_background(
    omr_sheet_ids: List[int],
    sheet_version: str
):
    """Background task for batch processing."""
    db = next(get_db())
    try:
        for omr_sheet_id in omr_sheet_ids:
            omr_sheet = await db_service.get_omr_sheet(db, omr_sheet_id)
            if not omr_sheet:
                continue
            
            # Process each sheet
            await process_omr_sheet_background(
                omr_sheet_id,
                omr_sheet.original_image_path,
                sheet_version,
                omr_sheet.student_id
            )
        
        logger.info(f"Batch processing completed for {len(omr_sheet_ids)} sheets")
        
    except Exception as e:
        logger.error(f"Error in batch processing background task: {e}")
    finally:
        db.close()

# Answer key management endpoints
@app.post("/api/answer-keys", response_model=AnswerKeyResponse)
async def create_answer_key(answer_key: AnswerKeyCreate, db: Session = Depends(get_db)):
    """Create a new answer key."""
    try:
        return await db_service.create_answer_key(db, answer_key)
    except Exception as e:
        logger.error(f"Error creating answer key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/answer-keys", response_model=List[AnswerKeyResponse])
async def get_answer_keys(active_only: bool = True, db: Session = Depends(get_db)):
    """Get list of answer keys."""
    try:
        return await db_service.get_answer_keys(db, active_only=active_only)
    except Exception as e:
        logger.error(f"Error getting answer keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Results and reporting endpoints
@app.get("/api/results/exam-session/{exam_session_id}")
async def get_exam_session_results(
    exam_session_id: int,
    include_details: bool = False,
    db: Session = Depends(get_db)
):
    """Get all results for an exam session."""
    try:
        results = await db_service.get_exam_session_results(
            db, exam_session_id, include_details
        )
        return results
    except Exception as e:
        logger.error(f"Error getting exam session results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/export/{exam_session_id}")
async def export_exam_results(
    exam_session_id: int,
    format: str = "csv",
    db: Session = Depends(get_db)
):
    """Export exam results in specified format."""
    try:
        if format == "csv":
            file_path = await db_service.export_results_csv(db, exam_session_id)
            return FileResponse(
                file_path, 
                media_type="text/csv",
                filename=f"exam_results_{exam_session_id}.csv"
            )
        elif format == "excel":
            file_path = await db_service.export_results_excel(db, exam_session_id)
            return FileResponse(
                file_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=f"exam_results_{exam_session_id}.xlsx"
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System configuration endpoints
@app.get("/api/config")
async def get_system_config(db: Session = Depends(get_db)):
    """Get system configuration."""
    try:
        configs = await db_service.get_system_configurations(db)
        return {config.key: config.value for config in configs}
    except Exception as e:
        logger.error(f"Error getting system config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/config/{key}")
async def update_system_config(
    key: str,
    value: str,
    db: Session = Depends(get_db)
):
    """Update system configuration."""
    try:
        await db_service.update_system_configuration(db, key, value)
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        logger.error(f"Error updating system config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
