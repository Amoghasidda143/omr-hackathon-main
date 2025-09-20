"""
Validation utilities for OMR Evaluation System.
Handles validation of inputs, files, and data.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import json
from PIL import Image
import magic


class ValidationManager:
    """Manages validation functionality for the OMR system."""
    
    def __init__(self):
        """Initialize validation manager."""
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.supported_document_formats = {'.pdf'}
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.min_image_dimensions = (100, 100)
        self.max_image_dimensions = (5000, 5000)
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate file for OMR processing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return False, f"File too large ({file_size / (1024*1024):.1f}MB). Maximum allowed: {self.max_file_size / (1024*1024):.1f}MB"
            
            if file_size == 0:
                return False, "File is empty"
            
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_image_formats and file_ext not in self.supported_document_formats:
                return False, f"Unsupported file format: {file_ext}. Supported formats: {', '.join(self.supported_image_formats | self.supported_document_formats)}"
            
            # Check MIME type
            try:
                mime_type = magic.from_file(file_path, mime=True)
                if file_ext in self.supported_image_formats:
                    if not mime_type.startswith('image/'):
                        return False, f"File appears to be {mime_type}, not an image"
                elif file_ext == '.pdf':
                    if mime_type != 'application/pdf':
                        return False, f"File appears to be {mime_type}, not a PDF"
            except Exception:
                pass  # MIME type detection failed, continue with other checks
            
            # Validate image content
            if file_ext in self.supported_image_formats:
                return self._validate_image_content(file_path)
            elif file_ext == '.pdf':
                return self._validate_pdf_content(file_path)
            
            return True, "File is valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_image_content(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate image content.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to load image with OpenCV
            image = cv2.imread(file_path)
            if image is None:
                return False, "Could not load image with OpenCV"
            
            # Check image dimensions
            height, width = image.shape[:2]
            if width < self.min_image_dimensions[0] or height < self.min_image_dimensions[1]:
                return False, f"Image too small ({width}x{height}). Minimum size: {self.min_image_dimensions[0]}x{self.min_image_dimensions[1]}"
            
            if width > self.max_image_dimensions[0] or height > self.max_image_dimensions[1]:
                return False, f"Image too large ({width}x{height}). Maximum size: {self.max_image_dimensions[0]}x{self.max_image_dimensions[1]}"
            
            # Check if image has content (not all black or white)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            if mean_intensity < 10 or mean_intensity > 245:
                return False, "Image appears to be too dark or too bright (likely blank)"
            
            # Check for sufficient contrast
            std_intensity = np.std(gray)
            if std_intensity < 10:
                return False, "Image has insufficient contrast (likely blank or uniform)"
            
            return True, "Image content is valid"
            
        except Exception as e:
            return False, f"Image validation error: {str(e)}"
    
    def _validate_pdf_content(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate PDF content.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic PDF validation - check if it can be opened
            import PyMuPDF
            doc = PyMuPDF.open(file_path)
            if doc.page_count == 0:
                return False, "PDF has no pages"
            
            # Check if first page has content
            page = doc[0]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            
            # Convert to numpy array for analysis
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return False, "Could not extract image from PDF"
            
            # Apply same image validation
            return self._validate_image_content_from_array(image)
            
        except Exception as e:
            return False, f"PDF validation error: {str(e)}"
    
    def _validate_image_content_from_array(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image content from numpy array.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check image dimensions
            height, width = image.shape[:2]
            if width < self.min_image_dimensions[0] or height < self.min_image_dimensions[1]:
                return False, f"Image too small ({width}x{height}). Minimum size: {self.min_image_dimensions[0]}x{self.min_image_dimensions[1]}"
            
            if width > self.max_image_dimensions[0] or height > self.max_image_dimensions[1]:
                return False, f"Image too large ({width}x{height}). Maximum size: {self.max_image_dimensions[0]}x{self.max_image_dimensions[1]}"
            
            # Check if image has content
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            if mean_intensity < 10 or mean_intensity > 245:
                return False, "Image appears to be too dark or too bright (likely blank)"
            
            # Check for sufficient contrast
            std_intensity = np.std(gray)
            if std_intensity < 10:
                return False, "Image has insufficient contrast (likely blank or uniform)"
            
            return True, "Image content is valid"
            
        except Exception as e:
            return False, f"Image validation error: {str(e)}"
    
    def validate_student_id(self, student_id: str) -> Tuple[bool, str]:
        """
        Validate student ID format.
        
        Args:
            student_id: Student ID string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not student_id:
            return False, "Student ID cannot be empty"
        
        if len(student_id) < 3:
            return False, "Student ID must be at least 3 characters long"
        
        if len(student_id) > 50:
            return False, "Student ID must be less than 50 characters"
        
        # Check for valid characters (alphanumeric, underscore, hyphen)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', student_id):
            return False, "Student ID can only contain letters, numbers, underscores, and hyphens"
        
        return True, "Student ID is valid"
    
    def validate_sheet_version(self, sheet_version: str) -> Tuple[bool, str]:
        """
        Validate sheet version format.
        
        Args:
            sheet_version: Sheet version string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sheet_version:
            return False, "Sheet version cannot be empty"
        
        if len(sheet_version) > 20:
            return False, "Sheet version must be less than 20 characters"
        
        # Check for valid characters
        import re
        if not re.match(r'^[a-zA-Z0-9_.-]+$', sheet_version):
            return False, "Sheet version can only contain letters, numbers, underscores, dots, and hyphens"
        
        return True, "Sheet version is valid"
    
    def validate_answer_key(self, answer_key_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate answer key data structure.
        
        Args:
            answer_key_data: Answer key dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check required fields
            required_fields = ['version', 'subjects', 'total_questions']
            for field in required_fields:
                if field not in answer_key_data:
                    return False, f"Missing required field: {field}"
            
            # Validate version
            version = answer_key_data['version']
            is_valid, message = self.validate_sheet_version(version)
            if not is_valid:
                return False, f"Invalid version: {message}"
            
            # Validate subjects
            subjects = answer_key_data['subjects']
            if not isinstance(subjects, dict):
                return False, "Subjects must be a dictionary"
            
            if not subjects:
                return False, "At least one subject is required"
            
            # Validate each subject
            for subject_name, subject_data in subjects.items():
                if not isinstance(subject_data, dict):
                    return False, f"Subject '{subject_name}' data must be a dictionary"
                
                if 'questions' not in subject_data or 'answers' not in subject_data:
                    return False, f"Subject '{subject_name}' missing 'questions' or 'answers' field"
                
                questions = subject_data['questions']
                answers = subject_data['answers']
                
                if not isinstance(questions, list) or not isinstance(answers, list):
                    return False, f"Subject '{subject_name}' questions and answers must be lists"
                
                if len(questions) != len(answers):
                    return False, f"Subject '{subject_name}' questions and answers count mismatch"
                
                # Validate answer format
                valid_answers = {'A', 'B', 'C', 'D', 'E'}
                for i, answer in enumerate(answers):
                    if answer not in valid_answers:
                        return False, f"Subject '{subject_name}', question {i+1}: Invalid answer '{answer}'. Must be A, B, C, D, or E"
            
            # Validate total questions
            total_questions = answer_key_data['total_questions']
            if not isinstance(total_questions, int) or total_questions <= 0:
                return False, "Total questions must be a positive integer"
            
            # Check if total questions matches sum of subject questions
            total_subject_questions = sum(len(subject_data['questions']) for subject_data in subjects.values())
            if total_subject_questions != total_questions:
                return False, f"Total questions ({total_questions}) does not match sum of subject questions ({total_subject_questions})"
            
            return True, "Answer key is valid"
            
        except Exception as e:
            return False, f"Answer key validation error: {str(e)}"
    
    def validate_exam_session_data(self, session_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate exam session data.
        
        Args:
            session_data: Exam session data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check required fields
            required_fields = ['session_name', 'sheet_version', 'subjects', 'answer_key_version']
            for field in required_fields:
                if field not in session_data:
                    return False, f"Missing required field: {field}"
            
            # Validate session name
            session_name = session_data['session_name']
            if not session_name or len(session_name) < 3:
                return False, "Session name must be at least 3 characters long"
            
            # Validate sheet version
            sheet_version = session_data['sheet_version']
            is_valid, message = self.validate_sheet_version(sheet_version)
            if not is_valid:
                return False, f"Invalid sheet version: {message}"
            
            # Validate subjects
            subjects = session_data['subjects']
            if not isinstance(subjects, list) or not subjects:
                return False, "Subjects must be a non-empty list"
            
            # Validate answer key version
            answer_key_version = session_data['answer_key_version']
            is_valid, message = self.validate_sheet_version(answer_key_version)
            if not is_valid:
                return False, f"Invalid answer key version: {message}"
            
            # Validate optional fields
            if 'total_questions' in session_data:
                total_questions = session_data['total_questions']
                if not isinstance(total_questions, int) or total_questions <= 0:
                    return False, "Total questions must be a positive integer"
            
            if 'questions_per_subject' in session_data:
                questions_per_subject = session_data['questions_per_subject']
                if not isinstance(questions_per_subject, int) or questions_per_subject <= 0:
                    return False, "Questions per subject must be a positive integer"
            
            return True, "Exam session data is valid"
            
        except Exception as e:
            return False, f"Exam session validation error: {str(e)}"
    
    def validate_batch_upload_data(self, 
                                  files: List[Any], 
                                  student_ids: List[str], 
                                  exam_session_id: int, 
                                  sheet_version: str) -> Tuple[bool, str]:
        """
        Validate batch upload data.
        
        Args:
            files: List of uploaded files
            student_ids: List of student IDs
            exam_session_id: Exam session ID
            sheet_version: Sheet version
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if files and student IDs are provided
            if not files:
                return False, "No files provided"
            
            if not student_ids:
                return False, "No student IDs provided"
            
            # Check if counts match
            if len(files) != len(student_ids):
                return False, f"Number of files ({len(files)}) does not match number of student IDs ({len(student_ids)})"
            
            # Validate exam session ID
            if not isinstance(exam_session_id, int) or exam_session_id <= 0:
                return False, "Exam session ID must be a positive integer"
            
            # Validate sheet version
            is_valid, message = self.validate_sheet_version(sheet_version)
            if not is_valid:
                return False, f"Invalid sheet version: {message}"
            
            # Validate each student ID
            for i, student_id in enumerate(student_ids):
                is_valid, message = self.validate_student_id(student_id)
                if not is_valid:
                    return False, f"Student ID {i+1}: {message}"
            
            return True, "Batch upload data is valid"
            
        except Exception as e:
            return False, f"Batch upload validation error: {str(e)}"
    
    def validate_email(self, email: str) -> Tuple[bool, str]:
        """
        Validate email address format.
        
        Args:
            email: Email address string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not email:
            return False, "Email cannot be empty"
        
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            return False, "Invalid email format"
        
        if len(email) > 100:
            return False, "Email must be less than 100 characters"
        
        return True, "Email is valid"
    
    def validate_phone(self, phone: str) -> Tuple[bool, str]:
        """
        Validate phone number format.
        
        Args:
            phone: Phone number string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not phone:
            return False, "Phone number cannot be empty"
        
        import re
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone)
        
        if len(digits_only) < 7:
            return False, "Phone number must have at least 7 digits"
        
        if len(digits_only) > 15:
            return False, "Phone number must have at most 15 digits"
        
        return True, "Phone number is valid"
