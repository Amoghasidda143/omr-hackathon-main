"""
Main OMR processor that orchestrates the complete evaluation pipeline.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

from .image_preprocessor import ImagePreprocessor
from .bubble_detector import BubbleDetector, BubbleRegion
from .answer_evaluator import AnswerEvaluator, ExamResult


class OMRProcessor:
    """Main OMR processing pipeline."""
    
    def __init__(self, 
                 answer_keys_dir: str = "answer_keys",
                 results_dir: str = "results",
                 uploads_dir: str = "uploads"):
        """
        Initialize OMR processor.
        
        Args:
            answer_keys_dir: Directory containing answer keys
            results_dir: Directory for storing results
            uploads_dir: Directory for uploaded images
        """
        self.image_preprocessor = ImagePreprocessor()
        self.bubble_detector = BubbleDetector()
        self.answer_evaluator = AnswerEvaluator(answer_keys_dir)
        
        self.results_dir = results_dir
        self.uploads_dir = uploads_dir
        
        # Create directories if they don't exist
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(uploads_dir, exist_ok=True)
    
    def process_omr_sheet(self, 
                         image_path: str, 
                         sheet_version: str,
                         student_id: str = None) -> Dict:
        """
        Process a single OMR sheet through the complete pipeline.
        
        Args:
            image_path: Path to the OMR sheet image
            sheet_version: Version of the OMR sheet
            student_id: Optional student identifier
            
        Returns:
            Processing result dictionary
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Generate student ID if not provided
            if student_id is None:
                student_id = f"student_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Step 1: Preprocess image
            preprocessed_image = self.image_preprocessor.preprocess_image(image)
            
            # Step 2: Detect orientation and rotate if needed
            orientation = self.image_preprocessor.detect_orientation(preprocessed_image)
            if orientation != 0:
                preprocessed_image = self.image_preprocessor.rotate_image(preprocessed_image, orientation)
            
            # Step 3: Configure bubble grid
            grid_config = self._get_grid_config()
            
            # Step 4: Detect bubbles
            bubble_grid = self.bubble_detector.detect_bubbles(preprocessed_image, grid_config)
            
            # Step 5: Extract answers
            student_answers = self.bubble_detector.get_answer_choices(bubble_grid)
            
            # Step 6: Evaluate answers
            exam_result = self.answer_evaluator.evaluate_answers(
                student_answers, sheet_version, student_id
            )
            
            # Step 7: Save results
            result_data = self._save_results(exam_result, image_path, preprocessed_image)
            
            return {
                "success": True,
                "student_id": student_id,
                "sheet_version": sheet_version,
                "result": exam_result,
                "result_data": result_data,
                "processing_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "image_path": image_path,
                    "orientation_correction": orientation,
                    "bubbles_detected": sum(len(row) for row in bubble_grid),
                    "questions_processed": len(student_answers)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "student_id": student_id,
                "sheet_version": sheet_version,
                "processing_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "image_path": image_path,
                    "error_type": type(e).__name__
                }
            }
    
    def _get_grid_config(self) -> Dict:
        """Get bubble grid configuration."""
        return {
            "rows": 100,  # 100 questions
            "cols": 5,    # 5 options per question (A, B, C, D, E)
            "subjects": {
                "Mathematics": {"start": 1, "end": 20},
                "Physics": {"start": 21, "end": 40},
                "Chemistry": {"start": 41, "end": 60},
                "Biology": {"start": 61, "end": 80},
                "General_Knowledge": {"start": 81, "end": 100}
            }
        }
    
    def _save_results(self, 
                     exam_result: ExamResult, 
                     original_image_path: str,
                     processed_image: np.ndarray) -> Dict:
        """
        Save processing results and metadata.
        
        Args:
            exam_result: Exam evaluation result
            original_image_path: Path to original image
            processed_image: Preprocessed image
            
        Returns:
            Result data dictionary
        """
        # Create result directory for this student
        student_dir = os.path.join(self.results_dir, exam_result.student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        # Save processed image
        processed_image_path = os.path.join(student_dir, "processed_image.jpg")
        cv2.imwrite(processed_image_path, processed_image)
        
        # Save result data as JSON
        result_data = {
            "student_id": exam_result.student_id,
            "sheet_version": exam_result.sheet_version,
            "total_score": exam_result.total_score,
            "total_percentage": exam_result.total_percentage,
            "subject_scores": [
                {
                    "subject": score.subject_name,
                    "correct": score.correct_answers,
                    "total": score.total_questions,
                    "score": score.score,
                    "percentage": score.percentage
                }
                for score in exam_result.subject_scores
            ],
            "answers": exam_result.answers,
            "correct_answers": exam_result.correct_answers,
            "evaluation_metadata": exam_result.evaluation_metadata,
            "file_paths": {
                "original_image": original_image_path,
                "processed_image": processed_image_path
            }
        }
        
        # Save JSON result
        import json
        result_json_path = os.path.join(student_dir, "result.json")
        with open(result_json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        return result_data
    
    def batch_process(self, 
                     image_paths: List[str], 
                     sheet_version: str) -> List[Dict]:
        """
        Process multiple OMR sheets in batch.
        
        Args:
            image_paths: List of image file paths
            sheet_version: Version of the OMR sheets
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.process_omr_sheet(image_path, sheet_version)
            results.append(result)
            
            if result["success"]:
                print(f"✓ Successfully processed {result['student_id']}")
            else:
                print(f"✗ Failed to process {image_path}: {result['error']}")
        
        return results
    
    def get_processing_statistics(self, results: List[Dict]) -> Dict:
        """
        Get processing statistics from batch results.
        
        Args:
            results: List of processing results
            
        Returns:
            Statistics dictionary
        """
        total_processed = len(results)
        successful = sum(1 for r in results if r["success"])
        failed = total_processed - successful
        
        if successful > 0:
            successful_results = [r for r in results if r["success"]]
            total_scores = [r["result"].total_score for r in successful_results]
            avg_score = sum(total_scores) / len(total_scores)
            max_score = max(total_scores)
            min_score = min(total_scores)
        else:
            avg_score = max_score = min_score = 0
        
        return {
            "total_processed": total_processed,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total_processed) * 100 if total_processed > 0 else 0,
            "average_score": avg_score,
            "max_score": max_score,
            "min_score": min_score
        }
    
    def validate_image(self, image_path: str) -> Tuple[bool, str]:
        """
        Validate if image is suitable for OMR processing.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return False, "File does not exist"
            
            # Try to load image
            image = cv2.imread(image_path)
            if image is None:
                return False, "Could not load image file"
            
            # Check image dimensions
            height, width = image.shape[:2]
            if height < 100 or width < 100:
                return False, "Image too small (minimum 100x100 pixels)"
            
            if height > 5000 or width > 5000:
                return False, "Image too large (maximum 5000x5000 pixels)"
            
            # Check file size (basic check)
            file_size = os.path.getsize(image_path)
            if file_size > 50 * 1024 * 1024:  # 50MB
                return False, "File too large (maximum 50MB)"
            
            return True, "Image is valid for processing"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_available_answer_versions(self) -> List[str]:
        """Get list of available answer key versions."""
        return self.answer_evaluator.get_available_versions()
    
    def add_answer_key(self, version: str, answer_key: Dict):
        """Add a new answer key."""
        self.answer_evaluator.add_answer_key(version, answer_key)
