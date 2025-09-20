#!/usr/bin/env python3
"""
Comprehensive test script for OMR Evaluation System.
Tests all components including image preprocessing, bubble detection, and answer evaluation.
"""

import cv2
import numpy as np
import os
import sys
import time
import json
from typing import List, Dict, Any
import tempfile
import shutil

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omr_processor.image_preprocessor import ImagePreprocessor
from omr_processor.bubble_detector import BubbleDetector
from omr_processor.answer_evaluator import AnswerEvaluator
from omr_processor.omr_processor import OMRProcessor


class OMRSystemTester:
    """Comprehensive tester for OMR system components."""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = tempfile.mkdtemp()
        print(f"Test directory: {self.temp_dir}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_omr_sheet(self, width: int = 800, height: int = 1000, 
                            num_questions: int = 20, marked_answers: List[int] = None) -> np.ndarray:
        """Create a synthetic OMR sheet for testing."""
        if marked_answers is None:
            marked_answers = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
        
        # Create white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add border
        cv2.rectangle(image, (50, 50), (width-50, height-50), (0, 0, 0), 2)
        
        # Add title
        cv2.putText(image, "OMR TEST SHEET", (width//2 - 100, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # Add question numbers and bubbles
        bubble_radius = 12
        questions_per_row = 5
        rows = (num_questions + questions_per_row - 1) // questions_per_row
        
        for i in range(num_questions):
            row = i // questions_per_row
            col = i % questions_per_row
            
            y = 150 + row * 40
            x = 100 + col * 120
            
            # Question number
            cv2.putText(image, f"{i+1:2d}.", (x-30, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Answer bubbles (A, B, C, D)
            for j, letter in enumerate(['A', 'B', 'C', 'D']):
                bubble_x = x + j * 25
                cv2.circle(image, (bubble_x, y), bubble_radius, (0, 0, 0), 2)
                
                # Fill marked bubbles
                if i < len(marked_answers) and j == marked_answers[i]:
                    cv2.circle(image, (bubble_x, y), bubble_radius-2, (0, 0, 0), -1)
        
        return image
    
    def test_image_preprocessor(self) -> Dict[str, Any]:
        """Test image preprocessing functionality."""
        print("Testing Image Preprocessor...")
        
        try:
            preprocessor = ImagePreprocessor()
            
            # Create test image
            test_image = self.create_test_omr_sheet()
            
            # Test preprocessing
            start_time = time.time()
            processed_image = preprocessor.preprocess_image(test_image)
            processing_time = time.time() - start_time
            
            # Test orientation detection
            orientation = preprocessor.detect_orientation(processed_image)
            
            # Test rotation
            rotated_image = preprocessor.rotate_image(processed_image, 90)
            
            # Validate results
            success = (
                processed_image is not None and
                processed_image.shape[0] > 0 and
                processed_image.shape[1] > 0 and
                orientation in [0, 90, 180, 270] and
                rotated_image is not None
            )
            
            result = {
                "component": "ImagePreprocessor",
                "success": success,
                "processing_time": processing_time,
                "input_shape": test_image.shape,
                "output_shape": processed_image.shape if processed_image is not None else None,
                "orientation_detected": orientation,
                "rotation_test": rotated_image is not None
            }
            
            if success:
                print("✅ Image Preprocessor test passed")
            else:
                print("❌ Image Preprocessor test failed")
            
            return result
            
        except Exception as e:
            print(f"❌ Image Preprocessor test failed with error: {e}")
            return {
                "component": "ImagePreprocessor",
                "success": False,
                "error": str(e)
            }
    
    def test_bubble_detector(self) -> Dict[str, Any]:
        """Test bubble detection functionality."""
        print("Testing Bubble Detector...")
        
        try:
            detector = BubbleDetector()
            
            # Create test image
            test_image = self.create_test_omr_sheet()
            
            # Test bubble detection
            start_time = time.time()
            grid_config = {
                "rows": 20,
                "cols": 4,
                "subjects": {
                    "Test_Subject": {"start": 1, "end": 20}
                }
            }
            
            bubble_grid = detector.detect_bubbles(test_image, grid_config)
            processing_time = time.time() - start_time
            
            # Test answer extraction
            answers = detector.get_answer_choices(bubble_grid, 20)
            
            # Validate results
            success = (
                bubble_grid is not None and
                len(bubble_grid) > 0 and
                len(answers) > 0 and
                all(isinstance(row, list) for row in bubble_grid)
            )
            
            result = {
                "component": "BubbleDetector",
                "success": success,
                "processing_time": processing_time,
                "bubbles_detected": sum(len(row) for row in bubble_grid),
                "questions_processed": len(answers),
                "bubble_grid_rows": len(bubble_grid),
                "answers_extracted": len(answers)
            }
            
            if success:
                print("✅ Bubble Detector test passed")
            else:
                print("❌ Bubble Detector test failed")
            
            return result
            
        except Exception as e:
            print(f"❌ Bubble Detector test failed with error: {e}")
            return {
                "component": "BubbleDetector",
                "success": False,
                "error": str(e)
            }
    
    def test_answer_evaluator(self) -> Dict[str, Any]:
        """Test answer evaluation functionality."""
        print("Testing Answer Evaluator...")
        
        try:
            evaluator = AnswerEvaluator()
            
            # Create test answer key
            test_answer_key = {
                "version": "test_v1",
                "subjects": {
                    "Mathematics": {
                        "questions": list(range(1, 21)),
                        "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                                  "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
                    }
                }
            }
            
            evaluator.add_answer_key("test_v1", test_answer_key)
            
            # Create test student answers
            student_answers = [
                [0], [1], [2], [3], [0], [1], [2], [3], [0], [1],
                [2], [3], [0], [1], [2], [3], [0], [1], [2], [3]
            ]
            
            # Test evaluation
            start_time = time.time()
            result = evaluator.evaluate_answers(student_answers, "test_v1", "test_student")
            processing_time = time.time() - start_time
            
            # Validate results
            success = (
                result is not None and
                result.total_score >= 0 and
                result.total_percentage >= 0 and
                len(result.subject_scores) > 0 and
                result.student_id == "test_student"
            )
            
            result_data = {
                "component": "AnswerEvaluator",
                "success": success,
                "processing_time": processing_time,
                "total_score": result.total_score if result else None,
                "total_percentage": result.total_percentage if result else None,
                "subject_scores_count": len(result.subject_scores) if result else 0,
                "student_id": result.student_id if result else None
            }
            
            if success:
                print("✅ Answer Evaluator test passed")
            else:
                print("❌ Answer Evaluator test failed")
            
            return result_data
            
        except Exception as e:
            print(f"❌ Answer Evaluator test failed with error: {e}")
            return {
                "component": "AnswerEvaluator",
                "success": False,
                "error": str(e)
            }
    
    def test_omr_processor(self) -> Dict[str, Any]:
        """Test complete OMR processing pipeline."""
        print("Testing OMR Processor...")
        
        try:
            processor = OMRProcessor()
            
            # Create test image
            test_image = self.create_test_omr_sheet()
            
            # Save test image
            test_image_path = os.path.join(self.temp_dir, "test_omr.jpg")
            cv2.imwrite(test_image_path, test_image)
            
            # Test complete processing
            start_time = time.time()
            result = processor.process_omr_sheet(test_image_path, "demo_v1", "test_student")
            processing_time = time.time() - start_time
            
            # Validate results
            success = (
                result is not None and
                "success" in result and
                result["success"] and
                "result" in result and
                result["result"] is not None
            )
            
            result_data = {
                "component": "OMRProcessor",
                "success": success,
                "processing_time": processing_time,
                "student_id": result.get("student_id") if result else None,
                "total_score": result["result"].total_score if result and "result" in result else None,
                "total_percentage": result["result"].total_percentage if result and "result" in result else None,
                "subject_scores_count": len(result["result"].subject_scores) if result and "result" in result else 0
            }
            
            if success:
                print("✅ OMR Processor test passed")
            else:
                print("❌ OMR Processor test failed")
                if result and "error" in result:
                    print(f"   Error: {result['error']}")
            
            return result_data
            
        except Exception as e:
            print(f"❌ OMR Processor test failed with error: {e}")
            return {
                "component": "OMRProcessor",
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print("=" * 60)
        print("OMR EVALUATION SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run individual component tests
        tests = [
            self.test_image_preprocessor,
            self.test_bubble_detector,
            self.test_answer_evaluator,
            self.test_omr_processor
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                error_result = {
                    "component": test_func.__name__,
                    "success": False,
                    "error": str(e)
                }
                results.append(error_result)
                self.test_results.append(error_result)
                print(f"❌ {test_func.__name__} failed with error: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate overall statistics
        total_tests = len(results)
        successful_tests = sum(1 for result in results if result.get("success", False))
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Overall results
        overall_result = {
            "test_suite": "OMR Evaluation System",
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": success_rate,
            "total_execution_time": total_time,
            "component_results": results
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Time: {total_time:.2f}s")
        print("=" * 60)
        
        # Print individual results
        for result in results:
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            component = result.get("component", "Unknown")
            print(f"{status} {component}")
            if not result.get("success", False) and "error" in result:
                print(f"    Error: {result['error']}")
        
        return overall_result
    
    def save_results(self, filename: str = "test_results.json"):
        """Save test results to JSON file."""
        results_file = os.path.join(self.temp_dir, filename)
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"\nTest results saved to: {results_file}")
        return results_file


def main():
    """Main test execution function."""
    tester = OMRSystemTester()
    
    try:
        # Run all tests
        overall_results = tester.run_all_tests()
        
        # Save results
        results_file = tester.save_results()
        
        # Print final status
        if overall_results["success_rate"] >= 80:
            print("\n🎉 Overall test suite PASSED!")
            return 0
        else:
            print("\n💥 Overall test suite FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite execution failed: {e}")
        return 1
    
    finally:
        # Cleanup
        tester.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)