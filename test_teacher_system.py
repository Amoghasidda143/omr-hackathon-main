"""
Test script for Teacher OMR Evaluation System.
Tests the complete workflow from answer key upload to OMR processing.
"""

import cv2
import numpy as np
import json
import os
import tempfile
from datetime import datetime

# Import OMR processing modules
from omr_processor.image_preprocessor import ImagePreprocessor
from omr_processor.bubble_detector import BubbleDetector
from omr_processor.answer_evaluator import AnswerEvaluator
from omr_processor.omr_processor import OMRProcessor

def create_test_answer_key():
    """Create a test answer key."""
    return {
        "version": "test_v1",
        "name": "Test Exam Answer Key",
        "description": "Test answer key for system validation",
        "subjects": {
            "Mathematics": {
                "questions": list(range(1, 21)),
                "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                          "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
            },
            "Physics": {
                "questions": list(range(21, 41)),
                "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                          "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
            },
            "Chemistry": {
                "questions": list(range(41, 61)),
                "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                          "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
            },
            "Biology": {
                "questions": list(range(61, 81)),
                "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                          "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
            },
            "General_Knowledge": {
                "questions": list(range(81, 101)),
                "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                          "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
            }
        }
    }

def create_test_omr_image():
    """Create a test OMR sheet image."""
    # Create a white background
    image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    # Add border
    cv2.rectangle(image, (50, 50), (750, 950), (0, 0, 0), 3)
    
    # Add title
    cv2.putText(image, "TEST OMR SHEET", (250, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Add student ID section
    cv2.putText(image, "Student ID: TEST_STUDENT_001", (100, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add question numbers and bubbles
    for i in range(20):  # First 20 questions for test
        y = 200 + i * 30
        
        # Question number
        cv2.putText(image, f"{i+1:2d}.", (80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Answer bubbles (A, B, C, D)
        for j, letter in enumerate(['A', 'B', 'C', 'D']):
            x = 150 + j * 60
            cv2.circle(image, (x, y), 10, (0, 0, 0), 2)
            
            # Fill some bubbles for test (simulate student answers)
            if i < 10 and j == 0:  # Fill A for first 10 questions
                cv2.circle(image, (x, y), 8, (0, 0, 0), -1)
            elif i >= 10 and j == 1:  # Fill B for next 10 questions
                cv2.circle(image, (x, y), 8, (0, 0, 0), -1)
    
    return image

def test_answer_key_validation():
    """Test answer key validation."""
    print("🧪 Testing answer key validation...")
    
    # Test valid answer key
    valid_key = create_test_answer_key()
    evaluator = AnswerEvaluator()
    
    is_valid, errors = evaluator.validate_answer_key(valid_key)
    if is_valid:
        print("✅ Valid answer key validation passed")
    else:
        print(f"❌ Valid answer key validation failed: {errors}")
        return False
    
    # Test invalid answer key
    invalid_key = {"version": "test", "subjects": {}}
    is_valid, errors = evaluator.validate_answer_key(invalid_key)
    if not is_valid:
        print("✅ Invalid answer key validation passed")
    else:
        print("❌ Invalid answer key validation failed")
        return False
    
    return True

def test_omr_processing():
    """Test OMR processing pipeline."""
    print("🧪 Testing OMR processing pipeline...")
    
    try:
        # Create test answer key
        answer_key = create_test_answer_key()
        
        # Create test OMR image
        test_image = create_test_omr_image()
        
        # Save test image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, test_image)
            temp_path = tmp_file.name
        
        try:
            # Initialize OMR processor
            processor = OMRProcessor()
            
            # Add answer key
            processor.answer_evaluator.add_answer_key("test_v1", answer_key)
            
            # Process OMR sheet
            result = processor.process_omr_sheet(temp_path, "test_v1", "TEST_STUDENT_001")
            
            if result["success"]:
                print("✅ OMR processing successful")
                print(f"   Student ID: {result['student_id']}")
                print(f"   Total Score: {result['result'].total_score}")
                print(f"   Total Percentage: {result['result'].total_percentage:.1f}%")
                
                # Check subject scores
                for score in result["result"].subject_scores:
                    print(f"   {score.subject_name}: {score.correct_answers}/{score.total_questions} ({score.percentage:.1f}%)")
                
                return True
            else:
                print(f"❌ OMR processing failed: {result['error']}")
                return False
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f"❌ OMR processing test failed with exception: {str(e)}")
        return False

def test_image_preprocessing():
    """Test image preprocessing."""
    print("🧪 Testing image preprocessing...")
    
    try:
        # Create test image
        test_image = create_test_omr_image()
        
        # Initialize preprocessor
        preprocessor = ImagePreprocessor()
        
        # Test preprocessing
        processed_image = preprocessor.preprocess_image(test_image)
        
        if processed_image is not None and processed_image.shape == test_image.shape:
            print("✅ Image preprocessing successful")
            return True
        else:
            print("❌ Image preprocessing failed")
            return False
            
    except Exception as e:
        print(f"❌ Image preprocessing test failed with exception: {str(e)}")
        return False

def test_bubble_detection():
    """Test bubble detection."""
    print("🧪 Testing bubble detection...")
    
    try:
        # Create test image
        test_image = create_test_omr_image()
        
        # Initialize bubble detector
        detector = BubbleDetector()
        
        # Test bubble detection
        grid_config = {
            "rows": 100,
            "cols": 5,
            "subjects": {
                "Mathematics": {"start": 1, "end": 20},
                "Physics": {"start": 21, "end": 40},
                "Chemistry": {"start": 41, "end": 60},
                "Biology": {"start": 61, "end": 80},
                "General_Knowledge": {"start": 81, "end": 100}
            }
        }
        
        bubble_grid = detector.detect_bubbles(test_image, grid_config)
        
        if bubble_grid is not None:
            print("✅ Bubble detection successful")
            return True
        else:
            print("❌ Bubble detection failed")
            return False
            
    except Exception as e:
        print(f"❌ Bubble detection test failed with exception: {str(e)}")
        return False

def test_answer_evaluation():
    """Test answer evaluation."""
    print("🧪 Testing answer evaluation...")
    
    try:
        # Create test answer key
        answer_key = create_test_answer_key()
        
        # Initialize evaluator
        evaluator = AnswerEvaluator()
        evaluator.add_answer_key("test_v1", answer_key)
        
        # Create test student answers (simulate some correct and incorrect answers)
        student_answers = []
        for i in range(100):
            if i < 10:
                student_answers.append([0])  # Answer A for first 10 questions
            elif i < 20:
                student_answers.append([1])  # Answer B for next 10 questions
            else:
                student_answers.append([])  # No answer for remaining questions
        
        # Evaluate answers
        result = evaluator.evaluate_answers(student_answers, "test_v1", "TEST_STUDENT_001")
        
        if result is not None:
            print("✅ Answer evaluation successful")
            print(f"   Total Score: {result.total_score}")
            print(f"   Total Percentage: {result.total_percentage:.1f}%")
            return True
        else:
            print("❌ Answer evaluation failed")
            return False
            
    except Exception as e:
        print(f"❌ Answer evaluation test failed with exception: {str(e)}")
        return False

def test_system_integration():
    """Test complete system integration."""
    print("🧪 Testing complete system integration...")
    
    try:
        # Create test data
        answer_key = create_test_answer_key()
        test_image = create_test_omr_image()
        
        # Save test image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, test_image)
            temp_path = tmp_file.name
        
        try:
            # Initialize complete system
            processor = OMRProcessor()
            processor.answer_evaluator.add_answer_key("test_v1", answer_key)
            
            # Process complete pipeline
            result = processor.process_omr_sheet(temp_path, "test_v1", "INTEGRATION_TEST_STUDENT")
            
            if result["success"]:
                print("✅ System integration test successful")
                print(f"   Processing time: {result['processing_metadata'].get('processing_time_seconds', 0):.2f}s")
                print(f"   Bubbles detected: {result['processing_metadata'].get('bubbles_detected', 0)}")
                print(f"   Questions processed: {result['processing_metadata'].get('questions_processed', 0)}")
                return True
            else:
                print(f"❌ System integration test failed: {result['error']}")
                return False
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f"❌ System integration test failed with exception: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Teacher OMR System Tests")
    print("=" * 50)
    
    tests = [
        ("Answer Key Validation", test_answer_key_validation),
        ("Image Preprocessing", test_image_preprocessing),
        ("Bubble Detection", test_bubble_detection),
        ("Answer Evaluation", test_answer_evaluation),
        ("OMR Processing", test_omr_processing),
        ("System Integration", test_system_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for use.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the system configuration.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
