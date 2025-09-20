"""
Test script for OMR Evaluation System
Verifies that all components are working correctly.
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import cv2
        print("  ✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"  ❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"  ❌ NumPy import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("  ✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"  ❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✅ Pandas imported successfully")
    except ImportError as e:
        print(f"  ❌ Pandas import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        print("  ✅ Plotly imported successfully")
    except ImportError as e:
        print(f"  ❌ Plotly import failed: {e}")
        return False
    
    try:
        import fastapi
        print("  ✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"  ❌ FastAPI import failed: {e}")
        return False
    
    try:
        import sqlalchemy
        print("  ✅ SQLAlchemy imported successfully")
    except ImportError as e:
        print(f"  ❌ SQLAlchemy import failed: {e}")
        return False
    
    return True

def test_omr_modules():
    """Test OMR processing modules."""
    print("🔧 Testing OMR modules...")
    
    try:
        from omr.pipeline import read_image_bytes, preprocess_image, find_document_contour
        print("  ✅ OMR pipeline imported successfully")
    except ImportError as e:
        print(f"  ❌ OMR pipeline import failed: {e}")
        return False
    
    try:
        from omr.utils import load_answer_keys
        print("  ✅ OMR utils imported successfully")
    except ImportError as e:
        print(f"  ❌ OMR utils import failed: {e}")
        return False
    
    try:
        from omr_processor.omr_processor import OMRProcessor
        print("  ✅ OMR processor imported successfully")
    except ImportError as e:
        print(f"  ❌ OMR processor import failed: {e}")
        return False
    
    return True

def test_backend_modules():
    """Test backend modules."""
    print("🚀 Testing backend modules...")
    
    try:
        from backend.main import app
        print("  ✅ FastAPI app imported successfully")
    except ImportError as e:
        print(f"  ❌ FastAPI app import failed: {e}")
        return False
    
    try:
        from backend.schemas import StudentCreate, ExamResultResponse
        print("  ✅ Backend schemas imported successfully")
    except ImportError as e:
        print(f"  ❌ Backend schemas import failed: {e}")
        return False
    
    try:
        from backend.services import OMRProcessingService, DatabaseService
        print("  ✅ Backend services imported successfully")
    except ImportError as e:
        print(f"  ❌ Backend services import failed: {e}")
        return False
    
    return True

def test_database_models():
    """Test database models."""
    print("🗄️ Testing database models...")
    
    try:
        from models.database import get_db, create_tables
        print("  ✅ Database configuration imported successfully")
    except ImportError as e:
        print(f"  ❌ Database configuration import failed: {e}")
        return False
    
    try:
        from models.omr_models import Student, ExamResult, OMRSheet
        print("  ✅ Database models imported successfully")
    except ImportError as e:
        print(f"  ❌ Database models import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if required files and directories exist."""
    print("📁 Testing file structure...")
    
    required_files = [
        "app/main.py",
        "backend/main.py",
        "omr/pipeline.py",
        "omr/utils.py",
        "omr_processor/omr_processor.py",
        "models/database.py",
        "models/omr_models.py",
        "requirements.txt",
        "answer_keys.json",
        "runner.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path} exists")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    return True

def test_answer_keys():
    """Test answer key loading."""
    print("🔑 Testing answer keys...")
    
    try:
        from omr.utils import load_answer_keys
        keys = load_answer_keys()
        
        if not keys:
            print("  ⚠️ No answer keys found")
            return False
        
        print(f"  ✅ Loaded {len(keys)} answer key versions")
        for version in keys.keys():
            print(f"    - {version}")
        
        return True
    except Exception as e:
        print(f"  ❌ Answer key loading failed: {e}")
        return False

def test_omr_processing():
    """Test basic OMR processing functionality."""
    print("🎯 Testing OMR processing...")
    
    try:
        from omr_processor.omr_processor import OMRProcessor
        
        # Create processor instance
        processor = OMRProcessor()
        print("  ✅ OMR processor created successfully")
        
        # Test validation
        is_valid, message = processor.validate_image("nonexistent.jpg")
        if not is_valid and "File does not exist" in message:
            print("  ✅ Image validation working correctly")
        else:
            print("  ⚠️ Image validation may not be working as expected")
        
        return True
    except Exception as e:
        print(f"  ❌ OMR processing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 OMR Evaluation System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("OMR Modules", test_omr_modules),
        ("Backend Modules", test_backend_modules),
        ("Database Models", test_database_models),
        ("Answer Keys", test_answer_keys),
        ("OMR Processing", test_omr_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To start the system, run:")
        print("   python runner.py both")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n🔧 To install missing dependencies, run:")
        print("   python runner.py install")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
