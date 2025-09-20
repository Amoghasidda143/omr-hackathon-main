"""
Teacher Launcher for OMR Evaluation System.
Simple script to start the teacher interface.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import opencv_python
        import numpy
        import pandas
        import plotly
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_teacher_interface():
    """Start the teacher interface."""
    print("👨‍🏫 Starting Teacher OMR Evaluation System...")
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("answer_keys", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("📁 Created necessary directories")
    
    # Start teacher interface
    try:
        subprocess.run([
            "streamlit", "run", "teacher_interface.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start teacher interface: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Teacher interface stopped")

if __name__ == "__main__":
    start_teacher_interface()
