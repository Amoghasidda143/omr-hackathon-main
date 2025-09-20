"""
OMR Evaluation System Runner
Comprehensive script to install dependencies and run the OMR evaluation system.
"""

import subprocess
import sys
import os
import argparse
import time
import threading
from pathlib import Path

def print_banner():
    """Print system banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    OMR Evaluation System                      ║
    ║              Automated OMR Sheet Processing                  ║
    ║                        Version 1.0.0                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = [
        "uploads",
        "results", 
        "logs",
        "models",
        "answer_keys",
        "static"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created: {directory}/")

def run_backend():
    """Run FastAPI backend."""
    print("🚀 Starting FastAPI backend...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "backend.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error running backend: {e}")

def run_frontend():
    """Run Streamlit frontend."""
    print("🎨 Starting Streamlit frontend...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/main.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error running frontend: {e}")

def run_both():
    """Run both backend and frontend."""
    print("🚀 Starting OMR Evaluation System...")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend
    run_frontend()

def run_tests():
    """Run system tests."""
    print("🧪 Running tests...")
    try:
        subprocess.run([sys.executable, "test_system.py"])
    except Exception as e:
        print(f"❌ Error running tests: {e}")

def show_status():
    """Show system status."""
    print("📊 System Status:")
    
    # Check if directories exist
    directories = ["uploads", "results", "logs", "models", "answer_keys"]
    for directory in directories:
        if Path(directory).exists():
            print(f"  ✅ {directory}/ - exists")
        else:
            print(f"  ❌ {directory}/ - missing")
    
    # Check if key files exist
    key_files = [
        "app/main.py",
        "backend/main.py", 
        "omr/pipeline.py",
        "requirements.txt",
        "answer_keys.json"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path} - exists")
        else:
            print(f"  ❌ {file_path} - missing")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="OMR Evaluation System Runner")
    parser.add_argument(
        "mode",
        choices=["install", "backend", "frontend", "both", "test", "status"],
        help="Mode to run the system"
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip installation of requirements"
    )
    
    args = parser.parse_args()
    
    print_banner()
    check_python_version()
    
    if args.mode == "install":
        if not args.no_install:
            install_requirements()
        create_directories()
        print("✅ Installation completed!")
        
    elif args.mode == "backend":
        if not args.no_install:
            install_requirements()
        create_directories()
        run_backend()
        
    elif args.mode == "frontend":
        if not args.no_install:
            install_requirements()
        create_directories()
        run_frontend()
        
    elif args.mode == "both":
        if not args.no_install:
            install_requirements()
        create_directories()
        run_both()
        
    elif args.mode == "test":
        if not args.no_install:
            install_requirements()
        create_directories()
        run_tests()
        
    elif args.mode == "status":
        show_status()

if __name__ == "__main__":
    main()