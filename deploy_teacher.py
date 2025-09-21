"""
Teacher Deployment Script for OMR Evaluation System.
Provides easy deployment options for teachers.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if required tools are installed."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check if we're in the right directory
    if not os.path.exists("teacher_interface.py"):
        print("❌ teacher_interface.py not found. Please run from the project root directory.")
        return False
    
    print("✅ Project files found")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        # Check if virtual environment exists
        if not os.path.exists("venv"):
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        
        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_path = "venv\\Scripts\\pip"
            python_path = "venv\\Scripts\\python"
        else:  # Unix/Linux/macOS
            pip_path = "venv/bin/pip"
            python_path = "venv/bin/python"
        
        # Install dependencies
        print("Installing Python packages...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = ["uploads", "results", "logs", "answer_keys", "models"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")
    
    return True

def create_env_file():
    """Create environment configuration file."""
    print("⚙️ Creating configuration file...")
    
    env_content = """# OMR Evaluation System Configuration
# Database
DATABASE_URL=sqlite:///./omr_evaluation.db

# File Upload
UPLOAD_DIR=uploads
RESULTS_DIR=results
LOGS_DIR=logs
MAX_FILE_SIZE_MB=50

# Processing
BUBBLE_DETECTION_THRESHOLD=0.15
PROCESSING_TIMEOUT_SECONDS=300

# API
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_PORT=8501

# Teacher Settings
TEACHER_MODE=true
ENABLE_SAMPLE_DATA=true
MAX_STUDENTS_PER_BATCH=100
AUTO_SAVE_RESULTS=true
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Configuration file created (.env)")
    return True

def start_teacher_interface():
    """Start the teacher interface."""
    print("🚀 Starting teacher interface...")
    
    try:
        # Determine python path
        if os.name == 'nt':  # Windows
            python_path = "venv\\Scripts\\python"
        else:  # Unix/Linux/macOS
            python_path = "venv/bin/python"
        
        # Start teacher interface
        subprocess.run([python_path, "teacher_launcher.py"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start teacher interface: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Teacher interface stopped")
        return True

def deploy_docker():
    """Deploy using Docker."""
    print("🐳 Deploying with Docker...")
    
    try:
        # Check if Docker is installed
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker detected")
        
        # Check if Docker Compose is installed
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
        print("✅ Docker Compose detected")
        
        # Start services
        print("Starting Docker services...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        
        print("✅ Docker deployment successful!")
        print("🌐 Access the system at: http://localhost")
        print("👨‍🏫 Teacher interface: http://localhost:8501")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker deployment failed: {e}")
        print("Please ensure Docker and Docker Compose are installed")
        return False

def check_system_health():
    """Check if the system is running properly."""
    print("🔍 Checking system health...")
    
    try:
        import requests
        
        # Check backend
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend API is running")
            else:
                print("⚠️ Backend API responded with error")
        except:
            print("⚠️ Backend API not accessible")
        
        # Check frontend
        try:
            response = requests.get("http://localhost:8501", timeout=5)
            if response.status_code == 200:
                print("✅ Frontend is running")
            else:
                print("⚠️ Frontend responded with error")
        except:
            print("⚠️ Frontend not accessible")
        
        return True
        
    except ImportError:
        print("⚠️ Requests library not available for health check")
        return True

def show_usage_instructions():
    """Show usage instructions."""
    print("\n" + "="*60)
    print("🎉 DEPLOYMENT SUCCESSFUL!")
    print("="*60)
    print("\n📋 How to use the system:")
    print("1. Open your web browser")
    print("2. Go to: http://localhost:8501")
    print("3. Follow the 3-step workflow:")
    print("   - Step 1: Upload Answer Key")
    print("   - Step 2: Upload Student OMR Sheets")
    print("   - Step 3: View Results & Analytics")
    print("\n📚 Documentation:")
    print("- Teacher Guide: TEACHER_GUIDE.md")
    print("- Deployment Guide: TEACHER_DEPLOYMENT_GUIDE.md")
    print("\n🔧 Management:")
    print("- Stop system: Ctrl+C")
    print("- Restart: python deploy_teacher.py --start")
    print("- Docker: docker-compose down")
    print("\n📞 Support:")
    print("- Check logs in logs/ directory")
    print("- Run tests: python test_teacher_system.py")
    print("- Review troubleshooting guide")

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy OMR Evaluation System for Teachers")
    parser.add_argument("--mode", choices=["local", "docker"], default="local",
                       help="Deployment mode (default: local)")
    parser.add_argument("--start", action="store_true",
                       help="Start the teacher interface")
    parser.add_argument("--health", action="store_true",
                       help="Check system health")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only setup, don't start")
    
    args = parser.parse_args()
    
    print("👨‍🏫 OMR Evaluation System - Teacher Deployment")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    if args.mode == "local":
        print("\n🏠 Local Deployment Mode")
        print("-" * 30)
        
        # Install dependencies
        if not install_dependencies():
            sys.exit(1)
        
        # Create directories
        if not create_directories():
            sys.exit(1)
        
        # Create configuration
        if not create_env_file():
            sys.exit(1)
        
        if not args.setup_only:
            if args.start:
                # Start teacher interface
                start_teacher_interface()
            else:
                show_usage_instructions()
                print("\n🚀 To start the system, run:")
                print("python deploy_teacher.py --start")
    
    elif args.mode == "docker":
        print("\n🐳 Docker Deployment Mode")
        print("-" * 30)
        
        if not deploy_docker():
            sys.exit(1)
        
        if not args.setup_only:
            show_usage_instructions()
    
    if args.health:
        check_system_health()

if __name__ == "__main__":
    main()


