"""
Main entry point for OMR Evaluation System.
Provides commands to start the API server, web interface, or both.
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import streamlit
        import opencv_python
        import numpy
        import pandas
        import sqlalchemy
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_api_server(host="0.0.0.0", port=8000, workers=1, reload=False):
    """Start the FastAPI server."""
    print(f"🚀 Starting API server on {host}:{port}")
    
    cmd = [
        "uvicorn",
        "backend.main:app",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")

def start_web_interface(host="localhost", port=8501):
    """Start the Streamlit web interface."""
    print(f"🌐 Starting web interface on {host}:{port}")
    
    cmd = [
        "streamlit",
        "run",
        "app.py",
        "--server.address", host,
        "--server.port", str(port)
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start web interface: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped")

def start_both(api_host="0.0.0.0", api_port=8000, web_host="localhost", web_port=8501, reload=False):
    """Start both API server and web interface."""
    print("🚀 Starting OMR Evaluation System...")
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    from config import create_directories
    create_directories()
    print("📁 Created necessary directories")
    
    # Start API server in background
    print("🔧 Starting API server...")
    api_process = subprocess.Popen([
        "uvicorn", "backend.main:app",
        "--host", api_host,
        "--port", str(api_port),
        "--workers", "1"
    ] + (["--reload"] if reload else []))
    
    # Wait for API server to start
    print("⏳ Waiting for API server to start...")
    time.sleep(5)
    
    # Check if API server is running
    try:
        import requests
        response = requests.get(f"http://{api_host}:{api_port}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print("⚠️ API server may not be ready yet")
    except:
        print("⚠️ Could not verify API server status")
    
    # Start web interface
    print("🌐 Starting web interface...")
    try:
        web_process = subprocess.Popen([
            "streamlit", "run", "app.py",
            "--server.address", web_host,
            "--server.port", str(web_port)
        ])
        
        print(f"✅ OMR Evaluation System is running!")
        print(f"📊 Web Interface: http://{web_host}:{web_port}")
        print(f"🔧 API Server: http://{api_host}:{api_port}")
        print(f"📚 API Documentation: http://{api_host}:{api_port}/api/docs")
        print("\nPress Ctrl+C to stop both services")
        
        # Wait for processes
        try:
            web_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            api_process.terminate()
            web_process.terminate()
            api_process.wait()
            web_process.wait()
            print("✅ Shutdown complete")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start web interface: {e}")
        api_process.terminate()
        api_process.wait()
        sys.exit(1)

def setup_database():
    """Initialize the database."""
    print("🗄️ Setting up database...")
    
    try:
        from backend.database import create_tables
        create_tables()
        print("✅ Database tables created successfully")
        
        # Initialize default data
        from backend.main import initialize_default_configs
        import asyncio
        asyncio.run(initialize_default_configs())
        print("✅ Default configurations initialized")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OMR Evaluation System")
    parser.add_argument("command", choices=["api", "web", "both", "setup"], 
                       help="Command to run")
    parser.add_argument("--api-host", default="0.0.0.0", help="API server host")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port")
    parser.add_argument("--web-host", default="localhost", help="Web interface host")
    parser.add_argument("--web-port", type=int, default=8501, help="Web interface port")
    parser.add_argument("--workers", type=int, default=1, help="Number of API workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_database()
    elif args.command == "api":
        if not check_dependencies():
            sys.exit(1)
        start_api_server(args.api_host, args.api_port, args.workers, args.reload)
    elif args.command == "web":
        if not check_dependencies():
            sys.exit(1)
        start_web_interface(args.web_host, args.web_port)
    elif args.command == "both":
        start_both(args.api_host, args.api_port, args.web_host, args.web_port, args.reload)

if __name__ == "__main__":
    main()
