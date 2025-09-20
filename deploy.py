"""
Deployment helper script for OMR Evaluation System.
Prepares the application for deployment on Streamlit Cloud or HuggingFace Spaces.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def create_deployment_structure():
    """Create the necessary structure for deployment."""
    print("🚀 Preparing OMR Evaluation System for deployment...")
    
    # Create deployment directory
    deploy_dir = Path("deployment")
    deploy_dir.mkdir(exist_ok=True)
    
    # Files to copy for deployment
    files_to_copy = [
        "streamlit_app.py",
        "requirements_streamlit.txt",
        "packages.txt",
        ".streamlit/config.toml",
        "README_DEPLOYMENT.md"
    ]
    
    # Directories to copy
    dirs_to_copy = [
        "omr_processor",
        "utils"
    ]
    
    print("📁 Copying files...")
    
    # Copy files
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            dest_path = deploy_dir / file_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)
            print(f"  ✅ Copied {file_path}")
        else:
            print(f"  ⚠️  File not found: {file_path}")
    
    # Copy directories
    for dir_path in dirs_to_copy:
        if os.path.exists(dir_path):
            dest_path = deploy_dir / dir_path
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(dir_path, dest_path)
            print(f"  ✅ Copied directory {dir_path}")
        else:
            print(f"  ⚠️  Directory not found: {dir_path}")
    
    # Create __init__.py files
    init_files = [
        "deployment/omr_processor/__init__.py",
        "deployment/utils/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"  ✅ Created {init_file}")
    
    print("✅ Deployment structure created successfully!")
    return deploy_dir

def create_gitignore():
    """Create .gitignore for deployment."""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Results and uploads (for deployment)
results/
uploads/
answer_keys/
models/
"""
    
    with open("deployment/.gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore for deployment")

def create_readme():
    """Create README for deployment."""
    readme_content = """# OMR Evaluation System

Automated OMR sheet evaluation and scoring system.

## Quick Start

This application is ready to run on Streamlit Cloud or HuggingFace Spaces.

## Features

- Upload and process OMR sheets
- Real-time processing and scoring
- Comprehensive analytics and reporting
- Export results as CSV/Excel
- Batch processing capabilities

## Usage

1. Upload OMR sheet images (JPG, PNG)
2. Select processing options
3. View results and analytics
4. Export data as needed

## Technical Details

- Built with Streamlit
- Uses OpenCV for image processing
- Machine learning for bubble detection
- Comprehensive error handling

For more information, see the full documentation.
"""
    
    with open("deployment/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ Created README for deployment")

def test_deployment():
    """Test the deployment locally."""
    print("🧪 Testing deployment locally...")
    
    try:
        # Change to deployment directory
        os.chdir("deployment")
        
        # Install dependencies
        print("📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"], 
                      check=True, capture_output=True)
        
        # Test import
        print("🔍 Testing imports...")
        import streamlit_app
        print("  ✅ streamlit_app imports successfully")
        
        # Test OMR processor
        from omr_processor.omr_processor import OMRProcessor
        print("  ✅ OMR processor imports successfully")
        
        print("✅ Local test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False
    finally:
        os.chdir("..")

def create_deployment_instructions():
    """Create deployment instructions."""
    instructions = """
# 🚀 Deployment Instructions

## Option 1: Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy OMR Evaluation System"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path to: `streamlit_app.py`
   - Click "Deploy!"

## Option 2: HuggingFace Spaces

1. **Create Space**:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Streamlit" SDK
   - Set space name: `omr-evaluation-system`

2. **Upload Files**:
   - Clone your space repository
   - Copy all files from `deployment/` folder
   - Commit and push

## Verification

After deployment, verify:
- [ ] App loads without errors
- [ ] Upload functionality works
- [ ] Processing completes successfully
- [ ] Results display properly
- [ ] Export functions work

## Troubleshooting

- Check logs for errors
- Verify all dependencies are installed
- Test locally first
- Check file paths and structure
"""
    
    with open("deployment/DEPLOYMENT_INSTRUCTIONS.md", "w", encoding="utf-8") as f:
        f.write(instructions)
    
    print("✅ Created deployment instructions")

def main():
    """Main deployment function."""
    print("🎯 OMR Evaluation System - Deployment Helper")
    print("=" * 50)
    
    # Create deployment structure
    deploy_dir = create_deployment_structure()
    
    # Create additional files
    create_gitignore()
    create_readme()
    create_deployment_instructions()
    
    # Test deployment
    if test_deployment():
        print("\n🎉 Deployment preparation completed successfully!")
        print(f"📁 Deployment files are in: {deploy_dir}")
        print("\n📋 Next steps:")
        print("1. Review the deployment files")
        print("2. Push to GitHub repository")
        print("3. Deploy on Streamlit Cloud or HuggingFace Spaces")
        print("4. Test the deployed application")
        print("\n📖 See DEPLOYMENT_INSTRUCTIONS.md for detailed steps")
    else:
        print("\n❌ Deployment preparation failed!")
        print("Please check the errors above and try again.")

if __name__ == "__main__":
    main()