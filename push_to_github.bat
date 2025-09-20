@echo off
echo ========================================
echo    OMR Evaluation System - GitHub Push
echo ========================================
echo.

echo Checking Git installation...
git --version
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/download/win
    echo and restart your terminal
    pause
    exit /b 1
)

echo.
echo Git is installed! Proceeding with GitHub push...
echo.

echo Initializing Git repository...
git init

echo.
echo Adding all files...
git add .

echo.
echo Creating initial commit...
git commit -m "Initial commit: OMR Evaluation System ready for deployment"

echo.
echo ========================================
echo    IMPORTANT: Set up your GitHub repository
echo ========================================
echo.
echo 1. Go to https://github.com
echo 2. Create a new repository named: omr-hackathon-main
echo 3. Make it PUBLIC
echo 4. Don't initialize with README
echo 5. Copy the repository URL
echo.
echo Your repository URL should look like:
echo https://github.com/YOUR_USERNAME/omr-hackathon-main.git
echo.
echo Enter your GitHub repository URL:
set /p REPO_URL=

echo.
echo Adding remote origin...
git remote add origin %REPO_URL%

echo.
echo Pushing to GitHub...
git branch -M main
git push -u origin main

echo.
echo ========================================
echo    SUCCESS! Your code is now on GitHub
echo ========================================
echo.
echo Next steps:
echo 1. Go to https://share.streamlit.io/
echo 2. Deploy your app using the repository
echo 3. Main file: deploy_streamlit.py
echo 4. Requirements: requirements_streamlit_cloud.txt
echo.
pause
