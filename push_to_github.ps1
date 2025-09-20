# OMR Evaluation System - GitHub Push Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   OMR Evaluation System - GitHub Push" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/download/win" -ForegroundColor Red
    Write-Host "and restart your terminal" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Git is installed! Proceeding with GitHub push..." -ForegroundColor Green
Write-Host ""

Write-Host "Initializing Git repository..." -ForegroundColor Yellow
git init

Write-Host ""
Write-Host "Adding all files..." -ForegroundColor Yellow
git add .

Write-Host ""
Write-Host "Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: OMR Evaluation System ready for deployment"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   IMPORTANT: Set up your GitHub repository" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Go to https://github.com" -ForegroundColor White
Write-Host "2. Create a new repository named: omr-hackathon-main" -ForegroundColor White
Write-Host "3. Make it PUBLIC" -ForegroundColor White
Write-Host "4. Don't initialize with README" -ForegroundColor White
Write-Host "5. Copy the repository URL" -ForegroundColor White
Write-Host ""
Write-Host "Your repository URL should look like:" -ForegroundColor White
Write-Host "https://github.com/YOUR_USERNAME/omr-hackathon-main.git" -ForegroundColor White
Write-Host ""

$repoUrl = Read-Host "Enter your GitHub repository URL"

Write-Host ""
Write-Host "Adding remote origin..." -ForegroundColor Yellow
git remote add origin $repoUrl

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git branch -M main
git push -u origin main

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   SUCCESS! Your code is now on GitHub" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Go to https://share.streamlit.io/" -ForegroundColor White
Write-Host "2. Deploy your app using the repository" -ForegroundColor White
Write-Host "3. Main file: deploy_streamlit.py" -ForegroundColor White
Write-Host "4. Requirements: requirements_streamlit_cloud.txt" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
