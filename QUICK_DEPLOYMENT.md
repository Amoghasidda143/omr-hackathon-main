# 🚀 Quick Deployment Guide - OMR Evaluation System

## 📋 Choose Your Deployment Method

### **Option 1: Streamlit Cloud (Recommended - 5 minutes)**

This is the fastest way to get your app online:

#### **Step 1: Prepare Your Code**
```bash
# Make sure all files are in your project directory
cd omr-hackathon-main

# Check that you have these key files:
# - deploy_streamlit.py (main app file)
# - requirements_streamlit_cloud.txt (dependencies)
# - omr_processor/ (all processing modules)
# - .streamlit/config.toml (configuration)
```

#### **Step 2: Push to GitHub**
```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Ready for deployment"

# Create GitHub repository and push
git remote add origin https://github.com/YOUR_USERNAME/omr-hackathon-main.git
git branch -M main
git push -u origin main
```

#### **Step 3: Deploy to Streamlit Cloud**
1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository: `YOUR_USERNAME/omr-hackathon-main`
5. Set main file path: `deploy_streamlit.py`
6. Set requirements file: `requirements_streamlit_cloud.txt`
7. Click "Deploy!"

#### **Step 4: Get Your URL**
Your app will be available at:
`https://YOUR_USERNAME-omr-hackathon-main-deploy-streamlit-XXXXXX.streamlit.app`

---

### **Option 2: Local Deployment (For Testing)**

#### **Step 1: Install Dependencies**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements_streamlit_cloud.txt
```

#### **Step 2: Run the App**
```bash
# Run the Streamlit app
streamlit run deploy_streamlit.py

# Or use the original app
streamlit run streamlit_app.py
```

#### **Step 3: Access the App**
- Open your browser to: `http://localhost:8501`

---

### **Option 3: Docker Deployment**

#### **Step 1: Build Docker Image**
```bash
# Build the image
docker build -t omr-evaluation .

# Run the container
docker run -p 8501:8501 omr-evaluation
```

#### **Step 2: Access the App**
- Open your browser to: `http://localhost:8501`

---

## 🔧 Troubleshooting

### **Common Issues & Solutions**

#### **1. Import Errors**
```bash
# Make sure all dependencies are installed
pip install -r requirements_streamlit_cloud.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### **2. Streamlit Cloud Deployment Issues**
- Check that `deploy_streamlit.py` is in the root directory
- Verify `requirements_streamlit_cloud.txt` has all dependencies
- Ensure all `omr_processor` modules are included
- Check Streamlit Cloud logs for specific errors

#### **3. File Upload Issues**
- Ensure file size is under 50MB
- Check file format (JPG, PNG, PDF supported)
- Verify image quality and contrast

#### **4. Processing Errors**
- Check that answer keys are properly configured
- Verify image preprocessing parameters
- Review error logs for specific issues

---

## 📊 Testing Your Deployment

### **1. Basic Functionality Test**
1. Open your deployed app
2. Navigate to "Upload & Process"
3. Click "Use Sample OMR Sheet"
4. Generate and process a sample
5. Verify results are displayed correctly

### **2. File Upload Test**
1. Prepare a test OMR sheet image
2. Upload the image using "Upload Image File"
3. Process the sheet
4. Check results and analytics

### **3. Performance Test**
1. Test processing speed (should be 2-3 seconds)
2. Check accuracy with known answers
3. Verify error handling with invalid files

---

## 🎯 Final Checklist

### **Before Submission**
- [ ] App is deployed and accessible
- [ ] All features work correctly
- [ ] Sample processing works
- [ ] File upload works
- [ ] Results display properly
- [ ] Analytics are functional
- [ ] Error handling works

### **Submission URLs**
1. **GitHub Repository**: `https://github.com/YOUR_USERNAME/omr-hackathon-main`
2. **Web Application**: `https://YOUR_USERNAME-omr-hackathon-main-deploy-streamlit-XXXXXX.streamlit.app`
3. **Video Presentation**: `https://youtube.com/watch?v=YOUR_VIDEO_ID`

---

## 🚀 Quick Start Commands

```bash
# 1. Test locally first
streamlit run deploy_streamlit.py

# 2. Push to GitHub
git add .
git commit -m "Ready for deployment"
git push origin main

# 3. Deploy to Streamlit Cloud
# Go to https://share.streamlit.io/
# Follow the deployment steps above
```

---

## 📞 Need Help?

If you encounter any issues:

1. **Check the logs** in Streamlit Cloud dashboard
2. **Test locally** first to identify issues
3. **Review the code** for any missing dependencies
4. **Check file paths** and module imports
5. **Verify all files** are in the repository

---

**Your OMR Evaluation System is ready to deploy! 🎉**
