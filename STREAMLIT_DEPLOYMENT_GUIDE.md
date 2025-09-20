# 🚀 Streamlit Cloud Deployment Guide - OMR Evaluation System

## 📋 Quick Deployment to Streamlit Cloud

This guide will help you deploy the OMR Evaluation System to Streamlit Cloud in just a few steps.

## 🎯 Prerequisites

- GitHub account
- Streamlit Cloud account
- Repository with the OMR system code

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Fork or Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd OMR_Evaluation_System_Complete
   ```

2. **Ensure Required Files Are Present**
   - `streamlit_cloud_app.py` - Main Streamlit app for cloud
   - `requirements_streamlit_cloud.txt` - Dependencies
   - `omr_processor/` - OMR processing modules
   - `config.py` - Configuration settings

### Step 2: Push to GitHub

1. **Initialize Git Repository** (if not already done)
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Streamlit Cloud deployment"
   ```

2. **Create GitHub Repository**
   - Go to [GitHub](https://github.com)
   - Click "New Repository"
   - Name it: `omr-evaluation-system`
   - Make it public (required for free Streamlit Cloud)

3. **Push to GitHub**
   ```bash
   git remote add origin https://github.com/yourusername/omr-evaluation-system.git
   git branch -M main
   git push -u origin main
   ```

### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Select your repository: `omr-evaluation-system`
   - Select branch: `main`
   - Set main file path: `streamlit_cloud_app.py`
   - Click "Deploy"

3. **Wait for Deployment**
   - Streamlit will automatically install dependencies
   - This may take 5-10 minutes for the first deployment
   - You'll see the build logs in real-time

### Step 4: Access Your App

- Your app will be available at: `https://your-app-name.streamlit.app`
- Share this URL with teachers and students

## 🔧 Configuration Options

### Environment Variables (Optional)

You can set environment variables in Streamlit Cloud:

1. **Go to App Settings**
   - Click on your app in Streamlit Cloud
   - Go to "Settings" tab
   - Scroll to "Environment variables"

2. **Add Variables** (if needed)
   ```
   LOG_LEVEL=INFO
   MAX_FILE_SIZE_MB=50
   BUBBLE_DETECTION_THRESHOLD=0.15
   ```

### Custom Domain (Optional)

1. **Go to App Settings**
2. **Scroll to "Custom domain"**
3. **Add your domain** (requires domain verification)

## 📊 App Features

### What Teachers Can Do:

1. **Upload Answer Keys**
   - JSON format answer keys
   - Sample answer keys for testing
   - Manual answer key creation

2. **Process OMR Sheets**
   - Upload individual OMR images
   - Batch upload multiple sheets
   - Generate sample OMR sheets for testing

3. **View Results & Analytics**
   - Individual student scores
   - Class statistics
   - Subject-wise performance
   - Export results (CSV, Excel)

### Supported File Formats:
- **Images**: JPG, JPEG, PNG, BMP, TIFF, WEBP
- **Documents**: PDF
- **Answer Keys**: JSON

## 🚨 Troubleshooting

### Common Issues

#### 1. Build Failures
**Problem**: App fails to build
**Solution**: 
- Check `requirements_streamlit_cloud.txt` for compatibility
- Ensure all dependencies are listed
- Check for version conflicts

#### 2. Import Errors
**Problem**: Module not found errors
**Solution**:
- Ensure all Python files are in the repository
- Check file paths in imports
- Verify `omr_processor/` directory structure

#### 3. Memory Issues
**Problem**: App crashes due to memory limits
**Solution**:
- Optimize image processing
- Reduce batch sizes
- Use smaller file uploads

#### 4. Slow Performance
**Problem**: App is slow to load
**Solution**:
- Optimize dependencies
- Use smaller images
- Implement caching

### Debug Mode

To enable debug logging:

1. **Add Environment Variable**
   ```
   LOG_LEVEL=DEBUG
   ```

2. **Check Logs**
   - Go to your app in Streamlit Cloud
   - Click "Logs" tab
   - Review error messages

## 🔄 Updates and Maintenance

### Updating Your App

1. **Make Changes Locally**
   ```bash
   # Edit your files
   # Test locally
   streamlit run streamlit_cloud_app.py
   ```

2. **Push Changes**
   ```bash
   git add .
   git commit -m "Update OMR system"
   git push origin main
   ```

3. **Streamlit Auto-Deploys**
   - Streamlit automatically detects changes
   - App redeploys within minutes
   - No manual intervention needed

### Monitoring Usage

1. **View Analytics**
   - Go to your app in Streamlit Cloud
   - Click "Analytics" tab
   - See usage statistics

2. **Check Logs**
   - Monitor app performance
   - Debug issues
   - Track errors

## 📈 Performance Optimization

### For Better Performance:

1. **Optimize Dependencies**
   - Use minimal required packages
   - Avoid heavy ML libraries if not needed
   - Use specific versions

2. **Image Processing**
   - Resize images before processing
   - Use efficient algorithms
   - Implement caching

3. **Memory Management**
   - Clear session state regularly
   - Use generators for large datasets
   - Optimize data structures

## 🎯 Best Practices

### Code Organization
- Keep main app file simple
- Separate processing logic into modules
- Use proper error handling

### User Experience
- Provide clear instructions
- Show progress indicators
- Handle errors gracefully

### Security
- Validate file uploads
- Sanitize user inputs
- Use secure file handling

## 📞 Support

### Getting Help

1. **Check Documentation**
   - Streamlit Cloud docs
   - OMR system documentation
   - GitHub repository

2. **Community Support**
   - Streamlit Community Forum
   - GitHub Issues
   - Stack Overflow

3. **Debug Steps**
   - Check build logs
   - Review error messages
   - Test locally first

## 🎉 Success Checklist

After deployment, verify:
- [ ] App loads at your Streamlit URL
- [ ] Can upload answer keys
- [ ] Can process OMR sheets
- [ ] Results display correctly
- [ ] Export functionality works
- [ ] No error messages in logs

## 🚀 Advanced Features

### Custom Styling
- Modify CSS in the app
- Add custom themes
- Implement responsive design

### Integration
- Connect to external databases
- Use cloud storage
- Implement authentication

### Scaling
- Use Streamlit Cloud Pro for more resources
- Implement caching strategies
- Optimize for multiple users

---

## 🎯 Quick Commands

### Local Testing
```bash
# Test the cloud app locally
streamlit run streamlit_cloud_app.py
```

### Deploy Updates
```bash
# Make changes and push
git add .
git commit -m "Update app"
git push origin main
```

### Check Status
```bash
# Check git status
git status

# View recent commits
git log --oneline -5
```

---

**🎉 Your OMR Evaluation System is now live on Streamlit Cloud!**

*Teachers can access it from anywhere and process OMR sheets efficiently.*
