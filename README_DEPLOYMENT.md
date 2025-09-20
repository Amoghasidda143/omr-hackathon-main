# 🚀 OMR Evaluation System - Deployment Guide

## 📋 Deployment Options

This guide covers deploying the OMR Evaluation System using **Streamlit Cloud** or **HuggingFace Spaces** as required.

## 🌐 Streamlit Cloud Deployment

### Step 1: Prepare Repository

1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Create Streamlit App**: The main entry point is `streamlit_app.py`
3. **Dependencies**: Use `requirements_streamlit.txt` for dependencies

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Configure deployment**:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` or `master`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose a custom URL (optional)

5. **Advanced Settings**:
   - **Python version**: 3.8 or higher
   - **Dependencies**: `requirements_streamlit.txt`

### Step 3: Deploy

Click **"Deploy!"** and wait for the deployment to complete.

## 🤗 HuggingFace Spaces Deployment

### Step 1: Create Space

1. **Go to HuggingFace Spaces**: https://huggingface.co/spaces
2. **Click "Create new Space"**
3. **Configure**:
   - **Space name**: `omr-evaluation-system`
   - **License**: MIT
   - **SDK**: Streamlit
   - **Hardware**: CPU Basic (free) or upgrade if needed

### Step 2: Upload Files

1. **Clone the space**:
   ```bash
   git clone https://huggingface.co/spaces/yourusername/omr-evaluation-system
   cd omr-evaluation-system
   ```

2. **Copy files**:
   - Copy `streamlit_app.py` as `app.py`
   - Copy `requirements_streamlit.txt` as `requirements.txt`
   - Copy all necessary directories (`omr_processor/`, `utils/`, etc.)

3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push
   ```

## 📁 Required Files for Deployment

Ensure these files are in your repository root:

```
├── streamlit_app.py          # Main Streamlit application
├── requirements_streamlit.txt # Dependencies
├── packages.txt              # Alternative dependencies file
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── omr_processor/           # OMR processing modules
├── utils/                   # Utility functions
└── README_DEPLOYMENT.md     # This file
```

## 🔧 Configuration

### Streamlit Configuration (`.streamlit/config.toml`)

```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[browser]
gatherUsageStats = false
```

### Environment Variables

Set these in your deployment platform:

```bash
# Optional: Custom configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🧪 Testing Deployment

### Local Testing

1. **Install dependencies**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Run locally**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Test functionality**:
   - Upload sample OMR sheets
   - Process and view results
   - Test export functionality

### Production Testing

1. **Access deployed URL**
2. **Test all features**:
   - Dashboard loads correctly
   - Upload functionality works
   - Processing completes successfully
   - Results display properly
   - Export functions work

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
- Ensure all required modules are in `requirements_streamlit.txt`
- Check that all Python files are properly uploaded

**2. File Not Found Errors**
- Verify all directories and files are included
- Check file paths are correct

**3. Memory Issues**
- Reduce image processing size if needed
- Optimize batch processing

**4. Slow Loading**
- Check if all dependencies are installed
- Monitor resource usage

### Debug Mode

Add debug information to your app:

```python
import streamlit as st

# Add debug info
st.sidebar.write("Debug Info:")
st.sidebar.write(f"Python version: {sys.version}")
st.sidebar.write(f"OpenCV version: {cv2.__version__}")
```

## 📊 Performance Optimization

### For Streamlit Cloud

1. **Optimize imports**: Import only what you need
2. **Cache expensive operations**: Use `@st.cache_data`
3. **Limit file sizes**: Compress images before processing
4. **Batch processing**: Process multiple files efficiently

### For HuggingFace Spaces

1. **Use appropriate hardware**: Upgrade if needed
2. **Optimize memory usage**: Process files in chunks
3. **Implement timeouts**: Prevent long-running processes

## 🔒 Security Considerations

1. **File upload limits**: Set reasonable file size limits
2. **Input validation**: Validate all user inputs
3. **Error handling**: Don't expose sensitive information
4. **Rate limiting**: Implement if needed

## 📈 Monitoring

### Streamlit Cloud

- Monitor usage in the Streamlit Cloud dashboard
- Check logs for errors
- Monitor resource usage

### HuggingFace Spaces

- Check the "Logs" tab for errors
- Monitor "Metrics" for performance
- Use "Files" tab to verify file structure

## 🎯 Final Checklist

Before submitting:

- [ ] App is deployed and accessible publicly
- [ ] All features work correctly
- [ ] Upload functionality works
- [ ] Processing completes successfully
- [ ] Results display properly
- [ ] Export functions work
- [ ] App is responsive and user-friendly
- [ ] No critical errors in logs
- [ ] Performance is acceptable

## 📞 Support

If you encounter issues:

1. Check the deployment platform logs
2. Test locally first
3. Verify all dependencies are correct
4. Check file structure and paths
5. Contact platform support if needed

---

**🎉 Your OMR Evaluation System should now be deployed and accessible publicly!**
