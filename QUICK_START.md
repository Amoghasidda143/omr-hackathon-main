# 🚀 OMR Evaluation System - Quick Start Guide

## ⚡ Get Started in 5 Minutes

### **Option 1: Local Development (Recommended for Testing)**

```bash
# 1. Navigate to project directory
cd omr-hackathon

# 2. Install and start everything
python runner.py both

# 3. Access the application
# Frontend: http://localhost:8501
# Backend: http://localhost:8000
```

### **Option 2: Docker Deployment (Recommended for Production)**

```bash
# 1. Make sure Docker is installed
docker --version
docker-compose --version

# 2. Deploy with Docker
python deploy.py docker

# 3. Access the application
# Frontend: http://localhost:8501
# Backend: http://localhost:8000
```

---

## 🎯 What You'll See

### **Frontend Dashboard (http://localhost:8501)**
- **Dashboard**: Real-time metrics and analytics
- **Upload & Process**: Upload OMR sheets for processing
- **Results Analysis**: View detailed results and export data
- **System Settings**: Configure answer keys and settings

### **Backend API (http://localhost:8000)**
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health
- **REST Endpoints**: 20+ endpoints for all operations

---

## 📤 How to Use

### **1. Upload OMR Sheets**
1. Go to "Upload & Process" tab
2. Choose "Single Upload" or "Batch Upload"
3. Select OMR image files (JPG, PNG, PDF)
4. Choose sheet version
5. Click "Process Sheet(s)"

### **2. View Results**
1. Go to "Results Analysis" tab
2. View score distributions and analytics
3. Export results as CSV, Excel, or JSON
4. Analyze subject-wise performance

### **3. Configure System**
1. Go to "System Settings" tab
2. Manage answer keys
3. Adjust processing parameters
4. View system information

---

## 🔧 Troubleshooting

### **Common Issues**

**"Backend API not available"**
```bash
# Check if backend is running
python runner.py status

# Restart backend
python runner.py backend
```

**"Docker services not starting"**
```bash
# Check Docker status
docker-compose ps

# View logs
docker-compose logs

# Restart services
docker-compose restart
```

**"Import errors"**
```bash
# Install dependencies
python runner.py install

# Or manually
pip install -r requirements.txt
```

### **Check System Status**
```bash
# Check all services
python runner.py status

# Or use deployment script
python deploy.py status
```

---

## 📊 Sample Data

### **Test with Sample OMR Sheets**
1. Create sample OMR sheets with 100 questions (5 subjects × 20 questions)
2. Use the provided answer key format
3. Upload and test the processing

### **Answer Key Format**
```json
{
  "v1": {
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "A",
    "6": "B", "7": "C", "8": "D", "9": "A", "10": "B",
    // ... continue for all 100 questions
  }
}
```

---

## 🎉 Success!

Once everything is running, you'll have:

✅ **Complete OMR Processing System**  
✅ **Modern Web Dashboard**  
✅ **REST API Backend**  
✅ **Batch Processing**  
✅ **Real-time Analytics**  
✅ **Export Capabilities**  
✅ **Production-Ready Architecture**  

**Ready to process thousands of OMR sheets efficiently!** 🎯

---

## 📞 Need Help?

- **Documentation**: Check `README.md` for detailed docs
- **Deployment**: See `DEPLOYMENT_GUIDE.md` for production setup
- **Issues**: Run `python test_system.py` to diagnose problems
- **Support**: Check logs in the `logs/` directory
