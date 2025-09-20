# 🚀 Teacher Deployment Guide - OMR Evaluation System

## 📋 Quick Deployment Options

This guide provides multiple deployment options for teachers to get the OMR Evaluation System running quickly and easily.

## 🎯 Deployment Options

### Option 1: Local Development (Easiest)
**Best for**: Testing, small classes, personal use

### Option 2: Docker Deployment (Recommended)
**Best for**: Production use, multiple users, server deployment

### Option 3: Cloud Deployment
**Best for**: Large-scale deployment, remote access, enterprise use

---

## 🏠 Option 1: Local Development Deployment

### Prerequisites
- Python 3.8+ installed
- 4GB+ RAM
- 2GB+ free disk space

### Step 1: Download and Setup
```bash
# Download the project
git clone <repository-url>
cd OMR_Evaluation_System_Complete

# Or extract from ZIP file
# Unzip the downloaded file to a folder
```

### Step 2: Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Start the Teacher Interface
```bash
# Start the teacher interface
python teacher_launcher.py

# Or start the full system
python run.py both
```

### Step 4: Access the System
- **Teacher Interface**: http://localhost:8501
- **Full System**: http://localhost:8501 (Frontend) + http://localhost:8000 (Backend)

---

## 🐳 Option 2: Docker Deployment (Recommended)

### Prerequisites
- Docker installed
- Docker Compose installed
- 8GB+ RAM recommended

### Step 1: Setup Docker Environment
```bash
# Navigate to project directory
cd OMR_Evaluation_System_Complete

# Create environment file
cat > .env << EOF
# Database
DATABASE_URL=sqlite:///./omr_evaluation.db

# File Upload
UPLOAD_DIR=uploads
RESULTS_DIR=results
LOGS_DIR=logs

# Processing
MAX_FILE_SIZE_MB=50
BUBBLE_DETECTION_THRESHOLD=0.15
PROCESSING_TIMEOUT_SECONDS=300

# API
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_PORT=8501
EOF
```

### Step 2: Deploy with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Step 3: Access the System
- **Web Interface**: http://localhost
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs

### Step 4: Teacher Interface
```bash
# Start teacher interface specifically
docker-compose up -d omr-frontend

# Access teacher interface
# http://localhost:8501
```

---

## ☁️ Option 3: Cloud Deployment

### A. Streamlit Cloud (Easiest for Teachers)

#### Step 1: Prepare for Streamlit Cloud
```bash
# Use the minimal requirements
cp requirements_streamlit_cloud.txt requirements.txt

# Create streamlit_cloud_app.py
cp streamlit_cloud_app.py app.py
```

#### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Select the repository and branch
4. Set the main file path to `app.py`
5. Click "Deploy"

#### Step 3: Access Your App
- Your app will be available at: `https://your-app-name.streamlit.app`

### B. Heroku Deployment

#### Step 1: Create Heroku App
```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run teacher_interface.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Create runtime.txt
echo "python-3.9.18" > runtime.txt
```

#### Step 2: Deploy to Heroku
```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-omr-app

# Deploy
git add .
git commit -m "Deploy OMR system"
git push heroku main

# Open app
heroku open
```

### C. AWS EC2 Deployment

#### Step 1: Launch EC2 Instance
1. Launch Ubuntu 20.04 LTS instance
2. Configure security groups (ports 22, 80, 443, 8000, 8501)
3. Connect via SSH

#### Step 2: Setup on EC2
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker $USER

# Clone repository
git clone <your-repository-url>
cd OMR_Evaluation_System_Complete

# Deploy with Docker
docker-compose up -d
```

#### Step 3: Configure Domain (Optional)
```bash
# Install Nginx
sudo apt install nginx -y

# Configure Nginx
sudo nano /etc/nginx/sites-available/omr-evaluation
```

Nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/omr-evaluation /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🔧 Configuration Options

### Environment Variables
Create a `.env` file in your project root:

```bash
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

# Security (for production)
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,localhost
```

### Teacher-Specific Configuration
```bash
# Teacher interface settings
TEACHER_MODE=true
ENABLE_SAMPLE_DATA=true
MAX_STUDENTS_PER_BATCH=100
AUTO_SAVE_RESULTS=true
```

---

## 📊 Monitoring and Maintenance

### Health Checks
```bash
# Check if services are running
curl http://localhost:8000/health
curl http://localhost:8501

# Check Docker containers
docker-compose ps
docker-compose logs
```

### Backup Strategy
```bash
# Backup results
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/

# Backup uploads
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz uploads/

# Backup database (if using SQLite)
cp omr_evaluation.db omr_evaluation_backup_$(date +%Y%m%d).db
```

### Updates
```bash
# Update application
git pull origin main

# Restart services
docker-compose down
docker-compose up -d

# Or for local deployment
# Restart the teacher interface
python teacher_launcher.py
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port
sudo lsof -i :8000
sudo lsof -i :8501

# Kill process
sudo kill -9 <PID>
```

#### 2. Docker Issues
```bash
# Clean up Docker
docker-compose down
docker system prune -a
docker-compose up -d
```

#### 3. Permission Issues
```bash
# Fix permissions
sudo chown -R $USER:$USER .
chmod -R 755 uploads results logs
```

#### 4. Memory Issues
```bash
# Check memory usage
free -h
htop

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python teacher_launcher.py
```

---

## 🎯 Recommended Deployment for Teachers

### For Small Classes (1-50 students)
**Recommended**: Local Development
- Easy to setup
- No cloud costs
- Full control

### For Medium Classes (50-200 students)
**Recommended**: Docker Deployment
- Better performance
- Easy to scale
- Professional setup

### For Large Classes (200+ students)
**Recommended**: Cloud Deployment
- Scalable
- Remote access
- Professional hosting

---

## 📞 Support

### Getting Help
1. **Check logs**: Look in `logs/` directory for error messages
2. **Run tests**: `python test_teacher_system.py`
3. **Check requirements**: Ensure all dependencies are installed
4. **Review configuration**: Check `.env` file settings

### Contact Support
- Check the documentation
- Review troubleshooting guide
- Contact the development team

---

## 🎉 Success Checklist

After deployment, verify:
- [ ] Teacher interface loads at http://localhost:8501
- [ ] Can upload answer keys
- [ ] Can process OMR sheets
- [ ] Results are displayed correctly
- [ ] Export functionality works
- [ ] System handles multiple students

---

**🚀 Your OMR Evaluation System is ready for teachers to use!**

*Choose the deployment option that best fits your needs and start processing OMR sheets efficiently.*
