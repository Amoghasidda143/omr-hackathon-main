# 🚀 OMR Evaluation System - Deployment Guide

This guide provides comprehensive instructions for deploying the Automated OMR Evaluation & Scoring System in various environments.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Configuration](#configuration)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **CPU**: 2 cores minimum, 4 cores recommended

### Recommended Requirements
- **OS**: Ubuntu 20.04+ or Windows Server 2019+
- **Python**: 3.9 or higher
- **RAM**: 16GB or more
- **Storage**: 10GB+ SSD
- **CPU**: 8 cores or more
- **GPU**: Optional, for ML acceleration

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd omr-hackathon-main
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the System
```bash
# Start the complete system (Frontend + Backend)
python runner.py both

# Or start components separately:
# Frontend only
python runner.py frontend

# Backend only
python runner.py backend
```

### 4. Access the Application
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs

## 🏭 Production Deployment

### 1. Environment Setup

Create a production environment file:
```bash
# Create .env file
cat > .env << EOF
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/omr_evaluation

# File Upload
UPLOAD_DIR=/var/omr/uploads
RESULTS_DIR=/var/omr/results
LOGS_DIR=/var/omr/logs

# Processing
MAX_FILE_SIZE_MB=50
BUBBLE_DETECTION_THRESHOLD=0.15
PROCESSING_TIMEOUT_SECONDS=300

# API
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_PORT=8501

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,localhost
EOF
```

### 2. Database Setup

#### PostgreSQL (Recommended)
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE omr_evaluation;
CREATE USER omr_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE omr_evaluation TO omr_user;
\q
```

#### SQLite (Development)
```bash
# SQLite is used by default for development
# No additional setup required
```

### 3. Web Server Setup

#### Using Nginx (Recommended)
```bash
# Install Nginx
sudo apt-get install nginx

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/omr-evaluation
```

Nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend (Streamlit)
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

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/omr-evaluation /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Process Management

#### Using systemd
```bash
# Create systemd service for backend
sudo nano /etc/systemd/system/omr-backend.service
```

Backend service file:
```ini
[Unit]
Description=OMR Evaluation Backend
After=network.target

[Service]
Type=simple
User=omr
WorkingDirectory=/opt/omr-evaluation
Environment=PATH=/opt/omr-evaluation/venv/bin
ExecStart=/opt/omr-evaluation/venv/bin/python backend/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Create systemd service for frontend
sudo nano /etc/systemd/system/omr-frontend.service
```

Frontend service file:
```ini
[Unit]
Description=OMR Evaluation Frontend
After=network.target

[Service]
Type=simple
User=omr
WorkingDirectory=/opt/omr-evaluation
Environment=PATH=/opt/omr-evaluation/venv/bin
ExecStart=/opt/omr-evaluation/venv/bin/streamlit run streamlit_app.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start services
sudo systemctl enable omr-backend omr-frontend
sudo systemctl start omr-backend omr-frontend
```

## 🐳 Docker Deployment

### 1. Build Docker Images

```bash
# Build the application image
docker build -t omr-evaluation:latest .

# Build with specific tag
docker build -t omr-evaluation:v1.0.0 .
```

### 2. Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Docker Compose Configuration

```yaml
version: '3.8'

services:
  omr-backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://omr_user:password@db:5432/omr_evaluation
    depends_on:
      - db
    volumes:
      - ./uploads:/app/uploads
      - ./results:/app/results
      - ./logs:/app/logs

  omr-frontend:
    build: .
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://omr-backend:8000
    depends_on:
      - omr-backend
    command: streamlit run streamlit_app.py --server.port=8501

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=omr_evaluation
      - POSTGRES_USER=omr_user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - omr-frontend
      - omr-backend

volumes:
  postgres_data:
```

## ☁️ Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance Setup
```bash
# Launch EC2 instance (Ubuntu 20.04)
# Install Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Clone repository
git clone <repository-url>
cd omr-evaluation

# Deploy with Docker Compose
docker-compose up -d
```

#### 2. RDS Database Setup
```bash
# Create RDS PostgreSQL instance
# Update DATABASE_URL in .env file
DATABASE_URL=postgresql://username:password@your-rds-endpoint:5432/omr_evaluation
```

#### 3. S3 Storage Setup
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Create S3 bucket for file storage
aws s3 mb s3://your-omr-bucket
```

### Google Cloud Platform

#### 1. App Engine Deployment
```yaml
# app.yaml
runtime: python39

env_variables:
  DATABASE_URL: "postgresql://username:password@/database?host=/cloudsql/project:region:instance"

handlers:
- url: /.*
  script: auto
```

#### 2. Cloud Run Deployment
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/omr-evaluation

# Deploy to Cloud Run
gcloud run deploy omr-evaluation --image gcr.io/PROJECT-ID/omr-evaluation --platform managed
```

### Azure Deployment

#### 1. Container Instances
```bash
# Create resource group
az group create --name omr-rg --location eastus

# Deploy container
az container create --resource-group omr-rg --name omr-evaluation --image omr-evaluation:latest --ports 8000 8501
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | Database connection string | `sqlite:///./omr_evaluation.db` | No |
| `UPLOAD_DIR` | Directory for uploaded files | `uploads` | No |
| `RESULTS_DIR` | Directory for results | `results` | No |
| `LOGS_DIR` | Directory for logs | `logs` | No |
| `MAX_FILE_SIZE_MB` | Maximum file size in MB | `50` | No |
| `BUBBLE_DETECTION_THRESHOLD` | Bubble detection threshold | `0.15` | No |
| `PROCESSING_TIMEOUT_SECONDS` | Processing timeout | `300` | No |
| `API_HOST` | API host address | `0.0.0.0` | No |
| `API_PORT` | API port | `8000` | No |
| `FRONTEND_PORT` | Frontend port | `8501` | No |
| `SECRET_KEY` | Secret key for security | Random | Yes (Production) |
| `ALLOWED_HOSTS` | Allowed host names | `*` | Yes (Production) |

### Answer Key Configuration

Answer keys are stored in JSON format in the `answer_keys/` directory:

```json
{
  "version": "v1",
  "subjects": {
    "Mathematics": {
      "questions": [1, 2, 3, ..., 20],
      "answers": ["A", "B", "C", "D", ...]
    },
    "Physics": {
      "questions": [21, 22, 23, ..., 40],
      "answers": ["A", "B", "C", "D", ...]
    }
  }
}
```

## 📊 Monitoring & Maintenance

### 1. Logging

The system generates logs in the following locations:
- **Application logs**: `logs/app.log`
- **Error logs**: `logs/error.log`
- **Processing logs**: `logs/processing.log`

### 2. Health Checks

```bash
# Check backend health
curl http://localhost:8000/health

# Check frontend
curl http://localhost:8501
```

### 3. Performance Monitoring

```bash
# Monitor system resources
htop

# Monitor disk usage
df -h

# Monitor database
psql -d omr_evaluation -c "SELECT * FROM pg_stat_activity;"
```

### 4. Backup Strategy

```bash
# Database backup
pg_dump omr_evaluation > backup_$(date +%Y%m%d).sql

# File backup
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz uploads/
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/
```

### 5. Updates

```bash
# Update application
git pull origin main
pip install -r requirements.txt

# Restart services
sudo systemctl restart omr-backend omr-frontend

# Or with Docker
docker-compose pull
docker-compose up -d
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port
sudo lsof -i :8000
sudo lsof -i :8501

# Kill process
sudo kill -9 <PID>
```

#### 2. Database Connection Issues
```bash
# Check database status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U omr_user -d omr_evaluation
```

#### 3. File Permission Issues
```bash
# Fix permissions
sudo chown -R omr:omr /opt/omr-evaluation
sudo chmod -R 755 /opt/omr-evaluation
```

#### 4. Memory Issues
```bash
# Check memory usage
free -h

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python runner.py both
```

### Support

For additional support:
1. Check the logs in `logs/` directory
2. Run the test suite: `python test_omr_system.py`
3. Check system requirements
4. Review configuration settings

## 📈 Performance Optimization

### 1. Database Optimization
- Use connection pooling
- Create appropriate indexes
- Regular VACUUM and ANALYZE

### 2. File Storage Optimization
- Use SSD storage for better I/O
- Implement file compression
- Regular cleanup of old files

### 3. Processing Optimization
- Use multiprocessing for batch operations
- Implement caching for frequently accessed data
- Optimize image processing parameters

### 4. Network Optimization
- Use CDN for static files
- Implement gzip compression
- Use HTTP/2 if available

---

**Built with ❤️ for automated education assessment**