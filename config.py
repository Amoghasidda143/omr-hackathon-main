
"""
Configuration settings for OMR Evaluation System.
"""

import os
from typing import Dict, Any

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./omr_evaluation.db")

# File upload configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
EXPORTS_DIR = os.getenv("EXPORTS_DIR", "results/exports")
ANSWER_KEYS_DIR = os.getenv("ANSWER_KEYS_DIR", "answer_keys")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")

# File size limits
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Supported file formats
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
SUPPORTED_DOCUMENT_FORMATS = {'.pdf'}
SUPPORTED_FORMATS = SUPPORTED_IMAGE_FORMATS | SUPPORTED_DOCUMENT_FORMATS

# Image processing configuration
TARGET_IMAGE_WIDTH = int(os.getenv("TARGET_IMAGE_WIDTH", "800"))
TARGET_IMAGE_HEIGHT = int(os.getenv("TARGET_IMAGE_HEIGHT", "1000"))
MIN_IMAGE_DIMENSIONS = (100, 100)
MAX_IMAGE_DIMENSIONS = (5000, 5000)

# OMR processing configuration
BUBBLE_DETECTION_THRESHOLD = float(os.getenv("BUBBLE_DETECTION_THRESHOLD", "0.15"))
MIN_BUBBLE_AREA = int(os.getenv("MIN_BUBBLE_AREA", "50"))
MAX_BUBBLE_AREA = int(os.getenv("MAX_BUBBLE_AREA", "500"))
ASPECT_RATIO_TOLERANCE = float(os.getenv("ASPECT_RATIO_TOLERANCE", "0.3"))

# Processing timeouts
PROCESSING_TIMEOUT_SECONDS = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", "300"))
BATCH_PROCESSING_TIMEOUT_SECONDS = int(os.getenv("BATCH_PROCESSING_TIMEOUT_SECONDS", "1800"))

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

# Streamlit configuration
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE_MAX_SIZE = int(os.getenv("LOG_FILE_MAX_SIZE", "10485760"))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
CORS_ALLOW_METHODS = os.getenv("CORS_ALLOW_METHODS", "*").split(",")
CORS_ALLOW_HEADERS = os.getenv("CORS_ALLOW_HEADERS", "*").split(",")

# Export configuration
EXPORT_RETENTION_DAYS = int(os.getenv("EXPORT_RETENTION_DAYS", "30"))
MAX_EXPORT_RECORDS = int(os.getenv("MAX_EXPORT_RECORDS", "10000"))

# Performance configuration
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CONCURRENT_PROCESSING = int(os.getenv("MAX_CONCURRENT_PROCESSING", "5"))

# Quality assurance configuration
ENABLE_QUALITY_CHECKS = os.getenv("ENABLE_QUALITY_CHECKS", "true").lower() == "true"
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.8"))
ENABLE_AUDIT_LOGGING = os.getenv("ENABLE_AUDIT_LOGGING", "true").lower() == "true"

# Default answer key configuration
DEFAULT_SUBJECTS = [
    "Mathematics",
    "Physics", 
    "Chemistry",
    "Biology",
    "General_Knowledge"
]
DEFAULT_QUESTIONS_PER_SUBJECT = 20
DEFAULT_TOTAL_QUESTIONS = 100

# Error tolerance configuration
MAX_ERROR_RATE = float(os.getenv("MAX_ERROR_RATE", "0.005"))  # 0.5%
ENABLE_AUTO_RETRY = os.getenv("ENABLE_AUTO_RETRY", "true").lower() == "true"
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))

# Notification configuration
ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
NOTIFICATION_EMAIL = os.getenv("NOTIFICATION_EMAIL", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# Development configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
RELOAD = os.getenv("RELOAD", "false").lower() == "true"
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "true").lower() == "true"

# Create directories if they don't exist
def create_directories():
    """Create necessary directories."""
    directories = [
        UPLOAD_DIR,
        RESULTS_DIR,
        EXPORTS_DIR,
        ANSWER_KEYS_DIR,
        MODELS_DIR,
        LOGS_DIR,
        "static",
        "static/css",
        "static/js",
        "static/images"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Configuration validation
def validate_config() -> Dict[str, Any]:
    """
    Validate configuration settings.
    
    Returns:
        Dictionary of validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required directories
    try:
        create_directories()
    except Exception as e:
        validation_results["errors"].append(f"Failed to create directories: {str(e)}")
        validation_results["valid"] = False
    
    # Check file size limits
    if MAX_FILE_SIZE_MB < 1 or MAX_FILE_SIZE_MB > 100:
        validation_results["warnings"].append("MAX_FILE_SIZE_MB should be between 1 and 100")
    
    # Check processing timeouts
    if PROCESSING_TIMEOUT_SECONDS < 30:
        validation_results["warnings"].append("PROCESSING_TIMEOUT_SECONDS should be at least 30")
    
    # Check API configuration
    if API_PORT < 1024 or API_PORT > 65535:
        validation_results["errors"].append("API_PORT must be between 1024 and 65535")
        validation_results["valid"] = False
    
    # Check Streamlit configuration
    if STREAMLIT_PORT < 1024 or STREAMLIT_PORT > 65535:
        validation_results["errors"].append("STREAMLIT_PORT must be between 1024 and 65535")
        validation_results["valid"] = False
    
    # Check security settings
    if SECRET_KEY == "your-secret-key-change-in-production":
        validation_results["warnings"].append("SECRET_KEY should be changed in production")
    
    # Check notification settings
    if ENABLE_NOTIFICATIONS and not NOTIFICATION_EMAIL:
        validation_results["warnings"].append("NOTIFICATION_EMAIL should be set if notifications are enabled")
    
    return validation_results

# Get configuration as dictionary
def get_config_dict() -> Dict[str, Any]:
    """
    Get all configuration as a dictionary.
    Returns:
        Configuration dictionary
    """
    return {
        "database_url": DATABASE_URL,
        "upload_dir": UPLOAD_DIR,
        "results_dir": RESULTS_DIR,
        "exports_dir": EXPORTS_DIR,
        "answer_keys_dir": ANSWER_KEYS_DIR,
        "models_dir": MODELS_DIR,
        "logs_dir": LOGS_DIR,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "supported_formats": list(SUPPORTED_FORMATS),
        "target_image_width": TARGET_IMAGE_WIDTH,
        "target_image_height": TARGET_IMAGE_HEIGHT,
        "bubble_detection_threshold": BUBBLE_DETECTION_THRESHOLD,
        "processing_timeout_seconds": PROCESSING_TIMEOUT_SECONDS,
        "api_host": API_HOST,
        "api_port": API_PORT,
        "streamlit_host": STREAMLIT_HOST,
        "streamlit_port": STREAMLIT_PORT,
        "log_level": LOG_LEVEL,
        "cors_origins": CORS_ORIGINS,
        "export_retention_days": EXPORT_RETENTION_DAYS,
        "enable_caching": ENABLE_CACHING,
        "enable_quality_checks": ENABLE_QUALITY_CHECKS,
        "min_confidence_threshold": MIN_CONFIDENCE_THRESHOLD,
        "max_error_rate": MAX_ERROR_RATE,
        "debug": DEBUG,
        "reload": RELOAD,
        "enable_docs": ENABLE_DOCS
    }
