"""
Utility functions for OMR Evaluation System.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Default log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"omr_evaluation_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def validate_json_file(file_path: str) -> tuple[bool, str]:
    """
    Validate JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True, "Valid JSON file"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except FileNotFoundError:
        return False, "File not found"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def create_directory_structure():
    """Create necessary directory structure."""
    directories = [
        "uploads",
        "results",
        "results/exports",
        "results/processed_images",
        "answer_keys",
        "logs",
        "models",
        "static",
        "static/css",
        "static/js",
        "static/images"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (with dot)
    """
    return os.path.splitext(filename)[1].lower()


def is_image_file(filename: str) -> bool:
    """
    Check if file is an image.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if file is an image
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return get_file_extension(filename) in image_extensions


def is_pdf_file(filename: str) -> bool:
    """
    Check if file is a PDF.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if file is a PDF
    """
    return get_file_extension(filename) == '.pdf'


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    """
    Generate unique filename.
    
    Args:
        original_filename: Original filename
        prefix: Optional prefix
        
    Returns:
        Unique filename
    """
    name, ext = os.path.splitext(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = os.urandom(4).hex()
    
    if prefix:
        return f"{prefix}_{timestamp}_{unique_id}{ext}"
    else:
        return f"{name}_{timestamp}_{unique_id}{ext}"


def safe_filename(filename: str) -> str:
    """
    Make filename safe for filesystem.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    safe_name = filename
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip(' .')
    
    # Ensure filename is not empty
    if not safe_name:
        safe_name = "unnamed_file"
    
    return safe_name


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        System information dictionary
    """
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
    }


def calculate_processing_time(start_time: datetime, end_time: Optional[datetime] = None) -> float:
    """
    Calculate processing time in seconds.
    
    Args:
        start_time: Start time
        end_time: End time (defaults to now)
        
    Returns:
        Processing time in seconds
    """
    if end_time is None:
        end_time = datetime.now()
    
    return (end_time - start_time).total_seconds()


def format_processing_time(seconds: float) -> str:
    """
    Format processing time in human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    elif seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} h"


def validate_email(email: str) -> bool:
    """
    Validate email address.
    
    Args:
        email: Email address
        
    Returns:
        True if email is valid
    """
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone(phone: str) -> bool:
    """
    Validate phone number.
    
    Args:
        phone: Phone number
        
    Returns:
        True if phone is valid
    """
    import re
    
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    # Check if it's a valid length (7-15 digits)
    return 7 <= len(digits_only) <= 15


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().isoformat()


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse timestamp string to datetime object.
    
    Args:
        timestamp_str: Timestamp string
        
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        return None


def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: Input list
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def retry_on_exception(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on exception.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator