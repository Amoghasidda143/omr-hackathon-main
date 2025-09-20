"""
OMR Evaluation System Backend Package.
"""

from .main import app
from .database import get_db, create_tables
from .models import *
from .schemas import *
from .services import OMRProcessingService, DatabaseService, FileService
from .utils import setup_logging, get_logger

__version__ = "1.0.0"
__author__ = "OMR Evaluation System Team"
