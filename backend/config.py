import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    # API settings
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', 30))
    API_URL = "https://api.huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On"
    
    # Upload settings
    UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', 'temp'))
    MAX_FILESIZE = int(os.getenv('MAX_FILESIZE', 5 * 1024 * 1024))  # Default 5MB
    ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'jpg,jpeg,png').split(','))
    
    # Rate limiting
    UPLOAD_RATE_LIMIT = "10 per minute"
    PROCESS_RATE_LIMIT = "5 per minute"
    
    # Image compression
    DEFAULT_QUALITY = 70
    MAX_DIMENSION = 1024
    
    @classmethod
    def init_app(cls):
        """Initialize application configuration"""
        cls.UPLOAD_FOLDER.mkdir(exist_ok=True)
        return cls 