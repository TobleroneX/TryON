from config import Config

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Sicherheitseinstellungen
    CSRF_ENABLED = True
    SSL_REQUIRED = True
    
    # Performance
    CACHE_TYPE = "redis"
    CACHE_REDIS_URL = os.getenv('REDIS_URL')
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_TO_STDOUT = True 