import pytest
import requests
from PIL import Image
import io
import os
import redis

def test_production_config():
    """Test production configuration"""
    assert os.getenv('FLASK_ENV') == 'production'
    assert os.getenv('HUGGINGFACE_API_KEY') is not None
    assert os.getenv('REDIS_URL') is not None

def test_ssl_redirect():
    """Test SSL redirect in production"""
    response = requests.get('http://yourdomain.com', allow_redirects=False)
    assert response.status_code == 301
    assert response.headers['Location'].startswith('https://')

def test_redis_connection():
    """Test Redis connection"""
    r = redis.from_url(os.getenv('REDIS_URL'))
    assert r.ping()

def test_file_permissions():
    """Test file permissions in production"""
    temp_dir = Path('/app/temp')
    assert temp_dir.exists()
    assert os.access(temp_dir, os.W_OK)

@pytest.mark.production
def test_production_upload():
    """Test file upload in production"""
    # Create test image
    img = Image.new('RGB', (100, 100), color='red')
    output = io.BytesIO()
    img.save(output, format='JPEG')
    output.seek(0)
    
    response = requests.post(
        'https://yourdomain.com/upload-person-image',
        files={'image': output}
    )
    assert response.status_code == 200

@pytest.mark.production
def test_production_processing():
    """Test image processing in production"""
    response = requests.post(
        'https://yourdomain.com/process-images',
        json={
            'person_image': 'test.jpg',
            'garment_image': 'test.jpg'
        }
    )
    assert response.status_code == 200

@pytest.mark.production
def test_rate_limiting():
    """Test rate limiting in production"""
    responses = []
    for _ in range(12):  # Over the limit
        responses.append(
            requests.post('https://yourdomain.com/upload-person-image')
        )
    
    assert any(r.status_code == 429 for r in responses)

@pytest.mark.production
def test_error_logging():
    """Test error logging in production"""
    import logging
    
    # Verify log configuration
    logger = logging.getLogger('app')
    assert logger.level == logging.INFO
    assert any(h.stream.name == '<stdout>' for h in logger.handlers) 