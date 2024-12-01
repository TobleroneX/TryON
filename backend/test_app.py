import pytest
from app import app
import io
from pathlib import Path
import requests
from PIL import Image
import os
import unittest.mock
import logging
from skimage.metrics import structural_similarity as ssim
import numpy as np
import time

@pytest.fixture
def client():
    """Create a test client for the app"""
    with app.test_client() as client:
        yield client

@pytest.fixture(autouse=True)
def setup_temp_dirs(tmp_path):
    """Automatically set up temporary directories before each test"""
    # Verwende temporäres Verzeichnis für Tests
    app.config['UPLOAD_FOLDER'] = tmp_path / 'temp'
    app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
    yield
    # Cleanup passiert automatisch mit tmp_path

@pytest.fixture
def test_images(tmp_path):
    """Create test images in a temporary directory"""
    # Create test images directory
    images_dir = tmp_path / "test_images"
    images_dir.mkdir()
    
    # Create test images
    jpeg_path = images_dir / "test.jpg"
    png_path = images_dir / "test.png"
    bmp_path = images_dir / "test.bmp"
    
    # Create JPEG test image
    img = Image.new('RGB', (100, 100), color='red')
    img.save(jpeg_path, 'JPEG')
    
    # Create PNG test image
    img = Image.new('RGB', (100, 100), color='blue')
    img.save(png_path, 'PNG')
    
    # Create BMP test image
    img = Image.new('RGB', (100, 100), color='green')
    img.save(bmp_path, 'BMP')
    
    yield {
        'dir': images_dir,
        'jpeg': jpeg_path,
        'png': png_path,
        'bmp': bmp_path
    }
    
    # Cleanup happens automatically with tmp_path

@pytest.mark.upload
@pytest.mark.parametrize('image_type', ['jpeg', 'png'])
def test_valid_image_upload(client, test_images, image_type):
    """Test uploading valid image formats"""
    with open(test_images[image_type], 'rb') as img:
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, f'test.{image_type}')}
        )
    assert response.status_code == 200
    assert 'filename' in response.json

@pytest.mark.upload
def test_invalid_format_upload(client, test_images):
    """Test uploading invalid image format (BMP)"""
    with open(test_images['bmp'], 'rb') as img:
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, 'test.bmp')}
        )
    assert response.status_code == 400
    assert 'Invalid image format' in response.json['error']

def test_upload_non_image(client):
    """Test uploading non-image content"""
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (io.BytesIO(b"NotAnImage"), 'test.txt')}
    )
    assert response.status_code == 400
    assert 'Invalid image' in response.json['error']

def test_upload_large_image(client, tmp_path):
    """Test uploading image exceeding size limit"""
    large_file = tmp_path / "large.jpg"
    with open(large_file, 'wb') as f:
        f.write(b'0' * (6 * 1024 * 1024))  # 6MB file
    
    with open(large_file, 'rb') as img:
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, 'large.jpg')}
        )
    assert response.status_code == 400
    assert 'size exceeds' in response.json['error']

@pytest.mark.process
def test_successful_processing(client, test_images, monkeypatch):
    """Test successful end-to-end image processing"""
    # Upload person image
    with open(test_images['jpeg'], 'rb') as img:
        person_response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, 'person.jpg')}
        )
    assert person_response.status_code == 200
    person_filename = person_response.json['filename']
    
    # Upload garment image
    with open(test_images['jpeg'], 'rb') as img:
        garment_response = client.post(
            '/upload-garment-image',
            content_type='multipart/form-data',
            data={'image': (img, 'garment.jpg')}
        )
    assert garment_response.status_code == 200
    garment_filename = garment_response.json['filename']
    
    # Mock API response
    mock_response = b'mock_processed_image_data'
    
    class MockResponse:
        status_code = 200
        content = mock_response
        
        def raise_for_status(self):
            pass
    
    def mock_post(*args, **kwargs):
        return MockResponse()
    
    monkeypatch.setattr(requests, 'post', mock_post)
    
    # Process images
    response = client.post(
        '/process-images',
        json={
            'person_image': person_filename,
            'garment_image': garment_filename
        }
    )
    
    assert response.status_code == 200
    assert 'result' in response.json
    assert response.json['result'] == mock_response.decode('latin1')

@pytest.mark.api
def test_api_error(client, test_images, monkeypatch):
    """Test handling of API errors"""
    def mock_post(*args, **kwargs):
        raise requests.exceptions.RequestException("API Error")
    
    monkeypatch.setattr(requests, 'post', mock_post)
    
    response = client.post(
        '/process-images',
        json={
            'person_image': 'test.jpg',
            'garment_image': 'test.jpg'
        }
    )
    assert response.status_code == 500
    assert 'API error' in response.json['error']

@pytest.mark.api
def test_rate_limit_upload(client, test_images):
    """Test rate limiting for upload endpoints"""
    # Mock the limiter to make tests faster
    with unittest.mock.patch('flask_limiter.Limiter.hit') as mock_hit:
        mock_hit.return_value = True
        for i in range(11):
            with open(test_images['jpeg'], 'rb') as img:
                response = client.post(
                    '/upload-person-image',
                    content_type='multipart/form-data',
                    data={'image': (img, 'test.jpg')}
                )
                if i < 10:
                    assert response.status_code == 200
                else:
                    assert response.status_code == 429

@pytest.mark.api
def test_rate_limit_process(client):
    """Test rate limiting for process endpoint"""
    # Make 6 requests (1 over limit)
    for i in range(6):
        response = client.post(
            '/process-images',
            json={
                'person_image': 'test.jpg',
                'garment_image': 'test.jpg'
            }
        )
        if i < 5:
            assert response.status_code in [200, 400, 404]  # Actual response doesn't matter
        else:
            assert response.status_code == 429
            assert 'Rate limit exceeded' in response.json['error']

def test_image_compression(client, test_images):
    """Test that uploaded images are compressed"""
    # Create a large but valid image
    large_image = Image.new('RGB', (2000, 2000), color='red')
    output = io.BytesIO()
    large_image.save(output, format='JPEG', quality=100)
    output.seek(0)
    
    # Upload the image
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'large.jpg')}
    )
    
    assert response.status_code == 200
    filename = response.json['filename']
    
    # Check the saved file size
    saved_path = app.config['UPLOAD_FOLDER'] / filename
    assert saved_path.exists()
    
    # Verify file size is under limit
    assert saved_path.stat().st_size <= 500 * 1024  # 500KB

def test_compression_maintains_quality(client, test_images):
    """Test that compression maintains reasonable image quality"""
    with open(test_images['jpeg'], 'rb') as img:
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, 'test.jpg')}
        )
    
    assert response.status_code == 200
    filename = response.json['filename']
    
    # Open compressed image and check dimensions
    saved_path = app.config['UPLOAD_FOLDER'] / filename
    compressed_img = Image.open(saved_path)
    
    # Verify dimensions are maintained
    original_img = Image.open(test_images['jpeg'])
    assert compressed_img.size == original_img.size 

def test_missing_image_file(client):
    """Test error when image file is missing"""
    response = client.post('/upload-person-image')
    assert response.status_code == 400
    assert response.json['error'] == 'No image provided'

def test_empty_filename(client):
    """Test error when filename is empty"""
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (io.BytesIO(b""), '')}
    )
    assert response.status_code == 400
    assert response.json['error'] == 'No file selected'

def test_process_invalid_json(client):
    """Test processing with invalid JSON data"""
    response = client.post(
        '/process-images',
        data='invalid json',
        content_type='application/json'
    )
    assert response.status_code == 400
    assert 'Invalid JSON format' in response.json['error']

def test_cleanup_error_handling(client, test_images, monkeypatch):
    """Test handling of cleanup errors"""
    # Mock unlink to raise an error
    def mock_unlink(*args):
        raise OSError("Cleanup error")
    
    monkeypatch.setattr(Path, 'unlink', mock_unlink)
    
    # Upload and process images
    with open(test_images['jpeg'], 'rb') as img:
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, 'test.jpg')}
        )
        assert response.status_code == 200

@pytest.mark.integration
def test_end_to_end_flow(client, test_images, monkeypatch):
    """Test complete end-to-end flow with realistic data"""
    # Mock API response with realistic data
    class MockResponse:
        status_code = 200
        content = b'processed_image_data_base64'
        
        def raise_for_status(self):
            pass
    
    # Mock API call
    def mock_post(*args, **kwargs):
        # Verify API is called with correct data
        assert 'files' in kwargs
        assert 'person_image' in kwargs['files']
        assert 'garment_image' in kwargs['files']
        return MockResponse()
    
    monkeypatch.setattr(requests, 'post', mock_post)
    
    # Step 1: Upload person image
    with open(test_images['jpeg'], 'rb') as img:
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, 'person.jpg')}
        )
        assert response.status_code == 200
        assert 'filename' in response.json
        person_filename = response.json['filename']
        
        # Verify file was saved
        person_path = app.config['UPLOAD_FOLDER'] / person_filename
        assert person_path.exists()
        assert person_path.stat().st_size > 0
    
    # Step 2: Upload garment image
    with open(test_images['jpeg'], 'rb') as img:
        response = client.post(
            '/upload-garment-image',
            content_type='multipart/form-data',
            data={'image': (img, 'garment.jpg')}
        )
        assert response.status_code == 200
        assert 'filename' in response.json
        garment_filename = response.json['filename']
        
        # Verify file was saved
        garment_path = app.config['UPLOAD_FOLDER'] / garment_filename
        assert garment_path.exists()
        assert garment_path.stat().st_size > 0
    
    # Step 3: Process images
    response = client.post(
        '/process-images',
        json={
            'person_image': person_filename,
            'garment_image': garment_filename
        }
    )
    
    # Verify successful processing
    assert response.status_code == 200
    assert 'result' in response.json
    assert response.json['result'] == MockResponse.content.decode('latin1')
    
    # Verify cleanup
    assert not person_path.exists()
    assert not garment_path.exists()

@pytest.mark.integration
def test_end_to_end_error_handling(client, test_images, monkeypatch):
    """Test error handling in end-to-end flow"""
    # Step 1: Upload invalid person image
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (io.BytesIO(b"NotAnImage"), 'person.txt')}
    )
    assert response.status_code == 400
    assert 'Invalid image' in response.json['error']
    
    # Step 2: Upload valid images but API fails
    def mock_post_error(*args, **kwargs):
        raise requests.exceptions.RequestException("API Error")
    
    monkeypatch.setattr(requests, 'post', mock_post_error)
    
    with open(test_images['jpeg'], 'rb') as img:
        # Upload person image
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, 'person.jpg')}
        )
        person_filename = response.json['filename']
        
        # Upload garment image
        response = client.post(
            '/upload-garment-image',
            content_type='multipart/form-data',
            data={'image': (img, 'garment.jpg')}
        )
        garment_filename = response.json['filename']
        
        # Try processing
        response = client.post(
            '/process-images',
            json={
                'person_image': person_filename,
                'garment_image': garment_filename
            }
        )
        assert response.status_code == 500
        assert 'API error' in response.json['error']

@pytest.mark.api
def test_api_timeout(client, test_images, monkeypatch):
    """Test handling of API timeout"""
    def mock_post(*args, **kwargs):
        raise requests.exceptions.Timeout("Request timed out")
    
    monkeypatch.setattr(requests, 'post', mock_post)
    
    # First upload test images
    with open(test_images['jpeg'], 'rb') as img:
        person_response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, 'person.jpg')}
        )
        assert person_response.status_code == 200
        person_filename = person_response.json['filename']
        
        garment_response = client.post(
            '/upload-garment-image',
            content_type='multipart/form-data',
            data={'image': (img, 'garment.jpg')}
        )
        assert garment_response.status_code == 200
        garment_filename = garment_response.json['filename']
    
    # Test processing with mocked timeout
    response = client.post(
        '/process-images',
        json={
            'person_image': person_filename,
            'garment_image': garment_filename
        }
    )
    
    assert response.status_code == 500  # Internal Server Error
    assert 'timed out' in response.json['error'].lower()

@pytest.mark.api
def test_invalid_input_data(client):
    """Test processing with invalid input data types"""
    test_cases = [
        {
            'person_image': 12345,  # Invalid type (int)
            'garment_image': 'test.jpg'
        },
        {
            'person_image': 'test.jpg',
            'garment_image': None  # Invalid type (None)
        },
        {
            'person_image': '',  # Empty string
            'garment_image': 'test.jpg'
        },
        {
            'person_image': ['test.jpg'],  # Invalid type (list)
            'garment_image': 'test.jpg'
        }
    ]
    
    for test_data in test_cases:
        response = client.post(
            '/process-images',
            json=test_data
        )
        assert response.status_code == 400
        assert 'Invalid input data' in response.json['error']

def test_image_compression_quality(client, test_images):
    """Test that compression maintains acceptable quality"""
    # Create a large test image
    large_image = Image.new('RGB', (2000, 2000), color='red')
    output = io.BytesIO()
    large_image.save(output, format='JPEG', quality=100)
    output.seek(0)
    original_size = output.tell()
    
    # Upload and compress
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'large.jpg')}
    )
    
    assert response.status_code == 200
    
    # Verify compression
    saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
    compressed_size = saved_path.stat().st_size
    assert compressed_size < original_size
    
    # Verify image is still valid
    compressed_img = Image.open(saved_path)
    assert compressed_img.format == 'JPEG'

def test_image_format_conversion(client):
    """Test format conversion from PNG/RGBA to JPEG/RGB"""
    # Create RGBA PNG image
    img = Image.new('RGBA', (500, 500), (255, 0, 0, 128))
    output = io.BytesIO()
    img.save(output, format='PNG')
    output.seek(0)
    
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'transparent.png')}
    )
    
    assert response.status_code == 200
    
    # Verify conversion
    saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
    converted_img = Image.open(saved_path)
    assert converted_img.mode == 'RGB'
    assert converted_img.format == 'JPEG'

def test_small_image_preservation(client):
    """Test that small images maintain original dimensions"""
    # Create small image
    small_img = Image.new('RGB', (500, 300), color='green')
    output = io.BytesIO()
    small_img.save(output, format='JPEG', quality=85)
    output.seek(0)
    
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'small.jpg')}
    )
    
    assert response.status_code == 200
    
    # Verify dimensions are preserved
    saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
    result_img = Image.open(saved_path)
    assert result_img.size == (500, 300)

def test_upload_with_compression(client):
    """Test successful image upload with compression"""
    # Create test image
    img = Image.new('RGB', (1500, 1500), color='blue')
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=100)
    output.seek(0)
    original_size = output.tell()
    
    # Upload image
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'test.jpg')}
    )
    
    assert response.status_code == 200
    assert 'filename' in response.json
    
    # Verify compressed file
    saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
    assert saved_path.exists()
    
    # Check compression results
    compressed_size = saved_path.stat().st_size
    assert compressed_size < original_size
    
    # Verify image is still valid
    compressed_img = Image.open(saved_path)
    assert compressed_img.format == 'JPEG'
    assert compressed_img.size[0] <= 1024  # Max dimension
    assert compressed_img.size[1] <= 1024

def test_upload_invalid_image(client):
    """Test upload with invalid image data"""
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (io.BytesIO(b'invalid data'), 'test.jpg')}
    )
    
    assert response.status_code == 400
    assert 'error' in response.json

def test_upload_no_file(client):
    """Test upload without file"""
    response = client.post('/upload-person-image')
    assert response.status_code == 400
    assert response.json['error'] == 'No image file provided'

def test_upload_empty_filename(client):
    """Test upload with empty filename"""
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (io.BytesIO(b''), '')}
    )
    
    assert response.status_code == 400
    assert response.json['error'] == 'No file selected'

def test_upload_with_error_handling(client, caplog):
    """Test error handling during upload"""
    caplog.set_level(logging.ERROR)
    
    # Create invalid image data
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (io.BytesIO(b'bad data'), 'test.jpg')}
    )
    
    assert response.status_code == 400
    assert 'error' in response.json
    assert any('Validation error' in record.message for record in caplog.records)

def test_large_image_compression(client):
    """Test compression of very large images (>5MB)"""
    # Create a large high-quality image
    large_img = Image.new('RGB', (4000, 3000), color='blue')  # 12MP image
    output = io.BytesIO()
    large_img.save(output, format='JPEG', quality=100)
    output.seek(0)
    original_size = output.tell()
    
    assert original_size > 5 * 1024 * 1024  # Verify test image is >5MB
    
    # Upload and compress
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'large.jpg')}
    )
    
    assert response.status_code == 200
    
    # Verify compression results
    saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
    compressed_size = saved_path.stat().st_size
    
    # Check significant size reduction
    assert compressed_size < original_size * 0.5  # At least 50% reduction
    assert compressed_size <= 5 * 1024 * 1024  # Under 5MB limit

def test_invalid_image_formats(client):
    """Test handling of various invalid image formats"""
    test_cases = [
        # Text file
        (io.BytesIO(b'This is not an image'), 'test.txt', 'Invalid image'),
        # GIF file
        (io.BytesIO(b'GIF89a...'), 'test.gif', 'Invalid format'),
        # Empty file
        (io.BytesIO(b''), 'empty.jpg', 'Invalid image'),
        # Corrupted JPEG
        (io.BytesIO(b'\xff\xd8\xff\xe0' + os.urandom(100)), 'corrupt.jpg', 'Invalid image'),
        # BMP file
        (io.BytesIO(b'BM...'), 'test.bmp', 'Invalid format')
    ]
    
    for file_data, filename, expected_error in test_cases:
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (file_data, filename)}
        )
        
        assert response.status_code == 400
        assert expected_error in response.json['error']

def test_small_image_handling(client):
    """Test handling of already small and optimized images"""
    # Create small, well-optimized image
    small_img = Image.new('RGB', (400, 300), color='red')
    output = io.BytesIO()
    small_img.save(output, format='JPEG', quality=85)  # Already optimized quality
    output.seek(0)
    original_size = output.tell()
    
    # Upload image
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'small.jpg')}
    )
    
    assert response.status_code == 200
    
    # Verify minimal compression
    saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
    final_size = saved_path.stat().st_size
    
    # Size should not increase significantly
    assert final_size <= original_size * 1.1  # Allow 10% margin
    
    # Original dimensions should be preserved
    compressed_img = Image.open(saved_path)
    assert compressed_img.size == (400, 300)

def test_compression_quality_thresholds(client):
    """Test different quality thresholds for compression"""
    # Create test images with different characteristics
    test_cases = [
        # High quality, large dimensions
        (Image.new('RGB', (2000, 1500), color='blue'), 100, 'high_quality.jpg'),
        # Medium quality, medium dimensions
        (Image.new('RGB', (1200, 800), color='green'), 85, 'medium_quality.jpg'),
        # Low quality, small dimensions
        (Image.new('RGB', (800, 600), color='red'), 60, 'low_quality.jpg')
    ]
    
    for img, quality, filename in test_cases:
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality)
        output.seek(0)
        original_size = output.tell()
        
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (output, filename)}
        )
        
        assert response.status_code == 200
        saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
        
        # Verify compression behavior
        if original_size > 1024 * 1024:  # If original > 1MB
            assert saved_path.stat().st_size < original_size
        else:
            # Small files shouldn't be compressed much
            assert saved_path.stat().st_size <= original_size * 1.1

def test_image_dimensions_preservation(client):
    """Test aspect ratio and dimension preservation during compression"""
    # Test various aspect ratios
    dimensions = [
        (800, 600),    # 4:3
        (1920, 1080),  # 16:9
        (1000, 1000),  # 1:1
        (600, 800),    # Portrait
        (1500, 500)    # Panorama
    ]
    
    for width, height in dimensions:
        img = Image.new('RGB', (width, height), color='blue')
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (output, f'{width}x{height}.jpg')}
        )
        
        assert response.status_code == 200
        
        # Check compressed image
        saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
        compressed_img = Image.open(saved_path)
        
        # Verify aspect ratio preservation
        original_ratio = width / height
        compressed_ratio = compressed_img.size[0] / compressed_img.size[1]
        assert abs(original_ratio - compressed_ratio) < 0.01  # Allow small rounding difference

def test_compression_scenarios(client):
    """Test compression with various image types and sizes"""
    test_cases = [
        # Großes hochauflösendes Bild
        {
            'size': (3000, 2000),
            'color': 'blue',
            'quality': 100,
            'format': 'JPEG',
            'expected_max_size': 5 * 1024 * 1024  # 5MB
        },
        # PNG mit Transparenz
        {
            'size': (800, 600),
            'color': (255, 0, 0, 128),
            'quality': 95,
            'format': 'PNG',
            'mode': 'RGBA'
        },
        # Kleines optimiertes Bild
        {
            'size': (400, 300),
            'color': 'red',
            'quality': 85,
            'format': 'JPEG'
        }
    ]
    
    for case in test_cases:
        # Erstelle Testbild
        if case.get('mode') == 'RGBA':
            img = Image.new('RGBA', case['size'], case['color'])
        else:
            img = Image.new('RGB', case['size'], case['color'])
            
        output = io.BytesIO()
        img.save(output, format=case['format'], quality=case.get('quality', 95))
        output.seek(0)
        original_size = output.tell()
        original_ratio = case['size'][0] / case['size'][1]
        
        # Upload und komprimiere
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (output, f'test.{case["format"].lower()}')}
        )
        
        assert response.status_code == 200
        
        # Prüfe komprimiertes Bild
        saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
        compressed_img = Image.open(saved_path)
        
        # Prüfe Größenbeschränkungen
        if 'expected_max_size' in case:
            assert saved_path.stat().st_size <= case['expected_max_size']
            
        # Prüfe Seitenverhältnis
        compressed_ratio = compressed_img.size[0] / compressed_img.size[1]
        assert abs(original_ratio - compressed_ratio) < 0.01
        
        # Prüfe Bildqualität mit SSIM wenn möglich
        if case['format'] == 'JPEG':
            original_img = Image.open(output)
            # Konvertiere Bilder zu gleicher Größe für Vergleich
            original_img.thumbnail(compressed_img.size)
            assert compare_images(original_img, compressed_img) > 0.9  # Mindestens 90% Ähnlichkeit

def compare_images(img1, img2):
    """Vergleiche zwei Bilder mit SSIM"""
    # Konvertiere zu Numpy Arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    # Berechne SSIM
    return ssim(img1_array, img2_array, multichannel=True)

def test_error_cases(client):
    """Test various error scenarios"""
    error_cases = [
        # Zu große Datei
        {
            'data': b'0' * (6 * 1024 * 1024),
            'filename': 'large.jpg',
            'expected_error': 'size exceeds'
        },
        # Textdatei
        {
            'data': b'This is not an image',
            'filename': 'test.txt',
            'expected_error': 'Invalid image'
        },
        # Korrupte JPEG-Datei
        {
            'data': b'\xff\xd8\xff' + os.urandom(100),
            'filename': 'corrupt.jpg',
            'expected_error': 'Invalid image'
        },
        # Leere Datei
        {
            'data': b'',
            'filename': 'empty.jpg',
            'expected_error': 'Invalid image'
        }
    ]
    
    for case in error_cases:
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (io.BytesIO(case['data']), case['filename'])}
        )
        
        assert response.status_code == 400
        assert case['expected_error'] in response.json['error']

def test_complete_workflow(client, monkeypatch):
    """Test complete workflow from upload to processing"""
    # Mock API response
    mock_response = b'processed_image_data'
    class MockResponse:
        status_code = 200
        content = mock_response
        def raise_for_status(self): pass
    
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: MockResponse())
    
    # 1. Upload Personenbild
    person_img = Image.new('RGB', (800, 600), 'blue')
    person_output = io.BytesIO()
    person_img.save(person_output, format='JPEG')
    person_output.seek(0)
    
    person_response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (person_output, 'person.jpg')}
    )
    assert person_response.status_code == 200
    person_filename = person_response.json['filename']
    
    # 2. Upload Kleidungsbild
    garment_img = Image.new('RGB', (400, 600), 'red')
    garment_output = io.BytesIO()
    garment_img.save(garment_output, format='JPEG')
    garment_output.seek(0)
    
    garment_response = client.post(
        '/upload-garment-image',
        content_type='multipart/form-data',
        data={'image': (garment_output, 'garment.jpg')}
    )
    assert garment_response.status_code == 200
    garment_filename = garment_response.json['filename']
    
    # 3. Verarbeite Bilder
    process_response = client.post(
        '/process-images',
        json={
            'person_image': person_filename,
            'garment_image': garment_filename
        }
    )
    
    assert process_response.status_code == 200
    assert 'result' in process_response.json
    assert process_response.json['result'] == mock_response.decode('latin1')
    
    # 4. Prüfe Cleanup
    person_path = app.config['UPLOAD_FOLDER'] / person_filename
    garment_path = app.config['UPLOAD_FOLDER'] / garment_filename
    assert not person_path.exists()
    assert not garment_path.exists()

def test_logging_details(client, caplog):
    """Test detailed logging information"""
    caplog.set_level(logging.INFO)
    
    # Create test image
    img = Image.new('RGB', (1500, 1000), color='blue')
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=100)
    output.seek(0)
    
    # Upload image
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'test.jpg')}
    )
    
    assert response.status_code == 200
    
    # Verify log messages
    log_messages = [record.message for record in caplog.records]
    
    # Check required log entries
    assert any("Starting person image upload request" in msg 
              for msg in log_messages)
    assert any("Image validation successful" in msg 
              for msg in log_messages)
    assert any("Image compressed and saved" in msg 
              for msg in log_messages)
    assert any("Request completed" in msg 
              for msg in log_messages)
    
    # Check compression details are logged
    compression_logs = [msg for msg in log_messages 
                       if "Compression complete" in msg]
    assert len(compression_logs) == 1
    assert "KB" in compression_logs[0]
    assert "quality=" in compression_logs[0]

def test_error_logging(client, caplog):
    """Test logging of error scenarios"""
    caplog.set_level(logging.ERROR)
    
    # Test invalid file
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (io.BytesIO(b'invalid'), 'test.jpg')}
    )
    
    assert response.status_code == 400
    
    # Verify error logs
    error_logs = [record for record in caplog.records 
                 if record.levelname == 'ERROR']
    assert len(error_logs) > 0
    assert any("Invalid image" in record.message 
              for record in error_logs)

def test_file_validation(client):
    """Test file validation with various cases"""
    test_cases = [
        # Valid cases
        {
            'filename': 'test.jpg',
            'content': b'valid_jpg_data',
            'content_type': 'image/jpeg',
            'should_pass': True
        },
        {
            'filename': 'test.png',
            'content': b'valid_png_data',
            'content_type': 'image/png',
            'should_pass': True
        },
        # Invalid cases
        {
            'filename': 'test.gif',
            'content': b'gif_data',
            'content_type': 'image/gif',
            'should_pass': False,
            'error': 'Invalid file extension'
        },
        {
            'filename': 'test.jpg',
            'content': b'0' * (MAX_FILESIZE + 1),
            'content_type': 'image/jpeg',
            'should_pass': False,
            'error': 'File too large'
        },
        {
            'filename': 'test',  # No extension
            'content': b'data',
            'content_type': 'image/jpeg',
            'should_pass': False,
            'error': 'Invalid file extension'
        },
        {
            'filename': 'test.jpg',
            'content': b'data',
            'content_type': 'text/plain',
            'should_pass': False,
            'error': 'Invalid file type'
        }
    ]
    
    for case in test_cases:
        file = io.BytesIO(case['content'])
        file.filename = case['filename']
        file.content_type = case['content_type']
        
        try:
            validate_file(file)
            assert case['should_pass'], f"Validation should have failed for {case['filename']}"
        except ValueError as e:
            assert not case['should_pass'], f"Validation should have passed for {case['filename']}"
            assert case['error'] in str(e)

def test_allowed_file():
    """Test allowed_file function"""
    valid_files = [
        'test.jpg',
        'test.JPG',
        'test.jpeg',
        'test.png',
        'path/to/test.jpg'
    ]
    
    invalid_files = [
        'test',          # No extension
        '.jpg',          # No filename
        'test.gif',      # Wrong extension
        'test.jpg.exe',  # Multiple extensions
        'test.JPG.GIF'   # Multiple extensions
    ]
    
    for filename in valid_files:
        assert allowed_file(filename), f"{filename} should be allowed"
        
    for filename in invalid_files:
        assert not allowed_file(filename), f"{filename} should not be allowed"

def test_rate_limits(client, test_images):
    """Test rate limiting for all endpoints"""
    endpoints = [
        {
            'url': '/upload-person-image',
            'method': 'post',
            'data': lambda: {
                'content_type': 'multipart/form-data',
                'data': {'image': (open(test_images['jpeg'], 'rb'), 'test.jpg')}
            },
            'limit': 10
        },
        {
            'url': '/upload-garment-image',
            'method': 'post',
            'data': lambda: {
                'content_type': 'multipart/form-data',
                'data': {'image': (open(test_images['jpeg'], 'rb'), 'test.jpg')}
            },
            'limit': 10
        },
        {
            'url': '/process-images',
            'method': 'post',
            'data': lambda: {
                'json': {
                    'person_image': 'test.jpg',
                    'garment_image': 'test.jpg'
                }
            },
            'limit': 5
        }
    ]
    
    for endpoint in endpoints:
        # Test successful requests up to limit
        for i in range(endpoint['limit']):
            response = getattr(client, endpoint['method'])(
                endpoint['url'],
                **endpoint['data']()
            )
            assert response.status_code in [200, 400, 404]  # Allow other valid errors
        
        # Test rate limit exceeded
        response = getattr(client, endpoint['method'])(
            endpoint['url'],
            **endpoint['data']()
        )
        assert response.status_code == 429
        assert 'error' in response.json
        assert 'limit' in response.json
        assert 'reset' in response.json

def test_rate_limit_reset(client, test_images):
    """Test rate limit reset after waiting"""
    import time
    
    # Make requests until limit
    for _ in range(10):
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (open(test_images['jpeg'], 'rb'), 'test.jpg')}
        )
        assert response.status_code == 200
    
    # Verify limit is reached
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (open(test_images['jpeg'], 'rb'), 'test.jpg')}
    )
    assert response.status_code == 429
    
    # Wait for reset (in test we use a shorter time)
    time.sleep(2)  # Adjust based on your test configuration
    
    # Verify limit is reset
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (open(test_images['jpeg'], 'rb'), 'test.jpg')}
    )
    assert response.status_code == 200

def test_rate_limit_per_endpoint(client, test_images):
    """Test that rate limits are per endpoint"""
    # Max out upload-person-image
    for _ in range(10):
        client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (open(test_images['jpeg'], 'rb'), 'test.jpg')}
        )
    
    # Should still be able to use garment upload
    with open(test_images['jpeg'], 'rb') as img:
        response = client.post(
            '/upload-garment-image',
            content_type='multipart/form-data',
            data={'image': (img, 'garment.jpg')}
        )
    assert response.status_code == 200

def test_config():
    """Test application configuration"""
    assert app.config['HUGGINGFACE_API_KEY'] is not None
    assert app.config['MAX_FILESIZE'] > 0
    assert 'jpg' in app.config['ALLOWED_EXTENSIONS']
    assert app.config['UPLOAD_FOLDER'].exists()

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing"""
    env_vars = {
        'HUGGINGFACE_API_KEY': 'test_key',
        'MAX_FILESIZE': '1048576',  # 1MB
        'ALLOWED_EXTENSIONS': 'jpg,png',
        'API_TIMEOUT': '10'
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    # Reload configuration
    config = Config.init_app()
    app.config.from_object(config)
    yield env_vars

def test_env_config(mock_env):
    """Test configuration from environment variables"""
    assert app.config['HUGGINGFACE_API_KEY'] == mock_env['HUGGINGFACE_API_KEY']
    assert app.config['MAX_FILESIZE'] == 1048576
    assert app.config['ALLOWED_EXTENSIONS'] == {'jpg', 'png'}
    assert app.config['API_TIMEOUT'] == 10

@pytest.mark.parametrize('test_case', [
    {
        'name': 'huge_image',
        'size': (8000, 6000),
        'color': 'blue',
        'should_resize': True
    },
    {
        'name': 'tiny_image',
        'size': (50, 50),
        'color': 'red',
        'should_resize': False
    },
    {
        'name': 'wide_image',
        'size': (3000, 500),
        'color': 'green',
        'should_resize': True
    }
])
def test_image_dimensions(client, test_case):
    """Test image resizing for various dimensions"""
    # Create test image
    img = Image.new('RGB', test_case['size'], test_case['color'])
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=100)
    output.seek(0)
    
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, f"{test_case['name']}.jpg")}
    )
    
    assert response.status_code == 200
    
    # Check dimensions
    saved_path = app.config['UPLOAD_FOLDER'] / response.json['data']['filename']
    processed_img = Image.open(saved_path)
    
    if test_case['should_resize']:
        assert processed_img.size[0] <= 1024
        assert processed_img.size[1] <= 1024
    else:
        assert processed_img.size == test_case['size']

@pytest.mark.parametrize('api_error', [
    requests.exceptions.ConnectionError("Connection failed"),
    requests.exceptions.Timeout("Request timed out"),
    requests.exceptions.RequestException("API error"),
    ValueError("Invalid response")
])
def test_api_error_handling(client, test_images, monkeypatch, api_error):
    """Test handling of various API errors"""
    def mock_api_error(*args, **kwargs):
        raise api_error
    
    monkeypatch.setattr(requests, 'post', mock_api_error)
    
    # Upload and process images
    with open(test_images['jpeg'], 'rb') as img:
        # Upload images
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, 'person.jpg')}
        )
        person_filename = response.json['data']['filename']
        
        response = client.post(
            '/upload-garment-image',
            content_type='multipart/form-data',
            data={'image': (img, 'garment.jpg')}
        )
        garment_filename = response.json['data']['filename']
    
    # Process images
    response = client.post(
        '/process-images',
        json={
            'person_image': person_filename,
            'garment_image': garment_filename
        }
    )
    
    assert response.status_code == 500
    assert 'error' in response.json

@pytest.mark.parametrize('batch_size', [1, 5, 10])
def test_batch_processing_sizes(client, test_images, batch_size):
    """Test batch processing with different sizes"""
    # Prepare batch request
    image_pairs = []
    for i in range(batch_size):
        # Upload images first
        with open(test_images['jpeg'], 'rb') as img:
            response = client.post(
                '/upload-person-image',
                content_type='multipart/form-data',
                data={'image': (img, f'person_{i}.jpg')}
            )
            person_filename = response.json['data']['filename']
            
            response = client.post(
                '/upload-garment-image',
                content_type='multipart/form-data',
                data={'image': (img, f'garment_{i}.jpg')}
            )
            garment_filename = response.json['data']['filename']
            
            image_pairs.append({
                'person_image': person_filename,
                'garment_image': garment_filename
            })
    
    # Process batch
    response = client.post(
        '/batch-process-images',
        json={'images': image_pairs}
    )
    
    assert response.status_code == 200
    assert response.json['data']['total'] == batch_size
    assert response.json['data']['successful'] == batch_size

def test_batch_processing_partial_failure(client, test_images):
    """Test batch processing with some failing items"""
    # Prepare mixed batch (valid and invalid files)
    image_pairs = [
        {
            'person_image': 'nonexistent.jpg',
            'garment_image': 'missing.jpg'
        }
    ]
    
    # Add one valid pair
    with open(test_images['jpeg'], 'rb') as img:
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (img, 'valid_person.jpg')}
        )
        person_filename = response.json['data']['filename']
        
        response = client.post(
            '/upload-garment-image',
            content_type='multipart/form-data',
            data={'image': (img, 'valid_garment.jpg')}
        )
        garment_filename = response.json['data']['filename']
        
        image_pairs.append({
            'person_image': person_filename,
            'garment_image': garment_filename
        })
    
    # Process batch
    response = client.post(
        '/batch-process-images',
        json={'images': image_pairs}
    )
    
    assert response.status_code == 200
    assert response.json['data']['total'] == 2
    assert response.json['data']['successful'] == 1
    assert response.json['data']['failed'] == 1

def test_rate_limit_reset_behavior(client, test_images):
    """Test rate limit reset behavior"""
    # Make requests until limit
    responses = []
    for _ in range(11):  # One over limit
        with open(test_images['jpeg'], 'rb') as img:
            response = client.post(
                '/upload-person-image',
                content_type='multipart/form-data',
                data={'image': (img, 'test.jpg')}
            )
            responses.append(response.status_code)
    
    # Verify rate limiting
    assert responses.count(200) == 10  # First 10 should succeed
    assert responses[-1] == 429  # Last one should fail
    
    # Get reset time from response
    last_response = responses[-1]
    assert 'reset' in last_response.json

def test_rate_limit_independence(client, test_images):
    """Test rate limits are independent per endpoint"""
    # Max out person upload
    for _ in range(10):
        with open(test_images['jpeg'], 'rb') as img:
            client.post(
                '/upload-person-image',
                content_type='multipart/form-data',
                data={'image': (img, 'person.jpg')}
            )
    
    # Should still be able to use garment upload
    with open(test_images['jpeg'], 'rb') as img:
        response = client.post(
            '/upload-garment-image',
            content_type='multipart/form-data',
            data={'image': (img, 'garment.jpg')}
        )
    assert response.status_code == 200

@pytest.mark.performance
def test_image_processing_performance(client, benchmark):
    """Test performance of image processing"""
    def process_image():
        # Create large test image
        img = Image.new('RGB', (4000, 3000), 'blue')
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=100)
        output.seek(0)
        
        return client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (output, 'test.jpg')}
        )
    
    result = benchmark(process_image)
    assert result.status_code == 200
    assert result.elapsed.total_seconds() < 2.0  # Max 2 Sekunden

@pytest.mark.performance
def test_batch_processing_performance(client, benchmark):
    """Test performance of batch processing"""
    def prepare_batch(size=5):
        pairs = []
        for i in range(size):
            img = Image.new('RGB', (800, 600), 'red')
            output = io.BytesIO()
            img.save(output, format='JPEG')
            output.seek(0)
            pairs.append({
                'person_image': f'person_{i}.jpg',
                'garment_image': f'garment_{i}.jpg'
            })
        return pairs
    
    def process_batch():
        return client.post(
            '/batch-process-images',
            json={'images': prepare_batch()}
        )
    
    result = benchmark(process_batch)
    assert result.status_code == 200
    assert result.elapsed.total_seconds() < 5.0  # Max 5 Sekunden

@pytest.mark.security
def test_csrf_protection(client):
    """Test CSRF protection"""
    # Test without CSRF token
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (io.BytesIO(b'test'), 'test.jpg')},
        headers={'X-CSRF-Token': None}
    )
    assert response.status_code == 403

@pytest.mark.security
def test_input_sanitization(client):
    """Test input sanitization"""
    malicious_filenames = [
        '../../../etc/passwd',
        'shell.jpg;rm -rf /',
        '<script>alert("xss")</script>.jpg',
        'image.jpg\x00.exe'
    ]
    
    for filename in malicious_filenames:
        img = Image.new('RGB', (100, 100), 'red')
        output = io.BytesIO()
        img.save(output, format='JPEG')
        output.seek(0)
        
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (output, filename)}
        )
        
        if response.status_code == 200:
            # Check sanitized filename
            assert '/' not in response.json['filename']
            assert ';' not in response.json['filename']
            assert '<' not in response.json['filename']
            assert '\x00' not in response.json['filename']

@pytest.mark.security
def test_api_key_protection(client, monkeypatch):
    """Test API key protection"""
    # Test with invalid API key
    monkeypatch.setenv('HUGGINGFACE_API_KEY', 'invalid_key')
    
    img = Image.new('RGB', (100, 100), 'red')
    output = io.BytesIO()
    img.save(output, format='JPEG')
    output.seek(0)
    
    response = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'test.jpg')}
    )
    
    # API key should not be exposed in response
    assert 'HUGGINGFACE_API_KEY' not in str(response.data)

@pytest.mark.security
def test_rate_limit_effectiveness(client):
    """Test rate limiting effectiveness"""
    responses = []
    
    # Make many requests quickly
    for _ in range(20):
        img = Image.new('RGB', (100, 100), 'red')
        output = io.BytesIO()
        img.save(output, format='JPEG')
        output.seek(0)
        
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (output, 'test.jpg')}
        )
        responses.append(response.status_code)
    
    # Check rate limiting
    assert 429 in responses  # Some requests should be rate limited
    assert responses.count(200) <= 10  # Max 10 successful requests

@pytest.mark.security
def test_file_type_validation(client):
    """Test file type validation"""
    invalid_files = [
        (b'GIF89a...', 'image.gif'),
        (b'%PDF-1.5', 'document.pdf'),
        (b'PK\x03\x04', 'archive.zip'),
        (b'\x89PNG\r\n\x1a\n', 'image.jpg')  # PNG with .jpg extension
    ]
    
    for content, filename in invalid_files:
        response = client.post(
            '/upload-person-image',
            content_type='multipart/form-data',
            data={'image': (io.BytesIO(content), filename)}
        )
        
        assert response.status_code == 400
        assert 'Invalid' in response.json['error']

@pytest.mark.performance
def test_caching_effectiveness(client):
    """Test caching effectiveness for repeated requests"""
    # First request
    img = Image.new('RGB', (800, 600), 'red')
    output = io.BytesIO()
    img.save(output, format='JPEG')
    output.seek(0)
    
    response1 = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'test.jpg')}
    )
    time1 = response1.elapsed.total_seconds()
    
    # Second request with same image
    output.seek(0)
    response2 = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output, 'test.jpg')}
    )
    time2 = response2.elapsed.total_seconds()
    
    # Second request should be faster due to caching
    assert time2 < time1

@pytest.mark.performance
def test_cache_invalidation(client):
    """Test cache invalidation for modified images"""
    # Original image
    img1 = Image.new('RGB', (800, 600), 'red')
    output1 = io.BytesIO()
    img1.save(output1, format='JPEG')
    output1.seek(0)
    
    response1 = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output1, 'test.jpg')}
    )
    
    # Modified image
    img2 = Image.new('RGB', (800, 600), 'blue')
    output2 = io.BytesIO()
    img2.save(output2, format='JPEG')
    output2.seek(0)
    
    response2 = client.post(
        '/upload-person-image',
        content_type='multipart/form-data',
        data={'image': (output2, 'test.jpg')}
    )
    
    # Results should be different
    assert response1.json['data']['size'] != response2.json['data']['size']

def test_successful_upload(client, test_image):
    """Test successful image upload"""
    response = client.post(
        '/upload',
        content_type='multipart/form-data',
        data={'file': (test_image, 'test.jpg')}
    )
    
    assert response.status_code == 200
    assert 'filename' in response.json
    assert 'size' in response.json
    
    # Verify file was saved
    saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

def test_no_file(client):
    """Test upload without file"""
    response = client.post('/upload')
    assert response.status_code == 400
    assert response.json['error'] == 'No file provided'

def test_empty_filename(client):
    """Test upload with empty filename"""
    response = client.post(
        '/upload',
        content_type='multipart/form-data',
        data={'file': (io.BytesIO(b''), '')}
    )
    assert response.status_code == 400
    assert response.json['error'] == 'No file selected'

def test_invalid_file_type(client):
    """Test upload with invalid file type"""
    response = client.post(
        '/upload',
        content_type='multipart/form-data',
        data={'file': (io.BytesIO(b'test'), 'test.txt')}
    )
    assert response.status_code == 400
    assert 'Invalid file type' in response.json['error']

def test_large_file(client):
    """Test upload with file exceeding size limit"""
    large_data = b'0' * (5 * 1024 * 1024 + 1)  # 5MB + 1 byte
    response = client.post(
        '/upload',
        content_type='multipart/form-data',
        data={'file': (io.BytesIO(large_data), 'large.jpg')}
    )
    assert response.status_code == 400
    assert 'File too large' in response.json['error']

def test_invalid_image(client):
    """Test upload with invalid image data"""
    response = client.post(
        '/upload',
        content_type='multipart/form-data',
        data={'file': (io.BytesIO(b'not an image'), 'test.jpg')}
    )
    assert response.status_code == 400
    assert 'Invalid image' in response.json['error']

def test_rgba_conversion(client):
    """Test RGBA to RGB conversion"""
    # Create RGBA image
    img = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    response = client.post(
        '/upload',
        content_type='multipart/form-data',
        data={'file': (img_io, 'test.png')}
    )
    
    assert response.status_code == 200
    
    # Verify conversion
    saved_path = app.config['UPLOAD_FOLDER'] / response.json['filename']
    saved_img = Image.open(saved_path)
    assert saved_img.mode == 'RGB'

def test_rate_limit(client, test_image):
    """Test rate limiting"""
    responses = []
    
    # Make 11 requests (1 over limit)
    for _ in range(11):
        response = client.post(
            '/upload',
            content_type='multipart/form-data',
            data={'file': (test_image, 'test.jpg')}
        )
        responses.append(response.status_code)
    
    # First 10 should succeed, 11th should fail
    assert responses.count(200) == 10
    assert responses[-1] == 429

def test_filename_security(client, test_image):
    """Test filename sanitization"""
    malicious_filenames = [
        '../../../etc/passwd.jpg',
        'shell.jpg;rm -rf /',
        '<script>alert("xss")</script>.jpg',
        'image.jpg\x00.exe'
    ]
    
    for filename in malicious_filenames:
        response = client.post(
            '/upload',
            content_type='multipart/form-data',
            data={'file': (test_image, filename)}
        )
        
        if response.status_code == 200:
            # Check sanitized filename
            assert '/' not in response.json['filename']
            assert ';' not in response.json['filename']
            assert '<' not in response.json['filename']
            assert '\x00' not in response.json['filename']

def test_concurrent_uploads(client, test_image):
    """Test concurrent uploads"""
    from concurrent.futures import ThreadPoolExecutor
    
    def upload():
        return client.post(
            '/upload',
            content_type='multipart/form-data',
            data={'file': (test_image, 'test.jpg')}
        )
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        responses = list(executor.map(lambda _: upload(), range(3)))
    
    # All uploads should succeed
    assert all(r.status_code == 200 for r in responses)