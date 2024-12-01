import pytest
from app import app
from pathlib import Path
from PIL import Image

@pytest.fixture(autouse=True)
def setup_temp_dirs(tmp_path):
    """Automatically set up temporary directories before each test"""
    app.config['UPLOAD_FOLDER'] = tmp_path / 'temp'
    app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
    yield
    # Cleanup happens automatically with tmp_path

@pytest.fixture
def client():
    """Create a test client for the app"""
    with app.test_client() as client:
        yield client

@pytest.fixture
def test_images(tmp_path):
    """Create test images in a temporary directory"""
    # ... existing code from test_app.py ... 

@pytest.fixture(autouse=True)
def clean_temp_files():
    """Clean up temporary files after each test"""
    yield
    temp_dir = Path('temp')
    if temp_dir.exists():
        for file in temp_dir.glob('*'):
            try:
                file.unlink()
            except OSError:
                pass

@pytest.fixture
def mock_api_response():
    """Mock successful API response"""
    class MockResponse:
        status_code = 200
        content = b'mock_processed_image_data'
        
        def raise_for_status(self):
            pass
    
    return MockResponse()