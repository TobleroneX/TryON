from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from werkzeug.utils import secure_filename
from pathlib import Path
from PIL import Image
import logging

app = Flask(__name__)
CORS(app)

# Configure rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configure upload settings
UPLOAD_FOLDER = Path('temp')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILESIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file):
    """Validate image file"""
    try:
        img = Image.open(file)
        img.verify()  # Verify it's a valid image
        file.seek(0)  # Reset file pointer
        return True
    except Exception:
        return False

@app.route('/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_file():
    """Handle file upload with validation"""
    try:
        # Check if file exists in request
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({
                'error': 'No file provided',
                'details': 'File must be provided in form data with key "file"'
            }), 400

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            logger.warning("Empty filename provided")
            return jsonify({
                'error': 'No file selected',
                'details': 'A file must be selected for upload'
            }), 400

        # Check file extension
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                'error': 'Invalid file type',
                'details': f'Allowed file types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Check file size
        file.seek(0, 2)  # Seek to end of file
        size = file.tell()
        file.seek(0)  # Reset file pointer
        
        if size > MAX_FILESIZE:
            logger.warning(f"File too large: {size} bytes")
            return jsonify({
                'error': 'File too large',
                'details': f'Maximum file size is {MAX_FILESIZE/1024/1024}MB'
            }), 400

        # Validate image
        if not validate_image(file):
            logger.warning("Invalid image file")
            return jsonify({
                'error': 'Invalid image',
                'details': 'File must be a valid image'
            }), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        
        # Process and save image
        try:
            img = Image.open(file)
            
            # Convert RGBA to RGB if needed
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, 'white')
                background.paste(img, mask=img.split()[-1])
                img = background
            
            # Compress and save
            img.save(filepath, 'JPEG', quality=85, optimize=True)
            
            logger.info(f"Successfully saved file: {filename}")
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'size': os.path.getsize(filepath)
            }), 200
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({
                'error': 'Error saving file',
                'details': str(e)
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit errors"""
    return jsonify({
        'error': 'Rate limit exceeded',
        'details': f'Too many requests. {str(e.description)}'
    }), 429

if __name__ == '__main__':
    app.run(debug=True) 