from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from werkzeug.utils import secure_filename
import requests
from pathlib import Path
from PIL import Image
import imghdr
from flask_limiter.errors import RateLimitExceeded
import io
import logging
from datetime import datetime
from config import Config
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Dict
import concurrent.futures

app = Flask(__name__)
CORS(app)

# Load configuration
config = Config.init_app()
app.config.from_object(config)

# Configure rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day"],
    storage_uri="memory://",
    strategy="fixed-window-elastic-expiry"
)

# Globaler ThreadPool für asynchrone Operationen
thread_pool = ThreadPoolExecutor(max_workers=4)

def allowed_file(filename):
    """Check if filename has allowed extension"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in app.config['ALLOWED_EXTENSIONS']

def validate_file(file):
    """Validate file type, size and extension
    
    Args:
        file: FileStorage object from request
        
    Raises:
        ValueError: If file is invalid
    """
    # Check if file exists
    if not file:
        raise ValueError("No file provided")
        
    # Check filename
    if file.filename == '':
        raise ValueError("No filename provided")
        
    # Check extension
    if not allowed_file(file.filename):
        raise ValueError(f"Invalid file extension. Allowed: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")
        
    # Check MIME type
    if file.content_type not in app.config['ALLOWED_MIME_TYPES']:
        raise ValueError(f"Invalid file type: {file.content_type}")
        
    # Check file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset position
    
    if size > app.config['MAX_FILESIZE']:
        raise ValueError(f"File too large. Maximum size: {app.config['MAX_FILESIZE']/1024/1024:.1f}MB")

def ensure_temp_directory():
    """Ensure temp directory exists with proper permissions"""
    temp_dir = Path(app.config['UPLOAD_FOLDER'])
    test_dir = Path('test_images')
    
    for directory in [temp_dir, test_dir]:
        directory.mkdir(exist_ok=True)
        # Set directory permissions to 777 (read/write/execute for all)
        directory.chmod(0o777)

# Call this function when app starts
ensure_temp_directory()

def compress_image(image_file, max_size=(1024, 1024), quality=70):
    """Compress and resize image with logging"""
    logger.info(f"Starting image compression: format={image_file.content_type}")
    
    try:
        img = Image.open(image_file)
        original_size = img.size
        original_mode = img.mode
        
        # Log original image details
        logger.debug(f"Original image: size={original_size}, mode={original_mode}")
        
        # Resize if needed
        img.thumbnail(max_size, Image.ANTIALIAS)
        if img.size != original_size:
            logger.info(f"Image resized: {original_size} -> {img.size}")

        # Convert format if needed
        if img.mode in ("RGBA", "P"):
            logger.info(f"Converting image mode: {img.mode} -> RGB")
            img = img.convert("RGB")

        # Compress
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', optimize=True, quality=quality)
        compressed_size = buffer.tell()
        
        # Log compression results
        logger.info(f"Compression complete: "
                   f"size={compressed_size/1024:.1f}KB, "
                   f"quality={quality}")
        
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Error during compression: {str(e)}", exc_info=True)
        raise

def save_compressed_image(file, filepath, max_size=(1024, 1024), quality=70):
    """Speichert ein komprimiertes Bild
    
    Args:
        file: Hochgeladenes Bild
        filepath: Zielpfad für das komprimierte Bild
        max_size: Maximale Bildgröße
        quality: JPEG-Qualität
    """
    compressed = compress_image(file, max_size=max_size, quality=quality)
    with open(filepath, 'wb') as f:
        f.write(compressed.getvalue())

def create_error_response(error_code, message, details=None):
    """Create standardized error response
    
    Args:
        error_code: String error code (e.g. 'INVALID_FORMAT')
        message: Human readable error message
        details: Optional additional error details
    """
    response = {
        'error': {
            'code': error_code,
            'message': message,
            'details': details
        }
    }
    return jsonify(response)

async def compress_image_async(image_file, max_size=(1024, 1024), quality=70):
    """Asynchrone Bildkomprimierung
    
    Args:
        image_file: Hochgeladenes Bild
        max_size: Maximale Bildgröße
        quality: JPEG-Qualität
    """
    loop = asyncio.get_event_loop()
    
    # Führe CPU-intensive Operationen im ThreadPool aus
    return await loop.run_in_executor(
        thread_pool,
        partial(compress_image, image_file, max_size, quality)
    )

async def process_images_async(person_image_path, garment_image_path):
    """Asynchrone Bildverarbeitung
    
    Args:
        person_image_path: Pfad zum Personenbild
        garment_image_path: Pfad zum Kleidungsbild
    """
    try:
        logger.info("Starting async image processing")
        
        # API-Request im ThreadPool ausführen
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            thread_pool,
            partial(send_to_kolors_api, person_image_path, garment_image_path)
        )
        
        logger.info("Async processing completed")
        return result
        
    except Exception as e:
        logger.error(f"Async processing error: {str(e)}")
        raise

async def batch_process_images_async(images: List[Dict[str, str]]) -> List[Dict]:
    """Process multiple image pairs asynchronously
    
    Args:
        images: List of image pair dictionaries with person_image and garment_image
    Returns:
        List of processing results
    """
    tasks = []
    results = []
    
    try:
        logger.info(f"Starting batch processing of {len(images)} image pairs")
        
        # Create processing tasks
        for img_pair in images:
            person_path = app.config['UPLOAD_FOLDER'] / img_pair['person_image']
            garment_path = app.config['UPLOAD_FOLDER'] / img_pair['garment_image']
            
            if not person_path.exists() or not garment_path.exists():
                continue
                
            # Create task for this image pair
            task = process_images_async(person_path, garment_path)
            tasks.append((img_pair, task))
        
        # Process all tasks concurrently
        for img_pair, task in tasks:
            try:
                result = await task
                results.append({
                    'person_image': img_pair['person_image'],
                    'garment_image': img_pair['garment_image'],
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                logger.error(f"Error processing pair: {str(e)}")
                results.append({
                    'person_image': img_pair['person_image'],
                    'garment_image': img_pair['garment_image'],
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise

@app.route('/upload-person-image', methods=['POST'])
@limiter.limit("10 per minute")
async def upload_person_image():
    """Handle person image upload with comprehensive error handling"""
    request_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    logger.info(f"[{request_id}] Starting person image upload")
    
    try:
        # Validate request
        if 'image' not in request.files:
            logger.warning(f"[{request_id}] No image file in request")
            return create_error_response(
                'NO_IMAGE',
                'No image file provided',
                {'accepted_formats': list(app.config['ALLOWED_EXTENSIONS'])}
            ), 400

        file = request.files['image']
        if not file or file.filename == '':
            logger.warning(f"[{request_id}] Empty filename")
            return create_error_response(
                'EMPTY_FILENAME',
                'No file selected'
            ), 400

        # Validate and process image
        try:
            validate_file(file)
            filename = secure_filename(file.filename)
            filepath = app.config['UPLOAD_FOLDER'] / filename
            
            # Compress and save image
            try:
                compressed = await compress_image_async(file)
                with open(filepath, 'wb') as f:
                    f.write(compressed.getvalue())
                
                file_size = filepath.stat().st_size
                logger.info(f"[{request_id}] Successfully saved image: {filename} ({file_size/1024:.1f}KB)")
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'filename': filename,
                        'type': file.content_type,
                        'size': file_size
                    }
                })
                
            except IOError as e:
                logger.error(f"[{request_id}] Failed to save file: {str(e)}")
                return create_error_response(
                    'SAVE_ERROR',
                    'Failed to save uploaded file',
                    {'error': str(e)}
                ), 500
                
        except ValueError as e:
            logger.warning(f"[{request_id}] Validation error: {str(e)}")
            return create_error_response(
                'VALIDATION_ERROR',
                str(e),
                {
                    'max_size': app.config['MAX_FILESIZE'],
                    'allowed_types': list(app.config['ALLOWED_EXTENSIONS'])
                }
            ), 400
            
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        return create_error_response(
            'INTERNAL_ERROR',
            'An unexpected error occurred',
            {'error': str(e)}
        ), 500

@app.route('/upload-garment-image', methods=['POST'])
@limiter.limit("10 per minute", error_message="Too many uploads")
def upload_garment_image():
    """Handle garment image upload with rate limiting"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    filepath = app.config['UPLOAD_FOLDER'] / filename
    save_compressed_image(file, filepath)
    
    return jsonify({'filename': filename})

def validate_input_data(data):
    """Validate input data types and format
    Args:
        data: Dictionary containing input data
    Raises:
        ValueError: If input data is invalid
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid input data format")
        
    person_image = data.get('person_image')
    garment_image = data.get('garment_image')
    
    # Check if both fields exist
    if not person_image or not garment_image:
        raise ValueError("Both images are required")
        
    # Check data types
    if not isinstance(person_image, str) or not isinstance(garment_image, str):
        raise ValueError("Invalid input data type - filenames must be strings")
        
    # Check for empty strings
    if not person_image.strip() or not garment_image.strip():
        raise ValueError("Empty filename not allowed")
        
    # Check filename format
    if not allowed_file(person_image) or not allowed_file(garment_image):
        raise ValueError("Invalid filename format")

@app.route('/process-images', methods=['POST'])
@limiter.limit("5 per minute")
async def process_images():
    """Process uploaded person and garment images
    ---
    tags:
      - processing
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - person_image
            - garment_image
          properties:
            person_image:
              type: string
              example: person.jpg
            garment_image:
              type: string
              example: garment.jpg
    responses:
      200:
        description: Images processed successfully
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            data:
              type: object
              properties:
                result:
                  type: string
                  description: Base64 encoded result image
                format:
                  type: string
                  example: base64
      400:
        description: Invalid input
        schema:
          type: object
          properties:
            error:
              type: object
              properties:
                code:
                  type: string
                  example: INVALID_INPUT
                message:
                  type: string
                  example: Both images are required
      404:
        description: Image file not found
      429:
        description: Too many requests
    """
    try:
        data = request.get_json()
        if not data:
            return create_error_response(
                'INVALID_JSON',
                'Invalid JSON data'
            ), 400
            
        try:
            validate_input_data(data)
        except ValueError as e:
            return create_error_response(
                'INVALID_INPUT',
                str(e)
            ), 400
            
        person_image = data.get('person_image')
        garment_image = data.get('garment_image')
        
        # Check files exist
        person_path = app.config['UPLOAD_FOLDER'] / person_image
        garment_path = app.config['UPLOAD_FOLDER'] / garment_image
        
        if not person_path.exists():
            return create_error_response(
                'FILE_NOT_FOUND',
                'Person image file not found',
                {'filename': person_image}
            ), 404
            
        if not garment_path.exists():
            return create_error_response(
                'FILE_NOT_FOUND',
                'Garment image file not found',
                {'filename': garment_image}
            ), 404

        # Verarbeite Bilder asynchron
        result = await process_images_async(person_path, garment_path)
        
        # Cleanup
        try:
            person_path.unlink()
            garment_path.unlink()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
            
        return jsonify({
            'status': 'success',
            'data': {
                'result': result,
                'format': 'base64'
            }
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return create_error_response(
            'PROCESSING_FAILED',
            'Failed to process images',
            {'error': str(e)}
        ), 500

def send_to_kolors_api(person_image_path, garment_image_path):
    """Send images to Kolors API"""
    headers = {
        'Authorization': f'Bearer {app.config["HUGGINGFACE_API_KEY"]}'
    }
    
    try:
        logger.info(f"Starting API request with images: {person_image_path}, {garment_image_path}")
        start_time = datetime.now()
        
        if not person_image_path.exists():
            logger.error(f"Person image not found: {person_image_path}")
            raise FileNotFoundError("Person image file not found")
        if not garment_image_path.exists():
            logger.error(f"Garment image not found: {garment_image_path}")
            raise FileNotFoundError("Garment image file not found")
            
        with open(person_image_path, 'rb') as f1, open(garment_image_path, 'rb') as f2:
            files = {
                'person_image': f1,
                'garment_image': f2
            }
            try:
                response = requests.post(
                    app.config['API_URL'],
                    headers=headers,
                    files=files,
                    timeout=app.config['API_TIMEOUT']
                )
                response.raise_for_status()
                
                if not response.content:
                    logger.error("API returned empty response")
                    raise ValueError("No image data received from API")
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"API request completed in {duration:.2f} seconds")
                return response.content
                
            except requests.exceptions.Timeout:
                logger.error(f"API request timed out after {(datetime.now() - start_time).total_seconds():.2f} seconds")
                raise TimeoutError("API request timed out. Please try again later.")
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {str(e)}")
                raise ConnectionError(f"API error: {str(e)}")
                
    except IOError as e:
        logger.error(f"File operation error: {str(e)}")
        raise IOError(f"File error: {str(e)}")

@app.errorhandler(Exception)
def handle_error(e):
    if isinstance(e, RateLimitExceeded):
        return create_error_response(
            'RATE_LIMIT_EXCEEDED',
            'Too many requests. Please try again later.',
            {
                'limit': str(e.description),
                'reset': int(e.reset_time.timestamp())
            }
        ), 429
        
    return create_error_response(
        'INTERNAL_ERROR',
        'An unexpected error occurred',
        {'type': e.__class__.__name__}
    ), 500

@app.errorhandler(RateLimitExceeded)
def handle_rate_limit_error(e):
    logger.warning(f"Rate limit exceeded: {str(e.description)}")
    return jsonify({
        'error': 'Rate limit exceeded',
        'limit': str(e.description),
        'reset': int(e.reset_time.timestamp())
    }), 429

@app.route('/batch-process-images', methods=['POST'])
@limiter.limit("2 per minute")
async def batch_process_images():
    """Handle batch processing with progress tracking"""
    request_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    logger.info(f"[{request_id}] Starting batch processing")
    
    try:
        # Validate request
        data = request.get_json()
        if not data or not isinstance(data.get('images'), list):
            return create_error_response(
                'INVALID_INPUT',
                'Invalid input format',
                {'expected': {'images': [{'person_image': 'string', 'garment_image': 'string'}]}}
            ), 400
            
        images = data['images']
        total_images = len(images)
        logger.info(f"[{request_id}] Processing {total_images} image pairs")

        # Track progress
        processed = 0
        successful = 0
        failed = 0
        results = []

        # Process images
        for img_pair in images:
            try:
                # Validate pair
                if not isinstance(img_pair, dict):
                    raise ValueError("Invalid image pair format")
                    
                for key in ['person_image', 'garment_image']:
                    if key not in img_pair:
                        raise ValueError(f"Missing {key}")
                    if not isinstance(img_pair[key], str):
                        raise ValueError(f"Invalid {key} type")
                    if not allowed_file(img_pair[key]):
                        raise ValueError(f"Invalid {key} format")
                        
                # Process pair
                person_path = app.config['UPLOAD_FOLDER'] / img_pair['person_image']
                garment_path = app.config['UPLOAD_FOLDER'] / img_pair['garment_image']

                if not person_path.exists() or not garment_path.exists():
                    raise FileNotFoundError("One or more image files not found")

                result = await process_images_async(person_path, garment_path)
                
                results.append({
                    'person_image': img_pair['person_image'],
                    'garment_image': img_pair['garment_image'],
                    'status': 'success',
                    'result': result
                })
                successful += 1

            except Exception as e:
                logger.error(f"[{request_id}] Error processing pair: {str(e)}")
                results.append({
                    'person_image': img_pair.get('person_image'),
                    'garment_image': img_pair.get('garment_image'),
                    'status': 'error',
                    'error': str(e)
                })
                failed += 1

            processed += 1
            logger.info(f"[{request_id}] Progress: {processed}/{total_images}")

        # Cleanup
        cleanup_files = set()
        for img_pair in images:
            cleanup_files.add(app.config['UPLOAD_FOLDER'] / img_pair['person_image'])
            cleanup_files.add(app.config['UPLOAD_FOLDER'] / img_pair['garment_image'])
            
        for filepath in cleanup_files:
            try:
                if filepath.exists():
                    filepath.unlink()
            except Exception as e:
                logger.warning(f"[{request_id}] Cleanup error for {filepath}: {e}")

        logger.info(f"[{request_id}] Batch processing completed: {successful} successful, {failed} failed")
        
        return jsonify({
            'status': 'success',
            'data': {
                'total': total_images,
                'successful': successful,
                'failed': failed,
                'results': results
            }
        })
        
    except Exception as e:
        logger.error(f"[{request_id}] Batch processing error: {str(e)}", exc_info=True)
        return create_error_response(
            'BATCH_PROCESSING_ERROR',
            'Failed to process image batch',
            {'error': str(e)}
        ), 500

if __name__ == '__main__':
    app.run(debug=True) 