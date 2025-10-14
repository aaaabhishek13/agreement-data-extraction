#!/usr/bin/env python3
"""
REST API for Lease Agreement Data Extraction
Provides comprehensive API endpoints for document processing and data extraction
"""

import os
import sys
import uuid
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from dotenv import load_dotenv
load_dotenv()

# Import our services
from src.services.extraction_service import ExtractionService
from src.models.lease_models import LeaseAgreementData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)  # Enable CORS for all routes

# Configure upload settings
UPLOAD_FOLDER = Path('api_uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}

# Initialize extraction service
try:
    llm_provider = os.getenv('LLM_PROVIDER', 'gemini').lower()
    extraction_service = ExtractionService(
        llm_provider=llm_provider,
        enable_fallback=True
    )
    logger.info(f"Initialized extraction service with {llm_provider} provider")
except Exception as e:
    logger.error(f"Failed to initialize extraction service: {e}")
    extraction_service = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def create_error_response(error_message: str, status_code: int = 400) -> tuple:
    """Create standardized error response."""
    return jsonify({
        'success': False,
        'error': error_message,
        'timestamp': datetime.now().isoformat()
    }), status_code


def create_success_response(data: Dict[str, Any], message: str = "Success") -> Dict[str, Any]:
    """Create standardized success response."""
    return {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Test service connectivity
        if extraction_service:
            test_results = extraction_service.test_services()
            return jsonify({
                'status': 'healthy',
                'services': test_results,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'unhealthy',
                'error': 'Extraction service not initialized',
                'timestamp': datetime.now().isoformat()
            }), 503
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503


@app.route('/api/extract', methods=['POST'])
def extract_lease_data():
    """
    Main API endpoint for lease data extraction.
    
    Accepts:
    - file: Document file (PDF, DOCX, DOC, TXT)
    - llm_provider: Optional ('openai' or 'gemini', defaults to env setting)
    - processing_method: Optional ('text_extraction' or 'direct_upload', defaults to 'text_extraction')
    - include_citations: Optional (boolean, defaults to True)
    - include_token_usage: Optional (boolean, defaults to True)
    
    Returns:
    - extracted_data: Complete lease agreement data
    - citations: Page/section references for each field
    - token_usage: LLM API usage statistics
    - processing_metadata: File info and processing details
    """
    try:
        # Validate service availability
        if not extraction_service:
            return create_error_response("Extraction service not available", 503)
        
        # Validate request
        if 'file' not in request.files:
            return create_error_response("No file provided")
        
        file = request.files['file']
        if file.filename == '':
            return create_error_response("No file selected")
        
        if not allowed_file(file.filename):
            return create_error_response(
                f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Get optional parameters
        llm_provider = request.form.get('llm_provider', '').lower()
        processing_method = request.form.get('processing_method', 'text_extraction').lower()
        include_citations = request.form.get('include_citations', 'true').lower() == 'true'
        include_token_usage = request.form.get('include_token_usage', 'true').lower() == 'true'
        
        # Validate parameters
        if llm_provider and llm_provider not in ['openai', 'gemini']:
            return create_error_response("Invalid LLM provider. Use 'openai' or 'gemini'")
        
        if processing_method not in ['text_extraction', 'direct_upload']:
            return create_error_response("Invalid processing method. Use 'text_extraction' or 'direct_upload'")
        
        # Use default provider if not specified
        if not llm_provider:
            llm_provider = os.getenv('LLM_PROVIDER', 'gemini').lower()
        
        # Generate unique filename and save file
        file_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        file_extension = Path(original_filename).suffix
        saved_filename = f"{file_id}_{original_filename}"
        file_path = UPLOAD_FOLDER / saved_filename
        
        file.save(file_path)
        logger.info(f"Saved uploaded file: {saved_filename}")
        
        # Process the document
        start_time = time.time()
        
        # Update extraction service provider if different from default
        if llm_provider != extraction_service.primary_provider:
            logger.info(f"Switching LLM provider from {extraction_service.primary_provider} to {llm_provider}")
            extraction_service = ExtractionService(
                llm_provider=llm_provider,
                enable_fallback=True
            )
        
        # Determine processing method
        direct_upload = (processing_method == 'direct_upload')
        
        # Extract data
        result = extraction_service.extract_lease_data_from_file(
            file_path=str(file_path),
            direct_upload=direct_upload
        )
        
        processing_time = time.time() - start_time
        
        # Prepare response data
        response_data = {
            'extraction_result': {
                'success': result.success,
                'status': 'completed' if result.success else 'failed',
                'error': result.error if result.error else None
            },
            'file_metadata': {
                'original_filename': original_filename,
                'file_id': file_id,
                'file_size': file_path.stat().st_size,
                'file_extension': file_extension,
                'processing_method': processing_method,
                'llm_provider': llm_provider
            },
            'processing_metadata': {
                'processing_time_seconds': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'file_info': result.file_info
            }
        }
        
        # Add extracted data if successful
        if result.success and result.data:
            data_dict = result.data.model_dump()
            
            # Main extracted data (excluding metadata fields)
            extracted_data = {}
            for key, value in data_dict.items():
                if key not in ['extraction_status', 'provider', 'error', 'token_usage', 'citations']:
                    extracted_data[key] = value
            
            response_data['extracted_data'] = extracted_data
            
            # Add citations if requested and available
            if include_citations and 'citations' in data_dict and data_dict['citations']:
                response_data['citations'] = data_dict['citations']
            
            # Add token usage if requested and available
            if include_token_usage and 'token_usage' in data_dict and data_dict['token_usage']:
                response_data['token_usage'] = data_dict['token_usage']
            
            # Add summary statistics
            if hasattr(result.data, 'summary'):
                response_data['summary_statistics'] = result.data.summary()
        
        # Clean up uploaded file
        try:
            file_path.unlink()
            logger.info(f"Cleaned up file: {saved_filename}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup file {saved_filename}: {cleanup_error}")
        
        # Return success response
        status_code = 200 if result.success else 422  # 422 for processing errors
        return jsonify(create_success_response(
            response_data, 
            "Data extraction completed successfully" if result.success else "Data extraction failed"
        )), status_code
        
    except RequestEntityTooLarge:
        return create_error_response("File too large. Maximum size is 16MB", 413)
    except Exception as e:
        logger.error(f"API extraction error: {str(e)}", exc_info=True)
        return create_error_response(f"Internal server error: {str(e)}", 500)


@app.route('/api/extract/formats', methods=['GET'])
def get_supported_formats():
    """Get list of supported file formats."""
    try:
        formats_info = {
            'supported_extensions': list(ALLOWED_EXTENSIONS),
            'max_file_size_mb': 16,
            'descriptions': {
                '.pdf': 'Portable Document Format - Full page citation support',
                '.docx': 'Microsoft Word Document - Section-based citations',
                '.doc': 'Legacy Microsoft Word Document - Section-based citations',
                '.txt': 'Plain Text File - Line-based citations'
            }
        }
        
        return jsonify(create_success_response(formats_info, "Supported formats retrieved"))
    except Exception as e:
        return create_error_response(f"Error retrieving formats: {str(e)}", 500)


@app.route('/api/extract/providers', methods=['GET'])
def get_llm_providers():
    """Get available LLM providers and their capabilities."""
    try:
        providers_info = {
            'available_providers': ['openai', 'gemini'],
            'default_provider': os.getenv('LLM_PROVIDER', 'gemini'),
            'capabilities': {
                'openai': {
                    'text_extraction': True,
                    'direct_file_upload': False,
                    'citation_support': True,
                    'token_tracking': True,
                    'models': ['gpt-4', 'gpt-3.5-turbo']
                },
                'gemini': {
                    'text_extraction': True,
                    'direct_file_upload': True,
                    'citation_support': True,
                    'token_tracking': True,
                    'models': ['gemini-pro', 'gemini-2.5-pro']
                }
            }
        }
        
        return jsonify(create_success_response(providers_info, "LLM providers information retrieved"))
    except Exception as e:
        return create_error_response(f"Error retrieving providers: {str(e)}", 500)


@app.route('/api/docs', methods=['GET'])
def api_documentation():
    """API documentation endpoint."""
    docs = {
        'api_version': '1.0.0',
        'description': 'Lease Agreement Data Extraction API',
        'endpoints': {
            'POST /api/extract': {
                'description': 'Extract structured data from lease agreement documents',
                'parameters': {
                    'file': 'Document file (required)',
                    'llm_provider': 'LLM provider - "openai" or "gemini" (optional)',
                    'processing_method': 'Processing method - "text_extraction" or "direct_upload" (optional)',
                    'include_citations': 'Include page citations - true/false (optional, default: true)',
                    'include_token_usage': 'Include token usage stats - true/false (optional, default: true)'
                },
                'response': {
                    'extracted_data': 'Structured lease agreement data',
                    'citations': 'Page/section references for each field',
                    'token_usage': 'LLM API usage statistics',
                    'processing_metadata': 'File and processing information'
                }
            },
            'GET /api/health': {
                'description': 'Service health check',
                'response': 'Service status and availability'
            },
            'GET /api/extract/formats': {
                'description': 'Get supported file formats',
                'response': 'List of supported file extensions and descriptions'
            },
            'GET /api/extract/providers': {
                'description': 'Get available LLM providers and capabilities',
                'response': 'LLM provider information and capabilities'
            }
        },
        'example_request': {
            'curl': """curl -X POST http://localhost:8000/api/extract \\
  -F "file=@lease_agreement.pdf" \\
  -F "llm_provider=gemini" \\
  -F "processing_method=text_extraction" \\
  -F "include_citations=true\"""",
            'python': """
import requests

with open('lease_agreement.pdf', 'rb') as f:
    response = requests.post('http://localhost:8000/api/extract', 
        files={'file': f},
        data={
            'llm_provider': 'gemini',
            'processing_method': 'text_extraction',
            'include_citations': 'true'
        }
    )
    
result = response.json()
"""
        }
    }
    
    return jsonify(create_success_response(docs, "API documentation retrieved"))


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file too large errors."""
    return create_error_response("File too large. Maximum size is 16MB", 413)


@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors."""
    return create_error_response("Endpoint not found", 404)


@app.errorhandler(405)
def handle_method_not_allowed(e):
    """Handle method not allowed errors."""
    return create_error_response("Method not allowed", 405)


@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}", exc_info=True)
    return create_error_response("Internal server error", 500)


if __name__ == '__main__':
    print("🚀 Starting Lease Agreement Data Extraction API")
    print("=" * 50)
    print(f"📋 Supported file formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"🤖 Default LLM provider: {os.getenv('LLM_PROVIDER', 'gemini')}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📊 Max file size: 16MB")
    print("=" * 50)
    print("API Endpoints:")
    print("  POST /api/extract - Extract lease data from documents")
    print("  GET  /api/health - Service health check") 
    print("  GET  /api/extract/formats - Supported file formats")
    print("  GET  /api/extract/providers - Available LLM providers")
    print("  GET  /api/docs - API documentation")
    print("=" * 50)
    print("🌐 Starting server on http://localhost:8000")
    
    app.run(host='0.0.0.0', port=8000, debug=False)
