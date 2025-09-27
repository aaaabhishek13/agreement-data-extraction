"""
Flask web application for the lease agreement extraction tool.
"""

import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Import our services
import sys
sys.path.append(str(Path(__file__).parent.parent))

from services.extraction_service import LeaseExtractionService
from models.lease_models import ExtractionResult


class LeaseExtractionApp:
    """Flask application for lease extraction."""
    
    def __init__(self, upload_folder: str = "uploads", llm_provider: str = "openai"):
        """Initialize the Flask app."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.FileHandler('extraction.log') if os.getenv('LOG_TO_FILE', 'false').lower() == 'true' else logging.NullHandler()
            ]
        )
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        
        self.upload_folder = Path(upload_folder).absolute()
        self.upload_folder.mkdir(exist_ok=True)
        
        # Initialize extraction service with fallback enabled
        try:
            self.extraction_service = LeaseExtractionService(llm_provider, enable_fallback=True)
        except Exception as e:
            print(f"Failed to initialize extraction service: {e}")
            self.extraction_service = None
        
        # In-memory storage for results (in production, use Redis or database)
        self.results_storage = {}
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page with upload form."""
            return render_template('index.html')
        
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            """Handle file upload and processing."""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                if not self.extraction_service:
                    return jsonify({'error': 'Extraction service not available'}), 500
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                task_id = str(uuid.uuid4())
                file_path = self.upload_folder / f"{task_id}_{filename}"
                
                file.save(file_path)
                
                # Get LLM provider from request
                llm_provider = request.form.get('llm_provider', 'openai')
                
                # Process the file
                try:
                    if llm_provider != self.extraction_service.llm_service.provider.value:
                        # Reinitialize service with new provider and fallback enabled
                        self.extraction_service = LeaseExtractionService(llm_provider, enable_fallback=True)
                    
                    result = self.extraction_service.extract_from_file(str(file_path))
                    
                    # Store result
                    self.results_storage[task_id] = {
                        'result': result,
                        'filename': filename,
                        'timestamp': datetime.now().isoformat(),
                        'llm_provider': llm_provider
                    }
                    
                    # Clean up uploaded file
                    try:
                        file_path.unlink()
                    except:
                        pass  # Ignore cleanup errors
                    
                    # Enhanced error message handling
                    message = 'Processing completed'
                    if not result.success:
                        error_msg = result.error or 'Unknown error'
                        
                        # Provide user-friendly error messages
                        if 'JSON parsing error' in error_msg or 'parsing_failed' in str(result.data):
                            message = "The AI model's response couldn't be parsed. This sometimes happens with complex documents. Try using a different LLM provider or a simpler document format."
                        elif 'safety' in error_msg.lower() or 'blocked' in error_msg.lower():
                            message = "The document was blocked by content safety filters. Try using the OpenAI provider instead, or check if the document contains sensitive information."
                        elif 'quota' in error_msg.lower() or 'limit' in error_msg.lower():
                            message = "API usage limit exceeded. Please check your API quota or try again later."
                        elif 'authentication' in error_msg.lower() or 'api key' in error_msg.lower():
                            message = "API authentication failed. Please check your API keys in the .env file."
                        elif 'Both providers failed' in error_msg:
                            message = "Both AI providers failed to process this document. The document might be too complex or contain unsupported content."
                        else:
                            message = f"Processing failed: {error_msg}"
                    
                    response_data = {
                        'task_id': task_id,
                        'success': result.success,
                        'message': message
                    }
                    
                    # Add additional info for debugging if extraction partially succeeded
                    if result.data and hasattr(result.data, 'extraction_status'):
                        if result.data.extraction_status == 'parsing_failed':
                            response_data['parsing_issue'] = True
                        elif result.data.extraction_status == 'validation_warning':
                            response_data['validation_warning'] = True
                    
                    return jsonify(response_data)
                    
                except Exception as e:
                    # Clean up uploaded file
                    try:
                        file_path.unlink()
                    except:
                        pass
                    
                    return jsonify({'error': f'Processing failed: {str(e)}'}), 500
                
            except Exception as e:
                return jsonify({'error': f'Upload failed: {str(e)}'}), 500
        
        @self.app.route('/result/<task_id>')
        def get_result(task_id):
            """Get extraction result."""
            if task_id not in self.results_storage:
                return jsonify({'error': 'Result not found'}), 404
            
            stored_result = self.results_storage[task_id]
            result = stored_result['result']
            
            # Convert to dict for JSON serialization
            result_dict = {
                'success': result.success,
                'processing_time': result.processing_time,
                'file_info': result.file_info,
                'filename': stored_result['filename'],
                'timestamp': stored_result['timestamp'],
                'llm_provider': stored_result['llm_provider']
            }
            
            if result.error:
                result_dict['error'] = result.error
            
            if result.data:
                result_dict['data'] = result.data.model_dump()
                result_dict['summary'] = result.data.summary()
            
            return jsonify(result_dict)
        
        @self.app.route('/results')
        def results_page():
            """Results display page."""
            task_id = request.args.get('task_id')
            if not task_id:
                return redirect(url_for('index'))
            
            if task_id not in self.results_storage:
                flash('Result not found', 'error')
                return redirect(url_for('index'))
            
            return render_template('results.html', task_id=task_id)
        
        @self.app.route('/api/test')
        def test_services():
            """Test service connectivity."""
            if not self.extraction_service:
                return jsonify({
                    'status': 'error',
                    'message': 'Extraction service not initialized'
                }), 500
            
            try:
                test_results = self.extraction_service.test_services()
                return jsonify({
                    'status': 'success',
                    'results': test_results
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Service test failed: {str(e)}'
                }), 500
        
        @self.app.route('/api/supported-formats')
        def get_supported_formats():
            """Get supported file formats."""
            if not self.extraction_service:
                return jsonify({'formats': []}), 500
            
            try:
                formats = self.extraction_service.get_supported_formats()
                return jsonify({'formats': formats})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.errorhandler(413)
        def too_large(e):
            """Handle file too large error."""
            return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
        
        @self.app.errorhandler(404)
        def not_found(e):
            """Handle 404 errors."""
            return render_template('404.html'), 404
        
        @self.app.errorhandler(500)
        def internal_error(e):
            """Handle internal errors."""
            return render_template('500.html'), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application."""
        self.app.run(host=host, port=port, debug=debug)


def create_app(upload_folder: str = "uploads", llm_provider: str = "openai") -> Flask:
    """Factory function to create Flask app."""
    lease_app = LeaseExtractionApp(upload_folder, llm_provider)
    return lease_app.app


if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get configuration from environment
    llm_provider = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
    upload_folder = os.getenv('UPLOAD_FOLDER', 'uploads')
    debug = os.getenv('FLASK_ENV') == 'development'
    
    # Create and run app
    app = LeaseExtractionApp(upload_folder, llm_provider)
    print(f"Starting Lease Agreement Extraction Tool...")
    print(f"LLM Provider: {llm_provider}")
    print(f"Upload folder: {upload_folder}")
    print("Open http://localhost:5000 in your browser")
    
    app.run(debug=debug)
