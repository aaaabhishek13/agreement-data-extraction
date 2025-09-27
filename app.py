"""
Main application entry point for the lease agreement extraction tool.
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv

def main():
    """Main application entry point."""
    # Load environment variables
    load_dotenv()
    
    # Import here to ensure environment variables are loaded
    from src.web.app import LeaseExtractionApp
    
    # Get configuration from environment
    llm_provider = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
    upload_folder = os.getenv('UPLOAD_FOLDER', 'uploads')
    debug = os.getenv('FLASK_ENV') == 'development'
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', '5000'))
    
    print("="*60)
    print("🏢 Lease Agreement Data Extraction Tool")
    print("="*60)
    print(f"🤖 LLM Provider: {llm_provider}")
    print(f"📁 Upload folder: {upload_folder}")
    print(f"🌐 Server: http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print("="*60)
    print("📝 Supported formats: PDF, DOCX, DOC, TXT")
    print("📏 Max file size: 16MB")
    print("="*60)
    
    # Verify API keys
    missing_keys = []
    if llm_provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if llm_provider == 'gemini' and not os.getenv('GEMINI_API_KEY'):
        missing_keys.append('GEMINI_API_KEY')
    
    if missing_keys:
        print(f"⚠️  Warning: Missing API keys: {', '.join(missing_keys)}")
        print("   Please set them in your .env file or environment variables")
        print("="*60)
    
    try:
        # Create and run app
        app = LeaseExtractionApp(upload_folder, llm_provider)
        app.run(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
