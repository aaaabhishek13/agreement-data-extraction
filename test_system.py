#!/usr/bin/env python3
"""
Test script for the lease agreement extraction tool.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Test core dependencies
        import flask
        print("✅ Flask imported successfully")
        
        import pydantic
        print("✅ Pydantic imported successfully")
        
        import dotenv
        print("✅ Python-dotenv imported successfully")
        
        # Test optional dependencies
        try:
            import PyPDF2
            print("✅ PyPDF2 imported successfully")
        except ImportError:
            print("⚠️  PyPDF2 not available")
        
        try:
            import docx
            print("✅ Python-docx imported successfully")
        except ImportError:
            print("⚠️  Python-docx not available")
        
        try:
            import openai
            print("✅ OpenAI library imported successfully")
        except ImportError:
            print("⚠️  OpenAI library not available")
        
        try:
            import google.generativeai
            print("✅ Google GenerativeAI imported successfully")
        except ImportError:
            print("⚠️  Google GenerativeAI not available")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_project_structure():
    """Test if the project structure is correct."""
    print("\n🏗️  Testing project structure...")
    
    required_files = [
        "src/models/__init__.py",
        "src/models/lease_models.py",
        "src/services/__init__.py",
        "src/services/document_processor.py",
        "src/services/llm_service.py",
        "src/services/extraction_service.py",
        "src/web/app.py",
        "src/web/templates/index.html",
        "src/web/templates/results.html",
        "requirements.txt",
        "README.md",
        ".env.example"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    # Check if uploads directory exists or can be created
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        try:
            uploads_dir.mkdir()
            print("✅ Created uploads directory")
        except Exception as e:
            print(f"❌ Failed to create uploads directory: {e}")
            return False
    else:
        print("✅ uploads directory exists")
    
    return True

def test_models():
    """Test if the Pydantic models work correctly."""
    print("\n📋 Testing Pydantic models...")
    
    try:
        from src.models.lease_models import LeaseAgreementData, ExtractionResult
        
        # Test basic model creation
        lease_data = LeaseAgreementData()
        print("✅ LeaseAgreementData model created")
        
        result = ExtractionResult(success=True)
        print("✅ ExtractionResult model created")
        
        # Test model serialization
        data_dict = lease_data.model_dump()
        print("✅ Model serialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_services():
    """Test if the services can be initialized."""
    print("\n🔧 Testing services...")
    
    try:
        from src.services.document_processor import DocumentProcessor
        
        # Test document processor
        doc_processor = DocumentProcessor()
        supported_formats = doc_processor.SUPPORTED_FORMATS
        print(f"✅ DocumentProcessor initialized (supports: {', '.join(supported_formats)})")
        
        # Test file validation
        validation = doc_processor.validate_file("nonexistent.pdf")
        assert not validation['valid']
        print("✅ File validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Services test failed: {e}")
        return False

def test_environment():
    """Test environment setup."""
    print("\n🌍 Testing environment...")
    
    # Check if .env.example exists
    if Path(".env.example").exists():
        print("✅ .env.example file exists")
    else:
        print("⚠️  .env.example file missing")
    
    # Check if .env file exists
    if Path(".env").exists():
        print("✅ .env file exists")
        
        # Load and check environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for API keys (warn if missing)
        if not os.getenv('OPENAI_API_KEY') and not os.getenv('GEMINI_API_KEY'):
            print("⚠️  No API keys found in environment. Add them to .env file to test LLM functionality.")
        else:
            if os.getenv('OPENAI_API_KEY'):
                print("✅ OpenAI API key found")
            if os.getenv('GEMINI_API_KEY'):
                print("✅ Gemini API key found")
    else:
        print("⚠️  .env file not found. Copy .env.example to .env and add your API keys.")
    
    return True

def create_sample_env():
    """Create a sample .env file if it doesn't exist."""
    env_path = Path(".env")
    if not env_path.exists():
        print("\n📝 Creating sample .env file...")
        try:
            with open(".env.example", "r") as example_file:
                content = example_file.read()
            
            with open(".env", "w") as env_file:
                env_file.write(content)
            
            print("✅ Created .env file from .env.example")
            print("📝 Please edit .env file and add your API keys")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    return True

def main():
    """Run all tests."""
    print("🚀 Lease Agreement Extraction Tool - System Test")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Pydantic Models", test_models),
        ("Services", test_services),
        ("Environment", test_environment),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
        
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The system is ready to use.")
        
        # Try to create .env if it doesn't exist
        create_sample_env()
        
        print("\n🚀 To start the application:")
        print("   1. Add your API keys to the .env file")
        print("   2. Run: python app.py")
        print("   3. Open http://localhost:5000 in your browser")
        
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
