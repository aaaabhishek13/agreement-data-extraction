#!/usr/bin/env python3
"""
Test script specifically for testing Gemini API integration with safety filters.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_gemini_safety_handling():
    """Test Gemini API with improved safety filter handling."""
    print("🧪 Testing Gemini API with Enhanced Safety Handling")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  GEMINI_API_KEY not found in environment. Skipping Gemini test.")
        return True
    
    try:
        from services.llm_service import LLMService
        
        # Initialize Gemini service
        print("🤖 Initializing Gemini LLM service...")
        llm_service = LLMService(provider='gemini')
        print("✅ Gemini service initialized successfully")
        
        # Test with a simple lease document text
        test_document = """
        LEASE AGREEMENT
        
        This lease agreement is made between ABC Properties (Landlord) and XYZ Corporation (Tenant).
        
        Property Details:
        - Unit Number: 501
        - Floor: 5th Floor
        - Area: 1200 sq ft
        - Monthly Rent: $5,000
        - Security Deposit: $15,000
        - Lease Period: 36 months
        - Start Date: January 1, 2025
        - End Date: December 31, 2027
        
        The tenant agrees to pay monthly rent of $5,000 on the 1st of each month.
        Security deposit of $15,000 is required before occupancy.
        """
        
        print("📄 Testing extraction with sample lease document...")
        result = llm_service.extract_lease_data(test_document)
        
        if result.get('extraction_status') == 'failed':
            print(f"❌ Extraction failed: {result.get('error')}")
            
            # Check if it's a safety filter issue
            if 'safety' in result.get('error', '').lower() or 'blocked' in result.get('error', '').lower():
                print("🛡️  This appears to be a Gemini safety filter issue")
                print("💡 Suggestions:")
                print("   - Try using OpenAI provider instead")
                print("   - Check if document contains sensitive content")
                print("   - The system will automatically fallback to OpenAI if available")
            
            return False
        else:
            print("✅ Extraction completed successfully!")
            print(f"📊 Provider used: {result.get('provider', 'unknown')}")
            
            # Check if fallback was used
            if result.get('fallback_used'):
                print(f"🔄 Fallback was used from {result.get('primary_provider')} to {result.get('fallback_provider')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

def test_fallback_mechanism():
    """Test the automatic fallback mechanism."""
    print("\n🔄 Testing Automatic Fallback Mechanism")
    print("=" * 60)
    
    # Check if both API keys are available
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    if not openai_key and not gemini_key:
        print("⚠️  No API keys found. Skipping fallback test.")
        return True
    
    if not (openai_key and gemini_key):
        print(f"⚠️  Only one API key found. Fallback requires both providers.")
        print(f"   OpenAI key: {'✅' if openai_key else '❌'}")
        print(f"   Gemini key: {'✅' if gemini_key else '❌'}")
        return True
    
    try:
        from services.extraction_service import LeaseExtractionService
        
        print("🤖 Testing fallback with Gemini as primary...")
        
        # Create service with Gemini as primary and fallback enabled
        extraction_service = LeaseExtractionService('gemini', enable_fallback=True)
        
        test_text = """
        COMMERCIAL LEASE AGREEMENT
        
        Landlord: ABC Real Estate LLC
        Tenant: Tech Solutions Inc.
        Property: Office Suite 1200, Tech Tower
        Monthly Rent: $8,500
        Lease Term: 60 months
        Security Deposit: $25,500
        """
        
        result = extraction_service.extract_from_text(test_text)
        
        if result.success:
            print("✅ Extraction successful!")
            if result.data and hasattr(result.data, 'fallback_used'):
                print("🔄 Fallback mechanism working correctly")
            else:
                print("✅ Primary provider worked without fallback needed")
        else:
            print(f"❌ Extraction failed: {result.error}")
            
        return result.success
        
    except Exception as e:
        print(f"❌ Fallback test failed: {str(e)}")
        return False

def main():
    """Run Gemini-specific tests."""
    print("🚀 Gemini API Integration Tests")
    print("=" * 60)
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    tests = [
        ("Gemini Safety Handling", test_gemini_safety_handling),
        ("Fallback Mechanism", test_fallback_mechanism),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"💥 {test_name} - CRASHED: {str(e)}")
        
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Gemini tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        
        if not os.getenv('GEMINI_API_KEY'):
            print("\n💡 Note: Make sure you have a valid GEMINI_API_KEY in your .env file")
            print("   You can get one from: https://makersuite.google.com/app/apikey")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
