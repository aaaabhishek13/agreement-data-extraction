#!/usr/bin/env python3
"""
Simple test to demonstrate JSON parsing logging without API calls.
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def setup_logging():
    """Setup detailed logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def test_json_parsing_with_logging():
    """Test the JSON parsing with detailed logging."""
    print("🔍 Testing JSON Parsing with Enhanced Logging")
    print("=" * 60)
    
    setup_logging()
    
    try:
        from services.llm_service import LLMService
        
        # Create a mock service to test parsing methods
        import os
        os.environ['OPENAI_API_KEY'] = 'dummy-key-for-testing'
        
        # We'll create the service but not call the API
        service = LLMService.__new__(LLMService)
        service.provider = type('MockProvider', (), {'value': 'test'})()
        
        # Get the parsing methods
        real_service = LLMService.__new__(LLMService)
        real_service.provider = service.provider
        service._parse_response = real_service._parse_response.__get__(service)
        service._robust_json_parse = real_service._robust_json_parse.__get__(service)
        service._fix_common_json_issues = real_service._fix_common_json_issues.__get__(service)
        
        print("\n🧪 Testing with malformed JSON response...")
        
        # Test with a malformed JSON that would come from an LLM
        malformed_response = '''Here is the extracted data:
```json
{
  "agreement_details": {
    "document_number": "DOC-2025-001",
    "project_name": "Tech Plaza"
  },
  "parties": {
    "landlord": {
      "name": "ABC Properties Inc.",
      "representative_name": "John Smith"
    },
    "tenant": {
      "name": "XYZ Business Corp."
    }
  },
  "lease_terms": {
    "monthly_rent": 2500,
    "lease_start_date": "2025-10-01",
  }
}
```
End of response.'''

        print(f"\nSimulated LLM Response (first 200 chars):")
        print(f"{malformed_response[:200]}...")
        
        print("\n" + "="*60)
        print("📝 DETAILED JSON PARSING LOG OUTPUT:")
        print("="*60)
        
        # This will show all our enhanced logging
        result = service._parse_response(malformed_response)
        
        print("\n" + "="*60)
        print("📊 PARSING RESULTS:")
        print("="*60)
        
        if result.get('extraction_status') == 'parsing_failed':
            print(f"❌ Parsing failed: {result.get('error')}")
            print("🔧 This demonstrates how parsing errors are logged")
        else:
            print("✅ Parsing succeeded!")
            if result.get('json_fixed'):
                print("🔧 JSON was automatically fixed during parsing")
            
            # Show some parsed data
            if 'parties' in result:
                parties = result.get('parties', {})
                if parties and isinstance(parties, dict):
                    landlord = parties.get('landlord', {})
                    tenant = parties.get('tenant', {})
                    print(f"🏢 Landlord: {landlord.get('name', 'N/A')}")
                    print(f"🏬 Tenant: {tenant.get('name', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the JSON parsing logging test."""
    print("🔍 Enhanced JSON Parsing Logging Demonstration")
    print("=" * 60)
    print("This demonstrates the detailed logging added for:")
    print("  📝 Response cleaning and preparation")
    print("  🔧 Multiple JSON parsing strategies")
    print("  🚨 Error detection and troubleshooting")
    print("  ✅ Success indicators and metadata")
    print("=" * 60)
    
    success = test_json_parsing_with_logging()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 JSON parsing logging demonstration completed!")
        print("\n💡 When processing real documents, you'll see:")
        print("  📤 Complete prompt details (truncated)")
        print("  📥 Raw LLM response details (truncated)")
        print("  🔧 Step-by-step JSON parsing attempts")
        print("  🚨 Detailed error messages for troubleshooting")
        print("  🔄 Retry attempt information")
    else:
        print("⚠️  Demonstration encountered issues.")
    
    print(f"\n📁 To see full logging during extraction:")
    print(f"   Set LOG_TO_FILE=true in .env to save logs to extraction.log")
    print(f"   Or run the Flask app to see console logging during uploads")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
