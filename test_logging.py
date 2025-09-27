#!/usr/bin/env python3
"""
Test script to demonstrate the detailed logging of LLM prompts and responses.
"""

import os
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

def test_prompt_and_response_logging():
    """Test the enhanced logging of prompts and responses."""
    print("🔍 Testing Enhanced LLM Logging")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if we have API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    if not (openai_key or gemini_key):
        print("⚠️  No API keys found. Skipping live test.")
        return True
    
    # Choose a provider that has an API key
    provider = 'openai' if openai_key else 'gemini'
    print(f"🤖 Using {provider.upper()} provider for logging test")
    
    try:
        from services.llm_service import LLMService
        
        # Initialize service
        llm_service = LLMService(provider=provider)
        print(f"✅ Initialized {provider} service")
        
        # Test with a simple lease document
        test_document = """
        SIMPLE LEASE AGREEMENT
        
        This lease agreement is between:
        - Landlord: ABC Properties Inc.
        - Tenant: XYZ Business Corp.
        
        Property Details:
        - Unit: Suite 100
        - Monthly Rent: $2,500
        - Security Deposit: $5,000
        - Lease Term: 24 months
        - Start Date: 2025-10-01
        - End Date: 2027-09-30
        
        Additional Terms:
        - Parking: 2 spaces included
        - Utilities: Tenant responsible
        """
        
        print("\n" + "="*60)
        print("📝 STARTING EXTRACTION WITH DETAILED LOGGING")
        print("="*60)
        
        # This will trigger all the detailed logging we added
        result = llm_service.extract_lease_data(test_document)
        
        print("\n" + "="*60)
        print("📊 EXTRACTION RESULTS")
        print("="*60)
        
        if result.get('extraction_status') == 'failed':
            print(f"❌ Extraction failed: {result.get('error')}")
            return False
        else:
            print("✅ Extraction completed successfully!")
            
            # Show some key extracted data
            if 'parties' in result and result['parties']:
                print(f"🏢 Landlord: {result['parties'].get('landlord', {}).get('name', 'N/A')}")
                print(f"🏬 Tenant: {result['parties'].get('tenant', {}).get('name', 'N/A')}")
            
            if 'lease_terms' in result and result['lease_terms']:
                print(f"💰 Monthly Rent: {result['lease_terms'].get('monthly_rent', 'N/A')}")
                print(f"📅 Start Date: {result['lease_terms'].get('lease_start_date', 'N/A')}")
            
            # Check if any special parsing was used
            if result.get('json_fixed'):
                print("🔧 JSON was automatically fixed during parsing")
            if result.get('retry_attempts'):
                print(f"🔄 Required {result['retry_attempts']} attempts to succeed")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

def main():
    """Run the logging test."""
    setup_logging()
    
    print("🔍 LLM Prompt and Response Logging Test")
    print("=" * 60)
    print("This test will show detailed logging of:")
    print("  📤 Prompts sent to the LLM")
    print("  📥 Raw responses received")
    print("  🔧 JSON parsing steps")
    print("  🔄 Retry attempts (if needed)")
    print("=" * 60)
    
    success = test_prompt_and_response_logging()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Logging test completed successfully!")
        print("\n💡 You can now see detailed logs when processing documents:")
        print("  - Full prompt content (truncated for readability)")
        print("  - Raw LLM responses (truncated for readability)")
        print("  - JSON parsing strategies attempted")
        print("  - Error details and retry attempts")
    else:
        print("⚠️  Logging test encountered issues.")
        print("Check the log output above for detailed information.")
    
    print("\n📝 Note: In production, you may want to reduce log level")
    print("      to WARNING or ERROR to avoid verbose output.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
