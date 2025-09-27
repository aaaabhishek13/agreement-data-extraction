#!/usr/bin/env python3
"""
Test script for enhanced JSON parsing capabilities.
"""

import sys
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_json_parsing_fixes():
    """Test the enhanced JSON parsing with common malformed JSON examples."""
    print("🧪 Testing Enhanced JSON Parsing")
    print("=" * 50)
    
    from services.llm_service import LLMService
    
    # Create a dummy service just to test the parsing methods
    import os
    os.environ['OPENAI_API_KEY'] = 'dummy'  # Set dummy key for testing
    
    try:
        llm_service = LLMService('openai')
    except:
        # If real initialization fails, we'll test the parsing methods directly
        llm_service = type('MockLLMService', (), {})()
        llm_service.provider = type('MockProvider', (), {'value': 'test'})()
        
        # Import the parsing methods directly
        from services.llm_service import LLMService
        real_service = LLMService.__new__(LLMService)
        real_service.provider = llm_service.provider
        llm_service._parse_response = real_service._parse_response.__get__(llm_service)
        llm_service._robust_json_parse = real_service._robust_json_parse.__get__(llm_service)
        llm_service._fix_common_json_issues = real_service._fix_common_json_issues.__get__(llm_service)
    
    # Test cases with common JSON issues
    test_cases = [
        {
            'name': 'Valid JSON',
            'input': '{"agreement_details": {"document_number": "DOC123"}, "parties": null}',
            'should_succeed': True
        },
        {
            'name': 'JSON with markdown wrapper',
            'input': '```json\n{"agreement_details": {"document_number": "DOC123"}}\n```',
            'should_succeed': True
        },
        {
            'name': 'JSON with trailing comma',
            'input': '{"agreement_details": {"document_number": "DOC123",}, "parties": null,}',
            'should_succeed': True
        },
        {
            'name': 'JSON with unquoted keys',
            'input': '{agreement_details: {"document_number": "DOC123"}, parties: null}',
            'should_succeed': True
        },
        {
            'name': 'JSON with single quotes',
            'input': "{'agreement_details': {'document_number': 'DOC123'}, 'parties': null}",
            'should_succeed': True
        },
        {
            'name': 'JSON with Python booleans',
            'input': '{"agreement_details": {"serviced_office": True}, "parties": None}',
            'should_succeed': True
        },
        {
            'name': 'JSON with extra text',
            'input': 'Here is the extracted data:\n{"agreement_details": {"document_number": "DOC123"}}\nEnd of response.',
            'should_succeed': True
        },
        {
            'name': 'Completely malformed JSON',
            'input': 'This is not JSON at all, just random text without any structure',
            'should_succeed': False
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        print(f"   Input: {test_case['input'][:100]}{'...' if len(test_case['input']) > 100 else ''}")
        
        try:
            result = llm_service._parse_response(test_case['input'])
            
            if test_case['should_succeed']:
                if result.get('extraction_status') in ['success', None] and 'error' not in result:
                    print("   ✅ PASSED - Successfully parsed")
                    passed += 1
                elif result.get('json_fixed'):
                    print("   ✅ PASSED - Parsed after fixing issues")
                    passed += 1
                else:
                    print(f"   ❌ FAILED - Expected success but got: {result.get('error', 'Unknown error')}")
                    failed += 1
            else:
                if result.get('extraction_status') == 'parsing_failed':
                    print("   ✅ PASSED - Correctly failed to parse malformed JSON")
                    passed += 1
                else:
                    print("   ❌ FAILED - Expected parsing failure but succeeded")
                    failed += 1
                    
        except Exception as e:
            if test_case['should_succeed']:
                print(f"   ❌ FAILED - Unexpected exception: {str(e)}")
                failed += 1
            else:
                print("   ✅ PASSED - Correctly threw exception for malformed input")
                passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 JSON Parsing Test Results: {passed} passed, {failed} failed")
    
    return failed == 0

def test_retry_mechanism():
    """Test the retry mechanism logic."""
    print("\n🔄 Testing Retry Mechanism Logic")
    print("=" * 50)
    
    # This would require mocking the LLM calls, so we'll just verify the method exists
    # and has the right signature
    from services.llm_service import LLMService
    import inspect
    
    # Check if extract_lease_data has max_retries parameter
    sig = inspect.signature(LLMService.extract_lease_data)
    if 'max_retries' in sig.parameters:
        print("✅ extract_lease_data method has max_retries parameter")
        print(f"   Default value: {sig.parameters['max_retries'].default}")
        return True
    else:
        print("❌ extract_lease_data method missing max_retries parameter")
        return False

def main():
    """Run JSON parsing enhancement tests."""
    print("🔧 JSON Parsing Enhancement Tests")
    print("=" * 60)
    
    tests = [
        ("JSON Parsing Fixes", test_json_parsing_fixes),
        ("Retry Mechanism", test_retry_mechanism),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All JSON parsing enhancement tests passed!")
        print("\n💡 The enhanced parsing should now handle:")
        print("   - Malformed JSON from LLM responses")
        print("   - Markdown code block wrappers")
        print("   - Trailing commas and unquoted keys")
        print("   - Python-style booleans and None values")
        print("   - Automatic retry on parsing failures")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
