"""
Large Language Model integration service for data extraction.
Supports both OpenAI GPT and Google Gemini models.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from enum import Enum

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMService:
    """Service for interacting with Large Language Models."""
    
    def __init__(self, provider: Union[str, LLMProvider] = LLMProvider.OPENAI):
        """
        Initialize LLM service.
        
        Args:
            provider: LLM provider to use (openai or gemini)
        """
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        self.provider = provider
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == LLMProvider.OPENAI:
            if openai is None:
                raise ImportError("openai package is required for OpenAI provider")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            self.client = openai.OpenAI(api_key=api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4")
            
        elif self.provider == LLMProvider.GEMINI:
            if genai is None:
                raise ImportError("google-generativeai package is required for Gemini provider")
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required")
            
            genai.configure(api_key=api_key)
            self.model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
            self.client = genai.GenerativeModel(self.model_name)
            
        logger.info(f"Initialized {self.provider.value} LLM service")
    
    def extract_lease_data(self, document_text: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Extract structured data from lease agreement text with retry logic.
        
        Args:
            document_text: Raw text extracted from the lease document
            max_retries: Maximum number of retry attempts for parsing failures
            
        Returns:
            Dict containing extracted structured data
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                prompt = self._create_extraction_prompt(document_text)
                
                # Add retry-specific instructions for subsequent attempts
                if attempt > 0:
                    prompt += f"\n\nIMPORTANT: This is retry attempt {attempt + 1}. Please ensure your response is valid JSON format without any markdown formatting or extra text. Return ONLY the JSON object."
                
                # Log the prompt being sent
                logger.info(f"=== PROMPT SENT TO {self.provider.value.upper()} (Attempt {attempt + 1}) ===")
                logger.info(f"Prompt length: {len(prompt)} characters")
                logger.info(f"Document text length: {len(document_text)} characters")
                
                # Check if full content logging is enabled
                log_full_content = os.getenv('LOG_FULL_CONTENT', 'false').lower() == 'true'
                
                if log_full_content:
                    logger.info("=== COMPLETE PROMPT ===")
                    logger.info(prompt)
                    logger.info("=== END COMPLETE PROMPT ===")
                else:
                    # Log first and last parts of the prompt for debugging
                    prompt_preview = prompt[:500] + "...[TRUNCATED]..." + prompt[-200:] if len(prompt) > 700 else prompt
                    logger.info(f"Prompt preview: {prompt_preview}")
                
                if self.provider == LLMProvider.OPENAI:
                    response = self._call_openai(prompt)
                elif self.provider == LLMProvider.GEMINI:
                    response = self._call_gemini(prompt)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                
                # Log the raw response received
                logger.info(f"=== RAW RESPONSE FROM {self.provider.value.upper()} (Attempt {attempt + 1}) ===")
                logger.info(f"Response length: {len(response)} characters")
                
                # Check if full content logging is enabled
                log_full_content = os.getenv('LOG_FULL_CONTENT', 'false').lower() == 'true'
                
                if log_full_content:
                    logger.info("=== COMPLETE RAW RESPONSE ===")
                    logger.info(response)
                    logger.info("=== END COMPLETE RAW RESPONSE ===")
                else:
                    # Log first and last parts of the response for debugging
                    response_preview = response[:500] + "...[TRUNCATED]..." + response[-200:] if len(response) > 700 else response
                    logger.info(f"Raw response: {response_preview}")
                
                # Parse the JSON response
                extracted_data = self._parse_response(response)
                
                # If parsing was successful, return the data
                if extracted_data.get('extraction_status') != 'parsing_failed':
                    if attempt > 0:
                        extracted_data['retry_attempts'] = attempt + 1
                    
                    # Add token usage information if available
                    if hasattr(self, '_last_token_usage'):
                        extracted_data['token_usage'] = self._last_token_usage.copy()
                    
                    logger.info(f"Successfully extracted lease data using LLM (attempt {attempt + 1})")
                    return extracted_data
                
                # If parsing failed, save the error and retry
                last_error = extracted_data.get('error', 'JSON parsing failed')
                logger.warning(f"Attempt {attempt + 1} failed with parsing error: {last_error}")
                
                if attempt < max_retries:
                    logger.info(f"Retrying extraction (attempt {attempt + 2}/{max_retries + 1})...")
                    continue
                else:
                    # Return the parsing error if all retries exhausted
                    return extracted_data
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt + 1} failed with error: {last_error}")
                
                if attempt < max_retries:
                    logger.info(f"Retrying extraction (attempt {attempt + 2}/{max_retries + 1})...")
                    continue
        
        # If all attempts failed, return the last error
        return {
            'error': f'All {max_retries + 1} attempts failed. Last error: {last_error}',
            'extraction_status': 'failed',
            'provider': self.provider.value,
            'total_attempts': max_retries + 1
        }
    
    def extract_from_file(self, file_path: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Extract lease data by sending file directly to LLM (Gemini only for now).
        
        Args:
            file_path: Path to the document file
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict containing extracted data or error information
        """
        try:
            if self.provider != LLMProvider.GEMINI:
                raise ValueError(f"Direct file upload not supported for {self.provider.value}. Only Gemini supports file uploads currently.")
            
            import mimetypes
            from pathlib import Path
            
            file_path = Path(file_path)
            if not file_path.exists():
                return {
                    'error': f'File not found: {file_path}',
                    'extraction_status': 'failed',
                    'provider': self.provider.value
                }
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                # Default MIME types for common document formats
                ext = file_path.suffix.lower()
                mime_map = {
                    '.pdf': 'application/pdf',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.doc': 'application/msword',
                    '.txt': 'text/plain'
                }
                mime_type = mime_map.get(ext, 'application/octet-stream')
            
            # Validate file format for Gemini
            supported_formats = ['.pdf', '.docx', '.doc', '.txt']
            if file_path.suffix.lower() not in supported_formats:
                raise ValueError(f"Unsupported file format {file_path.suffix}. Supported formats: {supported_formats}")
            
            # Check file size (Gemini has limits)
            file_size = file_path.stat().st_size
            max_size = 20 * 1024 * 1024  # 20MB limit for Gemini
            if file_size > max_size:
                raise ValueError(f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum size: {max_size / 1024 / 1024}MB")
            
            logger.info(f"=== DIRECT FILE UPLOAD TO GEMINI ===")
            logger.info(f"File: {file_path.name}")
            logger.info(f"Size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
            logger.info(f"MIME type: {mime_type}")
            
            try:
                # For DOCX files, try with explicit MIME type first
                if file_path.suffix.lower() == '.docx':
                    logger.info("Attempting DOCX upload with explicit MIME type")
                    try:
                        uploaded_file = genai.upload_file(
                            path=str(file_path), 
                            mime_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                        )
                        logger.info(f"DOCX upload successful. URI: {uploaded_file.uri}")
                    except Exception as docx_error:
                        logger.warning(f"DOCX upload failed with explicit MIME type: {str(docx_error)}")
                        # Try with generic application type
                        try:
                            uploaded_file = genai.upload_file(path=str(file_path), mime_type='application/octet-stream')
                            logger.info(f"DOCX upload successful with generic MIME type. URI: {uploaded_file.uri}")
                        except Exception as generic_error:
                            logger.error(f"DOCX upload failed with both MIME types: {str(generic_error)}")
                            raise ValueError(f"DOCX file upload not supported by Gemini API: {str(generic_error)}")
                else:
                    # For other file types, use detected MIME type
                    uploaded_file = genai.upload_file(path=str(file_path), mime_type=mime_type)
                    logger.info(f"Uploaded file URI: {uploaded_file.uri}")
            except Exception as upload_error:
                raise ValueError(f"Failed to upload file to Gemini: {str(upload_error)}")
            
            # Create prompt for file-based extraction
            prompt = self._create_file_extraction_prompt()
            
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"=== GEMINI FILE EXTRACTION (Attempt {attempt + 1}) ===")
                    
                    # Call Gemini with uploaded file
                    response = self._call_gemini_with_file(prompt, uploaded_file)
                    
                    # Log the raw response received
                    logger.info(f"=== RAW RESPONSE FROM GEMINI FILE (Attempt {attempt + 1}) ===")
                    logger.info(f"Response length: {len(response)} characters")
                    
                    # Check if full content logging is enabled
                    log_full_content = os.getenv('LOG_FULL_CONTENT', 'false').lower() == 'true'
                    
                    if log_full_content:
                        logger.info("=== COMPLETE RAW RESPONSE ===")
                        logger.info(response)
                        logger.info("=== END COMPLETE RAW RESPONSE ===")
                    else:
                        # Log first and last parts of the response for debugging
                        response_preview = response[:500] + "...[TRUNCATED]..." + response[-200:] if len(response) > 700 else response
                        logger.info(f"Raw response: {response_preview}")
                    
                    # Parse the JSON response
                    extracted_data = self._parse_response(response)
                    
                    # If parsing was successful, return the data
                    if extracted_data.get('extraction_status') != 'parsing_failed':
                        if attempt > 0:
                            extracted_data['retry_attempts'] = attempt + 1
                        extracted_data['extraction_method'] = 'direct_file_upload'
                        
                        # Add token usage information if available
                        if hasattr(self, '_last_token_usage'):
                            extracted_data['token_usage'] = self._last_token_usage.copy()
                        
                        logger.info(f"Successfully extracted lease data from file (attempt {attempt + 1})")
                        
                        # Clean up uploaded file
                        try:
                            genai.delete_file(uploaded_file.name)
                            logger.info("Cleaned up uploaded file")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup uploaded file: {cleanup_error}")
                        
                        return extracted_data
                    
                    # If parsing failed, save the error and retry
                    last_error = extracted_data.get('error', 'JSON parsing failed')
                    logger.warning(f"Attempt {attempt + 1} failed with parsing error: {last_error}")
                    
                    if attempt < max_retries:
                        logger.info(f"Retrying file extraction (attempt {attempt + 2}/{max_retries + 1})...")
                        continue
                    else:
                        # Clean up uploaded file
                        try:
                            genai.delete_file(uploaded_file.name)
                        except Exception:
                            pass
                        return extracted_data
                        
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"File extraction attempt {attempt + 1} failed with error: {last_error}")
                    
                    if attempt < max_retries:
                        logger.info(f"Retrying file extraction (attempt {attempt + 2}/{max_retries + 1})...")
                        continue
            
            # Clean up uploaded file
            try:
                genai.delete_file(uploaded_file.name)
            except Exception:
                pass
            
            # If all attempts failed, return the last error with special handling for DOCX
            if file_path.suffix.lower() == '.docx' and last_error and ('invalid argument' in str(last_error).lower() or '400' in str(last_error)):
                return {
                    'error': f'DOCX direct upload failed after {max_retries + 1} attempts. This file format may not be fully supported by Gemini file API. Please try text extraction method instead. Last error: {last_error}',
                    'extraction_status': 'failed',
                    'provider': self.provider.value,
                    'total_attempts': max_retries + 1,
                    'extraction_method': 'direct_file_upload',
                    'suggested_method': 'text_extraction'
                }
            
            return {
                'error': f'All {max_retries + 1} file extraction attempts failed. Last error: {last_error}',
                'extraction_status': 'failed',
                'provider': self.provider.value,
                'total_attempts': max_retries + 1,
                'extraction_method': 'direct_file_upload'
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"File upload extraction failed: {error_msg}")
            
            # Special handling for DOCX files with upload issues
            if file_path.suffix.lower() == '.docx' and ('invalid argument' in error_msg.lower() or '400' in error_msg):
                logger.info("DOCX direct upload failed - this file format may have compatibility issues with Gemini file API")
                return {
                    'error': f'DOCX direct upload not supported by Gemini API. Please use text extraction method instead. Original error: {error_msg}',
                    'extraction_status': 'failed',
                    'provider': self.provider.value,
                    'extraction_method': 'direct_file_upload',
                    'suggested_method': 'text_extraction',
                    'note': 'DOCX files may work better with text extraction method'
                }
            
            return {
                'error': f'File upload extraction failed: {error_msg}',
                'extraction_status': 'failed',
                'provider': self.provider.value,
                'extraction_method': 'direct_file_upload'
            }
    
    def _create_extraction_prompt(self, document_text: str) -> str:
        """Create a detailed prompt for lease data extraction with citation support."""
        
        schema = {
            "agreement_details": {
                "serviced_office": "boolean",
                "document_number": "string",
                "project_name": "string",
                "location": {
                    "survey_block_plot_no": "string",
                    "district_sector": "string",
                    "village": "string",
                    "zone_khewat": "string",
                    "mandal_municipality_khata": "string",
                    "registrar_office_id": "string"
                },
                "agreement_type": "string",
                "agreement_date": "date (YYYY-MM-DD)"
            },
            "parties": {
                "landlord": {
                    "name": "string",
                    "representative_name": "string",
                    "representative_role": "string"
                },
                "tenant": {
                    "name": "string",
                    "representative_name": "string",
                    "representative_role": "string"
                }
            },
            "unit_details": {
                "unit_number": "string",
                "floor_number": "string",
                "wing": "string",
                "other_info": "string",
                "abstract_area_type": "string",
                "abstract_area": "number",
                "abstract_rate": "number",
                "chargeable_area_type": "string",
                "super_built_up_area": "number",
                "built_up_area": "number",
                "carpet_area": "number"
            },
            "lease_terms": {
                "lease_start_date": "date (YYYY-MM-DD)",
                "lease_expiry_date": "date (YYYY-MM-DD)",
                "license_duration_months": "number",
                "monthly_rent": "number",
                "rate_per_sqft": "number",
                "consideration_value": "number",
                "escalation": {
                    "period_months": "number",
                    "percentage": "number"
                },
                "unit_condition": "string",
                "fit_outs": "boolean",
                "furnished_rate": "number",
                "rent_free_period_months": "number",
                "lock_in_period_landlord_months": "number",
                "lock_in_period_tenant_months": "number"
            },
            "financials": {
                "security_deposit": "number",
                "monthly_rental_equivalent_of_deposit": "number",
                "market_value": "number",
                "stamp_duty_amount": "number",
                "registration_amount": "number"
            },
            "parking_cam": {
                "car_parking_slots": "number",
                "car_parking_type": "string",
                "additional_car_parking_charges": "number",
                "two_wheeler_parking_slots": "number",
                "additional_two_wheeler_parking_charges": "number",
                "monthly_cam_charges": "number",
                "cam_paid_by": "string"
            },
            "property_tax": {
                "total_property_tax": "number",
                "property_tax": "number",
                "paid_by": "string"
            },
            "miscellaneous": {
                "comments": "string",
                "approver_comments": "string",
                "floor_plan": "string",
                "abstract": "string",
                "agreement_file": "string",
                "other_documents": "array of strings"
            },
            "citations": {
                "Note": "For each field extracted, provide the page number or section where the information was found. Use format like 'Page 1', 'Page 2-3', 'Section 1', 'Table 1', etc.",
                "agreement_details": "object with page/section references for each field",
                "parties": "object with page/section references for each field", 
                "unit_details": "object with page/section references for each field",
                "lease_terms": "object with page/section references for each field",
                "financials": "object with page/section references for each field",
                "parking_cam": "object with page/section references for each field",
                "property_tax": "object with page/section references for each field",
                "miscellaneous": "object with page/section references for each field"
            }
        }
        
        # Check if the document has page markers
        has_page_markers = "=== PAGE" in document_text or "--- Page" in document_text
        has_section_markers = "=== SECTION" in document_text or "=== TABLES SECTION" in document_text
        
        citation_instruction = ""
        if has_page_markers:
            citation_instruction = """
**CITATION REQUIREMENTS:**
- The document contains page markers (=== PAGE X ===). 
- For each extracted field, note which page the information came from
- Use format: "Page 1", "Page 2", "Page 1-2" for multi-page info
- If information spans multiple pages, list all relevant pages
"""
        elif has_section_markers:
            citation_instruction = """
**CITATION REQUIREMENTS:**
- The document contains section markers (=== SECTION X ===).
- For each extracted field, note which section the information came from
- Use format: "Section 1", "Section 2", "Table 1", etc.
- For tables, use "Tables Section"
"""
        else:
            citation_instruction = """
**CITATION REQUIREMENTS:**
- Provide line-based or paragraph-based references where possible
- Use format: "Lines 1-5", "Paragraph 2", "Beginning", "Middle", "End"
- Be as specific as possible about location in document
"""

        prompt = f"""
You are an expert document analyzer specializing in lease agreements. Your task is to extract structured data from the provided lease agreement text WITH SOURCE CITATIONS.

**CRITICAL INSTRUCTIONS:**
1. Carefully analyze the lease agreement document text provided below
2. Extract all relevant information according to the specified JSON schema
3. For missing information, use null values
4. For dates, use YYYY-MM-DD format
5. For numbers, extract only numeric values (remove currency symbols, commas, etc.)
6. Be as accurate as possible and avoid making assumptions
7. Return ONLY valid JSON without any additional text, explanations, or markdown formatting
8. Do NOT wrap your response in ```json blocks or any other formatting
9. Ensure all strings are properly quoted and all commas are correctly placed
10. Double-check that your JSON is syntactically correct before responding
11. **IMPORTANT: Include citations for each extracted field showing where in the document you found the information**

{citation_instruction}

**JSON SCHEMA:**
{json.dumps(schema, indent=2)}

**EXAMPLE CITATION FORMAT:**
```json
{{
  "agreement_details": {{
    "project_name": "Example Tower",
    "agreement_date": "2024-01-15"
  }},
  "citations": {{
    "agreement_details": {{
      "project_name": "Page 1",
      "agreement_date": "Page 1"
    }}
  }}
}}
```

**LEASE AGREEMENT DOCUMENT TEXT:**
{document_text}

**RESPONSE FORMAT:** Return only a valid JSON object that matches the schema above, including the citations section.
"""
        
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API for text extraction."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional document analyzer that extracts structured data from lease agreements. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=8000,
                temperature=0.1,  # Low temperature for consistent output
                response_format={"type": "json_object"}
            )
            
            # Log token usage
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                logger.info(f"=== OPENAI TOKEN USAGE ===")
                logger.info(f"Prompt tokens: {usage.prompt_tokens}")
                logger.info(f"Completion tokens: {usage.completion_tokens}")
                logger.info(f"Total tokens: {usage.total_tokens}")
                
                # Store token usage for later retrieval
                self._last_token_usage = {
                    'provider': 'openai',
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens,
                    'model': self.model
                }
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API for text extraction."""
        try:
            # Configure generation parameters with safety settings
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=8000,
            )
            
            # Configure safety settings to be more permissive for business documents
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Log token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                logger.info(f"=== GEMINI TOKEN USAGE ===")
                logger.info(f"Prompt tokens: {getattr(usage, 'prompt_token_count', 'N/A')}")
                logger.info(f"Completion tokens: {getattr(usage, 'candidates_token_count', 'N/A')}")
                logger.info(f"Total tokens: {getattr(usage, 'total_token_count', 'N/A')}")
                
                # Store token usage for later retrieval
                self._last_token_usage = {
                    'provider': 'gemini',
                    'prompt_tokens': getattr(usage, 'prompt_token_count', 0),
                    'completion_tokens': getattr(usage, 'candidates_token_count', 0),
                    'total_tokens': getattr(usage, 'total_token_count', 0),
                    'model': self.model_name
                }
            else:
                # For older versions or when usage data is not available
                logger.info("=== GEMINI TOKEN USAGE ===")
                logger.info("Token usage data not available (may require newer Gemini API version)")
                
                # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                estimated_prompt_tokens = len(prompt) // 4
                response_text = ""
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                        response_text = candidate.content.parts[0].text
                estimated_completion_tokens = len(response_text) // 4
                
                logger.info(f"Estimated prompt tokens: {estimated_prompt_tokens}")
                logger.info(f"Estimated completion tokens: {estimated_completion_tokens}")
                logger.info(f"Estimated total tokens: {estimated_prompt_tokens + estimated_completion_tokens}")
                
                self._last_token_usage = {
                    'provider': 'gemini',
                    'prompt_tokens': estimated_prompt_tokens,
                    'completion_tokens': estimated_completion_tokens,
                    'total_tokens': estimated_prompt_tokens + estimated_completion_tokens,
                    'model': self.model_name,
                    'estimated': True
                }
            
            # Check if response was blocked
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                # Check if content was blocked due to safety filters
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    if candidate.finish_reason.name in ['SAFETY', 'BLOCKED']:
                        # Log safety ratings for debugging
                        safety_info = []
                        if hasattr(candidate, 'safety_ratings'):
                            for rating in candidate.safety_ratings:
                                safety_info.append(f"{rating.category.name}: {rating.probability.name}")
                        
                        raise Exception(f"Content blocked by safety filters. Safety ratings: {'; '.join(safety_info)}")
                
                # Check if response has valid content
                if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text
                elif response.text:
                    return response.text
                else:
                    raise Exception("No valid content in response")
            else:
                raise Exception("No candidates in response")
            
        except Exception as e:
            # Try to provide more helpful error messages
            error_msg = str(e)
            if "safety" in error_msg.lower() or "blocked" in error_msg.lower():
                error_msg += ". Try using OpenAI provider for this document, or check if the document contains sensitive content."
            elif "quota" in error_msg.lower():
                error_msg += ". API quota exceeded. Please check your Gemini API usage limits."
            elif "authentication" in error_msg.lower():
                error_msg += ". Please verify your GEMINI_API_KEY in the .env file."
            
            raise Exception(f"Gemini API error: {error_msg}")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate the LLM response with robust JSON handling."""
        logger.info("=== PARSING RESPONSE ===")
        logger.info(f"Original response length: {len(response)} characters")
        
        try:
            # Clean the response - sometimes models include extra text
            original_response = response
            response = response.strip()
            
            # Find JSON content if wrapped in other text
            if not response.startswith('{'):
                logger.info("Response doesn't start with '{', searching for JSON content...")
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    logger.info(f"Found JSON content from position {start_idx} to {end_idx}")
                    response = response[start_idx:end_idx + 1]
                else:
                    logger.warning("No JSON braces found in response")
            
            logger.info(f"Cleaned response length: {len(response)} characters")
            
            # Check if full content logging is enabled
            log_full_content = os.getenv('LOG_FULL_CONTENT', 'false').lower() == 'true'
            
            if log_full_content:
                logger.info("=== CLEANED RESPONSE FOR PARSING ===")
                logger.info(response)
                logger.info("=== END CLEANED RESPONSE ===")
            else:
                logger.info(f"Cleaned response preview: {response[:200]}{'...' if len(response) > 200 else ''}")
            
            # Try multiple JSON parsing strategies
            data = self._robust_json_parse(response)
            
            logger.info("Successfully parsed JSON response")
            
            # Add metadata
            data['extraction_status'] = 'success'
            data['provider'] = self.provider.value
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"JSON error at line {getattr(e, 'lineno', 'unknown')}, column {getattr(e, 'colno', 'unknown')}")
            
            # Try to fix common JSON issues and retry
            logger.info("Attempting to fix common JSON issues...")
            fixed_response = self._fix_common_json_issues(response)
            
            if fixed_response != response:
                logger.info("Applied JSON fixes, attempting to parse again...")
                logger.info(f"Fixed response preview: {fixed_response[:200]}{'...' if len(fixed_response) > 200 else ''}")
                
                try:
                    data = json.loads(fixed_response)
                    data['extraction_status'] = 'success'
                    data['provider'] = self.provider.value
                    data['json_fixed'] = True
                    logger.info("Successfully parsed JSON after fixing common issues")
                    return data
                except json.JSONDecodeError as fix_error:
                    logger.error(f"JSON still invalid after fixes: {str(fix_error)}")
            else:
                logger.info("No fixes were applied to the JSON")
            
            logger.error("All JSON parsing strategies failed")
            return {
                'error': f'JSON parsing error: {str(e)}',
                'raw_response': response[:1000] + '...' if len(response) > 1000 else response,
                'extraction_status': 'parsing_failed',
                'provider': self.provider.value,
                'parsing_attempts': 'multiple_strategies_failed'
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {str(e)}")
            return {
                'error': f'Response parsing error: {str(e)}',
                'extraction_status': 'failed',
                'provider': self.provider.value
            }
    
    def _robust_json_parse(self, response: str) -> Dict[str, Any]:
        """Try multiple strategies to parse potentially malformed JSON."""
        logger.info("=== ROBUST JSON PARSING ===")
        
        # Strategy 1: Direct parsing
        logger.info("Strategy 1: Direct JSON parsing...")
        try:
            result = json.loads(response)
            logger.info("Strategy 1: SUCCESS - Direct parsing worked")
            return result
        except json.JSONDecodeError as e:
            logger.info(f"Strategy 1: FAILED - {str(e)}")
        
        # Strategy 2: Remove common prefixes/suffixes
        logger.info("Strategy 2: Removing common prefixes/suffixes...")
        cleaned = response
        prefixes_to_remove = ['```json', '```', 'JSON:', 'json:', 'Response:', 'Output:']
        suffixes_to_remove = ['```', '```json']
        
        original_cleaned = cleaned
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                logger.info(f"Removed prefix: {prefix}")
        
        for suffix in suffixes_to_remove:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
                logger.info(f"Removed suffix: {suffix}")
        
        if cleaned != original_cleaned:
            logger.info(f"Cleaned response preview: {cleaned[:200]}{'...' if len(cleaned) > 200 else ''}")
        
        try:
            result = json.loads(cleaned)
            logger.info("Strategy 2: SUCCESS - Parsing worked after cleaning")
            return result
        except json.JSONDecodeError as e:
            logger.info(f"Strategy 2: FAILED - {str(e)}")
        
        # Strategy 3: Find the largest valid JSON object
        logger.info("Strategy 3: Searching for valid JSON substring...")
        for i in range(len(response)):
            if response[i] == '{':
                for j in range(len(response) - 1, i, -1):
                    if response[j] == '}':
                        try:
                            candidate = response[i:j+1]
                            result = json.loads(candidate)
                            logger.info(f"Strategy 3: SUCCESS - Found valid JSON substring from {i} to {j}")
                            return result
                        except json.JSONDecodeError:
                            continue
        
        logger.info("Strategy 3: FAILED - No valid JSON substring found")
        
        # Strategy 4: Try fixing common issues
        logger.info("Strategy 4: Applying common JSON fixes...")
        fixed = self._fix_common_json_issues(response)
        
        try:
            result = json.loads(fixed)
            logger.info("Strategy 4: SUCCESS - Parsing worked after applying fixes")
            return result
        except json.JSONDecodeError as e:
            logger.info(f"Strategy 4: FAILED - {str(e)}")
            logger.error("All JSON parsing strategies exhausted")
            raise  # This will raise JSONDecodeError if still invalid
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues from LLM responses."""
        import re
        
        # Remove markdown code blocks
        json_str = re.sub(r'```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'(\w+)(\s*:\s*)', r'"\1"\2', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        # Fix common value issues
        json_str = re.sub(r':\s*True\b', ': true', json_str)
        json_str = re.sub(r':\s*False\b', ': false', json_str)
        json_str = re.sub(r':\s*None\b', ': null', json_str)
        
        # Fix unquoted string values (basic cases)
        json_str = re.sub(r':\s*([A-Za-z][A-Za-z0-9\s]*[A-Za-z0-9])\s*([,}])', r': "\1"\2', json_str)
        
        # Remove any non-printable characters
        json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in ['\n', '\r', '\t'])
        
        return json_str.strip()
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the LLM service connection."""
        try:
            test_prompt = "Return a JSON object with a single key 'status' and value 'connected'."
            
            if self.provider == LLMProvider.OPENAI:
                response = self._call_openai(test_prompt)
            elif self.provider == LLMProvider.GEMINI:
                response = self._call_gemini(test_prompt)
            
            result = self._parse_response(response)
            
            return {
                'connected': True,
                'provider': self.provider.value,
                'model': getattr(self, 'model', getattr(self, 'model_name', 'unknown')),
                'test_response': result
            }
            
        except Exception as e:
            return {
                'connected': False,
                'provider': self.provider.value,
                'error': str(e)
            }
    
    def _create_file_extraction_prompt(self) -> str:
        """Create a prompt for file-based lease data extraction with citations."""
        
        schema = {
            "agreement_details": {
                "serviced_office": "boolean",
                "document_number": "string",
                "project_name": "string",
                "location": {
                    "survey_block_plot_no": "string",
                    "district_sector": "string",
                    "village": "string",
                    "zone_khewat": "string",
                    "mandal_municipality_khata": "string",
                    "registrar_office_id": "string"
                },
                "agreement_type": "string",
                "agreement_date": "date (YYYY-MM-DD)"
            },
            "parties": {
                "landlord": {
                    "name": "string",
                    "representative_name": "string",
                    "representative_role": "string"
                },
                "tenant": {
                    "name": "string",
                    "representative_name": "string",
                    "representative_role": "string"
                }
            },
            "unit_details": {
                "unit_number": "string",
                "floor_number": "string",
                "wing": "string",
                "other_info": "string",
                "abstract_area_type": "string",
                "abstract_area": "number",
                "abstract_rate": "number",
                "chargeable_area_type": "string",
                "super_built_up_area": "number",
                "built_up_area": "number",
                "carpet_area": "number"
            },
            "lease_terms": {
                "lease_start_date": "date (YYYY-MM-DD)",
                "lease_expiry_date": "date (YYYY-MM-DD)",
                "license_duration_months": "number",
                "monthly_rent": "number",
                "rate_per_sqft": "number",
                "consideration_value": "number",
                "escalation": {
                    "period_months": "number",
                    "percentage": "number"
                },
                "unit_condition": "string",
                "fit_outs": "boolean",
                "furnished_rate": "number",
                "rent_free_period_months": "number",
                "lock_in_period_landlord_months": "number",
                "lock_in_period_tenant_months": "number"
            },
            "financials": {
                "security_deposit": "number",
                "monthly_rental_equivalent_of_deposit": "number",
                "market_value": "number",
                "stamp_duty_amount": "number",
                "registration_amount": "number"
            },
            "parking_cam": {
                "car_parking_slots": "number",
                "car_parking_type": "string",
                "additional_car_parking_charges": "number",
                "two_wheeler_parking_slots": "number",
                "additional_two_wheeler_parking_charges": "number",
                "monthly_cam_charges": "number",
                "cam_paid_by": "string"
            },
            "property_tax": {
                "total_property_tax": "number",
                "property_tax": "number",
                "paid_by": "string"
            },
            "miscellaneous": {
                "comments": "string",
                "approver_comments": "string",
                "floor_plan": "string",
                "abstract": "string",
                "agreement_file": "string",
                "other_documents": "array of strings"
            },
            "citations": {
                "Note": "For each field extracted, provide the page number where the information was found. Use format like 'Page 1', 'Page 2-3', etc.",
                "agreement_details": "object with page references for each field",
                "parties": "object with page references for each field", 
                "unit_details": "object with page references for each field",
                "lease_terms": "object with page references for each field",
                "financials": "object with page references for each field",
                "parking_cam": "object with page references for each field",
                "property_tax": "object with page references for each field",
                "miscellaneous": "object with page references for each field"
            }
        }
        
        prompt = f"""
You are an expert document analyzer specializing in lease agreements. Your task is to extract structured data from the provided lease agreement document WITH SOURCE CITATIONS.

**CRITICAL INSTRUCTIONS:**
1. Carefully analyze the lease agreement document provided as a file
2. Extract all relevant information according to the specified JSON schema
3. For missing information, use null values
4. For dates, use YYYY-MM-DD format
5. For numbers, extract only numeric values (remove currency symbols, commas, etc.)
6. Be as accurate as possible and avoid making assumptions
7. Return ONLY valid JSON without any additional text, explanations, or markdown formatting
8. Do NOT wrap your response in ```json blocks or any other formatting
9. Ensure all strings are properly quoted and all commas are correctly placed
10. Double-check that your JSON is syntactically correct before responding
11. **IMPORTANT: Include citations for each extracted field showing which page you found the information**

**CITATION REQUIREMENTS:**
- For each extracted field, note which page the information came from
- Use format: "Page 1", "Page 2", "Page 1-2" for multi-page info
- If information spans multiple pages, list all relevant pages
- If page numbers are not visible, use "Page 1", "Page 2" etc. based on document structure

**JSON SCHEMA:**
{json.dumps(schema, indent=2)}

**EXAMPLE CITATION FORMAT:**
```json
{{
  "agreement_details": {{
    "project_name": "Example Tower",
    "agreement_date": "2024-01-15"
  }},
  "citations": {{
    "agreement_details": {{
      "project_name": "Page 1",
      "agreement_date": "Page 1"
    }}
  }}
}}
```

**RESPONSE FORMAT:** Return only a valid JSON object that matches the schema above, including the citations section.
"""
        
        return prompt
    
    def _call_gemini_with_file(self, prompt: str, uploaded_file) -> str:
        """Call Google Gemini API with uploaded file for text extraction."""
        try:
            # Configure generation parameters with safety settings
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=8000,
            )
            
            # Configure safety settings to be more permissive for business documents
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
            
            response = self.client.generate_content(
                [prompt, uploaded_file],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Log token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                logger.info(f"=== GEMINI FILE TOKEN USAGE ===")
                logger.info(f"Prompt tokens: {getattr(usage, 'prompt_token_count', 'N/A')}")
                logger.info(f"Completion tokens: {getattr(usage, 'candidates_token_count', 'N/A')}")
                logger.info(f"Total tokens: {getattr(usage, 'total_token_count', 'N/A')}")
                
                # Store token usage for later retrieval
                self._last_token_usage = {
                    'provider': 'gemini',
                    'prompt_tokens': getattr(usage, 'prompt_token_count', 0),
                    'completion_tokens': getattr(usage, 'candidates_token_count', 0),
                    'total_tokens': getattr(usage, 'total_token_count', 0),
                    'model': self.model_name,
                    'extraction_method': 'direct_file_upload'
                }
            else:
                logger.info("=== GEMINI FILE TOKEN USAGE ===")
                logger.info("Token usage data not available for file upload")
                
                # For file uploads, we can't easily estimate tokens
                self._last_token_usage = {
                    'provider': 'gemini',
                    'prompt_tokens': 'N/A',
                    'completion_tokens': 'N/A',
                    'total_tokens': 'N/A',
                    'model': self.model_name,
                    'extraction_method': 'direct_file_upload',
                    'note': 'Token usage not available for file uploads'
                }
            
            # Check if response was blocked by safety filters
            if response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                block_reason = response.prompt_feedback.block_reason
                if block_reason:
                    raise Exception(f"Gemini blocked the request due to safety concerns: {block_reason}")
            
            # Check if response has valid content
            if not response.candidates or len(response.candidates) == 0:
                raise Exception("Gemini returned no response candidates")
            
            candidate = response.candidates[0]
            
            # Check for safety issues in the response
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                if 'SAFETY' in str(candidate.finish_reason):
                    raise Exception(f"Gemini response blocked for safety: {candidate.finish_reason}")
            
            if not candidate.content or not candidate.content.parts:
                raise Exception("Gemini returned empty content")
            
            return candidate.content.parts[0].text
            
        except Exception as e:
            raise Exception(f"Gemini file API error: {str(e)}")
