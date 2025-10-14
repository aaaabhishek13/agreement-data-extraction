"""
Main extraction service that combines document processing and LLM extraction.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from models.lease_models import LeaseAgreementData, ExtractionResult
from .document_processor import DocumentProcessor
from .llm_service import LLMService, LLMProvider

logger = logging.getLogger(__name__)


class LeaseExtractionService:
    """Main service for extracting lease agreement data."""
    
    def __init__(self, llm_provider: str = "openai", enable_fallback: bool = True):
        """
        Initialize the extraction service.
        
        Args:
            llm_provider: LLM provider to use ('openai' or 'gemini')
            enable_fallback: Whether to enable automatic fallback to other provider
        """
        self.document_processor = DocumentProcessor()
        self.primary_provider = llm_provider
        self.enable_fallback = enable_fallback
        
        try:
            self.llm_service = LLMService(provider=llm_provider)
            logger.info(f"Initialized extraction service with {llm_provider} provider")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise
    
    def extract_from_file(self, file_path: str, max_file_size_mb: int = 16, direct_upload: bool = False) -> ExtractionResult:
        """
        Extract lease data from a document file.
        
        Args:
            file_path: Path to the document file
            max_file_size_mb: Maximum file size limit in MB
            direct_upload: If True, send file directly to LLM instead of extracting text locally
            
        Returns:
            ExtractionResult containing the extracted data or error information
        """
        start_time = time.time()
        
        try:
            # Validate file
            validation_result = self.document_processor.validate_file(file_path, max_file_size_mb)
            
            if not validation_result['valid']:
                return ExtractionResult(
                    success=False,
                    error=validation_result['error'],
                    processing_time=time.time() - start_time,
                    file_info={'validation_failed': True}
                )

            # Initialize variables
            llm_result = None
            file_info = {}

            if direct_upload:
                # Send file directly to LLM
                logger.info(f"Sending file directly to LLM: {Path(file_path).name}")
                llm_result = self._extract_with_direct_upload(file_path)
                
                # Create basic file info for direct upload
                file_info = {
                    'file_name': Path(file_path).name,
                    'file_size': Path(file_path).stat().st_size,
                    'processing_method': 'direct_upload',
                    'processing_status': 'success'
                }
            else:
                # Extract text from document locally first
                logger.info(f"Processing document: {Path(file_path).name}")
                doc_result = self.document_processor.extract_text(file_path)
                
                if doc_result['processing_status'] == 'failed':
                    return ExtractionResult(
                        success=False,
                        error=f"Document processing failed: {doc_result.get('error', 'Unknown error')}",
                        processing_time=time.time() - start_time,
                        file_info=doc_result
                    )
                
                # Check if we have text to process
                if not doc_result['text'] or len(doc_result['text'].strip()) < 100:
                    return ExtractionResult(
                        success=False,
                        error="Document contains insufficient text for processing",
                        processing_time=time.time() - start_time,
                        file_info=doc_result
                    )
                
                # Extract structured data using LLM with fallback
                logger.info(f"Extracting structured data using {self.llm_service.provider.value} LLM")
                llm_result = self._extract_with_fallback(doc_result['text'])
                file_info = doc_result
            
            if llm_result.get('extraction_status') == 'failed':
                return ExtractionResult(
                    success=False,
                    error=f"LLM extraction failed: {llm_result.get('error', 'Unknown error')}",
                    processing_time=time.time() - start_time,
                    file_info=file_info
                )
            
            # Validate and create structured data
            try:
                # Add extracted text to the result for document viewing
                if not direct_upload and 'text' in file_info:
                    llm_result['extracted_text'] = file_info['text']
                
                lease_data = LeaseAgreementData(**llm_result)
                
                return ExtractionResult(
                    success=True,
                    data=lease_data,
                    processing_time=time.time() - start_time,
                    file_info=file_info
                )
                
            except Exception as validation_error:
                logger.error(f"Data validation error: {str(validation_error)}")
                
                # Return raw data if validation fails
                return ExtractionResult(
                    success=True,  # Partial success
                    data=LeaseAgreementData(
                        extraction_status='validation_warning',
                        error=f'Data validation warning: {str(validation_error)}',
                        provider=llm_result.get('provider')
                    ),
                    processing_time=time.time() - start_time,
                    file_info=file_info,
                    error=f"Data validation warning: {str(validation_error)}"
                )
        
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {str(e)}")
            return ExtractionResult(
                success=False,
                error=f"Extraction failed: {str(e)}",
                processing_time=time.time() - start_time,
                file_info=file_info if 'file_info' in locals() else {}
            )
    
    def _extract_with_fallback(self, text: str) -> Dict[str, Any]:
        """
        Extract data with automatic fallback to alternative provider if primary fails.
        
        Args:
            text: Document text to extract data from
            
        Returns:
            Dict containing extracted data or error information
        """
        primary_result = self.llm_service.extract_lease_data(text)
        
        # If primary extraction succeeded, return it
        if primary_result.get('extraction_status') != 'failed':
            return primary_result
        
        # If fallback is disabled or we don't have both API keys, return the error
        if not self.enable_fallback:
            return primary_result
        
        # Determine fallback provider
        fallback_provider = 'gemini' if self.primary_provider == 'openai' else 'openai'
        
        # Check if fallback provider is configured
        import os
        fallback_key = 'GEMINI_API_KEY' if fallback_provider == 'gemini' else 'OPENAI_API_KEY'
        if not os.getenv(fallback_key):
            logger.warning(f"Fallback to {fallback_provider} not possible: {fallback_key} not configured")
            return primary_result
        
        try:
            logger.warning(f"Primary provider ({self.primary_provider}) failed, attempting fallback to {fallback_provider}")
            
            # Create fallback LLM service
            fallback_service = LLMService(provider=fallback_provider)
            fallback_result = fallback_service.extract_lease_data(text)
            
            if fallback_result.get('extraction_status') != 'failed':
                logger.info(f"Fallback to {fallback_provider} succeeded")
                # Add a note about the fallback
                fallback_result['fallback_used'] = True
                fallback_result['primary_provider'] = self.primary_provider
                fallback_result['fallback_provider'] = fallback_provider
                return fallback_result
            else:
                logger.error(f"Both providers failed. Primary: {primary_result.get('error')}, Fallback: {fallback_result.get('error')}")
                # Return combined error
                return {
                    'extraction_status': 'failed',
                    'error': f"Both providers failed. {self.primary_provider}: {primary_result.get('error', 'Unknown error')}; {fallback_provider}: {fallback_result.get('error', 'Unknown error')}",
                    'provider': f"{self.primary_provider} (failed), {fallback_provider} (failed)"
                }
                
        except Exception as fallback_error:
            logger.error(f"Fallback provider initialization failed: {str(fallback_error)}")
            return {
                'extraction_status': 'failed',
                'error': f"Primary provider failed: {primary_result.get('error', 'Unknown error')}. Fallback failed: {str(fallback_error)}",
                'provider': self.primary_provider
            }

    def _extract_with_direct_upload(self, file_path: str) -> Dict[str, Any]:
        """
        Extract data by sending file directly to LLM (for providers that support file uploads).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing extracted data or error information
        """
        try:
            # Check if the current provider supports direct file upload
            if hasattr(self.llm_service, 'extract_from_file'):
                primary_result = self.llm_service.extract_from_file(file_path)
                
                # Special handling for DOCX files that fail with Gemini direct upload
                if (primary_result.get('extraction_status') == 'failed' and 
                    primary_result.get('suggested_method') == 'text_extraction' and
                    Path(file_path).suffix.lower() == '.docx'):
                    
                    logger.info("DOCX direct upload failed, automatically falling back to text extraction")
                    doc_result = self.document_processor.extract_text(file_path)
                    
                    if doc_result['processing_status'] == 'failed':
                        return {
                            'extraction_status': 'failed',
                            'error': f"Both direct upload and text extraction failed for DOCX: {doc_result.get('error', 'Unknown error')}",
                            'provider': self.primary_provider
                        }
                    
                    if not doc_result['text'] or len(doc_result['text'].strip()) < 100:
                        return {
                            'extraction_status': 'failed',
                            'error': "DOCX document contains insufficient text for processing",
                            'provider': self.primary_provider
                        }
                    
                    # Extract using text method
                    primary_result = self.llm_service.extract_lease_data(doc_result['text'])
                    primary_result['note'] = 'Automatically switched from direct upload to text extraction for DOCX file'
            else:
                # Fallback to text extraction for providers that don't support file upload
                logger.warning(f"Provider {self.primary_provider} doesn't support direct file upload, falling back to text extraction")
                doc_result = self.document_processor.extract_text(file_path)
                
                if doc_result['processing_status'] == 'failed':
                    return {
                        'extraction_status': 'failed',
                        'error': f"Document processing failed: {doc_result.get('error', 'Unknown error')}",
                        'provider': self.primary_provider
                    }
                
                if not doc_result['text'] or len(doc_result['text'].strip()) < 100:
                    return {
                        'extraction_status': 'failed',
                        'error': "Document contains insufficient text for processing",
                        'provider': self.primary_provider
                    }
                
                primary_result = self.llm_service.extract_lease_data(doc_result['text'])
            
            # If primary extraction succeeded, return it
            if primary_result.get('extraction_status') != 'failed':
                return primary_result
            
            # If fallback is disabled, return the error
            if not self.enable_fallback:
                return primary_result
            
            # Attempt fallback with direct upload if possible
            fallback_provider = 'gemini' if self.primary_provider == 'openai' else 'openai'
            
            # Check if fallback provider is configured
            import os
            fallback_key = 'GEMINI_API_KEY' if fallback_provider == 'gemini' else 'OPENAI_API_KEY'
            if not os.getenv(fallback_key):
                logger.warning(f"Fallback to {fallback_provider} not possible: {fallback_key} not configured")
                return primary_result
            
            try:
                logger.warning(f"Direct upload with {self.primary_provider} failed, attempting fallback to {fallback_provider}")
                
                # Create fallback LLM service
                fallback_service = LLMService(provider=fallback_provider)
                
                if hasattr(fallback_service, 'extract_from_file'):
                    fallback_result = fallback_service.extract_from_file(file_path)
                else:
                    # Fallback to text extraction
                    doc_result = self.document_processor.extract_text(file_path)
                    if doc_result['processing_status'] == 'failed':
                        return primary_result
                    fallback_result = fallback_service.extract_lease_data(doc_result['text'])
                
                if fallback_result.get('extraction_status') != 'failed':
                    logger.info(f"Fallback to {fallback_provider} succeeded")
                    fallback_result['fallback_used'] = True
                    fallback_result['primary_provider'] = self.primary_provider
                    fallback_result['fallback_provider'] = fallback_provider
                    return fallback_result
                else:
                    return {
                        'extraction_status': 'failed',
                        'error': f"Both providers failed. {self.primary_provider}: {primary_result.get('error', 'Unknown error')}; {fallback_provider}: {fallback_result.get('error', 'Unknown error')}",
                        'provider': f"{self.primary_provider} (failed), {fallback_provider} (failed)"
                    }
                    
            except Exception as fallback_error:
                logger.error(f"Fallback provider initialization failed: {str(fallback_error)}")
                return {
                    'extraction_status': 'failed',
                    'error': f"Primary provider failed: {primary_result.get('error', 'Unknown error')}. Fallback failed: {str(fallback_error)}",
                    'provider': self.primary_provider
                }
                
        except Exception as e:
            logger.error(f"Direct upload extraction failed: {str(e)}")
            return {
                'extraction_status': 'failed',
                'error': f"Direct upload extraction failed: {str(e)}",
                'provider': self.primary_provider
            }

    def extract_from_text(self, text: str, file_info: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        """
        Extract lease data from raw text.
        
        Args:
            text: Raw text content
            file_info: Optional file information
            
        Returns:
            ExtractionResult containing the extracted data
        """
        start_time = time.time()
        
        try:
            if not text or len(text.strip()) < 100:
                return ExtractionResult(
                    success=False,
                    error="Insufficient text content for processing",
                    processing_time=time.time() - start_time
                )
            
            # Extract structured data using LLM with fallback
            logger.info(f"Extracting structured data from text using {self.llm_service.provider.value} LLM")
            llm_result = self._extract_with_fallback(text)
            
            if llm_result.get('extraction_status') == 'failed':
                return ExtractionResult(
                    success=False,
                    error=f"LLM extraction failed: {llm_result.get('error', 'Unknown error')}",
                    processing_time=time.time() - start_time,
                    file_info=file_info
                )
            
            # Create structured data
            lease_data = LeaseAgreementData(**llm_result)
            
            return ExtractionResult(
                success=True,
                data=lease_data,
                processing_time=time.time() - start_time,
                file_info=file_info or {'source': 'text_input', 'text_length': len(text)}
            )
            
        except Exception as e:
            logger.error(f"Error extracting from text: {str(e)}")
            return ExtractionResult(
                success=False,
                error=f"Text extraction failed: {str(e)}",
                processing_time=time.time() - start_time,
                file_info=file_info
            )
    
    def get_supported_formats(self) -> list:
        """Get list of supported file formats."""
        return list(self.document_processor.SUPPORTED_FORMATS)
    
    def test_services(self) -> Dict[str, Any]:
        """Test all service components."""
        results = {
            'document_processor': {'available': True},
            'llm_service': self.llm_service.test_connection()
        }
        
        # Test document processor with supported formats
        try:
            supported_formats = self.get_supported_formats()
            results['document_processor']['supported_formats'] = supported_formats
        except Exception as e:
            results['document_processor']['error'] = str(e)
            results['document_processor']['available'] = False
        
        # Overall status
        results['overall_status'] = (
            results['document_processor']['available'] and 
            results['llm_service']['connected']
        )
        
        return results


class BatchExtractionService:
    """Service for processing multiple lease documents."""
    
    def __init__(self, llm_provider: str = "openai"):
        """Initialize batch extraction service."""
        self.extraction_service = LeaseExtractionService(llm_provider)
    
    def extract_from_files(self, file_paths: list, max_file_size_mb: int = 16) -> Dict[str, ExtractionResult]:
        """
        Extract data from multiple files.
        
        Args:
            file_paths: List of file paths to process
            max_file_size_mb: Maximum file size limit
            
        Returns:
            Dictionary mapping file paths to extraction results
        """
        results = {}
        
        for file_path in file_paths:
            logger.info(f"Processing file {len(results) + 1}/{len(file_paths)}: {Path(file_path).name}")
            
            try:
                result = self.extraction_service.extract_from_file(file_path, max_file_size_mb)
                results[file_path] = result
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                results[file_path] = ExtractionResult(
                    success=False,
                    error=f"Processing failed: {str(e)}",
                    file_info={'file_path': file_path}
                )
        
        return results
    
    def get_batch_summary(self, results: Dict[str, ExtractionResult]) -> Dict[str, Any]:
        """Get summary statistics for batch processing."""
        total_files = len(results)
        successful = sum(1 for result in results.values() if result.success)
        failed = total_files - successful
        
        total_processing_time = sum(
            result.processing_time or 0 for result in results.values()
        )
        
        summary = {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'success_rate': round(successful / total_files * 100, 1) if total_files > 0 else 0,
            'total_processing_time': round(total_processing_time, 2),
            'average_processing_time': round(total_processing_time / total_files, 2) if total_files > 0 else 0,
            'failed_files': [
                {
                    'file_path': path,
                    'error': result.error
                }
                for path, result in results.items() if not result.success
            ]
        }
        
        return summary
