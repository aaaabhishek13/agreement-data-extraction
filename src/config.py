"""
Configuration management for the lease agreement extraction tool.
"""

import os
from pathlib import Path
from typing import Optional

class Config:
    """Application configuration class."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Flask configuration
        self.SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
        self.FLASK_ENV = os.getenv('FLASK_ENV', 'development')
        self.FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
        self.FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
        
        # File upload configuration
        self.UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', 'uploads')).absolute()
        self.MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))  # 16MB
        
        # LLM configuration
        self.DEFAULT_LLM_PROVIDER = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        self.GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-pro')
        
        # Ensure upload folder exists
        self.UPLOAD_FOLDER.mkdir(exist_ok=True)
    
    @property
    def debug(self) -> bool:
        """Check if running in debug mode."""
        return self.FLASK_ENV == 'development'
    
    def validate(self) -> list:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check API keys based on provider
        if self.DEFAULT_LLM_PROVIDER == 'openai':
            if not self.OPENAI_API_KEY:
                issues.append("OPENAI_API_KEY is required when using OpenAI provider")
        elif self.DEFAULT_LLM_PROVIDER == 'gemini':
            if not self.GEMINI_API_KEY:
                issues.append("GEMINI_API_KEY is required when using Gemini provider")
        else:
            issues.append(f"Invalid LLM provider: {self.DEFAULT_LLM_PROVIDER}")
        
        # Check file upload settings
        if self.MAX_CONTENT_LENGTH < 1024:
            issues.append("MAX_CONTENT_LENGTH is too small (minimum 1KB)")
        
        return issues
    
    def get_llm_config(self, provider: Optional[str] = None) -> dict:
        """Get LLM configuration for specified provider."""
        provider = provider or self.DEFAULT_LLM_PROVIDER
        
        if provider == 'openai':
            return {
                'provider': 'openai',
                'api_key': self.OPENAI_API_KEY,
                'model': self.OPENAI_MODEL
            }
        elif provider == 'gemini':
            return {
                'provider': 'gemini',
                'api_key': self.GEMINI_API_KEY,
                'model': self.GEMINI_MODEL
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            'flask_env': self.FLASK_ENV,
            'flask_host': self.FLASK_HOST,
            'flask_port': self.FLASK_PORT,
            'upload_folder': str(self.UPLOAD_FOLDER),
            'max_content_length': self.MAX_CONTENT_LENGTH,
            'default_llm_provider': self.DEFAULT_LLM_PROVIDER,
            'openai_model': self.OPENAI_MODEL,
            'gemini_model': self.GEMINI_MODEL,
            'has_openai_key': bool(self.OPENAI_API_KEY),
            'has_gemini_key': bool(self.GEMINI_API_KEY),
            'debug': self.debug
        }


# Global configuration instance
config = Config()
