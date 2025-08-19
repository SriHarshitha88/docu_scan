from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    app_name: str = Field(default="DocuScan", description="Application name")
    log_level: str = Field(default="INFO", description="Logging level")
    upload_max_size_mb: int = Field(default=10, description="Maximum upload size in MB")
    
    # API Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_vision_model: str = Field(default="gpt-4o", description="OpenAI model for vision tasks")
    openai_text_model: str = Field(default="gpt-4o-mini", description="OpenAI model for text tasks")
    ocr_max_tokens: int = Field(default=2000, description="Maximum tokens for OCR response")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()