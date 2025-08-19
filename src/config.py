from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    app_name: str = Field(default="DocuScan", description="Application name")
    log_level: str = Field(default="INFO", description="Logging level")
    upload_max_size_mb: int = Field(default=10, description="Maximum upload size in MB")
    
    # Future API keys can be added here
    # openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    # google_api_key: Optional[str] = Field(default=None, description="Google API key")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()