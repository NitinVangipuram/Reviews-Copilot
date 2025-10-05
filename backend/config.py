"""
Configuration management for Reviews Copilot API
Handles environment variables, database settings, and feature flags
"""

import os
import sys
import logging
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    api_key: str = Field(default="demo-key-123", env="API_KEY")
    api_title: str = Field(default="Reviews Copilot API", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    api_description: str = Field(
        default="AI-powered review management system for multi-location businesses",
        env="API_DESCRIPTION"
    )
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    huggingface_api_token: str = Field(default="", description="Hugging Face API Token")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./reviews.db", env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080","https://reviewscopilot.vercel.app"],
        env="ALLOWED_ORIGINS"
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        env="ALLOWED_METHODS"
    )
    allowed_headers: List[str] = Field(
        default=["*"],
        env="ALLOWED_HEADERS"
    )
    
    # AI Configuration
    ai_enabled: bool = Field(default=True, env="AI_ENABLED")
    ai_model_cache_size: int = Field(default=100, env="AI_MODEL_CACHE_SIZE")
    ai_timeout: int = Field(default=30, env="AI_TIMEOUT")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=False, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Feature Flags
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    enable_search: bool = Field(default=True, env="ENABLE_SEARCH")
    enable_ai_replies: bool = Field(default=True, env="ENABLE_AI_REPLIES")
    enable_batch_processing: bool = Field(default=True, env="ENABLE_BATCH_PROCESSING")
    
    # Pagination
    default_page_size: int = Field(default=10, env="DEFAULT_PAGE_SIZE")
    max_page_size: int = Field(default=100, env="MAX_PAGE_SIZE")
    
    # Search Configuration
    search_max_results: int = Field(default=20, env="SEARCH_MAX_RESULTS")
    search_min_similarity: float = Field(default=0.1, env="SEARCH_MIN_SIMILARITY")
    
    @validator('allowed_origins', pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('allowed_methods', pre=True)
    def parse_allowed_methods(cls, v):
        if isinstance(v, str):
            return [method.strip() for method in v.split(',')]
        return v
    
    @validator('allowed_headers', pre=True)
    def parse_allowed_headers(cls, v):
        if isinstance(v, str):
            return [header.strip() for header in v.split(',')]
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            logger.warning(f"Invalid log level {v}, defaulting to INFO")
            return "INFO"
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings

def setup_logging():
    """Configure logging based on settings"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", mode="a") if not settings.debug else logging.NullHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    if settings.debug:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        logging.getLogger("transformers").setLevel(logging.WARNING)

# Initialize logging
setup_logging()
