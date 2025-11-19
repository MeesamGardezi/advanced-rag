"""
Configuration management for the RAG backend.
Loads environment variables and provides centralized access to all settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.2
    openai_max_tokens: int = 1000
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "construction_jobs"
    
    # Firebase Configuration
    firebase_credentials_path: str
    firebase_database_url: str
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dimension: int = 384
    
    # RAG Configuration
    top_k_results: int = 8
    similarity_threshold: float = 0.7
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Convenience function to access settings
settings = get_settings()