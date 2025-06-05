"""
Configuration settings for VoiceFlow Studio Backend
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # OpenAI Configuration
    openai_api_key: Optional[str] = None

    # Stripe Configuration
    stripe_api_key: Optional[str] = None
    stripe_publishable_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None

    # JWT Configuration
    jwt_secret_key: str = "your-super-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Database Configuration
    database_url: str = "sqlite:///./voiceflow_studio.db"

    # Frontend URL for CORS
    frontend_url: str = "http://localhost:3000"

    # Audio/Voice Configuration
    audio_storage_path: str = "./storage/audio"
    max_audio_file_size: int = 50 * 1024 * 1024  # 50MB

    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()

# Legacy exports for backward compatibility
openai_api_key: Optional[str] = settings.openai_api_key
stripe_api_key: Optional[str] = settings.stripe_api_key
stripe_publishable_key: Optional[str] = settings.stripe_publishable_key
stripe_webhook_secret: Optional[str] = settings.stripe_webhook_secret
jwt_secret_key: str = settings.jwt_secret_key
jwt_algorithm: str = settings.jwt_algorithm
jwt_expiration_hours: int = settings.jwt_expiration_hours
database_url: str = settings.database_url
frontend_url: str = settings.frontend_url

# Storage Configuration
storage_type: str = os.getenv("STORAGE_TYPE", "local")  # local, s3, gcs
storage_bucket: Optional[str] = os.getenv("STORAGE_BUCKET")
storage_region: Optional[str] = os.getenv("STORAGE_REGION")

# Audio Processing Configuration
audio_sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "22050"))
audio_format: str = os.getenv("AUDIO_FORMAT", "wav")

# Chatterbox TTS Configuration (Note: No API key needed - runs locally)
# The service will automatically detect available hardware (CUDA/CPU)
chatterbox_device: Optional[str] = os.getenv(
    "CHATTERBOX_DEVICE"
)  # Optional: force "cuda" or "cpu"
chatterbox_model_cache: str = os.getenv("CHATTERBOX_MODEL_CACHE", "./storage/models")
