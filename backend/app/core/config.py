"""
Configuration settings for VoiceFlow Studio Backend
"""

import os
import warnings
from typing import Optional
from contextlib import contextmanager
from pydantic_settings import BaseSettings


# Configure warnings suppression for deprecated dependencies
def configure_warnings():
    """Configure warning filters for dependencies"""
    # Suppress specific deprecation warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
    warnings.filterwarnings(
        "ignore", message=".*LoRACompatibleLinear.*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message=".*torch.backends.cuda.sdp_kernel.*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message=".*scaled_dot_product_attention.*", category=UserWarning
    )

    # Suppress diffusers v0.29+ deprecation warnings
    warnings.filterwarnings(
        "ignore", message=".*Transformer2DModelOutput.*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message=".*VQEncoderOutput.*", category=FutureWarning
    )
    warnings.filterwarnings("ignore", message=".*VQModel.*", category=FutureWarning)

    # Suppress transformers attention deprecation warnings
    warnings.filterwarnings(
        "ignore",
        message=".*was not found in transformers version.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore", message=".*You are using a model.*", category=UserWarning
    )

    # Suppress PyTorch 2.7+ compatibility warnings
    warnings.filterwarnings(
        "ignore", message=".*torch.nn.attention.*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*torch.nn.functional.scaled_dot_product_attention.*",
        category=UserWarning,
    )


# Apply warning configuration
configure_warnings()


@contextmanager
def suppress_model_warnings():
    """Context manager to suppress model loading and generation warnings"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


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

    # Auth compatibility properties
    @property
    def secret_key(self) -> str:
        return self.jwt_secret_key

    @property
    def algorithm(self) -> str:
        return self.jwt_algorithm

    @property
    def access_token_expire_minutes(self) -> int:
        return self.jwt_expiration_hours * 60

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
