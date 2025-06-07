"""
Configuration settings for VoiceFlow Studio Backend
"""

import os
import warnings
from typing import Optional
from contextlib import contextmanager
from pydantic_settings import BaseSettings
from pydantic import Field


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

    # App settings
    app_name: str = "VoiceFlow Studio"
    debug: bool = False
    version: str = "1.0.0"

    # GPU Requirements - MANDATORY for production performance
    require_gpu: bool = True  # Force GPU acceleration, no CPU fallback
    min_gpu_memory_gb: int = 4  # Minimum 4GB VRAM required

    # API settings
    api_v1_str: str = "/api/v1"

    # Database
    database_url: str = Field(default="sqlite:///./app.db")

    # Security
    secret_key: str = Field(default="your-secret-key-here")
    access_token_expire_minutes: int = 30

    # JWT settings
    jwt_secret_key: str = Field(default="your-jwt-secret-key")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Frontend URL
    frontend_url: str = Field(default="http://localhost:3000")

    # Stripe (for credits/payments)
    stripe_publishable_key: str = Field(default="")
    stripe_secret_key: str = Field(default="")
    stripe_webhook_secret: str = Field(default="")

    # External APIs
    openai_api_key: str = Field(default="")

    # Audio processing settings
    max_audio_duration_seconds: int = 1800  # 30 minutes max
    max_file_size_mb: int = 100

    # TTS Performance settings
    tts_production_mode: bool = True  # Enable ultra-fast GPU mode
    tts_cache_enabled: bool = True  # Enable audio caching
    tts_max_inference_steps: int = 10  # Ultra-fast inference for production

    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Validate GPU requirements on startup if required
        if self.require_gpu:
            self._validate_gpu_requirements()

    def _validate_gpu_requirements(self):
        """Validate GPU requirements at application startup"""
        try:
            import torch

            if not torch.cuda.is_available():
                error_msg = (
                    f"🚨 STARTUP FAILURE: GPU acceleration required but not available!\n"
                    f"   App: {self.app_name} v{self.version}\n"
                    f"   Requirement: GPU with {self.min_gpu_memory_gb}GB+ VRAM\n"
                    f"   Found: No CUDA/GPU detected\n\n"
                    f"Production Performance Requirements:\n"
                    f"   - CPU-based TTS: 4-5x slower (40+ min for 10-min podcast)\n"
                    f"   - GPU-based TTS: <2x ratio (15-20 min for 10-min podcast)\n\n"
                    f"Required Setup:\n"
                    f"   1. Install NVIDIA GPU drivers\n"
                    f"   2. Install CUDA toolkit\n"
                    f"   3. Install PyTorch with CUDA support\n"
                    f"   4. Ensure sufficient GPU memory\n\n"
                    f"To disable GPU requirement (NOT RECOMMENDED):\n"
                    f"   Set REQUIRE_GPU=false in environment"
                )
                print(error_msg)
                raise RuntimeError(
                    "GPU acceleration required for production performance"
                )

            # Check GPU memory
            gpu_properties = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_properties.total_memory // 1024**3

            if gpu_memory_gb < self.min_gpu_memory_gb:
                error_msg = (
                    f"🚨 STARTUP FAILURE: Insufficient GPU memory!\n"
                    f"   App: {self.app_name} v{self.version}\n"
                    f"   Required: {self.min_gpu_memory_gb}GB VRAM minimum\n"
                    f"   Found: {gpu_memory_gb}GB VRAM\n"
                    f"   GPU: {torch.cuda.get_device_name(0)}\n\n"
                    f"Audio generation requires significant GPU memory for optimal performance."
                )
                print(error_msg)
                raise RuntimeError(
                    f"Insufficient GPU memory: {gpu_memory_gb}GB < {self.min_gpu_memory_gb}GB required"
                )

            # Success
            print(f"✅ GPU Requirements Met:")
            print(f"   🎯 GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"   💾 VRAM: {gpu_memory_gb}GB (required: {self.min_gpu_memory_gb}GB+)"
            )
            print(f"   🚀 {self.app_name} ready for production performance")

        except ImportError:
            error_msg = (
                f"🚨 STARTUP FAILURE: PyTorch not available!\n"
                f"   GPU validation requires PyTorch with CUDA support.\n"
                f"   Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )
            print(error_msg)
            raise RuntimeError("PyTorch with CUDA support required")


# Create global settings instance
settings = Settings()

# Legacy exports for backward compatibility
openai_api_key: Optional[str] = settings.openai_api_key
stripe_api_key: Optional[str] = settings.stripe_secret_key  # Map to stripe_secret_key
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
