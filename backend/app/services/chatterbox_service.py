"""
Chatterbox Text-to-Speech Service

This service provides integration with the Chatterbox TTS library for text-to-speech conversion,
voice management, and audio generation. Chatterbox is an open-source alternative to ElevenLabs
that supports voice cloning and custom voice prompts.
"""

import logging
import os
import io
import hashlib
import tempfile
import time
import warnings
from typing import Optional, Dict, List, Any, BinaryIO
from pathlib import Path
from dataclasses import dataclass

# Import warning suppression context manager
from app.core.config import suppress_model_warnings

# Suppress specific deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings(
    "ignore", message=".*LoRACompatibleLinear.*", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message=".*torch.backends.cuda.sdp_kernel.*", category=FutureWarning
)
# Suppress diffusers v0.29+ deprecation warnings
warnings.filterwarnings(
    "ignore", message=".*Transformer2DModelOutput.*", category=FutureWarning
)
warnings.filterwarnings("ignore", message=".*VQEncoderOutput.*", category=FutureWarning)
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

import torch
import torchaudio

try:
    from chatterbox.tts import ChatterboxTTS

    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    ChatterboxTTS = None

logger = logging.getLogger(__name__)


@dataclass
class TTSResponse:
    """Response object for TTS operations"""

    audio_data: bytes
    audio_format: str
    sample_rate: int
    duration: float
    voice_id: str
    text: str
    success: bool
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


@dataclass
class VoiceProfile:
    """Voice profile for podcast hosts"""

    id: str
    name: str
    description: str
    gender: str
    style: str
    audio_prompt_path: Optional[str] = None
    is_custom: bool = False


class ChatterboxService:
    """
    Chatterbox TTS service for text-to-speech conversion and voice management.

    This service provides local TTS generation using the Chatterbox library,
    supporting voice cloning and custom voice prompts.
    """

    def __init__(self):
        """Initialize the Chatterbox service"""
        self.model: Optional[ChatterboxTTS] = None
        # Prefer CUDA if available, fallback to CPU
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(
                f"CUDA detected: {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB VRAM"
            )
        else:
            self.device = "cpu"
            logger.info("CUDA not available, using CPU")

        self.sample_rate = 22050  # Default sample rate
        self.voice_profiles = self._initialize_voice_profiles()
        self.audio_cache_dir = Path("storage/audio/cache")
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_loaded = False

        logger.info(f"Chatterbox service initialized on device: {self.device}")

    def _initialize_voice_profiles(self) -> Dict[str, VoiceProfile]:
        """Initialize predefined voice profiles for podcast hosts"""
        return {
            "alex": VoiceProfile(
                id="alex",
                name="Alex",
                description="Professional male voice, authoritative and clear",
                gender="male",
                style="professional",
            ),
            "sarah": VoiceProfile(
                id="sarah",
                name="Sarah",
                description="Friendly female voice, conversational and warm",
                gender="female",
                style="conversational",
            ),
            "tech_expert": VoiceProfile(
                id="tech_expert",
                name="Tech Expert",
                description="Technical specialist voice, confident and knowledgeable",
                gender="male",
                style="technical",
            ),
            "interviewer": VoiceProfile(
                id="interviewer",
                name="Interviewer",
                description="Curious and engaging voice, perfect for asking questions",
                gender="female",
                style="curious",
            ),
        }

    def _load_model(self):
        """Load the Chatterbox TTS model with proper device mapping"""
        if self._model_loaded and self.model is not None:
            return self.model

        if not CHATTERBOX_AVAILABLE:
            raise ImportError(
                "Chatterbox TTS is not installed. Please install with: pip install chatterbox-tts"
            )

        try:
            logger.info(f"Loading Chatterbox TTS model on {self.device}...")
            start_time = time.time()

            # Suppress deprecation warnings during model loading
            with suppress_model_warnings():
                # Load model with proper device mapping and CUDA compatibility
                try:
                    if self.device == "cuda" and torch.cuda.is_available():
                        logger.info(f"Loading Chatterbox TTS with CUDA support")
                        # Use modern PyTorch attention implementation during model loading
                        try:
                            with torch.backends.cuda.sdp_kernel(
                                enable_flash=True,
                                enable_math=True,
                                enable_mem_efficient=True,
                            ):
                                self.model = ChatterboxTTS.from_pretrained(
                                    device="cuda"
                                )
                        except (AttributeError, RuntimeError):
                            # Fallback without SDPA if not supported
                            self.model = ChatterboxTTS.from_pretrained(device="cuda")
                    else:
                        logger.info(f"Loading Chatterbox TTS on CPU")
                        # Force CPU loading with map_location to avoid CUDA deserialization issues
                        original_load = torch.load

                        def patched_load(*args, **kwargs):
                            kwargs["map_location"] = "cpu"
                            return original_load(*args, **kwargs)

                        torch.load = patched_load
                        try:
                            self.model = ChatterboxTTS.from_pretrained(device="cpu")
                        finally:
                            torch.load = original_load

                        self.device = "cpu"

                except Exception as device_error:
                    logger.warning(f"Failed to load on {self.device}: {device_error}")

                    # Force CPU fallback with explicit map_location patching
                    if self.device == "cuda":
                        logger.info(
                            "Attempting CPU fallback with map_location patching..."
                        )
                        original_load = torch.load

                        def patched_load(*args, **kwargs):
                            kwargs["map_location"] = "cpu"
                            return original_load(*args, **kwargs)

                        torch.load = patched_load
                        try:
                            self.model = ChatterboxTTS.from_pretrained(device="cpu")
                            self.device = "cpu"
                            logger.info(
                                "Successfully loaded model on CPU with map_location patching"
                            )
                        finally:
                            torch.load = original_load
                    else:
                        raise device_error

            load_time = time.time() - start_time
            self.sample_rate = self.model.sr
            self._model_loaded = True

            logger.info(
                f"Chatterbox TTS model loaded successfully on {self.device} in {load_time:.2f} seconds"
            )
            logger.info(f"Model sample rate: {self.sample_rate}")

            return self.model

        except Exception as e:
            logger.error(f"Failed to load Chatterbox TTS model: {e}")
            self._model_loaded = False
            raise

    def is_available(self) -> bool:
        """Check if the Chatterbox service is available"""
        try:
            self._load_model()
            return self._model_loaded
        except Exception:
            return False

    async def test_connection(self) -> Dict[str, Any]:
        """Test the Chatterbox TTS functionality"""
        try:
            model = self._load_model()

            return {
                "status": "success",
                "message": "Chatterbox TTS connection successful",
                "device": self.device,
                "sample_rate": self.sample_rate,
                "model_loaded": self._model_loaded,
                "cuda_available": torch.cuda.is_available(),
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Chatterbox TTS test failed: {str(e)}",
                "device": self.device,
                "model_loaded": False,
            }

    async def convert_text_to_speech(
        self,
        text: str,
        voice_id: str = "default",
        audio_prompt_path: Optional[str] = None,
        speed: float = 1.0,
        stability: float = 0.5,
        similarity_boost: float = 0.8,
        style: float = 0.0,
    ) -> TTSResponse:
        """Convert text to speech using Chatterbox TTS"""
        try:
            model = self._load_model()

            start_time = time.time()

            # Suppress warnings during generation
            with suppress_model_warnings():
                # Generate audio using Chatterbox with modern attention
                if self.device == "cuda" and torch.cuda.is_available():
                    # Use modern PyTorch attention for CUDA (updated syntax for PyTorch 2.7+)
                    try:
                        # Use the modern SDPA context manager
                        with torch.backends.cuda.sdp_kernel(
                            enable_flash=True,
                            enable_math=True,
                            enable_mem_efficient=True,
                        ):
                            if audio_prompt_path:
                                wav = model.generate(
                                    text, audio_prompt_path=audio_prompt_path
                                )
                            else:
                                wav = model.generate(text)
                    except (AttributeError, ImportError):
                        # Fallback for older PyTorch versions or missing SDPA
                        if audio_prompt_path:
                            wav = model.generate(
                                text, audio_prompt_path=audio_prompt_path
                            )
                        else:
                            wav = model.generate(text)
                else:
                    # CPU generation
                    if audio_prompt_path:
                        wav = model.generate(text, audio_prompt_path=audio_prompt_path)
                    else:
                        wav = model.generate(text)

            processing_time = time.time() - start_time

            # Convert to bytes
            with io.BytesIO() as buffer:
                torchaudio.save(buffer, wav, model.sr, format="wav")
                audio_data = buffer.getvalue()

            duration = wav.shape[1] / model.sr

            return TTSResponse(
                success=True,
                audio_data=audio_data,
                audio_format="wav",
                sample_rate=model.sr,
                duration=duration,
                voice_id=voice_id,
                text=text,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Text-to-speech conversion failed: {e}")
            return TTSResponse(
                success=False,
                audio_data=b"",
                audio_format="wav",
                sample_rate=22050,
                duration=0.0,
                voice_id=voice_id,
                text=text,
                error_message=str(e),
            )

    def _tensor_to_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """Convert audio tensor to bytes"""
        # Ensure tensor is on CPU and in correct format
        audio_tensor = audio_tensor.cpu().float()

        # Create a temporary file to save audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            torchaudio.save(temp_file.name, audio_tensor.unsqueeze(0), self.sample_rate)

            # Read the file back as bytes
            with open(temp_file.name, "rb") as f:
                audio_bytes = f.read()

            # Clean up temporary file
            os.unlink(temp_file.name)

        return audio_bytes

    async def convert_text_to_speech_stream(
        self,
        text: str,
        voice_id: str = "alex",
        audio_prompt_path: Optional[str] = None,
    ) -> BinaryIO:
        """Convert text to speech and return as stream"""
        response = await self.convert_text_to_speech(text, voice_id, audio_prompt_path)
        return io.BytesIO(response.audio_data)

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voice profiles"""
        voices = []
        for voice_id, profile in self.voice_profiles.items():
            voices.append(
                {
                    "voice_id": voice_id,
                    "name": profile.name,
                    "description": profile.description,
                    "gender": profile.gender,
                    "style": profile.style,
                    "is_custom": profile.is_custom,
                    "preview_url": None,  # Could add preview generation
                }
            )
        return voices

    def get_voice_details(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific voice"""
        if voice_id not in self.voice_profiles:
            return None

        profile = self.voice_profiles[voice_id]
        return {
            "voice_id": voice_id,
            "name": profile.name,
            "description": profile.description,
            "gender": profile.gender,
            "style": profile.style,
            "is_custom": profile.is_custom,
            "audio_prompt_path": profile.audio_prompt_path,
            "settings": {
                "sample_rate": self.sample_rate,
                "device": self.device,
            },
        }

    def get_podcast_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get voice profiles optimized for podcast generation"""
        return {
            "host1": {
                "id": "alex",
                "name": "Alex (Host 1)",
                "role": "primary_host",
                "personality": "authoritative, analytical",
                "voice_settings": {"style": "professional"},
            },
            "host2": {
                "id": "sarah",
                "name": "Sarah (Host 2)",
                "role": "co_host",
                "personality": "conversational, curious",
                "voice_settings": {"style": "conversational"},
            },
            "expert": {
                "id": "tech_expert",
                "name": "Tech Expert",
                "role": "guest_expert",
                "personality": "knowledgeable, confident",
                "voice_settings": {"style": "technical"},
            },
            "interviewer": {
                "id": "interviewer",
                "name": "Interviewer",
                "role": "moderator",
                "personality": "engaging, inquisitive",
                "voice_settings": {"style": "curious"},
            },
        }

    async def generate_podcast_segment(
        self,
        text: str,
        speaker_id: str,
        segment_type: str = "dialogue",
        voice_settings: Optional[Dict[str, Any]] = None,
    ) -> TTSResponse:
        """Generate audio for a podcast segment with specific speaker"""
        voice_profiles = self.get_podcast_voices()

        # Get voice ID from speaker mapping
        voice_id = "alex"  # default
        audio_prompt_path = None

        if speaker_id in voice_profiles:
            voice_id = voice_profiles[speaker_id]["id"]

        # Check if there's a custom voice prompt for this speaker
        voice_profile = self.voice_profiles.get(voice_id)
        if voice_profile and voice_profile.audio_prompt_path:
            audio_prompt_path = voice_profile.audio_prompt_path

        return await self.convert_text_to_speech(
            text=text,
            voice_id=voice_id,
            audio_prompt_path=audio_prompt_path,
        )

    def estimate_cost(self, text: str) -> Dict[str, Any]:
        """
        Estimate processing cost for text generation
        Since Chatterbox is local/free, this returns computational cost estimate
        """
        character_count = len(text)

        # Estimate processing time based on text length and hardware
        base_time_per_char = 0.01 if self.device == "cuda" else 0.05  # seconds
        estimated_time = character_count * base_time_per_char

        return {
            "character_count": character_count,
            "estimated_processing_time": estimated_time,
            "computational_cost": "local_processing",
            "api_cost": 0.0,  # Free for local processing
            "total_cost": 0.0,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for Chatterbox service"""
        health_status = {
            "service": "chatterbox",
            "status": "unknown",
            "timestamp": None,
            "details": {},
        }

        try:
            # Check if model can be initialized
            model_available = self._load_model()

            if model_available:
                # Test generation
                test_result = await self.test_connection()

                health_status.update(
                    {
                        "status": "healthy"
                        if test_result["status"] == "success"
                        else "unhealthy",
                        "details": {
                            "model_loaded": True,
                            "device": self.device,
                            "sample_rate": self.sample_rate,
                            "cuda_available": torch.cuda.is_available(),
                            "test_result": test_result,
                        },
                    }
                )
            else:
                health_status.update(
                    {
                        "status": "unhealthy",
                        "details": {
                            "model_loaded": False,
                            "error": "Failed to load Chatterbox model",
                        },
                    }
                )

        except Exception as e:
            health_status.update(
                {
                    "status": "error",
                    "details": {"error": str(e)},
                }
            )

        return health_status

    def add_custom_voice(
        self,
        voice_id: str,
        name: str,
        description: str,
        audio_prompt_path: str,
        gender: str = "unknown",
        style: str = "custom",
    ) -> bool:
        """Add a custom voice profile with audio prompt"""
        try:
            if not os.path.exists(audio_prompt_path):
                logger.error(f"Audio prompt file not found: {audio_prompt_path}")
                return False

            self.voice_profiles[voice_id] = VoiceProfile(
                id=voice_id,
                name=name,
                description=description,
                gender=gender,
                style=style,
                audio_prompt_path=audio_prompt_path,
                is_custom=True,
            )

            logger.info(f"Added custom voice profile: {voice_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add custom voice: {e}")
            return False


# Global service instance
chatterbox_service = ChatterboxService()
