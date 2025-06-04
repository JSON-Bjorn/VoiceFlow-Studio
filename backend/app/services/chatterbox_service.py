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
from typing import Optional, Dict, List, Any, BinaryIO
from pathlib import Path
from dataclasses import dataclass

import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

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
        self.sample_rate = 22050  # Default sample rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.voice_profiles = self._initialize_voice_profiles()
        self.audio_cache_dir = Path("storage/audio/cache")
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)

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

    async def initialize_model(self) -> bool:
        """Initialize the Chatterbox TTS model"""
        try:
            if self.model is None:
                logger.info("Loading Chatterbox TTS model...")
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                self.sample_rate = self.model.sr
                logger.info(f"Chatterbox model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Chatterbox model: {e}")
            return False

    def is_available(self) -> bool:
        """Check if the Chatterbox service is available"""
        return self.model is not None or torch.cuda.is_available()

    async def test_connection(self) -> Dict[str, Any]:
        """Test the Chatterbox TTS functionality"""
        try:
            if not await self.initialize_model():
                return {
                    "status": "error",
                    "message": "Failed to initialize Chatterbox model",
                    "available": False,
                }

            # Generate a short test audio
            test_text = "Hello, this is a test of Chatterbox TTS."
            wav = self.model.generate(test_text)

            return {
                "status": "success",
                "message": "Chatterbox TTS connection successful",
                "available": True,
                "device": self.device,
                "sample_rate": self.sample_rate,
                "test_audio_duration": len(wav) / self.sample_rate,
            }
        except Exception as e:
            logger.error(f"Chatterbox TTS test failed: {e}")
            return {
                "status": "error",
                "message": f"Chatterbox TTS test failed: {str(e)}",
                "available": False,
            }

    async def convert_text_to_speech(
        self,
        text: str,
        voice_id: str = "alex",
        audio_prompt_path: Optional[str] = None,
        speed: float = 1.0,
        stability: float = 0.5,
        similarity_boost: float = 0.8,
        style: float = 0.0,
    ) -> TTSResponse:
        """
        Convert text to speech using Chatterbox TTS

        Args:
            text: Text to convert to speech
            voice_id: Voice profile ID to use
            audio_prompt_path: Path to audio file for voice cloning
            speed: Speech speed (not used in current Chatterbox version)
            stability: Voice stability (not used in current Chatterbox version)
            similarity_boost: Voice similarity boost (not used in current Chatterbox version)
            style: Voice style (not used in current Chatterbox version)
        """
        try:
            if not await self.initialize_model():
                raise Exception("Chatterbox model not available")

            # Generate audio using Chatterbox
            if audio_prompt_path and os.path.exists(audio_prompt_path):
                wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
            else:
                wav = self.model.generate(text)

            # Convert tensor to bytes
            audio_bytes = self._tensor_to_bytes(wav)
            duration = len(wav) / self.sample_rate

            return TTSResponse(
                audio_data=audio_bytes,
                audio_format="wav",
                sample_rate=self.sample_rate,
                duration=duration,
                voice_id=voice_id,
                text=text,
                success=True,
            )

        except Exception as e:
            logger.error(f"TTS conversion failed: {e}")
            return TTSResponse(
                audio_data=b"",
                audio_format="wav",
                sample_rate=self.sample_rate,
                duration=0.0,
                voice_id=voice_id,
                text=text,
                success=False,
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
            model_available = await self.initialize_model()

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
