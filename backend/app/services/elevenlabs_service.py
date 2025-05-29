"""
ElevenLabs Text-to-Speech Service

This service provides integration with the ElevenLabs API for text-to-speech conversion,
voice management, and audio generation. It follows the official ElevenLabs documentation
patterns and best practices.
"""

import os
import logging
from typing import Optional, List, Dict, Any, IO
from io import BytesIO
import asyncio
from datetime import datetime

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings, Voice, play
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    """Text-to-Speech request model"""

    text: str
    voice_id: Optional[str] = None
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_128"
    voice_settings: Optional[Dict[str, Any]] = None


class TTSResponse(BaseModel):
    """Text-to-Speech response model"""

    audio_data: bytes
    character_count: int
    request_id: Optional[str] = None
    voice_id: str
    model_id: str


class VoiceProfile(BaseModel):
    """Voice profile model"""

    voice_id: str
    name: str
    description: Optional[str] = None
    category: str
    labels: Dict[str, str]
    samples: Optional[List[Dict[str, Any]]] = None


class ElevenLabsService:
    """
    ElevenLabs API service for text-to-speech conversion and voice management.

    This service provides:
    - Text-to-speech conversion with various models and voices
    - Voice profile management
    - Streaming audio generation
    - Cost tracking and rate limiting
    """

    def __init__(self):
        """Initialize the ElevenLabs service with API key from environment"""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not found in environment variables")
            self.client = None
        else:
            try:
                self.client = ElevenLabs(api_key=self.api_key)
                logger.info("ElevenLabs client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ElevenLabs client: {e}")
                self.client = None

        # Default voice settings for consistent quality
        self.default_voice_settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        )

        # Predefined voice profiles for podcast hosts
        self.voice_profiles = {
            "host_1": {
                "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Callum - Clear, lighter male voice
                "name": "Felix",
                "description": "Professional male host with clear, lighter articulation",
                "category": "podcast_host",
                "personality": "analytical, thoughtful, engaging",
            },
            "host_2": {
                "voice_id": "pNInz6obpgDQGcFmaJgB",  # Adam - Deep, warm male voice
                "name": "Bjorn",
                "description": "Professional male host with deep, warm tone",
                "category": "podcast_host",
                "personality": "warm, curious, conversational",
            },
        }

    def is_available(self) -> bool:
        """Check if the ElevenLabs service is available"""
        return self.client is not None

    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection to ElevenLabs API"""
        if not self.is_available():
            return {
                "status": "error",
                "message": "ElevenLabs API key not configured",
                "available": False,
            }

        try:
            # Test by getting available voices
            voices = self.client.voices.get_all()
            return {
                "status": "success",
                "message": "ElevenLabs API connection successful",
                "available": True,
                "voice_count": len(voices.voices) if voices else 0,
            }
        except Exception as e:
            logger.error(f"ElevenLabs API test failed: {e}")
            return {
                "status": "error",
                "message": f"ElevenLabs API test failed: {str(e)}",
                "available": False,
            }

    async def convert_text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
        voice_settings: Optional[Dict[str, Any]] = None,
        enable_logging: bool = False,
    ) -> TTSResponse:
        """
        Convert text to speech using ElevenLabs API

        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use (defaults to host_1)
            model_id: Model to use for generation
            output_format: Audio output format
            voice_settings: Custom voice settings
            enable_logging: Whether to enable logging (set to False for privacy)

        Returns:
            TTSResponse with audio data and metadata
        """
        if not self.is_available():
            raise Exception("ElevenLabs API not available")

        # Use default voice if none specified
        if not voice_id:
            voice_id = self.voice_profiles["host_1"]["voice_id"]

        # Use default voice settings if none provided
        if not voice_settings:
            voice_settings_obj = self.default_voice_settings
        else:
            voice_settings_obj = VoiceSettings(**voice_settings)

        try:
            logger.info(
                f"Converting text to speech: {len(text)} characters, voice: {voice_id}"
            )

            # Convert text to speech
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                output_format=output_format,
                text=text,
                model_id=model_id,
                voice_settings=voice_settings_obj,
                enable_logging=enable_logging,
            )

            # Collect audio data
            audio_data = b"".join(chunk for chunk in audio_generator)

            logger.info(f"TTS conversion successful: {len(audio_data)} bytes generated")

            return TTSResponse(
                audio_data=audio_data,
                character_count=len(text),
                voice_id=voice_id,
                model_id=model_id,
            )

        except Exception as e:
            logger.error(f"TTS conversion failed: {e}")
            raise Exception(f"Text-to-speech conversion failed: {str(e)}")

    async def convert_text_to_speech_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
        voice_settings: Optional[Dict[str, Any]] = None,
    ) -> IO[bytes]:
        """
        Convert text to speech and return as a stream

        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use
            model_id: Model to use for generation
            output_format: Audio output format
            voice_settings: Custom voice settings

        Returns:
            BytesIO stream containing audio data
        """
        if not self.is_available():
            raise Exception("ElevenLabs API not available")

        # Use default voice if none specified
        if not voice_id:
            voice_id = self.voice_profiles["host_1"]["voice_id"]

        # Use default voice settings if none provided
        if not voice_settings:
            voice_settings_obj = self.default_voice_settings
        else:
            voice_settings_obj = VoiceSettings(**voice_settings)

        try:
            logger.info(f"Converting text to speech stream: {len(text)} characters")

            # Perform the text-to-speech conversion
            response = self.client.text_to_speech.convert_as_stream(
                voice_id=voice_id,
                output_format=output_format,
                text=text,
                model_id=model_id,
                voice_settings=voice_settings_obj,
            )

            # Create a BytesIO object to hold the audio data in memory
            audio_stream = BytesIO()

            # Write each chunk of audio data to the stream
            for chunk in response:
                if chunk:
                    audio_stream.write(chunk)

            # Reset stream position to the beginning
            audio_stream.seek(0)

            logger.info(
                f"TTS stream conversion successful: {audio_stream.getbuffer().nbytes} bytes"
            )

            return audio_stream

        except Exception as e:
            logger.error(f"TTS stream conversion failed: {e}")
            raise Exception(f"Text-to-speech stream conversion failed: {str(e)}")

    async def get_available_voices(self) -> List[VoiceProfile]:
        """Get all available voices from ElevenLabs"""
        if not self.is_available():
            raise Exception("ElevenLabs API not available")

        try:
            voices_response = self.client.voices.get_all()
            voice_profiles = []

            for voice in voices_response.voices:
                profile = VoiceProfile(
                    voice_id=voice.voice_id,
                    name=voice.name,
                    description=voice.description,
                    category=voice.category,
                    labels=voice.labels or {},
                )
                voice_profiles.append(profile)

            logger.info(f"Retrieved {len(voice_profiles)} available voices")
            return voice_profiles

        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            raise Exception(f"Failed to get available voices: {str(e)}")

    async def get_voice_details(self, voice_id: str) -> VoiceProfile:
        """Get detailed information about a specific voice"""
        if not self.is_available():
            raise Exception("ElevenLabs API not available")

        try:
            voice = self.client.voices.get(voice_id)

            profile = VoiceProfile(
                voice_id=voice.voice_id,
                name=voice.name,
                description=voice.description,
                category=voice.category,
                labels=voice.labels or {},
                samples=[
                    {
                        "sample_id": sample.sample_id,
                        "file_name": sample.file_name,
                        "mime_type": sample.mime_type,
                        "size_bytes": sample.size_bytes,
                        "hash": sample.hash,
                    }
                    for sample in (voice.samples or [])
                ],
            )

            logger.info(f"Retrieved voice details for: {voice_id}")
            return profile

        except Exception as e:
            logger.error(f"Failed to get voice details for {voice_id}: {e}")
            raise Exception(f"Failed to get voice details: {str(e)}")

    def get_podcast_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined podcast host voice profiles"""
        return self.voice_profiles

    async def generate_podcast_segment(
        self,
        text: str,
        speaker: str = "host_1",
        model_id: str = "eleven_multilingual_v2",
    ) -> TTSResponse:
        """
        Generate audio for a podcast segment with specific speaker

        Args:
            text: Text to convert to speech
            speaker: Speaker identifier (host_1, host_2, etc.)
            model_id: Model to use for generation

        Returns:
            TTSResponse with audio data
        """
        if speaker not in self.voice_profiles:
            raise ValueError(f"Unknown speaker: {speaker}")

        voice_profile = self.voice_profiles[speaker]
        voice_id = voice_profile["voice_id"]

        logger.info(
            f"Generating podcast segment for {speaker} ({voice_profile['name']})"
        )

        return await self.convert_text_to_speech(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            enable_logging=False,  # Privacy mode for podcast content
        )

    async def estimate_cost(self, text: str) -> Dict[str, Any]:
        """
        Estimate the cost for text-to-speech conversion

        Args:
            text: Text to estimate cost for

        Returns:
            Cost estimation details
        """
        character_count = len(text)

        # ElevenLabs pricing (approximate, check current pricing)
        # Standard voices: $0.30 per 1K characters
        # Professional voices: $0.30 per 1K characters
        cost_per_1k_chars = 0.30
        estimated_cost = (character_count / 1000) * cost_per_1k_chars

        return {
            "character_count": character_count,
            "estimated_cost_usd": round(estimated_cost, 4),
            "cost_per_1k_chars": cost_per_1k_chars,
            "currency": "USD",
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for ElevenLabs service"""
        health_status = {
            "service": "elevenlabs",
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "details": {},
        }

        try:
            if not self.is_available():
                health_status.update(
                    {
                        "status": "error",
                        "details": {
                            "api_key_configured": False,
                            "client_initialized": False,
                            "error": "API key not configured",
                        },
                    }
                )
                return health_status

            # Test API connection
            connection_test = await self.test_connection()

            health_status.update(
                {
                    "status": "healthy" if connection_test["available"] else "error",
                    "details": {
                        "api_key_configured": True,
                        "client_initialized": True,
                        "api_connection": connection_test["status"],
                        "voice_count": connection_test.get("voice_count", 0),
                        "voice_profiles_loaded": len(self.voice_profiles),
                    },
                }
            )

        except Exception as e:
            health_status.update(
                {
                    "status": "error",
                    "details": {
                        "error": str(e),
                        "api_key_configured": self.api_key is not None,
                        "client_initialized": self.client is not None,
                    },
                }
            )

        return health_status


# Global service instance
elevenlabs_service = ElevenLabsService()
