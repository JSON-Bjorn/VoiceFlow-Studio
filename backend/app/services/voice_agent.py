"""
Voice Agent for VoiceFlow Studio

This agent handles text-to-speech conversion using ElevenLabs API, integrating with
the enhanced 6-agent pipeline system. It manages voice generation for podcast segments
with proper speaker assignment and audio quality control.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from io import BytesIO

from .elevenlabs_service import elevenlabs_service, TTSResponse
from .openai_service import openai_service

# Configure logging
logger = logging.getLogger(__name__)


class VoiceSegment:
    """Represents a voice segment with speaker and audio data"""

    def __init__(
        self,
        text: str,
        speaker: str,
        voice_id: str,
        audio_data: bytes,
        duration_estimate: float,
        character_count: int,
        segment_id: str,
    ):
        self.text = text
        self.speaker = speaker
        self.voice_id = voice_id
        self.audio_data = audio_data
        self.duration_estimate = duration_estimate
        self.character_count = character_count
        self.segment_id = segment_id
        self.timestamp = datetime.utcnow()


class VoiceGenerationResult:
    """Result of voice generation process"""

    def __init__(self):
        self.segments: List[VoiceSegment] = []
        self.total_duration: float = 0.0
        self.total_characters: int = 0
        self.total_cost: float = 0.0
        self.success: bool = False
        self.error_message: Optional[str] = None
        self.generation_time: float = 0.0


class VoiceAgent:
    """
    Voice Agent for converting script segments to speech using ElevenLabs TTS.

    This agent:
    - Converts script segments to high-quality speech
    - Manages speaker voice assignments
    - Handles audio timing and pacing
    - Integrates with the enhanced pipeline system
    - Provides cost tracking and quality control
    """

    def __init__(self):
        """Initialize the Voice Agent"""
        self.agent_name = "Voice Agent"
        self.agent_type = "voice_generation"
        self.version = "1.0.0"

        # Voice configuration
        self.voice_profiles = {
            "host_1": {
                "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Callum - Clear, lighter male voice
                "name": "Felix",
                "personality": "analytical, thoughtful, engaging",
                "speaking_rate": 1.0,
                "pause_duration": 0.5,
            },
            "host_2": {
                "voice_id": "pNInz6obpgDQGcFmaJgB",  # Adam - Deep, warm male voice
                "name": "Bjorn",
                "personality": "warm, curious, conversational",
                "speaking_rate": 1.0,
                "pause_duration": 0.5,
            },
        }

        # Audio settings
        self.default_model = "eleven_multilingual_v2"
        self.default_format = "mp3_44100_128"

        # Quality thresholds
        self.max_segment_length = 2000  # characters
        self.min_segment_length = 10  # characters

        logger.info(f"{self.agent_name} initialized successfully")

    def is_available(self) -> bool:
        """Check if the Voice Agent is available"""
        return elevenlabs_service.is_available()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for Voice Agent"""
        health_status = {
            "agent": self.agent_name,
            "type": self.agent_type,
            "version": self.version,
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
                            "elevenlabs_available": False,
                            "error": "ElevenLabs service not available",
                        },
                    }
                )
                return health_status

            # Test ElevenLabs connection
            elevenlabs_health = await elevenlabs_service.health_check()

            health_status.update(
                {
                    "status": "healthy"
                    if elevenlabs_health["status"] == "healthy"
                    else "error",
                    "details": {
                        "elevenlabs_available": True,
                        "elevenlabs_status": elevenlabs_health["status"],
                        "voice_profiles_loaded": len(self.voice_profiles),
                        "supported_models": [self.default_model],
                        "supported_formats": [self.default_format],
                    },
                }
            )

        except Exception as e:
            health_status.update(
                {
                    "status": "error",
                    "details": {
                        "error": str(e),
                        "elevenlabs_available": self.is_available(),
                    },
                }
            )

        return health_status

    async def generate_voice_segments(
        self,
        script_segments: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> VoiceGenerationResult:
        """
        Generate voice audio for script segments

        Args:
            script_segments: List of script segments with speaker and text
            context: Additional context for voice generation

        Returns:
            VoiceGenerationResult with generated audio segments
        """
        start_time = datetime.utcnow()
        result = VoiceGenerationResult()

        try:
            logger.info(
                f"Starting voice generation for {len(script_segments)} segments"
            )

            if not self.is_available():
                result.error_message = "ElevenLabs service not available"
                return result

            # Validate segments
            validated_segments = await self._validate_segments(script_segments)
            if not validated_segments:
                result.error_message = "No valid segments to process"
                return result

            # Generate audio for each segment
            for i, segment in enumerate(validated_segments):
                try:
                    voice_segment = await self._generate_segment_audio(segment, i)
                    result.segments.append(voice_segment)
                    result.total_characters += voice_segment.character_count
                    result.total_duration += voice_segment.duration_estimate

                    logger.info(
                        f"Generated segment {i + 1}/{len(validated_segments)}: {voice_segment.character_count} chars"
                    )

                except Exception as e:
                    logger.error(f"Failed to generate segment {i}: {e}")
                    # Continue with other segments
                    continue

            # Calculate total cost
            result.total_cost = await self._calculate_total_cost(
                result.total_characters
            )

            # Mark as successful if we generated at least some segments
            result.success = len(result.segments) > 0

            if not result.success:
                result.error_message = "Failed to generate any voice segments"

            end_time = datetime.utcnow()
            result.generation_time = (end_time - start_time).total_seconds()

            logger.info(
                f"Voice generation completed: {len(result.segments)} segments, {result.total_duration:.1f}s duration"
            )

        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
            result.error_message = str(e)
            result.success = False

        return result

    async def _validate_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and prepare segments for voice generation"""
        validated = []

        for segment in segments:
            # Check required fields
            if not all(key in segment for key in ["text", "speaker"]):
                logger.warning(f"Skipping segment missing required fields: {segment}")
                continue

            text = segment["text"].strip()
            speaker = segment["speaker"]

            # Validate text length
            if len(text) < self.min_segment_length:
                logger.warning(f"Skipping segment too short: {len(text)} chars")
                continue

            if len(text) > self.max_segment_length:
                logger.warning(f"Truncating segment too long: {len(text)} chars")
                text = text[: self.max_segment_length]

            # Validate speaker
            if speaker not in self.voice_profiles:
                logger.warning(f"Unknown speaker {speaker}, using host_1")
                speaker = "host_1"

            validated.append(
                {"text": text, "speaker": speaker, "original_segment": segment}
            )

        return validated

    async def _generate_segment_audio(
        self, segment: Dict[str, Any], segment_index: int
    ) -> VoiceSegment:
        """Generate audio for a single segment"""
        text = segment["text"]
        speaker = segment["speaker"]
        voice_profile = self.voice_profiles[speaker]

        # Generate TTS audio
        tts_response = await elevenlabs_service.generate_podcast_segment(
            text=text, speaker=speaker, model_id=self.default_model
        )

        # Estimate duration (rough calculation: ~150 words per minute)
        word_count = len(text.split())
        duration_estimate = (word_count / 150) * 60  # seconds

        # Apply speaking rate adjustment
        speaking_rate = voice_profile.get("speaking_rate", 1.0)
        duration_estimate = duration_estimate / speaking_rate

        # Create voice segment
        voice_segment = VoiceSegment(
            text=text,
            speaker=speaker,
            voice_id=voice_profile["voice_id"],
            audio_data=tts_response.audio_data,
            duration_estimate=duration_estimate,
            character_count=tts_response.character_count,
            segment_id=f"segment_{segment_index:03d}_{speaker}",
        )

        logger.info(
            f"Generating podcast segment for {speaker} ({voice_profile['name']})"
        )

        return voice_segment

    async def _calculate_total_cost(self, total_characters: int) -> float:
        """Calculate total cost for voice generation"""
        cost_estimate = await elevenlabs_service.estimate_cost("x" * total_characters)
        return cost_estimate.get("estimated_cost_usd", 0.0)

    async def generate_single_voice(
        self, text: str, speaker: str = "host_1", model_id: Optional[str] = None
    ) -> VoiceSegment:
        """
        Generate voice for a single text segment

        Args:
            text: Text to convert to speech
            speaker: Speaker identifier
            model_id: TTS model to use

        Returns:
            VoiceSegment with generated audio
        """
        if not self.is_available():
            raise Exception("ElevenLabs service not available")

        if speaker not in self.voice_profiles:
            raise ValueError(f"Unknown speaker: {speaker}")

        if len(text.strip()) < self.min_segment_length:
            raise ValueError(
                f"Text too short: minimum {self.min_segment_length} characters"
            )

        if len(text) > self.max_segment_length:
            text = text[: self.max_segment_length]
            logger.warning(f"Text truncated to {self.max_segment_length} characters")

        model_id = model_id or self.default_model
        voice_profile = self.voice_profiles[speaker]

        # Generate TTS audio
        tts_response = await elevenlabs_service.generate_podcast_segment(
            text=text, speaker=speaker, model_id=model_id
        )

        # Estimate duration
        word_count = len(text.split())
        duration_estimate = (word_count / 150) * 60  # seconds
        speaking_rate = voice_profile.get("speaking_rate", 1.0)
        duration_estimate = duration_estimate / speaking_rate

        # Create voice segment
        voice_segment = VoiceSegment(
            text=text,
            speaker=speaker,
            voice_id=voice_profile["voice_id"],
            audio_data=tts_response.audio_data,
            duration_estimate=duration_estimate,
            character_count=tts_response.character_count,
            segment_id=f"single_{speaker}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        )

        logger.info(
            f"Generated single voice segment: {speaker}, {len(text)} chars, {duration_estimate:.1f}s"
        )

        return voice_segment

    def get_voice_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get available voice profiles"""
        return self.voice_profiles

    async def estimate_generation_cost(
        self, script_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate cost for generating voice from script segments"""
        total_characters = 0

        for segment in script_segments:
            if "text" in segment:
                text = segment["text"].strip()
                if len(text) >= self.min_segment_length:
                    # Truncate if too long
                    if len(text) > self.max_segment_length:
                        text = text[: self.max_segment_length]
                    total_characters += len(text)

        if total_characters == 0:
            return {
                "total_characters": 0,
                "estimated_cost_usd": 0.0,
                "segments_count": 0,
                "currency": "USD",
            }

        cost_estimate = await elevenlabs_service.estimate_cost("x" * total_characters)

        return {
            "total_characters": total_characters,
            "estimated_cost_usd": cost_estimate.get("estimated_cost_usd", 0.0),
            "segments_count": len(
                [
                    s
                    for s in script_segments
                    if "text" in s and len(s["text"].strip()) >= self.min_segment_length
                ]
            ),
            "cost_per_1k_chars": cost_estimate.get("cost_per_1k_chars", 0.30),
            "currency": "USD",
        }


# Global agent instance
voice_agent = VoiceAgent()
