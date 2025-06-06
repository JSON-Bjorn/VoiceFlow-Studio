"""
Voice Agent Service

This agent handles text-to-speech conversion using Chatterbox TTS, integrating with
the podcast generation pipeline to convert script segments into high-quality audio.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .chatterbox_service import chatterbox_service, TTSResponse
from .storage_service import storage_service

logger = logging.getLogger(__name__)


@dataclass
class VoiceGenerationResult:
    """Result of voice generation operation"""

    success: bool
    audio_segments: List[Dict[str, Any]]
    total_duration: float
    total_cost: float
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class VoiceSegment:
    """Individual voice segment"""

    text: str
    speaker_id: str
    voice_id: str
    audio_data: Optional[bytes] = None
    duration: float = 0.0
    success: bool = False
    error_message: Optional[str] = None


class VoiceAgent:
    """
    Voice Agent for converting script segments to speech using Chatterbox TTS.

    This agent handles the conversion of podcast scripts into audio segments,
    managing voice profiles, timing, and audio quality optimization.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VoiceAgent")
        self.voice_profiles = {
            "host1": {
                "voice_id": "alex",
                "name": "Alex (Host 1)",
                "style": "professional",
                "role": "primary_host",
            },
            "host2": {
                "voice_id": "sarah",
                "name": "Sarah (Host 2)",
                "style": "conversational",
                "role": "co_host",
            },
            "expert": {
                "voice_id": "tech_expert",
                "name": "Tech Expert",
                "style": "technical",
                "role": "guest_expert",
            },
            "interviewer": {
                "voice_id": "interviewer",
                "name": "Interviewer",
                "style": "curious",
                "role": "moderator",
            },
        }

    def is_available(self) -> bool:
        """Check if the voice generation service is available"""
        return chatterbox_service.is_available()

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            if not self.is_available():
                return {
                    "status": "unhealthy",
                    "chatterbox_available": False,
                    "error": "Chatterbox TTS service not available",
                    "timestamp": time.time(),
                }

            # Test Chatterbox connection
            chatterbox_health = await chatterbox_service.health_check()

            return {
                "status": "healthy"
                if chatterbox_health["status"] == "healthy"
                else "unhealthy",
                "chatterbox_available": True,
                "chatterbox_status": chatterbox_health["status"],
                "service_info": {
                    "device": chatterbox_service.device,
                    "sample_rate": chatterbox_service.sample_rate,
                },
                "voice_profiles": len(self.voice_profiles),
                "timestamp": time.time(),
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "chatterbox_available": self.is_available(),
                "error": str(e),
                "timestamp": time.time(),
            }

    async def generate_voice_segments(
        self,
        script_segments: List[Dict[str, Any]],
        voice_settings: Optional[Dict[str, Any]] = None,
        include_cost_estimate: bool = True,
    ) -> VoiceGenerationResult:
        """
        Generate voice audio for multiple script segments

        Args:
            script_segments: List of script segments with text and speaker info
            voice_settings: Optional voice generation settings
            include_cost_estimate: Whether to include cost estimation
        """
        start_time = time.time()

        try:
            if not self.is_available():
                result = VoiceGenerationResult(
                    success=False,
                    audio_segments=[],
                    total_duration=0.0,
                    total_cost=0.0,
                    processing_time=0.0,
                )
                result.error_message = "Chatterbox TTS service not available"
                return result

            audio_segments = []
            total_duration = 0.0
            total_cost = 0.0

            self.logger.info(f"Generating voice for {len(script_segments)} segments")

            for i, segment in enumerate(script_segments):
                segment_start = time.time()

                try:
                    # Extract segment information
                    text = segment.get("text", "")
                    speaker_id = segment.get("speaker", "host1")
                    segment_type = segment.get("type", "dialogue")

                    if not text.strip():
                        self.logger.warning(f"Skipping empty segment {i}")
                        continue

                    # Get voice profile for speaker
                    voice_profile = self.voice_profiles.get(
                        speaker_id, self.voice_profiles["host1"]
                    )
                    voice_id = voice_profile["voice_id"]

                    self.logger.info(
                        f"Generating audio for segment {i}: {speaker_id} ({voice_id})"
                    )

                    # Generate audio using Chatterbox
                    tts_response = await chatterbox_service.generate_podcast_segment(
                        text=text,
                        speaker_id=speaker_id,
                        segment_type=segment_type,
                        voice_settings=voice_settings,
                    )

                    if tts_response.success:
                        # Save audio data using storage service for proper path handling
                        # Create a unique segment identifier
                        segment_id = f"segment_{i}_{speaker_id}_{int(time.time())}"

                        # Save using storage service with proper podcast ID structure
                        # Use a temporary podcast ID for segments if none provided
                        temp_podcast_id = "segments"

                        file_path = await storage_service.save_audio_file(
                            audio_data=tts_response.audio_data,
                            podcast_id=temp_podcast_id,
                            segment_id=segment_id,
                            file_type="wav",
                            metadata={
                                "speaker_id": speaker_id,
                                "text_preview": text[:100] + "..."
                                if len(text) > 100
                                else text,
                                "duration": tts_response.duration,
                                "segment_index": i,
                                "voice_id": voice_id,
                            },
                        )

                        # Get the filename from the file path for URL generation
                        filename = file_path.split("/")[-1]

                        audio_segments.append(
                            {
                                "segment_index": i,
                                "segment_id": f"seg_{i}",
                                "text": text,
                                "speaker": speaker_id,
                                "speaker_id": speaker_id,
                                "voice_id": voice_id,
                                "audio_data": tts_response.audio_data,
                                "duration": tts_response.duration,
                                "duration_estimate": tts_response.duration,
                                "character_count": len(text),
                                "file_path": file_path,  # This is now the correct relative path
                                "file_url": f"/api/storage/files/{file_path}",
                                "timestamp": time.time(),
                                "success": True,
                                "processing_time": time.time() - segment_start,
                            }
                        )
                        total_duration += tts_response.duration

                        # Estimate cost (free for Chatterbox but track computational cost)
                        if include_cost_estimate:
                            cost_info = chatterbox_service.estimate_cost(text)
                            total_cost += cost_info.get("total_cost", 0.0)

                        self.logger.info(
                            f"Generated {tts_response.duration:.2f}s audio for segment {i}"
                        )
                    else:
                        audio_segments.append(
                            {
                                "segment_index": i,
                                "text": text,
                                "speaker_id": speaker_id,
                                "voice_id": voice_id,
                                "audio_data": None,
                                "duration": 0.0,
                                "success": False,
                                "error_message": tts_response.error_message,
                                "processing_time": time.time() - segment_start,
                            }
                        )
                        self.logger.error(
                            f"Failed to generate audio for segment {i}: {tts_response.error_message}"
                        )

                except Exception as segment_error:
                    self.logger.error(f"Error processing segment {i}: {segment_error}")
                    audio_segments.append(
                        {
                            "segment_index": i,
                            "text": segment.get("text", ""),
                            "speaker_id": segment.get("speaker", "unknown"),
                            "voice_id": "unknown",
                            "audio_data": None,
                            "duration": 0.0,
                            "success": False,
                            "error_message": str(segment_error),
                            "processing_time": time.time() - segment_start,
                        }
                    )

            processing_time = time.time() - start_time
            successful_segments = sum(1 for seg in audio_segments if seg["success"])

            self.logger.info(
                f"Voice generation completed: {successful_segments}/{len(audio_segments)} segments successful"
            )

            return VoiceGenerationResult(
                success=successful_segments > 0,
                audio_segments=audio_segments,
                total_duration=total_duration,
                total_cost=total_cost,
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Voice generation failed: {e}")

            result = VoiceGenerationResult(
                success=False,
                audio_segments=[],
                total_duration=0.0,
                total_cost=0.0,
                processing_time=processing_time,
            )
            result.error_message = str(e)
            return result

    async def estimate_generation_cost(
        self, script_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate the cost of generating voice for script segments"""
        try:
            if not self.is_available():
                raise Exception("Chatterbox TTS service not available")

            total_characters = 0
            segment_estimates = []

            for segment in script_segments:
                text = segment.get("text", "")
                character_count = len(text)
                total_characters += character_count

                # Get cost estimate from Chatterbox service (free but includes computational cost)
                cost_estimate = chatterbox_service.estimate_cost("x" * character_count)
                segment_estimates.append(
                    {
                        "text_preview": text[:50] + "..." if len(text) > 50 else text,
                        "character_count": character_count,
                        "estimated_time": cost_estimate.get(
                            "estimated_processing_time", 0
                        ),
                        "cost": cost_estimate.get("total_cost", 0.0),
                    }
                )

            return {
                "total_segments": len(script_segments),
                "total_characters": total_characters,
                "estimated_total_cost": 0.0,  # Free for Chatterbox
                "estimated_processing_time": sum(
                    est["estimated_time"] for est in segment_estimates
                ),
                "computational_cost": "local_processing",
                "segment_estimates": segment_estimates,
            }

        except Exception as e:
            self.logger.error(f"Cost estimation failed: {e}")
            return {
                "error": str(e),
                "total_segments": len(script_segments) if script_segments else 0,
                "total_characters": 0,
                "estimated_total_cost": 0.0,
            }

    async def generate_single_segment(
        self,
        text: str,
        speaker_id: str = "host1",
        voice_settings: Optional[Dict[str, Any]] = None,
    ) -> VoiceSegment:
        """Generate voice for a single text segment"""
        try:
            if not self.is_available():
                raise Exception("Chatterbox TTS service not available")

            # Get voice profile
            voice_profile = self.voice_profiles.get(
                speaker_id, self.voice_profiles["host1"]
            )
            voice_id = voice_profile["voice_id"]

            # Generate audio
            tts_response = await chatterbox_service.generate_podcast_segment(
                text=text,
                speaker_id=speaker_id,
                voice_settings=voice_settings,
            )

            return VoiceSegment(
                text=text,
                speaker_id=speaker_id,
                voice_id=voice_id,
                audio_data=tts_response.audio_data if tts_response.success else None,
                duration=tts_response.duration,
                success=tts_response.success,
                error_message=tts_response.error_message,
            )

        except Exception as e:
            self.logger.error(f"Single segment generation failed: {e}")
            return VoiceSegment(
                text=text,
                speaker_id=speaker_id,
                voice_id="unknown",
                success=False,
                error_message=str(e),
            )

    def get_voice_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get available voice profiles"""
        return self.voice_profiles.copy()

    def get_voice_profile(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Get specific voice profile"""
        return self.voice_profiles.get(speaker_id)

    async def test_voice_generation(
        self, text: str = "Hello, this is a test of voice generation."
    ) -> Dict[str, Any]:
        """Test voice generation with sample text"""
        try:
            segment = await self.generate_single_segment(text, "host1")

            return {
                "success": segment.success,
                "text": text,
                "duration": segment.duration,
                "voice_id": segment.voice_id,
                "error_message": segment.error_message,
                "audio_data_size": len(segment.audio_data) if segment.audio_data else 0,
            }

        except Exception as e:
            return {
                "success": False,
                "text": text,
                "error_message": str(e),
            }


# Global voice agent instance
voice_agent = VoiceAgent()
