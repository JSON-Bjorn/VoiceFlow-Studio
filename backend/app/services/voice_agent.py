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
from .agents.voice_personality_agent import VoicePersonalityAgent

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
        # NO DEFAULT VOICE PROFILES - Users must select voices explicitly
        self.voice_profiles = {}
        # Add intelligent voice personality agent
        self.voice_personality_agent = VoicePersonalityAgent()
        logger.info(
            "🚫 NO DEFAULT VOICES: Users must explicitly select voices for podcast generation"
        )
        logger.info(
            "🎭 VoicePersonalityAgent initialized for intelligent voice optimization"
        )

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
        podcast_id: Optional[str] = None,
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
            # MANDATORY: Check if voice profiles are set up
            if not self.voice_profiles:
                result = VoiceGenerationResult(
                    success=False,
                    audio_segments=[],
                    total_duration=0.0,
                    total_cost=0.0,
                    processing_time=0.0,
                )
                result.error_message = "❌ VOICE SELECTION REQUIRED: No voice profiles configured. Users must explicitly select voices before generating audio."
                return result

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
            self.logger.info(
                f"🎯 Using user-selected voice profiles: {list(self.voice_profiles.keys())}"
            )

            for i, segment in enumerate(script_segments):
                segment_start = time.time()

                try:
                    # Extract segment information
                    text = segment.get("text", "")
                    speaker_id = segment.get("speaker", "host_1")
                    segment_type = segment.get("type", "dialogue")

                    if not text.strip():
                        self.logger.warning(f"Skipping empty segment {i}")
                        continue

                    # Get voice profile for speaker - NO FALLBACKS
                    if speaker_id not in self.voice_profiles:
                        raise ValueError(
                            f"❌ SPEAKER NOT CONFIGURED: Speaker '{speaker_id}' not found in voice profiles. "
                            f"Available speakers: {list(self.voice_profiles.keys())}. No fallback voices provided."
                        )

                    voice_profile = self.voice_profiles[speaker_id]
                    voice_id = voice_profile["voice_id"]

                    self.logger.info(
                        f"Generating audio for segment {i}: {speaker_id} ({voice_id})"
                    )
                    logger.error(
                        f"🔊 SEGMENT {i}: speaker={speaker_id}, voice_id={voice_id}"
                    )
                    logger.error(
                        f"🔊 PASSING PROFILES TO CHATTERBOX: {self.voice_profiles}"
                    )

                    # Generate audio using Chatterbox with dynamic voice profiles
                    tts_response = await chatterbox_service.generate_podcast_segment(
                        text=text,
                        speaker_id=speaker_id,
                        segment_type=segment_type,
                        voice_settings=voice_settings,
                        dynamic_voice_profiles=self.voice_profiles,  # Pass current voice profiles
                    )

                    if tts_response.success:
                        # Save audio data using storage service for proper path handling
                        # Create a unique segment identifier
                        segment_id = f"segment_{i}_{speaker_id}_{int(time.time())}"

                        # Save using storage service with proper podcast ID structure
                        # Use the provided podcast ID or fallback to "segments"
                        temp_podcast_id = podcast_id if podcast_id else "segments"

                        file_path = await storage_service.save_audio_file(
                            audio_data=tts_response.audio_data,
                            podcast_id=temp_podcast_id,
                            segment_id=segment_id,
                            file_type="mp3",
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

    async def generate_audio_with_intelligence(
        self,
        script_data: dict,
        speakers: dict,
        user_id: int,
        voice_settings: Optional[Dict[str, Any]] = None,
        podcast_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate audio with intelligent voice personality optimization"""

        start_time = time.time()
        audio_segments = []

        try:
            # Check if voice profiles are set up
            if not self.voice_profiles:
                return {
                    "success": False,
                    "error_message": "❌ VOICE SELECTION REQUIRED: No voice profiles configured",
                    "audio_segments": [],
                    "intelligence_summary": {},
                }

            self.logger.info(
                f"🎭 Starting intelligent voice generation for user {user_id}"
            )

            for i, segment in enumerate(script_data.get("segments", [])):
                for j, line in enumerate(segment.get("dialogue", [])):
                    speaker_id = line.get("speaker")
                    text = line.get("text", "")

                    if not text.strip():
                        continue

                    # Get speaker info
                    speaker_info = speakers.get(speaker_id, {})
                    speaker_role = speaker_info.get("role", "host")

                    # Intelligent emotion optimization
                    emotion_context = {
                        "decision_type": "emotion_optimization",
                        "content_text": text,
                        "speaker_role": speaker_role,
                        "content_type": self._determine_content_type(
                            i, len(script_data.get("segments", []))
                        ),
                        "user_id": user_id,
                    }

                    try:
                        emotion_decision = (
                            await self.voice_personality_agent.make_decision(
                                emotion_context
                            )
                        )
                        optimized_voice_settings = emotion_decision.data[
                            "voice_settings"
                        ]

                        self.logger.info(
                            f"🎭 Optimized voice for segment {i}-{j}: {emotion_decision.reasoning}"
                        )

                        # Get voice profile for speaker
                        if speaker_id not in self.voice_profiles:
                            raise ValueError(
                                f"Speaker '{speaker_id}' not found in voice profiles"
                            )

                        voice_profile = self.voice_profiles[speaker_id]
                        voice_id = voice_profile["voice_id"]

                        # Create enhanced voice settings combining user settings with AI optimization
                        enhanced_voice_settings = (
                            voice_settings.copy() if voice_settings else {}
                        )
                        enhanced_voice_settings.update(
                            {
                                "speed": optimized_voice_settings.get(
                                    "speed_factor", 1.0
                                ),
                                "emotion": optimized_voice_settings.get(
                                    "emotion_mode", "CONVERSATIONAL"
                                ),
                                "exaggeration": optimized_voice_settings.get(
                                    "exaggeration", 0.6
                                ),
                                "temperature": optimized_voice_settings.get(
                                    "temperature", 0.6
                                ),
                            }
                        )

                        # Generate audio with Chatterbox using optimized settings
                        tts_response = (
                            await chatterbox_service.generate_podcast_segment(
                                text=text,
                                speaker_id=speaker_id,
                                segment_type="dialogue",
                                voice_settings=enhanced_voice_settings,
                                dynamic_voice_profiles=self.voice_profiles,
                            )
                        )

                        if tts_response.success:
                            # Save audio data
                            segment_id = (
                                f"segment_{i}_{j}_{speaker_id}_{int(time.time())}"
                            )
                            temp_podcast_id = (
                                podcast_id if podcast_id else "intelligent_segments"
                            )

                            audio_file_path = await storage_service.save_audio(
                                audio_data=tts_response.audio_data,
                                podcast_id=temp_podcast_id,
                                segment_id=segment_id,
                                file_format="wav",
                            )

                            # Store audio segment with intelligence metadata
                            audio_segments.append(
                                {
                                    "segment_index": i,
                                    "line_index": j,
                                    "speaker_id": speaker_id,
                                    "text": text,
                                    "audio_file_path": audio_file_path,
                                    "duration": tts_response.duration,
                                    "success": True,
                                    "voice_settings": enhanced_voice_settings,
                                    "intelligence_metadata": {
                                        "agent_decision": emotion_decision.data,
                                        "confidence": emotion_decision.confidence,
                                        "reasoning": emotion_decision.reasoning,
                                        "optimizations_applied": {
                                            "emotion_mode": optimized_voice_settings.get(
                                                "emotion_mode"
                                            ),
                                            "speed_factor": optimized_voice_settings.get(
                                                "speed_factor"
                                            ),
                                            "exaggeration": optimized_voice_settings.get(
                                                "exaggeration"
                                            ),
                                        },
                                    },
                                }
                            )

                            self.logger.info(
                                f"✅ Generated intelligent audio for segment {i}-{j} with {emotion_decision.confidence:.2f} confidence"
                            )

                        else:
                            self.logger.error(
                                f"❌ TTS generation failed for segment {i}-{j}: {tts_response.error_message}"
                            )
                            audio_segments.append(
                                {
                                    "segment_index": i,
                                    "line_index": j,
                                    "speaker_id": speaker_id,
                                    "text": text,
                                    "success": False,
                                    "error_message": tts_response.error_message,
                                    "intelligence_metadata": {
                                        "agent_decision": emotion_decision.data,
                                        "confidence": emotion_decision.confidence,
                                        "reasoning": emotion_decision.reasoning,
                                    },
                                }
                            )

                    except Exception as e:
                        self.logger.error(
                            f"❌ Intelligent voice generation failed for segment {i}-{j}: {e}"
                        )
                        audio_segments.append(
                            {
                                "segment_index": i,
                                "line_index": j,
                                "speaker_id": speaker_id,
                                "text": text,
                                "success": False,
                                "error_message": str(e),
                            }
                        )

            # Optimize speaker dynamics across all segments if multiple speakers
            if len(speakers) > 1:
                try:
                    dynamics_context = {
                        "decision_type": "speaker_dynamics",
                        "speakers": list(speakers.values()),
                        "conversation_segments": audio_segments,
                        "target_dynamic": "balanced",
                    }

                    dynamics_decision = (
                        await self.voice_personality_agent.make_decision(
                            dynamics_context
                        )
                    )

                    # Apply dynamic optimizations (this would be for future enhancement)
                    self.logger.info(
                        f"🎭 Speaker dynamics optimized: {dynamics_decision.reasoning}"
                    )

                except Exception as e:
                    self.logger.warning(f"Speaker dynamics optimization failed: {e}")

            processing_time = time.time() - start_time
            successful_segments = sum(
                1 for seg in audio_segments if seg.get("success", False)
            )

            intelligence_summary = {
                "voice_optimizations_applied": len(audio_segments),
                "successful_optimizations": successful_segments,
                "average_confidence": sum(
                    seg.get("intelligence_metadata", {}).get("confidence", 0.0)
                    for seg in audio_segments
                    if seg.get("intelligence_metadata")
                )
                / max(
                    1,
                    len(
                        [
                            seg
                            for seg in audio_segments
                            if seg.get("intelligence_metadata")
                        ]
                    ),
                ),
                "processing_time": processing_time,
                "ai_agent_used": "VoicePersonalityAgent",
                "optimization_strategies": [
                    seg.get("intelligence_metadata", {}).get("reasoning")
                    for seg in audio_segments[:3]
                    if seg.get("intelligence_metadata")
                ],
            }

            self.logger.info(
                f"🎭 Intelligent voice generation completed: {successful_segments}/{len(audio_segments)} segments successful"
            )

            return {
                "success": successful_segments > 0,
                "audio_segments": audio_segments,
                "intelligence_summary": intelligence_summary,
                "total_duration": sum(seg.get("duration", 0) for seg in audio_segments),
                "processing_time": processing_time,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Intelligent voice generation failed: {e}")

            return {
                "success": False,
                "error_message": str(e),
                "audio_segments": audio_segments,
                "intelligence_summary": {},
                "processing_time": processing_time,
            }

    def _determine_content_type(self, segment_index: int, total_segments: int) -> str:
        """Determine content type based on position in podcast"""

        if segment_index == 0:
            return "introduction"
        elif segment_index == total_segments - 1:
            return "conclusion"
        elif segment_index == 1:
            return "main_content"
        else:
            return "transition"

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
        speaker_id: str = "host_1",
        voice_settings: Optional[Dict[str, Any]] = None,
    ) -> VoiceSegment:
        """Generate voice for a single text segment"""
        try:
            if not self.is_available():
                raise Exception("Chatterbox TTS service not available")

            # Get voice profile
            voice_profile = self.voice_profiles.get(
                speaker_id, self.voice_profiles["host_1"]
            )
            voice_id = voice_profile["voice_id"]

            # Generate audio
            tts_response = await chatterbox_service.generate_podcast_segment(
                text=text,
                speaker_id=speaker_id,
                voice_settings=voice_settings,
                dynamic_voice_profiles=self.voice_profiles,  # Pass current voice profiles
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

    def get_clean_speaker_names(self) -> Dict[str, str]:
        """Get clean speaker names for script generation"""
        logger.error(f"🎭 GET_CLEAN_NAMES called with profiles: {self.voice_profiles}")

        clean_names = {}
        for profile_id, profile_data in self.voice_profiles.items():
            full_name = profile_data.get("name", f"Host {profile_id[-1]}")

            # Handle different types of voice names:
            # 1. System voices like "David Professional" → "David"
            # 2. Custom user voices → use exactly as named
            # 3. Generic fallback names like "Host 1" → keep as is

            if full_name.startswith("Host "):
                # Keep generic host names as distinct identifiers
                clean_name = full_name  # "Host 1", "Host 2" etc.
            elif " " in full_name:
                # For names with spaces (like "David Professional", "Marcus Conversational")
                # Use just the first name to keep it natural for podcast dialogue
                parts = full_name.split()
                clean_name = parts[0]  # "David Professional" → "David"
            else:
                # Single word names (custom user names) - use exactly as is
                clean_name = full_name

            clean_names[profile_id] = clean_name

        logger.error(f"🎭 CLEAN NAMES RESULT: {clean_names}")
        return clean_names

    def get_voice_profile(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Get specific voice profile"""
        return self.voice_profiles.get(speaker_id)

    async def test_voice_generation(
        self, text: str = "Hello, this is a test of voice generation."
    ) -> Dict[str, Any]:
        """Test voice generation with sample text"""
        try:
            segment = await self.generate_single_segment(text, "host_1")

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
