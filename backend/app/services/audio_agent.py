"""
Audio Agent for VoiceFlow Studio

This agent handles audio file assembly and processing, combining individual voice segments
into complete podcast episodes with proper transitions, timing, and audio quality optimization.
"""

import logging
import asyncio
import io
import tempfile
import os
from typing import Dict, List, Any, Optional, Union, BinaryIO
from datetime import datetime, timedelta
from pathlib import Path

try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    from pydub.silence import split_on_silence, detect_leading_silence

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

from .storage_service import storage_service

# Configure logging
logger = logging.getLogger(__name__)


class AudioProcessingResult:
    """Result of audio processing operation"""

    def __init__(self):
        self.success: bool = False
        self.final_audio_path: Optional[str] = None
        self.final_audio_url: Optional[str] = None
        self.total_duration: float = 0.0
        self.segments_processed: int = 0
        self.processing_time: float = 0.0
        self.file_size_bytes: int = 0
        self.error_message: Optional[str] = None
        self.metadata: Dict[str, Any] = {}


class AudioSegmentInfo:
    """Information about an audio segment"""

    def __init__(
        self,
        segment_id: str,
        file_path: str,
        speaker: str,
        text: str,
        duration: float,
        order: int = 0,
    ):
        self.segment_id = segment_id
        self.file_path = file_path
        self.speaker = speaker
        self.text = text
        self.duration = duration
        self.order = order
        self.audio_data: Optional[AudioSegment] = None


class AudioAgent:
    """
    Audio Agent for assembling and processing podcast episodes

    This agent:
    - Combines individual voice segments into complete episodes
    - Adds proper audio transitions and timing
    - Handles audio quality optimization
    - Manages intro/outro music integration
    - Exports final podcast episodes
    """

    def __init__(self):
        """Initialize the Audio Agent"""
        self.agent_name = "Audio Agent"
        self.agent_type = "audio_processing"
        self.version = "1.0.0"

        # Audio processing settings
        self.default_format = "mp3"
        self.default_bitrate = "128k"
        self.default_sample_rate = 44100
        self.default_channels = 2  # Stereo

        # Timing settings (in milliseconds)
        self.speaker_transition_pause = 500  # 0.5 seconds between speakers
        self.segment_transition_pause = 300  # 0.3 seconds between segments
        self.intro_outro_fade = 2000  # 2 seconds fade for intro/outro

        # Audio processing settings
        self.normalize_audio = True
        self.apply_compression = True
        self.remove_silence_threshold = -50  # dB
        self.max_silence_duration = 1000  # 1 second max silence

        # Check dependencies
        if not PYDUB_AVAILABLE:
            logger.warning("PyDub not available. Audio processing will be limited.")

        logger.info(f"{self.agent_name} initialized successfully")

    def is_available(self) -> bool:
        """Check if the Audio Agent is available"""
        return PYDUB_AVAILABLE

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for Audio Agent"""
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
                        "status": "warning",
                        "details": {
                            "pydub_available": False,
                            "error": "PyDub not available - audio processing limited",
                            "recommendation": "Install PyDub: pip install pydub",
                        },
                    }
                )
                return health_status

            # Test basic audio processing
            test_audio = AudioSegment.silent(duration=100)  # 100ms silence

            health_status.update(
                {
                    "status": "healthy",
                    "details": {
                        "pydub_available": True,
                        "supported_formats": ["mp3", "wav", "m4a", "ogg"],
                        "audio_processing_enabled": True,
                        "default_settings": {
                            "format": self.default_format,
                            "bitrate": self.default_bitrate,
                            "sample_rate": self.default_sample_rate,
                            "channels": self.default_channels,
                        },
                    },
                }
            )

        except Exception as e:
            health_status.update(
                {
                    "status": "error",
                    "details": {"error": str(e), "pydub_available": PYDUB_AVAILABLE},
                }
            )

        return health_status

    async def assemble_podcast_episode(
        self,
        voice_segments: List[Dict[str, Any]],
        podcast_id: str,
        episode_metadata: Optional[Dict[str, Any]] = None,
    ) -> AudioProcessingResult:
        """
        Assemble individual voice segments into a complete podcast episode

        Args:
            voice_segments: List of voice segment data with file paths
            podcast_id: Podcast identifier
            episode_metadata: Optional metadata for the episode

        Returns:
            AudioProcessingResult with final episode file
        """
        start_time = datetime.utcnow()
        result = AudioProcessingResult()

        try:
            logger.info(f"Starting podcast episode assembly for podcast {podcast_id}")

            if not self.is_available():
                result.error_message = "Audio processing not available - PyDub required"
                return result

            if not voice_segments:
                result.error_message = "No voice segments provided for assembly"
                return result

            # Step 1: Load and validate audio segments
            audio_segments = await self._load_audio_segments(voice_segments)
            if not audio_segments:
                result.error_message = "Failed to load audio segments"
                return result

            # Step 2: Sort segments by order/speaker
            ordered_segments = self._order_audio_segments(audio_segments)

            # Step 3: Process and combine segments
            combined_audio = await self._combine_audio_segments(ordered_segments)
            if not combined_audio:
                result.error_message = "Failed to combine audio segments"
                return result

            # Step 4: Apply audio processing and optimization
            processed_audio = self._apply_audio_processing(combined_audio)

            # Step 5: Add intro/outro if available
            final_audio = await self._add_intro_outro(processed_audio, podcast_id)

            # Step 6: Export final episode
            export_result = await self._export_final_episode(
                final_audio, podcast_id, episode_metadata
            )

            if not export_result["success"]:
                result.error_message = (
                    f"Failed to export episode: {export_result['error']}"
                )
                return result

            # Update result
            result.success = True
            result.final_audio_path = export_result["file_path"]
            result.final_audio_url = export_result["file_url"]
            result.total_duration = len(final_audio) / 1000.0  # Convert to seconds
            result.segments_processed = len(ordered_segments)
            result.file_size_bytes = export_result["file_size"]
            result.processing_time = (datetime.utcnow() - start_time).total_seconds()

            result.metadata = {
                "original_segments": len(voice_segments),
                "processed_segments": len(ordered_segments),
                "total_duration_ms": len(final_audio),
                "sample_rate": final_audio.frame_rate,
                "channels": final_audio.channels,
                "format": self.default_format,
                "processing_applied": {
                    "normalization": self.normalize_audio,
                    "compression": self.apply_compression,
                    "silence_removal": True,
                },
            }

            logger.info(
                f"Episode assembly completed: {result.total_duration:.1f}s, "
                f"{result.segments_processed} segments, {result.file_size_bytes} bytes"
            )

        except Exception as e:
            logger.error(f"Episode assembly failed: {e}")
            result.error_message = str(e)
            result.success = False

        return result

    async def _load_audio_segments(
        self, voice_segments: List[Dict[str, Any]]
    ) -> List[AudioSegmentInfo]:
        """Load audio segments from storage"""
        audio_segments = []

        for i, segment_data in enumerate(voice_segments):
            try:
                # Extract segment information
                segment_id = segment_data.get("segment_id", f"segment_{i}")
                file_path = segment_data.get("file_path")
                speaker = segment_data.get("speaker", "unknown")
                text = segment_data.get("text", "")
                duration = segment_data.get("duration_estimate", 0)

                if not file_path:
                    logger.warning(f"No file path for segment {segment_id}")
                    continue

                # Load audio data from storage
                audio_data = await storage_service.get_audio_file(file_path)
                if not audio_data:
                    logger.warning(f"Could not load audio data for {file_path}")
                    continue

                # Convert to AudioSegment
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(audio_data), format="mp3"
                )

                # Create segment info
                segment_info = AudioSegmentInfo(
                    segment_id=segment_id,
                    file_path=file_path,
                    speaker=speaker,
                    text=text,
                    duration=duration,
                    order=i,
                )
                segment_info.audio_data = audio_segment

                audio_segments.append(segment_info)
                logger.debug(
                    f"Loaded audio segment: {segment_id} ({len(audio_segment)}ms)"
                )

            except Exception as e:
                logger.error(f"Failed to load audio segment {i}: {e}")
                continue

        logger.info(f"Loaded {len(audio_segments)} audio segments")
        return audio_segments

    def _order_audio_segments(
        self, segments: List[AudioSegmentInfo]
    ) -> List[AudioSegmentInfo]:
        """Order segments for proper episode flow"""
        # For now, use the original order
        # Could be enhanced to optimize speaker transitions, etc.
        return sorted(segments, key=lambda x: x.order)

    async def _combine_audio_segments(
        self, segments: List[AudioSegmentInfo]
    ) -> Optional[AudioSegment]:
        """Combine audio segments with proper transitions"""
        if not segments:
            return None

        try:
            combined = AudioSegment.empty()
            current_speaker = None

            for i, segment in enumerate(segments):
                audio = segment.audio_data

                # Remove leading/trailing silence
                audio = self._trim_silence(audio)

                # Add transition pause if speaker changed
                if current_speaker and current_speaker != segment.speaker:
                    pause_duration = self.speaker_transition_pause
                    pause = AudioSegment.silent(duration=pause_duration)
                    combined += pause
                    logger.debug(f"Added speaker transition pause: {pause_duration}ms")
                elif i > 0:
                    # Add smaller pause between segments from same speaker
                    pause_duration = self.segment_transition_pause
                    pause = AudioSegment.silent(duration=pause_duration)
                    combined += pause
                    logger.debug(f"Added segment transition pause: {pause_duration}ms")

                # Add the audio segment
                combined += audio
                current_speaker = segment.speaker

                logger.debug(
                    f"Added segment {segment.segment_id}: {len(audio)}ms "
                    f"(total: {len(combined)}ms)"
                )

            logger.info(
                f"Combined {len(segments)} segments into {len(combined)}ms audio"
            )
            return combined

        except Exception as e:
            logger.error(f"Failed to combine audio segments: {e}")
            return None

    def _trim_silence(self, audio: AudioSegment) -> AudioSegment:
        """Remove leading and trailing silence from audio"""
        try:
            # Detect leading silence
            leading_silence = detect_leading_silence(
                audio, silence_threshold=self.remove_silence_threshold
            )

            # Detect trailing silence
            trailing_silence = detect_leading_silence(
                audio.reverse(), silence_threshold=self.remove_silence_threshold
            )

            # Trim silence but keep some padding
            padding = 100  # 100ms padding
            start_trim = max(0, leading_silence - padding)
            end_trim = max(0, trailing_silence - padding)

            if end_trim > 0:
                audio = audio[start_trim:-end_trim]
            else:
                audio = audio[start_trim:]

            return audio

        except Exception as e:
            logger.warning(f"Failed to trim silence: {e}")
            return audio

    def _apply_audio_processing(self, audio: AudioSegment) -> AudioSegment:
        """Apply audio processing and optimization"""
        try:
            processed = audio

            # Normalize audio levels
            if self.normalize_audio:
                processed = normalize(processed)
                logger.debug("Applied audio normalization")

            # Apply dynamic range compression
            if self.apply_compression:
                processed = compress_dynamic_range(
                    processed, threshold=-20.0, ratio=2.0
                )
                logger.debug("Applied dynamic range compression")

            # Ensure consistent sample rate and channels
            if processed.frame_rate != self.default_sample_rate:
                processed = processed.set_frame_rate(self.default_sample_rate)
                logger.debug(f"Set sample rate to {self.default_sample_rate}")

            if processed.channels != self.default_channels:
                if self.default_channels == 1:
                    processed = processed.set_channels(1)  # Mono
                else:
                    processed = processed.set_channels(2)  # Stereo
                logger.debug(f"Set channels to {self.default_channels}")

            return processed

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return audio

    async def _add_intro_outro(
        self, audio: AudioSegment, podcast_id: str
    ) -> AudioSegment:
        """Add intro/outro music if available"""
        try:
            # Check for intro/outro files in storage
            # For now, just return the audio as-is
            # This will be enhanced in Task 6.5
            return audio

        except Exception as e:
            logger.warning(f"Failed to add intro/outro: {e}")
            return audio

    async def _export_final_episode(
        self,
        audio: AudioSegment,
        podcast_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export final episode to storage"""
        try:
            # Export to bytes
            buffer = io.BytesIO()
            audio.export(
                buffer,
                format=self.default_format,
                bitrate=self.default_bitrate,
                parameters=["-ar", str(self.default_sample_rate)],
            )
            audio_data = buffer.getvalue()
            buffer.close()

            # Save to storage
            file_metadata = {
                "type": "complete_episode",
                "duration_seconds": len(audio) / 1000.0,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "format": self.default_format,
                "bitrate": self.default_bitrate,
                **(metadata or {}),
            }

            file_path = await storage_service.save_audio_file(
                audio_data=audio_data,
                podcast_id=podcast_id,
                segment_id=None,  # This is a complete episode, not a segment
                file_type=self.default_format,
                metadata=file_metadata,
            )

            file_url = await storage_service.get_file_url(file_path)

            return {
                "success": True,
                "file_path": file_path,
                "file_url": file_url,
                "file_size": len(audio_data),
            }

        except Exception as e:
            logger.error(f"Failed to export episode: {e}")
            return {"success": False, "error": str(e)}


# Global agent instance
audio_agent = AudioAgent()
