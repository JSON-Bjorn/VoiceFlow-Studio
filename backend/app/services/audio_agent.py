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
    from pydub.effects import (
        normalize,
        compress_dynamic_range,
        low_pass_filter,
        high_pass_filter,
    )
    from pydub.silence import split_on_silence, detect_leading_silence
    from pydub.generators import Sine, Square, Sawtooth, Triangle, WhiteNoise

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

from .storage_service import storage_service

# Configure logging
logger = logging.getLogger(__name__)


class AudioAsset:
    """Represents an audio asset (intro, outro, transition, etc.)"""

    def __init__(
        self,
        asset_id: str,
        asset_type: str,  # "intro", "outro", "transition", "background"
        file_path: Optional[str] = None,
        audio_data: Optional[AudioSegment] = None,
        duration: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.asset_id = asset_id
        self.asset_type = asset_type
        self.file_path = file_path
        self.audio_data = audio_data
        self.duration = duration
        self.metadata = metadata or {}


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
    - Applies audio effects and enhancements
    - Exports final podcast episodes
    """

    def __init__(self):
        """Initialize the Audio Agent"""
        self.agent_name = "Audio Agent"
        self.agent_type = "audio_processing"
        self.version = "1.1.0"

        # Audio processing settings
        self.default_format = "mp3"
        self.default_bitrate = "128k"
        self.default_sample_rate = 44100
        self.default_channels = 2  # Stereo

        # Timing settings (in milliseconds)
        self.speaker_transition_pause = 500  # 0.5 seconds between speakers
        self.segment_transition_pause = 300  # 0.3 seconds between segments
        self.intro_outro_fade = 2000  # 2 seconds fade for intro/outro
        self.music_fade_in = 1500  # 1.5 seconds fade in for music
        self.music_fade_out = 1500  # 1.5 seconds fade out for music

        # Audio processing settings
        self.normalize_audio = True
        self.apply_compression = True
        self.remove_silence_threshold = -50  # dB
        self.max_silence_duration = 1000  # 1 second max silence

        # Music and effects settings
        self.intro_music_volume = -20  # dB reduction for background music
        self.outro_music_volume = -18  # dB reduction for outro music
        self.transition_effect_volume = -25  # dB reduction for transition effects
        self.background_music_volume = -30  # dB reduction for background music

        # Asset storage
        self.audio_assets: Dict[str, AudioAsset] = {}
        self.asset_directory = "storage/audio/assets"

        # Check dependencies
        if not PYDUB_AVAILABLE:
            logger.warning("PyDub not available. Audio processing will be limited.")

        # Initialize default assets when first used (lazy initialization)
        self._assets_initialized = False

        logger.info(f"{self.agent_name} v{self.version} initialized successfully")

    async def _ensure_assets_initialized(self):
        """Ensure default assets are initialized (lazy loading)"""
        if not self._assets_initialized and PYDUB_AVAILABLE:
            await self._initialize_default_assets()
            self._assets_initialized = True

    async def _initialize_default_assets(self):
        """Initialize default audio assets"""
        try:
            # Create assets directory if it doesn't exist
            os.makedirs(self.asset_directory, exist_ok=True)

            # Create default intro music (professional podcast intro)
            await self._create_default_intro()

            # Create default outro music
            await self._create_default_outro()

            # Create transition sound effects
            await self._create_transition_effects()

            logger.info("Default audio assets initialized")

        except Exception as e:
            logger.error(f"Failed to initialize default assets: {e}")

    async def _create_default_intro(self):
        """Create a default professional podcast intro"""
        try:
            if not PYDUB_AVAILABLE:
                return

            # Create a professional-sounding intro with layered tones
            base_tone = Sine(220).to_audio_segment(duration=3000)  # A3 note
            harmony = Sine(330).to_audio_segment(duration=3000)  # E4 note
            bass = Sine(110).to_audio_segment(duration=3000)  # A2 note

            # Layer the tones with different volumes
            intro = (
                base_tone.apply_gain(-6) + harmony.apply_gain(-9) + bass.apply_gain(-12)
            )

            # Add fade in and out
            intro = intro.fade_in(500).fade_out(1000)

            # Add some reverb effect by overlaying delayed versions
            delayed = intro.apply_gain(-15)
            intro = intro.overlay(delayed, position=100)  # 100ms delay
            intro = intro.overlay(delayed, position=200)  # 200ms delay

            # Create asset
            asset = AudioAsset(
                asset_id="default_intro",
                asset_type="intro",
                audio_data=intro,
                duration=len(intro) / 1000.0,
                metadata={
                    "name": "Professional Podcast Intro",
                    "description": "Default professional intro with harmonic tones",
                    "generated": True,
                },
            )

            self.audio_assets["default_intro"] = asset
            logger.debug("Created default intro music")

        except Exception as e:
            logger.error(f"Failed to create default intro: {e}")

    async def _create_default_outro(self):
        """Create a default professional podcast outro"""
        try:
            if not PYDUB_AVAILABLE:
                return

            # Create a warm, concluding outro
            base_tone = Sine(196).to_audio_segment(duration=4000)  # G3 note
            harmony = Sine(294).to_audio_segment(duration=4000)  # D4 note

            # Create a gentle fade out outro
            outro = base_tone.apply_gain(-8) + harmony.apply_gain(-12)
            outro = outro.fade_in(1000).fade_out(2000)

            # Add subtle modulation
            for i in range(0, len(outro), 200):
                if i + 200 <= len(outro):
                    segment = outro[i : i + 200]
                    modulated = segment.apply_gain(
                        -1 + (i / len(outro)) * 2
                    )  # Slight volume variation
                    outro = outro[:i] + modulated + outro[i + 200 :]

            asset = AudioAsset(
                asset_id="default_outro",
                asset_type="outro",
                audio_data=outro,
                duration=len(outro) / 1000.0,
                metadata={
                    "name": "Professional Podcast Outro",
                    "description": "Default warm concluding outro",
                    "generated": True,
                },
            )

            self.audio_assets["default_outro"] = asset
            logger.debug("Created default outro music")

        except Exception as e:
            logger.error(f"Failed to create default outro: {e}")

    async def _create_transition_effects(self):
        """Create default transition sound effects"""
        try:
            if not PYDUB_AVAILABLE:
                return

            # Create a subtle chime transition
            chime = Sine(880).to_audio_segment(duration=500)  # A5 note
            chime = chime.fade_in(50).fade_out(200).apply_gain(-15)

            chime_asset = AudioAsset(
                asset_id="default_transition",
                asset_type="transition",
                audio_data=chime,
                duration=len(chime) / 1000.0,
                metadata={
                    "name": "Subtle Chime Transition",
                    "description": "Default transition sound effect",
                    "generated": True,
                },
            )

            self.audio_assets["default_transition"] = chime_asset

            # Create a whoosh effect for dramatic transitions
            noise = WhiteNoise().to_audio_segment(duration=800)
            # Apply high-pass filter to create whoosh effect
            whoosh = high_pass_filter(noise, 1000).apply_gain(-20)
            whoosh = whoosh.fade_in(100).fade_out(300)

            whoosh_asset = AudioAsset(
                asset_id="whoosh_transition",
                asset_type="transition",
                audio_data=whoosh,
                duration=len(whoosh) / 1000.0,
                metadata={
                    "name": "Whoosh Transition",
                    "description": "Dramatic whoosh transition effect",
                    "generated": True,
                },
            )

            self.audio_assets["whoosh_transition"] = whoosh_asset
            logger.debug("Created transition effects")

        except Exception as e:
            logger.error(f"Failed to create transition effects: {e}")

    def is_available(self) -> bool:
        """Check if the Audio Agent is available"""
        return PYDUB_AVAILABLE

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health and availability of the Audio Agent

        Returns:
            Health status information
        """
        health_status = {
            "agent": self.agent_name,
            "version": self.version,
            "status": "healthy",
            "dependencies": {"pydub": PYDUB_AVAILABLE},
            "capabilities": {
                "audio_assembly": PYDUB_AVAILABLE,
                "effects_processing": PYDUB_AVAILABLE,
                "music_integration": PYDUB_AVAILABLE,
            },
            "settings": {
                "default_format": self.default_format,
                "default_bitrate": self.default_bitrate,
                "default_sample_rate": self.default_sample_rate,
                "normalization": self.normalize_audio,
                "compression": self.apply_compression,
            },
            "last_check": datetime.utcnow().isoformat(),
        }

        try:
            # Check if we can create audio segments (requires pydub)
            if PYDUB_AVAILABLE:
                test_audio = AudioSegment.silent(duration=100)  # 100ms silence
                health_status.update(
                    {
                        "status": "healthy",
                        "test_results": {
                            "audio_creation": True,
                            "audio_processing": len(test_audio) == 100,
                        },
                    }
                )
            else:
                health_status.update(
                    {
                        "status": "limited",
                        "details": {"reason": "PyDub not available"},
                        "test_results": {"audio_creation": False},
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

    async def assemble_podcast(
        self,
        segments: List[Dict[str, Any]],
        podcast_id: str,
        audio_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble voice segments into a complete podcast episode

        This is an alias for assemble_podcast_episode that matches the interface
        expected by the Enhanced Pipeline Orchestrator.

        Args:
            segments: List of voice segment data with file paths
            podcast_id: Podcast identifier
            audio_options: Optional audio processing options (intro/outro, effects, etc.)

        Returns:
            Dict with success status and assembled audio data
        """
        try:
            # Call the main assembly method
            result = await self.assemble_podcast_episode(
                voice_segments=segments,
                podcast_id=podcast_id,
                episode_metadata=None,
                audio_options=audio_options,
            )

            # Convert AudioProcessingResult to expected dict format
            return {
                "success": result.success,
                "data": {
                    "final_audio_path": result.final_audio_path,
                    "final_audio_url": result.final_audio_url,
                    "duration": result.total_duration,
                    "file_size": result.file_size_bytes,
                    "processing_time": result.processing_time,
                    "segments_processed": result.segments_processed,
                    "metadata": result.metadata,
                },
                "error": result.error_message if not result.success else None,
            }

        except Exception as e:
            logger.error(f"Podcast assembly failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None,
            }

    async def assemble_podcast_episode(
        self,
        voice_segments: List[Dict[str, Any]],
        podcast_id: str,
        episode_metadata: Optional[Dict[str, Any]] = None,
        audio_options: Optional[Dict[str, Any]] = None,
    ) -> AudioProcessingResult:
        """
        Assemble individual voice segments into a complete podcast episode

        Args:
            voice_segments: List of voice segment data with file paths
            podcast_id: Podcast identifier
            episode_metadata: Optional metadata for the episode
            audio_options: Optional audio processing options (intro/outro, effects, etc.)

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

            # Ensure assets are initialized
            await self._ensure_assets_initialized()

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

            # Step 5: Add intro/outro and effects if available
            final_audio = await self._add_intro_outro(
                processed_audio, podcast_id, audio_options
            )

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

                # Load audio data from storage - handle both old and new path formats
                audio_data = None

                # Try the file path as-is first (new format)
                try:
                    audio_data = await storage_service.get_audio_file(file_path)
                except Exception as e:
                    logger.debug(f"Failed to load {file_path} as storage path: {e}")

                # If that fails, try converting old format to new format
                if not audio_data and file_path.startswith("storage/"):
                    # Convert old format "storage/audio/segments/filename.wav" to "audio/segments/filename.wav"
                    new_path = file_path.replace("storage/", "", 1)
                    try:
                        audio_data = await storage_service.get_audio_file(new_path)
                        logger.info(f"Found file using converted path: {new_path}")
                    except Exception as e:
                        logger.debug(f"Failed to load {new_path} as storage path: {e}")

                # If still no data, try direct filesystem access (legacy support)
                if not audio_data:
                    import os
                    from pathlib import Path

                    # Try absolute path
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, "rb") as f:
                                audio_data = f.read()
                            logger.info(
                                f"Found file using direct filesystem access: {file_path}"
                            )
                        except Exception as e:
                            logger.error(f"Failed to read file directly: {e}")

                    # Try relative to current directory
                    elif os.path.exists(f"./{file_path}"):
                        try:
                            with open(f"./{file_path}", "rb") as f:
                                audio_data = f.read()
                            logger.info(
                                f"Found file using relative path: ./{file_path}"
                            )
                        except Exception as e:
                            logger.error(f"Failed to read relative file: {e}")

                if not audio_data:
                    logger.error(
                        f"Could not load audio data for {file_path} using any method"
                    )
                    continue

                # Convert to AudioSegment - auto-detect format from file extension
                try:
                    if file_path.lower().endswith(".wav"):
                        audio_segment = AudioSegment.from_file(
                            io.BytesIO(audio_data), format="wav"
                        )
                    elif file_path.lower().endswith(".mp3"):
                        audio_segment = AudioSegment.from_file(
                            io.BytesIO(audio_data), format="mp3"
                        )
                    elif file_path.lower().endswith(".m4a"):
                        audio_segment = AudioSegment.from_file(
                            io.BytesIO(audio_data), format="m4a"
                        )
                    else:
                        # Try to auto-detect format
                        logger.debug(
                            f"Unknown format for {file_path}, trying auto-detection"
                        )
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
                except Exception as format_error:
                    logger.error(
                        f"Failed to load audio format for {file_path}: {format_error}"
                    )
                    continue

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
        self,
        audio: AudioSegment,
        podcast_id: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> AudioSegment:
        """Add intro/outro music and effects to the podcast episode"""
        try:
            if not PYDUB_AVAILABLE:
                logger.warning("PyDub not available, skipping intro/outro")
                return audio

            options = options or {}
            final_audio = audio

            # Add intro music if requested
            if options.get("add_intro", True):
                intro_asset_id = options.get("intro_asset_id", "default_intro")
                intro_asset = self.audio_assets.get(intro_asset_id)

                if intro_asset and intro_asset.audio_data:
                    logger.info(f"Adding intro music: {intro_asset_id}")

                    # Prepare intro music
                    intro_music = intro_asset.audio_data
                    intro_music = intro_music.apply_gain(self.intro_music_volume)

                    # Option 1: Intro music alone, then fade into content
                    if options.get("intro_style", "overlay") == "sequential":
                        # Add intro music followed by main content
                        intro_with_fade = intro_music.fade_out(self.music_fade_out)
                        silence_gap = AudioSegment.silent(duration=500)  # 0.5s pause
                        final_audio = intro_with_fade + silence_gap + final_audio

                    # Option 2: Intro music overlaid with beginning of content
                    else:
                        # Overlay intro music with the beginning of the main content
                        intro_duration = min(len(intro_music), 10000)  # Max 10 seconds
                        intro_overlay = intro_music[:intro_duration]

                        # Create a version that fades out as voice starts
                        intro_overlay = intro_overlay.fade_out(self.music_fade_out)

                        # Overlay with main content
                        final_audio = final_audio.overlay(intro_overlay, position=0)

                else:
                    logger.warning(f"Intro asset not found: {intro_asset_id}")

            # Add outro music if requested
            if options.get("add_outro", True):
                outro_asset_id = options.get("outro_asset_id", "default_outro")
                outro_asset = self.audio_assets.get(outro_asset_id)

                if outro_asset and outro_asset.audio_data:
                    logger.info(f"Adding outro music: {outro_asset_id}")

                    # Prepare outro music
                    outro_music = outro_asset.audio_data
                    outro_music = outro_music.apply_gain(self.outro_music_volume)

                    # Option 1: Content ends, then outro music
                    if options.get("outro_style", "overlay") == "sequential":
                        silence_gap = AudioSegment.silent(duration=500)  # 0.5s pause
                        outro_with_fade = outro_music.fade_in(self.music_fade_in)
                        final_audio = final_audio + silence_gap + outro_with_fade

                    # Option 2: Outro music overlaid with end of content
                    else:
                        # Overlay outro music with the end of the content
                        outro_duration = min(len(outro_music), 8000)  # Max 8 seconds
                        outro_overlay = outro_music[:outro_duration]
                        outro_overlay = outro_overlay.fade_in(self.music_fade_in)

                        # Calculate position to start outro (near the end)
                        start_position = max(
                            0, len(final_audio) - outro_duration + 2000
                        )
                        final_audio = final_audio.overlay(
                            outro_overlay, position=start_position
                        )

                else:
                    logger.warning(f"Outro asset not found: {outro_asset_id}")

            # Add transition effects if requested
            if options.get("add_transitions", False):
                transition_asset_id = options.get(
                    "transition_asset_id", "default_transition"
                )
                transition_asset = self.audio_assets.get(transition_asset_id)

                if transition_asset and transition_asset.audio_data:
                    logger.info(f"Adding transition effects: {transition_asset_id}")

                    # This would require more complex logic to detect good transition points
                    # For now, we'll skip automatic transition insertion
                    # This could be enhanced to detect speaker changes or topic shifts
                    pass

            # Add background music if requested
            if options.get("add_background_music", False):
                await self._add_background_music(final_audio, options)

            logger.info(
                f"Intro/outro processing complete. Final duration: {len(final_audio) / 1000:.2f}s"
            )
            return final_audio

        except Exception as e:
            logger.error(f"Failed to add intro/outro: {e}")
            return audio

    async def _add_background_music(
        self, audio: AudioSegment, options: Dict[str, Any]
    ) -> AudioSegment:
        """Add subtle background music throughout the episode"""
        try:
            background_asset_id = options.get("background_asset_id")
            if not background_asset_id:
                return audio

            background_asset = self.audio_assets.get(background_asset_id)
            if not background_asset or not background_asset.audio_data:
                return audio

            logger.info(f"Adding background music: {background_asset_id}")

            background_music = background_asset.audio_data
            background_music = background_music.apply_gain(self.background_music_volume)

            # Loop background music to match content length
            content_duration = len(audio)
            if len(background_music) < content_duration:
                loops_needed = (content_duration // len(background_music)) + 1
                background_music = background_music * loops_needed

            # Trim to exact length and add fades
            background_music = background_music[:content_duration]
            background_music = background_music.fade_in(2000).fade_out(2000)

            # Overlay with main content
            return audio.overlay(background_music)

        except Exception as e:
            logger.error(f"Failed to add background music: {e}")
            return audio

    async def get_available_assets(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available audio assets"""
        # Ensure assets are initialized
        await self._ensure_assets_initialized()

        assets = {}
        for asset_id, asset in self.audio_assets.items():
            assets[asset_id] = {
                "id": asset_id,
                "type": asset.asset_type,
                "duration": asset.duration,
                "metadata": asset.metadata,
            }
        return assets

    async def load_custom_asset(
        self,
        asset_id: str,
        asset_type: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Load a custom audio asset from file"""
        try:
            if not PYDUB_AVAILABLE:
                return False

            # Load audio file
            audio_data = AudioSegment.from_file(file_path)

            asset = AudioAsset(
                asset_id=asset_id,
                asset_type=asset_type,
                file_path=file_path,
                audio_data=audio_data,
                duration=len(audio_data) / 1000.0,
                metadata=metadata or {},
            )

            self.audio_assets[asset_id] = asset
            logger.info(f"Loaded custom asset: {asset_id} ({asset_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to load custom asset {asset_id}: {e}")
            return False

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
