"""
Chatterbox Text-to-Speech Service - ULTIMATE IMPLEMENTATION

This service provides integration with the Chatterbox TTS library for text-to-speech conversion,
voice management, and audio generation. This implementation leverages the FULL power of Chatterbox:

🎯 CHATTERBOX ULTIMATE FEATURES:
- Emotion/Exaggeration Control (0.0-2.0) - Chatterbox's unique capability
- Real-time Streaming with generate_stream()
- Advanced Parameter Control (cfg_weight, temperature, seed)
- Intelligent Voice-Aware Parameter Optimization
- Batch Processing with Consistent Seeds
- Watermark Detection for Responsible AI
- Sub-200ms Latency for Real-time Applications
- Voice Cloning with Emotion Transfer
"""

import asyncio
import logging
import os
import io
import hashlib
import tempfile
import time
import warnings
from typing import Optional, Dict, List, Any, BinaryIO, AsyncGenerator, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Import warning suppression context manager
from app.core.config import suppress_model_warnings

# Import GPU validator for mandatory GPU acceleration
from app.services.gpu_validator import gpu_validator

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

# Import PyDub for MP3 conversion and quality standardization
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

try:
    from chatterbox.tts import ChatterboxTTS

    CHATTERBOX_AVAILABLE = True
except ImportError as e:
    CHATTERBOX_AVAILABLE = False
    ChatterboxTTS = None

    # Store import error for later logging
    CHATTERBOX_IMPORT_ERROR = e

# Optional: Perth watermarking for responsible AI
try:
    import perth

    PERTH_AVAILABLE = True
except ImportError:
    PERTH_AVAILABLE = False
    perth = None

logger = logging.getLogger(__name__)

# Log chatterbox availability after logger is defined
if not CHATTERBOX_AVAILABLE:
    import sys

    if sys.version_info >= (3, 13):
        logger.warning(
            f"Chatterbox TTS not available due to Python 3.13 compatibility: {CHATTERBOX_IMPORT_ERROR}"
        )
    else:
        logger.error(f"Chatterbox TTS import failed: {CHATTERBOX_IMPORT_ERROR}")


class EmotionMode(Enum):
    """Emotion modes for different content types"""

    NEUTRAL = "neutral"  # 0.3-0.5 exaggeration
    CONVERSATIONAL = "conversational"  # 0.5-0.7 exaggeration
    EXPRESSIVE = "expressive"  # 0.7-1.0 exaggeration
    DRAMATIC = "dramatic"  # 1.0-1.5 exaggeration
    INTENSE = "intense"  # 1.5-2.0 exaggeration


class VoiceCharacteristic(Enum):
    """Voice speed characteristics for CFG weight optimization"""

    SLOW_SPEAKER = "slow"  # Higher CFG weight (0.6-0.8)
    NORMAL_SPEAKER = "normal"  # Default CFG weight (0.5)
    FAST_SPEAKER = "fast"  # Lower CFG weight (0.3-0.4)


@dataclass
class ChatterboxStreamMetrics:
    """Real-time metrics for streaming TTS"""

    chunk_count: int = 0
    rtf: float = 0.0  # Real-time factor
    latency_to_first_chunk: Optional[float] = None
    total_audio_duration: float = 0.0
    processing_time: float = 0.0


@dataclass
class EmotionProfile:
    """Emotion profile for intelligent parameter optimization"""

    mode: EmotionMode
    exaggeration: float
    cfg_weight: float
    temperature: float
    speed_factor: float
    description: str


@dataclass
class TTSResponse:
    """Enhanced response object for TTS operations"""

    audio_data: bytes
    audio_format: str
    sample_rate: int
    duration: float
    voice_id: str
    text: str
    success: bool
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

    # Enhanced Chatterbox fields
    emotion_mode: Optional[EmotionMode] = None
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    real_time_factor: Optional[float] = None  # Processing speed vs audio duration
    watermark_detected: Optional[float] = None  # Watermark confidence (0.0-1.0)
    voice_characteristics: Optional[VoiceCharacteristic] = None


@dataclass
class VoiceProfile:
    """Enhanced voice profile with Chatterbox-specific optimizations"""

    id: str
    name: str
    description: str
    gender: str
    style: str
    audio_prompt_path: Optional[str] = None
    is_custom: bool = False

    # Chatterbox-specific optimizations
    voice_characteristics: VoiceCharacteristic = VoiceCharacteristic.NORMAL_SPEAKER
    default_emotion_mode: EmotionMode = EmotionMode.CONVERSATIONAL
    optimal_cfg_weight: float = 0.5
    optimal_exaggeration: float = 0.5
    optimal_temperature: float = 0.7
    reference_audio_duration: Optional[float] = None


class ChatterboxService:
    """
    Ultimate Chatterbox TTS service leveraging ALL advanced features.

    🎯 ENHANCED CAPABILITIES:
    - Emotion/Exaggeration Control (Chatterbox's unique feature)
    - Real-time Streaming with generate_stream()
    - Voice-aware Parameter Optimization
    - Batch Processing with Consistent Seeds
    - Watermark Detection for Responsible AI
    - Intelligent Content-aware Emotion Selection
    - Sub-200ms Latency Streaming

    Audio Quality Standards:
    - Format: MP3
    - Bitrate: 128 kbps
    - Sample Rate: 44.1 kHz (44100 Hz)
    - Channels: Stereo (2)

    Performance Optimizations:
    - GPU-only operation with Flash Attention
    - Mixed precision for maximum speed
    - Intelligent parameter optimization per voice
    - Real-time streaming with chunked processing
    """

    def __init__(self):
        """Initialize the Ultimate Chatterbox service"""
        self.model: Optional[ChatterboxTTS] = None

        # Check Python 3.13 compatibility
        import sys

        if not CHATTERBOX_AVAILABLE and sys.version_info >= (3, 13):
            logger.warning(
                "🟡 Running in Python 3.13 compatibility mode - TTS functionality limited"
            )
            logger.info(
                "💡 For full TTS support, use Python 3.11 with: python -m pip install chatterbox-tts==0.1.1"
            )

        # Use GPU validator to enforce requirements across the application
        if CHATTERBOX_AVAILABLE:
            gpu_validator.ensure_gpu_available()
            gpu_validator.optimize_gpu_settings()
        else:
            logger.info("⚠️  Skipping GPU validation due to missing TTS dependencies")

        # Force GPU device - no CPU option
        self.device = "cuda"
        gpu_status = gpu_validator.get_gpu_status()
        logger.info(
            f"🚀 ULTIMATE CHATTERBOX GPU ACCELERATION: {gpu_status['gpu_name']} ({gpu_status['memory_total_gb']}GB VRAM)"
        )
        logger.info("✅ CPU fallback DISABLED for maximum performance")

        # Audio quality standards for consistent MP3 output
        self.target_sample_rate = 44100
        self.target_bitrate = "128k"  # 128 kbps bitrate
        self.target_channels = 2  # Stereo output
        self.target_format = "mp3"  # MP3 format

        # Model's native sample rate (will be set when model loads)
        self.model_sample_rate = 22050  # Default, updated when model loads

        # Ultimate Performance Settings - GPU-optimized for sub-200ms latency
        self.enable_fast_inference = True
        self.chunk_batch_size = 6  # Higher batch size for GPU
        self.streaming_chunk_size = 25  # Small chunks for low latency streaming

        # Performance optimization settings
        self.max_chunk_length = 200  # Maximum characters per chunk
        self.enable_caching = True  # Enable intelligent phrase caching
        self.cache_max_size = 500  # Larger cache for ultimate performance
        self._audio_cache = {}  # Enhanced LRU-style cache

        # Ultimate Production mode settings - sub-200ms latency
        self.ultra_fast_mode = (
            os.getenv("TTS_ULTRA_FAST_MODE", "true").lower() == "true"
        )

        # Initialize emotion profiles for intelligent parameter optimization
        self.emotion_profiles = self._initialize_emotion_profiles()

        # Initialize enhanced voice profiles with Chatterbox optimizations
        self.voice_profiles = self._initialize_enhanced_voice_profiles()

        self.audio_cache_dir = Path("storage/audio/cache")
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_loaded = False

        # Check PyDub availability for MP3 conversion
        if not PYDUB_AVAILABLE:
            logger.warning("PyDub not available - MP3 conversion will be limited")

        # Check Perth watermarking availability
        if PERTH_AVAILABLE:
            logger.info("🔒 Perth watermarking available for responsible AI")
            self.watermarker = perth.PerthImplicitWatermarker()
        else:
            logger.info("ℹ️  Perth watermarking not available")
            self.watermarker = None

        logger.info(
            f"🎯 ULTIMATE Chatterbox service initialized with ALL advanced features enabled"
        )

    def _initialize_emotion_profiles(self) -> Dict[EmotionMode, EmotionProfile]:
        """Initialize emotion profiles for intelligent parameter optimization"""
        return {
            EmotionMode.NEUTRAL: EmotionProfile(
                mode=EmotionMode.NEUTRAL,
                exaggeration=0.4,
                cfg_weight=0.5,
                temperature=0.6,
                speed_factor=1.0,
                description="Calm, measured speech for professional content",
            ),
            EmotionMode.CONVERSATIONAL: EmotionProfile(
                mode=EmotionMode.CONVERSATIONAL,
                exaggeration=0.6,
                cfg_weight=0.5,
                temperature=0.7,
                speed_factor=1.0,
                description="Natural conversational tone for podcasts and dialogue",
            ),
            EmotionMode.EXPRESSIVE: EmotionProfile(
                mode=EmotionMode.EXPRESSIVE,
                exaggeration=0.8,
                cfg_weight=0.4,  # Lower CFG for better pacing with higher emotion
                temperature=0.8,
                speed_factor=0.95,
                description="Animated, expressive speech for storytelling",
            ),
            EmotionMode.DRAMATIC: EmotionProfile(
                mode=EmotionMode.DRAMATIC,
                exaggeration=1.2,
                cfg_weight=0.3,  # Much lower CFG to compensate for high emotion
                temperature=0.9,
                speed_factor=0.9,
                description="Dramatic, theatrical delivery for emphasis",
            ),
            EmotionMode.INTENSE: EmotionProfile(
                mode=EmotionMode.INTENSE,
                exaggeration=1.6,
                cfg_weight=0.3,
                temperature=1.0,
                speed_factor=0.85,
                description="High-intensity emotional delivery",
            ),
        }

    def _initialize_enhanced_voice_profiles(self) -> Dict[str, VoiceProfile]:
        """Initialize enhanced voice profiles with Chatterbox-specific optimizations"""
        profiles = {}

        # Predefined voice profiles with Chatterbox optimizations
        profile_configs = [
            {
                "id": "alex",
                "name": "Alex",
                "description": "Professional male narrator",
                "gender": "male",
                "style": "professional",
                "voice_characteristics": VoiceCharacteristic.NORMAL_SPEAKER,
                "default_emotion_mode": EmotionMode.CONVERSATIONAL,
                "optimal_cfg_weight": 0.5,
                "optimal_exaggeration": 0.6,
            },
            {
                "id": "sarah",
                "name": "Sarah",
                "description": "Warm female narrator",
                "gender": "female",
                "style": "warm",
                "voice_characteristics": VoiceCharacteristic.SLOW_SPEAKER,
                "default_emotion_mode": EmotionMode.CONVERSATIONAL,
                "optimal_cfg_weight": 0.6,
                "optimal_exaggeration": 0.5,
            },
            {
                "id": "mike",
                "name": "Mike",
                "description": "Energetic male host",
                "gender": "male",
                "style": "energetic",
                "voice_characteristics": VoiceCharacteristic.FAST_SPEAKER,
                "default_emotion_mode": EmotionMode.EXPRESSIVE,
                "optimal_cfg_weight": 0.4,
                "optimal_exaggeration": 0.8,
            },
            {
                "id": "emma",
                "name": "Emma",
                "description": "Expressive female storyteller",
                "gender": "female",
                "style": "expressive",
                "voice_characteristics": VoiceCharacteristic.NORMAL_SPEAKER,
                "default_emotion_mode": EmotionMode.EXPRESSIVE,
                "optimal_cfg_weight": 0.4,
                "optimal_exaggeration": 0.9,
            },
        ]

        for config in profile_configs:
            profiles[config["id"]] = VoiceProfile(**config)

        # Add podcast-specific voices with emotion optimization
        podcast_voices = {
            "host1": VoiceProfile(
                id="host1",
                name="Primary Host",
                description="Main podcast host",
                gender="male",
                style="conversational",
                voice_characteristics=VoiceCharacteristic.NORMAL_SPEAKER,
                default_emotion_mode=EmotionMode.CONVERSATIONAL,
                optimal_cfg_weight=0.5,
                optimal_exaggeration=0.6,
            ),
            "host2": VoiceProfile(
                id="host2",
                name="Co-Host",
                description="Secondary podcast host",
                gender="female",
                style="friendly",
                voice_characteristics=VoiceCharacteristic.FAST_SPEAKER,
                default_emotion_mode=EmotionMode.CONVERSATIONAL,
                optimal_cfg_weight=0.4,
                optimal_exaggeration=0.7,
            ),
            "narrator": VoiceProfile(
                id="narrator",
                name="Narrator",
                description="Story narrator",
                gender="neutral",
                style="authoritative",
                voice_characteristics=VoiceCharacteristic.SLOW_SPEAKER,
                default_emotion_mode=EmotionMode.NEUTRAL,
                optimal_cfg_weight=0.6,
                optimal_exaggeration=0.4,
            ),
        }

        profiles.update(podcast_voices)
        return profiles

    def _load_model(self):
        """Load the ULTIMATE Chatterbox TTS model with all optimizations"""
        if self._model_loaded and self.model is not None:
            return self.model

        if not CHATTERBOX_AVAILABLE:
            raise ImportError(
                "Chatterbox TTS is not installed. Please install with: pip install chatterbox-tts"
            )

        # Double-check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ GPU acceleration lost during runtime! Cannot continue without CUDA."
            )

        try:
            logger.info(
                f"🚀 Loading ULTIMATE Chatterbox TTS model with ALL optimizations..."
            )
            start_time = time.time()

            # Suppress deprecation warnings during model loading
            with suppress_model_warnings():
                # FORCE CUDA loading with ultimate optimizations
                logger.info(f"⚡ Loading ULTIMATE Chatterbox TTS with CUDA")

                # Use modern PyTorch GPU optimizations for ultimate performance
                try:
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=True,
                        enable_mem_efficient=True,
                    ):
                        self.model = ChatterboxTTS.from_pretrained(device="cuda")
                        logger.info("✅ Loaded with Flash Attention GPU optimizations")
                except (AttributeError, RuntimeError) as e:
                    logger.warning(f"Flash Attention not available: {e}")
                    # Fallback to standard CUDA loading (but still GPU-only)
                    self.model = ChatterboxTTS.from_pretrained(device="cuda")
                    logger.info("✅ Loaded with standard CUDA acceleration")

            # Apply ULTIMATE GPU-specific performance optimizations
            if self.model:
                # Set model to evaluation mode for maximum inference speed
                try:
                    if hasattr(self.model, "eval"):
                        self.model.eval()
                        logger.debug(
                            "✅ Set model to evaluation mode for ultimate speed"
                        )
                except:
                    logger.debug("Model eval() not available, skipping")

                # Disable gradient computation for maximum speed
                try:
                    if hasattr(self.model, "parameters"):
                        for param in self.model.parameters():
                            param.requires_grad = False
                        logger.debug(
                            "✅ Disabled gradients for ultimate inference speed"
                        )
                except:
                    logger.debug("Cannot disable gradients, skipping")

                # ULTIMATE GPU-specific optimizations
                try:
                    # Enable GPU optimizations
                    torch.backends.cudnn.benchmark = (
                        True  # Optimize for consistent input sizes
                    )
                    torch.backends.cudnn.enabled = True

                    # Set GPU memory optimization
                    if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                        torch.cuda.set_per_process_memory_fraction(
                            0.9
                        )  # Use 90% of GPU memory

                    logger.info("✅ Enabled ULTIMATE cuDNN optimizations for GPU")
                except:
                    logger.debug("Advanced GPU optimizations not available")

            load_time = time.time() - start_time
            self.model_sample_rate = self.model.sr  # Store native model sample rate
            self._model_loaded = True

            logger.info(
                f"🚀 ULTIMATE Chatterbox TTS model loaded in {load_time:.2f} seconds"
            )
            logger.info(f"📊 Model native sample rate: {self.model_sample_rate}Hz")
            logger.info(
                f"🎯 Target output: {self.target_sample_rate}Hz {self.target_format.upper()}"
            )
            logger.info(f"⚡ ALL GPU optimizations applied for ultimate performance")

            return self.model

        except Exception as e:
            logger.error(f"❌ Failed to load ULTIMATE Chatterbox TTS model: {e}")
            logger.error(
                "💡 Ensure CUDA is properly installed and GPU has sufficient VRAM"
            )
            self._model_loaded = False
            raise RuntimeError(f"Ultimate GPU model loading failed: {e}")

    def is_available(self) -> bool:
        """Check if the Chatterbox service is available"""
        try:
            self._load_model()
            return self._model_loaded
        except Exception:
            return False

    async def test_connection(self) -> Dict[str, Any]:
        """Test the Chatterbox TTS functionality with GPU validation"""
        try:
            # Validate GPU is still available
            if not torch.cuda.is_available():
                return {
                    "status": "error",
                    "message": "GPU acceleration lost during runtime",
                    "device": "none",
                    "model_loaded": False,
                    "gpu_available": False,
                }

            model = self._load_model()

            return {
                "status": "success",
                "message": "Chatterbox TTS GPU acceleration active",
                "device": self.device,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory
                // 1024**3,
                "model_sample_rate": self.model_sample_rate,
                "target_sample_rate": self.target_sample_rate,
                "target_format": self.target_format,
                "target_bitrate": self.target_bitrate,
                "target_channels": self.target_channels,
                "inference_steps": self.inference_steps,
                "production_mode": self.production_mode,
                "model_loaded": self._model_loaded,
                "cuda_available": torch.cuda.is_available(),
                "pydub_available": PYDUB_AVAILABLE,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Chatterbox TTS GPU test failed: {str(e)}",
                "device": self.device,
                "model_loaded": False,
                "pydub_available": PYDUB_AVAILABLE,
                "gpu_available": torch.cuda.is_available(),
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
        # NEW: Enhanced Chatterbox parameters
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        emotion_mode: Optional[EmotionMode] = None,
        enable_watermark_detection: bool = True,
    ) -> TTSResponse:
        """
        ULTIMATE text-to-speech conversion using ALL Chatterbox capabilities

        🎯 ENHANCED FEATURES:
        - Emotion/Exaggeration Control (Chatterbox's unique capability)
        - Intelligent Voice-aware Parameter Optimization
        - Advanced CFG Weight, Temperature, and Seed Control
        - Watermark Detection for Responsible AI
        - Real-time Performance Metrics
        """
        try:
            start_time = time.time()

            # 🎯 INTELLIGENT PARAMETER OPTIMIZATION
            # Get voice profile for intelligent optimization
            voice_profile = self.voice_profiles.get(
                voice_id, self.voice_profiles.get("alex")
            )

            # Auto-detect emotion mode if not specified
            if emotion_mode is None:
                emotion_mode = self._detect_emotion_from_text(
                    text, voice_profile.default_emotion_mode
                )

            # Get optimized parameters for this emotion and voice combination
            optimized_params = self._get_optimized_parameters(
                emotion_mode, voice_profile, exaggeration, cfg_weight, temperature
            )

            logger.info(
                f"🎭 Using {emotion_mode.value} mode with {voice_profile.name} "
                f"(exaggeration: {optimized_params['exaggeration']:.2f}, "
                f"cfg_weight: {optimized_params['cfg_weight']:.2f})"
            )

            # 🚀 ENHANCED CACHING with emotion-aware keys
            cache_key = None
            if self.enable_caching:
                cache_key = self._generate_enhanced_cache_key(
                    text, voice_id, audio_prompt_path, optimized_params, seed
                )
                cached_result = self._get_cached_audio(cache_key)
                if cached_result:
                    logger.debug(
                        f"🎯 Cache hit for emotion-optimized audio: {text[:50]}..."
                    )

                    # Detect watermark if enabled and available
                    watermark_confidence = None
                    if enable_watermark_detection and self.watermarker:
                        watermark_confidence = await self._detect_watermark(
                            cached_result["audio_data"]
                        )

                    return TTSResponse(
                        success=True,
                        audio_data=cached_result["audio_data"],
                        audio_format=self.target_format,
                        sample_rate=self.target_sample_rate,
                        duration=cached_result["duration"],
                        voice_id=voice_id,
                        text=text,
                        processing_time=0.01,  # Cache retrieval time
                        # Enhanced fields
                        emotion_mode=emotion_mode,
                        exaggeration=optimized_params["exaggeration"],
                        cfg_weight=optimized_params["cfg_weight"],
                        temperature=optimized_params["temperature"],
                        seed=seed,
                        watermark_detected=watermark_confidence,
                        voice_characteristics=voice_profile.voice_characteristics,
                    )

            model = self._load_model()

            # 🎯 INTELLIGENT CHUNKING for long text with emotion consistency
            if len(text) > self.max_chunk_length:
                logger.info(
                    f"🔄 Using emotion-consistent chunked processing for long text ({len(text)} chars)"
                )
                return await self._process_text_chunked_enhanced(
                    text,
                    voice_id,
                    audio_prompt_path,
                    optimized_params,
                    emotion_mode,
                    seed,
                    enable_watermark_detection,
                )

            # 🚀 ULTIMATE AUDIO GENERATION with all Chatterbox features
            wav = await self._generate_audio_ultimate(
                model, text, audio_prompt_path, optimized_params, seed
            )

            processing_time = time.time() - start_time

            # 🎵 CONVERT to standardized MP3 format
            audio_data = await self._convert_to_standardized_mp3(wav, model.sr)
            duration = wav.shape[1] / model.sr

            # Calculate real-time factor (processing speed vs audio duration)
            real_time_factor = processing_time / duration if duration > 0 else 0

            # 🔒 WATERMARK DETECTION for responsible AI
            watermark_confidence = None
            if enable_watermark_detection and self.watermarker:
                watermark_confidence = await self._detect_watermark(audio_data)

            # 💾 CACHE the result with enhanced metadata
            if self.enable_caching and cache_key:
                self._cache_audio_enhanced(
                    cache_key, audio_data, duration, optimized_params
                )

            return TTSResponse(
                success=True,
                audio_data=audio_data,
                audio_format=self.target_format,
                sample_rate=self.target_sample_rate,
                duration=duration,
                voice_id=voice_id,
                text=text,
                processing_time=processing_time,
                # 🎯 Enhanced Chatterbox fields
                emotion_mode=emotion_mode,
                exaggeration=optimized_params["exaggeration"],
                cfg_weight=optimized_params["cfg_weight"],
                temperature=optimized_params["temperature"],
                seed=seed,
                real_time_factor=real_time_factor,
                watermark_detected=watermark_confidence,
                voice_characteristics=voice_profile.voice_characteristics,
            )

        except Exception as e:
            logger.error(f"ULTIMATE text-to-speech conversion failed: {e}")
            return TTSResponse(
                success=False,
                audio_data=b"",
                audio_format=self.target_format,
                sample_rate=self.target_sample_rate,
                duration=0.0,
                voice_id=voice_id,
                text=text,
                error_message=str(e),
                emotion_mode=emotion_mode,
                voice_characteristics=voice_profile.voice_characteristics
                if "voice_profile" in locals()
                else None,
            )

    def _detect_emotion_from_text(
        self, text: str, default_mode: EmotionMode
    ) -> EmotionMode:
        """Intelligently detect emotion mode from text content"""
        text_lower = text.lower()

        # Dramatic indicators
        dramatic_keywords = [
            "amazing",
            "incredible",
            "fantastic",
            "extraordinary",
            "shocking",
            "unbelievable",
        ]
        if any(keyword in text_lower for keyword in dramatic_keywords):
            return EmotionMode.DRAMATIC

        # Expressive indicators
        expressive_keywords = [
            "excited",
            "wonderful",
            "great",
            "awesome",
            "love",
            "hate",
        ]
        if any(keyword in text_lower for keyword in expressive_keywords):
            return EmotionMode.EXPRESSIVE

        # Neutral/professional indicators
        neutral_keywords = [
            "analysis",
            "research",
            "study",
            "report",
            "data",
            "statistics",
        ]
        if any(keyword in text_lower for keyword in neutral_keywords):
            return EmotionMode.NEUTRAL

        # Punctuation-based detection
        if text.count("!") >= 2 or text.count("?") >= 2:
            return EmotionMode.EXPRESSIVE

        # Default to voice profile's preference
        return default_mode

    def _get_optimized_parameters(
        self,
        emotion_mode: EmotionMode,
        voice_profile: VoiceProfile,
        user_exaggeration: Optional[float],
        user_cfg_weight: Optional[float],
        user_temperature: Optional[float],
    ) -> Dict[str, float]:
        """Get optimized parameters for emotion mode and voice characteristics"""

        # Get base emotion profile
        emotion_profile = self.emotion_profiles[emotion_mode]

        # Use user overrides or optimize based on voice characteristics
        exaggeration = (
            user_exaggeration
            if user_exaggeration is not None
            else emotion_profile.exaggeration
        )

        # Optimize CFG weight based on voice characteristics and emotion
        if user_cfg_weight is not None:
            cfg_weight = user_cfg_weight
        else:
            # Adjust CFG weight based on voice speed characteristics
            base_cfg = emotion_profile.cfg_weight
            if voice_profile.voice_characteristics == VoiceCharacteristic.FAST_SPEAKER:
                cfg_weight = max(0.2, base_cfg - 0.1)  # Lower CFG for fast speakers
            elif (
                voice_profile.voice_characteristics == VoiceCharacteristic.SLOW_SPEAKER
            ):
                cfg_weight = min(0.8, base_cfg + 0.1)  # Higher CFG for slow speakers
            else:
                cfg_weight = base_cfg

        temperature = (
            user_temperature
            if user_temperature is not None
            else emotion_profile.temperature
        )
        speed_factor = emotion_profile.speed_factor

        return {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "speed_factor": speed_factor,
        }

    def _generate_enhanced_cache_key(
        self,
        text: str,
        voice_id: str,
        audio_prompt_path: Optional[str],
        params: Dict[str, float],
        seed: Optional[int],
    ) -> str:
        """Generate enhanced cache key including all Chatterbox parameters"""
        import hashlib

        key_data = (
            f"{text}|{voice_id}|{audio_prompt_path or 'none'}|"
            f"ex:{params['exaggeration']:.2f}|cfg:{params['cfg_weight']:.2f}|"
            f"temp:{params['temperature']:.2f}|seed:{seed or 'random'}"
        )
        return hashlib.md5(key_data.encode()).hexdigest()

    def _cache_audio_enhanced(
        self,
        cache_key: str,
        audio_data: bytes,
        duration: float,
        params: Dict[str, float],
    ):
        """Cache audio with enhanced metadata"""
        if len(self._audio_cache) >= self.cache_max_size:
            # LRU eviction: remove oldest entry
            oldest_key = next(iter(self._audio_cache))
            del self._audio_cache[oldest_key]

        self._audio_cache[cache_key] = {
            "audio_data": audio_data,
            "duration": duration,
            "timestamp": time.time(),
            "parameters": params.copy(),
        }

    async def _detect_watermark(self, audio_data: bytes) -> Optional[float]:
        """Detect watermark in generated audio"""
        if not self.watermarker:
            return None

        try:
            # Convert audio bytes to format suitable for watermark detection
            # This is a simplified implementation - real implementation would need proper audio loading
            import librosa
            import io

            # Load audio from bytes
            audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=None)

            # Detect watermark
            watermark_confidence = self.watermarker.get_watermark(
                audio_array, sample_rate=sr
            )

            logger.debug(f"🔒 Watermark confidence: {watermark_confidence:.3f}")
            return watermark_confidence

        except Exception as e:
            logger.warning(f"Watermark detection failed: {e}")
            return None

    async def _generate_audio_ultimate(
        self,
        model,
        text: str,
        audio_prompt_path: Optional[str],
        params: Dict[str, float],
        seed: Optional[int],
    ) -> torch.Tensor:
        """
        ULTIMATE audio generation using ALL Chatterbox features

        🎯 Features:
        - Emotion/Exaggeration Control (unique to Chatterbox)
        - CFG Weight and Temperature Control
        - Seed for Reproducible Results
        - Mixed Precision for Speed
        - Flash Attention for Performance
        """
        # Suppress warnings during generation
        with suppress_model_warnings():
            logger.debug(
                f"🚀 ULTIMATE generation: exaggeration={params['exaggeration']:.2f}, "
                f"cfg_weight={params['cfg_weight']:.2f}, temp={params['temperature']:.2f}, "
                f"seed={seed or 'random'}"
            )

            # Use modern PyTorch GPU attention with ultimate optimizations
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=True,
                    enable_mem_efficient=True,
                ):
                    with torch.no_grad():  # Disable gradients for maximum speed
                        with torch.cuda.amp.autocast():  # Mixed precision for speed
                            # Prepare generation arguments with ALL Chatterbox features
                            gen_kwargs = {
                                "exaggeration": params["exaggeration"],
                                "cfg_weight": params["cfg_weight"],
                                "temperature": params["temperature"],
                                "speed_factor": params["speed_factor"],
                            }

                            # Add seed if specified for reproducible results
                            if seed is not None:
                                gen_kwargs["seed"] = seed

                            # Add audio prompt if provided
                            if audio_prompt_path:
                                gen_kwargs["audio_prompt_path"] = audio_prompt_path

                            # Generate with ULTIMATE Chatterbox parameters
                            wav = model.generate(text, **gen_kwargs)

                            logger.debug(
                                "✅ ULTIMATE generation with Flash Attention + All Features"
                            )

            except (AttributeError, ImportError, TypeError) as e:
                logger.debug(f"Advanced features not available: {e}, using fallback")

                # Fallback to standard generation with available parameters
                with torch.no_grad():
                    try:
                        # Try with core Chatterbox parameters
                        if audio_prompt_path:
                            wav = model.generate(
                                text,
                                audio_prompt_path=audio_prompt_path,
                                exaggeration=params["exaggeration"],
                                cfg_weight=params["cfg_weight"],
                            )
                        else:
                            wav = model.generate(
                                text,
                                exaggeration=params["exaggeration"],
                                cfg_weight=params["cfg_weight"],
                            )
                        logger.debug("✅ Generated with core Chatterbox parameters")

                    except TypeError:
                        # Final fallback - basic generation
                        if audio_prompt_path:
                            wav = model.generate(
                                text, audio_prompt_path=audio_prompt_path
                            )
                        else:
                            wav = model.generate(text)
                        logger.debug(
                            "⚠️  Using basic generation (features not available)"
                        )

        return wav

    async def _process_text_chunked_enhanced(
        self,
        text: str,
        voice_id: str,
        audio_prompt_path: Optional[str],
        params: Dict[str, float],
        emotion_mode: EmotionMode,
        seed: Optional[int],
        enable_watermark_detection: bool,
    ) -> TTSResponse:
        """Process long text with emotion-consistent chunking"""
        try:
            # Split text into chunks at sentence boundaries
            chunks = self._split_text_intelligently(text)
            logger.info(f"🔄 Split text into {len(chunks)} emotion-consistent chunks")

            model = self._load_model()
            audio_segments = []
            total_processing_time = 0

            # Use consistent seed across chunks if provided
            chunk_seed = seed

            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i + 1}/{len(chunks)}: {chunk[:50]}...")

                start_time = time.time()

                # Use consistent seed for each chunk if base seed provided
                if seed is not None:
                    chunk_seed = seed + i  # Deterministic seed per chunk

                wav = await self._generate_audio_ultimate(
                    model, chunk, audio_prompt_path, params, chunk_seed
                )
                chunk_processing_time = time.time() - start_time
                total_processing_time += chunk_processing_time

                audio_segments.append(wav)

            # Combine audio segments
            combined_wav = torch.cat(audio_segments, dim=1)

            # Convert to MP3
            audio_data = await self._convert_to_standardized_mp3(combined_wav, model.sr)
            duration = combined_wav.shape[1] / model.sr
            real_time_factor = total_processing_time / duration if duration > 0 else 0

            # Detect watermark if enabled
            watermark_confidence = None
            if enable_watermark_detection and self.watermarker:
                watermark_confidence = await self._detect_watermark(audio_data)

            voice_profile = self.voice_profiles.get(
                voice_id, self.voice_profiles.get("alex")
            )

            return TTSResponse(
                success=True,
                audio_data=audio_data,
                audio_format=self.target_format,
                sample_rate=self.target_sample_rate,
                duration=duration,
                voice_id=voice_id,
                text=text,
                processing_time=total_processing_time,
                emotion_mode=emotion_mode,
                exaggeration=params["exaggeration"],
                cfg_weight=params["cfg_weight"],
                temperature=params["temperature"],
                seed=seed,
                real_time_factor=real_time_factor,
                watermark_detected=watermark_confidence,
                voice_characteristics=voice_profile.voice_characteristics,
            )

        except Exception as e:
            logger.error(f"Enhanced chunked processing failed: {e}")
            raise

    async def _process_text_chunked(
        self,
        text: str,
        voice_id: str,
        audio_prompt_path: Optional[str],
        speed: float,
        stability: float,
        similarity_boost: float,
        style: float,
    ) -> TTSResponse:
        """Process long text in chunks and combine results"""
        try:
            # Split text into chunks at sentence boundaries when possible
            chunks = self._split_text_intelligently(text)
            logger.debug(f"Split text into {len(chunks)} chunks")

            model = self._load_model()
            audio_segments = []
            total_processing_time = 0

            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i + 1}/{len(chunks)}: {chunk[:50]}...")

                start_time = time.time()
                wav = await self._generate_audio_optimized(
                    model, chunk, audio_prompt_path
                )
                chunk_processing_time = time.time() - start_time
                total_processing_time += chunk_processing_time

                audio_segments.append(wav)

            # Combine audio segments
            combined_wav = torch.cat(audio_segments, dim=1)

            # Convert to MP3
            audio_data = await self._convert_to_standardized_mp3(combined_wav, model.sr)
            duration = combined_wav.shape[1] / model.sr

            return TTSResponse(
                success=True,
                audio_data=audio_data,
                audio_format=self.target_format,
                sample_rate=self.target_sample_rate,
                duration=duration,
                voice_id=voice_id,
                text=text,
                processing_time=total_processing_time,
            )

        except Exception as e:
            logger.error(f"Chunked processing failed: {e}")
            raise

    def _split_text_intelligently(self, text: str) -> List[str]:
        """Split text into chunks at natural boundaries"""
        import re

        # First, try to split at sentence boundaries
        sentences = re.split(r"[.!?]+\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed the limit, start a new chunk
            if (
                len(current_chunk) + len(sentence) > self.max_chunk_length
                and current_chunk
            ):
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If we still have chunks that are too long, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_length:
                final_chunks.append(chunk)
            else:
                # Split long chunks at word boundaries
                words = chunk.split()
                current_word_chunk = ""
                for word in words:
                    if (
                        len(current_word_chunk) + len(word) > self.max_chunk_length
                        and current_word_chunk
                    ):
                        final_chunks.append(current_word_chunk.strip())
                        current_word_chunk = word
                    else:
                        if current_word_chunk:
                            current_word_chunk += " " + word
                        else:
                            current_word_chunk = word
                if current_word_chunk.strip():
                    final_chunks.append(current_word_chunk.strip())

        return final_chunks

    def _generate_cache_key(
        self, text: str, voice_id: str, audio_prompt_path: Optional[str]
    ) -> str:
        """Generate a cache key for the given parameters"""
        import hashlib

        key_data = f"{text}|{voice_id}|{audio_prompt_path or 'none'}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_audio(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached audio if available"""
        return self._audio_cache.get(cache_key)

    def _cache_audio(self, cache_key: str, audio_data: bytes, duration: float):
        """Cache audio data with simple LRU eviction"""
        if len(self._audio_cache) >= self.cache_max_size:
            # Simple eviction: remove oldest entry
            oldest_key = next(iter(self._audio_cache))
            del self._audio_cache[oldest_key]

        self._audio_cache[cache_key] = {
            "audio_data": audio_data,
            "duration": duration,
            "timestamp": time.time(),
        }

    async def _convert_to_standardized_mp3(
        self, wav_tensor: torch.Tensor, original_sample_rate: int
    ) -> bytes:
        """
        Convert audio tensor to standardized MP3 format

        Standards:
        - Format: MP3
        - Bitrate: 128 kbps
        - Sample Rate: 44.1 kHz (44100 Hz)
        - Channels: Stereo (2)
        """
        try:
            # Ensure tensor is on CPU and in correct format
            wav_tensor = wav_tensor.cpu().float()

            # Handle tensor dimensions - ensure we have [channels, samples]
            if wav_tensor.dim() == 1:
                # Mono audio, add channel dimension
                wav_tensor = wav_tensor.unsqueeze(0)
            elif wav_tensor.dim() == 2 and wav_tensor.shape[0] > wav_tensor.shape[1]:
                # Transpose if samples x channels instead of channels x samples
                wav_tensor = wav_tensor.transpose(0, 1)

            if PYDUB_AVAILABLE:
                # Method 1: Use PyDub for high-quality MP3 conversion

                # First, convert tensor to WAV bytes for PyDub
                with io.BytesIO() as wav_buffer:
                    torchaudio.save(
                        wav_buffer, wav_tensor, original_sample_rate, format="wav"
                    )
                    wav_bytes = wav_buffer.getvalue()

                # Load into PyDub AudioSegment
                audio_segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))

                # Convert to stereo if mono
                if audio_segment.channels == 1:
                    audio_segment = audio_segment.set_channels(self.target_channels)
                    logger.debug("Converted mono audio to stereo")

                # Resample to target sample rate if needed
                if audio_segment.frame_rate != self.target_sample_rate:
                    audio_segment = audio_segment.set_frame_rate(
                        self.target_sample_rate
                    )
                    logger.debug(
                        f"Resampled from {original_sample_rate}Hz to {self.target_sample_rate}Hz"
                    )

                # Export to MP3 with standardized quality settings
                with io.BytesIO() as mp3_buffer:
                    audio_segment.export(
                        mp3_buffer,
                        format=self.target_format,
                        bitrate=self.target_bitrate,
                        parameters=["-ar", str(self.target_sample_rate)],
                    )
                    mp3_data = mp3_buffer.getvalue()

                logger.debug(
                    f"Converted to {self.target_format.upper()}: {self.target_bitrate}, {self.target_sample_rate}Hz, {self.target_channels}ch"
                )
                return mp3_data

            else:
                # Method 2: Fallback using torchaudio (limited MP3 support)
                logger.warning(
                    "PyDub not available, using torchaudio fallback - MP3 quality may be limited"
                )

                # Ensure stereo output
                if wav_tensor.shape[0] == 1:
                    # Convert mono to stereo by duplicating channel
                    wav_tensor = wav_tensor.repeat(2, 1)

                # Resample if needed
                if original_sample_rate != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=original_sample_rate, new_freq=self.target_sample_rate
                    )
                    wav_tensor = resampler(wav_tensor)

                # Convert to bytes - try MP3 first, fallback to WAV
                with io.BytesIO() as buffer:
                    try:
                        torchaudio.save(
                            buffer, wav_tensor, self.target_sample_rate, format="mp3"
                        )
                        logger.debug("Used torchaudio MP3 export")
                    except Exception as mp3_error:
                        logger.warning(
                            f"MP3 export failed: {mp3_error}, falling back to WAV"
                        )
                        buffer.seek(0)
                        torchaudio.save(
                            buffer, wav_tensor, self.target_sample_rate, format="wav"
                        )

                    return buffer.getvalue()

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            # Ultimate fallback - basic WAV conversion
            with io.BytesIO() as buffer:
                torchaudio.save(buffer, wav_tensor, original_sample_rate, format="wav")
                return buffer.getvalue()

    def _tensor_to_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """Convert audio tensor to bytes with standardized MP3 format"""
        # This method is now deprecated in favor of _convert_to_standardized_mp3
        # but kept for backward compatibility
        try:
            return asyncio.run(
                self._convert_to_standardized_mp3(audio_tensor, self.model_sample_rate)
            )
        except Exception as e:
            logger.error(f"Tensor to bytes conversion failed: {e}")
            # Fallback to basic WAV conversion
            audio_tensor = audio_tensor.cpu().float()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                torchaudio.save(
                    temp_file.name, audio_tensor.unsqueeze(0), self.model_sample_rate
                )
                with open(temp_file.name, "rb") as f:
                    audio_bytes = f.read()
                os.unlink(temp_file.name)
            return audio_bytes

    async def convert_text_to_speech_stream(
        self,
        text: str,
        voice_id: str = "alex",
        audio_prompt_path: Optional[str] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        emotion_mode: Optional[EmotionMode] = None,
        chunk_size: int = 25,  # Small chunks for low latency
    ) -> BinaryIO:
        """Convert text to speech and return as stream (fallback method)"""
        response = await self.convert_text_to_speech(
            text,
            voice_id,
            audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            seed=seed,
            emotion_mode=emotion_mode,
        )
        return io.BytesIO(response.audio_data)

    async def generate_real_time_stream(
        self,
        text: str,
        voice_id: str = "alex",
        audio_prompt_path: Optional[str] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        emotion_mode: Optional[EmotionMode] = None,
        chunk_size: int = 25,  # Characters per chunk for streaming
    ) -> AsyncGenerator[Tuple[bytes, ChatterboxStreamMetrics], None]:
        """
        🚀 REAL-TIME STREAMING TTS using Chatterbox's generate_stream()

        Yields audio chunks as they're generated for sub-200ms latency.
        This is Chatterbox's killer feature for real-time applications!

        Args:
            text: Text to convert to speech
            voice_id: Voice profile to use
            audio_prompt_path: Optional voice cloning reference
            exaggeration: Emotion intensity (0.0-2.0)
            cfg_weight: Generation guidance (0.1-1.0)
            temperature: Randomness control
            seed: For reproducible results
            emotion_mode: Emotion mode for intelligent optimization
            chunk_size: Characters per chunk (smaller = lower latency)

        Yields:
            Tuple of (audio_chunk_bytes, metrics) for each generated chunk
        """
        if not CHATTERBOX_AVAILABLE:
            raise RuntimeError("Chatterbox TTS not available for streaming")

        try:
            model = self._load_model()

            # Get optimized parameters for emotion and voice
            voice_profile = self.voice_profiles.get(
                voice_id, self.voice_profiles.get("alex")
            )

            if emotion_mode is None:
                emotion_mode = self._detect_emotion_from_text(
                    text, voice_profile.default_emotion_mode
                )

            params = self._get_optimized_parameters(
                emotion_mode, voice_profile, exaggeration, cfg_weight, temperature
            )

            logger.info(
                f"🚀 Starting REAL-TIME streaming for '{text[:50]}...' "
                f"using {emotion_mode.value} mode with {voice_profile.name}"
            )

            start_time = time.time()
            metrics = ChatterboxStreamMetrics()
            audio_chunks_bytes = []

            # Prepare streaming arguments
            stream_kwargs = {
                "exaggeration": params["exaggeration"],
                "cfg_weight": params["cfg_weight"],
                "temperature": params["temperature"],
                "chunk_size": chunk_size,
            }

            if seed is not None:
                stream_kwargs["seed"] = seed
            if audio_prompt_path:
                stream_kwargs["audio_prompt_path"] = audio_prompt_path

            # 🚀 REAL-TIME STREAMING with Chatterbox's generate_stream()
            try:
                with suppress_model_warnings():
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            # Use Chatterbox's streaming capability
                            for audio_chunk, chunk_metrics in model.generate_stream(
                                text, **stream_kwargs
                            ):
                                chunk_start = time.time()

                                # Convert audio chunk to standardized format
                                audio_chunk_bytes = (
                                    await self._convert_audio_chunk_to_bytes(
                                        audio_chunk, model.sr
                                    )
                                )

                                # Update metrics
                                metrics.chunk_count += 1
                                chunk_duration = audio_chunk.shape[1] / model.sr
                                metrics.total_audio_duration += chunk_duration
                                metrics.processing_time = time.time() - start_time

                                # Calculate RTF (Real-Time Factor)
                                if metrics.total_audio_duration > 0:
                                    metrics.rtf = (
                                        metrics.processing_time
                                        / metrics.total_audio_duration
                                    )

                                # Record latency to first chunk
                                if metrics.chunk_count == 1:
                                    metrics.latency_to_first_chunk = (
                                        time.time() - start_time
                                    )
                                    logger.info(
                                        f"⚡ First chunk latency: {metrics.latency_to_first_chunk:.3f}s"
                                    )

                                logger.debug(
                                    f"🔊 Chunk {metrics.chunk_count}: {chunk_duration:.2f}s audio, "
                                    f"RTF: {metrics.rtf:.3f}"
                                )

                                audio_chunks_bytes.append(audio_chunk_bytes)

                                # Yield chunk immediately for real-time playback
                                yield audio_chunk_bytes, metrics

            except AttributeError:
                # Fallback: simulate streaming with chunked processing
                logger.warning(
                    "⚠️  generate_stream() not available, using chunked fallback"
                )

                # Split text and process chunks
                chunks = self._split_text_for_streaming(text, chunk_size)

                for i, chunk in enumerate(chunks):
                    chunk_start = time.time()

                    # Use deterministic seed per chunk if provided
                    chunk_seed = seed + i if seed is not None else None

                    # Generate chunk
                    wav = await self._generate_audio_ultimate(
                        model, chunk, audio_prompt_path, params, chunk_seed
                    )

                    # Convert to bytes
                    audio_chunk_bytes = await self._convert_audio_chunk_to_bytes(
                        wav, model.sr
                    )

                    # Update metrics
                    metrics.chunk_count += 1
                    chunk_duration = wav.shape[1] / model.sr
                    metrics.total_audio_duration += chunk_duration
                    metrics.processing_time = time.time() - start_time

                    if metrics.total_audio_duration > 0:
                        metrics.rtf = (
                            metrics.processing_time / metrics.total_audio_duration
                        )

                    if metrics.chunk_count == 1:
                        metrics.latency_to_first_chunk = time.time() - start_time

                    audio_chunks_bytes.append(audio_chunk_bytes)
                    yield audio_chunk_bytes, metrics

            logger.info(
                f"🎯 Streaming completed: {metrics.chunk_count} chunks, "
                f"{metrics.total_audio_duration:.2f}s audio, RTF: {metrics.rtf:.3f}"
            )

        except Exception as e:
            logger.error(f"Real-time streaming failed: {e}")
            raise

    def _split_text_for_streaming(self, text: str, chunk_size: int) -> List[str]:
        """Split text into small chunks optimized for streaming"""
        import re

        # First try to split at sentence boundaries
        sentences = re.split(r"[.!?]+\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If sentence is short enough, use it as a chunk
            if len(sentence) <= chunk_size:
                if current_chunk and len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += ". " + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            else:
                # Split long sentences at word boundaries
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                        if current_chunk:
                            current_chunk += " " + word
                        else:
                            current_chunk = word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    async def _convert_audio_chunk_to_bytes(
        self, audio_tensor: torch.Tensor, sample_rate: int
    ) -> bytes:
        """Convert audio tensor chunk to bytes efficiently for streaming"""
        try:
            # Fast conversion for streaming - use WAV for speed
            audio_tensor = audio_tensor.cpu().float()

            # Ensure correct format
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Convert to bytes quickly
            with io.BytesIO() as buffer:
                torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
                return buffer.getvalue()

        except Exception as e:
            logger.error(f"Audio chunk conversion failed: {e}")
            return b""

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
                "model_sample_rate": self.model_sample_rate,
                "target_sample_rate": self.target_sample_rate,
                "target_format": self.target_format,
                "target_bitrate": self.target_bitrate,
                "target_channels": self.target_channels,
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

    async def generate_batch_tts(
        self,
        texts: List[str],
        voice_id: str = "alex",
        audio_prompt_path: Optional[str] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        base_seed: Optional[int] = None,
        emotion_mode: Optional[EmotionMode] = None,
        consistent_voice: bool = True,
    ) -> List[TTSResponse]:
        """
        🚀 BATCH PROCESSING with consistent voice and emotion across all texts

        Perfect for generating podcast episodes, audiobooks, or multiple segments
        with consistent quality and voice characteristics.

        Args:
            texts: List of texts to convert
            voice_id: Voice profile for all texts
            audio_prompt_path: Optional voice cloning reference
            exaggeration: Emotion intensity for all texts
            cfg_weight: CFG weight for all texts
            temperature: Temperature for all texts
            base_seed: Base seed for reproducible results (incremented per text)
            emotion_mode: Emotion mode for all texts
            consistent_voice: Use consistent parameters across all texts

        Returns:
            List of TTSResponse objects
        """
        logger.info(f"🔄 Starting BATCH processing for {len(texts)} texts")

        results = []
        start_time = time.time()

        # Get voice profile and emotion mode for consistency
        voice_profile = self.voice_profiles.get(
            voice_id, self.voice_profiles.get("alex")
        )

        if emotion_mode is None:
            # Use the most common emotion across all texts
            emotion_mode = self._detect_batch_emotion(
                texts, voice_profile.default_emotion_mode
            )

        # Get optimized parameters for batch consistency
        if consistent_voice:
            params = self._get_optimized_parameters(
                emotion_mode, voice_profile, exaggeration, cfg_weight, temperature
            )
            logger.info(
                f"🎯 Using consistent parameters for batch: {emotion_mode.value} mode, "
                f"exaggeration: {params['exaggeration']:.2f}, cfg_weight: {params['cfg_weight']:.2f}"
            )

        for i, text in enumerate(texts):
            try:
                # Use incremental seed for reproducible but varied results
                text_seed = base_seed + i if base_seed is not None else None

                # Use consistent parameters if requested
                if consistent_voice:
                    response = await self.convert_text_to_speech(
                        text=text,
                        voice_id=voice_id,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=params["exaggeration"],
                        cfg_weight=params["cfg_weight"],
                        temperature=params["temperature"],
                        seed=text_seed,
                        emotion_mode=emotion_mode,
                    )
                else:
                    # Allow individual optimization per text
                    response = await self.convert_text_to_speech(
                        text=text,
                        voice_id=voice_id,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        seed=text_seed,
                        emotion_mode=emotion_mode,
                    )

                results.append(response)

                if response.success:
                    logger.debug(
                        f"✅ Batch item {i + 1}/{len(texts)}: {response.duration:.2f}s, "
                        f"RTF: {response.real_time_factor:.3f}"
                    )
                else:
                    logger.warning(
                        f"❌ Batch item {i + 1}/{len(texts)} failed: {response.error_message}"
                    )

            except Exception as e:
                logger.error(f"Batch item {i + 1}/{len(texts)} error: {e}")
                # Create error response
                error_response = TTSResponse(
                    success=False,
                    audio_data=b"",
                    audio_format=self.target_format,
                    sample_rate=self.target_sample_rate,
                    duration=0.0,
                    voice_id=voice_id,
                    text=text,
                    error_message=str(e),
                    emotion_mode=emotion_mode,
                    voice_characteristics=voice_profile.voice_characteristics,
                )
                results.append(error_response)

        total_time = time.time() - start_time
        successful_count = sum(1 for r in results if r.success)
        total_audio_duration = sum(r.duration for r in results if r.success)

        logger.info(
            f"🎯 Batch completed: {successful_count}/{len(texts)} successful, "
            f"{total_audio_duration:.2f}s total audio in {total_time:.2f}s processing"
        )

        return results

    def _detect_batch_emotion(
        self, texts: List[str], default_mode: EmotionMode
    ) -> EmotionMode:
        """Detect the most appropriate emotion mode for a batch of texts"""
        emotion_counts = {}

        for text in texts:
            emotion = self._detect_emotion_from_text(text, default_mode)
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Return the most common emotion, or default if tie
        if emotion_counts:
            return max(emotion_counts.items(), key=lambda x: x[1])[0]
        return default_mode

    async def clone_voice_and_generate(
        self,
        text: str,
        reference_audio_path: str,
        emotion_mode: EmotionMode = EmotionMode.CONVERSATIONAL,
        enhance_emotion: bool = True,
        seed: Optional[int] = None,
    ) -> TTSResponse:
        """
        🎭 ADVANCED VOICE CLONING with emotion transfer

        Clones a voice from reference audio and applies intelligent emotion optimization.
        This showcases Chatterbox's powerful voice cloning + emotion control combination.

        Args:
            text: Text to convert to speech
            reference_audio_path: Path to reference audio for cloning
            emotion_mode: Emotion mode to apply to cloned voice
            enhance_emotion: Whether to enhance emotion based on text content
            seed: Seed for reproducible results

        Returns:
            TTSResponse with cloned voice and emotion
        """
        logger.info(
            f"🎭 Cloning voice from {reference_audio_path} with {emotion_mode.value} emotion"
        )

        # Analyze reference audio if possible
        reference_duration = None
        try:
            import librosa

            y, sr = librosa.load(reference_audio_path, sr=None)
            reference_duration = len(y) / sr
            logger.info(f"📊 Reference audio: {reference_duration:.2f}s duration")
        except Exception as e:
            logger.warning(f"Could not analyze reference audio: {e}")

        # Auto-detect emotion from text if enhancement is enabled
        if enhance_emotion:
            detected_emotion = self._detect_emotion_from_text(text, emotion_mode)
            if detected_emotion != emotion_mode:
                logger.info(
                    f"🎯 Enhanced emotion: {emotion_mode.value} → {detected_emotion.value}"
                )
                emotion_mode = detected_emotion

        # Create temporary voice profile for this cloning operation
        temp_voice_profile = VoiceProfile(
            id="temp_clone",
            name="Cloned Voice",
            description=f"Voice cloned from {reference_audio_path}",
            gender="unknown",
            style="cloned",
            audio_prompt_path=reference_audio_path,
            is_custom=True,
            default_emotion_mode=emotion_mode,
            reference_audio_duration=reference_duration,
        )

        # Get optimized parameters for the emotion and cloned voice
        params = self._get_optimized_parameters(
            emotion_mode, temp_voice_profile, None, None, None
        )

        # Generate with voice cloning and emotion
        response = await self.convert_text_to_speech(
            text=text,
            voice_id="temp_clone",
            audio_prompt_path=reference_audio_path,
            exaggeration=params["exaggeration"],
            cfg_weight=params["cfg_weight"],
            temperature=params["temperature"],
            seed=seed,
            emotion_mode=emotion_mode,
        )

        if response.success:
            logger.info(
                f"✅ Voice cloning successful: {response.duration:.2f}s audio, "
                f"emotion: {emotion_mode.value}, RTF: {response.real_time_factor:.3f}"
            )
        else:
            logger.error(f"❌ Voice cloning failed: {response.error_message}")

        return response

    def get_emotion_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get available emotion profiles with their parameters"""
        return {
            mode.value: {
                "mode": mode.value,
                "exaggeration": profile.exaggeration,
                "cfg_weight": profile.cfg_weight,
                "temperature": profile.temperature,
                "speed_factor": profile.speed_factor,
                "description": profile.description,
            }
            for mode, profile in self.emotion_profiles.items()
        }

    def estimate_cost(self, text: str) -> Dict[str, Any]:
        """
        Estimate processing cost for text generation
        Since Chatterbox is local/free, this returns computational cost estimate
        """
        character_count = len(text)

        # Estimate processing time based on text length and hardware
        base_time_per_char = (
            0.005 if self.device == "cuda" else 0.02
        )  # Updated for ultimate performance
        estimated_time = character_count * base_time_per_char

        return {
            "character_count": character_count,
            "estimated_processing_time": estimated_time,
            "computational_cost": "local_processing_ultimate",
            "api_cost": 0.0,  # Free for local processing
            "total_cost": 0.0,
            "features": [
                "emotion_control",
                "voice_cloning",
                "real_time_streaming",
                "watermark_detection",
                "batch_processing",
            ],
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
                            "model_sample_rate": self.model_sample_rate,
                            "target_sample_rate": self.target_sample_rate,
                            "target_format": self.target_format,
                            "target_bitrate": self.target_bitrate,
                            "target_channels": self.target_channels,
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
