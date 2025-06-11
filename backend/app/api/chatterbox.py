"""
ULTIMATE Chatterbox API Endpoints - Leveraging ALL Advanced Features

This module provides REST API endpoints for the ULTIMATE Chatterbox TTS implementation,
including emotion control, real-time streaming, voice cloning, batch processing,
and all advanced Chatterbox capabilities.

🎯 ULTIMATE FEATURES:
- Emotion/Exaggeration Control (Chatterbox's unique capability)
- Real-time Streaming with generate_stream()
- Voice Cloning with Emotion Transfer
- Batch Processing with Consistent Seeds
- Watermark Detection for Responsible AI
- Advanced Parameter Control (cfg_weight, temperature, seed)
- Intelligent Voice-aware Optimization
"""

import logging
from typing import Dict, Any, List, Optional
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..services.chatterbox_service import (
    chatterbox_service,
    TTSResponse,
    EmotionMode,
    VoiceCharacteristic,
    ChatterboxStreamMetrics,
)
from ..models.user import User
from ..models.voice_profile import VoiceProfile
from ..core.auth import get_current_user
from ..core.database import get_db
from ..services.storage_service import storage_service
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Enhanced Request Models with ALL Chatterbox features
class UltimateTTSRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=10000, description="Text to convert to speech"
    )
    voice_id: str = Field(default="bjorn", description="Voice profile ID")
    audio_prompt_path: Optional[str] = Field(
        default=None, description="Path to custom voice prompt audio for cloning"
    )

    # 🎯 ULTIMATE CHATTERBOX PARAMETERS
    exaggeration: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Emotion intensity (0.0-2.0) - Chatterbox's unique feature!",
    )
    cfg_weight: Optional[float] = Field(
        default=None, ge=0.1, le=1.0, description="Generation guidance/pacing (0.1-1.0)"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.1, le=2.0, description="Randomness control (0.1-2.0)"
    )
    seed: Optional[int] = Field(
        default=None, ge=-1, description="Seed for reproducible results (-1 for random)"
    )
    emotion_mode: Optional[str] = Field(
        default=None,
        description="Emotion mode: neutral, conversational, expressive, dramatic, intense",
    )
    enable_watermark_detection: bool = Field(
        default=True, description="Enable watermark detection for responsible AI"
    )

    # Legacy parameters for backward compatibility
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier (legacy)"
    )
    stability: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Voice stability (legacy)"
    )
    similarity_boost: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Voice similarity boost (legacy)"
    )
    style: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Voice style strength (legacy)"
    )


class RealTimeStreamRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text for real-time streaming"
    )
    voice_id: str = Field(default="bjorn", description="Voice profile ID")
    audio_prompt_path: Optional[str] = Field(
        default=None, description="Path to voice cloning reference"
    )

    # 🚀 STREAMING PARAMETERS
    chunk_size: int = Field(
        default=25,
        ge=10,
        le=100,
        description="Characters per chunk (smaller = lower latency)",
    )
    exaggeration: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Emotion intensity"
    )
    cfg_weight: Optional[float] = Field(
        default=None, ge=0.1, le=1.0, description="Generation guidance"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.1, le=2.0, description="Randomness control"
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for reproducible results"
    )
    emotion_mode: Optional[str] = Field(default=None, description="Emotion mode")


class VoiceCloningRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to convert")
    reference_audio_path: str = Field(
        ..., description="Path to reference audio for voice cloning"
    )
    emotion_mode: str = Field(
        default="conversational", description="Emotion mode for cloned voice"
    )
    enhance_emotion: bool = Field(
        default=True, description="Auto-enhance emotion based on text content"
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for reproducible results"
    )


class BatchTTSRequest(BaseModel):
    texts: List[str] = Field(
        ..., min_items=1, max_items=50, description="List of texts to process"
    )
    voice_id: str = Field(default="bjorn", description="Voice profile for all texts")
    audio_prompt_path: Optional[str] = Field(
        default=None, description="Voice cloning reference for all texts"
    )

    # 🎯 BATCH PARAMETERS
    exaggeration: Optional[float] = Field(
        default=None, description="Emotion intensity for all"
    )
    cfg_weight: Optional[float] = Field(default=None, description="CFG weight for all")
    temperature: Optional[float] = Field(
        default=None, description="Temperature for all"
    )
    base_seed: Optional[int] = Field(
        default=None, description="Base seed (incremented per text)"
    )
    emotion_mode: Optional[str] = Field(
        default=None, description="Emotion mode for all"
    )
    consistent_voice: bool = Field(
        default=True, description="Use consistent parameters across all texts"
    )


class PodcastSegmentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text content for the segment")
    speaker_id: str = Field(
        ..., description="Speaker ID (host1, host2, expert, interviewer)"
    )
    segment_type: str = Field(default="dialogue", description="Type of segment")
    emotion_mode: Optional[str] = Field(
        default=None, description="Emotion mode override"
    )
    voice_settings: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom voice settings"
    )


# Enhanced Response Models
class UltimateTTSResponse(BaseModel):
    success: bool
    audio_url: Optional[str] = None
    duration: float
    voice_id: str
    processing_time: Optional[float] = None
    cost_estimate: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # 🎯 ULTIMATE CHATTERBOX FIELDS
    emotion_mode: Optional[str] = None
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    real_time_factor: Optional[float] = None
    watermark_detected: Optional[float] = None
    voice_characteristics: Optional[str] = None


class StreamMetricsResponse(BaseModel):
    chunk_count: int
    rtf: float
    latency_to_first_chunk: Optional[float] = None
    total_audio_duration: float
    processing_time: float


class EmotionProfilesResponse(BaseModel):
    emotion_profiles: Dict[str, Dict[str, Any]]


class VoiceCloningResponse(BaseModel):
    success: bool
    audio_url: Optional[str] = None
    duration: float
    cloned_voice_characteristics: Optional[Dict[str, Any]] = None
    emotion_applied: str
    processing_time: float
    error_message: Optional[str] = None


class BatchTTSResponse(BaseModel):
    success: bool
    total_items: int
    successful_items: int
    total_duration: float
    total_processing_time: float
    results: List[Dict[str, Any]]


# Router Setup
router = APIRouter(prefix="/api/chatterbox", tags=["ultimate-chatterbox"])


@router.get("/health")
async def check_health():
    """Check the health status of the ULTIMATE Chatterbox service"""
    try:
        health_status = await chatterbox_service.health_check()
        return {
            "status": health_status["status"],
            "service": "ultimate-chatterbox",
            "timestamp": health_status.get("timestamp"),
            "ultimate_features": [
                "emotion_control",
                "real_time_streaming",
                "voice_cloning",
                "batch_processing",
                "watermark_detection",
                "advanced_parameters",
            ],
            "details": health_status.get("details", {}),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/test")
async def test_connection():
    """Test the ULTIMATE Chatterbox TTS connection and features"""
    try:
        test_result = await chatterbox_service.test_connection()
        if test_result["status"] == "success":
            return {
                "status": "success",
                "message": "ULTIMATE Chatterbox TTS connection successful",
                "ultimate_features_available": True,
                "emotion_modes": [
                    "neutral",
                    "conversational",
                    "expressive",
                    "dramatic",
                    "intense",
                ],
                "advanced_parameters": [
                    "exaggeration",
                    "cfg_weight",
                    "temperature",
                    "seed",
                ],
                "details": test_result,
            }
        else:
            raise HTTPException(
                status_code=503,
                detail=f"ULTIMATE Chatterbox TTS test failed: {test_result.get('message', 'Unknown error')}",
            )
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")


@router.get("/emotion-profiles", response_model=EmotionProfilesResponse)
async def get_emotion_profiles():
    """Get all available emotion profiles with their parameters"""
    try:
        emotion_profiles = chatterbox_service.get_emotion_profiles()
        return EmotionProfilesResponse(emotion_profiles=emotion_profiles)
    except Exception as e:
        logger.error(f"Failed to get emotion profiles: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get emotion profiles: {str(e)}"
        )


@router.get("/voices")
async def get_available_voices():
    """Get all available voices with enhanced Chatterbox optimizations"""
    try:
        voices = chatterbox_service.get_available_voices()
        return {
            "voices": voices,
            "voice_optimization_features": [
                "emotion_aware_parameters",
                "speed_characteristic_optimization",
                "voice_cloning_support",
                "consistent_batch_processing",
            ],
        }
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")


@router.post("/ultimate-generate", response_model=UltimateTTSResponse)
async def ultimate_text_to_speech(request: UltimateTTSRequest):
    """
    🎯 ULTIMATE text-to-speech conversion using ALL Chatterbox capabilities

    Features:
    - Emotion/Exaggeration Control (unique to Chatterbox)
    - Intelligent Voice-aware Parameter Optimization
    - Advanced CFG Weight, Temperature, and Seed Control
    - Watermark Detection for Responsible AI
    - Real-time Performance Metrics
    """
    try:
        # Convert emotion mode string to enum
        emotion_mode = None
        if request.emotion_mode:
            try:
                emotion_mode = EmotionMode(request.emotion_mode.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid emotion mode: {request.emotion_mode}. "
                    "Valid modes: neutral, conversational, expressive, dramatic, intense",
                )

        # Get cost estimate
        cost_estimate = chatterbox_service.estimate_cost(request.text)

        # Generate audio with ULTIMATE features
        tts_response = await chatterbox_service.convert_text_to_speech(
            text=request.text,
            voice_id=request.voice_id,
            audio_prompt_path=request.audio_prompt_path,
            speed=request.speed,
            stability=request.stability,
            similarity_boost=request.similarity_boost,
            style=request.style,
            # ULTIMATE parameters
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            seed=request.seed,
            emotion_mode=emotion_mode,
            enable_watermark_detection=request.enable_watermark_detection,
        )

        if not tts_response.success:
            raise HTTPException(
                status_code=400,
                detail=f"ULTIMATE TTS generation failed: {tts_response.error_message}",
            )

        # TODO: Save audio file and return URL - implement proper file storage
        return UltimateTTSResponse(
            success=True,
            audio_url=None,  # Would contain actual audio URL
            duration=tts_response.duration,
            voice_id=tts_response.voice_id,
            processing_time=tts_response.processing_time,
            cost_estimate=cost_estimate,
            # ULTIMATE fields
            emotion_mode=tts_response.emotion_mode.value
            if tts_response.emotion_mode
            else None,
            exaggeration=tts_response.exaggeration,
            cfg_weight=tts_response.cfg_weight,
            temperature=tts_response.temperature,
            seed=tts_response.seed,
            real_time_factor=tts_response.real_time_factor,
            watermark_detected=tts_response.watermark_detected,
            voice_characteristics=tts_response.voice_characteristics.value
            if tts_response.voice_characteristics
            else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ULTIMATE TTS generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"ULTIMATE TTS generation failed: {str(e)}"
        )


@router.post("/real-time-stream")
async def real_time_streaming_tts(
    request: RealTimeStreamRequest, current_user: User = Depends(get_current_user)
):
    """
    🚀 REAL-TIME STREAMING TTS using Chatterbox's generate_stream()

    Streams audio chunks as they're generated for sub-200ms latency.
    This is Chatterbox's killer feature for real-time applications!
    """
    try:
        # Convert emotion mode
        emotion_mode = None
        if request.emotion_mode:
            try:
                emotion_mode = EmotionMode(request.emotion_mode.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid emotion mode: {request.emotion_mode}",
                )

        async def stream_generator():
            try:
                async for (
                    audio_chunk,
                    metrics,
                ) in chatterbox_service.generate_real_time_stream(
                    text=request.text,
                    voice_id=request.voice_id,
                    audio_prompt_path=request.audio_prompt_path,
                    exaggeration=request.exaggeration,
                    cfg_weight=request.cfg_weight,
                    temperature=request.temperature,
                    seed=request.seed,
                    emotion_mode=emotion_mode,
                    chunk_size=request.chunk_size,
                ):
                    yield audio_chunk
            except Exception as e:
                logger.error(f"Streaming failed: {e}")
                raise

        return StreamingResponse(
            stream_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=realtime_tts_{request.voice_id}.wav",
                "X-Chatterbox-Feature": "real-time-streaming",
                "X-Emotion-Mode": request.emotion_mode or "auto-detected",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Real-time streaming failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Real-time streaming failed: {str(e)}"
        )


@router.post("/voice-cloning", response_model=VoiceCloningResponse)
async def advanced_voice_cloning(
    request: VoiceCloningRequest, current_user: User = Depends(get_current_user)
):
    """
    🎭 ADVANCED VOICE CLONING with emotion transfer

    Clones a voice from reference audio and applies intelligent emotion optimization.
    This showcases Chatterbox's powerful voice cloning + emotion control combination.
    """
    try:
        # Convert emotion mode
        try:
            emotion_mode = EmotionMode(request.emotion_mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid emotion mode: {request.emotion_mode}"
            )

        # Generate with voice cloning and emotion
        response = await chatterbox_service.clone_voice_and_generate(
            text=request.text,
            reference_audio_path=request.reference_audio_path,
            emotion_mode=emotion_mode,
            enhance_emotion=request.enhance_emotion,
            seed=request.seed,
        )

        if not response.success:
            raise HTTPException(
                status_code=400,
                detail=f"Voice cloning failed: {response.error_message}",
            )

        return VoiceCloningResponse(
            success=True,
            audio_url=None,  # Would contain actual audio URL
            duration=response.duration,
            cloned_voice_characteristics={
                "emotion_mode": response.emotion_mode.value
                if response.emotion_mode
                else None,
                "voice_characteristics": response.voice_characteristics.value
                if response.voice_characteristics
                else None,
                "exaggeration": response.exaggeration,
                "cfg_weight": response.cfg_weight,
            },
            emotion_applied=response.emotion_mode.value
            if response.emotion_mode
            else "unknown",
            processing_time=response.processing_time or 0.0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed with unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")


@router.post("/batch-generate", response_model=BatchTTSResponse)
async def batch_tts_generation(
    request: BatchTTSRequest, current_user: User = Depends(get_current_user)
):
    """
    🚀 BATCH PROCESSING with consistent voice and emotion across all texts

    Perfect for generating podcast episodes, audiobooks, or multiple segments
    with consistent quality and voice characteristics.
    """
    try:
        # Convert emotion mode
        emotion_mode = None
        if request.emotion_mode:
            try:
                emotion_mode = EmotionMode(request.emotion_mode.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid emotion mode: {request.emotion_mode}",
                )

        # Generate batch
        results = await chatterbox_service.generate_batch_tts(
            texts=request.texts,
            voice_id=request.voice_id,
            audio_prompt_path=request.audio_prompt_path,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            base_seed=request.base_seed,
            emotion_mode=emotion_mode,
            consistent_voice=request.consistent_voice,
        )

        # Process results
        successful_results = [r for r in results if r.success]
        total_duration = sum(r.duration for r in successful_results)
        total_processing_time = sum(r.processing_time or 0 for r in results)

        result_summaries = []
        for i, result in enumerate(results):
            result_summaries.append(
                {
                    "index": i,
                    "text_preview": request.texts[i][:100] + "..."
                    if len(request.texts[i]) > 100
                    else request.texts[i],
                    "success": result.success,
                    "duration": result.duration,
                    "processing_time": result.processing_time,
                    "emotion_mode": result.emotion_mode.value
                    if result.emotion_mode
                    else None,
                    "exaggeration": result.exaggeration,
                    "cfg_weight": result.cfg_weight,
                    "real_time_factor": result.real_time_factor,
                    "error_message": result.error_message,
                }
            )

        return BatchTTSResponse(
            success=len(successful_results) > 0,
            total_items=len(request.texts),
            successful_items=len(successful_results),
            total_duration=total_duration,
            total_processing_time=total_processing_time,
            results=result_summaries,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch generation failed: {str(e)}"
        )


# Legacy endpoints for backward compatibility
@router.post("/generate", response_model=UltimateTTSResponse)
async def convert_text_to_speech(
    request: UltimateTTSRequest, current_user: User = Depends(get_current_user)
):
    """Legacy endpoint - redirects to ultimate-generate"""
    return await ultimate_text_to_speech(request, current_user)


@router.post("/generate-stream")
async def convert_text_to_speech_stream(
    request: RealTimeStreamRequest, current_user: User = Depends(get_current_user)
):
    """Legacy streaming endpoint - redirects to real-time-stream"""
    return await real_time_streaming_tts(request, current_user)


@router.get("/config")
async def get_service_config():
    """Get ULTIMATE Chatterbox service configuration"""
    try:
        test_result = await chatterbox_service.test_connection()
        return {
            "service": "ultimate-chatterbox",
            "version": "ultimate-1.0",
            "features": {
                "emotion_control": True,
                "real_time_streaming": True,
                "voice_cloning": True,
                "batch_processing": True,
                "watermark_detection": True,
                "advanced_parameters": True,
            },
            "emotion_modes": [
                "neutral",
                "conversational",
                "expressive",
                "dramatic",
                "intense",
            ],
            "max_text_length": 10000,
            "max_batch_size": 50,
            "streaming_chunk_sizes": [10, 15, 25, 50, 100],
            "status": test_result.get("status", "unknown"),
            "gpu_acceleration": test_result.get("cuda_available", False),
        }
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


@router.post("/estimate-cost")
async def estimate_cost(request: Dict[str, str]):
    """Estimate computational cost for ULTIMATE TTS generation"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        cost_estimate = chatterbox_service.estimate_cost(text)
        return cost_estimate
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")


@router.post("/clone-voice")
async def clone_voice_endpoint(
    audio: UploadFile = File(...),
    voice_name: str = Form(...),
    description: str = Form(""),
    current_user: User = Depends(get_current_user),
):
    """
    🎭 Clone a voice from uploaded audio sample

    Creates a custom voice profile from an audio file upload with user isolation.
    """
    try:
        from ..services.storage_service import storage_service
        from ..models.voice_profile import VoiceProfile
        from ..core.database import get_db
        from sqlalchemy.orm import Session
        import time
        import hashlib

        logger.info(f"Voice cloning request from user {current_user.id}: {voice_name}")

        # Check if user already has a voice with this name
        db: Session = next(get_db())
        existing_voice = (
            db.query(VoiceProfile)
            .filter(
                VoiceProfile.user_id == current_user.id,
                VoiceProfile.name == voice_name.strip(),
            )
            .first()
        )

        if existing_voice:
            raise HTTPException(
                status_code=400,
                detail=f"You already have a voice named '{voice_name}'. Please choose a different name.",
            )

        # Validate audio file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Validate file type
        if not audio.content_type or not audio.content_type.startswith("audio/"):
            logger.error(f"Invalid content type: {audio.content_type}")
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file.",
            )

        # Validate voice name
        if not voice_name or len(voice_name.strip()) < 1:
            raise HTTPException(status_code=400, detail="Voice name is required")

        # Read and validate file size (10MB limit)
        content = await audio.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400, detail="File too large. Maximum size is 10MB."
            )

        if len(content) < 1000:  # At least 1KB
            raise HTTPException(
                status_code=400,
                detail="File too small. Please upload a proper audio file.",
            )

        logger.info(
            f"Processing audio file: {audio.filename}, size: {len(content)} bytes"
        )

        # Create temporary file for the audio
        import tempfile
        import os

        # Get file extension safely
        file_extension = "wav"  # default
        if audio.filename and "." in audio.filename:
            file_extension = audio.filename.split(".")[-1].lower()

        # Validate extension
        allowed_extensions = ["mp3", "wav", "m4a", "aac", "ogg", "flac"]
        if file_extension not in allowed_extensions:
            file_extension = "wav"  # fallback

        # Save the original voice sample to user-specific storage
        voice_sample_path = await storage_service.save_audio_file(
            content,
            f"voice_clone_{current_user.id}",
            segment_id=f"voice_sample_{int(time.time())}",
            file_type=file_extension,
            metadata={
                "user_id": current_user.id,
                "voice_name": voice_name,
                "description": description,
                "original_filename": audio.filename,
                "content_type": audio.content_type,
                "purpose": "voice_cloning_sample",
            },
        )

        logger.info(f"Saved voice sample to: {voice_sample_path}")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file_extension}", prefix="voice_clone_"
        ) as temp_file:
            temp_file.write(content)
            temp_audio_path = temp_file.name

        logger.info(f"Temporary file created: {temp_audio_path}")

        try:
            # Generate unique voice ID with user isolation
            voice_id = f"custom_{current_user.id}_{int(time.time())}_{hashlib.md5(voice_name.encode()).hexdigest()[:8]}"

            logger.info(f"Generated voice ID: {voice_id}")

            # Add custom voice to chatterbox service
            success = chatterbox_service.add_custom_voice(
                voice_id=voice_id,
                name=voice_name.strip(),
                description=description.strip() or f"Custom cloned voice: {voice_name}",
                audio_prompt_path=temp_audio_path,
                gender="unknown",
                style="custom",
            )

            if not success:
                logger.error(f"Chatterbox service failed to add custom voice")
                raise HTTPException(
                    status_code=500, detail="Failed to clone voice. Please try again."
                )

            logger.info(
                f"Successfully cloned voice '{voice_name}' for user {current_user.id}"
            )

            # Generate a test sample using the cloned voice
            logger.info(f"Generating test sample with cloned voice: {voice_id}")

            test_response = await chatterbox_service.convert_text_to_speech(
                text=f"Hello! This is a test of your cloned voice, {voice_name}. I'm speaking with natural intonation and varied sentence structures to demonstrate the quality and versatility of this voice clone. Notice how I can express different emotions, from excitement to calm professionalism. The technology captures not just the tone, but the unique characteristics that make your voice distinctly yours. How authentic does this sound to you?",
                voice_id="bjorn",  # Use base voice for now
                audio_prompt_path=temp_audio_path,  # Use cloned voice as prompt
                # Optimized parameters for better voice similarity
                speed=1.0,  # Keep natural speed
                stability=0.7,  # Higher stability for consistency
                similarity_boost=0.95,  # Much higher similarity to source
                style=0.2,  # Lower style for more authentic reproduction
                exaggeration=0.3,  # Lower exaggeration for more natural sound
                cfg_weight=0.7,  # Higher CFG weight for better prompt following
                temperature=0.5,  # Lower temperature for more consistent results
                emotion_mode=EmotionMode.NEUTRAL,  # Neutral emotion to match source better
            )

            test_audio_url = None
            test_sample_path = None
            if test_response.success and test_response.audio_data:
                # Save the test sample to user-specific storage
                test_sample_path = await storage_service.save_audio_file(
                    test_response.audio_data,
                    f"voice_clone_{current_user.id}",
                    segment_id=f"test_sample_{int(time.time())}",
                    file_type="mp3",
                    metadata={
                        "user_id": current_user.id,
                        "voice_id": voice_id,
                        "voice_name": voice_name,
                        "purpose": "voice_cloning_test_sample",
                        "text": f"Hello! This is a test of your cloned voice, {voice_name}...",
                    },
                )

                # Get the URL for the test sample
                test_audio_url = await storage_service.get_file_url(test_sample_path)
                logger.info(f"Test sample saved and accessible at: {test_audio_url}")
            else:
                logger.warning(
                    f"Test sample generation failed: {test_response.error_message}"
                )

            # Save voice profile to database with user isolation
            voice_profile = VoiceProfile(
                user_id=current_user.id,
                voice_id=voice_id,
                name=voice_name.strip(),
                description=description.strip() or f"Custom cloned voice: {voice_name}",
                original_audio_path=voice_sample_path,
                test_audio_path=test_sample_path,
                gender="unknown",
                style="custom",
                is_active=True,
                # Store optimized parameters
                optimal_similarity_boost=0.95,
                optimal_stability=0.7,
                optimal_style=0.2,
                optimal_exaggeration=0.3,
                optimal_cfg_weight=0.7,
                optimal_temperature=0.5,
                # Metadata
                file_size=len(content),
                original_filename=audio.filename,
                content_type=audio.content_type,
            )

            db.add(voice_profile)
            db.commit()
            db.refresh(voice_profile)

            logger.info(f"Voice profile saved to database with ID: {voice_profile.id}")

            return {
                "success": True,
                "voice_id": voice_id,
                "voice_name": voice_name,
                "description": description,
                "message": "Voice cloned successfully!",
                "voice_sample_url": await storage_service.get_file_url(
                    voice_sample_path
                ),
                "test_audio_url": test_audio_url,
                "file_info": {
                    "filename": audio.filename,
                    "content_type": audio.content_type,
                    "size": len(content),
                },
                "voice_profile_id": voice_profile.id,
            }

        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    logger.info(f"Cleaned up temporary file: {temp_audio_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed with unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")


@router.post("/clone-voice-no-auth")
async def clone_voice_no_auth_endpoint(
    audio: UploadFile = File(...),
    voice_name: str = Form(...),
    description: str = Form(""),
):
    """
    🎭 Clone a voice from uploaded audio sample (No authentication required - TEMPORARY)

    Creates a custom voice profile from an audio file upload without authentication.
    This is a temporary endpoint to preserve recordings during auth issues.
    """
    try:
        from ..services.storage_service import storage_service
        import time
        import hashlib

        # Use temporary user ID for testing
        temp_user_id = 999

        logger.info(f"Voice cloning request (no-auth) for voice: {voice_name}")

        # Validate audio file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Validate file type
        if not audio.content_type or not audio.content_type.startswith("audio/"):
            logger.error(f"Invalid content type: {audio.content_type}")
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file.",
            )

        # Validate voice name
        if not voice_name or len(voice_name.strip()) < 1:
            raise HTTPException(status_code=400, detail="Voice name is required")

        # Read and validate file size (10MB limit)
        content = await audio.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400, detail="File too large. Maximum size is 10MB."
            )

        if len(content) < 1000:  # At least 1KB
            raise HTTPException(
                status_code=400,
                detail="File too small. Please upload a proper audio file.",
            )

        logger.info(
            f"Processing audio file: {audio.filename}, size: {len(content)} bytes"
        )

        # Create temporary file for the audio
        import tempfile
        import os

        # Get file extension safely
        file_extension = "wav"  # default
        if audio.filename and "." in audio.filename:
            file_extension = audio.filename.split(".")[-1].lower()

        # Validate extension
        allowed_extensions = ["mp3", "wav", "m4a", "aac", "ogg", "flac"]
        if file_extension not in allowed_extensions:
            file_extension = "wav"  # fallback

        # Save the original voice sample to storage for the voice profile
        voice_sample_path = await storage_service.save_audio_file(
            content,
            f"voice_clone_{temp_user_id}",
            segment_id=f"voice_sample_{int(time.time())}",
            file_type=file_extension,
            metadata={
                "user_id": temp_user_id,
                "voice_name": voice_name,
                "description": description,
                "original_filename": audio.filename,
                "content_type": audio.content_type,
                "purpose": "voice_cloning_sample_no_auth",
            },
        )

        logger.info(f"Saved voice sample to: {voice_sample_path}")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file_extension}", prefix="voice_clone_"
        ) as temp_file:
            temp_file.write(content)
            temp_audio_path = temp_file.name

        logger.info(f"Temporary file created: {temp_audio_path}")

        try:
            # Generate unique voice ID
            voice_id = f"custom_{temp_user_id}_{int(time.time())}_{hashlib.md5(voice_name.encode()).hexdigest()[:8]}"

            logger.info(f"Generated voice ID: {voice_id}")

            # Add custom voice to chatterbox service
            success = chatterbox_service.add_custom_voice(
                voice_id=voice_id,
                name=voice_name.strip(),
                description=description.strip() or f"Custom cloned voice: {voice_name}",
                audio_prompt_path=temp_audio_path,
                gender="unknown",
                style="custom",
            )

            if not success:
                logger.error(f"Chatterbox service failed to add custom voice")
                raise HTTPException(
                    status_code=500, detail="Failed to clone voice. Please try again."
                )

            logger.info(f"Successfully cloned voice '{voice_name}' (no-auth)")

            # Generate a test sample using the cloned voice
            logger.info(f"Generating test sample with cloned voice: {voice_id}")

            test_response = await chatterbox_service.convert_text_to_speech(
                text=f"Hello! This is a test of your cloned voice, {voice_name}. I'm speaking with natural intonation and varied sentence structures to demonstrate the quality and versatility of this voice clone. Notice how I can express different emotions, from excitement to calm professionalism. The technology captures not just the tone, but the unique characteristics that make your voice distinctly yours. How authentic does this sound to you?",
                voice_id="bjorn",  # Use base voice for now
                audio_prompt_path=temp_audio_path,  # Use cloned voice as prompt
                # Optimized parameters for better voice similarity
                speed=1.0,  # Keep natural speed
                stability=0.7,  # Higher stability for consistency
                similarity_boost=0.95,  # Much higher similarity to source
                style=0.2,  # Lower style for more authentic reproduction
                exaggeration=0.3,  # Lower exaggeration for more natural sound
                cfg_weight=0.7,  # Higher CFG weight for better prompt following
                temperature=0.5,  # Lower temperature for more consistent results
                emotion_mode=EmotionMode.NEUTRAL,  # Neutral emotion to match source better
            )

            test_audio_url = None
            if test_response.success and test_response.audio_data:
                # Save the test sample to storage
                test_sample_path = await storage_service.save_audio_file(
                    test_response.audio_data,
                    f"voice_clone_{temp_user_id}",
                    segment_id=f"test_sample_{int(time.time())}",
                    file_type="mp3",
                    metadata={
                        "user_id": temp_user_id,
                        "voice_id": voice_id,
                        "voice_name": voice_name,
                        "purpose": "voice_cloning_test_sample_no_auth",
                        "text": f"Hello! This is a test of your cloned voice, {voice_name}...",
                    },
                )

                # Get the URL for the test sample
                test_audio_url = await storage_service.get_file_url(test_sample_path)
                logger.info(f"Test sample saved and accessible at: {test_audio_url}")
            else:
                logger.warning(
                    f"Test sample generation failed: {test_response.error_message}"
                )

            return {
                "success": True,
                "voice_id": voice_id,
                "voice_name": voice_name,
                "description": description,
                "message": "Voice cloned successfully! (No authentication)",
                "voice_sample_url": await storage_service.get_file_url(
                    voice_sample_path
                ),
                "test_audio_url": test_audio_url,
                "file_info": {
                    "filename": audio.filename,
                    "content_type": audio.content_type,
                    "size": len(content),
                },
            }

        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    logger.info(f"Cleaned up temporary file: {temp_audio_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Voice cloning (no-auth) failed with unexpected error: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")


@router.get("/my-voices")
async def get_user_voice_profiles(
    current_user: User = Depends(get_current_user),
):
    """
    Get all voice profiles for the current user
    """
    try:
        from ..models.voice_profile import VoiceProfile
        from ..core.database import get_db
        from sqlalchemy.orm import Session

        db: Session = next(get_db())

        voice_profiles = (
            db.query(VoiceProfile)
            .filter(
                VoiceProfile.user_id == current_user.id, VoiceProfile.is_active == True
            )
            .order_by(VoiceProfile.created_at.desc())
            .all()
        )

        result = []
        for profile in voice_profiles:
            # Get file URLs for the voice samples
            voice_sample_url = None
            test_audio_url = None

            if profile.original_audio_path:
                voice_sample_url = await storage_service.get_file_url(
                    profile.original_audio_path
                )

            if profile.test_audio_path:
                test_audio_url = await storage_service.get_file_url(
                    profile.test_audio_path
                )

            result.append(
                {
                    "id": profile.id,
                    "voice_id": profile.voice_id,
                    "name": profile.name,
                    "description": profile.description,
                    "gender": profile.gender,
                    "style": profile.style,
                    "voice_sample_url": voice_sample_url,
                    "test_audio_url": test_audio_url,
                    "created_at": profile.created_at.isoformat(),
                    "file_info": {
                        "filename": profile.original_filename,
                        "content_type": profile.content_type,
                        "size": profile.file_size,
                        "duration": profile.duration,
                    },
                    "optimization_params": {
                        "similarity_boost": profile.optimal_similarity_boost,
                        "stability": profile.optimal_stability,
                        "style": profile.optimal_style,
                        "exaggeration": profile.optimal_exaggeration,
                        "cfg_weight": profile.optimal_cfg_weight,
                        "temperature": profile.optimal_temperature,
                    },
                }
            )

        return {
            "success": True,
            "voices": result,
            "total": len(result),
        }

    except Exception as e:
        logger.error(f"Failed to get user voice profiles: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get voice profiles: {str(e)}"
        )


@router.delete("/my-voices/{voice_profile_id}")
async def delete_user_voice_profile(
    voice_profile_id: int,
    current_user: User = Depends(get_current_user),
):
    """
    Delete a user's voice profile
    """
    try:
        from ..models.voice_profile import VoiceProfile
        from ..core.database import get_db
        from sqlalchemy.orm import Session

        db: Session = next(get_db())

        voice_profile = (
            db.query(VoiceProfile)
            .filter(
                VoiceProfile.id == voice_profile_id,
                VoiceProfile.user_id
                == current_user.id,  # Ensure user can only delete their own voices
            )
            .first()
        )

        if not voice_profile:
            raise HTTPException(
                status_code=404,
                detail="Voice profile not found or you don't have permission to delete it",
            )

        # Mark as inactive instead of deleting (soft delete)
        voice_profile.is_active = False
        db.commit()

        # Also remove from chatterbox service
        # Note: This depends on how the chatterbox service handles voice removal
        # For now, we'll just log it
        logger.info(
            f"Voice profile {voice_profile.voice_id} marked as inactive for user {current_user.id}"
        )

        return {
            "success": True,
            "message": f"Voice '{voice_profile.name}' deleted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete voice profile: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete voice profile: {str(e)}"
        )


@router.post("/create-system-voice")
async def create_system_voice_endpoint(
    voice_name: str = Form(...),
    description: str = Form(""),
    voice_type: str = Form(...),  # "cloned" or "synthetic"
    gender: str = Form("male"),  # "male", "female", "neutral"
    accent: str = Form(
        "american"
    ),  # "american", "british", "australian", "indian", "french", "german", etc.
    is_system_default: bool = Form(False),  # Whether this is a system default voice
    generate_variations: int = Form(1),  # Number of variations to generate (1-5)
    audio: UploadFile = File(None),  # Optional for synthetic voices
):
    """
    🎭 Create enhanced system voices with multiple variations

    Supports two modes:
    1. Voice Cloning: Clone from uploaded audio with multiple random seed variations
    2. Synthetic Generation: Create voices based on gender/accent characteristics

    Perfect for creating baseline system voices like Bjorn and Felix!
    """
    try:
        from ..services.storage_service import storage_service
        from ..models.voice_profile import VoiceProfile
        from ..core.database import get_db
        from sqlalchemy.orm import Session
        import time
        import hashlib
        import random
        import tempfile
        import os

        logger.info(
            f"Creating system voice '{voice_name}' - Type: {voice_type}, Gender: {gender}, Accent: {accent}"
        )

        # Validate parameters
        if voice_type not in ["cloned", "synthetic"]:
            raise HTTPException(
                status_code=400, detail="voice_type must be 'cloned' or 'synthetic'"
            )

        if gender not in ["male", "female", "neutral"]:
            raise HTTPException(
                status_code=400, detail="gender must be 'male', 'female', or 'neutral'"
            )

        if generate_variations < 1 or generate_variations > 5:
            raise HTTPException(
                status_code=400, detail="generate_variations must be between 1 and 5"
            )

        # Validate audio file for cloned voices
        if voice_type == "cloned":
            if not audio or not audio.filename:
                raise HTTPException(
                    status_code=400, detail="Audio file required for cloned voices"
                )

            if not audio.content_type or not audio.content_type.startswith("audio/"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Please upload an audio file.",
                )

        # Read audio content if provided
        temp_audio_path = None
        voice_sample_path = None
        content = None

        if voice_type == "cloned" and audio:
            content = await audio.read()

            if len(content) > 15 * 1024 * 1024:  # 15MB limit for system voices
                raise HTTPException(
                    status_code=400,
                    detail="File too large. Maximum size is 15MB for system voices.",
                )

            if len(content) < 1000:
                raise HTTPException(
                    status_code=400,
                    detail="File too small. Please upload a proper audio file.",
                )

            # Get file extension
            file_extension = "wav"
            if audio.filename and "." in audio.filename:
                file_extension = audio.filename.split(".")[-1].lower()

            allowed_extensions = ["mp3", "wav", "m4a", "aac", "ogg", "flac"]
            if file_extension not in allowed_extensions:
                file_extension = "wav"

            # Save original voice sample for system voices
            voice_sample_path = await storage_service.save_audio_file(
                content,
                "system_voices",
                segment_id=f"baseline_{voice_name}_{int(time.time())}",
                file_type=file_extension,
                metadata={
                    "voice_name": voice_name,
                    "voice_type": voice_type,
                    "gender": gender,
                    "accent": accent,
                    "is_system_default": is_system_default,
                    "purpose": "system_baseline_voice",
                    "original_filename": audio.filename,
                    "content_type": audio.content_type,
                },
            )

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_extension}", prefix="system_voice_"
            ) as temp_file:
                temp_file.write(content)
                temp_audio_path = temp_file.name

        db: Session = next(get_db())
        results = []

        # Generate multiple variations
        for variation_num in range(generate_variations):
            try:
                # Generate random seed for each variation
                variation_seed = random.randint(1000, 99999)

                # Create unique voice ID for each variation
                base_voice_id = (
                    f"system_{voice_name.lower().replace(' ', '_')}_{gender}_{accent}"
                )
                if generate_variations > 1:
                    voice_id = f"{base_voice_id}_v{variation_num + 1}_{variation_seed}"
                else:
                    voice_id = f"{base_voice_id}_{variation_seed}"

                logger.info(
                    f"Creating variation {variation_num + 1}/{generate_variations} with seed {variation_seed}"
                )

                # Enhanced test text for better voice evaluation
                test_texts = {
                    "analytical": f"Hello, I'm {voice_name}. Let me analyze this fascinating topic with you. When we examine the underlying principles, we discover remarkable insights that challenge conventional thinking. How do you think this applies to real-world scenarios?",
                    "conversational": f"Hey there! I'm {voice_name}, and I'm excited to chat with you today. You know what's really interesting? The way complex ideas can become simple when we break them down together. What questions do you have about this topic?",
                    "professional": f"Good day. I'm {voice_name}, and I'll be your guide through today's discussion. We'll explore key concepts systematically, examining both theoretical frameworks and practical applications. Are you ready to dive deep into this subject?",
                }

                # Select appropriate test text based on voice characteristics
                if "bjorn" in voice_name.lower():
                    test_text = test_texts["analytical"]
                elif "felix" in voice_name.lower():
                    test_text = test_texts["conversational"]
                else:
                    test_text = test_texts["professional"]

                # Enhanced parameters for system voice quality
                enhanced_params = {
                    "speed": 1.0,
                    "stability": 0.8,  # Higher stability for system voices
                    "similarity_boost": 0.98
                    if voice_type == "cloned"
                    else 0.7,  # Maximum similarity for cloned
                    "style": 0.15,  # Lower style for more natural reproduction
                    "exaggeration": 0.25,  # Subtle exaggeration for natural expression
                    "cfg_weight": 0.75,  # High CFG for better control
                    "temperature": 0.4,  # Lower temperature for consistency
                    "seed": variation_seed,  # Random seed for variation
                    "emotion_mode": EmotionMode.CONVERSATIONAL,  # Good baseline emotion
                }

                # Adjust parameters based on accent and gender
                accent_adjustments = {
                    "british": {"stability": 0.85, "style": 0.2, "exaggeration": 0.2},
                    "australian": {
                        "stability": 0.75,
                        "style": 0.25,
                        "exaggeration": 0.35,
                    },
                    "indian": {"stability": 0.8, "style": 0.3, "exaggeration": 0.4},
                    "french": {"stability": 0.7, "style": 0.35, "exaggeration": 0.45},
                    "german": {"stability": 0.85, "style": 0.2, "exaggeration": 0.25},
                    "american": {"stability": 0.8, "style": 0.15, "exaggeration": 0.25},
                }

                if accent in accent_adjustments:
                    enhanced_params.update(accent_adjustments[accent])

                # Generate test sample
                if voice_type == "cloned":
                    test_response = await chatterbox_service.convert_text_to_speech(
                        text=test_text,
                        voice_id="bjorn",  # Use base voice for cloning
                        audio_prompt_path=temp_audio_path,
                        **enhanced_params,
                    )
                else:
                    # For synthetic voices, use the best available voice as base
                    base_voice = "bjorn" if gender == "male" else "felix"
                    test_response = await chatterbox_service.convert_text_to_speech(
                        text=test_text, voice_id=base_voice, **enhanced_params
                    )

                # Save test sample and create voice profile
                test_audio_url = None
                test_sample_path = None

                if test_response.success and test_response.audio_data:
                    # Save test sample
                    test_sample_path = await storage_service.save_audio_file(
                        test_response.audio_data,
                        "system_voices",
                        segment_id=f"test_{voice_id}_{int(time.time())}",
                        file_type="mp3",
                        metadata={
                            "voice_id": voice_id,
                            "voice_name": voice_name,
                            "variation_number": variation_num + 1,
                            "seed": variation_seed,
                            "voice_type": voice_type,
                            "gender": gender,
                            "accent": accent,
                            "purpose": "system_voice_test_sample",
                            "test_text_preview": test_text[:100] + "...",
                        },
                    )

                    test_audio_url = await storage_service.get_file_url(
                        test_sample_path
                    )

                # Add to chatterbox service
                success = chatterbox_service.add_custom_voice(
                    voice_id=voice_id,
                    name=f"{voice_name} (V{variation_num + 1})"
                    if generate_variations > 1
                    else voice_name,
                    description=f"{description} | {voice_type.title()} {gender} voice with {accent} accent | Seed: {variation_seed}",
                    audio_prompt_path=temp_audio_path
                    if voice_type == "cloned"
                    else None,
                    gender=gender,
                    style="system_baseline",
                )

                if not success:
                    logger.error(
                        f"Failed to add voice {voice_id} to chatterbox service"
                    )
                    continue

                # Create database record (but don't tie to specific user for system voices)
                voice_profile = VoiceProfile(
                    user_id=None,  # System voices have no user (no auth required)
                    voice_id=voice_id,
                    name=f"{voice_name} (V{variation_num + 1})"
                    if generate_variations > 1
                    else voice_name,
                    description=f"{description} | {voice_type.title()} {gender} voice with {accent} accent",
                    original_audio_path=voice_sample_path,
                    test_audio_path=test_sample_path,
                    gender=gender,
                    style="system_baseline",
                    is_active=True,
                    # Store optimized parameters
                    optimal_similarity_boost=enhanced_params["similarity_boost"],
                    optimal_stability=enhanced_params["stability"],
                    optimal_style=enhanced_params["style"],
                    optimal_exaggeration=enhanced_params["exaggeration"],
                    optimal_cfg_weight=enhanced_params["cfg_weight"],
                    optimal_temperature=enhanced_params["temperature"],
                    # Metadata
                    file_size=len(content) if content else 0,
                    original_filename=audio.filename if audio else None,
                    content_type=audio.content_type if audio else "synthetic",
                    # Custom fields for system voices
                    accent=accent,
                    voice_type=voice_type,
                    generation_seed=variation_seed,
                    is_system_default=is_system_default,
                )

                db.add(voice_profile)
                db.commit()
                db.refresh(voice_profile)

                results.append(
                    {
                        "variation_number": variation_num + 1,
                        "voice_id": voice_id,
                        "seed": variation_seed,
                        "voice_profile_id": voice_profile.id,
                        "test_audio_url": test_audio_url,
                        "voice_sample_url": await storage_service.get_file_url(
                            voice_sample_path
                        )
                        if voice_sample_path
                        else None,
                        "parameters_used": enhanced_params,
                        "success": test_response.success
                        if "test_response" in locals()
                        else False,
                        "duration": test_response.duration
                        if "test_response" in locals() and test_response.success
                        else 0,
                        "processing_time": test_response.processing_time
                        if "test_response" in locals() and test_response.success
                        else 0,
                    }
                )

                logger.info(
                    f"Successfully created voice variation {variation_num + 1}: {voice_id}"
                )

            except Exception as variation_error:
                logger.error(
                    f"Failed to create variation {variation_num + 1}: {variation_error}"
                )
                results.append(
                    {
                        "variation_number": variation_num + 1,
                        "success": False,
                        "error": str(variation_error),
                    }
                )

        # Cleanup
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")

        successful_variations = [r for r in results if r.get("success", False)]

        return {
            "success": len(successful_variations) > 0,
            "voice_name": voice_name,
            "voice_type": voice_type,
            "gender": gender,
            "accent": accent,
            "is_system_default": is_system_default,
            "variations_requested": generate_variations,
            "variations_created": len(successful_variations),
            "variations": results,
            "message": f"Created {len(successful_variations)}/{generate_variations} voice variations for {voice_name}",
            "file_info": {
                "filename": audio.filename if audio else "synthetic",
                "content_type": audio.content_type if audio else "synthetic",
                "size": len(content) if content else 0,
            }
            if content
            else None,
            "next_steps": [
                "Test each variation by playing the test samples",
                "Select the best variation based on quality and character fit",
                "Mark the chosen variation as the primary system voice",
                "Optionally delete unused variations to keep the system clean",
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"System voice creation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"System voice creation failed: {str(e)}"
        )


@router.get("/system-voices")
async def get_system_voices():
    """
    Get all available system voices (both default and user-created system voices)
    """
    try:
        from ..models.voice_profile import VoiceProfile
        from ..core.database import get_db
        from sqlalchemy.orm import Session

        db: Session = next(get_db())

        # Get all system voices (voices with no user_id or marked as system default)
        system_voices = (
            db.query(VoiceProfile)
            .filter(
                (VoiceProfile.user_id == None)
                | (VoiceProfile.is_system_default == True),
                VoiceProfile.is_active == True,
            )
            .order_by(
                VoiceProfile.is_system_default.desc(), VoiceProfile.created_at.desc()
            )
            .all()
        )

        result = []
        for profile in system_voices:
            # Get file URLs for the voice samples
            voice_sample_url = None
            test_audio_url = None

            if profile.original_audio_path:
                voice_sample_url = await storage_service.get_file_url(
                    profile.original_audio_path
                )

            if profile.test_audio_path:
                test_audio_url = await storage_service.get_file_url(
                    profile.test_audio_path
                )

            result.append(
                {
                    "id": profile.id,
                    "voice_id": profile.voice_id,
                    "name": profile.name,
                    "description": profile.description,
                    "gender": profile.gender,
                    "accent": profile.accent,
                    "voice_type": profile.voice_type,
                    "style": profile.style,
                    "is_system_default": profile.is_system_default,
                    "generation_seed": profile.generation_seed,
                    "voice_sample_url": voice_sample_url,
                    "test_audio_url": test_audio_url,
                    "created_at": profile.created_at.isoformat(),
                    "file_info": {
                        "filename": profile.original_filename,
                        "content_type": profile.content_type,
                        "size": profile.file_size,
                        "duration": profile.duration,
                    },
                    "optimization_params": {
                        "similarity_boost": profile.optimal_similarity_boost,
                        "stability": profile.optimal_stability,
                        "style": profile.optimal_style,
                        "exaggeration": profile.optimal_exaggeration,
                        "cfg_weight": profile.optimal_cfg_weight,
                        "temperature": profile.optimal_temperature,
                    },
                }
            )

        return {
            "success": True,
            "system_voices": result,
            "total": len(result),
            "categories": {
                "default": [v for v in result if v["is_system_default"]],
                "cloned": [v for v in result if v["voice_type"] == "cloned"],
                "synthetic": [v for v in result if v["voice_type"] == "synthetic"],
            },
        }

    except Exception as e:
        logger.error(f"Failed to get system voices: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system voices: {str(e)}"
        )


@router.post("/promote-to-system-voice/{voice_profile_id}")
async def promote_voice_to_system(
    voice_profile_id: int,
    make_default: bool = Form(False),
    current_user: User = Depends(get_current_user),
):
    """
    Promote a user's voice to a system voice (available to all users)
    Only voice owners can promote their voices to system status
    """
    try:
        from ..models.voice_profile import VoiceProfile
        from ..core.database import get_db
        from sqlalchemy.orm import Session

        db: Session = next(get_db())

        voice_profile = (
            db.query(VoiceProfile)
            .filter(
                VoiceProfile.id == voice_profile_id,
                VoiceProfile.user_id
                == current_user.id,  # Only allow user to promote their own voices
                VoiceProfile.is_active == True,
            )
            .first()
        )

        if not voice_profile:
            raise HTTPException(
                status_code=404,
                detail="Voice profile not found or you don't have permission to modify it",
            )

        # Promote to system voice
        voice_profile.user_id = None  # Remove user ownership (makes it system-wide)
        voice_profile.is_system_default = make_default
        voice_profile.style = "system_baseline"

        db.commit()
        db.refresh(voice_profile)

        logger.info(
            f"Voice '{voice_profile.name}' promoted to system voice by user {current_user.id}"
        )

        return {
            "success": True,
            "voice_id": voice_profile.voice_id,
            "voice_name": voice_profile.name,
            "is_system_default": voice_profile.is_system_default,
            "message": f"Voice '{voice_profile.name}' successfully promoted to system voice",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to promote voice to system: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to promote voice to system: {str(e)}"
        )


@router.post("/generate-synthetic-voice")
async def generate_synthetic_voice_endpoint(
    voice_name: str = Form(...),
    description: str = Form(""),
    gender: str = Form("male"),  # "male", "female", "neutral"
    accent: str = Form("american"),  # "american", "british", "australian", etc.
    personality: str = Form(
        "professional"
    ),  # "professional", "casual", "energetic", "calm"
    generate_variations: int = Form(3),  # Number of variations (1-5)
    is_system_default: bool = Form(False),
):
    """
    🎭 Generate synthetic voices based on characteristics (no audio file needed)

    Creates voices using AI characteristics like gender, accent, and personality.
    Perfect for creating diverse system voices without source audio!
    """
    try:
        from ..services.storage_service import storage_service
        from ..models.voice_profile import VoiceProfile
        from ..core.database import get_db
        from sqlalchemy.orm import Session
        import time
        import random

        logger.info(
            f"Generating synthetic voice '{voice_name}' - Gender: {gender}, Accent: {accent}, Personality: {personality}"
        )

        # Validate parameters
        if gender not in ["male", "female", "neutral"]:
            raise HTTPException(
                status_code=400, detail="gender must be 'male', 'female', or 'neutral'"
            )

        if generate_variations < 1 or generate_variations > 5:
            raise HTTPException(
                status_code=400, detail="generate_variations must be between 1 and 5"
            )

        # Enhanced synthetic voice prompts based on characteristics
        personality_prompts = {
            "professional": "Clear, authoritative speaking style with measured pace and professional tone",
            "casual": "Relaxed, conversational tone with natural inflections and friendly warmth",
            "energetic": "Dynamic, enthusiastic delivery with varied pace and expressive intonation",
            "calm": "Soothing, steady voice with gentle pace and peaceful tone",
            "analytical": "Thoughtful, precise speaking with deliberate pacing and intellectual depth",
            "storyteller": "Engaging, narrative style with dramatic pauses and emotional range",
        }

        accent_adjustments = {
            "american": {"stability": 0.8, "style": 0.15, "exaggeration": 0.25},
            "british": {"stability": 0.85, "style": 0.2, "exaggeration": 0.2},
            "australian": {"stability": 0.75, "style": 0.25, "exaggeration": 0.35},
            "indian": {"stability": 0.8, "style": 0.3, "exaggeration": 0.4},
            "french": {"stability": 0.7, "style": 0.35, "exaggeration": 0.45},
            "german": {"stability": 0.85, "style": 0.2, "exaggeration": 0.25},
            "canadian": {"stability": 0.8, "style": 0.18, "exaggeration": 0.22},
            "scottish": {"stability": 0.75, "style": 0.3, "exaggeration": 0.4},
            "irish": {"stability": 0.7, "style": 0.28, "exaggeration": 0.38},
        }

        # Get database session without authentication requirements
        try:
            db: Session = next(get_db())
            logger.info("✅ Database connection established")
        except Exception as db_error:
            logger.error(f"❌ Database connection failed: {db_error}", exc_info=True)
            return {
                "success": False,
                "voice_name": voice_name,
                "voice_type": "synthetic",
                "error": f"Database connection failed: {str(db_error)}",
                "variations_requested": generate_variations,
                "variations_created": 0,
                "variations": [],
                "message": f"Failed to connect to database for {voice_name}",
            }
        results = []

        # Generate multiple variations
        for variation_num in range(generate_variations):
            try:
                # Generate random seed for each variation
                variation_seed = random.randint(10000, 99999)

                # Create unique voice ID for each variation
                base_voice_id = f"synthetic_{voice_name.lower().replace(' ', '_')}_{gender}_{accent}_{personality}"
                if generate_variations > 1:
                    voice_id = f"{base_voice_id}_v{variation_num + 1}_{variation_seed}"
                else:
                    voice_id = f"{base_voice_id}_{variation_seed}"

                logger.info(
                    f"Creating synthetic variation {variation_num + 1}/{generate_variations} with seed {variation_seed}"
                )

                # Create enhanced test text for synthetic voice
                test_text = (
                    f"Hello, I'm {voice_name}, a {personality} {gender} voice with a {accent} accent. "
                    f"I can adapt my speaking style to match various content types, from professional presentations "
                    f"to casual conversations. My voice combines clarity with natural expression, making complex "
                    f"topics accessible and engaging. How does this synthetic voice sound to you?"
                )

                # Enhanced parameters for synthetic voice quality
                base_params = {
                    "speed": 1.0,
                    "stability": 0.75,
                    "similarity_boost": 0.7,  # Lower for synthetic (no source to match)
                    "style": 0.2,
                    "exaggeration": 0.3,
                    "cfg_weight": 0.6,
                    "temperature": 0.6,
                    "seed": variation_seed,
                    "emotion_mode": EmotionMode.CONVERSATIONAL,
                }

                # Apply accent-specific adjustments
                if accent in accent_adjustments:
                    base_params.update(accent_adjustments[accent])

                # Apply personality adjustments
                personality_adjustments = {
                    "professional": {
                        "exaggeration": 0.2,
                        "cfg_weight": 0.7,
                        "temperature": 0.5,
                    },
                    "casual": {
                        "exaggeration": 0.4,
                        "cfg_weight": 0.5,
                        "temperature": 0.7,
                    },
                    "energetic": {
                        "exaggeration": 0.6,
                        "cfg_weight": 0.4,
                        "temperature": 0.8,
                    },
                    "calm": {
                        "exaggeration": 0.15,
                        "cfg_weight": 0.8,
                        "temperature": 0.4,
                    },
                    "analytical": {
                        "exaggeration": 0.25,
                        "cfg_weight": 0.75,
                        "temperature": 0.45,
                    },
                    "storyteller": {
                        "exaggeration": 0.5,
                        "cfg_weight": 0.45,
                        "temperature": 0.75,
                    },
                }

                if personality in personality_adjustments:
                    base_params.update(personality_adjustments[personality])

                # Select base voice for generation
                base_voice = "bjorn" if gender == "male" else "felix"

                # Generate synthetic voice test sample
                logger.info(
                    f"Generating TTS for variation {variation_num + 1} with base voice '{base_voice}'"
                )
                test_response = await chatterbox_service.convert_text_to_speech(
                    text=test_text, voice_id=base_voice, **base_params
                )
                logger.info(
                    f"TTS generation result: success={test_response.success}, duration={test_response.duration if test_response.success else 'N/A'}"
                )

                # Save test sample
                test_audio_url = None
                test_sample_path = None

                if test_response.success and test_response.audio_data:
                    test_sample_path = await storage_service.save_audio_file(
                        test_response.audio_data,
                        "synthetic_voices",
                        segment_id=f"synthetic_{voice_id}_{int(time.time())}",
                        file_type="mp3",
                        metadata={
                            "voice_id": voice_id,
                            "voice_name": voice_name,
                            "variation_number": variation_num + 1,
                            "seed": variation_seed,
                            "voice_type": "synthetic",
                            "gender": gender,
                            "accent": accent,
                            "personality": personality,
                            "purpose": "synthetic_voice_sample",
                            "test_text_preview": test_text[:100] + "...",
                        },
                    )

                    test_audio_url = await storage_service.get_file_url(
                        test_sample_path
                    )

                # Add to chatterbox service
                success = chatterbox_service.add_custom_voice(
                    voice_id=voice_id,
                    name=f"{voice_name} (Synthetic V{variation_num + 1})"
                    if generate_variations > 1
                    else f"{voice_name} (Synthetic)",
                    description=f"{description} | Synthetic {gender} voice with {accent} accent and {personality} personality | Seed: {variation_seed}",
                    audio_prompt_path=None,  # No audio prompt for synthetic voices
                    gender=gender,
                    style="synthetic",
                )

                if not success:
                    logger.error(
                        f"Failed to add synthetic voice {voice_id} to chatterbox service"
                    )
                    continue

                # Create database record
                logger.info(f"Creating database record for voice {voice_id}")
                voice_profile = VoiceProfile(
                    user_id=None,  # System voices have no user (no auth required)
                    voice_id=voice_id,
                    name=f"{voice_name} (Synthetic V{variation_num + 1})"
                    if generate_variations > 1
                    else f"{voice_name} (Synthetic)",
                    description=f"{description} | Synthetic {gender} voice with {accent} accent and {personality} personality",
                    original_audio_path=None,  # No original audio for synthetic
                    test_audio_path=test_sample_path,
                    gender=gender,
                    style="synthetic",
                    is_active=True,
                    # Enhanced fields
                    accent=accent,
                    voice_type="synthetic",
                    generation_seed=variation_seed,
                    is_system_default=is_system_default,
                    # Optimization parameters
                    optimal_similarity_boost=base_params["similarity_boost"],
                    optimal_stability=base_params["stability"],
                    optimal_style=base_params["style"],
                    optimal_exaggeration=base_params["exaggeration"],
                    optimal_cfg_weight=base_params["cfg_weight"],
                    optimal_temperature=base_params["temperature"],
                    # Metadata
                    file_size=0,  # No original file
                    original_filename=None,
                    content_type="synthetic",
                )

                logger.info(f"Adding voice profile to database")
                db.add(voice_profile)
                logger.info(f"Committing database transaction")
                db.commit()
                logger.info(f"Refreshing voice profile from database")
                db.refresh(voice_profile)
                logger.info(
                    f"Successfully created voice profile with ID: {voice_profile.id}"
                )

                results.append(
                    {
                        "variation_number": variation_num + 1,
                        "voice_id": voice_id,
                        "seed": variation_seed,
                        "voice_profile_id": voice_profile.id,
                        "test_audio_url": test_audio_url,
                        "parameters_used": base_params,
                        "characteristics": {
                            "gender": gender,
                            "accent": accent,
                            "personality": personality,
                            "base_voice": base_voice,
                        },
                        "success": test_response.success
                        if "test_response" in locals()
                        else False,
                        "duration": test_response.duration
                        if "test_response" in locals() and test_response.success
                        else 0,
                        "processing_time": test_response.processing_time
                        if "test_response" in locals() and test_response.success
                        else 0,
                    }
                )

                logger.info(
                    f"Successfully created synthetic voice variation {variation_num + 1}: {voice_id}"
                )

            except Exception as variation_error:
                logger.error(
                    f"Failed to create synthetic variation {variation_num + 1}: {variation_error}",
                    exc_info=True,
                )
                results.append(
                    {
                        "variation_number": variation_num + 1,
                        "success": False,
                        "error": str(variation_error),
                        "voice_id": f"synthetic_{voice_name.lower().replace(' ', '_')}_{gender}_{accent}_{personality}_v{variation_num + 1}"
                        if "voice_name" in locals()
                        else "unknown",
                    }
                )

        successful_variations = [r for r in results if r.get("success", False)]

        return {
            "success": len(successful_variations) > 0,
            "voice_name": voice_name,
            "voice_type": "synthetic",
            "characteristics": {
                "gender": gender,
                "accent": accent,
                "personality": personality,
            },
            "is_system_default": is_system_default,
            "variations_requested": generate_variations,
            "variations_created": len(successful_variations),
            "variations": results,
            "message": f"Created {len(successful_variations)}/{generate_variations} synthetic voice variations for {voice_name}",
            "next_steps": [
                "Test each synthetic variation by playing the test samples",
                "Select the best variation based on quality and character fit",
                "Consider promoting to system voice for all users",
                "Use in podcast generation or other TTS applications",
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthetic voice generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Synthetic voice generation failed: {str(e)}"
        )
