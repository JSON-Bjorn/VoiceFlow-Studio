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

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
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
from ..core.auth import get_current_user

logger = logging.getLogger(__name__)


# Enhanced Request Models with ALL Chatterbox features
class UltimateTTSRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=10000, description="Text to convert to speech"
    )
    voice_id: str = Field(default="alex", description="Voice profile ID")
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
    voice_id: str = Field(default="alex", description="Voice profile ID")
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
    voice_id: str = Field(default="alex", description="Voice profile for all texts")
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
async def ultimate_text_to_speech(
    request: UltimateTTSRequest, current_user: User = Depends(get_current_user)
):
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
        logger.error(f"Voice cloning failed: {e}")
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
