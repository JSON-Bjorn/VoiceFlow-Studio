"""
Chatterbox API Endpoints

This module provides REST API endpoints for Chatterbox text-to-speech functionality,
including voice synthesis, voice management, and audio generation for podcast production.
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
)
from ..models.user import User
from ..api.auth import get_current_user

logger = logging.getLogger(__name__)


# Request/Response Models
class TTSRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=10000, description="Text to convert to speech"
    )
    voice_id: str = Field(default="alex", description="Voice profile ID")
    audio_prompt_path: Optional[str] = Field(
        default=None, description="Path to custom voice prompt audio"
    )
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier"
    )
    stability: float = Field(default=0.5, ge=0.0, le=1.0, description="Voice stability")
    similarity_boost: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Voice similarity boost"
    )
    style: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Voice style strength"
    )


class StreamTTSRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text to convert to speech"
    )
    voice_id: str = Field(default="alex", description="Voice profile ID")
    audio_prompt_path: Optional[str] = Field(
        default=None, description="Path to custom voice prompt audio"
    )


class PodcastSegmentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text content for the segment")
    speaker_id: str = Field(
        ..., description="Speaker ID (host1, host2, expert, interviewer)"
    )
    segment_type: str = Field(default="dialogue", description="Type of segment")
    voice_settings: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom voice settings"
    )


class BatchTTSRequest(BaseModel):
    segments: List[PodcastSegmentRequest] = Field(
        ..., description="List of text segments to process"
    )
    include_cost_estimate: bool = Field(
        default=True, description="Include cost estimation"
    )


class CostEstimateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to estimate cost for")


class CustomVoiceRequest(BaseModel):
    voice_id: str = Field(
        ..., min_length=1, max_length=50, description="Unique voice ID"
    )
    name: str = Field(..., min_length=1, max_length=100, description="Voice name")
    description: str = Field(..., max_length=500, description="Voice description")
    gender: str = Field(default="unknown", description="Voice gender")
    style: str = Field(default="custom", description="Voice style")


# Response Models
class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: Optional[str] = None
    details: Dict[str, Any]


class VoiceResponse(BaseModel):
    voices: List[Dict[str, Any]]


class TTSAudioResponse(BaseModel):
    success: bool
    audio_url: Optional[str] = None
    duration: float
    voice_id: str
    processing_time: Optional[float] = None
    cost_estimate: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# Router Setup
router = APIRouter(prefix="/api/chatterbox", tags=["chatterbox"])


@router.get("/health", response_model=HealthResponse)
async def check_health():
    """
    Check the health status of the Chatterbox service
    """
    try:
        health_status = await chatterbox_service.health_check()
        return HealthResponse(
            status=health_status["status"],
            service="chatterbox",
            timestamp=health_status.get("timestamp"),
            details=health_status.get("details", {}),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/test")
async def test_connection():
    """
    Test the connection to Chatterbox TTS
    """
    try:
        test_result = await chatterbox_service.test_connection()
        if test_result["status"] == "success":
            return {
                "status": "success",
                "message": "Chatterbox TTS connection successful",
                "details": test_result,
            }
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Chatterbox TTS test failed: {test_result.get('message', 'Unknown error')}",
            )
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")


@router.get("/voices", response_model=VoiceResponse)
async def get_available_voices():
    """
    Get all available voices from Chatterbox
    """
    try:
        voices = chatterbox_service.get_available_voices()
        return VoiceResponse(voices=voices)
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")


@router.get("/voices/{voice_id}")
async def get_voice_details(voice_id: str):
    """
    Get detailed information about a specific voice
    """
    try:
        voice_profile = chatterbox_service.get_voice_details(voice_id)
        if not voice_profile:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
        return voice_profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get voice details: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get voice details: {str(e)}"
        )


@router.get("/podcast-voices")
async def get_podcast_voices():
    """
    Get voice profiles optimized for podcast generation
    """
    try:
        voice_profiles = chatterbox_service.get_podcast_voices()
        return {"voices": voice_profiles}
    except Exception as e:
        logger.error(f"Failed to get podcast voices: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get podcast voices: {str(e)}"
        )


@router.post("/generate", response_model=TTSAudioResponse)
async def convert_text_to_speech(
    request: TTSRequest, current_user: User = Depends(get_current_user)
):
    """
    Convert text to speech using Chatterbox TTS
    """
    try:
        # Get cost estimate first
        cost_estimate = chatterbox_service.estimate_cost(request.text)

        # Generate audio
        tts_response = await chatterbox_service.convert_text_to_speech(
            text=request.text,
            voice_id=request.voice_id,
            audio_prompt_path=request.audio_prompt_path,
            speed=request.speed,
            stability=request.stability,
            similarity_boost=request.similarity_boost,
            style=request.style,
        )

        if not tts_response.success:
            raise HTTPException(
                status_code=400,
                detail=f"TTS generation failed: {tts_response.error_message}",
            )

        # TODO: Save audio file and return URL
        # For now, return success with metadata
        return TTSAudioResponse(
            success=True,
            audio_url=None,  # Would contain actual audio URL
            duration=tts_response.duration,
            voice_id=tts_response.voice_id,
            processing_time=tts_response.processing_time,
            cost_estimate=cost_estimate,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@router.post("/generate-stream")
async def convert_text_to_speech_stream(
    request: StreamTTSRequest, current_user: User = Depends(get_current_user)
):
    """
    Convert text to speech and return audio stream
    """
    try:
        audio_stream = await chatterbox_service.convert_text_to_speech_stream(
            text=request.text,
            voice_id=request.voice_id,
            audio_prompt_path=request.audio_prompt_path,
        )

        return StreamingResponse(
            audio_stream,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_audio_{request.voice_id}.wav"
            },
        )

    except Exception as e:
        logger.error(f"Stream TTS generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Stream TTS generation failed: {str(e)}"
        )


@router.post("/podcast-segment", response_model=TTSAudioResponse)
async def generate_podcast_segment(
    request: PodcastSegmentRequest, current_user: User = Depends(get_current_user)
):
    """
    Generate audio for a podcast segment with specific speaker
    """
    try:
        # Generate cost estimate
        cost_estimate = chatterbox_service.estimate_cost(request.text)

        # Generate audio for the segment
        tts_response = await chatterbox_service.generate_podcast_segment(
            text=request.text,
            speaker_id=request.speaker_id,
            segment_type=request.segment_type,
            voice_settings=request.voice_settings,
        )

        if not tts_response.success:
            raise HTTPException(
                status_code=400,
                detail=f"Podcast segment generation failed: {tts_response.error_message}",
            )

        return TTSAudioResponse(
            success=True,
            audio_url=None,  # Would contain actual audio URL
            duration=tts_response.duration,
            voice_id=tts_response.voice_id,
            processing_time=tts_response.processing_time,
            cost_estimate=cost_estimate,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Podcast segment generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Podcast segment generation failed: {str(e)}"
        )


@router.post("/batch-generate")
async def generate_batch_audio(
    request: BatchTTSRequest, current_user: User = Depends(get_current_user)
):
    """
    Generate audio for multiple podcast segments in batch
    """
    try:
        results = []
        total_cost = 0.0
        total_duration = 0.0

        for segment in request.segments:
            # Calculate cost if requested
            if request.include_cost_estimate:
                cost_info = chatterbox_service.estimate_cost(segment.text)
                total_cost += cost_info.get("total_cost", 0.0)

            # Generate audio for segment
            tts_response = await chatterbox_service.generate_podcast_segment(
                text=segment.text,
                speaker_id=segment.speaker_id,
                segment_type=segment.segment_type,
                voice_settings=segment.voice_settings,
            )

            if tts_response.success:
                total_duration += tts_response.duration

            results.append(
                {
                    "text": segment.text[:100] + "..."
                    if len(segment.text) > 100
                    else segment.text,
                    "speaker_id": segment.speaker_id,
                    "success": tts_response.success,
                    "duration": tts_response.duration,
                    "voice_id": tts_response.voice_id,
                    "error_message": tts_response.error_message,
                }
            )

        return {
            "success": True,
            "total_segments": len(request.segments),
            "successful_segments": sum(1 for r in results if r["success"]),
            "total_duration": total_duration,
            "total_cost": total_cost,
            "results": results,
        }

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch generation failed: {str(e)}"
        )


@router.post("/estimate-cost")
async def estimate_cost(request: CostEstimateRequest):
    """
    Estimate the processing cost for text generation
    """
    try:
        cost_estimate = chatterbox_service.estimate_cost(request.text)
        return cost_estimate
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")


@router.post("/custom-voice")
async def add_custom_voice(
    request: CustomVoiceRequest,
    audio_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Add a custom voice profile with audio prompt
    """
    try:
        # Validate audio file
        if not audio_file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid audio file format")

        # Save the uploaded audio file
        audio_dir = Path("storage/audio/custom_voices")
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_path = audio_dir / f"{request.voice_id}_{current_user.id}.wav"

        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        # Add custom voice profile
        success = chatterbox_service.add_custom_voice(
            voice_id=request.voice_id,
            name=request.name,
            description=request.description,
            audio_prompt_path=str(audio_path),
            gender=request.gender,
            style=request.style,
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to add custom voice")

        return {
            "success": True,
            "message": f"Custom voice '{request.voice_id}' added successfully",
            "voice_id": request.voice_id,
            "audio_path": str(audio_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Custom voice creation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Custom voice creation failed: {str(e)}"
        )


@router.get("/config")
async def get_service_config():
    """
    Get Chatterbox service configuration and status
    """
    try:
        voice_profiles = chatterbox_service.get_podcast_voices()
        health_status = await chatterbox_service.health_check()

        return {
            "service": "chatterbox",
            "available": chatterbox_service.is_available(),
            "device": chatterbox_service.device,
            "sample_rate": chatterbox_service.sample_rate,
            "voice_profiles": voice_profiles,
            "health": health_status,
        }

    except Exception as e:
        logger.error(f"Config retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Config retrieval failed: {str(e)}"
        )
