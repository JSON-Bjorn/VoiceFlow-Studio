"""
ElevenLabs API Endpoints

This module provides REST API endpoints for ElevenLabs text-to-speech functionality,
including voice management, audio generation, and service health checks.
"""

from fastapi import APIRouter, HTTPException, Depends, Response
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import logging
from io import BytesIO

from ..services.elevenlabs_service import (
    elevenlabs_service,
    TTSRequest,
    TTSResponse,
    VoiceProfile,
)
from ..core.auth import get_current_user
from ..models.user import User

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/elevenlabs", tags=["elevenlabs"])


# Request/Response Models
class TTSRequestModel(BaseModel):
    """Text-to-Speech request model for API"""

    text: str
    voice_id: Optional[str] = None
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_128"
    voice_settings: Optional[Dict[str, Any]] = None


class TTSResponseModel(BaseModel):
    """Text-to-Speech response model for API"""

    success: bool
    message: str
    character_count: int
    voice_id: str
    model_id: str
    audio_url: Optional[str] = None
    estimated_cost: Optional[Dict[str, Any]] = None


class VoiceListResponse(BaseModel):
    """Voice list response model"""

    success: bool
    voices: List[VoiceProfile]
    count: int


class PodcastSegmentRequest(BaseModel):
    """Podcast segment generation request"""

    text: str
    speaker: str = "host_1"
    model_id: str = "eleven_multilingual_v2"


class ServiceStatusResponse(BaseModel):
    """Service status response model"""

    success: bool
    service: str
    status: str
    details: Dict[str, Any]


# Health and Status Endpoints
@router.get("/health", response_model=ServiceStatusResponse)
async def health_check():
    """
    Check the health status of the ElevenLabs service
    """
    try:
        health_status = await elevenlabs_service.health_check()

        return ServiceStatusResponse(
            success=health_status["status"] == "healthy",
            service="elevenlabs",
            status=health_status["status"],
            details=health_status["details"],
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/test")
async def test_connection():
    """
    Test the connection to ElevenLabs API
    """
    try:
        test_result = await elevenlabs_service.test_connection()

        if test_result["available"]:
            return {
                "success": True,
                "message": test_result["message"],
                "voice_count": test_result.get("voice_count", 0),
            }
        else:
            raise HTTPException(status_code=503, detail=test_result["message"])
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")


# Voice Management Endpoints
@router.get("/voices", response_model=VoiceListResponse)
async def get_available_voices(current_user: User = Depends(get_current_user)):
    """
    Get all available voices from ElevenLabs
    """
    try:
        voices = await elevenlabs_service.get_available_voices()

        return VoiceListResponse(success=True, voices=voices, count=len(voices))
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")


@router.get("/voices/{voice_id}", response_model=VoiceProfile)
async def get_voice_details(
    voice_id: str, current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific voice
    """
    try:
        voice_profile = await elevenlabs_service.get_voice_details(voice_id)
        return voice_profile
    except Exception as e:
        logger.error(f"Failed to get voice details: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get voice details: {str(e)}"
        )


@router.get("/voices/podcast/profiles")
async def get_podcast_voices():
    """
    Get predefined podcast host voice profiles
    """
    try:
        voice_profiles = elevenlabs_service.get_podcast_voices()
        return {"success": True, "voices": voice_profiles, "count": len(voice_profiles)}
    except Exception as e:
        logger.error(f"Failed to get podcast voices: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get podcast voices: {str(e)}"
        )


# Text-to-Speech Endpoints
@router.post("/tts", response_model=TTSResponseModel)
async def convert_text_to_speech(
    request: TTSRequestModel, current_user: User = Depends(get_current_user)
):
    """
    Convert text to speech using ElevenLabs API
    """
    try:
        # Validate text length
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Text too long. Maximum 5000 characters allowed.",
            )

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Get cost estimation
        cost_estimate = await elevenlabs_service.estimate_cost(request.text)

        # Convert text to speech
        tts_response = await elevenlabs_service.convert_text_to_speech(
            text=request.text,
            voice_id=request.voice_id,
            model_id=request.model_id,
            output_format=request.output_format,
            voice_settings=request.voice_settings,
        )

        # TODO: Save audio to storage and return URL
        # For now, we'll return the response without audio URL

        return TTSResponseModel(
            success=True,
            message="Text-to-speech conversion successful",
            character_count=tts_response.character_count,
            voice_id=tts_response.voice_id,
            model_id=tts_response.model_id,
            estimated_cost=cost_estimate,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")


@router.post("/tts/stream")
async def convert_text_to_speech_stream(
    request: TTSRequestModel, current_user: User = Depends(get_current_user)
):
    """
    Convert text to speech and return as streaming audio
    """
    try:
        # Validate text length
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Text too long. Maximum 5000 characters allowed.",
            )

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Convert text to speech stream
        audio_stream = await elevenlabs_service.convert_text_to_speech_stream(
            text=request.text,
            voice_id=request.voice_id,
            model_id=request.model_id,
            output_format=request.output_format,
            voice_settings=request.voice_settings,
        )

        # Return streaming response
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=tts_audio.mp3",
                "X-Character-Count": str(len(request.text)),
                "X-Voice-ID": request.voice_id or "default",
                "X-Model-ID": request.model_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS stream conversion failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"TTS stream conversion failed: {str(e)}"
        )


@router.post("/tts/podcast-segment")
async def generate_podcast_segment(
    request: PodcastSegmentRequest, current_user: User = Depends(get_current_user)
):
    """
    Generate audio for a podcast segment with specific speaker
    """
    try:
        # Validate text length
        if len(request.text) > 2000:
            raise HTTPException(
                status_code=400,
                detail="Segment text too long. Maximum 2000 characters allowed.",
            )

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Segment text cannot be empty")

        # Generate podcast segment
        tts_response = await elevenlabs_service.generate_podcast_segment(
            text=request.text, speaker=request.speaker, model_id=request.model_id
        )

        # Get cost estimation
        cost_estimate = await elevenlabs_service.estimate_cost(request.text)

        return {
            "success": True,
            "message": f"Podcast segment generated for {request.speaker}",
            "character_count": tts_response.character_count,
            "voice_id": tts_response.voice_id,
            "model_id": tts_response.model_id,
            "speaker": request.speaker,
            "estimated_cost": cost_estimate,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Podcast segment generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Podcast segment generation failed: {str(e)}"
        )


@router.post("/tts/podcast-segment/stream")
async def generate_podcast_segment_stream(
    request: PodcastSegmentRequest, current_user: User = Depends(get_current_user)
):
    """
    Generate audio for a podcast segment and return as streaming audio
    """
    try:
        # Validate text length
        if len(request.text) > 2000:
            raise HTTPException(
                status_code=400,
                detail="Segment text too long. Maximum 2000 characters allowed.",
            )

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Segment text cannot be empty")

        # Get voice profile for speaker
        voice_profiles = elevenlabs_service.get_podcast_voices()
        if request.speaker not in voice_profiles:
            raise HTTPException(
                status_code=400, detail=f"Unknown speaker: {request.speaker}"
            )

        voice_profile = voice_profiles[request.speaker]
        voice_id = voice_profile["voice_id"]

        # Convert text to speech stream
        audio_stream = await elevenlabs_service.convert_text_to_speech_stream(
            text=request.text, voice_id=voice_id, model_id=request.model_id
        )

        # Return streaming response
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename={request.speaker}_segment.mp3",
                "X-Character-Count": str(len(request.text)),
                "X-Voice-ID": voice_id,
                "X-Model-ID": request.model_id,
                "X-Speaker": request.speaker,
                "X-Speaker-Name": voice_profile["name"],
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Podcast segment stream generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Podcast segment stream generation failed: {str(e)}",
        )


# Cost Estimation Endpoints
@router.post("/estimate-cost")
async def estimate_tts_cost(text: str, current_user: User = Depends(get_current_user)):
    """
    Estimate the cost for text-to-speech conversion
    """
    try:
        cost_estimate = await elevenlabs_service.estimate_cost(text)

        return {"success": True, "cost_estimate": cost_estimate}

    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")


# Configuration Endpoints
@router.get("/config")
async def get_service_config():
    """
    Get ElevenLabs service configuration and status
    """
    try:
        voice_profiles = elevenlabs_service.get_podcast_voices()
        health_status = await elevenlabs_service.health_check()

        return {
            "success": True,
            "service": "elevenlabs",
            "available": elevenlabs_service.is_available(),
            "status": health_status["status"],
            "voice_profiles": voice_profiles,
            "supported_models": [
                "eleven_multilingual_v2",
                "eleven_turbo_v2",
                "eleven_flash_v2_5",
            ],
            "supported_formats": [
                "mp3_44100_128",
                "mp3_44100_192",
                "pcm_16000",
                "pcm_22050",
                "pcm_24000",
                "pcm_44100",
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get service config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get service config: {str(e)}"
        )
