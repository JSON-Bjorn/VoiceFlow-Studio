"""
Audio Agent API endpoints for VoiceFlow Studio

This module provides endpoints for audio processing operations:
- Audio assembly (combining voice segments into complete episodes)
- Audio health checks
- Audio processing status
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..services.audio_agent import audio_agent
from ..core.auth import get_current_user
from ..models.user import User

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audio", tags=["audio"])


class AudioAssemblyRequest(BaseModel):
    """Request model for audio assembly"""

    voice_segments: List[Dict[str, Any]] = Field(
        ..., description="List of voice segments to assemble"
    )
    podcast_id: str = Field(..., description="Podcast identifier")
    episode_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional episode metadata"
    )
    processing_options: Optional[Dict[str, Any]] = Field(
        None, description="Optional audio processing options"
    )


class AudioAssemblyResponse(BaseModel):
    """Response model for audio assembly"""

    success: bool
    final_audio_path: Optional[str] = None
    final_audio_url: Optional[str] = None
    total_duration: float = 0.0
    segments_processed: int = 0
    processing_time: float = 0.0
    file_size_bytes: int = 0
    metadata: Dict[str, Any] = {}
    error_message: Optional[str] = None


class AudioHealthResponse(BaseModel):
    """Response model for audio health check"""

    agent: str
    type: str
    version: str
    status: str
    timestamp: str
    details: Dict[str, Any]


@router.get("/health", response_model=AudioHealthResponse)
async def get_audio_health():
    """Get Audio Agent health status"""
    try:
        health = await audio_agent.health_check()
        return AudioHealthResponse(**health)

    except Exception as e:
        logger.error(f"Audio health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assemble", response_model=AudioAssemblyResponse)
async def assemble_podcast_episode(
    request: AudioAssemblyRequest, current_user: User = Depends(get_current_user)
):
    """
    Assemble voice segments into a complete podcast episode

    This endpoint:
    - Takes individual voice segments
    - Combines them with proper transitions
    - Applies audio processing (normalization, compression)
    - Exports final episode file
    """
    try:
        logger.info(f"Starting audio assembly for podcast {request.podcast_id}")

        # Check if Audio Agent is available
        if not audio_agent.is_available():
            raise HTTPException(
                status_code=503,
                detail="Audio processing unavailable - PyDub dependency required",
            )

        # Validate voice segments
        if not request.voice_segments:
            raise HTTPException(
                status_code=400, detail="No voice segments provided for assembly"
            )

        # Perform audio assembly
        result = await audio_agent.assemble_podcast_episode(
            voice_segments=request.voice_segments,
            podcast_id=request.podcast_id,
            episode_metadata=request.episode_metadata,
        )

        # Prepare response
        response_data = {
            "success": result.success,
            "final_audio_path": result.final_audio_path,
            "final_audio_url": result.final_audio_url,
            "total_duration": result.total_duration,
            "segments_processed": result.segments_processed,
            "processing_time": result.processing_time,
            "file_size_bytes": result.file_size_bytes,
            "metadata": result.metadata,
            "error_message": result.error_message,
        }

        if result.success:
            logger.info(
                f"Audio assembly completed for podcast {request.podcast_id}: "
                f"{result.total_duration:.1f}s episode, {result.file_size_bytes} bytes"
            )
        else:
            logger.error(f"Audio assembly failed: {result.error_message}")

        return AudioAssemblyResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio assembly endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/status")
async def get_audio_agents_status(current_user: User = Depends(get_current_user)):
    """Get status of all audio processing agents"""
    try:
        return {
            "audio_agent": {
                "available": audio_agent.is_available(),
                "health": await audio_agent.health_check(),
                "settings": {
                    "default_format": audio_agent.default_format,
                    "default_bitrate": audio_agent.default_bitrate,
                    "default_sample_rate": audio_agent.default_sample_rate,
                    "default_channels": audio_agent.default_channels,
                    "processing_enabled": {
                        "normalization": audio_agent.normalize_audio,
                        "compression": audio_agent.apply_compression,
                        "silence_removal": True,
                    },
                    "timing_settings": {
                        "speaker_transition_ms": audio_agent.speaker_transition_pause,
                        "segment_transition_ms": audio_agent.segment_transition_pause,
                        "intro_outro_fade_ms": audio_agent.intro_outro_fade,
                    },
                },
            }
        }

    except Exception as e:
        logger.error(f"Failed to get audio agents status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing/options")
async def get_audio_processing_options():
    """Get available audio processing options and settings"""

    if not audio_agent.is_available():
        raise HTTPException(status_code=503, detail="Audio processing unavailable")

    return {
        "supported_formats": ["mp3", "wav", "m4a", "ogg"],
        "supported_bitrates": ["96k", "128k", "192k", "256k", "320k"],
        "supported_sample_rates": [22050, 44100, 48000],
        "supported_channels": [1, 2],  # Mono, Stereo
        "processing_options": {
            "normalization": {
                "enabled": audio_agent.normalize_audio,
                "description": "Normalize audio levels for consistent volume",
            },
            "compression": {
                "enabled": audio_agent.apply_compression,
                "description": "Apply dynamic range compression for better listening experience",
            },
            "silence_removal": {
                "enabled": True,
                "threshold_db": audio_agent.remove_silence_threshold,
                "max_duration_ms": audio_agent.max_silence_duration,
                "description": "Remove excessive silence while preserving natural pauses",
            },
            "speaker_transitions": {
                "pause_duration_ms": audio_agent.speaker_transition_pause,
                "description": "Add natural pauses between different speakers",
            },
            "segment_transitions": {
                "pause_duration_ms": audio_agent.segment_transition_pause,
                "description": "Add brief pauses between segments from same speaker",
            },
        },
        "quality_settings": {
            "default": {
                "format": audio_agent.default_format,
                "bitrate": audio_agent.default_bitrate,
                "sample_rate": audio_agent.default_sample_rate,
                "channels": audio_agent.default_channels,
            },
            "high_quality": {
                "format": "mp3",
                "bitrate": "320k",
                "sample_rate": 48000,
                "channels": 2,
            },
            "compact": {
                "format": "mp3",
                "bitrate": "96k",
                "sample_rate": 22050,
                "channels": 1,
            },
        },
    }
