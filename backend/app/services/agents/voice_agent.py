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

from ..chatterbox_service import chatterbox_service
from ..storage_service import storage_service
from .voice_personality_agent import VoicePersonalityAgent

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

    # ... (rest of the class implementation from oice_agent.py) ...


voice_agent = VoiceAgent()
