from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import logging
import asyncio
from datetime import datetime

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..services.conversation_orchestrator import ConversationOrchestrator
from ..services.openai_service import OpenAIService
from ..services.research_agent import ResearchAgent
from ..services.script_agent import ScriptAgent
from ..services.podcast_service import PodcastService
from ..services.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai-pipeline", tags=["ai_pipeline"])


# Request/Response models
class TestPipelineResponse(BaseModel):
    openai_connection: bool
    research_agent: bool
    script_agent: bool
    pipeline_ready: bool
    error: Optional[str] = None


class ResearchRequest(BaseModel):
    topic: str
    target_length: int = 10
    depth: str = "standard"


class ScriptGenerationRequest(BaseModel):
    podcast_id: int
    custom_settings: Optional[Dict[str, Any]] = None


class EnhancedGenerationRequest(BaseModel):
    podcast_id: int
    custom_settings: Optional[Dict[str, Any]] = None


class GenerationStatusResponse(BaseModel):
    generation_id: Optional[str]
    status: str
    progress: int
    current_phase: str
    podcast_id: Optional[int] = None


@router.get("/test", response_model=TestPipelineResponse)
async def test_ai_pipeline(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Test the AI pipeline components"""
    try:
        orchestrator = ConversationOrchestrator(db)
        test_result = orchestrator.test_pipeline()

        return TestPipelineResponse(**test_result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline test failed: {str(e)}",
        )


@router.post("/research")
async def generate_research(
    request: ResearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate research for a topic using the Research Agent"""
    try:
        research_agent = ResearchAgent()

        research_data = research_agent.research_topic(
            main_topic=request.topic,
            target_length=request.target_length,
            depth=request.depth,
        )

        if not research_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate research data",
            )

        # Validate the research
        validation = research_agent.validate_research(research_data)

        return {
            "success": True,
            "research_data": research_data,
            "validation": validation,
            "summary": research_agent.get_research_summary(research_data),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research generation failed: {str(e)}",
        )


@router.post("/script/generate")
async def generate_script_from_research(
    research_data: Dict[str, Any],
    target_length: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a script from research data using the Script Agent"""
    try:
        script_agent = ScriptAgent()

        script_data = script_agent.generate_script(
            research_data=research_data, target_length=target_length
        )

        if not script_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate script",
            )

        # Validate the script
        validation = script_agent.validate_script(script_data)

        return {
            "success": True,
            "script_data": script_data,
            "validation": validation,
            "summary": script_agent.get_script_summary(script_data),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Script generation failed: {str(e)}",
        )


@router.post("/generate/podcast")
async def generate_complete_podcast(
    request: ScriptGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a complete podcast using the full AI pipeline"""
    try:
        # Verify the podcast exists and belongs to the user
        podcast_service = PodcastService(db)
        podcast = podcast_service.get_podcast_by_id(request.podcast_id)

        if not podcast:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Podcast not found"
            )

        if podcast.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
            )

        # Update podcast status to generating
        podcast_service.update_podcast_status(
            request.podcast_id, "generating", current_user.id
        )

        # Start generation in background
        orchestrator = ConversationOrchestrator(db)

        # For now, we'll run it synchronously, but in production this should be async
        generation_result = await orchestrator.generate_podcast(
            podcast_id=request.podcast_id, custom_settings=request.custom_settings
        )

        return {
            "success": generation_result["success"],
            "generation_id": generation_result.get("generation_id"),
            "message": "Podcast generation completed"
            if generation_result["success"]
            else "Podcast generation failed",
            "result": generation_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Podcast generation failed: {str(e)}",
        )


@router.get("/generation/status")
async def get_generation_status(
    generation_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the status of a podcast generation"""
    try:
        orchestrator = ConversationOrchestrator(db)
        status_data = orchestrator.get_generation_status(generation_id)

        return {"success": True, "status": status_data}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get generation status: {str(e)}",
        )


@router.post("/test/simple")
async def test_simple_generation(
    topic: str = "The future of renewable energy",
    length: int = 5,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Test the pipeline with a simple topic (for development/testing)"""
    try:
        # Test research generation
        research_agent = ResearchAgent()
        research_data = research_agent.research_topic(
            main_topic=topic, target_length=length, depth="light"
        )

        if not research_data:
            return {"success": False, "error": "Research generation failed"}

        # Test script generation
        script_agent = ScriptAgent()
        script_data = script_agent.generate_script(
            research_data=research_data, target_length=length
        )

        if not script_data:
            return {"success": False, "error": "Script generation failed"}

        return {
            "success": True,
            "topic": topic,
            "research_summary": research_agent.get_research_summary(research_data),
            "script_summary": script_agent.get_script_summary(script_data),
            "full_data": {"research": research_data, "script": script_data},
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simple generation test failed: {str(e)}",
        )


@router.get("/config")
async def get_ai_config(current_user: User = Depends(get_current_user)):
    """Get AI pipeline configuration and status"""
    try:
        openai_service = OpenAIService()
        connection_test = openai_service.test_connection()

        return {
            "openai_configured": connection_test,
            "available_models": ["gpt-4o-mini", "gpt-4o"],
            "default_settings": {
                "research_depth": "standard",
                "script_style": {
                    "tone": "conversational",
                    "formality": "casual",
                    "humor_level": "light",
                },
                "quality_threshold": 70,
            },
            "pipeline_status": "ready" if connection_test else "not_configured",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI configuration: {str(e)}",
        )


@router.post("/enhanced/generate/podcast")
async def generate_enhanced_podcast(
    request: EnhancedGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a complete podcast using the Enhanced AI Pipeline with voice generation"""
    try:
        # Verify the podcast exists and belongs to the user
        podcast_service = PodcastService(db)
        podcast = podcast_service.get_podcast_by_id(request.podcast_id, current_user.id)

        if not podcast:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Podcast not found"
            )

        if podcast.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
            )

        # Update podcast status to generating
        podcast_service.update_podcast_status(
            request.podcast_id, "generating", current_user.id
        )

        # Initialize Enhanced Pipeline Orchestrator
        enhanced_orchestrator = EnhancedPipelineOrchestrator(db)

        # Extract user inputs from custom settings
        custom_settings = request.custom_settings or {}

        # Create user inputs for enhanced pipeline
        user_inputs = {
            "topic": podcast.topic,
            "target_duration": podcast.length,
            "generate_voice": custom_settings.get("generate_voice", False),
            "assemble_audio": custom_settings.get("assemble_audio", True),
            "hosts": custom_settings.get(
                "hosts",
                {
                    "host_1": {
                        "name": "Felix",
                        "personality": "analytical, thoughtful, engaging",
                        "role": "primary_questioner",
                    },
                    "host_2": {
                        "name": "Bjorn",
                        "personality": "warm, curious, conversational",
                        "role": "storyteller",
                    },
                },
            ),
            "style_preferences": custom_settings.get(
                "style_preferences",
                {
                    "tone": "conversational",
                    "complexity": "accessible",
                    "humor_level": "light",
                    "pacing": "moderate",
                },
            ),
            "content_preferences": custom_settings.get(
                "content_preferences",
                {
                    "focus_areas": ["practical applications", "recent developments"],
                    "target_audience": "general public",
                },
            ),
            "audio_options": custom_settings.get(
                "audio_options",
                {
                    "add_intro": True,
                    "add_outro": True,
                    "intro_style": "overlay",
                    "outro_style": "overlay",
                    "intro_asset_id": "default_intro",
                    "outro_asset_id": "default_outro",
                    "add_transitions": False,
                    "transition_asset_id": "default_transition",
                    "add_background_music": False,
                    "background_asset_id": "",
                },
            ),
        }

        # Start enhanced generation
        logger.info(
            f"Starting enhanced podcast generation for podcast ID: {request.podcast_id}"
        )

        generation_result = await enhanced_orchestrator.generate_enhanced_podcast(
            podcast_id=request.podcast_id,
            user_id=current_user.id,
            user_inputs=user_inputs,
            progress_callback=None,  # Could implement WebSocket progress updates later
            quality_settings=custom_settings.get("quality_settings"),
        )

        return {
            "success": generation_result["success"],
            "generation_id": generation_result.get("generation_id"),
            "message": "Enhanced podcast generation completed"
            if generation_result["success"]
            else f"Enhanced podcast generation failed: {generation_result.get('error', 'Unknown error')}",
            "result": generation_result,
            "voice_generated": generation_result.get("voice_data") is not None,
            "total_duration": generation_result.get("voice_data", {}).get(
                "total_duration"
            )
            if generation_result.get("voice_data")
            else None,
            "cost_estimate": generation_result.get("voice_data", {}).get("total_cost")
            if generation_result.get("voice_data")
            else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced podcast generation failed: {e}")

        # Update podcast status to failed
        try:
            podcast_service.update_podcast_status(
                request.podcast_id, "failed", current_user.id
            )
        except Exception as update_error:
            logger.error(f"Failed to update podcast status: {update_error}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced podcast generation failed: {str(e)}",
        )
