from typing import Dict, List, Optional, Any, Callable
from .research_agent import ResearchAgent
from .script_agent import ScriptAgent
from .openai_service import OpenAIService
from ..models.podcast import Podcast
from ..services.podcast_service import PodcastService
from sqlalchemy.orm import Session
import logging
import asyncio
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """
    Orchestrates the entire podcast generation pipeline

    This orchestrator coordinates:
    1. Research Agent - Topic research and subtopic generation
    2. Script Agent - Dialogue and script generation
    3. Memory/State Store - Tracking generation progress
    4. Error handling and retry logic
    5. Progress callbacks and status updates
    """

    def __init__(self, db: Session):
        self.db = db
        self.research_agent = ResearchAgent()
        self.script_agent = ScriptAgent()
        self.openai_service = OpenAIService()
        self.podcast_service = PodcastService(db)

        # Generation state tracking
        self.current_generation = None
        self.generation_history = []

    async def generate_podcast(
        self,
        podcast_id: int,
        progress_callback: Optional[Callable] = None,
        custom_settings: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete podcast from research to script

        Args:
            podcast_id: ID of the podcast to generate
            progress_callback: Optional callback for progress updates
            custom_settings: Optional custom generation settings

        Returns:
            Generation result with status and data
        """
        logger.info(f"Starting podcast generation for ID: {podcast_id}")

        # Initialize generation state
        generation_state = self._initialize_generation_state(
            podcast_id, custom_settings
        )
        self.current_generation = generation_state

        try:
            # Step 1: Get podcast details
            await self._update_progress(
                progress_callback, "Retrieving podcast details", 5
            )
            # For conversation orchestrator, we'll allow the call without user_id since
            # it's an internal service - this is a temporary fix
            # In production, we should pass user_id to this method too
            podcast_query = (
                self.db.query(Podcast).filter(Podcast.id == podcast_id).first()
            )
            if not podcast_query:
                raise ValueError(f"Podcast with ID {podcast_id} not found")
            podcast = podcast_query

            generation_state["podcast"] = {
                "id": podcast.id,
                "title": podcast.title,
                "topic": podcast.topic,
                "length": podcast.length,
            }

            # Step 2: Research phase
            await self._update_progress(
                progress_callback, "Conducting topic research", 15
            )
            research_result = await self._conduct_research(
                podcast.topic, podcast.length, generation_state
            )

            if not research_result["success"]:
                raise Exception(f"Research failed: {research_result['error']}")

            # Step 3: Script generation phase
            await self._update_progress(
                progress_callback, "Generating podcast script", 50
            )
            script_result = await self._generate_script(
                research_result["data"], podcast.length, generation_state
            )

            if not script_result["success"]:
                raise Exception(f"Script generation failed: {script_result['error']}")

            # Step 4: Validation and quality check
            await self._update_progress(
                progress_callback, "Validating generated content", 80
            )
            validation_result = await self._validate_generation(
                research_result["data"], script_result["data"], generation_state
            )

            # Step 5: Save results
            await self._update_progress(
                progress_callback, "Saving generated content", 90
            )
            save_result = await self._save_generation_results(
                podcast,
                research_result["data"],
                script_result["data"],
                validation_result,
                generation_state,
            )

            # Step 6: Complete
            await self._update_progress(progress_callback, "Generation completed", 100)

            final_result = {
                "success": True,
                "podcast_id": podcast_id,
                "generation_id": generation_state["id"],
                "research_data": research_result["data"],
                "script_data": script_result["data"],
                "validation": validation_result,
                "metadata": generation_state,
                "completed_at": datetime.utcnow().isoformat(),
            }

            # Add to history
            self.generation_history.append(final_result)
            self.current_generation = None

            logger.info(
                f"Podcast generation completed successfully for ID: {podcast_id}"
            )
            return final_result

        except Exception as e:
            logger.error(f"Podcast generation failed for ID {podcast_id}: {str(e)}")

            # Update podcast status to failed (no user_id needed for internal service)
            try:
                self.podcast_service.update_podcast_status(podcast_id, "failed")
            except Exception as update_error:
                logger.error(f"Failed to update podcast status: {update_error}")

            error_result = {
                "success": False,
                "podcast_id": podcast_id,
                "error": str(e),
                "generation_state": generation_state,
                "failed_at": datetime.utcnow().isoformat(),
            }

            self.generation_history.append(error_result)
            self.current_generation = None

            return error_result

    def _initialize_generation_state(
        self, podcast_id: int, custom_settings: Optional[Dict]
    ) -> Dict[str, Any]:
        """Initialize generation state tracking"""
        return {
            "id": f"gen_{podcast_id}_{int(datetime.utcnow().timestamp())}",
            "podcast_id": podcast_id,
            "started_at": datetime.utcnow().isoformat(),
            "current_phase": "initialization",
            "progress": 0,
            "settings": custom_settings or self._get_default_settings(),
            "phases_completed": [],
            "errors": [],
            "warnings": [],
        }

    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default generation settings"""
        return {
            "research_depth": "standard",
            "script_style": {
                "tone": "conversational",
                "formality": "casual",
                "humor_level": "light",
            },
            "quality_threshold": 70,
            "retry_attempts": 2,
            "enable_validation": True,
        }

    async def _update_progress(
        self, callback: Optional[Callable], message: str, progress: int
    ):
        """Update generation progress"""
        if self.current_generation:
            self.current_generation["current_phase"] = message
            self.current_generation["progress"] = progress

        logger.info(f"Progress: {progress}% - {message}")

        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress, message)
                else:
                    callback(progress, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def _conduct_research(
        self, topic: str, length: int, generation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Conduct research phase"""
        try:
            settings = generation_state["settings"]
            research_depth = settings.get("research_depth", "standard")

            research_data = self.research_agent.research_topic(
                main_topic=topic, target_length=length, depth=research_depth
            )

            if not research_data:
                return {"success": False, "error": "Research agent returned no data"}

            # Validate research quality
            validation = self.research_agent.validate_research(research_data)
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "error": f"Research validation failed: {validation['issues']}",
                }

            # Check quality threshold
            quality_threshold = settings.get("quality_threshold", 70)
            if validation["quality_score"] < quality_threshold:
                generation_state["warnings"].append(
                    f"Research quality ({validation['quality_score']}) below threshold ({quality_threshold})"
                )

            generation_state["phases_completed"].append("research")

            return {"success": True, "data": research_data, "validation": validation}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_script(
        self,
        research_data: Dict[str, Any],
        length: int,
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate script phase"""
        try:
            settings = generation_state["settings"]
            style_preferences = settings.get("script_style", {})

            script_data = self.script_agent.generate_script(
                research_data=research_data,
                target_length=length,
                style_preferences=style_preferences,
            )

            if not script_data:
                return {"success": False, "error": "Script agent returned no data"}

            # Validate script quality
            validation = self.script_agent.validate_script(script_data)
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "error": f"Script validation failed: {validation['issues']}",
                }

            # Check quality threshold
            quality_threshold = settings.get("quality_threshold", 70)
            if validation["quality_score"] < quality_threshold:
                generation_state["warnings"].append(
                    f"Script quality ({validation['quality_score']}) below threshold ({quality_threshold})"
                )

            generation_state["phases_completed"].append("script_generation")

            return {"success": True, "data": script_data, "validation": validation}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _validate_generation(
        self,
        research_data: Dict[str, Any],
        script_data: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate the complete generation"""
        validation_result = {
            "overall_valid": True,
            "research_validation": None,
            "script_validation": None,
            "coherence_check": None,
            "recommendations": [],
        }

        try:
            # Re-validate research
            research_validation = self.research_agent.validate_research(research_data)
            validation_result["research_validation"] = research_validation

            # Re-validate script
            script_validation = self.script_agent.validate_script(script_data)
            validation_result["script_validation"] = script_validation

            # Check coherence between research and script
            coherence_check = self._check_research_script_coherence(
                research_data, script_data
            )
            validation_result["coherence_check"] = coherence_check

            # Overall validation
            if not research_validation["is_valid"] or not script_validation["is_valid"]:
                validation_result["overall_valid"] = False

            # Generate recommendations
            recommendations = []
            if research_validation["quality_score"] < 80:
                recommendations.append(
                    "Consider regenerating research with deeper analysis"
                )
            if script_validation["quality_score"] < 80:
                recommendations.append("Consider refining script for better quality")
            if not coherence_check["is_coherent"]:
                recommendations.append("Research and script topics may not align well")

            validation_result["recommendations"] = recommendations
            generation_state["phases_completed"].append("validation")

            return validation_result

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "overall_valid": False,
                "error": str(e),
                "research_validation": None,
                "script_validation": None,
                "coherence_check": None,
                "recommendations": [
                    "Validation process failed - manual review recommended"
                ],
            }

    def _check_research_script_coherence(
        self, research_data: Dict[str, Any], script_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if script aligns with research"""
        coherence_result = {
            "is_coherent": True,
            "alignment_score": 0,
            "issues": [],
            "topic_coverage": {},
        }

        try:
            # Check if main topics align
            research_topic = research_data.get("main_topic", "").lower()
            script_title = script_data.get("title", "").lower()

            # Simple keyword overlap check
            research_words = set(research_topic.split())
            script_words = set(script_title.split())
            overlap = len(research_words.intersection(script_words))

            if overlap == 0:
                coherence_result["issues"].append(
                    "Script title doesn't reflect research topic"
                )
                coherence_result["is_coherent"] = False

            # Check subtopic coverage
            research_subtopics = research_data.get("subtopics", [])
            script_segments = script_data.get("segments", [])

            covered_topics = 0
            for subtopic in research_subtopics:
                subtopic_title = subtopic.get("title", "").lower()
                for segment in script_segments:
                    segment_content = json.dumps(segment).lower()
                    if any(word in segment_content for word in subtopic_title.split()):
                        covered_topics += 1
                        break

            if research_subtopics:
                coverage_ratio = covered_topics / len(research_subtopics)
                coherence_result["topic_coverage"]["ratio"] = coverage_ratio
                coherence_result["alignment_score"] = coverage_ratio * 100

                if coverage_ratio < 0.5:
                    coherence_result["issues"].append(
                        "Script covers less than 50% of research topics"
                    )
                    coherence_result["is_coherent"] = False

            return coherence_result

        except Exception as e:
            logger.error(f"Coherence check failed: {e}")
            return {
                "is_coherent": False,
                "alignment_score": 0,
                "issues": [f"Coherence check failed: {str(e)}"],
                "topic_coverage": {},
            }

    async def _save_generation_results(
        self,
        podcast: Podcast,
        research_data: Dict[str, Any],
        script_data: Dict[str, Any],
        validation_result: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save generation results to database"""
        try:
            # Update podcast with generated script
            script_text = self._convert_script_to_text(script_data)

            updated_podcast = self.podcast_service.update_podcast(
                podcast_id=podcast.id,
                updates={"script": script_text, "status": "completed"},
            )

            # TODO: Save research data and metadata to separate tables
            # For now, we'll store it in the podcast script field as JSON
            full_data = {
                "script_data": script_data,
                "research_data": research_data,
                "validation": validation_result,
                "generation_metadata": generation_state,
            }

            generation_state["phases_completed"].append("save_results")

            return {
                "success": True,
                "podcast": updated_podcast,
                "saved_data": full_data,
            }

        except Exception as e:
            logger.error(f"Failed to save generation results: {e}")
            return {"success": False, "error": str(e)}

    def _convert_script_to_text(self, script_data: Dict[str, Any]) -> str:
        """Convert structured script data to readable text"""
        if not script_data or "segments" not in script_data:
            return "Script generation failed"

        text_parts = []
        title = script_data.get("title", "Untitled Podcast")
        text_parts.append(f"# {title}\n")

        for segment in script_data["segments"]:
            segment_type = segment.get("type", "segment")
            text_parts.append(f"\n## {segment_type.title()}\n")

            for line in segment.get("dialogue", []):
                speaker = line.get("speaker", "Unknown")
                text = line.get("text", "")
                text_parts.append(f"**{speaker}:** {text}\n")

        return "\n".join(text_parts)

    def get_generation_status(
        self, generation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current or specific generation status"""
        if generation_id:
            # Find specific generation in history
            for gen in self.generation_history:
                if gen.get("generation_id") == generation_id:
                    return gen
            return {"error": "Generation not found"}
        else:
            # Return current generation status
            return self.current_generation or {"status": "No active generation"}

    def test_pipeline(self) -> Dict[str, Any]:
        """Test the entire pipeline with a simple example"""
        try:
            # Test OpenAI connection
            openai_test = self.openai_service.test_connection()

            # Test research agent
            research_test = self.research_agent.research_topic(
                "The future of artificial intelligence", target_length=5
            )

            # Test script agent if research works
            script_test = None
            if research_test:
                script_test = self.script_agent.generate_script(
                    research_data=research_test, target_length=5
                )

            return {
                "openai_connection": openai_test,
                "research_agent": research_test is not None,
                "script_agent": script_test is not None,
                "pipeline_ready": all([openai_test, research_test, script_test]),
            }

        except Exception as e:
            return {"error": str(e), "pipeline_ready": False}
