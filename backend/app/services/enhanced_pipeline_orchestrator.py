from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
import asyncio
import time
from .websocket_manager import websocket_manager
from ..models.podcast import Podcast
from ..services.podcast_service import PodcastService
from ..schemas.podcast import PodcastUpdate
from .research_agent import ResearchAgent
from .script_agent import ScriptAgent
from .content_planning_agent import ContentPlanningAgent
from .conversation_flow_agent import ConversationFlowAgent
from .dialogue_distribution_agent import DialogueDistributionAgent
from .personality_adaptation_agent import PersonalityAdaptationAgent
from .voice_agent import VoiceAgent
from .audio_agent import AudioAgent
from sqlalchemy.orm import Session
from .error_handler import (
    error_handler,
    RetryConfig,
    ErrorCategory,
    with_error_handling,
)

logger = logging.getLogger(__name__)


class EnhancedPipelineOrchestrator:
    """
    Enhanced orchestrator for the 6-agent podcast generation pipeline
    with advanced error handling, retry mechanisms, and recovery options
    """

    def __init__(self, db: Session):
        self.db = db
        self.podcast_service = PodcastService(db)

        # Initialize all agents
        self.research_agent = ResearchAgent()
        self.content_planning_agent = ContentPlanningAgent()
        self.conversation_flow_agent = ConversationFlowAgent()
        self.script_agent = ScriptAgent()
        self.dialogue_distribution_agent = DialogueDistributionAgent()
        self.personality_adaptation_agent = PersonalityAdaptationAgent()
        self.voice_agent = VoiceAgent()
        self.audio_agent = AudioAgent()

        # Enhanced error handling
        self.error_handler = error_handler
        self.recovery_strategies = self._initialize_recovery_strategies()

        # Generation state
        self.current_generation = None
        self.generation_history = []

        # Quality thresholds
        self.quality_thresholds = {
            "minimum_research_quality": 0.7,
            "minimum_content_plan_quality": 0.75,
            "minimum_script_quality": 0.8,
            "minimum_coherence_score": 0.75,
            "content_plan_quality": 0.7,
            "maximum_iterations": 3,
            "target_script_length_variance": 0.15,  # 15% variance allowed
        }

    def _initialize_recovery_strategies(self) -> Dict[str, Dict]:
        """Initialize recovery strategies for different error types"""
        return {
            "research_failure": {
                "fallback_strategy": "use_simplified_research",
                "quality_reduction": 0.1,
                "user_message": "Using simplified research approach to continue generation",
            },
            "content_planning_failure": {
                "fallback_strategy": "use_basic_structure",
                "quality_reduction": 0.15,
                "user_message": "Using basic content structure to continue generation",
            },
            "script_generation_failure": {
                "fallback_strategy": "regenerate_with_lower_standards",
                "quality_reduction": 0.2,
                "user_message": "Adjusting quality standards to complete generation",
            },
            "voice_generation_failure": {
                "fallback_strategy": "skip_voice_generation",
                "quality_reduction": 0.0,
                "user_message": "Continuing with text-only output",
            },
        }

    async def generate_enhanced_podcast(
        self,
        podcast_id: int,
        user_id: int,
        user_inputs: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        quality_settings: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate enhanced podcast with comprehensive error handling and recovery
        """
        logger.info(f"Starting enhanced podcast generation for ID: {podcast_id}")

        # Get podcast and user info for WebSocket updates
        podcast = self.podcast_service.get_podcast_by_id(podcast_id, user_id)
        if not podcast:
            raise ValueError(f"Podcast with ID {podcast_id} not found")

        # Initialize enhanced generation state
        generation_state = self._initialize_enhanced_state(
            podcast_id, user_inputs, quality_settings
        )
        self.current_generation = generation_state

        # Helper function to send progress updates via WebSocket with error context
        async def send_progress(
            phase: str, progress: int, message: str, metadata: Optional[Dict] = None
        ):
            await websocket_manager.send_progress_update(
                user_id=podcast.user_id,
                generation_id=generation_state["id"],
                phase=phase,
                progress=progress,
                message=message,
                metadata=metadata,
            )
            # Also call the optional callback
            if progress_callback:
                await progress_callback(phase, progress, message)

        # Enhanced error callback for retry notifications
        async def error_progress_callback(
            message: str, retry_info: Optional[Dict] = None
        ):
            metadata = {"retry_info": retry_info} if retry_info else None
            await send_progress(
                "error_recovery", generation_state.get("progress", 0), message, metadata
            )

        try:
            # Create debug file to track pipeline start
            import time

            pipeline_debug = f"pipeline_start_{int(time.time())}.txt"
            with open(pipeline_debug, "w") as f:
                f.write(f"Enhanced pipeline started at {time.time()}\n")
                f.write(f"user_inputs: {user_inputs}\n")
                f.write(f"generate_voice: {user_inputs.get('generate_voice')}\n")

            # Phase 1: Enhanced Research with Error Recovery
            await send_progress("research", 10, "Conducting contextual research")
            research_result = await self._execute_with_recovery(
                self._enhanced_research_phase,
                "research_failure",
                error_progress_callback,
                user_inputs,
                generation_state,
            )

            if not research_result["success"]:
                await websocket_manager.send_error_notification(
                    user_id=podcast.user_id,
                    generation_id=generation_state["id"],
                    error_message="Research phase failed",
                    error_details=research_result,
                )
                return await self._handle_phase_failure(
                    "research", research_result, generation_state, podcast.user_id
                )

            # Phase 2: Strategic Content Planning with Error Recovery
            await send_progress("planning", 25, "Creating strategic content plan")
            planning_result = await self._execute_with_recovery(
                self._content_planning_phase,
                "content_planning_failure",
                error_progress_callback,
                research_result["data"],
                user_inputs,
                generation_state,
            )

            if not planning_result["success"]:
                await websocket_manager.send_error_notification(
                    user_id=podcast.user_id,
                    generation_id=generation_state["id"],
                    error_message="Content planning failed",
                    error_details=planning_result,
                )
                return await self._handle_phase_failure(
                    "planning", planning_result, generation_state, podcast.user_id
                )

            # Phase 3: Iterative Script Generation with Enhanced Error Handling
            await send_progress(
                "script_generation", 50, "Generating and refining script"
            )
            script_result = await self._execute_with_recovery(
                self._iterative_script_generation,
                "script_generation_failure",
                error_progress_callback,
                research_result["data"],
                planning_result["data"],
                user_inputs,
                generation_state,
                progress_update_callback=send_progress,
            )

            if not script_result["success"]:
                await websocket_manager.send_error_notification(
                    user_id=podcast.user_id,
                    generation_id=generation_state["id"],
                    error_message="Script generation failed",
                    error_details=script_result,
                )
                return await self._handle_phase_failure(
                    "script", script_result, generation_state, podcast.user_id
                )

            # Phase 4: Voice Generation with Circuit Breaker Protection
            voice_result = None
            generate_voice_enabled = user_inputs.get("generate_voice", False)
            voice_agent_available = self.voice_agent.is_available()

            # Create debug file to track condition check
            import time

            debug_file = f"voice_condition_check_{int(time.time())}.txt"
            with open(debug_file, "w") as f:
                f.write(f"Voice condition check at {time.time()}\n")
                f.write(f"user_inputs keys: {list(user_inputs.keys())}\n")
                f.write(f"generate_voice value: {user_inputs.get('generate_voice')}\n")
                f.write(f"generate_voice_enabled: {generate_voice_enabled}\n")
                f.write(f"voice_agent_available: {voice_agent_available}\n")
                f.write(
                    f"condition result: {generate_voice_enabled and voice_agent_available}\n"
                )

            logger.info(
                f"Voice generation check: generate_voice={generate_voice_enabled}, voice_agent_available={voice_agent_available}"
            )

            if generate_voice_enabled and voice_agent_available:
                logger.info("Starting voice generation phase")
                await send_progress("voice_generation", 65, "Generating voice audio")
                try:
                    voice_result = await self._execute_with_recovery(
                        self._voice_generation_phase,
                        "voice_generation_failure",
                        error_progress_callback,
                        script_result["data"],
                        user_inputs,
                        generation_state,
                        send_progress,
                    )
                    logger.info(
                        f"Voice generation result: success={voice_result.get('success') if voice_result else None}"
                    )
                except Exception as e:
                    # Voice generation failure is non-critical, continue without voice
                    logger.warning(
                        f"Voice generation failed, continuing without voice: {e}"
                    )
                    await send_progress(
                        "voice_generation", 65, "Voice generation skipped due to error"
                    )
            else:
                logger.warning(
                    f"Voice generation skipped: generate_voice={generate_voice_enabled}, voice_agent_available={voice_agent_available}"
                )

            # Phase 5: Audio Assembly with Error Recovery
            audio_result = None
            logger.info(f"Checking audio assembly conditions:")
            logger.info(
                f"  - voice_result success: {voice_result and voice_result.get('success')}"
            )
            logger.info(f"  - audio_agent available: {self.audio_agent.is_available()}")
            logger.info(
                f"  - assemble_audio setting: {user_inputs.get('assemble_audio', True)}"
            )

            if (
                voice_result
                and voice_result["success"]
                and self.audio_agent.is_available()
                and user_inputs.get("assemble_audio", True)
            ):
                await send_progress(
                    "audio_assembly", 75, "Assembling final audio episode"
                )
                logger.info("Starting audio assembly phase - all conditions met")
                try:
                    audio_result = await self._execute_with_recovery(
                        self._audio_assembly_phase,
                        "audio_assembly_failure",
                        error_progress_callback,
                        voice_result,
                        user_inputs,
                        generation_state,
                    )
                    logger.info(
                        f"Audio assembly result: success={audio_result.get('success') if audio_result else None}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Audio assembly failed, continuing without assembled audio: {e}"
                    )
                    await send_progress(
                        "audio_assembly", 75, "Audio assembly skipped due to error"
                    )
            else:
                logger.warning("Audio assembly skipped - conditions not met")
                if not voice_result or not voice_result.get("success"):
                    logger.warning("  - Voice generation was not successful")
                if not self.audio_agent.is_available():
                    logger.warning("  - Audio agent not available (PyDub missing?)")
                if not user_inputs.get("assemble_audio", True):
                    logger.warning("  - Audio assembly disabled in user inputs")

            # Phase 6: Final Quality Validation and Optimization
            await send_progress("validation", 85, "Final quality validation")
            validation_result = await self._comprehensive_validation(
                research_result["data"],
                planning_result["data"],
                script_result["data"],
                generation_state,
            )

            # Phase 7: Save and Complete
            await send_progress("saving", 95, "Saving optimized content")
            save_result = await self._save_enhanced_results(
                podcast_id,
                user_id,
                research_result["data"],
                planning_result["data"],
                script_result["data"],
                validation_result,
                voice_result,
                audio_result,
                generation_state,
            )

            await send_progress("completed", 100, "Generation completed successfully")

            final_result = {
                "success": True,
                "podcast_id": podcast_id,
                "generation_id": generation_state["id"],
                "research_data": research_result["data"],
                "content_plan": planning_result["data"],
                "script_data": script_result["data"],
                "voice_data": voice_result["data"]
                if voice_result and voice_result["success"]
                else None,
                "audio_data": audio_result["data"]
                if audio_result and audio_result["success"]
                else None,
                "validation": validation_result,
                "quality_metrics": self._calculate_overall_quality(
                    research_result["data"],
                    planning_result["data"],
                    script_result["data"],
                ),
                "iterations_performed": generation_state.get("iterations", {}),
                "errors_encountered": generation_state.get("errors", []),
                "recovery_actions": generation_state.get("recovery_actions", []),
                "metadata": generation_state,
                "completed_at": datetime.utcnow().isoformat(),
            }

            # Send completion notification via WebSocket
            await websocket_manager.send_generation_complete(
                user_id=podcast.user_id,
                generation_id=generation_state["id"],
                success=True,
                result=final_result,
            )

            self.generation_history.append(final_result)
            self.current_generation = None

            logger.info(f"Enhanced podcast generation completed for ID: {podcast_id}")
            return final_result

        except Exception as e:
            logger.error(
                f"Enhanced podcast generation failed for ID {podcast_id}: {str(e)}"
            )

            # Classify error for better user messaging
            error_details = self.error_handler.classify_error(e)

            # Send detailed error notification via WebSocket
            await websocket_manager.send_error_notification(
                user_id=podcast.user_id,
                generation_id=generation_state["id"],
                error_message=error_details.user_message,
                error_details={
                    "error_code": error_details.error_code,
                    "category": error_details.category.value,
                    "severity": error_details.severity.value,
                    "suggested_action": error_details.suggested_action,
                    "generation_state": generation_state,
                },
            )

            return await self._handle_generation_failure(
                podcast_id, user_id, str(e), generation_state, error_details
            )

    def _initialize_enhanced_state(
        self,
        podcast_id: int,
        user_inputs: Dict[str, Any],
        quality_settings: Optional[Dict],
    ) -> Dict[str, Any]:
        """Initialize enhanced generation state with user inputs"""

        # Merge quality settings
        quality_thresholds = self.quality_thresholds.copy()
        if quality_settings:
            quality_thresholds.update(quality_settings)

        return {
            "id": f"enhanced_gen_{podcast_id}_{int(datetime.utcnow().timestamp())}",
            "podcast_id": podcast_id,
            "started_at": datetime.utcnow().isoformat(),
            "current_phase": "initialization",
            "progress": 0,
            "user_inputs": user_inputs,
            "quality_thresholds": quality_thresholds,
            "phases_completed": [],
            "iterations": {
                "research": 0,
                "planning": 0,
                "script": 0,
                "total_refinements": 0,
            },
            "quality_scores": {},
            "feedback_loops": [],
            "errors": [],
            "warnings": [],
        }

    async def _enhanced_research_phase(
        self, user_inputs: Dict[str, Any], generation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced research phase with user context and iterative improvement"""

        max_iterations = 2
        current_iteration = 0
        best_research = None
        best_score = 0

        while current_iteration < max_iterations:
            try:
                # Extract research parameters from user inputs
                topic = user_inputs.get("topic", "")
                target_duration = user_inputs.get("target_duration", 10)
                content_prefs = user_inputs.get("content_preferences", {})

                # Enhanced research with user context
                research_data = await self._contextual_research(
                    topic, target_duration, content_prefs, generation_state
                )

                if not research_data:
                    current_iteration += 1
                    continue

                # Validate research quality
                validation = self.research_agent.validate_research(research_data)
                quality_score = validation.get("quality_score", 0)

                # Track best result
                if quality_score > best_score:
                    best_research = research_data
                    best_score = quality_score

                # Check if quality threshold is met
                threshold = generation_state["quality_thresholds"]["research_quality"]
                if quality_score >= threshold:
                    generation_state["quality_scores"]["research"] = quality_score
                    generation_state["iterations"]["research"] = current_iteration + 1
                    generation_state["phases_completed"].append("enhanced_research")
                    return {
                        "success": True,
                        "data": research_data,
                        "validation": validation,
                    }

                current_iteration += 1
                generation_state["iterations"]["research"] = current_iteration

                # Add feedback for next iteration
                if current_iteration < max_iterations:
                    feedback = self._generate_research_feedback(
                        validation, content_prefs
                    )
                    generation_state["feedback_loops"].append(
                        {
                            "phase": "research",
                            "iteration": current_iteration,
                            "feedback": feedback,
                            "quality_score": quality_score,
                        }
                    )

            except Exception as e:
                logger.error(f"Research iteration {current_iteration} failed: {e}")
                current_iteration += 1

        # Use best result if no iteration met threshold
        if best_research:
            generation_state["quality_scores"]["research"] = best_score
            generation_state["iterations"]["research"] = max_iterations
            generation_state["warnings"].append(
                f"Research quality ({best_score}) below threshold, using best attempt"
            )
            return {
                "success": True,
                "data": best_research,
                "validation": {"quality_score": best_score},
            }

        return {
            "success": False,
            "error": "Failed to generate acceptable research after all iterations",
        }

    async def _contextual_research(
        self,
        topic: str,
        target_duration: int,
        content_prefs: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Conduct research with user context and preferences"""

        # Build enhanced research prompt with user context
        focus_areas = content_prefs.get("focus_areas", [])
        avoid_topics = content_prefs.get("avoid_topics", [])
        target_audience = content_prefs.get("target_audience", "general public")

        enhanced_prompt = f"""
        Conduct comprehensive research for a {target_duration}-minute podcast about: "{topic}"
        
        USER CONTEXT:
        - Target Audience: {target_audience}
        - Focus Areas: {", ".join(focus_areas) if focus_areas else "general coverage"}
        - Avoid: {", ".join(avoid_topics) if avoid_topics else "none specified"}
        
        Research Requirements:
        1. Generate 3-5 subtopics that align with focus areas
        2. Ensure content is appropriate for target audience
        3. Avoid or minimize coverage of specified topics to avoid
        4. Include recent developments and practical applications
        5. Provide engaging facts and discussion angles
        
        Format as comprehensive research data with subtopics, facts, and discussion angles.
        """

        # Use existing research agent with enhanced context
        return self.research_agent.research_topic(
            main_topic=topic, target_length=target_duration, depth="standard"
        )

    async def _content_planning_phase(
        self,
        research_data: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Strategic content planning with user preferences"""

        try:
            target_duration = user_inputs.get("target_duration", 10)
            style_prefs = user_inputs.get("style_preferences", {})

            # Create audience preferences for content planning
            audience_preferences = {
                "engagement_style": style_prefs.get("tone", "conversational"),
                "complexity_level": style_prefs.get("complexity", "accessible"),
                "pacing": style_prefs.get("pacing", "moderate"),
                "humor_level": style_prefs.get("humor_level", "light"),
            }

            content_plan = self.content_planning_agent.plan_content(
                research_data=research_data,
                target_duration=target_duration,
                audience_preferences=audience_preferences,
            )

            if not content_plan:
                return {"success": False, "error": "Failed to generate content plan"}

            # Validate content plan
            validation = self.content_planning_agent.validate_content_plan(content_plan)
            quality_score = validation.get("quality_score", 0)

            # Check quality threshold
            threshold = generation_state["quality_thresholds"]["content_plan_quality"]
            if quality_score >= threshold:
                generation_state["quality_scores"]["content_plan"] = quality_score
                generation_state["phases_completed"].append("content_planning")
                return {"success": True, "data": content_plan, "validation": validation}
            else:
                generation_state["warnings"].append(
                    f"Content plan quality ({quality_score}) below threshold"
                )
                return {"success": True, "data": content_plan, "validation": validation}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _iterative_script_generation(
        self,
        research_data: Dict[str, Any],
        content_plan: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
        progress_update_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Enhanced script generation with progress updates"""
        try:
            max_iterations = generation_state["quality_thresholds"].get(
                "maximum_iterations", 3
            )

            for iteration in range(max_iterations):
                if progress_update_callback:
                    progress = 50 + (
                        iteration * 10
                    )  # 50-80% range for script generation
                    await progress_update_callback(
                        "script_generation",
                        progress,
                        f"Script generation iteration {iteration + 1}/{max_iterations}",
                    )

                # Generate script (adapting to ScriptAgent's method signature)
                script_data = self.script_agent.generate_script(
                    research_data=research_data,
                    target_length=user_inputs.get("target_duration", 10),
                    host_personalities=user_inputs.get("hosts"),
                    style_preferences=user_inputs.get("style_preferences"),
                )

                script_result = {
                    "success": script_data is not None,
                    "data": script_data,
                    "error": "Script generation failed"
                    if script_data is None
                    else None,
                }

                if not script_result["success"]:
                    continue

                # Quality validation
                quality_scores = self._validate_script_quality(
                    script_result["data"], user_inputs, generation_state
                )

                # Check if quality meets threshold
                if quality_scores["overall_score"] >= generation_state[
                    "quality_thresholds"
                ].get("minimum_coherence_score", 0.75):
                    return {
                        "success": True,
                        "data": script_result["data"],
                        "quality_scores": quality_scores,
                        "iterations_used": iteration + 1,
                    }

                # Store feedback for next iteration
                generation_state["script_feedback"] = quality_scores.get("feedback", {})

            # If we reach here, max iterations exceeded
            return {
                "success": False,
                "error": f"Script generation did not meet quality standards after {max_iterations} iterations",
                "last_attempt": script_result if "script_result" in locals() else None,
            }

        except Exception as e:
            logger.error(f"Error in iterative script generation: {e}")
            return {"success": False, "error": str(e)}

    def _validate_script_quality(
        self,
        script_data: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Basic script quality validation"""
        if not script_data:
            return {"overall_score": 0.0, "feedback": {"error": "No script data"}}

        # Use ScriptAgent's built-in validation
        validation_result = self.script_agent.validate_script(script_data)

        # Return adapted quality scores
        return {
            "overall_score": validation_result.get("quality_score", 0.0)
            / 100.0,  # Convert to 0-1 scale
            "feedback": validation_result,
            "is_valid": validation_result.get("is_valid", False),
        }

    async def _voice_generation_phase(
        self,
        script_data: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
        progress_update_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate voice audio for script segments using ElevenLabs TTS

        Args:
            script_data: Generated script with segments
            user_inputs: User inputs including host configurations
            generation_state: Current generation state

        Returns:
            Voice generation result
        """
        try:
            logger.info("Starting voice generation phase")

            # Create a debug marker file to prove this method was called
            import time

            debug_marker = f"voice_generation_called_{int(time.time())}.txt"
            with open(debug_marker, "w") as f:
                f.write(f"Voice generation phase called at {time.time()}")
            logger.info(f"Created debug marker: {debug_marker}")

            # Check if voice agent is available
            if not self.voice_agent.is_available():
                return {
                    "success": False,
                    "error": "Voice generation service not available",
                    "data": None,
                }

            # Extract script segments for voice generation
            logger.info(
                f"Extracting voice segments from script_data with keys: {list(script_data.keys())}"
            )
            script_segments = self._extract_voice_segments(script_data, user_inputs)
            logger.info(f"Extracted {len(script_segments)} voice segments")

            if not script_segments:
                logger.error(
                    f"No voice segments extracted from script_data: {script_data}"
                )
                return {
                    "success": False,
                    "error": "No valid script segments found for voice generation",
                    "data": None,
                }

            # Estimate cost before generation
            cost_estimate = await self.voice_agent.estimate_generation_cost(
                script_segments
            )
            logger.info(
                f"Voice generation cost estimate: ${cost_estimate.get('estimated_cost_usd', 0):.4f}"
            )

            # Generate voice segments
            voice_result = await self.voice_agent.generate_voice_segments(
                script_segments=script_segments,
                voice_settings=user_inputs.get("voice_settings"),
                include_cost_estimate=True,
            )

            if voice_result.success:
                generation_state["phases_completed"].append("voice_generation")
                generation_state["quality_scores"]["voice_generation"] = (
                    100  # Successful generation
                )

                # Prepare voice data for response
                voice_data = {
                    "segments": [
                        {
                            "segment_id": seg.get(
                                "segment_id", f"seg_{seg.get('segment_index', i)}"
                            ),
                            "text": seg.get("text", ""),
                            "speaker": seg.get("speaker", seg.get("speaker_id", "")),
                            "voice_id": seg.get("voice_id", ""),
                            "duration_estimate": seg.get(
                                "duration_estimate", seg.get("duration", 0)
                            ),
                            "character_count": seg.get(
                                "character_count", len(seg.get("text", ""))
                            ),
                            "audio_size_bytes": len(seg.get("audio_data", b"")),
                            "file_path": seg.get("file_path", ""),
                            "file_url": seg.get("file_url", ""),
                            "timestamp": seg.get("timestamp", time.time()),
                        }
                        for i, seg in enumerate(voice_result.audio_segments)
                    ],
                    "total_duration": voice_result.total_duration,
                    "total_characters": sum(
                        len(seg.get("text", "")) for seg in voice_result.audio_segments
                    ),
                    "total_cost": voice_result.total_cost,
                    "generation_time": voice_result.processing_time,
                    "segments_count": len(voice_result.audio_segments),
                }

                logger.info(
                    f"Voice generation completed: {len(voice_result.audio_segments)} segments, {voice_result.total_duration:.1f}s total"
                )

                return {
                    "success": True,
                    "data": voice_data,
                    "cost_estimate": cost_estimate,
                    "generation_result": voice_result,
                }
            else:
                logger.error(f"Voice generation failed: {voice_result.error_message}")
                return {
                    "success": False,
                    "error": voice_result.error_message,
                    "data": None,
                    "cost_estimate": cost_estimate,
                }

        except Exception as e:
            logger.error(f"Voice generation phase failed: {e}")
            return {"success": False, "error": str(e), "data": None}

    def _extract_voice_segments(
        self, script_data: Dict[str, Any], user_inputs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract voice segments from script data for TTS generation

        Args:
            script_data: Generated script with segments
            user_inputs: User inputs including host configurations

        Returns:
            List of voice segments ready for TTS
        """
        voice_segments = []

        try:
            # Get host configurations
            hosts_config = user_inputs.get("hosts", {})

            # Extract segments from script
            script_segments = script_data.get("segments", [])

            for segment in script_segments:
                # Skip non-dialogue segments
                if segment.get("type") not in [
                    "intro",
                    "dialogue",
                    "main_content",
                    "outro",
                ]:
                    continue

                # Extract dialogue items from segment
                dialogue_items = segment.get("dialogue", [])

                # If no dialogue array, try to extract from segment directly
                if not dialogue_items:
                    speaker = segment.get("speaker", "host_1")
                    text = segment.get("text", "").strip()
                    if text and len(text) >= 10:
                        dialogue_items = [{"speaker": speaker, "text": text}]

                # Process each dialogue item
                for dialogue_item in dialogue_items:
                    speaker = dialogue_item.get("speaker", "host_1")
                    text = dialogue_item.get("text", "").strip()

                    # Skip empty or very short segments
                    if len(text) < 10:
                        continue

                    # Map speaker to voice configuration
                    voice_id = None
                    host_config = None

                    # First try direct speaker name lookup
                    if speaker in hosts_config:
                        host_config = hosts_config[speaker]
                        voice_id = host_config.get("voice_id")
                    else:
                        # Try to find speaker by name in host configurations
                        for host_key, host_data in hosts_config.items():
                            if host_data.get("name") == speaker:
                                host_config = host_data
                                voice_id = host_data.get("voice_id")
                                break

                    # If still no match, use default mapping
                    if not host_config:
                        voice_id = None

                    voice_segments.append(
                        {
                            "text": text,
                            "speaker": speaker,
                            "voice_id": voice_id,
                            "segment_type": segment.get("type", "dialogue"),
                            "subtopic": segment.get("subtopic", ""),
                            "original_segment": segment,
                            "original_dialogue_item": dialogue_item,
                        }
                    )

            logger.info(f"Extracted {len(voice_segments)} voice segments from script")
            return voice_segments

        except Exception as e:
            logger.error(f"Failed to extract voice segments: {e}")
            return []

    async def _comprehensive_validation(
        self,
        research_data: Dict[str, Any],
        content_plan: Dict[str, Any],
        script_data: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Comprehensive validation of all components"""

        validation_result = {
            "overall_valid": True,
            "component_validations": {},
            "coherence_analysis": {},
            "quality_summary": {},
            "recommendations": [],
        }

        # Validate each component
        research_validation = self.research_agent.validate_research(research_data)
        content_validation = self.content_planning_agent.validate_content_plan(
            content_plan
        )

        from .script_agent import ScriptAgent

        script_agent = ScriptAgent()
        script_validation = script_agent.validate_script(script_data)

        validation_result["component_validations"] = {
            "research": research_validation,
            "content_plan": content_validation,
            "script": script_validation,
        }

        # Analyze coherence between components
        coherence_score = self._analyze_multi_component_coherence(
            research_data, content_plan, script_data
        )
        validation_result["coherence_analysis"] = coherence_score

        # Generate quality summary
        quality_scores = generation_state.get("quality_scores", {})
        validation_result["quality_summary"] = {
            "research_quality": quality_scores.get("research", 0),
            "content_plan_quality": quality_scores.get("content_plan", 0),
            "script_quality": quality_scores.get("script", 0),
            "coherence_score": coherence_score.get("overall_coherence", 0),
            "iterations_used": generation_state.get("iterations", {}),
        }

        return validation_result

    def _analyze_multi_component_coherence(
        self,
        research_data: Dict[str, Any],
        content_plan: Dict[str, Any],
        script_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze coherence across all pipeline components"""

        coherence_result = {
            "research_to_plan": 0,
            "plan_to_script": 0,
            "research_to_script": 0,
            "overall_coherence": 0,
            "alignment_issues": [],
        }

        try:
            # Check research to content plan alignment
            research_topics = [
                st.get("title", "") for st in research_data.get("subtopics", [])
            ]
            plan_topics = [
                section.get("subtopic", "")
                for section in content_plan.get("content_structure", {}).get(
                    "main_content", []
                )
            ]

            if research_topics and plan_topics:
                topic_overlap = len(set(research_topics).intersection(set(plan_topics)))
                coherence_result["research_to_plan"] = (
                    topic_overlap / len(research_topics)
                ) * 100

            # Check content plan to script alignment
            script_segments = script_data.get("segments", [])
            script_topics = [
                seg.get("subtopic", "")
                for seg in script_segments
                if seg.get("type") == "main_content"
            ]

            if plan_topics and script_topics:
                script_overlap = len(set(plan_topics).intersection(set(script_topics)))
                coherence_result["plan_to_script"] = (
                    script_overlap / len(plan_topics)
                ) * 100

            # Overall coherence
            coherence_result["overall_coherence"] = (
                coherence_result["research_to_plan"] * 0.4
                + coherence_result["plan_to_script"] * 0.6
            )

        except Exception as e:
            logger.error(f"Coherence analysis failed: {e}")
            coherence_result["overall_coherence"] = 0
            coherence_result["alignment_issues"].append(f"Analysis failed: {str(e)}")

        return coherence_result

    def _calculate_overall_quality(
        self,
        research_data: Dict[str, Any],
        content_plan: Dict[str, Any],
        script_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""

        return {
            "research_completeness": len(research_data.get("subtopics", [])),
            "content_structure_quality": len(
                content_plan.get("content_structure", {}).get("main_content", [])
            ),
            "script_word_count": script_data.get("script_metadata", {}).get(
                "estimated_word_count", 0
            ),
            "estimated_quality_score": (
                self.current_generation.get("quality_scores", {}).get("research", 0)
                * 0.3
                + self.current_generation.get("quality_scores", {}).get(
                    "content_plan", 0
                )
                * 0.3
                + self.current_generation.get("quality_scores", {}).get("script", 0)
                * 0.4
            ),
        }

    async def _handle_phase_failure(
        self,
        phase_name: str,
        result: Dict[str, Any],
        generation_state: Dict[str, Any],
        user_id: int,
    ) -> Dict[str, Any]:
        """Enhanced phase failure handling with user notifications"""

        error_msg = result.get("error", f"{phase_name} phase failed")
        generation_state["errors"].append(
            {
                "phase": phase_name,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Send user-friendly error notification
        await websocket_manager.send_error_notification(
            user_id=user_id,
            generation_id=generation_state["id"],
            error_message=f"Generation failed during {phase_name} phase",
            error_details={
                "phase": phase_name,
                "user_message": f"We encountered an issue during the {phase_name} phase. Please try again.",
                "technical_error": error_msg,
                "retry_suggested": True,
            },
        )

        return {
            "success": False,
            "failed_phase": phase_name,
            "error": error_msg,
            "generation_state": generation_state,
            "failed_at": datetime.utcnow().isoformat(),
        }

    async def _handle_generation_failure(
        self,
        podcast_id: int,
        user_id: int,
        error: str,
        generation_state: Dict[str, Any],
        error_details: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Enhanced generation failure handling with detailed error information"""

        try:
            self.podcast_service.update_podcast_status(podcast_id, "failed", user_id)
        except Exception as update_error:
            logger.error(f"Failed to update podcast status: {update_error}")

        error_result = {
            "success": False,
            "podcast_id": podcast_id,
            "error": error,
            "error_details": {
                "code": error_details.error_code if error_details else "UNKNOWN_ERROR",
                "category": error_details.category.value
                if error_details
                else "unknown",
                "user_message": error_details.user_message
                if error_details
                else "An unexpected error occurred",
                "retry_recommended": error_details.is_retryable
                if error_details
                else True,
            },
            "generation_state": generation_state,
            "failed_at": datetime.utcnow().isoformat(),
        }

        self.generation_history.append(error_result)
        self.current_generation = None

        return error_result

    async def _save_enhanced_results(
        self,
        podcast_id: int,
        user_id: int,
        research_data: Dict[str, Any],
        content_plan: Dict[str, Any],
        script_data: Dict[str, Any],
        validation_result: Dict[str, Any],
        voice_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save enhanced generation results"""

        try:
            # Convert script to text
            script_text = self._convert_enhanced_script_to_text(
                script_data, content_plan
            )

            # Create comprehensive metadata
            metadata = {
                "research_data": research_data,
                "content_plan": content_plan,
                "script_data": script_data,
                "validation": validation_result,
                "generation_metadata": generation_state,
                "quality_metrics": self._calculate_overall_quality(
                    research_data, content_plan, script_data
                ),
            }

            # Prepare audio data for database update
            audio_url = None
            audio_file_paths = []
            has_audio = False
            audio_segments_count = 0
            audio_total_duration = 0
            voice_generation_cost = None

            # Process voice generation results
            if (
                voice_result
                and voice_result.get("success")
                and voice_result.get("data")
            ):
                voice_data = voice_result["data"]

                # Extract voice segments file paths
                voice_segments = voice_data.get("segments", [])
                for segment in voice_segments:
                    if segment.get("file_path"):
                        audio_file_paths.append(segment["file_path"])

                audio_segments_count = len(voice_segments)
                audio_total_duration = voice_data.get("total_duration", 0)
                voice_generation_cost = str(voice_data.get("total_cost", 0))

            # Process audio assembly results (final podcast episode)
            if (
                audio_result
                and audio_result.get("success")
                and audio_result.get("data")
            ):
                audio_data = audio_result["data"]

                # Add the final assembled audio file path
                if audio_data.get("final_audio_path"):
                    audio_file_paths.append(audio_data["final_audio_path"])
                    audio_url = audio_data.get("final_audio_url")
                    has_audio = True

                    # Use assembly duration if available, otherwise use voice duration
                    if audio_data.get("total_duration"):
                        audio_total_duration = audio_data["total_duration"]

            elif voice_result and voice_result.get("success"):
                # We have voice segments but no assembled episode
                has_audio = len(audio_file_paths) > 0

            # Update podcast with all data including audio information
            update_data = PodcastUpdate(
                script=script_text,
                status="completed",
                audio_url=audio_url,
                has_audio=has_audio,
                audio_file_paths=audio_file_paths if audio_file_paths else None,
                audio_segments_count=audio_segments_count
                if audio_segments_count > 0
                else None,
                audio_total_duration=audio_total_duration
                if audio_total_duration > 0
                else None,
                voice_generation_cost=voice_generation_cost,
            )

            updated_podcast = self.podcast_service.update_podcast(
                podcast_id=podcast_id,
                user_id=user_id,
                update_data=update_data,
            )

            generation_state["phases_completed"].append("save_enhanced_results")

            logger.info(
                f"Saved podcast results: has_audio={has_audio}, file_paths={len(audio_file_paths)}, duration={audio_total_duration}s"
            )

            return {
                "success": True,
                "podcast": updated_podcast,
                "saved_metadata": metadata,
                "audio_info": {
                    "has_audio": has_audio,
                    "audio_url": audio_url,
                    "file_paths_count": len(audio_file_paths),
                    "segments_count": audio_segments_count,
                    "total_duration": audio_total_duration,
                },
            }

        except Exception as e:
            logger.error(f"Failed to save enhanced results: {e}")
            return {"success": False, "error": str(e)}

    def _convert_enhanced_script_to_text(
        self, script_data: Dict[str, Any], content_plan: Dict[str, Any]
    ) -> str:
        """Convert enhanced script data to readable text with content plan context"""

        if not script_data or "segments" not in script_data:
            return "Enhanced script generation failed"

        text_parts = []
        title = script_data.get("title", "Untitled Podcast")
        text_parts.append(f"# {title}\n")

        # Add content plan summary
        if content_plan:
            text_parts.append("## Content Plan Summary")
            plan_summary = self.content_planning_agent.get_content_plan_summary(
                content_plan
            )
            text_parts.append(plan_summary + "\n")

        # Add script content
        text_parts.append("## Podcast Script\n")

        for segment in script_data["segments"]:
            segment_type = segment.get("type", "segment")
            text_parts.append(f"\n### {segment_type.title()}\n")

            # Add timing information if available
            duration = segment.get("duration_estimate", 0)
            if duration:
                text_parts.append(f"*Estimated duration: {duration} minutes*\n")

            for line in segment.get("dialogue", []):
                speaker = line.get("speaker", "Unknown")
                text = line.get("text", "")
                text_parts.append(f"**{speaker}:** {text}\n")

        return "\n".join(text_parts)

    async def _execute_with_recovery(
        self,
        func: Callable,
        recovery_type: str,
        progress_callback: Callable,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute function with recovery strategy on failure"""

        # Configure retry based on function type
        if "research" in func.__name__:
            retry_config = RetryConfig(max_attempts=3, base_delay=2.0)
        elif "content_planning" in func.__name__:
            retry_config = RetryConfig(max_attempts=2, base_delay=1.5)
        elif "script_generation" in func.__name__:
            retry_config = RetryConfig(max_attempts=3, base_delay=3.0, max_delay=30.0)
        elif "voice_generation" in func.__name__:
            retry_config = RetryConfig(max_attempts=2, base_delay=5.0)
        else:
            retry_config = RetryConfig(max_attempts=2, base_delay=1.0)

        try:
            result = await self.error_handler.execute_with_retry(
                func,
                *args,
                retry_config=retry_config,
                context={"function": func.__name__, "recovery_type": recovery_type},
                progress_callback=progress_callback,
                **kwargs,
            )
            return result

        except Exception as e:
            logger.warning(
                f"Function {func.__name__} failed after retries, attempting recovery"
            )

            # Attempt recovery strategy
            recovery_strategy = self.recovery_strategies.get(recovery_type)
            if recovery_strategy and recovery_strategy["fallback_strategy"]:
                try:
                    await progress_callback(
                        f"Attempting recovery: {recovery_strategy['user_message']}"
                    )

                    # Apply recovery strategy
                    recovery_result = await self._apply_recovery_strategy(
                        func.__name__, recovery_strategy, *args, **kwargs
                    )

                    if recovery_result and recovery_result.get("success"):
                        # Record recovery action
                        if "recovery_actions" not in kwargs.get("generation_state", {}):
                            kwargs["generation_state"]["recovery_actions"] = []
                        kwargs["generation_state"]["recovery_actions"].append(
                            {
                                "phase": func.__name__,
                                "strategy": recovery_strategy["fallback_strategy"],
                                "message": recovery_strategy["user_message"],
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

                        await progress_callback(
                            "Recovery successful, continuing generation..."
                        )
                        return recovery_result

                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")

            # If recovery fails, re-raise original error
            raise e

    async def _apply_recovery_strategy(
        self, function_name: str, strategy: Dict, *args, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Apply specific recovery strategy based on function and error type"""

        fallback_strategy = strategy["fallback_strategy"]

        if (
            function_name == "_enhanced_research_phase"
            and fallback_strategy == "use_simplified_research"
        ):
            # Simplified research with reduced requirements
            return await self._simplified_research_fallback(*args, **kwargs)

        elif (
            function_name == "_content_planning_phase"
            and fallback_strategy == "use_basic_structure"
        ):
            # Basic content structure fallback
            return await self._basic_content_planning_fallback(*args, **kwargs)

        elif (
            function_name == "_iterative_script_generation"
            and fallback_strategy == "regenerate_with_lower_standards"
        ):
            # Lower quality standards for script generation
            return await self._simplified_script_generation_fallback(*args, **kwargs)

        elif (
            function_name == "_voice_generation_phase"
            and fallback_strategy == "skip_voice_generation"
        ):
            # Skip voice generation and return success
            return {
                "success": True,
                "data": None,
                "message": "Voice generation skipped due to errors",
            }

        return None

    async def _simplified_research_fallback(
        self, user_inputs: Dict, generation_state: Dict
    ) -> Dict[str, Any]:
        """Simplified research strategy with basic topic exploration"""
        try:
            topic = user_inputs.get("topic", "General Discussion")

            # Basic research data structure
            simplified_research = {
                "main_topic": topic,
                "subtopics": [
                    f"Introduction to {topic}",
                    f"Key aspects of {topic}",
                    f"Discussion points about {topic}",
                    f"Conclusion on {topic}",
                ],
                "key_points": [
                    f"Understanding {topic}",
                    f"Exploring different perspectives",
                    f"Practical applications",
                    f"Future considerations",
                ],
                "sources": ["Fallback research strategy"],
                "research_quality": "simplified",
            }

            return {
                "success": True,
                "data": simplified_research,
                "validation": {"quality_score": 0.6},
            }

        except Exception as e:
            logger.error(f"Simplified research fallback failed: {e}")
            return {"success": False, "error": str(e)}

    async def _basic_content_planning_fallback(
        self, research_data: Dict, user_inputs: Dict, generation_state: Dict
    ) -> Dict[str, Any]:
        """Basic content planning with simple structure"""
        try:
            topic = research_data.get("main_topic", "Discussion")
            duration = user_inputs.get("target_duration", 10)

            basic_plan = {
                "title": f"Discussion on {topic}",
                "estimated_duration": duration,
                "content_structure": {
                    "introduction": f"Welcome to our discussion about {topic}",
                    "main_content": [
                        {"section": "Overview", "duration": duration * 0.2},
                        {"section": "Key Discussion", "duration": duration * 0.6},
                        {"section": "Conclusion", "duration": duration * 0.2},
                    ],
                },
                "key_themes": research_data.get("subtopics", [topic]),
                "planning_quality": "basic",
            }

            return {
                "success": True,
                "data": basic_plan,
                "validation": {"quality_score": 0.65},
            }

        except Exception as e:
            logger.error(f"Basic content planning fallback failed: {e}")
            return {"success": False, "error": str(e)}

    async def _simplified_script_generation_fallback(
        self,
        research_data: Dict,
        content_plan: Dict,
        user_inputs: Dict,
        generation_state: Dict,
        **kwargs,
    ) -> Dict[str, Any]:
        """Simplified script generation with lower quality standards"""
        try:
            # Lower quality thresholds for fallback
            original_threshold = generation_state["quality_thresholds"][
                "minimum_script_quality"
            ]
            generation_state["quality_thresholds"]["minimum_script_quality"] = 0.6

            # Generate basic script (adapting to ScriptAgent's method signature)
            script_data = self.script_agent.generate_script(
                research_data=research_data,
                target_length=user_inputs.get("target_duration", 10),
                host_personalities=user_inputs.get("hosts"),
                style_preferences=user_inputs.get("style_preferences"),
            )

            script_result = {
                "success": script_data is not None,
                "data": script_data,
                "error": "Script generation failed" if script_data is None else None,
            }

            # Restore original threshold
            generation_state["quality_thresholds"]["minimum_script_quality"] = (
                original_threshold
            )

            if script_result.get("success"):
                return {
                    "success": True,
                    "data": script_result["data"],
                    "quality_scores": {"overall_score": 0.6},
                    "iterations_used": 1,
                    "fallback_used": True,
                }
            else:
                return {
                    "success": False,
                    "error": "Simplified script generation failed",
                }

        except Exception as e:
            logger.error(f"Simplified script generation fallback failed: {e}")
            return {"success": False, "error": str(e)}

    async def _update_progress(
        self, callback: Optional[Callable], message: str, progress: int
    ):
        """Update generation progress"""
        if self.current_generation:
            self.current_generation["current_phase"] = message
            self.current_generation["progress"] = progress

        logger.info(f"Enhanced Progress: {progress}% - {message}")

        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress, message)
                else:
                    callback(progress, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def get_enhanced_generation_status(
        self, generation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get enhanced generation status with detailed metrics"""

        if generation_id:
            for gen in self.generation_history:
                if gen.get("generation_id") == generation_id:
                    return gen
            return {"error": "Generation not found"}
        else:
            return self.current_generation or {"status": "No active generation"}

    async def _audio_assembly_phase(
        self,
        voice_result: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assemble voice segments into a complete podcast episode using Audio Agent

        Args:
            voice_result: Voice generation result with segments
            user_inputs: User inputs for customization
            generation_state: Current generation state

        Returns:
            Audio assembly result
        """
        try:
            logger.info("Starting audio assembly phase")

            # Check if audio agent is available
            if not self.audio_agent.is_available():
                return {
                    "success": False,
                    "error": "Audio assembly not available - PyDub required",
                    "data": None,
                }

            # Extract voice segments for assembly
            voice_segments = voice_result.get("data", {}).get("segments", [])
            if not voice_segments:
                return {
                    "success": False,
                    "error": "No voice segments found for assembly",
                    "data": None,
                }

            # Prepare episode metadata
            episode_metadata = {
                "title": user_inputs.get("topic", "Untitled Podcast"),
                "generation_id": generation_state["id"],
                "user_inputs": user_inputs,
                "voice_generation_cost": voice_result.get("data", {}).get(
                    "total_cost", 0
                ),
                "voice_segments_count": len(voice_segments),
                "voice_total_duration": voice_result.get("data", {}).get(
                    "total_duration", 0
                ),
            }

            # Prepare audio processing options
            audio_options = {
                # Default intro/outro settings
                "add_intro": user_inputs.get("audio_options", {}).get(
                    "add_intro", True
                ),
                "add_outro": user_inputs.get("audio_options", {}).get(
                    "add_outro", True
                ),
                "intro_style": user_inputs.get("audio_options", {}).get(
                    "intro_style", "overlay"
                ),
                "outro_style": user_inputs.get("audio_options", {}).get(
                    "outro_style", "overlay"
                ),
                # Audio asset selection
                "intro_asset_id": user_inputs.get("audio_options", {}).get(
                    "intro_asset_id", "default_intro"
                ),
                "outro_asset_id": user_inputs.get("audio_options", {}).get(
                    "outro_asset_id", "default_outro"
                ),
                # Effects settings
                "add_transitions": user_inputs.get("audio_options", {}).get(
                    "add_transitions", False
                ),
                "transition_asset_id": user_inputs.get("audio_options", {}).get(
                    "transition_asset_id", "default_transition"
                ),
                # Background music (optional)
                "add_background_music": user_inputs.get("audio_options", {}).get(
                    "add_background_music", False
                ),
                "background_asset_id": user_inputs.get("audio_options", {}).get(
                    "background_asset_id"
                ),
            }

            # Assemble podcast episode with audio options
            assembly_result = await self.audio_agent.assemble_podcast_episode(
                voice_segments=voice_segments,
                podcast_id=str(generation_state["podcast_id"]),
                episode_metadata=episode_metadata,
                audio_options=audio_options,
            )

            if assembly_result.success:
                generation_state["phases_completed"].append("audio_assembly")
                generation_state["quality_scores"]["audio_assembly"] = (
                    100  # Successful assembly
                )

                # Prepare assembly data for response
                assembly_data = {
                    "final_audio_path": assembly_result.final_audio_path,
                    "final_audio_url": assembly_result.final_audio_url,
                    "total_duration": assembly_result.total_duration,
                    "segments_processed": assembly_result.segments_processed,
                    "processing_time": assembly_result.processing_time,
                    "file_size_bytes": assembly_result.file_size_bytes,
                    "metadata": assembly_result.metadata,
                    "audio_processing_applied": {
                        "normalization": True,
                        "compression": True,
                        "silence_removal": True,
                        "speaker_transitions": True,
                        "intro_music": audio_options.get("add_intro", False),
                        "outro_music": audio_options.get("add_outro", False),
                        "transition_effects": audio_options.get(
                            "add_transitions", False
                        ),
                        "background_music": audio_options.get(
                            "add_background_music", False
                        ),
                    },
                    "audio_options_used": audio_options,
                }

                logger.info(
                    f"Audio assembly completed: {assembly_result.total_duration:.1f}s final episode, "
                    f"{assembly_result.segments_processed} segments processed with enhanced audio processing"
                )

                return {
                    "success": True,
                    "data": assembly_data,
                    "processing_result": assembly_result,
                }
            else:
                logger.error(f"Audio assembly failed: {assembly_result.error_message}")
                return {
                    "success": False,
                    "error": assembly_result.error_message,
                    "data": None,
                }

        except Exception as e:
            logger.error(f"Audio assembly phase failed: {e}")
            return {"success": False, "error": str(e), "data": None}
