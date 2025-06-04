from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
import asyncio
from .websocket_manager import websocket_manager
from ..models.podcast import Podcast
from ..services.podcast_service import PodcastService
from .research_agent import ResearchAgent
from .script_agent import ScriptAgent
from .content_planning_agent import ContentPlanningAgent
from .conversation_flow_agent import ConversationFlowAgent
from .dialogue_distribution_agent import DialogueDistributionAgent
from .personality_adaptation_agent import PersonalityAdaptationAgent
from .voice_agent import VoiceAgent
from .audio_agent import AudioAgent
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class EnhancedPipelineOrchestrator:
    """
    Enhanced Pipeline Orchestrator with 6-agent system and quality assurance
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

        # Quality thresholds
        self.quality_thresholds = {
            "minimum_coherence_score": 0.75,
            "minimum_engagement_score": 0.70,
            "maximum_iterations": 3,
            "target_script_length_variance": 0.15,  # 15% variance allowed
        }

        # Generation state tracking
        self.current_generation = None
        self.generation_history = []

    async def generate_enhanced_podcast(
        self,
        podcast_id: int,
        user_inputs: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        quality_settings: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate podcast with enhanced flow and user inputs

        Args:
            podcast_id: ID of the podcast to generate
            user_inputs: Enhanced user inputs including host details, preferences, etc.
            progress_callback: Optional callback for progress updates
            quality_settings: Optional quality thresholds and settings

        Expected user_inputs format:
        {
            "topic": "Main topic",
            "target_duration": 10,
            "hosts": {
                "host_1": {
                    "name": "Felix",
                    "personality": "analytical, curious",
                    "voice_id": "voice_123",
                    "role": "primary_questioner"
                },
                "host_2": {
                    "name": "Bjorn",
                    "personality": "enthusiastic, relatable",
                    "voice_id": "voice_456",
                    "role": "storyteller"
                }
            },
            "style_preferences": {
                "tone": "conversational",
                "complexity": "accessible",
                "humor_level": "light",
                "pacing": "moderate"
            },
            "content_preferences": {
                "focus_areas": ["practical applications", "recent developments"],
                "avoid_topics": ["overly technical details"],
                "target_audience": "general public"
            }
        }
        """
        logger.info(f"Starting enhanced podcast generation for ID: {podcast_id}")

        # Get podcast and user info for WebSocket updates
        podcast = self.podcast_service.get_podcast_by_id(podcast_id)
        if not podcast:
            raise ValueError(f"Podcast with ID {podcast_id} not found")

        # Initialize enhanced generation state
        generation_state = self._initialize_enhanced_state(
            podcast_id, user_inputs, quality_settings
        )
        self.current_generation = generation_state

        # Helper function to send progress updates via WebSocket
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

        try:
            # Phase 1: Enhanced Research with User Context
            await send_progress("research", 10, "Conducting contextual research")
            research_result = await self._enhanced_research_phase(
                user_inputs, generation_state
            )

            if not research_result["success"]:
                await websocket_manager.send_error_notification(
                    user_id=podcast.user_id,
                    generation_id=generation_state["id"],
                    error_message="Research phase failed",
                    error_details=research_result,
                )
                return await self._handle_phase_failure(
                    "research", research_result, generation_state
                )

            # Phase 2: Strategic Content Planning
            await send_progress("planning", 25, "Creating strategic content plan")
            planning_result = await self._content_planning_phase(
                research_result["data"], user_inputs, generation_state
            )

            if not planning_result["success"]:
                await websocket_manager.send_error_notification(
                    user_id=podcast.user_id,
                    generation_id=generation_state["id"],
                    error_message="Content planning failed",
                    error_details=planning_result,
                )
                return await self._handle_phase_failure(
                    "planning", planning_result, generation_state
                )

            # Phase 3: Iterative Script Generation with Feedback
            await send_progress(
                "script_generation", 50, "Generating and refining script"
            )
            script_result = await self._iterative_script_generation(
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
                    "script", script_result, generation_state
                )

            # Phase 4: Voice Generation (Optional)
            voice_result = None
            if (
                user_inputs.get("generate_voice", False)
                and self.voice_agent.is_available()
            ):
                await send_progress("voice_generation", 65, "Generating voice audio")
                voice_result = await self._voice_generation_phase(
                    script_result["data"], user_inputs, generation_state, send_progress
                )

                if voice_result and not voice_result["success"]:
                    logger.warning(
                        f"Voice generation failed: {voice_result.get('error', 'Unknown error')}"
                    )
                    # Continue without voice - this is optional

            # Phase 5: Audio Assembly (Optional - only if voice was generated)
            audio_result = None
            if (
                voice_result
                and voice_result["success"]
                and self.audio_agent.is_available()
                and user_inputs.get(
                    "assemble_audio", True
                )  # Default to True if voice was generated
            ):
                await send_progress(
                    "audio_assembly", 75, "Assembling final audio episode"
                )
                audio_result = await self._audio_assembly_phase(
                    voice_result, user_inputs, generation_state
                )

                if audio_result and not audio_result["success"]:
                    logger.warning(
                        f"Audio assembly failed: {audio_result.get('error', 'Unknown error')}"
                    )
                    # Continue without assembled audio - voice segments are still available

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

            # Send error notification via WebSocket
            await websocket_manager.send_error_notification(
                user_id=podcast.user_id,
                generation_id=generation_state["id"],
                error_message=str(e),
                error_details={"generation_state": generation_state},
            )

            return await self._handle_generation_failure(
                podcast_id, str(e), generation_state
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

                # Generate script
                script_result = await self.script_agent.generate_enhanced_script(
                    research_data=research_data,
                    content_plan=content_plan,
                    user_inputs=user_inputs,
                    iteration=iteration,
                    feedback_from_previous=generation_state.get("script_feedback"),
                )

                if not script_result["success"]:
                    continue

                # Quality validation
                quality_scores = await self._validate_script_quality(
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

            # Check if voice agent is available
            if not self.voice_agent.is_available():
                return {
                    "success": False,
                    "error": "Voice generation service not available",
                    "data": None,
                }

            # Extract script segments for voice generation
            script_segments = self._extract_voice_segments(script_data, user_inputs)

            if not script_segments:
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
                script_segments,
                context={
                    "podcast_id": generation_state["podcast_id"],
                    "generation_id": generation_state["id"],
                    "user_inputs": user_inputs,
                },
                podcast_id=str(generation_state["podcast_id"]),
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
                            "segment_id": seg.segment_id,
                            "text": seg.text,
                            "speaker": seg.speaker,
                            "voice_id": seg.voice_id,
                            "duration_estimate": seg.duration_estimate,
                            "character_count": seg.character_count,
                            "audio_size_bytes": len(seg.audio_data),
                            "file_path": seg.file_path,
                            "file_url": seg.file_url,
                            "timestamp": seg.timestamp.isoformat(),
                        }
                        for seg in voice_result.segments
                    ],
                    "total_duration": voice_result.total_duration,
                    "total_characters": voice_result.total_characters,
                    "total_cost": voice_result.total_cost,
                    "generation_time": voice_result.generation_time,
                    "segments_count": len(voice_result.segments),
                }

                logger.info(
                    f"Voice generation completed: {len(voice_result.segments)} segments, {voice_result.total_duration:.1f}s total"
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
                if segment.get("type") not in ["dialogue", "main_content"]:
                    continue

                # Extract speaker and text
                speaker = segment.get("speaker", "host_1")
                text = segment.get("text", "").strip()

                # Skip empty or very short segments
                if len(text) < 10:
                    continue

                # Map speaker to voice configuration
                if speaker in hosts_config:
                    host_config = hosts_config[speaker]
                    # Use configured voice_id if available
                    voice_id = host_config.get("voice_id")
                else:
                    # Use default mapping
                    voice_id = None

                voice_segments.append(
                    {
                        "text": text,
                        "speaker": speaker,
                        "voice_id": voice_id,
                        "segment_type": segment.get("type", "dialogue"),
                        "subtopic": segment.get("subtopic", ""),
                        "original_segment": segment,
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
        self, phase_name: str, result: Dict[str, Any], generation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle failure in a specific phase"""

        error_msg = result.get("error", f"{phase_name} phase failed")
        generation_state["errors"].append(f"{phase_name}: {error_msg}")

        return {
            "success": False,
            "failed_phase": phase_name,
            "error": error_msg,
            "generation_state": generation_state,
            "failed_at": datetime.utcnow().isoformat(),
        }

    async def _handle_generation_failure(
        self, podcast_id: int, error: str, generation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle overall generation failure"""

        try:
            self.podcast_service.update_podcast_status(podcast_id, "failed")
        except Exception as update_error:
            logger.error(f"Failed to update podcast status: {update_error}")

        error_result = {
            "success": False,
            "podcast_id": podcast_id,
            "error": error,
            "generation_state": generation_state,
            "failed_at": datetime.utcnow().isoformat(),
        }

        self.generation_history.append(error_result)
        self.current_generation = None

        return error_result

    async def _save_enhanced_results(
        self,
        podcast_id: int,
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

            # Update podcast
            updated_podcast = self.podcast_service.update_podcast(
                podcast_id=podcast_id,
                updates={
                    "script": script_text,
                    "status": "completed",
                    # Could add metadata field to store full generation data
                },
            )

            generation_state["phases_completed"].append("save_enhanced_results")

            return {
                "success": True,
                "podcast": updated_podcast,
                "saved_metadata": metadata,
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
