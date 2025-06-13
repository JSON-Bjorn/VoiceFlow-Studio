from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import asyncio
import json
import time
import hashlib
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
from .duration_calculator import DurationCalculator
from .content_depth_analyzer import ContentDepthAnalyzer
from .voice_agent import voice_agent
from .audio_agent import AudioAgent
from .voice_name_resolver import VoiceNameResolver
from .voice_assignment_validator import VoiceAssignmentValidator
from .agents.resource_management_agent import ResourceManagementAgent
from .agents.reliability_agent import ReliabilityAgent
from .agents.voice_personality_agent import VoicePersonalityAgent
from .agents.agent_coordination import AgentCoordinator, IntelligentPipelineOrchestrator
from .database.agent_data_store import AgentDataStore
from sqlalchemy.orm import Session
from .error_handler import (
    error_handler,
    RetryConfig,
    ErrorCategory,
    with_error_handling,
)

logger = logging.getLogger(__name__)

# Set up file logging for detailed debugging
import os

os.makedirs("storage", exist_ok=True)
file_handler = logging.FileHandler("storage/generation_debug.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Create a separate logger for file-only detailed logging
file_logger = logging.getLogger("generation_debug")
file_logger.setLevel(logging.DEBUG)
file_logger.addHandler(file_handler)
file_logger.propagate = False  # Don't send to console


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
        self.voice_agent = voice_agent
        self.audio_agent = AudioAgent()

        # Initialize intelligent agents
        self.resource_agent = ResourceManagementAgent()
        self.reliability_agent = ReliabilityAgent()
        self.voice_personality_agent = VoicePersonalityAgent()
        self.agent_coordinator = AgentCoordinator()
        self.intelligent_orchestrator = IntelligentPipelineOrchestrator(db)
        self.agent_data_store = AgentDataStore(db)

        # Initialize duration-aware components
        self.duration_calculator = DurationCalculator()
        self.content_analyzer = ContentDepthAnalyzer()

        # Enhanced error handling
        self.error_handler = error_handler
        self.recovery_strategies = self._initialize_recovery_strategies()

        # Generation state
        self.current_generation = None
        self.generation_history = []

        # NEW: Voice profile cache to prevent unnecessary refreshes
        self.voice_profile_cache = {
            "profiles": None,
            "cache_time": None,
            "user_inputs_hash": None,
        }

        # Quality thresholds
        self.quality_thresholds = {
            "research_quality": 0.7,
            "minimum_research_quality": 0.7,
            "minimum_content_plan_quality": 0.75,
            "minimum_script_quality": 0.8,
            "minimum_coherence_score": 0.75,
            "content_plan_quality": 0.7,
            "maximum_iterations": 3,
            "target_script_length_variance": 0.15,  # 15% variance allowed
            "duration_accuracy_threshold": 85.0,  # 85% accuracy required
            "content_completeness_threshold": 80.0,  # 80% content completeness required
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

    def setup_voice_profiles_from_user_inputs(
        self, user_inputs: Dict[str, Any]
    ) -> None:
        """
        Setup voice profiles from user inputs with intelligent caching

        Args:
            user_inputs: User configuration including host voice selections
        """
        try:
            logger.info("🎭 Setting up voice profiles from user inputs")

            # Extract hosts configuration
            hosts = user_inputs.get("hosts", {})

            if not hosts:
                logger.warning("❌ No hosts configuration found in user inputs")
                return

            # Convert hosts configuration to voice agent format
            voice_profiles = {}

            for host_id, host_config in hosts.items():
                if "voice_id" in host_config:
                    # Look up the actual voice name from the database using voice_id
                    actual_voice_name = self._get_voice_name_from_id(
                        host_config["voice_id"]
                    )

                    # If database lookup fails, extract a clean name from the voice_id
                    if not actual_voice_name:
                        actual_voice_name = self._extract_clean_name_from_voice_id(
                            host_config["voice_id"],
                            host_config.get("name", f"Host {host_id.split('_')[-1]}"),
                        )

                    voice_profiles[host_id] = {
                        "voice_id": host_config["voice_id"],
                        "name": actual_voice_name
                        or host_config.get("name", f"Host {host_id.split('_')[-1]}"),
                        "personality": host_config.get("personality", "conversational"),
                        "role": host_config.get("role", "speaker"),
                    }
                    logger.info(
                        f"🎯 Configured voice profile for {host_id}: {host_config['voice_id']} -> '{voice_profiles[host_id]['name']}'"
                    )

            if voice_profiles:
                # Set voice profiles in voice agent
                self.voice_agent.voice_profiles = voice_profiles

                # Update cache
                user_inputs_hash = hashlib.md5(
                    str(sorted(user_inputs.items())).encode()
                ).hexdigest()
                self.voice_profile_cache = {
                    "profiles": voice_profiles.copy(),
                    "cache_time": time.time(),
                    "user_inputs_hash": user_inputs_hash,
                }

                logger.info(
                    f"✅ Voice profiles setup completed: {list(voice_profiles.keys())}"
                )
                logger.info(
                    f"🔍 Voice agent profiles after setup: {self.voice_agent.voice_profiles}"
                )
            else:
                logger.warning(
                    "❌ No valid voice profiles could be extracted from user inputs"
                )

        except Exception as e:
            logger.error(f"❌ Failed to setup voice profiles: {e}")
            # Don't raise exception - let pipeline continue with empty profiles if needed

    def _get_voice_name_from_id(self, voice_id: str) -> Optional[str]:
        """
        Look up the actual voice name from the database using voice_id

        Args:
            voice_id: The voice ID to look up

        Returns:
            The actual voice name or None if not found
        """
        try:
            from ..models.voice_profile import VoiceProfile

            # Query the database for the voice profile
            voice_profile = (
                self.db.query(VoiceProfile)
                .filter(VoiceProfile.voice_id == voice_id)
                .first()
            )

            if voice_profile:
                return voice_profile.voice_name
            else:
                logger.warning(f"Voice profile not found for voice_id: {voice_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to look up voice name for {voice_id}: {e}")
            return None

    def get_cached_voice_profiles(
        self, user_inputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get cached voice profiles or setup new ones if cache is invalid

        Args:
            user_inputs: User configuration for cache validation

        Returns:
            Dictionary of voice profiles
        """
        try:
            # Generate hash of current user inputs for cache validation
            user_inputs_hash = hashlib.md5(
                str(sorted(user_inputs.items())).encode()
            ).hexdigest()

            # Check if cache is valid
            cache_valid = (
                self.voice_profile_cache.get("profiles") is not None
                and self.voice_profile_cache.get("user_inputs_hash") == user_inputs_hash
                and self.voice_profile_cache.get("cache_time") is not None
                and (time.time() - self.voice_profile_cache["cache_time"])
                < 3600  # 1 hour cache
            )

            if cache_valid:
                logger.info("🎭 Using cached voice profiles")
                return self.voice_profile_cache["profiles"].copy()
            else:
                logger.info(
                    "🎭 Cache invalid or expired, setting up fresh voice profiles"
                )
                self.setup_voice_profiles_from_user_inputs(user_inputs)
                return self.voice_profile_cache.get("profiles", {})

        except Exception as e:
            logger.error(f"❌ Failed to get cached voice profiles: {e}")
            # Fallback to voice agent's current profiles
            return (
                self.voice_agent.get_voice_profiles()
                if hasattr(self.voice_agent, "get_voice_profiles")
                else {}
            )

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

        # Setup user-selected voice profiles before pipeline execution
        self.setup_voice_profiles_from_user_inputs(user_inputs)
        logger.error(f"🔍 CHECKPOINT after setup: {self.voice_agent.voice_profiles}")

        # Log pipeline start with user inputs for debugging
        debug_file = f"pipeline_start_{int(time.time() * 1000000)}.txt"
        try:
            with open(debug_file, "w") as f:
                f.write(f"Enhanced pipeline started at {time.time()}\\n")
                f.write(f"user_inputs: {user_inputs}\\n")
                f.write(f"generate_voice: {user_inputs.get('generate_voice')}\\n")
        except Exception as e:
            logger.warning(f"Could not write debug file: {e}")

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
            logger.error(
                f"🔍 CHECKPOINT before script generation: {self.voice_agent.voice_profiles}"
            )
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
            logger.error(
                f"🔍 CHECKPOINT before voice generation: {self.voice_agent.voice_profiles}"
            )
            voice_result = None
            generate_voice_enabled = user_inputs.get("generate_voice", False)
            voice_agent_available = self.voice_agent.is_available()

            # Create debug file to track condition check
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

            # Log detailed debug info to file
            file_logger.info(f"=== AUDIO ASSEMBLY DEBUG - Podcast {podcast_id} ===")
            file_logger.info(
                f"Voice result: {json.dumps(voice_result, indent=2, default=str)}"
            )
            file_logger.info(
                f"User inputs: {json.dumps(user_inputs, indent=2, default=str)}"
            )
            file_logger.info(
                f"Audio agent available: {self.audio_agent.is_available()}"
            )

            # DEBUG: Log voice_result structure
            if voice_result:
                logger.info(f"  - voice_result keys: {list(voice_result.keys())}")
                if voice_result.get("data"):
                    logger.info(
                        f"  - voice_result.data keys: {list(voice_result['data'].keys())}"
                    )
                    if voice_result["data"].get("segments"):
                        logger.info(
                            f"  - segments count: {len(voice_result['data']['segments'])}"
                        )
                        logger.info(
                            f"  - first segment keys: {list(voice_result['data']['segments'][0].keys()) if voice_result['data']['segments'] else 'no segments'}"
                        )
                        # Log file paths to debug file
                        file_logger.info(f"Voice segments file paths:")
                        for i, seg in enumerate(
                            voice_result["data"]["segments"][:5]
                        ):  # Log first 5
                            file_logger.info(
                                f"  Segment {i}: {seg.get('file_path', 'NO FILE PATH')}"
                            )
                    else:
                        logger.info(f"  - NO SEGMENTS in voice_result.data")
                else:
                    logger.info(f"  - NO DATA in voice_result")
            else:
                logger.info(f"  - voice_result is None")

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
                    logger.info(
                        f"DEBUG: Calling audio assembly with voice_result segments: {len(voice_result.get('data', {}).get('segments', []))}"
                    )
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
                    if audio_result and not audio_result.get("success"):
                        logger.error(
                            f"Audio assembly failed with error: {audio_result.get('error')}"
                        )
                    elif audio_result and audio_result.get("success"):
                        logger.info(
                            f"Audio assembly succeeded! Final path: {audio_result.get('data', {}).get('final_audio_path')}"
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
                podcast_id,
                user_id,
                str(e),
                generation_state,
                error_details,
            )

    async def generate_podcast_with_intelligent_resource_management(
        self,
        podcast_id: int,
        user_id: int,
        user_inputs: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        quality_settings: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced generation with intelligent resource management
        """
        logger.info(f"Starting intelligent podcast generation for ID: {podcast_id}")

        # Get podcast and user info
        podcast = self.podcast_service.get_podcast_by_id(podcast_id, user_id)
        if not podcast:
            raise ValueError(f"Podcast with ID {podcast_id} not found")

        try:
            # Phase 0: Intelligent Resource Planning
            topic = user_inputs.get("topic", "")
            target_length = user_inputs.get("length", 10)
            user_budget = user_inputs.get("budget", 5.0)

            resource_context = {
                "topic": topic,
                "target_length": target_length,
                "user_budget": user_budget,
                "user_id": user_id,
                "quality_requirements": quality_settings or {},
            }

            resource_decision = await self.resource_agent.make_decision(
                resource_context
            )

            # Store decision for learning
            decision_id = await self.agent_data_store.store_decision(
                agent_name="ResourceManagement",
                decision_type="resource_allocation",
                context=resource_context,
                decision=resource_decision.data,
                confidence=resource_decision.confidence,
            )

            logger.info(f"Resource allocation decision: {resource_decision.reasoning}")

            # Helper function to send progress updates
            async def send_progress(
                phase: str, progress: int, message: str, metadata: Optional[Dict] = None
            ):
                if progress_callback:
                    await progress_callback(phase, progress, message)

            # Phase 1: Execute generation with intelligent allocation
            token_allocation = resource_decision.data["token_allocation"]
            generation_start_time = datetime.utcnow()

            await send_progress(
                "resource_planning",
                5,
                f"Allocated {token_allocation['total_tokens']} tokens intelligently",
            )

            # Research phase with allocated tokens
            await send_progress(
                "research", 15, "Conducting research with optimized token allocation"
            )
            research_result = await self._generate_research_with_budget(
                topic, target_length, token_allocation["research_tokens"]
            )

            # Monitor resource usage after research
            research_tokens_used = research_result.get("tokens_used", 0)
            progress_monitor = await self.resource_agent.monitor_generation_progress(
                str(podcast_id), "research", research_tokens_used
            )

            if progress_monitor.get("adjustment_needed"):
                logger.info(
                    f"Resource adjustment recommended: {progress_monitor['recommendation']}"
                )
                # Apply adjustment to remaining allocation
                if progress_monitor["recommendation"] == "reduce_remaining_allocation":
                    scale_factor = progress_monitor["suggested_reduction"]
                    token_allocation["script_tokens"] = int(
                        token_allocation["script_tokens"] * scale_factor
                    )

            # Script phase with allocated tokens
            await send_progress(
                "script", 40, "Generating script with optimized resource allocation"
            )
            script_result = await self._generate_script_with_budget(
                research_result["data"],
                target_length,
                token_allocation["script_tokens"],
                user_inputs,
            )

            # Voice generation (if requested)
            voice_result = None
            if user_inputs.get("generate_voice", False):
                await send_progress(
                    "voice", 70, "Generating voice with standard allocation"
                )
                voice_result = await self.voice_agent.generate_voice_from_script(
                    script_result["data"], user_inputs
                )

            # Audio assembly
            audio_result = None
            if voice_result:
                await send_progress("audio", 90, "Assembling final audio")
                audio_result = await self.audio_agent.assemble_podcast(voice_result)

            # Calculate actual costs and outcomes
            actual_cost = research_result.get("cost", 0) + script_result.get("cost", 0)
            total_tokens_used = research_result.get(
                "tokens_used", 0
            ) + script_result.get("tokens_used", 0)
            generation_time = (
                datetime.utcnow() - generation_start_time
            ).total_seconds()

            # Assess generation quality (simplified)
            quality_score = await self._assess_generation_quality(script_result["data"])

            # Learn from outcome
            outcome = {
                "actual_cost": actual_cost,
                "actual_tokens_used": total_tokens_used,
                "quality_score": quality_score,
                "generation_time": generation_time,
                "success": True,
            }

            await self.resource_agent.learn_from_outcome(resource_decision, outcome)

            # Update decision outcome in database
            await self.agent_data_store.update_decision_outcome(
                decision_id, outcome, actual_cost, quality_score, True
            )

            await send_progress(
                "complete", 100, "Generation completed with intelligent optimization"
            )

            return {
                "success": True,
                "generation_id": f"intelligent_{podcast_id}_{int(datetime.utcnow().timestamp())}",
                "research_data": research_result["data"],
                "script_data": script_result["data"],
                "voice_result": voice_result,
                "audio_result": audio_result,
                "resource_optimization": {
                    "predicted_cost": resource_decision.execution_cost,
                    "actual_cost": actual_cost,
                    "cost_accuracy": abs(
                        resource_decision.execution_cost - actual_cost
                    ),
                    "token_efficiency": total_tokens_used
                    / token_allocation["total_tokens"],
                    "optimization_strategy": resource_decision.data[
                        "optimization_strategy"
                    ],
                    "complexity_score": resource_decision.data["complexity_score"],
                },
                "intelligence_summary": {
                    "agent_confidence": resource_decision.confidence,
                    "reasoning": resource_decision.reasoning,
                    "adjustments_made": 1
                    if progress_monitor.get("adjustment_needed")
                    else 0,
                },
            }

        except Exception as e:
            # Learn from failures too
            failure_outcome = {
                "actual_cost": 0.0,
                "actual_tokens_used": 0,
                "quality_score": 0.0,
                "success": False,
                "error": str(e),
            }

            if "resource_decision" in locals():
                await self.resource_agent.learn_from_outcome(
                    resource_decision, failure_outcome
                )
                if "decision_id" in locals():
                    await self.agent_data_store.update_decision_outcome(
                        decision_id, failure_outcome, 0.0, 0.0, False
                    )

            logger.error(f"Intelligent podcast generation failed: {str(e)}")
            raise e

    async def _generate_research_with_budget(
        self, topic: str, target_length: int, token_budget: int
    ):
        """Generate research with token budget constraint"""
        # This would modify the existing research generation to respect token budget
        # For now, we'll use the existing method and add budget tracking
        research_data = self.research_agent.research_topic(
            main_topic=topic,
            target_length=target_length,
            depth="adaptive",  # Let the agent decide depth based on budget
        )

        # Simulate token usage and cost tracking
        estimated_tokens = min(
            token_budget, len(str(research_data)) // 4
        )  # Rough estimate
        estimated_cost = (estimated_tokens / 1000) * 0.002  # GPT-4 pricing

        return {
            "data": research_data,
            "tokens_used": estimated_tokens,
            "cost": estimated_cost,
        }

    async def _generate_script_with_budget(
        self,
        research_data: dict,
        target_length: int,
        token_budget: int,
        user_inputs: Optional[Dict[str, Any]] = None,
    ):
        """Generate script with token budget constraint"""
        # This would modify the existing script generation to respect token budget

        # Get voice profiles and set up clean voice names if available
        voice_profiles = None
        if hasattr(self, "voice_agent") and self.voice_agent:
            voice_profiles = self.voice_agent.get_voice_profiles()
            # Pass voice agent reference for clean voice names
            self.script_agent._voice_agent_ref = self.voice_agent

        script_data = self.script_agent.generate_script(
            research_data=research_data,
            target_length=target_length,
            voice_profiles=voice_profiles,
            user_inputs=user_inputs,
            use_clean_voice_names=True,
        )

        # Simulate token usage and cost tracking
        estimated_tokens = min(
            token_budget, len(str(script_data)) // 4
        )  # Rough estimate
        estimated_cost = (estimated_tokens / 1000) * 0.002  # GPT-4 pricing

        return {
            "data": script_data,
            "tokens_used": estimated_tokens,
            "cost": estimated_cost,
        }

    async def _assess_generation_quality(self, script_data: dict) -> float:
        """Assess the quality of generated content"""
        # Simple quality assessment - can be improved
        quality_factors = {
            "has_segments": "segments" in script_data
            and len(script_data["segments"]) > 0,
            "has_dialogue": any(
                "dialogue" in segment for segment in script_data.get("segments", [])
            ),
            "reasonable_length": len(str(script_data)) > 1000,
            "has_title": "title" in script_data and len(script_data["title"]) > 0,
        }

        quality_score = sum(quality_factors.values()) / len(quality_factors)
        return quality_score

    # Helper methods for intelligent failure recovery
    def _classify_failure(self, exception) -> str:
        """Classify failure type for intelligent recovery"""
        error_str = str(exception).lower()

        if "rate limit" in error_str or "429" in error_str:
            return "api_rate_limit"
        elif "timeout" in error_str:
            return "api_timeout"
        elif "400" in error_str or "bad request" in error_str:
            return "api_error"
        elif "401" in error_str or "unauthorized" in error_str:
            return "auth_error"
        elif "500" in error_str or "internal server" in error_str:
            return "server_error"
        elif "budget" in error_str or "credit" in error_str:
            return "budget_exceeded"
        else:
            return "unknown_error"

    async def _execute_recovery_strategy(
        self, strategy: dict, generation_state: dict, original_func, *args, **kwargs
    ):
        """Execute the recovery strategy determined by ReliabilityAgent"""

        action = strategy.get("action")

        if action == "exponential_backoff":
            wait_time = strategy.get("wait_time", 5)
            await asyncio.sleep(wait_time)
            # Retry original operation
            return await original_func(*args, **kwargs)

        elif action == "retry_with_smaller_chunks":
            # Reduce token allocation and retry
            if len(args) > 2:  # Check if we have token budget parameter
                reduced_budget = int(args[2] * strategy.get("chunk_reduction", 0.7))
                new_args = args[:2] + (reduced_budget,) + args[3:]
                return await original_func(*new_args, **kwargs)
            else:
                return await original_func(*args, **kwargs)

        elif action == "rollback_to_checkpoint":
            # Restore from checkpoint
            if "checkpoints" in generation_state and generation_state["checkpoints"]:
                latest_checkpoint = list(generation_state["checkpoints"].values())[-1]
                generation_state.update(latest_checkpoint["state"])
                return {"success": True, "data": {}, "cost": 0, "tokens_used": 0}

        elif action == "use_cached_content":
            # Return minimal successful result
            return {
                "success": True,
                "data": {"segments": []},
                "cost": 0,
                "tokens_used": 0,
            }

        elif action == "skip_voice_generation":
            # Skip this phase entirely
            return None

        # Default: return empty result
        return {"success": False, "data": {}, "cost": 0, "tokens_used": 0}

    async def _execute_research_with_intelligence(
        self, user_inputs, generation_state, token_budget
    ):
        """Execute research phase with intelligent monitoring"""
        # Track API call for reliability agent
        self.reliability_agent.track_api_call()

        research_result = await self._generate_research_with_budget(
            user_inputs.get("topic", ""),
            user_inputs.get("duration_minutes", 10),
            token_budget,
        )

        return research_result

    async def _execute_script_with_intelligence(
        self, research_result, content_plan, user_inputs, generation_state, token_budget
    ):
        """Execute script phase with intelligent monitoring"""
        # Track API call for reliability agent
        self.reliability_agent.track_api_call()

        script_result = await self._generate_script_with_budget(
            research_result["data"],
            user_inputs.get("duration_minutes", 10),
            token_budget,
            user_inputs,
        )

        return script_result

    async def _execute_intelligent_refund(
        self, podcast_id: int, user_id: int, context: Dict[str, Any]
    ):
        """Execute intelligent refund based on failure analysis"""
        # This would integrate with the payment/credit system
        # For now, just log the refund recommendation
        logger.info(f"AI-recommended refund for podcast {podcast_id} (user {user_id})")
        # Implementation would calculate partial vs full refund based on completed phases

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
            "target_duration": user_inputs.get(
                "target_duration", 10
            ),  # Add target duration to state
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
            "duration_tracking": {  # Add duration tracking
                "research_completeness": 0,
                "content_plan_allocation": 0,
                "script_accuracy": 0,
                "pipeline_coherence": 0,
            },
        }

    async def _enhanced_research_phase(
        self, user_inputs: Dict[str, Any], generation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced research phase with comprehensive error handling and quality assurance"""
        try:
            topic = user_inputs.get("topic", "")
            target_duration = user_inputs.get("target_duration", 10)
            content_prefs = user_inputs.get("content_preferences", {})

            # NEW: Early content adequacy check
            estimated_content_need = (
                target_duration * 100
            )  # Rough words per minute estimate
            logger.info(
                f"Estimated content need for {target_duration}min: ~{estimated_content_need} words"
            )

            # Try contextual research first
            research_data = await self._contextual_research(
                topic, target_duration, content_prefs, generation_state
            )

            if not research_data:
                # Fallback to enhanced research
                research_data = self.research_agent.research_topic(
                    main_topic=topic,
                    target_length=target_duration,
                    depth="enhanced",
                )

            if not research_data:
                return {"success": False, "error": "Research agent returned no data"}

            # NEW: Extract and store research completeness
            research_completeness = self._extract_research_completeness(
                research_data, target_duration
            )
            generation_state["research_completeness"] = research_completeness

            logger.info(f"Research completeness calculated: {research_completeness}%")

            # NEW: Early termination check for insufficient content
            if research_completeness < 30:
                return {
                    "success": False,
                    "error": "Insufficient research content for target duration",
                    "early_termination": True,
                    "recommendation": f"Consider reducing target duration to {max(5, target_duration * 0.6):.0f} minutes or expanding research topic",
                    "research_completeness": research_completeness,
                }

            # Warning for low content but continue
            if research_completeness < 50:
                generation_state["warnings"].append(
                    f"Research content may be insufficient ({research_completeness}%) - expect shorter duration"
                )

            # Validate research quality
            validation = self.research_agent.validate_research(research_data)
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "error": f"Research validation failed: {validation['issues']}",
                }

            # Check quality threshold
            quality_threshold = generation_state["quality_thresholds"].get(
                "minimum_research_quality", 0.7
            )
            quality_score = validation["quality_score"] / 100.0

            if quality_score < quality_threshold:
                feedback = self._generate_research_feedback(validation, content_prefs)
                generation_state["research_feedback"] = feedback

                # Try research refinement if quality is borderline
                if quality_score >= (quality_threshold - 0.1):
                    # Attempt refinement
                    refined_data = await self._contextual_research(
                        topic, target_duration, content_prefs, generation_state
                    )
                    if refined_data:
                        research_data = refined_data
                        validation = self.research_agent.validate_research(
                            research_data
                        )
                        quality_score = validation["quality_score"] / 100.0

            generation_state["phases_completed"].append("research")
            generation_state["quality_scores"]["research"] = validation["quality_score"]

            return {"success": True, "data": research_data, "validation": validation}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_research_completeness(
        self, research_data: Dict[str, Any], target_duration: int
    ) -> float:
        """Extract research completeness percentage from research data"""

        # Check if research already has completeness data
        if "duration_readiness" in research_data:
            return research_data["duration_readiness"].get("final_completeness", 0)

        if "content_depth_analysis" in research_data:
            return research_data["content_depth_analysis"]["analysis_summary"].get(
                "completeness_percentage", 0
            )

        if "final_content_analysis" in research_data:
            return research_data["final_content_analysis"]["analysis_summary"].get(
                "completeness_percentage", 0
            )

        # Fallback: estimate based on content volume
        total_words = 0
        subtopics = research_data.get("subtopics", [])

        for subtopic in subtopics:
            content = subtopic.get("content", "")
            key_points = subtopic.get("key_points", [])
            total_words += len(content.split()) + sum(
                len(point.split()) for point in key_points
            )

        # Rough estimation: 150 words per minute of content needed
        estimated_words_needed = target_duration * 150
        completeness = min(100, (total_words / estimated_words_needed) * 100)

        logger.info(
            f"Estimated research completeness: {completeness}% ({total_words} words for {target_duration}min target)"
        )

        return completeness

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

        # Use duration-aware research with content depth validation
        try:
            # First attempt: duration-targeted research
            research_data = self.research_agent.research_with_duration_target(
                main_topic=topic,
                target_duration=target_duration,
                depth="moderate_depth",
                quality_threshold=self.quality_thresholds[
                    "content_completeness_threshold"
                ],
            )

            if research_data:
                logger.info(
                    f"Duration-aware research completed with {research_data.get('duration_readiness', {}).get('final_completeness', 0)}% completeness"
                )
                return research_data

        except Exception as e:
            logger.warning(
                f"Duration-aware research failed, falling back to standard: {e}"
            )

        # Fallback: standard research with post-analysis
        standard_research = self.research_agent.research_topic(
            main_topic=topic, target_length=target_duration, depth="standard"
        )

        if standard_research:
            # Analyze content depth for duration target
            depth_analysis = self.content_analyzer.analyze_topic_depth(
                standard_research, target_duration
            )
            standard_research["content_depth_analysis"] = depth_analysis

            # Log content readiness
            completeness = depth_analysis["analysis_summary"]["completeness_percentage"]
            logger.info(
                f"Standard research completed with {completeness}% content completeness"
            )

            if completeness < self.quality_thresholds["content_completeness_threshold"]:
                logger.warning(
                    f"Research content may be insufficient for {target_duration}min target"
                )

        return standard_research

    def _generate_research_feedback(
        self, validation: Dict[str, Any], content_prefs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate feedback for research iteration improvement"""
        feedback = {
            "quality_issues": validation.get("issues", []),
            "suggestions": validation.get("suggestions", []),
            "content_preferences": content_prefs,
            "improvement_areas": [],
        }

        # Add specific improvement suggestions based on quality score
        quality_score = validation.get("quality_score", 0)
        if quality_score < 60:
            feedback["improvement_areas"].append("Increase depth of research")
            feedback["improvement_areas"].append("Add more factual content")
        elif quality_score < 80:
            feedback["improvement_areas"].append("Improve discussion angles")
            feedback["improvement_areas"].append("Enhance subtopic coverage")

        return feedback

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

            # Use duration-aware content planning
            try:
                content_plan = (
                    self.content_planning_agent.plan_content_with_duration_awareness(
                        research_data=research_data,
                        target_duration=target_duration,
                        audience_preferences=audience_preferences,
                        quality_threshold=self.quality_thresholds[
                            "content_completeness_threshold"
                        ],
                    )
                )

                if content_plan:
                    # Log duration validation results
                    duration_validation = content_plan.get("duration_validation", {})
                    if duration_validation.get("validation_passed", False):
                        logger.info(
                            f"Content plan duration validation passed: {duration_validation.get('allocated_duration', 0)}min"
                        )
                    else:
                        logger.warning(
                            f"Content plan duration validation issues: {duration_validation}"
                        )

            except Exception as e:
                logger.warning(
                    f"Duration-aware content planning failed, falling back to standard: {e}"
                )

                # Fallback to standard content planning
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
        """Enhanced script generation with progress updates and cached voice profiles"""
        try:
            max_iterations = generation_state["quality_thresholds"].get(
                "maximum_iterations", 3
            )

            # NEW: Get cached voice profiles once at the start
            voice_profiles = self.get_cached_voice_profiles(user_inputs)
            logger.info(
                f"🎭 Using voice profiles for script generation: {list(voice_profiles.keys())}"
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

                # Generate enhanced script with personality adaptation
                from .script_agent import ScriptAgent
                from .personality_adaptation_agent import PersonalityAdaptationAgent

                script_agent = ScriptAgent()
                personality_agent = PersonalityAdaptationAgent()

                # Pass voice agent reference for clean voice names
                script_agent._voice_agent_ref = self.voice_agent

                # NEW: Use cached profiles instead of forcing refresh
                logger.info(
                    f"🎭 Script generation iteration {iteration + 1} using cached voice profiles"
                )

                # Base script generation with voice-based speaker names
                base_script = script_agent.generate_script(
                    research_data=research_data,
                    target_length=user_inputs.get("target_duration", 10),
                    voice_profiles=voice_profiles,  # Use cached profiles
                    user_inputs=user_inputs,
                    use_clean_voice_names=True,  # Enable clean voice names (e.g., "David", "Marcus")
                )

                script_result = {
                    "success": base_script is not None,
                    "data": base_script,
                    "error": "Script generation failed"
                    if base_script is None
                    else None,
                }

                if not script_result["success"]:
                    logger.warning(
                        f"Script generation attempt {iteration + 1} failed, trying next iteration"
                    )
                    continue

                # NEW: Smart duration auto-adjustment before validation
                duration_adjustment = self._auto_adjust_target_duration(
                    script_result["data"], user_inputs, generation_state
                )
                if duration_adjustment.get("adjusted"):
                    logger.info(f"📏 Duration auto-adjusted: {duration_adjustment}")

                # Quality validation with adaptive thresholds
                quality_scores = self._validate_script_quality(
                    script_result["data"], user_inputs, generation_state
                )

                # NEW: Progressive quality threshold - becomes more lenient with each iteration
                base_threshold = generation_state["quality_thresholds"].get(
                    "minimum_coherence_score", 0.75
                )
                iteration_threshold = base_threshold - (
                    iteration * 0.05
                )  # Reduce by 5% each iteration
                iteration_threshold = max(
                    0.5, iteration_threshold
                )  # Never go below 50%

                logger.info(
                    f"Iteration {iteration + 1}: Quality score {quality_scores['overall_score']:.3f} vs threshold {iteration_threshold:.3f}"
                )

                # Check if quality meets progressive threshold
                if quality_scores["overall_score"] >= iteration_threshold:
                    logger.info(
                        f"✅ Script generation succeeded on iteration {iteration + 1}"
                    )
                    return {
                        "success": True,
                        "data": script_result["data"],
                        "quality_scores": quality_scores,
                        "iterations_used": iteration + 1,
                    }

                # Store feedback for next iteration
                generation_state["script_feedback"] = quality_scores.get("feedback", {})
                logger.info(
                    f"⚠️ Iteration {iteration + 1} below threshold, trying again..."
                )

            # If we reach here, max iterations exceeded
            # NEW: Return the last attempt instead of failing completely
            if "script_result" in locals() and script_result.get("success"):
                logger.warning(
                    f"⚠️ Script generation completed with lower quality after {max_iterations} iterations"
                )
                return {
                    "success": True,
                    "data": script_result["data"],
                    "quality_scores": quality_scores
                    if "quality_scores" in locals()
                    else {"overall_score": 0.5},
                    "iterations_used": max_iterations,
                    "quality_warning": "Script quality below optimal threshold",
                }

            return {
                "success": False,
                "error": f"Script generation failed after {max_iterations} iterations",
                "last_attempt": script_result if "script_result" in locals() else None,
            }

        except Exception as e:
            import traceback

            logger.error(f"Error in iterative script generation: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _validate_script_quality(
        self,
        script_data: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enhanced script quality validation with adaptive duration thresholds"""
        if not script_data:
            return {"overall_score": 0.0, "feedback": {"error": "No script data"}}

        # Use ScriptAgent's built-in validation
        validation_result = self.script_agent.validate_script(script_data)

        # Enhanced duration validation with adaptive thresholds
        target_duration = user_inputs.get("target_duration", 10)
        duration_validation = script_data.get("duration_validation", {})

        # NEW: Calculate adaptive threshold based on research quality
        research_completeness = generation_state.get("research_completeness", 100)
        adaptive_threshold = self._calculate_adaptive_duration_threshold(
            research_completeness
        )

        logger.info(
            f"Using adaptive duration threshold: {adaptive_threshold}% (research completeness: {research_completeness}%)"
        )

        # Calculate duration accuracy score with adaptive threshold
        duration_score = 0.0
        if duration_validation:
            duration_analysis = duration_validation.get("duration_analysis", {})
            accuracy_info = duration_analysis.get("accuracy_info", {})
            duration_accuracy = accuracy_info.get("accuracy_percentage", 0)

            # Use adaptive threshold instead of fixed 85%
            if duration_accuracy >= adaptive_threshold:
                duration_score = min(1.0, duration_accuracy / 100.0)
            else:
                # More forgiving penalty calculation
                duration_score = max(
                    0.4, (duration_accuracy / adaptive_threshold) * 0.9
                )

            logger.info(
                f"Script duration validation: {duration_accuracy}% accuracy (target: {target_duration}min, threshold: {adaptive_threshold}%)"
            )

        # Get base quality score
        base_quality = validation_result.get("quality_score", 0.0) / 100.0

        # NEW: Adaptive weighting based on content availability
        if research_completeness < 60:
            # Prioritize duration over quality when content is limited
            overall_score = (base_quality * 0.5) + (duration_score * 0.5)
        else:
            # Original weighting for good content
            overall_score = (base_quality * 0.7) + (duration_score * 0.3)

        enhanced_feedback = validation_result.copy()
        enhanced_feedback.update(
            {
                "duration_validation": duration_validation,
                "duration_accuracy": duration_accuracy,
                "duration_score": duration_score,
                "base_quality_score": base_quality,
                "adaptive_threshold_used": adaptive_threshold,
                "research_completeness": research_completeness,
                "meets_duration_threshold": duration_score
                >= 0.6,  # More reasonable threshold
            }
        )

        return {
            "overall_score": overall_score,
            "feedback": enhanced_feedback,
            "is_valid": validation_result.get("is_valid", False),
            "duration_accurate": duration_score >= 0.6,  # Lowered from 0.8 to 0.6
        }

    def _calculate_adaptive_duration_threshold(
        self, research_completeness: float
    ) -> float:
        """Calculate adaptive duration threshold based on research quality"""
        base_threshold = 85.0

        if research_completeness < 40:
            return base_threshold * 0.5  # 42.5% - Very forgiving for poor content
        elif research_completeness < 60:
            return base_threshold * 0.6  # 51% - Forgiving for limited content
        elif research_completeness < 80:
            return base_threshold * 0.75  # 63.75% - Moderate for decent content
        else:
            return base_threshold  # 85% - Full standard for good content

    async def _voice_generation_phase(
        self,
        script_data: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
        progress_update_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate voice audio for script segments using Chatterbox TTS

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

            # NEW: Validate voice assignments BEFORE generation
            logger.info("Validating voice assignments...")
            validator = VoiceAssignmentValidator()
            voice_profiles = self.voice_agent.get_voice_profiles()

            validation_result = validator.comprehensive_validation(
                script_segments, voice_profiles, user_inputs
            )

            # Log validation results with detailed debug information
            logger.info(
                f"Voice validation debug info: {validation_result.get('validations', {}).get('speaker_mapping', {}).get('debug_info', {})}"
            )

            if validation_result["overall_valid"]:
                logger.info("✅ Voice assignment validation PASSED")
                if validation_result["summary"]["total_warnings"] > 0:
                    logger.warning(
                        f"Voice validation warnings ({validation_result['summary']['total_warnings']}): {validation_result['summary']['recommendations']}"
                    )
            else:
                logger.error("❌ Voice assignment validation FAILED")
                critical_issues = validation_result["summary"]["critical_issues"]

                # Log each critical issue for debugging
                for issue in critical_issues:
                    logger.error(f"  - {issue}")

                # Check if this is a total failure or if we can continue with warnings
                speaker_mapping = validation_result.get("validations", {}).get(
                    "speaker_mapping", {}
                )
                debug_info = speaker_mapping.get("debug_info", {})

                # Allow continuation if we have at least some mapped speakers
                mapped_speakers = debug_info.get("speaker_voice_mapping", {})
                all_speakers = debug_info.get("all_speakers_found", [])

                if mapped_speakers and len(mapped_speakers) > 0:
                    logger.warning(
                        f"⚠️  Validation failed but continuing anyway - {len(mapped_speakers)} speakers mapped out of {len(all_speakers)}"
                    )
                    logger.warning(
                        f"   Mapped speakers: {list(mapped_speakers.keys())}"
                    )
                    logger.warning(
                        f"   Unmapped speakers: {debug_info.get('unmapped_speakers', [])}"
                    )
                else:
                    # Total failure - no speakers mapped at all
                    logger.error(
                        "🚫 Complete validation failure - no speakers can be mapped to voices"
                    )
                    return {
                        "success": False,
                        "error": f"Voice assignment validation failed completely: {critical_issues}",
                        "validation_details": validation_result,
                        "data": None,
                    }

            # Send validation success notification
            if progress_update_callback:
                await progress_update_callback(
                    "voice_validation",
                    25,
                    f"Voice assignment validation passed: {len(script_segments)} segments validated",
                )

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
                podcast_id=str(generation_state["podcast_id"]),
            )

            if voice_result.success:
                generation_state["phases_completed"].append("voice_generation")
                generation_state["quality_scores"]["voice_generation"] = (
                    100  # Successful generation
                )

                # Post-generation validation: Check that both hosts got voice assignments
                post_validation = validator.validate_voice_balance(
                    voice_result.audio_segments
                )
                if not post_validation["balanced"]:
                    logger.warning(
                        f"Voice balance issues detected: {post_validation['warnings']}"
                    )
                else:
                    logger.info("✅ Voice balance validation passed")

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
                    # Add validation metadata
                    "validation_results": {
                        "pre_generation": validation_result,
                        "post_generation": post_validation,
                    },
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
                }

        except Exception as e:
            logger.error(f"Error in voice generation phase: {e}")
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
            # Get voice profiles for direct mapping
            voice_profiles = self.voice_agent.get_voice_profiles()
            logger.error(f"🎯 VOICE PROFILES FOR EXTRACTION: {voice_profiles}")

            # Create mapping from actual speaker names to voice profile keys
            speaker_name_mapping = {}
            user_hosts = user_inputs.get("hosts", {})

            # Get clean voice names for mapping
            clean_voice_names = self.voice_agent.get_clean_speaker_names()

            # Build mapping from clean voice names to host_1/host_2
            for host_id in ["host_1", "host_2"]:
                # Map clean voice names (e.g., "David" -> "host_1")
                if host_id in clean_voice_names:
                    clean_name = clean_voice_names[host_id]
                    speaker_name_mapping[clean_name] = host_id

                # Also map host_id to itself for direct matches
                speaker_name_mapping[host_id] = host_id

                # Handle custom user names if provided (backwards compatibility)
                if host_id in user_hosts and "name" in user_hosts[host_id]:
                    actual_name = user_hosts[host_id]["name"]
                    speaker_name_mapping[actual_name] = host_id

                # Map voice profile names to host IDs (full names like "David Professional")
                if host_id in voice_profiles and "name" in voice_profiles[host_id]:
                    profile_name = voice_profiles[host_id]["name"]
                    speaker_name_mapping[profile_name] = host_id

            logger.info(f"Speaker name mapping: {speaker_name_mapping}")
            logger.info(f"Clean voice names: {clean_voice_names}")

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
                    original_speaker = dialogue_item.get("speaker", "host_1")
                    text = dialogue_item.get("text", "").strip()

                    # Skip empty or very short segments
                    if len(text) < 10:
                        continue

                    # Map actual speaker name to voice profile key
                    mapped_speaker = speaker_name_mapping.get(
                        original_speaker, "host_1"
                    )

                    # If we still can't find a mapping, try fuzzy matching
                    if (
                        mapped_speaker == "host_1"
                        and original_speaker not in speaker_name_mapping
                    ):
                        # Try to find a partial match
                        for name, host_id in speaker_name_mapping.items():
                            if (
                                name.lower() in original_speaker.lower()
                                or original_speaker.lower() in name.lower()
                            ):
                                mapped_speaker = host_id
                                break

                    # Get voice ID for the mapped speaker
                    if mapped_speaker in voice_profiles:
                        voice_id = voice_profiles[mapped_speaker]["voice_id"]
                    else:
                        # Final fallback to host_1
                        voice_id = voice_profiles.get("host_1", {}).get("voice_id")
                        mapped_speaker = "host_1"

                    voice_segments.append(
                        {
                            "text": text,
                            "speaker": mapped_speaker,  # Use mapped speaker for validation
                            "original_speaker": original_speaker,  # Keep original for reference
                            "voice_id": voice_id,
                            "segment_type": segment.get("type", "dialogue"),
                            "subtopic": segment.get("subtopic", ""),
                            "original_segment": segment,
                            "original_dialogue_item": dialogue_item,
                        }
                    )

            logger.info(f"Extracted {len(voice_segments)} voice segments from script")
            logger.info(
                f"Speaker mapping summary: {set(vs['speaker'] for vs in voice_segments)}"
            )

            # Debug: Log first few segments
            for i, seg in enumerate(voice_segments[:3]):
                logger.error(
                    f"🎯 SEGMENT {i}: speaker={seg.get('speaker')}, voice_id={seg.get('voice_id')}"
                )

            return voice_segments

        except Exception as e:
            logger.error(f"Failed to extract voice segments: {e}")
            return []

    def _comprehensive_duration_validation(
        self,
        research_data: Dict[str, Any],
        content_plan: Dict[str, Any],
        script_data: Dict[str, Any],
        target_duration: float,
    ) -> Dict[str, Any]:
        """
        Comprehensive duration validation across all pipeline components

        Args:
            research_data: Research data with content analysis
            content_plan: Content plan with duration allocation
            script_data: Script data with duration validation
            target_duration: Target duration in minutes

        Returns:
            Comprehensive duration validation results
        """

        validation_result = {
            "overall_duration_valid": False,
            "target_duration": target_duration,
            "component_validations": {},
            "duration_accuracy": {},
            "recommendations": [],
            "pipeline_duration_coherence": 0.0,
        }

        try:
            # 1. Research Content Depth Validation
            research_analysis = research_data.get(
                "content_depth_analysis"
            ) or research_data.get("final_content_analysis")
            if research_analysis:
                research_completeness = research_analysis["analysis_summary"][
                    "completeness_percentage"
                ]
                validation_result["component_validations"]["research"] = {
                    "content_completeness": research_completeness,
                    "sufficient_for_duration": research_completeness
                    >= self.quality_thresholds["content_completeness_threshold"],
                    "word_count": research_analysis["analysis_summary"][
                        "current_words"
                    ],
                    "required_words": research_analysis["analysis_summary"][
                        "required_words"
                    ],
                }

                if (
                    research_completeness
                    < self.quality_thresholds["content_completeness_threshold"]
                ):
                    validation_result["recommendations"].append(
                        f"Research content insufficient ({research_completeness}%) - consider expanding research"
                    )

            # 2. Content Plan Duration Allocation Validation
            content_duration_validation = content_plan.get("duration_validation", {})
            if content_duration_validation:
                plan_duration_valid = content_duration_validation.get(
                    "validation_passed", False
                )
                allocated_duration = content_duration_validation.get(
                    "allocated_duration", 0
                )

                validation_result["component_validations"]["content_plan"] = {
                    "duration_allocation_valid": plan_duration_valid,
                    "allocated_duration": allocated_duration,
                    "target_duration": target_duration,
                    "allocation_accuracy": (allocated_duration / target_duration * 100)
                    if target_duration > 0
                    else 0,
                }

                if not plan_duration_valid:
                    validation_result["recommendations"].append(
                        f"Content plan duration allocation needs adjustment ({allocated_duration}min vs {target_duration}min target)"
                    )

            # 3. Script Duration Accuracy Validation
            script_duration_validation = script_data.get("duration_validation", {})
            if script_duration_validation:
                duration_analysis = script_duration_validation.get(
                    "duration_analysis", {}
                )
                accuracy_info = duration_analysis.get("accuracy_info", {})

                script_duration_accuracy = accuracy_info.get("accuracy_percentage", 0)
                estimated_duration = accuracy_info.get("estimated_duration", 0)

                validation_result["component_validations"]["script"] = {
                    "duration_accuracy": script_duration_accuracy,
                    "estimated_duration": estimated_duration,
                    "target_duration": target_duration,
                    "meets_threshold": script_duration_accuracy
                    >= self.quality_thresholds["duration_accuracy_threshold"],
                    "duration_breakdown": duration_analysis.get(
                        "duration_breakdown", {}
                    ),
                }

                if (
                    script_duration_accuracy
                    < self.quality_thresholds["duration_accuracy_threshold"]
                ):
                    validation_result["recommendations"].append(
                        f"Script duration accuracy below threshold ({script_duration_accuracy}% vs {self.quality_thresholds['duration_accuracy_threshold']}% required)"
                    )

            # 4. Calculate Overall Duration Accuracy
            accuracies = []

            # Research component (content completeness as proxy for duration support)
            if research_analysis:
                research_score = min(
                    100,
                    research_analysis["analysis_summary"]["completeness_percentage"],
                )
                accuracies.append(research_score)

            # Content plan component (allocation accuracy)
            if content_duration_validation:
                plan_score = min(
                    100,
                    validation_result["component_validations"]["content_plan"][
                        "allocation_accuracy"
                    ],
                )
                accuracies.append(plan_score)

            # Script component (duration accuracy)
            if script_duration_validation:
                script_score = validation_result["component_validations"]["script"][
                    "duration_accuracy"
                ]
                accuracies.append(script_score)

            # Overall pipeline coherence for duration
            if accuracies:
                overall_accuracy = sum(accuracies) / len(accuracies)
                validation_result["duration_accuracy"]["overall_percentage"] = (
                    overall_accuracy
                )
                validation_result["duration_accuracy"]["component_scores"] = {
                    "research_content": accuracies[0] if len(accuracies) > 0 else 0,
                    "content_planning": accuracies[1] if len(accuracies) > 1 else 0,
                    "script_generation": accuracies[2] if len(accuracies) > 2 else 0,
                }
                validation_result["pipeline_duration_coherence"] = overall_accuracy

                # Determine if overall validation passes
                validation_result["overall_duration_valid"] = (
                    overall_accuracy
                    >= self.quality_thresholds["duration_accuracy_threshold"]
                )

            # 5. Generate Summary Recommendations
            if validation_result["overall_duration_valid"]:
                validation_result["recommendations"].append(
                    "✅ Pipeline duration validation passed - excellent duration accuracy achieved"
                )
            else:
                validation_result["recommendations"].append(
                    "⚠️ Pipeline duration validation needs improvement"
                )

                # Add specific improvement suggestions
                if (
                    research_analysis
                    and research_analysis["analysis_summary"]["completeness_percentage"]
                    < 80
                ):
                    validation_result["recommendations"].append(
                        "• Expand research content to support target duration"
                    )

                if script_duration_validation:
                    script_accuracy = validation_result["component_validations"][
                        "script"
                    ]["duration_accuracy"]
                    if script_accuracy < 85:
                        validation_result["recommendations"].append(
                            "• Optimize script content for better duration accuracy"
                        )

        except Exception as e:
            logger.error(f"Duration validation failed: {e}")
            validation_result["recommendations"].append(
                f"Duration validation error: {str(e)}"
            )

        return validation_result

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

        # Add comprehensive duration validation
        target_duration = generation_state.get("target_duration", 10)
        duration_validation = self._comprehensive_duration_validation(
            research_data, content_plan, script_data, target_duration
        )
        validation_result["duration_validation"] = duration_validation

        # Update overall validity with duration consideration
        duration_valid = duration_validation.get("overall_duration_valid", False)
        if not duration_valid:
            validation_result["overall_valid"] = False
            validation_result["recommendations"].extend(
                duration_validation.get("recommendations", [])
            )

        # Generate quality summary
        quality_scores = generation_state.get("quality_scores", {})
        validation_result["quality_summary"] = {
            "research_quality": quality_scores.get("research", 0),
            "content_plan_quality": quality_scores.get("content_plan", 0),
            "script_quality": quality_scores.get("script", 0),
            "coherence_score": coherence_score.get("overall_coherence", 0),
            "duration_accuracy": duration_validation.get("duration_accuracy", {}).get(
                "overall_percentage", 0
            ),
            "duration_valid": duration_validation.get("overall_duration_valid", False),
            "pipeline_duration_coherence": duration_validation.get(
                "pipeline_duration_coherence", 0
            ),
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

    def _auto_adjust_target_duration(
        self,
        script_data: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Smart duration auto-adjustment based on content density and pacing
        """
        try:
            original_duration = user_inputs.get("target_duration", 10)
            estimated_duration = script_data.get("duration_validation", {}).get(
                "estimated_duration", 0
            )

            if not estimated_duration:
                return {"adjusted": False, "reason": "No duration estimate available"}

            # Calculate deviation percentage
            deviation = abs(estimated_duration - original_duration) / original_duration

            # Only adjust if deviation is significant (>15%)
            if deviation <= 0.15:
                return {"adjusted": False, "reason": "Duration within acceptable range"}

            # Analyze content density for adjustment strategy
            segments = script_data.get("segments", [])
            total_words = sum(
                len(line.get("text", "").split())
                for segment in segments
                for line in segment.get("dialogue", [])
            )

            # Calculate words per minute (average speaking pace: 150-180 wpm)
            if estimated_duration > 0:
                current_wpm = total_words / estimated_duration
            else:
                current_wpm = 150  # Default

            # Determine optimal speaking pace based on content type
            content_complexity = self._assess_content_complexity(script_data)
            if content_complexity == "high":
                target_wpm = 140  # Slower for complex topics
            elif content_complexity == "low":
                target_wpm = 170  # Faster for simple topics
            else:
                target_wpm = 155  # Standard pace

            # Calculate adjusted duration
            adjusted_duration = total_words / target_wpm

            # Apply constraints (±30% of original duration)
            min_duration = original_duration * 0.7
            max_duration = original_duration * 1.3
            adjusted_duration = max(min_duration, min(max_duration, adjusted_duration))

            # Update user inputs and generation state
            user_inputs["target_duration"] = adjusted_duration
            generation_state["duration_adjustments"] = generation_state.get(
                "duration_adjustments", []
            )
            generation_state["duration_adjustments"].append(
                {
                    "original": original_duration,
                    "estimated": estimated_duration,
                    "adjusted": adjusted_duration,
                    "reason": "Smart auto-adjustment",
                    "content_complexity": content_complexity,
                    "wpm_target": target_wpm,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            logger.info(
                f"🎯 Duration auto-adjusted: {original_duration}min → {adjusted_duration:.1f}min "
                f"(complexity: {content_complexity}, target WPM: {target_wpm})"
            )

            return {
                "adjusted": True,
                "original_duration": original_duration,
                "adjusted_duration": adjusted_duration,
                "adjustment_factor": adjusted_duration / original_duration,
                "reason": f"Optimized for {content_complexity} complexity content",
            }

        except Exception as e:
            logger.error(f"Duration auto-adjustment failed: {e}")
            return {"adjusted": False, "error": str(e)}

    def _assess_content_complexity(self, script_data: Dict[str, Any]) -> str:
        """Assess content complexity for duration adjustment"""
        try:
            segments = script_data.get("segments", [])

            # Analyze vocabulary complexity
            all_text = " ".join(
                line.get("text", "")
                for segment in segments
                for line in segment.get("dialogue", [])
            )

            words = all_text.split()
            if not words:
                return "medium"

            # Simple complexity metrics
            avg_word_length = sum(len(word) for word in words) / len(words)
            long_words_ratio = sum(1 for word in words if len(word) > 6) / len(words)

            # Check for technical terms or complex concepts
            technical_indicators = [
                "algorithm",
                "methodology",
                "implementation",
                "optimization",
                "framework",
                "architecture",
                "paradigm",
                "theoretical",
                "comprehensive",
                "sophisticated",
                "fundamental",
                "systematic",
            ]

            technical_ratio = sum(
                1 for word in words if word.lower() in technical_indicators
            ) / len(words)

            # Determine complexity
            if (
                avg_word_length > 5.5
                or long_words_ratio > 0.3
                or technical_ratio > 0.02
            ):
                return "high"
            elif avg_word_length < 4.5 and long_words_ratio < 0.15:
                return "low"
            else:
                return "medium"

        except Exception:
            return "medium"

    async def _intelligent_error_recovery(
        self,
        error: Exception,
        phase: str,
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
        attempt_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Intelligent error recovery with contextual strategies
        """
        try:
            error_type = type(error).__name__
            error_message = str(error)

            logger.info(
                f"🔧 Attempting intelligent recovery for {phase} (attempt {attempt_count})"
            )

            # Analyze error context
            recovery_strategy = self._analyze_error_context(
                error_type, error_message, phase, generation_state
            )

            if not recovery_strategy:
                return {"recovered": False, "reason": "No recovery strategy available"}

            # Apply recovery strategy
            if recovery_strategy["type"] == "parameter_adjustment":
                return await self._apply_parameter_recovery(
                    recovery_strategy, user_inputs, generation_state
                )
            elif recovery_strategy["type"] == "fallback_method":
                return await self._apply_fallback_recovery(
                    recovery_strategy, phase, user_inputs, generation_state
                )
            elif recovery_strategy["type"] == "resource_optimization":
                return await self._apply_resource_recovery(
                    recovery_strategy, user_inputs, generation_state
                )
            else:
                return {"recovered": False, "reason": "Unknown recovery type"}

        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {recovery_error}")
            return {"recovered": False, "error": str(recovery_error)}

    def _analyze_error_context(
        self,
        error_type: str,
        error_message: str,
        phase: str,
        generation_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Analyze error context to determine best recovery strategy"""

        # Duration validation failures
        if "duration" in error_message.lower() and phase == "script_generation":
            return {
                "type": "parameter_adjustment",
                "action": "relax_duration_constraints",
                "parameters": {"duration_tolerance": 0.4},  # Allow 40% deviation
                "reason": "Duration validation too strict",
            }

        # Quality threshold failures
        if "quality" in error_message.lower() or "coherence" in error_message.lower():
            return {
                "type": "parameter_adjustment",
                "action": "reduce_quality_thresholds",
                "parameters": {"quality_reduction": 0.15},  # Reduce by 15%
                "reason": "Quality thresholds too high for content",
            }

        # Research failures
        if phase == "research" and (
            "timeout" in error_message.lower() or "api" in error_message.lower()
        ):
            return {
                "type": "fallback_method",
                "action": "use_cached_research",
                "reason": "Research API issues",
            }

        # Memory/resource failures
        if "memory" in error_message.lower() or "resource" in error_message.lower():
            return {
                "type": "resource_optimization",
                "action": "reduce_processing_load",
                "parameters": {"batch_size_reduction": 0.5},
                "reason": "Resource constraints",
            }

        # Voice generation failures
        if phase == "voice_generation" and "elevenlabs" in error_message.lower():
            return {
                "type": "fallback_method",
                "action": "use_alternative_voice_service",
                "reason": "Primary voice service unavailable",
            }

        return None

    async def _apply_parameter_recovery(
        self,
        strategy: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply parameter adjustment recovery"""
        try:
            action = strategy["action"]
            parameters = strategy.get("parameters", {})

            if action == "relax_duration_constraints":
                # Increase duration tolerance
                tolerance = parameters.get("duration_tolerance", 0.3)
                generation_state["quality_thresholds"]["duration_tolerance"] = tolerance
                logger.info(f"🔧 Relaxed duration tolerance to {tolerance * 100}%")

            elif action == "reduce_quality_thresholds":
                # Reduce quality thresholds across the board
                reduction = parameters.get("quality_reduction", 0.1)
                thresholds = generation_state["quality_thresholds"]

                for threshold_key in [
                    "minimum_coherence_score",
                    "minimum_script_quality",
                ]:
                    if threshold_key in thresholds:
                        original = thresholds[threshold_key]
                        thresholds[threshold_key] = max(0.4, original - reduction)
                        logger.info(
                            f"🔧 Reduced {threshold_key}: {original} → {thresholds[threshold_key]}"
                        )

            return {
                "recovered": True,
                "strategy": action,
                "adjustments": parameters,
                "reason": strategy["reason"],
            }

        except Exception as e:
            return {"recovered": False, "error": str(e)}

    async def _apply_fallback_recovery(
        self,
        strategy: Dict[str, Any],
        phase: str,
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply fallback method recovery"""
        try:
            action = strategy["action"]

            if action == "use_cached_research" and phase == "research":
                # Use simplified research approach
                topic = user_inputs.get("topic", "General Discussion")
                fallback_research = {
                    "main_topic": topic,
                    "subtopics": [
                        f"Introduction to {topic}",
                        f"Key aspects",
                        "Discussion",
                        "Conclusion",
                    ],
                    "key_points": ["Overview", "Analysis", "Insights", "Summary"],
                    "sources": ["Fallback research"],
                    "research_quality": "simplified_fallback",
                    "completeness_score": 0.6,
                }

                generation_state["fallback_research"] = fallback_research
                logger.info("🔧 Applied fallback research strategy")

                return {
                    "recovered": True,
                    "strategy": action,
                    "data": fallback_research,
                    "reason": strategy["reason"],
                }

            elif action == "use_alternative_voice_service":
                # Could implement alternative voice services here
                logger.info(
                    "🔧 Alternative voice service not implemented, continuing without voice"
                )
                return {
                    "recovered": True,
                    "strategy": action,
                    "data": None,
                    "reason": "Voice generation skipped",
                }

            return {"recovered": False, "reason": f"Unknown fallback action: {action}"}

        except Exception as e:
            return {"recovered": False, "error": str(e)}

    async def _apply_resource_recovery(
        self,
        strategy: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply resource optimization recovery"""
        try:
            action = strategy["action"]
            parameters = strategy.get("parameters", {})

            if action == "reduce_processing_load":
                # Reduce batch sizes and complexity
                reduction_factor = parameters.get("batch_size_reduction", 0.5)

                # Apply to various processing parameters
                if "max_iterations" in generation_state["quality_thresholds"]:
                    original_iterations = generation_state["quality_thresholds"][
                        "max_iterations"
                    ]
                    generation_state["quality_thresholds"]["max_iterations"] = max(
                        1, int(original_iterations * reduction_factor)
                    )

                # Reduce target duration if very long
                original_duration = user_inputs.get("target_duration", 10)
                if original_duration > 15:
                    user_inputs["target_duration"] = min(15, original_duration)
                    logger.info(
                        f"🔧 Reduced target duration: {original_duration} → {user_inputs['target_duration']}"
                    )

                logger.info(
                    f"🔧 Applied resource optimization (reduction: {reduction_factor})"
                )

                return {
                    "recovered": True,
                    "strategy": action,
                    "optimizations": {
                        "reduction_factor": reduction_factor,
                        "duration_capped": original_duration > 15,
                    },
                    "reason": strategy["reason"],
                }

            return {"recovered": False, "reason": f"Unknown resource action: {action}"}

        except Exception as e:
            return {"recovered": False, "error": str(e)}

    def _initialize_performance_metrics(self, generation_state: Dict[str, Any]) -> None:
        """Initialize performance metrics tracking"""
        generation_state["performance_metrics"] = {
            "phase_timings": {},
            "iteration_counts": {},
            "quality_improvements": {},
            "resource_usage": {},
            "cache_hits": {},
            "error_recovery_count": 0,
            "optimization_applied": [],
            "start_time": time.time(),
        }

    def _track_phase_performance(
        self,
        phase_name: str,
        start_time: float,
        end_time: float,
        generation_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track performance metrics for a pipeline phase"""
        try:
            if "performance_metrics" not in generation_state:
                self._initialize_performance_metrics(generation_state)

            metrics = generation_state["performance_metrics"]

            # Record timing
            duration = end_time - start_time
            metrics["phase_timings"][phase_name] = {
                "duration_seconds": duration,
                "start_time": start_time,
                "end_time": end_time,
                "metadata": metadata or {},
            }

            # Track iteration counts
            if metadata and "iterations" in metadata:
                metrics["iteration_counts"][phase_name] = metadata["iterations"]

            # Track quality improvements
            if metadata and "quality_score" in metadata:
                if phase_name not in metrics["quality_improvements"]:
                    metrics["quality_improvements"][phase_name] = []
                metrics["quality_improvements"][phase_name].append(
                    metadata["quality_score"]
                )

            # Track cache usage
            if metadata and "cache_used" in metadata:
                metrics["cache_hits"][phase_name] = metadata["cache_used"]

            logger.info(f"📊 Phase '{phase_name}' completed in {duration:.2f}s")

        except Exception as e:
            logger.warning(f"Failed to track performance for {phase_name}: {e}")

    def _calculate_efficiency_score(
        self, generation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall pipeline efficiency score"""
        try:
            metrics = generation_state.get("performance_metrics", {})

            if not metrics:
                return {"efficiency_score": 0.0, "breakdown": {}}

            total_time = time.time() - metrics.get("start_time", time.time())
            phase_timings = metrics.get("phase_timings", {})

            # Time efficiency (faster is better)
            expected_time = 180  # 3 minutes baseline
            time_efficiency = min(1.0, expected_time / max(total_time, 1))

            # Iteration efficiency (fewer iterations is better)
            iteration_counts = metrics.get("iteration_counts", {})
            total_iterations = sum(iteration_counts.values())
            expected_iterations = 3  # 1 iteration per major phase
            iteration_efficiency = min(
                1.0, expected_iterations / max(total_iterations, 1)
            )

            # Quality consistency (stable quality is better)
            quality_improvements = metrics.get("quality_improvements", {})
            quality_efficiency = 1.0  # Default
            if quality_improvements:
                quality_scores = []
                for phase_scores in quality_improvements.values():
                    quality_scores.extend(phase_scores)
                if quality_scores:
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    quality_efficiency = avg_quality

            # Cache efficiency (more cache hits is better)
            cache_hits = metrics.get("cache_hits", {})
            cache_efficiency = sum(1 for hit in cache_hits.values() if hit) / max(
                len(cache_hits), 1
            )

            # Error recovery penalty
            error_recovery_count = metrics.get("error_recovery_count", 0)
            error_penalty = max(0, 1.0 - (error_recovery_count * 0.1))

            # Overall efficiency score
            efficiency_score = (
                time_efficiency * 0.3
                + iteration_efficiency * 0.25
                + quality_efficiency * 0.25
                + cache_efficiency * 0.1
                + error_penalty * 0.1
            )

            return {
                "efficiency_score": round(efficiency_score, 3),
                "breakdown": {
                    "time_efficiency": round(time_efficiency, 3),
                    "iteration_efficiency": round(iteration_efficiency, 3),
                    "quality_efficiency": round(quality_efficiency, 3),
                    "cache_efficiency": round(cache_efficiency, 3),
                    "error_penalty": round(error_penalty, 3),
                },
                "total_time_seconds": round(total_time, 2),
                "total_iterations": total_iterations,
                "error_recoveries": error_recovery_count,
            }

        except Exception as e:
            logger.error(f"Failed to calculate efficiency score: {e}")
            return {"efficiency_score": 0.0, "breakdown": {}, "error": str(e)}

    def _generate_optimization_recommendations(
        self, generation_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for pipeline optimization"""
        try:
            recommendations = []
            metrics = generation_state.get("performance_metrics", {})

            if not metrics:
                return recommendations

            phase_timings = metrics.get("phase_timings", {})
            iteration_counts = metrics.get("iteration_counts", {})

            # Check for slow phases
            for phase, timing in phase_timings.items():
                duration = timing.get("duration_seconds", 0)
                if duration > 60:  # More than 1 minute
                    recommendations.append(
                        {
                            "type": "performance",
                            "priority": "high",
                            "phase": phase,
                            "issue": f"Phase taking {duration:.1f}s (>60s threshold)",
                            "recommendation": "Consider caching, parallel processing, or algorithm optimization",
                            "estimated_impact": "20-50% time reduction",
                        }
                    )

            # Check for excessive iterations
            for phase, iterations in iteration_counts.items():
                if iterations > 2:
                    recommendations.append(
                        {
                            "type": "quality",
                            "priority": "medium",
                            "phase": phase,
                            "issue": f"Phase required {iterations} iterations",
                            "recommendation": "Review quality thresholds or improve initial generation",
                            "estimated_impact": "30-60% iteration reduction",
                        }
                    )

            # Check cache usage
            cache_hits = metrics.get("cache_hits", {})
            cache_misses = sum(1 for hit in cache_hits.values() if not hit)
            if cache_misses > 2:
                recommendations.append(
                    {
                        "type": "caching",
                        "priority": "low",
                        "phase": "general",
                        "issue": f"{cache_misses} cache misses detected",
                        "recommendation": "Improve caching strategy or cache invalidation logic",
                        "estimated_impact": "10-25% time reduction",
                    }
                )

            # Check error recovery usage
            error_count = metrics.get("error_recovery_count", 0)
            if error_count > 1:
                recommendations.append(
                    {
                        "type": "reliability",
                        "priority": "high",
                        "phase": "general",
                        "issue": f"{error_count} error recoveries required",
                        "recommendation": "Investigate root causes and improve error prevention",
                        "estimated_impact": "Improved stability and user experience",
                    }
                )

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []

    async def _execute_with_recovery(
        self,
        phase_func: Callable,
        failure_type: str,
        error_callback: Callable,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a pipeline phase with intelligent error recovery

        Args:
            phase_func: The phase function to execute
            failure_type: Type of failure for recovery strategy selection
            error_callback: Callback for error progress updates
            *args: Arguments to pass to the phase function
            **kwargs: Keyword arguments to pass to the phase function

        Returns:
            Result dictionary with success status and data/error information
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                logger.info(
                    f"🔄 Executing {phase_func.__name__} (attempt {retry_count + 1})"
                )

                # Execute the phase function
                result = await phase_func(*args, **kwargs)

                if result and result.get("success", False):
                    logger.info(f"✅ {phase_func.__name__} completed successfully")
                    return result
                else:
                    # Phase failed, try recovery
                    logger.warning(
                        f"⚠️ {phase_func.__name__} failed: {result.get('error', 'Unknown error')}"
                    )

                    if retry_count < max_retries - 1:
                        # Apply recovery strategy
                        recovery_strategy = self.recovery_strategies.get(
                            failure_type, {}
                        )
                        if recovery_strategy:
                            await error_callback(
                                f"Applying recovery strategy for {failure_type}",
                                {
                                    "strategy": recovery_strategy,
                                    "attempt": retry_count + 1,
                                },
                            )

                            # Apply the recovery strategy
                            recovery_result = await self._execute_recovery_strategy(
                                recovery_strategy,
                                kwargs.get("generation_state", {}),
                                phase_func,
                                *args,
                                **kwargs,
                            )

                            if recovery_result and recovery_result.get("success"):
                                return recovery_result

                        retry_count += 1
                        await asyncio.sleep(
                            min(2**retry_count, 10)
                        )  # Exponential backoff
                    else:
                        return result or {
                            "success": False,
                            "error": f"{phase_func.__name__} failed after all retries",
                        }

            except Exception as e:
                logger.error(f"❌ Exception in {phase_func.__name__}: {e}")

                if retry_count < max_retries - 1:
                    await error_callback(
                        f"Error in {phase_func.__name__}, attempting recovery",
                        {"error": str(e), "attempt": retry_count + 1},
                    )

                    # Try intelligent error recovery
                    recovery_result = await self._intelligent_error_recovery(
                        e,
                        phase_func.__name__,
                        kwargs.get("user_inputs", {}),
                        kwargs.get("generation_state", {}),
                        retry_count + 1,
                    )

                    if recovery_result.get("recovered", False):
                        retry_count += 1
                        continue

                # Final attempt failed
                return {
                    "success": False,
                    "error": f"{phase_func.__name__} failed with exception: {str(e)}",
                    "exception_type": type(e).__name__,
                    "retry_count": retry_count,
                }

        return {
            "success": False,
            "error": f"{phase_func.__name__} failed after {max_retries} attempts",
        }

    async def _handle_generation_failure(
        self,
        podcast_id: int,
        user_id: int,
        error_message: str,
        generation_state: Dict[str, Any],
        error_details: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Handle complete generation failure with user notification and cleanup

        Args:
            podcast_id: ID of the podcast being generated
            user_id: ID of the user
            error_message: Error message describing the failure
            generation_state: Current generation state
            error_details: Additional error details

        Returns:
            Failure result dictionary
        """
        try:
            logger.error(
                f"🚨 Generation failure for podcast {podcast_id}: {error_message}"
            )

            # Update podcast status to failed
            try:
                self.podcast_service.update_podcast_status(
                    podcast_id, "failed", user_id
                )
            except Exception as status_error:
                logger.error(f"Failed to update podcast status: {status_error}")

            # Send final error notification via WebSocket
            try:
                await websocket_manager.send_error_notification(
                    user_id=user_id,
                    generation_id=generation_state.get("id", f"failed_{podcast_id}"),
                    error_message="Podcast generation failed",
                    error_details={
                        "message": error_message,
                        "podcast_id": podcast_id,
                        "generation_state": generation_state,
                        "error_details": str(error_details) if error_details else None,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            except Exception as ws_error:
                logger.error(f"Failed to send WebSocket error notification: {ws_error}")

            # Calculate performance metrics if available
            efficiency_metrics = {}
            if generation_state.get("performance_metrics"):
                try:
                    efficiency_metrics = self._calculate_efficiency_score(
                        generation_state
                    )
                except Exception as metrics_error:
                    logger.error(
                        f"Failed to calculate efficiency metrics: {metrics_error}"
                    )

            # Execute intelligent refund if applicable
            try:
                refund_result = await self._execute_intelligent_refund(
                    podcast_id,
                    user_id,
                    {
                        "error_message": error_message,
                        "generation_state": generation_state,
                        "error_details": error_details,
                    },
                )
                logger.info(f"Refund processing result: {refund_result}")
            except Exception as refund_error:
                logger.error(f"Failed to process refund: {refund_error}")

            # Return comprehensive failure result
            return {
                "success": False,
                "error": error_message,
                "error_type": "generation_failure",
                "podcast_id": podcast_id,
                "generation_id": generation_state.get("id"),
                "failure_phase": generation_state.get("current_phase", "unknown"),
                "progress": generation_state.get("progress", 0),
                "timestamp": datetime.utcnow().isoformat(),
                "efficiency_metrics": efficiency_metrics,
                "recovery_attempted": True,
                "user_notified": True,
                "status_updated": True,
                "error_details": {
                    "message": error_message,
                    "details": str(error_details) if error_details else None,
                    "generation_state": generation_state,
                },
            }

        except Exception as handler_error:
            logger.error(f"Error in generation failure handler: {handler_error}")
            return {
                "success": False,
                "error": f"Generation failed and error handler also failed: {error_message}",
                "handler_error": str(handler_error),
                "podcast_id": podcast_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _audio_assembly_phase(
        self,
        voice_result: Dict[str, Any],
        user_inputs: Dict[str, Any],
        generation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assemble final audio from voice segments

        Args:
            voice_result: Voice generation result with segments
            user_inputs: User configuration
            generation_state: Current generation state

        Returns:
            Audio assembly result
        """
        try:
            logger.info("Starting audio assembly phase")

            if not voice_result.get("success") or not voice_result.get("data"):
                return {"success": False, "error": "Voice result is invalid"}

            voice_data = voice_result["data"]
            segments = voice_data.get("segments", [])

            if not segments:
                return {"success": False, "error": "No voice segments to assemble"}

            logger.info(f"Assembling {len(segments)} voice segments")

            # Get audio options from user inputs
            audio_options = user_inputs.get("audio_options", {})

            # Prepare assembly data
            assembly_data = {
                "voice_segments": segments,
                "audio_options": audio_options,
                "total_duration": voice_data.get("total_duration", 0),
                "podcast_id": generation_state.get("podcast_id"),
            }

            # Call audio agent to assemble podcast
            if not self.audio_agent.is_available():
                logger.warning("Audio agent not available, skipping assembly")
                return {
                    "success": False,
                    "error": "Audio agent not available (PyDub missing?)",
                    "data": assembly_data,
                }

            logger.info("Audio agent is available, proceeding with assembly")

            # Use audio agent to assemble the podcast
            assembly_result = await self.audio_agent.assemble_podcast(
                segments=segments,
                podcast_id=generation_state.get("podcast_id"),
                audio_options=audio_options,
            )

            if assembly_result.get("success"):
                logger.info(f"✅ Audio assembly completed successfully")
                logger.info(
                    f"Final audio path: {assembly_result.get('data', {}).get('final_audio_path')}"
                )

                return {
                    "success": True,
                    "data": {
                        "final_audio_path": assembly_result.get("data", {}).get(
                            "final_audio_path"
                        ),
                        "final_audio_url": assembly_result.get("data", {}).get(
                            "final_audio_url"
                        ),
                        "assembled_duration": assembly_result.get("data", {}).get(
                            "duration"
                        ),
                        "file_size": assembly_result.get("data", {}).get("file_size"),
                        "segments_count": len(segments),
                        "assembly_time": assembly_result.get("data", {}).get(
                            "processing_time"
                        ),
                        "audio_format": "mp3",
                        "metadata": assembly_data,
                    },
                    "assembly_result": assembly_result,
                }
            else:
                logger.error(f"Audio assembly failed: {assembly_result.get('error')}")
                return {
                    "success": False,
                    "error": assembly_result.get("error", "Audio assembly failed"),
                    "data": assembly_data,
                }

        except Exception as e:
            logger.error(f"Audio assembly phase failed: {e}")
            return {
                "success": False,
                "error": f"Audio assembly exception: {str(e)}",
                "exception_type": type(e).__name__,
            }

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
        """
        Save enhanced generation results to database and storage

        Args:
            podcast_id: ID of the podcast
            user_id: ID of the user
            research_data: Research results
            content_plan: Content planning results
            script_data: Script generation results
            validation_result: Validation results
            voice_result: Voice generation results (optional)
            audio_result: Audio assembly results (optional)
            generation_state: Current generation state

        Returns:
            Save operation result
        """
        try:
            logger.info(f"Saving enhanced results for podcast {podcast_id}")

            # Prepare comprehensive result data
            save_data = {
                "podcast_id": podcast_id,
                "user_id": user_id,
                "generation_id": generation_state.get("id"),
                "research_data": research_data,
                "content_plan": content_plan,
                "script_data": script_data,
                "validation_result": validation_result,
                "voice_result": voice_result,
                "audio_result": audio_result,
                "generation_state": generation_state,
                "completed_at": datetime.utcnow().isoformat(),
                "success": True,
            }

            # Update podcast with final content
            try:
                # Save script content to podcast
                if script_data:
                    script_content = {
                        "title": script_data.get("title", "Generated Podcast"),
                        "segments": script_data.get("segments", []),
                        "estimated_duration": script_data.get("estimated_duration", 0),
                        "script_metadata": script_data.get("script_metadata", {}),
                        "validation": validation_result,
                        "generation_metadata": {
                            "generation_id": generation_state.get("id"),
                            "completed_at": datetime.utcnow().isoformat(),
                            "quality_scores": generation_state.get(
                                "quality_scores", {}
                            ),
                            "iterations": generation_state.get("iterations", {}),
                        },
                    }

                    # Update podcast status and content
                    self.podcast_service.update_podcast_content(
                        podcast_id, script_content, user_id
                    )

                    # Update status to completed
                    self.podcast_service.update_podcast_status(
                        podcast_id, "completed", user_id
                    )

                    logger.info(f"✅ Podcast {podcast_id} content and status updated")

            except Exception as update_error:
                logger.error(f"Failed to update podcast content: {update_error}")
                # Continue with save operation even if update fails

            # Save voice files metadata if available
            voice_files_saved = 0
            if voice_result and voice_result.get("success"):
                voice_data = voice_result.get("data", {})
                segments = voice_data.get("segments", [])

                for segment in segments:
                    if segment.get("file_path"):
                        voice_files_saved += 1

                logger.info(f"✅ {voice_files_saved} voice files metadata saved")

            # Save audio file metadata if available
            audio_file_saved = False
            if audio_result and audio_result.get("success"):
                audio_data = audio_result.get("data", {})
                final_audio_path = audio_data.get("final_audio_path")

                if final_audio_path:
                    try:
                        # Update podcast with audio file information
                        from sqlalchemy.orm import Session
                        from ..models.podcast import Podcast

                        # Get podcast from database
                        podcast = (
                            self.db.query(Podcast)
                            .filter(Podcast.id == podcast_id)
                            .first()
                        )
                        if podcast:
                            # Initialize audio_file_paths as list if None
                            if not podcast.audio_file_paths:
                                podcast.audio_file_paths = []

                            # Add the final audio file path to the list
                            if final_audio_path not in podcast.audio_file_paths:
                                podcast.audio_file_paths.append(final_audio_path)

                            # Also add all voice segment files if available
                            if voice_result and voice_result.get("success"):
                                voice_data = voice_result.get("data", {})
                                segments = voice_data.get("segments", [])

                                for segment in segments:
                                    segment_file_path = segment.get("file_path")
                                    if (
                                        segment_file_path
                                        and segment_file_path
                                        not in podcast.audio_file_paths
                                    ):
                                        podcast.audio_file_paths.append(
                                            segment_file_path
                                        )

                            # Set audio flags
                            podcast.has_audio = True
                            podcast.audio_segments_count = voice_files_saved
                            podcast.audio_total_duration = audio_data.get(
                                "assembled_duration"
                            )

                            # Set audio URL to the final assembled file
                            try:
                                from ..services.storage_service import storage_service

                                podcast.audio_url = await storage_service.get_file_url(
                                    final_audio_path
                                )
                            except Exception as url_error:
                                logger.warning(
                                    f"Could not generate audio URL: {url_error}"
                                )
                                # Set a relative URL as fallback
                                podcast.audio_url = (
                                    f"/api/podcasts/{podcast_id}/download"
                                )

                            # Mark the database session as modified
                            self.db.add(podcast)
                            self.db.commit()

                            audio_file_saved = True
                            logger.info(
                                f"✅ Podcast {podcast_id} updated with audio file paths: {len(podcast.audio_file_paths)} files"
                            )
                            logger.info(
                                f"✅ Final audio file saved to database: {final_audio_path}"
                            )
                        else:
                            logger.error(
                                f"Could not find podcast {podcast_id} in database"
                            )

                    except Exception as audio_save_error:
                        logger.error(
                            f"Failed to save audio file metadata to database: {audio_save_error}"
                        )
                        # Still mark as saved for metrics, but log the error
                        audio_file_saved = True
                        logger.info(
                            f"✅ Final audio file metadata saved: {final_audio_path}"
                        )
                else:
                    logger.warning(
                        "Audio assembly successful but no final_audio_path found"
                    )
            else:
                logger.info("No audio assembly result to save")

            # Calculate generation metrics
            metrics = {
                "voice_generation_success": voice_result.get("success", False)
                if voice_result
                else False,
                "audio_assembly_success": audio_result.get("success", False)
                if audio_result
                else False,
                "voice_files_count": voice_files_saved,
                "audio_file_available": audio_file_saved,
                "total_duration": voice_result.get("data", {}).get("total_duration", 0)
                if voice_result
                else 0,
                "script_segments_count": len(script_data.get("segments", []))
                if script_data
                else 0,
                "validation_passed": validation_result.get("overall_valid", False),
                "quality_score": validation_result.get("quality_summary", {}).get(
                    "script_quality", 0
                ),
                "generation_time": (
                    datetime.utcnow()
                    - datetime.fromisoformat(
                        generation_state.get(
                            "started_at", datetime.utcnow().isoformat()
                        )
                    )
                ).total_seconds(),
            }

            logger.info(f"📊 Generation metrics: {metrics}")

            return {
                "success": True,
                "podcast_id": podcast_id,
                "save_data": save_data,
                "metrics": metrics,
                "files_saved": {
                    "voice_files": voice_files_saved,
                    "audio_file": audio_file_saved,
                    "script_content": script_data is not None,
                },
                "database_updated": True,
                "status_updated": True,
                "message": f"Enhanced results saved successfully for podcast {podcast_id}",
            }

        except Exception as e:
            logger.error(f"Failed to save enhanced results: {e}")
            return {
                "success": False,
                "error": f"Save operation failed: {str(e)}",
                "exception_type": type(e).__name__,
                "podcast_id": podcast_id,
                "user_id": user_id,
            }

    # Add this static method for clean name extraction
    @staticmethod
    def _extract_clean_name_from_voice_id(
        voice_id: str, fallback_name: str = "Host"
    ) -> str:
        """
        Extract a clean, human-friendly name from a voice_id string.
        Mimics the logic in VoiceAgent.get_clean_speaker_names.
        """
        if not voice_id:
            return fallback_name
        if voice_id.startswith("system_") and "_" in voice_id:
            parts = voice_id.split("_")
            if len(parts) >= 2:
                return parts[1].capitalize()
            else:
                return fallback_name
        elif "_" in voice_id and not voice_id.startswith("system_"):
            return voice_id.split("_")[0].capitalize()
        else:
            return voice_id.capitalize()
