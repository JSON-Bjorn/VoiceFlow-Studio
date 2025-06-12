from typing import Dict, List, Optional, Any
from .openai_service import OpenAIService
from .voice_name_resolver import VoiceNameResolver
from .duration_calculator import DurationCalculator
import logging
import json

logger = logging.getLogger(__name__)


class ScriptAgent:
    """
    Specialized agent for generating podcast scripts using OpenAI

    This agent is responsible for:
    1. Converting research data into natural dialogue
    2. Maintaining consistent host personalities
    3. Creating smooth transitions between topics
    4. Ensuring proper pacing and timing
    """

    def __init__(self):
        self.openai_service = OpenAIService()
        self.duration_calculator = DurationCalculator()
        # Remove hardcoded names - will get names dynamically from voice profiles
        self.default_hosts = {
            "host_1": {
                "name": "Host 1",  # Will be replaced dynamically
                "personality": "analytical and engaging host",
                "role": "primary_questioner",
            },
            "host_2": {
                "name": "Host 2",  # Will be replaced dynamically
                "personality": "warm and curious host",
                "role": "storyteller",
            },
        }

    def get_dynamic_host_config(self, voice_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Get host configuration from voice profiles dynamically"""
        return {
            "host_1": {
                "name": voice_profiles.get("host_1", {}).get("name", "Host 1"),
                "personality": "analytical and engaging host",
                "role": "primary_questioner",
            },
            "host_2": {
                "name": voice_profiles.get("host_2", {}).get("name", "Host 2"),
                "personality": "warm and curious host",
                "role": "storyteller",
            },
        }

    def get_clean_voice_host_config(self, voice_agent) -> Dict[str, Any]:
        """Get host configuration using clean voice names (e.g., 'David', 'Marcus')"""
        clean_names = voice_agent.get_clean_speaker_names()
        return {
            "host_1": {
                "name": clean_names.get("host_1", "Host 1"),
                "personality": "analytical and engaging host",
                "role": "primary_questioner",
            },
            "host_2": {
                "name": clean_names.get("host_2", "Host 2"),
                "personality": "warm and curious host",
                "role": "storyteller",
            },
        }

    def _get_default_host_personalities(self) -> Dict[str, Any]:
        """Define default host personalities"""
        return {
            "host_1": {
                "name": "Host 1",
                "personality": "Curious and analytical, asks probing questions, likes to dig deeper into topics",
                "speaking_style": "Thoughtful, uses phrases like 'That's fascinating' and 'Help me understand'",
                "role": "primary_questioner",
                "voice_characteristics": "calm, measured, slightly lower pitch",
            },
            "host_2": {
                "name": "Host 2",
                "personality": "Enthusiastic and relatable, brings topics down to earth, shares personal connections",
                "speaking_style": "Energetic, uses phrases like 'Oh wow' and 'That reminds me of'",
                "role": "enthusiastic_responder",
                "voice_characteristics": "upbeat, expressive, slightly higher pitch",
            },
        }

    def generate_script(
        self,
        research_data: Dict[str, Any],
        target_length: int = 10,
        host_personalities: Optional[Dict] = None,
        style_preferences: Optional[Dict] = None,
        voice_profiles: Optional[Dict[str, Any]] = None,
        user_inputs: Optional[Dict[str, Any]] = None,
        use_clean_voice_names: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a complete podcast script from research data

        Args:
            research_data: Research data from ResearchAgent
            target_length: Target length in minutes
            host_personalities: Custom host personalities (optional)
            style_preferences: Style preferences for the script
            voice_profiles: Voice profiles for dynamic name resolution
            user_inputs: User inputs for enhanced name resolution
            use_clean_voice_names: Whether to use clean voice names (e.g., "David") as speakers

        Returns:
            Complete podcast script with dialogue and timing
        """
        logger.info(
            f"Generating script for topic: {research_data.get('main_topic', 'Unknown')}"
        )

        # Determine host configuration strategy
        if use_clean_voice_names and hasattr(self, "_voice_agent_ref"):
            # Use clean voice names from voice agent
            logger.error(f"🎭 SCRIPT AGENT: Using clean voice names from voice agent")
            logger.error(
                f"🎭 SCRIPT AGENT: Voice agent profiles: {self._voice_agent_ref.voice_profiles}"
            )
            hosts = self.get_clean_voice_host_config(self._voice_agent_ref)
            logger.error(f"🎭 SCRIPT AGENT: Generated host config: {hosts}")
            logger.info(
                f"Using clean voice names as speakers: {[h['name'] for h in hosts.values()]}"
            )

        elif voice_profiles or user_inputs:
            # Use Voice Name Resolver for enhanced dynamic name resolution (custom names)
            resolver = VoiceNameResolver()

            # Resolve host names dynamically
            if user_inputs:
                resolved_host_names = resolver.resolve_host_names(
                    user_inputs, voice_profiles
                )
                enhanced_host_config = resolver.create_enhanced_host_config(
                    user_inputs, voice_profiles
                )
                hosts = enhanced_host_config

                logger.info(
                    f"Using enhanced dynamic host configuration: {resolved_host_names}"
                )
            else:
                # Fallback to basic voice profile dynamic configuration
                hosts = self.get_dynamic_host_config(voice_profiles)
                logger.info(
                    f"Using basic dynamic host configuration from voice profiles"
                )
        else:
            # Use custom hosts or defaults
            hosts = host_personalities or self.default_hosts
            logger.info(f"Using provided host personalities or defaults")

        # Apply style preferences
        style = self._apply_style_preferences(style_preferences)

        # Generate the script using OpenAI with resolved host names
        script_data = self.openai_service.generate_podcast_script(
            research_data=research_data,
            target_length=target_length,
            host_personalities=hosts,
        )

        if not script_data:
            logger.error("Failed to generate script")
            return None

        # Enhance script with additional metadata and structure
        enhanced_script = self._enhance_script(
            script_data, research_data, style, target_length
        )

        # Add voice resolution metadata for debugging and validation
        enhanced_script["voice_resolution_metadata"] = {
            "resolver_used": True,
            "voice_profiles_available": voice_profiles is not None,
            "user_inputs_provided": user_inputs is not None,
            "use_clean_voice_names": use_clean_voice_names,
            "resolved_hosts": list(hosts.keys()) if hosts else [],
            "host_names": {k: v.get("name", "Unknown") for k, v in hosts.items()}
            if hosts
            else {},
        }

        # Add duration validation
        if "segments" in enhanced_script:
            all_dialogue = []
            for segment in enhanced_script["segments"]:
                all_dialogue.extend(segment.get("dialogue", []))

            duration_validation = self.duration_calculator.validate_duration_accuracy(
                all_dialogue, target_length, tolerance_percentage=15
            )

            enhanced_script["duration_validation"] = duration_validation
            enhanced_script["estimated_duration"] = duration_validation[
                "duration_analysis"
            ]["estimated_duration"]

            # Log duration accuracy
            accuracy_info = duration_validation["duration_analysis"]["accuracy_info"]
            logger.info(
                f"Script duration: {accuracy_info['estimated_duration']:.1f}min (target: {target_length}min, accuracy: {accuracy_info['accuracy_percentage']:.1f}%)"
            )

        logger.info(
            "Script generation completed with voice-based speaker names and duration validation"
        )
        return enhanced_script

    def _apply_style_preferences(
        self, style_preferences: Optional[Dict]
    ) -> Dict[str, Any]:
        """Apply style preferences to script generation"""
        default_style = {
            "tone": "conversational",
            "formality": "casual",
            "pace": "moderate",
            "humor_level": "light",
            "technical_depth": "accessible",
            "interaction_style": "collaborative",
        }

        if style_preferences:
            default_style.update(style_preferences)

        return default_style

    def _enhance_script(
        self,
        script_data: Dict[str, Any],
        research_data: Dict[str, Any],
        style: Dict[str, Any],
        target_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Enhance script with additional metadata and structure

        Args:
            script_data: Base script from OpenAI
            research_data: Original research data
            style: Applied style preferences

        Returns:
            Enhanced script with metadata
        """
        if not script_data:
            return script_data

        # Add script metadata
        script_data["script_metadata"] = {
            "generated_from_research": research_data.get("main_topic"),
            "style_applied": style,
            "total_segments": len(script_data.get("segments", [])),
            "estimated_word_count": self._estimate_word_count(script_data),
            "dialogue_balance": self._analyze_dialogue_balance(script_data),
            "quality_metrics": self._calculate_quality_metrics(script_data),
        }

        # Enhance segments with additional data
        if "segments" in script_data:
            segment_target_duration = (
                target_length / len(script_data["segments"])
                if target_length and script_data["segments"]
                else None
            )
            for segment in script_data["segments"]:
                self._enhance_segment(segment, segment_target_duration)

        return script_data

    def _enhance_segment(
        self, segment: Dict[str, Any], target_duration: Optional[float] = None
    ) -> None:
        """Enhance individual segment with metadata"""
        if "dialogue" not in segment:
            return

        dialogue = segment["dialogue"]

        # Add segment metadata with enhanced duration analysis
        duration_result = self.duration_calculator.estimate_dialogue_duration(
            dialogue, target_duration=target_duration
        )

        segment["segment_metadata"] = {
            "line_count": len(dialogue),
            "estimated_words": duration_result["word_count"],
            "speaker_distribution": self._get_speaker_distribution(dialogue),
            "estimated_reading_time": duration_result["estimated_duration"],
            "duration_breakdown": duration_result,
        }

    def _estimate_word_count(self, script_data: Dict[str, Any]) -> int:
        """Estimate total word count of the script"""
        total_words = 0
        for segment in script_data.get("segments", []):
            for line in segment.get("dialogue", []):
                total_words += len(line.get("text", "").split())
        return total_words

    def _analyze_dialogue_balance(self, script_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the balance of dialogue between speakers"""
        speaker_stats = {}

        for segment in script_data.get("segments", []):
            for line in segment.get("dialogue", []):
                speaker = line.get("speaker", "unknown")
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {"lines": 0, "words": 0}

                speaker_stats[speaker]["lines"] += 1
                speaker_stats[speaker]["words"] += len(line.get("text", "").split())

        return speaker_stats

    def _get_speaker_distribution(
        self, dialogue: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Get speaker distribution for a segment"""
        distribution = {}
        for line in dialogue:
            speaker = line.get("speaker", "unknown")
            distribution[speaker] = distribution.get(speaker, 0) + 1
        return distribution

    def _estimate_reading_time(
        self, dialogue: List[Dict[str, Any]], target_duration: Optional[float] = None
    ) -> float:
        """Estimate reading time for dialogue using advanced duration calculator"""
        duration_result = self.duration_calculator.estimate_dialogue_duration(
            dialogue, target_duration=target_duration
        )
        return duration_result["estimated_duration"]

    def _calculate_quality_metrics(self, script_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the script"""
        segments = script_data.get("segments", [])

        if not segments:
            return {"overall_score": 0, "issues": ["No segments found"]}

        metrics = {
            "overall_score": 0,
            "segment_balance": 0,
            "dialogue_flow": 0,
            "speaker_balance": 0,
            "issues": [],
            "strengths": [],
        }

        # Check segment balance
        segment_durations = [seg.get("duration_estimate", 0) for seg in segments]
        if segment_durations:
            avg_duration = sum(segment_durations) / len(segment_durations)
            variance = sum((d - avg_duration) ** 2 for d in segment_durations) / len(
                segment_durations
            )
            metrics["segment_balance"] = max(0, 100 - variance * 10)

        # Check speaker balance
        speaker_stats = self._analyze_dialogue_balance(script_data)
        if len(speaker_stats) >= 2:
            speaker_words = [stats["words"] for stats in speaker_stats.values()]
            word_balance = (
                min(speaker_words) / max(speaker_words) if max(speaker_words) > 0 else 0
            )
            metrics["speaker_balance"] = word_balance * 100

        # Calculate overall score
        metrics["overall_score"] = (
            metrics["segment_balance"] * 0.3
            + metrics["speaker_balance"] * 0.4
            + 80  # Base score for successful generation
        ) / 1.7

        # Add qualitative assessments
        if metrics["speaker_balance"] > 70:
            metrics["strengths"].append("Good speaker balance")
        else:
            metrics["issues"].append("Uneven speaker distribution")

        if metrics["segment_balance"] > 70:
            metrics["strengths"].append("Well-balanced segments")
        else:
            metrics["issues"].append("Uneven segment timing")

        return metrics

    def refine_script(
        self,
        script_data: Dict[str, Any],
        feedback: str,
        target_aspect: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Refine script based on feedback

        Args:
            script_data: Original script data
            feedback: Feedback on what to improve
            target_aspect: Specific aspect to focus on (dialogue, pacing, etc.)

        Returns:
            Refined script or None if error
        """
        logger.info(f"Refining script based on feedback: {feedback[:50]}...")

        if target_aspect and target_aspect in ["segment", "dialogue"]:
            # Refine specific segments
            return self._refine_segments(script_data, feedback, target_aspect)
        else:
            # Refine entire script
            return self._refine_entire_script(script_data, feedback)

    def _refine_segments(
        self, script_data: Dict[str, Any], feedback: str, target_aspect: str
    ) -> Optional[Dict[str, Any]]:
        """Refine specific segments of the script"""
        # Implementation for segment-specific refinement
        # This could target specific segments based on feedback
        return script_data  # Placeholder

    def _refine_entire_script(
        self, script_data: Dict[str, Any], feedback: str
    ) -> Optional[Dict[str, Any]]:
        """Refine the entire script based on feedback"""
        # Use OpenAI to refine the script
        refined_script = self.openai_service.refine_script_segment(
            segment=script_data,
            feedback=feedback,
            context="Complete podcast script refinement",
        )

        if refined_script:
            # Re-enhance the refined script
            return self._enhance_script(
                refined_script,
                {"main_topic": script_data.get("title", "Unknown")},
                {"tone": "conversational"},
            )

        return None

    def validate_script(self, script_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate script structure and content

        Args:
            script_data: Script data to validate

        Returns:
            Validation results with issues and suggestions
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "quality_score": 0,
        }

        if not script_data:
            validation["is_valid"] = False
            validation["issues"].append("No script data provided")
            return validation

        # Check required fields
        required_fields = ["title", "segments"]
        for field in required_fields:
            if field not in script_data:
                validation["is_valid"] = False
                validation["issues"].append(f"Missing required field: {field}")

        # Validate segments
        segments = script_data.get("segments", [])
        if not segments:
            validation["is_valid"] = False
            validation["issues"].append("No segments found in script")
        else:
            for i, segment in enumerate(segments):
                if "dialogue" not in segment:
                    validation["issues"].append(f"Segment {i + 1} missing dialogue")
                elif not segment["dialogue"]:
                    validation["issues"].append(f"Segment {i + 1} has empty dialogue")

        # Calculate quality score from metadata if available
        if "script_metadata" in script_data:
            quality_metrics = script_data["script_metadata"].get("quality_metrics", {})
            validation["quality_score"] = quality_metrics.get("overall_score", 0)

        return validation

    def get_script_summary(self, script_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the script"""
        if not script_data:
            return "No script data available"

        title = script_data.get("title", "Untitled Podcast")
        segments = script_data.get("segments", [])

        summary = f"Script Summary: '{title}'\n\n"
        summary += f"Total segments: {len(segments)}\n"

        if "script_metadata" in script_data:
            metadata = script_data["script_metadata"]
            summary += f"Estimated duration: {script_data.get('estimated_duration', 'unknown')} minutes\n"
            summary += (
                f"Word count: {metadata.get('estimated_word_count', 'unknown')}\n"
            )
            summary += f"Quality score: {metadata.get('quality_metrics', {}).get('overall_score', 'unknown'):.1f}/100\n\n"

        summary += "Segments:\n"
        for i, segment in enumerate(segments, 1):
            segment_type = segment.get("type", "unknown")
            duration = segment.get("duration_estimate", "?")
            dialogue_count = len(segment.get("dialogue", []))
            summary += (
                f"{i}. {segment_type.title()} ({duration}min, {dialogue_count} lines)\n"
            )

        return summary
