from typing import Dict, List, Optional, Any
from .openai_service import OpenAIService
from .duration_calculator import DurationCalculator
import logging

logger = logging.getLogger(__name__)


class ConversationFlowAgent:
    """
    Conversation Flow Agent - Final dialogue polishing and natural flow

    Purpose: Final dialogue polishing and natural flow
    Tasks:
    - Smooth transitions between topics
    - Add natural conversation elements (agreements, disagreements, build-ups)
    - Insert cross-references where hosts mention each other's points
    - Add timing cues and pause indicators
    - Ensure the dialogue sounds like a genuine discussion

    Output: Final podcast script ready for voice synthesis
    """

    def __init__(self):
        self.openai_service = OpenAIService()
        self.duration_calculator = DurationCalculator()

    def polish_conversation_flow(
        self,
        personality_adapted_dialogue: Dict[str, Any],
        content_plan: Dict[str, Any],
        flow_preferences: Optional[Dict] = None,
        polish_settings: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Polish conversation flow for final podcast script

        Args:
            personality_adapted_dialogue: Personality-adapted dialogue from PersonalityAdaptationAgent
            content_plan: Content plan for structure reference
            flow_preferences: Optional flow preferences
            polish_settings: Optional polishing settings

        Returns:
            Final polished podcast script ready for voice synthesis
        """
        logger.info("Starting conversation flow polishing")

        # Apply polish settings
        settings = self._apply_polish_settings(polish_settings)

        # Analyze current flow state
        flow_analysis = self._analyze_current_flow(
            personality_adapted_dialogue, content_plan, settings
        )

        # Generate flow improvements
        flow_improvements = self._generate_flow_improvements(
            personality_adapted_dialogue, flow_analysis, settings
        )

        if not flow_improvements:
            logger.error("Failed to generate flow improvements")
            return None

        # Apply final polishing
        polished_script = self._apply_final_polishing(
            flow_improvements, flow_analysis, settings
        )

        logger.info("Conversation flow polishing completed")
        return polished_script

    def _apply_polish_settings(self, polish_settings: Optional[Dict]) -> Dict[str, Any]:
        """Apply polish settings with defaults"""
        default_settings = {
            "transition_smoothness": "high",
            "natural_elements": True,
            "timing_precision": "moderate",
            "pause_indicators": True,
            "cross_reference_enhancement": True,
            "energy_flow_management": True,
            "conversation_authenticity": "high",
            "final_quality_target": 95,
        }

        if polish_settings:
            default_settings.update(polish_settings)

        return default_settings

    def _analyze_current_flow(
        self,
        personality_adapted_dialogue: Dict[str, Any],
        content_plan: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze current conversation flow state"""

        flow_analysis = {
            "transition_quality": {},
            "natural_elements": {},
            "timing_analysis": {},
            "energy_flow": {},
            "authenticity_assessment": {},
            "improvement_areas": [],
        }

        segments = personality_adapted_dialogue.get("adapted_segments", [])

        # Analyze transitions between segments
        flow_analysis["transition_quality"] = self._analyze_transitions(segments)

        # Analyze natural conversation elements
        flow_analysis["natural_elements"] = self._analyze_natural_elements(segments)

        # Analyze timing and pacing
        flow_analysis["timing_analysis"] = self._analyze_timing_and_pacing(
            segments, content_plan
        )

        # Analyze energy flow throughout conversation
        flow_analysis["energy_flow"] = self._analyze_energy_flow(segments)

        # Assess conversation authenticity
        flow_analysis["authenticity_assessment"] = (
            self._assess_conversation_authenticity(segments)
        )

        # Identify improvement areas
        flow_analysis["improvement_areas"] = self._identify_improvement_areas(
            flow_analysis, settings
        )

        return flow_analysis

    def _analyze_transitions(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality of transitions between segments"""

        transition_analysis = {
            "total_transitions": len(segments) - 1 if len(segments) > 1 else 0,
            "smooth_transitions": 0,
            "abrupt_transitions": 0,
            "missing_transitions": 0,
            "transition_quality_score": 0,
            "transition_details": [],
        }

        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]

            transition_detail = self._analyze_single_transition(
                current_segment, next_segment
            )
            transition_analysis["transition_details"].append(transition_detail)

            # Count transition types
            if transition_detail["quality"] == "smooth":
                transition_analysis["smooth_transitions"] += 1
            elif transition_detail["quality"] == "abrupt":
                transition_analysis["abrupt_transitions"] += 1
            else:
                transition_analysis["missing_transitions"] += 1

        # Calculate overall transition quality score
        if transition_analysis["total_transitions"] > 0:
            smooth_ratio = (
                transition_analysis["smooth_transitions"]
                / transition_analysis["total_transitions"]
            )
            transition_analysis["transition_quality_score"] = smooth_ratio * 100

        return transition_analysis

    def _analyze_single_transition(
        self, current_segment: Dict[str, Any], next_segment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a single transition between segments"""

        transition_detail = {
            "from_segment": current_segment.get("segment_id", "unknown"),
            "to_segment": next_segment.get("segment_id", "unknown"),
            "quality": "missing",
            "has_bridge": False,
            "natural_flow": False,
            "improvement_needed": True,
        }

        # Check for existing cross-references or bridges
        current_cross_refs = current_segment.get("cross_references", [])
        next_cross_refs = next_segment.get("cross_references", [])

        if current_cross_refs or next_cross_refs:
            transition_detail["has_bridge"] = True
            transition_detail["quality"] = "smooth"
            transition_detail["improvement_needed"] = False

        # Check dialogue flow at segment boundaries
        current_dialogue = current_segment.get("adapted_dialogue", [])
        next_dialogue = next_segment.get("adapted_dialogue", [])

        if current_dialogue and next_dialogue:
            last_speaker = current_dialogue[-1].get("speaker", "")
            first_speaker = next_dialogue[0].get("speaker", "")

            # Natural if different speakers or if there's a clear topic shift
            if last_speaker != first_speaker:
                transition_detail["natural_flow"] = True
                if transition_detail["quality"] == "missing":
                    transition_detail["quality"] = "abrupt"

        return transition_detail

    def _analyze_natural_elements(
        self, segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze natural conversation elements"""

        natural_analysis = {
            "total_agreements": 0,
            "total_disagreements": 0,
            "total_build_ups": 0,
            "total_reactions": 0,
            "natural_interjections": 0,
            "conversation_markers": 0,
            "naturalness_score": 0,
        }

        for segment in segments:
            # Count personality enhancements that add naturalness
            enhancements = segment.get("personality_enhancements", [])
            for enhancement in enhancements:
                enhancement_type = enhancement.get("type", "")
                if enhancement_type == "agreement":
                    natural_analysis["total_agreements"] += 1
                elif enhancement_type == "disagreement":
                    natural_analysis["total_disagreements"] += 1
                elif enhancement_type == "build_up":
                    natural_analysis["total_build_ups"] += 1
                elif enhancement_type == "reaction":
                    natural_analysis["total_reactions"] += 1
                elif enhancement_type == "interjection":
                    natural_analysis["natural_interjections"] += 1

            # Count conversation markers in dialogue
            dialogue = segment.get("adapted_dialogue", [])
            for dialogue_item in dialogue:
                text = dialogue_item.get("text", "").lower()
                if any(
                    marker in text
                    for marker in ["you know", "i mean", "actually", "well", "so"]
                ):
                    natural_analysis["conversation_markers"] += 1

        # Calculate naturalness score
        total_elements = sum(
            [
                natural_analysis["total_agreements"],
                natural_analysis["total_build_ups"],
                natural_analysis["total_reactions"],
                natural_analysis["natural_interjections"],
                natural_analysis["conversation_markers"],
            ]
        )

        if segments:
            elements_per_segment = total_elements / len(segments)
            natural_analysis["naturalness_score"] = min(100, elements_per_segment * 20)

        return natural_analysis

    def _analyze_timing_and_pacing(
        self, segments: List[Dict[str, Any]], content_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze timing and pacing throughout conversation"""

        timing_analysis = {
            "total_estimated_duration": 0,
            "segment_durations": [],
            "pacing_consistency": 0,
            "timing_accuracy": 0,
            "pause_indicators": 0,
            "timing_cues": 0,
        }

        target_duration = content_plan.get("target_duration", 10)

        for segment in segments:
            segment_duration = segment.get("estimated_duration", 0)
            timing_analysis["segment_durations"].append(segment_duration)
            timing_analysis["total_estimated_duration"] += segment_duration

            # Count timing-related elements
            dialogue = segment.get("adapted_dialogue", [])
            for dialogue_item in dialogue:
                timing_cue = dialogue_item.get("timing_cue", "")
                if timing_cue:
                    timing_analysis["timing_cues"] += 1

                # Check for pause indicators in text
                text = dialogue_item.get("text", "")
                if "..." in text or "[pause]" in text.lower():
                    timing_analysis["pause_indicators"] += 1

        # Calculate pacing consistency
        if len(timing_analysis["segment_durations"]) > 1:
            durations = timing_analysis["segment_durations"]
            avg_duration = sum(durations) / len(durations)
            variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
            consistency = max(0, 100 - (variance * 10))
            timing_analysis["pacing_consistency"] = consistency

        # Calculate timing accuracy
        duration_diff = abs(
            timing_analysis["total_estimated_duration"] - target_duration
        )
        timing_analysis["timing_accuracy"] = max(0, 100 - (duration_diff * 10))

        return timing_analysis

    def _analyze_energy_flow(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze energy flow throughout the conversation"""

        energy_analysis = {
            "energy_levels": [],
            "energy_consistency": 0,
            "energy_peaks": 0,
            "energy_valleys": 0,
            "overall_energy_flow": "unknown",
        }

        for i, segment in enumerate(segments):
            # Estimate energy level based on dialogue characteristics
            energy_level = self._estimate_segment_energy(segment)
            energy_analysis["energy_levels"].append(
                {
                    "segment_index": i,
                    "segment_id": segment.get("segment_id", f"segment_{i}"),
                    "energy_level": energy_level,
                }
            )

            # Count energy peaks and valleys
            if energy_level > 75:
                energy_analysis["energy_peaks"] += 1
            elif energy_level < 40:
                energy_analysis["energy_valleys"] += 1

        # Analyze overall energy flow pattern
        if len(energy_analysis["energy_levels"]) > 2:
            levels = [e["energy_level"] for e in energy_analysis["energy_levels"]]

            # Check for good energy flow (start high, maintain, end strong)
            start_energy = levels[0]
            end_energy = levels[-1]
            mid_energy = (
                sum(levels[1:-1]) / len(levels[1:-1]) if len(levels) > 2 else levels[0]
            )

            if start_energy > 60 and end_energy > 60 and mid_energy > 50:
                energy_analysis["overall_energy_flow"] = "good"
            elif start_energy > 70 or end_energy > 70:
                energy_analysis["overall_energy_flow"] = "moderate"
            else:
                energy_analysis["overall_energy_flow"] = "needs_improvement"

        return energy_analysis

    def _estimate_segment_energy(self, segment: Dict[str, Any]) -> int:
        """Estimate energy level of a segment"""

        energy_score = 50  # Base energy level

        dialogue = segment.get("adapted_dialogue", [])
        for dialogue_item in dialogue:
            text = dialogue_item.get("text", "").lower()
            personality_elements = dialogue_item.get("personality_elements", [])

            # High energy indicators
            if any(
                word in text
                for word in ["amazing", "incredible", "wow", "fantastic", "exciting"]
            ):
                energy_score += 10
            if "!" in dialogue_item.get("text", ""):
                energy_score += 5
            if (
                "energetic" in personality_elements
                or "enthusiastic" in personality_elements
            ):
                energy_score += 8

            # Low energy indicators
            if any(
                word in text
                for word in ["however", "actually", "specifically", "technically"]
            ):
                energy_score -= 3
            if (
                "methodical" in personality_elements
                or "analytical" in personality_elements
            ):
                energy_score -= 2

        # Factor in enhancements
        enhancements = segment.get("personality_enhancements", [])
        for enhancement in enhancements:
            if enhancement.get("type") == "reaction":
                energy_score += 5
            elif enhancement.get("type") == "interjection":
                energy_score += 3

        return max(0, min(100, energy_score))

    def _assess_conversation_authenticity(
        self, segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess how authentic the conversation sounds"""

        authenticity_assessment = {
            "natural_interruptions": 0,
            "organic_topic_shifts": 0,
            "genuine_reactions": 0,
            "conversational_markers": 0,
            "authenticity_score": 0,
        }

        for segment in segments:
            dialogue = segment.get("adapted_dialogue", [])

            # Look for natural conversation patterns
            for i, dialogue_item in enumerate(dialogue):
                text = dialogue_item.get("text", "")

                # Count conversational markers
                markers = [
                    "well",
                    "you know",
                    "i mean",
                    "actually",
                    "so",
                    "but",
                    "and",
                    "oh",
                ]
                for marker in markers:
                    if f" {marker} " in f" {text.lower()} ":
                        authenticity_assessment["conversational_markers"] += 1

                # Look for interruption patterns (incomplete sentences followed by different speaker)
                if i < len(dialogue) - 1:
                    next_item = dialogue[i + 1]
                    if (
                        text.endswith("...") or text.endswith("-")
                    ) and dialogue_item.get("speaker") != next_item.get("speaker"):
                        authenticity_assessment["natural_interruptions"] += 1

            # Count genuine reactions from enhancements
            enhancements = segment.get("personality_enhancements", [])
            for enhancement in enhancements:
                if enhancement.get("type") in ["reaction", "interjection"]:
                    authenticity_assessment["genuine_reactions"] += 1

        # Calculate authenticity score
        total_authentic_elements = sum(
            [
                authenticity_assessment["natural_interruptions"],
                authenticity_assessment["genuine_reactions"],
                min(
                    authenticity_assessment["conversational_markers"], 10
                ),  # Cap markers to avoid over-counting
            ]
        )

        if segments:
            elements_per_segment = total_authentic_elements / len(segments)
            authenticity_assessment["authenticity_score"] = min(
                100, elements_per_segment * 15
            )

        return authenticity_assessment

    def _identify_improvement_areas(
        self, flow_analysis: Dict[str, Any], settings: Dict[str, Any]
    ) -> List[str]:
        """Identify areas that need improvement"""

        improvement_areas = []

        # Check transition quality
        transition_quality = flow_analysis.get("transition_quality", {}).get(
            "transition_quality_score", 0
        )
        if transition_quality < 80:
            improvement_areas.append("smooth_transitions")

        # Check natural elements
        naturalness_score = flow_analysis.get("natural_elements", {}).get(
            "naturalness_score", 0
        )
        if naturalness_score < 70:
            improvement_areas.append("natural_conversation_elements")

        # Check timing
        timing_accuracy = flow_analysis.get("timing_analysis", {}).get(
            "timing_accuracy", 0
        )
        if timing_accuracy < 85:
            improvement_areas.append("timing_precision")

        # Check energy flow
        energy_flow = flow_analysis.get("energy_flow", {}).get(
            "overall_energy_flow", "unknown"
        )
        if energy_flow == "needs_improvement":
            improvement_areas.append("energy_management")

        # Check authenticity
        authenticity_score = flow_analysis.get("authenticity_assessment", {}).get(
            "authenticity_score", 0
        )
        if authenticity_score < 75:
            improvement_areas.append("conversation_authenticity")

        return improvement_areas

    def _generate_flow_improvements(
        self,
        personality_adapted_dialogue: Dict[str, Any],
        flow_analysis: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate flow improvements using OpenAI"""

        improvement_areas = flow_analysis.get("improvement_areas", [])

        prompt = f"""
        You are a podcast conversation flow specialist. Polish the following personality-adapted dialogue to create the most natural, engaging conversation possible.

        PERSONALITY-ADAPTED DIALOGUE:
        {personality_adapted_dialogue}

        FLOW ANALYSIS:
        {flow_analysis}

        IMPROVEMENT AREAS NEEDED:
        {improvement_areas}

        POLISH SETTINGS:
        {settings}

        Apply final polishing by:
        1. Smoothing transitions between topics with natural bridges
        2. Adding natural conversation elements (agreements, build-ups, reactions)
        3. Inserting appropriate cross-references between hosts
        4. Adding timing cues and pause indicators for natural delivery
        5. Ensuring authentic, genuine conversation flow
        6. Managing energy levels throughout the conversation

        Target Quality: {settings.get("final_quality_target", 95)}%

        Format as JSON:
        {{
            "polishing_metadata": {{
                "improvements_applied": [],
                "flow_quality_score": 95,
                "naturalness_score": 93,
                "timing_precision": 90,
                "overall_polish_score": 93
            }},
            "polished_segments": [
                {{
                    "segment_id": "intro",
                    "original_dialogue": [],
                    "polished_dialogue": [
                        {{
                            "speaker": "Host Name",
                            "text": "Polished dialogue with natural flow",
                            "timing_cues": {{
                                "pace": "moderate",
                                "pause_before": 0.5,
                                "pause_after": 0.3,
                                "emphasis": ["key", "words"]
                            }},
                            "delivery_notes": "Natural, conversational tone",
                            "flow_elements": ["smooth_transition", "natural_bridge"]
                        }}
                    ],
                    "transition_elements": [
                        {{
                            "type": "topic_bridge",
                            "from_topic": "previous_topic",
                            "to_topic": "current_topic",
                            "bridge_text": "Speaking of that...",
                            "speaker": "Host Name"
                        }}
                    ],
                    "natural_enhancements": [
                        {{
                            "type": "agreement",
                            "speaker": "Host Name",
                            "content": "Absolutely, that's exactly right",
                            "trigger": "after_key_point"
                        }}
                    ]
                }}
            ],
            "conversation_flow": {{
                "opening_energy": "engaging",
                "energy_progression": ["high", "moderate", "high", "strong_finish"],
                "transition_quality": "smooth",
                "overall_authenticity": "very_natural"
            }},
            "final_script_metadata": {{
                "total_duration": 10.2,
                "word_count": 1650,
                "speaker_balance": {{"Host1": 52, "Host2": 48}},
                "conversation_quality": 94
            }}
        }}
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert podcast conversation flow specialist who creates the most natural, engaging dialogue possible.",
            },
            {"role": "user", "content": prompt},
        ]

        response = self.openai_service._make_request(
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        if response:
            try:
                import json

                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse flow improvements JSON: {e}")
                return None

        return None

    def _apply_final_polishing(
        self,
        flow_improvements: Dict[str, Any],
        flow_analysis: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply final polishing touches to the conversation"""

        if not flow_improvements:
            return flow_improvements

        # Add final script metadata
        flow_improvements["final_script_metadata"] = (
            self._generate_final_script_metadata(flow_improvements, flow_analysis)
        )

        # Add production notes
        flow_improvements["production_notes"] = self._generate_production_notes(
            flow_improvements, settings
        )

        # Add quality validation
        flow_improvements["quality_validation"] = self._validate_final_quality(
            flow_improvements, settings
        )

        return flow_improvements

    def _generate_final_script_metadata(
        self, flow_improvements: Dict[str, Any], flow_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for the final script"""

        polished_segments = flow_improvements.get("polished_segments", [])

        metadata = {
            "script_version": "final_polished",
            "total_segments": len(polished_segments),
            "estimated_duration": 0,
            "total_words": 0,
            "total_dialogue_items": 0,
            "timing_cues_count": 0,
            "natural_elements_count": 0,
            "transition_bridges": 0,
        }

        for segment in polished_segments:
            polished_dialogue = segment.get("polished_dialogue", [])
            metadata["total_dialogue_items"] += len(polished_dialogue)

            for dialogue_item in polished_dialogue:
                text = dialogue_item.get("text", "")
                word_count = len(text.split())
                metadata["total_words"] += word_count

                # Count timing cues
                if dialogue_item.get("timing_cues"):
                    metadata["timing_cues_count"] += 1

            # Count natural enhancements
            natural_enhancements = segment.get("natural_enhancements", [])
            metadata["natural_elements_count"] += len(natural_enhancements)

            # Count transition elements
            transition_elements = segment.get("transition_elements", [])
            metadata["transition_bridges"] += len(transition_elements)

        # Estimate duration using advanced duration calculator
        if metadata["total_words"] > 0:
            # Collect all dialogue for duration calculation
            all_dialogue = []
            for segment in polished_segments:
                polished_dialogue = segment.get("polished_dialogue", [])
                all_dialogue.extend(polished_dialogue)

            # Use advanced duration calculator for more accurate estimation
            duration_result = self.duration_calculator.estimate_dialogue_duration(
                all_dialogue, conversation_style="conversational"
            )

            metadata["estimated_duration"] = duration_result["estimated_duration"]
            metadata["duration_breakdown"] = duration_result
            metadata["speech_time_minutes"] = duration_result["speech_time_minutes"]
            metadata["pause_time_minutes"] = duration_result["pause_time_minutes"]
            metadata["flow_adjustment_factor"] = duration_result[
                "flow_adjustment_factor"
            ]

        return metadata

    def _generate_production_notes(
        self, flow_improvements: Dict[str, Any], settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate production notes for voice synthesis"""

        return {
            "voice_synthesis_ready": True,
            "timing_precision": settings.get("timing_precision", "moderate"),
            "natural_delivery_required": True,
            "energy_management_notes": "Follow energy progression in conversation_flow",
            "pause_indicators_included": settings.get("pause_indicators", True),
            "cross_reference_emphasis": "Emphasize host name mentions for clarity",
            "overall_tone": "Natural, engaging conversation",
        }

    def _validate_final_quality(
        self, flow_improvements: Dict[str, Any], settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate final script quality"""

        validation = {
            "meets_quality_target": False,
            "quality_scores": {},
            "validation_issues": [],
            "recommendations": [],
        }

        polishing_metadata = flow_improvements.get("polishing_metadata", {})
        target_quality = settings.get("final_quality_target", 95)

        # Extract quality scores
        flow_quality = polishing_metadata.get("flow_quality_score", 0)
        naturalness = polishing_metadata.get("naturalness_score", 0)
        timing_precision = polishing_metadata.get("timing_precision", 0)
        overall_polish = polishing_metadata.get("overall_polish_score", 0)

        validation["quality_scores"] = {
            "flow_quality": flow_quality,
            "naturalness": naturalness,
            "timing_precision": timing_precision,
            "overall_polish": overall_polish,
        }

        # Check if target is met
        validation["meets_quality_target"] = overall_polish >= target_quality

        # Identify issues
        if flow_quality < target_quality:
            validation["validation_issues"].append("Flow quality below target")
        if naturalness < target_quality:
            validation["validation_issues"].append("Naturalness below target")
        if timing_precision < 85:
            validation["validation_issues"].append("Timing precision needs improvement")

        # Generate recommendations
        if not validation["meets_quality_target"]:
            validation["recommendations"].append(
                "Consider additional polishing iteration"
            )
        if flow_quality < 90:
            validation["recommendations"].append("Focus on smoother topic transitions")
        if naturalness < 90:
            validation["recommendations"].append(
                "Add more natural conversation elements"
            )

        return validation

    def validate_conversation_flow(
        self, polished_script: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate final conversation flow quality"""

        validation = {
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "flow_score": 0,
            "naturalness_score": 0,
            "timing_score": 0,
            "overall_score": 0,
        }

        if not polished_script:
            validation["is_valid"] = False
            validation["issues"].append("No polished script provided")
            return validation

        # Extract scores from metadata
        polishing_metadata = polished_script.get("polishing_metadata", {})

        validation["flow_score"] = polishing_metadata.get("flow_quality_score", 0)
        validation["naturalness_score"] = polishing_metadata.get("naturalness_score", 0)
        validation["timing_score"] = polishing_metadata.get("timing_precision", 0)

        # Check quality validation
        quality_validation = polished_script.get("quality_validation", {})
        if not quality_validation.get("meets_quality_target", False):
            validation["issues"].extend(quality_validation.get("validation_issues", []))
            validation["suggestions"].extend(
                quality_validation.get("recommendations", [])
            )

        # Calculate overall score
        scores = [
            validation["flow_score"],
            validation["naturalness_score"],
            validation["timing_score"],
        ]
        validation["overall_score"] = sum(scores) / len(scores) if scores else 0

        return validation

    def get_flow_summary(self, polished_script: Dict[str, Any]) -> str:
        """Generate human-readable summary of conversation flow polishing"""

        if not polished_script:
            return "No polished script available"

        polishing_metadata = polished_script.get("polishing_metadata", {})
        final_metadata = polished_script.get("final_script_metadata", {})
        conversation_flow = polished_script.get("conversation_flow", {})

        summary = f"Conversation Flow Polishing Summary\n\n"

        # Quality scores
        flow_quality = polishing_metadata.get("flow_quality_score", 0)
        naturalness = polishing_metadata.get("naturalness_score", 0)
        timing_precision = polishing_metadata.get("timing_precision", 0)
        overall_polish = polishing_metadata.get("overall_polish_score", 0)

        summary += f"Quality Scores:\n"
        summary += f"  Flow Quality: {flow_quality}%\n"
        summary += f"  Naturalness: {naturalness}%\n"
        summary += f"  Timing Precision: {timing_precision}%\n"
        summary += f"  Overall Polish: {overall_polish}%\n"

        # Script statistics
        total_duration = final_metadata.get("estimated_duration", 0)
        total_words = final_metadata.get("total_words", 0)
        total_segments = final_metadata.get("total_segments", 0)

        summary += f"\nScript Statistics:\n"
        summary += f"  Estimated Duration: {total_duration:.1f} minutes\n"
        summary += f"  Total Words: {total_words}\n"
        summary += f"  Total Segments: {total_segments}\n"

        # Flow characteristics
        opening_energy = conversation_flow.get("opening_energy", "unknown")
        transition_quality = conversation_flow.get("transition_quality", "unknown")
        authenticity = conversation_flow.get("overall_authenticity", "unknown")

        summary += f"\nFlow Characteristics:\n"
        summary += f"  Opening Energy: {opening_energy}\n"
        summary += f"  Transition Quality: {transition_quality}\n"
        summary += f"  Authenticity: {authenticity}\n"

        # Improvements applied
        improvements = polishing_metadata.get("improvements_applied", [])
        if improvements:
            summary += f"\nImprovements Applied:\n"
            for improvement in improvements:
                summary += f"  - {improvement}\n"

        return summary
