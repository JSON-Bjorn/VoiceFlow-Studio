from typing import Dict, List, Optional, Any
from ..openai_service import OpenAIService
from ..duration_calculator import DurationCalculator
import logging

logger = logging.getLogger(__name__)


class PersonalityAdaptationAgent:
    """
    Personality Adaptation Agent - Apply host personalities and create natural dialogue

    Purpose: Apply host personalities and create natural dialogue
    Tasks:
    - Apply hardcoded personality traits to each host's segments
    - Rewrite content in each host's unique voice and style
    - Add personality-specific reactions, jokes, or commentary
    - Create natural interjections and cross-references between hosts
    - Add conversational elements like "As [Host Name] mentioned..." or "That reminds me of..."

    Output: Fully personalized dialogue script
    """

    def __init__(self):
        self.openai_service = OpenAIService()

    def adapt_personalities(
        self,
        dialogue_distribution: Dict[str, Any],
        host_personalities: Dict[str, Any],
        style_preferences: Optional[Dict] = None,
        adaptation_settings: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Apply host personalities to create natural, personalized dialogue

        Args:
            dialogue_distribution: Dialogue distribution from DialogueDistributionAgent
            host_personalities: Host personality definitions
            style_preferences: Style preferences for adaptation
            adaptation_settings: Optional adaptation settings

        Returns:
            Fully personalized dialogue script with personality-driven content
        """
        logger.info("Starting personality adaptation for dialogue")

        # Apply adaptation settings
        settings = self._apply_adaptation_settings(adaptation_settings)

        # Analyze personality traits for adaptation
        personality_analysis = self._analyze_personalities_for_adaptation(
            host_personalities, style_preferences, settings
        )

        # Generate personality-adapted dialogue
        adapted_dialogue = self._generate_personality_adapted_dialogue(
            dialogue_distribution, personality_analysis, settings
        )

        if not adapted_dialogue:
            logger.error("Failed to generate personality-adapted dialogue")
            return None

        # Enhance with personality-specific elements
        enhanced_dialogue = self._enhance_with_personality_elements(
            adapted_dialogue, personality_analysis, settings
        )

        logger.info("Personality adaptation completed")
        return enhanced_dialogue

    def _apply_adaptation_settings(
        self, adaptation_settings: Optional[Dict]
    ) -> Dict[str, Any]:
        """Apply adaptation settings with defaults"""
        default_settings = {
            "personality_intensity": "moderate",  # light, moderate, strong
            "voice_consistency": "high",
            "natural_interjections": True,
            "cross_references": True,
            "personality_specific_reactions": True,
            "humor_integration": True,
            "speaking_style_adaptation": True,
        }

        if adaptation_settings:
            default_settings.update(adaptation_settings)

        return default_settings

    def _analyze_personalities_for_adaptation(
        self,
        host_personalities: Dict[str, Any],
        style_preferences: Optional[Dict],
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze host personalities for detailed adaptation"""

        personality_analysis = {
            "hosts": {},
            "interaction_styles": {},
            "adaptation_rules": {},
        }

        for host_key, host_data in host_personalities.items():
            name = host_data.get("name", f"Host {host_key[-1]}")
            personality = host_data.get("personality", "")
            role = host_data.get("role", "co-host")

            # Detailed personality analysis
            detailed_analysis = self._perform_detailed_personality_analysis(
                personality, role, style_preferences
            )

            personality_analysis["hosts"][host_key] = {
                "name": name,
                "personality": personality,
                "role": role,
                "voice_characteristics": detailed_analysis["voice_characteristics"],
                "speaking_patterns": detailed_analysis["speaking_patterns"],
                "reaction_styles": detailed_analysis["reaction_styles"],
                "humor_style": detailed_analysis["humor_style"],
                "interjection_patterns": detailed_analysis["interjection_patterns"],
                "cross_reference_style": detailed_analysis["cross_reference_style"],
            }

        # Determine interaction styles between hosts
        if len(personality_analysis["hosts"]) == 2:
            host_keys = list(personality_analysis["hosts"].keys())
            personality_analysis["interaction_styles"] = (
                self._determine_interaction_styles(
                    personality_analysis["hosts"][host_keys[0]],
                    personality_analysis["hosts"][host_keys[1]],
                    settings,
                )
            )

        # Create adaptation rules
        personality_analysis["adaptation_rules"] = self._create_adaptation_rules(
            personality_analysis["hosts"], settings
        )

        return personality_analysis

    def _perform_detailed_personality_analysis(
        self, personality: str, role: str, style_preferences: Optional[Dict]
    ) -> Dict[str, Any]:
        """Perform detailed analysis of a single personality"""

        personality_lower = personality.lower()
        analysis = {
            "voice_characteristics": [],
            "speaking_patterns": [],
            "reaction_styles": [],
            "humor_style": "none",
            "interjection_patterns": [],
            "cross_reference_style": "neutral",
        }

        # Voice characteristics based on personality
        if "analytical" in personality_lower:
            analysis["voice_characteristics"].extend(
                ["precise", "methodical", "fact-focused"]
            )
            analysis["speaking_patterns"].extend(
                ["structured_responses", "detailed_explanations"]
            )
            analysis["reaction_styles"].extend(
                ["thoughtful_pauses", "clarifying_questions"]
            )

        if "enthusiastic" in personality_lower:
            analysis["voice_characteristics"].extend(
                ["energetic", "expressive", "animated"]
            )
            analysis["speaking_patterns"].extend(
                ["exclamatory_phrases", "rapid_delivery"]
            )
            analysis["reaction_styles"].extend(["excited_responses", "building_energy"])

        if "curious" in personality_lower:
            analysis["voice_characteristics"].extend(
                ["inquisitive", "engaged", "probing"]
            )
            analysis["speaking_patterns"].extend(
                ["frequent_questions", "follow_up_inquiries"]
            )
            analysis["reaction_styles"].extend(
                ["surprised_discoveries", "deeper_exploration"]
            )

        if "relatable" in personality_lower:
            analysis["voice_characteristics"].extend(
                ["warm", "accessible", "down-to-earth"]
            )
            analysis["speaking_patterns"].extend(
                ["personal_examples", "simple_language"]
            )
            analysis["reaction_styles"].extend(
                ["empathetic_responses", "shared_experiences"]
            )

        if "friendly" in personality_lower:
            analysis["voice_characteristics"].extend(
                ["warm", "approachable", "supportive"]
            )
            analysis["speaking_patterns"].extend(
                ["encouraging_phrases", "inclusive_language"]
            )
            analysis["reaction_styles"].extend(
                ["positive_affirmations", "supportive_comments"]
            )

        # Humor style analysis
        if any(word in personality_lower for word in ["witty", "humorous", "funny"]):
            analysis["humor_style"] = "witty"
        elif "enthusiastic" in personality_lower:
            analysis["humor_style"] = "playful"
        elif "relatable" in personality_lower:
            analysis["humor_style"] = "self_deprecating"
        elif "analytical" in personality_lower:
            analysis["humor_style"] = "dry"
        else:
            analysis["humor_style"] = "light"

        # Interjection patterns
        if "enthusiastic" in personality_lower:
            analysis["interjection_patterns"].extend(
                ["Wow!", "That's amazing!", "Incredible!"]
            )
        if "analytical" in personality_lower:
            analysis["interjection_patterns"].extend(
                ["Interesting...", "Let me think about that", "Actually..."]
            )
        if "curious" in personality_lower:
            analysis["interjection_patterns"].extend(
                ["Wait, what?", "Tell me more", "How so?"]
            )

        # Cross-reference style
        if "analytical" in personality_lower:
            analysis["cross_reference_style"] = "methodical"
        elif "enthusiastic" in personality_lower:
            analysis["cross_reference_style"] = "energetic"
        elif "friendly" in personality_lower:
            analysis["cross_reference_style"] = "supportive"
        else:
            analysis["cross_reference_style"] = "neutral"

        return analysis

    def _determine_interaction_styles(
        self,
        host1_analysis: Dict[str, Any],
        host2_analysis: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine how personalities interact with each other"""

        interaction_styles = {
            "dynamic_type": "balanced",
            "energy_balance": "equal",
            "conversation_flow": "collaborative",
            "interruption_style": "respectful",
            "building_patterns": [],
            "contrast_elements": [],
        }

        host1_voice = host1_analysis.get("voice_characteristics", [])
        host2_voice = host2_analysis.get("voice_characteristics", [])

        # Determine dynamic type
        if "energetic" in host1_voice and "methodical" in host2_voice:
            interaction_styles["dynamic_type"] = "energetic_analytical"
            interaction_styles["energy_balance"] = "host1_higher"
        elif "methodical" in host1_voice and "energetic" in host2_voice:
            interaction_styles["dynamic_type"] = "analytical_energetic"
            interaction_styles["energy_balance"] = "host2_higher"
        elif "energetic" in host1_voice and "energetic" in host2_voice:
            interaction_styles["dynamic_type"] = "high_energy"
            interaction_styles["energy_balance"] = "equal_high"
        elif "methodical" in host1_voice and "methodical" in host2_voice:
            interaction_styles["dynamic_type"] = "analytical_pair"
            interaction_styles["energy_balance"] = "equal_moderate"

        # Building patterns
        if "supportive" in host1_voice or "supportive" in host2_voice:
            interaction_styles["building_patterns"].append("mutual_support")
        if "inquisitive" in host1_voice or "inquisitive" in host2_voice:
            interaction_styles["building_patterns"].append("question_building")

        # Contrast elements
        if "precise" in host1_voice and "expressive" in host2_voice:
            interaction_styles["contrast_elements"].append("precision_vs_expression")
        if "fact-focused" in host1_voice and "warm" in host2_voice:
            interaction_styles["contrast_elements"].append("facts_vs_warmth")

        return interaction_styles

    def _create_adaptation_rules(
        self, hosts_analysis: Dict[str, Any], settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create specific adaptation rules for personality application"""

        rules = {
            "voice_adaptation": {},
            "reaction_adaptation": {},
            "interjection_rules": {},
            "cross_reference_rules": {},
            "humor_integration": {},
        }

        for host_key, host_data in hosts_analysis.items():
            host_name = host_data["name"]

            # Voice adaptation rules
            rules["voice_adaptation"][host_name] = {
                "characteristics": host_data["voice_characteristics"],
                "patterns": host_data["speaking_patterns"],
                "intensity": settings.get("personality_intensity", "moderate"),
            }

            # Reaction adaptation rules
            rules["reaction_adaptation"][host_name] = {
                "styles": host_data["reaction_styles"],
                "frequency": "moderate"
                if settings.get("personality_specific_reactions")
                else "low",
            }

            # Interjection rules
            rules["interjection_rules"][host_name] = {
                "patterns": host_data["interjection_patterns"],
                "enabled": settings.get("natural_interjections", True),
                "frequency": "moderate",
            }

            # Cross-reference rules
            rules["cross_reference_rules"][host_name] = {
                "style": host_data["cross_reference_style"],
                "enabled": settings.get("cross_references", True),
            }

            # Humor integration
            rules["humor_integration"][host_name] = {
                "style": host_data["humor_style"],
                "enabled": settings.get("humor_integration", True),
                "appropriateness": "contextual",
            }

        return rules

    def _generate_personality_adapted_dialogue(
        self,
        dialogue_distribution: Dict[str, Any],
        personality_analysis: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate personality-adapted dialogue using OpenAI"""

        prompt = f"""
        You are a podcast personality adaptation specialist. Transform the following dialogue distribution into natural, personality-driven conversation.

        DIALOGUE DISTRIBUTION:
        {dialogue_distribution}

        PERSONALITY ANALYSIS:
        {personality_analysis}

        ADAPTATION SETTINGS:
        {settings}

        Transform the dialogue by:
        1. Applying each host's unique voice characteristics and speaking patterns
        2. Adding personality-specific reactions, comments, and interjections
        3. Creating natural cross-references between hosts
        4. Integrating appropriate humor based on personality styles
        5. Ensuring voice consistency throughout

        Personality Intensity: {settings.get("personality_intensity", "moderate")}

        Format as JSON:
        {{
            "adaptation_metadata": {{
                "total_adaptations": 0,
                "personality_consistency_score": 95,
                "natural_flow_score": 90,
                "adaptation_strategy": "strategy_description"
            }},
            "adapted_segments": [
                {{
                    "segment_id": "intro",
                    "original_content": "Original segment content",
                    "adapted_dialogue": [
                        {{
                            "speaker": "Host Name",
                            "text": "Personality-adapted dialogue",
                            "personality_elements": ["voice_characteristic", "speaking_pattern"],
                            "adaptation_type": "voice_adaptation",
                            "timing_cue": "natural_pace"
                        }}
                    ],
                    "personality_enhancements": [
                        {{
                            "type": "interjection",
                            "host": "Host Name",
                            "content": "Natural interjection",
                            "trigger": "after_fact_presentation"
                        }}
                    ],
                    "cross_references": [
                        {{
                            "from_host": "Host 1",
                            "reference_type": "callback",
                            "content": "As Host 2 mentioned...",
                            "context": "building_on_previous_point"
                        }}
                    ]
                }}
            ],
            "personality_consistency": {{
                "host1_voice_score": 95,
                "host2_voice_score": 93,
                "interaction_naturalness": 92,
                "overall_consistency": 93
            }},
            "enhancement_summary": {{
                "total_interjections": 0,
                "total_cross_references": 0,
                "humor_instances": 0,
                "personality_adaptations": 0
            }}
        }}
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert podcast personality adaptation specialist who creates natural, engaging dialogue that reflects each host's unique personality.",
            },
            {"role": "user", "content": prompt},
        ]

        response = self.openai_service._make_request(
            messages=messages,
            temperature=0.8,  # Higher temperature for more personality variation
            response_format={"type": "json_object"},
        )

        if response:
            try:
                import json

                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse personality adaptation JSON: {e}")
                return None

        return None

    def _enhance_with_personality_elements(
        self,
        adapted_dialogue: Dict[str, Any],
        personality_analysis: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enhance adapted dialogue with additional personality elements"""

        if not adapted_dialogue:
            return adapted_dialogue

        # Add personality metadata
        adapted_dialogue["personality_metadata"] = {
            "adaptation_quality": self._assess_adaptation_quality(
                adapted_dialogue, personality_analysis
            ),
            "voice_consistency": self._assess_voice_consistency(
                adapted_dialogue, personality_analysis
            ),
            "natural_flow": self._assess_natural_flow(adapted_dialogue),
            "personality_coverage": self._assess_personality_coverage(
                adapted_dialogue, personality_analysis
            ),
        }

        # Enhance segments with additional personality touches
        for segment in adapted_dialogue.get("adapted_segments", []):
            self._enhance_segment_personality(segment, personality_analysis, settings)

        return adapted_dialogue

    def _enhance_segment_personality(
        self,
        segment: Dict[str, Any],
        personality_analysis: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> None:
        """Enhance individual segment with additional personality elements"""

        # Add timing and delivery cues
        for dialogue_item in segment.get("adapted_dialogue", []):
            speaker = dialogue_item.get("speaker", "")

            # Find speaker's personality data
            speaker_data = None
            for host_data in personality_analysis.get("hosts", {}).values():
                if host_data.get("name") == speaker:
                    speaker_data = host_data
                    break

            if speaker_data:
                # Add delivery cues based on personality
                dialogue_item["delivery_cues"] = self._generate_delivery_cues(
                    dialogue_item, speaker_data, settings
                )

                # Add emotional context
                dialogue_item["emotional_context"] = self._determine_emotional_context(
                    dialogue_item, speaker_data
                )

    def _generate_delivery_cues(
        self,
        dialogue_item: Dict[str, Any],
        speaker_data: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate delivery cues based on personality"""

        voice_characteristics = speaker_data.get("voice_characteristics", [])
        speaking_patterns = speaker_data.get("speaking_patterns", [])

        delivery_cues = {
            "pace": "moderate",
            "energy": "moderate",
            "emphasis": [],
            "pauses": [],
        }

        # Pace based on personality
        if "energetic" in voice_characteristics:
            delivery_cues["pace"] = "quick"
            delivery_cues["energy"] = "high"
        elif "methodical" in voice_characteristics:
            delivery_cues["pace"] = "measured"
            delivery_cues["energy"] = "moderate"

        # Emphasis patterns
        if "expressive" in voice_characteristics:
            delivery_cues["emphasis"].append("emotional_words")
        if "fact-focused" in voice_characteristics:
            delivery_cues["emphasis"].append("key_facts")

        # Pause patterns
        if "thoughtful_pauses" in speaker_data.get("reaction_styles", []):
            delivery_cues["pauses"].append("before_responses")
        if "structured_responses" in speaking_patterns:
            delivery_cues["pauses"].append("between_points")

        return delivery_cues

    def _determine_emotional_context(
        self, dialogue_item: Dict[str, Any], speaker_data: Dict[str, Any]
    ) -> str:
        """Determine emotional context for dialogue delivery"""

        voice_characteristics = speaker_data.get("voice_characteristics", [])
        text = dialogue_item.get("text", "").lower()

        # Determine base emotional context
        if "enthusiastic" in voice_characteristics:
            if any(word in text for word in ["amazing", "incredible", "wow"]):
                return "excited"
            else:
                return "energetic"
        elif "analytical" in voice_characteristics:
            if any(word in text for word in ["interesting", "however", "actually"]):
                return "thoughtful"
            else:
                return "focused"
        elif "warm" in voice_characteristics:
            return "friendly"
        else:
            return "neutral"

    def _assess_adaptation_quality(
        self, adapted_dialogue: Dict[str, Any], personality_analysis: Dict[str, Any]
    ) -> int:
        """Assess overall adaptation quality"""

        segments = adapted_dialogue.get("adapted_segments", [])
        if not segments:
            return 0

        quality_factors = []

        # Check personality element coverage
        total_personality_elements = 0
        for segment in segments:
            for dialogue_item in segment.get("adapted_dialogue", []):
                total_personality_elements += len(
                    dialogue_item.get("personality_elements", [])
                )

        if segments:
            elements_per_segment = total_personality_elements / len(segments)
            quality_factors.append(min(100, elements_per_segment * 20))

        # Check enhancement coverage
        total_enhancements = sum(
            len(segment.get("personality_enhancements", [])) for segment in segments
        )
        if segments:
            enhancements_per_segment = total_enhancements / len(segments)
            quality_factors.append(min(100, enhancements_per_segment * 30))

        return (
            int(sum(quality_factors) / len(quality_factors)) if quality_factors else 0
        )

    def _assess_voice_consistency(
        self, adapted_dialogue: Dict[str, Any], personality_analysis: Dict[str, Any]
    ) -> int:
        """Assess voice consistency across segments"""

        # Simple heuristic: check if personality elements are consistently applied
        hosts = personality_analysis.get("hosts", {})
        if not hosts:
            return 0

        consistency_scores = []

        for host_data in hosts.values():
            host_name = host_data.get("name", "")
            expected_characteristics = host_data.get("voice_characteristics", [])

            # Count how often this host's characteristics appear
            appearances = 0
            total_dialogue_items = 0

            for segment in adapted_dialogue.get("adapted_segments", []):
                for dialogue_item in segment.get("adapted_dialogue", []):
                    if dialogue_item.get("speaker") == host_name:
                        total_dialogue_items += 1
                        item_elements = dialogue_item.get("personality_elements", [])
                        if any(
                            char in item_elements for char in expected_characteristics
                        ):
                            appearances += 1

            if total_dialogue_items > 0:
                consistency_score = (appearances / total_dialogue_items) * 100
                consistency_scores.append(consistency_score)

        return (
            int(sum(consistency_scores) / len(consistency_scores))
            if consistency_scores
            else 0
        )

    def _assess_natural_flow(self, adapted_dialogue: Dict[str, Any]) -> int:
        """Assess natural flow of adapted dialogue"""

        segments = adapted_dialogue.get("adapted_segments", [])
        if not segments:
            return 0

        flow_factors = []

        # Check cross-reference usage
        total_cross_refs = sum(
            len(segment.get("cross_references", [])) for segment in segments
        )
        if len(segments) > 1:
            cross_ref_score = min(100, (total_cross_refs / (len(segments) - 1)) * 50)
            flow_factors.append(cross_ref_score)

        # Check interjection distribution
        total_interjections = sum(
            len(segment.get("personality_enhancements", [])) for segment in segments
        )
        interjection_score = min(100, (total_interjections / len(segments)) * 25)
        flow_factors.append(interjection_score)

        return int(sum(flow_factors) / len(flow_factors)) if flow_factors else 0

    def _assess_personality_coverage(
        self, adapted_dialogue: Dict[str, Any], personality_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess how well personalities are covered"""

        coverage = {"hosts": {}, "overall_coverage": 0}

        hosts = personality_analysis.get("hosts", {})
        for host_data in hosts.values():
            host_name = host_data.get("name", "")
            expected_elements = (
                host_data.get("voice_characteristics", [])
                + host_data.get("speaking_patterns", [])
                + host_data.get("reaction_styles", [])
            )

            # Count coverage for this host
            covered_elements = set()
            total_appearances = 0

            for segment in adapted_dialogue.get("adapted_segments", []):
                for dialogue_item in segment.get("adapted_dialogue", []):
                    if dialogue_item.get("speaker") == host_name:
                        total_appearances += 1
                        item_elements = dialogue_item.get("personality_elements", [])
                        covered_elements.update(item_elements)

            coverage_percentage = 0
            if expected_elements:
                covered_count = len(
                    covered_elements.intersection(set(expected_elements))
                )
                coverage_percentage = (covered_count / len(expected_elements)) * 100

            coverage["hosts"][host_name] = {
                "coverage_percentage": coverage_percentage,
                "total_appearances": total_appearances,
                "covered_elements": list(covered_elements),
            }

        # Calculate overall coverage
        host_coverages = [
            data["coverage_percentage"] for data in coverage["hosts"].values()
        ]
        coverage["overall_coverage"] = (
            sum(host_coverages) / len(host_coverages) if host_coverages else 0
        )

        return coverage

    def validate_personality_adaptation(
        self, adapted_dialogue: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate personality adaptation quality"""

        validation = {
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "adaptation_score": 0,
            "consistency_score": 0,
            "naturalness_score": 0,
            "overall_score": 0,
        }

        if not adapted_dialogue:
            validation["is_valid"] = False
            validation["issues"].append("No adapted dialogue provided")
            return validation

        # Check adaptation metadata
        metadata = adapted_dialogue.get("personality_metadata", {})

        adaptation_quality = metadata.get("adaptation_quality", 0)
        voice_consistency = metadata.get("voice_consistency", 0)
        natural_flow = metadata.get("natural_flow", 0)

        validation["adaptation_score"] = adaptation_quality
        validation["consistency_score"] = voice_consistency
        validation["naturalness_score"] = natural_flow

        # Check for issues
        if adaptation_quality < 70:
            validation["issues"].append(
                "Low adaptation quality - personalities not well represented"
            )
        if voice_consistency < 80:
            validation["issues"].append(
                "Inconsistent voice characteristics across segments"
            )
        if natural_flow < 75:
            validation["issues"].append(
                "Unnatural dialogue flow - needs more interjections and cross-references"
            )

        # Generate suggestions
        if adaptation_quality < 85:
            validation["suggestions"].append(
                "Increase personality element density in dialogue"
            )
        if voice_consistency < 90:
            validation["suggestions"].append(
                "Ensure consistent application of voice characteristics"
            )
        if natural_flow < 85:
            validation["suggestions"].append(
                "Add more natural interjections and cross-references"
            )

        # Calculate overall score
        validation["overall_score"] = (
            adaptation_quality * 0.4 + voice_consistency * 0.3 + natural_flow * 0.3
        )

        return validation

    def get_adaptation_summary(self, adapted_dialogue: Dict[str, Any]) -> str:
        """Generate human-readable summary of personality adaptation"""

        if not adapted_dialogue:
            return "No personality adaptation available"

        metadata = adapted_dialogue.get("adaptation_metadata", {})
        enhancement_summary = adapted_dialogue.get("enhancement_summary", {})

        summary = f"Personality Adaptation Summary\n\n"

        # Adaptation statistics
        total_adaptations = metadata.get("total_adaptations", 0)
        summary += f"Total Adaptations: {total_adaptations}\n"

        # Quality scores
        consistency_score = metadata.get("personality_consistency_score", 0)
        flow_score = metadata.get("natural_flow_score", 0)
        summary += f"Personality Consistency: {consistency_score}%\n"
        summary += f"Natural Flow: {flow_score}%\n"

        # Enhancement details
        interjections = enhancement_summary.get("total_interjections", 0)
        cross_refs = enhancement_summary.get("total_cross_references", 0)
        humor_instances = enhancement_summary.get("humor_instances", 0)

        summary += f"\nEnhancements:\n"
        summary += f"  Interjections: {interjections}\n"
        summary += f"  Cross-references: {cross_refs}\n"
        summary += f"  Humor instances: {humor_instances}\n"

        # Strategy
        strategy = metadata.get("adaptation_strategy", "Not specified")
        summary += f"\nAdaptation Strategy: {strategy}\n"

        return summary
