from typing import Dict, List, Optional, Any
from ..openai_service import OpenAIService
from ..duration_calculator import DurationCalculator
import logging

logger = logging.getLogger(__name__)


class DialogueDistributionAgent:
    """
    Dialogue Distribution Agent - Assign content between hosts strategically

    Purpose: Assign content between hosts strategically
    Tasks:
    - Divide content between Host 1 and Host 2
    - Create natural back-and-forth conversation patterns
    - Assign different perspectives or roles to each host
    - Ensure balanced speaking time (roughly 50/50 split)
    - Plan interruptions, questions, and responses

    Output: Content mapped to specific hosts with interaction cues
    """

    def __init__(self):
        self.openai_service = OpenAIService()

    def distribute_dialogue(
        self,
        script_content: Dict[str, Any],
        content_plan: Dict[str, Any],
        host_personalities: Dict[str, Any],
        distribution_preferences: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Distribute dialogue content between hosts strategically

        Args:
            script_content: Raw script content from Script Generation Agent
            content_plan: Content plan with structure and timing
            host_personalities: Host personality definitions
            distribution_preferences: Optional distribution preferences

        Returns:
            Dialogue distribution with host assignments and interaction cues
        """
        logger.info("Starting dialogue distribution between hosts")

        # Apply distribution preferences
        preferences = self._apply_distribution_preferences(distribution_preferences)

        # Analyze host roles and capabilities
        host_analysis = self._analyze_host_roles(host_personalities, preferences)

        # Generate dialogue distribution
        distribution = self._generate_dialogue_distribution(
            script_content, content_plan, host_analysis, preferences
        )

        if not distribution:
            logger.error("Failed to generate dialogue distribution")
            return None

        # Enhance distribution with interaction patterns
        enhanced_distribution = self._enhance_with_interaction_patterns(
            distribution, host_analysis, preferences
        )

        logger.info("Dialogue distribution completed")
        return enhanced_distribution

    def _apply_distribution_preferences(
        self, distribution_preferences: Optional[Dict]
    ) -> Dict[str, Any]:
        """Apply distribution preferences with defaults"""
        default_preferences = {
            "balance_target": 50,  # 50/50 split
            "interaction_style": "collaborative",
            "role_specialization": True,
            "natural_interruptions": True,
            "cross_references": True,
            "question_response_patterns": True,
        }

        if distribution_preferences:
            default_preferences.update(distribution_preferences)

        return default_preferences

    def _analyze_host_roles(
        self, host_personalities: Dict[str, Any], preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze host personalities to determine optimal roles"""

        host_analysis = {
            "hosts": {},
            "role_assignments": {},
            "interaction_dynamics": {},
        }

        for host_key, host_data in host_personalities.items():
            personality = host_data.get("personality", "")
            role = host_data.get("role", "co-host")
            name = host_data.get("name", f"Host {host_key[-1]}")

            # Analyze personality for role assignment
            role_analysis = self._analyze_personality_for_role(personality, role)

            host_analysis["hosts"][host_key] = {
                "name": name,
                "personality": personality,
                "assigned_role": role,
                "strengths": role_analysis["strengths"],
                "content_types": role_analysis["content_types"],
                "interaction_style": role_analysis["interaction_style"],
            }

        # Determine interaction dynamics
        if len(host_analysis["hosts"]) == 2:
            host_keys = list(host_analysis["hosts"].keys())
            host1_data = host_analysis["hosts"][host_keys[0]]
            host2_data = host_analysis["hosts"][host_keys[1]]

            host_analysis["interaction_dynamics"] = (
                self._determine_interaction_dynamics(
                    host1_data, host2_data, preferences
                )
            )

        return host_analysis

    def _analyze_personality_for_role(
        self, personality: str, assigned_role: str
    ) -> Dict[str, Any]:
        """Analyze personality traits to determine content assignment strengths"""

        personality_lower = personality.lower()

        # Define personality-based content preferences
        role_analysis = {
            "strengths": [],
            "content_types": [],
            "interaction_style": "balanced",
        }

        # Analytical personalities
        if any(
            word in personality_lower
            for word in ["analytical", "logical", "technical", "detailed"]
        ):
            role_analysis["strengths"].extend(["facts", "analysis", "explanations"])
            role_analysis["content_types"].extend(
                ["research_heavy", "technical_details", "comparisons"]
            )
            role_analysis["interaction_style"] = "methodical"

        # Enthusiastic personalities
        if any(
            word in personality_lower
            for word in ["enthusiastic", "energetic", "passionate", "excited"]
        ):
            role_analysis["strengths"].extend(
                ["engagement", "storytelling", "reactions"]
            )
            role_analysis["content_types"].extend(
                ["anecdotes", "examples", "emotional_responses"]
            )
            role_analysis["interaction_style"] = "dynamic"

        # Curious personalities
        if any(
            word in personality_lower
            for word in ["curious", "questioning", "inquisitive"]
        ):
            role_analysis["strengths"].extend(
                ["questions", "exploration", "clarification"]
            )
            role_analysis["content_types"].extend(
                ["follow_up_questions", "deeper_dives", "clarifications"]
            )

        # Relatable personalities
        if any(
            word in personality_lower
            for word in ["relatable", "friendly", "approachable", "warm"]
        ):
            role_analysis["strengths"].extend(
                ["connection", "examples", "accessibility"]
            )
            role_analysis["content_types"].extend(
                ["personal_examples", "simplification", "audience_connection"]
            )

        # Storyteller personalities
        if any(
            word in personality_lower
            for word in ["storyteller", "narrative", "creative"]
        ):
            role_analysis["strengths"].extend(["narratives", "examples", "metaphors"])
            role_analysis["content_types"].extend(
                ["stories", "analogies", "creative_explanations"]
            )

        # Default assignments if no specific traits found
        if not role_analysis["strengths"]:
            role_analysis["strengths"] = ["general_discussion", "balanced_contribution"]
            role_analysis["content_types"] = ["general_content"]

        return role_analysis

    def _determine_interaction_dynamics(
        self,
        host1_data: Dict[str, Any],
        host2_data: Dict[str, Any],
        preferences: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine how hosts should interact based on their personalities"""

        dynamics = {
            "primary_questioner": None,
            "primary_explainer": None,
            "interaction_pattern": "balanced",
            "natural_roles": {},
        }

        host1_style = host1_data.get("interaction_style", "balanced")
        host2_style = host2_data.get("interaction_style", "balanced")

        # Determine primary roles
        if "questions" in host1_data.get(
            "strengths", []
        ) and "explanations" in host2_data.get("strengths", []):
            dynamics["primary_questioner"] = host1_data["name"]
            dynamics["primary_explainer"] = host2_data["name"]
            dynamics["interaction_pattern"] = "question_answer"
        elif "questions" in host2_data.get(
            "strengths", []
        ) and "explanations" in host1_data.get("strengths", []):
            dynamics["primary_questioner"] = host2_data["name"]
            dynamics["primary_explainer"] = host1_data["name"]
            dynamics["interaction_pattern"] = "question_answer"
        elif host1_style == "methodical" and host2_style == "dynamic":
            dynamics["primary_explainer"] = host1_data["name"]
            dynamics["primary_questioner"] = host2_data["name"]
            dynamics["interaction_pattern"] = "analytical_enthusiastic"
        elif host2_style == "methodical" and host1_style == "dynamic":
            dynamics["primary_explainer"] = host2_data["name"]
            dynamics["primary_questioner"] = host1_data["name"]
            dynamics["interaction_pattern"] = "analytical_enthusiastic"
        else:
            dynamics["interaction_pattern"] = "collaborative"

        # Assign natural roles
        dynamics["natural_roles"] = {
            host1_data["name"]: self._assign_natural_role(host1_data, dynamics),
            host2_data["name"]: self._assign_natural_role(host2_data, dynamics),
        }

        return dynamics

    def _assign_natural_role(
        self, host_data: Dict[str, Any], dynamics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assign natural conversation role to a host"""

        host_name = host_data["name"]
        strengths = host_data.get("strengths", [])

        role_assignment = {
            "primary_functions": [],
            "secondary_functions": [],
            "interaction_triggers": [],
        }

        # Primary functions based on strengths
        if "facts" in strengths:
            role_assignment["primary_functions"].append("fact_presentation")
        if "analysis" in strengths:
            role_assignment["primary_functions"].append("analysis_delivery")
        if "questions" in strengths:
            role_assignment["primary_functions"].append("question_asking")
        if "storytelling" in strengths:
            role_assignment["primary_functions"].append("example_sharing")
        if "engagement" in strengths:
            role_assignment["primary_functions"].append("audience_engagement")

        # Secondary functions
        if "connection" in strengths:
            role_assignment["secondary_functions"].append("topic_bridging")
        if "reactions" in strengths:
            role_assignment["secondary_functions"].append("response_reactions")
        if "clarification" in strengths:
            role_assignment["secondary_functions"].append("clarification_requests")

        # Interaction triggers
        if dynamics["primary_questioner"] == host_name:
            role_assignment["interaction_triggers"].extend(
                ["new_topic_questions", "follow_up_questions"]
            )
        if dynamics["primary_explainer"] == host_name:
            role_assignment["interaction_triggers"].extend(
                ["detailed_explanations", "fact_elaboration"]
            )

        return role_assignment

    def _generate_dialogue_distribution(
        self,
        script_content: Dict[str, Any],
        content_plan: Dict[str, Any],
        host_analysis: Dict[str, Any],
        preferences: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate dialogue distribution using OpenAI"""

        prompt = f"""
        You are a podcast dialogue distribution specialist. Distribute the following script content between hosts strategically.

        SCRIPT CONTENT:
        {script_content}

        CONTENT PLAN:
        {content_plan}

        HOST ANALYSIS:
        {host_analysis}

        DISTRIBUTION PREFERENCES:
        {preferences}

        Create a strategic dialogue distribution that:
        1. Assigns content to hosts based on their strengths and personalities
        2. Creates natural back-and-forth conversation patterns
        3. Ensures balanced speaking time (target: {preferences.get("balance_target", 50)}% each)
        4. Plans natural interruptions, questions, and responses
        5. Includes cross-references where hosts mention each other's points

        Format as JSON:
        {{
            "distribution_metadata": {{
                "total_segments": 0,
                "host_balance": {{"host1": 50, "host2": 50}},
                "interaction_points": 0,
                "distribution_strategy": "strategy_description"
            }},
            "distributed_segments": [
                {{
                    "segment_id": "intro",
                    "content_type": "introduction",
                    "primary_host": "Host Name",
                    "secondary_host": "Other Host Name",
                    "content": "Segment content",
                    "interaction_cues": [
                        {{
                            "type": "question",
                            "from_host": "Host 1",
                            "to_host": "Host 2",
                            "trigger": "After explaining X",
                            "content": "What do you think about Y?"
                        }}
                    ],
                    "speaking_distribution": {{"host1": 60, "host2": 40}},
                    "estimated_duration": 1.5
                }}
            ],
            "interaction_patterns": [
                {{
                    "pattern_type": "question_response",
                    "frequency": "high",
                    "hosts_involved": ["Host 1", "Host 2"],
                    "typical_triggers": ["After fact presentation", "Before topic transition"]
                }}
            ],
            "cross_references": [
                {{
                    "reference_type": "callback",
                    "from_segment": "segment_2",
                    "to_segment": "segment_1",
                    "host": "Host 2",
                    "content": "As Host 1 mentioned earlier..."
                }}
            ]
        }}
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert podcast dialogue distribution specialist who creates natural, engaging conversations between hosts.",
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
                logger.error(f"Failed to parse dialogue distribution JSON: {e}")
                return None

        return None

    def _enhance_with_interaction_patterns(
        self,
        distribution: Dict[str, Any],
        host_analysis: Dict[str, Any],
        preferences: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enhance distribution with additional interaction patterns"""

        if not distribution:
            return distribution

        # Add interaction metadata
        distribution["interaction_metadata"] = {
            "natural_interruptions": self._plan_natural_interruptions(
                distribution, host_analysis
            ),
            "question_response_chains": self._plan_question_response_chains(
                distribution, host_analysis
            ),
            "topic_transitions": self._plan_topic_transitions(
                distribution, host_analysis
            ),
            "energy_management": self._plan_energy_management(
                distribution, host_analysis
            ),
        }

        # Enhance segments with detailed interaction cues
        for segment in distribution.get("distributed_segments", []):
            self._enhance_segment_interactions(segment, host_analysis, preferences)

        return distribution

    def _plan_natural_interruptions(
        self, distribution: Dict[str, Any], host_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Plan natural interruption points"""

        interruptions = []
        segments = distribution.get("distributed_segments", [])

        for i, segment in enumerate(segments):
            if (
                segment.get("estimated_duration", 0) > 2
            ):  # Longer segments need interruptions
                interruptions.append(
                    {
                        "segment_id": segment.get("segment_id", f"segment_{i}"),
                        "interruption_point": "mid_segment",
                        "type": "clarification_question",
                        "interrupting_host": self._determine_interrupting_host(
                            segment, host_analysis
                        ),
                        "purpose": "maintain_engagement",
                    }
                )

        return interruptions

    def _plan_question_response_chains(
        self, distribution: Dict[str, Any], host_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Plan question-response conversation chains"""

        chains = []
        dynamics = host_analysis.get("interaction_dynamics", {})
        questioner = dynamics.get("primary_questioner")
        explainer = dynamics.get("primary_explainer")

        if questioner and explainer:
            chains.append(
                {
                    "chain_type": "primary_q_and_a",
                    "questioner": questioner,
                    "explainer": explainer,
                    "frequency": "high",
                    "typical_pattern": f"{questioner} asks -> {explainer} explains -> {questioner} follows up",
                }
            )

        return chains

    def _plan_topic_transitions(
        self, distribution: Dict[str, Any], host_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Plan smooth topic transitions"""

        transitions = []
        segments = distribution.get("distributed_segments", [])

        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]

            transitions.append(
                {
                    "from_segment": current_segment.get("segment_id"),
                    "to_segment": next_segment.get("segment_id"),
                    "transition_host": self._determine_transition_host(
                        current_segment, next_segment, host_analysis
                    ),
                    "transition_type": "bridge",
                    "suggested_approach": "Connect current topic to next topic naturally",
                }
            )

        return transitions

    def _plan_energy_management(
        self, distribution: Dict[str, Any], host_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan energy level management throughout the conversation"""

        return {
            "opening_energy": "high",
            "mid_point_boost": "moderate_increase",
            "closing_energy": "strong_finish",
            "energy_host": self._determine_energy_host(host_analysis),
            "energy_techniques": ["enthusiasm", "questions", "reactions", "examples"],
        }

    def _enhance_segment_interactions(
        self,
        segment: Dict[str, Any],
        host_analysis: Dict[str, Any],
        preferences: Dict[str, Any],
    ) -> None:
        """Enhance individual segment with detailed interaction cues"""

        # Add detailed speaking cues
        if "interaction_cues" not in segment:
            segment["interaction_cues"] = []

        # Add natural conversation elements
        segment["conversation_elements"] = {
            "agreements": self._generate_agreement_cues(segment, host_analysis),
            "build_ups": self._generate_build_up_cues(segment, host_analysis),
            "reactions": self._generate_reaction_cues(segment, host_analysis),
        }

    def _determine_interrupting_host(
        self, segment: Dict[str, Any], host_analysis: Dict[str, Any]
    ) -> str:
        """Determine which host should interrupt in a segment"""

        primary_host = segment.get("primary_host", "")
        secondary_host = segment.get("secondary_host", "")

        # The non-primary host typically interrupts
        return secondary_host if secondary_host else primary_host

    def _determine_transition_host(
        self,
        current_segment: Dict[str, Any],
        next_segment: Dict[str, Any],
        host_analysis: Dict[str, Any],
    ) -> str:
        """Determine which host should handle topic transition"""

        # Prefer the host who will be primary in the next segment
        next_primary = next_segment.get("primary_host", "")
        if next_primary:
            return next_primary

        # Fallback to current primary
        return current_segment.get("primary_host", "Host 1")

    def _determine_energy_host(self, host_analysis: Dict[str, Any]) -> str:
        """Determine which host should manage energy levels"""

        for host_key, host_data in host_analysis.get("hosts", {}).items():
            if "engagement" in host_data.get("strengths", []):
                return host_data.get("name", "Host 1")

        # Fallback to first host
        hosts = list(host_analysis.get("hosts", {}).values())
        return hosts[0].get("name", "Host 1") if hosts else "Host 1"

    def _generate_agreement_cues(
        self, segment: Dict[str, Any], host_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate natural agreement cues"""

        return [
            "Absolutely",
            "That's exactly right",
            "I completely agree",
            "That's a great point",
            "Exactly what I was thinking",
        ]

    def _generate_build_up_cues(
        self, segment: Dict[str, Any], host_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate build-up conversation cues"""

        return [
            "And building on that...",
            "What's even more interesting is...",
            "That reminds me of...",
            "Speaking of which...",
            "On a related note...",
        ]

    def _generate_reaction_cues(
        self, segment: Dict[str, Any], host_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate natural reaction cues"""

        return [
            "Wow, that's fascinating",
            "I had no idea",
            "That's incredible",
            "Really? Tell me more",
            "That's surprising",
        ]

    def validate_dialogue_distribution(
        self, distribution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate dialogue distribution quality and balance"""

        validation = {
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "balance_score": 0,
            "interaction_score": 0,
            "overall_score": 0,
        }

        if not distribution:
            validation["is_valid"] = False
            validation["issues"].append("No distribution data provided")
            return validation

        # Check host balance
        metadata = distribution.get("distribution_metadata", {})
        host_balance = metadata.get("host_balance", {})

        if host_balance:
            balance_values = list(host_balance.values())
            if len(balance_values) == 2:
                balance_diff = abs(balance_values[0] - balance_values[1])
                validation["balance_score"] = max(0, 100 - balance_diff * 2)

                if balance_diff > 20:
                    validation["issues"].append(
                        f"Host balance is uneven: {balance_diff}% difference"
                    )

        # Check interaction quality
        segments = distribution.get("distributed_segments", [])
        total_interactions = sum(
            len(seg.get("interaction_cues", [])) for seg in segments
        )

        if segments:
            interactions_per_segment = total_interactions / len(segments)
            validation["interaction_score"] = min(100, interactions_per_segment * 25)

            if interactions_per_segment < 1:
                validation["suggestions"].append(
                    "Add more interaction cues between hosts"
                )

        # Calculate overall score
        validation["overall_score"] = (
            validation["balance_score"] * 0.6 + validation["interaction_score"] * 0.4
        )

        return validation

    def get_distribution_summary(self, distribution: Dict[str, Any]) -> str:
        """Generate human-readable summary of dialogue distribution"""

        if not distribution:
            return "No dialogue distribution available"

        metadata = distribution.get("distribution_metadata", {})
        segments = distribution.get("distributed_segments", [])

        summary = f"Dialogue Distribution Summary\n\n"
        summary += f"Total Segments: {metadata.get('total_segments', len(segments))}\n"

        # Host balance
        host_balance = metadata.get("host_balance", {})
        if host_balance:
            summary += f"Host Balance:\n"
            for host, percentage in host_balance.items():
                summary += f"  {host}: {percentage}%\n"

        # Interaction points
        interaction_points = metadata.get("interaction_points", 0)
        summary += f"Interaction Points: {interaction_points}\n"

        # Strategy
        strategy = metadata.get("distribution_strategy", "Not specified")
        summary += f"Distribution Strategy: {strategy}\n"

        # Segments overview
        summary += f"\nSegments Overview:\n"
        for i, segment in enumerate(segments[:5], 1):  # Show first 5 segments
            primary_host = segment.get("primary_host", "Unknown")
            content_type = segment.get("content_type", "general")
            duration = segment.get("estimated_duration", 0)
            summary += f"  {i}. {content_type.title()} - Primary: {primary_host} ({duration} min)\n"

        if len(segments) > 5:
            summary += f"  ... and {len(segments) - 5} more segments\n"

        return summary
