from .base_agent import BaseIntelligentAgent, AgentDecision
from typing import Dict, Any, Optional, List
from datetime import datetime


class VoicePersonalityAgent(BaseIntelligentAgent):
    """
    Intelligent voice emotion and personality adaptation agent

    Responsibilities:
    - Analyze content for optimal emotion settings
    - Learn user voice preferences
    - Optimize speaker dynamics and interactions
    - Adapt voice characteristics based on content context
    - Balance naturalness with engagement
    """

    def __init__(self):
        super().__init__("VoicePersonality")
        self.user_voice_preferences = {}  # user_id -> preferences
        self.content_emotion_patterns = {}  # content_type -> optimal_emotion
        self.speaker_interaction_patterns = {}  # speaker_combo -> interaction_style
        self.naturalness_feedback = {}  # voice_settings -> user_satisfaction

    async def make_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """
        Make intelligent voice personality decision

        Context types:
        - emotion_optimization: Select optimal emotion for content
        - speaker_dynamics: Optimize speaker interactions
        - voice_adaptation: Adapt voice settings for user preferences
        """

        decision_type = context.get("decision_type")

        if decision_type == "emotion_optimization":
            return await self._optimize_content_emotion(context)
        elif decision_type == "speaker_dynamics":
            return await self._optimize_speaker_dynamics(context)
        elif decision_type == "voice_adaptation":
            return await self._adapt_voice_settings(context)
        else:
            raise ValueError(f"Unknown decision type: {decision_type}")

    async def _optimize_content_emotion(self, context: Dict[str, Any]) -> AgentDecision:
        """Optimize emotion settings for specific content"""

        content_text = context.get("content_text", "")
        speaker_role = context.get("speaker_role", "host")
        content_type = context.get("content_type", "general")  # intro, main, conclusion
        user_id = context.get("user_id")

        # Analyze content characteristics
        content_analysis = self._analyze_content_characteristics(content_text)

        # Get user preferences if available
        user_preferences = self.user_voice_preferences.get(user_id, {})

        # Select optimal emotion mode
        optimal_emotion = self._select_optimal_emotion_mode(
            content_analysis, speaker_role, content_type, user_preferences
        )

        # Calculate optimal exaggeration within emotion mode
        optimal_exaggeration = self._calculate_optimal_exaggeration(
            optimal_emotion, content_analysis, speaker_role
        )

        # Determine other voice parameters
        voice_settings = {
            "emotion_mode": optimal_emotion,
            "exaggeration": optimal_exaggeration,
            "speed_factor": self._calculate_optimal_speed(speaker_role, content_type),
            "cfg_weight": self._calculate_optimal_cfg_weight(
                optimal_emotion, speaker_role
            ),
            "temperature": self._calculate_optimal_temperature(content_analysis),
        }

        # Calculate confidence based on data availability
        confidence = self._calculate_emotion_confidence(
            content_analysis, user_preferences
        )

        reasoning = f"Selected {optimal_emotion} emotion (exaggeration: {optimal_exaggeration:.2f}) for {speaker_role} based on content analysis"

        return AgentDecision(
            agent_name=self.agent_name,
            decision_type="emotion_optimization",
            confidence=confidence,
            reasoning=reasoning,
            data={
                "voice_settings": voice_settings,
                "content_analysis": content_analysis,
                "reasoning_factors": self._get_reasoning_factors(
                    content_analysis, speaker_role
                ),
            },
            timestamp=datetime.utcnow(),
        )

    def _analyze_content_characteristics(self, content_text: str) -> Dict[str, Any]:
        """Analyze content to understand emotional requirements"""

        text_lower = content_text.lower()

        characteristics = {
            "excitement_level": 0.0,
            "technical_complexity": 0.0,
            "emotional_intensity": 0.0,
            "conversational_tone": 0.0,
            "formality_level": 0.0,
            "question_density": 0.0,
            "enthusiasm_indicators": 0.0,
        }

        # Excitement indicators
        excitement_words = [
            "amazing",
            "incredible",
            "fantastic",
            "exciting",
            "wonderful",
            "awesome",
        ]
        excitement_count = sum(1 for word in excitement_words if word in text_lower)
        characteristics["excitement_level"] = min(1.0, excitement_count / 3.0)

        # Technical complexity
        technical_indicators = [
            "analysis",
            "research",
            "study",
            "methodology",
            "algorithm",
            "framework",
        ]
        tech_count = sum(1 for word in technical_indicators if word in text_lower)
        characteristics["technical_complexity"] = min(1.0, tech_count / 5.0)

        # Emotional intensity (exclamation marks, strong words)
        exclamation_count = content_text.count("!")
        strong_words = ["love", "hate", "incredible", "terrible", "brilliant", "awful"]
        strong_word_count = sum(1 for word in strong_words if word in text_lower)
        characteristics["emotional_intensity"] = min(
            1.0, (exclamation_count + strong_word_count) / 5.0
        )

        # Conversational tone
        conversation_indicators = [
            "you know",
            "I think",
            "what do you",
            "let's",
            "we can",
        ]
        conv_count = sum(
            1 for phrase in conversation_indicators if phrase in text_lower
        )
        characteristics["conversational_tone"] = min(1.0, conv_count / 3.0)

        # Formality level
        formal_words = [
            "furthermore",
            "consequently",
            "therefore",
            "however",
            "moreover",
        ]
        formal_count = sum(1 for word in formal_words if word in text_lower)
        characteristics["formality_level"] = min(1.0, formal_count / 3.0)

        # Question density
        question_count = content_text.count("?")
        word_count = len(content_text.split())
        characteristics["question_density"] = min(
            1.0, (question_count / max(1, word_count)) * 100
        )

        # Enthusiasm indicators
        enthusiasm_words = ["great", "love", "excited", "thrilled", "passionate"]
        enthusiasm_count = sum(1 for word in enthusiasm_words if word in text_lower)
        characteristics["enthusiasm_indicators"] = min(1.0, enthusiasm_count / 3.0)

        return characteristics

    def _select_optimal_emotion_mode(
        self,
        content_analysis: Dict[str, Any],
        speaker_role: str,
        content_type: str,
        user_preferences: Dict[str, Any],
    ) -> str:
        """Select the most appropriate emotion mode"""

        # User preference override
        if user_preferences.get("preferred_emotion"):
            return user_preferences["preferred_emotion"]

        # Role-based constraints
        if speaker_role == "expert":
            # Experts should be more measured
            max_emotion = "CONVERSATIONAL"
        elif speaker_role == "narrator":
            # Narrators can be more expressive
            max_emotion = "EXPRESSIVE"
        else:
            # Hosts can use full range
            max_emotion = "DRAMATIC"

        # Content-driven selection
        excitement = content_analysis["excitement_level"]
        emotional_intensity = content_analysis["emotional_intensity"]
        formality = content_analysis["formality_level"]

        # Calculate emotion score
        emotion_score = (excitement + emotional_intensity) / 2.0 - (formality * 0.3)

        # Map to emotion modes
        if emotion_score < 0.2:
            selected_emotion = "NEUTRAL"
        elif emotion_score < 0.4:
            selected_emotion = "CONVERSATIONAL"
        elif emotion_score < 0.7:
            selected_emotion = "EXPRESSIVE"
        else:
            selected_emotion = "DRAMATIC"

        # Apply role constraints
        emotion_levels = ["NEUTRAL", "CONVERSATIONAL", "EXPRESSIVE", "DRAMATIC"]
        max_level_index = emotion_levels.index(max_emotion)
        selected_level_index = emotion_levels.index(selected_emotion)

        if selected_level_index > max_level_index:
            selected_emotion = max_emotion

        return selected_emotion

    def _calculate_optimal_exaggeration(
        self, emotion_mode: str, content_analysis: Dict[str, Any], speaker_role: str
    ) -> float:
        """Calculate optimal exaggeration within the emotion mode range"""

        # Base exaggeration from emotion mode
        base_exaggerations = {
            "NEUTRAL": 0.4,
            "CONVERSATIONAL": 0.6,
            "EXPRESSIVE": 0.8,
            "DRAMATIC": 1.2,
        }

        base_exaggeration = base_exaggerations[emotion_mode]

        # Adjust based on content characteristics
        excitement_factor = content_analysis["excitement_level"]
        technical_factor = content_analysis["technical_complexity"]

        # More technical content = less exaggeration
        # More exciting content = slightly more exaggeration
        adjustment = (excitement_factor * 0.1) - (technical_factor * 0.15)

        # Role-based fine-tuning
        role_adjustments = {
            "expert": -0.1,  # More measured
            "host": 0.0,  # Baseline
            "narrator": -0.05,  # Slightly more controlled
        }

        role_adjustment = role_adjustments.get(speaker_role, 0.0)

        final_exaggeration = base_exaggeration + adjustment + role_adjustment

        # Ensure within reasonable bounds
        return max(0.1, min(1.5, final_exaggeration))

    def _calculate_optimal_speed(self, speaker_role: str, content_type: str) -> float:
        """Calculate optimal speaking speed"""

        base_speeds = {
            "expert": 0.95,  # Slightly slower for clarity
            "host": 1.0,  # Normal speed
            "narrator": 1.05,  # Slightly faster for engagement
        }

        base_speed = base_speeds.get(speaker_role, 1.0)

        # Content type adjustments
        type_adjustments = {
            "introduction": -0.05,  # Slower for introductions
            "conclusion": -0.03,  # Slightly slower for conclusions
            "transition": 0.05,  # Faster for transitions
            "main_content": 0.0,  # Normal for main content
        }

        type_adjustment = type_adjustments.get(content_type, 0.0)

        return base_speed + type_adjustment

    def _calculate_optimal_cfg_weight(
        self, emotion_mode: str, speaker_role: str
    ) -> float:
        """Calculate optimal CFG weight for control vs naturalness balance"""

        # Base CFG weights by emotion (higher = more control, less natural variation)
        base_cfg_weights = {
            "NEUTRAL": 0.7,  # High control for professional tone
            "CONVERSATIONAL": 0.5,  # Balanced
            "EXPRESSIVE": 0.4,  # Lower control for natural expressiveness
            "DRAMATIC": 0.3,  # Lowest control for natural drama
        }

        base_cfg = base_cfg_weights[emotion_mode]

        # Role adjustments
        if speaker_role == "expert":
            base_cfg += 0.1  # Experts need more control
        elif speaker_role == "narrator":
            base_cfg -= 0.05  # Narrators can be more natural

        return max(0.2, min(0.8, base_cfg))

    def _calculate_optimal_temperature(self, content_analysis: Dict[str, Any]) -> float:
        """Calculate optimal temperature for voice generation"""

        base_temperature = 0.6

        # Technical content needs more consistency (lower temperature)
        if content_analysis["technical_complexity"] > 0.5:
            base_temperature -= 0.2

        # Conversational content can have more variation
        if content_analysis["conversational_tone"] > 0.5:
            base_temperature += 0.1

        return max(0.3, min(0.9, base_temperature))

    def _calculate_emotion_confidence(
        self, content_analysis: Dict[str, Any], user_preferences: Dict[str, Any]
    ) -> float:
        """Calculate confidence in emotion selection"""

        base_confidence = 0.7

        # Higher confidence if we have user preferences
        if user_preferences:
            base_confidence += 0.15

        # Higher confidence for clear content indicators
        clear_indicators = (
            content_analysis["excitement_level"] > 0.7
            or content_analysis["technical_complexity"] > 0.7
            or content_analysis["formality_level"] > 0.7
        )

        if clear_indicators:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _get_reasoning_factors(
        self, content_analysis: Dict[str, Any], speaker_role: str
    ) -> Dict[str, Any]:
        """Get detailed reasoning factors for decision transparency"""

        return {
            "content_factors": {
                "excitement_detected": content_analysis["excitement_level"] > 0.3,
                "technical_content": content_analysis["technical_complexity"] > 0.4,
                "high_formality": content_analysis["formality_level"] > 0.5,
                "conversational_style": content_analysis["conversational_tone"] > 0.4,
            },
            "role_factors": {
                "speaker_role": speaker_role,
                "role_constraints": self._get_role_constraints(speaker_role),
            },
            "optimization_strategy": "content_aware_emotion_selection",
        }

    def _get_role_constraints(self, speaker_role: str) -> Dict[str, Any]:
        """Get constraints based on speaker role"""

        constraints = {
            "expert": {
                "max_emotion": "CONVERSATIONAL",
                "preferred_exaggeration_range": [0.2, 0.5],
                "speed_preference": "measured",
            },
            "host": {
                "max_emotion": "DRAMATIC",
                "preferred_exaggeration_range": [0.3, 0.8],
                "speed_preference": "normal",
            },
            "narrator": {
                "max_emotion": "EXPRESSIVE",
                "preferred_exaggeration_range": [0.4, 0.7],
                "speed_preference": "engaging",
            },
        }

        return constraints.get(speaker_role, constraints["host"])

    async def _optimize_speaker_dynamics(
        self, context: Dict[str, Any]
    ) -> AgentDecision:
        """Optimize interactions between multiple speakers"""

        speakers = context.get("speakers", [])
        conversation_segments = context.get("conversation_segments", [])
        target_dynamic = context.get(
            "target_dynamic", "balanced"
        )  # balanced, contrasting, harmonious

        # Analyze current speaker characteristics
        speaker_analysis = {}
        for speaker in speakers:
            speaker_analysis[speaker["id"]] = {
                "role": speaker.get("role", "host"),
                "current_emotion": speaker.get("emotion_mode", "CONVERSATIONAL"),
                "personality_traits": speaker.get("personality_traits", []),
            }

        # Optimize speaker contrast and balance
        optimized_dynamics = self._calculate_optimal_speaker_dynamics(
            speaker_analysis, target_dynamic
        )

        # Ensure good conversation flow
        interaction_patterns = self._design_interaction_patterns(
            speakers, conversation_segments
        )

        confidence = (
            0.8 if len(speakers) <= 3 else 0.6
        )  # Less confident with many speakers

        reasoning = f"Optimized dynamics for {len(speakers)} speakers with {target_dynamic} target"

        return AgentDecision(
            agent_name=self.agent_name,
            decision_type="speaker_dynamics",
            confidence=confidence,
            reasoning=reasoning,
            data={
                "optimized_speakers": optimized_dynamics,
                "interaction_patterns": interaction_patterns,
                "dynamic_strategy": target_dynamic,
            },
            timestamp=datetime.utcnow(),
        )

    def _calculate_optimal_speaker_dynamics(
        self, speaker_analysis: Dict[str, Any], target_dynamic: str
    ) -> Dict[str, Any]:
        """Calculate optimal dynamics between speakers"""

        if target_dynamic == "contrasting":
            # Ensure speakers have different characteristics
            return self._create_contrasting_dynamics(speaker_analysis)
        elif target_dynamic == "harmonious":
            # Ensure speakers complement each other
            return self._create_harmonious_dynamics(speaker_analysis)
        else:  # balanced
            # Balance between contrast and harmony
            return self._create_balanced_dynamics(speaker_analysis)

    def _create_contrasting_dynamics(
        self, speaker_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create contrasting speaker characteristics"""

        optimized = {}

        # Assign contrasting emotions and speeds
        emotion_levels = ["NEUTRAL", "CONVERSATIONAL", "EXPRESSIVE"]
        speed_levels = [0.9, 1.0, 1.1]

        for i, (speaker_id, analysis) in enumerate(speaker_analysis.items()):
            optimized[speaker_id] = {
                "emotion_mode": emotion_levels[i % len(emotion_levels)],
                "speed_factor": speed_levels[i % len(speed_levels)],
                "personality_emphasis": "contrasting",
            }

        return optimized

    def _create_harmonious_dynamics(
        self, speaker_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create harmonious speaker characteristics"""

        optimized = {}

        # Use similar emotion levels with slight variations
        base_emotion = "CONVERSATIONAL"
        base_speed = 1.0

        for i, (speaker_id, analysis) in enumerate(speaker_analysis.items()):
            optimized[speaker_id] = {
                "emotion_mode": base_emotion,
                "speed_factor": base_speed + (i * 0.05),  # Slight speed variations
                "personality_emphasis": "harmonious",
            }

        return optimized

    def _create_balanced_dynamics(
        self, speaker_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create balanced speaker characteristics"""

        # Mix of contrasting and harmonious elements
        optimized = {}

        for i, (speaker_id, analysis) in enumerate(speaker_analysis.items()):
            role = analysis["role"]

            if role == "expert":
                emotion = "NEUTRAL"
                speed = 0.95
            elif role == "host":
                emotion = "CONVERSATIONAL"
                speed = 1.0
            else:
                emotion = "CONVERSATIONAL"
                speed = 1.05

            optimized[speaker_id] = {
                "emotion_mode": emotion,
                "speed_factor": speed,
                "personality_emphasis": "balanced",
            }

        return optimized

    def _design_interaction_patterns(
        self, speakers: List[Dict], conversation_segments: List[Dict]
    ) -> Dict[str, Any]:
        """Design optimal interaction patterns between speakers"""

        patterns = {
            "turn_taking_style": "natural",  # natural, structured, dynamic
            "interruption_frequency": "low",  # low, medium, high
            "response_timing": "measured",  # quick, measured, thoughtful
            "energy_flow": "building",  # consistent, building, varied
        }

        # Adjust based on number of speakers
        if len(speakers) > 2:
            patterns["turn_taking_style"] = "structured"
            patterns["interruption_frequency"] = "low"

        return patterns

    async def _adapt_voice_settings(self, context: Dict[str, Any]) -> AgentDecision:
        """Adapt voice settings based on user preferences and feedback"""

        user_id = context.get("user_id")
        current_settings = context.get("current_settings", {})
        user_feedback = context.get("user_feedback", {})

        # Get existing user preferences
        user_preferences = self.user_voice_preferences.get(user_id, {})

        # Update preferences based on feedback
        updated_preferences = self._update_user_preferences(
            user_preferences, user_feedback, current_settings
        )

        # Generate adapted settings
        adapted_settings = self._generate_adapted_settings(
            current_settings, updated_preferences
        )

        # Store updated preferences
        self.user_voice_preferences[user_id] = updated_preferences

        confidence = 0.9 if user_feedback else 0.6
        reasoning = f"Adapted voice settings based on user feedback and preferences"

        return AgentDecision(
            agent_name=self.agent_name,
            decision_type="voice_adaptation",
            confidence=confidence,
            reasoning=reasoning,
            data={
                "adapted_settings": adapted_settings,
                "user_preferences": updated_preferences,
                "adaptation_factors": user_feedback,
            },
            timestamp=datetime.utcnow(),
        )

    def _update_user_preferences(
        self,
        current_preferences: Dict[str, Any],
        feedback: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update user preferences based on feedback"""

        updated = current_preferences.copy()

        # Process different types of feedback
        if "too_dramatic" in feedback:
            updated["max_exaggeration"] = min(updated.get("max_exaggeration", 0.8), 0.5)

        if "too_boring" in feedback:
            updated["min_exaggeration"] = max(updated.get("min_exaggeration", 0.2), 0.4)

        if "too_fast" in feedback:
            updated["max_speed"] = min(updated.get("max_speed", 1.2), 0.9)

        if "too_slow" in feedback:
            updated["min_speed"] = max(updated.get("min_speed", 0.8), 1.1)

        if "preferred_emotion" in feedback:
            updated["preferred_emotion"] = feedback["preferred_emotion"]

        return updated

    def _generate_adapted_settings(
        self, current_settings: Dict[str, Any], preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate adapted settings based on preferences"""

        adapted = current_settings.copy()

        # Apply preference constraints
        if "max_exaggeration" in preferences:
            adapted["exaggeration"] = min(
                adapted.get("exaggeration", 0.6), preferences["max_exaggeration"]
            )

        if "min_exaggeration" in preferences:
            adapted["exaggeration"] = max(
                adapted.get("exaggeration", 0.6), preferences["min_exaggeration"]
            )

        if "max_speed" in preferences:
            adapted["speed_factor"] = min(
                adapted.get("speed_factor", 1.0), preferences["max_speed"]
            )

        if "min_speed" in preferences:
            adapted["speed_factor"] = max(
                adapted.get("speed_factor", 1.0), preferences["min_speed"]
            )

        if "preferred_emotion" in preferences:
            adapted["emotion_mode"] = preferences["preferred_emotion"]

        return adapted

    async def learn_from_outcome(
        self, decision: AgentDecision, outcome: Dict[str, Any]
    ):
        """Learn from voice personality decision outcomes"""

        decision_type = decision.decision_type
        success = outcome.get("success", False)
        user_satisfaction = outcome.get("user_satisfaction", 0.5)  # 0.0-1.0
        naturalness_score = outcome.get("naturalness_score", 0.5)  # 0.0-1.0

        if decision_type == "emotion_optimization":
            # Learn emotion selection effectiveness
            voice_settings = decision.data["voice_settings"]

            learning_entry = {
                "settings": voice_settings,
                "content_characteristics": decision.data["content_analysis"],
                "user_satisfaction": user_satisfaction,
                "naturalness_score": naturalness_score,
                "timestamp": datetime.utcnow(),
            }

            # Store in naturalness feedback
            settings_key = (
                f"{voice_settings['emotion_mode']}_{voice_settings['exaggeration']:.1f}"
            )
            if settings_key not in self.naturalness_feedback:
                self.naturalness_feedback[settings_key] = []

            self.naturalness_feedback[settings_key].append(learning_entry)

            # Update content emotion patterns
            content_type = decision.data.get("content_type", "general")
            if content_type not in self.content_emotion_patterns:
                self.content_emotion_patterns[content_type] = []

            self.content_emotion_patterns[content_type].append(
                {
                    "emotion_mode": voice_settings["emotion_mode"],
                    "satisfaction": user_satisfaction,
                    "naturalness": naturalness_score,
                }
            )

        elif decision_type == "speaker_dynamics":
            # Learn speaker interaction effectiveness
            speakers = decision.data["optimized_speakers"]
            interaction_success = outcome.get("interaction_quality", 0.5)

            speaker_combo_key = "_".join(sorted(speakers.keys()))
            if speaker_combo_key not in self.speaker_interaction_patterns:
                self.speaker_interaction_patterns[speaker_combo_key] = []

            self.speaker_interaction_patterns[speaker_combo_key].append(
                {
                    "dynamics": decision.data["optimized_speakers"],
                    "patterns": decision.data["interaction_patterns"],
                    "success": interaction_success,
                    "timestamp": datetime.utcnow(),
                }
            )

        elif decision_type == "voice_adaptation":
            # Learn user preference adaptation effectiveness
            user_id = outcome.get("user_id")
            if user_id and user_id in self.user_voice_preferences:
                # Update confidence in user preferences
                self.user_voice_preferences[user_id]["confidence"] = user_satisfaction
                self.user_voice_preferences[user_id]["last_updated"] = datetime.utcnow()

        self.logger.info(
            f"Learned from {decision_type}: satisfaction {user_satisfaction:.2f}, naturalness {naturalness_score:.2f}"
        )
