from typing import Dict, List, Optional, Any
from ..openai_service import OpenAIService
from ..duration_calculator import DurationCalculator
from ..content_depth_analyzer import ContentDepthAnalyzer
import logging

logger = logging.getLogger(__name__)


class ContentPlanningAgent:
    """
    Content Planning Agent - Strategic content organization

    Purpose: Strategic content organization
    Tasks:
    - Analyze research findings for relevance and importance
    - Prioritize subtopics based on audience interest and topic significance
    - Create content hierarchy (what to cover first, second, etc.)
    - Estimate time allocation for each section to fit 10-minute constraint
    - Identify natural conversation transition points

    Output: Content outline with time estimates and priority rankings
    """

    def __init__(self):
        self.openai_service = OpenAIService()
        self.duration_calculator = DurationCalculator()
        self.content_analyzer = ContentDepthAnalyzer()

    def plan_content(
        self,
        research_data: Dict[str, Any],
        target_duration: int = 10,
        audience_preferences: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create strategic content plan from research data

        Args:
            research_data: Research data from ResearchAgent
            target_duration: Target duration in minutes
            audience_preferences: Optional audience preferences

        Returns:
            Content plan with hierarchy, timing, and transitions
        """
        logger.info(
            f"Planning content for topic: {research_data.get('main_topic', 'Unknown')}"
        )

        # Apply audience preferences
        preferences = self._apply_audience_preferences(audience_preferences)

        # Generate content plan using OpenAI
        content_plan = self._generate_content_plan(
            research_data, target_duration, preferences
        )

        if not content_plan:
            logger.error("Failed to generate content plan")
            return None

        # Enhance plan with additional analysis
        enhanced_plan = self._enhance_content_plan(
            content_plan, research_data, target_duration
        )

        # Validate duration allocation and adjust if needed
        duration_validated_plan = self._validate_and_adjust_duration(
            enhanced_plan, research_data, target_duration
        )

        logger.info("Content planning completed")
        return duration_validated_plan

    def _apply_audience_preferences(
        self, audience_preferences: Optional[Dict]
    ) -> Dict[str, Any]:
        """Apply audience preferences to content planning"""
        default_preferences = {
            "engagement_style": "conversational",
            "complexity_level": "accessible",
            "pacing": "moderate",
            "priority_factors": ["relevance", "interest", "uniqueness"],
            "transition_style": "smooth",
        }

        if audience_preferences:
            default_preferences.update(audience_preferences)

        return default_preferences

    def _generate_content_plan(
        self,
        research_data: Dict[str, Any],
        target_duration: int,
        preferences: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate content plan using OpenAI"""

        prompt = f"""
        You are a podcast content strategist. Create a strategic content plan for a {target_duration}-minute podcast.

        RESEARCH DATA:
        {research_data}

        AUDIENCE PREFERENCES:
        {preferences}

        Create a content plan with:
        1. Content hierarchy (order of topics based on importance and flow)
        2. Time allocation for each section (must total {target_duration} minutes)
        3. Priority rankings for each subtopic
        4. Natural transition points between topics
        5. Engagement hooks and key moments

        Consider:
        - Audience engagement (start strong, maintain interest)
        - Logical flow (build complexity gradually)
        - Time constraints (fit within {target_duration} minutes)
        - Natural conversation breaks

        Format as JSON:
        {{
            "main_topic": "{research_data.get("main_topic", "")}",
            "target_duration": {target_duration},
            "content_structure": {{
                "intro": {{
                    "duration_minutes": 1.0,
                    "purpose": "Hook audience and introduce topic",
                    "key_elements": ["greeting", "topic_introduction", "preview"]
                }},
                "main_content": [
                    {{
                        "subtopic": "Subtopic Title",
                        "priority_rank": 1,
                        "duration_minutes": 2.5,
                        "placement_rationale": "Why this comes first",
                        "engagement_hooks": ["Interesting fact", "Question"],
                        "transition_to_next": "How to smoothly move to next topic"
                    }}
                ],
                "outro": {{
                    "duration_minutes": 1.0,
                    "purpose": "Wrap up and call to action",
                    "key_elements": ["summary", "takeaways", "closing"]
                }}
            }},
            "transition_points": [
                {{
                    "from_topic": "Topic A",
                    "to_topic": "Topic B",
                    "transition_type": "bridge",
                    "suggested_phrases": ["Speaking of...", "This connects to..."]
                }}
            ],
            "timing_breakdown": {{
                "intro_percentage": 10,
                "main_content_percentage": 80,
                "outro_percentage": 10
            }},
            "engagement_strategy": {{
                "opening_hook": "Strong opening statement or question",
                "mid_point_energizer": "Keep energy up halfway through",
                "closing_impact": "Memorable ending"
            }}
        }}
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert podcast content strategist who creates engaging, well-structured content plans.",
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
                logger.error(f"Failed to parse content plan JSON: {e}")
                return None

        return None

    def _enhance_content_plan(
        self,
        content_plan: Dict[str, Any],
        research_data: Dict[str, Any],
        target_duration: int,
    ) -> Dict[str, Any]:
        """Enhance content plan with additional analysis"""

        if not content_plan:
            return content_plan

        # Analyze content requirements for target duration
        content_requirements = (
            self.duration_calculator.estimate_required_content_for_duration(
                target_duration, "conversational"
            )
        )

        # Assess current content depth
        content_depth_assessment = self.content_analyzer.analyze_topic_depth(
            research_data, target_duration
        )

        # Add planning metadata
        content_plan["planning_metadata"] = {
            "target_duration": target_duration,
            "total_subtopics": len(
                content_plan.get("content_structure", {}).get("main_content", [])
            ),
            "content_density": self._calculate_content_density(content_plan),
            "flow_quality": self._assess_flow_quality(content_plan),
            "timing_validation": self._validate_timing(content_plan),
            "duration_requirements": content_requirements,
            "content_depth_assessment": content_depth_assessment,
            "duration_allocation": self._calculate_duration_allocation(content_plan),
        }

        # Enhance each content section
        if (
            "content_structure" in content_plan
            and "main_content" in content_plan["content_structure"]
        ):
            for section in content_plan["content_structure"]["main_content"]:
                self._enhance_content_section(section, research_data)

        return content_plan

    def _enhance_content_section(
        self, section: Dict[str, Any], research_data: Dict[str, Any]
    ) -> None:
        """Enhance individual content section with additional data"""

        # Find matching research subtopic
        subtopic_title = section.get("subtopic", "")
        matching_research = None

        for subtopic in research_data.get("subtopics", []):
            if subtopic.get("title", "").lower() in subtopic_title.lower():
                matching_research = subtopic
                break

        if matching_research:
            section["research_support"] = {
                "key_facts_count": len(matching_research.get("key_facts", [])),
                "discussion_angles_count": len(
                    matching_research.get("discussion_angles", [])
                ),
                "content_richness": self._assess_content_richness(matching_research),
            }

        # Add section metadata
        section["section_metadata"] = {
            "estimated_word_count": int(
                section.get("duration_minutes", 0) * 150
            ),  # 150 WPM
            "complexity_level": self._assess_section_complexity(section),
            "engagement_potential": self._assess_engagement_potential(section),
        }

    def _calculate_content_density(self, content_plan: Dict[str, Any]) -> str:
        """Calculate content density (light, moderate, dense)"""
        main_content = content_plan.get("content_structure", {}).get("main_content", [])
        total_duration = sum(
            section.get("duration_minutes", 0) for section in main_content
        )

        if total_duration == 0:
            return "unknown"

        topics_per_minute = len(main_content) / total_duration

        if topics_per_minute > 0.5:
            return "dense"
        elif topics_per_minute > 0.3:
            return "moderate"
        else:
            return "light"

    def _assess_flow_quality(self, content_plan: Dict[str, Any]) -> str:
        """Assess the quality of content flow"""
        main_content = content_plan.get("content_structure", {}).get("main_content", [])

        if len(main_content) < 2:
            return "insufficient"

        # Check if priority rankings are logical
        priorities = [section.get("priority_rank", 0) for section in main_content]
        is_logical_order = priorities == sorted(priorities)

        # Check if transitions are defined
        transitions = content_plan.get("transition_points", [])
        has_transitions = len(transitions) >= len(main_content) - 1

        if is_logical_order and has_transitions:
            return "excellent"
        elif is_logical_order or has_transitions:
            return "good"
        else:
            return "needs_improvement"

    def _validate_timing(self, content_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate timing allocation"""
        structure = content_plan.get("content_structure", {})
        target_duration = content_plan.get("target_duration", 10)

        intro_time = structure.get("intro", {}).get("duration_minutes", 0)
        outro_time = structure.get("outro", {}).get("duration_minutes", 0)
        main_content_time = sum(
            section.get("duration_minutes", 0)
            for section in structure.get("main_content", [])
        )

        total_time = intro_time + outro_time + main_content_time

        return {
            "total_allocated_time": total_time,
            "target_time": target_duration,
            "time_difference": total_time - target_duration,
            "is_within_tolerance": abs(total_time - target_duration) <= 0.5,
            "breakdown": {
                "intro_time": intro_time,
                "main_content_time": main_content_time,
                "outro_time": outro_time,
            },
        }

    def _assess_content_richness(self, subtopic: Dict[str, Any]) -> str:
        """Assess richness of content for a subtopic"""
        facts_count = len(subtopic.get("key_facts", []))
        angles_count = len(subtopic.get("discussion_angles", []))
        summary_length = len(subtopic.get("summary", "").split())

        richness_score = facts_count * 2 + angles_count * 1.5 + summary_length / 10

        if richness_score > 15:
            return "rich"
        elif richness_score > 8:
            return "moderate"
        else:
            return "light"

    def _assess_section_complexity(self, section: Dict[str, Any]) -> str:
        """Assess complexity level of a content section"""
        # Simple heuristic based on duration and content
        duration = section.get("duration_minutes", 0)
        hooks_count = len(section.get("engagement_hooks", []))

        if duration > 3 and hooks_count > 2:
            return "complex"
        elif duration > 2 or hooks_count > 1:
            return "moderate"
        else:
            return "simple"

    def _assess_engagement_potential(self, section: Dict[str, Any]) -> str:
        """Assess engagement potential of a content section"""
        hooks = section.get("engagement_hooks", [])
        priority = section.get("priority_rank", 5)

        if len(hooks) > 2 and priority <= 2:
            return "high"
        elif len(hooks) > 1 or priority <= 3:
            return "medium"
        else:
            return "low"

    def _calculate_duration_allocation(
        self, content_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate detailed duration allocation breakdown"""

        structure = content_plan.get("content_structure", {})
        target_duration = content_plan.get("target_duration", 10)

        # Calculate actual allocations
        intro_duration = structure.get("intro", {}).get("duration_minutes", 0)
        outro_duration = structure.get("outro", {}).get("duration_minutes", 0)

        main_content_sections = structure.get("main_content", [])
        main_content_duration = sum(
            section.get("duration_minutes", 0) for section in main_content_sections
        )

        total_duration = intro_duration + main_content_duration + outro_duration

        # Calculate percentages
        intro_percentage = (
            (intro_duration / total_duration * 100) if total_duration > 0 else 0
        )
        main_percentage = (
            (main_content_duration / total_duration * 100) if total_duration > 0 else 0
        )
        outro_percentage = (
            (outro_duration / total_duration * 100) if total_duration > 0 else 0
        )

        # Ideal allocation (intro 10%, main 80%, outro 10%)
        ideal_allocation = {
            "intro": target_duration * 0.1,
            "main_content": target_duration * 0.8,
            "outro": target_duration * 0.1,
        }

        # Calculate deviations
        deviations = {
            "intro": abs(intro_duration - ideal_allocation["intro"]),
            "main_content": abs(
                main_content_duration - ideal_allocation["main_content"]
            ),
            "outro": abs(outro_duration - ideal_allocation["outro"]),
        }

        return {
            "actual_allocation": {
                "intro": intro_duration,
                "main_content": main_content_duration,
                "outro": outro_duration,
                "total": total_duration,
            },
            "percentage_breakdown": {
                "intro": round(intro_percentage, 1),
                "main_content": round(main_percentage, 1),
                "outro": round(outro_percentage, 1),
            },
            "ideal_allocation": ideal_allocation,
            "deviations": deviations,
            "needs_rebalancing": any(dev > 0.5 for dev in deviations.values()),
            "section_details": [
                {
                    "section": section.get("subtopic", f"Section {i + 1}"),
                    "duration": section.get("duration_minutes", 0),
                    "priority": section.get("priority_rank", i + 1),
                    "percentage_of_main": (
                        section.get("duration_minutes", 0) / main_content_duration * 100
                    )
                    if main_content_duration > 0
                    else 0,
                }
                for i, section in enumerate(main_content_sections)
            ],
        }

    def _validate_and_adjust_duration(
        self,
        content_plan: Dict[str, Any],
        research_data: Dict[str, Any],
        target_duration: int,
    ) -> Dict[str, Any]:
        """
        Enhanced duration validation with adaptive thresholds and detailed analysis

        Args:
            content_plan: Current content plan
            research_data: Research data for context
            target_duration: Target duration in minutes

        Returns:
            Adjusted content plan with enhanced duration validation
        """

        logger.info(
            f"🎯 ENHANCED: Validating duration allocation for {target_duration}min target"
        )

        # Get current duration allocation
        duration_allocation = content_plan["planning_metadata"]["duration_allocation"]
        total_allocated = duration_allocation["actual_allocation"]["total"]

        # Enhanced adaptive tolerance based on target duration
        tolerance = self._calculate_adaptive_tolerance(target_duration)
        needs_adjustment = abs(total_allocated - target_duration) > tolerance

        # Calculate accuracy percentage
        accuracy_percentage = 100 - (
            abs(total_allocated - target_duration) / target_duration * 100
        )

        logger.info(
            f"📊 DURATION ANALYSIS: Allocated: {total_allocated:.1f}min, Target: {target_duration}min, Accuracy: {accuracy_percentage:.1f}%"
        )

        if needs_adjustment:
            logger.info(
                f"🔧 ADJUSTMENT NEEDED: {total_allocated:.1f}min allocated vs {target_duration}min target (tolerance: {tolerance:.1f}min)"
            )
            content_plan = self._enhanced_adjust_content_duration(
                content_plan, target_duration, research_data
            )

        # Enhanced content-to-duration mapping
        content_requirements = (
            self._calculate_enhanced_content_requirements_for_duration(
                target_duration, research_data
            )
        )

        # Advanced research completeness validation
        research_completeness = self._assess_research_completeness_for_duration(
            research_data, target_duration
        )

        # Enhanced duration validation with multiple metrics
        content_plan["enhanced_duration_validation"] = {
            "target_duration": target_duration,
            "allocated_duration": total_allocated,
            "accuracy_percentage": accuracy_percentage,
            "adaptive_tolerance": tolerance,
            "within_tolerance": not needs_adjustment,
            "adjustment_applied": needs_adjustment,
            "validation_passed": accuracy_percentage >= 95.0,  # Stricter threshold
            "quality_grade": self._calculate_duration_quality_grade(
                accuracy_percentage
            ),
            "content_requirements": content_requirements,
            "research_completeness": research_completeness,
            "duration_prediction_confidence": self._calculate_duration_confidence(
                content_plan, research_data, target_duration
            ),
            "recommended_adjustments": self._generate_duration_recommendations(
                total_allocated, target_duration, research_completeness
            ),
        }

        # Validate research content depth with enhanced metrics
        depth_validation = self.duration_calculator.validate_duration_accuracy(
            research_data.get("research_sections", []),
            target_duration,
            tolerance_percentage=10,  # Stricter tolerance
        )

        content_plan["enhanced_content_readiness"] = {
            "content_requirements": content_requirements,
            "depth_validation": depth_validation,
            "ready_for_script_generation": (
                depth_validation.get("validation_passed", False)
                and accuracy_percentage >= 90.0
                and research_completeness["completeness_score"] >= 80.0
            ),
            "readiness_score": self._calculate_readiness_score(
                depth_validation, accuracy_percentage, research_completeness
            ),
            "recommended_actions": depth_validation.get("recommendations", []),
            "enhancement_applied": True,
        }

        logger.info(
            f"✅ DURATION VALIDATION: {accuracy_percentage:.1f}% accuracy, readiness: {content_plan['enhanced_content_readiness']['ready_for_script_generation']}"
        )

        return content_plan

    def _calculate_adaptive_tolerance(self, target_duration: float) -> float:
        """Calculate adaptive tolerance based on target duration"""
        # Shorter podcasts need tighter tolerance
        if target_duration <= 5:
            return 0.25  # 15 seconds
        elif target_duration <= 10:
            return 0.5  # 30 seconds
        else:
            return 0.75  # 45 seconds

    def _calculate_enhanced_content_requirements_for_duration(
        self, target_duration: float, research_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced content requirement calculation based on research data"""

        # Analyze research complexity to determine speech rate
        research_complexity = self._assess_research_complexity(research_data)

        # Adaptive words per minute based on content complexity
        base_wpm = {
            "simple": 150,  # Simple, casual topics
            "moderate": 140,  # Balanced complexity
            "complex": 130,  # Technical, analytical topics
            "very_complex": 120,  # Highly technical topics
        }

        effective_wpm = (
            base_wpm.get(research_complexity, 140) * 0.85
        )  # Account for natural conversation flow
        required_words = int(target_duration * effective_wpm)

        # Calculate segment requirements
        estimated_segments = max(3, int(target_duration / 2.5))  # ~2.5 min per segment
        words_per_segment = required_words // estimated_segments

        return {
            "required_words": required_words,
            "effective_wpm": effective_wpm,
            "research_complexity": research_complexity,
            "estimated_segments": estimated_segments,
            "words_per_segment": words_per_segment,
            "content_density": "high"
            if required_words > (target_duration * 150)
            else "moderate",
            "conversation_style": "podcast_dialogue",
        }

    def _assess_research_complexity(self, research_data: Dict[str, Any]) -> str:
        """Assess complexity of research data"""
        complexity_score = 0

        # Check subtopic complexity
        subtopics = research_data.get("subtopics", [])
        for subtopic in subtopics:
            if "statistics" in subtopic and len(subtopic["statistics"]) > 2:
                complexity_score += 2
            if "technical_details" in subtopic:
                complexity_score += 3
            if len(subtopic.get("key_facts", [])) > 5:
                complexity_score += 1

        # Check for technical sections
        research_sections = research_data.get("research_sections", [])
        for section in research_sections:
            if "technical" in section.get("content_type", "").lower():
                complexity_score += 2
            if "analysis" in section.get("content_type", "").lower():
                complexity_score += 1

        # Determine complexity level
        if complexity_score >= 15:
            return "very_complex"
        elif complexity_score >= 10:
            return "complex"
        elif complexity_score >= 5:
            return "moderate"
        else:
            return "simple"

    def _assess_research_completeness_for_duration(
        self, research_data: Dict[str, Any], target_duration: float
    ) -> Dict[str, Any]:
        """Assess if research is complete enough for target duration"""

        completeness_score = 0
        max_score = 0

        # Check subtopic coverage
        subtopics = research_data.get("subtopics", [])
        for subtopic in subtopics:
            max_score += 10
            if subtopic.get("key_facts"):
                completeness_score += 3
            if subtopic.get("discussion_angles"):
                completeness_score += 2
            if subtopic.get("examples"):
                completeness_score += 2
            if subtopic.get("statistics"):
                completeness_score += 2
            if len(subtopic.get("summary", "")) > 50:
                completeness_score += 1

        # Check additional research sections
        research_sections = research_data.get("research_sections", [])
        max_score += len(research_sections) * 5
        completeness_score += len(research_sections) * 5

        # Check key points
        key_points = research_data.get("key_points", [])
        max_score += 20
        completeness_score += min(20, len(key_points) * 2)

        completeness_percentage = (
            (completeness_score / max_score * 100) if max_score > 0 else 0
        )

        return {
            "completeness_score": completeness_percentage,
            "sufficient_for_duration": completeness_percentage
            >= (80 if target_duration <= 10 else 85),
            "content_gaps": self._identify_content_gaps(research_data),
            "expansion_needed": completeness_percentage < 75,
        }

    def _calculate_duration_quality_grade(self, accuracy_percentage: float) -> str:
        """Calculate quality grade for duration accuracy"""
        if accuracy_percentage >= 98:
            return "A+ (Excellent)"
        elif accuracy_percentage >= 95:
            return "A (Very Good)"
        elif accuracy_percentage >= 90:
            return "B+ (Good)"
        elif accuracy_percentage >= 85:
            return "B (Acceptable)"
        elif accuracy_percentage >= 80:
            return "C+ (Needs Improvement)"
        else:
            return "C (Poor)"

    def _calculate_duration_confidence(
        self,
        content_plan: Dict[str, Any],
        research_data: Dict[str, Any],
        target_duration: float,
    ) -> Dict[str, Any]:
        """Calculate confidence level in duration prediction"""

        confidence_factors = {
            "research_completeness": 0.3,
            "content_structure_quality": 0.25,
            "topic_complexity_match": 0.2,
            "historical_accuracy": 0.25,
        }

        # Research completeness factor
        research_completeness = self._assess_research_completeness_for_duration(
            research_data, target_duration
        )
        completeness_factor = min(
            1.0, research_completeness["completeness_score"] / 100
        )

        # Content structure quality
        structure_quality = self._assess_content_structure_quality(content_plan)

        # Topic complexity match
        complexity_match = self._assess_complexity_duration_match(
            research_data, target_duration
        )

        # Historical accuracy (simulated - in real implementation would use actual data)
        historical_accuracy = 0.85  # Placeholder

        overall_confidence = (
            completeness_factor * confidence_factors["research_completeness"]
            + structure_quality * confidence_factors["content_structure_quality"]
            + complexity_match * confidence_factors["topic_complexity_match"]
            + historical_accuracy * confidence_factors["historical_accuracy"]
        )

        confidence_level = (
            "High"
            if overall_confidence >= 0.8
            else "Medium"
            if overall_confidence >= 0.6
            else "Low"
        )

        return {
            "overall_confidence": overall_confidence,
            "confidence_level": confidence_level,
            "confidence_percentage": overall_confidence * 100,
            "contributing_factors": {
                "research_completeness": completeness_factor,
                "content_structure_quality": structure_quality,
                "topic_complexity_match": complexity_match,
                "historical_accuracy": historical_accuracy,
            },
        }

    def _assess_content_structure_quality(self, content_plan: Dict[str, Any]) -> float:
        """Assess quality of content structure for duration prediction"""
        quality_score = 0.0

        structure = content_plan.get("content_structure", {})

        # Check if has proper intro/outro structure
        if structure.get("intro") and structure.get("outro"):
            quality_score += 0.3

        # Check main content organization
        main_content = structure.get("main_content", [])
        if len(main_content) >= 3:  # Good number of sections
            quality_score += 0.3

        # Check for transition planning
        if content_plan.get("transition_points"):
            quality_score += 0.2

        # Check for engagement strategy
        if content_plan.get("engagement_strategy"):
            quality_score += 0.2

        return min(1.0, quality_score)

    def _assess_complexity_duration_match(
        self, research_data: Dict[str, Any], target_duration: float
    ) -> float:
        """Assess if research complexity matches target duration"""
        complexity = self._assess_research_complexity(research_data)

        # Optimal duration ranges for different complexities
        optimal_ranges = {
            "simple": (3, 8),
            "moderate": (5, 12),
            "complex": (8, 20),
            "very_complex": (10, 30),
        }

        min_optimal, max_optimal = optimal_ranges.get(complexity, (5, 15))

        if min_optimal <= target_duration <= max_optimal:
            return 1.0  # Perfect match
        elif target_duration < min_optimal:
            return max(0.3, 1.0 - (min_optimal - target_duration) / min_optimal)
        else:
            return max(0.3, 1.0 - (target_duration - max_optimal) / max_optimal)

    def _generate_duration_recommendations(
        self,
        allocated_duration: float,
        target_duration: float,
        research_completeness: Dict[str, Any],
    ) -> List[str]:
        """Generate specific recommendations for duration optimization"""
        recommendations = []

        duration_diff = allocated_duration - target_duration
        accuracy = 100 - (abs(duration_diff) / target_duration * 100)

        if accuracy < 95:
            if duration_diff > 0.5:
                recommendations.extend(
                    [
                        f"Content allocation is {duration_diff:.1f} minutes over target",
                        "Consider reducing detail in lower-priority sections",
                        "Streamline transitions between topics",
                        "Focus on most engaging content elements",
                    ]
                )
            elif duration_diff < -0.5:
                recommendations.extend(
                    [
                        f"Content allocation is {abs(duration_diff):.1f} minutes under target",
                        "Add more detailed examples or case studies",
                        "Expand discussion angles for key topics",
                        "Include additional expert insights",
                    ]
                )

        if not research_completeness.get("sufficient_for_duration", True):
            recommendations.extend(
                [
                    "Research content may be insufficient for target duration",
                    "Consider expanding research before script generation",
                    f"Current completeness: {research_completeness.get('completeness_score', 0):.1f}%",
                ]
            )

        return recommendations

    def _calculate_readiness_score(
        self,
        depth_validation: Dict[str, Any],
        accuracy_percentage: float,
        research_completeness: Dict[str, Any],
    ) -> float:
        """Calculate overall readiness score for script generation"""

        # Weight different factors
        weights = {
            "duration_accuracy": 0.4,
            "research_completeness": 0.4,
            "depth_validation": 0.2,
        }

        duration_score = accuracy_percentage / 100
        completeness_score = research_completeness.get("completeness_score", 0) / 100
        depth_score = 1.0 if depth_validation.get("validation_passed", False) else 0.5

        overall_score = (
            duration_score * weights["duration_accuracy"]
            + completeness_score * weights["research_completeness"]
            + depth_score * weights["depth_validation"]
        )

        return overall_score

    def _identify_content_gaps(self, research_data: Dict[str, Any]) -> List[str]:
        """Identify gaps in research content"""
        gaps = []

        subtopics = research_data.get("subtopics", [])
        for i, subtopic in enumerate(subtopics):
            if not subtopic.get("key_facts"):
                gaps.append(f"Subtopic {i + 1} lacks key facts")
            if not subtopic.get("discussion_angles"):
                gaps.append(f"Subtopic {i + 1} lacks discussion angles")
            if not subtopic.get("examples"):
                gaps.append(f"Subtopic {i + 1} lacks concrete examples")

        if len(research_data.get("research_sections", [])) < 2:
            gaps.append("Insufficient additional research sections")

        if len(research_data.get("key_points", [])) < 5:
            gaps.append("Insufficient key points for comprehensive coverage")

        return gaps

    def _enhanced_adjust_content_duration(
        self,
        content_plan: Dict[str, Any],
        target_duration: int,
        research_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enhanced content duration adjustment with research-aware optimization"""

        structure = content_plan.get("content_structure", {})
        research_complexity = self._assess_research_complexity(research_data)

        # Calculate current total
        current_total = (
            structure.get("intro", {}).get("duration_minutes", 0)
            + sum(
                section.get("duration_minutes", 0)
                for section in structure.get("main_content", [])
            )
            + structure.get("outro", {}).get("duration_minutes", 0)
        )

        if current_total == 0:
            return content_plan

        logger.info(
            f"🔧 ENHANCED ADJUSTMENT: From {current_total:.1f}min to {target_duration}min (complexity: {research_complexity})"
        )

        # Adaptive timing allocation based on research complexity
        if research_complexity in ["complex", "very_complex"]:
            # More time for main content in complex topics
            intro_ratio, main_ratio, outro_ratio = 0.08, 0.84, 0.08
        else:
            # Standard allocation for simpler topics
            intro_ratio, main_ratio, outro_ratio = 0.10, 0.80, 0.10

        # Adjust intro with complexity awareness
        ideal_intro = target_duration * intro_ratio
        if "intro" in structure:
            structure["intro"]["duration_minutes"] = round(ideal_intro, 1)

        # Adjust outro
        ideal_outro = target_duration * outro_ratio
        if "outro" in structure:
            structure["outro"]["duration_minutes"] = round(ideal_outro, 1)

        # Smart main content adjustment based on priority and complexity
        remaining_time = target_duration * main_ratio
        main_content = structure.get("main_content", [])

        if main_content:
            # Prioritize sections based on research richness and importance
            for section in main_content:
                priority_rank = section.get("priority_rank", 5)
                # Higher priority (lower rank number) gets more time
                priority_weight = 1.0 + (5 - priority_rank) * 0.1
                section["priority_weight"] = priority_weight

            total_weight = sum(
                section.get("priority_weight", 1.0) for section in main_content
            )

            for section in main_content:
                weight_proportion = section.get("priority_weight", 1.0) / total_weight
                section["duration_minutes"] = round(
                    remaining_time * weight_proportion, 1
                )

        # Recalculate duration allocation after adjustment
        content_plan["planning_metadata"]["duration_allocation"] = (
            self._calculate_duration_allocation(content_plan)
        )

        logger.info(
            f"✅ ENHANCED ADJUSTMENT COMPLETE: Optimized for {research_complexity} complexity"
        )

        return content_plan

    def plan_content_with_duration_awareness(
        self,
        research_data: Dict[str, Any],
        target_duration: int = 10,
        audience_preferences: Optional[Dict] = None,
        quality_threshold: float = 80.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Create content plan with strict duration awareness and quality validation

        Args:
            research_data: Research data from ResearchAgent
            target_duration: Target duration in minutes
            audience_preferences: Optional audience preferences
            quality_threshold: Minimum quality threshold percentage

        Returns:
            Duration-aware content plan that meets quality requirements
        """

        logger.info(f"Creating duration-aware content plan for {target_duration}min")

        # First check if research data has sufficient content
        content_depth = self.content_analyzer.analyze_topic_depth(
            research_data, target_duration
        )

        if (
            content_depth["analysis_summary"]["completeness_percentage"]
            < quality_threshold
        ):
            logger.warning(
                f"Research content may be insufficient for {target_duration}min target"
            )

        # Create initial content plan
        content_plan = self.plan_content(
            research_data, target_duration, audience_preferences
        )

        if not content_plan:
            return None

        # Iterative improvement for duration accuracy
        max_iterations = 3
        iteration = 0

        while iteration < max_iterations:
            # Check duration validation
            duration_validation = content_plan.get("duration_validation", {})

            if duration_validation.get("validation_passed", False):
                logger.info(f"Duration validation passed on iteration {iteration + 1}")
                break

            # Adjust timing if needed
            logger.info(f"Iteration {iteration + 1}: Adjusting content timing")
            content_plan = self._fine_tune_content_timing(content_plan, target_duration)

            iteration += 1

        # Final validation
        final_validation = self.validate_content_plan(content_plan)
        content_plan["final_validation"] = final_validation

        logger.info(
            f"Duration-aware content planning completed. Quality score: {final_validation['quality_score']}"
        )

        return content_plan

    def _fine_tune_content_timing(
        self, content_plan: Dict[str, Any], target_duration: int
    ) -> Dict[str, Any]:
        """Fine-tune content timing for better duration accuracy"""

        # Re-validate and adjust duration
        content_plan = self._validate_and_adjust_duration(
            content_plan,
            {},
            target_duration,  # Empty research_data for timing-only adjustment
        )

        # Optimize section priorities based on duration constraints
        main_content = content_plan.get("content_structure", {}).get("main_content", [])

        # Sort by priority and adjust durations for better balance
        main_content.sort(key=lambda x: x.get("priority_rank", 999))

        # Redistribute time more evenly among high-priority sections
        high_priority_sections = [
            s for s in main_content if s.get("priority_rank", 999) <= 2
        ]
        medium_priority_sections = [
            s for s in main_content if 3 <= s.get("priority_rank", 999) <= 4
        ]

        available_main_time = target_duration * 0.8  # 80% for main content

        if high_priority_sections and medium_priority_sections:
            # Allocate 60% to high priority, 40% to medium priority
            high_priority_time = available_main_time * 0.6
            medium_priority_time = available_main_time * 0.4

            # Redistribute within each group
            for section in high_priority_sections:
                section["duration_minutes"] = round(
                    high_priority_time / len(high_priority_sections), 1
                )

            for section in medium_priority_sections:
                section["duration_minutes"] = round(
                    medium_priority_time / len(medium_priority_sections), 1
                )

        # Recalculate metadata
        content_plan["planning_metadata"]["duration_allocation"] = (
            self._calculate_duration_allocation(content_plan)
        )

        return content_plan

    def validate_content_plan(self, content_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content plan structure and quality"""
        validation = {
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "quality_score": 0,
        }

        if not content_plan:
            validation["is_valid"] = False
            validation["issues"].append("No content plan provided")
            return validation

        # Check required fields
        required_fields = ["main_topic", "target_duration", "content_structure"]
        for field in required_fields:
            if field not in content_plan:
                validation["is_valid"] = False
                validation["issues"].append(f"Missing required field: {field}")

        # Validate timing
        if "planning_metadata" in content_plan:
            timing_validation = content_plan["planning_metadata"].get(
                "timing_validation", {}
            )
            if not timing_validation.get("is_within_tolerance", False):
                validation["issues"].append("Timing allocation exceeds target duration")

        # Calculate quality score
        quality_factors = []
        if "planning_metadata" in content_plan:
            metadata = content_plan["planning_metadata"]

            # Flow quality
            flow_quality = metadata.get("flow_quality", "needs_improvement")
            if flow_quality == "excellent":
                quality_factors.append(25)
            elif flow_quality == "good":
                quality_factors.append(20)
            else:
                quality_factors.append(10)

            # Content density
            density = metadata.get("content_density", "unknown")
            if density == "moderate":
                quality_factors.append(25)
            elif density in ["light", "dense"]:
                quality_factors.append(20)
            else:
                quality_factors.append(10)

            # Timing validation
            if metadata.get("timing_validation", {}).get("is_within_tolerance", False):
                quality_factors.append(25)
            else:
                quality_factors.append(10)

            # Structure completeness
            structure = content_plan.get("content_structure", {})
            if all(key in structure for key in ["intro", "main_content", "outro"]):
                quality_factors.append(25)
            else:
                quality_factors.append(15)

        validation["quality_score"] = sum(quality_factors) if quality_factors else 0

        return validation

    def get_content_plan_summary(self, content_plan: Dict[str, Any]) -> str:
        """Generate human-readable summary of content plan"""
        if not content_plan:
            return "No content plan available"

        main_topic = content_plan.get("main_topic", "Unknown Topic")
        target_duration = content_plan.get("target_duration", "unknown")

        summary = f"Content Plan for '{main_topic}' ({target_duration} minutes)\n\n"

        structure = content_plan.get("content_structure", {})

        # Intro
        intro = structure.get("intro", {})
        summary += f"Intro ({intro.get('duration_minutes', 0)} min): {intro.get('purpose', 'N/A')}\n"

        # Main content
        main_content = structure.get("main_content", [])
        summary += f"\nMain Content ({len(main_content)} sections):\n"
        for i, section in enumerate(main_content, 1):
            duration = section.get("duration_minutes", 0)
            priority = section.get("priority_rank", "?")
            title = section.get("subtopic", "Untitled")
            summary += f"  {i}. {title} ({duration} min, Priority: {priority})\n"

        # Outro
        outro = structure.get("outro", {})
        summary += f"\nOutro ({outro.get('duration_minutes', 0)} min): {outro.get('purpose', 'N/A')}\n"

        # Metadata
        if "planning_metadata" in content_plan:
            metadata = content_plan["planning_metadata"]
            summary += f"\nQuality Metrics:\n"
            summary += (
                f"  Content Density: {metadata.get('content_density', 'unknown')}\n"
            )
            summary += f"  Flow Quality: {metadata.get('flow_quality', 'unknown')}\n"

            timing = metadata.get("timing_validation", {})
            if timing:
                summary += (
                    f"  Total Time: {timing.get('total_allocated_time', 0)} min\n"
                )

        return summary
