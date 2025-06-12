from typing import Dict, List, Optional, Any
from .openai_service import OpenAIService
from .duration_calculator import DurationCalculator
from .content_depth_analyzer import ContentDepthAnalyzer
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
        Validate duration allocation and suggest adjustments if needed

        Args:
            content_plan: Current content plan
            research_data: Research data for context
            target_duration: Target duration in minutes

        Returns:
            Adjusted content plan with duration validation
        """

        logger.info(f"Validating duration allocation for {target_duration}min target")

        # Get current duration allocation
        duration_allocation = content_plan["planning_metadata"]["duration_allocation"]
        total_allocated = duration_allocation["actual_allocation"]["total"]

        # Check if adjustment is needed
        tolerance = 0.5  # 30 seconds tolerance
        needs_adjustment = abs(total_allocated - target_duration) > tolerance

        if needs_adjustment:
            logger.info(
                f"Duration adjustment needed: {total_allocated}min allocated vs {target_duration}min target"
            )
            content_plan = self._adjust_content_duration(content_plan, target_duration)

        # Add duration validation metadata
        content_plan["duration_validation"] = {
            "target_duration": target_duration,
            "allocated_duration": total_allocated,
            "within_tolerance": not needs_adjustment,
            "adjustment_applied": needs_adjustment,
            "tolerance_minutes": tolerance,
            "validation_passed": abs(
                content_plan["planning_metadata"]["duration_allocation"][
                    "actual_allocation"
                ]["total"]
                - target_duration
            )
            <= tolerance,
        }

        # Estimate content requirements for validation
        content_requirements = (
            self.duration_calculator.estimate_required_content_for_duration(
                target_duration, "conversational"
            )
        )

        # Validate research content depth
        depth_validation = self.duration_calculator.validate_duration_accuracy(
            research_data.get("research_sections", []),
            target_duration,
            tolerance_percentage=15,
        )

        content_plan["content_readiness"] = {
            "content_requirements": content_requirements,
            "depth_validation": depth_validation,
            "ready_for_script_generation": depth_validation.get(
                "validation_passed", False
            ),
            "recommended_actions": depth_validation.get("recommendations", []),
        }

        return content_plan

    def _adjust_content_duration(
        self, content_plan: Dict[str, Any], target_duration: int
    ) -> Dict[str, Any]:
        """Adjust content durations to meet target"""

        structure = content_plan.get("content_structure", {})

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

        # Calculate adjustment factor
        adjustment_factor = target_duration / current_total

        # Adjust intro (keep to ideal 10%)
        ideal_intro = target_duration * 0.1
        if "intro" in structure:
            structure["intro"]["duration_minutes"] = round(ideal_intro, 1)

        # Adjust outro (keep to ideal 10%)
        ideal_outro = target_duration * 0.1
        if "outro" in structure:
            structure["outro"]["duration_minutes"] = round(ideal_outro, 1)

        # Adjust main content proportionally (remaining 80%)
        remaining_time = target_duration * 0.8
        main_content = structure.get("main_content", [])

        if main_content:
            current_main_total = sum(
                section.get("duration_minutes", 0) for section in main_content
            )

            for section in main_content:
                if current_main_total > 0:
                    section_proportion = (
                        section.get("duration_minutes", 0) / current_main_total
                    )
                    section["duration_minutes"] = round(
                        remaining_time * section_proportion, 1
                    )

        # Recalculate duration allocation after adjustment
        content_plan["planning_metadata"]["duration_allocation"] = (
            self._calculate_duration_allocation(content_plan)
        )

        logger.info(
            f"Duration adjusted from {current_total}min to {target_duration}min"
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
