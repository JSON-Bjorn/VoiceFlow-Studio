from typing import Dict, List, Optional, Any
from .openai_service import OpenAIService
from .content_depth_analyzer import ContentDepthAnalyzer
from .duration_calculator import DurationCalculator
import logging

logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Specialized agent for podcast topic research using OpenAI

    This agent is responsible for:
    1. Breaking down main topics into engaging subtopics
    2. Gathering key facts and statistics
    3. Identifying discussion angles and questions
    4. Creating a narrative structure for the podcast
    """

    def __init__(self):
        self.openai_service = OpenAIService()
        self.content_analyzer = ContentDepthAnalyzer()
        self.duration_calculator = DurationCalculator()

    def research_topic(
        self, main_topic: str, target_length: int = 10, depth: str = "standard"
    ) -> Optional[Dict[str, Any]]:
        """
        Conduct comprehensive research on a podcast topic

        Args:
            main_topic: The main topic to research
            target_length: Target podcast length in minutes
            depth: Research depth ("light", "standard", "deep")

        Returns:
            Comprehensive research data structure
        """
        logger.info(f"Starting research for topic: {main_topic}")

        # Adjust research depth based on target length and depth parameter
        num_subtopics = self._calculate_subtopics(target_length, depth)

        research_data = self.openai_service.generate_research_topics(
            main_topic=main_topic, target_length=target_length
        )

        if not research_data:
            logger.error(f"Failed to generate research for topic: {main_topic}")
            return None

        # Enhance research with additional context if needed
        enhanced_research = self._enhance_research(research_data, depth, target_length)

        # Validate content depth for target duration
        depth_analysis = self.content_analyzer.analyze_topic_depth(
            enhanced_research, target_length
        )

        # If content is insufficient, expand research
        if depth_analysis["needs_expansion"]:
            logger.info(f"Content expansion needed for {target_length}min target")
            enhanced_research = self._expand_research_for_duration(
                enhanced_research, depth_analysis, target_length
            )

        logger.info(f"Research completed for topic: {main_topic}")
        return enhanced_research

    def _calculate_subtopics(self, target_length: int, depth: str) -> int:
        """Calculate optimal number of subtopics based on length and depth"""
        base_subtopics = {
            "light": max(2, target_length // 4),
            "standard": max(3, target_length // 3),
            "deep": max(4, target_length // 2),
        }
        return min(base_subtopics.get(depth, 3), 6)  # Cap at 6 subtopics

    def _enhance_research(
        self, research_data: Dict[str, Any], depth: str, target_length: int = 10
    ) -> Dict[str, Any]:
        """
        Enhance research data with additional context and structure

        Args:
            research_data: Base research data from OpenAI
            depth: Research depth level

        Returns:
            Enhanced research data
        """
        if not research_data:
            return research_data

        # Calculate content requirements for target duration
        content_requirements = (
            self.duration_calculator.estimate_required_content_for_duration(
                target_length, "conversational"
            )
        )

        # Add metadata
        research_data["research_metadata"] = {
            "depth_level": depth,
            "target_length": target_length,
            "generated_at": None,  # Will be set by the calling service
            "subtopic_count": len(research_data.get("subtopics", [])),
            "estimated_research_quality": self._assess_research_quality(research_data),
            "content_requirements": content_requirements,
        }

        # Enhance subtopics with additional structure
        if "subtopics" in research_data:
            for i, subtopic in enumerate(research_data["subtopics"]):
                subtopic["order"] = i + 1
                subtopic["estimated_duration"] = self._estimate_subtopic_duration(
                    subtopic, research_data.get("estimated_segments", 3)
                )

        return research_data

    def _assess_research_quality(self, research_data: Dict[str, Any]) -> str:
        """Assess the quality of generated research"""
        if not research_data or "subtopics" not in research_data:
            return "poor"

        subtopics = research_data["subtopics"]

        # Check for completeness
        has_summaries = all("summary" in st for st in subtopics)
        has_facts = all(
            "key_facts" in st and len(st["key_facts"]) > 0 for st in subtopics
        )
        has_angles = all("discussion_angles" in st for st in subtopics)

        if has_summaries and has_facts and has_angles:
            return "excellent"
        elif has_summaries and (has_facts or has_angles):
            return "good"
        elif has_summaries:
            return "fair"
        else:
            return "poor"

    def _estimate_subtopic_duration(
        self, subtopic: Dict[str, Any], total_segments: int
    ) -> float:
        """Estimate duration for a subtopic based on content complexity"""
        base_duration = 8 / total_segments  # 8 minutes for main content

        # Adjust based on content richness
        content_score = 0
        content_score += len(subtopic.get("key_facts", [])) * 0.5
        content_score += len(subtopic.get("discussion_angles", [])) * 0.3
        content_score += (
            len(subtopic.get("summary", "").split()) / 20
        )  # Word count factor

        # Apply multiplier (0.8 to 1.4)
        multiplier = min(1.4, max(0.8, 1 + (content_score - 3) * 0.1))

        return round(base_duration * multiplier, 1)

    def validate_research(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate research data and provide feedback

        Args:
            research_data: Research data to validate

        Returns:
            Validation results with suggestions
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "quality_score": 0,
        }

        if not research_data:
            validation_result["is_valid"] = False
            validation_result["issues"].append("No research data provided")
            return validation_result

        # Check required fields
        required_fields = ["main_topic", "subtopics"]
        for field in required_fields:
            if field not in research_data:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Missing required field: {field}")

        # Check subtopics quality
        subtopics = research_data.get("subtopics", [])
        if len(subtopics) < 2:
            validation_result["issues"].append("Too few subtopics (minimum 2 required)")
            validation_result["suggestions"].append(
                "Add more subtopics for better content coverage"
            )

        if len(subtopics) > 6:
            validation_result["issues"].append(
                "Too many subtopics (maximum 6 recommended)"
            )
            validation_result["suggestions"].append(
                "Consider consolidating some subtopics"
            )

        # Calculate quality score
        quality_score = 0
        for subtopic in subtopics:
            if "summary" in subtopic and len(subtopic["summary"]) > 50:
                quality_score += 2
            if "key_facts" in subtopic and len(subtopic["key_facts"]) >= 2:
                quality_score += 2
            if (
                "discussion_angles" in subtopic
                and len(subtopic["discussion_angles"]) >= 1
            ):
                quality_score += 1

        validation_result["quality_score"] = min(
            100, (quality_score / (len(subtopics) * 5)) * 100
        )

        return validation_result

    def _expand_research_for_duration(
        self,
        research_data: Dict[str, Any],
        depth_analysis: Dict[str, Any],
        target_duration: float,
    ) -> Dict[str, Any]:
        """
        Expand research data to meet target duration requirements

        Args:
            research_data: Current research data
            depth_analysis: Analysis from ContentDepthAnalyzer
            target_duration: Target duration in minutes

        Returns:
            Expanded research data
        """

        logger.info(f"Expanding research for {target_duration}min target duration")

        # Get expansion recommendations with safety checks
        expansion_recommendations = depth_analysis.get("expansion_recommendations", [])
        word_deficit = depth_analysis.get("analysis_summary", {}).get("word_deficit", 0)

        # Expand existing subtopics
        if "subtopics" in research_data:
            self._expand_existing_subtopics(
                research_data["subtopics"], expansion_recommendations
            )

        # Add new research sections if major expansion is needed
        if word_deficit > 300:
            self._add_additional_research_sections(
                research_data, expansion_recommendations, target_duration
            )

        # Add enhanced key points
        if "key_points" not in research_data:
            research_data["key_points"] = []

        self._expand_key_points(research_data, expansion_recommendations)

        # Update metadata to reflect expansion
        if "research_metadata" not in research_data:
            research_data["research_metadata"] = {}
        research_data["research_metadata"]["expanded_for_duration"] = True
        research_data["research_metadata"]["expansion_applied"] = [
            rec.get("description", "expansion applied")
            for rec in expansion_recommendations[:3]
        ]

        # Re-analyze content depth after expansion
        final_analysis = self.content_analyzer.analyze_topic_depth(
            research_data, target_duration
        )
        research_data["final_content_analysis"] = final_analysis

        logger.info(
            f"Research expansion completed. New completeness: {final_analysis['analysis_summary']['completeness_percentage']}%"
        )

        return research_data

    def _expand_existing_subtopics(
        self, subtopics: List[Dict[str, Any]], recommendations: List[Dict[str, Any]]
    ) -> None:
        """Expand existing subtopics with more detailed content"""

        for subtopic in subtopics:
            # Add more key facts
            if "key_facts" not in subtopic:
                subtopic["key_facts"] = []

            # Expand based on recommendations
            for rec in recommendations:
                rec_type = rec.get("type", "")
                rec_description = rec.get("description", "")

                if rec_type == "content_type_addition":
                    if "examples" in rec_description:
                        if "examples" not in subtopic:
                            subtopic["examples"] = []
                        subtopic["examples"].extend(
                            [
                                f"Example related to {subtopic.get('title', 'this topic')}",
                                f"Case study demonstrating {subtopic.get('title', 'this concept')}",
                            ]
                        )

                    if "analysis" in rec_description:
                        if "deeper_analysis" not in subtopic:
                            subtopic["deeper_analysis"] = []
                        subtopic["deeper_analysis"].extend(
                            [
                                f"Technical analysis of {subtopic.get('title', 'this topic')}",
                                f"Implications and consequences of {subtopic.get('title', 'this development')}",
                            ]
                        )

                elif rec_type == "quality_improvement":
                    if "statistics" in rec_description:
                        if "statistics" not in subtopic:
                            subtopic["statistics"] = []
                        subtopic["statistics"].extend(
                            [
                                f"Key statistics related to {subtopic.get('title', 'this topic')}",
                                f"Market data and trends for {subtopic.get('title', 'this area')}",
                            ]
                        )

    def _add_additional_research_sections(
        self,
        research_data: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        target_duration: float,
    ) -> None:
        """Add additional research sections for major content expansion"""

        if "research_sections" not in research_data:
            research_data["research_sections"] = []

        # Add background context section
        research_data["research_sections"].append(
            {
                "title": "Background and Context",
                "content": f"Historical background and context for {research_data.get('main_topic', 'the topic')}",
                "summary": "Provides foundational understanding and historical perspective",
                "content_type": "background",
            }
        )

        # Add expert perspectives section
        research_data["research_sections"].append(
            {
                "title": "Expert Insights and Analysis",
                "content": f"Expert opinions and professional analysis of {research_data.get('main_topic', 'the topic')}",
                "summary": "Professional insights and expert commentary",
                "content_type": "expert_analysis",
            }
        )

        # Add implications section for longer content
        if target_duration > 8:
            research_data["research_sections"].append(
                {
                    "title": "Future Implications and Trends",
                    "content": f"Future implications and emerging trends related to {research_data.get('main_topic', 'the topic')}",
                    "summary": "Forward-looking analysis and trend predictions",
                    "content_type": "future_analysis",
                }
            )

    def _expand_key_points(
        self, research_data: Dict[str, Any], recommendations: List[Dict[str, Any]]
    ) -> None:
        """Expand key points with additional content"""

        main_topic = research_data.get("main_topic", "the topic")

        # Add content-type specific key points
        for rec in recommendations:
            rec_priority = rec.get("priority", "low")
            rec_description = rec.get("description", "")

            if rec_priority in ["high", "medium"]:
                if "examples" in rec_description:
                    research_data["key_points"].extend(
                        [
                            f"Real-world example: How {main_topic} impacts everyday life",
                            f"Case study: Successful implementation of {main_topic}",
                            f"Practical demonstration: {main_topic} in action",
                        ]
                    )

                if "analysis" in rec_description:
                    research_data["key_points"].extend(
                        [
                            f"Technical analysis: The mechanics behind {main_topic}",
                            f"Critical evaluation: Benefits and drawbacks of {main_topic}",
                            f"Comparative analysis: {main_topic} vs alternatives",
                        ]
                    )

                if "statistics" in rec_description:
                    research_data["key_points"].extend(
                        [
                            f"Key statistic: Market size and growth for {main_topic}",
                            f"Performance metrics: Success rates in {main_topic}",
                            f"Trend data: {main_topic} adoption over time",
                        ]
                    )

    def research_with_duration_target(
        self,
        main_topic: str,
        target_duration: float,
        depth: str = "moderate_depth",
        quality_threshold: float = 80.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Research topic with specific duration target and quality requirements

        Args:
            main_topic: Topic to research
            target_duration: Target duration in minutes
            depth: Research depth level
            quality_threshold: Minimum content completeness percentage

        Returns:
            Research data that meets duration and quality requirements
        """

        logger.info(
            f"Starting duration-targeted research: {main_topic} for {target_duration}min"
        )

        # Initial research
        research_data = self.research_topic(main_topic, int(target_duration), depth)

        if not research_data:
            return None

        # Iterative improvement to meet quality threshold
        max_iterations = 3
        iteration = 0

        while iteration < max_iterations:
            # Analyze content depth
            depth_analysis = self.content_analyzer.analyze_topic_depth(
                research_data, target_duration, depth
            )

            completeness = depth_analysis["analysis_summary"]["completeness_percentage"]

            if completeness >= quality_threshold:
                logger.info(
                    f"Quality threshold met: {completeness}% >= {quality_threshold}%"
                )
                break

            # Expand research for better quality
            logger.info(
                f"Iteration {iteration + 1}: Expanding research (current: {completeness}%)"
            )
            research_data = self._expand_research_for_duration(
                research_data, depth_analysis, target_duration
            )

            iteration += 1

        # Final validation
        final_analysis = self.content_analyzer.analyze_topic_depth(
            research_data, target_duration, depth
        )
        research_data["duration_readiness"] = {
            "target_duration": target_duration,
            "final_completeness": final_analysis["analysis_summary"][
                "completeness_percentage"
            ],
            "meets_threshold": final_analysis["analysis_summary"][
                "completeness_percentage"
            ]
            >= quality_threshold,
            "iterations_used": iteration,
            "ready_for_script_generation": final_analysis["ready_for_generation"],
        }

        logger.info(
            f"Duration-targeted research completed. Final quality: {final_analysis['analysis_summary']['completeness_percentage']}%"
        )

        return research_data

    def get_research_summary(self, research_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the research"""
        if not research_data:
            return "No research data available"

        main_topic = research_data.get("main_topic", "Unknown Topic")
        subtopics = research_data.get("subtopics", [])

        summary = f"Research Summary for '{main_topic}':\n\n"
        summary += f"Total subtopics: {len(subtopics)}\n"

        if "research_metadata" in research_data:
            metadata = research_data["research_metadata"]
            summary += f"Research depth: {metadata.get('depth_level', 'unknown')}\n"
            summary += f"Quality assessment: {metadata.get('estimated_research_quality', 'unknown')}\n\n"

        summary += "Subtopics:\n"
        for i, subtopic in enumerate(subtopics, 1):
            summary += f"{i}. {subtopic.get('title', 'Untitled')}\n"
            if "summary" in subtopic:
                summary += f"   {subtopic['summary'][:100]}...\n"

        return summary
