from typing import Dict, List, Optional, Any
from .openai_service import OpenAIService
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
        enhanced_research = self._enhance_research(research_data, depth)

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
        self, research_data: Dict[str, Any], depth: str
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

        # Add metadata
        research_data["research_metadata"] = {
            "depth_level": depth,
            "generated_at": None,  # Will be set by the calling service
            "subtopic_count": len(research_data.get("subtopics", [])),
            "estimated_research_quality": self._assess_research_quality(research_data),
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
