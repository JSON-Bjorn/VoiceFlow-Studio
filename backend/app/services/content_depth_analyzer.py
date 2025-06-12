from typing import Dict, List, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class ContentDepthAnalyzer:
    """Analyzes if topic has sufficient depth for target duration"""

    def __init__(self):
        # Content density factors (words needed per minute for different content types)
        self.content_density_factors = {
            "surface_level": 120,  # Basic discussion, simple topics
            "moderate_depth": 140,  # Standard podcast depth
            "deep_analysis": 160,  # In-depth technical or analytical content
            "comprehensive": 180,  # Very detailed, comprehensive coverage
            "expert_level": 200,  # Expert-level deep dives
        }

        # Research quality indicators
        self.quality_indicators = {
            "high_quality": [
                "research",
                "study",
                "data",
                "evidence",
                "analysis",
                "expert",
                "scientific",
            ],
            "examples": [
                "example",
                "case",
                "instance",
                "story",
                "experience",
                "demonstration",
            ],
            "statistics": [
                "percent",
                "%",
                "number",
                "rate",
                "increase",
                "decrease",
                "compared",
            ],
            "quotes": [
                "said",
                "according",
                "stated",
                "mentioned",
                "explained",
                "noted",
            ],
            "technical_terms": [
                "process",
                "method",
                "system",
                "algorithm",
                "technique",
                "approach",
            ],
        }

        # Content expansion opportunities
        self.expansion_areas = {
            "examples": "Add real-world examples and case studies",
            "background": "Provide more background context and history",
            "technical_details": "Include technical explanations and processes",
            "comparisons": "Add comparisons with alternatives or competitors",
            "implications": "Discuss implications and future impact",
            "expert_opinions": "Include expert perspectives and quotes",
            "statistics": "Add relevant statistics and data points",
            "personal_stories": "Include personal anecdotes and experiences",
            "step_by_step": "Break down complex processes step-by-step",
            "different_perspectives": "Present multiple viewpoints on the topic",
        }

    def analyze_topic_depth(
        self,
        research_data: Dict[str, Any],
        target_duration: float,
        desired_depth: str = "moderate_depth",
    ) -> Dict[str, Any]:
        """
        Analyze if research data has sufficient content depth for target duration

        Args:
            research_data: Research data from ResearchAgent
            target_duration: Target duration in minutes
            desired_depth: Desired content depth level

        Returns:
            Analysis results with recommendations
        """

        logger.info(f"Analyzing content depth for {target_duration}min target duration")

        # Calculate required content density
        required_words = target_duration * self.content_density_factors.get(
            desired_depth, self.content_density_factors["moderate_depth"]
        )

        # Analyze current research content
        current_analysis = self._analyze_current_content(research_data)

        # Calculate content gaps
        content_gaps = self._identify_content_gaps(current_analysis, required_words)

        # Generate expansion recommendations
        expansion_recommendations = self._generate_expansion_recommendations(
            research_data, content_gaps, target_duration
        )

        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(
            current_analysis, required_words
        )

        return {
            "analysis_summary": {
                "target_duration": target_duration,
                "required_words": required_words,
                "current_words": current_analysis.get("total_words", 0),
                "word_deficit": max(
                    0, required_words - current_analysis.get("total_words", 0)
                ),
                "completeness_percentage": completeness_score,
                "depth_level": desired_depth,
            },
            "content_analysis": current_analysis,
            "content_gaps": content_gaps,
            "expansion_recommendations": expansion_recommendations,
            "needs_expansion": completeness_score < 80,
            "ready_for_generation": completeness_score >= 80,
            "quality_assessment": self._assess_content_quality(research_data),
        }

    def _analyze_current_content(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the current research content comprehensively"""

        analysis = {
            "total_words": 0,
            "content_sections": 0,
            "quality_indicators": {
                "high_quality": 0,
                "examples": 0,
                "statistics": 0,
                "quotes": 0,
                "technical_terms": 0,
            },
            "topic_coverage": [],
            "content_types": {
                "factual": 0,
                "narrative": 0,
                "analytical": 0,
                "technical": 0,
            },
            "depth_indicators": {
                "surface_mentions": 0,
                "detailed_explanations": 0,
                "expert_level_content": 0,
            },
        }

        # Extract and analyze all text content
        all_text = self._extract_all_text_content(research_data)
        analysis["total_words"] = len(all_text.split()) if all_text else 0

        # Count content sections
        analysis["content_sections"] = len(research_data.get("research_sections", []))

        # Analyze quality indicators
        for indicator_type, keywords in self.quality_indicators.items():
            if all_text:  # Only analyze if text exists
                count = sum(
                    1 for keyword in keywords if keyword.lower() in all_text.lower()
                )
                analysis["quality_indicators"][indicator_type] = count
            else:
                analysis["quality_indicators"][indicator_type] = 0

        # Analyze content types and depth
        analysis["content_types"] = self._classify_content_types(all_text)
        analysis["depth_indicators"] = self._analyze_content_depth(all_text)
        analysis["topic_coverage"] = self._analyze_topic_coverage(research_data)

        return analysis

    def _extract_all_text_content(self, research_data: Dict[str, Any]) -> str:
        """Extract all text content from research data"""

        text_parts = []

        # Main content
        if "main_content" in research_data:
            text_parts.append(str(research_data["main_content"]))

        # Research sections
        for section in research_data.get("research_sections", []):
            if isinstance(section, dict):
                text_parts.append(section.get("content", ""))
                text_parts.append(section.get("summary", ""))
            else:
                text_parts.append(str(section))

        # Key points
        for point in research_data.get("key_points", []):
            text_parts.append(str(point))

        # Additional data
        if "additional_context" in research_data:
            text_parts.append(str(research_data["additional_context"]))

        return " ".join(text_parts)

    def _classify_content_types(self, text: str) -> Dict[str, int]:
        """Classify content into different types"""

        content_types = {
            "factual": 0,
            "narrative": 0,
            "analytical": 0,
            "technical": 0,
            "examples": 0,  # Add examples key
        }

        text_lower = text.lower()

        # Factual content indicators
        factual_patterns = [
            "fact",
            "data",
            "research shows",
            "study found",
            "according to",
        ]
        content_types["factual"] = sum(
            1 for pattern in factual_patterns if pattern in text_lower
        )

        # Narrative content indicators
        narrative_patterns = ["story", "example", "experience", "happened", "once"]
        content_types["narrative"] = sum(
            1 for pattern in narrative_patterns if pattern in text_lower
        )

        # Examples content indicators
        example_patterns = [
            "example",
            "for instance",
            "case study",
            "demonstration",
            "illustration",
        ]
        content_types["examples"] = sum(
            1 for pattern in example_patterns if pattern in text_lower
        )

        # Analytical content indicators
        analytical_patterns = [
            "analysis",
            "conclusion",
            "therefore",
            "because",
            "result",
        ]
        content_types["analytical"] = sum(
            1 for pattern in analytical_patterns if pattern in text_lower
        )

        # Technical content indicators
        technical_patterns = [
            "algorithm",
            "process",
            "system",
            "method",
            "technical",
            "implementation",
        ]
        content_types["technical"] = sum(
            1 for pattern in technical_patterns if pattern in text_lower
        )

        return content_types

    def _analyze_content_depth(self, text: str) -> Dict[str, int]:
        """Analyze the depth level of content"""

        depth_indicators = {
            "surface_mentions": 0,
            "detailed_explanations": 0,
            "expert_level_content": 0,
        }

        # Count sentences for depth analysis
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) < 20:
                continue

            # Surface level - short, simple statements
            if len(sentence_lower) < 50:
                depth_indicators["surface_mentions"] += 1
            # Detailed explanations - medium length with explanatory words
            elif any(
                word in sentence_lower
                for word in [
                    "because",
                    "however",
                    "therefore",
                    "which means",
                    "in other words",
                ]
            ):
                depth_indicators["detailed_explanations"] += 1
            # Expert level - complex sentences with technical terms
            elif any(
                word in sentence_lower
                for word in [
                    "algorithm",
                    "methodology",
                    "implementation",
                    "architecture",
                    "framework",
                ]
            ):
                depth_indicators["expert_level_content"] += 1
            else:
                depth_indicators["detailed_explanations"] += 1

        return depth_indicators

    def _analyze_topic_coverage(self, research_data: Dict[str, Any]) -> List[str]:
        """Analyze what topics are covered in the research"""

        topics_covered = []

        # Extract main topic
        main_topic = research_data.get("main_topic", "")
        if main_topic:
            topics_covered.append(main_topic)

        # Extract subtopics from sections
        for section in research_data.get("research_sections", []):
            if isinstance(section, dict):
                section_title = section.get("title", "")
                if section_title and section_title not in topics_covered:
                    topics_covered.append(section_title)

        # Extract topics from key points
        for point in research_data.get("key_points", []):
            # Simple topic extraction from key points
            point_str = str(point)
            if len(point_str) > 10 and point_str not in topics_covered:
                topics_covered.append(
                    point_str[:50] + "..." if len(point_str) > 50 else point_str
                )

        return topics_covered[:10]  # Limit to top 10 topics

    def _identify_content_gaps(
        self, current_analysis: Dict[str, Any], required_words: int
    ) -> Dict[str, Any]:
        """Identify specific content gaps that need to be filled"""

        gaps = {
            "word_deficit": max(
                0, required_words - current_analysis.get("total_words", 0)
            ),
            "missing_content_types": [],
            "weak_areas": [],
            "expansion_opportunities": [],
        }

        # Identify missing content types
        content_types = current_analysis.get("content_types", {})
        if content_types.get("examples", 0) < 2:
            gaps["missing_content_types"].append("examples")
        if content_types.get("narrative", 0) < 1:
            gaps["missing_content_types"].append("storytelling")
        if content_types.get("analytical", 0) < 2:
            gaps["missing_content_types"].append("analysis")

        # Identify weak areas based on quality indicators
        quality_indicators = current_analysis.get("quality_indicators", {})
        if quality_indicators.get("examples", 0) < 3:
            gaps["weak_areas"].append("real_world_examples")
        if quality_indicators.get("statistics", 0) < 2:
            gaps["weak_areas"].append("data_and_statistics")
        if quality_indicators.get("quotes", 0) < 1:
            gaps["weak_areas"].append("expert_quotes")

        # Identify expansion opportunities
        depth_indicators = current_analysis.get("depth_indicators", {})
        if depth_indicators.get("detailed_explanations", 0) < 5:
            gaps["expansion_opportunities"].append("detailed_explanations")
        if depth_indicators.get("expert_level_content", 0) < 2:
            gaps["expansion_opportunities"].append("expert_insights")

        return gaps

    def _generate_expansion_recommendations(
        self,
        research_data: Dict[str, Any],
        content_gaps: Dict[str, Any],
        target_duration: float,
    ) -> List[Dict[str, Any]]:
        """Generate specific recommendations for content expansion"""

        recommendations = []

        # Calculate priority based on word deficit
        word_deficit = content_gaps.get("word_deficit", 0)

        if word_deficit > 0:
            # High priority recommendations for significant gaps
            if word_deficit > 500:
                recommendations.append(
                    {
                        "priority": "high",
                        "type": "major_expansion",
                        "description": "Significant content expansion needed",
                        "suggested_words": word_deficit // 2,
                        "areas": [
                            "background_context",
                            "detailed_examples",
                            "expert_analysis",
                        ],
                    }
                )

            # Specific recommendations based on gaps
            for content_type in content_gaps.get("missing_content_types", []):
                recommendations.append(
                    {
                        "priority": "medium",
                        "type": "content_type_addition",
                        "description": f"Add {content_type} content",
                        "suggested_words": 150,
                        "specific_suggestion": self.expansion_areas.get(
                            content_type, f"Add {content_type}"
                        ),
                    }
                )

            for weak_area in content_gaps.get("weak_areas", []):
                recommendations.append(
                    {
                        "priority": "medium",
                        "type": "quality_improvement",
                        "description": f"Strengthen {weak_area}",
                        "suggested_words": 100,
                        "specific_suggestion": self.expansion_areas.get(
                            weak_area, f"Improve {weak_area}"
                        ),
                    }
                )

        # Always suggest some enhancements for better content
        recommendations.append(
            {
                "priority": "low",
                "type": "enhancement",
                "description": "Add engaging elements",
                "suggested_words": 50,
                "specific_suggestion": "Add personal anecdotes, analogies, or interesting side facts",
            }
        )

        return recommendations

    def _calculate_completeness_score(
        self, current_analysis: Dict[str, Any], required_words: int
    ) -> float:
        """Calculate a completeness score for the content"""

        # Word count score (40% of total)
        word_score = (
            min(100, (current_analysis.get("total_words", 0) / required_words) * 100)
            if required_words > 0
            else 0
        )

        # Quality score (30% of total)
        quality_indicators = current_analysis.get("quality_indicators", {})
        quality_score = min(
            100,
            (
                quality_indicators.get("high_quality", 0) * 10
                + quality_indicators.get("examples", 0) * 15
                + quality_indicators.get("statistics", 0) * 10
                + quality_indicators.get("quotes", 0) * 10
                + quality_indicators.get("technical_terms", 0) * 5
            ),
        )

        # Depth score (30% of total)
        depth_indicators = current_analysis.get("depth_indicators", {})
        total_depth_content = sum(depth_indicators.values()) if depth_indicators else 0
        depth_score = (
            min(
                100,
                (
                    depth_indicators.get("detailed_explanations", 0) * 10
                    + depth_indicators.get("expert_level_content", 0) * 15
                ),
            )
            if total_depth_content > 0
            else 0
        )

        # Weighted average
        completeness_score = word_score * 0.4 + quality_score * 0.3 + depth_score * 0.3

        return round(completeness_score, 1)

    def _assess_content_quality(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall content quality"""

        quality_assessment = {
            "overall_rating": "unknown",
            "strengths": [],
            "weaknesses": [],
            "readiness_for_podcast": False,
        }

        # Extract text for analysis
        all_text = self._extract_all_text_content(research_data)

        # Assess based on length
        word_count = len(all_text.split())
        if word_count > 1000:
            quality_assessment["strengths"].append("Substantial content volume")
        elif word_count < 300:
            quality_assessment["weaknesses"].append("Limited content volume")

        # Assess based on structure
        sections_count = len(research_data.get("research_sections", []))
        if sections_count >= 3:
            quality_assessment["strengths"].append("Well-structured content")
        elif sections_count < 2:
            quality_assessment["weaknesses"].append("Needs better structure")

        # Assess based on key points
        key_points_count = len(research_data.get("key_points", []))
        if key_points_count >= 5:
            quality_assessment["strengths"].append("Rich in key insights")
        elif key_points_count < 3:
            quality_assessment["weaknesses"].append("Needs more key points")

        # Overall rating
        strength_count = len(quality_assessment["strengths"])
        weakness_count = len(quality_assessment["weaknesses"])

        if strength_count > weakness_count:
            quality_assessment["overall_rating"] = "good"
            quality_assessment["readiness_for_podcast"] = True
        elif strength_count == weakness_count:
            quality_assessment["overall_rating"] = "fair"
            quality_assessment["readiness_for_podcast"] = word_count > 500
        else:
            quality_assessment["overall_rating"] = "needs_improvement"
            quality_assessment["readiness_for_podcast"] = False

        return quality_assessment

    def recommend_research_expansion(
        self,
        research_data: Dict[str, Any],
        target_duration: float,
        focus_areas: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Recommend specific research expansion to meet duration target

        Args:
            research_data: Current research data
            target_duration: Target duration in minutes
            focus_areas: Specific areas to focus expansion on

        Returns:
            Detailed expansion recommendations
        """

        analysis = self.analyze_topic_depth(research_data, target_duration)

        expansion_plan = {
            "current_status": analysis["analysis_summary"],
            "expansion_needed": analysis["needs_expansion"],
            "recommended_additions": [],
            "priority_areas": [],
            "estimated_additional_research_time": 0,
        }

        if analysis["needs_expansion"]:
            word_deficit = analysis["analysis_summary"]["word_deficit"]

            # Calculate additional research time (rough estimate)
            expansion_plan["estimated_additional_research_time"] = max(
                5, word_deficit // 50
            )

            # Prioritize based on focus areas if provided
            if focus_areas:
                for area in focus_areas:
                    expansion_plan["priority_areas"].append(
                        {
                            "area": area,
                            "suggested_approach": self.expansion_areas.get(
                                area, f"Research {area} in depth"
                            ),
                            "estimated_words": 200,
                        }
                    )

            # Add general recommendations
            for recommendation in analysis["expansion_recommendations"]:
                if recommendation["priority"] in ["high", "medium"]:
                    expansion_plan["recommended_additions"].append(recommendation)

        return expansion_plan
