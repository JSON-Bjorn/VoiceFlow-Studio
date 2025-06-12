from typing import Dict, List, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class DurationCalculator:
    """Advanced duration calculation with speech pattern analysis"""

    def __init__(self):
        # Realistic conversation speech rates (words per minute)
        # Based on actual podcast conversation analysis
        self.speech_rates = {
            "conversational": 140,  # WPM for natural conversation
            "analytical": 120,  # WPM for complex explanations
            "storytelling": 160,  # WPM for narrative sections
            "Q&A": 130,  # WPM for question-answer format
            "technical": 110,  # WPM for technical content
            "casual": 150,  # WPM for light conversation
            "excited": 170,  # WPM for enthusiastic discussion
            "thoughtful": 100,  # WPM for deep contemplation
        }

        # Pause factors for natural speech patterns (in seconds)
        self.pause_factors = {
            "topic_transition": 2.0,  # seconds between major topics
            "speaker_change": 1.0,  # seconds for speaker transitions
            "emphasis": 0.5,  # seconds for emphasis pauses
            "natural_pause": 0.3,  # seconds per sentence
            "question_pause": 1.5,  # seconds after questions
            "thoughtful_pause": 1.0,  # seconds for contemplative moments
        }

        # Content type detection patterns
        self.content_patterns = {
            "question": r"[?]|what|how|why|when|where|who|could you|can you",
            "technical": r"algorithm|process|method|system|technical|data|code|API",
            "storytelling": r"story|example|remember|once|happened|experience",
            "analytical": r"analysis|research|study|evidence|conclusion|therefore",
            "excited": r"amazing|incredible|wow|fantastic|awesome|love|excited",
            "thoughtful": r"think|consider|ponder|reflect|contemplate|interesting",
        }

    def estimate_dialogue_duration(
        self,
        dialogue: List[Dict[str, Any]],
        target_duration: Optional[float] = None,
        conversation_style: str = "conversational",
    ) -> Dict[str, Any]:
        """
        Estimate duration for dialogue with natural conversation timing

        Args:
            dialogue: List of dialogue items with text and speaker
            target_duration: Target duration in minutes (for adjustment calculations)
            conversation_style: Style of conversation for rate adjustment

        Returns:
            Dict with duration estimates and breakdown
        """

        if not dialogue:
            return {"estimated_duration": 0, "word_count": 0, "breakdown": {}}

        total_words = 0
        total_speech_time = 0
        total_pause_time = 0
        content_analysis = {
            "conversational": 0,
            "analytical": 0,
            "storytelling": 0,
            "Q&A": 0,
            "technical": 0,
            "excited": 0,
            "thoughtful": 0,
            "question": 0,  # Add missing question key
        }

        segment_breakdown = []

        for i, item in enumerate(dialogue):
            text = item.get("text", "")
            speaker = item.get("speaker", "unknown")

            # Count words
            words = len(text.split())
            total_words += words

            # Analyze content type for this dialogue item
            content_type = self._analyze_content_type(text)
            content_analysis[content_type] += words

            # Calculate speech time based on content type
            wpm = self.speech_rates.get(
                content_type, self.speech_rates["conversational"]
            )
            speech_time = words / wpm if words > 0 else 0

            # Add natural pauses
            pause_time = self._calculate_pause_time(text, i, dialogue)

            total_speech_time += speech_time
            total_pause_time += pause_time

            segment_breakdown.append(
                {
                    "speaker": speaker,
                    "words": words,
                    "content_type": content_type,
                    "speech_time": speech_time,
                    "pause_time": pause_time,
                    "wpm_used": wpm,
                }
            )

        # Calculate total duration in minutes
        total_duration_minutes = total_speech_time + total_pause_time

        # Conversation flow adjustment (conversations are naturally slower)
        flow_adjustment = self._calculate_conversation_flow_adjustment(
            len(dialogue), content_analysis, conversation_style
        )

        adjusted_duration = total_duration_minutes * flow_adjustment

        # Calculate accuracy vs target if provided
        accuracy_info = {}
        if target_duration:
            accuracy_info = self._calculate_accuracy_metrics(
                adjusted_duration, target_duration
            )

        return {
            "estimated_duration": round(adjusted_duration, 2),
            "raw_duration": round(total_duration_minutes, 2),
            "word_count": total_words,
            "speech_time_minutes": round(total_speech_time, 2),
            "pause_time_minutes": round(total_pause_time, 2),
            "flow_adjustment_factor": flow_adjustment,
            "content_breakdown": content_analysis,
            "accuracy_info": accuracy_info,
            "segment_breakdown": segment_breakdown,
        }

    def _analyze_content_type(self, text: str) -> str:
        """Analyze text to determine content type for appropriate speech rate"""
        text_lower = text.lower()

        # Check for patterns in order of specificity
        for content_type, pattern in self.content_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return content_type

        # Default to conversational
        return "conversational"

    def _calculate_pause_time(
        self, text: str, position: int, full_dialogue: List[Dict]
    ) -> float:
        """Calculate natural pause time for a dialogue item"""
        pause_time = 0

        # Natural pauses based on sentence count
        sentence_count = len(re.findall(r"[.!?]+", text))
        pause_time += sentence_count * self.pause_factors["natural_pause"]

        # Speaker change pause (if not first item)
        if position > 0:
            pause_time += self.pause_factors["speaker_change"]

        # Question pause
        if "?" in text:
            pause_time += self.pause_factors["question_pause"]

        # Emphasis pause for exclamations
        if "!" in text:
            pause_time += self.pause_factors["emphasis"]

        # Topic transition detection (simple heuristic)
        if position > 0 and position < len(full_dialogue) - 1:
            current_text = text.lower()
            prev_text = full_dialogue[position - 1].get("text", "").lower()

            # Simple topic transition detection
            transition_keywords = [
                "now",
                "moving on",
                "speaking of",
                "another",
                "also",
                "next",
            ]
            if any(keyword in current_text for keyword in transition_keywords):
                pause_time += self.pause_factors["topic_transition"]

        return pause_time / 60  # Convert to minutes

    def _calculate_conversation_flow_adjustment(
        self,
        dialogue_count: int,
        content_analysis: Dict[str, int],
        conversation_style: str,
    ) -> float:
        """Calculate adjustment factor for natural conversation flow"""

        base_adjustment = 1.2  # Base 20% increase for natural conversation

        # Adjust based on dialogue complexity
        if dialogue_count > 50:
            base_adjustment += 0.1  # Longer conversations are naturally slower

        # Adjust based on content type distribution
        total_words = sum(content_analysis.values())
        if total_words > 0:
            technical_ratio = content_analysis.get("technical", 0) / total_words
            analytical_ratio = content_analysis.get("analytical", 0) / total_words

            # More technical/analytical content = slower delivery
            base_adjustment += (technical_ratio + analytical_ratio) * 0.3

        # Conversation style adjustments
        style_adjustments = {
            "casual": 1.0,
            "professional": 1.1,
            "educational": 1.2,
            "interview": 1.15,
            "debate": 0.95,  # Debates tend to be faster
        }

        style_factor = style_adjustments.get(conversation_style, 1.0)

        return base_adjustment * style_factor

    def _calculate_accuracy_metrics(
        self, estimated_duration: float, target_duration: float
    ) -> Dict[str, Any]:
        """Calculate accuracy metrics comparing estimated vs target duration"""

        difference = estimated_duration - target_duration
        percentage_diff = (
            (difference / target_duration) * 100 if target_duration > 0 else 0
        )
        accuracy_percentage = max(0, 100 - abs(percentage_diff))

        # Determine quality rating
        if accuracy_percentage >= 95:
            quality = "excellent"
        elif accuracy_percentage >= 90:
            quality = "very_good"
        elif accuracy_percentage >= 80:
            quality = "good"
        elif accuracy_percentage >= 70:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "target_duration": target_duration,
            "estimated_duration": estimated_duration,
            "difference_minutes": round(difference, 2),
            "percentage_difference": round(percentage_diff, 1),
            "accuracy_percentage": round(accuracy_percentage, 1),
            "quality_rating": quality,
            "meets_target": abs(percentage_diff)
            <= 10,  # Within 10% is considered meeting target
        }

    def estimate_required_content_for_duration(
        self, target_duration: float, content_style: str = "conversational"
    ) -> Dict[str, Any]:
        """
        Estimate how much content is needed to reach target duration

        Args:
            target_duration: Target duration in minutes
            content_style: Style of content for rate estimation

        Returns:
            Dict with content requirements
        """

        # Get base speech rate for style
        base_wpm = self.speech_rates.get(
            content_style, self.speech_rates["conversational"]
        )

        # Account for conversation flow adjustment
        flow_adjustment = self._calculate_conversation_flow_adjustment(
            30, {content_style: 100}, content_style
        )

        # Calculate target speech time (accounting for pauses)
        pause_ratio = 0.15  # Assume 15% of time is pauses
        target_speech_time = target_duration * (1 - pause_ratio)

        # Adjust for conversation flow
        adjusted_speech_time = target_speech_time / flow_adjustment

        # Calculate required words
        required_words = int(adjusted_speech_time * base_wpm)

        # Estimate dialogue items needed (average words per dialogue item)
        avg_words_per_item = 25  # Typical podcast dialogue length
        estimated_dialogue_items = int(required_words / avg_words_per_item)

        return {
            "target_duration": target_duration,
            "required_words": required_words,
            "estimated_dialogue_items": estimated_dialogue_items,
            "base_wpm": base_wpm,
            "flow_adjustment_factor": flow_adjustment,
            "content_style": content_style,
            "recommendations": {
                "words_per_minute_target": base_wpm,
                "dialogue_items_per_minute": estimated_dialogue_items / target_duration
                if target_duration > 0
                else 0,
                "average_words_per_dialogue": avg_words_per_item,
            },
        }

    def validate_duration_accuracy(
        self,
        dialogue: List[Dict[str, Any]],
        target_duration: float,
        tolerance_percentage: float = 10,
    ) -> Dict[str, Any]:
        """
        Validate if dialogue meets duration target within tolerance

        Args:
            dialogue: List of dialogue items
            target_duration: Target duration in minutes
            tolerance_percentage: Acceptable variance percentage

        Returns:
            Validation results with recommendations
        """

        duration_result = self.estimate_dialogue_duration(dialogue, target_duration)
        estimated_duration = duration_result["estimated_duration"]

        accuracy_info = duration_result["accuracy_info"]
        meets_tolerance = (
            abs(accuracy_info["percentage_difference"]) <= tolerance_percentage
        )

        recommendations = []
        if not meets_tolerance:
            if estimated_duration < target_duration:
                shortage = target_duration - estimated_duration
                additional_words = self.estimate_required_content_for_duration(
                    shortage
                )["required_words"]
                recommendations.append(
                    f"Add approximately {additional_words} more words of content"
                )
                recommendations.append(
                    "Consider expanding existing topics or adding new discussion points"
                )
            else:
                excess = estimated_duration - target_duration
                excess_words = int(excess * self.speech_rates["conversational"])
                recommendations.append(
                    f"Remove approximately {excess_words} words of content"
                )
                recommendations.append(
                    "Consider condensing explanations or removing tangential discussions"
                )

        return {
            "meets_target": meets_tolerance,
            "validation_passed": meets_tolerance,
            "duration_analysis": duration_result,
            "tolerance_percentage": tolerance_percentage,
            "recommendations": recommendations,
            "action_required": "none" if meets_tolerance else "content_adjustment",
        }
