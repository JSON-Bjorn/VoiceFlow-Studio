import openai
from typing import Dict, List, Optional, Any
from ..core.config import settings
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = openai.OpenAI(api_key=settings.openai_api_key)


class OpenAIService:
    """Service for handling OpenAI API interactions"""

    @staticmethod
    def _make_request(
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Make a request to OpenAI API with error handling

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: OpenAI model to use
            temperature: Creativity level (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            Response content or None if error
        """
        try:
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            if max_tokens:
                request_params["max_tokens"] = max_tokens

            if response_format:
                request_params["response_format"] = response_format

            response = client.chat.completions.create(**request_params)

            return response.choices[0].message.content

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI request: {e}")
            return None

    @staticmethod
    def generate_research_topics(
        main_topic: str, target_length: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Generate research subtopics and summaries for a podcast topic

        Args:
            main_topic: The main podcast topic
            target_length: Target podcast length in minutes

        Returns:
            Dictionary with subtopics and research data
        """
        prompt = f"""
        You are a podcast research assistant. Generate comprehensive research for a {target_length}-minute podcast about: "{main_topic}"

        Create a structured research plan with:
        1. 3-5 key subtopics that would make for engaging discussion
        2. For each subtopic, provide:
           - A brief summary (2-3 sentences)
           - 2-3 interesting facts or statistics
           - Potential discussion angles or questions

        Format your response as JSON with this structure:
        {{
            "main_topic": "{main_topic}",
            "estimated_segments": 3-5,
            "subtopics": [
                {{
                    "title": "Subtopic Title",
                    "summary": "Brief summary of this subtopic",
                    "key_facts": ["Fact 1", "Fact 2", "Fact 3"],
                    "discussion_angles": ["Angle 1", "Angle 2"]
                }}
            ],
            "overall_narrative": "How these subtopics connect to tell a complete story"
        }}
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert podcast researcher who creates engaging, factual content.",
            },
            {"role": "user", "content": prompt},
        ]

        response = OpenAIService._make_request(
            messages=messages, temperature=0.8, response_format={"type": "json_object"}
        )

        if response:
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse research JSON: {e}")
                return None

        return None

    @staticmethod
    def generate_podcast_script(
        research_data: Dict[str, Any],
        target_length: int = 10,
        host_personalities: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a podcast script based on research data

        Args:
            research_data: Research data from generate_research_topics
            target_length: Target length in minutes
            host_personalities: Optional host personality definitions

        Returns:
            Dictionary with complete podcast script
        """

        # Default host personalities if none provided
        if not host_personalities:
            host_personalities = {
                "host_1": {
                    "name": "Bjorn",
                    "personality": "analytical and engaging",
                    "role": "primary_questioner",
                },
                "host_2": {
                    "name": "Felix",
                    "personality": "warm and curious",
                    "role": "storyteller",
                },
            }

        prompt = f"""
        You are a podcast script writer. Create a natural, engaging {target_length}-minute podcast script based on this research:

        RESEARCH DATA:
        {json.dumps(research_data, indent=2)}

        HOST PERSONALITIES:
        {json.dumps(host_personalities, indent=2)}

        Create a script with:
        1. Natural intro (30-60 seconds)
        2. Main content segments covering each subtopic
        3. Smooth transitions between topics
        4. Natural outro (30-60 seconds)
        5. Realistic dialogue that matches each host's personality

        Format as JSON:
        {{
            "title": "Podcast Episode Title",
            "estimated_duration": {target_length},
            "segments": [
                {{
                    "type": "intro",
                    "duration_estimate": 1,
                    "dialogue": [
                        {{"speaker": "Bjorn", "text": "Welcome to our show..."}},
                        {{"speaker": "Felix", "text": "Today we're exploring..."}},
                        {{"speaker": "Bjorn", "text": "That's fascinating..."}},
                        {{"speaker": "Felix", "text": "Let me break that down..."}},
                        {{"speaker": "Bjorn", "text": "So let's start with..."}},
                        {{"speaker": "Felix", "text": "That's really interesting because..."}}
                    ]
                }},
                {{
                    "type": "main_content",
                    "subtopic": "Subtopic Title",
                    "duration_estimate": 3,
                    "dialogue": [
                        {{"speaker": "Bjorn", "text": "So let's start with..."}},
                        {{"speaker": "Felix", "text": "That's really interesting because..."}}
                    ]
                }}
            ]
        }}

        Make the conversation feel natural and spontaneous, not scripted. Include natural speech patterns, interruptions, and reactions.
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert podcast script writer who creates natural, engaging dialogue.",
            },
            {"role": "user", "content": prompt},
        ]

        response = OpenAIService._make_request(
            messages=messages,
            temperature=0.9,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )

        if response:
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse script JSON: {e}")
                return None

        return None

    @staticmethod
    def refine_script_segment(
        segment: Dict[str, Any], feedback: str, context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Refine a specific script segment based on feedback

        Args:
            segment: The script segment to refine
            feedback: Feedback on what to improve
            context: Optional context about the overall podcast

        Returns:
            Refined segment or None if error
        """
        prompt = f"""
        Refine this podcast script segment based on the feedback:

        ORIGINAL SEGMENT:
        {json.dumps(segment, indent=2)}

        FEEDBACK:
        {feedback}

        {f"CONTEXT: {context}" if context else ""}

        Return the improved segment in the same JSON format, maintaining the natural dialogue style.
        """

        messages = [
            {
                "role": "system",
                "content": "You are a podcast script editor who improves dialogue based on feedback.",
            },
            {"role": "user", "content": prompt},
        ]

        response = OpenAIService._make_request(
            messages=messages, temperature=0.7, response_format={"type": "json_object"}
        )

        if response:
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse refined segment JSON: {e}")
                return None

        return None

    @staticmethod
    def test_connection() -> bool:
        """Test if OpenAI API is working"""
        try:
            response = OpenAIService._make_request(
                messages=[
                    {"role": "user", "content": "Say 'API connection successful'"}
                ],
                model="gpt-4o-mini",
                max_tokens=10,
            )
            return response is not None and "successful" in response.lower()
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
