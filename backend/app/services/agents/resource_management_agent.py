from .base_agent import BaseIntelligentAgent, AgentDecision
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime, timedelta


class ResourceManagementAgent(BaseIntelligentAgent):
    """
    Intelligent API resource allocation and cost optimization agent

    Responsibilities:
    - Learn optimal token allocation patterns per topic type and length
    - Predict generation costs before execution
    - Dynamically adjust resource allocation during generation
    - Optimize prompt efficiency for maximum token utilization
    - Learn from cost/quality tradeoffs in historical data
    """

    def __init__(self):
        super().__init__("ResourceManagement")
        self.cost_history = {}  # topic_type -> [cost_data]
        self.token_efficiency_patterns = {}  # length -> optimal_allocation
        self.topic_complexity_scores = {}  # topic -> complexity_score
        self.user_budget_patterns = {}  # user_id -> budget_preferences

    async def make_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """
        Make intelligent resource allocation decision

        Context expected:
        - topic: str
        - target_length: int (minutes)
        - user_budget: float
        - user_id: int
        - quality_requirements: dict
        """
        topic = context.get("topic", "")
        target_length = context.get("target_length", 10)
        user_budget = context.get("user_budget", 5.0)
        quality_requirements = context.get("quality_requirements", {})

        # 1. Analyze topic complexity
        complexity_score = await self._analyze_topic_complexity(topic)

        # 2. Predict optimal token allocation
        optimal_allocation = await self._predict_optimal_allocation(
            target_length, complexity_score, user_budget
        )

        # 3. Calculate confidence based on historical data
        confidence = await self._calculate_confidence(topic, target_length)

        # 4. Generate cost prediction
        cost_prediction = await self._predict_total_cost(
            optimal_allocation, complexity_score
        )

        decision_data = {
            "token_allocation": optimal_allocation,
            "complexity_score": complexity_score,
            "predicted_cost": cost_prediction,
            "quality_confidence": confidence,
            "optimization_strategy": await self._select_optimization_strategy(context),
        }

        reasoning = f"Allocated {optimal_allocation['total_tokens']} tokens based on complexity {complexity_score:.2f} and length {target_length}min. Predicted cost: ${cost_prediction:.2f}"

        return AgentDecision(
            agent_name=self.agent_name,
            decision_type="resource_allocation",
            confidence=confidence,
            reasoning=reasoning,
            data=decision_data,
            timestamp=datetime.utcnow(),
            execution_cost=cost_prediction,
        )

    async def _analyze_topic_complexity(self, topic: str) -> float:
        """Analyze topic complexity to predict resource needs"""

        # Check if we've seen this topic before
        if topic in self.topic_complexity_scores:
            return self.topic_complexity_scores[topic]

        # Complexity indicators
        complexity_factors = {
            "technical_terms": len([word for word in topic.split() if len(word) > 8]),
            "abstract_concepts": any(
                word in topic.lower()
                for word in ["philosophy", "quantum", "metaphysics", "consciousness"]
            ),
            "specialized_domain": any(
                word in topic.lower()
                for word in ["medical", "legal", "scientific", "engineering"]
            ),
            "controversial_topic": any(
                word in topic.lower()
                for word in ["politics", "religion", "controversial", "debate"]
            ),
            "word_count": len(topic.split()),
        }

        # Calculate complexity score (0.0-1.0)
        base_score = 0.3  # Minimum complexity

        if complexity_factors["technical_terms"] > 2:
            base_score += 0.2
        if complexity_factors["abstract_concepts"]:
            base_score += 0.15
        if complexity_factors["specialized_domain"]:
            base_score += 0.2
        if complexity_factors["controversial_topic"]:
            base_score += 0.1
        if complexity_factors["word_count"] > 10:
            base_score += 0.05

        complexity_score = min(1.0, base_score)

        # Cache for future use
        self.topic_complexity_scores[topic] = complexity_score

        return complexity_score

    async def _predict_optimal_allocation(
        self, target_length: int, complexity: float, budget: float
    ) -> Dict[str, int]:
        """Predict optimal token allocation based on learning data"""

        # Base allocation formula learned from historical data
        base_tokens_per_minute = 600  # Average tokens per minute of content
        complexity_multiplier = 1.0 + (complexity * 0.5)  # 1.0-1.5x based on complexity

        total_content_tokens = int(
            target_length * base_tokens_per_minute * complexity_multiplier
        )

        # Intelligent phase allocation based on learned patterns
        if target_length <= 5:
            # Short podcasts: focus on quality over quantity
            research_ratio = 0.4  # More research for short content
            script_ratio = 0.6
        elif target_length <= 15:
            # Medium podcasts: balanced approach
            research_ratio = 0.3
            script_ratio = 0.7
        else:
            # Long podcasts: efficient content generation
            research_ratio = 0.25
            script_ratio = 0.75

        # Adjust for complexity
        if complexity > 0.7:
            research_ratio += 0.1  # More research for complex topics
            script_ratio -= 0.1

        allocation = {
            "research_tokens": int(total_content_tokens * research_ratio),
            "script_tokens": int(total_content_tokens * script_ratio),
            "total_tokens": total_content_tokens,
            "reserve_tokens": int(total_content_tokens * 0.1),  # 10% reserve
            "allocation_strategy": "adaptive_learning",
        }

        # Budget constraint check
        estimated_cost = self._estimate_cost_from_tokens(allocation["total_tokens"])
        if estimated_cost > budget:
            # Scale down proportionally
            scale_factor = budget / estimated_cost
            for key in allocation:
                if key.endswith("_tokens"):
                    allocation[key] = int(allocation[key] * scale_factor)

        return allocation

    async def _calculate_confidence(self, topic: str, target_length: int) -> float:
        """Calculate confidence based on historical data similarity"""

        # Base confidence
        confidence = 0.6

        # Increase confidence if we have similar historical data
        similar_topics = self._find_similar_topics(topic)
        if similar_topics:
            confidence += 0.2

        # Length-based confidence (we're more confident with common lengths)
        common_lengths = [5, 10, 15, 20, 30]
        if target_length in common_lengths:
            confidence += 0.1

        # Historical performance boost
        if len(self.cost_history) > 10:
            confidence += 0.1

        return min(1.0, confidence)

    async def _predict_total_cost(
        self, allocation: Dict[str, int], complexity: float
    ) -> float:
        """Predict total generation cost"""

        # OpenAI pricing (approximate - update with real pricing)
        cost_per_1k_tokens = 0.002  # GPT-4 pricing

        total_tokens = allocation.get("total_tokens", 0)
        base_cost = (total_tokens / 1000) * cost_per_1k_tokens

        # Complexity can increase cost due to retries/refinements
        complexity_cost_factor = 1.0 + (complexity * 0.3)

        total_cost = base_cost * complexity_cost_factor

        return round(total_cost, 4)

    async def _select_optimization_strategy(self, context: Dict[str, Any]) -> str:
        """Select the best optimization strategy for this context"""

        budget = context.get("user_budget", 5.0)
        quality_requirements = context.get("quality_requirements", {})

        if budget < 2.0:
            return "cost_optimized"
        elif quality_requirements.get("high_quality", False):
            return "quality_optimized"
        else:
            return "balanced"

    async def learn_from_outcome(
        self, decision: AgentDecision, outcome: Dict[str, Any]
    ):
        """Learn from the outcome of resource allocation decisions"""

        actual_cost = outcome.get("actual_cost", 0.0)
        actual_tokens = outcome.get("actual_tokens_used", 0)
        quality_score = outcome.get("quality_score", 0.0)  # User feedback
        generation_success = outcome.get("success", False)

        # Store learning data
        learning_entry = {
            "decision": decision,
            "outcome": outcome,
            "cost_accuracy": abs(decision.execution_cost - actual_cost),
            "token_efficiency": actual_tokens
            / decision.data["token_allocation"]["total_tokens"],
            "quality_achieved": quality_score,
            "timestamp": datetime.utcnow(),
        }

        # Update learning models
        topic = decision.data.get("topic", "unknown")
        if topic not in self.cost_history:
            self.cost_history[topic] = []

        self.cost_history[topic].append(learning_entry)

        # Update token efficiency patterns
        length = decision.data.get("target_length", 0)
        if length not in self.token_efficiency_patterns:
            self.token_efficiency_patterns[length] = []

        self.token_efficiency_patterns[length].append(
            {
                "allocation": decision.data["token_allocation"],
                "efficiency": learning_entry["token_efficiency"],
                "quality": quality_score,
            }
        )

        self.logger.info(
            f"Learned from outcome: Cost accuracy {learning_entry['cost_accuracy']:.4f}, Token efficiency {learning_entry['token_efficiency']:.2f}"
        )

    def _estimate_cost_from_tokens(self, tokens: int) -> float:
        """Estimate cost from token count"""
        return (tokens / 1000) * 0.002  # Approximate GPT-4 pricing

    def _find_similar_topics(self, topic: str) -> List[str]:
        """Find similar topics in historical data"""
        # Simple keyword matching - could be improved with semantic similarity
        topic_words = set(topic.lower().split())
        similar_topics = []

        for historical_topic in self.topic_complexity_scores.keys():
            historical_words = set(historical_topic.lower().split())
            overlap = len(topic_words.intersection(historical_words))
            if overlap >= 2:  # At least 2 words in common
                similar_topics.append(historical_topic)

        return similar_topics

    # Monitoring and Real-time Adjustment Methods
    async def monitor_generation_progress(
        self, generation_id: str, current_phase: str, tokens_used: int
    ):
        """Monitor ongoing generation and suggest adjustments"""

        # Check if we're on track with token budget
        expected_usage = await self._get_expected_token_usage(
            generation_id, current_phase
        )

        if tokens_used > expected_usage * 1.2:  # 20% over budget
            return {
                "adjustment_needed": True,
                "recommendation": "reduce_remaining_allocation",
                "suggested_reduction": 0.8,  # Scale down to 80%
            }
        elif tokens_used < expected_usage * 0.7:  # Using much less than expected
            return {
                "adjustment_needed": True,
                "recommendation": "increase_quality_focus",
                "suggested_quality_boost": 1.2,  # Allow 20% more tokens for quality
            }

        return {"adjustment_needed": False}

    async def _get_expected_token_usage(
        self, generation_id: str, current_phase: str
    ) -> int:
        """Get expected token usage for current phase"""
        # Implementation depends on tracking system
        # Return expected tokens used by this point in generation
        return 1000  # Placeholder
