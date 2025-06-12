from .base_agent import BaseIntelligentAgent, AgentDecision
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime, timedelta
import aiohttp
import psutil
import logging


class ReliabilityAgent(BaseIntelligentAgent):
    """
    Intelligent failure prediction, recovery, and system resilience agent

    Responsibilities:
    - Predict API failure likelihood based on system health
    - Create intelligent checkpoint strategies
    - Select optimal recovery approaches for different failure types
    - Monitor external service health (OpenAI, etc.)
    - Learn failure patterns and prevention strategies
    """

    def __init__(self):
        super().__init__("Reliability")
        self.failure_patterns = {}
        self.recovery_strategies = {}
        self.system_health_metrics = {}
        self.checkpoint_strategies = {}
        self.api_health_history = {}
        self.recent_api_calls = []  # Track recent API calls for rate limit prediction

    async def make_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """
        Make intelligent reliability decision

        Context types:
        - checkpoint_strategy: When/what to checkpoint
        - failure_recovery: How to recover from specific failure
        - risk_assessment: Assess failure risk before operation
        """

        decision_type = context.get("decision_type")

        if decision_type == "checkpoint_strategy":
            return await self._decide_checkpoint_strategy(context)
        elif decision_type == "failure_recovery":
            return await self._decide_recovery_strategy(context)
        elif decision_type == "risk_assessment":
            return await self._assess_failure_risk(context)
        else:
            raise ValueError(f"Unknown decision type: {decision_type}")

    async def _decide_checkpoint_strategy(
        self, context: Dict[str, Any]
    ) -> AgentDecision:
        """Decide when and what to checkpoint"""

        generation_cost = context.get("generation_cost", 0.0)
        current_phase = context.get("current_phase", "unknown")
        completion_percentage = context.get("completion_percentage", 0.0)

        # Risk-based checkpointing
        failure_risk = await self._calculate_current_failure_risk()

        # Cost-benefit analysis of checkpointing
        checkpoint_cost = generation_cost * 0.05  # 5% overhead for checkpointing
        potential_loss = generation_cost * (1.0 - completion_percentage)

        should_checkpoint = (failure_risk * potential_loss) > checkpoint_cost

        # Determine checkpoint type and frequency based on risk
        if failure_risk > 0.8:
            checkpoint_type = "full"
            checkpoint_frequency = "every_operation"
        elif failure_risk > 0.6:
            checkpoint_type = "full"
            checkpoint_frequency = "every_phase"
        elif failure_risk > 0.4:
            checkpoint_type = "incremental"
            checkpoint_frequency = "major_phases_only"
        else:
            checkpoint_type = "minimal"
            checkpoint_frequency = "major_phases_only"

        strategy = {
            "should_checkpoint": should_checkpoint,
            "checkpoint_type": checkpoint_type,
            "checkpoint_frequency": checkpoint_frequency,
            "risk_factors": await self._get_current_risk_factors(),
            "estimated_overhead": checkpoint_cost,
            "potential_savings": potential_loss if should_checkpoint else 0.0,
        }

        confidence = 0.8 if len(self.failure_patterns) > 10 else 0.6

        reasoning = f"Checkpoint {checkpoint_type} strategy based on {failure_risk:.2f} failure risk and ${potential_loss:.2f} potential loss"

        return AgentDecision(
            agent_name=self.agent_name,
            decision_type="checkpoint_strategy",
            confidence=confidence,
            reasoning=reasoning,
            data=strategy,
            timestamp=datetime.utcnow(),
        )

    async def _decide_recovery_strategy(self, context: Dict[str, Any]) -> AgentDecision:
        """Decide how to recover from a specific failure"""

        failure_type = context.get("failure_type", "unknown")
        failure_details = context.get("failure_details", {})
        generation_state = context.get("generation_state", {})
        generation_cost_so_far = context.get("generation_cost_so_far", 0.0)

        # Learn from similar past failures
        similar_failures = self._find_similar_failures(failure_type, failure_details)

        # Select recovery strategy based on failure type and history
        if failure_type == "api_rate_limit":
            strategy = {
                "action": "exponential_backoff",
                "wait_time": await self._calculate_optimal_wait_time(failure_details),
                "retry_count": 3,
                "fallback_strategy": "reduce_token_allocation",
                "cost_impact": "minimal",
            }
        elif failure_type == "api_timeout":
            strategy = {
                "action": "retry_with_smaller_chunks",
                "chunk_reduction": 0.7,
                "retry_count": 2,
                "fallback_strategy": "checkpoint_and_resume",
                "cost_impact": "low",
            }
        elif failure_type == "api_error":
            error_code = failure_details.get("error_code")
            if error_code == 400:  # Bad request
                strategy = {
                    "action": "prompt_revision",
                    "revision_type": "simplify",
                    "retry_count": 1,
                    "fallback_strategy": "use_cached_content",
                    "cost_impact": "moderate",
                }
            elif error_code in [401, 403]:  # Auth errors
                strategy = {
                    "action": "refresh_credentials",
                    "retry_count": 1,
                    "fallback_strategy": "full_refund",
                    "cost_impact": "none",
                }
            else:
                strategy = {
                    "action": "immediate_retry",
                    "retry_count": 2,
                    "fallback_strategy": "rollback_to_checkpoint",
                    "cost_impact": "low",
                }
        elif failure_type == "budget_exceeded":
            strategy = {
                "action": "reduce_scope",
                "scope_reduction": 0.8,  # Reduce to 80% of original scope
                "retry_count": 1,
                "fallback_strategy": "partial_refund",
                "cost_impact": "significant",
            }
        elif failure_type == "system_overload":
            strategy = {
                "action": "delayed_retry",
                "delay_minutes": 5,
                "retry_count": 2,
                "fallback_strategy": "queue_for_later",
                "cost_impact": "none",
            }
        else:
            # Unknown failure - conservative approach
            strategy = {
                "action": "rollback_to_checkpoint",
                "retry_count": 1,
                "fallback_strategy": "full_refund",
                "cost_impact": "none",
            }

        # Add learning from similar failures
        if similar_failures:
            strategy["learned_optimizations"] = self._extract_successful_patterns(
                similar_failures
            )
            strategy["success_probability"] = (
                self._calculate_strategy_success_probability(
                    similar_failures, strategy["action"]
                )
            )
        else:
            strategy["success_probability"] = 0.5  # Default for unknown patterns

        # Consider generation cost when choosing strategy
        if generation_cost_so_far > 2.0:  # High cost generation
            if strategy["cost_impact"] == "significant":
                strategy["recommended_action"] = "escalate_to_user"

        confidence = 0.9 if similar_failures else 0.5
        reasoning = f"Recovery strategy '{strategy['action']}' for {failure_type} based on {len(similar_failures)} similar past failures"

        return AgentDecision(
            agent_name=self.agent_name,
            decision_type="failure_recovery",
            confidence=confidence,
            reasoning=reasoning,
            data=strategy,
            timestamp=datetime.utcnow(),
        )

    async def _assess_failure_risk(self, context: Dict[str, Any]) -> AgentDecision:
        """Assess the risk of failure for an upcoming operation"""

        operation_type = context.get("operation_type", "unknown")
        api_calls_needed = context.get("api_calls_needed", 1)
        token_requirement = context.get("token_requirement", 1000)
        user_budget = context.get("user_budget", 5.0)

        # Assess multiple risk factors
        risk_factors = {
            "api_health": await self._check_api_health(),
            "rate_limit_risk": await self._assess_rate_limit_risk(api_calls_needed),
            "token_availability": await self._assess_token_availability(
                token_requirement
            ),
            "budget_risk": await self._assess_budget_risk(
                token_requirement, user_budget
            ),
            "historical_failure_rate": self._get_historical_failure_rate(
                operation_type
            ),
            "system_load": await self._check_system_load(),
        }

        # Calculate overall risk (0.0-1.0)
        risk_weights = {
            "api_health": 0.25,
            "rate_limit_risk": 0.2,
            "token_availability": 0.15,
            "budget_risk": 0.15,
            "historical_failure_rate": 0.15,
            "system_load": 0.1,
        }

        overall_risk = sum(
            risk_factors[factor] * risk_weights[factor] for factor in risk_factors
        )

        # Risk mitigation recommendations
        mitigations = []
        if risk_factors["api_health"] > 0.7:
            mitigations.append("delay_operation")
        if risk_factors["rate_limit_risk"] > 0.6:
            mitigations.append("reduce_concurrent_calls")
        if risk_factors["token_availability"] > 0.8:
            mitigations.append("optimize_token_usage")
        if risk_factors["budget_risk"] > 0.7:
            mitigations.append("reduce_scope")
        if risk_factors["system_load"] > 0.8:
            mitigations.append("wait_for_lower_load")

        # Determine risk level and proceed recommendation
        if overall_risk > 0.8:
            risk_level = "critical"
            proceed_recommendation = False
        elif overall_risk > 0.6:
            risk_level = "high"
            proceed_recommendation = (
                len(mitigations) > 0
            )  # Only if mitigations available
        elif overall_risk > 0.4:
            risk_level = "medium"
            proceed_recommendation = True
        else:
            risk_level = "low"
            proceed_recommendation = True

        assessment = {
            "overall_risk": overall_risk,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommended_mitigations": mitigations,
            "proceed_recommendation": proceed_recommendation,
            "estimated_success_probability": 1.0 - overall_risk,
            "risk_breakdown": {
                factor: {
                    "score": risk_factors[factor],
                    "weight": risk_weights[factor],
                    "contribution": risk_factors[factor] * risk_weights[factor],
                }
                for factor in risk_factors
            },
        }

        confidence = 0.8 if len(self.api_health_history) > 20 else 0.6
        reasoning = f"Risk assessment: {overall_risk:.2f} overall risk ({risk_level}) with {len(mitigations)} recommended mitigations"

        return AgentDecision(
            agent_name=self.agent_name,
            decision_type="risk_assessment",
            confidence=confidence,
            reasoning=reasoning,
            data=assessment,
            timestamp=datetime.utcnow(),
        )

    async def learn_from_outcome(
        self, decision: AgentDecision, outcome: Dict[str, Any]
    ):
        """Learn from reliability decision outcomes"""

        decision_type = decision.decision_type
        success = outcome.get("success", False)

        if decision_type == "checkpoint_strategy":
            # Learn checkpoint effectiveness
            checkpoint_used = outcome.get("checkpoint_used", False)
            recovery_time = outcome.get("recovery_time", 0)

            if checkpoint_used and success:
                # Successful checkpoint recovery
                self._update_checkpoint_strategy_success(decision.data, recovery_time)
            elif not checkpoint_used and not success:
                # Could have benefited from checkpointing
                self._update_checkpoint_missed_opportunity(decision.data)

        elif decision_type == "failure_recovery":
            # Learn recovery strategy effectiveness
            recovery_success = success
            recovery_time = outcome.get("recovery_time", 0)
            recovery_cost = outcome.get("recovery_cost", 0)

            self._update_recovery_strategy_effectiveness(
                decision.data, recovery_success, recovery_time, recovery_cost
            )

        elif decision_type == "risk_assessment":
            # Learn risk prediction accuracy
            actual_failure = outcome.get("failure_occurred", False)
            predicted_risk = decision.data["overall_risk"]

            self._update_risk_prediction_accuracy(predicted_risk, actual_failure)

        # Store learning data
        learning_key = (
            f"{decision_type}_{decision.data.get('operation_type', 'general')}"
        )
        learning_data = {
            "decision": decision.data,
            "outcome": outcome,
            "accuracy": self._calculate_decision_accuracy(decision, outcome),
            "timestamp": datetime.utcnow(),
        }

        if learning_key not in self.failure_patterns:
            self.failure_patterns[learning_key] = []

        self.failure_patterns[learning_key].append(learning_data)

        # Keep only recent learning data (last 100 entries per pattern)
        self.failure_patterns[learning_key] = self.failure_patterns[learning_key][-100:]

        self.logger.info(
            f"Learned from {decision_type}: accuracy {learning_data['accuracy']:.2f}"
        )

    # Helper methods for risk assessment
    async def _check_api_health(self) -> float:
        """Check OpenAI API health (0.0 = perfect, 1.0 = completely down)"""
        try:
            # Simple health check - could be expanded
            async with aiohttp.ClientSession() as session:
                start_time = datetime.utcnow()
                async with session.get(
                    "https://api.openai.com/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    response_time = (datetime.utcnow() - start_time).total_seconds()

                    if response.status == 200:
                        # Good response, but check response time
                        if response_time < 2.0:
                            health_score = 0.0  # Excellent health
                        elif response_time < 5.0:
                            health_score = 0.3  # Acceptable health
                        else:
                            health_score = 0.6  # Slow but working
                    else:
                        health_score = 0.8  # API issues

                    # Store in history
                    self.api_health_history[datetime.utcnow()] = {
                        "response_time": response_time,
                        "status_code": response.status,
                        "health_score": health_score,
                    }

                    return health_score

        except Exception as e:
            self.logger.warning(f"API health check failed: {e}")
            return 1.0  # API unreachable

    async def _assess_rate_limit_risk(self, api_calls_needed: int) -> float:
        """Assess risk of hitting rate limits"""
        # Track recent API call frequency
        recent_calls = self._get_recent_api_calls_count(minutes=60)

        # Estimate rate limit based on tier (simplified)
        estimated_rate_limit = 3000  # calls per hour for tier 1

        projected_usage = recent_calls + api_calls_needed
        usage_ratio = projected_usage / estimated_rate_limit

        if usage_ratio < 0.5:
            return 0.0  # Safe
        elif usage_ratio < 0.8:
            return 0.4  # Moderate risk
        elif usage_ratio < 0.95:
            return 0.7  # High risk
        else:
            return 1.0  # Almost certain rate limit

    async def _assess_token_availability(self, token_requirement: int) -> float:
        """Assess if we have sufficient token quota"""
        # This would check actual OpenAI token limits
        # Simplified implementation
        estimated_monthly_limit = 1000000  # tokens per month
        tokens_used_this_month = self._get_monthly_token_usage()

        remaining_tokens = estimated_monthly_limit - tokens_used_this_month

        if remaining_tokens > token_requirement * 5:
            return 0.0  # Plenty available
        elif remaining_tokens > token_requirement * 2:
            return 0.3  # Sufficient
        elif remaining_tokens > token_requirement:
            return 0.7  # Tight but possible
        else:
            return 1.0  # Insufficient

    async def _assess_budget_risk(
        self, token_requirement: int, user_budget: float
    ) -> float:
        """Assess risk of exceeding user budget"""
        estimated_cost = (token_requirement / 1000) * 0.002  # Rough GPT-4 pricing

        if estimated_cost < user_budget * 0.5:
            return 0.0  # Well within budget
        elif estimated_cost < user_budget * 0.8:
            return 0.3  # Moderate budget usage
        elif estimated_cost < user_budget:
            return 0.7  # Close to budget limit
        else:
            return 1.0  # Exceeds budget

    def _get_historical_failure_rate(self, operation_type: str) -> float:
        """Get historical failure rate for operation type"""
        if operation_type not in self.failure_patterns:
            return 0.3  # Default moderate risk for unknown operations

        recent_operations = self.failure_patterns[operation_type][
            -50:
        ]  # Last 50 operations
        if not recent_operations:
            return 0.3

        failures = sum(
            1
            for op in recent_operations
            if not op.get("outcome", {}).get("success", True)
        )
        failure_rate = failures / len(recent_operations)

        return failure_rate

    async def _check_system_load(self) -> float:
        """Check current system load"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            max_usage = max(cpu_percent, memory_percent)

            if max_usage < 50:
                return 0.0  # Low load
            elif max_usage < 75:
                return 0.3  # Moderate load
            elif max_usage < 90:
                return 0.7  # High load
            else:
                return 1.0  # Critical load
        except Exception:
            return 0.5  # Unknown load

    # Learning system methods
    def _find_similar_failures(
        self, failure_type: str, failure_details: dict
    ) -> List[Dict]:
        """Find similar past failures for learning"""
        similar = []

        for pattern_key, patterns in self.failure_patterns.items():
            for pattern in patterns:
                if pattern.get("decision", {}).get("failure_type") == failure_type:
                    similar.append(pattern)

        return similar[-10:]  # Return last 10 similar failures

    def _extract_successful_patterns(self, similar_failures: List[Dict]) -> Dict:
        """Extract successful recovery patterns from similar failures"""
        successful_strategies = {}

        for failure in similar_failures:
            if failure.get("outcome", {}).get("success", False):
                strategy = failure.get("decision", {}).get("action", "unknown")
                if strategy not in successful_strategies:
                    successful_strategies[strategy] = 0
                successful_strategies[strategy] += 1

        return successful_strategies

    def _calculate_strategy_success_probability(
        self, similar_failures: List[Dict], strategy: str
    ) -> float:
        """Calculate success probability for a specific strategy"""
        strategy_attempts = [
            f
            for f in similar_failures
            if f.get("decision", {}).get("action") == strategy
        ]

        if not strategy_attempts:
            return 0.5  # Default probability for unknown strategies

        successes = sum(
            1
            for attempt in strategy_attempts
            if attempt.get("outcome", {}).get("success", False)
        )

        return successes / len(strategy_attempts)

    def _update_checkpoint_strategy_success(self, strategy: dict, recovery_time: float):
        """Update checkpoint strategy based on successful usage"""
        strategy_key = f"{strategy.get('checkpoint_type', 'unknown')}_{strategy.get('checkpoint_frequency', 'unknown')}"

        if strategy_key not in self.checkpoint_strategies:
            self.checkpoint_strategies[strategy_key] = {
                "successes": 0,
                "total_recovery_time": 0.0,
            }

        self.checkpoint_strategies[strategy_key]["successes"] += 1
        self.checkpoint_strategies[strategy_key]["total_recovery_time"] += recovery_time

    def _update_checkpoint_missed_opportunity(self, strategy: dict):
        """Learn from situations where checkpointing would have helped"""
        # This could adjust future checkpoint thresholds to be more aggressive
        pass

    def _update_recovery_strategy_effectiveness(
        self, strategy: dict, success: bool, time: float, cost: float
    ):
        """Update recovery strategy effectiveness"""
        strategy_action = strategy.get("action", "unknown")

        if strategy_action not in self.recovery_strategies:
            self.recovery_strategies[strategy_action] = {
                "attempts": 0,
                "successes": 0,
                "total_time": 0.0,
                "total_cost": 0.0,
            }

        self.recovery_strategies[strategy_action]["attempts"] += 1
        if success:
            self.recovery_strategies[strategy_action]["successes"] += 1
        self.recovery_strategies[strategy_action]["total_time"] += time
        self.recovery_strategies[strategy_action]["total_cost"] += cost

    def _update_risk_prediction_accuracy(
        self, predicted_risk: float, actual_failure: bool
    ):
        """Update risk prediction model accuracy"""
        # Simple accuracy tracking - could be improved with more sophisticated models
        prediction_correct = (predicted_risk > 0.5) == actual_failure

        if "risk_predictions" not in self.system_health_metrics:
            self.system_health_metrics["risk_predictions"] = {"correct": 0, "total": 0}

        self.system_health_metrics["risk_predictions"]["total"] += 1
        if prediction_correct:
            self.system_health_metrics["risk_predictions"]["correct"] += 1

    def _calculate_decision_accuracy(
        self, decision: AgentDecision, outcome: Dict[str, Any]
    ) -> float:
        """Calculate how accurate the decision was"""
        if decision.decision_type == "risk_assessment":
            predicted_risk = decision.data["overall_risk"]
            actual_failure = outcome.get("failure_occurred", False)

            # Calculate accuracy based on how well risk prediction matched reality
            if actual_failure:
                return predicted_risk  # Higher risk prediction = more accurate
            else:
                return 1.0 - predicted_risk  # Lower risk prediction = more accurate

        elif decision.decision_type == "failure_recovery":
            return 1.0 if outcome.get("success", False) else 0.0

        elif decision.decision_type == "checkpoint_strategy":
            checkpoint_used = outcome.get("checkpoint_used", False)
            needed_checkpoint = outcome.get("needed_checkpoint", False)

            if checkpoint_used == needed_checkpoint:
                return 1.0  # Correct prediction
            else:
                return 0.0  # Incorrect prediction

        return 0.5  # Default for unknown decision types

    def _get_recent_api_calls_count(self, minutes: int) -> int:
        """Get API calls made in recent time period"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return len([call for call in self.recent_api_calls if call > cutoff_time])

    def _get_monthly_token_usage(self) -> int:
        """Get tokens used this month"""
        # This would integrate with actual token tracking
        return 50000  # Placeholder

    async def _calculate_current_failure_risk(self) -> float:
        """Calculate current overall failure risk"""
        api_health = await self._check_api_health()
        system_load = await self._check_system_load()

        return (api_health + system_load) / 2

    async def _get_current_risk_factors(self) -> Dict[str, float]:
        """Get all current risk factors"""
        return {
            "api_health": await self._check_api_health(),
            "system_load": await self._check_system_load(),
            "recent_failures": self._get_recent_failure_rate(),
        }

    def _get_recent_failure_rate(self) -> float:
        """Get failure rate in recent operations"""
        recent_failures = []
        cutoff_time = datetime.utcnow() - timedelta(hours=24)

        for patterns in self.failure_patterns.values():
            for pattern in patterns:
                if pattern.get("timestamp", datetime.min) > cutoff_time:
                    recent_failures.append(pattern)

        if not recent_failures:
            return 0.1  # Default low failure rate

        failed_operations = sum(
            1 for f in recent_failures if not f.get("outcome", {}).get("success", True)
        )

        return failed_operations / len(recent_failures)

    async def _calculate_optimal_wait_time(self, failure_details: dict) -> float:
        """Calculate optimal wait time for rate limit recovery"""
        # Exponential backoff with jitter
        retry_count = failure_details.get("retry_count", 0)
        base_wait = min(2**retry_count, 60)  # Cap at 60 seconds
        jitter = base_wait * 0.1  # 10% jitter

        return base_wait + jitter

    def track_api_call(self):
        """Track an API call for rate limit monitoring"""
        self.recent_api_calls.append(datetime.utcnow())

        # Keep only recent calls (last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.recent_api_calls = [
            call for call in self.recent_api_calls if call > cutoff_time
        ]
