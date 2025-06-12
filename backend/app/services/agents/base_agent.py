from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from dataclasses import dataclass


@dataclass
class AgentDecision:
    """Standard decision object returned by all agents"""

    agent_name: str
    decision_type: str
    confidence: float  # 0.0-1.0
    reasoning: str
    data: Dict[str, Any]
    timestamp: datetime
    execution_cost: Optional[float] = None


class BaseIntelligentAgent(ABC):
    """Base class for all intelligent agents with learning capabilities"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"agent.{agent_name}")
        self.decision_history = []
        self.performance_metrics = {}
        self.learning_data = {}

    @abstractmethod
    async def make_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """Make an intelligent decision based on context and learning"""
        pass

    @abstractmethod
    async def learn_from_outcome(
        self, decision: AgentDecision, outcome: Dict[str, Any]
    ):
        """Learn from the outcome of a previous decision"""
        pass

    async def update_performance_metrics(
        self, decision: AgentDecision, outcome: Dict[str, Any]
    ):
        """Update performance tracking"""
        # Track decision accuracy, cost efficiency, user satisfaction
        pass

    def get_confidence_score(self, context: Dict[str, Any]) -> float:
        """Calculate confidence in decision-making for this context"""
        # Base implementation - override in specific agents
        return 0.7
