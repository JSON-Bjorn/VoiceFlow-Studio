from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from ...models.agent_models import AgentDecisionHistory, AgentLearningData


class AgentDataStore:
    """Data access layer for agent learning and decision history"""

    def __init__(self, db: Session):
        self.db = db

    async def store_decision(
        self,
        agent_name: str,
        decision_type: str,
        context: Dict[str, Any],
        decision: Dict[str, Any],
        confidence: float,
    ) -> int:
        """Store an agent decision"""

        decision_record = AgentDecisionHistory(
            agent_name=agent_name,
            decision_type=decision_type,
            context_data=context,
            decision_data=decision,
            confidence_score=confidence,
        )

        self.db.add(decision_record)
        self.db.commit()
        self.db.refresh(decision_record)

        return decision_record.id

    async def update_decision_outcome(
        self,
        decision_id: int,
        outcome: Dict[str, Any],
        actual_cost: float,
        quality_score: float,
        success: bool,
    ):
        """Update a decision with its outcome"""

        decision_record = (
            self.db.query(AgentDecisionHistory)
            .filter(AgentDecisionHistory.id == decision_id)
            .first()
        )

        if decision_record:
            decision_record.outcome_data = outcome
            decision_record.actual_cost = actual_cost
            decision_record.quality_score = quality_score
            decision_record.success = success
            self.db.commit()

    async def get_agent_history(
        self, agent_name: str, days: int = 30
    ) -> List[AgentDecisionHistory]:
        """Get recent decision history for an agent"""

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        return (
            self.db.query(AgentDecisionHistory)
            .filter(
                AgentDecisionHistory.agent_name == agent_name,
                AgentDecisionHistory.timestamp >= cutoff_date,
            )
            .all()
        )

    async def store_learning_data(
        self,
        agent_name: str,
        learning_type: str,
        key: str,
        data: Dict[str, Any],
        confidence: float,
    ):
        """Store agent learning data"""

        # Check if learning data already exists
        existing = (
            self.db.query(AgentLearningData)
            .filter(
                AgentLearningData.agent_name == agent_name,
                AgentLearningData.learning_type == learning_type,
                AgentLearningData.key == key,
            )
            .first()
        )

        if existing:
            existing.data = data
            existing.confidence = confidence
            existing.last_updated = datetime.utcnow()
        else:
            learning_record = AgentLearningData(
                agent_name=agent_name,
                learning_type=learning_type,
                key=key,
                data=data,
                confidence=confidence,
            )
            self.db.add(learning_record)

        self.db.commit()

    async def get_learning_data(
        self, agent_name: str, learning_type: str, key: Optional[str] = None
    ) -> List[AgentLearningData]:
        """Retrieve agent learning data"""

        query = self.db.query(AgentLearningData).filter(
            AgentLearningData.agent_name == agent_name,
            AgentLearningData.learning_type == learning_type,
        )

        if key:
            query = query.filter(AgentLearningData.key == key)

        return query.all()
