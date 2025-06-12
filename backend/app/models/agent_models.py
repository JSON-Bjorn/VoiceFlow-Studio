from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from datetime import datetime
from ..core.database import Base


class AgentDecisionHistory(Base):
    """Store agent decisions and outcomes for learning"""

    __tablename__ = "agent_decision_history"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String, index=True)
    decision_type = Column(String, index=True)
    context_data = Column(JSON)  # Input context
    decision_data = Column(JSON)  # Agent's decision
    outcome_data = Column(JSON)  # Actual outcome
    confidence_score = Column(Float)
    actual_cost = Column(Float)
    quality_score = Column(Float)
    success = Column(Boolean)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class AgentLearningData(Base):
    """Store agent learning patterns and models"""

    __tablename__ = "agent_learning_data"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String, index=True)
    learning_type = Column(String, index=True)  # "pattern", "model", "metric"
    key = Column(String, index=True)  # Pattern identifier
    data = Column(JSON)  # Learning data
    confidence = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
