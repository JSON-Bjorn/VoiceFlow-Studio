from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    Float,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base


class VoiceProfile(Base):
    __tablename__ = "voice_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id"), nullable=True, index=True
    )  # Nullable for system voices
    voice_id = Column(
        String, unique=True, index=True, nullable=False
    )  # Unique identifier for TTS
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    # Audio file paths
    original_audio_path = Column(
        String,
        nullable=True,  # Nullable for synthetic voices
    )  # Path to original voice sample
    test_audio_path = Column(String, nullable=True)  # Path to test sample

    # Voice characteristics
    gender = Column(String, default="unknown")
    style = Column(String, default="custom")
    is_active = Column(Boolean, default=True)

    # Enhanced system voice fields
    accent = Column(String, nullable=True)  # "american", "british", "australian", etc.
    voice_type = Column(String, default="custom")  # "cloned", "synthetic", "custom"
    generation_seed = Column(Integer, nullable=True)  # Seed used for generation
    is_system_default = Column(
        Boolean, default=False
    )  # Whether this is a system default voice

    # Optimization parameters (stored for consistency)
    optimal_similarity_boost = Column(Float, default=0.95)
    optimal_stability = Column(Float, default=0.7)
    optimal_style = Column(Float, default=0.2)
    optimal_exaggeration = Column(Float, default=0.3)
    optimal_cfg_weight = Column(Float, default=0.7)
    optimal_temperature = Column(Float, default=0.5)

    # Metadata
    file_size = Column(Integer, nullable=True)  # Size in bytes
    duration = Column(Float, nullable=True)  # Duration in seconds
    original_filename = Column(String, nullable=True)
    content_type = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="voice_profiles")
