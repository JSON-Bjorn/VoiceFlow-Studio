from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
    Enum,
    JSON,
    Boolean,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from ..core.database import Base


class PodcastStatus(enum.Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class Podcast(Base):
    __tablename__ = "podcasts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    length = Column(Integer, nullable=False)  # Length in minutes
    status = Column(Enum(PodcastStatus), default=PodcastStatus.PENDING)
    audio_url = Column(String, nullable=True)
    script = Column(Text, nullable=True)

    # Audio storage fields
    has_audio = Column(Boolean, default=False)
    audio_segments_count = Column(Integer, default=0)
    audio_total_duration = Column(Integer, nullable=True)  # Duration in seconds
    voice_generation_cost = Column(String, nullable=True)  # Cost in USD
    audio_file_paths = Column(JSON, nullable=True)  # List of file paths

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    user = relationship("User", back_populates="podcasts")
