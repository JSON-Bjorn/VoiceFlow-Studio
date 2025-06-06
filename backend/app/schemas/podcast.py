from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from ..models.podcast import PodcastStatus


class PodcastBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    topic: str = Field(..., min_length=1, max_length=500)
    length: int = Field(..., ge=1, le=60)  # 1-60 minutes


class PodcastCreate(PodcastBase):
    pass


class PodcastUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    topic: Optional[str] = Field(None, min_length=1, max_length=500)
    length: Optional[int] = Field(None, ge=1, le=60)
    status: Optional[PodcastStatus] = None
    audio_url: Optional[str] = None
    script: Optional[str] = None
    has_audio: Optional[bool] = None
    audio_file_paths: Optional[List[str]] = None
    audio_segments_count: Optional[int] = None
    audio_total_duration: Optional[float] = None
    voice_generation_cost: Optional[str] = None


class PodcastResponse(PodcastBase):
    id: int
    user_id: int
    status: PodcastStatus
    audio_url: Optional[str] = None
    script: Optional[str] = None
    has_audio: Optional[bool] = None
    audio_file_paths: Optional[List[str]] = None
    audio_segments_count: Optional[int] = None
    audio_total_duration: Optional[float] = None
    voice_generation_cost: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PodcastListResponse(BaseModel):
    podcasts: list[PodcastResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


class PodcastSummary(BaseModel):
    total_podcasts: int
    completed_podcasts: int
    pending_podcasts: int
    failed_podcasts: int
