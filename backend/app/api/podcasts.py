from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
import math

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..models.podcast import PodcastStatus
from ..schemas.podcast import (
    PodcastCreate,
    PodcastUpdate,
    PodcastResponse,
    PodcastListResponse,
    PodcastSummary,
)
from ..services.podcast_service import PodcastService

router = APIRouter(prefix="/api/podcasts", tags=["podcasts"])


@router.post("/", response_model=PodcastResponse)
async def create_podcast(
    podcast_data: PodcastCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new podcast"""
    try:
        service = PodcastService(db)
        podcast = service.create_podcast(current_user.id, podcast_data)
        return podcast
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=PodcastListResponse)
async def get_user_podcasts(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50),
    status: Optional[PodcastStatus] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's podcasts with pagination and optional status filter"""
    service = PodcastService(db)
    podcasts, total = service.get_user_podcasts(
        user_id=current_user.id, page=page, per_page=per_page, status_filter=status
    )

    total_pages = math.ceil(total / per_page) if total > 0 else 1

    return PodcastListResponse(
        podcasts=podcasts,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
    )


@router.get("/summary", response_model=PodcastSummary)
async def get_podcast_summary(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get summary statistics for user's podcasts"""
    service = PodcastService(db)
    return service.get_podcast_summary(current_user.id)


@router.get("/recent", response_model=list[PodcastResponse])
async def get_recent_podcasts(
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's most recent podcasts"""
    service = PodcastService(db)
    podcasts = service.get_recent_podcasts(current_user.id, limit)
    return podcasts


@router.get("/{podcast_id}", response_model=PodcastResponse)
async def get_podcast(
    podcast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a specific podcast by ID"""
    service = PodcastService(db)
    podcast = service.get_podcast_by_id(podcast_id, current_user.id)

    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    return podcast


@router.put("/{podcast_id}", response_model=PodcastResponse)
async def update_podcast(
    podcast_id: int,
    update_data: PodcastUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update a podcast"""
    service = PodcastService(db)
    podcast = service.update_podcast(podcast_id, current_user.id, update_data)

    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    return podcast


@router.delete("/{podcast_id}")
async def delete_podcast(
    podcast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a podcast"""
    service = PodcastService(db)
    success = service.delete_podcast(podcast_id, current_user.id)

    if not success:
        raise HTTPException(status_code=404, detail="Podcast not found")

    return {"message": "Podcast deleted successfully"}


@router.post("/{podcast_id}/generate", response_model=PodcastResponse)
async def simulate_podcast_generation(
    podcast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Simulate podcast generation (for testing purposes)"""
    service = PodcastService(db)
    podcast = service.simulate_podcast_generation(podcast_id, current_user.id)

    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    return podcast
