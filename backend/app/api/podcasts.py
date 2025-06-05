from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
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
from ..services.storage_service import storage_service

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


@router.get("/{podcast_id}/download")
async def download_podcast(
    podcast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Download a completed podcast as an audio file"""
    service = PodcastService(db)
    podcast = service.get_podcast_by_id(podcast_id, current_user.id)

    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    if podcast.status != PodcastStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Podcast is not ready for download")

    if not podcast.has_audio or not podcast.audio_file_paths:
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        # Get the main episode file (not segments)
        main_audio_files = []
        if podcast.audio_file_paths:
            for file_path in podcast.audio_file_paths:
                if not "/segments/" in file_path and file_path.endswith(".mp3"):
                    main_audio_files.append(file_path)

        if not main_audio_files:
            raise HTTPException(
                status_code=404, detail="No downloadable audio file found"
            )

        # Use the most recent main audio file
        audio_file_path = main_audio_files[-1]

        # Get file data
        file_data = await storage_service.get_audio_file(audio_file_path)

        # Create safe filename
        safe_title = "".join(
            c for c in podcast.title if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        filename = f"{safe_title}.mp3"

        # Return file with download headers
        return Response(
            content=file_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(file_data)),
                "Cache-Control": "private, max-age=0",
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error downloading podcast: {str(e)}"
        )


@router.get("/{podcast_id}/share-info")
async def get_podcast_share_info(
    podcast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get sharing information for a podcast"""
    service = PodcastService(db)
    podcast = service.get_podcast_by_id(podcast_id, current_user.id)

    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")

    # For now, we'll generate sharing info even for non-completed podcasts
    share_url = f"https://voiceflow-studio.com/shared/podcast/{podcast_id}"

    return {
        "podcast_id": podcast_id,
        "title": podcast.title,
        "topic": podcast.topic,
        "duration": f"{podcast.length} min" if podcast.length else "Unknown",
        "share_url": share_url,
        "downloadable": podcast.status == PodcastStatus.COMPLETED and podcast.has_audio,
        "social_shares": {
            "twitter": f'https://twitter.com/intent/tweet?text=Check out this AI-generated podcast: "{podcast.title}" - {share_url}',
            "facebook": f"https://www.facebook.com/sharer/sharer.php?u={share_url}",
            "linkedin": f"https://www.linkedin.com/sharing/share-offsite/?url={share_url}",
            "reddit": f"https://www.reddit.com/submit?url={share_url}&title={podcast.title}",
        },
    }
