from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import Optional, List
from ..models.podcast import Podcast, PodcastStatus
from ..models.user import User
from ..schemas.podcast import PodcastCreate, PodcastUpdate, PodcastSummary
from ..services.credit_service import CreditService
from ..models.credit_transaction import TransactionType
import math


class PodcastService:
    def __init__(self, db: Session):
        self.db = db
        self.credit_service = CreditService(db)

    def create_podcast(self, user_id: int, podcast_data: PodcastCreate) -> Podcast:
        """Create a new podcast for a user"""
        # Check if user has enough credits (1 credit per podcast)
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        if user.credits < 1:
            raise ValueError("Insufficient credits to create podcast")

        # Create the podcast
        podcast = Podcast(
            user_id=user_id,
            title=podcast_data.title,
            topic=podcast_data.topic,
            length=podcast_data.length,
            status=PodcastStatus.PENDING,
        )

        self.db.add(podcast)
        self.db.flush()  # Get the podcast ID

        # Deduct credit
        self.credit_service.use_credits(
            user_id=user_id,
            amount=1,
            description=f"Created podcast: {podcast_data.title}",
            transaction_type=TransactionType.USAGE,
        )

        self.db.commit()
        self.db.refresh(podcast)
        return podcast

    def get_user_podcasts(
        self,
        user_id: int,
        page: int = 1,
        per_page: int = 10,
        status_filter: Optional[PodcastStatus] = None,
    ) -> tuple[List[Podcast], int]:
        """Get paginated podcasts for a user with optional status filter"""
        query = self.db.query(Podcast).filter(Podcast.user_id == user_id)

        if status_filter:
            query = query.filter(Podcast.status == status_filter)

        # Get total count
        total = query.count()

        # Apply pagination and ordering
        podcasts = (
            query.order_by(desc(Podcast.created_at))
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )

        return podcasts, total

    def get_podcast_by_id(self, podcast_id: int, user_id: int) -> Optional[Podcast]:
        """Get a specific podcast by ID, ensuring it belongs to the user"""
        return (
            self.db.query(Podcast)
            .filter(Podcast.id == podcast_id, Podcast.user_id == user_id)
            .first()
        )

    def update_podcast(
        self, podcast_id: int, user_id: int, update_data: PodcastUpdate
    ) -> Optional[Podcast]:
        """Update a podcast"""
        podcast = self.get_podcast_by_id(podcast_id, user_id)
        if not podcast:
            return None

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(podcast, field, value)

        self.db.commit()
        self.db.refresh(podcast)
        return podcast

    def delete_podcast(self, podcast_id: int, user_id: int) -> bool:
        """Delete a podcast"""
        podcast = self.get_podcast_by_id(podcast_id, user_id)
        if not podcast:
            return False

        self.db.delete(podcast)
        self.db.commit()
        return True

    def get_podcast_summary(self, user_id: int) -> PodcastSummary:
        """Get summary statistics for user's podcasts"""
        base_query = self.db.query(Podcast).filter(Podcast.user_id == user_id)

        total_podcasts = base_query.count()
        completed_podcasts = base_query.filter(
            Podcast.status == PodcastStatus.COMPLETED
        ).count()
        pending_podcasts = base_query.filter(
            Podcast.status == PodcastStatus.PENDING
        ).count()
        failed_podcasts = base_query.filter(
            Podcast.status == PodcastStatus.FAILED
        ).count()

        return PodcastSummary(
            total_podcasts=total_podcasts,
            completed_podcasts=completed_podcasts,
            pending_podcasts=pending_podcasts,
            failed_podcasts=failed_podcasts,
        )

    def simulate_podcast_generation(
        self, podcast_id: int, user_id: int
    ) -> Optional[Podcast]:
        """Simulate podcast generation process (for testing)"""
        podcast = self.get_podcast_by_id(podcast_id, user_id)
        if not podcast:
            return None

        # Update status to generating
        podcast.status = PodcastStatus.GENERATING
        self.db.commit()

        # Simulate completion with dummy data
        podcast.status = PodcastStatus.COMPLETED
        podcast.audio_url = f"https://example.com/audio/podcast_{podcast_id}.mp3"
        podcast.script = f"This is a simulated script for the podcast '{podcast.title}' about {podcast.topic}. The podcast would be approximately {podcast.length} minutes long."

        self.db.commit()
        self.db.refresh(podcast)
        return podcast

    def get_recent_podcasts(self, user_id: int, limit: int = 5) -> List[Podcast]:
        """Get the most recent podcasts for a user"""
        return (
            self.db.query(Podcast)
            .filter(Podcast.user_id == user_id)
            .order_by(desc(Podcast.created_at))
            .limit(limit)
            .all()
        )
