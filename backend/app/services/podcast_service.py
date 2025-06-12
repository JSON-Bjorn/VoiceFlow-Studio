from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import Optional, List
from ..models.podcast import Podcast, PodcastStatus
from ..models.user import User
from ..schemas.podcast import PodcastCreate, PodcastUpdate, PodcastSummary
from ..services.credit_service import CreditService
import math
from datetime import datetime


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

        # Validate voice settings if provided
        voice_settings_dict = None
        if podcast_data.voice_settings:
            voice_settings_dict = self._validate_and_prepare_voice_settings(
                user_id, podcast_data.voice_settings
            )

        # Create the podcast
        podcast = Podcast(
            user_id=user_id,
            title=podcast_data.title,
            topic=podcast_data.topic,
            length=podcast_data.length,
            status=PodcastStatus.PENDING,
            voice_settings=voice_settings_dict,
        )

        self.db.add(podcast)
        self.db.flush()  # Get the podcast ID

        # Deduct credit
        self.credit_service.use_credits(
            user_id=user_id,
            amount=1,
            description=f"Created podcast: {podcast_data.title}",
        )

        self.db.commit()
        self.db.refresh(podcast)
        return podcast

    def _validate_and_prepare_voice_settings(
        self, user_id: int, voice_settings
    ) -> dict:
        """Validate voice settings and ensure user has access to selected voices"""
        from ..models.voice_profile import VoiceProfile

        settings_dict = voice_settings.model_dump()

        # If using custom voices, validate that the user owns the selected voices
        if settings_dict.get("use_custom_voices", False):
            voice_ids = [
                settings_dict.get("host1_voice_id"),
                settings_dict.get("host2_voice_id"),
            ]

            # Filter out None values
            voice_ids = [vid for vid in voice_ids if vid is not None]

            if voice_ids:
                # Check that all custom voice IDs belong to the user
                user_voices = (
                    self.db.query(VoiceProfile)
                    .filter(
                        VoiceProfile.user_id == user_id,
                        VoiceProfile.voice_id.in_(voice_ids),
                        VoiceProfile.is_active == True,
                    )
                    .all()
                )

                user_voice_ids = {voice.voice_id for voice in user_voices}

                # Validate each voice ID
                for voice_id in voice_ids:
                    if voice_id not in user_voice_ids:
                        raise ValueError(
                            f"Voice '{voice_id}' not found or not accessible"
                        )

        return settings_dict

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

    def update_podcast_status(
        self, podcast_id: int, status: str, user_id: Optional[int] = None
    ) -> bool:
        """Update podcast status (for internal use)"""
        query = self.db.query(Podcast).filter(Podcast.id == podcast_id)

        # If user_id is provided, ensure the podcast belongs to the user
        if user_id is not None:
            query = query.filter(Podcast.user_id == user_id)

        podcast = query.first()
        if not podcast:
            return False

        # Convert string status to enum if needed
        if isinstance(status, str):
            try:
                status_enum = PodcastStatus(status.lower())
                podcast.status = status_enum
            except ValueError:
                # If status string doesn't match enum, treat as raw string
                podcast.status = status
        else:
            podcast.status = status

        self.db.commit()
        return True

    def update_podcast_content(
        self, podcast_id: int, script_content: dict, user_id: Optional[int] = None
    ) -> bool:
        """
        Update podcast content with generated script and metadata

        Args:
            podcast_id: ID of the podcast to update
            script_content: Dictionary containing script data, title, segments, etc.
            user_id: Optional user ID for access validation

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            query = self.db.query(Podcast).filter(Podcast.id == podcast_id)

            # If user_id is provided, ensure the podcast belongs to the user
            if user_id is not None:
                query = query.filter(Podcast.user_id == user_id)

            podcast = query.first()
            if not podcast:
                return False

            # Update podcast with script content
            if script_content.get("title"):
                podcast.title = script_content["title"]

            # Convert segments to script text if available
            if script_content.get("segments"):
                segments = script_content["segments"]
                script_text = ""

                for segment in segments:
                    speaker = segment.get("speaker", "Speaker")
                    text = segment.get("text", "")
                    script_text += f"{speaker}: {text}\n\n"

                podcast.script = script_text.strip()

            # Update estimated duration if available
            if script_content.get("estimated_duration"):
                # Convert minutes to integer if needed
                duration = script_content["estimated_duration"]
                if isinstance(duration, (int, float)):
                    podcast.length = int(duration)

            # Store additional metadata in voice_settings field (reusing existing JSON field)
            if not podcast.voice_settings:
                podcast.voice_settings = {}

            podcast.voice_settings.update(
                {
                    "generation_metadata": script_content.get(
                        "generation_metadata", {}
                    ),
                    "script_metadata": script_content.get("script_metadata", {}),
                    "validation": script_content.get("validation", {}),
                    "last_updated": datetime.utcnow().isoformat(),
                }
            )

            self.db.commit()
            return True

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to update podcast content: {e}")
            self.db.rollback()
            return False
