"""Add audio storage fields to podcast

Revision ID: d1b2c3e4f5a6
Revises: 793ea163bbaa
Create Date: 2024-12-29 10:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "d1b2c3e4f5a6"
down_revision = "793ea163bbaa"
branch_labels = None
depends_on = None


def upgrade():
    # Add audio storage fields to podcasts table
    op.add_column("podcasts", sa.Column("has_audio", sa.Boolean(), default=False))
    op.add_column(
        "podcasts", sa.Column("audio_segments_count", sa.Integer(), default=0)
    )
    op.add_column(
        "podcasts", sa.Column("audio_total_duration", sa.Integer(), nullable=True)
    )
    op.add_column(
        "podcasts", sa.Column("voice_generation_cost", sa.String(), nullable=True)
    )
    op.add_column("podcasts", sa.Column("audio_file_paths", sa.JSON(), nullable=True))


def downgrade():
    # Remove audio storage fields from podcasts table
    op.drop_column("podcasts", "audio_file_paths")
    op.drop_column("podcasts", "voice_generation_cost")
    op.drop_column("podcasts", "audio_total_duration")
    op.drop_column("podcasts", "audio_segments_count")
    op.drop_column("podcasts", "has_audio")
