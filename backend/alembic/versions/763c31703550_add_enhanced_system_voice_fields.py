"""Add enhanced system voice fields

Revision ID: 763c31703550
Revises: 9b1e7434b103
Create Date: 2025-06-08 08:49:49.416977

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "763c31703550"
down_revision: Union[str, None] = "9b1e7434b103"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # For SQLite, we need to recreate the table to modify NOT NULL constraints

    # Create new table with updated schema
    op.create_table(
        "voice_profiles_new",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "user_id", sa.Integer(), nullable=True
        ),  # Now nullable for system voices
        sa.Column("voice_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "original_audio_path", sa.String(), nullable=True
        ),  # Now nullable for synthetic voices
        sa.Column("test_audio_path", sa.String(), nullable=True),
        sa.Column("gender", sa.String(), nullable=True),
        sa.Column("style", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        # New enhanced system voice fields
        sa.Column("accent", sa.String(), nullable=True),
        sa.Column("voice_type", sa.String(), nullable=True),
        sa.Column("generation_seed", sa.Integer(), nullable=True),
        sa.Column("is_system_default", sa.Boolean(), nullable=True),
        # Optimization parameters
        sa.Column("optimal_similarity_boost", sa.Float(), nullable=True),
        sa.Column("optimal_stability", sa.Float(), nullable=True),
        sa.Column("optimal_style", sa.Float(), nullable=True),
        sa.Column("optimal_exaggeration", sa.Float(), nullable=True),
        sa.Column("optimal_cfg_weight", sa.Float(), nullable=True),
        sa.Column("optimal_temperature", sa.Float(), nullable=True),
        # Metadata
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("original_filename", sa.String(), nullable=True),
        sa.Column("content_type", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Copy data from old table to new table
    op.execute("""
        INSERT INTO voice_profiles_new (
            id, user_id, voice_id, name, description, original_audio_path, test_audio_path,
            gender, style, is_active, optimal_similarity_boost, optimal_stability,
            optimal_style, optimal_exaggeration, optimal_cfg_weight, optimal_temperature,
            file_size, duration, original_filename, content_type, created_at, updated_at
        )
        SELECT 
            id, user_id, voice_id, name, description, original_audio_path, test_audio_path,
            gender, style, is_active, optimal_similarity_boost, optimal_stability,
            optimal_style, optimal_exaggeration, optimal_cfg_weight, optimal_temperature,
            file_size, duration, original_filename, content_type, created_at, updated_at
        FROM voice_profiles
    """)

    # Create indexes
    op.create_index(
        op.f("ix_voice_profiles_new_id"), "voice_profiles_new", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_voice_profiles_new_user_id"),
        "voice_profiles_new",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_voice_profiles_new_voice_id"),
        "voice_profiles_new",
        ["voice_id"],
        unique=True,
    )

    # Drop old table and rename new table
    op.drop_table("voice_profiles")
    op.rename_table("voice_profiles_new", "voice_profiles")


def downgrade() -> None:
    """Downgrade schema."""
    # Create old table structure
    op.create_table(
        "voice_profiles_old",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),  # Back to NOT NULL
        sa.Column("voice_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "original_audio_path", sa.String(), nullable=False
        ),  # Back to NOT NULL
        sa.Column("test_audio_path", sa.String(), nullable=True),
        sa.Column("gender", sa.String(), nullable=True),
        sa.Column("style", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        # Optimization parameters
        sa.Column("optimal_similarity_boost", sa.Float(), nullable=True),
        sa.Column("optimal_stability", sa.Float(), nullable=True),
        sa.Column("optimal_style", sa.Float(), nullable=True),
        sa.Column("optimal_exaggeration", sa.Float(), nullable=True),
        sa.Column("optimal_cfg_weight", sa.Float(), nullable=True),
        sa.Column("optimal_temperature", sa.Float(), nullable=True),
        # Metadata
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("original_filename", sa.String(), nullable=True),
        sa.Column("content_type", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Copy data back (only records that have user_id and original_audio_path)
    op.execute("""
        INSERT INTO voice_profiles_old (
            id, user_id, voice_id, name, description, original_audio_path, test_audio_path,
            gender, style, is_active, optimal_similarity_boost, optimal_stability,
            optimal_style, optimal_exaggeration, optimal_cfg_weight, optimal_temperature,
            file_size, duration, original_filename, content_type, created_at, updated_at
        )
        SELECT 
            id, user_id, voice_id, name, description, original_audio_path, test_audio_path,
            gender, style, is_active, optimal_similarity_boost, optimal_stability,
            optimal_style, optimal_exaggeration, optimal_cfg_weight, optimal_temperature,
            file_size, duration, original_filename, content_type, created_at, updated_at
        FROM voice_profiles
        WHERE user_id IS NOT NULL AND original_audio_path IS NOT NULL
    """)

    # Create indexes
    op.create_index(
        op.f("ix_voice_profiles_old_id"), "voice_profiles_old", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_voice_profiles_old_user_id"),
        "voice_profiles_old",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_voice_profiles_old_voice_id"),
        "voice_profiles_old",
        ["voice_id"],
        unique=True,
    )

    # Drop current table and rename old table
    op.drop_table("voice_profiles")
    op.rename_table("voice_profiles_old", "voice_profiles")
