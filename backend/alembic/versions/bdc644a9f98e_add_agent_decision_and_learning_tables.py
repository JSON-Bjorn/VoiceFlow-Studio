"""Add agent decision and learning tables

Revision ID: bdc644a9f98e
Revises: 64ab723f283e
Create Date: 2025-06-12 10:58:55.945318

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "bdc644a9f98e"
down_revision: Union[str, None] = "64ab723f283e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create agent_decision_history table
    op.create_table(
        "agent_decision_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_name", sa.String(), nullable=True),
        sa.Column("decision_type", sa.String(), nullable=True),
        sa.Column("context_data", sa.JSON(), nullable=True),
        sa.Column("decision_data", sa.JSON(), nullable=True),
        sa.Column("outcome_data", sa.JSON(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("actual_cost", sa.Float(), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_agent_decision_history_id"),
        "agent_decision_history",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_decision_history_agent_name"),
        "agent_decision_history",
        ["agent_name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_decision_history_decision_type"),
        "agent_decision_history",
        ["decision_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_decision_history_timestamp"),
        "agent_decision_history",
        ["timestamp"],
        unique=False,
    )

    # Create agent_learning_data table
    op.create_table(
        "agent_learning_data",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_name", sa.String(), nullable=True),
        sa.Column("learning_type", sa.String(), nullable=True),
        sa.Column("key", sa.String(), nullable=True),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("last_updated", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_agent_learning_data_id"), "agent_learning_data", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_agent_learning_data_agent_name"),
        "agent_learning_data",
        ["agent_name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_learning_data_learning_type"),
        "agent_learning_data",
        ["learning_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_learning_data_key"), "agent_learning_data", ["key"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop agent tables
    op.drop_index(op.f("ix_agent_learning_data_key"), table_name="agent_learning_data")
    op.drop_index(
        op.f("ix_agent_learning_data_learning_type"), table_name="agent_learning_data"
    )
    op.drop_index(
        op.f("ix_agent_learning_data_agent_name"), table_name="agent_learning_data"
    )
    op.drop_index(op.f("ix_agent_learning_data_id"), table_name="agent_learning_data")
    op.drop_table("agent_learning_data")

    op.drop_index(
        op.f("ix_agent_decision_history_timestamp"), table_name="agent_decision_history"
    )
    op.drop_index(
        op.f("ix_agent_decision_history_decision_type"),
        table_name="agent_decision_history",
    )
    op.drop_index(
        op.f("ix_agent_decision_history_agent_name"),
        table_name="agent_decision_history",
    )
    op.drop_index(
        op.f("ix_agent_decision_history_id"), table_name="agent_decision_history"
    )
    op.drop_table("agent_decision_history")
