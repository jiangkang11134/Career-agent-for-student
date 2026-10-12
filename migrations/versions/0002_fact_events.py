"""Add tenant-scoped append-only fact events."""

import sqlalchemy as sa
from alembic import op

revision = "0002_fact_events"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fact_events",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("fact_id", sa.String(64), nullable=False, unique=True),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("source_turn_id", sa.String(128), nullable=False),
        sa.Column("supersedes_fact_id", sa.String(64), nullable=False),
        sa.Column("reason", sa.String(255), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index(
        "ix_fact_events_tenant_session_created",
        "fact_events",
        ["tenant_id", "session_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_fact_events_tenant_session_created", table_name="fact_events")
    op.drop_table("fact_events")
