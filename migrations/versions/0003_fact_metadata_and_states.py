"""Add structured fact metadata used by the long-term memory state machine."""

import sqlalchemy as sa
from alembic import op

revision = "0003_fact_metadata"
down_revision = "0002_fact_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("fact_events", sa.Column("fact_key", sa.String(128), nullable=False, server_default=""))
    op.add_column("fact_events", sa.Column("confidence", sa.Float(), nullable=False, server_default="1"))
    op.add_column("fact_events", sa.Column("actor_type", sa.String(32), nullable=False, server_default="user"))
    op.add_column("fact_events", sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"))


def downgrade() -> None:
    op.drop_column("fact_events", "metadata_json")
    op.drop_column("fact_events", "actor_type")
    op.drop_column("fact_events", "confidence")
    op.drop_column("fact_events", "fact_key")

