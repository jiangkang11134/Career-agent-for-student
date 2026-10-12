"""add tenant-scoped model connection configs"""

from alembic import op
import sqlalchemy as sa


revision = "0004_tenant_model_configs"
down_revision = "0003_fact_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenant_model_configs",
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("agent_role", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("base_url", sa.String(length=1024), nullable=False),
        sa.Column("api_key", sa.String(length=1024), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("tenant_id", "agent_role"),
    )


def downgrade() -> None:
    op.drop_table("tenant_model_configs")
