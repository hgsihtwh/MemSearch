"""Добавлены прогресс обработки, доступность оригинала и дата обновления ролика.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("clips", sa.Column("processing_percent", sa.Integer(), nullable=True))
    op.add_column(
        "clips",
        sa.Column("original_status", sa.String(16), server_default="unknown", nullable=False),
    )
    op.add_column(
        "clips", sa.Column("original_checked_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("clips", sa.Column("original_reason", sa.String(32), nullable=True))
    op.add_column(
        "clips",
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    # Дата последнего изменения раньше не хранилась; сохраняем известную дату добавления.
    op.execute("UPDATE clips SET updated_at = added_at")
    op.create_check_constraint(
        "ck_clips_processing_percent", "clips", "processing_percent BETWEEN 0 AND 100"
    )
    op.create_check_constraint(
        "ck_clips_original_status",
        "clips",
        "original_status IN ('available', 'unavailable', 'unknown')",
    )
    op.create_check_constraint(
        "ck_clips_original_reason",
        "clips",
        "original_reason IN ('removed', 'private', 'region_blocked', 'embed_blocked', 'network_error')",
    )
    op.create_index("ix_clips_collection_updated", "clips", ["collection_id", "updated_at", "id"])


def downgrade() -> None:
    # Поля можно убрать, пока они не содержат новых измерений или истории обновлений.
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM clips
                WHERE processing_percent IS NOT NULL
                    OR original_status <> 'unknown'
                    OR original_checked_at IS NOT NULL
                    OR original_reason IS NOT NULL
                    OR updated_at IS DISTINCT FROM added_at
            ) THEN
                RAISE EXCEPTION 'Cannot downgrade 0004 while clip metadata exists';
            END IF;
        END $$
    """)
    op.drop_index("ix_clips_collection_updated", table_name="clips")
    op.drop_constraint("ck_clips_original_reason", "clips", type_="check")
    op.drop_constraint("ck_clips_original_status", "clips", type_="check")
    op.drop_constraint("ck_clips_processing_percent", "clips", type_="check")
    for column in (
        "updated_at",
        "original_reason",
        "original_checked_at",
        "original_status",
        "processing_percent",
    ):
        op.drop_column("clips", column)
