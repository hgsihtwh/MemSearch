"""Добавлены ручные подборки, метки и исключения автоматических тем.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-26 14:14:33.398858
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint("uq_clips_id_collection", "clips", ["id", "collection_id"])
    op.create_unique_constraint("uq_tags_id_collection", "tags", ["id", "collection_id"])
    op.add_column(
        "topics",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.execute("UPDATE topics SET created_at = computed_at")
    op.create_unique_constraint("uq_topics_id_collection", "topics", ["id", "collection_id"])
    op.create_table(
        "playlists",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=60), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("name ~ '[^[:space:]]'", name="ck_playlists_nonempty_name"),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "collection_id", name="uq_playlists_id_collection"),
    )
    op.create_index(
        "ix_playlists_collection_created",
        "playlists",
        ["collection_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "playlist_tags",
        sa.Column("playlist_id", sa.UUID(), nullable=False),
        sa.Column("tag_id", sa.Integer(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["playlist_id", "collection_id"],
            ["playlists.id", "playlists.collection_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tag_id", "collection_id"], ["tags.id", "tags.collection_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("playlist_id", "tag_id"),
    )
    op.create_index("ix_playlist_tags_tag_id", "playlist_tags", ["tag_id"], unique=False)
    op.create_table(
        "topic_tags",
        sa.Column("topic_id", sa.Integer(), nullable=False),
        sa.Column("tag_id", sa.Integer(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["tag_id", "collection_id"], ["tags.id", "tags.collection_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["topic_id", "collection_id"], ["topics.id", "topics.collection_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("topic_id", "tag_id"),
    )
    op.create_index("ix_topic_tags_tag_id", "topic_tags", ["tag_id"], unique=False)
    op.create_table(
        "playlist_clips",
        sa.Column("playlist_id", sa.UUID(), nullable=False),
        sa.Column("clip_id", sa.BigInteger(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column(
            "added_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["clip_id", "collection_id"], ["clips.id", "clips.collection_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["playlist_id", "collection_id"],
            ["playlists.id", "playlists.collection_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("playlist_id", "clip_id"),
    )
    op.create_index("ix_playlist_clips_clip_id", "playlist_clips", ["clip_id"], unique=False)
    op.create_table(
        "topic_exclusions",
        sa.Column("topic_id", sa.Integer(), nullable=False),
        sa.Column("clip_id", sa.BigInteger(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["clip_id", "collection_id"], ["clips.id", "clips.collection_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["topic_id", "collection_id"], ["topics.id", "topics.collection_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("topic_id", "clip_id"),
    )
    op.create_index("ix_topic_exclusions_clip_id", "topic_exclusions", ["clip_id"], unique=False)


def downgrade() -> None:
    # Откат не должен незаметно уничтожать новые подборки, метки или исключения.
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM playlists)
                OR EXISTS (SELECT 1 FROM playlist_clips)
                OR EXISTS (SELECT 1 FROM playlist_tags)
                OR EXISTS (SELECT 1 FROM topic_tags)
                OR EXISTS (SELECT 1 FROM topic_exclusions) THEN
                RAISE EXCEPTION 'Cannot downgrade 0003 while playlist or topic-link data exists';
            END IF;
        END $$
    """)
    op.drop_index("ix_topic_exclusions_clip_id", table_name="topic_exclusions")
    op.drop_table("topic_exclusions")
    op.drop_index("ix_playlist_clips_clip_id", table_name="playlist_clips")
    op.drop_table("playlist_clips")
    op.drop_index("ix_topic_tags_tag_id", table_name="topic_tags")
    op.drop_table("topic_tags")
    op.drop_index("ix_playlist_tags_tag_id", table_name="playlist_tags")
    op.drop_table("playlist_tags")
    op.drop_index("ix_playlists_collection_created", table_name="playlists")
    op.drop_table("playlists")
    op.drop_constraint("uq_topics_id_collection", "topics", type_="unique")
    op.drop_column("topics", "created_at")
    op.drop_constraint("uq_tags_id_collection", "tags", type_="unique")
    op.drop_constraint("uq_clips_id_collection", "clips", type_="unique")
