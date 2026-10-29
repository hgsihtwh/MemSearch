"""Начальная схема по ER-модели (MEM-26, MEM-28)

Revision ID: 0001
Revises:
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "collections",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("onboarding_done_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "topics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "collection_id",
            sa.Uuid(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("name_source", sa.String(8), nullable=False),
        sa.Column("keywords", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column(
            "computed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("name_source IN ('auto', 'user')"),
    )
    op.create_index("ix_topics_collection_id", "topics", ["collection_id"])

    op.create_table(
        "clips",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "collection_id",
            sa.Uuid(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("platform", sa.String(8), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("author", sa.Text()),
        sa.Column("title", sa.Text()),
        sa.Column("caption", sa.Text()),
        sa.Column("duration", sa.Float()),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("thumbnail_key", sa.Text()),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("topic_id", sa.Integer(), sa.ForeignKey("topics.id", ondelete="SET NULL")),
        sa.Column("summary", sa.Text()),
        sa.Column(
            "added_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("collection_id", "platform", "external_id"),
        sa.CheckConstraint("platform IN ('tiktok', 'vk')"),
        sa.CheckConstraint(
            "status IN ('queued', 'downloading', 'transcribing', 'indexing', 'enriching', "
            "'ready', 'failed')"
        ),
    )
    op.create_index("ix_clips_collection_added", "clips", ["collection_id", "added_at"])
    op.create_index("ix_clips_topic_id", "clips", ["topic_id"])

    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "clip_id",
            sa.BigInteger(),
            sa.ForeignKey("clips.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("start", sa.Float(), nullable=False),
        sa.Column("end", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source", sa.String(10), nullable=False),
        sa.CheckConstraint("source IN ('asr', 'subtitles')"),
    )
    op.create_index("ix_transcript_segments_clip_id", "transcript_segments", ["clip_id"])

    op.create_table(
        "frames",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "clip_id",
            sa.BigInteger(),
            sa.ForeignKey("clips.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("t", sa.Float(), nullable=False),
        sa.Column("ocr_text", sa.Text()),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(512)),
    )
    op.create_index("ix_frames_clip_id", "frames", ["clip_id"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "clip_id",
            sa.BigInteger(),
            sa.ForeignKey("clips.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(8), nullable=False),
        sa.Column("start", sa.Float()),
        sa.Column("end", sa.Float()),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(1024)),
        sa.Column("tsv", postgresql.TSVECTOR()),
        sa.UniqueConstraint("clip_id", "ordinal"),
        sa.CheckConstraint("source IN ('caption', 'speech', 'ocr', 'scene')"),
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.create_index(
        "ix_chunks_text_trgm",
        "chunks",
        ["text"],
        postgresql_using="gin",
        postgresql_ops={"text": "gin_trgm_ops"},
    )

    op.create_table(
        "tags",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "collection_id",
            sa.Uuid(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.UniqueConstraint("collection_id", "name"),
    )
    op.create_table(
        "clip_tags",
        sa.Column(
            "clip_id",
            sa.BigInteger(),
            sa.ForeignKey("clips.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "tag_id", sa.Integer(), sa.ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("source", sa.String(8), nullable=False),
        sa.CheckConstraint("source IN ('auto', 'user')"),
    )

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "clip_id",
            sa.BigInteger(),
            sa.ForeignKey("clips.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(8), nullable=False),
        sa.Column("queue", sa.String(16), nullable=False),
        sa.Column("status", sa.String(8), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("stage IN ('fast', 'full', 'enrich')"),
        sa.CheckConstraint("status IN ('queued', 'running', 'done', 'failed')"),
    )
    op.create_index("ix_processing_jobs_clip_id", "processing_jobs", ["clip_id"])
    op.create_index("ix_processing_jobs_status", "processing_jobs", ["status"])

    op.create_table(
        "import_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "collection_id",
            sa.Uuid(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("platform", sa.String(8), nullable=False),
        sa.Column("scope", sa.String(8), nullable=False),
        sa.Column("total", sa.Integer()),
        sa.Column("sent", sa.Integer(), nullable=False),
        sa.Column("cursor", sa.Text()),
        sa.Column("finished", sa.Boolean(), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("platform IN ('tiktok', 'vk')"),
        sa.CheckConstraint("scope IN ('last200', 'year', 'all')"),
    )
    op.create_index("ix_import_sessions_collection_id", "import_sessions", ["collection_id"])

    op.create_table(
        "telegram_links",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "collection_id",
            sa.Uuid(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("telegram_user_id", sa.BigInteger(), unique=True),
        sa.Column("link_code", sa.String(32), unique=True),
        sa.Column("code_expires_at", sa.DateTime(timezone=True)),
        sa.Column("linked_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_telegram_links_collection_id", "telegram_links", ["collection_id"])

    op.create_table(
        "search_feedback",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "collection_id",
            sa.Uuid(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query_id", sa.Uuid(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("clip_id", sa.BigInteger(), sa.ForeignKey("clips.id", ondelete="SET NULL")),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("relevant", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_search_feedback_collection_id", "search_feedback", ["collection_id"])


def downgrade() -> None:
    for table in (
        "search_feedback",
        "telegram_links",
        "import_sessions",
        "processing_jobs",
        "clip_tags",
        "tags",
        "chunks",
        "frames",
        "transcript_segments",
        "clips",
        "topics",
        "collections",
    ):
        op.drop_table(table)
