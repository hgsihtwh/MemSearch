"""SQLAlchemy-модели по ER-модели docs/architecture.md (раздел 7)."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Размерность векторов фрагментов: multilingual-e5-large (решение 14, MEM-76)
CHUNK_EMBEDDING_DIM = 1024
# Векторы кадров: модель «картинка — текст» (версия 1.0, MEM-87)
FRAME_EMBEDDING_DIM = 512

CLIP_STATUSES = (
    "queued",
    "downloading",
    "transcribing",
    "indexing",
    "enriching",
    "ready",
    "failed",
)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class Collection(Base):
    """Коллекция роликов; вместо учётной записи — ключ доступа (хранится только хеш)."""

    __tablename__ = "collections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    onboarding_done_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    clips: Mapped[list["Clip"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan", passive_deletes=True
    )


class Topic(Base):
    __tablename__ = "topics"
    __table_args__ = (CheckConstraint(_in("name_source", ("auto", "user"))),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    name_source: Mapped[str] = mapped_column(String(8), default="auto")
    keywords: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Clip(Base):
    __tablename__ = "clips"
    __table_args__ = (
        UniqueConstraint("collection_id", "platform", "external_id"),
        CheckConstraint(_in("platform", ("tiktok", "vk"))),
        CheckConstraint(_in("status", CLIP_STATUSES)),
        Index("ix_clips_collection_added", "collection_id", "added_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE")
    )
    platform: Mapped[str] = mapped_column(String(8))
    external_id: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    duration: Mapped[float | None] = mapped_column(Float)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    thumbnail_key: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL"), index=True
    )
    summary: Mapped[str | None] = mapped_column(Text)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    collection: Mapped[Collection] = relationship(back_populates="clips")
    segments: Mapped[list["TranscriptSegment"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True, order_by="TranscriptSegment.start"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True, order_by="Chunk.ordinal"
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (CheckConstraint(_in("source", ("asr", "subtitles"))),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    clip_id: Mapped[int] = mapped_column(ForeignKey("clips.id", ondelete="CASCADE"), index=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(10), default="asr")


class Frame(Base):
    """Ключевой кадр (версия 1.0): хранятся только текст в кадре и вектор, сам кадр — нет."""

    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    clip_id: Mapped[int] = mapped_column(ForeignKey("clips.id", ondelete="CASCADE"), index=True)
    t: Mapped[float] = mapped_column(Float)
    ocr_text: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(FRAME_EMBEDDING_DIM))


class Chunk(Base):
    """Фрагмент поиска: подпись, окно речи, текст в кадре или описание сцены."""

    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint(_in("source", ("caption", "speech", "ocr", "scene"))),
        UniqueConstraint("clip_id", "ordinal"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_chunks_text_trgm",
            "text",
            postgresql_using="gin",
            postgresql_ops={"text": "gin_trgm_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    clip_id: Mapped[int] = mapped_column(ForeignKey("clips.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(8))
    start: Mapped[float | None] = mapped_column(Float)
    end: Mapped[float | None] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(CHUNK_EMBEDDING_DIM))
    tsv: Mapped[str | None] = mapped_column(TSVECTOR)


class Tag(Base):
    __tablename__ = "tags"
    __table_args__ = (UniqueConstraint("collection_id", "name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(Text)


class ClipTag(Base):
    __tablename__ = "clip_tags"
    __table_args__ = (CheckConstraint(_in("source", ("auto", "user"))),)

    clip_id: Mapped[int] = mapped_column(
        ForeignKey("clips.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)
    source: Mapped[str] = mapped_column(String(8), default="auto")


class ProcessingJob(Base):
    """Задача обработки ролика; состояние не зависит от Celery (ADR-001)."""

    __tablename__ = "processing_jobs"
    __table_args__ = (
        CheckConstraint(_in("stage", ("fast", "full", "enrich"))),
        CheckConstraint(_in("status", ("queued", "running", "done", "failed"))),
        Index("ix_processing_jobs_status", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    clip_id: Mapped[int] = mapped_column(ForeignKey("clips.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(8))
    queue: Mapped[str] = mapped_column(String(16), default="fast")
    status: Mapped[str] = mapped_column(String(8), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ImportSession(Base):
    """Перенос «Избранного» расширением с возможностью продолжить (ФТ-1.6)."""

    __tablename__ = "import_sessions"
    __table_args__ = (
        CheckConstraint(_in("platform", ("tiktok", "vk"))),
        CheckConstraint(_in("scope", ("last200", "year", "all"))),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    platform: Mapped[str] = mapped_column(String(8))
    scope: Mapped[str] = mapped_column(String(8))
    total: Mapped[int | None] = mapped_column(Integer)
    sent: Mapped[int] = mapped_column(Integer, default=0)
    cursor: Mapped[str | None] = mapped_column(Text)
    finished: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class TelegramLink(Base):
    __tablename__ = "telegram_links"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    telegram_user_id: Mapped[int | None] = mapped_column(BigInteger, unique=True)
    link_code: Mapped[str | None] = mapped_column(String(32), unique=True)
    code_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    linked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SearchFeedback(Base):
    """Оценка результата поиска «подходит / не подходит» (ФТ-3.8)."""

    __tablename__ = "search_feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    query_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    query: Mapped[str] = mapped_column(Text)
    clip_id: Mapped[int | None] = mapped_column(ForeignKey("clips.id", ondelete="SET NULL"))
    position: Mapped[int] = mapped_column(Integer)
    relevant: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
