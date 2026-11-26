"""Build a collection index from existing transcripts; never modify source files."""

import json
import re
from pathlib import Path

import numpy as np

from ml.chunking import chunk_transcript
from ml.embeddings import DIM, MODEL


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_transcripts(directory):
    """Load numbered JSON transcripts, rejecting ambiguous IDs and invalid times."""
    directory = Path(directory)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError("no transcripts found")
    transcripts = {}
    for path in paths:
        if path.is_symlink() or not re.fullmatch(r"[0-9]+", path.stem):
            raise ValueError(f"invalid transcript path: {path}")
        clip_id = int(path.stem)
        if clip_id in transcripts or clip_id < 1:
            raise ValueError(f"duplicate or invalid ID: {path}")
        try:
            item = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"invalid transcript {path}: {exc}") from exc
        if type(item) is not dict or item.get("clip_id") != clip_id:
            raise ValueError(f"transcript ID mismatch: {path}")
        try:
            chunk_transcript(item)
        except ValueError as exc:
            raise ValueError(f"invalid transcript {path}: {exc}") from exc
        transcripts[clip_id] = item
    return transcripts


def prepare_rows(transcripts, captions, adapter):
    """Embed chunks in deterministic ID order, preserving vector alignment."""
    rows = []
    if set(transcripts) != set(captions):
        raise ValueError("captions and transcripts have different IDs")
    for clip_id, transcript in sorted(transcripts.items()):
        if transcript is None:
            transcript = {
                "clip_id": clip_id,
                "model": "",
                "language": "",
                "duration": 0,
                "segments": [],
            }
        if not isinstance(transcript, dict) or transcript.get("clip_id") != clip_id:
            raise ValueError("transcript ID mismatch")
        chunks = chunk_transcript(transcript, captions[clip_id])["chunks"]
        if not chunks:
            raise ValueError(f"clip {clip_id} has no chunks")
        matrix, metadata = adapter.embed_passages(clip_id, chunks)
        if (
            metadata.get("chunk_ids") != [chunk["chunk_id"] for chunk in chunks]
            or metadata.get("clip_id") != clip_id
            or metadata.get("model") != MODEL
            or metadata.get("dim") != DIM
            or metadata.get("count") != len(chunks)
        ):
            raise ValueError("embedding metadata or order mismatch")
        matrix = np.asarray(matrix, dtype=np.float32)
        if (
            matrix.shape != (len(chunks), DIM)
            or not np.isfinite(matrix).all()
            or not np.allclose(
                np.linalg.norm(matrix.astype(np.float64), axis=1), 1, rtol=1e-5, atol=1e-5
            )
        ):
            raise ValueError("invalid embedding shape, values or normalization")
        for ordinal, (chunk, vector) in enumerate(zip(chunks, matrix, strict=True)):
            rows.append(
                {
                    "clip_id": clip_id,
                    "ordinal": ordinal,
                    "source": chunk["source"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                    "embedding": vector,
                }
            )
    return rows


async def index_collection(connection, collection_id, transcripts, clip_ids, adapter):
    """Replace caption/speech chunks for explicit collection-owned clips atomically.

    ``clip_ids`` maps transcript IDs to existing database clip IDs. A transcript
    value of None means caption-only and preserves any existing speech chunks.
    The connection must be an asyncpg connection to the migrated schema; no
    schema or clips are created here. Other chunk sources and the source
    transcript files are untouched. E5 is supplied via an EmbeddingAdapter; the
    adapter's encoder must actually be intfloat/multilingual-e5-large. Query
    results must join chunks.clip_id to clips.url to return the original link.
    All selected clips are locked in ID order through the transaction. Callers
    writing chunks concurrently must also lock their parent clip first.
    """
    if not transcripts or set(transcripts) != set(clip_ids):
        raise ValueError("transcripts and clip mapping must contain the same nonempty IDs")
    if any(type(i) is not int or i < 1 for i in clip_ids.values()):
        raise ValueError("database clip IDs must be positive integers")
    if len(set(clip_ids.values())) != len(clip_ids):
        raise ValueError("database clip IDs must be unique")

    async with connection.transaction():
        selected = await connection.fetch(
            "SELECT id, title, caption FROM clips "
            "WHERE collection_id = $1 AND id = ANY($2::bigint[]) ORDER BY id FOR UPDATE",
            collection_id,
            sorted(clip_ids.values()),
        )
        clips = {record["id"]: record for record in selected}
        if set(clips) != set(clip_ids.values()):
            raise ValueError("clip not found in the selected collection")
        captions = {}
        for source_id, db_id in clip_ids.items():
            clip = clips[db_id]
            parts = [
                part.strip() for part in (clip["title"], clip["caption"]) if part and part.strip()
            ]
            captions[source_id] = " ".join(dict.fromkeys(parts))
        rows = prepare_rows(transcripts, captions, adapter)
        full_ids = sorted(
            clip_ids[source_id]
            for source_id, transcript in transcripts.items()
            if transcript is not None
        )
        caption_only_ids = sorted(set(clip_ids.values()) - set(full_ids))
        reserved = await connection.fetch(
            "SELECT clip_id, ordinal, source FROM chunks "
            "WHERE clip_id = ANY($1::bigint[]) AND "
            "(source NOT IN ('caption', 'speech') OR "
            "(source = 'speech' AND clip_id = ANY($2::bigint[])))",
            sorted(clip_ids.values()),
            caption_only_ids,
        )
        used = {db_id: set() for db_id in clip_ids.values()}
        for record in reserved:
            used[record["clip_id"]].add(record["ordinal"])
        next_ordinal = dict.fromkeys(used, 0)
        payload = []
        for row in rows:
            db_id = clip_ids[row["clip_id"]]
            ordinal = next_ordinal[db_id]
            while ordinal in used[db_id]:
                ordinal += 1
            next_ordinal[db_id] = ordinal + 1
            vector = "[" + ",".join(str(float(value)) for value in row["embedding"]) + "]"
            payload.append(
                (db_id, ordinal, row["source"], row["start"], row["end"], row["text"], vector)
            )
        await connection.execute(
            "DELETE FROM chunks WHERE (clip_id = ANY($1::bigint[]) "
            "AND source IN ('caption', 'speech')) OR "
            "(clip_id = ANY($2::bigint[]) AND source = 'caption')",
            full_ids,
            caption_only_ids,
        )
        await connection.executemany(
            'INSERT INTO chunks (clip_id, ordinal, source, start, "end", text, embedding, tsv) '
            "VALUES ($1, $2, $3, $4, $5, $6, $7::vector, to_tsvector('russian', $6))",
            payload,
        )
    return {"clips": len(clips), "chunks": len(payload)}
