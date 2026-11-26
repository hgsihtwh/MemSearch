import asyncio
import copy
import json

import numpy as np
import pytest

from ml.embeddings import DIM, EmbeddingAdapter
from ml.index_collection import index_collection, load_transcripts, prepare_rows


class Encoder:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        vectors = np.zeros((len(texts), DIM), dtype=np.float32)
        for i in range(len(texts)):
            vectors[i, i] = 2
        return vectors


def test_prepare_rows_uses_caption_then_speech_and_real_adapter():
    encoder = Encoder()
    rows = prepare_rows({1: transcript()}, {1: "caption"}, EmbeddingAdapter(encoder))
    assert [row["source"] for row in rows] == ["caption", "speech"]
    assert encoder.seen == ["passage: caption", "passage: hello"]
    assert [row["ordinal"] for row in rows] == [0, 1]
    assert np.allclose([np.linalg.norm(row["embedding"]) for row in rows], 1)
    assert rows[0]["embedding"][0] == 1 and rows[1]["embedding"][1] == 1


def test_empty_clip_is_rejected_instead_of_silent_skip():
    item = transcript()
    item["segments"] = []
    with pytest.raises(ValueError, match="no chunks"):
        prepare_rows({1: item}, {1: None}, EmbeddingAdapter(Encoder()))


def transcript(clip_id=1):
    return {
        "clip_id": clip_id,
        "model": "large-v3",
        "language": "ru",
        "duration": 3,
        "segments": [{"start": 0, "end": 2, "text": "hello"}],
    }


def test_loads_padded_transcript_files_without_altering_them(tmp_path):
    path = tmp_path / "01.json"
    original = json.dumps(transcript(), ensure_ascii=False)
    path.write_text(original, encoding="utf-8")
    result = load_transcripts(tmp_path)
    assert result == {1: transcript()}
    assert path.read_text(encoding="utf-8") == original


def test_rejects_mismatched_id_and_invalid_timestamps(tmp_path):
    item = transcript(2)
    (tmp_path / "01.json").write_text(json.dumps(item), encoding="utf-8")
    with pytest.raises(ValueError, match="ID"):
        load_transcripts(tmp_path)


class MemoryConnection:
    def __init__(self):
        self.rows = [{"clip_id": 10, "ordinal": 0, "source": "ocr"}]
        self.fail_insert = False

    def transaction(self):
        connection = self

        class Transaction:
            async def __aenter__(self):
                self.snapshot = copy.deepcopy(connection.rows)

            async def __aexit__(self, typ, value, traceback):
                if typ:
                    connection.rows = self.snapshot

        return Transaction()

    async def fetch(self, sql, *args):
        if "FROM clips" in sql:
            return [{"id": 10, "title": "Title", "caption": "caption"}]
        if "FROM chunks" in sql:
            return [
                r
                for r in self.rows
                if r["source"] not in ("caption", "speech")
                or (r["source"] == "speech" and r["clip_id"] in args[1])
            ]
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        if "DELETE" in sql:
            self.rows = [
                r
                for r in self.rows
                if not (
                    (r["clip_id"] in args[0] and r["source"] in ("caption", "speech"))
                    or (r["clip_id"] in args[1] and r["source"] == "caption")
                )
            ]
        else:
            raise AssertionError(sql)

    async def executemany(self, sql, args):
        if self.fail_insert:
            raise RuntimeError("insert failed")
        self.rows.extend(
            {"clip_id": r[0], "ordinal": r[1], "source": r[2], "text": r[5]} for r in args
        )


def test_index_is_atomic_and_retries_preserve_other_sources():
    connection = MemoryConnection()
    source = {1: transcript()}
    kwargs = dict(
        connection=connection,
        collection_id="collection",
        transcripts=source,
        clip_ids={1: 10},
        adapter=EmbeddingAdapter(Encoder()),
    )
    asyncio.run(index_collection(**kwargs))
    assert [r["source"] for r in connection.rows] == ["ocr", "caption", "speech"]
    assert connection.rows[1]["text"] == "Title caption"
    first = copy.deepcopy(connection.rows)
    asyncio.run(index_collection(**kwargs))
    assert connection.rows == first
    connection.fail_insert = True
    with pytest.raises(RuntimeError, match="insert failed"):
        asyncio.run(index_collection(**kwargs))
    assert connection.rows == first


def test_rejects_corrupt_embeddings_before_replacement():
    class BadAdapter:
        def embed_passages(self, clip_id, chunks):
            return (
                np.full((len(chunks), DIM), np.nan),
                {
                    "model": "intfloat/multilingual-e5-large",
                    "dim": DIM,
                    "clip_id": clip_id,
                    "count": len(chunks),
                    "chunk_ids": [c["chunk_id"] for c in chunks],
                },
            )

    connection = MemoryConnection()
    first = copy.deepcopy(connection.rows)
    with pytest.raises(ValueError, match="embedding"):
        asyncio.run(
            index_collection(connection, "collection", {1: transcript()}, {1: 10}, BadAdapter())
        )
    assert connection.rows == first


def test_missing_transcript_can_use_caption_without_inventing_speech():
    connection = MemoryConnection()
    result = asyncio.run(
        index_collection(connection, "collection", {1: None}, {1: 10}, EmbeddingAdapter(Encoder()))
    )
    assert result == {"clips": 1, "chunks": 1}
    assert [r["source"] for r in connection.rows] == ["ocr", "caption"]


def test_caption_only_does_not_erase_existing_speech():
    connection = MemoryConnection()
    connection.rows.append({"clip_id": 10, "ordinal": 3, "source": "speech", "text": "old"})
    result = asyncio.run(
        index_collection(connection, "collection", {1: None}, {1: 10}, EmbeddingAdapter(Encoder()))
    )
    assert result["chunks"] == 1
    assert any(r["source"] == "speech" and r["text"] == "old" for r in connection.rows)


def test_rejects_duplicate_json_keys_in_transcript(tmp_path):
    (tmp_path / "01.json").write_text(
        '{"clip_id": 1, "clip_id": 1, "model": "x", "language": "ru", '
        '"duration": 1, "segments": []}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_transcripts(tmp_path)


def test_rejects_clip_outside_collection_before_deleting():
    class MissingClip(MemoryConnection):
        async def fetch(self, sql, *args):
            if "FROM clips" in sql:
                return []
            return await super().fetch(sql, *args)

    connection = MissingClip()
    first = copy.deepcopy(connection.rows)
    with pytest.raises(ValueError, match="selected collection"):
        asyncio.run(
            index_collection(
                connection, "other", {1: transcript()}, {1: 10}, EmbeddingAdapter(Encoder())
            )
        )
    assert connection.rows == first


def test_rejects_transcript_with_time_outside_duration_before_deleting():
    connection = MemoryConnection()
    first = copy.deepcopy(connection.rows)
    item = transcript()
    item["segments"][0]["end"] = 4
    with pytest.raises(ValueError, match="segment"):
        asyncio.run(
            index_collection(
                connection, "collection", {1: item}, {1: 10}, EmbeddingAdapter(Encoder())
            )
        )
    assert connection.rows == first
