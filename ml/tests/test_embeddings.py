"""Unit tests for the in-memory E5 adapter."""

import importlib.util

import numpy as np
import pytest


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, *, batch_size, normalize_embeddings):
        self.calls.append((list(texts), batch_size, normalize_embeddings))
        rows = np.zeros((len(texts), 1024), dtype=np.float32)
        for index, text in enumerate(texts):
            rows[index, 0] = len(text)
            rows[index, 1] = index + 1
        return rows


def test_adapter_module_exists():
    assert importlib.util.find_spec("ml.embeddings") is not None, "embedding adapter is missing"


def test_passages_keep_row_order_and_normalize():
    from ml.embeddings import EmbeddingAdapter

    encoder = FakeEncoder()
    adapter = EmbeddingAdapter(encoder, batch_size=2)
    chunks = [{"chunk_id": "7_002", "text": "первый"}, {"chunk_id": "7_000", "text": "второй!"}]
    matrix, metadata = adapter.embed_passages(7, chunks)
    assert encoder.calls == [(["passage: первый", "passage: второй!"], 2, False)]
    assert metadata == {
        "clip_id": 7,
        "model": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "count": 2,
        "chunk_ids": ["7_002", "7_000"],
    }
    assert matrix.shape == (2, 1024)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, rtol=1e-6)
    assert matrix[0, 1] < matrix[1, 1]


def test_queries_prefix_and_empty_input():
    from ml.embeddings import EmbeddingAdapter

    enc = FakeEncoder()
    adapter = EmbeddingAdapter(enc)
    assert adapter.embed_queries([]).shape == (0, 1024)
    assert not enc.calls
    assert adapter.embed_queries(["тест"]).shape == (1, 1024)
    assert enc.calls[0][0] == ["query: тест"]
    assert adapter.embed_passages(3, [])[0].shape == (0, 1024)


@pytest.mark.parametrize(
    "output", [np.zeros((1, 1024)), np.ones((1, 3)), np.full((1, 1024), np.nan)]
)
def test_rejects_bad_vectors(output):
    from ml.embeddings import EmbeddingAdapter

    class InvalidEncoder:
        def encode(self, *args, **kwargs):
            return output

    with pytest.raises(ValueError):
        EmbeddingAdapter(InvalidEncoder()).embed_queries(["query"])


def test_rejects_duplicate_ids_and_empty_query():
    from ml.embeddings import EmbeddingAdapter

    adapter = EmbeddingAdapter(FakeEncoder())
    with pytest.raises(ValueError):
        adapter.embed_passages(1, [{"chunk_id": "1_001", "text": "x"}] * 2)
    with pytest.raises(ValueError):
        adapter.embed_queries([" "])
