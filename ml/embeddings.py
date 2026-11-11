"""E5 embeddings: validated float32 unit vectors."""

import re

import numpy as np

MODEL = "intfloat/multilingual-e5-large"
DIM = 1024


class EmbeddingAdapter:
    def __init__(self, encoder, batch_size=32):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.encoder = encoder
        self.batch_size = batch_size

    def _encode(self, texts, prefix):
        if not texts:
            return np.empty((0, DIM), dtype=np.float32)
        vectors = np.asarray(
            self.encoder.encode(
                [prefix + text for text in texts],
                batch_size=self.batch_size,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )
        if vectors.shape != (len(texts), DIM) or not np.isfinite(vectors).all():
            raise ValueError("invalid embedding dimensions or values")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError("invalid embedding norm")
        return (vectors / norms[:, None]).astype(np.float32)

    def embed_passages(self, clip_id, chunks):
        if type(clip_id) is not int or clip_id < 1:
            raise ValueError("clip_id must be a positive integer")
        ids, texts = [], []
        for chunk in chunks:
            identifier = chunk.get("chunk_id")
            text = chunk.get("text")
            if not isinstance(identifier, str) or not re.fullmatch(
                rf"{clip_id}_[0-9]{{3}}", identifier
            ):
                raise ValueError("invalid chunk ID")
            if identifier in ids or not isinstance(text, str) or not text.strip():
                raise ValueError("duplicate ID or empty text")
            ids.append(identifier)
            texts.append(text)
        matrix = self._encode(texts, "passage: ")
        return matrix, {
            "clip_id": clip_id,
            "model": MODEL,
            "dim": DIM,
            "count": len(ids),
            "chunk_ids": ids,
        }

    def embed_queries(self, queries):
        if any(not isinstance(text, str) or not text.strip() for text in queries):
            raise ValueError("query must be nonempty text")
        return self._encode(queries, "query: ")
