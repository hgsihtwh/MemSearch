"""Deterministic MEM-76 speech chunking; no embedding or persistence.

``chunk_transcript(transcript, caption=None)`` returns the chunks-schema object.
The 15-second windows advance by 7 seconds and select whole overlapping ASR
segments. Consequently chunk timecodes may extend outside the window. Duplicate
texts are deduplicated when the same ASR segments occupy consecutive windows.
Unlike MEM-76, matching caption and speech are kept as separate sources, and
repeated phrases at different timestamps are retained. Caption cleanup removes
hashtags and ``Clip by @...``, normalizes
whitespace, then truncates to 1000 characters. Text is stored without an E5
prefix; add ``passage: `` only when encoding the text.

Inputs must match the transcript schema. Non-finite timestamps, reversed or
out-of-order segments and segment ends beyond the duration are rejected rather
than repaired. More than 1000 chunks cannot be represented by the current
three-digit ID schema, so the function raises ``ValueError`` in that case.
This module is a pure in-memory API; it does not read or write files or a DB.
"""

import math
import re

WINDOW = 15
STRIDE = 7


def _clean_caption(caption):
    text = re.sub(r"#\S+", " ", caption or "")
    text = re.sub(r"^Clip by @\S+$", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()[:1000]


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validate_transcript(transcript):
    if not isinstance(transcript, dict) or set(transcript) != {
        "clip_id",
        "model",
        "language",
        "duration",
        "segments",
    }:
        raise ValueError("transcript must match the transcript schema")
    clip_id = transcript["clip_id"]
    duration = transcript["duration"]
    if type(clip_id) is not int or clip_id < 1:
        raise ValueError("clip_id must be a positive integer")
    if not isinstance(transcript["model"], str) or not isinstance(transcript["language"], str):
        raise ValueError("model and language must be strings")
    if not _number(duration) or not isinstance(transcript["segments"], list):
        raise ValueError("invalid duration or segments")
    previous_start = previous_end = 0
    for segment in transcript["segments"]:
        if not isinstance(segment, dict) or set(segment) != {"start", "end", "text"}:
            raise ValueError("segment must match the transcript schema")
        start, end, text = segment["start"], segment["end"], segment["text"]
        if (
            not _number(start)
            or not _number(end)
            or start >= end
            or end > duration
            or start < previous_start
            or end < previous_end
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError("invalid or out-of-order segment")
        previous_start = start
        previous_end = end


def chunk_transcript(transcript, caption=None):
    """Build caption and speech chunks in their embedding-row order."""
    _validate_transcript(transcript)
    if caption is not None and not isinstance(caption, str):
        raise ValueError("caption must be a string or None")
    clip_id = transcript["clip_id"]
    chunks = []

    def add(source, start, end, text):
        if len(chunks) >= 1000:
            raise ValueError("chunk schema permits at most 1000 chunks")
        chunks.append(
            {
                "chunk_id": f"{clip_id}_{len(chunks):03d}",
                "source": source,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    caption = _clean_caption(caption)
    if caption:
        add("caption", None, None, caption)
    segments = transcript["segments"]
    previous_indices = None
    if segments:
        t = 0
        while True:
            indices = tuple(
                i for i, s in enumerate(segments) if s["end"] > t and s["start"] < t + WINDOW
            )
            part = [segments[i] for i in indices]
            text = " ".join(s["text"] for s in part)
            if part and indices != previous_indices:
                add("speech", part[0]["start"], part[-1]["end"], text)
                previous_indices = indices
            if t + WINDOW >= segments[-1]["end"]:
                break
            t += STRIDE
    return {"clip_id": clip_id, "chunking": {"window": WINDOW, "stride": STRIDE}, "chunks": chunks}
