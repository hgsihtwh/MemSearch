"""Tests for transcript chunking and its data-format contract."""

import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs" / "data-format" / "examples" / "77"


def test_real_example_matches_documented_chunks():
    assert importlib.util.find_spec("ml.chunking") is not None, "chunking module is missing"
    from ml.chunking import chunk_transcript

    transcript = json.loads((EXAMPLE / "transcript.json").read_text(encoding="utf-8"))
    expected = json.loads((EXAMPLE / "chunks.json").read_text(encoding="utf-8"))
    result = chunk_transcript(transcript)
    assert result == expected
    assert any(chunk["end"] - chunk["start"] > 15 for chunk in result["chunks"])


def test_caption_cleanup_and_empty_segments():
    from ml.chunking import chunk_transcript

    transcript = {"clip_id": 7, "model": "x", "language": "ru", "duration": 0, "segments": []}
    result = chunk_transcript(transcript, "  Привет #тег\n мир  ")
    assert result["chunks"] == [
        {"chunk_id": "7_000", "source": "caption", "start": None, "end": None, "text": "Привет мир"}
    ]
    assert chunk_transcript(transcript, "#only_tag")["chunks"] == []
    assert chunk_transcript(transcript, "Clip by @name")["chunks"] == []
    assert chunk_transcript(transcript, "a" * 1002)["chunks"][0]["text"] == "a" * 1000


def make_transcript(segments, duration=60):
    return {
        "clip_id": 3,
        "model": "asr",
        "language": "ru",
        "duration": duration,
        "segments": segments,
    }


def test_windows_overlap_boundaries_deduplication_and_silence():
    from ml.chunking import chunk_transcript

    segments = [
        {"start": 0, "end": 7, "text": "А"},
        {"start": 7, "end": 15, "text": "Б"},
        {"start": 15, "end": 29, "text": "В"},
        {"start": 50, "end": 51, "text": "Г"},
    ]
    transcript = make_transcript(segments)
    saved = deepcopy(transcript)
    result = chunk_transcript(transcript, "caption")
    assert result == chunk_transcript(transcript, "caption")
    assert transcript == saved
    assert [(c["source"], c["start"], c["end"], c["text"]) for c in result["chunks"]] == [
        ("caption", None, None, "caption"),
        ("speech", 0, 15, "А Б"),
        ("speech", 7, 29, "Б В"),
        ("speech", 15, 29, "В"),
        ("speech", 50, 51, "Г"),
    ]
    assert [c["chunk_id"] for c in result["chunks"]] == [f"3_{i:03d}" for i in range(5)]


def test_duplicate_caption_and_speech_preserves_both_sources():
    from ml.chunking import chunk_transcript

    transcript = make_transcript([{"start": 0, "end": 2, "text": "same"}])
    assert [c["source"] for c in chunk_transcript(transcript, "same")["chunks"]] == [
        "caption",
        "speech",
    ]


def test_repeated_speech_after_pause_is_not_deduplicated():
    from ml.chunking import chunk_transcript

    transcript = make_transcript(
        [{"start": 0, "end": 1, "text": "same"}, {"start": 40, "end": 41, "text": "same"}]
    )
    chunks = chunk_transcript(transcript)["chunks"]
    assert [(c["start"], c["end"]) for c in chunks] == [(0, 1), (40, 41)]


@pytest.mark.parametrize("clip_id", [0, -1, True, 1.5, "1"])
def test_reject_invalid_clip_id(clip_id):
    from ml.chunking import chunk_transcript

    transcript = make_transcript([])
    transcript["clip_id"] = clip_id
    with pytest.raises(ValueError):
        chunk_transcript(transcript)


@pytest.mark.parametrize(
    "segments",
    [
        [{"start": 3, "end": 2, "text": "invalid"}],
        [{"start": -1, "end": 2, "text": "invalid"}],
        [{"start": 1, "end": math.nan, "text": "invalid"}],
        [{"start": 2, "end": 3, "text": "a"}, {"start": 1, "end": 2, "text": "b"}],
        [{"start": 0, "end": 2, "text": ""}],
        [{"start": 0, "end": 2, "text": "a", "extra": 1}],
        [{"start": 0, "end": 61, "text": "late"}],
    ],
)
def test_reject_invalid_segments_without_repairing_them(segments):
    from ml.chunking import chunk_transcript

    with pytest.raises(ValueError):
        chunk_transcript(make_transcript(segments))


@pytest.mark.parametrize("timestamp", [0, 15])
def test_reject_zero_duration_segments_without_losing_them(timestamp):
    from ml.chunking import chunk_transcript

    transcript = make_transcript([{"start": timestamp, "end": timestamp, "text": "speech"}])
    with pytest.raises(ValueError):
        chunk_transcript(transcript)


def test_reject_decreasing_ends_even_when_starts_increase():
    from ml.chunking import chunk_transcript

    transcript = make_transcript(
        [
            {"start": 0, "end": 40, "text": "long"},
            {"start": 1, "end": 2, "text": "short"},
        ]
    )
    with pytest.raises(ValueError):
        chunk_transcript(transcript)


def test_reject_invalid_transcript_shape_and_caption():
    from ml.chunking import chunk_transcript

    for change in (
        {"duration": -1},
        {"duration": math.inf},
        {"model": None},
        {"language": 42},
        {"segments": None},
        {"unexpected": "field"},
    ):
        with pytest.raises(ValueError):
            chunk_transcript(make_transcript([]) | change)
    for key in ("clip_id", "model", "language", "duration", "segments"):
        document = make_transcript([])
        del document[key]
        with pytest.raises(ValueError):
            chunk_transcript(document)
    with pytest.raises(ValueError):
        chunk_transcript(make_transcript([]), 42)


def test_reject_more_than_one_thousand_chunks():
    from ml.chunking import chunk_transcript

    segments = [{"start": i * 7, "end": i * 7 + 1, "text": str(i)} for i in range(1003)]
    with pytest.raises(ValueError, match="1000"):
        chunk_transcript(make_transcript(segments, duration=segments[-1]["end"]))
