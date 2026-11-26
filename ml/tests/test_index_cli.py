import asyncio
import csv
import hashlib
import json

import pytest


def transcript(clip_id=1, end=1):
    return {
        "clip_id": clip_id,
        "model": "asr",
        "language": "ru",
        "duration": 2,
        "segments": [{"start": 0, "end": end, "text": "hello"}],
    }


def test_reports_invalid_missing_and_unmatched_transcripts(tmp_path):
    from ml.index_cli import inspect_transcripts

    (tmp_path / "01.json").write_text(json.dumps(transcript(end=3)), encoding="utf-8")
    (tmp_path / "03.json").write_text(json.dumps(transcript(3)), encoding="utf-8")
    sources, report = inspect_transcripts(tmp_path, {1, 2})
    assert sources == {}
    assert report["missing"] == [2]
    assert report["unmatched"] == [3]
    assert "segment" in report["invalid"]["1"]
    assert json.loads((tmp_path / "01.json").read_text())["segments"][0]["end"] == 3


def test_mapping_requires_collection_and_exact_identity():
    from ml.index_cli import resolve_mapping

    class Connection:
        async def fetchval(self, sql, *args):
            return True

        async def fetch(self, sql, *args):
            assert "collection_id = $1" in sql
            return [{"id": 40, "platform": "tiktok", "external_id": "123", "url": "u"}]

    passport = {
        1: {"platform": "tiktok", "external_id": "123", "url": "u"},
        2: {"platform": "vk", "external_id": "-1_2", "url": "v"},
    }
    mapping, report = asyncio.run(resolve_mapping(Connection(), "collection", passport))
    assert mapping == {1: 40}
    assert report["unmatched_passport"] == [2]
    assert report["unmatched_database"] == []


def test_safe_index_requires_confirmation_of_mapping_and_rechecks_under_lock():
    from ml.index_cli import index_selected, mapping_digest

    class Connection:
        def transaction(self):
            class Tx:
                async def __aenter__(self):
                    pass

                async def __aexit__(self, *args):
                    pass

            return Tx()

        async def fetchval(self, *args):
            return True

        async def fetch(self, sql, *args):
            assert "FOR UPDATE" in sql
            return [{"id": 40, "platform": "tiktok", "external_id": "other", "url": "u"}]

    passport = {1: {"platform": "tiktok", "external_id": "123", "url": "u"}}
    mapping = {1: 40}
    digest = mapping_digest("collection", mapping, passport)
    assert len(digest) == hashlib.sha256().digest_size * 2
    with pytest.raises(ValueError, match="confirmation"):
        asyncio.run(
            index_selected(
                Connection(),
                "collection",
                passport,
                mapping,
                {1: transcript()},
                "wrong",
                lambda: None,
            )
        )
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(
            index_selected(
                Connection(),
                "collection",
                passport,
                mapping,
                {1: transcript()},
                digest,
                lambda: None,
            )
        )


def test_search_filters_collection_returns_url_time_and_measures_encode():
    from ml.index_cli import search_collection

    class Adapter:
        def embed_queries(self, queries):
            import numpy as np

            assert queries == ["cats"]
            v = np.zeros((1, 1024), dtype="float32")
            v[0, 0] = 1
            return v

    class Connection:
        async def fetch(self, sql, collection, vector, limit):
            assert "collection_id = $1" in sql
            assert "<=>" in sql and limit == 5 and collection == "collection"
            assert vector.startswith("[1.0,")
            return [
                {
                    "url": "https://example.test",
                    "start": 1.2,
                    "end": 3.0,
                    "text": "cats",
                    "distance": 0.1,
                    "clip_id": 1,
                    "source": "speech",
                }
            ]

    result = asyncio.run(search_collection(Connection(), "collection", "cats", Adapter(), 5))
    assert result["results"][0]["url"] == "https://example.test"
    assert result["results"][0]["start"] == 1.2
    assert result["timing_ms"]["end_to_end"] >= result["timing_ms"]["query_encode"]


def test_cli_refuses_dsn_missing_and_help_does_not_load_model(monkeypatch, capsys):
    from ml.index_cli import main

    monkeypatch.delenv("MEMSEARCH_DSN", raising=False)
    assert (
        main(["search", "--collection", "d09b87bd-39bb-46df-a997-dced7d84b788", "--query", "cats"])
        != 0
    )
    assert "MEMSEARCH_DSN" in capsys.readouterr().err


def test_index_success_delegates_only_confirmed_clips(monkeypatch):
    from ml import index_cli

    captured = {}

    class Connection:
        def transaction(self):
            class Tx:
                async def __aenter__(self):
                    pass

                async def __aexit__(self, *args):
                    pass

            return Tx()

        async def fetch(self, sql, collection, ids):
            return [{"id": 40, "platform": "tiktok", "external_id": "123", "url": "u"}]

    async def index(connection, collection, transcripts, mapping, adapter):
        captured.update(
            collection=collection, transcripts=transcripts, mapping=mapping, adapter=adapter
        )
        return {"clips": 1, "chunks": 2}

    monkeypatch.setattr(index_cli, "index_collection", index)
    passport = {1: {"platform": "tiktok", "external_id": "123", "url": "u"}}
    mapping = {1: 40}
    digest = index_cli.mapping_digest("collection", mapping, passport)
    out = asyncio.run(
        index_cli.index_selected(
            Connection(),
            "collection",
            passport,
            mapping,
            {1: transcript()},
            digest,
            lambda: "adapter",
        )
    )
    assert out == {"clips": 1, "chunks": 2}
    assert captured == {
        "collection": "collection",
        "transcripts": {1: transcript()},
        "mapping": {1: 40},
        "adapter": "adapter",
    }


def test_real_transcript_quality_is_reported_without_mutation():
    from pathlib import Path

    from ml.index_cli import inspect_transcripts, read_passport

    root = Path(__file__).resolve().parents[2]
    passport = read_passport(root / "research/collection/passport.csv")
    valid, report = inspect_transcripts(
        root / "research/embeddings_check/results/transcripts", set(passport)
    )
    assert len(valid) == 74
    assert len(report["invalid"]) == 14
    assert len(report["missing"]) == 3
    assert not report["unmatched"]


def test_passport_validates_ids_and_url(tmp_path):
    from ml.index_cli import read_passport

    path = tmp_path / "passport.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["№", "ID", "Платформа", "Ссылка"])
        writer.writerow(["1", "123", "TikTok", "https://www.tiktok.com/@a/video/123"])
    assert read_passport(path) == {
        1: {
            "platform": "tiktok",
            "external_id": "123",
            "url": "https://www.tiktok.com/@a/video/123",
        }
    }
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["2", "999", "TikTok", "https://www.tiktok.com/@a/video/123"])
    with pytest.raises(ValueError, match="URL"):
        read_passport(path)
