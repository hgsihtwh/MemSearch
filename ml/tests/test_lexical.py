import asyncio
import csv
import os
from pathlib import Path

import pytest

from ml import lexical


def test_metrics_use_first_relevant_result_and_full_rank():
    queries = [
        {"answers": [2, 3], "ranking": [8, 2, 3]},
        {"answers": [7], "ranking": list(range(10, 20)) + [7]},
        {"answers": [42], "ranking": []},
        {"answers": [], "ranking": [1]},
    ]
    result = lexical.calculate_metrics(queries)
    assert result["evaluated"] == 3
    assert result["without_answer"] == 1
    assert result["recall_at_10"] == pytest.approx(1 / 3)
    assert result["mrr"] == pytest.approx((1 / 2 + 1 / 11) / 3)


def test_load_real_collection_and_control_queries():
    root = Path(__file__).resolve().parents[2]
    docs, missing = lexical.load_collection(root)
    queries = lexical.load_queries(root, {d["id"] for d in docs})
    assert len(docs) == 91
    assert missing == [30, 55, 56]
    assert len(queries) == 110
    assert sum(bool(q["answers"]) for q in queries) == 107
    assert any("Therac" in d["caption"] for d in docs if d["id"] == 1)
    assert next(d for d in docs if d["id"] == 1)["transcript"]


def test_invalid_query_answer_reference_rejected(tmp_path):
    root = tmp_path
    path = root / "research/queries/control_queries.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["№", "Запрос", "Правильные ролики (№ по паспорту)"])
        writer.writeheader()
        writer.writerow({"№": "1", "Запрос": "тест", "Правильные ролики (№ по паспорту)": "999"})
    with pytest.raises(ValueError, match="unknown"):
        lexical.load_queries(root, {1})


def test_integration_rejects_locale_without_russian_case_folding():
    dsn = os.getenv("MEM80_BAD_LOCALE_DSN")
    if not dsn:
        pytest.skip("set MEM80_BAD_LOCALE_DSN to an isolated database with C locale")

    async def run():
        import asyncpg

        conn = await asyncpg.connect(dsn, timeout=5)
        try:
            with pytest.raises(ValueError, match="case folding"):
                await lexical.evaluate(conn, [], [])
        finally:
            await conn.close()

    asyncio.run(run())


def test_integration_real_data_report():
    dsn = os.getenv("MEM80_TEST_DSN")
    if not dsn:
        pytest.skip("set MEM80_TEST_DSN to a PostgreSQL database with TEMP privilege")
    report = asyncio.run(lexical._main(dsn, Path(__file__).resolve().parents[2]))
    assert report["collection"]["clips"] == 91
    assert report["metrics"]["evaluated"] == 107
    assert len(report["queries"]) == 110
    assert report["method"]["lc_ctype"]
    assert report["mem76"]["multilingual-e5-large"]["recall_at_10"] == pytest.approx(0.963)
    assert report["mem76"]["multilingual-e5-large"]["mrr"] == pytest.approx(0.912)


def test_integration_fts_is_temporary_and_searches_russian():
    dsn = os.getenv("MEM80_TEST_DSN")
    if not dsn:
        pytest.skip("set MEM80_TEST_DSN to a PostgreSQL database with TEMP privilege")

    async def run():
        import asyncpg

        conn = await asyncpg.connect(dsn, timeout=5)
        try:
            docs = [
                {
                    "id": 1,
                    "title": "",
                    "caption": "Красная кошка",
                    "transcript": "Прыгает по столу",
                },
                {"id": 2, "title": "Собаки", "caption": "", "transcript": "кошка спит"},
                {"id": 3, "title": "", "caption": "", "transcript": ""},
            ]
            async with conn.transaction():
                await lexical.build_index(conn, docs)
                assert await conn.fetchval("SELECT count(*) FROM pg_temp.mem80_docs") == 3
                assert (
                    await conn.fetchval(
                        "SELECT count(*) FROM pg_indexes WHERE schemaname LIKE 'pg_temp_%' AND tablename = 'mem80_docs' AND indexdef LIKE '%USING gin%'"
                    )
                    == 1
                )
                assert set(await lexical.search(conn, "кошки")) == {1, 2}
                assert await lexical.search(conn, "несуществующееслово") == []
                assert await lexical.search(conn, "собаку") == [2]
                assert set(await lexical.search(conn, "кошки неизвестное")) == {1, 2}
                assert await lexical.search(conn, "и в на") == []
            assert await conn.fetchval("SELECT to_regclass('pg_temp.mem80_docs') IS NULL")
            report = await lexical.evaluate(
                conn,
                docs,
                [
                    {"id": 1, "query": "собаку", "answers": [2]},
                    {"id": 2, "query": "несуществующееслово", "answers": [1]},
                    {"id": 3, "query": "кошки", "answers": []},
                ],
            )
            assert report["metrics"]["evaluated"] == 2
            assert report["metrics"]["without_answer"] == 1
            assert report["metrics"]["recall_at_10"] == pytest.approx(0.5)
            assert report["metrics"]["mrr"] == pytest.approx(0.5)
            assert report["queries"][0]["ranking"] == [2]
            assert await conn.fetchval("SELECT to_regclass('pg_temp.mem80_docs') IS NULL")
        finally:
            await conn.close()

    asyncio.run(run())
