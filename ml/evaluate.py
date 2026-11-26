"""Оценка качества семантического поиска по контрольным запросам MEM-17 (MEM-39).

Каждый запрос векторизуется моделью e5, в pgvector ищутся 100 ближайших фрагментов
коллекции (``index_cli.search_collection``), ролик оценивается лучшим фрагментом.
Считаются Recall@1, Recall@10 (доля запросов с правильным роликом в первой десятке)
и MRR; запросы без ответа исключаются. Отчёт — JSON с метриками, временем и
ранжированием каждого запроса.

    MEMSEARCH_DSN=postgresql://... python -m ml.evaluate --collection UUID \\
        --out research/search_eval/report.json [--model ПУТЬ_ИЛИ_ИМЯ]
"""

import argparse
import asyncio
import importlib
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

from ml.embeddings import MODEL, EmbeddingAdapter
from ml.index_cli import read_passport, search_collection
from ml.lexical import calculate_metrics, load_queries

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = 100


def rank_clips(rows, to_passport):
    """Collapse nearest chunks into clips, keeping the best chunk of each clip."""
    ranking = []
    for row in rows:
        clip = to_passport[row["clip_id"]]
        if clip not in ranking:
            ranking.append(clip)
    return ranking


def _stats(values):
    values = sorted(values)
    return {
        "median": round(statistics.median(values), 1),
        "p95": round(values[max(0, int(0.95 * len(values)) - 1)], 1),
        "max": round(values[-1], 1),
    }


async def evaluate(connection, collection_id, adapter, root=ROOT):
    passport = read_passport(root / "research/collection/passport.csv")
    queries = load_queries(root, set(passport))
    identity = {(v["platform"], v["external_id"]): k for k, v in passport.items()}
    rows = await connection.fetch(
        "SELECT id, platform, external_id FROM clips WHERE collection_id = $1", collection_id
    )
    to_passport = {
        r["id"]: identity[(r["platform"], r["external_id"])]
        for r in rows
        if (r["platform"], r["external_id"]) in identity
    }
    await search_collection(connection, collection_id, "прогрев", adapter, 10)
    results, timings = [], []
    for query in queries:
        found = await search_collection(
            connection, collection_id, query["query"], adapter, CANDIDATES
        )
        timings.append(found["timing_ms"]["end_to_end"])
        results.append({**query, "ranking": rank_clips(found["results"], to_passport)[:10]})
    metrics = calculate_metrics(results)
    answerable = [r for r in results if r["answers"]]
    metrics["recall_at_1"] = sum(
        bool(r["ranking"]) and r["ranking"][0] in r["answers"] for r in answerable
    ) / len(answerable)
    missed = [
        {"id": r["id"], "query": r["query"], "answers": r["answers"], "top3": r["ranking"][:3]}
        for r in answerable
        if not set(r["ranking"]) & set(r["answers"])
    ]
    return {
        "metrics": metrics,
        "search_ms": _stats(timings),
        "missed_at_10": missed,
        "queries": results,
    }


async def _main(args, dsn):
    import asyncpg

    sentence_transformers = importlib.import_module("sentence_transformers")
    started = time.perf_counter()
    adapter = EmbeddingAdapter(sentence_transformers.SentenceTransformer(args.model))
    load_ms = (time.perf_counter() - started) * 1000
    connection = await asyncpg.connect(dsn)
    try:
        report = await evaluate(connection, args.collection, adapter)
    finally:
        await connection.close()
    report["method"] = {
        "model": MODEL,
        "model_source": args.model,
        "candidates": CANDIDATES,
        "clip_score": "best chunk",
        "model_load_ms": round(load_ms),
        "collection_id": str(args.collection),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collection", type=uuid.UUID, required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--model", default=MODEL, help="e5 model name or local directory")
    args = parser.parse_args(argv)
    dsn = os.environ.get("MEMSEARCH_DSN")
    if not dsn:
        parser.exit(2, "error: MEMSEARCH_DSN must be set explicitly\n")
    report = asyncio.run(_main(args, dsn))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    m = report["metrics"]
    print(
        f"Recall@1 {m['recall_at_1']:.3f}  Recall@10 {m['recall_at_10']:.3f}  MRR {m['mrr']:.3f}"
        f"  поиск медиана {report['search_ms']['median']} мс  → {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
