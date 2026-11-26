"""PostgreSQL lexical baseline for the MEM-17 control queries."""

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path


def load_collection(root):
    """Read the exact MEM-76 passport captions and saved ASR transcripts.

    The passport combines title/description in one field. Preserve that field
    as caption, rather than substituting fuller downloaded metadata, so the
    compared baselines use the same source text.
    """
    root = Path(root)
    path = root / "research/collection/passport.csv"
    documents = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            clip_id = int(row["№"])
            if clip_id < 1 or clip_id in documents:
                raise ValueError(f"duplicate or invalid clip ID: {clip_id}")
            documents[clip_id] = {
                "id": clip_id,
                "title": "",
                "caption": row["Название / подпись"] or "",
                "transcript": "",
            }
    if not documents:
        raise ValueError("no clips in passport")
    found = set()
    directory = root / "research/embeddings_check/results/transcripts"
    if not directory.is_dir():
        raise ValueError(f"missing transcripts directory: {directory}")
    for path in directory.glob("*.json"):
        try:
            clip_id = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"invalid transcript filename: {path}") from exc
        if clip_id not in documents or clip_id in found:
            raise ValueError(f"unknown or duplicate transcript ID: {clip_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["clip_id"] != clip_id or not isinstance(data["segments"], list):
            raise ValueError(f"invalid transcript: {path}")
        texts = [segment["text"] for segment in data["segments"]]
        if any(not isinstance(text, str) for text in texts):
            raise ValueError(f"invalid transcript text: {path}")
        documents[clip_id]["transcript"] = " ".join(texts)
        found.add(clip_id)
    return [documents[k] for k in sorted(documents)], sorted(set(documents) - found)


def load_queries(root, clip_ids):
    """Read MEM-17, rejecting malformed and out-of-collection answers."""
    path = Path(root) / "research/queries/control_queries.csv"
    queries = []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            number = int(row["№"])
            if number < 1 or number in seen or not row["Запрос"].strip():
                raise ValueError(f"invalid query {number}")
            seen.add(number)
            answer = row["Правильные ролики (№ по паспорту)"].strip()
            answers = [int(value.strip()) for value in answer.split(",")] if answer else []
            unknown = set(answers) - set(clip_ids)
            if unknown:
                raise ValueError(f"unknown answer IDs: {sorted(unknown)}")
            queries.append({"id": number, "query": row["Запрос"], "answers": answers})
    if not queries:
        raise ValueError("no control queries")
    return queries


async def build_index(conn, documents):
    """Create a transaction-scoped, GIN-indexed document corpus in pg_temp.

    Must run inside an explicit transaction: ON COMMIT DROP protects existing
    databases and ensures repeatable, isolated evaluation.
    """
    if not conn.is_in_transaction():
        raise ValueError("build_index requires an explicit transaction")
    await conn.execute("""
        CREATE TEMP TABLE mem80_docs (
            id integer PRIMARY KEY,
            search_vector tsvector NOT NULL
        ) ON COMMIT DROP
    """)
    await conn.executemany(
        """
        INSERT INTO pg_temp.mem80_docs (id, search_vector)
        VALUES ($1, to_tsvector('russian', $2 || ' ' || $3 || ' ' || $4))
    """,
        [(doc["id"], doc["title"], doc["caption"], doc["transcript"]) for doc in documents],
    )
    await conn.execute("CREATE INDEX ON pg_temp.mem80_docs USING GIN (search_vector)")
    await conn.execute("ANALYZE pg_temp.mem80_docs")


async def search(conn, query):
    """Rank matching clips by ts_rank, OR-ing non-stopword Russian lexemes.

    A parameterized query is used, with clip ID as deterministic tie-breaker.
    Empty and stopword-only queries match nothing.
    """
    return [
        record["id"]
        for record in await conn.fetch(
            """
        WITH parsed AS (
            SELECT to_tsquery('russian',
                array_to_string(tsvector_to_array(to_tsvector('russian', $1)), ' | ')
            ) AS query
        )
        SELECT d.id FROM pg_temp.mem80_docs AS d CROSS JOIN parsed AS p
        WHERE d.search_vector @@ p.query
        ORDER BY ts_rank(d.search_vector, p.query) DESC, d.id ASC
    """,
            query,
        )
    ]


def calculate_metrics(queries):
    """MEM-76 hit@10 and full MRR, excluding no-answer queries."""
    ranks = []
    without_answer = 0
    for query in queries:
        answers = set(query["answers"])
        if not answers:
            without_answer += 1
            continue
        rank = next((i for i, clip in enumerate(query["ranking"], 1) if clip in answers), None)
        ranks.append(rank)
    if not ranks:
        raise ValueError("no answerable queries")
    return {
        "evaluated": len(ranks),
        "without_answer": without_answer,
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank if rank else 0 for rank in ranks) / len(ranks),
    }


async def evaluate(conn, documents, queries):
    """Evaluate all queries, returning full rankings for reproducibility."""
    if conn.is_in_transaction():
        raise ValueError("evaluate requires a connection without an active transaction")
    if not await conn.fetchval(
        "SELECT to_tsvector('russian', 'Собаки') @@ plainto_tsquery('russian', 'собаку')"
    ):
        raise ValueError("PostgreSQL locale does not support Russian case folding")
    async with conn.transaction():
        await build_index(conn, documents)
        results = []
        for query in queries:
            results.append({**query, "ranking": await search(conn, query["query"])})
        return {"metrics": calculate_metrics(results), "queries": results}


async def _main(dsn, root):
    import asyncpg

    documents, missing = load_collection(root)
    queries = load_queries(root, {doc["id"] for doc in documents})
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        report = await evaluate(conn, documents, queries)
        report["collection"] = {"clips": len(documents), "missing_transcripts": missing}
        comparison_path = root / "research/embeddings_check/results/comparison.csv"
        with comparison_path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        names = ["multilingual-e5-large", "BM25 (по словам)"]
        report["mem76"] = {}
        for name in names:
            row = next(
                r for r in rows if r["модель"] == name and r["разбиение"] == "окно 15 с / шаг 7 с"
            )
            report["mem76"][name] = {
                "recall_at_10": float(row["Recall@10"]),
                "mrr": float(row["MRR"]),
                "source": "research/embeddings_check/results/comparison.csv",
                "chunking": row["разбиение"],
            }
        report["method"] = {
            "postgres_version": await conn.fetchval("SHOW server_version"),
            "lc_ctype": await conn.fetchval(
                "SELECT datctype FROM pg_database WHERE datname = current_database()"
            ),
            "text_config": "russian",
            "query": "OR lexemes",
            "ranking": "ts_rank",
            "comparison": "MEM-76 passport captions and saved transcripts; no chunks",
        }
        return report
    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    dsn = os.environ.get("MEM80_TEST_DSN")
    if not dsn:
        parser.error("set MEM80_TEST_DSN to a PostgreSQL DSN with TEMP privilege")
    print(json.dumps(asyncio.run(_main(dsn, args.root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
