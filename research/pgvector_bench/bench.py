"""Замер pgvector и индекса HNSW на синтетических векторах (MEM-78).

Для каждой размерности и объёма скрипт создаёт временную таблицу, заполняет её случайными
нормированными векторами (генерация на стороне PostgreSQL, без numpy), строит индекс HNSW
(vector_cosine_ops), измеряет время построения и объём индекса, время поиска топ-10 при разных
hnsw.ef_search и recall@10 относительно точного поиска. После замера таблицы удаляются.

Запуск из корня репозитория (нужен поднятый `docker compose up -d --wait postgres`):

    uv run --locked python research/pgvector_bench/bench.py

Результаты печатаются и сохраняются в results.csv и results.md рядом со скриптом.
Для параллельных запусков задавайте разные --output, чтобы не перезаписать результаты.
Временные таблицы изолированы по соединениям; построение их индексов не параллелится.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import asyncpg

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TABLE_PREFIX = "bench"
BATCH = 10_000
# Модель синтетических данных: темы -> ролики -> фрагменты (в среднем 9 фрагментов на ролик).
# Разброс подобран так, что косинусная близость фрагментов одного ролика ~0.9, одной темы ~0.7,
# разных тем ~0 — похоже на распределение у multilingual-e5-large.
FRAGMENTS_PER_VIDEO = 9
VIDEOS_PER_TOPIC = 100
VIDEO_SPREAD = 0.5
FRAGMENT_SPREAD = 0.35


@dataclass
class Row:
    dim: int
    n_vectors: int
    load_s: float
    build_s: float
    index_mb: float
    table_mb: float
    ef_search: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    recall_at_10: float
    exact_p50_ms: float


def load_dotenv(path: Path) -> dict[str, str]:
    """Читает простой .env (KEY=VALUE) без внешних зависимостей."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def db_settings() -> dict[str, str | int]:
    env = load_dotenv(REPO_ROOT / ".env") | dict(os.environ)
    password = env.get("POSTGRES_PASSWORD", "")
    if not password:
        raise SystemExit("POSTGRES_PASSWORD не задан: заполните .env (см. .env.example)")
    return {
        "host": env.get("POSTGRES_HOST", "localhost"),
        "port": int(env.get("POSTGRES_PORT", "5432")),
        "user": env.get("POSTGRES_USER", "memsearch"),
        "password": password,
        "database": env.get("POSTGRES_DB", "memsearch"),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


async def fill_uniform(conn: asyncpg.Connection, table: str, dim: int, n: int) -> None:
    """Изотропный шум: n случайных векторов единичной длины (худший случай для HNSW)."""
    for start in range(1, n + 1, BATCH):
        stop = min(n, start + BATCH - 1)
        # Ссылка на g внутри подзапроса заставляет генерировать новый вектор для каждой строки.
        await conn.execute(
            f"""
            INSERT INTO {table} (id, embedding)
            SELECT g, l2_normalize(
                ARRAY(SELECT random_normal() FROM generate_series(1, {dim}) WHERE g > 0)::vector
            )
            FROM generate_series({start}, {stop}) AS g
            """
        )


async def make_centers(conn: asyncpg.Connection, dim: int, n_topics: int, n_videos: int) -> None:
    """Центры тем и роликов во временных таблицах bench_topic / bench_video."""
    await conn.execute(
        f"""
        CREATE TEMP TABLE bench_topic AS
        SELECT t AS id,
               ARRAY(SELECT random_normal() / sqrt({dim})
                     FROM generate_series(1, {dim}) WHERE t >= 0)::real[] AS c
        FROM generate_series(0, {n_topics - 1}) AS t
        """
    )
    await conn.execute(
        f"""
        CREATE TEMP TABLE bench_video AS
        SELECT v AS id,
               ARRAY(SELECT x + {VIDEO_SPREAD} * random_normal() / sqrt({dim})
                     FROM unnest(t.c) AS x)::real[] AS c
        FROM generate_series(0, {n_videos - 1}) AS v
        JOIN pg_temp.bench_topic t ON t.id = v % {n_topics}
        """
    )
    await conn.execute("ALTER TABLE pg_temp.bench_video ADD PRIMARY KEY (id)")


async def fill_clustered(
    conn: asyncpg.Connection, table: str, dim: int, n: int, n_videos: int, *, random_video: bool
) -> None:
    """Фрагменты роликов: центр ролика + шум, нормировка.

    Для коллекции фрагменты идут подряд по FRAGMENTS_PER_VIDEO на ролик; для запросов ролик
    выбирается случайно (запрос «про содержание» какого-то ролика).
    """
    video_expr = (
        f"floor(random() * {n_videos})::int"
        if random_video
        else (f"((g - 1) / {FRAGMENTS_PER_VIDEO})::int")
    )
    for start in range(1, n + 1, BATCH):
        stop = min(n, start + BATCH - 1)
        await conn.execute(
            f"""
            INSERT INTO {table} (id, embedding)
            SELECT s.g, l2_normalize(
                ARRAY(SELECT x + {FRAGMENT_SPREAD} * random_normal() / sqrt({dim})
                      FROM unnest(v.c) AS x)::vector
            )
            FROM (SELECT g, {video_expr} AS vid FROM generate_series({start}, {stop}) AS g) s
            JOIN pg_temp.bench_video v ON v.id = s.vid
            """
        )


async def generate(
    conn: asyncpg.Connection, args: argparse.Namespace, table: str, qtable: str, dim: int, n: int
) -> None:
    n_queries = max(args.queries, args.recall_queries)
    if args.data == "uniform":
        await fill_uniform(conn, table, dim, n)
        await fill_uniform(conn, qtable, dim, n_queries)
        return
    n_videos = max(1, -(-n // FRAGMENTS_PER_VIDEO))
    n_topics = max(1, n_videos // VIDEOS_PER_TOPIC)
    await make_centers(conn, dim, n_topics, n_videos)
    await fill_clustered(conn, table, dim, n, n_videos, random_video=False)
    await fill_clustered(conn, qtable, dim, n_queries, n_videos, random_video=True)
    await conn.execute("DROP TABLE IF EXISTS pg_temp.bench_topic, pg_temp.bench_video")


async def search_ids(conn: asyncpg.Connection, table: str, query: str) -> list[int]:
    rows = await conn.fetch(
        f"SELECT id FROM {table} ORDER BY embedding <=> $1::vector LIMIT 10", query
    )
    return [r["id"] for r in rows]


async def timed_search(conn: asyncpg.Connection, table: str, query: str) -> tuple[float, list[int]]:
    t0 = time.perf_counter()
    ids = await search_ids(conn, table, query)
    return (time.perf_counter() - t0) * 1000, ids


async def exact_search(
    conn: asyncpg.Connection, table: str, queries: list[str]
) -> tuple[list[list[int]], list[float]]:
    """Точный поиск полным перебором (индекс отключён)."""
    results: list[list[int]] = []
    times: list[float] = []
    async with conn.transaction():
        indexscan = await conn.fetchval("SELECT current_setting('enable_indexscan')")
        bitmapscan = await conn.fetchval("SELECT current_setting('enable_bitmapscan')")
        await conn.execute("SET LOCAL enable_indexscan = off")
        await conn.execute("SET LOCAL enable_bitmapscan = off")
        for q in queries:
            ms, ids = await timed_search(conn, table, q)
            results.append(ids)
            times.append(ms)
        await conn.execute("SELECT set_config('enable_indexscan', $1, true)", indexscan)
        await conn.execute("SELECT set_config('enable_bitmapscan', $1, true)", bitmapscan)
    return results, times


async def bench_one(
    conn: asyncpg.Connection, args: argparse.Namespace, dim: int, n: int
) -> list[Row]:
    name = f"{TABLE_PREFIX}_{dim}_{n}"
    qname = f"{name}_q"
    table = f"pg_temp.{name}"
    qtable = f"pg_temp.{qname}"
    print(f"\n== dim={dim}, n={n:,} ==", flush=True)
    # В transaction pooling одна серверная сессия закреплена за всей транзакцией.
    # При ошибке PostgreSQL откатит создание таблиц, без SQL в прерванной транзакции.
    async with conn.transaction():
        await conn.execute(
            f"CREATE TEMP TABLE {name} (id bigint PRIMARY KEY, embedding vector({dim}))"
        )

        await conn.execute(
            f"CREATE TEMP TABLE {qname} (id bigint PRIMARY KEY, embedding vector({dim}))"
        )

        await conn.execute("SELECT setseed($1)", args.seed)

        t0 = time.perf_counter()
        await generate(conn, args, table, qtable, dim, n)
        await conn.execute(f"ANALYZE {table}")
        load_s = time.perf_counter() - t0
        print(f"  генерация и загрузка: {load_s:.1f} с", flush=True)

        await conn.execute(
            "SELECT set_config('maintenance_work_mem', $1, true)", args.maintenance_work_mem
        )
        await conn.execute(f"SET LOCAL max_parallel_maintenance_workers = {args.parallel_workers}")
        t0 = time.perf_counter()
        await conn.execute(
            f"CREATE INDEX {name}_hnsw ON {table} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {args.m}, ef_construction = {args.ef_construction})"
        )
        build_s = time.perf_counter() - t0
        index_mb = await conn.fetchval(f"SELECT pg_relation_size('pg_temp.{name}_hnsw')") / 2**20
        table_mb = await conn.fetchval(f"SELECT pg_table_size('{table}')") / 2**20
        print(f"  HNSW: {build_s:.1f} с, индекс {index_mb:.1f} МБ, таблица {table_mb:.1f} МБ")

        queries = [
            r["e"]
            for r in await conn.fetch(f"SELECT embedding::text AS e FROM {qtable} ORDER BY id")
        ]
        recall_queries = queries[: args.recall_queries]
        truth, exact_times = await exact_search(conn, table, recall_queries)
        exact_p50 = statistics.median(exact_times)
        print(f"  точный поиск (перебор): p50 {exact_p50:.1f} мс", flush=True)

        # Планировщик на малых таблицах иногда выбирает полный перебор, и замер смешивает два
        # плана. Принудительно меряем именно HNSW и проверяем план через EXPLAIN.
        await conn.execute("SET LOCAL enable_seqscan = off")
        await conn.execute("SET LOCAL plan_cache_mode = force_custom_plan")
        plan = "\n".join(
            r[0]
            for r in await conn.fetch(
                f"EXPLAIN SELECT id FROM {table} ORDER BY embedding <=> $1::vector LIMIT 10",
                queries[0],
            )
        )
        if f"{name}_hnsw" not in plan:
            raise RuntimeError(f"Поиск идёт не по HNSW:\n{plan}")

        rows: list[Row] = []
        for ef in args.ef_search:
            await conn.execute(f"SET LOCAL hnsw.ef_search = {ef}")
            for q in queries[: args.warmup]:
                await search_ids(conn, table, q)
            times: list[float] = []
            for q in queries[: args.queries]:
                ms, _ = await timed_search(conn, table, q)
                times.append(ms)
            hits = 0
            for q, true_ids in zip(recall_queries, truth, strict=True):
                hits += len(set(await search_ids(conn, table, q)) & set(true_ids))
            recall = hits / (10 * len(recall_queries))
            row = Row(
                dim=dim,
                n_vectors=n,
                load_s=round(load_s, 1),
                build_s=round(build_s, 1),
                index_mb=round(index_mb, 1),
                table_mb=round(table_mb, 1),
                ef_search=ef,
                p50_ms=round(statistics.median(times), 2),
                p95_ms=round(percentile(times, 0.95), 2),
                max_ms=round(max(times), 2),
                recall_at_10=round(recall, 3),
                exact_p50_ms=round(exact_p50, 1),
            )
            print(
                f"  ef_search={ef}: p50 {row.p50_ms} мс, p95 {row.p95_ms} мс, "
                f"max {row.max_ms} мс, recall@10 {row.recall_at_10}",
                flush=True,
            )
            rows.append(row)
        await conn.execute(f"DROP TABLE IF EXISTS {table}, {qtable}")
        return rows


def write_results(rows: list[Row], meta: str, stem: str) -> None:
    fields = list(asdict(rows[0]).keys())
    with (HERE / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)

    header = (
        "| Размерность | Векторов | Загрузка, с | Построение HNSW, с | Индекс, МБ | Таблица, МБ "
        "| ef_search | p50, мс | p95, мс | max, мс | recall@10 | Перебор p50, мс |"
    )
    lines = [
        "# Результаты замера pgvector + HNSW (MEM-78)",
        "",
        meta,
        "",
        header,
        "|" + "---|" * 12,
    ]
    for r in rows:
        lines.append(
            f"| {r.dim} | {r.n_vectors:_} | {r.load_s} | {r.build_s} | {r.index_mb} | {r.table_mb} "
            f"| {r.ef_search} | {r.p50_ms} | {r.p95_ms} | {r.max_ms} | {r.recall_at_10} "
            f"| {r.exact_p50_ms} |".replace("_", " ")
        )
    (HERE / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[4:]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data",
        choices=["clustered", "uniform"],
        default="clustered",
        help="clustered — темы/ролики/фрагменты как у реальных эмбеддингов; "
        "uniform — изотропный шум (худший случай для HNSW)",
    )
    p.add_argument("--dims", type=int, nargs="+", default=[768, 1024])
    p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 50_000, 100_000])
    p.add_argument("--ef-search", type=int, nargs="+", default=[40, 100])
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=64)
    p.add_argument("--queries", type=int, default=100, help="запросов для замера времени")
    p.add_argument("--recall-queries", type=int, default=20, help="запросов для recall@10")
    p.add_argument("--warmup", type=int, default=10, help="прогревочных запросов (не в замере)")
    p.add_argument("--seed", type=float, default=0.42, help="setseed() для воспроизводимости")
    p.add_argument("--output", default="results", help="имя файлов результатов без расширения")
    p.add_argument("--maintenance-work-mem", default="1GB")
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="max_parallel_maintenance_workers (для временных таблиц индекс строится без workers)",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    conn = await asyncpg.connect(**db_settings())
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        pg_version = await conn.fetchval("SHOW server_version")
        vec_version = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        meta = (
            f"PostgreSQL {pg_version}, pgvector {vec_version}; HNSW m={args.m}, "
            f"ef_construction={args.ef_construction}, vector_cosine_ops; "
            f"{args.queries} запросов на замер времени, {args.recall_queries} на recall@10; "
            f"данные {args.data} (seed {args.seed}), {FRAGMENTS_PER_VIDEO} фрагментов на ролик. "
            f"Дата: {time.strftime('%Y-%m-%d')}."
        )
        print(meta)
        rows: list[Row] = []
        for dim in args.dims:
            for n in args.sizes:
                rows.extend(await bench_one(conn, args, dim, n))
    finally:
        await conn.close()
    write_results(rows, meta, args.output)


if __name__ == "__main__":
    asyncio.run(main())
