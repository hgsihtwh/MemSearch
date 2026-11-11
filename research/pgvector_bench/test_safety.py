"""Regression tests use only an in-memory fake; no PostgreSQL connection is made."""

import argparse
import asyncio
import re

import bench
import pytest


class FakeDB:
    def __init__(self):
        self.public = {"bench_3_10", "bench_3_10_q", "bench_topic", "bench_video"}
        self.dropped_public = set()


class FakeConnection:
    def __init__(self, db, fail_at=None):
        self.db = db
        self.temporary = set()
        self.fail_at = fail_at
        self.sql = []
        self.executions = []
        self.events = []
        self.transactions = []
        self.aborted = False

    async def execute(self, sql, *args):
        sql = " ".join(sql.split())
        self.sql.append(sql)
        self.executions.append((sql, args))
        self.events.append(("sql", len(self.transactions), sql))
        await asyncio.sleep(0)
        if self.aborted:
            raise RuntimeError("transaction is aborted")
        if self.fail_at and self.fail_at in sql:
            self.aborted = bool(self.transactions)
            raise RuntimeError("injected failure")
        match = re.match(r"CREATE (TEMP(?:ORARY)? )?TABLE ([\w.]+)", sql, re.I)
        if match:
            temp, name = match.groups()
            name = name.removeprefix("pg_temp.")
            relations = self.temporary if temp else self.db.public
            if name in relations:
                self.aborted = bool(self.transactions)
                raise RuntimeError("relation already exists")
            relations.add(name)
        match = re.match(r"DROP TABLE(?: IF EXISTS)? (.+)", sql, re.I)
        if match:
            for name in match.group(1).split(", "):
                if name.startswith("pg_temp."):
                    self.temporary.discard(name.removeprefix("pg_temp."))
                elif name in self.temporary:
                    self.temporary.remove(name)
                elif name in self.db.public:
                    self.db.public.remove(name)
                    self.db.dropped_public.add(name)

    async def fetchval(self, sql, *args):
        self.sql.append(" ".join(sql.split()))
        self.events.append(("sql", len(self.transactions), " ".join(sql.split())))
        if "current_setting" in sql:
            return "on"
        return 1048576

    async def fetch(self, sql, *args):
        self.sql.append(" ".join(sql.split()))
        self.events.append(("sql", len(self.transactions), " ".join(sql.split())))
        if sql.startswith("EXPLAIN"):
            return [("Index Scan using bench_3_10_hnsw",)]
        if "embedding::text" in sql:
            return [{"e": "[1,0,0]"}]
        return [{"id": i} for i in range(10)]

    def transaction(self):
        return FakeTransaction(self)


class FakeTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.transactions.append((self.conn.temporary.copy(), self.conn.aborted))
        self.conn.events.append(("begin", len(self.conn.transactions), ""))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        depth = len(self.conn.transactions)
        snapshot, was_aborted = self.conn.transactions.pop()
        self.conn.events.append(("rollback" if exc_type else "commit", depth, ""))
        if exc_type:
            self.conn.temporary = snapshot
            self.conn.aborted = was_aborted
        return False


def arguments():
    return argparse.Namespace(
        seed=0.42,
        data="clustered",
        queries=1,
        recall_queries=1,
        warmup=1,
        maintenance_work_mem="1GB",
        parallel_workers=0,
        m=16,
        ef_construction=64,
        ef_search=[40],
    )


def test_maintenance_work_mem_is_passed_as_a_local_parameter():
    conn = FakeConnection(FakeDB())
    args = arguments()
    args.maintenance_work_mem = "1GB'; DROP TABLE public.bench_topic; --"
    asyncio.run(bench.bench_one(conn, args, 3, 10))
    config = [(sql, params) for sql, params in conn.executions if "maintenance_work_mem" in sql]
    assert config == [
        ("SELECT set_config('maintenance_work_mem', $1, true)", (args.maintenance_work_mem,))
    ]


def test_benchmark_uses_one_transaction_with_nested_exact_search():
    conn = FakeConnection(FakeDB())
    asyncio.run(bench.bench_one(conn, arguments(), 3, 10))
    assert conn.events[0] == ("begin", 1, "")
    assert conn.events[-1] == ("commit", 1, "")
    assert [(action, depth) for action, depth, _ in conn.events if action != "sql"] == [
        ("begin", 1),
        ("begin", 2),
        ("commit", 2),
        ("commit", 1),
    ]
    assert all(depth >= 1 for action, depth, _ in conn.events if action == "sql")
    assert any(depth == 2 and "enable_indexscan" in sql for _, depth, sql in conn.events)
    assert all(sql.startswith("SET LOCAL ") for sql in conn.sql if sql.startswith("SET "))
    assert not any(sql.startswith("RESET ") for sql in conn.sql)
    assert not conn.temporary


@pytest.mark.parametrize(
    "fail_at",
    [
        "CREATE TEMP TABLE bench_video",
        "INSERT INTO pg_temp.bench_3_10 ",
        "SET LOCAL enable_indexscan",
        "DROP TABLE IF EXISTS pg_temp.bench_topic",
        "DROP TABLE IF EXISTS pg_temp.bench_3_10",
    ],
)
def test_benchmark_failure_rolls_back_without_masking_original_error(fail_at):
    conn = FakeConnection(FakeDB(), fail_at=fail_at)
    with pytest.raises(RuntimeError, match="^injected failure$"):
        asyncio.run(bench.bench_one(conn, arguments(), 3, 10))
    assert conn.events[-1] == ("rollback", 1, "")
    assert not conn.temporary
    assert not conn.db.dropped_public


def test_exact_search_restores_original_planner_settings_in_outer_transaction():
    conn = FakeConnection(FakeDB())

    async def run():
        async with conn.transaction():
            await bench.exact_search(conn, "pg_temp.bench_3_10", ["[1,0,0]"])

    asyncio.run(run())
    restores = [(sql, args) for sql, args in conn.executions if "set_config" in sql]
    assert restores == [
        ("SELECT set_config('enable_indexscan', $1, true)", ("on",)),
        ("SELECT set_config('enable_bitmapscan', $1, true)", ("on",)),
    ]


def test_bench_failure_does_not_drop_existing_public_tables():
    db = FakeDB()
    conn = FakeConnection(db, fail_at="SELECT setseed")
    with pytest.raises(RuntimeError, match="injected failure"):
        asyncio.run(bench.bench_one(conn, argparse.Namespace(seed=0.42), 3, 10))
    assert not db.dropped_public
    assert db.public == {"bench_3_10", "bench_3_10_q", "bench_topic", "bench_video"}
    assert not conn.temporary


def test_make_centers_failure_cleans_only_created_tables():
    db = FakeDB()
    conn = FakeConnection(db, fail_at="CREATE TEMP TABLE bench_video")

    async def run():
        async with conn.transaction():
            await bench.make_centers(conn, 3, 2, 4)

    with pytest.raises(RuntimeError, match="injected failure"):
        asyncio.run(run())
    assert not db.dropped_public
    assert not conn.temporary


def test_preexisting_temporary_table_is_not_removed_on_creation_failure():
    db = FakeDB()
    conn = FakeConnection(db)
    conn.temporary.add("bench_3_10")
    with pytest.raises(RuntimeError, match="relation already exists"):
        asyncio.run(bench.bench_one(conn, arguments(), 3, 10))
    assert conn.temporary == {"bench_3_10"}
    assert not db.dropped_public


def test_second_table_collision_only_removes_created_table():
    db = FakeDB()
    conn = FakeConnection(db)
    conn.temporary.add("bench_3_10_q")
    with pytest.raises(RuntimeError, match="relation already exists"):
        asyncio.run(bench.bench_one(conn, arguments(), 3, 10))
    assert conn.temporary == {"bench_3_10_q"}
    assert not db.dropped_public


def test_center_collision_preserves_preexisting_temporary_video_table():
    db = FakeDB()
    conn = FakeConnection(db)
    conn.temporary.add("bench_video")

    async def run():
        async with conn.transaction():
            await bench.make_centers(conn, 3, 2, 4)

    with pytest.raises(RuntimeError, match="relation already exists"):
        asyncio.run(run())
    assert conn.temporary == {"bench_video"}
    assert not db.dropped_public


def test_generation_failure_cleans_centers_and_benchmark_tables():
    db = FakeDB()
    conn = FakeConnection(db, fail_at="INSERT INTO pg_temp.bench_3_10 ")
    with pytest.raises(RuntimeError, match="injected failure"):
        asyncio.run(bench.bench_one(conn, arguments(), 3, 10))
    assert not conn.temporary
    assert not db.dropped_public


def test_two_parallel_sessions_use_private_tables_and_hnsw():
    db = FakeDB()
    one, two = FakeConnection(db), FakeConnection(db)

    async def run_both():
        return await asyncio.gather(
            bench.bench_one(one, arguments(), 3, 10),
            bench.bench_one(two, arguments(), 3, 10),
        )

    results = asyncio.run(run_both())
    assert [len(rows) for rows in results] == [1, 1]
    assert not db.dropped_public
    for conn in (one, two):
        assert not conn.temporary
        assert any(
            "CREATE INDEX bench_3_10_hnsw ON pg_temp.bench_3_10 USING hnsw" in sql
            for sql in conn.sql
        )
        assert any("pg_relation_size('pg_temp.bench_3_10_hnsw')" in sql for sql in conn.sql)
        drops = [sql for sql in conn.sql if sql.startswith("DROP TABLE")]
        assert drops
        for sql in drops:
            names = sql.removeprefix("DROP TABLE IF EXISTS ").split(", ")
            assert all(name.startswith("pg_temp.") for name in names)
