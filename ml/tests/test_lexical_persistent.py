"""Tests for the collection-scoped persistent lexical index."""

import asyncio
import os
import secrets
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest


class FakeConnection:
    def __init__(self):
        self.calls = []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return [{"clip_id": 7, "score": 0.25}]


def test_search_binds_user_input_and_collection():
    from ml.lexical_persistent import search

    conn = FakeConnection()
    collection = uuid.uuid4()
    query = "кошка'); DROP TABLE public.clips; --"
    assert asyncio.run(search(conn, collection, query, 5)) == [{"clip_id": 7, "score": 0.25}]
    sql, args = conn.calls[0]
    assert query not in sql
    assert args == (collection, query, 5)


@pytest.mark.parametrize("limit", [0, -1, True, 10001, 1.5])
def test_search_rejects_bad_limit(limit):
    from ml.lexical_persistent import search

    with pytest.raises(ValueError):
        asyncio.run(search(FakeConnection(), uuid.uuid4(), "кот", limit))


def test_search_requires_explicit_collection():
    from ml.lexical_persistent import search

    class Connection:
        async def fetch(self, *args):
            raise AssertionError("must validate before querying")

    try:
        asyncio.run(search(Connection(), None, "кошка"))
    except ValueError:
        pass
    else:
        raise AssertionError("missing collection must be rejected")


@pytest.fixture(scope="module")
def isolated_postgres():
    """Opt-in only: create and remove our own empty password-protected container."""
    if os.environ.get("MEM80_RUN_DOCKER_TESTS") != "1":
        pytest.skip("set MEM80_RUN_DOCKER_TESTS=1 to start an isolated container")
    name = "mem80-test-" + secrets.token_hex(12)
    password = secrets.token_hex(24)
    user = "mem80_" + secrets.token_hex(4)
    db = "mem80_" + secrets.token_hex(4)
    root = Path(__file__).resolve().parents[2]
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                "-e",
                f"POSTGRES_USER={user}",
                "-e",
                f"POSTGRES_PASSWORD={password}",
                "-e",
                f"POSTGRES_DB={db}",
                "-p",
                "127.0.0.1::5432",
                "pgvector/pgvector:0.8.0-pg16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        port_text = subprocess.check_output(["docker", "port", name, "5432/tcp"], text=True).strip()
        port = int(port_text.rsplit(":", 1)[1])
        for _ in range(60):
            result = subprocess.run(
                ["docker", "exec", name, "pg_isready", "-U", user, "-d", db],
                capture_output=True,
            )
            if result.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("isolated PostgreSQL did not start")
        env = os.environ.copy()
        env.update(
            POSTGRES_HOST="127.0.0.1",
            POSTGRES_PORT=str(port),
            POSTGRES_DB=db,
            POSTGRES_USER=user,
            POSTGRES_PASSWORD=password,
            PYTHONPATH=str(root / "backend"),
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                str(root / "backend/alembic.ini"),
                "upgrade",
                "head",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError("isolated project migration failed")
        yield dict(host="127.0.0.1", port=port, database=db, user=user, password=password)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def test_persistent_index_real_postgres(isolated_postgres):
    from ml.lexical_persistent import index_collection, install, search

    async def run():
        import asyncpg

        conn = await asyncpg.connect(**isolated_postgres)
        try:
            a, b = uuid.uuid4(), uuid.uuid4()
            await conn.executemany(
                "INSERT INTO public.collections(id,key_hash) VALUES($1,$2)",
                [(a, "a"), (b, "b")],
            )
            await conn.executemany(
                "INSERT INTO public.clips(id,collection_id,platform,external_id,url,status,title,caption) "
                "VALUES($1,$2,'vk',$3,'https://example.org','ready',$4,$5)",
                [
                    (1, a, "1", "Красная кошка", None),
                    (2, a, "2", None, "Горные вершины"),
                    (3, a, "3", None, None),
                    (4, b, "4", "кошка", None),
                    (5, a, "5", "кот", None),
                    (6, a, "6", "кот", None),
                ],
            )
            await conn.executemany(
                'INSERT INTO public.transcript_segments(clip_id,start,"end",text,source) '
                "VALUES($1,0,1,$2,$3)",
                [(2, "Собаки бегут", "asr"), (3, "Самолёты", "subtitles")],
            )
            await conn.execute(
                "INSERT INTO public.chunks(clip_id,ordinal,source,text) VALUES(3,0,'ocr','Осьминоги')"
            )
            await install(conn)
            await install(conn)
            assert await index_collection(conn, a) == 5
            assert await index_collection(conn, b) == 1
            assert await index_collection(conn, a) == 5
            assert await conn.fetchval("SELECT count(*) FROM mem80_lexical.clip_documents") == 6
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM pg_indexes WHERE schemaname='mem80_lexical' "
                    "AND indexdef LIKE '%USING gin%'"
                )
                == 1
            )
            assert [r["clip_id"] for r in await search(conn, a, "кошки")] == [1]
            assert [r["clip_id"] for r in await search(conn, b, "кошки")] == [4]
            assert [r["clip_id"] for r in await search(conn, a, "собаку")] == [2]
            assert [r["clip_id"] for r in await search(conn, a, "самолёт")] == [3]
            assert [r["clip_id"] for r in await search(conn, a, "осьминога")] == [3]
            assert {r["clip_id"] for r in await search(conn, a, "кошки вершина")} == {1, 2}
            assert [r["clip_id"] for r in await search(conn, a, "кот")] == [5, 6]
            assert await search(conn, a, "и в на") == []
            await search(conn, a, "кошка'); DROP TABLE public.clips; --")
            assert await conn.fetchval("SELECT count(*) FROM public.clips") == 6
            await conn.execute("UPDATE public.clips SET title='киты' WHERE id=1")
            await index_collection(conn, a)
            assert await search(conn, a, "кошки") == []
            assert [r["clip_id"] for r in await search(conn, a, "кита")] == [1]
            assert [r["clip_id"] for r in await search(conn, b, "кошки")] == [4]
            await conn.close()
            conn = await asyncpg.connect(**isolated_postgres)
            assert [r["clip_id"] for r in await search(conn, a, "киты")] == [1]
        finally:
            await conn.close()

    asyncio.run(run())
