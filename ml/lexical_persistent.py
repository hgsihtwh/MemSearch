"""Persistent collection-scoped PostgreSQL lexical search.

Explicit installation creates only the named ``mem80_lexical`` schema and its
owned objects; it never modifies the source tables. Indexing replaces derived
data for one collection atomically. Callers must authorize access to the UUID
before invoking these APIs: a collection UUID is not an authentication token.
The index is a snapshot, not a trigger: rerun indexing after source changes.
Run ``python -m ml.lexical_persistent --help`` for the explicit CLI operations.
"""

import argparse
import asyncio
import json
import os
import uuid

SCHEMA = "mem80_lexical"
# Serializes concurrent installs; arbitrary constant private to this module.
_INSTALL_LOCK = 0x4D454D80

# One document per clip: title, caption, transcript segments (ASR or platform
# subtitles) and text-like chunks that are not derived from the former
# (``ocr`` — text in frame, ``scene`` — scene description). ``caption`` and
# ``speech`` chunks duplicate clip fields and segments, so they are skipped.
_DOCUMENT_SQL = """
    SELECT c.id, c.collection_id, to_tsvector('pg_catalog.russian', concat_ws(' ',
        c.title,
        c.caption,
        (SELECT string_agg(s.text, ' ' ORDER BY s.start, s.id)
           FROM public.transcript_segments s WHERE s.clip_id = c.id),
        (SELECT string_agg(k.text, ' ' ORDER BY k.ordinal)
           FROM public.chunks k WHERE k.clip_id = c.id AND k.source IN ('ocr', 'scene'))
    ))
    FROM public.clips c
    WHERE c.collection_id = $1
"""


def _collection_id(value):
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("an explicit collection UUID is required") from exc


async def install(connection):
    """Create the schema, document table and its single GIN index; idempotent."""
    async with connection.transaction():
        await connection.execute("SELECT pg_advisory_xact_lock($1)", _INSTALL_LOCK)
        await connection.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        await connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.clip_documents (
                clip_id bigint PRIMARY KEY
                    REFERENCES public.clips(id) ON DELETE CASCADE,
                collection_id uuid NOT NULL
                    REFERENCES public.collections(id) ON DELETE CASCADE,
                search_vector tsvector NOT NULL,
                indexed_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        await connection.execute(
            f"CREATE INDEX IF NOT EXISTS clip_documents_collection_idx "
            f"ON {SCHEMA}.clip_documents (collection_id)"
        )
        await connection.execute(
            f"CREATE INDEX IF NOT EXISTS clip_documents_search_gin "
            f"ON {SCHEMA}.clip_documents USING gin (search_vector)"
        )


async def index_collection(connection, collection_id):
    """Atomically rebuild documents of one collection; return the number of clips."""
    collection_id = _collection_id(collection_id)
    async with connection.transaction():
        if not await connection.fetchval(
            "SELECT EXISTS(SELECT 1 FROM public.collections WHERE id = $1)", collection_id
        ):
            raise ValueError("collection does not exist")
        # Lock the collection's clips so the snapshot is consistent with them.
        await connection.execute(
            "SELECT id FROM public.clips WHERE collection_id = $1 ORDER BY id FOR SHARE",
            collection_id,
        )
        await connection.execute(
            f"DELETE FROM {SCHEMA}.clip_documents WHERE collection_id = $1", collection_id
        )
        status = await connection.execute(
            f"INSERT INTO {SCHEMA}.clip_documents (clip_id, collection_id, search_vector) "
            + _DOCUMENT_SQL,
            collection_id,
        )
    return int(status.rsplit(" ", 1)[1])


async def search(connection, collection_id, query, limit=100):
    collection_id = _collection_id(collection_id)
    if not isinstance(query, str):
        raise ValueError("query must be text")
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError("limit must be an integer from 1 to 10000")
    rows = await connection.fetch(
        """
        WITH parsed AS (
            SELECT to_tsquery('pg_catalog.russian', coalesce(
                (SELECT string_agg(quote_literal(lexeme), ' | ' ORDER BY lexeme)
                 FROM unnest(tsvector_to_array(to_tsvector('pg_catalog.russian', $2)))
                      AS lexemes(lexeme)), '')) AS query
        )
        SELECT d.clip_id, ts_rank(d.search_vector, parsed.query) AS score
        FROM mem80_lexical.clip_documents d
        JOIN public.clips c ON c.id = d.clip_id AND c.collection_id = d.collection_id
        CROSS JOIN parsed
        WHERE d.collection_id = $1 AND d.search_vector @@ parsed.query
        ORDER BY score DESC, d.clip_id ASC
        LIMIT $3
        """,
        collection_id,
        query,
        limit,
    )
    return [dict(row) for row in rows]


async def _run(args, dsn):
    import asyncpg

    connection = await asyncpg.connect(dsn)
    try:
        if args.command == "install":
            await install(connection)
            return {"installed": SCHEMA}
        if args.command == "index":
            return {
                "collection_id": str(args.collection),
                "indexed_clips": await index_collection(connection, args.collection),
            }
        return {
            "collection_id": str(args.collection),
            "query": args.query,
            "results": await search(connection, args.collection, args.query, args.limit),
        }
    finally:
        await connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="MEMSEARCH_DSN must be set explicitly. Order: install, index, search.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install", help="create the mem80_lexical schema (idempotent)")
    index = commands.add_parser("index", help="rebuild documents of one collection")
    index.add_argument("--collection", type=uuid.UUID, required=True)
    find = commands.add_parser("search", help="lexical search within one collection")
    find.add_argument("--collection", type=uuid.UUID, required=True)
    find.add_argument("--query", required=True)
    find.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    dsn = os.environ.get("MEMSEARCH_DSN")
    if not dsn:
        parser.exit(2, "error: MEMSEARCH_DSN must be set explicitly\n")
    try:
        result = asyncio.run(_run(args, dsn))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
