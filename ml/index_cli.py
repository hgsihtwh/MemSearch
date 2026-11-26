"""Safe CLI for indexing existing collection clips. Run: python -m ml.index_cli --help."""

import argparse
import asyncio
import csv
import hashlib
import hmac
import importlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from ml.chunking import chunk_transcript
from ml.index_collection import _unique_object, index_collection


def read_passport(path):
    """Read source IDs and require platform, external ID and canonical URL to agree."""
    result = {}
    identities = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"№", "ID", "Платформа", "Ссылка"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("passport missing required columns")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("duplicate passport columns")
        for row in reader:
            n, external, platform, url = (
                row[k].strip() for k in ("№", "ID", "Платформа", "Ссылка")
            )
            if not re.fullmatch(r"[1-9][0-9]*", n):
                raise ValueError("invalid passport ID")
            platform = platform.lower()
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
                raise ValueError("invalid URL")
            if platform == "tiktok":
                pattern = r"/@[^/]+/video/([0-9]+)/?"
                hosts = {"www.tiktok.com", "tiktok.com"}
            elif platform == "vk":
                pattern = r"/clip(-?[0-9]+_[0-9]+)/?"
                hosts = {"vk.com", "www.vk.com"}
            else:
                raise ValueError("invalid platform")
            match = re.fullmatch(pattern, parsed.path)
            if parsed.hostname not in hosts or not match or match.group(1) != external:
                raise ValueError("URL and external ID disagree")
            identifier = int(n)
            if identifier in result or (platform, external) in identities:
                raise ValueError("duplicate passport identity")
            identities.add((platform, external))
            result[identifier] = {"platform": platform, "external_id": external, "url": url}
    if not result:
        raise ValueError("empty passport")
    return result


def inspect_transcripts(directory, expected):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("transcripts directory does not exist")
    sources, paths = {}, {}
    report = {"missing": [], "unmatched": [], "invalid": {}}
    for path in sorted(directory.glob("*.json")):
        if not re.fullmatch(r"[0-9]+", path.stem) or int(path.stem) < 1:
            report["invalid"][path.name] = "invalid filename"
            continue
        identifier = int(path.stem)
        paths.setdefault(identifier, []).append(path)
    report["missing"] = sorted(expected - paths.keys())
    report["unmatched"] = sorted(paths.keys() - expected)
    for identifier in sorted(expected & paths.keys()):
        files = paths[identifier]
        try:
            if len(files) != 1 or files[0].is_symlink():
                raise ValueError("duplicate or symlink transcript")
            item = json.loads(
                files[0].read_text(encoding="utf-8"), object_pairs_hook=_unique_object
            )
            if (
                not isinstance(item, dict)
                or type(item.get("clip_id")) is not int
                or item["clip_id"] != identifier
            ):
                raise ValueError("transcript ID mismatch")
            chunk_transcript(item)
            sources[identifier] = item
        except (ValueError, UnicodeError, OSError) as exc:
            report["invalid"][str(identifier)] = str(exc)
    return sources, report


def mapping_digest(collection_id, mapping, passport):
    payload = {
        "collection_id": str(collection_id),
        "mapping": [
            [identifier, db_id, passport[identifier]]
            for identifier, db_id in sorted(mapping.items())
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


async def index_selected(
    connection, collection_id, passport, mapping, transcripts, confirmation, adapter_factory
):
    if not mapping or set(mapping) != set(transcripts):
        raise ValueError("empty or inconsistent mapping")
    if not hmac.compare_digest(mapping_digest(collection_id, mapping, passport), confirmation):
        raise ValueError("mapping confirmation mismatch")
    async with connection.transaction():
        records = await connection.fetch(
            "SELECT id, platform, external_id, url FROM clips WHERE collection_id = $1 "
            "AND id = ANY($2::bigint[]) ORDER BY id FOR UPDATE",
            collection_id,
            sorted(mapping.values()),
        )
        by_id = {r["id"]: r for r in records}
        if len(by_id) != len(mapping):
            raise ValueError("mapping changed")
        for identifier, db_id in mapping.items():
            record, source = by_id[db_id], passport[identifier]
            if (record["platform"], record["external_id"]) != (
                source["platform"],
                source["external_id"],
            ):
                raise ValueError("mapping changed")
        adapter = adapter_factory()
        return await index_collection(connection, collection_id, transcripts, mapping, adapter)


async def resolve_mapping(connection, collection_id, passport):
    if not await connection.fetchval(
        "SELECT EXISTS(SELECT 1 FROM collections WHERE id = $1)", collection_id
    ):
        raise ValueError("collection does not exist")
    records = await connection.fetch(
        "SELECT id, platform, external_id, url FROM clips WHERE collection_id = $1", collection_id
    )
    by_identity = {(r["platform"], r["external_id"]): r for r in records}
    mapping = {}
    for identifier, item in passport.items():
        found = by_identity.get((item["platform"], item["external_id"]))
        if found:
            mapping[identifier] = found["id"]
    matched = set(mapping.values())
    return mapping, {
        "unmatched_passport": sorted(set(passport) - mapping.keys()),
        "unmatched_database": sorted(r["id"] for r in records if r["id"] not in matched),
        "matches": [
            {
                "passport_id": i,
                "clip_id": db,
                "passport_url": passport[i]["url"],
                "database_url": by_identity[(passport[i]["platform"], passport[i]["external_id"])][
                    "url"
                ],
            }
            for i, db in sorted(mapping.items())
        ],
    }


def load_adapter():
    # Optional heavy dependency: imported lazily, like faster_whisper in transcribe.py.
    sentence_transformers = importlib.import_module("sentence_transformers")

    from ml.embeddings import MODEL, EmbeddingAdapter

    return EmbeddingAdapter(sentence_transformers.SentenceTransformer(MODEL))


async def search_collection(connection, collection_id, query, adapter, limit):
    start = time.perf_counter()
    vector = adapter.embed_queries([query])[0]
    encoded = time.perf_counter()
    literal = "[" + ",".join(str(float(x)) for x in vector) + "]"
    records = await connection.fetch(
        'SELECT clips.id AS clip_id, clips.url, chunks.source, chunks.start, chunks."end", '
        "chunks.text, chunks.embedding <=> $2::vector AS distance "
        "FROM chunks JOIN clips ON clips.id = chunks.clip_id "
        "WHERE clips.collection_id = $1 AND chunks.embedding IS NOT NULL "
        "AND chunks.source IN ('caption', 'speech') "
        "ORDER BY chunks.embedding <=> $2::vector LIMIT $3",
        collection_id,
        literal,
        limit,
    )
    finished = time.perf_counter()
    return {
        "results": [dict(row) for row in records],
        "timing_ms": {
            "query_encode": (encoded - start) * 1000,
            "database_and_serialization": (finished - encoded) * 1000,
            "end_to_end": (finished - start) * 1000,
        },
    }


async def run_command(args, dsn):
    import asyncpg

    started = time.perf_counter()
    connection = await asyncpg.connect(dsn)
    try:
        if args.command == "search":
            if not await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM collections WHERE id = $1)", args.collection
            ):
                raise ValueError("collection does not exist")
            load_start = time.perf_counter()
            adapter = load_adapter()
            load_ms = (time.perf_counter() - load_start) * 1000
            runs = [
                await search_collection(
                    connection, args.collection, args.query, adapter, args.limit
                )
                for _ in range(args.repeat)
            ]
            return {
                "model_load_ms": load_ms,
                "runs": runs,
                "first_run": "first encode after model load; OS/model caches may already be warm",
                "subsequent_runs": "same model and connection, warm process",
                "process_total_ms": (time.perf_counter() - started) * 1000,
                "note": "No latency guarantee; process_total includes connection and model load.",
            }
        passport = read_passport(args.passport)
        transcripts, quality = inspect_transcripts(args.transcripts, set(passport))
        mapping, mapping_report = await resolve_mapping(connection, args.collection, passport)
        eligible = {i: db for i, db in mapping.items() if i in transcripts}
        report = {
            "collection_id": str(args.collection),
            "model": "intfloat/multilingual-e5-large",
            "dimension": 1024,
            "window_s": 15,
            "stride_s": 7,
            "transcript_report": quality,
            "mapping_report": mapping_report,
            "eligible_ids": sorted(eligible),
            "confirmation_sha256": mapping_digest(args.collection, eligible, passport),
            "note": "Only valid transcripts are indexed; missing or invalid are not modified.",
        }
        if args.command == "index" and not args.dry_run:
            if quality["invalid"] or quality["unmatched"] or mapping_report["unmatched_passport"]:
                raise ValueError(
                    "report contains invalid or unmatched inputs; run plan and resolve issues"
                )
            report["indexed"] = await index_selected(
                connection,
                args.collection,
                passport,
                eligible,
                {i: transcripts[i] for i in eligible},
                args.confirm_mapping or "",
                load_adapter,
            )
        return report
    finally:
        await connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "MEMSEARCH_DSN must be explicitly set. Plan prints a SHA-256 confirmation of the mapping; "
            "inspect the IDs before running index --confirm-mapping HASH. Index replaces caption and "
            "speech chunks only for selected clips, atomically; source files and other collections "
            "are untouched. Missing/invalid transcripts are not repaired or silently indexed. "
            "Install sentence-transformers separately. Search timings include query encoding and DB; "
            "model loading/connection are reported separately. Model provenance is not in DB schema."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "index"):
        sub = commands.add_parser(name)
        sub.add_argument("--collection", type=uuid.UUID, required=True)
        sub.add_argument("--passport", type=Path, required=True)
        sub.add_argument("--transcripts", type=Path, required=True)
        if name == "index":
            sub.add_argument("--dry-run", action="store_true")
            sub.add_argument("--confirm-mapping")
    sub = commands.add_parser("search")
    sub.add_argument("--collection", type=uuid.UUID, required=True)
    sub.add_argument("--query", required=True)
    sub.add_argument("--limit", type=int, default=10)
    sub.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args(argv)
    dsn = os.environ.get("MEMSEARCH_DSN")
    if not dsn:
        print("MEMSEARCH_DSN must be set explicitly", file=sys.stderr)
        return 2
    if args.command == "index" and not args.dry_run and not args.confirm_mapping:
        print("index requires --confirm-mapping from plan", file=sys.stderr)
        return 2
    if args.command == "search" and (
        not args.query.strip() or not 1 <= args.limit <= 100 or not 1 <= args.repeat <= 100
    ):
        print("invalid query, limit or repeat", file=sys.stderr)
        return 2
    try:
        report = asyncio.run(run_command(args, dsn))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
