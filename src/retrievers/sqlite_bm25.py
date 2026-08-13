"""Disk-backed global BM25 for corpora that must not be materialized in RAM.

``sqlite_bm25_v1`` is intentionally independent from the project's BM25S
backend.  Construction streams the canonical ``chunks.jsonl`` once, retains
only one document's :class:`collections.Counter` in Python, and persists all
global state in SQLite.  Retrieval aggregates Lucene-style BM25 scores in SQL
over the complete corpus.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.persistence.artifact_io import (
    atomic_write_json_object,
    decode_chunk_record_line,
    read_json_object,
)
from src.persistence.artifact_validation import VerifiedBuild
from src.provenance import json_sha256, sha256_file
from src.records import ChunkRecord, RetrievalTrace, SearchHit
from src.query_plan import validate_k
from src.retrievers.chunk_store import ChunkStore, as_chunk_store


BACKEND = "sqlite_bm25_v1"
ANALYZER = "english_regex_casefold_v1"
SCHEMA_VERSION = 1
DATABASE_FILE = "index.sqlite3"
MANIFEST_FILE = "manifest.json"
CHECKPOINT_FILE = "checkpoint.json"

_TOKEN_PATTERN_TEXT = r"[a-z0-9]+(?:['’][a-z0-9]+)?"
_TOKEN_PATTERN = re.compile(_TOKEN_PATTERN_TEXT, flags=re.ASCII | re.IGNORECASE)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# This list is deliberately owned and versioned by sqlite_bm25_v1.  It is not
# inherited from bm25s, NLTK, scikit-learn, or the host environment.
_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

_ANALYZER_SPEC = {
    "name": ANALYZER,
    "casefold": True,
    "token_pattern": _TOKEN_PATTERN_TEXT,
}

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE docs (
        vector_id INTEGER PRIMARY KEY,
        length INTEGER NOT NULL CHECK (length >= 0)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE postings (
        term TEXT NOT NULL,
        vector_id INTEGER NOT NULL,
        tf INTEGER NOT NULL CHECK (tf > 0),
        PRIMARY KEY (term, vector_id),
        FOREIGN KEY (vector_id) REFERENCES docs(vector_id)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE term_stats (
        term TEXT PRIMARY KEY,
        df INTEGER NOT NULL CHECK (df > 0),
        idf REAL NOT NULL CHECK (idf > 0.0)
    ) WITHOUT ROWID
    """,
)

_SEARCH_SQL = """
SELECT
    postings.vector_id AS vector_id,
    SUM(
        term_stats.idf
        * (
            postings.tf * (? + 1.0)
            / (
                postings.tf
                + ? * (
                    1.0 - ?
                    + ? * docs.length / CASE WHEN ? > 0.0 THEN ? ELSE 1.0 END
                )
            )
        )
    ) AS score
FROM temp.query_terms
JOIN main.postings ON postings.term = query_terms.term
JOIN main.term_stats ON term_stats.term = postings.term
JOIN main.docs ON docs.vector_id = postings.vector_id
GROUP BY postings.vector_id
HAVING score > 0.0
ORDER BY score DESC, postings.vector_id ASC
LIMIT ?
"""


@dataclass(frozen=True, slots=True)
class SQLiteBM25Artifact:
    directory: Path
    manifest: dict[str, Any]
    database: Path


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_parameters(k1: Any, b: Any) -> tuple[float, float]:
    if isinstance(k1, bool) or not isinstance(k1, (int, float)):
        raise ValueError("k1 must be a positive finite number")
    if isinstance(b, bool) or not isinstance(b, (int, float)):
        raise ValueError("b must be a finite number between 0 and 1")
    k1_value = float(k1)
    b_value = float(b)
    if not math.isfinite(k1_value) or k1_value <= 0.0:
        raise ValueError("k1 must be a positive finite number")
    if not math.isfinite(b_value) or not 0.0 <= b_value <= 1.0:
        raise ValueError("b must be a finite number between 0 and 1")
    return k1_value, b_value


def _analyze(text: str) -> Iterator[str]:
    folded = text.casefold()
    for match in _TOKEN_PATTERN.finditer(folded):
        token = match.group(0)
        if token not in _ENGLISH_STOPWORDS:
            yield token


def _source_identity(verified_build: VerifiedBuild) -> tuple[Path, dict[str, Any]]:
    if not isinstance(verified_build, VerifiedBuild):
        raise TypeError("verified_build must be a VerifiedBuild")
    chunks_path = Path(verified_build.files["chunks"]).resolve()
    descriptor = verified_build.manifest.get("artifacts", {}).get("chunks")
    if not isinstance(descriptor, Mapping):
        raise ValueError("Verified build has no chunk artifact descriptor")
    rows = _positive_integer("source chunk rows", descriptor.get("rows"))
    size_bytes = descriptor.get("size_bytes")
    sha256 = descriptor.get("sha256")
    filename = descriptor.get("file")
    if (
        not isinstance(filename, str)
        or not filename
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or not isinstance(sha256, str)
        or _SHA256_PATTERN.fullmatch(sha256) is None
    ):
        raise ValueError("Verified build chunk descriptor is invalid")
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk artifact is missing: {chunks_path}")
    if chunks_path.stat().st_size != size_bytes or sha256_file(chunks_path) != sha256:
        raise ValueError("Chunk artifact no longer matches its verified descriptor")
    source_build_id = verified_build.manifest.get("build_id")
    if not isinstance(source_build_id, str) or not source_build_id:
        raise ValueError("Verified build has no build identity")
    return chunks_path, {
        "file": filename,
        "rows": rows,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def sqlite_bm25_identity(
    verified_build: VerifiedBuild,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    analyzer: str = ANALYZER,
) -> tuple[str, str, dict[str, Any]]:
    """Return the stable identity bound to chunks, analyzer, and BM25 params."""

    if analyzer != ANALYZER:
        raise ValueError(f"analyzer must be {ANALYZER}")
    k1_value, b_value = _validated_parameters(k1, b)
    _, chunks = _source_identity(verified_build)
    spec = {
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND,
        "implementation_sha256": sha256_file(Path(__file__)),
        "source_build_id": verified_build.manifest["build_id"],
        "source_chunks": chunks,
        "analyzer": dict(_ANALYZER_SPEC),
        "bm25": {
            "method": "lucene",
            "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))",
            "k1": k1_value,
            "b": b_value,
        },
    }
    digest = json_sha256(spec)
    return f"sqlite_bm25_{digest[:16]}", digest, spec


def _metadata_values(connection: sqlite3.Connection) -> dict[str, Any]:
    try:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError("SQLite BM25 metadata table is invalid") from exc
    values: dict[str, Any] = {}
    for key, raw in rows:
        if not isinstance(key, str) or not isinstance(raw, str):
            raise ValueError("SQLite BM25 metadata entries must be strings")
        try:
            values[key] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SQLite BM25 metadata value is invalid: {key}") from exc
    return values


def _set_metadata(connection: sqlite3.Connection, **values: Any) -> None:
    connection.executemany(
        """
        INSERT INTO metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (
            (
                key,
                json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False),
            )
            for key, value in values.items()
        ),
    )


def _configure_build_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=30000")
    mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if str(mode).casefold() != "wal":
        raise RuntimeError("SQLite BM25 build requires WAL journal mode")
    connection.execute("PRAGMA wal_autocheckpoint=1000")


def _checkpoint_value(
    identity: Mapping[str, Any],
    identity_sha256: str,
    *,
    next_vector_id: int,
    byte_offset: int,
    total_length: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "building",
        "backend": BACKEND,
        "identity_sha256": identity_sha256,
        "identity": dict(identity),
        "next_vector_id": next_vector_id,
        "byte_offset": byte_offset,
        "total_length": total_length,
        "database_file": DATABASE_FILE,
    }


def _validate_empty_database_checkpoint(
    checkpoint_path: Path,
    identity: Mapping[str, Any],
    identity_sha256: str,
) -> None:
    if not checkpoint_path.exists():
        return
    checkpoint = read_json_object(checkpoint_path, label="SQLite BM25 checkpoint")
    if (
        checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("status") != "building"
        or checkpoint.get("backend") != BACKEND
        or checkpoint.get("identity_sha256") != identity_sha256
        or checkpoint.get("identity") != dict(identity)
        or checkpoint.get("database_file") != DATABASE_FILE
        or checkpoint.get("next_vector_id") != 0
        or checkpoint.get("byte_offset") != 0
        or checkpoint.get("total_length") != 0
    ):
        raise ValueError("SQLite BM25 checkpoint cannot initialize an empty database")


def _initialize_database(
    connection: sqlite3.Connection,
    identity: Mapping[str, Any],
    identity_sha256: str,
) -> tuple[int, int, int]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        _set_metadata(
            connection,
            schema_version=SCHEMA_VERSION,
            backend=BACKEND,
            status="building",
            identity_sha256=identity_sha256,
            identity=dict(identity),
            next_vector_id=0,
            byte_offset=0,
            total_length=0,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return 0, 0, 0


def _integer_state(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"SQLite BM25 {name} is invalid")
    return value


def _resume_database(
    connection: sqlite3.Connection,
    identity: Mapping[str, Any],
    identity_sha256: str,
    checkpoint_path: Path,
) -> tuple[int, int, int]:
    metadata = _metadata_values(connection)
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("backend") != BACKEND
        or metadata.get("identity_sha256") != identity_sha256
        or metadata.get("identity") != dict(identity)
        or metadata.get("status") not in {"building", "complete"}
    ):
        raise ValueError("SQLite BM25 partial database identity does not match this build")
    next_vector_id = _integer_state("next_vector_id", metadata.get("next_vector_id"))
    byte_offset = _integer_state("byte_offset", metadata.get("byte_offset"))
    total_length = _integer_state("total_length", metadata.get("total_length"))

    actual = connection.execute(
        "SELECT COUNT(*), COALESCE(MIN(vector_id), -1), "
        "COALESCE(MAX(vector_id), -1), COALESCE(SUM(length), 0) FROM docs"
    ).fetchone()
    expected_min = 0 if next_vector_id else -1
    expected_max = next_vector_id - 1
    if actual != (next_vector_id, expected_min, expected_max, total_length):
        raise ValueError("SQLite BM25 database progress is inconsistent")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ValueError("SQLite BM25 partial database has invalid postings")

    if checkpoint_path.exists():
        checkpoint = read_json_object(
            checkpoint_path,
            label="SQLite BM25 checkpoint",
        )
        if (
            checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("backend") != BACKEND
            or checkpoint.get("status") != "building"
            or checkpoint.get("identity_sha256") != identity_sha256
            or checkpoint.get("identity") != dict(identity)
            or checkpoint.get("database_file") != DATABASE_FILE
        ):
            raise ValueError("SQLite BM25 checkpoint identity is incompatible")
        checkpoint_next = _integer_state(
            "checkpoint next_vector_id",
            checkpoint.get("next_vector_id"),
        )
        if checkpoint_next > next_vector_id:
            raise ValueError("SQLite BM25 checkpoint is ahead of the committed database")
        if checkpoint_next == next_vector_id and (
            checkpoint.get("byte_offset") != byte_offset
            or checkpoint.get("total_length") != total_length
        ):
            raise ValueError("SQLite BM25 checkpoint progress is inconsistent")
    return next_vector_id, byte_offset, total_length


def _index_record(connection: sqlite3.Connection, record: ChunkRecord) -> int:
    frequencies = Counter(_analyze(record.text))
    length = sum(frequencies.values())
    connection.execute(
        "INSERT INTO docs(vector_id, length) VALUES (?, ?)",
        (record.vector_id, length),
    )
    connection.executemany(
        "INSERT INTO postings(term, vector_id, tf) VALUES (?, ?, ?)",
        (
            (term, record.vector_id, frequency)
            for term, frequency in frequencies.items()
        ),
    )
    return length


def _commit_progress(
    connection: sqlite3.Connection,
    checkpoint_path: Path,
    identity: Mapping[str, Any],
    identity_sha256: str,
    *,
    next_vector_id: int,
    byte_offset: int,
    total_length: int,
) -> None:
    _set_metadata(
        connection,
        status="building",
        next_vector_id=next_vector_id,
        byte_offset=byte_offset,
        total_length=total_length,
    )
    connection.commit()
    atomic_write_json_object(
        checkpoint_path,
        _checkpoint_value(
            identity,
            identity_sha256,
            next_vector_id=next_vector_id,
            byte_offset=byte_offset,
            total_length=total_length,
        ),
    )


def _stream_chunks_into_database(
    connection: sqlite3.Connection,
    chunks_path: Path,
    checkpoint_path: Path,
    identity: Mapping[str, Any],
    identity_sha256: str,
    *,
    transaction_documents: int,
    next_vector_id: int,
    byte_offset: int,
    total_length: int,
) -> tuple[int, int, int]:
    expected_rows = int(identity["source_chunks"]["rows"])
    if next_vector_id > expected_rows or byte_offset > chunks_path.stat().st_size:
        raise ValueError("SQLite BM25 checkpoint exceeds the source corpus")

    with chunks_path.open("rb") as handle:
        handle.seek(byte_offset)
        connection.execute("BEGIN IMMEDIATE")
        pending = 0
        try:
            while next_vector_id < expected_rows:
                raw = handle.readline()
                record = decode_chunk_record_line(raw, next_vector_id)
                total_length += _index_record(connection, record)
                next_vector_id += 1
                byte_offset = handle.tell()
                pending += 1
                if pending >= transaction_documents:
                    _commit_progress(
                        connection,
                        checkpoint_path,
                        identity,
                        identity_sha256,
                        next_vector_id=next_vector_id,
                        byte_offset=byte_offset,
                        total_length=total_length,
                    )
                    connection.execute("BEGIN IMMEDIATE")
                    pending = 0

            for trailing in handle:
                if trailing.strip():
                    raise ValueError("Chunk artifact contains more rows than its manifest")
            _commit_progress(
                connection,
                checkpoint_path,
                identity,
                identity_sha256,
                next_vector_id=next_vector_id,
                byte_offset=byte_offset,
                total_length=total_length,
            )
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
    return next_vector_id, byte_offset, total_length


def _finalize_database(
    connection: sqlite3.Connection,
    *,
    document_count: int,
    total_length: int,
) -> float:
    average_length = total_length / document_count

    def lucene_idf(df: int) -> float:
        return math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))

    connection.create_function("lucene_idf", 1, lucene_idf, deterministic=True)
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("DELETE FROM term_stats")
        connection.execute(
            """
            INSERT INTO term_stats(term, df, idf)
            SELECT term, COUNT(*), 1.0
            FROM postings
            GROUP BY term
            """
        )
        connection.execute("UPDATE term_stats SET idf=lucene_idf(df)")
        invalid = connection.execute(
            "SELECT 1 FROM term_stats "
            "WHERE df <= 0 OR df > ? OR idf <= 0.0 LIMIT 1",
            (document_count,),
        ).fetchone()
        if invalid is not None:
            raise ValueError("SQLite BM25 term statistics are invalid")
        _set_metadata(
            connection,
            status="complete",
            document_count=document_count,
            total_length=total_length,
            average_document_length=average_length,
        )
        connection.execute("ANALYZE")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return average_length


def _close_final_database(connection: sqlite3.Connection) -> None:
    checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint is None or int(checkpoint[0]) != 0:
        raise RuntimeError("Could not checkpoint the SQLite BM25 WAL")
    mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
    if str(mode).casefold() != "delete":
        raise RuntimeError("Could not finalize the SQLite BM25 journal")
    connection.close()


def _database_descriptor(database: Path) -> dict[str, Any]:
    return {
        "file": DATABASE_FILE,
        "size_bytes": database.stat().st_size,
        "sha256": sha256_file(database),
    }


def build_sqlite_bm25_index(
    verified_build: VerifiedBuild,
    index_dir: str | Path,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    analyzer: str = ANALYZER,
    transaction_documents: int = 1000,
) -> SQLiteBM25Artifact:
    """Build or resume one global, disk-backed exact BM25 index."""

    transaction_documents = _positive_integer(
        "transaction_documents",
        transaction_documents,
    )
    sparse_index_id, identity_sha256, identity = sqlite_bm25_identity(
        verified_build,
        k1=k1,
        b=b,
        analyzer=analyzer,
    )
    chunks_path = Path(verified_build.files["chunks"]).resolve()
    directory = Path(index_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST_FILE
    checkpoint_path = directory / CHECKPOINT_FILE
    database = directory / DATABASE_FILE

    if manifest_path.exists():
        return validate_sqlite_bm25_index(
            directory,
            expected_identity_sha256=identity_sha256,
        )
    if checkpoint_path.exists() and not database.exists():
        raise ValueError("SQLite BM25 checkpoint exists without its database")

    new_database = not database.exists() or database.stat().st_size == 0
    connection = sqlite3.connect(database, isolation_level=None)
    closed = False
    started = time.perf_counter()
    try:
        _configure_build_connection(connection)
        if new_database:
            _validate_empty_database_checkpoint(
                checkpoint_path,
                identity,
                identity_sha256,
            )
            next_vector_id, byte_offset, total_length = _initialize_database(
                connection,
                identity,
                identity_sha256,
            )
        else:
            next_vector_id, byte_offset, total_length = _resume_database(
                connection,
                identity,
                identity_sha256,
                checkpoint_path,
            )
        atomic_write_json_object(
            checkpoint_path,
            _checkpoint_value(
                identity,
                identity_sha256,
                next_vector_id=next_vector_id,
                byte_offset=byte_offset,
                total_length=total_length,
            ),
        )

        next_vector_id, byte_offset, total_length = _stream_chunks_into_database(
            connection,
            chunks_path,
            checkpoint_path,
            identity,
            identity_sha256,
            transaction_documents=transaction_documents,
            next_vector_id=next_vector_id,
            byte_offset=byte_offset,
            total_length=total_length,
        )
        expected_rows = int(identity["source_chunks"]["rows"])
        if next_vector_id != expected_rows:
            raise RuntimeError("SQLite BM25 build did not index every source document")
        if sha256_file(chunks_path) != identity["source_chunks"]["sha256"]:
            raise RuntimeError("Source chunks changed while SQLite BM25 was being built")

        average_length = _finalize_database(
            connection,
            document_count=expected_rows,
            total_length=total_length,
        )
        _close_final_database(connection)
        closed = True

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "backend": BACKEND,
            "sparse_index_id": sparse_index_id,
            "identity_sha256": identity_sha256,
            "identity": identity,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "document_count": expected_rows,
            "total_document_length": total_length,
            "average_document_length": average_length,
            "artifacts": {"database": _database_descriptor(database)},
            "timings_ms": {
                "total_before_manifest": (time.perf_counter() - started) * 1000
            },
        }
        atomic_write_json_object(manifest_path, manifest)
        checkpoint_path.unlink(missing_ok=True)
        return validate_sqlite_bm25_index(
            directory,
            expected_identity_sha256=identity_sha256,
        )
    finally:
        if not closed:
            connection.close()


def _open_read_only(database: Path) -> sqlite3.Connection:
    uri = f"{database.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def _required_table_columns(connection: sqlite3.Connection) -> None:
    expected = {
        "metadata": ["key", "value"],
        "docs": ["vector_id", "length"],
        "postings": ["term", "vector_id", "tf"],
        "term_stats": ["term", "df", "idf"],
    }
    for table, columns in expected.items():
        actual = [
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        ]
        if actual != columns:
            raise ValueError(f"SQLite BM25 table schema is invalid: {table}")


def validate_sqlite_bm25_index(
    index_dir: str | Path,
    *,
    expected_identity_sha256: str | None = None,
) -> SQLiteBM25Artifact:
    """Validate the complete manifest, immutable database, and global stats."""

    directory = Path(index_dir).resolve()
    manifest = read_json_object(
        directory / MANIFEST_FILE,
        label="SQLite BM25 manifest",
    )
    manifest_schema = manifest.get("schema_version")
    if (
        isinstance(manifest_schema, bool)
        or not isinstance(manifest_schema, int)
        or manifest_schema != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("backend") != BACKEND
    ):
        raise ValueError("SQLite BM25 manifest is incomplete or incompatible")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("SQLite BM25 manifest has no identity")
    identity_sha256 = json_sha256(identity)
    if manifest.get("identity_sha256") != identity_sha256:
        raise ValueError("SQLite BM25 manifest identity hash is invalid")
    if (
        expected_identity_sha256 is not None
        and identity_sha256 != expected_identity_sha256
    ):
        raise ValueError("SQLite BM25 identity does not match the requested build")
    if manifest.get("sparse_index_id") != f"sqlite_bm25_{identity_sha256[:16]}":
        raise ValueError("SQLite BM25 sparse index id is not derived from its identity")
    analyzer = identity.get("analyzer")
    if (
        identity.get("schema_version") != SCHEMA_VERSION
        or identity.get("backend") != BACKEND
        or not isinstance(analyzer, Mapping)
        or any(analyzer.get(key) != value for key, value in _ANALYZER_SPEC.items())
        or not set(analyzer)
        <= set(_ANALYZER_SPEC) | {"stopwords_count", "stopwords_sha256"}
    ):
        raise ValueError("SQLite BM25 identity backend or analyzer is invalid")
    bm25 = identity.get("bm25")
    source_chunks = identity.get("source_chunks")
    if not isinstance(bm25, Mapping) or not isinstance(source_chunks, Mapping):
        raise ValueError("SQLite BM25 identity is incomplete")
    k1, b = _validated_parameters(bm25.get("k1"), bm25.get("b"))
    if (
        bm25.get("method") != "lucene"
        or bm25.get("idf") != "ln(1 + (N - df + 0.5) / (df + 0.5))"
    ):
        raise ValueError("SQLite BM25 identity must use Lucene BM25")
    rows = _positive_integer("SQLite BM25 source rows", source_chunks.get("rows"))
    source_size = source_chunks.get("size_bytes")
    source_sha = source_chunks.get("sha256")
    if (
        isinstance(source_size, bool)
        or not isinstance(source_size, int)
        or source_size <= 0
        or not isinstance(source_sha, str)
        or _SHA256_PATTERN.fullmatch(source_sha) is None
    ):
        raise ValueError("SQLite BM25 source chunk identity is invalid")
    if manifest.get("document_count") != rows:
        raise ValueError("SQLite BM25 document count does not match its source")

    artifacts = manifest.get("artifacts")
    database_descriptor = (
        artifacts.get("database") if isinstance(artifacts, Mapping) else None
    )
    if (
        not isinstance(database_descriptor, Mapping)
        or database_descriptor.get("file") != DATABASE_FILE
    ):
        raise ValueError("SQLite BM25 database descriptor is invalid")
    database = directory / DATABASE_FILE
    if database.is_symlink() or not database.is_file():
        raise FileNotFoundError(f"SQLite BM25 database is missing: {database}")
    if any(
        database.with_name(f"{database.name}{suffix}").exists()
        for suffix in ("-wal", "-shm")
    ):
        raise ValueError("Completed SQLite BM25 database must not depend on WAL sidecars")
    expected_size = database_descriptor.get("size_bytes")
    expected_sha256 = database_descriptor.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
        or database.stat().st_size != expected_size
        or sha256_file(database) != expected_sha256
    ):
        raise ValueError("SQLite BM25 database hash or size is invalid")

    connection = _open_read_only(database)
    try:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("SQLite BM25 database integrity check failed")
        _required_table_columns(connection)
        metadata = _metadata_values(connection)
        average_length = manifest.get("average_document_length")
        total_length = manifest.get("total_document_length")
        if (
            isinstance(total_length, bool)
            or not isinstance(total_length, int)
            or total_length < 0
            or isinstance(average_length, bool)
            or not isinstance(average_length, (int, float))
            or not math.isfinite(float(average_length))
            or float(average_length) < 0.0
        ):
            raise ValueError("SQLite BM25 global length statistics are invalid")
        if (
            metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("backend") != BACKEND
            or metadata.get("status") != "complete"
            or metadata.get("identity_sha256") != identity_sha256
            or metadata.get("identity") != dict(identity)
            or metadata.get("document_count") != rows
            or metadata.get("next_vector_id") != rows
            or metadata.get("total_length") != total_length
            or metadata.get("average_document_length") != average_length
        ):
            raise ValueError("SQLite BM25 database metadata is inconsistent")
        stats = connection.execute(
            "SELECT COUNT(*), COALESCE(MIN(vector_id), -1), "
            "COALESCE(MAX(vector_id), -1), COALESCE(SUM(length), 0) FROM docs"
        ).fetchone()
        if stats != (rows, 0, rows - 1, total_length):
            raise ValueError("SQLite BM25 document statistics are inconsistent")
        invalid_terms = connection.execute(
            "SELECT 1 FROM term_stats "
            "WHERE df <= 0 OR df > ? OR idf <= 0.0 LIMIT 1",
            (rows,),
        ).fetchone()
        if invalid_terms is not None:
            raise ValueError("SQLite BM25 term statistics are invalid")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("SQLite BM25 postings reference unknown documents")
        if not math.isclose(
            float(average_length),
            float(total_length) / rows,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("SQLite BM25 average document length is invalid")
        # Keep variables live in this validation scope to prove they parsed.
        _ = (k1, b)
    except sqlite3.DatabaseError as exc:
        raise ValueError("SQLite BM25 database is invalid") from exc
    finally:
        connection.close()
    return SQLiteBM25Artifact(
        directory=directory,
        manifest=manifest,
        database=database,
    )


class SQLiteBM25Retriever:
    """Read-only SQL BM25 aggregation mapped through the canonical ChunkStore."""

    def __init__(
        self,
        chunks: ChunkStore | Iterable[ChunkRecord | dict],
        index: str | Path | SQLiteBM25Artifact,
        *,
        top_k: int = 5,
        sparse_index_id: str | None = None,
    ) -> None:
        top_k = validate_k(top_k)
        artifact = (
            index
            if isinstance(index, SQLiteBM25Artifact)
            else validate_sqlite_bm25_index(index)
        )
        if (
            sparse_index_id is not None
            and artifact.manifest.get("sparse_index_id") != sparse_index_id
        ):
            raise ValueError("SQLite BM25 sparse index id does not match")
        store = as_chunk_store(chunks)
        document_count = int(artifact.manifest["document_count"])
        if len(store) != document_count:
            raise ValueError(
                f"Chunk count {len(store)} does not match SQLite BM25 "
                f"document count {document_count}"
            )

        bm25 = artifact.manifest["identity"]["bm25"]
        self.chunk_store = store
        self.top_k = top_k
        self.k1 = float(bm25["k1"])
        self.b = float(bm25["b"])
        self.average_document_length = float(
            artifact.manifest["average_document_length"]
        )
        self.sparse_index_id = str(artifact.manifest["sparse_index_id"])
        self.backend = BACKEND
        self._connection = _open_read_only(artifact.database)
        self._connection.execute(
            "CREATE TEMP TABLE query_terms(term TEXT PRIMARY KEY) WITHOUT ROWID"
        )

    @classmethod
    def load(
        cls,
        chunks: ChunkStore | Iterable[ChunkRecord | dict],
        index: str | Path | SQLiteBM25Artifact,
        *,
        top_k: int = 5,
        sparse_index_id: str | None = None,
    ) -> "SQLiteBM25Retriever":
        return cls(
            chunks,
            index,
            top_k=top_k,
            sparse_index_id=sparse_index_id,
        )

    def retrieve_trace(
        self,
        query: str,
        top_k: int | None = None,
        *,
        search_params: Mapping[str, Any] | None = None,
    ) -> RetrievalTrace:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        effective_top_k = self.top_k if top_k is None else top_k
        effective_top_k = validate_k(effective_top_k)
        if search_params:
            raise ValueError("SQLite BM25 retrieval does not accept ANN search parameters")
        if effective_top_k == 0:
            return RetrievalTrace(
                top_k=0,
                results=(),
                timings_ms={
                    "query_tokenization_ms": 0.0,
                    "sparse_search_ms": 0.0,
                    "chunk_mapping_ms": 0.0,
                    "total_ms": 0.0,
                },
            )

        started = time.perf_counter()
        tokenization_started = time.perf_counter()
        terms = sorted(set(_analyze(query.strip())))
        tokenization_ms = (time.perf_counter() - tokenization_started) * 1000

        search_started = time.perf_counter()
        ranked: list[tuple[int, float]] = []
        if terms:
            self._connection.execute("DELETE FROM temp.query_terms")
            self._connection.executemany(
                "INSERT INTO temp.query_terms(term) VALUES (?)",
                ((term,) for term in terms),
            )
            rows = self._connection.execute(
                _SEARCH_SQL,
                (
                    self.k1,
                    self.k1,
                    self.b,
                    self.b,
                    self.average_document_length,
                    self.average_document_length,
                    min(effective_top_k, len(self.chunk_store)),
                ),
            ).fetchall()
            ranked = [
                (int(vector_id), float(score))
                for vector_id, score in rows
                if math.isfinite(float(score)) and float(score) > 0.0
            ]
        search_ms = (time.perf_counter() - search_started) * 1000

        vector_ids = [vector_id for vector_id, _ in ranked]
        if len(vector_ids) != len(set(vector_ids)):
            raise ValueError("SQLite BM25 returned duplicate document ids")
        mapping_started = time.perf_counter()
        try:
            records = self.chunk_store.get_many(vector_ids)
        except KeyError as exc:
            raise ValueError(
                f"SQLite BM25 returned unknown vector id: {exc.args[0]}"
            ) from exc
        results = tuple(
            SearchHit(rank=rank, chunk=chunk, score=score)
            for rank, ((_, score), chunk) in enumerate(zip(ranked, records), start=1)
        )
        mapping_ms = (time.perf_counter() - mapping_started) * 1000
        return RetrievalTrace(
            top_k=effective_top_k,
            results=results,
            timings_ms={
                "query_tokenization_ms": tokenization_ms,
                "sparse_search_ms": search_ms,
                "chunk_mapping_ms": mapping_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "SQLiteBM25Retriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
