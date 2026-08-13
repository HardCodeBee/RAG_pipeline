"""Prepare selected BEIR archives as immutable, independently loadable units.

Official corpus and qrels rows are preserved in a structured layout.  Regular
query files are copied exactly; CQADupStack queries are reduced to their BEIR
``_id``/``text`` contract because the official archive repeats a global
metadata sidecar in every row.  Validation uses a temporary on-disk SQLite
database so corpus and query identifiers are never accumulated in Python
memory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import stat
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.persistence.artifact_io import (
    describe_artifact,
    read_json_object,
    write_manifest,
)
from src.persistence.artifact_validation import verify_artifact_descriptor
from src.provenance import sha256_file


PROTOCOL = "beir_corpus_queries_v2"
MANIFEST_SCHEMA_VERSION = 2
TEXT_FORMAT = "title_newline_text_v1"
CQA_QUERY_FORMAT = "beir_id_text_only_v1"
DEFAULT_OUTPUT_ROOT = Path("data/beir")
OFFICIAL_BASE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
)
CQADUPSTACK_FORUMS = (
    "android",
    "english",
    "gaming",
    "gis",
    "mathematica",
    "physics",
    "programmers",
    "stats",
    "tex",
    "unix",
    "webmasters",
    "wordpress",
)


@dataclass(frozen=True, slots=True)
class BeirDatasetSpec:
    """Immutable identity of one selected official BEIR archive."""

    name: str
    official_md5: str

    @property
    def archive_name(self) -> str:
        return f"{self.name}.zip"

    @property
    def url(self) -> str:
        return f"{OFFICIAL_BASE_URL}/{self.archive_name}"


BEIR_DATASETS: dict[str, BeirDatasetSpec] = {
    "nq": BeirDatasetSpec("nq", "d4d3d2e48787a744b6f6e691ff534307"),
    "hotpotqa": BeirDatasetSpec(
        "hotpotqa", "f412724f78b0d91183a0e86805e16114"
    ),
    "fiqa": BeirDatasetSpec("fiqa", "17918ed23cd04fb15047f73e6c3bd9d9"),
    "nfcorpus": BeirDatasetSpec(
        "nfcorpus", "a89dba18a62ef92f7d323ec890a0d38d"
    ),
    "arguana": BeirDatasetSpec(
        "arguana", "8ad3e3c2a5867cdced806d6503f29b99"
    ),
    "webis-touche2020": BeirDatasetSpec(
        "webis-touche2020", "46f650ba5a527fc69e0a6521c5a23563"
    ),
    "cqadupstack": BeirDatasetSpec(
        "cqadupstack", "4e41456d7df8ee7760a7f866133bda78"
    ),
}

_DATASET_ALIASES = {
    "fiqa-2018": "fiqa",
    "touche": "webis-touche2020",
    "touche-2020": "webis-touche2020",
    "touché": "webis-touche2020",
    "touché-2020": "webis-touche2020",
}


def canonical_dataset_name(value: str) -> str:
    """Return the BEIR archive name accepted by the local integration."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("dataset must be a non-empty string")
    normalized = value.strip().lower()
    canonical = _DATASET_ALIASES.get(normalized, normalized)
    if canonical not in BEIR_DATASETS:
        choices = ", ".join(sorted(BEIR_DATASETS))
        raise ValueError(f"Unsupported BEIR dataset {value!r}; choose one of: {choices}")
    return canonical


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: str | Path) -> str:
    """Compute the archive checksum incrementally without loading it in RAM."""

    return _hash_file(Path(path), "md5")


def _validate_archive(path: Path, spec: BeirDatasetSpec) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"BEIR archive does not exist: {path}")
    actual_md5 = md5_file(path)
    if actual_md5.lower() != spec.official_md5.lower():
        raise ValueError(
            f"BEIR archive MD5 mismatch for {spec.name}: "
            f"expected {spec.official_md5}, found {actual_md5}"
        )
    return {
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "md5": actual_md5,
        "sha256": sha256_file(path),
    }


def _download_archive(spec: BeirDatasetSpec, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_archive(destination, spec)
        return destination
    partial = destination.with_name(f".{destination.name}.part")
    if partial.exists():
        partial.unlink()
    try:
        with urllib.request.urlopen(spec.url) as response, partial.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
        _validate_archive(partial, spec)
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract regular files while rejecting traversal and archive symlinks."""

    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            member_path = Path(member.filename.replace("\\", "/"))
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in BEIR archive: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symlink is not allowed in BEIR archive: {member.filename}")
            target = (destination / member_path).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Archive member escapes extraction root: {member.filename}"
                ) from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_handle, target.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def _unit_source_directories(extracted: Path, dataset: str) -> dict[str, Path]:
    candidates = sorted(
        {
            path.parent.resolve()
            for path in extracted.rglob("corpus.jsonl")
            if (path.parent / "queries.jsonl").is_file()
            and (path.parent / "qrels").is_dir()
        },
        key=lambda path: path.as_posix(),
    )
    if dataset != "cqadupstack":
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one BEIR unit in {dataset} archive; found {len(candidates)}"
            )
        return {dataset: candidates[0]}

    by_forum = {path.name.lower(): path for path in candidates}
    missing = [forum for forum in CQADUPSTACK_FORUMS if forum not in by_forum]
    extras = sorted(set(by_forum) - set(CQADUPSTACK_FORUMS))
    if missing or extras:
        raise ValueError(
            "CQADupStack archive does not contain exactly the 12 expected forums; "
            f"missing={missing}, extras={extras}"
        )
    return {forum: by_forum[forum] for forum in CQADUPSTACK_FORUMS}


def _jsonl_objects(path: Path, label: str) -> Iterator[tuple[int, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid {label} JSON at line {line_number}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{label} line {line_number} must be a JSON object")
            yield line_number, value


def _insert_id_batch(
    connection: sqlite3.Connection,
    table: str,
    values: list[tuple[str]],
    label: str,
) -> None:
    if not values:
        return
    try:
        connection.executemany(f"INSERT INTO {table}(id) VALUES (?)", values)
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"{label} contains a duplicate _id") from exc
    values.clear()


def _insert_corpus_batch(
    connection: sqlite3.Connection,
    values: list[tuple[str, int]],
) -> None:
    if not values:
        return
    try:
        connection.executemany(
            "INSERT INTO corpus_ids(id, retrievable) VALUES (?, ?)", values
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError("corpus contains a duplicate _id") from exc
    values.clear()


def _id_query_sha256(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[Any, ...] = (),
) -> str:
    """Hash a sorted SQLite id result as a canonical JSON string array."""

    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for (raw_id,) in connection.execute(query, parameters):
        if not first:
            digest.update(b",")
        digest.update(
            json.dumps(
                raw_id,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _index_and_filter_corpus(
    connection: sqlite3.Connection,
    source_path: Path,
    destination_path: Path,
) -> dict[str, Any]:
    source_rows = 0
    retrievable_rows = 0
    excluded_empty_rows = 0
    pending: list[tuple[str, int]] = []
    source_digest = hashlib.sha256()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source, destination_path.open("wb") as destination:
        for line_number, raw_line in enumerate(source, start=1):
            source_digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid corpus JSON at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"corpus line {line_number} must be a JSON object")
            raw_id = row.get("_id")
            title = row.get("title", "")
            text = row.get("text", "")
            if not isinstance(raw_id, str) or not raw_id:
                raise ValueError(f"corpus line {line_number} has an invalid _id")
            if not isinstance(title, str) or not isinstance(text, str):
                raise ValueError(f"corpus line {line_number} title/text must be strings")
            retrievable = bool(title.strip() or text.strip())
            pending.append((raw_id, int(retrievable)))
            source_rows += 1
            if retrievable:
                # Preserve the complete official JSONL row bytes for every
                # retained document; no placeholder text or rewritten ids.
                destination.write(raw_line)
                retrievable_rows += 1
            else:
                excluded_empty_rows += 1
            if len(pending) >= 10_000:
                _insert_corpus_batch(connection, pending)
    _insert_corpus_batch(connection, pending)
    if source_rows == 0:
        raise ValueError("BEIR corpus is empty")
    if retrievable_rows == 0:
        raise ValueError("BEIR corpus has no retrievable non-empty rows")
    return {
        "source_corpus_rows": source_rows,
        "retrievable_corpus_rows": retrievable_rows,
        "excluded_empty_rows": excluded_empty_rows,
        "excluded_empty_ids_sha256": _id_query_sha256(
            connection,
            "SELECT id FROM corpus_ids WHERE retrievable = 0 ORDER BY id",
        ),
        "source_file": {
            "file_name": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": source_digest.hexdigest(),
        },
    }


def _index_query_ids(connection: sqlite3.Connection, path: Path) -> int:
    count = 0
    pending: list[tuple[str]] = []
    for line_number, row in _jsonl_objects(path, "queries"):
        raw_id = row.get("_id")
        text = row.get("text")
        if not isinstance(raw_id, str) or not raw_id:
            raise ValueError(f"queries line {line_number} has an invalid _id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"queries line {line_number} has invalid text")
        pending.append((raw_id,))
        count += 1
        if len(pending) >= 10_000:
            _insert_id_batch(connection, "query_ids", pending, "queries")
    _insert_id_batch(connection, "query_ids", pending, "queries")
    if count == 0:
        raise ValueError("BEIR queries are empty")
    return count


def _validate_qrels(
    connection: sqlite3.Connection,
    path: Path,
    split: str,
) -> dict[str, Any]:
    count = 0
    dangling_qrels = 0
    excluded_empty_qrels = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError(
                f"BEIR qrels {path.name} must have query-id, corpus-id, score header"
            )
        for line_number, row in enumerate(reader, start=2):
            query_id = row["query-id"]
            corpus_id = row["corpus-id"]
            if not query_id or not corpus_id:
                raise ValueError(f"qrels {path.name} line {line_number} has an empty id")
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"qrels {path.name} line {line_number} has an invalid score"
                ) from exc
            if not math.isfinite(score):
                raise ValueError(
                    f"qrels {path.name} line {line_number} has a non-finite score"
                )
            if connection.execute(
                "SELECT 1 FROM query_ids WHERE id = ?", (query_id,)
            ).fetchone() is None:
                raise ValueError(
                    f"qrels {path.name} references absent query id {query_id!r}"
                )
            corpus_row = connection.execute(
                "SELECT retrievable FROM corpus_ids WHERE id = ?", (corpus_id,)
            ).fetchone()
            if corpus_row is None:
                dangling_qrels += 1
                connection.execute(
                    "INSERT OR IGNORE INTO qrel_issues(split, kind, corpus_id) "
                    "VALUES (?, 'dangling', ?)",
                    (split, corpus_id),
                )
            elif int(corpus_row[0]) == 0:
                excluded_empty_qrels += 1
                connection.execute(
                    "INSERT OR IGNORE INTO qrel_issues(split, kind, corpus_id) "
                    "VALUES (?, 'excluded_empty', ?)",
                    (split, corpus_id),
                )
            try:
                connection.execute(
                    "INSERT INTO qrels(split, query_id, corpus_id) VALUES (?, ?, ?)",
                    (split, query_id, corpus_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"qrels {path.name} duplicates query/corpus pair "
                    f"{query_id!r}/{corpus_id!r}"
                ) from exc
            count += 1
    if count == 0:
        raise ValueError(f"BEIR qrels split is empty: {path.name}")
    dangling_ids = int(
        connection.execute(
            "SELECT COUNT(*) FROM qrel_issues "
            "WHERE split = ? AND kind = 'dangling'",
            (split,),
        ).fetchone()[0]
    )
    excluded_empty_ids = int(
        connection.execute(
            "SELECT COUNT(*) FROM qrel_issues "
            "WHERE split = ? AND kind = 'excluded_empty'",
            (split,),
        ).fetchone()[0]
    )
    return {
        "rows": count,
        "dangling_corpus_qrels": dangling_qrels,
        "dangling_corpus_ids": dangling_ids,
        "dangling_corpus_ids_sha256": _id_query_sha256(
            connection,
            "SELECT corpus_id FROM qrel_issues "
            "WHERE split = ? AND kind = 'dangling' ORDER BY corpus_id",
            (split,),
        ),
        "excluded_empty_corpus_qrels": excluded_empty_qrels,
        "excluded_empty_corpus_ids": excluded_empty_ids,
        "excluded_empty_corpus_ids_sha256": _id_query_sha256(
            connection,
            "SELECT corpus_id FROM qrel_issues "
            "WHERE split = ? AND kind = 'excluded_empty' ORDER BY corpus_id",
            (split,),
        ),
    }


def _write_id_text_queries(source: Path, destination: Path) -> None:
    """Drop CQADupStack's repeated global metadata sidecar per query row."""

    with source.open("r", encoding="utf-8") as input_handle, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid queries JSON at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"queries line {line_number} must be a JSON object"
                )
            raw_id = row.get("_id")
            text = row.get("text")
            if not isinstance(raw_id, str) or not raw_id:
                raise ValueError(
                    f"queries line {line_number} has an invalid _id"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"queries line {line_number} has invalid text")
            output_handle.write(
                json.dumps(
                    {"_id": raw_id, "text": text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            output_handle.write("\n")


def _validate_id_text_queries(path: Path, *, expected_rows: int) -> None:
    """Validate the compact CQADupStack query representation."""

    count = 0
    for line_number, row in _jsonl_objects(path, "queries"):
        if set(row) != {"_id", "text"}:
            raise ValueError(
                "Compact CQADupStack queries must contain exactly _id and text; "
                f"invalid keys at line {line_number}"
            )
        raw_id = row["_id"]
        text = row["text"]
        if not isinstance(raw_id, str) or not raw_id:
            raise ValueError(f"queries line {line_number} has an invalid _id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"queries line {line_number} has invalid text")
        count += 1
    if count != expected_rows:
        raise ValueError(
            "Compact CQADupStack query row count mismatch: "
            f"expected {expected_rows}, found {count}"
        )


def _copy_unit_sources(
    source: Path,
    destination: Path,
    *,
    query_format: str | None = None,
) -> dict[str, Path]:
    corpus = destination / "corpus" / "corpus.jsonl"
    queries = destination / "queries" / "queries.jsonl"
    qrels_dir = destination / "qrels"
    corpus.parent.mkdir(parents=True)
    queries.parent.mkdir(parents=True)
    qrels_dir.mkdir(parents=True)
    if query_format is None:
        shutil.copyfile(source / "queries.jsonl", queries)
    elif query_format == CQA_QUERY_FORMAT:
        _write_id_text_queries(source / "queries.jsonl", queries)
    else:
        raise ValueError(f"Unsupported prepared query format: {query_format}")
    qrel_sources = sorted((source / "qrels").glob("*.tsv"), key=lambda p: p.name)
    if not qrel_sources or not any(path.stem == "test" for path in qrel_sources):
        raise ValueError(f"BEIR unit has no test qrels: {source}")
    copied: dict[str, Path] = {}
    for qrel_source in qrel_sources:
        qrel_destination = qrels_dir / qrel_source.name
        shutil.copyfile(qrel_source, qrel_destination)
        copied[qrel_source.stem] = qrel_destination
    return {"corpus": corpus, "queries": queries, **copied}


def _validate_and_describe_unit(
    destination: Path,
    *,
    source_corpus_path: Path,
    dataset: str,
    unit: str,
    archive: Mapping[str, Any],
    spec: BeirDatasetSpec,
) -> dict[str, Any]:
    database = destination / ".validation.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-32768")
        connection.execute(
            "CREATE TABLE corpus_ids("
            "id TEXT PRIMARY KEY, retrievable INTEGER NOT NULL "
            "CHECK(retrievable IN (0, 1))) WITHOUT ROWID"
        )
        connection.execute("CREATE TABLE query_ids(id TEXT PRIMARY KEY) WITHOUT ROWID")
        connection.execute(
            "CREATE TABLE qrels("
            "split TEXT NOT NULL, query_id TEXT NOT NULL, corpus_id TEXT NOT NULL, "
            "PRIMARY KEY(split, query_id, corpus_id)) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE qrel_issues("
            "split TEXT NOT NULL, kind TEXT NOT NULL, corpus_id TEXT NOT NULL, "
            "PRIMARY KEY(split, kind, corpus_id)) WITHOUT ROWID"
        )
        corpus_path = destination / "corpus" / "corpus.jsonl"
        queries_path = destination / "queries" / "queries.jsonl"
        corpus_stats = _index_and_filter_corpus(
            connection,
            source_corpus_path,
            corpus_path,
        )
        query_rows = _index_query_ids(connection, queries_path)
        qrel_stats: dict[str, dict[str, Any]] = {}
        for qrel_path in sorted((destination / "qrels").glob("*.tsv")):
            qrel_stats[qrel_path.stem] = _validate_qrels(
                connection, qrel_path, qrel_path.stem
            )
    finally:
        connection.close()
        if database.exists():
            database.unlink()

    source_manifest = {
        "status": "complete",
        "protocol": PROTOCOL,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "unit": unit,
        "official_url": spec.url,
        "official_md5": spec.official_md5,
        "archive": dict(archive),
        "source_corpus": dict(corpus_stats["source_file"]),
    }
    source_manifest_path = destination / "source_manifest.json"
    write_manifest(source_manifest_path, source_manifest)
    qrel_descriptors = {
        split: describe_artifact(
            destination / "qrels" / f"{split}.tsv",
            relative_to=destination,
            rows=stats["rows"],
            extra={key: value for key, value in stats.items() if key != "rows"},
        )
        for split, stats in sorted(qrel_stats.items())
    }
    manifest = {
        "status": "complete",
        "protocol": PROTOCOL,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "dataset_unit",
        "dataset": dataset,
        "unit": unit,
        "text_format": TEXT_FORMAT,
        "counts": {
            "source_corpus_rows": corpus_stats["source_corpus_rows"],
            "retrievable_corpus_rows": corpus_stats["retrievable_corpus_rows"],
            "excluded_empty_rows": corpus_stats["excluded_empty_rows"],
            "queries": query_rows,
            "qrels": {
                split: stats["rows"] for split, stats in sorted(qrel_stats.items())
            },
        },
        "source_corpus": {
            **dict(corpus_stats["source_file"]),
            "excluded_empty_ids_sha256": corpus_stats[
                "excluded_empty_ids_sha256"
            ],
        },
        "artifacts": {
            "source_manifest": describe_artifact(
                source_manifest_path, relative_to=destination
            ),
            "corpus": describe_artifact(
                destination / "corpus" / "corpus.jsonl",
                relative_to=destination,
                rows=corpus_stats["retrievable_corpus_rows"],
            ),
            "queries": describe_artifact(
                destination / "queries" / "queries.jsonl",
                relative_to=destination,
                rows=query_rows,
            ),
            "qrels": qrel_descriptors,
        },
    }
    if dataset == "cqadupstack":
        manifest["query_format"] = CQA_QUERY_FORMAT
    write_manifest(destination / "manifest.json", manifest)
    return manifest


def _artifact_descriptors(manifest: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("BEIR manifest artifacts must be an object")
    for name in ("source_manifest", "corpus", "queries"):
        descriptor = artifacts.get(name)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"BEIR manifest is missing artifact {name}")
        yield descriptor
    qrels = artifacts.get("qrels")
    if not isinstance(qrels, Mapping) or "test" not in qrels:
        raise ValueError("BEIR manifest must contain test qrels")
    for descriptor in qrels.values():
        if not isinstance(descriptor, Mapping):
            raise ValueError("BEIR qrels artifact descriptor must be an object")
        yield descriptor


def validate_beir_unit_directory(path: str | Path) -> dict[str, Any]:
    """Validate immutable files for one regular dataset or CQA forum."""

    root = Path(path).resolve()
    manifest = read_json_object(root / "manifest.json", label="BEIR unit manifest")
    if (
        manifest.get("status") != "complete"
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != "dataset_unit"
    ):
        raise ValueError(f"Invalid BEIR unit manifest: {root}")
    for descriptor in _artifact_descriptors(manifest):
        verify_artifact_descriptor(root, descriptor, label="BEIR")
    counts = manifest.get("counts")
    artifacts = manifest.get("artifacts")
    corpus_descriptor = (
        artifacts.get("corpus") if isinstance(artifacts, Mapping) else None
    )
    source_corpus = manifest.get("source_corpus")
    if (
        not isinstance(counts, Mapping)
        or not isinstance(corpus_descriptor, Mapping)
        or not isinstance(source_corpus, Mapping)
        or counts.get("retrievable_corpus_rows") != corpus_descriptor.get("rows")
        or not isinstance(counts.get("source_corpus_rows"), int)
        or not isinstance(counts.get("retrievable_corpus_rows"), int)
        or not isinstance(counts.get("excluded_empty_rows"), int)
        or counts["source_corpus_rows"]
        != counts["retrievable_corpus_rows"] + counts["excluded_empty_rows"]
        or source_corpus.get("size_bytes", -1) < 0
        or not isinstance(source_corpus.get("sha256"), str)
        or len(source_corpus["sha256"]) != 64
        or not isinstance(source_corpus.get("excluded_empty_ids_sha256"), str)
        or len(source_corpus["excluded_empty_ids_sha256"]) != 64
    ):
        raise ValueError(f"Invalid BEIR source/retrievable corpus counts: {root}")
    qrel_descriptors = artifacts.get("qrels")
    if not isinstance(qrel_descriptors, Mapping):
        raise ValueError("BEIR qrels descriptors must be an object")
    qrel_counts = counts.get("qrels")
    for split, descriptor in qrel_descriptors.items():
        if (
            not isinstance(descriptor, Mapping)
            or not isinstance(qrel_counts, Mapping)
            or qrel_counts.get(split) != descriptor.get("rows")
            or any(
                not isinstance(descriptor.get(name), int)
                or descriptor[name] < 0
                for name in (
                    "dangling_corpus_qrels",
                    "dangling_corpus_ids",
                    "excluded_empty_corpus_qrels",
                    "excluded_empty_corpus_ids",
                )
            )
            or any(
                not isinstance(descriptor.get(name), str)
                or len(descriptor[name]) != 64
                for name in (
                    "dangling_corpus_ids_sha256",
                    "excluded_empty_corpus_ids_sha256",
                )
            )
        ):
            raise ValueError(f"Invalid BEIR qrels issue metadata for split {split}")
    dataset = manifest.get("dataset")
    if dataset not in BEIR_DATASETS:
        raise ValueError(f"Unsupported dataset identity in BEIR manifest: {dataset!r}")
    query_format = manifest.get("query_format")
    if dataset == "cqadupstack" and query_format not in {
        None,
        CQA_QUERY_FORMAT,
    }:
        raise ValueError(f"Unsupported CQADupStack query format: {query_format!r}")
    if dataset == "cqadupstack" and query_format == CQA_QUERY_FORMAT:
        queries_descriptor = (
            artifacts.get("queries") if isinstance(artifacts, Mapping) else None
        )
        if (
            not isinstance(queries_descriptor, Mapping)
            or not isinstance(queries_descriptor.get("rows"), int)
        ):
            raise ValueError(f"Invalid compact CQADupStack query metadata: {root}")
        _validate_id_text_queries(
            root / queries_descriptor["file"],
            expected_rows=queries_descriptor["rows"],
        )
    source_path = root / manifest["artifacts"]["source_manifest"]["file"]
    source = read_json_object(source_path, label="BEIR source manifest")
    spec = BEIR_DATASETS[dataset]
    archive = source.get("archive")
    if (
        source.get("status") != "complete"
        or source.get("protocol") != PROTOCOL
        or source.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or source.get("dataset") != dataset
        or source.get("unit") != manifest.get("unit")
        or source.get("official_url") != spec.url
        or source.get("official_md5") != spec.official_md5
        or not isinstance(archive, Mapping)
        or archive.get("md5") != spec.official_md5
        or not isinstance(archive.get("sha256"), str)
        or len(archive["sha256"]) != 64
    ):
        raise ValueError(f"Invalid BEIR source identity: {root}")
    return manifest


def validate_beir_dataset_directory(path: str | Path) -> dict[str, Any]:
    """Validate one prepared dataset, including all 12 CQADupStack forums."""

    root = Path(path).resolve()
    manifest = read_json_object(root / "manifest.json", label="BEIR dataset manifest")
    if manifest.get("kind") == "dataset_unit":
        return validate_beir_unit_directory(root)
    if (
        manifest.get("status") != "complete"
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != "dataset_collection"
        or manifest.get("dataset") != "cqadupstack"
        or manifest.get("query_format") not in {None, CQA_QUERY_FORMAT}
    ):
        raise ValueError(f"Invalid BEIR dataset manifest: {root}")
    source_descriptor = manifest.get("source_manifest")
    if not isinstance(source_descriptor, Mapping):
        raise ValueError("CQADupStack collection source manifest is missing")
    verify_artifact_descriptor(root, source_descriptor, label="BEIR collection source")
    units = manifest.get("units")
    if not isinstance(units, Mapping) or tuple(sorted(units)) != tuple(
        sorted(CQADUPSTACK_FORUMS)
    ):
        raise ValueError("CQADupStack manifest must list exactly 12 forums")
    for forum in CQADUPSTACK_FORUMS:
        descriptor = units[forum]
        if not isinstance(descriptor, Mapping) or descriptor.get("path") != forum:
            raise ValueError(f"Invalid CQADupStack unit descriptor: {forum}")
        unit_manifest = validate_beir_unit_directory(root / forum)
        manifest_path = root / forum / "manifest.json"
        if sha256_file(manifest_path) != descriptor.get("manifest_sha256"):
            raise ValueError(f"CQADupStack unit manifest changed: {forum}")
        if unit_manifest.get("unit") != f"cqadupstack/{forum}":
            raise ValueError(f"CQADupStack unit identity mismatch: {forum}")
    return manifest


def prepare_beir_dataset(
    dataset: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    archive_path: str | Path | None = None,
    download: bool = False,
) -> dict[str, Any]:
    """Prepare one selected official BEIR archive without loading it into RAM.

    Downloads are opt-in.  Without ``archive_path`` the function reuses
    ``<output_root>/_archives/<dataset>.zip`` and downloads it only when
    ``download=True``.
    """

    canonical = canonical_dataset_name(dataset)
    spec = BEIR_DATASETS[canonical]
    output = Path(output_root).resolve()
    target = output / canonical
    if target.exists():
        manifest = validate_beir_dataset_directory(target)
        if manifest.get("dataset") != canonical:
            raise ValueError(f"Existing BEIR output has a different identity: {target}")
        return manifest

    if archive_path is None:
        archive = output / "_archives" / spec.archive_name
        if not archive.exists():
            if not download:
                raise FileNotFoundError(
                    f"Cached BEIR archive is missing: {archive}; pass --download"
                )
            archive = _download_archive(spec, archive)
    else:
        archive = Path(archive_path).resolve()
    archive_descriptor = _validate_archive(archive, spec)

    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{canonical}-prepare-", dir=output))
    extracted = work / "extracted"
    staging = work / canonical
    extracted.mkdir()
    staging.mkdir()
    try:
        _safe_extract(archive, extracted)
        sources = _unit_source_directories(extracted, canonical)
        unit_manifests: dict[str, dict[str, Any]] = {}
        for unit_name, source in sources.items():
            destination = staging if canonical != "cqadupstack" else staging / unit_name
            destination.mkdir(parents=True, exist_ok=True)
            _copy_unit_sources(
                source,
                destination,
                query_format=(
                    CQA_QUERY_FORMAT if canonical == "cqadupstack" else None
                ),
            )
            qualified_unit = (
                canonical
                if canonical != "cqadupstack"
                else f"cqadupstack/{unit_name}"
            )
            unit_manifests[unit_name] = _validate_and_describe_unit(
                destination,
                source_corpus_path=source / "corpus.jsonl",
                dataset=canonical,
                unit=qualified_unit,
                archive=archive_descriptor,
                spec=spec,
            )

        if canonical == "cqadupstack":
            source_manifest = {
                "status": "complete",
                "protocol": PROTOCOL,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "dataset": canonical,
                "official_url": spec.url,
                "official_md5": spec.official_md5,
                "archive": archive_descriptor,
            }
            write_manifest(staging / "source_manifest.json", source_manifest)
            collection_manifest = {
                "status": "complete",
                "protocol": PROTOCOL,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "kind": "dataset_collection",
                "dataset": canonical,
                "query_format": CQA_QUERY_FORMAT,
                "source_manifest": describe_artifact(
                    staging / "source_manifest.json", relative_to=staging
                ),
                "units": {
                    forum: {
                        "path": forum,
                        "manifest_sha256": sha256_file(staging / forum / "manifest.json"),
                        "counts": unit_manifests[forum]["counts"],
                    }
                    for forum in CQADUPSTACK_FORUMS
                },
            }
            write_manifest(staging / "manifest.json", collection_manifest)

        os.replace(staging, target)
        return validate_beir_dataset_directory(target)
    finally:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
