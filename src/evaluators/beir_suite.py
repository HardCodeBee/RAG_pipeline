"""Shared first-stage candidates and aggregation for the four-condition BEIR suite."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from src.evaluators.beir_evaluation import FirstStageCandidateBatch, score_beir_query
from src.persistence.artifact_io import atomic_write_npz, read_json_object
from src.persistence.run_output_writer import write_metadata_json
from src.provenance import json_sha256, sha256_file
from src.records import SearchHit
from src.rerankers.reranker_contract import RerankResult, RerankTrace, reranked_hits


SHARED_CANDIDATE_SCHEMA_VERSION = 1
SUITE_EVALUATION_PROTOCOL = "beir_retrieval_only_v2_ignore_identical_pre_candidate"
SHARED_CANDIDATE_FILE = "shared_first_stage_candidates.npz"
SHARED_CANDIDATE_MANIFEST = "shared_first_stage_candidates.manifest.json"
SHARED_RERANK_SCHEMA_VERSION = 1
SHARED_RERANK_FILE = "shared_bge_scores.npz"
SHARED_RERANK_MANIFEST = "shared_bge_scores.manifest.json"
SHARED_CACHE_REFERENCE_SCHEMA_VERSION = 1
BM25_CHECKPOINT_SCHEMA_VERSION = 1


def _cache_snapshot(
    *,
    kind: str,
    manifest_path: Path,
    data_path: Path,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = read_json_object(manifest_path, label=f"{kind} cache manifest")
    artifact = manifest.get("artifact")
    if (
        manifest.get("identity") != dict(expected_identity)
        or not isinstance(artifact, Mapping)
        or artifact.get("file") != data_path.name
        or not data_path.is_file()
    ):
        raise ValueError(f"{kind} cache descriptor is incompatible")
    size_bytes = data_path.stat().st_size
    artifact_sha256 = sha256_file(data_path)
    if (
        artifact.get("size_bytes") != size_bytes
        or artifact.get("sha256") != artifact_sha256
    ):
        raise ValueError(f"{kind} cache artifact changed after validation")
    return {
        "schema_version": SHARED_CACHE_REFERENCE_SCHEMA_VERSION,
        "kind": kind,
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": size_bytes,
    }


def _unit_relative_path(path: Path, unit_directory: str | Path) -> str:
    root = Path(unit_directory).resolve()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Cache artifact is outside its suite unit: {path}") from exc
    return relative.as_posix()


def _cache_descriptor(
    *,
    snapshot: Mapping[str, Any] | None,
    kind: str,
    manifest_path: Path,
    data_path: Path,
    expected_identity: Mapping[str, Any],
    unit_directory: str | Path,
) -> dict[str, Any]:
    if snapshot is None:
        raise RuntimeError(f"{kind} cache must be loaded or written before describing it")
    current = _cache_snapshot(
        kind=kind,
        manifest_path=manifest_path,
        data_path=data_path,
        expected_identity=expected_identity,
    )
    if current != dict(snapshot):
        raise ValueError(f"{kind} cache changed after it was loaded or written")
    return {
        **current,
        "manifest_path": _unit_relative_path(manifest_path, unit_directory),
        "data_path": _unit_relative_path(data_path, unit_directory),
    }


@dataclass(frozen=True, slots=True)
class SharedCandidateBatch:
    """A variable-width CSR batch that works for dense and sparse retrieval."""

    scores: np.ndarray
    vector_ids: np.ndarray
    indptr: np.ndarray
    question_ids: tuple[str, ...]
    timings_ms: Mapping[str, float]
    per_question_timings_ms: tuple[Mapping[str, float], ...]
    reused_cache: bool = False

    def row(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("position must be an integer")
        if not 0 <= position < len(self.question_ids):
            raise IndexError("candidate row position is out of range")
        start = int(self.indptr[position])
        stop = int(self.indptr[position + 1])
        return self.scores[start:stop], self.vector_ids[start:stop]


@dataclass(frozen=True, slots=True)
class SharedRerankScoreBatch:
    """Cross-encoder scores aligned one-to-one with a shared candidate batch."""

    scores: np.ndarray
    timing_ms: np.ndarray
    question_ids: tuple[str, ...]
    reused_cache: bool = False

    def row(
        self,
        position: int,
        candidates: SharedCandidateBatch,
    ) -> tuple[np.ndarray, float]:
        if self.question_ids != candidates.question_ids:
            raise ValueError("Rerank scores do not align with shared candidates")
        start = int(candidates.indptr[position])
        stop = int(candidates.indptr[position + 1])
        return self.scores[start:stop], float(self.timing_ms[position])


def _timing_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"{label}.{raw_key} must be numeric")
        number = float(raw_value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{label}.{raw_key} must be finite and non-negative")
        result[raw_key] = number
    return result


def validate_shared_candidate_batch(
    batch: SharedCandidateBatch,
    *,
    expected_question_ids: Sequence[str],
    candidate_k: int,
    corpus_count: int,
    require_full_width: bool,
) -> SharedCandidateBatch:
    """Validate shape, order, ids, and timing metadata before trust or commit."""

    if not isinstance(batch, SharedCandidateBatch):
        raise TypeError("batch must be a SharedCandidateBatch")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
        raise ValueError("candidate_k must be a positive integer")
    if isinstance(corpus_count, bool) or not isinstance(corpus_count, int) or corpus_count <= 0:
        raise ValueError("corpus_count must be a positive integer")
    if not isinstance(require_full_width, bool):
        raise TypeError("require_full_width must be a boolean")

    expected_ids = tuple(expected_question_ids)
    if not expected_ids or any(not isinstance(value, str) or not value for value in expected_ids):
        raise ValueError("expected_question_ids must contain non-empty strings")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected_question_ids must be unique")
    actual_ids = tuple(batch.question_ids)
    if actual_ids != expected_ids:
        raise ValueError("Candidate cache question order does not match the selected set")

    scores = np.asarray(batch.scores)
    vector_ids = np.asarray(batch.vector_ids)
    indptr = np.asarray(batch.indptr)
    if scores.dtype != np.dtype("float64") or scores.ndim != 1:
        raise ValueError("Shared candidate scores must be a one-dimensional float64 array")
    if vector_ids.dtype != np.dtype("int64") or vector_ids.shape != scores.shape:
        raise ValueError("Shared candidate vector ids must be int64 and align with scores")
    if indptr.dtype != np.dtype("int64") or indptr.shape != (len(expected_ids) + 1,):
        raise ValueError("Shared candidate indptr has an invalid dtype or shape")
    if int(indptr[0]) != 0 or int(indptr[-1]) != len(scores):
        raise ValueError("Shared candidate indptr boundaries are invalid")
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("Shared candidate indptr must be monotonic")
    if not np.isfinite(scores).all():
        raise ValueError("Shared candidate cache contains non-finite scores")
    if np.any(vector_ids < 0) or np.any(vector_ids >= corpus_count):
        raise ValueError("Shared candidate cache contains out-of-range vector ids")

    full_width = min(candidate_k, corpus_count)
    for position in range(len(expected_ids)):
        start = int(indptr[position])
        stop = int(indptr[position + 1])
        row_scores = scores[start:stop]
        row_ids = vector_ids[start:stop]
        width = stop - start
        if width > full_width or (require_full_width and width != full_width):
            qualifier = "exactly" if require_full_width else "at most"
            raise ValueError(f"Candidate rows must contain {qualifier} {full_width} results")
        if len(np.unique(row_ids)) != width:
            raise ValueError("Shared candidate cache contains duplicate ids within a query")
        if width:
            expected_order = np.arange(width)
            actual_order = np.lexsort((row_ids, -row_scores))
            if not np.array_equal(actual_order, expected_order):
                raise ValueError(
                    "Shared candidates must use score-descending/id-ascending order"
                )

    timings = _timing_mapping(batch.timings_ms, label="timings_ms")
    if len(batch.per_question_timings_ms) != len(expected_ids):
        raise ValueError("Per-question timings must align with question ids")
    row_timings = tuple(
        _timing_mapping(value, label=f"per_question_timings_ms[{position}]")
        for position, value in enumerate(batch.per_question_timings_ms)
    )
    return SharedCandidateBatch(
        scores=np.ascontiguousarray(scores),
        vector_ids=np.ascontiguousarray(vector_ids),
        indptr=np.ascontiguousarray(indptr),
        question_ids=actual_ids,
        timings_ms=timings,
        per_question_timings_ms=row_timings,
        reused_cache=batch.reused_cache,
    )


def shared_batch_from_dense(
    batch: FirstStageCandidateBatch,
    *,
    questions: Sequence[Mapping[str, Any]] | None = None,
    chunk_store: Any | None = None,
    retained_k: int | None = None,
) -> SharedCandidateBatch:
    """Convert dense results and apply BEIR's self-match filter."""

    if not isinstance(batch, FirstStageCandidateBatch):
        raise TypeError("batch must be a FirstStageCandidateBatch")
    if batch.scores.ndim != 2 or batch.vector_ids.shape != batch.scores.shape:
        raise ValueError("Dense candidate arrays must be aligned matrices")
    num_questions, width = batch.scores.shape
    if questions is None or chunk_store is None:
        raise ValueError("Self-match filtering requires questions and chunk_store")
    if len(questions) != num_questions:
        raise ValueError("Dense questions must align with the candidate batch")
    if isinstance(retained_k, bool) or not isinstance(retained_k, int) or retained_k <= 0:
        raise ValueError("retained_k must be a positive integer")
    if width < min(retained_k + 1, len(chunk_store)):
        raise ValueError("Dense self-match filtering requires one extra search result")

    started = time.perf_counter()
    retained_scores: list[float] = []
    retained_ids: list[int] = []
    indptr = [0]
    removed = 0
    for position, question in enumerate(questions):
        query_id = str(question["question_id"])
        row_ids = [int(value) for value in batch.vector_ids[position]]
        chunks = chunk_store.get_many(row_ids)
        for score, vector_id, chunk in zip(batch.scores[position], row_ids, chunks):
            if chunk.doc_id == query_id:
                removed += 1
                continue
            if len(retained_ids) - indptr[-1] >= retained_k:
                break
            retained_scores.append(float(score))
            retained_ids.append(vector_id)
        indptr.append(len(retained_ids))
    timings = dict(batch.timings_ms)
    timings.update(
        {
            "identical_id_filter_ms": (time.perf_counter() - started) * 1000,
            "identical_id_matches_removed": float(removed),
        }
    )
    return SharedCandidateBatch(
        scores=np.asarray(retained_scores, dtype=np.float64),
        vector_ids=np.asarray(retained_ids, dtype=np.int64),
        indptr=np.asarray(indptr, dtype=np.int64),
        question_ids=tuple(batch.question_ids),
        timings_ms=timings,
        per_question_timings_ms=tuple({} for _ in range(num_questions)),
        reused_cache=batch.reused_cache,
    )


def _compute_bm25_group(
    pipeline: Any,
    questions: Sequence[Mapping[str, Any]],
    *,
    candidate_k: int,
    retained_k: int,
) -> SharedCandidateBatch:
    started = time.perf_counter()
    scores: list[float] = []
    vector_ids: list[int] = []
    indptr = [0]
    per_question_timings: list[dict[str, float]] = []
    timing_totals: defaultdict[str, float] = defaultdict(float)
    removed = 0
    for question in questions:
        trace = pipeline.retriever.retrieve_trace(
            str(question["question"]),
            top_k=candidate_k,
        )
        if [hit.rank for hit in trace.results] != list(range(1, len(trace.results) + 1)):
            raise ValueError("BM25 candidate ranks must be consecutive")
        retained = []
        query_id = str(question["question_id"])
        for hit in trace.results:
            if hit.chunk.doc_id == query_id:
                removed += 1
                continue
            retained.append(hit)
            if len(retained) >= retained_k:
                break
        scores.extend(float(hit.score) for hit in retained)
        vector_ids.extend(int(hit.chunk.vector_id) for hit in retained)
        indptr.append(len(scores))
        row_timings = _timing_mapping(trace.timings_ms, label="BM25 timings")
        per_question_timings.append(row_timings)
        for key, value in row_timings.items():
            timing_totals[key] += value
    timing_totals["batch_wall_clock_ms"] = (time.perf_counter() - started) * 1000
    timing_totals["identical_id_matches_removed"] = float(removed)
    return SharedCandidateBatch(
        scores=np.asarray(scores, dtype=np.float64),
        vector_ids=np.asarray(vector_ids, dtype=np.int64),
        indptr=np.asarray(indptr, dtype=np.int64),
        question_ids=tuple(str(question["question_id"]) for question in questions),
        timings_ms=dict(timing_totals),
        per_question_timings_ms=tuple(per_question_timings),
    )


def _combine_bm25_groups(
    groups: Sequence[SharedCandidateBatch],
    *,
    question_ids: Sequence[str],
) -> SharedCandidateBatch:
    scores = np.concatenate([group.scores for group in groups])
    vector_ids = np.concatenate([group.vector_ids for group in groups])
    indptr = [0]
    offset = 0
    per_question_timings: list[Mapping[str, float]] = []
    timing_totals: defaultdict[str, float] = defaultdict(float)
    actual_question_ids: list[str] = []
    for group in groups:
        actual_question_ids.extend(group.question_ids)
        for value in group.indptr[1:]:
            indptr.append(offset + int(value))
        offset += len(group.scores)
        per_question_timings.extend(group.per_question_timings_ms)
        for key, value in group.timings_ms.items():
            timing_totals[key] += float(value)
    if tuple(actual_question_ids) != tuple(question_ids):
        raise ValueError("BM25 checkpoint groups do not match the selected question order")
    return SharedCandidateBatch(
        scores=np.asarray(scores, dtype=np.float64),
        vector_ids=np.asarray(vector_ids, dtype=np.int64),
        indptr=np.asarray(indptr, dtype=np.int64),
        question_ids=tuple(actual_question_ids),
        timings_ms=dict(timing_totals),
        per_question_timings_ms=tuple(per_question_timings),
        reused_cache=False,
    )


def compute_bm25_first_stage(
    pipeline: Any,
    questions: Sequence[Mapping[str, Any]],
    *,
    candidate_k: int,
    retained_k: int | None = None,
    checkpoint_store: BM25FirstStageCheckpointStore | None = None,
) -> SharedCandidateBatch:
    """Issue one BM25 retrieval per query, optionally resuming verified groups."""

    if not questions:
        raise ValueError("questions must not be empty")
    if pipeline.config["retrieval"]["method"] != "bm25":
        raise ValueError("BM25 first-stage computation requires a BM25 pipeline")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
        raise ValueError("candidate_k must be a positive integer")
    if retained_k is None:
        retained_k = candidate_k
    if isinstance(retained_k, bool) or not isinstance(retained_k, int) or retained_k <= 0:
        raise ValueError("retained_k must be a positive integer")
    if retained_k > candidate_k:
        raise ValueError("retained_k cannot exceed candidate_k")
    if checkpoint_store is None:
        return _compute_bm25_group(
            pipeline,
            questions,
            candidate_k=candidate_k,
            retained_k=retained_k,
        )

    checkpoint_store.validate_compute_request(
        question_ids=[str(question["question_id"]) for question in questions],
        candidate_k=candidate_k,
        retained_k=retained_k,
        corpus_count=len(pipeline.chunk_store),
    )
    groups: list[SharedCandidateBatch] = []
    for start in range(0, len(questions), checkpoint_store.query_group_size):
        stop = min(start + checkpoint_store.query_group_size, len(questions))
        group = checkpoint_store.load(start=start, stop=stop)
        if group is None:
            group = _compute_bm25_group(
                pipeline,
                questions[start:stop],
                candidate_k=candidate_k,
                retained_k=retained_k,
            )
            checkpoint_store.write(group, start=start, stop=stop)
        groups.append(group)
    return _combine_bm25_groups(
        groups,
        question_ids=[str(question["question_id"]) for question in questions],
    )


class SharedCandidateStore:
    """Atomic, identity-bound storage for a reusable CSR candidate batch."""

    def __init__(
        self,
        directory: str | Path,
        *,
        identity: Mapping[str, Any],
        question_ids: Sequence[str],
        candidate_k: int,
        corpus_count: int,
        require_full_width: bool,
    ) -> None:
        self.directory = Path(directory).resolve()
        self.data_path = self.directory / SHARED_CANDIDATE_FILE
        self.manifest_path = self.directory / SHARED_CANDIDATE_MANIFEST
        self.identity = dict(identity)
        self.question_ids = tuple(question_ids)
        self.candidate_k = candidate_k
        self.corpus_count = corpus_count
        self.require_full_width = require_full_width
        self._validated_cache_snapshot: dict[str, Any] | None = None

    def _validated(self, batch: SharedCandidateBatch) -> SharedCandidateBatch:
        return validate_shared_candidate_batch(
            batch,
            expected_question_ids=self.question_ids,
            candidate_k=self.candidate_k,
            corpus_count=self.corpus_count,
            require_full_width=self.require_full_width,
        )

    def write(self, batch: SharedCandidateBatch) -> SharedCandidateBatch:
        committed = self._validated(batch)
        if self.manifest_path.exists():
            raise FileExistsError(f"Shared candidate manifest already exists: {self.manifest_path}")
        atomic_write_npz(
            self.data_path,
            scores=committed.scores,
            vector_ids=committed.vector_ids,
            indptr=committed.indptr,
            question_ids=np.asarray(committed.question_ids, dtype=np.str_),
        )

        artifact = {
            "file": SHARED_CANDIDATE_FILE,
            "size_bytes": self.data_path.stat().st_size,
            "sha256": sha256_file(self.data_path),
        }
        manifest = {
            "status": "complete",
            "schema_version": SHARED_CANDIDATE_SCHEMA_VERSION,
            "identity": self.identity,
            "artifact": artifact,
            "timings_ms": dict(committed.timings_ms),
            "per_question_timings_ms": [
                dict(value) for value in committed.per_question_timings_ms
            ],
        }
        write_metadata_json(self.manifest_path, manifest, overwrite=False)
        self._validated_cache_snapshot = _cache_snapshot(
            kind="shared_first_stage_candidates",
            manifest_path=self.manifest_path,
            data_path=self.data_path,
            expected_identity=self.identity,
        )
        return committed

    def load(self) -> SharedCandidateBatch | None:
        if not self.manifest_path.exists():
            self._validated_cache_snapshot = None
            return None
        if not self.data_path.is_file():
            raise FileNotFoundError(
                "Shared candidate manifest exists but its NPZ artifact is missing"
            )
        manifest = read_json_object(
            self.manifest_path,
            label="Shared first-stage candidate manifest",
        )
        artifact = manifest.get("artifact")
        if (
            manifest.get("status") != "complete"
            or manifest.get("schema_version") != SHARED_CANDIDATE_SCHEMA_VERSION
            or manifest.get("identity") != self.identity
            or not isinstance(artifact, Mapping)
            or artifact.get("file") != SHARED_CANDIDATE_FILE
        ):
            raise ValueError("Shared first-stage candidate cache identity is incompatible")
        if (
            self.data_path.stat().st_size != artifact.get("size_bytes")
            or sha256_file(self.data_path) != artifact.get("sha256")
        ):
            raise ValueError("Shared first-stage candidate cache artifact is corrupted")
        try:
            with np.load(self.data_path, allow_pickle=False) as arrays:
                if set(arrays.files) != {
                    "scores",
                    "vector_ids",
                    "indptr",
                    "question_ids",
                }:
                    raise ValueError("Shared candidate NPZ has unexpected arrays")
                batch = SharedCandidateBatch(
                    scores=np.array(arrays["scores"], copy=True),
                    vector_ids=np.array(arrays["vector_ids"], copy=True),
                    indptr=np.array(arrays["indptr"], copy=True),
                    question_ids=tuple(str(value) for value in arrays["question_ids"]),
                    timings_ms=manifest.get("timings_ms", {}),
                    per_question_timings_ms=tuple(
                        manifest.get("per_question_timings_ms", ())
                    ),
                    reused_cache=True,
                )
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot load shared first-stage candidate cache NPZ") from exc
        validated = self._validated(batch)
        self._validated_cache_snapshot = _cache_snapshot(
            kind="shared_first_stage_candidates",
            manifest_path=self.manifest_path,
            data_path=self.data_path,
            expected_identity=self.identity,
        )
        return SharedCandidateBatch(
            scores=validated.scores,
            vector_ids=validated.vector_ids,
            indptr=validated.indptr,
            question_ids=validated.question_ids,
            timings_ms=validated.timings_ms,
            per_question_timings_ms=validated.per_question_timings_ms,
            reused_cache=True,
        )

    def descriptor(self, *, unit_directory: str | Path) -> dict[str, Any]:
        """Return the exact, unit-relative cache generation just validated."""

        return _cache_descriptor(
            snapshot=self._validated_cache_snapshot,
            kind="shared_first_stage_candidates",
            manifest_path=self.manifest_path,
            data_path=self.data_path,
            expected_identity=self.identity,
            unit_directory=unit_directory,
        )


def _validate_rerank_score_batch(
    batch: SharedRerankScoreBatch,
    candidates: SharedCandidateBatch,
) -> SharedRerankScoreBatch:
    if not isinstance(batch, SharedRerankScoreBatch):
        raise TypeError("batch must be a SharedRerankScoreBatch")
    if tuple(batch.question_ids) != candidates.question_ids:
        raise ValueError("Rerank score question ids do not match shared candidates")
    scores = np.asarray(batch.scores)
    timing_ms = np.asarray(batch.timing_ms)
    if scores.dtype != np.dtype("float64") or scores.shape != candidates.scores.shape:
        raise ValueError("Rerank scores must be float64 and align with all candidates")
    if timing_ms.dtype != np.dtype("float64") or timing_ms.shape != (
        len(candidates.question_ids),
    ):
        raise ValueError("Rerank timings must be one float64 value per question")
    if not np.isfinite(scores).all():
        raise ValueError("Rerank scores contain non-finite values")
    if not np.isfinite(timing_ms).all() or np.any(timing_ms < 0.0):
        raise ValueError("Rerank timings must be finite and non-negative")
    return SharedRerankScoreBatch(
        scores=np.ascontiguousarray(scores),
        timing_ms=np.ascontiguousarray(timing_ms),
        question_ids=tuple(batch.question_ids),
        reused_cache=batch.reused_cache,
    )


class BM25FirstStageCheckpointStore:
    """Identity-bound, resumable NPZ groups for BM25 first-stage retrieval."""

    def __init__(
        self,
        directory: str | Path,
        *,
        identity: Mapping[str, Any],
        question_ids: Sequence[str],
        candidate_k: int,
        retained_k: int,
        corpus_count: int,
        query_group_size: int,
    ) -> None:
        if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")
        if isinstance(retained_k, bool) or not isinstance(retained_k, int) or retained_k <= 0:
            raise ValueError("retained_k must be a positive integer")
        if retained_k > candidate_k:
            raise ValueError("retained_k cannot exceed candidate_k")
        if isinstance(corpus_count, bool) or not isinstance(corpus_count, int) or corpus_count <= 0:
            raise ValueError("corpus_count must be a positive integer")
        if (
            isinstance(query_group_size, bool)
            or not isinstance(query_group_size, int)
            or query_group_size <= 0
        ):
            raise ValueError("query_group_size must be a positive integer")
        normalized_ids = tuple(question_ids)
        if (
            not normalized_ids
            or any(not isinstance(value, str) or not value for value in normalized_ids)
            or len(normalized_ids) != len(set(normalized_ids))
        ):
            raise ValueError("question_ids must contain unique, non-empty strings")

        self.directory = Path(directory).resolve()
        self.question_ids = normalized_ids
        self.candidate_k = candidate_k
        self.retained_k = retained_k
        self.corpus_count = corpus_count
        self.query_group_size = query_group_size
        self.identity = {
            "schema_version": BM25_CHECKPOINT_SCHEMA_VERSION,
            "first_stage_identity": dict(identity),
            "question_ids_sha256": json_sha256(list(normalized_ids)),
            "candidate_k": candidate_k,
            "retained_k": retained_k,
            "corpus_count": corpus_count,
            "query_group_size": query_group_size,
        }

    def _data_path(self, start: int, stop: int) -> Path:
        return self.directory / f"{start:08d}_{stop:08d}.npz"

    def _manifest_path(self, start: int, stop: int) -> Path:
        return self.directory / f"{start:08d}_{stop:08d}.manifest.json"

    def _expected_names(self) -> tuple[set[str], set[str]]:
        data_names: set[str] = set()
        manifest_names: set[str] = set()
        for start in range(0, len(self.question_ids), self.query_group_size):
            stop = min(start + self.query_group_size, len(self.question_ids))
            data_names.add(self._data_path(start, stop).name)
            manifest_names.add(self._manifest_path(start, stop).name)
        return data_names, manifest_names

    def _validate_layout(self) -> None:
        if not self.directory.is_dir():
            return
        expected_data, expected_manifests = self._expected_names()
        unexpected = sorted(
            [path.name for path in self.directory.glob("*.npz") if path.name not in expected_data]
            + [
                path.name
                for path in self.directory.glob("*.manifest.json")
                if path.name not in expected_manifests
            ]
        )
        if unexpected:
            raise ValueError("Unexpected BM25 checkpoint files: " + ", ".join(unexpected))

    def validate_compute_request(
        self,
        *,
        question_ids: Sequence[str],
        candidate_k: int,
        retained_k: int,
        corpus_count: int,
    ) -> None:
        if (
            tuple(question_ids) != self.question_ids
            or candidate_k != self.candidate_k
            or retained_k != self.retained_k
            or corpus_count != self.corpus_count
        ):
            raise ValueError("BM25 checkpoint store is incompatible with this computation")
        self._validate_layout()

    def load(self, *, start: int, stop: int) -> SharedCandidateBatch | None:
        data_path = self._data_path(start, stop)
        manifest_path = self._manifest_path(start, stop)
        if not manifest_path.exists():
            # A data file without its atomic manifest is an incomplete group,
            # so it is safe to replace when the group is recomputed.
            return None
        if not data_path.is_file():
            raise FileNotFoundError(
                f"BM25 checkpoint manifest exists but its NPZ is missing: {data_path}"
            )
        manifest = read_json_object(manifest_path, label="BM25 checkpoint manifest")
        artifact = manifest.get("artifact")
        expected_ids = self.question_ids[start:stop]
        if (
            manifest.get("status") != "complete"
            or manifest.get("schema_version") != BM25_CHECKPOINT_SCHEMA_VERSION
            or manifest.get("identity") != self.identity
            or manifest.get("group_start") != start
            or manifest.get("group_stop") != stop
            or not isinstance(artifact, Mapping)
            or artifact.get("file") != data_path.name
        ):
            raise ValueError(f"BM25 checkpoint identity is incompatible: {data_path}")
        if (
            data_path.stat().st_size != artifact.get("size_bytes")
            or sha256_file(data_path) != artifact.get("sha256")
        ):
            raise ValueError(f"BM25 checkpoint artifact is corrupted: {data_path}")
        try:
            with np.load(data_path, allow_pickle=False) as arrays:
                expected_arrays = {
                    "scores",
                    "vector_ids",
                    "indptr",
                    "question_ids",
                    "timings_json",
                    "per_question_timings_json",
                }
                legacy_arrays = {"identity_sha256", "group_start", "group_stop"}
                array_names = set(arrays.files)
                extras = frozenset(array_names - expected_arrays)
                if not expected_arrays <= array_names or extras not in {
                    frozenset(),
                    frozenset(legacy_arrays),
                }:
                    raise ValueError("BM25 checkpoint has unexpected arrays")
                if "identity_sha256" in arrays.files and str(
                    np.asarray(arrays["identity_sha256"]).item()
                ) != json_sha256(self.identity):
                    raise ValueError(f"BM25 checkpoint identity is incompatible: {data_path}")
                if "group_start" in arrays.files and (
                    int(np.asarray(arrays["group_start"]).item()) != start
                    or int(np.asarray(arrays["group_stop"]).item()) != stop
                ):
                    raise ValueError(f"BM25 checkpoint identity is incompatible: {data_path}")
                timings = json.loads(str(np.asarray(arrays["timings_json"]).item()))
                per_question_timings = tuple(
                    json.loads(str(value))
                    for value in arrays["per_question_timings_json"]
                )
                batch = SharedCandidateBatch(
                    scores=np.array(arrays["scores"], copy=True),
                    vector_ids=np.array(arrays["vector_ids"], copy=True),
                    indptr=np.array(arrays["indptr"], copy=True),
                    question_ids=tuple(str(value) for value in arrays["question_ids"]),
                    timings_ms=timings,
                    per_question_timings_ms=per_question_timings,
                    reused_cache=True,
                )
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot load BM25 checkpoint NPZ: {data_path}") from exc
        return validate_shared_candidate_batch(
            batch,
            expected_question_ids=expected_ids,
            candidate_k=self.retained_k,
            corpus_count=self.corpus_count,
            require_full_width=False,
        )

    def write(
        self,
        batch: SharedCandidateBatch,
        *,
        start: int,
        stop: int,
    ) -> None:
        expected_ids = self.question_ids[start:stop]
        committed = validate_shared_candidate_batch(
            batch,
            expected_question_ids=expected_ids,
            candidate_k=self.retained_k,
            corpus_count=self.corpus_count,
            require_full_width=False,
        )
        data_path = self._data_path(start, stop)
        manifest_path = self._manifest_path(start, stop)
        if manifest_path.exists():
            raise FileExistsError(f"BM25 checkpoint manifest already exists: {manifest_path}")
        atomic_write_npz(
            data_path,
            scores=committed.scores,
            vector_ids=committed.vector_ids,
            indptr=committed.indptr,
            question_ids=np.asarray(committed.question_ids, dtype=np.str_),
            timings_json=np.asarray(
                json.dumps(
                    dict(committed.timings_ms),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                dtype=np.str_,
            ),
            per_question_timings_json=np.asarray(
                [
                    json.dumps(
                        dict(value),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    for value in committed.per_question_timings_ms
                ],
                dtype=np.str_,
            ),
        )
        manifest = {
            "status": "complete",
            "schema_version": BM25_CHECKPOINT_SCHEMA_VERSION,
            "identity": self.identity,
            "group_start": start,
            "group_stop": stop,
            "artifact": {
                "file": data_path.name,
                "size_bytes": data_path.stat().st_size,
                "sha256": sha256_file(data_path),
            },
        }
        write_metadata_json(manifest_path, manifest, overwrite=False)

    def cleanup(self) -> None:
        """Best-effort cleanup after the final candidate manifest commits."""

        if not self.directory.is_dir():
            return
        for pattern in ("*.npz", "*.manifest.json", ".*.npz.*.tmp", ".*.json.*.tmp"):
            for path in self.directory.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    pass
        try:
            self.directory.rmdir()
        except OSError:
            pass


class SharedRerankScoreStore:
    """Atomic BGE score cache with resumable query-group checkpoints."""

    def __init__(
        self,
        directory: str | Path,
        *,
        identity: Mapping[str, Any],
        candidates: SharedCandidateBatch,
    ) -> None:
        self.directory = Path(directory).resolve()
        self.data_path = self.directory / SHARED_RERANK_FILE
        self.manifest_path = self.directory / SHARED_RERANK_MANIFEST
        self.checkpoints_dir = self.directory / "checkpoints"
        self.identity = dict(identity)
        self.identity_sha256 = json_sha256(self.identity)
        self.candidates = candidates
        self._validated_cache_snapshot: dict[str, Any] | None = None

    def load(self) -> SharedRerankScoreBatch | None:
        if not self.manifest_path.exists():
            self._validated_cache_snapshot = None
            return None
        if not self.data_path.is_file():
            raise FileNotFoundError("BGE score manifest exists but its NPZ artifact is missing")
        manifest = read_json_object(self.manifest_path, label="Shared BGE score manifest")
        artifact = manifest.get("artifact")
        if (
            manifest.get("status") != "complete"
            or manifest.get("schema_version") != SHARED_RERANK_SCHEMA_VERSION
            or manifest.get("identity") != self.identity
            or not isinstance(artifact, Mapping)
            or artifact.get("file") != SHARED_RERANK_FILE
        ):
            raise ValueError("Shared BGE score cache identity is incompatible")
        if (
            self.data_path.stat().st_size != artifact.get("size_bytes")
            or sha256_file(self.data_path) != artifact.get("sha256")
        ):
            raise ValueError("Shared BGE score cache artifact is corrupted")
        try:
            with np.load(self.data_path, allow_pickle=False) as arrays:
                if set(arrays.files) != {"scores", "timing_ms", "question_ids"}:
                    raise ValueError("Shared BGE score NPZ has unexpected arrays")
                batch = SharedRerankScoreBatch(
                    scores=np.array(arrays["scores"], copy=True),
                    timing_ms=np.array(arrays["timing_ms"], copy=True),
                    question_ids=tuple(str(value) for value in arrays["question_ids"]),
                    reused_cache=True,
                )
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot load shared BGE score NPZ") from exc
        validated = _validate_rerank_score_batch(batch, self.candidates)
        self._validated_cache_snapshot = _cache_snapshot(
            kind="shared_bge_scores",
            manifest_path=self.manifest_path,
            data_path=self.data_path,
            expected_identity=self.identity,
        )
        return SharedRerankScoreBatch(
            scores=validated.scores,
            timing_ms=validated.timing_ms,
            question_ids=validated.question_ids,
            reused_cache=True,
        )

    def write(self, batch: SharedRerankScoreBatch) -> SharedRerankScoreBatch:
        committed = _validate_rerank_score_batch(batch, self.candidates)
        if self.manifest_path.exists():
            raise FileExistsError(f"BGE score manifest already exists: {self.manifest_path}")
        atomic_write_npz(
            self.data_path,
            scores=committed.scores,
            timing_ms=committed.timing_ms,
            question_ids=np.asarray(committed.question_ids, dtype=np.str_),
        )
        manifest = {
            "status": "complete",
            "schema_version": SHARED_RERANK_SCHEMA_VERSION,
            "identity": self.identity,
            "artifact": {
                "file": SHARED_RERANK_FILE,
                "size_bytes": self.data_path.stat().st_size,
                "sha256": sha256_file(self.data_path),
            },
        }
        write_metadata_json(self.manifest_path, manifest, overwrite=False)
        self._validated_cache_snapshot = _cache_snapshot(
            kind="shared_bge_scores",
            manifest_path=self.manifest_path,
            data_path=self.data_path,
            expected_identity=self.identity,
        )
        for checkpoint in self.checkpoints_dir.glob("*.npz") if self.checkpoints_dir.exists() else ():
            try:
                checkpoint.unlink()
            except OSError:
                pass
        if self.checkpoints_dir.exists():
            try:
                self.checkpoints_dir.rmdir()
            except OSError:
                pass
        return committed

    def descriptor(self, *, unit_directory: str | Path) -> dict[str, Any]:
        """Return the exact, unit-relative BGE cache generation just validated."""

        return _cache_descriptor(
            snapshot=self._validated_cache_snapshot,
            kind="shared_bge_scores",
            manifest_path=self.manifest_path,
            data_path=self.data_path,
            expected_identity=self.identity,
            unit_directory=unit_directory,
        )

    def _checkpoint_path(self, start: int, stop: int) -> Path:
        return self.checkpoints_dir / f"{start:08d}_{stop:08d}.npz"

    def _load_checkpoint(
        self,
        path: Path,
        *,
        start: int,
        stop: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        expected_ids = self.candidates.question_ids[start:stop]
        expected_scores = int(self.candidates.indptr[stop] - self.candidates.indptr[start])
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != {
                    "scores",
                    "timing_ms",
                    "question_ids",
                    "identity_sha256",
                }:
                    raise ValueError("BGE checkpoint has unexpected arrays")
                identity = str(np.asarray(arrays["identity_sha256"]).item())
                scores = np.array(arrays["scores"], copy=True)
                timings = np.array(arrays["timing_ms"], copy=True)
                question_ids = tuple(str(value) for value in arrays["question_ids"])
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot load BGE score checkpoint: {path}") from exc
        if (
            identity != self.identity_sha256
            or question_ids != expected_ids
            or scores.dtype != np.dtype("float64")
            or scores.shape != (expected_scores,)
            or timings.dtype != np.dtype("float64")
            or timings.shape != (stop - start,)
            or not np.isfinite(scores).all()
            or not np.isfinite(timings).all()
            or np.any(timings < 0.0)
        ):
            raise ValueError(f"BGE score checkpoint is incompatible: {path}")
        return scores, timings

    def compute(
        self,
        *,
        questions: Sequence[Mapping[str, Any]],
        chunk_store: Any,
        reranker: Any,
        final_k: int,
        query_group_size: int,
    ) -> SharedRerankScoreBatch:
        """Compute or resume small query groups, preferring rerank_many."""

        if len(questions) != len(self.candidates.question_ids):
            raise ValueError("Questions must align with shared candidates")
        if (
            isinstance(query_group_size, bool)
            or not isinstance(query_group_size, int)
            or query_group_size <= 0
        ):
            raise ValueError("query_group_size must be a positive integer")
        all_scores = np.empty_like(self.candidates.scores, dtype=np.float64)
        all_timings = np.empty(len(questions), dtype=np.float64)
        for start in range(0, len(questions), query_group_size):
            stop = min(start + query_group_size, len(questions))
            checkpoint_path = self._checkpoint_path(start, stop)
            if checkpoint_path.is_file():
                group_scores, group_timings = self._load_checkpoint(
                    checkpoint_path,
                    start=start,
                    stop=stop,
                )
            else:
                queries: list[str] = []
                hits_by_query: list[tuple[SearchHit, ...]] = []
                for position in range(start, stop):
                    row_scores, row_ids = self.candidates.row(position)
                    chunks = chunk_store.get_many([int(value) for value in row_ids])
                    hits_by_query.append(
                        tuple(
                            SearchHit(rank=rank, chunk=chunk, score=float(score))
                            for rank, (chunk, score) in enumerate(
                                zip(chunks, row_scores), start=1
                            )
                        )
                    )
                    queries.append(str(questions[position]["question"]))
                rerank_many = getattr(reranker, "rerank_many", None)
                if callable(rerank_many):
                    results = tuple(
                        rerank_many(queries, hits_by_query, final_k=final_k)
                    )
                else:
                    results = tuple(
                        reranker.rerank(query, hits, final_k=final_k)
                        for query, hits in zip(queries, hits_by_query)
                    )
                if len(results) != stop - start:
                    raise ValueError("rerank_many returned the wrong number of results")
                score_parts: list[np.ndarray] = []
                timing_values: list[float] = []
                for hits, result in zip(hits_by_query, results):
                    if not isinstance(result, RerankResult):
                        raise ValueError("BGE reranking must return RerankResult values")
                    if not hits:
                        if result.results or result.trace is not None:
                            raise ValueError("Empty BGE candidates must produce an empty result")
                        score_parts.append(np.empty(0, dtype=np.float64))
                        timing_values.append(float(result.timing_ms))
                        continue
                    if result.trace is None:
                        raise ValueError("Non-empty BGE reranking must return a trace")
                    if result.trace.candidates != hits:
                        raise ValueError("BGE trace candidates differ from shared candidates")
                    score_parts.append(np.asarray(result.trace.scores, dtype=np.float64))
                    timing_values.append(float(result.timing_ms))
                group_scores = (
                    np.concatenate(score_parts)
                    if score_parts
                    else np.empty(0, dtype=np.float64)
                )
                group_timings = np.asarray(timing_values, dtype=np.float64)
                atomic_write_npz(
                    checkpoint_path,
                    scores=group_scores,
                    timing_ms=group_timings,
                    question_ids=np.asarray(
                        self.candidates.question_ids[start:stop], dtype=np.str_
                    ),
                    identity_sha256=np.asarray(self.identity_sha256, dtype=np.str_),
                )
            score_start = int(self.candidates.indptr[start])
            score_stop = int(self.candidates.indptr[stop])
            all_scores[score_start:score_stop] = group_scores
            all_timings[start:stop] = group_timings
        return _validate_rerank_score_batch(
            SharedRerankScoreBatch(
                scores=all_scores,
                timing_ms=all_timings,
                question_ids=self.candidates.question_ids,
            ),
            self.candidates,
        )


def rerank_result_from_scores(
    hits: Sequence[SearchHit],
    scores: Sequence[float],
    *,
    final_k: int,
    timing_ms: float,
) -> RerankResult:
    """Reconstruct the standard BGE trace from cached per-candidate logits."""

    candidates = tuple(hits)
    values = tuple(float(value) for value in scores)
    if len(values) != len(candidates) or any(not math.isfinite(value) for value in values):
        raise ValueError("Cached rerank scores must align with candidates and be finite")
    if not candidates:
        return RerankResult(results=(), timing_ms=float(timing_ms))
    effective_final_k = min(final_k, len(candidates))
    order = tuple(sorted(range(len(candidates)), key=lambda position: -values[position]))
    trace = RerankTrace(
        candidates=candidates,
        scores=values,
        order=order,
        final_k=effective_final_k,
        score_kind="raw_logit",
        max_sequence_length=512,
    )
    ordered = [(candidates[position], values[position]) for position in order]
    return RerankResult(
        results=reranked_hits(ordered, effective_final_k),
        timing_ms=float(timing_ms),
        trace=trace,
    )


def _cache_ref_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def beir_row_from_shared_candidates(
    *,
    question: Mapping[str, Any],
    position: int,
    batch: SharedCandidateBatch,
    chunk_store: Any,
    reranker: Any,
    method: str,
    physical_candidate_k: int,
    condition_candidate_k: int,
    final_k: int,
    build_id: str,
    run_spec_sha256: str,
    save_text: bool,
    split: str,
    dataset: str,
    unit: str,
    shared_candidate_cache_ref_sha256: str,
    rerank_scores: Sequence[float] | None = None,
    rerank_timing_ms: float | None = None,
    shared_rerank_cache_ref_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one standard BEIR result row from a shared Top-k candidate prefix."""

    if method not in {"bm25", "dense"}:
        raise ValueError("method must be bm25 or dense")
    if condition_candidate_k > physical_candidate_k:
        raise ValueError("condition_candidate_k cannot exceed the shared candidate depth")
    candidate_cache_ref = _cache_ref_sha256(
        shared_candidate_cache_ref_sha256,
        label="shared_candidate_cache_ref_sha256",
    )
    if rerank_scores is not None:
        rerank_cache_ref = _cache_ref_sha256(
            shared_rerank_cache_ref_sha256,
            label="shared_rerank_cache_ref_sha256",
        )
    elif shared_rerank_cache_ref_sha256 is not None:
        raise ValueError("A rerank cache reference requires cached rerank scores")
    else:
        rerank_cache_ref = None
    started = time.perf_counter()
    scores, vector_ids = batch.row(position)
    scores = scores[:condition_candidate_k]
    vector_ids = vector_ids[:condition_candidate_k]
    mapping_started = time.perf_counter()
    chunks = chunk_store.get_many([int(value) for value in vector_ids])
    candidate_hits = tuple(
        SearchHit(rank=rank, chunk=chunk, score=float(score))
        for rank, (chunk, score) in enumerate(zip(chunks, scores), start=1)
    )
    mapping_ms = (time.perf_counter() - mapping_started) * 1000
    if rerank_scores is None:
        reranked = reranker.rerank(
            str(question["question"]),
            candidate_hits,
            final_k=final_k,
        )
    else:
        if rerank_timing_ms is None:
            raise ValueError("Cached rerank scores require rerank_timing_ms")
        reranked = rerank_result_from_scores(
            candidate_hits,
            rerank_scores,
            final_k=final_k,
            timing_ms=rerank_timing_ms,
        )
    final_hits = reranked.results
    qrels = question["qrels"]
    candidate_doc_ids = [hit.chunk.doc_id for hit in candidate_hits]
    metrics = score_beir_query(
        [hit.chunk.doc_id for hit in final_hits], qrels, k=final_k
    )
    first_stage_metrics = score_beir_query(candidate_doc_ids, qrels, k=final_k)
    candidate_pool_metrics = score_beir_query(
        candidate_doc_ids, qrels, k=condition_candidate_k
    )
    row_first_stage_timings = dict(batch.per_question_timings_ms[position])
    if "total_ms" in row_first_stage_timings:
        allocated_first_stage_ms = float(row_first_stage_timings["total_ms"])
        latency_accounting = "measured_per_query_first_stage_plus_rerank"
    else:
        batch_first_stage_ms = float(batch.timings_ms.get("total_ms", 0.0))
        batch_first_stage_ms += float(
            batch.timings_ms.get("identical_id_filter_ms", 0.0)
        )
        allocated_first_stage_ms = batch_first_stage_ms / len(batch.question_ids)
        latency_accounting = "equal_share_of_dense_batch_plus_rerank"
    post_cache_ms = (time.perf_counter() - started) * 1000
    accounted_total_ms = (
        allocated_first_stage_ms + mapping_ms + float(reranked.timing_ms)
    )
    retrieval: dict[str, Any] = {
        "method": method,
        "candidate_k": condition_candidate_k,
        "top_k": final_k,
        "results": [hit.to_dict(include_text=save_text) for hit in final_hits],
        "shared_first_stage": {
            "physical_candidate_k": physical_candidate_k,
            "reused_cache": batch.reused_cache,
            "cache_ref_sha256": candidate_cache_ref,
            "cache_position": position,
            "condition_candidate_count": len(candidate_hits),
        },
        "timings_ms": {
            **{
                f"first_stage_{key}": value
                for key, value in row_first_stage_timings.items()
            },
            "allocated_first_stage_ms": allocated_first_stage_ms,
            "chunk_mapping_from_cache_ms": mapping_ms,
            "rerank_ms": float(reranked.timing_ms),
            "post_cache_total_ms": post_cache_ms,
        },
        "latency_accounting": latency_accounting,
    }
    if reranked.trace is not None:
        if rerank_scores is not None:
            # Candidate ids/retrieval scores and all BGE logits already live in
            # the identity-bound NPZ caches. A second 50-row JSON trace per
            # question only duplicates those arrays and becomes very large on
            # BEIR. Keep the exact row positions needed to reconstruct it.
            retrieval["rerank"] = {
                "candidate_k": len(candidate_hits),
                "final_k": reranked.trace.final_k,
                "score_kind": reranked.trace.score_kind,
                "max_sequence_length": reranked.trace.max_sequence_length,
                "shared_score_cache": True,
                "cache_ref_sha256": rerank_cache_ref,
                "cache_position": position,
            }
        else:
            retrieval["rerank"] = reranked.trace.to_dict()
    return {
        "status": "success",
        "question_id": question["question_id"],
        "question": question["question"],
        "identity": {
            "build_id": build_id,
            "run_spec_sha256": run_spec_sha256,
        },
        "qrels": dict(qrels),
        "retrieval": retrieval,
        "metrics": metrics,
        "first_stage_metrics": first_stage_metrics,
        "candidate_pool_metrics": candidate_pool_metrics,
        "total_latency_ms": accounted_total_ms,
        "evaluation": {
            "protocol": SUITE_EVALUATION_PROTOCOL,
            "dataset": dataset,
            "unit": unit,
            "split": split,
            "shared_first_stage_candidates": True,
        },
    }


_METRIC_PREFIXES = (
    "ndcg_at_",
    "map_at_",
    "recall_at_",
    "precision_at_",
    "mrr_at_",
    "hit_at_",
    "first_stage_",
    "candidate_pool_",
)


def macro_metric_values(summaries: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not summaries:
        return {}
    names = set.intersection(
        *(
            {
                key
                for key, value in summary.items()
                if (
                    key.startswith(_METRIC_PREFIXES)
                    or key == "rerank_delta_ndcg"
                )
                and not key.endswith("_valid_count")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
            for summary in summaries
        )
    )
    return {
        name: mean(float(summary[name]) for summary in summaries)
        for name in sorted(names)
    }


def aggregate_suite_summaries(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return per-unit results plus unit- and family-weighted macro averages."""

    if not records:
        raise ValueError("records must not be empty")
    normalized: list[dict[str, Any]] = []
    for record in records:
        dataset = record.get("dataset")
        unit = record.get("unit")
        condition = record.get("condition")
        summary = record.get("summary")
        if any(not isinstance(value, str) or not value for value in (dataset, unit, condition)):
            raise ValueError("Every suite record requires dataset, unit, and condition")
        if not isinstance(summary, Mapping):
            raise TypeError("Every suite record requires a summary mapping")
        normalized.append(
            {
                "dataset": dataset,
                "unit": unit,
                "condition": condition,
                "summary": dict(summary),
            }
        )

    family_rows: list[dict[str, Any]] = []
    by_family_condition: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in normalized:
        by_family_condition[(record["dataset"], record["condition"])].append(
            record["summary"]
        )
    for (dataset, condition), summaries in sorted(by_family_condition.items()):
        family_rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "num_units": len(summaries),
                "metrics": macro_metric_values(summaries),
            }
        )

    unit_macro: dict[str, dict[str, Any]] = {}
    by_condition: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in normalized:
        by_condition[record["condition"]].append(record["summary"])
    for condition, summaries in sorted(by_condition.items()):
        unit_macro[condition] = {
            "num_units": len(summaries),
            "metrics": macro_metric_values(summaries),
        }

    family_macro: dict[str, dict[str, Any]] = {}
    family_by_condition: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in family_rows:
        family_by_condition[row["condition"]].append(row["metrics"])
    for condition, summaries in sorted(family_by_condition.items()):
        family_macro[condition] = {
            "num_families": len(summaries),
            "metrics": macro_metric_values(summaries),
        }

    return {
        "schema_version": 1,
        "unit_results": normalized,
        "family_results": family_rows,
        "unit_macro": unit_macro,
        "family_macro": family_macro,
    }


__all__ = [
    "BM25FirstStageCheckpointStore",
    "SHARED_CANDIDATE_FILE",
    "SHARED_CANDIDATE_MANIFEST",
    "SHARED_CACHE_REFERENCE_SCHEMA_VERSION",
    "SHARED_RERANK_FILE",
    "SHARED_RERANK_MANIFEST",
    "SUITE_EVALUATION_PROTOCOL",
    "SharedCandidateBatch",
    "SharedCandidateStore",
    "SharedRerankScoreBatch",
    "SharedRerankScoreStore",
    "aggregate_suite_summaries",
    "beir_row_from_shared_candidates",
    "compute_bm25_first_stage",
    "shared_batch_from_dense",
    "rerank_result_from_scores",
    "validate_shared_candidate_batch",
]
