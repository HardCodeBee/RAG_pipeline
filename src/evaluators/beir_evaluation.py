"""BEIR retrieval-only question selection, candidate caching, and summaries."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation_runner import validate_questions
from src.evaluators.beir_metrics import (
    METRICS_VERSION,
    score_beir_query,
    summarize_beir_rows,
)
from src.loaders.beir_loader import BeirCorpusLoader
from src.persistence.artifact_io import atomic_write_npz, read_json_object
from src.persistence.run_output_writer import write_metadata_json
from src.provenance import json_sha256, sha256_file


EVALUATION_PROTOCOL = "beir_retrieval_only_v1"
CANDIDATE_CACHE_SCHEMA_VERSION = 1
CANDIDATE_CACHE_FILE = "first_stage_candidates.npz"
CANDIDATE_CACHE_MANIFEST = "first_stage_candidates.manifest.json"


@dataclass(frozen=True, slots=True)
class VerifiedBeirQuestionSet:
    dataset: str
    unit: str
    split: str
    questions: tuple[dict[str, Any], ...]
    dataset_manifest_path: Path
    dataset_manifest_sha256: str
    queries_path: Path
    queries_file_sha256: str
    qrels_path: Path
    qrels_file_sha256: str


@dataclass(frozen=True, slots=True)
class FirstStageCandidateBatch:
    scores: np.ndarray
    vector_ids: np.ndarray
    question_ids: tuple[str, ...]
    timings_ms: Mapping[str, float]
    reused_cache: bool = False


def load_beir_question_split(
    corpus_path: str | Path,
    *,
    split: str = "test",
    max_questions: int | None = None,
    expected_dataset: str | None = None,
    loader: BeirCorpusLoader | None = None,
) -> VerifiedBeirQuestionSet:
    """Select only queries referenced by one qrels split, in query-file order."""

    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    if max_questions is not None and (
        isinstance(max_questions, bool)
        or not isinstance(max_questions, int)
        or max_questions <= 0
    ):
        raise ValueError("max_questions must be a positive integer or None")

    root = Path(corpus_path).resolve()
    active_loader = loader or BeirCorpusLoader(expected_dataset=expected_dataset)
    manifest = active_loader.manifest(root)

    qrels_by_query: dict[str, dict[str, float]] = {}
    for qrel in active_loader.iter_qrels(root, split=split):
        query_qrels = qrels_by_query.setdefault(qrel.query_id, {})
        if qrel.corpus_id in query_qrels:
            raise ValueError(
                f"Duplicate BEIR qrel pair: {qrel.query_id!r}/{qrel.corpus_id!r}"
            )
        query_qrels[qrel.corpus_id] = float(qrel.score)
    if not qrels_by_query:
        raise RuntimeError(f"BEIR qrels split is empty: {split}")

    questions: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for query in active_loader.iter_queries(root):
        qrels = qrels_by_query.get(query.query_id)
        if qrels is None:
            continue
        questions.append(
            {
                "question_id": query.query_id,
                "question": query.text,
                "qrels": dict(qrels),
            }
        )
        selected_ids.add(query.query_id)
        if max_questions is not None and len(questions) >= max_questions:
            break
    if not questions:
        raise RuntimeError(f"No BEIR queries were selected for split {split}")
    if max_questions is None:
        missing = sorted(set(qrels_by_query) - selected_ids)
        if missing:
            raise ValueError(
                "BEIR qrels reference queries absent from queries.jsonl: "
                + ", ".join(missing[:5])
            )
    validated = validate_questions(questions)

    queries_descriptor = manifest["artifacts"]["queries"]
    qrels_descriptor = manifest["artifacts"]["qrels"][split]
    return VerifiedBeirQuestionSet(
        dataset=str(manifest["dataset"]),
        unit=str(manifest["unit"]),
        split=split,
        questions=tuple(validated),
        dataset_manifest_path=(root / "manifest.json").resolve(),
        dataset_manifest_sha256=sha256_file(root / "manifest.json"),
        queries_path=(root / queries_descriptor["file"]).resolve(),
        queries_file_sha256=str(queries_descriptor["sha256"]),
        qrels_path=(root / qrels_descriptor["file"]).resolve(),
        qrels_file_sha256=str(qrels_descriptor["sha256"]),
    )


def first_stage_identity(
    *,
    questions_sha256: str,
    question_ids: Sequence[str],
    build_id: str,
    run_spec_sha256: str,
    retrieval_method: str,
    index_type: str,
    candidate_k: int,
    query_batch_size: int,
) -> dict[str, Any]:
    """Describe everything allowed to select first-stage candidates."""

    if retrieval_method != "dense" or index_type != "streaming_flat_ip":
        raise ValueError("Candidate cache identity requires dense streaming_flat_ip")
    for value, label in (
        (questions_sha256, "questions_sha256"),
        (build_id, "build_id"),
        (run_spec_sha256, "run_spec_sha256"),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")
    for value, label in (
        (candidate_k, "candidate_k"),
        (query_batch_size, "query_batch_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    ids = list(question_ids)
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("question_ids must contain non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("question_ids must be unique")
    return {
        "schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
        "questions_sha256": questions_sha256,
        "question_ids_sha256": json_sha256(ids),
        "build_id": build_id,
        "run_spec_sha256": run_spec_sha256,
        "retrieval_method": retrieval_method,
        "index_type": index_type,
        "candidate_k": candidate_k,
        "query_batch_size": query_batch_size,
    }


def _validate_candidate_arrays(
    scores: np.ndarray,
    vector_ids: np.ndarray,
    question_ids: Sequence[str],
    *,
    expected_question_ids: Sequence[str],
    candidate_k: int,
    corpus_count: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    values = np.asarray(scores)
    ids = np.asarray(vector_ids)
    actual_question_ids = tuple(str(value) for value in question_ids)
    expected_ids = tuple(expected_question_ids)
    result_k = min(candidate_k, corpus_count)
    expected_shape = (len(expected_ids), result_k)
    if values.dtype != np.dtype("float32") or values.shape != expected_shape:
        raise ValueError(
            f"Candidate scores must be float32 with shape {expected_shape}"
        )
    if ids.dtype != np.dtype("int64") or ids.shape != expected_shape:
        raise ValueError(
            f"Candidate vector ids must be int64 with shape {expected_shape}"
        )
    if actual_question_ids != expected_ids:
        raise ValueError("Candidate cache question order does not match the selected set")
    if not np.isfinite(values).all():
        raise ValueError("Candidate cache contains non-finite scores")
    if np.any(ids < 0) or np.any(ids >= corpus_count):
        raise ValueError("Candidate cache contains out-of-range vector ids")
    expected_order = np.arange(result_k)
    for row in range(values.shape[0]):
        if len(np.unique(ids[row])) != result_k:
            raise ValueError("Candidate cache contains duplicate ids within a query")
        order = np.lexsort((ids[row], -values[row]))
        if not np.array_equal(order, expected_order):
            raise ValueError(
                "Candidate cache must use score-descending/id-ascending order"
            )
    return (
        np.ascontiguousarray(values),
        np.ascontiguousarray(ids),
        actual_question_ids,
    )


def compute_streaming_first_stage(
    pipeline: Any,
    questions: Sequence[Mapping[str, Any]],
    *,
    candidate_k: int,
    query_batch_size: int,
) -> FirstStageCandidateBatch:
    """Encode the selected query set once and scan the streaming index once."""

    if not questions:
        raise ValueError("questions must not be empty")
    if pipeline.embedder is None or pipeline.index is None:
        raise ValueError("Dense streaming retrieval requires an embedder and index")
    if getattr(pipeline.index, "index_type", None) != "streaming_flat_ip":
        raise ValueError("Batch first-stage search requires streaming_flat_ip")
    search_many = getattr(pipeline.index, "search_many", None)
    if not callable(search_many):
        raise TypeError("streaming_flat_ip index must implement search_many")

    question_ids = tuple(str(question["question_id"]) for question in questions)
    texts = [str(question["question"]).strip() for question in questions]
    started = time.perf_counter()
    embedding_started = time.perf_counter()
    query_embeddings = pipeline.embedder.encode_queries(texts)
    query_embedding_ms = (time.perf_counter() - embedding_started) * 1000
    search_started = time.perf_counter()
    scores, vector_ids = search_many(
        query_embeddings,
        candidate_k,
        query_batch_size=query_batch_size,
    )
    index_search_ms = (time.perf_counter() - search_started) * 1000
    del query_embeddings
    scores, vector_ids, question_ids = _validate_candidate_arrays(
        scores,
        vector_ids,
        question_ids,
        expected_question_ids=question_ids,
        candidate_k=candidate_k,
        corpus_count=int(pipeline.index.count),
    )
    return FirstStageCandidateBatch(
        scores=scores,
        vector_ids=vector_ids,
        question_ids=question_ids,
        timings_ms={
            "query_embedding_ms": query_embedding_ms,
            "index_search_ms": index_search_ms,
            "total_ms": (time.perf_counter() - started) * 1000,
        },
    )


class FirstStageCandidateStore:
    """Atomic, identity-bound storage for one dense batch search result."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        identity: Mapping[str, Any],
        question_ids: Sequence[str],
        corpus_count: int,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.data_path = self.run_dir / CANDIDATE_CACHE_FILE
        self.manifest_path = self.run_dir / CANDIDATE_CACHE_MANIFEST
        self.identity = dict(identity)
        self.question_ids = tuple(question_ids)
        self.corpus_count = int(corpus_count)
        self.candidate_k = int(self.identity["candidate_k"])
        if self.corpus_count <= 0:
            raise ValueError("corpus_count must be positive")

    def write(self, batch: FirstStageCandidateBatch) -> FirstStageCandidateBatch:
        scores, vector_ids, question_ids = _validate_candidate_arrays(
            batch.scores,
            batch.vector_ids,
            batch.question_ids,
            expected_question_ids=self.question_ids,
            candidate_k=self.candidate_k,
            corpus_count=self.corpus_count,
        )
        if self.manifest_path.exists():
            raise FileExistsError(f"Candidate cache manifest already exists: {self.manifest_path}")
        committed = FirstStageCandidateBatch(
            scores=scores,
            vector_ids=vector_ids,
            question_ids=question_ids,
            timings_ms=dict(batch.timings_ms),
            reused_cache=False,
        )
        atomic_write_npz(
            self.data_path,
            scores=committed.scores,
            vector_ids=committed.vector_ids,
            question_ids=np.asarray(committed.question_ids, dtype=np.str_),
        )
        manifest = {
            "status": "complete",
            "schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
            "identity": self.identity,
            "artifact": {
                "file": CANDIDATE_CACHE_FILE,
                "size_bytes": self.data_path.stat().st_size,
                "sha256": sha256_file(self.data_path),
            },
            "timings_ms": dict(batch.timings_ms),
        }
        # The manifest is committed last and is the cache completion marker.
        write_metadata_json(self.manifest_path, manifest, overwrite=False)
        return committed

    def load(self) -> FirstStageCandidateBatch | None:
        if not self.manifest_path.exists():
            return None
        if not self.data_path.is_file():
            raise FileNotFoundError(
                "Candidate cache manifest exists but its NPZ artifact is missing"
            )
        manifest = read_json_object(
            self.manifest_path,
            label="First-stage candidate manifest",
        )
        artifact = manifest.get("artifact")
        if (
            manifest.get("status") != "complete"
            or manifest.get("schema_version") != CANDIDATE_CACHE_SCHEMA_VERSION
            or manifest.get("identity") != self.identity
            or not isinstance(artifact, Mapping)
            or artifact.get("file") != CANDIDATE_CACHE_FILE
        ):
            raise ValueError("First-stage candidate cache identity is incompatible")
        if (
            self.data_path.stat().st_size != artifact.get("size_bytes")
            or sha256_file(self.data_path) != artifact.get("sha256")
        ):
            raise ValueError("First-stage candidate cache artifact is corrupted")
        try:
            with np.load(self.data_path, allow_pickle=False) as arrays:
                if set(arrays.files) != {"scores", "vector_ids", "question_ids"}:
                    raise ValueError("Candidate cache NPZ has unexpected arrays")
                scores = np.array(arrays["scores"], copy=True)
                vector_ids = np.array(arrays["vector_ids"], copy=True)
                question_ids = tuple(str(value) for value in arrays["question_ids"])
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot load first-stage candidate cache NPZ") from exc
        scores, vector_ids, question_ids = _validate_candidate_arrays(
            scores,
            vector_ids,
            question_ids,
            expected_question_ids=self.question_ids,
            candidate_k=self.candidate_k,
            corpus_count=self.corpus_count,
        )
        timings = manifest.get("timings_ms")
        if not isinstance(timings, Mapping):
            raise ValueError("Candidate cache manifest has no timing metadata")
        return FirstStageCandidateBatch(
            scores=scores,
            vector_ids=vector_ids,
            question_ids=question_ids,
            timings_ms=dict(timings),
            reused_cache=True,
        )


def summarize_beir_evaluation(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    unit: str,
    split: str,
    final_k: int,
    candidate_k: int,
    retrieval_method: str,
    reranker_provider: str,
) -> dict[str, Any]:
    """Summarize final, first-stage, and candidate-pool rankings separately."""

    summary = summarize_beir_rows(rows, k=final_k)
    first_stage_rows = [
        {"status": row.get("status"), "metrics": row.get("first_stage_metrics", {})}
        for row in rows
    ]
    first_stage = summarize_beir_rows(first_stage_rows, k=final_k)
    candidate_rows = [
        {"status": row.get("status"), "metrics": row.get("candidate_pool_metrics", {})}
        for row in rows
    ]
    candidate_pool = summarize_beir_rows(candidate_rows, k=candidate_k)
    for key, value in first_stage.items():
        if key.startswith(("ndcg_", "map_", "recall_", "precision_", "mrr_", "hit_")):
            summary[f"first_stage_{key}"] = value
    for key, value in candidate_pool.items():
        if key.startswith(("ndcg_", "map_", "recall_", "precision_", "mrr_", "hit_")):
            summary[f"candidate_pool_{key}"] = value
    ndcg_key = f"ndcg_at_{final_k}"
    summary["rerank_delta_ndcg"] = summary[ndcg_key] - first_stage[ndcg_key]
    summary.update(
        {
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "dataset": dataset,
            "unit": unit,
            "split": split,
            "retrieval_method": retrieval_method,
            "reranker_provider": reranker_provider,
            "candidate_k": candidate_k,
            "final_k": final_k,
        }
    )
    return summary


__all__ = [
    "CANDIDATE_CACHE_FILE",
    "CANDIDATE_CACHE_MANIFEST",
    "EVALUATION_PROTOCOL",
    "METRICS_VERSION",
    "FirstStageCandidateBatch",
    "FirstStageCandidateStore",
    "VerifiedBeirQuestionSet",
    "compute_streaming_first_stage",
    "first_stage_identity",
    "load_beir_question_split",
    "score_beir_query",
    "summarize_beir_evaluation",
    "summarize_beir_rows",
]
