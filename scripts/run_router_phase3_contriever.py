"""Run the 1,200-query BM25/Contriever F1 routing revision."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_router_phase1 import (  # noqa: E402
    GENERATOR_PRICES,
    _base_row,
    _core_config,
    _deduplicated_hits,
    _load_router_config,
    _prompt_for_row,
    _usage_cost,
)
from scripts.run_router_phase2 import _load_examples  # noqa: E402
from scripts.run_router_phase3 import _generator_view  # noqa: E402
from scripts.run_router_phase26_bd_f1_gate import (  # noqa: E402
    _fit_model,
    _group_folds,
    _new_model,
    _predict_scores,
    choose_dense_default_threshold,
)
from scripts.run_router_phase3_train import (  # noqa: E402
    _corpus_prototypes,
    _dense_summary,
    _lexical_features,
)
from src.embedders.text_embedder import create_embedder  # noqa: E402
from src.evaluators.beir_evaluation import compute_streaming_first_stage  # noqa: E402
from src.evaluators.hotpot_answer import answer_metrics  # noqa: E402
from src.persistence.artifact_validation import validate_build_directory  # noqa: E402
from src.records import EmbeddingSpaceSpec  # noqa: E402
from src.retrievers.chunk_store import JsonlOffsetChunkStore  # noqa: E402
from src.retrievers.sqlite_bm25 import (  # noqa: E402
    analyze_sqlite_bm25_text,
    read_sqlite_bm25_term_stats,
)
from src.text.token_counters import RegexTokenCounter  # noqa: E402
from src.vector_index_factory import create_index  # noqa: E402


ROUTED_ACTIONS = ("bm25", "contriever")
REFERENCE_ACTION = "bge"
SOURCE_ACTIONS = ("bm25", "dense")
TIE_ATOL = 1e-12


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="outputs/router/hotpotqa_bc_router_v1/config.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("audit", "prepare", "generate", "features", "fit", "status"),
        required=True,
    )
    parser.add_argument("--max-new-calls", type=int, default=None)
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_yaml(path)
    router = _load_router_config(_resolve(str(protocol["base_config"])))
    dense = protocol["contriever"]
    if not isinstance(dense, Mapping):
        raise TypeError("contriever must be a mapping")
    router["retrieval"]["dense"] = {
        key: dense[key]
        for key in (
            "backend",
            "family",
            "model",
            "revision",
            "pooling",
            "document_input_format",
            "normalize",
            "query_prefix",
            "document_prefix",
            "max_sequence_length",
            "batch_size",
            "encode_call_rows",
            "shard_rows",
            "device",
            "index",
        )
    }
    router["artifacts"]["encoded_corpus"] = str(dense["encoded_corpus"])
    _validate_protocol(protocol, router)
    return protocol, router


def _validate_protocol(protocol: Mapping[str, Any], router: Mapping[str, Any]) -> None:
    sample = protocol["sample"]
    actions = protocol["actions"]
    gate = protocol["gate"]
    if int(sample["query_prefix_size"]) != 1_200:
        raise ValueError("The Contriever revision must start at 1,200 queries")
    if int(sample["repeats_per_action"]) != 3:
        raise ValueError("Each action must have exactly three repeats")
    if tuple(actions["routed"]) != ROUTED_ACTIONS:
        raise ValueError(f"Routed actions must be {ROUTED_ACTIONS}")
    if actions["default"] != "bm25" or actions["external_reference"] != REFERENCE_ACTION:
        raise ValueError("BM25 must be default and BGE must be the external reference")
    if int(gate["answer_correctness_calls"]) != 0:
        raise ValueError("This Phase 3 revision must not call Answer Correctness")
    if protocol["features"].get("normalize_corpus_prototypes") is not False:
        raise ValueError("Raw-dot Contriever corpus prototypes must remain unnormalized")
    if bool(gate["fresh_dev_opened"]) or bool(gate["final_holdout_opened"]):
        raise ValueError("Fresh dev and final holdout must remain sealed")

    core = _core_config(router, method="dense")
    embedding = core["embedding"]
    required = {
        "backend": "hf_dense",
        "family": "contriever",
        "model_name": "facebook/contriever",
        "pooling": "masked_mean",
        "document_input_format": "title_space_text",
        "normalize": False,
        "max_sequence_length": 512,
    }
    if any(embedding.get(key) != value for key, value in required.items()):
        raise ValueError("The active Contriever embedding contract is incorrect")


def _run_dir(protocol: Mapping[str, Any]) -> Path:
    return _resolve(str(protocol["run_dir"]))


def _source_database(protocol: Mapping[str, Any]) -> Path:
    return _resolve(str(protocol["source_run"])) / "state.sqlite3"


def _readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def _source_rows(
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    limit = int(protocol["sample"]["query_prefix_size"])
    connection = _readonly_connection(_source_database(protocol))
    try:
        samples = connection.execute(
            """
            SELECT query_rank, query_id, group_id
            FROM sample_rows
            WHERE partition = 'train' AND query_rank < ?
            ORDER BY query_rank
            """,
            (limit,),
        ).fetchall()
        outcomes = connection.execute(
            """
            SELECT s.query_rank, s.query_id, s.group_id,
                   o.action, o.repeat_id, o.status, o.payload
            FROM sample_rows AS s
            JOIN outcomes AS o ON o.query_id = s.query_id
            WHERE s.partition = 'train' AND s.query_rank < ?
              AND o.action IN ('bm25', 'dense')
            ORDER BY s.query_rank,
                     CASE o.action WHEN 'bm25' THEN 0 ELSE 1 END,
                     o.repeat_id
            """,
            (limit,),
        ).fetchall()
    finally:
        connection.close()

    if len(samples) != limit:
        raise ValueError(f"Expected {limit} source queries, found {len(samples)}")
    expected_outcomes = limit * len(SOURCE_ACTIONS) * int(
        protocol["sample"]["repeats_per_action"]
    )
    if len(outcomes) != expected_outcomes:
        raise ValueError(f"Expected {expected_outcomes} source outcomes, found {len(outcomes)}")

    sample_rows = [
        {"rank": int(rank), "query_id": str(query_id), "group_id": str(group_id)}
        for rank, query_id, group_id in samples
    ]
    projected: list[dict[str, Any]] = []
    cells: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    for rank, query_id, group_id, action, repeat_id, status, payload in outcomes:
        if status != "success":
            raise ValueError("The source contains an incomplete BGE/BM25 outcome")
        row = json.loads(payload)
        f1 = float(row["metrics"]["normalized_token_f1"])
        if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
            raise ValueError("Source F1 must be finite and in [0, 1]")
        projected_action = "bm25" if action == "bm25" else REFERENCE_ACTION
        repeat_id = int(repeat_id)
        cells[(str(query_id), projected_action)].add(repeat_id)
        copied = copy.deepcopy(row)
        copied["action"] = projected_action
        projected.append(
            {
                "rank": int(rank),
                "query_id": str(query_id),
                "group_id": str(group_id),
                "action": projected_action,
                "repeat_id": repeat_id,
                "status": "success",
                "payload": copied,
            }
        )
    expected_repeats = set(range(int(protocol["sample"]["repeats_per_action"])))
    if any(repeats != expected_repeats for repeats in cells.values()):
        raise ValueError("Every source query/action must contain repeats 0/1/2")
    if len(cells) != limit * 2:
        raise ValueError("Source query/action cells are incomplete")
    return sample_rows, projected


def _connect(run_dir: Path) -> sqlite3.Connection:
    run_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(run_dir / "state.sqlite3", timeout=60.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sample_rows (
            query_rank INTEGER NOT NULL,
            query_id TEXT NOT NULL PRIMARY KEY,
            group_id TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS outcomes (
            query_id TEXT NOT NULL,
            action TEXT NOT NULL,
            repeat_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (query_id, action, repeat_id)
        )
        """
    )
    connection.commit()
    return connection


def _initialize_state(protocol: Mapping[str, Any]) -> dict[str, Any]:
    samples, outcomes = _source_rows(protocol)
    connection = _connect(_run_dir(protocol))
    try:
        for row in samples:
            connection.execute(
                "INSERT OR IGNORE INTO sample_rows VALUES (?, ?, ?)",
                (row["rank"], row["query_id"], row["group_id"]),
            )
        for row in outcomes:
            connection.execute(
                "INSERT OR IGNORE INTO outcomes VALUES (?, ?, ?, ?, ?)",
                (
                    row["query_id"],
                    row["action"],
                    row["repeat_id"],
                    row["status"],
                    json.dumps(row["payload"], ensure_ascii=False, allow_nan=False),
                ),
            )
        connection.commit()
        counts = {
            str(action): int(count)
            for action, count in connection.execute(
                "SELECT action, COUNT(*) FROM outcomes GROUP BY action"
            )
        }
    finally:
        connection.close()
    return {"queries": len(samples), "source_outcome_rows": counts}


def audit(protocol: Mapping[str, Any], router: Mapping[str, Any]) -> dict[str, Any]:
    initialized = _initialize_state(protocol)
    encoded = _resolve(str(protocol["contriever"]["encoded_corpus"]))
    manifest = json.loads((encoded / "manifest.json").read_text(encoding="utf-8"))
    space = manifest.get("embedding", {}).get("space", {})
    required_space = {
        "encoder_family": "contriever",
        "pooling": "masked_mean",
        "document_input_format": "title_space_text",
        "normalized": False,
        "dimension": 768,
    }
    if any(space.get(key) != value for key, value in required_space.items()):
        raise ValueError("The stored Contriever corpus space does not match the query encoder")
    parts = manifest.get("artifacts", {}).get("embeddings", {}).get("parts", [])
    if sum(int(part["rows"]) for part in parts) != 5_233_329:
        raise ValueError("The Contriever corpus embedding rows are incomplete")
    result = {
        **initialized,
        "contriever_contract": "validated",
        "contriever_embedding_dimensions": 768,
        "contriever_corpus_rows": 5_233_329,
        "fresh_dev_outcomes_read": 0,
        "final_holdout_outcomes_read": 0,
        "answer_correctness_calls": 0,
    }
    _write_json(_run_dir(protocol) / "validation.json", result)
    return result


def _sample_for_examples(protocol: Mapping[str, Any]) -> dict[str, Any]:
    samples, _ = _source_rows(protocol)
    return {
        "rows": [
            {
                "query_id": row["query_id"],
                "group_id": row["group_id"],
                "partition": "train",
            }
            for row in samples
        ]
    }


@contextmanager
def _verified_contriever_runtime(
    protocol: Mapping[str, Any],
    core: dict[str, Any],
):
    """Open the frozen legacy Contriever build without weakening core identity rules.

    The generic pipeline deliberately accepts only the build derived from the active
    source tree or one pinned in its registry.  This experiment predates that registry,
    so it opens the explicitly named immutable build at a narrow read-only boundary,
    while retaining full artifact, embedding-space, index, and row validation.
    """

    build_dir = _resolve(str(protocol["contriever"]["encoded_corpus"]))
    verified = validate_build_directory(build_dir, expected_build_id=build_dir.name)
    manifest = verified.manifest
    artifacts = manifest["artifacts"]
    expected_space = EmbeddingSpaceSpec.from_mapping(manifest["embedding"]["space"])

    with ExitStack() as stack:
        embedder = create_embedder(core, role="query")
        embedder_close = getattr(embedder, "close", None)
        if callable(embedder_close):
            stack.callback(embedder_close)
        actual_space = embedder.embedding_space(expected_space.similarity)
        if actual_space != expected_space:
            raise ValueError(
                "Contriever query encoder does not match the immutable corpus space: "
                f"{actual_space.to_dict()} != {expected_space.to_dict()}"
            )

        offsets = artifacts.get("chunk_offsets")
        if not isinstance(offsets, Mapping):
            raise ValueError("The frozen Contriever build requires a chunk-offset artifact")
        chunk_store = JsonlOffsetChunkStore(
            verified.files["chunks"],
            verified.files["chunk_offsets"],
            expected_rows=int(artifacts["chunks"]["rows"]),
        )
        stack.callback(chunk_store.close)
        manifest_index = manifest["index"]
        index = create_index(
            core,
            backend=str(manifest_index["backend"]),
            index_type=str(manifest_index["type"]),
            threads=(
                int(core["retrieval"].get("search_threads", 0))
                or int(core["index"].get("faiss_threads", 0))
            ),
        )
        index_close = getattr(index, "close", None)
        if callable(index_close):
            stack.callback(index_close)
        expected_build_params = dict(index.build_params)
        embeddings_path = verified.files["embeddings"]
        embedding_source = (
            embeddings_path.parent
            if artifacts["embeddings"].get("storage") == "sharded_npy"
            else embeddings_path
        )
        index.load(embedding_source)

        if index.backend != manifest_index["backend"] or index.index_type != manifest_index["type"]:
            raise ValueError("Loaded Contriever index backend/type differs from its manifest")
        if index.count != len(chunk_store) or manifest_index.get("count") not in {
            None,
            index.count,
        }:
            raise ValueError("Loaded Contriever index count differs from chunks or manifest")
        if manifest_index.get("dimension") not in {None, index.dimension}:
            raise ValueError("Loaded Contriever index dimension differs from its manifest")
        if index.dimension != expected_space.dimension:
            raise ValueError("Loaded Contriever index and query encoder dimensions differ")
        if index.build_params != expected_build_params:
            raise ValueError("Loaded Contriever index changed the requested build parameters")
        manifest_build_params = manifest_index.get("build_params")
        if manifest_build_params is not None and manifest_build_params != index.build_params:
            raise ValueError("Loaded Contriever index build parameters differ from its manifest")
        if index.ids is not None:
            raise ValueError("The streaming Contriever index must use implicit contiguous ids")

        runtime = SimpleNamespace(embedder=embedder, index=index, chunk_store=chunk_store)
        yield runtime


def prepare(protocol: Mapping[str, Any], router: Mapping[str, Any]) -> dict[str, Any]:
    audit(protocol, router)
    run_dir = _run_dir(protocol)
    examples, mapping = _load_examples(router, _sample_for_examples(protocol))
    repeats = int(protocol["sample"]["repeats_per_action"])
    connection = _connect(run_dir)
    try:
        existing = {
            str(query_id)
            for (query_id,) in connection.execute(
                """
                SELECT query_id FROM outcomes
                WHERE action = 'contriever'
                GROUP BY query_id HAVING COUNT(*) = ?
                """,
                (repeats,),
            )
        }
        pending_examples = [row for row in examples if str(row["query_id"]) not in existing]
        if pending_examples:
            questions = [
                {"question_id": row["query_id"], "question": row["question"], "qrels": {}}
                for row in pending_examples
            ]
            candidate_k = int(router["retrieval"]["candidate_k"])
            final_k = int(router["retrieval"]["final_k"])
            token_counter = RegexTokenCounter()
            core = _core_config(router, method="dense")
            with _verified_contriever_runtime(protocol, core) as pipeline:
                started = time.perf_counter()
                batch = compute_streaming_first_stage(
                    pipeline,
                    questions,
                    candidate_k=candidate_k,
                    query_batch_size=int(protocol["contriever"]["query_batch_size"]),
                )
                retrieval_ms = 1000.0 * (time.perf_counter() - started)
                for position, example in enumerate(pending_examples, start=1):
                    hits = _deduplicated_hits(
                        pipeline.chunk_store,
                        batch.vector_ids[position - 1],
                        batch.scores[position - 1],
                        final_k=final_k,
                    )
                    base = _base_row(
                        example,
                        action="contriever",
                        hits=hits,
                        retrieval_latency_ms=retrieval_ms / len(pending_examples),
                        token_counter=token_counter,
                        context_max_tokens=int(router["context"]["max_tokens"]),
                    )
                    base.pop("dataset_id", None)
                    for repeat_id in range(repeats):
                        row = copy.deepcopy(base)
                        row["repeat_id"] = repeat_id
                        connection.execute(
                            "INSERT OR IGNORE INTO outcomes VALUES (?, 'contriever', ?, ?, ?)",
                            (
                                str(example["query_id"]),
                                repeat_id,
                                str(row["status"]),
                                json.dumps(row, ensure_ascii=False, allow_nan=False),
                            ),
                        )
                    if position % 100 == 0:
                        connection.commit()
                        print(json.dumps({"contriever_queries_prepared": position}), flush=True)
            connection.commit()
        counts = _status_counts(connection)
    finally:
        connection.close()
    result = {
        "queries": len(examples),
        "mapped_queries": int(mapping["mapped"]),
        "status_counts": counts,
    }
    _write_json(run_dir / "prepare.json", result)
    return result


def _status_counts(connection: sqlite3.Connection) -> dict[str, dict[str, int]]:
    result: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for action, status, count in connection.execute(
        "SELECT action, status, COUNT(*) FROM outcomes GROUP BY action, status"
    ):
        result[str(action)][str(status)] = int(count)
    return dict(result)


def status(protocol: Mapping[str, Any]) -> dict[str, Any]:
    connection = _connect(_run_dir(protocol))
    try:
        return {"status_counts": _status_counts(connection)}
    finally:
        connection.close()


def _spent_generation_cost(connection: sqlite3.Connection) -> float:
    total = 0.0
    for (payload,) in connection.execute(
        "SELECT payload FROM outcomes WHERE action = 'contriever' AND status = 'success'"
    ):
        row = json.loads(payload)
        generation = row.get("generation")
        if isinstance(generation, Mapping):
            total += _usage_cost(generation.get("token_usage", {}), GENERATOR_PRICES)
    return total


def generate(
    protocol: Mapping[str, Any],
    router: Mapping[str, Any],
    *,
    max_new_calls: int | None,
) -> dict[str, Any]:
    from src.generators.answer_generator import LLMGenerator

    connection = _connect(_run_dir(protocol))
    created: list[LLMGenerator] = []
    try:
        pending = connection.execute(
            """
            SELECT o.query_id, o.repeat_id, o.payload
            FROM outcomes AS o
            JOIN sample_rows AS s ON s.query_id = o.query_id
            WHERE o.action = 'contriever' AND o.status = 'pending_generation'
            ORDER BY s.query_rank, o.repeat_id
            """
        ).fetchall()
        attempted = int(
            connection.execute(
                "SELECT COUNT(*) FROM outcomes WHERE action='contriever' AND status!='pending_generation'"
            ).fetchone()[0]
        )
        remaining = max(0, int(protocol["generation"]["call_limit"]) - attempted)
        if max_new_calls is not None:
            remaining = min(remaining, int(max_new_calls))
        pending = pending[:remaining]
        if not pending:
            return status(protocol)

        generation = _core_config(router, method="dense")["generation"]
        local = threading.local()
        created_lock = threading.Lock()

        def worker(item: Sequence[Any]) -> tuple[str, int, dict[str, Any]]:
            query_id, repeat_id, payload = item
            generator = getattr(local, "generator", None)
            if generator is None:
                generator = LLMGenerator(
                    provider=generation["provider"],
                    model=generation["model"],
                    temperature=generation["temperature"],
                    max_output_tokens=generation["max_output_tokens"],
                    timeout_seconds=generation["timeout_seconds"],
                    max_retries=generation["max_retries"],
                )
                local.generator = generator
                with created_lock:
                    created.append(generator)
            row = json.loads(payload)
            try:
                result = generator.generate_from_prompt(
                    _prompt_for_row(row), row["question"], []
                )
                prediction = result.answer.strip()
                metrics = answer_metrics(prediction, row["reference_answers"])
                metrics.pop("answer_correctness", None)
                row.update(
                    {
                        "status": "success",
                        "prediction": prediction,
                        "metrics": metrics,
                        "generation": _generator_view(result),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "generation_failure",
                        "generation": {
                            "attempted": True,
                            "status": "failure",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    }
                )
            return str(query_id), int(repeat_id), row

        spent = _spent_generation_cost(connection)
        completed = 0
        hard_budget = float(protocol["generation"]["hard_budget_usd"])
        workers = int(protocol["generation"]["provider_workers"])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(pending), 100):
                batch = pending[start : start + 100]
                futures = [pool.submit(worker, item) for item in batch]
                for future in as_completed(futures):
                    query_id, repeat_id, row = future.result()
                    connection.execute(
                        """
                        UPDATE outcomes SET status=?, payload=?
                        WHERE query_id=? AND action='contriever' AND repeat_id=?
                        """,
                        (
                            str(row["status"]),
                            json.dumps(row, ensure_ascii=False, allow_nan=False),
                            query_id,
                            repeat_id,
                        ),
                    )
                    completed += 1
                    view = row.get("generation")
                    if row["status"] == "success" and isinstance(view, Mapping):
                        spent += _usage_cost(view.get("token_usage", {}), GENERATOR_PRICES)
                connection.commit()
                print(
                    json.dumps(
                        {"generation_new_calls": completed, "estimated_cost_usd": spent}
                    ),
                    flush=True,
                )
                if spent > hard_budget:
                    raise RuntimeError(
                        f"Generation budget exceeded: {spent:.6f} > {hard_budget:.6f}"
                    )
        return status(protocol)
    finally:
        connection.commit()
        connection.close()
        for generator in created:
            close = getattr(getattr(generator, "_client", None), "close", None)
            if callable(close):
                close()


def build_features(
    protocol: Mapping[str, Any], router: Mapping[str, Any]
) -> dict[str, Any]:
    connection = _connect(_run_dir(protocol))
    try:
        selected = [
            {"query_id": str(query_id), "group_id": str(group_id), "partition": "train"}
            for query_id, group_id in connection.execute(
                "SELECT query_id, group_id FROM sample_rows ORDER BY query_rank"
            )
        ]
    finally:
        connection.close()
    expected = int(protocol["sample"]["query_prefix_size"])
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} feature rows, found {len(selected)}")
    examples, mapping = _load_examples(router, {"rows": selected})
    questions = [str(row["question"]) for row in examples]
    terms = {
        token for question in questions for token in analyze_sqlite_bm25_text(question)
    }
    database = _resolve(str(router["artifacts"]["bm25_index"])) / "index.sqlite3"
    stats = read_sqlite_bm25_term_stats(database, terms)
    lexical = np.asarray(
        [_lexical_features(question, stats)[0] for question in questions],
        dtype=np.float32,
    )
    if lexical.shape != (expected, 17):
        raise ValueError(f"Expected a 1200x17 lexical matrix, found {lexical.shape}")

    core = _core_config(router, method="dense")
    embedder = create_embedder(core, role="query")
    embedding = np.asarray(embedder.encode_queries(questions), dtype=np.float32)
    if embedding.shape != (expected, 768) or not np.isfinite(embedding).all():
        raise ValueError("Contriever query embeddings have an invalid shape or value")
    prototypes = _corpus_prototypes(
        router,
        _run_dir(protocol),
        clusters=int(protocol["features"]["corpus_prototypes"]),
        sample_rows=int(protocol["features"]["corpus_prototype_sample_rows"]),
        normalize_centers=bool(protocol["features"]["normalize_corpus_prototypes"]),
    )
    dense, _ = _dense_summary(embedding, prototypes)
    if dense.shape != (expected, 13) or not np.isfinite(dense).all():
        raise ValueError("Contriever corpus-summary features are invalid")

    feature_path = _run_dir(protocol) / "features.npz"
    np.savez_compressed(feature_path, lexical=lexical, dense=dense, embedding=embedding)
    result = {
        "queries": expected,
        "mapped_queries": int(mapping["mapped"]),
        "lexical_dimensions": 17,
        "dense_corpus_dimensions": 13,
        "query_embedding_dimensions": 768,
        "features_finite": True,
        "online_information": "query_and_corpus_static_only",
    }
    _write_json(_run_dir(protocol) / "feature_validation.json", result)
    return result


def _outcome_utilities(
    protocol: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    repeats = int(protocol["sample"]["repeats_per_action"])
    limit = int(protocol["sample"]["query_prefix_size"])
    connection = _connect(_run_dir(protocol))
    try:
        groups = np.asarray(
            [str(group) for (group,) in connection.execute(
                "SELECT group_id FROM sample_rows ORDER BY query_rank"
            )],
            dtype=np.str_,
        )
        rows = connection.execute(
            """
            SELECT s.query_rank, o.action, o.repeat_id, o.status, o.payload
            FROM sample_rows AS s JOIN outcomes AS o ON o.query_id=s.query_id
            WHERE o.action IN ('bm25','contriever','bge')
            ORDER BY s.query_rank,
                     CASE o.action WHEN 'bm25' THEN 0 WHEN 'contriever' THEN 1 ELSE 2 END,
                     o.repeat_id
            """
        ).fetchall()
    finally:
        connection.close()
    expected = limit * 3 * repeats
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} outcome rows, found {len(rows)}")
    values = np.full((limit, 3, repeats), np.nan, dtype=np.float64)
    action_index = {"bm25": 0, "contriever": 1, "bge": 2}
    for rank, action, repeat_id, status, payload in rows:
        if status != "success":
            raise ValueError("Training outcomes must all be successful")
        f1 = float(json.loads(payload)["metrics"]["normalized_token_f1"])
        if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
            raise ValueError("Training F1 must be finite and in [0, 1]")
        index = action_index[str(action)]
        if np.isfinite(values[int(rank), index, int(repeat_id)]):
            raise ValueError("Duplicate query/action/repeat outcome")
        values[int(rank), index, int(repeat_id)] = f1
    if not np.isfinite(values).all() or len(groups) != limit:
        raise ValueError("Outcome tensor or group alignment is incomplete")
    means = np.mean(values, axis=2)
    return means[:, 0], means[:, 1], means[:, 2], groups


def _bootstrap_ci(
    differences: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    by_group: defaultdict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)
    ordered = sorted(by_group)
    sums = np.asarray([np.sum(differences[by_group[group]]) for group in ordered])
    counts = np.asarray([len(by_group[group]) for group in ordered], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        sampled = rng.integers(0, len(ordered), size=(stop - start, len(ordered)))
        estimates[start:stop] = np.sum(sums[sampled], axis=1) / np.sum(
            counts[sampled], axis=1
        )
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _policy_metrics(
    bm25: np.ndarray,
    contriever: np.ndarray,
    bge: np.ndarray,
    switches: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    selected = np.where(switches, contriever, bm25)
    fixed = {
        "bm25": float(np.mean(bm25)),
        "contriever": float(np.mean(contriever)),
        "bge": float(np.mean(bge)),
    }
    best_pair = max(ROUTED_ACTIONS, key=lambda action: fixed[action])
    best_pair_value = fixed[best_pair]
    best_pair_per_query = bm25 if best_pair == "bm25" else contriever
    oracle = float(np.mean(np.maximum(bm25, contriever)))
    router = float(np.mean(selected))
    pair_differences = selected - best_pair_per_query
    bge_differences = selected - bge
    metrics = {
        "router_f1": router,
        "fixed": fixed,
        "best_pair_fixed": best_pair,
        "best_pair_fixed_f1": best_pair_value,
        "pair_oracle_f1": oracle,
        "pair_oracle_headroom": oracle - best_pair_value,
        "gain_over_best_pair_fixed": router - best_pair_value,
        "gain_over_fixed_bge": float(np.mean(bge_differences)),
        "switch_queries": int(np.sum(switches)),
        "switch_coverage": float(np.mean(switches)),
        "switch_precision": (
            float(np.mean((contriever - bm25)[switches] > TIE_ATOL))
            if np.any(switches)
            else None
        ),
        "conditional_gain_over_bm25": (
            float(np.mean((contriever - bm25)[switches])) if np.any(switches) else None
        ),
    }
    return metrics, pair_differences, bge_differences


def fit(protocol: Mapping[str, Any]) -> dict[str, Any]:
    feature_path = _run_dir(protocol) / "features.npz"
    if not feature_path.is_file():
        raise FileNotFoundError("Run the features stage before fit")
    with np.load(feature_path, allow_pickle=False) as stored:
        lexical = np.asarray(stored["lexical"], dtype=np.float32)
        dense = np.asarray(stored["dense"], dtype=np.float32)
        embedding = np.asarray(stored["embedding"], dtype=np.float32)
    bm25, contriever, bge, groups = _outcome_utilities(protocol)
    limit = int(protocol["sample"]["query_prefix_size"])
    if any(matrix.shape[0] != limit for matrix in (lexical, dense, embedding)):
        raise ValueError("Feature and outcome row counts differ")
    full = np.concatenate([lexical, dense, embedding], axis=1).astype(np.float32)
    matrices = {"full": full, "tier_a": lexical}
    gaps = contriever - bm25
    training = protocol["training"]
    outer_folds = _group_folds(
        groups,
        n_splits=int(training["outer_group_folds"]),
        seed=int(training["fold_seed"]),
    )
    model_seeds = [int(value) for value in training["model_seeds"]]
    candidates: dict[str, Any] = {}
    definitions = {
        "full_pairwise_xgboost": ("full", "pairwise"),
        "full_gain_xgboost": ("full", "gain"),
        "tier_a_pairwise_xgboost": ("tier_a", "pairwise"),
    }
    practical = float(protocol["gate"]["minimum_gain_over_best_pair_fixed"])
    global_practical = float(protocol["gate"]["minimum_gain_over_fixed_bge"])
    resamples = int(training["bootstrap_resamples"])
    bootstrap_seed = int(training["bootstrap_seed"])

    for candidate_position, name in enumerate(training["candidates"]):
        name = str(name)
        feature_tier, kind = definitions[name]
        print(f"candidate {candidate_position + 1}/3: {name}", flush=True)
        seed_switches: list[np.ndarray] = []
        seed_results: list[dict[str, Any]] = []
        for seed_position, seed in enumerate(model_seeds):
            print(f"  model seed {seed_position + 1}/5: {seed}", flush=True)
            switches, folds = _nested_oof_for_seed_configured(
                matrices[feature_tier],
                gaps,
                groups,
                kind=kind,
                model_seed=seed,
                outer_folds=outer_folds,
                inner_folds=int(training["inner_group_folds"]),
                fold_seed=int(training["fold_seed"]),
            )
            seed_switches.append(switches)
            metrics, _, _ = _policy_metrics(bm25, contriever, bge, switches)
            seed_results.append({"model_seed": seed, **metrics, "folds": folds})
        ensemble_switches = (
            np.mean(np.asarray(seed_switches, dtype=np.float64), axis=0) > 0.5
        )
        ensemble, pair_diff, bge_diff = _policy_metrics(
            bm25, contriever, bge, ensemble_switches
        )
        ensemble["gain_over_best_pair_fixed_ci95"] = _bootstrap_ci(
            pair_diff,
            groups,
            resamples=resamples,
            seed=bootstrap_seed + candidate_position * 10,
        )
        ensemble["gain_over_fixed_bge_ci95"] = _bootstrap_ci(
            bge_diff,
            groups,
            resamples=resamples,
            seed=bootstrap_seed + candidate_position * 10 + 1,
        )
        seed_gains = [float(row["gain_over_best_pair_fixed"]) for row in seed_results]
        seed_median = float(np.median(seed_gains))
        gate = {
            "pair_oracle_headroom_at_least_minimum": (
                ensemble["pair_oracle_headroom"]
                >= float(protocol["gate"]["minimum_pair_oracle_headroom"])
            ),
            "pair_gain_at_least_practical": (
                ensemble["gain_over_best_pair_fixed"] >= practical
            ),
            "pair_gain_ci_lower_above_zero": (
                ensemble["gain_over_best_pair_fixed_ci95"][0] > 0.0
            ),
            "global_gain_at_least_practical": (
                ensemble["gain_over_fixed_bge"] >= global_practical
            ),
            "global_gain_ci_lower_above_zero": (
                ensemble["gain_over_fixed_bge_ci95"][0] > 0.0
            ),
            "all_model_seeds_positive": all(gain > 0.0 for gain in seed_gains),
            "seed_median_at_least_practical": seed_median >= practical,
            "nondegenerate_switching": ensemble["switch_queries"] > 0,
        }
        gate["passed"] = all(gate.values())
        candidates[name] = {
            "feature_tier": feature_tier,
            "model_kind": kind,
            "ensemble": ensemble,
            "seed_gain_median": seed_median,
            "seed_gain_min": float(min(seed_gains)),
            "seed_gain_max": float(max(seed_gains)),
            "positive_model_seeds": int(sum(gain > 0.0 for gain in seed_gains)),
            "model_seeds": seed_results,
            "gate": gate,
        }

    passing = [name for name, value in candidates.items() if value["gate"]["passed"]]
    strongest = max(
        candidates,
        key=lambda name: float(candidates[name]["ensemble"]["gain_over_best_pair_fixed"]),
    )
    summary = {
        "phase": "3-contriever-revision",
        "status": "complete",
        "protocol": {
            "queries": limit,
            "routed_actions": list(ROUTED_ACTIONS),
            "default_action": "bm25",
            "external_reference": REFERENCE_ACTION,
            "metric": "normalized_token_f1",
            "outer_group_folds": int(training["outer_group_folds"]),
            "inner_group_folds": int(training["inner_group_folds"]),
            "answer_correctness_calls": 0,
            "fresh_dev_opened": False,
            "final_holdout_opened": False,
        },
        "candidates": candidates,
        "gate": {
            "passing_candidates": passing,
            "strongest_candidate": strongest,
            "decision": (
                "SUBSTANTIAL_F1_IMPROVEMENT" if passing else "ARCHITECTURE_DIAGNOSIS_REQUIRED"
            ),
        },
    }
    _write_json(_run_dir(protocol) / "summary.json", summary)
    return summary


def _nested_oof_for_seed_configured(
    matrix: np.ndarray,
    gaps: np.ndarray,
    groups: np.ndarray,
    *,
    kind: str,
    model_seed: int,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    inner_folds: int,
    fold_seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    switches = np.zeros(len(matrix), dtype=bool)
    filled = np.zeros(len(matrix), dtype=bool)
    fold_results: list[dict[str, Any]] = []
    for outer_id, (outer_train, outer_validation) in enumerate(outer_folds):
        inner = _group_folds(
            groups[outer_train],
            n_splits=inner_folds,
            seed=fold_seed + 100 + outer_id,
        )
        inner_scores = np.full(len(outer_train), np.nan, dtype=np.float64)
        for inner_train_local, inner_validation_local in inner:
            train = outer_train[inner_train_local]
            validation = outer_train[inner_validation_local]
            model = _new_model(kind, model_seed)
            _fit_model(model, kind, matrix[train], gaps[train])
            inner_scores[inner_validation_local] = _predict_scores(
                model, kind, matrix[validation]
            )
        if not np.isfinite(inner_scores).all():
            raise RuntimeError("Inner OOF scores were not completely filled")
        threshold = choose_dense_default_threshold(inner_scores, gaps[outer_train])
        model = _new_model(kind, model_seed)
        _fit_model(model, kind, matrix[outer_train], gaps[outer_train])
        outer_scores = _predict_scores(model, kind, matrix[outer_validation])
        fold_switches = (
            np.zeros(len(outer_scores), dtype=bool)
            if threshold["threshold"] is None
            else outer_scores > float(threshold["threshold"])
        )
        switches[outer_validation] = fold_switches
        filled[outer_validation] = True
        fold_results.append(
            {
                "outer_fold": outer_id,
                **threshold,
                "outer_queries": int(len(outer_validation)),
                "outer_gain_over_bm25": float(
                    np.mean(np.where(fold_switches, gaps[outer_validation], 0.0))
                ),
                "outer_switch_coverage": float(np.mean(fold_switches)),
            }
        )
    if not filled.all():
        raise RuntimeError("Outer OOF policy did not cover every query")
    return switches, fold_results


def main() -> int:
    args = _arguments()
    protocol, router = _load_protocol(_resolve(args.config))
    if args.stage == "audit":
        result: Any = audit(protocol, router)
    elif args.stage == "prepare":
        result = prepare(protocol, router)
    elif args.stage == "generate":
        result = generate(protocol, router, max_new_calls=args.max_new_calls)
    elif args.stage == "features":
        result = build_features(protocol, router)
    elif args.stage == "fit":
        result = fit(protocol)
    else:
        result = status(protocol)
    printable = result["gate"] if args.stage == "fit" else result
    print(json.dumps(printable, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
