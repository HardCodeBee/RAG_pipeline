"""Run a small paired HotpotQA prompt ablation on cached BM25/BGE contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluators.hotpot_answer import answer_metrics
from src.generators.answer_generator import LLMGenerator
from src.persistence.run_output_writer import (
    write_metadata_json,
    write_result_checkpoint,
    write_results,
)
from src.prompts.fixed_prompt import (
    HOTPOT_MULTIHOP_SHORT_ANSWER_VERSION,
    HOTPOT_SHORT_ANSWER_VERSION,
    build_prompt,
)
from src.records import ContextPackage


PROMPT_VERSIONS = (
    HOTPOT_SHORT_ANSWER_VERSION,
    HOTPOT_MULTIHOP_SHORT_ANSWER_VERSION,
)
RETRIEVERS = ("bm25", "dense")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="outputs/router/hotpotqa_bd_router_v1/config.yaml",
    )
    parser.add_argument(
        "--source-run",
        default=(
            "outputs/router/hotpotqa_bd_router_v1/runs/"
            "phase2_bd_headroom_repeated_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/router/hotpotqa_prompt_v1_v2_small_v1",
    )
    parser.add_argument("--partition", choices=("train", "dev"), default="dev")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--max-new-calls", type=int, default=None)
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry checkpoints whose previous provider call failed.",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "summarize", "all"),
        default="all",
    )
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _load_router_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_cells(source_run: Path, partition: str) -> dict[str, dict[str, dict[str, Any]]]:
    checkpoint_paths = sorted((source_run / "checkpoints").glob("*.json"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No cached source checkpoints: {source_run}")
    cells: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in checkpoint_paths:
        row = _read_json(path)
        if (
            row.get("split") != partition
            or row.get("action") not in RETRIEVERS
            or int(row.get("repeat_id", 0)) != 0
        ):
            continue
        query_id = str(row["query_id"])
        action = str(row["action"])
        if action in cells[query_id]:
            raise ValueError(f"Duplicate source cell: {(query_id, action)}")
        context = row.get("context")
        if not isinstance(context, Mapping) or not isinstance(context.get("text"), str):
            raise ValueError(f"Source cell has no cached context: {path}")
        cells[query_id][action] = row
    complete = {
        query_id: actions
        for query_id, actions in cells.items()
        if set(actions) == set(RETRIEVERS)
    }
    if not complete:
        raise ValueError(f"No complete BM25/BGE pairs in partition {partition!r}")
    return complete


def _sample_path(output_dir: Path) -> Path:
    return output_dir / "sample.json"


def _checkpoint_path(output_dir: Path, position: int) -> Path:
    return output_dir / "checkpoints" / f"{position:04d}.json"


def prepare(
    router: Mapping[str, Any],
    source_run: Path,
    output_dir: Path,
    *,
    partition: str,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    cells = _source_cells(source_run, partition)
    query_ids = sorted(cells)
    if sample_size > len(query_ids):
        raise ValueError(
            f"sample_size={sample_size} exceeds {len(query_ids)} complete source pairs"
        )

    sample_file = _sample_path(output_dir)
    if sample_file.is_file():
        sample = _read_json(sample_file)
        expected = {
            "partition": partition,
            "sample_size": sample_size,
            "seed": seed,
            "selection": "uniform_query_without_replacement_before_generation",
        }
        for key, value in expected.items():
            if sample.get(key) != value:
                raise ValueError(f"Existing sample has incompatible {key}: {sample.get(key)!r}")
        selected_ids = [str(value) for value in sample.get("query_ids", [])]
        if len(selected_ids) != sample_size or any(value not in cells for value in selected_ids):
            raise ValueError("Existing sample query ids do not match the source run")
    else:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(query_ids), size=sample_size, replace=False)
        selected_ids = [query_ids[int(position)] for position in chosen]
        write_metadata_json(
            sample_file,
            {
                "partition": partition,
                "sample_size": sample_size,
                "seed": seed,
                "selection": "uniform_query_without_replacement_before_generation",
                "query_ids": selected_ids,
            },
            overwrite=False,
        )

    existing = sorted((output_dir / "checkpoints").glob("*.json"))
    expected_rows = sample_size * len(RETRIEVERS) * len(PROMPT_VERSIONS)
    if existing:
        if len(existing) != expected_rows:
            raise RuntimeError("Partial ablation checkpoints exist; refusing an ambiguous prepare")
        return [_read_json(path) for path in existing]

    records: list[dict[str, Any]] = []
    for query_id in selected_ids:
        for action in RETRIEVERS:
            source = cells[query_id][action]
            context = source["context"]
            for prompt_version in PROMPT_VERSIONS:
                context_package = ContextPackage(
                    text=str(context["text"]),
                    results=(),
                    token_count=int(context["token_count"]),
                    truncated=bool(context["truncated"]),
                )
                prompt = build_prompt(
                    str(source["question"]), context_package, prompt_version
                )
                records.append(
                    {
                        "status": "pending_generation",
                        "dataset_id": "hotpotqa",
                        "partition": partition,
                        "query_id": query_id,
                        "group_id": str(source["group_id"]),
                        "retriever": "bge" if action == "dense" else action,
                        "source_action": action,
                        "question": str(source["question"]),
                        "reference_answers": list(source["reference_answers"]),
                        "context_sha256": _sha256_text(str(context["text"])),
                        "context_token_count": int(context["token_count"]),
                        "retrieval": dict(source["retrieval"]),
                        "prompt_version": prompt_version,
                        "prompt_sha256": prompt.sha256,
                    }
                )

    order_rng = np.random.default_rng(seed + 1)
    order_rng.shuffle(records)
    for position, row in enumerate(records):
        row["call_position"] = position
        write_result_checkpoint(_checkpoint_path(output_dir, position), row)

    generation = router["generation"]
    write_metadata_json(
        output_dir / "metadata.json",
        {
            "protocol": "hotpotqa_paired_prompt_ablation_small_v1",
            "exploratory": True,
            "source_run": str(source_run),
            "source_repeat_id": 0,
            "partition": partition,
            "sample_size": sample_size,
            "seed": seed,
            "retrievers": ["bm25", "bge"],
            "prompt_versions": list(PROMPT_VERSIONS),
            "generation": {
                "provider": generation["provider"],
                "model": generation["model"],
                "temperature": float(generation["temperature"]),
                "max_output_tokens": int(generation["max_output_tokens"]),
                "repeats_per_cell": 1,
            },
            "primary_metrics": ["normalized_exact_match", "normalized_token_f1"],
            "selection_uses_predictions_or_metrics": False,
            "cached_context_reused_across_prompts": True,
            "formal_significance_claim": False,
        },
        overwrite=False,
    )
    write_results(output_dir / "results.jsonl", records)
    return records


def _load_rows(output_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((output_dir / "checkpoints").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No ablation checkpoints: {output_dir}")
    return [_read_json(path) for path in paths]


def generate(
    router: Mapping[str, Any],
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    workers: int,
    max_new_calls: int | None,
    retry_failures: bool,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    generation = router["generation"]
    pending = []
    for position, row in enumerate(rows):
        generation_trace = row.get("generation")
        if not isinstance(generation_trace, Mapping) or not bool(
            generation_trace.get("attempted")
        ):
            pending.append(position)
        elif retry_failures and generation_trace.get("status") == "failure":
            pending.append(position)
    if max_new_calls is not None:
        if max_new_calls < 0:
            raise ValueError("max_new_calls must be non-negative")
        pending = pending[:max_new_calls]
    if not pending:
        return rows

    local = threading.local()
    created: list[LLMGenerator] = []
    created_lock = threading.Lock()

    def worker(position: int) -> tuple[int, dict[str, Any]]:
        generator = getattr(local, "generator", None)
        if generator is None:
            generator = LLMGenerator(
                provider=str(generation["provider"]),
                model=str(generation["model"]),
                temperature=float(generation["temperature"]),
                max_output_tokens=int(generation["max_output_tokens"]),
                timeout_seconds=60.0,
                max_retries=2,
            )
            local.generator = generator
            with created_lock:
                created.append(generator)
        row = rows[position]
        source_cells = _source_cells_cache[row["query_id"]]
        source = source_cells[str(row["source_action"])]
        context = source["context"]
        context_package = ContextPackage(
            text=str(context["text"]),
            results=(),
            token_count=int(context["token_count"]),
            truncated=bool(context["truncated"]),
        )
        prompt = build_prompt(
            str(row["question"]), context_package, str(row["prompt_version"])
        )
        if prompt.sha256 != row["prompt_sha256"]:
            raise RuntimeError(f"Prompt hash drift at checkpoint {position}")
        try:
            result = generator.generate_from_prompt(prompt.text, str(row["question"]), [])
            prediction = result.answer.strip()
            return position, {
                "status": "success",
                "prediction": prediction,
                "metrics": answer_metrics(prediction, row["reference_answers"]),
                "generation": {
                    "attempted": True,
                    "status": "success",
                    "requested_model": result.requested_model,
                    "model": result.model,
                    "response_id": result.response_id,
                    "latency_ms": result.latency_ms,
                    "token_usage": result.token_usage,
                },
            }
        except Exception as exc:
            return position, {
                "status": "generation_failure",
                "generation": {
                    "attempted": True,
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }

    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, position) for position in pending]
            for future in as_completed(futures):
                position, update = future.result()
                rows[position].update(update)
                write_result_checkpoint(_checkpoint_path(output_dir, position), rows[position])
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        json.dumps(
                            {"generation_new_calls": completed, "requested": len(pending)}
                        ),
                        flush=True,
                    )
    finally:
        for generator in created:
            close = getattr(getattr(generator, "_client", None), "close", None)
            if callable(close):
                close()
    write_results(output_dir / "results.jsonl", rows)
    return rows


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _bootstrap_ci(values: Sequence[float], resamples: int, seed: int) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or resamples <= 0:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return [float(lower), float(upper)]


def summarize(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    successes = [row for row in rows if row.get("status") == "success"]
    expected = len(rows)
    cell_summary: dict[str, Any] = {}
    for retriever in ("bm25", "bge"):
        for prompt_version in PROMPT_VERSIONS:
            cell = [
                row
                for row in successes
                if row["retriever"] == retriever
                and row["prompt_version"] == prompt_version
            ]
            key = f"{retriever}__{prompt_version}"
            cell_summary[key] = {
                "n": len(cell),
                "mean_em": _mean(
                    [float(row["metrics"]["normalized_exact_match"]) for row in cell]
                ),
                "mean_f1": _mean(
                    [float(row["metrics"]["normalized_token_f1"]) for row in cell]
                ),
                "mean_latency_ms": _mean(
                    [float(row["generation"]["latency_ms"]) for row in cell]
                ),
                "mean_input_tokens": _mean(
                    [
                        float(row["generation"]["token_usage"]["provider_reported"]["input_tokens"])
                        for row in cell
                    ]
                ),
                "mean_output_tokens": _mean(
                    [
                        float(row["generation"]["token_usage"]["provider_reported"]["output_tokens"])
                        for row in cell
                    ]
                ),
            }

    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in successes:
        grouped[(str(row["query_id"]), str(row["retriever"]))][
            str(row["prompt_version"])
        ] = row

    comparisons: list[dict[str, Any]] = []
    for (query_id, retriever), versions in sorted(grouped.items()):
        if set(versions) != set(PROMPT_VERSIONS):
            continue
        old = versions[HOTPOT_SHORT_ANSWER_VERSION]
        new = versions[HOTPOT_MULTIHOP_SHORT_ANSWER_VERSION]
        if old["context_sha256"] != new["context_sha256"]:
            raise RuntimeError(f"Context drift for {(query_id, retriever)}")
        old_f1 = float(old["metrics"]["normalized_token_f1"])
        new_f1 = float(new["metrics"]["normalized_token_f1"])
        old_em = float(old["metrics"]["normalized_exact_match"])
        new_em = float(new["metrics"]["normalized_exact_match"])
        comparisons.append(
            {
                "query_id": query_id,
                "retriever": retriever,
                "question": old["question"],
                "reference_answers": old["reference_answers"],
                "context_sha256": old["context_sha256"],
                "old_prediction": old["prediction"],
                "new_prediction": new["prediction"],
                "old_f1": old_f1,
                "new_f1": new_f1,
                "delta_f1": new_f1 - old_f1,
                "old_em": old_em,
                "new_em": new_em,
                "delta_em": new_em - old_em,
            }
        )

    paired: dict[str, Any] = {}
    query_f1: dict[str, list[float]] = defaultdict(list)
    query_em: dict[str, list[float]] = defaultdict(list)
    for retriever in ("bm25", "bge"):
        selected = [row for row in comparisons if row["retriever"] == retriever]
        f1_deltas = [float(row["delta_f1"]) for row in selected]
        em_deltas = [float(row["delta_em"]) for row in selected]
        paired[retriever] = {
            "paired_n": len(selected),
            "mean_delta_f1": _mean(f1_deltas),
            "delta_f1_bootstrap_95_ci": _bootstrap_ci(
                f1_deltas, bootstrap_resamples, seed + (1 if retriever == "bm25" else 2)
            ),
            "mean_delta_em": _mean(em_deltas),
            "delta_em_bootstrap_95_ci": _bootstrap_ci(
                em_deltas, bootstrap_resamples, seed + (3 if retriever == "bm25" else 4)
            ),
            "f1_wins_ties_losses": {
                "new_better": sum(value > 0.0 for value in f1_deltas),
                "tie": sum(value == 0.0 for value in f1_deltas),
                "new_worse": sum(value < 0.0 for value in f1_deltas),
            },
        }
        for row in selected:
            query_f1[str(row["query_id"])].append(float(row["delta_f1"]))
            query_em[str(row["query_id"])].append(float(row["delta_em"]))

    overall_f1 = [_mean(values) for values in query_f1.values() if len(values) == 2]
    overall_em = [_mean(values) for values in query_em.values() if len(values) == 2]
    paired["query_macro_across_retrievers"] = {
        "paired_query_n": len(overall_f1),
        "mean_delta_f1": _mean(overall_f1),
        "delta_f1_bootstrap_95_ci": _bootstrap_ci(
            overall_f1, bootstrap_resamples, seed + 5
        ),
        "mean_delta_em": _mean(overall_em),
        "delta_em_bootstrap_95_ci": _bootstrap_ci(
            overall_em, bootstrap_resamples, seed + 6
        ),
    }

    complete = len(successes) == expected and len(comparisons) * 2 == expected
    summary = {
        "status": "complete" if complete else "incomplete",
        "exploratory": True,
        "expected_generation_calls": expected,
        "successful_generation_calls": len(successes),
        "failed_generation_calls": expected - len(successes),
        "cell_metrics": cell_summary,
        "paired_new_minus_old": paired,
        "interpretation_boundary": (
            "Direction-only small-sample prompt screen; one generation per cell and no "
            "Answer Correctness judge. Do not treat the bootstrap interval as a formal gate."
        ),
    }
    write_results(output_dir / "comparisons.jsonl", comparisons)
    write_metadata_json(output_dir / "summary.json", summary)
    return summary


_source_cells_cache: dict[str, dict[str, dict[str, Any]]] = {}


def main() -> None:
    args = _arguments()
    config_path = _resolve(args.config)
    source_run = _resolve(args.source_run)
    output_dir = _resolve(args.output_dir)
    router = _load_router_config(config_path)

    global _source_cells_cache
    _source_cells_cache = _source_cells(source_run, args.partition)

    if args.stage in {"prepare", "all"}:
        rows = prepare(
            router,
            source_run,
            output_dir,
            partition=args.partition,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        if args.stage == "prepare":
            print(json.dumps({"prepared_rows": len(rows)}))
            return
    else:
        rows = _load_rows(output_dir)

    if args.stage in {"generate", "all"}:
        rows = generate(
            router,
            output_dir,
            rows,
            workers=args.workers,
            max_new_calls=args.max_new_calls,
            retry_failures=args.retry_failures,
        )
        if args.stage == "generate":
            print(json.dumps({"rows": len(rows)}))
            return

    summary = summarize(
        output_dir,
        rows,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
