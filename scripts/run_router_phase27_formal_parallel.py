#!/usr/bin/env python3
"""Run frozen Phase 2.7 candidate/split tasks in isolated worker processes."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_router_phase27_model_audit as audit


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="analysis/hotpotqa_router/phase27_9600_config.yaml"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    return parser.parse_args()


def task_stem(candidate_id: str, split_seed: int) -> str:
    return f"{candidate_id}__{split_seed}"


def worker(
    config_path: str,
    task_dir: str,
    candidate_id: str,
    split_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    config = audit._load_config(config_path)
    data, _ = audit.load_frozen_data(config)
    spec = audit._candidate_by_id(candidate_id)
    model_seeds = [int(value) for value in config["cross_validation"]["model_seeds"]]
    if spec.model_kind != "xgboost":
        model_seeds = [model_seeds[0]]
    result, scores, fold_ids = audit.run_candidate_split(
        config,
        data,
        spec,
        split_seed=int(split_seed),
        model_seeds=model_seeds,
        bootstrap_resamples=int(bootstrap_resamples),
    )
    directory = Path(task_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = task_stem(candidate_id, split_seed)
    audit._write_json(directory / f"{stem}.json", result)
    np.savez_compressed(directory / f"{stem}.npz", scores=scores, fold_ids=fold_ids)
    return {
        "candidate_id": candidate_id,
        "split_seed": int(split_seed),
        "gain": result["metrics"]["gain_over_fixed_dense"],
    }


def import_central_progress(
    output_dir: Path,
    task_dir: Path,
    candidate_ids: list[str],
    split_seeds: list[int],
) -> None:
    metrics_path = output_dir / "formal_metrics.json"
    predictions_path = output_dir / "formal_predictions.npz"
    if not metrics_path.exists() or not predictions_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(predictions_path, allow_pickle=False) as stored:
        for candidate_id in candidate_ids:
            for split_seed in split_seeds:
                result = metrics.get("candidate_splits", {}).get(candidate_id, {}).get(
                    str(split_seed)
                )
                prediction_key = f"prediction__{candidate_id}__{split_seed}"
                fold_key = f"fold_id__{candidate_id}__{split_seed}"
                if result is None or prediction_key not in stored or fold_key not in stored:
                    continue
                stem = task_stem(candidate_id, split_seed)
                audit._write_json(task_dir / f"{stem}.json", result)
                np.savez_compressed(
                    task_dir / f"{stem}.npz",
                    scores=np.asarray(stored[prediction_key]),
                    fold_ids=np.asarray(stored[fold_key]),
                )


def main() -> int:
    args = arguments()
    config_path = str(audit._resolve(args.config))
    config = audit._load_config(config_path)
    output_dir = audit._resolve(
        args.output_dir or config["planned_implementation"]["output_dir"]
    )
    freeze = json.loads((output_dir / "candidate_freeze.json").read_text(encoding="utf-8"))
    if freeze.get("protocol_id") != config["protocol"]["id"]:
        raise ValueError("Candidate freeze belongs to another protocol")
    candidate_ids = list(freeze["formal_candidates"])
    split_seeds = [int(value) for value in config["cross_validation"]["split_seeds"]]
    resamples = int(
        args.bootstrap_resamples
        if args.bootstrap_resamples is not None
        else config["evaluation"]["bootstrap"]["resamples"]
    )
    task_dir = output_dir / "formal_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    import_central_progress(output_dir, task_dir, candidate_ids, split_seeds)

    pending: list[tuple[str, int]] = []
    for candidate_id in candidate_ids:
        for split_seed in split_seeds:
            stem = task_stem(candidate_id, split_seed)
            if (task_dir / f"{stem}.json").exists() and (task_dir / f"{stem}.npz").exists():
                print(f"formal {candidate_id} split={split_seed} [isolated resume]", flush=True)
            else:
                pending.append((candidate_id, split_seed))

    if pending:
        with ProcessPoolExecutor(max_workers=int(args.max_workers)) as executor:
            futures = {
                executor.submit(
                    worker,
                    config_path,
                    str(task_dir),
                    candidate_id,
                    split_seed,
                    resamples,
                ): (candidate_id, split_seed)
                for candidate_id, split_seed in pending
            }
            for future in as_completed(futures):
                candidate_id, split_seed = futures[future]
                completed = future.result()
                print(
                    f"formal {candidate_id} split={split_seed} [complete] "
                    f"gain={completed['gain']:+.6f}",
                    flush=True,
                )

    completed: dict[str, dict[str, Any]] = {candidate_id: {} for candidate_id in candidate_ids}
    predictions: dict[str, np.ndarray] = {}
    for candidate_id in candidate_ids:
        for split_seed in split_seeds:
            stem = task_stem(candidate_id, split_seed)
            completed[candidate_id][str(split_seed)] = json.loads(
                (task_dir / f"{stem}.json").read_text(encoding="utf-8")
            )
            with np.load(task_dir / f"{stem}.npz", allow_pickle=False) as stored:
                predictions[f"prediction__{candidate_id}__{split_seed}"] = np.asarray(
                    stored["scores"]
                )
                predictions[f"fold_id__{candidate_id}__{split_seed}"] = np.asarray(
                    stored["fold_ids"]
                )

    aggregate: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        metrics = [
            completed[candidate_id][str(split_seed)]["metrics"]
            for split_seed in split_seeds
        ]
        aggregate[candidate_id] = {
            "mean_gain_over_split_seeds": float(
                np.mean([row["gain_over_fixed_dense"] for row in metrics])
            ),
            "all_split_seed_gains_positive": all(
                row["gain_over_fixed_dense"] > 0.0 for row in metrics
            ),
            "split_seed_metrics": metrics,
        }
    audit._write_json(
        output_dir / "formal_metrics.json",
        {
            "protocol_id": config["protocol"]["id"],
            "status": "complete",
            "formal_candidates": candidate_ids,
            "candidate_splits": completed,
            "aggregate": aggregate,
            "orchestration": "isolated_process_pool_then_deterministic_merge",
            "max_workers": int(args.max_workers),
            "external_calls": 0,
        },
    )
    np.savez_compressed(output_dir / "formal_predictions.npz", **predictions)
    audit._write_execution_state(config, output_dir)
    print(json.dumps({"formal": "complete", "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
