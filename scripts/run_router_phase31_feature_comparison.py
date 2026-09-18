"""Compare predefined structured feature blocks under the frozen M3 recipe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy
import sklearn
import yaml


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def common_bootstrap(groups: np.ndarray, contrasts: dict[str, np.ndarray], seed: int, resamples: int) -> dict:
    """Resample whole groups, using the same draws for every paired contrast."""
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.zeros((len(counts), len(contrasts)), dtype=np.float64)
    np.add.at(sums, inverse, np.column_stack(list(contrasts.values())))
    rng = np.random.default_rng(seed)
    estimates = np.empty((resamples, len(contrasts)), dtype=np.float64)
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        sampled = rng.integers(0, len(counts), size=(stop - start, len(counts)))
        estimates[start:stop] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1, keepdims=True)
    intervals = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {name: {"mean": float(np.mean(values)), "ci95": intervals[:, index].tolist()}
            for index, (name, values) in enumerate(contrasts.items())}


def historical_gate(base: dict, metrics: dict, split_metrics: list[dict]) -> dict:
    gate = base["formal_gate"]
    ratio = metrics["harmful_to_beneficial_mass_ratio"]
    checks = {
        "practical_gain": float(np.mean([m["gain_over_fixed_dense"] for m in split_metrics])) >= gate["practical_gain_minimum"],
        "grouped_bootstrap_ci_lower": metrics["gain_ci95"][0] > gate["grouped_bootstrap_ci_lower_must_exceed"],
        "all_split_seed_gains_positive": all(m["gain_over_fixed_dense"] > 0 for m in split_metrics),
        "harmful_to_beneficial_mass_ratio": ratio is not None and ratio <= gate["harmful_to_beneficial_mass_ratio_maximum"],
        "switch_coverage": gate["switch_coverage_minimum"] <= metrics["switch_coverage"] <= gate["switch_coverage_maximum"],
        "calibration_slope": gate["calibration_slope_minimum"] <= metrics["calibration"]["slope"] <= gate["calibration_slope_maximum"],
        "top_decile_realized_gap": metrics["calibration"]["top_decile_realized_gap"] > 0,
    }
    return {"checks": checks, "passed": all(checks.values()), "role": "historical_context_only"}


def decision_changes(data, old: np.ndarray, new: np.ndarray) -> dict:
    old_switch, new_switch = old > 0, new > 0
    delta = (new_switch.astype(float) - old_switch.astype(float)) * data.gap
    rows = []
    for old_value, new_value in [(False, False), (False, True), (True, False), (True, True)]:
        selected = (old_switch == old_value) & (new_switch == new_value)
        rows.append({"old_action": "bm25" if old_value else "dense", "new_action": "bm25" if new_value else "dense",
                     "queries": int(selected.sum()), "new_minus_old_mean_f1_contribution": float(np.where(selected, delta, 0).mean())})
    return {"transition_counts": rows, "changed_queries": int(np.sum(old_switch != new_switch)),
            "improved_queries": int(np.sum(delta > 1e-12)), "worsened_queries": int(np.sum(delta < -1e-12)),
            "neutral_changed_queries": int(np.sum((old_switch != new_switch) & (np.abs(delta) <= 1e-12))),
            "positive_gain_mass": float(np.maximum(delta, 0).mean()), "negative_loss_mass": float(np.maximum(-delta, 0).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/hotpotqa_router/phases/phase31/config.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    started = time.time()
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))
    from src.router_experiments.modeling import candidate_by_id, load_config, load_frozen_data, policy_metrics, run_candidate_split
    from router_phase31_features import extract_new_features

    config_path = resolve(repo, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path = resolve(repo, config["base_config"])
    base = load_config(base_path)
    run_dir = resolve(repo, str(args.output_dir or config["output_dir"]))
    results_dir = resolve(repo, str(args.results_dir or config["results_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
        "config": config_path, "base_config": base_path, "runner": Path(__file__),
        "feature_helper": Path(__file__).with_name("router_phase31_features.py")}.items()}
    freeze = {"protocol_id": config["protocol"]["id"], "sha256": hashes, "config": config,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}}
    freeze_path = run_dir / "effective_config.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text(encoding="utf-8")) != freeze:
        raise ValueError("Existing run has a different frozen config or implementation; use a new run directory")
    write_json(freeze_path, freeze)
    comparison = config["comparison"]
    seeds = base["cross_validation"]["split_seeds"]
    if seeds != comparison["split_seeds"] or [base["cross_validation"][k] for k in ("outer_folds", "inner_folds")] != [5, 4]:
        raise ValueError("The comparison must preserve the original 3-seed 5/4-fold protocol")
    spec = candidate_by_id(comparison["model"])
    if (spec.model_kind, spec.feature_kind, spec.pca_dim, comparison["ridge_alpha"], comparison["model_seed"]) != ("ridge", "pca_structured", 32, 10.0, 11):
        raise ValueError("Only the fixed M3 PCA32 Ridge alpha10 recipe is supported")
    bootstrap = config["bootstrap"]
    data, validation = load_frozen_data(base)
    write_json(run_dir / "input_validation.json", validation)
    if data.structured.shape != (9600, 30) or data.embedding.shape != (9600, 384):
        raise ValueError("Expected the original frozen 9600-query snapshot")

    def fit(label: str, current, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
        stem = run_dir / "fits" / f"{label}__{seed}"
        if stem.with_suffix(".json").exists() and stem.with_suffix(".npz").exists():
            print(f"{label} split={seed} [resume]", flush=True)
            result = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as stored:
                return result, stored["scores"].copy(), stored["fold_ids"].copy()
        print(f"{label} split={seed}", flush=True)
        result, scores, folds = run_candidate_split(base, current, spec, split_seed=seed, model_seeds=[11], bootstrap_resamples=bootstrap["resamples"])
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stem.with_suffix(".npz"), scores=scores, fold_ids=folds)
        write_json(stem.with_suffix(".json"), result)
        return result, scores, folds

    scores_by_model, split_results, arrays, reproduction = {}, {}, {}, []
    with np.load(resolve(repo, config["historical_predictions"]), allow_pickle=False) as historical:
        old_results = [fit("old30", data, seed) for seed in seeds]
        for seed, (_, scores, folds) in zip(seeds, old_results):
            expected = historical[f"prediction__{spec.candidate_id}__{seed}"]
            expected_folds = historical[f"fold_id__{spec.candidate_id}__{seed}"]
            check = {"split_seed": seed, "scores_exact": bool(np.array_equal(scores, expected)),
                     "max_abs_score_difference": float(np.max(np.abs(scores - expected))),
                     "scores_within_1e_12": bool(np.allclose(scores, expected, atol=1e-12, rtol=0)),
                     "folds_exact": bool(np.array_equal(folds, expected_folds)),
                     "decisions_exact": bool(np.array_equal(scores > 0, expected > 0))}
            reproduction.append(check)
        write_json(run_dir / "old_reproduction.json", {"checks": reproduction})
        if not all(c["scores_within_1e_12"] and c["folds_exact"] and c["decisions_exact"] for c in reproduction):
            raise RuntimeError("Old M3 reproduction failed; new feature training has not started")
    print("Old M3 reproduction passed for every split.", flush=True)
    matrix, names, feature_metadata = extract_new_features(data, config, repo, run_dir)
    if list(names) != config["features"] or matrix.shape != (9600, 35) or not np.isfinite(matrix).all():
        raise ValueError("The new matrix must be finite, aligned, and contain exactly the frozen 35 columns")
    write_json(run_dir / "feature_extraction.json", feature_metadata)
    new_data = replace(data, lexical=matrix, corpus=np.empty((len(matrix), 0), dtype=np.float64))
    new_results = [fit("new35", new_data, seed) for seed in seeds]
    for label, results in [("old30", old_results), ("new35", new_results)]:
        split_results[label] = {str(seed): result[0]["metrics"] for seed, result in zip(seeds, results)}
        scores_by_model[label] = np.mean(np.stack([result[1] for result in results]), axis=0)
        for seed, (_, scores, folds) in zip(seeds, results):
            arrays[f"prediction__{label}__{seed}"] = scores
            arrays[f"fold_id__{label}__{seed}"] = folds
        arrays[f"consensus__{label}"] = scores_by_model[label]
    if not all(np.array_equal(a[2], b[2]) for a, b in zip(old_results, new_results)):
        raise RuntimeError("Feature replacement changed fold assignments")
    np.savez_compressed(run_dir / "predictions.npz", query_ids=data.query_ids, group_ids=data.group_ids, **arrays)
    gains = {label: np.where(scores > 0, data.gap, 0.0) for label, scores in scores_by_model.items()}
    contrasts = common_bootstrap(data.group_ids, {"old_minus_dense": gains["old30"], "new_minus_dense": gains["new35"],
                                                "new_minus_old": gains["new35"] - gains["old30"]},
                                 bootstrap["seed"], bootstrap["resamples"])
    metrics, gates = {}, {}
    for label in ("old30", "new35"):
        metrics[label] = policy_metrics(data, scores_by_model[label], bootstrap_seed=bootstrap["seed"],
                                       bootstrap_resamples=bootstrap["resamples"], with_ci=False)
        metrics[label]["gain_ci95"] = contrasts["old_minus_dense" if label == "old30" else "new_minus_dense"]["ci95"]
        gates[label] = historical_gate(base, metrics[label], list(split_results[label].values()))
    lower, upper = contrasts["new_minus_old"]["ci95"]
    decision = "INTERNAL_IMPROVEMENT" if lower > 0 else "INTERNAL_DEGRADATION" if upper < 0 else "NO_CLEAR_INTERNAL_DIFFERENCE"
    summary = {"protocol_id": config["protocol"]["id"], "status": "complete", "data_role": config["protocol"]["data_role"],
               "deployable": False, "decision": decision, "queries": len(data.query_ids), "groups": len(np.unique(data.group_ids)),
               "dimensions": {"old30": 62, "new35": 67}, "baseline_reproduction": reproduction,
               "bootstrap": bootstrap, "consensus": metrics, "paired_contrasts": contrasts, "split_metrics": split_results,
               "split_new_minus_old": {str(seed): new[0]["metrics"]["gain_over_fixed_dense"] - old[0]["metrics"]["gain_over_fixed_dense"]
                                        for seed, old, new in zip(seeds, old_results, new_results)},
               "decision_changes": decision_changes(data, scores_by_model["old30"], scores_by_model["new35"]),
               "historical_gate_context": gates, "external_calls": 0, "natural_2000_outcome_rows": 0,
               "fresh_natural_rows": 0, "final_holdout_rows": 0, "elapsed_seconds": time.time() - started}
    write_json(results_dir / "summary.json", summary)
    fields = ["model", "split", "router_mean_f1", "gain_over_fixed_dense", "gain_ci95_low", "gain_ci95_high", "switch_coverage",
              "beneficial_switches", "neutral_switches", "harmful_switches", "harmful_to_beneficial_mass_ratio", "non_tie_auc"]
    with (results_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in ("old30", "new35"):
            for split, row in [("consensus", metrics[label]), *split_results[label].items()]:
                writer.writerow({"model": label, "split": split, **{k: row[k] for k in fields[2:] if k in row},
                                 "gain_ci95_low": row["gain_ci95"][0], "gain_ci95_high": row["gain_ci95"][1]})
    print(json.dumps({"decision": decision, "paired_contrasts": contrasts, "results_dir": str(results_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
