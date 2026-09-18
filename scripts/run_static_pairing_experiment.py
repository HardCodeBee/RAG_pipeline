"""One frozen three-arm, consumed-development experiment of corpus alignment.

No candidate/threshold search, new retrieval, answers, or reserved-set access.
The full-vocabulary feature bank and shared query encoding precede this fit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "work/router_research"
RUNS = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs"
OUT = RUNS / "static_pairing_effect_v1"
FEATURES = RUNS / "static_pairing_features_v1"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT))
import weighted_linear_probe_refined as probe

COMMON = [
    "log_unique_term_count", "corpus_known_fraction", "centroid_available",
    "idf_mean", "idf_std", "idf_max", "log_total_lexical_mass_per_document",
    "centered_centroid_squared_norm",
]
SPEC = {
    "scope": "one_fixed_mechanism_experiment_on_consumed_development_not_independent_confirmation",
    "arms": ["matched_base", "true_alignment", "coordinate_control"],
    "fits": 3, "training": {"old": 6144, "beir": 4165},
    "development": {"old": 1536, "beir": 1044},
    "roles": "reuse_m6_complete_cohort_extension_v1_source_labels_and_roles_no_new_split",
    "common_input": "cached_M6_384_plus_actual_Dense_max512_normalized_384_plus_eight_scalars",
    "common_scalars": COMMON,
    "scalar_preprocessing": "unweighted_fit_mean_and_population_std; zero_scale_below_1e-12_replaced_by_1",
    "embedding_preprocessing": "none; identical_saved_query_embeddings_for_all_arms",
    "added_feature": "IDF_mixture_of_termwise_BM25_centroids_centered_by_full_corpus_mean_dot_actual_Dense_query",
    "control": "same_centroid_fixed_signed_coordinate_derangement_seed2026091882",
    "residualization": "joint_two_RHS_unweighted_fit_lstsq_against_common_input_and_intercept_gelsd_rcond1e-12_no_outcomes",
    "residual_scale": "fit_RMS; below1e-12_zero_column; no_variance_or_effect_selection",
    "objective": "sum_abs_mean3_gap_weighted_BCE_over_sum_weights_plus_lambda_half_coef_norm_squared",
    "regularization": .001, "intercept_penalized": False, "tie_atol": 1e-12,
    "solver": "existing_weighted_linear_probe_refined.fit",
    "numerical_budget": probe.NUMERICAL_BUDGET, "refinement": probe.REFINEMENT,
    "accept_gradient_inf": probe.ACCEPT_GRAD_INF,
    "prediction": "FP64_native_linear_logit_threshold_zero_no_calibration",
    "threads": 2,
    "point_gate": {"true_over_each_fixed_each_source": .01,
                   "true_over_matched_base_each_source": .002,
                   "true_over_coordinate_control_each_source": .002},
    "intervals_if_point_gate_passes": {"group_bootstrap_draws": 10000, "seed": 2026091883,
                                      "simultaneous_comparisons": 8, "two_sided_confidence_each": .99375,
                                      "require_lower_positive_all": True},
    "failure_action": "close_recipe_no_new_feature_pooling_dimension_lambda_threshold_seed_or_dev_source_search",
    "success_action": "freeze_candidate_and_design_independent_confirmation_no_auto_paid_collection",
    "reserved895_accessed": False, "consumed304_accessed": False,
    "new_retrieval_calls": 0, "new_answer_calls": 0,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Protocol or results already exist")
    OUT.mkdir(parents=True, exist_ok=True)
    write_new(OUT / "protocol.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_new_alignment_feature_inspection_and_model_outcomes", "spec": SPEC})
    print("Frozen exactly three model fits and unchanged development roles.", flush=True)


def labels_for(features):
    n = len(features["query_ids"])
    utility = np.empty((n, 2), dtype=np.float64)
    old = features["source"] == "old"
    with np.load(BASE / "layer_pooling_v1/predictions.npz", allow_pickle=False) as saved:
        mapping = {str(q): i for i, q in enumerate(saved["query_ids"])}
        selected = np.array([mapping[str(q)] for q in features["query_ids"][old]])
        utility[old] = saved["utility"][selected]
    with np.load(RUNS / "m6_complete_cohort_extension_v1/source_labels_and_roles.npz", allow_pickle=False) as saved:
        mapping = {str(q): i for i, q in enumerate(saved["query_ids"])}
        selected = np.array([mapping[str(q)] for q in features["query_ids"][~old]])
        utility[~old] = saved["utility"][selected]
    if not np.isfinite(utility).all() or not ((utility >= 0) & (utility <= 1)).all():
        raise ValueError("Invalid existing paired utilities")
    return utility


def grouped_intervals(differences, groups):
    _, inv = np.unique(groups, return_inverse=True)
    k = int(inv.max()) + 1
    counts = np.bincount(inv)
    sums = np.vstack([np.bincount(inv, weights=column, minlength=k) for column in differences.T]).T
    config = SPEC["intervals_if_point_gate_passes"]
    rng = np.random.default_rng(config["seed"])
    draws = np.empty((config["group_bootstrap_draws"], differences.shape[1]))
    for start in range(0, len(draws), 200):
        selection = rng.integers(0, k, size=(min(200, len(draws) - start), k))
        draws[start:start + len(selection)] = sums[selection].sum(axis=1) / counts[selection].sum(axis=1)[:, None]
    tail = (1 - config["two_sided_confidence_each"]) / 2
    return np.quantile(draws, [tail, 1-tail], axis=0).T


def run():
    if read(OUT / "protocol.json")["spec"] != SPEC:
        raise ValueError("Frozen protocol differs")
    if {p.name for p in OUT.iterdir()} != {"protocol.json"}:
        raise RuntimeError("Prior execution exists; do not repeat automatically")
    feature_summary = read(FEATURES / "summary.json") if (FEATURES / "summary.json").exists() else {}
    if feature_summary.get("complete") is not True or feature_summary.get("status") != "complete":
        raise RuntimeError("Shared feature extraction must complete first")
    with np.load(FEATURES / "features.npz", allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    fit, dev = data["role"] == "fit", data["role"] == "dev"
    if set(map(str, data["group_ids"][fit])) & set(map(str, data["group_ids"][dev])):
        raise ValueError("Training and development groups overlap")
    for source in ("old", "beir"):
        if int(np.sum(fit & (data["source"] == source))) != SPEC["training"][source] or int(np.sum(dev & (data["source"] == source))) != SPEC["development"][source]:
            raise ValueError("Source roles differ from protocol")
    if not np.all(fit | dev) or len(np.unique(data["query_ids"])) != len(fit):
        raise ValueError("Unexpected or duplicated feature rows")
    names = list(map(str, data["scalar_names"]))
    scalars = data["scalars"][:, [names.index(name) for name in COMMON]].astype(np.float64)
    mean, scale = scalars[fit].mean(0), scalars[fit].std(0)
    scale[scale < 1e-12] = 1
    x = np.column_stack((data["M6"], data["Dense"], (scalars - mean) / scale)).astype(np.float64)
    added = data["scalars"][:, [names.index("aligned"), names.index("coordinate_control")]].astype(np.float64)
    augmented = np.column_stack((x, np.ones(len(x))))
    with threadpool_limits(limits=SPEC["threads"]):
        projection, _, rank, _ = linalg.lstsq(augmented[fit], added[fit], cond=1e-12, lapack_driver="gelsd")
        residual = added - augmented @ projection
    rms = np.sqrt(np.mean(residual[fit] ** 2, axis=0))
    for column in range(2):
        if rms[column] < 1e-12:
            residual[:, column] = 0
            rms[column] = 1
    residual /= rms
    utility = labels_for(data)
    gap = utility[:, 0] - utility[:, 1]
    weight = np.where(np.abs(gap[fit]) <= SPEC["tie_atol"], 0., np.abs(gap[fit]))
    target = (gap[fit] > 0).astype(np.float64)
    write_new(OUT / "started.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_queries": int(fit.sum()), "development_queries": int(dev.sum()),
        "projection_rank": int(rank), "residual_rms": rms.tolist()})
    began = time.perf_counter()
    arrays = dict(scalar_mean=mean, scalar_scale=scale, projection=projection, residual_rms=rms)
    results, logits = {}, {}
    for arm in SPEC["arms"]:
        xx = x if arm == "matched_base" else np.column_stack((x, residual[:, 0 if arm == "true_alignment" else 1]))
        model = probe.fit(xx[fit], target, weight, regularization=SPEC["regularization"])
        if not model["accepted"]:
            write_new(OUT / "numerical_failure.json", {"arm": arm, "status": model["status"], "gradient_inf": model["grad_inf"]})
            raise RuntimeError("Fixed numerical budget did not reach accepted solution")
        arrays[f"{arm}_coef"] = model["coef"]
        arrays[f"{arm}_intercept"] = np.array(model["intercept"])
        logits[arm] = xx[dev] @ model["coef"] + model["intercept"]
        results[arm] = {key: model[key] for key in ("loss", "data_loss", "penalty", "grad_inf", "status", "elapsed_seconds", "numerical_refinement")}
        print(json.dumps({"arm": arm, "gradient_inf": model["grad_inf"], "seconds": model["elapsed_seconds"]}), flush=True)
    dev_u = utility[dev]
    policy_u = {arm: np.where(value > 0, dev_u[:, 0], dev_u[:, 1]) for arm, value in logits.items()}
    comparisons = ["Dense", "BM25", "matched_base", "coordinate_control"]
    all_point_gates = True
    cohorts = {}
    for source in ("old", "beir"):
        selected = data["source"][dev] == source
        values = {"Dense": dev_u[selected, 1], "BM25": dev_u[selected, 0],
                  **{arm: value[selected] for arm, value in policy_u.items()}}
        difference = {name: float(np.mean(values["true_alignment"] - values[name])) for name in comparisons}
        gate = SPEC["point_gate"]
        passes = (min(difference["Dense"], difference["BM25"]) >= gate["true_over_each_fixed_each_source"]
                  and difference["matched_base"] >= gate["true_over_matched_base_each_source"]
                  and difference["coordinate_control"] >= gate["true_over_coordinate_control_each_source"])
        all_point_gates &= passes
        cohorts[source] = {"queries": int(selected.sum()), "F1": {key: float(value.mean()) for key, value in values.items()},
                           "true_minus": difference, "point_gate": bool(passes),
                           "bm25_switches": {arm: int(np.sum(value[selected] > 0)) for arm, value in logits.items()}}
    statistical_pass = False
    if all_point_gates:
        statistical_pass = True
        for source in ("old", "beir"):
            selected = data["source"][dev] == source
            references = [dev_u[:, 1], dev_u[:, 0], policy_u["matched_base"], policy_u["coordinate_control"]]
            differences = np.column_stack([policy_u["true_alignment"] - value for value in references])[selected]
            intervals = grouped_intervals(differences, data["group_ids"][dev][selected])
            cohorts[source]["intervals"] = {name: interval.tolist() for name, interval in zip(comparisons, intervals)}
            statistical_pass &= bool(np.all(intervals[:, 0] > 0))
    with (OUT / "models.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with (OUT / "development_predictions.npz").open("xb") as handle:
        np.savez_compressed(handle, query_ids=data["query_ids"][dev], group_ids=data["group_ids"][dev], source=data["source"][dev],
                            utility=dev_u, **{f"{arm}_logit": value for arm, value in logits.items()})
    summary = {"status": "complete_consumed_development_only", "cohorts": cohorts, "models": results,
               "fit_count": 3, "fit_seconds": time.perf_counter() - began, "point_gates_passed": bool(all_point_gates),
               "statistical_gates_passed": bool(statistical_pass),
               "bootstrap_draws_per_source": 10000 if all_point_gates else 0,
               "decision": "freeze_candidate_for_independent_confirmation_design" if statistical_pass else "close_fixed_alignment_recipe_no_search_extension",
               "independent_gain_established": False, "deployable": False,
               "reserved895_accessed": False, "consumed304_accessed": False,
               "new_answer_calls": 0, "new_retrieval_calls": 0}
    write_new(OUT / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument("--prepare", action="store_true")
    flags.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
