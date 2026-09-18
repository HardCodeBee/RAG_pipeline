"""One descriptive diagnosis of the closed static-pairing experiment.

No model or preprocessing is fitted. No bootstrap, threshold sweep, question
text, new outcome, reserved set, or reopening of the failed recipe is involved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs"
EFFECT = RUNS / "static_pairing_effect_v1"
FEATURES = RUNS / "static_pairing_features_v1"
OUT = RUNS / "static_pairing_transfer_description_v1"
COMMON = ["log_unique_term_count", "corpus_known_fraction", "centroid_available",
          "idf_mean", "idf_std", "idf_max", "log_total_lexical_mass_per_document",
          "centered_centroid_squared_norm"]
SPEC = {
    "role": "post_result_descriptive_diagnosis_not_confirmation_or_candidate_selection",
    "question": "how_many_decision_changes_drive_each_source_gain_and_are_residual_marginals_different",
    "comparisons": ["true_alignment_minus_matched_base", "true_alignment_minus_coordinate_control"],
    "cohorts": ["old_dev1536", "beir_dev1044"],
    "outcome_analysis": ["fixed_2x2_action_table", "benefit_harm_tie_counts_and_masses",
                         "top1_top5_positive_group_contribution_shares",
                         "positive_group_concentration_count_not_statistical_effective_sample_size",
                         "range_of_group_leave_one_out_means_without_changing_predictions"],
    "group_rule": "existing_canonical_group_ids; aggregate_query_utility_differences_within_group",
    "residual_analysis": "use_saved_scaler_projection_and_RMS; no_refit",
    "residual_summaries": "mean_population_std_p05_p25_p50_p75_p95_and_pooled_fit_quartile_bin_fractions",
    "bins": "four_intervals_defined_only_by_pooled_fit_residual_quartiles",
    "outcomes_by_residual_bins": False,
    "uncertainty_claim": "none; leave_one_group_out_range_is_not_a_confidence_interval",
    "new_fits": 0, "new_encoder_calls": 0, "bootstrap_draws": 0,
    "new_answer_calls": 0, "natural_query_text_reads": 0,
    "reserved895_accessed": False, "consumed304_accessed": False,
    "decision_rule": "record_descriptive_limits_only; failed_recipe_remains_closed_for_all_results",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Description was already prepared or executed")
    if read(EFFECT / "results.json")["decision"] != "close_fixed_alignment_recipe_no_search_extension":
        raise ValueError("This description applies to the closed three-arm recipe")
    OUT.mkdir(parents=True, exist_ok=True)
    write_new(OUT / "protocol.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "spec": SPEC,
        "status": "frozen_after_aggregate_results_before_decision_change_and_residual_analysis"})
    print("Frozen one descriptive analysis; no model changes or confirmation claims.", flush=True)


def contrast(utility, groups, reference_action, candidate_action):
    n = len(groups)
    change = candidate_action.astype(np.int8) - reference_action.astype(np.int8)
    difference = change * (utility[:, 0] - utility[:, 1])
    benefit, harm = difference > 1e-12, difference < -1e-12
    changed = change != 0
    unique, inverse = np.unique(groups, return_inverse=True)
    group_sizes = np.bincount(inverse)
    group_sum = np.bincount(inverse, weights=difference)
    group_positive_mass = np.bincount(inverse, weights=np.maximum(difference, 0))
    positive_groups = np.sort(group_positive_mass[group_positive_mass > 1e-12])[::-1]
    positive_total = float(positive_groups.sum())
    leave_out = (difference.sum() - group_sum) / (n - group_sizes)
    return {
        "queries": n, "groups": len(unique), "mean_increment": float(difference.mean()),
        "decision_changes": int(changed.sum()), "decision_change_fraction": float(changed.mean()),
        "groups_with_decision_changes": int(np.unique(inverse[changed]).size),
        "action_table_reference_to_candidate": {
            "Dense_to_Dense": int(np.sum(~reference_action & ~candidate_action)),
            "Dense_to_BM25": int(np.sum(~reference_action & candidate_action)),
            "BM25_to_Dense": int(np.sum(reference_action & ~candidate_action)),
            "BM25_to_BM25": int(np.sum(reference_action & candidate_action)),
        },
        "beneficial_changes": int(benefit.sum()), "harmful_changes": int(harm.sum()),
        "zero_utility_changes": int(np.sum(changed & ~(benefit | harm))),
        "benefit_mass_per_all_queries": float(difference[benefit].sum() / n),
        "harm_mass_per_all_queries": float(-difference[harm].sum() / n),
        "positive_net_groups": int(np.sum(group_sum > 1e-12)),
        "negative_net_groups": int(np.sum(group_sum < -1e-12)),
        "top1_share_of_positive_group_mass": float(positive_groups[:1].sum() / positive_total) if positive_total else None,
        "top5_share_of_positive_group_mass": float(positive_groups[:5].sum() / positive_total) if positive_total else None,
        "positive_group_concentration_count": float(positive_total**2 / (positive_groups @ positive_groups)) if positive_total else None,
        "concentration_count_is_statistical_effective_sample_size": False,
        "largest_absolute_group_contribution_per_all_queries": float(np.max(np.abs(group_sum)) / n),
        "leave_one_group_out_mean_range": [float(leave_out.min()), float(leave_out.max())],
        "leave_one_group_out_range_is_CI": False,
    }


def run():
    if read(OUT / "protocol.json")["spec"] != SPEC or {p.name for p in OUT.iterdir()} != {"protocol.json"}:
        raise RuntimeError("Protocol differs or results already exist")
    with np.load(EFFECT / "development_predictions.npz", allow_pickle=False) as saved:
        predictions = {key: saved[key].copy() for key in saved.files}
    contrasts = {}
    for source in ("old", "beir"):
        mask = predictions["source"] == source
        contrasts[source] = {}
        for reference in ("matched_base", "coordinate_control"):
            contrasts[source][reference] = contrast(
                predictions["utility"][mask], predictions["group_ids"][mask],
                predictions[f"{reference}_logit"][mask] > 0,
                predictions["true_alignment_logit"][mask] > 0,
            )
    with np.load(FEATURES / "features.npz", allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    with np.load(EFFECT / "models.npz", allow_pickle=False) as saved:
        model = {key: saved[key].copy() for key in ("scalar_mean", "scalar_scale", "projection", "residual_rms")}
    fit = data["role"] == "fit"
    names = list(map(str, data["scalar_names"]))
    common = data["scalars"][:, [names.index(name) for name in COMMON]]
    x = np.column_stack((data["M6"], data["Dense"], (common - model["scalar_mean"]) / model["scalar_scale"], np.ones(len(fit))))
    added = data["scalars"][:, [names.index("aligned"), names.index("coordinate_control")]]
    raw_residual = added - x @ model["projection"]
    raw_residual[:, np.sqrt(np.mean(raw_residual[fit] ** 2, axis=0)) < 1e-12] = 0
    residual = raw_residual / model["residual_rms"]
    descriptions = {}
    for column, name in enumerate(("true_alignment", "coordinate_control")):
        edges = np.quantile(residual[fit, column], [.25, .5, .75])
        values = {}
        for source in ("old", "beir"):
            for role in ("fit", "dev"):
                selected = residual[(data["source"] == source) & (data["role"] == role), column]
                bins = np.bincount(np.searchsorted(edges, selected, side="right"), minlength=4)
                values[f"{source}_{role}"] = {
                    "queries": len(selected), "mean": float(selected.mean()), "std": float(selected.std()),
                    "quantiles_p05_p25_p50_p75_p95": np.quantile(selected, [.05, .25, .5, .75, .95]).tolist(),
                    "pooled_fit_quartile_fractions": (bins / len(selected)).tolist(),
                }
        descriptions[name] = {"pooled_fit_quartile_edges": edges.tolist(), "cohorts": values}
    result = {"status": "complete_descriptive_only", "contrasts": contrasts, "residual_marginals": descriptions,
              "uncertainty_quantified": False, "source_effect_causally_identified": False,
              "recipe_reopened": False, "new_fits": 0, "bootstrap_draws": 0,
              "new_queries": 0, "new_answer_calls": 0, "reserved895_accessed": False, "consumed304_accessed": False}
    write_new(OUT / "results.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
