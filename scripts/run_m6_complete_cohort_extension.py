"""One fixed-lambda M6 fit adding complete BEIR training groups.

Both old calibration and the newly assigned BEIR development partition are
consumed evidence. The RAG environment, encoder, head family and threshold stay
fixed; neither source is reclaimed as independent confirmation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0")
import numpy as np
import torch
import yaml

from run_m6_capability_screen import require_execution_allowed, save_arrays, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "work/router_research"
BEIR = BASE / "m6_beir_validation_v1"
OUT = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/m6_complete_cohort_extension_v1"
IDENTITY = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1/old85000_group_identities.npz"
sys.path.insert(0, str(BASE))
import weighted_linear_probe_refined as probe
import lp_ft_training as bridge

SPEC = {
    "scope": "consumed_sources_development_only", "new_fits": 1,
    "baseline": "stored_fold0_M6_fit6144_head", "old_fit_queries": 6144, "old_cal_queries": 1536,
    "new_source": "complete_BEIR5209_mean3_pairs", "regularization": .001,
    "objective": "sum_abs_gap_weighted_BCE_over_sum_weights_plus_lambda_half_coef_norm_squared",
    "tie_atol": 1e-12, "intercept_penalized": False, "feature_transform": "none",
    "source_weights": "none; normalize absolute gap weights over each complete training arm",
    "BEIR_role_rule": "sort groups by SHA256(m6_complete_cohort_extension_v1|group); rank mod 5 == 0 is development, others training",
    "cross_source_rule": "require zero query/canonical group overlap; stop rather than silently merge duplicates",
    "solver": "existing_weighted_linear_probe_refined.fit", "numerical_budget": probe.NUMERICAL_BUDGET,
    "bounded_refinement": probe.REFINEMENT, "accept_gradient_inf": probe.ACCEPT_GRAD_INF,
    "solver_dtype": "float64", "cached_features_and_exported_head_dtype": "float32",
    "prediction": "existing_native_logits_cuda_bfloat16_autocast_batch8", "threshold": 0,
    "minimum_gain_over_old_head_each_cohort": .002,
    "minimum_gain_over_both_fixed_each_cohort": .01,
    "new_encoder_forwards": 0, "new_retrieval_calls": 0, "new_answer_calls": 0,
    "old_outer_test_evaluations": 0, "external_evaluations": 0,
    "no_automatic_extension": "no lambda search, source balancing, source intercept, threshold or partition changes",
}


def sources():
    return [BASE / "layer_pooling_v1/features.npz", BASE / "layer_pooling_v1/predictions.npz",
            BASE / "layer_pooling_v1/fold0_M6_head.pt", BASE / "layer_pooling_v1/fold0_M6.json",
            BASE / "e02_results/fold_indices.npz", BASE / "m6_pooled_offset_v1/cal_logits.npz",
            BEIR / "actions.npz", BEIR / "answers_v1/outcomes.jsonl", BEIR / "execution_contract.json",
            BEIR / "evaluation.json", IDENTITY]


def load_inputs():
    with np.load(BASE / "layer_pooling_v1/features.npz", allow_pickle=False) as archive:
        old_qids, old_groups, old_x = (archive[key].copy() for key in ("query_ids", "group_ids", "M6"))
    with np.load(BASE / "layer_pooling_v1/predictions.npz", allow_pickle=False) as archive:
        assert np.array_equal(old_qids, archive["query_ids"]) and np.array_equal(old_groups, archive["group_ids"])
        old_u = archive["utility"].copy()
    with np.load(BASE / "e02_results/fold_indices.npz", allow_pickle=False) as archive:
        fit, cal = archive["fold0_fit"].copy(), archive["fold0_calibration"].copy()
    assert (len(fit), len(cal)) == (SPEC["old_fit_queries"], SPEC["old_cal_queries"])
    with np.load(BASE / "m6_pooled_offset_v1/cal_logits.npz", allow_pickle=False) as archive:
        assert np.array_equal(cal, archive["fold0_cal_indices"])
        old_cal_logits = archive["fold0_O"].copy()
    with np.load(IDENTITY, allow_pickle=False) as archive:
        lookup = {str(q): (str(g), str(k)) for q, g, k in zip(archive["query_ids"], archive["original_group_ids"], archive["group_key_sha256"])}
    assert all(str(g) == lookup[str(q)][0] for q, g in zip(old_qids, old_groups))
    old_keys = np.array([lookup[str(q)][1] for q in old_qids])
    assert not set(old_keys[fit]) & set(old_keys[cal])
    assert not set(fit) & set(cal)
    with np.load(BEIR / "actions.npz", allow_pickle=False) as archive:
        qids, groups, x = (archive[key].copy() for key in ("query_ids", "group_ids", "M6"))
    assert len(qids) == 5209 and len(set(qids)) == 5209 and len(set(groups)) == 5206
    assert not set(qids) & set(old_qids) and not set(groups) & set(old_keys)
    ordered = sorted(set(map(str, groups)), key=lambda group: (hashlib.sha256(f"m6_complete_cohort_extension_v1|{group}".encode()).digest(), group))
    development_groups = set(ordered[::5])
    new_dev = np.flatnonzero(np.array([str(group) in development_groups for group in groups]))
    new_fit = np.flatnonzero(np.array([str(group) not in development_groups for group in groups]))
    assert not set(groups[new_fit]) & set(groups[new_dev])
    for values in (old_x, x):
        assert values.shape[1] == 384 and values.dtype == np.float32 and np.isfinite(values).all()
    cache = OUT / "source_labels_and_roles.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            for key, expected in (("query_ids", qids), ("group_ids", groups), ("new_fit", new_fit), ("new_dev", new_dev), ("old_fit", fit), ("old_cal", cal)):
                assert np.array_equal(archive[key], expected), key
            utility = archive["utility"].copy()
    else:
        # Numeric labels only; do not inspect answers, prompts or attempt ledgers.
        position = {str(q): i for i, q in enumerate(qids)}
        repeated = np.full((len(qids), 2, 3), np.nan, dtype=np.float64)
        with (BEIR / "answers_v1/outcomes.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                assert row["status"] == "ok" and row["model"] == "gpt-4.1-mini-2025-04-14"
                i = position[row["query_id"]]
                assert row["group_id"] == groups[i] and row["action"] in ("bm25", "dense")
                action = 0 if row["action"] == "bm25" else 1
                repeat = row["repeat_id"]
                assert repeat in (0, 1, 2) and np.isnan(repeated[i, action, repeat])
                value = float(row["f1"])
                assert np.isfinite(value) and 0 <= value <= 1
                repeated[i, action, repeat] = value
        assert np.isfinite(repeated).all()
        utility = repeated.mean(axis=2)
        reference = json.loads((BEIR / "evaluation.json").read_text(encoding="utf-8"))["absolute_descriptive_metrics"]
        assert all(abs(float(utility[:, i].mean())-reference[action]["F1"]) < 1e-12 for i, action in enumerate(("BM25", "Dense")))
        save_arrays(cache, query_ids=qids, group_ids=groups, utility=utility, new_fit=new_fit, new_dev=new_dev,
                    old_fit=fit, old_cal=cal, old_fit_canonical_keys=old_keys[fit], old_cal_canonical_keys=old_keys[cal])
    assert old_u.shape == (len(old_qids), 2) and utility.shape == (len(qids), 2)
    assert np.isfinite(old_u).all() and np.isfinite(utility).all()
    return dict(old_qids=old_qids, old_keys=old_keys, old_x=old_x, old_u=old_u, fit=fit, cal=cal,
                old_cal_logits=old_cal_logits, new_qids=qids, new_keys=groups, new_x=x, new_u=utility,
                new_fit=new_fit, new_dev=new_dev)


def summarize(logits, utility):
    gap = utility[:, 0]-utility[:, 1]
    action = logits > 0
    return dict(F1=float(np.where(action, utility[:, 0], utility[:, 1]).mean()), bm25_count=int(action.sum()),
                beneficial_count=int((action & (gap > SPEC["tie_atol"])).sum()),
                harmful_count=int((action & (gap < -SPEC["tie_atol"])).sum()),
                beneficial_mass=float(np.where(action, np.maximum(gap, 0), 0).mean()),
                harmful_mass=float(np.where(action, np.maximum(-gap, 0), 0).mean()))


def run():
    require_execution_allowed()
    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError("The existing native CUDA/BF16 head path is required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config_path = ROOT / "outputs/router/hotpotqa_bd_router_v1/config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = json.loads((BEIR / "execution_contract.json").read_text(encoding="utf-8"))
    assert config["retrieval"] == contract["retrieval"] and config["artifacts"] == contract["artifact_locations"]
    assert all(config["context"][key] == value for key, value in contract["context"].items())
    assert config["prompt"] == contract["prompt"]
    assert all(config["generation"][key] == contract["generation"][key] for key in ("provider", "model", "temperature", "max_output_tokens"))
    binding = {"spec": SPEC, "base_config": str(config_path), "execution_contract": contract,
               "source_metadata": {str(path): {"bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns} for path in sources()},
               "implementation": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                  (Path(__file__), BASE / "weighted_linear_probe.py", BASE / "weighted_linear_probe_refined.py", BASE / "lp_ft_training.py")}}
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "protocol.json"
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8"))["binding"] != binding:
            raise ValueError("Existing run differs from this fixed recipe")
    else:
        write_json(protocol, dict(created_at_utc=datetime.now(timezone.utc).isoformat(), binding=binding))
    if (OUT / "results.json").exists():
        print((OUT / "results.json").read_text(encoding="utf-8"))
        return
    data = load_inputs()
    fit, cal, new_fit, new_dev = (data[key] for key in ("fit", "cal", "new_fit", "new_dev"))
    old_gap = data["old_u"][fit, 0]-data["old_u"][fit, 1]
    new_gap = data["new_u"][new_fit, 0]-data["new_u"][new_fit, 1]
    combined_gap = np.r_[old_gap, new_gap]
    weights = np.where(np.abs(combined_gap) > SPEC["tie_atol"], np.abs(combined_gap), 0.)
    fit_path = OUT / "pooled_fit.json"
    if fit_path.exists():
        fitted = json.loads(fit_path.read_text(encoding="utf-8"))
    else:
        fitted = probe.fit(np.r_[data["old_x"][fit], data["new_x"][new_fit]], (combined_gap > 0).astype(float), weights,
                           regularization=SPEC["regularization"])
        fitted["coef"] = fitted["coef"].tolist()
        write_json(fit_path, fitted)
    if not fitted["accepted"]:
        write_json(OUT / "results.json", dict(status="numerical_fit_rejected_no_policy_evaluation", fit=fitted, decision="stop_no_automatic_retry"))
        return
    pooled = bridge.head_from_probe(fitted)
    torch.save(pooled, OUT / "pooled_head.pt")
    original = torch.load(BASE / "layer_pooling_v1/fold0_M6_head.pt", map_location="cpu", weights_only=True)
    scores = {"old_O": data["old_cal_logits"], "old_P": bridge.native_logits(data["old_x"][cal], pooled),
              "beir_O": bridge.native_logits(data["new_x"][new_dev], original),
              "beir_P": bridge.native_logits(data["new_x"][new_dev], pooled)}
    reports = {}
    for name, utility in (("old", data["old_u"][cal]), ("beir", data["new_u"][new_dev])):
        summary = {arm: summarize(scores[f"{name}_{arm}"], utility) for arm in ("O", "P")}
        fixed = {"BM25": float(utility[:, 0].mean()), "Dense": float(utility[:, 1].mean())}
        gain_old = summary["P"]["F1"]-summary["O"]["F1"]
        gain_fixed = summary["P"]["F1"]-max(fixed.values())
        reports[name] = dict(queries=len(utility), policies=summary, fixed_F1=fixed,
                             pooled_minus_old=gain_old, pooled_minus_best_fixed=gain_fixed,
                             development_gate=gain_old >= SPEC["minimum_gain_over_old_head_each_cohort"]
                             and gain_fixed >= SPEC["minimum_gain_over_both_fixed_each_cohort"])
    gate = all(value["development_gate"] for value in reports.values())
    save_arrays(OUT / "development_scores.npz", old_query_ids=data["old_qids"][cal], old_group_keys=data["old_keys"][cal],
                beir_query_ids=data["new_qids"][new_dev], beir_group_keys=data["new_keys"][new_dev],
                old_utility=data["old_u"][cal], beir_utility=data["new_u"][new_dev], **scores)
    support = {}
    for name, gap in (("old", old_gap), ("beir", new_gap)):
        weight = np.where(np.abs(gap) > SPEC["tie_atol"], np.abs(gap), 0.)
        support[name] = dict(queries=len(gap), positive_weight_queries=int((weight > 0).sum()),
                             absolute_gap_weight_mass=float(weight.sum()), share_of_combined_weight=float(weight.sum()/weights.sum()))
    result = dict(status="complete_two_consumed_cohort_development_screen", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  cohorts=reports, training_support=support, new_fit_queries=len(combined_gap),
                  beir_train_groups=len(set(data["new_keys"][new_fit])), beir_development_groups=len(set(data["new_keys"][new_dev])),
                  canonical_overlap_with_old=0, new_fits=1, old_head_refitted=False,
                  fit_summary={key: value for key, value in fitted.items() if key != "coef"},
                  development_gate=gate, independent_gain_established=False,
                  decision="promising_development_only_independent_protocol_required" if gate else "close_fixed_complete_cohort_recipe_no_rebalancing_lambda_or_threshold_extension")
    write_json(OUT / "results.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        run()
    else:
        print(json.dumps({"mode": "describe_only", "spec": SPEC}, indent=2))
