"""Prepare the frozen 2Wiki analysis; evaluate only a complete paired collection.

This script makes no model, retriever, or external calls. Synthetic checks are
limited to three mathematical/data-completeness checks and never save outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/2wiki_development_measurement_v1"
POLICIES = ("BM25", "Dense", "M6", "F", "leave_one_repeat_out", "empirical_mean3_oracle")
METRICS = ("f1", "em")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_frozen(run):
    spec = read_json(run / "sampling_protocol.json")["spec"]
    contract = read_json(run / "execution_contract.json")
    freeze = read_json(run / "actions_freeze.json")
    questions = [json.loads(line) for line in (run / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(run / "actions.npz", allow_pickle=False) as archive:
        actions = {key: archive[key] for key in ("query_ids", "group_ids", "M6_switch", "F_switch")}
    qids, gids = actions["query_ids"].astype(str), actions["group_ids"].astype(str)
    if len(qids) != 304 or len(set(qids)) != 304 or len(set(gids)) != 256:
        raise ValueError("Expected the frozen 304-query, 256-group sample")
    mapping = {row["query_id"]: row for row in questions}
    if len(mapping) != len(questions) or set(mapping) != set(qids):
        raise ValueError("Question and action query IDs must match exactly")
    for qid, gid in zip(qids, gids):
        row = mapping[qid]
        if row["group_id"] != gid or len(row["reference_answers"]) != 1:
            raise ValueError("Frozen group or single original reference differs")
    for policy in ("M6", "F"):
        values = actions[policy + "_switch"]
        if values.shape != (304,) or not np.isin(values, (False, True)).all():
            raise ValueError("Frozen switch vector must contain one Boolean per query")
        actions[policy + "_switch"] = values.astype(bool)
    if (spec["population_queries"], spec["population_components"], spec["sample_components"],
            spec["bootstrap_draws"], spec["bootstrap_seed"]) != (1199, 1006, 256, 10000, 2026091842):
        raise ValueError("Frozen sampling/bootstrap specification differs")
    if freeze["status"] != "frozen_pre_retrieval_predictions_before_new_answers":
        raise ValueError("Actions have not been frozen")
    return spec, contract, actions, mapping


def analysis_spec(spec, contract, actions):
    return {
        "scope": spec["scope"], "queries": 304, "groups": 256,
        "population_queries": 1199, "population_groups": 1006,
        "retained_queries": 895, "retained_groups": 750,
        "required_successful_outcomes": 1824,
        "outcome_schema": {"required": ["query_id", "group_id", "action", "repeat_id", "status", "answer", "f1", "em", "model"],
                           "actions": ["bm25", "dense"], "repeat_ids": [0, 1, 2], "status": "ok",
                           "model": contract["generation"]["model"],
                           "metrics": "rescore raw answer with existing evaluator and require saved f1/em agreement"},
        "missingness": "refuse any missing, duplicate, extra, failed, or mismatched record; no partial policy effects",
        "estimand": "query-weighted mean3 answer utility in the conditional 1199-query frame",
        "primary_estimator": "Hajek: sum sampled group utility totals / sum sampled group sizes",
        "sensitivity_estimator": "HT: population_groups / (sample_groups * population_queries) * sampled utility sum",
        "bootstrap": {"draws": spec["bootstrap_draws"], "seed": spec["bootstrap_seed"],
                      "rng": "numpy default_rng PCG64; integers group indexes; batches of 200",
                      "unit": "sampled dependency group; preserve all queries, actions, repeats and policies jointly",
                      "estimator_each_draw": "ratio of resampled query utility sum to resampled query count",
                      "finite_population_correction": False,
                      "interval": "percentile; approximate conservative no-FPC inference, not exact finite-sample coverage",
                      "reason_no_FPC": "Do not shrink generation-repeat noise with a finite-frame sampling correction"},
        "primary_family": {"policy": "F", "comparisons": ["Dense", "BM25"], "metric": "f1",
                           "two_sided_CI_each": 0.975, "Bonferroni_family_nominal_coverage": 0.95,
                           "gate": "both point gains >= 0.01 and both lower CI endpoints > 0",
                           "pass_meaning": "external development promising only"},
        "diagnostic_family": {"policy": "leave_one_repeat_out", "comparisons": ["Dense", "BM25"], "metric": "f1",
                              "two_sided_CI_each": 0.975, "Bonferroni_family_nominal_coverage": 0.95,
                              "gate": "both point gains >= 0.02 and both lower CI endpoints > 0",
                              "pass_meaning": "worth proposing one new query-observable mechanism; no query-only learnability claim"},
        "multiplicity": "primary and diagnostic families handled separately; no joint overall 95-percent FWER claim across both families",
        "M6": "descriptive reference; 95-percent pointwise intervals; never an acceptance candidate",
        "secondary_EM": "descriptive 95-percent pointwise intervals; never overrides F1 gates",
        "leave_one_repeat_out": "for each query and held repeat, choose larger mean F1 on other two repeats; tie Dense; score held repeat; average all three; repeat IDs are scheduling pairs, not shared random seeds",
        "empirical_mean3_oracle": "choose action using same query mean3 F1; report descriptively with optimistic selection bias; EM uses F1-selected action, not an EM oracle",
        "oracle_boundary": "leave-one-repeat-out is an answer-informed diagnostic, neither deployable query-only policy nor unbiased true expected oracle; failure does not rule out all routable gain",
        "outcome_free_gain_ceiling_vs_Dense": {p: {"switches": int(actions[p + "_switch"].sum()),
            "sample_Hajek_ceiling": float(actions[p + "_switch"].mean()),
            "interpretation": "support ceiling only; not observed utility or a population guarantee"} for p in ("M6", "F")},
        "next_actions": {"primary_pass": "freeze candidate; prepare separately authorized reserved-group protocol using measured precision",
                         "diagnostic_only_pass": "propose one distinct query-observable mechanism using measured sample as development; retain unused groups",
                         "neither_pass": "report negative or inconclusive evidence; no automatic sample expansion, threshold tuning, slicing, or mechanism rejection"},
        "objective_achieved": False, "new_training": False, "external_calls_by_this_script": 0,
    }


def prepare(run):
    spec, contract, actions, _ = load_frozen(run)
    payload = analysis_spec(spec, contract, actions)
    path = run / "analysis_protocol.json"
    if path.exists():
        if read_json(path)["spec"] != payload:
            raise ValueError("Analysis is already frozen with a different specification")
        return {"status": "already_prepared", "protocol": str(path), "external_calls": 0}
    write_new(path, {"status": "frozen_before_new_answers_local_preparation_only",
                     "created_at_utc": datetime.now(timezone.utc).isoformat(), "spec": payload})
    return {"status": "prepared_no_outcomes_evaluated", "protocol": str(path), "external_calls": 0}


def complete_records(records, qids, gids, model):
    expected_groups = dict(zip(qids, gids))
    expected = {(q, a, r) for q in qids for a in ("bm25", "dense") for r in range(3)}
    indexed = {}
    for row in records:
        if set(("query_id", "group_id", "action", "repeat_id", "status", "answer", "f1", "em", "model")) - set(row):
            raise ValueError("Outcome is missing required fields")
        if type(row["repeat_id"]) is not int:
            raise ValueError("repeat_id must be an integer")
        key = (row["query_id"], row["action"], row["repeat_id"])
        if key not in expected or key in indexed:
            raise ValueError("Unexpected or duplicate outcome key")
        if row["status"] != "ok" or row["group_id"] != expected_groups[row["query_id"]] or row["model"] != model:
            raise ValueError("Failed outcome, group mismatch, or model mismatch")
        if not isinstance(row["answer"], str):
            raise ValueError("A successful outcome must retain its raw answer string")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(f"Incomplete collection: {len(indexed)}/{len(expected)} successful unique outcomes; no partial evaluation")
    return indexed


def loo_choices(f1):
    other_two = (f1.sum(axis=2, keepdims=True) - f1) / 2
    return other_two[:, 0, :] > other_two[:, 1, :]


def ratio_from_totals(totals, sizes):
    return totals.sum(axis=0) / sizes.sum()


def evaluate(run, outcomes):
    spec, contract, actions, questions = load_frozen(run)
    protocol = read_json(run / "analysis_protocol.json")
    if protocol["spec"] != analysis_spec(spec, contract, actions):
        raise ValueError("Current inputs differ from frozen analysis specification")
    if (run / "evaluation.json").exists() or (run / "measurement_scores.npz").exists():
        raise ValueError("Evaluation artifacts already exist; do not rerun or overwrite them")
    qids, gids = actions["query_ids"].astype(str), actions["group_ids"].astype(str)
    records = [json.loads(line) for line in outcomes.read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = complete_records(records, qids, gids, contract["generation"]["model"])
    sys.path.insert(0, str(ROOT))
    from src.evaluators.hotpot_answer import answer_metrics
    values = np.empty((len(qids), 2, 3, 2), dtype=np.float64)
    for i, qid in enumerate(qids):
        for a, action in enumerate(("bm25", "dense")):
            for repeat in range(3):
                row = indexed[qid, action, repeat]
                score = answer_metrics(row["answer"], questions[qid]["reference_answers"])
                computed = (score["normalized_token_f1"], score["normalized_exact_match"])
                if not np.allclose([row["f1"], row["em"]], computed, rtol=0, atol=1e-12):
                    raise ValueError("Saved F1/EM does not match the fixed evaluator and original reference")
                values[i, a, repeat] = computed
    average = values.mean(axis=2)
    utilities = np.empty((len(qids), len(POLICIES), 2), dtype=np.float64)
    utilities[:, :2] = average
    for column, policy in ((2, "M6"), (3, "F")):
        utilities[:, column] = np.where(actions[policy + "_switch"][:, None], average[:, 0], average[:, 1])
    loo_bm25 = loo_choices(values[:, :, :, 0])
    utilities[:, 4] = np.where(loo_bm25[:, :, None], values[:, 0], values[:, 1]).mean(axis=1)
    oracle_bm25 = average[:, 0, 0] > average[:, 1, 0]
    utilities[:, 5] = np.where(oracle_bm25[:, None], average[:, 0], average[:, 1])
    _, groups = np.unique(gids, return_inverse=True)
    sizes = np.bincount(groups)
    totals = np.zeros((len(sizes), len(POLICIES), 2), dtype=np.float64)
    np.add.at(totals, groups, utilities)
    point = ratio_from_totals(totals, sizes)
    ht = spec["population_components"] / (len(sizes) * spec["population_queries"]) * totals.sum(axis=0)
    rng = np.random.default_rng(spec["bootstrap_seed"])
    bootstrap = np.empty((spec["bootstrap_draws"], len(POLICIES), 2), dtype=np.float64)
    for start in range(0, len(bootstrap), 200):
        stop = min(start + 200, len(bootstrap))
        selected = rng.integers(0, len(sizes), size=(stop - start, len(sizes)))
        bootstrap[start:stop] = totals[selected].sum(axis=1) / sizes[selected].sum(axis=1)[:, None, None]

    def interval(samples, confidence):
        tail = (1 - confidence) / 2
        return np.quantile(samples, [tail, 1 - tail]).tolist()

    policy_means = {}
    for p, name in enumerate(POLICIES):
        policy_means[name] = {metric: {"Hajek": float(point[p, k]), "HT": float(ht[p, k])}
                              for k, metric in enumerate(METRICS)}
        if name != "empirical_mean3_oracle":
            for k, metric in enumerate(METRICS):
                policy_means[name][metric]["descriptive_pointwise_95CI"] = interval(bootstrap[:, p, k], .95)

    def contrast(p, reference, metric, confidence):
        a, b, k = POLICIES.index(p), POLICIES.index(reference), METRICS.index(metric)
        return {"Hajek_gain": float(point[a, k] - point[b, k]), "HT_gain": float(ht[a, k] - ht[b, k]),
                "confidence": confidence, "two_sided_percentile_CI": interval(bootstrap[:, a, k] - bootstrap[:, b, k], confidence)}

    comparisons = {p: {reference: {metric: contrast(p, reference, metric, .975 if p != "M6" and metric == "f1" else .95)
                    for metric in METRICS} for reference in ("Dense", "BM25")} for p in ("F", "M6", "leave_one_repeat_out")}
    def passes(policy, minimum):
        return all(comparisons[policy][reference]["f1"]["Hajek_gain"] >= minimum and
                   comparisons[policy][reference]["f1"]["two_sided_percentile_CI"][0] > 0 for reference in ("Dense", "BM25"))
    primary_pass, diagnostic_pass = passes("F", .01), passes("leave_one_repeat_out", .02)
    switches = {}
    gap = average[:, 0, 0] - average[:, 1, 0]
    for policy in ("M6", "F"):
        switch = actions[policy + "_switch"]
        switches[policy] = {"bm25_count": int(switch.sum()), "beneficial_count": int(np.sum(switch & (gap > 0))),
                            "harmful_count": int(np.sum(switch & (gap < 0))), "tie_count": int(np.sum(switch & (gap == 0))),
                            "beneficial_F1_mass_Hajek": float(np.mean(switch * np.maximum(gap, 0))),
                            "harmful_F1_mass_Hajek": float(np.mean(switch * np.maximum(-gap, 0)))}
    decision = "primary_pass" if primary_pass else "diagnostic_only_pass" if diagnostic_pass else "neither_pass"
    result = {"status": "complete_external_development_measurement", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_protocol": str(run / "analysis_protocol.json"), "outcomes": str(outcomes),
              "queries": len(qids), "groups": len(sizes), "successful_outcomes": len(indexed),
              "policy_means": policy_means, "comparisons": comparisons, "switch_diagnostics": switches,
              "F_external_development_promising": primary_pass, "stable_answer_informed_diagnostic_gate": diagnostic_pass,
              "decision": decision, "next_action": protocol["spec"]["next_actions"][decision],
              "multiplicity": protocol["spec"]["multiplicity"], "oracle_boundary": protocol["spec"]["oracle_boundary"],
              "mean3_oracle_warning": protocol["spec"]["empirical_mean3_oracle"],
              "independent_HotpotQA_gain_established": False, "query_only_learnability_established": False,
              "objective_achieved": False, "external_calls_by_this_script": 0}
    with (run / "measurement_scores.npz").open("xb") as stream:
        np.savez_compressed(stream, query_ids=qids, group_ids=gids, outcomes=values,
                            policy_names=np.asarray(POLICIES), metric_names=np.asarray(METRICS),
                            query_policy_utilities=utilities, loo_bm25=loo_bm25,
                            oracle_bm25=oracle_bm25, bootstrap_policy_means=bootstrap)
    write_new(run / "evaluation.json", result)
    return {"status": result["status"], "decision": decision, "result": str(run / "evaluation.json")}


def self_check():
    # 1. Modifying a held-out outcome cannot change that held-out action choice.
    toy = np.array([[[1., 0., 0.], [0., 1., 0.]]])
    baseline = loo_choices(toy)
    for repeat in range(3):
        changed = toy.copy()
        changed[:, :, repeat] = [[.25, .75]]
        assert np.array_equal(loo_choices(changed)[:, repeat], baseline[:, repeat])
    held = np.where(baseline, toy[:, 0], toy[:, 1]).mean()
    assert held == 0 and toy.mean(axis=2).max() == 1 / 3
    # 2. Query-weighted cluster ratio differs from the equally weighted group mean.
    sizes, totals = np.array([1, 3]), np.array([1., 0.])
    assert ratio_from_totals(totals, sizes) == .25 and np.mean(totals / sizes) == .5
    # 3. One missing paired repeat is rejected; toy records never become artifacts.
    records = [{"query_id": "toy", "group_id": "g", "action": a, "repeat_id": r,
                "status": "ok", "answer": "synthetic", "f1": 0., "em": 0., "model": "toy-model"}
               for a in ("bm25", "dense") for r in range(3)]
    assert len(complete_records(records, ["toy"], ["g"], "toy-model")) == 6
    try:
        complete_records(records[:-1], ["toy"], ["g"], "toy-model")
    except ValueError as exc:
        assert "Incomplete collection" in str(exc)
    else:
        raise AssertionError("Incomplete records were accepted")
    return {"status": "passed_three_bounded_checks", "checks": ["LOO_no_held_repeat_selection", "query_weighted_cluster_ratio", "missing_repeat_refused"],
            "real_outcomes_evaluated": 0, "external_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    modes.add_argument("--self-check", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--outcomes", type=Path, help="Complete successful outcomes JSONL; default run/answers_v1/outcomes.jsonl")
    args = parser.parse_args()
    if args.self_check:
        receipt = self_check()
    elif args.prepare:
        receipt = prepare(args.run_dir)
    else:
        receipt = evaluate(args.run_dir, args.outcomes or args.run_dir / "answers_v1/outcomes.jsonl")
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
