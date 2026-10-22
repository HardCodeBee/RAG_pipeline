"""Explicit read-only native scoring; never imported by an execution gate.

``--self-test`` uses synthetic records only.  The normal entry point reads only
this experiment's checkpoint after preparation.check(), and writes its separate
DETAILS/PAIRED/REPORT artifacts.  It never constructs Cache, a provider or a
retrieval worker.  Historical predictions are local scoring references only.
"""
from collections import Counter
from contextlib import closing
import argparse
import json
import math
from pathlib import Path
import sqlite3

ARMS = ("N128", "N256", "F256", "R256")
DATASETS = ("hotpotqa", "2wiki", "musique")
TOKENS = ("prompt_tokens", "completion_tokens", "total_tokens")
QUALITY_TERMINAL = {"success", "success_truncated", "algorithm_failure"}
COMPARISONS = (("N128", "N256"), ("N128", "F256"), ("N128", "R256"), ("F256", "R256"))


def estimated_cny(usage):
    """Frozen Qwen pricing; logical usage, not another actual-paid ledger."""
    return usage["prompt_tokens"] * 0.8e-6 + usage["completion_tokens"] * 2e-6


def stage_usage(stages):
    result = dict.fromkeys(TOKENS, 0)
    for stage in stages:
        usage = stage["usage"]
        assert all(type(usage.get(k)) is int and usage[k] >= 0 for k in TOKENS), "Incomplete native usage"
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Native usage mismatch"
        for key in TOKENS:
            result[key] += usage[key]
    return result


def native_result(outcome, scorer, reference):
    """Algorithm terminals score zero; technical absence never reaches here."""
    assert outcome["status"] in QUALITY_TERMINAL
    if outcome["status"] == "algorithm_failure":
        return dict(f1=0.0, em=0.0)
    assert isinstance(outcome["prediction"], str) and outcome["prediction"].strip()
    native = scorer(outcome["prediction"], reference)
    result = {key: float(native[key]) for key in ("f1", "em")}
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in result.values())
    return result


def _mean(values):
    return sum(values) / len(values) if values else None


def _summary(index, arm, ids):
    selected = [index[i, arm] for i in sorted(ids) if (i, arm) in index]
    found = [r for r in selected if r.get("f1") is not None]
    n, available = len(ids), len(found)
    missing = n - available
    result = dict(planned=n, quality_terminal=available, missing=missing,
                  statuses=dict(Counter(r["status"] for r in selected)),
                  native_full_correct=sum(r["native_full_correct"] for r in found))
    for metric, title in (("f1", "F1"), ("em", "EM")):
        total = sum(r[metric] for r in found)
        result[title] = total / n if n and not missing else None
        result[title + "_available"] = total / available if available else None
        result[title + "_fixed_denominator_bounds"] = [total / n, (total + missing) / n] if n else None
    usage_known = [r for r in found if r["full_flow_logical_usage"] is not None]
    for key, title in (("prompt_tokens", "input"), ("completion_tokens", "output"), ("total_tokens", "total")):
        result["mean_" + title + "_tokens"] = _mean([r["full_flow_logical_usage"][key] for r in usage_known])
    result["mean_logical_estimated_CNY"] = _mean([r["full_flow_logical_estimated_CNY"] for r in found
                                                if r["full_flow_logical_estimated_CNY"] is not None])
    result["token_mean_denominator"] = len(usage_known)
    result["token_scope"] = "Quality-terminal flows including algorithm terminals; partial technical-held usage is separate in DETAILS"
    return result


def _pair(index, reference_arm, candidate_arm, ids):
    pairs = []
    bounds = {key: [0.0, 0.0] for key in ("f1", "em")}
    for ident in sorted(ids):
        ref, cand = index.get((ident, reference_arm)), index.get((ident, candidate_arm))
        valid_ref = ref is not None and ref.get("f1") is not None
        valid_cand = cand is not None and cand.get("f1") is not None
        for key in bounds:
            if valid_ref and valid_cand:
                lower = upper = cand[key] - ref[key]
            elif valid_ref:
                lower, upper = -ref[key], 1 - ref[key]
            elif valid_cand:
                lower, upper = cand[key] - 1, cand[key]
            else:
                lower, upper = -1, 1
            bounds[key][0] += lower
            bounds[key][1] += upper
        if not (valid_ref and valid_cand):
            continue
        row = dict(id=ident, dataset=ref["dataset"], reference_arm=reference_arm, candidate_arm=candidate_arm,
                   delta_F1=cand["f1"] - ref["f1"], delta_EM=cand["em"] - ref["em"])
        for key, name in (("prompt_tokens", "input"), ("completion_tokens", "output"), ("total_tokens", "total")):
            row["reference_" + name + "_tokens"] = ref["full_flow_logical_usage"][key] if ref["full_flow_logical_usage"] is not None else None
            row["candidate_" + name + "_tokens"] = cand["full_flow_logical_usage"][key] if cand["full_flow_logical_usage"] is not None else None
        row["delta_logical_estimated_CNY"] = (cand["full_flow_logical_estimated_CNY"] - ref["full_flow_logical_estimated_CNY"]
            if cand["full_flow_logical_estimated_CNY"] is not None and ref["full_flow_logical_estimated_CNY"] is not None else None)
        pairs.append(row)
    n = len(pairs)
    result = dict(reference_arm=reference_arm, candidate_arm=candidate_arm, planned=len(ids),
                  complete_pairs=n, missing_pairs=len(ids) - n,
                  delta_F1=_mean([r["delta_F1"] for r in pairs]), delta_EM=_mean([r["delta_EM"] for r in pairs]),
                  improved=sum(r["delta_F1"] > 0 for r in pairs), tied=sum(r["delta_F1"] == 0 for r in pairs),
                  harmed=sum(r["delta_F1"] < 0 for r in pairs),
                  scope="Available same-query quality-terminal pairs; missing technical conditions are not zero")
    for key, title in (("f1", "F1"), ("em", "EM")):
        result["delta_" + title + "_complete_planned"] = result["delta_" + title] if n == len(ids) and n else None
        result["delta_" + title + "_fixed_denominator_bounds"] = [v / len(ids) for v in bounds[key]] if ids else None
    for name in ("input", "output", "total"):
        known = [r for r in pairs if r["reference_" + name + "_tokens"] is not None
                 and r["candidate_" + name + "_tokens"] is not None]
        ref_sum = sum(r["reference_" + name + "_tokens"] for r in known)
        cand_sum = sum(r["candidate_" + name + "_tokens"] for r in known)
        result["ratio_" + name + "_tokens_on_matched_pairs"] = cand_sum / ref_sum if ref_sum else None
        result["mean_delta_" + name + "_tokens"] = (cand_sum - ref_sum) / len(known) if known else None
        result["matched_" + name + "_token_pairs"] = len(known)
    result["mean_delta_logical_estimated_CNY"] = _mean([r["delta_logical_estimated_CNY"] for r in pairs
                                                      if r["delta_logical_estimated_CNY"] is not None])
    return result, pairs


def _section(index, inputs, ids, arms=ARMS):
    by_dataset_ids = {d: {i for i in ids if inputs[i]["dataset"] == d} for d in DATASETS}
    by_dataset = {d: {a: _summary(index, a, selected) for a in arms} for d, selected in by_dataset_ids.items()}
    macro = {}
    for arm in arms:
        complete = [by_dataset[d][arm]["F1"] for d in DATASETS]
        macro[arm] = dict(F1=_mean(complete) if all(v is not None for v in complete) else None,
                         EM=_mean([by_dataset[d][arm]["EM"] for d in DATASETS]) if all(v is not None for v in complete) else None,
                         F1_available=_mean([by_dataset[d][arm]["F1_available"] for d in DATASETS])
                         if all(by_dataset[d][arm]["quality_terminal"] for d in DATASETS) else None,
                         EM_available=_mean([by_dataset[d][arm]["EM_available"] for d in DATASETS])
                         if all(by_dataset[d][arm]["quality_terminal"] for d in DATASETS) else None,
                         dataset_weights={d: 1 / len(DATASETS) for d in DATASETS},
                         available_macro_is_partial=any(by_dataset[d][arm]["missing"] for d in DATASETS))
    paired, paired_by_dataset, paired_macro, rows = {}, {}, {}, []
    for reference_arm, candidate_arm in COMPARISONS:
        if reference_arm not in arms or candidate_arm not in arms:
            continue
        key = reference_arm + "_to_" + candidate_arm
        paired[key], values = _pair(index, reference_arm, candidate_arm, ids)
        rows += values
        paired_by_dataset[key] = {d: _pair(index, reference_arm, candidate_arm, selected)[0]
                                  for d, selected in by_dataset_ids.items()}
        values = list(paired_by_dataset[key].values())
        paired_macro[key] = dict(
            delta_F1=_mean([v["delta_F1_complete_planned"] for v in values])
            if all(v["delta_F1_complete_planned"] is not None for v in values) else None,
            delta_EM=_mean([v["delta_EM_complete_planned"] for v in values])
            if all(v["delta_EM_complete_planned"] is not None for v in values) else None,
            delta_F1_available=_mean([v["delta_F1"] for v in values]) if all(v["complete_pairs"] for v in values) else None,
            delta_EM_available=_mean([v["delta_EM"] for v in values]) if all(v["complete_pairs"] for v in values) else None,
            complete_pairs_by_dataset={d: v["complete_pairs"] for d, v in paired_by_dataset[key].items()},
            available_macro_is_partial=any(v["missing_pairs"] for v in values))
    return dict(planned_queries=len(ids), arms={a: _summary(index, a, ids) for a in arms},
                overall_weighting="Query mean; equal-weight dataset macro is a separate field",
                by_dataset=by_dataset, equal_weight_dataset_macro=macro,
                paired_predeclared=paired, paired_by_dataset=paired_by_dataset,
                paired_equal_weight_dataset_macro=paired_macro), rows


def aggregate_report(details, inputs, historical_details):
    """Pure aggregation. Inputs contain no gold or answer text requirements."""
    items = {r["id"]: r for r in inputs}
    assert len(items) == len(inputs)
    ids = set(items)
    index = {(r["id"], r["arm"]): r for r in details}
    assert len(index) == len(details)
    assert all(i in ids and a in ARMS for i, a in index)
    assert all(r["dataset"] == items[r["id"]]["dataset"] for r in details)
    whole, pairs = _section(index, items, ids)
    ordinary_ids = {i for i in ids if items[i]["supporting_questions"]}
    ordinary, _ = _section(index, items, ordinary_ids)
    old_index = {(r["id"], "historical_P02"): r for r in historical_details}
    assert len(old_index) == len(historical_details) and all(i in ids for i, _ in old_index)
    assert not set(index).intersection(old_index)
    combined = {**index, **old_index}
    matched_ids = {i for i in ids if (i, "historical_P02") in old_index
                   and old_index[i, "historical_P02"].get("f1") is not None
                   and all((i, a) in index and index[i, a].get("f1") is not None for a in ARMS)}
    history = dict(scope="Nonconcurrent native P02 reference only; historical availability never selects the fixed cohort",
                   coverage=_summary(old_index, "historical_P02", ids),
                   by_dataset={d: _summary(old_index, "historical_P02", {i for i in ids if items[i]["dataset"] == d})
                               for d in DATASETS},
                   common_all_four_fresh_and_historical=dict(n=len(matched_ids), ids=sorted(matched_ids),
                       historical=_summary(combined, "historical_P02", matched_ids),
                       fresh={a: _summary(combined, a, matched_ids) for a in ARMS}),
                   paired_fresh_vs_historical={a: _pair(combined, "historical_P02", a, ids)[0] for a in ARMS})
    return dict(scope="Exposed fixed90 actual R02/v12 development cohort; not formal500 and not independent confirmation",
                all_planned=whole, arms=whole["arms"], by_dataset=whole["by_dataset"],
                equal_weight_dataset_macro=whole["equal_weight_dataset_macro"],
                paired_vs_fresh_N128={a: whole["paired_predeclared"]["N128_to_" + a] for a in ARMS if a != "N128"},
                paired_predeclared=whole["paired_predeclared"], paired_by_dataset=whole["paired_by_dataset"],
                paired_equal_weight_dataset_macro=whole["paired_equal_weight_dataset_macro"],
                ordinary_present_sensitivity=ordinary, historical_reference_only=history,
                algorithm_failures_score_zero=True, technical_missing_score_zero=False,
                no_semantic_truth_claim_from_native_score=True, main_evaluation_remains_paused=True,
                cost_scope="Logical per-flow stages including exact shared receipts once per logical use; excludes retired technical attempts, unresolved usage, GPU expense and decomposer cost. Actual newly paid once/unknown liability remain separate in STATUS.json; never add logical and paid costs"), pairs


def score():
    from . import preparation as prep
    prep.check()  # Mandatory before even opening the new experiment's checkpoint.
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    data, tasks = prep.rows(prep.INPUTS), prep.rows(prep.ORDER)
    assert len(data) == 90 and len(tasks) == 360 and sum(bool(r["supporting_questions"]) for r in data) == 84
    references = {r["id"]: r for r in prep.rows(prep.HERE / "references90.local.jsonl")}
    assert set(references) == {r["id"] for r in data}
    planned = {r["task_id"]: r for r in tasks}
    assert len(planned) == 360 and {r["arm"] for r in tasks} == set(ARMS)
    dbpath = prep.OUT / "checkpoint.sqlite3"
    if not dbpath.is_file():
        raise RuntimeError("This experiment has no checkpoint to score")
    held_paths = sorted((prep.OUT / "held").glob("*.json"))
    held_hashes = {str(p): prep.sha(p) for p in held_paths}
    held = {r["task_id"]: r for r in (prep.read(p) for p in held_paths)}
    with closing(sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        identity = db.execute("SELECT value FROM data WHERE kind='meta' AND key='identity'").fetchone()
        assert identity is not None
        identity = json.loads(identity[0])
        for key, path in (("protocol_sha256", prep.HERE / "PROTOCOL.frozen.json"),
                          ("inputs_sha256", prep.INPUTS), ("order_sha256", prep.ORDER)):
            assert identity[key] == prep.sha(path), "Own-checkpoint identity mismatch"
        outcomes = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM data WHERE kind='outcome'")}
        in_flight = db.execute("SELECT count(*) FROM requests WHERE status='in_flight'").fetchone()[0]
    assert held_hashes == {str(p): prep.sha(p) for p in sorted((prep.OUT / "held").glob("*.json"))}, "Held records changed during snapshot"
    assert not set(outcomes).intersection(held) and set(outcomes) <= set(planned) and set(held) <= set(planned)
    scorers = {d: metric(d) for d in DATASETS}  # Exactly one factory construction per dataset, reused for historical references.
    details = []
    for task in tasks:
        value = outcomes.get(task["task_id"])
        if value is not None:
            assert all(value[k] == task[k] for k in ("task_id", "id", "dataset", "arm"))
            native = native_result(value, scorers[task["dataset"]], references[task["id"]])
            usage = stage_usage(value["stages"])
            status, prediction = value["status"], value.get("prediction")
        else:
            value = held.get(task["task_id"])
            native = dict(f1=None, em=None)
            usage = stage_usage(value["stages"]) if value is not None and "stages" in value else None
            status, prediction = ("technical_held" if value is not None else "unstarted_or_incomplete"), None
        details.append({**{k: task[k] for k in ("task_id", "id", "dataset", "arm")},
                        "status": status, "prediction": prediction, **native,
                        "native_full_correct": native["f1"] == 1 or native["em"] == 1,
                        "full_flow_logical_usage": usage,
                        "full_flow_logical_estimated_CNY": estimated_cny(usage) if usage is not None else None,
                        "logical_usage_complete_for_terminal_flow": status in QUALITY_TERMINAL})
    old = []
    for r in prep.rows(prep.HERE / "HISTORICAL_REFERENCES90.local.jsonl"):
        if not r["historical_quality_available"]:
            continue
        native = native_result(dict(status="success", prediction=r["final_prediction_reference_only"]),
                               scorers[r["dataset"]], references[r["id"]])
        # Old source logical tokens are not present in this frozen reference file.
        # Do not invent them or pass reference_CNY off as measured token usage.
        old.append(dict(id=r["id"], dataset=r["dataset"], arm="historical_P02", status="historical_reference",
                        **{k: float(native[k]) for k in ("f1", "em")},
                        native_full_correct=native["f1"] == 1 or native["em"] == 1,
                        full_flow_logical_usage=None,
                        full_flow_logical_estimated_CNY=r["reference_CNY"],
                        historical_tokens_unavailable=True, historical_reference_source=r.get("reference_source")))
    report, paired = aggregate_report(details, data, old)
    history = report["historical_reference_only"]
    history["historical_token_usage_available"] = False
    report.update(scorer_factory_calls_once_per_dataset=len(scorers), checkpoint_in_flight_at_snapshot=in_flight,
                  protocol_sha256=prep.sha(prep.HERE / "PROTOCOL.frozen.json"),
                  input_sha256=prep.sha(prep.INPUTS), order_sha256=prep.sha(prep.ORDER),
                  scoring_execution_is_read_only_to_checkpoint_and_ledger=True)
    prep.jsonl(prep.OUT / "DETAILS.jsonl", details)
    prep.jsonl(prep.OUT / "PAIRED.jsonl", paired)
    prep.atomic_json(prep.OUT / "REPORT.json", report)
    return report


def self_test():
    """Small meaningful aggregate/denominator checks; no real inputs or scores."""
    data = [dict(id=i, dataset=d, supporting_questions=[] if i in {"h2", "w1"} else ["q"])
            for i, d in (("h0", "hotpotqa"), ("h1", "hotpotqa"), ("h2", "hotpotqa"),
                         ("w0", "2wiki"), ("w1", "2wiki"), ("m0", "musique"))]
    details = []
    for item in data:
        for arm in ARMS:
            value = float(item["id"] not in {"h1", "h2"})
            if arm in {"F256", "R256"} and item["id"] == "h1": value = 1.0
            status = "algorithm_failure" if arm == "N256" and item["id"] == "h1" else "success"
            if arm == "R256" and item["id"] == "h0": value, status = None, "technical_held"
            usage = dict(prompt_tokens=10, completion_tokens=2, total_tokens=12)
            details.append(dict(id=item["id"], dataset=item["dataset"], arm=arm, status=status,
                                f1=value, em=value, native_full_correct=value == 1,
                                full_flow_logical_usage=usage, full_flow_logical_estimated_CNY=estimated_cny(usage)))
    old = [{**r, "arm": "historical_P02", "status": "historical_reference"}
           for r in details if r["arm"] == "N128" and r["id"] != "w1"]
    report, _ = aggregate_report(details, data, old)
    assert report["arms"]["R256"]["F1"] is None and report["arms"]["R256"]["missing"] == 1
    assert math.isclose(report["arms"]["R256"]["F1_available"], 0.8)
    assert report["arms"]["R256"]["F1_fixed_denominator_bounds"] == [4 / 6, 5 / 6]
    assert report["paired_predeclared"]["F256_to_R256"]["complete_pairs"] == 5
    assert report["historical_reference_only"]["common_all_four_fresh_and_historical"]["n"] == 4
    assert report["ordinary_present_sensitivity"]["planned_queries"] == 4
    assert math.isclose(report["arms"]["N128"]["F1"], 4 / 6)
    assert math.isclose(report["equal_weight_dataset_macro"]["N128"]["F1"], (1 / 3 + 1 + 1) / 3)
    without_old_tokens = [{**r, "full_flow_logical_usage": None} for r in old]
    tokenless, _ = aggregate_report(details, data, without_old_tokens)
    assert tokenless["historical_reference_only"]["coverage"]["mean_input_tokens"] is None
    assert tokenless["historical_reference_only"]["paired_fresh_vs_historical"]["N128"]["matched_input_token_pairs"] == 0
    def forbidden_scorer(*args): raise AssertionError("Algorithm terminal incorrectly invoked scorer")
    assert native_result(dict(status="algorithm_failure"), forbidden_scorer, {}) == dict(f1=0.0, em=0.0)
    assert stage_usage([dict(usage=dict(prompt_tokens=1, completion_tokens=2, total_tokens=3))])["total_tokens"] == 3
    for invalid in (dict(prompt_tokens=True, completion_tokens=1, total_tokens=2),
                    dict(prompt_tokens=-1, completion_tokens=1, total_tokens=0),
                    dict(prompt_tokens=1, completion_tokens=1, total_tokens=3)):
        try:
            stage_usage([dict(usage=invalid)])
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid usage was accepted")
    return dict(passed=True, checks=["technical_missing_not_zero", "algorithm_zero_without_scorer",
                "matched_pair_denominators", "partial_historical_intersection", "ordinary_sensitivity",
                "macro_distinct_from_query_mean", "strict_usage_integrity"],
                actual_predictions_or_scores_read=False, real_API_SSH_GPU_calls=0, checkpoint_or_ledger_writes=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    print(json.dumps(self_test() if parser.parse_args().self_test else score(), ensure_ascii=True))
