"""Explicit read-only native scoring; never invoked by run/engineering/status."""
from contextlib import closing
import json
from .preparation import HERE, OUT, INPUTS, ORDER, HISTORY, base, read, rows, jsonl, check, atomic_json, templates


def score():
    check()
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    scorers = {dataset: metric(dataset) for dataset in base.DATASETS}
    references = {r["id"]: r for r in rows(HERE / "references90.local.jsonl")}
    with closing(base.ro(OUT / "checkpoint.sqlite3")) as db:
        outcomes = [json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='outcome'")]
    details = []
    for outcome in outcomes:
        native = scorers[outcome["dataset"]](outcome["prediction"], references[outcome["id"]]) if outcome["status"] in ("success", "success_truncated") else dict(f1=0., em=0.)
        usage = {key: sum(s["usage"][key] for s in outcome["stages"]) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        details.append({**{key: outcome[key] for key in ("task_id", "id", "dataset", "arm", "status", "prediction")}, **native,
                        "native_full_correct": native["f1"] == 1 or native["em"] == 1,
                        "full_flow_logical_usage": usage, "full_flow_logical_estimated_CNY": base.cost(usage)})
    index = {(r["id"], r["arm"]): r for r in details}
    def summary(arm, ids):
        found = [index[i, arm] for i in ids if (i, arm) in index]; planned = len(ids); count = len(found)
        sums = {key: sum(r[key] for r in found) for key in ("f1", "em")}
        return dict(planned=planned, quality_terminal=count, missing=planned-count,
            F1=sums["f1"]/planned if count == planned and planned else None,
            EM=sums["em"]/planned if count == planned and planned else None,
            F1_available=sums["f1"]/count if count else None, EM_available=sums["em"]/count if count else None,
            F1_fixed_denominator_bounds=[sums["f1"]/planned, (sums["f1"]+planned-count)/planned] if planned else None,
            mean_input_tokens=sum(r["full_flow_logical_usage"]["prompt_tokens"] for r in found)/count if count else None,
            mean_output_tokens=sum(r["full_flow_logical_usage"]["completion_tokens"] for r in found)/count if count else None,
            mean_logical_estimated_CNY=sum(r["full_flow_logical_estimated_CNY"] for r in found)/count if count else None)
    ids = set(references)
    pairs = [dict(id=i, dataset=references[i]["dataset"], candidate_arm=arm,
        reference_arm="C0", delta_F1=index[i, arm]["f1"]-index[i, "C0"]["f1"],
        delta_EM=index[i, arm]["em"]-index[i, "C0"]["em"],
        delta_input_tokens=index[i, arm]["full_flow_logical_usage"]["prompt_tokens"]-index[i, "C0"]["full_flow_logical_usage"]["prompt_tokens"],
        delta_output_tokens=index[i, arm]["full_flow_logical_usage"]["completion_tokens"]-index[i, "C0"]["full_flow_logical_usage"]["completion_tokens"])
        for arm in templates.ARMS if arm != "C0" for i in sorted(ids) if (i, arm) in index and (i, "C0") in index]
    def paired(arm, selected):
        found = [r for r in pairs if r["candidate_arm"] == arm and r["id"] in selected]; n = len(found)
        return dict(planned=len(selected), complete_pairs=n, missing_pairs=len(selected)-n,
            delta_F1=sum(r["delta_F1"] for r in found)/n if n else None,
            delta_EM=sum(r["delta_EM"] for r in found)/n if n else None,
            improved=sum(r["delta_F1"] > 0 for r in found), tied=sum(r["delta_F1"] == 0 for r in found), harmed=sum(r["delta_F1"] < 0 for r in found),
            scope="Available matched pairs; not a complete90 claim when missing_pairs>0")
    old = [r for r in rows(HISTORY) if r["historical_quality_available"]]
    old_values = [scorers[r["dataset"]](r["final_prediction_reference_only"], references[r["id"]]) for r in old]
    old_ids = {r["id"] for r in old}
    value = dict(scope="Same exposed actualR02 fixed90; separate ordinary prompt factors; technicalholds missing, algorithm failures zero; notformal500",
        arms={arm: summary(arm, ids) for arm in templates.ARMS},
        by_dataset={dataset: {arm: summary(arm, {i for i in ids if references[i]["dataset"] == dataset}) for arm in templates.ARMS} for dataset in base.DATASETS},
        paired_vs_fresh_C0={arm: paired(arm, ids) for arm in templates.ARMS if arm != "C0"},
        paired_by_dataset={dataset: {arm: paired(arm, {i for i in ids if references[i]["dataset"] == dataset}) for arm in templates.ARMS if arm != "C0"} for dataset in base.DATASETS},
        historical_reference_only=dict(available=len(old), missing=90-len(old),
            F1_available=sum(r["f1"] for r in old_values)/len(old_values) if old_values else None,
            EM_available=sum(r["em"] for r in old_values)/len(old_values) if old_values else None,
            same_historical_ids_fresh={arm: summary(arm, old_ids) for arm in templates.ARMS}),
        scorer_factory_calls_once_per_dataset=3, cost_scope="Logical per-flow usage; actual paid fees and unknown liabilities are in STATUS.json; do not add them",
        no_semantic_truth_claim_from_native_score=True, main_evaluation_remains_paused=True)
    jsonl(OUT / "DETAILS.jsonl", details); jsonl(OUT / "PAIRED.jsonl", pairs); atomic_json(OUT / "REPORT.json", value)
    return value


if __name__ == "__main__":
    print(json.dumps(score(), ensure_ascii=True))
