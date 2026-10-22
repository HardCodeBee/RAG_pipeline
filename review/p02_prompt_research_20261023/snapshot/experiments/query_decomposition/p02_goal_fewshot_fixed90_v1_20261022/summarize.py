"""Offline same-case native recount and human-readable report; no model access."""
from collections import Counter
import json
from . import run


def main():
    report = run.report()
    inputs = {r["id"]: r for r in run.rows(run.HERE / "INPUTS90.frozen.jsonl")}
    history = {r["id"]: r for r in run.rows(run.HERE / "HISTORY90.local.frozen.jsonl")}
    refs = {r["id"]: r for r in run.rows(run.pilot.COHORT / "references90.local.frozen.jsonl")}
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    scorers = {ds: metric(ds) for ds in run.pilot.DATASETS}
    details = run.rows(run.OUT / "DETAILS.scored.local.jsonl")
    indexed = {(r["arm"], r["id"]): r for r in details}
    common = report["common_ids"]
    table, changes, missing = [], [], []
    for ds in ("all",) + run.pilot.DATASETS:
        ids = [i for i in common if ds == "all" or inputs[i]["dataset"] == ds]
        if not ids:
            continue
        values = {}
        for arm in ("native_historical_P02",) + run.templates.ARMS:
            scores = [scorers[inputs[i]["dataset"]](history[i]["prediction"], refs[i])
                      if arm == "native_historical_P02" else indexed[(arm, i)]["native_metrics"] for i in ids]
            values[arm] = {"F1": sum(s["f1"] for s in scores) / len(ids),
                           "EM": sum(s["em"] for s in scores) / len(ids)}
        table.append({"dataset": ds, "n": len(ids), "scores": values})
    for ident, record in inputs.items():
        for arm in run.templates.ARMS:
            r = indexed.get((arm, ident))
            if not r or r["native_metrics"]["f1"] is None:
                missing.append({"id": ident, "dataset": record["dataset"], "arm": arm,
                                "status": r["status"] if r else "unstarted"})
        if ident not in common:
            continue
        predictions = {arm: indexed[(arm, ident)]["prediction"] for arm in run.templates.ARMS}
        if len(set(predictions.values())) > 1:
            changes.append({"id": ident, "dataset": record["dataset"], "question": record["question"],
                "reference": refs[ident]["answer"], "ordinary_nodes": record["ordinary_nodes"],
                "variants": {arm: {"prediction": indexed[(arm, ident)]["prediction"],
                    "goal_text": indexed[(arm, ident)]["goal_text"],
                    "native_metrics": indexed[(arm, ident)]["native_metrics"]} for arm in run.templates.ARMS}})
    # A returned length-ended goal is a scored failure, but has no final usage.
    # Compare completed goal+final costs on the same successful cases in all arms.
    cost_ids = [i for i in common if all(indexed[(a, i)]["status"] == "success" for a in run.templates.ARMS)]
    costs = {}
    for arm in run.templates.ARMS:
        selected = [indexed[(arm, i)] for i in cost_ids]
        if not selected:
            continue
        costs[arm] = {"n": len(selected),
            "mean_input_tokens": sum(r["input_tokens"] for r in selected) / len(selected),
            "mean_output_tokens": sum(r["output_tokens"] for r in selected) / len(selected),
            "mean_goal_final_CNY": sum(r["goal_final_CNY"] for r in selected) / len(selected),
            "mean_logical_fullflow_CNY": sum(r["logical_fullflow_CNY"] for r in selected) / len(selected)}
    dataset_rows = [r for r in table if r["dataset"] != "all"]
    macro = {arm: {key: sum(r["scores"][arm][key] for r in dataset_rows) / len(dataset_rows)
                   for key in ("F1", "EM")}
             for arm in ("native_historical_P02",) + run.templates.ARMS} if len(dataset_rows) == 3 else None
    native_pairs = {}
    for arm in run.templates.ARMS:
        counts, deltas = Counter(), []
        for ident in common:
            baseline = scorers[inputs[ident]["dataset"]](history[ident]["prediction"], refs[ident])
            delta = indexed[(arm, ident)]["native_metrics"]["f1"] - baseline["f1"]
            deltas.append(delta)
            counts["improved" if delta > 1e-12 else "harmed" if delta < -1e-12 else "tied"] += 1
        native_pairs[arm] = {"n": len(deltas), **dict(counts),
                             "mean_F1_delta": sum(deltas) / len(deltas) if deltas else None}
    algorithms = [r for r in details if r["status"] == "algorithm_failure"]
    summary = {"same_three_arm_cases": len(common), "table": table, "costs_same_cases": costs,
        "same_three_arm_successful_cost_cases": len(cost_ids),
        "algorithm_failures_included_as_zero": [{k:r[k] for k in ("id", "dataset", "arm", "status")} for r in algorithms],
        "dataset_equal_mean": macro, "paired_against_historical_native_P02": native_pairs,
        "fresh_actual_CNY": report["fresh_actual_CNY"], "unknown_reserved_CNY": report["unknown_reserved_CNY"],
        "old42_unknown_reserved_CNY": report["external_old42_unknown_reserved_CNY"],
        "missing_conditions": missing, "different_final_text_cases": len(changes),
        "paired_F1": report["paired_F1"], "scope": "Frozen-prefix exposed fixed90; use matched cases and disclose technical holds",
        "native_historical_reference_protocol_differs": True}
    run.save(run.OUT / "SUMMARY.json", summary)
    run.jsonl(run.OUT / "CHANGED_FINAL_CASES.local.jsonl", changes)
    run.save(run.REVIEW / "RESULTS.json", summary)
    lines = ["# Fixed90 goal few-shot experiment", "", "Frozen-prefix exposed development comparison; main evaluation remains paused.", "",
             f"Planned:90 cases/270 conditions. All270 conditions have terminal records. Three-arm scorable paired cases:{len(common)}. Unknown held conditions:{len(missing)} across4 questions.",
             f"Returned algorithm failures:{len(algorithms)}; retained as zero-score outcomes in the paired denominator. Unknown requests remain unscored.", "",
             "| Dataset | Matched n | Historical native P02 F1/EM | G2 F1/EM | I F1/EM | F F1/EM |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in table:
        fields = [row["dataset"], str(row["n"])]
        for arm in ("native_historical_P02",) + run.templates.ARMS:
            s = row["scores"][arm]
            fields.append(f"{s['F1']*100:.2f}/{s['EM']*100:.2f}")
        lines.append("| " + " | ".join(fields) + " |")
    if macro:
        fields = ["Dataset equal mean", "same matched subsets"]
        for arm in ("native_historical_P02",) + run.templates.ARMS:
            fields.append(f"{macro[arm]['F1']*100:.2f}/{macro[arm]['EM']*100:.2f}")
        lines.append("| " + " | ".join(fields) + " |")
    lines += ["", "Historical native P02 is a reference with its original goal/final protocol; G2/I/F use the common512/64 synthesis interface.",
        "The matched table never compares different denominators. Missing unknown requests are not zero or automatically reissued.", "",
        f"New estimated expense from returned usage:CNY{report['fresh_actual_CNY']:.6f}; current preserved request reservations:CNY{report['unknown_reserved_CNY']:.6f}; preserved old42 unknown reserve:CNY{report['external_old42_unknown_reserved_CNY']:.6f}. Reservations are not confirmed spend; current reservations conservatively include connect failures, detailed in OFFLINE_AUDIT.json.",
        "", f"Costs below use the same{len(cost_ids)} successful cases in all three arms. They are logical goal+final usage including exact cached work, rather than newly paid complete-RAG costs. The length-ended goal is excluded only from this completed-flow cost comparison; its returned usage is included in new actual returned-usage expense above.", "",
        "| Arm | Input tokens/query | Output tokens/query | Goal+final CNY/query | Logical full-flow CNY/query |",
        "|---|---:|---:|---:|---:|"]
    for arm, value in costs.items():
        lines.append(f"| {arm} | {value['mean_input_tokens']:.1f} | {value['mean_output_tokens']:.1f} | {value['mean_goal_final_CNY']:.6f} | {value['mean_logical_fullflow_CNY']:.6f} |")
    lines += ["", "No fresh retrieval, decomposer, GPU, external judge, timing campaign, method promotion or publication was performed.",
              "These are development results on exposed examples; no independent generalization or pure reasoning-effect claim is justified."]
    (run.OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
