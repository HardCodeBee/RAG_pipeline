"""Offline native F1 and document-coverage diagnostics; no generation calls."""

from collections import Counter, defaultdict
import hashlib
import json
import sqlite3

from .screen_musique import HERE, OUT, ROOT, OUTPUT, CAMPAIGN, rows, load, write
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric


def main():
    cases = rows(HERE / "cases.frozen.jsonl")
    protocol = load(HERE / "protocol.json")
    conditions = ["single_original", "dual_original_rewritten"] if protocol["matched_single_control"] else ["dual_original_rewritten"]
    references = {}
    scorers = {}
    historical = {}
    frozen_competitors = rows(HERE / "historical_competitor_references.jsonl")
    for dataset in ("hotpotqa", "2wiki", "musique"):
        references[dataset] = {r["qid"]: r for r in rows(ROOT / f"experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{dataset}/formal.references.local.jsonl")}
        scorers[dataset] = metric(dataset)
        historical[dataset] = {}
        for label in ("top10_v1", "react"):
            historical[dataset][label] = {r["qid"]: r for r in frozen_competitors if r["dataset"] == dataset and r["method"] == label}
        if dataset == "musique":
            historical[dataset]["frozen_s02"] = {r["qid"]: r for r in load(OUT / "MUSIQUE_SCREENING_RESULTS.json")["results"]}
        else:
            source = CAMPAIGN / dataset / "formal/m00_s02_child4_faithful/frozen_s02_child4/scores.jsonl"
            historical[dataset]["frozen_s02"] = {r["qid"]: r for r in rows(source)}
    details = []
    for case in cases:
        reference = references[case["dataset"]][case["qid"]]
        gold = set(reference["positive_doc_ids"])
        row = {k: case[k] for k in ("dataset", "qid", "question", "type", "historical_group", "historical_f1")}
        row["gold_answer"] = reference["answer"]
        row["historical_references"] = {
            label: {k: pool.get(case["qid"], {}).get(k) for k in ("prediction", "f1", "em", "status")}
            for label, pool in historical[case["dataset"]].items()
        }
        row["conditions"] = {}
        for condition in conditions:
            path = OUT / "cases" / case["dataset"] / condition / (case["qid"] + ".json")
            if not path.exists():
                row["conditions"][condition] = {"status": "unstarted_or_incomplete", "f1": None}
                continue
            result = load(path)
            with sqlite3.connect((OUT / "api" / case["dataset"] / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
                request = db.execute("SELECT result FROM requests WHERE id=?", (result["request_id"],)).fetchone()
                final_usage = json.loads(request[0]).get("usage", {}) if request else {}
            node_details = []
            original_ids = set(result["original_top10_doc_ids"])
            all_shown = set(result["original_top10_doc_ids"])
            for node in result["nodes"]:
                original = {d["doc_id"] for d in node["original_retrieval"]["docs"]}
                rewritten = {d["doc_id"] for d in node["rewritten_retrieval"]["docs"]} if node["rewritten_retrieval"] else set()
                shown = {d["doc_id"] for d in node["reader_packet"]["docs"]}
                all_shown.update(shown)
                node_details.append({
                    "step_id": node["step_id"], "question": node["request"], "rewritten_query": node["rewritten_query"], "answer": node["answer"],
                    "rewrite_changed": node["rewritten_retrieval"] is not None,
                    "original_gold_docs": sorted(original & gold),
                    "rewrite_added_gold_docs": sorted((rewritten - original) & gold),
                    "added_gold_docs_actually_shown": sorted((shown - original) & gold),
                    "original_gold_docs_displaced": sorted((original - shown) & gold),
                    "sentence_format_repair": node["sentence_format_request_id"] is not None,
                    "rewrite_format_rejected": node["rewrite_rejection"],
                    "original_gold_document_count": len(original & gold),
                    "reader_gold_document_count": len(shown & gold),
                    "reader_document_count": len(shown),
                    "added_documents_actually_shown": sorted(shown - original),
                })
            score = scorers[case["dataset"]](result["prediction"], reference)
            row["conditions"][condition] = {"status": result["status"], "prediction": result["prediction"], **score,
                "all_gold_docs_shown_to_some_reader": gold.issubset(all_shown), "gold_document_recall": len(gold & all_shown) / len(gold) if gold else None,
                "final_prompt_tokens": final_usage.get("prompt_tokens"), "node_diagnostics": node_details,
                "original_top10_gold_document_recall": len(original_ids & gold) / len(gold) if gold else None,
                "extra_gold_documents_shown_beyond_original_top10": sorted((all_shown - original_ids) & gold),
                "original_top10_preserved_in_final_prompt": all(
                    result["final_context"].count(doc["block"]) >= 1
                    for doc in result["nodes"][-1]["original_retrieval"]["docs"]
                ),
            }
        details.append(row)
    def summarize(records):
        summary = {"planned": len(records), "conditions": {}, "historical_references": {}}
        for label in ("frozen_s02", "top10_v1", "react"):
            valid = [r["historical_references"][label] for r in records if r["historical_references"][label]["f1"] is not None]
            summary["historical_references"][label] = {"available": len(valid), "missing": len(records) - len(valid), "mean_f1_available": sum(r["f1"] for r in valid) / len(valid) if valid else None}
        for condition in conditions:
            valid = [r["conditions"][condition] for r in records if r["conditions"][condition]["f1"] is not None]
            nodes = [n for r in valid for n in r["node_diagnostics"]]
            summary["conditions"][condition] = {
                "available": len(valid), "missing": len(records) - len(valid),
                "mean_f1_available": sum(r["f1"] for r in valid) / len(valid) if valid else None,
                "gain_vs_historical": sum(r["conditions"][condition]["f1"] > r["historical_f1"] for r in records if r["conditions"][condition]["f1"] is not None),
                "loss_vs_historical": sum(r["conditions"][condition]["f1"] < r["historical_f1"] for r in records if r["conditions"][condition]["f1"] is not None),
                "rewrites": sum(n["rewrite_changed"] for n in nodes),
                "steps_with_new_gold_docs_shown": sum(bool(n["added_gold_docs_actually_shown"]) for n in nodes),
                "steps_with_original_gold_docs_displaced": sum(bool(n["original_gold_docs_displaced"]) for n in nodes),
                "sentence_format_calls": sum(n["sentence_format_repair"] for n in nodes),
                "rejected_rewrite_formats": sum(bool(n["rewrite_format_rejected"]) for n in nodes),
                "questions_with_all_gold_docs_shown_to_some_reader": sum(r["all_gold_docs_shown_to_some_reader"] for r in valid),
                "questions_with_gold_docs_beyond_original_top10": sum(bool(r["extra_gold_documents_shown_beyond_original_top10"]) for r in valid),
                "all_original_top10_preserved_in_final_prompt": all(r["original_top10_preserved_in_final_prompt"] for r in valid),
                "mean_final_prompt_tokens": sum(r.get("final_prompt_tokens") or 0 for r in valid) / len(valid) if valid else None,
            }
        if len(conditions) == 2:
            pairs = [r for r in records if all(r["conditions"][c]["f1"] is not None for c in conditions)]
            differences = [r["conditions"][conditions[1]]["f1"] - r["conditions"][conditions[0]]["f1"] for r in pairs]
            summary["paired_dual_minus_single"] = {"pairs": len(pairs), "mean_difference": sum(differences) / len(differences) if differences else None,
                "gains": sum(d > 0 for d in differences), "ties": sum(d == 0 for d in differences), "losses": sum(d < 0 for d in differences)}
        return summary
    with sqlite3.connect((OUTPUT / "budget.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        actual, reserved = db.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace LIKE ?", (str(OUT.relative_to(OUTPUT)) + "%",)).fetchone()
    report = {"scope": "24 purposively selected diagnostic cases; not a generalization or significance claim", "protocol": protocol,
        "overall": summarize(details), "by_dataset": {ds: summarize([r for r in details if r["dataset"] == ds]) for ds in references},
        "by_historical_group": {group: summarize([r for r in details if r["historical_group"] == group]) for group in ("success", "failure")},
        "api_usage_list_price_cny": actual, "unresolved_liability_cny": reserved,
        "reference_plan_limit": "fixed query-only Codex-authored plans, ideal-correctness assumption rather than human-verified oracle", "gold_in_generation": False}
    api_stats = {}
    for dataset in references:
        checkpoint = OUT / "api" / dataset / "checkpoint.sqlite3"
        if not checkpoint.exists():
            continue
        with sqlite3.connect(checkpoint.resolve().as_uri() + "?mode=ro", uri=True) as db:
            statuses = dict(db.execute("SELECT status,COUNT(*) FROM requests GROUP BY status"))
            returned = [json.loads(raw) for raw, in db.execute("SELECT result FROM requests WHERE status='success'")]
            api_stats[dataset] = {"unique_request_statuses": statuses,
                "attempt_statuses": dict(db.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status")),
                "returned_prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in returned),
                "returned_completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in returned),
                "sum_returned_request_latency_seconds_not_wall_time": sum(r.get("latency_seconds", 0) for r in returned)}
    report["pilot_api_execution"] = api_stats
    frozen = load(OUT / "EXECUTION_FROZEN.json")
    report["frozen_hash_checks"] = {
        "protocol": hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest() == frozen["protocol_sha256"],
        "cases": hashlib.sha256((HERE / "cases.frozen.jsonl").read_bytes()).hexdigest() == frozen["cases_sha256"],
        "runner": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest() == frozen["code_sha256"],
        "child_and_final_prompts": hashlib.sha256((ROOT / "experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/prompts.py").read_bytes()).hexdigest() == frozen["child_and_final_prompts_sha256"],
    }
    write(OUT / "RESULTS.json", report)
    (OUT / "CASE_RESULTS.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
