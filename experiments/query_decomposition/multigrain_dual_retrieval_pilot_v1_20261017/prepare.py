"""Freeze targeted old-success/old-failure cases and query-only reference plans."""

from collections import Counter
import json
from pathlib import Path

from .screen_musique import HERE, OUT, CAMPAIGN, ROOT, rows, load, write


def main():
    path = HERE / "cases.frozen.jsonl"
    if path.exists():
        print("Existing cases preserved:", len(rows(path)))
        return
    reference = load(HERE / "reference_intermediates.json")
    cases = []
    for dataset in ("hotpotqa", "2wiki", "musique"):
        questions = {r["qid"]: r["question"] for r in rows(ROOT / f"experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{dataset}/formal.questions.jsonl")}
        if dataset == "musique":
            pool = load(OUT / "MUSIQUE_SCREENING_RESULTS.json")["results"]
            # Prefer coverage of different hop counts; ties keep the frozen
            # screening order. No candidate-method outcomes enter selection.
            chosen = []
            for score in (0, 1):
                eligible = [r for r in pool if r["f1"] == score]
                selected = []
                for kind in ("2-hop", "3-hop", "4-hop"):
                    candidates = [r for r in eligible if r["type"] == kind]
                    if candidates:
                        selected.append(candidates[0])
                for record in eligible:
                    if record not in selected:
                        selected.append(record)
                    if len(selected) >= 4:
                        break
                assert len(selected) >= 4, (score, len(selected))
                chosen.extend(selected[:4])
            plans = reference["musique_screening_candidates"]
        else:
            pool = rows(CAMPAIGN / dataset / "formal/m00_s02_child4_faithful/frozen_s02_child4/scores.jsonl")
            plans = reference[dataset]
            by_id = {r["qid"]: r for r in pool}
            chosen = [by_id[qid] for qid in plans]
        assert Counter("failure" if r["f1"] == 0 else "success" for r in chosen) == {"failure": 4, "success": 4}
        for record in chosen:
            qid = record["qid"]
            query = questions[qid]
            requests = plans[qid] + [query]
            assert all("#" not in request for request in requests)
            cases.append({
                "dataset": dataset,
                "qid": qid,
                "question": query,
                "type": record["type"],
                "historical_group": "failure" if record["f1"] == 0 else "success",
                "historical_f1": record["f1"],
                "qdmr": {"steps": [{"id": i, "request": q} for i, q in enumerate(requests, 1)], "goal": len(requests)},
                "plan_provenance": "Codex-authored from original question only; no factual intermediate answers inserted; not human gold or proof of correctness",
            })
    assert len(cases) == 24
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cases), encoding="utf-8")
    write(HERE / "SELECTION_FROZEN.json", {
        "cases": len(cases),
        "per_dataset": dict(Counter(r["dataset"] for r in cases)),
        "selection": "targeted diagnostic selection conditional on historical F1 endpoints and reasoning types; not a random population estimate",
        "success_definition": "historical final native Answer F1 equals 1",
        "failure_definition": "historical final native Answer F1 equals 0; infrastructure missingness excluded",
        "plan_semantics": "self-contained questions at multiple levels of granularity, no hard #N references; last step preserves original query",
        "gold_answers_or_supports_in_plan_inputs": False,
        "candidate_method_outputs_used_for_selection": False,
        "training_revision_resumed": False,
    })
    print(json.dumps({"prepared_cases": len(cases), "by_dataset": dict(Counter(r["dataset"] for r in cases))}))


if __name__ == "__main__":
    main()
