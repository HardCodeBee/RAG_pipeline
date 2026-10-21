"""Freeze twelve fresh cases per dataset, independent of answer outcomes."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import ROOT, OUTPUT, CAMPAIGN, rows, load, write

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
PILOT_SPEC = HERE.parent / "multigrain_dual_retrieval_pilot_v1_20261017"
PILOT_OUT = OUTPUT / PILOT_SPEC.name
INPUTS = ROOT / "experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen"
SEED = 20261017


def main():
    destination = HERE / "new36.questions.frozen.jsonl"
    if destination.exists():
        print("Preserve existing selection", len(rows(destination)))
        return
    old = rows(PILOT_SPEC / "cases.frozen.jsonl")
    excluded = {dataset: {r["qid"] for r in old if r["dataset"] == dataset} for dataset in ("hotpotqa", "2wiki", "musique")}
    excluded["musique"].update(load(PILOT_OUT / "MUSIQUE_SCREENING_FROZEN.json")["ids"])
    quotas = {"hotpotqa": {"bridge": 8, "comparison": 4}, "2wiki": {"bridge_comparison": 3, "compositional": 3, "inference": 3, "comparison": 3}, "musique": {"2-hop": 4, "3-hop": 4, "4-hop": 4}}
    selected = []
    for dataset in quotas:
        questions = rows(INPUTS / dataset / "formal.questions.jsonl")
        # Access only question-type metadata. Answers/supports never enter this
        # selection artifact or the reference-plan drafting inputs.
        types = {r["qid"]: r["type"] for r in rows(INPUTS / dataset / "formal.references.local.jsonl")}
        for kind, count in quotas[dataset].items():
            pool = sorted([q for q in questions if q["qid"] not in excluded[dataset] and types[q["qid"]] == kind], key=lambda q: q["qid"])
            seed = int(hashlib.sha256(f"{SEED}:{dataset}:{kind}".encode()).hexdigest()[:16], 16)
            chosen = random.Random(seed).sample(pool, count)
            selected.extend({"dataset": dataset, "qid": q["qid"], "question": q["question"], "type": kind, "sample_group": "fresh36_type_stratified_random"} for q in chosen)
    assert len(selected) == 36
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
    write(HERE / "SELECTION_FROZEN.json", {"target_total": 60, "old_diagnostic24_preserved": True, "fresh_cases": 36, "per_dataset_new": dict(Counter(r["dataset"] for r in selected)), "seed": SEED, "type_quotas_new": quotas, "exclude": {d: sorted(v) for d, v in excluded.items()}, "selection_uses_answers_or_method_outcomes": False, "reporting": "fresh36 primary expansion result and total60 mixed diagnostic result separately; type-stratified means are not population estimates"})
    protocol = load(PILOT_SPEC / "protocol.json")
    protocol.update({"status": "user_authorized_expansion_to_total60", "pilot_subcap_cny": 5, "generation_changes_from_pilot": [], "selection_change": "12 fresh type-stratified random cases per dataset; retain previous8 per dataset", "reference_plan_method": "current Codex drafts from frozen original questions only, reviews and freezes before generation", "fix_known_rewrite_issues": False})
    write(HERE / "protocol.json", protocol)
    (HERE / "DECISIONS.txt").write_text(
        "2026-10-17 | Expand the unchanged multigrain pilot to60 total: old24 plus36 fresh.\n"
        "User specified20 per dataset. Preserve original8 and add12 per dataset; report fresh36 separately from outcome-selected old24.\n"
        "New selection is fixed-seed stratified by query type and independent of S02 success/failure. Mu original24 screening candidates excluded.\n"
        "Same prompts, full-sentence propagation, self-contained query-only plans, goal execution, optional rewrite, interleave-original-first-deduplicate-top10, final originalTop10+helper. No fixes to known rewriting problems in this evaluation.\n"
        "Reference plans will be drafted by current Codex from question-only selection and frozen before candidate API calls. They are assumed-correct references, not human oracle labels. No annotation API or new model training.\n"
        "Three groups: frozen S02 with actual4B output, same-reference-plan single and dual. Prior Top10/ReAct scores retained when available, with missingness explicit. Existing exact completed outputs reused; no historical result changed.\n"
        "Fresh Mu S02 will run unchanged in an isolated namespace. Pilot whole-budget accounting retained, global cap200 CNY, protected cutoff195 CNY, expansion subcap5 CNY.\n",
        encoding="utf-8")
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
