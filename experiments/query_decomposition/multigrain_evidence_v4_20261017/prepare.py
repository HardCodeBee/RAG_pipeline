"""Freeze fresh cases, carried regressions and targeted diagnostic cases."""

import hashlib
import json
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import rows, load, write, CAMPAIGN
from experiments.query_decomposition.multigrain_prompt_v2_20261017.prepare import S02, OLD_OUT
from . import prompts

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
V2 = ROOT / "experiments/query_decomposition/multigrain_prompt_v2_20261017"
V3 = ROOT / "experiments/query_decomposition/multigrain_prompt_v3_20261017"
PLANS = {
 "hotpotqa": {
  "5abe4fd95542991f6610613d": ["Who recorded Swingin' Down Yonder?"],
  "5ab67ce2554299710c8d1ead": ["How many species of Disporum occur in Asia?", "How many species of Boltonia occur in Asia?"],
  "5abd0296554299114383a1d1": ["At the site of which battle is Douglas D. Scott most notable for his archaeological work?"],
  "5abec2ca5542997719eab5b9": ["When did the Carondelet Canal begin operating?", "When did the Miami and Erie Canal begin operating?"],
 },
 "2wiki": {
  "0c55bf3e0bda11eba7f7acde48001122": ["Who is the father of Charles The Younger?"],
  "bb9081ff086611ebbd5eac1f6bf848b6": ["From which country does The Bounty (1984 Film) originate?", "From which country does Rock 'N' Roll Revue originate?"],
  "d6e9771908dd11ebbd9eac1f6bf848b6": ["Who directed Haiducii (Film)?", "When did the director of Haiducii (Film) die?", "Who directed My Wife'S Best Friend?", "When did the director of My Wife'S Best Friend die?"],
  "bc702d440baf11ebab90acde48001122": ["Who is the father of Ahmose-Henutemipet?"],
 },
 "musique": {
  "2hop__471509_136043": ["Which band released the album Nicole and Natalie?"],
  "2hop__805610_392181": ["Who directed A Stitch for Time?"],
  "3hop1__136083_87694_64412": ["After which saint is Schleswig Cathedral named?", "Where is the Basilica named after the same saint as Schleswig Cathedral and the head of the Catholic religion located?"],
  "4hop3__152056_698586_57596_54362": ["Who recorded Dunn Dunn?", "In which city did the artist of Dunn Dunn die?", "During which war did income tax start?"],
 },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE / "protocol.json").exists():
        raise RuntimeError("Already frozen")
    prior_cases = [r for name in ("multigrain_dual_retrieval_pilot_v1_20261017", V2.name, V3.name)
        for r in rows(ROOT / "experiments/query_decomposition" / name / "cases.frozen.jsonl")]
    used = {r['qid'] for r in prior_cases}
    cases = []
    for ds, plans in PLANS.items():
        questions = {r['qid']: r for r in rows(ROOT / f"experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{ds}/formal.questions.jsonl")}
        refs = {r['qid']: r for r in rows(ROOT / f"experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{ds}/formal.references.local.jsonl")}
        for qid, intermediate in plans.items():
            assert qid not in used
            question = questions[qid]['question']
            steps = [{"id": i+1, "request": s} for i, s in enumerate(intermediate + [question])]
            cases.append({"dataset": ds, "qid": qid, "question": question, "type": refs[qid]['type'],
                "historical_f1": None, "evaluation_group": "fresh", "qdmr": {"steps": steps, "goal": len(steps)}})
    cases += load(OUTPUT / V3.name / "REGRESSIONS_FOR_NEXT_ROUND.json")
    for qid in ("5a8eb63455429917b4a5bdb7", "bc7073a80baf11ebab90acde48001122"):
        cases.append({**next(r for r in prior_cases if r['qid'] == qid), "evaluation_group": "competitor_gap"})
    assert len(cases) == len({r['qid'] for r in cases}) == 15
    (HERE / "cases.frozen.jsonl").write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in cases), encoding='utf-8')
    old = load(S02 / "examples.json")['child']
    teaching = []
    for i in (0, 2, 3):
        r = old[i]
        question = {0: "What is the length of the river into which Pack Creek flows?", 2: "How many species does Callicoma contain other than Callicoma serratifolia?", 3: "What is the capital of Logar province?"}[i]
        teaching.append({"source_qid": r['source_qid'], "request": question, "dependencies": r['dependencies'], "context": r['context'], "answer": "[complete] " + r['answer'], "origin": "authored teaching output, not model output"})
    source = load(OLD_OUT / "cases/hotpotqa/dual_original_rewritten/5a89706055429951533612f9.json")
    scott = next(d for n in source['nodes'] for d in n['reader_packet']['docs'] if d['source'] == 'Scott Sandelin')
    teaching.insert(1, {"source_qid": "5a89706055429951533612f9", "request": "In what town and Minnesota county did Scott Sandelin grow up?", "dependencies": [],
        "context": '[Scott Sandelin]\n' + scott['text'], "answer": "[partial] Scott Sandelin grew up in Hibbing, Minnesota; the county cannot be determined from this passage.",
        "origin": "real source paragraph, authored teaching question and partial answer; not full runtime retrieval"})
    assert not {r['qid'] for r in cases} & {r['source_qid'] for r in teaching}
    write(HERE / "child_examples.json", teaching)
    conditions = {f"{mode}_{branch}": {"child": "old" if mode == "baseline" else "partial", "reread": mode == "reuse", "dual": branch == "dual"}
        for mode in ("baseline", "partial", "reuse") for branch in ("dual", "single")}
    files = [HERE/'prompts.py', HERE/'cases.frozen.jsonl', HERE/'child_examples.json', V2/'prompts.py', V2/'final_examples.json', S02/'prompts.py', S02/'examples.json']
    protocol = {"scope": "12 fresh cases + all previous dual losses + 2 targeted ReAct-gap diagnostics; manual query-only plans, development only",
        "conditions": conditions, "model": "qwen-plus-2025-12-01", "rewrite_tokens": 256, "child_tokens": 128, "final_tokens": 64,
        "global_cap_cny": 200, "round_subcap_cny": 5, "reuse_added_document_cap": 5,
        "frozen_files": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "passing_gate": "all15 complete; reuse_dual mean F1 > reuse_single; zero per-query losses; no gold-based runtime choice",
        "retrieval": "unchanged original-first rank-interleave dedup Top10; no extra searches from reread",
        "reread": "Only partial/none/unparsed nodes, once, after fixed first pass; retain all current10 docs, append <=5 unique already-retrieved docs; freeze all first-pass histories"}
    write(HERE / 'protocol.json', protocol)
    (HERE / 'CHILD_TEMPLATE.txt').write_text(prompts.child('{original_query}', '{current_question}', [], '{retrieved_passages}', teaching), encoding='utf-8')
    (HERE / 'DECISIONS.txt').write_text(
        '2026-10-17 round3 frozen before API submission\n'
        'Revert failed v3 final-name reinforcement. Keep v2 final prompt and v2 rewriter exactly. No new model or retraining.\n'
        'A: preserve supported partial factual information; one status marker controls reread and is stripped before propagation/helper. Status is self-report, not independently verified evidence.\n'
        'A also aligns four child shots to self-contained questions; replaces the entirely unknown count shot with a real Scott Sandelin excerpt teaching partial town knowledge. Baseline controls retain old shots. Do not claim the status marker alone causes any gain.\n'
        'B: one reread of existing materials only for incomplete factual nodes. Keep original reader docs; append up to5 unique docs from original query and other fixed nodes, rank-round-robin. No extra queries, no iteration, no corrected answer feeds another retry.\n'
        'Branches remain self-contained and never stop because of earlier missing answers. Reread outputs replace their own first answer, never chosen by gold or F1.\n'
        'Six conditions isolate old versus new child prompt and the added reread; partial/reuse share byte-identical first-pass trajectories. All final prompts v2.\n'
        '12 new IDs exclude preceding48 unique IDs, selected in deterministic hash order with query-type screening. Mu restricted to existing ReAct availability, not its score. Add prior1 loss and2 observed ReAct-gap cases separately. Manual plans assume semantic correctness, not an independently annotated oracle.\n'
        'Audit retrieved fact availability separately from answer utilization; gold-name occurrence or qrels hit is not sufficient evidence. Full final-question sufficiency differs from partial local-node evidence.\n', encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'frozen': len(cases), 'conditions': len(conditions)}))


if __name__ == '__main__':
    main()
