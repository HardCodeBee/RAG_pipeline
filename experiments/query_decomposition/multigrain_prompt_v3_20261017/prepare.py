"""Freeze a new sample plus all previous losses; modify final naming only."""

import hashlib
import json
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import rows, load, write, CAMPAIGN
from experiments.query_decomposition.multigrain_prompt_v2_20261017.prepare import S02, OLD_OUT
from . import prompts

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / "multigrain_prompt_v3_20261017"
PREVIOUS = ROOT / "experiments/query_decomposition/multigrain_prompt_v2_20261017"
PREVIOUS_OUT = OUTPUT / "multigrain_prompt_v2_20261017"

PLANS = {
    "hotpotqa": {
        "5ade630e55429975fa854eb7": ["Which genus contains lions?"],
        "5a90ab995542990a98493683": ["Does the genus Osteomeles extend to Pacific Ocean islands?", "Does the genus Medinilla extend to Pacific Ocean islands?"],
        "5a77d31c55429967ab1052c0": ["Which pianist recorded Live at Shelly's Manne-Hole?"],
        "5ae383e15542992f92d822fb": ["In what year was The Apple Dumpling Gang released?", "In what year was Heavyweights released?"],
    },
    "2wiki": {
        "1bb538ec0bda11eba7f7acde48001122": ["Who directed the film Chronicles Of A Dive Bomber?"],
        "bade33cb08d211ebbd96ac1f6bf848b6": ["Who directed the film L'Arma?", "On what date did the director of L'Arma die?", "Who directed the film The Mill Of Good Luck?", "On what date did the director of The Mill Of Good Luck die?"],
        "c244022c0baf11ebab90acde48001122": ["Who is the mother of Sophie Of France (1734–1782)?"],
        "73b99793086f11ebbd62ac1f6bf848b6": ["From which country does the film The Shadow Of Evil originate?", "From which country does the film The Lone Wolf Takes A Chance originate?"],
    },
    "musique": {
        "4hop1__151650_5274_458768_33633": ["Which record label signed Desde El Principio?", "Which record label has as its only larger label the label that signed Desde El Principio?", "In which region is the headquarters of the record label whose only larger label is the one that signed Desde El Principio?"],
        "2hop__846844_47134": ["Which school owns Goss Stadium at Coleman Field?"],
        "2hop__341176_711757": ["In which county is Gmina Stężyca located?"],
        "3hop1__801799_547811_41132": ["Who composed La fida ninfa?", "In which city was the composer of La fida ninfa born?"],
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE / "protocol.json").exists():
        raise RuntimeError("Round already frozen")
    previous_result = load(PREVIOUS_OUT / "RESULTS.json")
    assert previous_result["complete"] and not previous_result["passing_gate_met"]
    used = {r["qid"] for source in (OLD_OUT / "CASE_RESULTS.jsonl", PREVIOUS_OUT / "CASE_RESULTS.jsonl") for r in rows(source)}
    cases = []
    inputs = ROOT / "experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen"
    for dataset, plans in PLANS.items():
        questions = {r["qid"]: r for r in rows(inputs / dataset / "formal.questions.jsonl")}
        refs = {r["qid"]: r for r in rows(inputs / dataset / "formal.references.local.jsonl")}
        scores = load(OLD_OUT / "MUSIQUE_SCREENING_RESULTS.json")["results"] if dataset == "musique" else rows(CAMPAIGN / dataset / "formal/m00_s02_child4_faithful/frozen_s02_child4/scores.jsonl")
        scores = {r["qid"]: r for r in scores}
        for qid, intermediates in plans.items():
            assert qid not in used
            q = questions[qid]["question"]
            steps = [{"id": i + 1, "request": request} for i, request in enumerate(intermediates + [q])]
            f1 = scores[qid]["f1"]
            cases.append({"dataset": dataset, "qid": qid, "question": q, "type": refs[qid]["type"],
                "historical_f1": f1, "historical_group": "success" if f1 == 1 else "failure_or_partial", "evaluation_group": "fresh",
                "qdmr": {"steps": steps, "goal": len(steps)}})
    cases.extend(load(PREVIOUS_OUT / "REGRESSIONS_FOR_NEXT_ROUND.json"))
    assert len({r["qid"] for r in cases}) == len(cases) == 13
    (HERE / "cases.frozen.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cases), encoding="utf-8")
    examples = load(PREVIOUS / "final_examples.json")
    first = examples[0]
    passage = first["context"].split("\n\n[Chunk 2")[0]
    question = "Which river does Pack Creek flow into after passing through Spanish Valley?"
    history = [{"step_id": 1, "request": question, "answer": "Pack Creek flows into the Colorado."}]
    examples[0] = {"source_qid": first["source_qid"], "question": question,
        "context": passage + "\n\n" + prompts.helper(history), "answer": "Colorado River", "qas": history,
        "fixture_origin": "same source excerpt as v2; authored short-alias QA for name-preservation teaching, not a runtime output"}
    assert not {r["qid"] for r in cases} & {e["source_qid"] for e in examples}
    assert "Jeanne" not in json.dumps(examples, ensure_ascii=False)
    write(HERE / "final_examples.json", examples)
    protocol = {
        "scope": "round2:12 different fresh questions plus all round1 paired losses; development evidence only",
        "conditions": {
            "dual_final_v2": {"rewrite": "frozen_v2", "final": "v2"},
            "dual_final_v3": {"rewrite": "frozen_v2", "final": "v3"},
            "single_final_v2": {"rewrite": "none", "final": "v2"},
            "single_final_v3": {"rewrite": "none", "final": "v3"},
        },
        "model": "qwen-plus-2025-12-01", "temperature": 0, "enable_thinking": False,
        "rewrite_tokens": 256, "child_tokens": 128, "final_tokens": 64,
        "global_cap_cny": 200, "protected_global_cutoff_cny": 195, "round_subcap_cny": 5,
        "cases_sha256": sha(HERE / "cases.frozen.jsonl"), "new_prompt_sha256": sha(HERE / "prompts.py"),
        "new_examples_sha256": sha(HERE / "final_examples.json"), "v2_prompts_sha256": sha(PREVIOUS / "prompts.py"),
        "v2_examples_sha256": sha(PREVIOUS / "final_examples.json"), "child_prompts_sha256": sha(S02 / "prompts.py"),
        "previous_results_sha256": sha(PREVIOUS_OUT / "CASE_RESULTS.jsonl"),
        "passing_gate": "all13 complete; dual_final_v3 mean F1 above single_final_v3; zero individual F1 losses, including the mandatory regression",
        "sample_notes": "Hotpot/2Wiki2 historical full successes+2 zeros each. Mu2 full successes+1 zero+1 partial(0.6667), explicitly retained as partial. New query IDs can share corpus facts/templates; not an unseen-corpus or independent generalization claim.",
    }
    write(HERE / "protocol.json", protocol)
    (HERE / "FINAL_TEMPLATE.txt").write_text(prompts.final("{original_query}", "{original_top10_plus_auxiliary}", examples), encoding="utf-8")
    (HERE / "DECISIONS.txt").write_text(
        "2026-10-17 round2, frozen before generation\n"
        "Round1 failed: all13 complete,4 gains/8 ties/1 loss; name regression remains. All four dual prompt combinations have identical aggregate F1, so no F1 improvement from the v2 prompt revision is claimed.\n"
        "Only final name selection changes. Retain the exact v2 rewriter, its3 shots, model, child prompt, decomposition process, retrieval, interleave limit, full-sentence propagation, goal execution and helper placement.\n"
        "N1 Add an explicit preference for the full identifying source name when an auxiliary uses a short alias; preserve compound surnames and disambiguating suffixes.\n"
        "N2 Replace shortest-complete-answer wording with concise-answer wording and restate name preservation in the final readout. Hypothesis: minimize-length wording competes with complete-name preservation; not established from model internals.\n"
        "N3 Replace only the first teaching fixture: the old Spanish Valley excerpt names Colorado River, while a complete auxiliary sentence calls it Colorado; final answer is Colorado River. This is a generic source-name example, not the regression entity or its gold answer. Keep the other3 v2 shots byte-for-byte.\n"
        "Four controls share upstream branches: dual+v2/v3 final, single+v2/v3 final. Compare new dual against single with the same final prompt; measure final-only changes separately.\n"
        "Sampling uses hash order prompt-v3-round2 plus query-only type/suitability screening, excluding all preceding24+12 fresh IDs. Hotpot skips the first two historical-success direct-fact questions (electoral division, actress role) to retain a real bridge question. TwoWiki rotates strata to compositional/bridge_comparison failures and inference/comparison successes. Mu's unused screened prefix has only one zero left, so one partial case is explicitly included. Prior loss is mandatory; no gold-based runtime output selection.\n",
        encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"questions": len(cases), "fresh": 12, "regression": 1, "conditions": 4}))


if __name__ == "__main__":
    main()
