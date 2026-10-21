"""Freeze revised prompt fixtures and a matched 2-by-2 diagnostic comparison."""

import hashlib
import json
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import rows, load, write
from . import prompts

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / "multigrain_prompt_v2_20261017"
OLD_SPEC = ROOT / "experiments/query_decomposition/multigrain_dual_retrieval_pilot_v1_20261017"
OLD_OUT = OUTPUT / "multigrain_dual_retrieval_pilot_v1_20261017"
S02 = ROOT / "experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful"

NEW_PLANS = {
    "hotpotqa": {
        "5a8eb63455429917b4a5bdb7": ["With which musicians did Jonny Browning collaborate?"],
        "5ae25d2b554299495565da46": ["How many singles titles did Brian Gottfried win?", "How many singles titles did Peter Fleming win?"],
        "5abad3445542996cc5e49ea0": ["Who is Horace Grant's twin brother?"],
        "5a8662da5542991e7718162f": ["Does Denis Bélanger play the guitar?", "Does Max Carl play the guitar?"],
    },
    "2wiki": {
        "bc7073a80baf11ebab90acde48001122": ["Who are Anurutha's parents?", "Who are the brothers of Anurutha's parents?"],
        "0878d86308e511ebbda4ac1f6bf848b6": ["Who directed the film Swaroopam?", "On what date was the director of Swaroopam born?", "Who directed the film A Ship Comes In?", "On what date was the director of A Ship Comes In born?"],
        "f98692b80bdc11eba7f7acde48001122": ["Who directed the film Saint Janet?"],
        "ed0838b508dc11ebbd9dac1f6bf848b6": ["From which country does the film Daisy Kenyon originate?", "From which country does the film Man Push Cart originate?"],
    },
    "musique": {
        "4hop1__726152_153080_159767_81096": ["Who performed Tijuana Moods?", "In which state was the performer of Tijuana Moods born?", "What is the largest city and capital of the state where the performer of Tijuana Moods was born?"],
        "3hop2__304722_423353_63959": ["For which sports team did Jose Dominguez play?", "Which team won the 1894-95 FA Cup?"],
        "2hop__73717_653666": ["Who found the sacred writings that became the Book of Mormon?"],
        "3hop1__488744_443779_52195": ["In which country is Jenggawah located?", "Which newly declared independent country is part of a Commission of Truth and Friendship with the country where Jenggawah is located?"],
    },
}


def freeze_cases():
    old_cases = {r["qid"]: r for r in rows(OLD_SPEC / "cases.frozen.jsonl")}
    selected = []
    inputs = ROOT / "experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen"
    for dataset, plans in NEW_PLANS.items():
        questions = {r["qid"]: r for r in rows(inputs / dataset / "formal.questions.jsonl")}
        refs = {r["qid"]: r for r in rows(inputs / dataset / "formal.references.local.jsonl")}
        scores = (
            load(OLD_OUT / "MUSIQUE_SCREENING_RESULTS.json")["results"] if dataset == "musique" else
            rows(OUTPUT / f"comprehensive_v1_20261015/campaign_api_v1/{dataset}/formal/m00_s02_child4_faithful/frozen_s02_child4/scores.jsonl")
        )
        scores = {r["qid"]: r for r in scores}
        for qid, intermediates in plans.items():
            assert qid not in old_cases
            question = questions[qid]["question"]
            steps = [{"id": i + 1, "request": request} for i, request in enumerate(intermediates + [question])]
            selected.append({"dataset": dataset, "qid": qid, "question": question, "type": refs[qid]["type"],
                "historical_f1": scores[qid]["f1"], "historical_group": "success" if scores[qid]["f1"] == 1 else "failure",
                "evaluation_group": "fresh", "qdmr": {"steps": steps, "goal": len(steps)}})
    regression = dict(old_cases["32f0334c0bb011ebab90acde48001122"])
    regression["evaluation_group"] = "regression"
    selected.append(regression)
    assert len(selected) == 13 and sum(r["evaluation_group"] == "fresh" for r in selected) == 12
    (HERE / "cases.frozen.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
    return selected


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE / "protocol.json").exists():
        raise RuntimeError("Protocol already frozen; do not regenerate it")
    old = load(S02 / "examples.json")
    examples = []
    answers = [
        [
            ("Which river does Pack Creek flow into after passing through Spanish Valley?", "Pack Creek flows into the Colorado River."),
            (old["final"][0]["question"], "The Colorado River, into which Pack Creek flows, is 1450 mi long."),
        ],
        [
            ("Is Leycesteria native to Australia?", "Leycesteria is native to temperate Asia, not Australia."),
            ("Is Anigozanthos native to Australia?", "Anigozanthos is native to Australia."),
            (old["final"][1]["question"], "Leycesteria and Anigozanthos are not both native to Australia, because Leycesteria is native to temperate Asia."),
        ],
        [
            ("In which province is Baraki Barak District situated?", "Baraki Barak District is in Logar Province."),
            (old["final"][2]["question"], "The capital of Logar Province is Kabul."),
        ],
        [
            ("How many species does Clianthus contain?", "Clianthus contains two species."),
            ("How many species does Callicoma contain?", "Callicoma contains one species."),
            (old["final"][3]["question"], "Clianthus has more species than Callicoma: Clianthus contains two species and Callicoma contains one."),
        ],
    ]
    for index, row in enumerate(old["final"]):
        history = [{"step_id": i + 1, "request": q, "answer": a} for i, (q, a) in enumerate(answers[index])]
        source_context = row["context"].split("\n\nAuxiliary question-answer pairs")[0]
        examples.append({
            "source_qid": row["qid"], "question": row["question"],
            "context": source_context + "\n\n" + prompts.helper(history),
            "answer": row["answer"], "qas": history,
            "fixture_origin": "authored teaching fixtures adapted from frozen S02; not runtime answers",
            "intentional_conflict": "Kabul contradicts the explicit Puli Alam passage" if index == 2 else None,
        })
    cases = freeze_cases()
    assert not {r["qid"] for r in cases} & {r["source_qid"] for r in examples}
    assert all("#" not in q["request"] for example in examples for q in example["qas"])
    assert all("Earlier retrieval" not in e["context"] for e in examples)
    write(HERE / "final_examples.json", examples)
    protocol = {
        "scope": "round1:12 fresh method-diagnostic questions plus the previous paired-regression question; development comparison, not independent confirmation",
        "conditions": {
            "c00_old_prompts": {"rewrite": "old", "final": "old"},
            "c01_final_only": {"rewrite": "old", "final": "new", "source": "exact c00 context; final prompt only differs"},
            "c02_rewrite_only": {"rewrite": "new", "final": "old"},
            "c03_both": {"rewrite": "new", "final": "new", "source": "exact c02 context; final prompt only differs"},
            "c04_single": {"rewrite": "none", "final": "new", "source": "same full-sentence process and final prompt as c03, no rewrite retrieval"},
        },
        "model": "qwen-plus-2025-12-01", "temperature": 0, "enable_thinking": False,
        "rewrite_tokens": 256, "child_tokens": 128, "final_tokens": 64,
        "rewrite_receives_original_query": False,
        "rewriter_verifies_prior_answers": False,
        "rewrite_examples": 3, "final_examples": 4,
        "child_prompt": "unchanged frozen S02 with4 ordinary examples; no change to child answer generation instructions",
        "retrieval": "same BGE resources; Top10 each route; original-first rank interleave, doc-id dedup, keep10",
        "original_top10": "exact frozen original packets retained in every final context",
        "goal": "execute and answer all fixed self-contained nodes including original-query goal",
        "full_sentence_propagation": True,
        "historical_reuse": "only successful exact full API payloads; original response receipts/provenance retained; no new charge claimed for a replay",
        "network_policy": "inherited bounded in-process retries; unknown attempt reservations retained",
        "global_cap_cny": 200, "protected_global_cutoff_cny": 195, "new_experiment_subcap_cny": 5,
        "cases_sha256": sha(HERE / "cases.frozen.jsonl"),
        "passing_gate": "all13 complete; candidate dual mean F1 above matched c04 single; zero per-question F1 losses, including mandatory previous-regression case",
        "next_round": "12 new questions not used in earlier rounds, plus accumulated previous-regression questions; select and freeze before any next-round generation",
        "old_results_sha256": sha(OLD_OUT / "CASE_RESULTS.jsonl"),
        "prompt_source_sha256": sha(HERE / "prompts.py"),
        "final_examples_sha256": sha(HERE / "final_examples.json"),
        "child_prompt_source_sha256": sha(S02 / "prompts.py"),
        "child_examples_sha256": sha(S02 / "examples.json"),
    }
    write(HERE / "protocol.json", protocol)
    (HERE / "REWRITE_TEMPLATE.txt").write_text(prompts.rewrite("{current_question}", [{"request": "{previous_question}", "answer": "{complete_previous_answer}"}]), encoding="utf-8")
    (HERE / "FINAL_TEMPLATE.txt").write_text(prompts.final("{original_query}", "{unchanged_original_top10_with_auxiliary_QA}", examples), encoding="utf-8")
    (HERE / "DECISIONS.txt").write_text(
        "2026-10-17 prompt revision v2, before API submission\n"
        "R1 Remove candidate/unverified language and prior-answer verification from rewriting. Rewriter only transforms the current retrieval question using supplied QA. No corrective model or extra verifier.\n"
        "R2 Remove the original global query from rewrite input because current questions are self-contained. Purpose: reduce jumps from a local question to the final goal. Previous QA and current question remain.\n"
        "R3 Add3 authored rewrite shots: explicit entity replacement; preserve a death-place lookup instead of inventing its answer; leave an independent question unchanged. These examples are disjoint from all evaluation questions.\n"
        "F1 Keep4 final-answer shots and existing passage excerpts. Rewrite every request as a self-contained question; retain complete answer sentences and include each example's goal.\n"
        "F2 Replace the artificial Clianthus missing-count/incorrect-comparison scenario with fully stated two-versus-one evidence and correct complete QA. Also replace the contradictory Boolean goal with a correct goal.\n"
        "F3 Retain one explicit direct conflict example (Kabul QA versus Puli Alam passage) to teach use of matching source evidence. It is an authored teaching fixture, not a runtime prediction.\n"
        "F4 Use one concise final task contract and one short readout: original target, matching entity/relation/conditions, evidence conflict, supported combination, full names/units, and explicit Cannot be determined when information is insufficient.\n"
        "E1 User steering before execution: each evaluation round must contain new questions and prior dual-versus-single regressions. Round1 has12 fresh questions (4 per dataset,2 historical successes and2 failures) plus the known name-shortening regression. Earlier24 are not the development test batch again.\n"
        "E2 Selection: deterministic hash order prompt-v2-round1 within dataset/history/type strata, with query-only suitability screening. Hotpot bridge failures ahead of Jonny Browning were skipped because Fort Orange initial name and Structure of Liberty philosophy are direct facts without a necessary intermediate, and the1977 horror-film question has no unique film identification in its wording. TwoWiki covers4 types across the4 new questions; Mu covers2/3/4 hops using previously untested questions from the original S02 screening prefix. No new-method answers were known at selection.\n"
        "E3 Use a2x2 prompt comparison plus a matched new-final single control. Each dual branch shares its upstream history between final variants. Single and new dual share the exact final prompt, original Top10, fixed decomposition, full-sentence answering and helper placement.\n"
        "E4 Passing requires complete denominators, positive mean dual-minus-single F1 and zero individual losses. Known regressions are mandatory and separately reported. No gold-based output selection at runtime. A finite zero-loss batch does not establish a guarantee on future queries.\n",
        encoding="utf-8",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"frozen": str(HERE / 'protocol.json'), "questions": len(cases), "new_conditions": 3}))


if __name__ == "__main__":
    main()
