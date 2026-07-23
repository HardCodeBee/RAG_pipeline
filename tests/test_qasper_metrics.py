from __future__ import annotations

import pytest

from src.evaluators.qasper_metrics import (
    normalize_qasper_answer,
    qasper_evidence_f1,
    qasper_reference_answer,
    qasper_token_f1,
    score_qasper_open_corpus,
    score_qasper_question,
    summarize_qasper_open_corpus,
    summarize_qasper_scores,
)


def _reference(*, answer: str, evidence: list[str]) -> dict:
    return {
        "unanswerable": False,
        "extractive_spans": [answer],
        "yes_no": None,
        "free_form_answer": "",
        "evidence": evidence,
    }


def test_qasper_answer_normalization_and_token_f1_follow_squad() -> None:
    assert normalize_qasper_answer("The Model, an Example!") == "model example"
    assert qasper_token_f1("retrieval retrieval model", "retrieval model") == pytest.approx(0.8)


def test_qasper_evidence_f1_matches_official_empty_and_list_rules() -> None:
    assert qasper_evidence_f1([], []) == 1.0
    assert qasper_evidence_f1(["paragraph"], []) == 0.0
    assert qasper_evidence_f1(["paragraph", "paragraph"], ["paragraph"]) == pytest.approx(2 / 3)


def test_qasper_answer_and_evidence_choose_references_independently() -> None:
    references = [
        _reference(answer="first answer", evidence=["wrong evidence"]),
        _reference(answer="second answer", evidence=["right evidence"]),
    ]

    score = score_qasper_question("first answer", ["right evidence"], references)

    assert score == {"answer_f1": 1.0, "answer_type": "extractive", "evidence_f1": 1.0}


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ({"unanswerable": True}, ("Unanswerable", "none")),
        ({"extractive_spans": ["one", "two"]}, ("one, two", "extractive")),
        ({"free_form_answer": "summary"}, ("summary", "abstractive")),
        ({"yes_no": True}, ("Yes", "boolean")),
        ({"yes_no": False}, ("No", "boolean")),
    ],
)
def test_qasper_reference_answer_types(reference, expected) -> None:
    assert qasper_reference_answer(reference) == expected


def test_qasper_summary_uses_macro_average_and_counts_missing_predictions() -> None:
    summary = summarize_qasper_scores(
        [
            {"answer_f1": 1.0, "answer_type": "extractive", "evidence_f1": 0.5},
            {"answer_f1": 0.5, "answer_type": "boolean", "evidence_f1": 1.0},
        ],
        missing_predictions=1,
    )

    assert summary["Answer F1"] == pytest.approx(0.5)
    assert summary["Evidence F1"] == pytest.approx(0.5)
    assert summary["Answer F1 by type"]["extractive"] == 1.0
    assert summary["Answer F1 by type"]["boolean"] == 0.5
    assert summary["Missing predictions"] == 1


def test_unanswerable_reference_uses_empty_official_evidence() -> None:
    reference = {"unanswerable": True, "evidence": ["ignored annotation text"]}

    score = score_qasper_question("Unanswerable", [], [reference])

    assert score["answer_f1"] == 1.0
    assert score["evidence_f1"] == 1.0


def test_open_corpus_metrics_penalize_units_from_wrong_papers() -> None:
    score = score_qasper_open_corpus(
        "answer",
        [
            {"doc_id": "target", "page_start": 1, "page_end": 1},
            {"doc_id": "wrong", "page_start": 1, "page_end": 1},
        ],
        [_reference(answer="answer", evidence=["gold evidence"])],
        "target",
        {"target": ["gold evidence"], "wrong": ["unrelated evidence"]},
    )

    assert score["qasper_target_paper_hit_at_k"] is True
    assert score["qasper_target_paper_rr"] == 1.0
    assert score["qasper_answer_f1"] == 1.0
    assert score["qasper_target_evidence_hit_at_k"] is True
    assert score["qasper_target_evidence_recall_at_k"] == 1.0
    assert score["qasper_target_evidence_f1_at_k"] == pytest.approx(2 / 3)


def test_open_corpus_summary_skips_only_undefined_evidence_recall() -> None:
    summary = summarize_qasper_open_corpus(
        [
            {
                "qasper_target_paper_hit_at_k": True,
                "qasper_target_paper_rr": 0.5,
                "qasper_answer_f1": 1.0,
                "qasper_target_evidence_hit_at_k": True,
                "qasper_target_evidence_recall_at_k": 0.5,
                "qasper_target_evidence_f1_at_k": 0.5,
            },
            {
                "qasper_target_paper_hit_at_k": False,
                "qasper_target_paper_rr": 0.0,
                "qasper_answer_f1": 0.0,
                "qasper_target_evidence_hit_at_k": False,
                "qasper_target_evidence_recall_at_k": None,
                "qasper_target_evidence_f1_at_k": 0.0,
            },
        ]
    )

    assert summary["qasper_target_paper_hit_rate_at_k"] == 0.5
    assert summary["qasper_target_paper_mrr"] == 0.25
    assert summary["qasper_target_evidence_hit_rate_at_k"] == 0.5
    assert summary["qasper_target_evidence_recall_valid_count"] == 1
    assert summary["qasper_answer_f1_when_evidence_hit"] == 1.0
    assert summary["qasper_answer_f1_when_evidence_miss"] == 0.0
    assert summary["qasper_evidence_hit_count"] == 1
    assert summary["qasper_evidence_miss_count"] == 1
