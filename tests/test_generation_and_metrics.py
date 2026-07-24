from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.evaluators.metrics import (
    answer_exact_match,
    answer_token_f1,
    evidence_all_hit,
    evidence_recall_at_k,
    expected_source_hit,
    source_precision_at_k,
    source_recall_at_k,
    summarize_results,
)
from src.generators.llm_generator import LLMGenerator


RETRIEVED = [
    {
        "rank": 1,
        "chunk_id": "c1",
        "source": "paper.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": "Evidence sentence.",
    }
]


def test_openai_failure_is_not_hidden(monkeypatch) -> None:
    generator = LLMGenerator(provider="openai", model="test")
    monkeypatch.setattr(generator, "_openai_generate", lambda prompt: (_ for _ in ()).throw(ConnectionError("offline")))
    with pytest.raises(ConnectionError, match="offline"):
        generator.generate_from_prompt("Prompt", "Question?", RETRIEVED)


def test_extractive_backend_is_explicit() -> None:
    generator = LLMGenerator(provider="extractive")
    result = generator.generate_from_prompt("Prompt", "Question?", RETRIEVED)
    assert result.provider == "extractive"
    assert result.model == "extractive"


def test_provider_model_response_id_and_environment_key_are_recorded(monkeypatch) -> None:
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text="answer",
                usage=None,
                model="resolved-model-2026-01-01",
                id="resp_test",
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    generator = LLMGenerator(provider="openai", model="requested-alias")
    result = generator.generate_from_prompt("Prompt", "Question?", RETRIEVED)
    assert captured["api_key"] == "private-key"
    assert result.requested_model == "requested-alias"
    assert result.model == "resolved-model-2026-01-01"
    assert result.response_id == "resp_test"
    assert "billable" not in result.token_usage


def test_environment_api_key_is_resolved_in_memory(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-plaintext-key")
    generator = LLMGenerator(provider="extractive")
    assert generator.api_key == "environment-plaintext-key"


def test_evidence_and_answer_metrics_use_valid_labels() -> None:
    results = [
        {"rank": 1, "source": "a.pdf", "page_start": 1, "page_end": 1},
        {"rank": 2, "source": "b.pdf", "page_start": 2, "page_end": 3},
    ]
    evidence = [
        {"source": "a.pdf", "page_start": 1, "page_end": 1},
        {"source": "b.pdf", "page_start": 3, "page_end": 3},
        {"source": "b.pdf", "page_start": 8, "page_end": 8},
    ]
    assert evidence_recall_at_k(results, evidence) == pytest.approx(2 / 3)
    assert evidence_all_hit(results, evidence) is False
    assert answer_exact_match("Indexing, retrieval!", "indexing retrieval") is True
    assert answer_token_f1("indexing retrieval generation", "indexing generation") == pytest.approx(0.8)
    alternative = [
        {"alternatives": [{"source": "missing.pdf", "page": 1}, {"source": "b.pdf", "page": 3}]}
    ]
    assert evidence_recall_at_k(results, alternative) == 1.0


def test_source_metrics_share_one_normalization_rule() -> None:
    results = [{"source": "Paper.PDF"}]
    expected = [" paper.pdf "]
    assert expected_source_hit(results, expected) is True
    assert source_recall_at_k(results, expected) == 1.0
    assert source_precision_at_k(results, expected) == 1.0


def test_status_counts_are_disjoint() -> None:
    summary = summarize_results(
        [
            {"status": "success", "metrics": {}},
            {"status": "error", "metrics": {}},
        ]
    )
    assert summary["num_successful_questions"] == 1
    assert summary["num_failed_questions"] == 1


def test_summary_reports_warm_latency_percentiles_and_valid_counts() -> None:
    rows = [
        {
            "status": "success",
            "answerable": True,
            "retrieval": {"latency_ms": 10.0},
            "generation": {"latency_ms": 20.0},
            "total_latency_ms": 35.0,
            "metrics": {"retrieval_evidence_recall_at_k": 1.0},
        },
        {
            "status": "success",
            "answerable": False,
            "retrieval": {"latency_ms": 30.0},
            "generation": {"latency_ms": 40.0},
            "total_latency_ms": 75.0,
            "metrics": {
                "retrieval_evidence_recall_at_k": None,
                "answerability_decision_accuracy": 1.0,
            },
        },
    ]
    summary = summarize_results(rows)
    assert summary["p50_retrieval_latency_ms"] == 20.0
    assert summary["p95_retrieval_latency_ms"] == pytest.approx(29.0)
    assert summary["retrieval_evidence_recall_at_k_valid_count"] == 1
    assert summary["answerability_decision_accuracy_valid_count"] == 1
    assert summary["unanswerable_refusal_valid_count"] == 1
    assert summary["num_answerable_questions"] == summary["num_unanswerable_questions"] == 1
