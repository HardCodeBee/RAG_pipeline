"""Run the fixed retrieval-focused QASPER protocol on the global paper corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.cli_utils import configure_utf8_output, positive_int, safe_run_id
from src.config import apply_cli_overrides, load_config, resolve_cli_path, validate_config
from src.evaluators.metrics import summarize_results
from src.evaluators.qasper_metrics import score_qasper_open_corpus, summarize_qasper_open_corpus
from src.evaluators.runner import run_evaluation, validate_questions
from src.index_builder import build_index
from src.loaders.qasper_loader import (
    QASPER_EVALUATION_SLICE,
    QASPER_SPLITS,
    load_qasper_dataset,
    qasper_evaluation_questions,
    qasper_evaluation_slice_stats,
    qasper_unit_records,
)
from src.pipeline import NaiveRAGPipeline
from src.provenance import (
    evaluation_spec,
    json_sha256,
    recorded_config,
    resolved_roots,
    source_group_sha256,
)


EVALUATION_PROTOCOL = "qasper_open_corpus_text_extractive_single_evidence_v2"


def qasper_eval_config(base_config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Reuse the pinned baseline backends with the QASPER corpus adapter."""

    config_path = resolve_cli_path(PROJECT_ROOT, base_config_path)
    config = deepcopy(load_config(config_path))
    config["paths"]["corpus"] = str(
        (PROJECT_ROOT / "data" / "processed" / "qasper" / "hf_dataset").resolve()
    )
    config["loader"] = {"type": "qasper", "split": "all", "max_documents": None}
    config["chunking"]["local_files_only"] = True
    config["embedding"]["local_files_only"] = True
    config["logging"]["save_retrieved_text"] = True
    config["logging"]["save_prompt"] = False
    config = validate_config(config)

    required = {
        "sentence_transformers": config["embedding"]["backend"] == "sentence_transformers",
        "faiss": config["index"]["backend"] == "faiss",
        "openai": config["generation"]["provider"] == "openai",
    }
    missing = [name for name, enabled in required.items() if not enabled]
    if missing:
        raise ValueError("QASPER full evaluation requires: " + ", ".join(missing))
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("QASPER full evaluation requires OPENAI_API_KEY")
    return config, config_path


def qasper_eval_inputs(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_questions: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Derive validation questions and the global train/validation/test paper map."""

    articles_by_id: dict[str, Mapping[str, Any]] = {}
    for split in QASPER_SPLITS:
        if split not in dataset:
            raise ValueError(f"QASPER split is not available: {split}")
        for article in dataset[split]:
            paper_id = str(article.get("id") or "").strip()
            if not paper_id or paper_id in articles_by_id:
                raise ValueError("QASPER paper ids must be non-empty and globally unique")
            articles_by_id[paper_id] = article

    questions: list[dict[str, Any]] = []
    for article in dataset["validation"]:
        for question in qasper_evaluation_questions(article):
            questions.append({**question, "expected_sources": [question["paper_id"]]})
            if max_questions is not None and len(questions) >= max_questions:
                return questions, articles_by_id
    return questions, articles_by_id


def _failed_score() -> dict[str, Any]:
    return {
        "qasper_target_paper_hit_at_k": False,
        "qasper_target_paper_rr": 0.0,
        "qasper_answer_f1": 0.0,
        "qasper_target_evidence_hit_at_k": False,
        "qasper_target_evidence_recall_at_k": 0.0,
        "qasper_target_evidence_f1_at_k": 0.0,
    }


def _evaluation_fields(question: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": EVALUATION_PROTOCOL,
        "question_slice": QASPER_EVALUATION_SLICE,
        "target_paper_id": question["paper_id"],
        "reference_count": len(question["references"]),
        "source_reference_count": question["source_reference_count"],
    }


def _provider_tokens(row: Mapping[str, Any], field: str) -> int:
    reported = (
        (row.get("generation") or {})
        .get("token_usage", {})
        .get("provider_reported", {})
    )
    value = reported.get(field)
    return int(value) if isinstance(value, (int, float)) else 0


def _qasper_summary(
    rows: list[dict[str, Any]],
    *,
    manifest: Mapping[str, Any],
    effective_top_k: int,
    slice_stats: Mapping[str, Any],
) -> dict[str, Any]:
    scores = [row.get("metrics") or _failed_score() for row in rows]
    generic = summarize_results(rows)
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "question_slice": QASPER_EVALUATION_SLICE,
        "question_split": "validation",
        "num_candidate_questions_before_slice": slice_stats["candidate_questions"],
        "num_eligible_questions": slice_stats["selected_questions"],
        "num_candidate_references_before_slice": slice_stats["candidate_references"],
        "num_eligible_references": slice_stats["selected_references"],
        "corpus_splits": "+".join(QASPER_SPLITS),
        "num_corpus_papers": manifest["corpus"]["num_documents"],
        "num_index_chunks": manifest["index"]["count"],
        "effective_top_k": effective_top_k,
        "embedding_backend": manifest["embedding"]["space"]["backend"],
        "index_backend": manifest["index"]["backend"],
        "generation_provider": "openai",
        **summarize_qasper_open_corpus(scores),
        "num_successful_questions": generic["num_successful_questions"],
        "num_failed_questions": generic["num_failed_questions"],
        "avg_retrieval_latency_ms": generic["avg_retrieval_latency_ms"],
        "avg_generation_latency_ms": generic["avg_generation_latency_ms"],
        "avg_total_latency_ms": generic["avg_total_latency_ms"],
        "p95_total_latency_ms": generic["p95_total_latency_ms"],
        "total_provider_input_tokens": sum(_provider_tokens(row, "input_tokens") for row in rows),
        "total_provider_output_tokens": sum(_provider_tokens(row, "output_tokens") for row in rows),
    }


def main(argv: Sequence[str] | None = None) -> Path:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Evaluate the retrieval-focused QASPER slice on the global paper corpus."
    )
    parser.add_argument("--base-config", default="configs/baseline.yaml")
    parser.add_argument("--run-id", type=safe_run_id, default=None)
    parser.add_argument("--top-k", type=positive_int, default=None)
    parser.add_argument("--max-questions", type=positive_int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    configure_utf8_output()
    if args.resume and args.run_id is None:
        parser.error("--resume requires an explicit --run-id")

    config, config_path = qasper_eval_config(args.base_config)
    config = apply_cli_overrides(config, top_k=args.top_k)
    dataset_path = resolved_roots(config)["corpus"]
    dataset = load_qasper_dataset(dataset_path)
    slice_stats = qasper_evaluation_slice_stats(dataset["validation"])
    raw_questions, articles_by_id = qasper_eval_inputs(
        dataset,
        max_questions=args.max_questions,
    )
    questions = validate_questions(raw_questions)
    if not questions:
        raise RuntimeError(f"QASPER evaluation slice is empty: {QASPER_EVALUATION_SLICE}")

    manifest = build_index(config)
    if manifest["index"]["backend"] != "faiss":
        raise RuntimeError("QASPER index build did not use FAISS")
    pipeline = NaiveRAGPipeline(config)
    questions_sha = json_sha256(questions)
    evaluation_source_sha = source_group_sha256(PROJECT_ROOT, "evaluation")
    evaluation_value = evaluation_spec(questions_sha, evaluation_source_sha)
    run_id = args.run_id or (
        "qasper_validation_text_extractive_single_evidence_"
        + ("full_" if args.max_questions is None else f"n{args.max_questions}_")
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    run_dir = resolved_roots(config)["outputs_root"] / run_id
    evidence_units_by_paper = {
        paper_id: [unit.evidence for unit in qasper_unit_records(article)]
        for paper_id, article in articles_by_id.items()
    }

    def evaluate_question(question: Mapping[str, Any]) -> dict[str, Any]:
        row = pipeline.query(question["question"], question_id=question["question_id"])
        generation = row["generation"]
        if generation["provider"] != "openai":
            raise RuntimeError("A successful QASPER row did not use the OpenAI provider")
        row["metrics"] = score_qasper_open_corpus(
            generation["answer"],
            row["retrieval"]["results"],
            question["references"],
            question["paper_id"],
            evidence_units_by_paper,
        )
        row["evaluation"] = _evaluation_fields(question)
        return row

    def error_fields(question: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "metrics": _failed_score(),
            "evaluation": _evaluation_fields(question),
        }

    metadata = {
        "run_id": run_id,
        "command": "run_qasper_eval",
        "status": "running",
        "config_path": str(config_path),
        "effective_config": recorded_config(config),
        "questions_path": None,
        "questions_source": f"{dataset_path}#validation/qas?slice={QASPER_EVALUATION_SLICE}",
        "questions_sha256": questions_sha,
        "evaluation_spec": evaluation_value,
        "evaluation_spec_sha256": json_sha256(evaluation_value),
        "evaluation_source_sha256": evaluation_source_sha,
        **pipeline.runtime_metadata,
        "requested_top_k": args.top_k,
        "effective_top_k": config["retrieval"]["top_k"],
        "resume": args.resume,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "num_question_records": len(questions),
        "num_rows_written": 0,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "question_slice": QASPER_EVALUATION_SLICE,
        "question_selection": dict(slice_stats),
        "question_split": "validation",
        "corpus_splits": list(QASPER_SPLITS),
    }
    _, summary = run_evaluation(
        questions=questions,
        run_dir=run_dir,
        metadata=metadata,
        resume=args.resume,
        evaluate_question=evaluate_question,
        summarize_rows=lambda rows: _qasper_summary(
            rows,
            manifest=manifest,
            effective_top_k=config["retrieval"]["top_k"],
            slice_stats=slice_stats,
        ),
        error_fields=error_fields,
        process_started=process_started,
    )
    print(f"Saved run: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir


if __name__ == "__main__":
    main()
