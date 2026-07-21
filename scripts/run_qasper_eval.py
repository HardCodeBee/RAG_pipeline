"""Run the real baseline backends on question-only, open-corpus QASPER evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# QASPER and the pinned BGE revision are already local. These flags also stop
# optional processor-file probes that some Transformers versions issue even
# when model loading receives local_files_only=True.
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from scripts.run_eval import main as run_eval_main
from src.cli_utils import configure_utf8_output, positive_int, safe_run_id
from src.config import load_config, resolve_cli_path, validate_config
from src.evaluators.logger import write_metadata_json, write_results, write_summary_csv
from src.evaluators.metrics import summarize_results
from src.evaluators.qasper_metrics import score_qasper_open_corpus, summarize_qasper_open_corpus
from src.index_builder import build_index
from src.io_utils import read_jsonl
from src.loaders.qasper_loader import (
    QASPER_SPLITS,
    load_qasper_dataset,
    qasper_questions,
    qasper_unit_records,
)
from src.provenance import resolved_roots


EVALUATION_PROTOCOL = "qasper_question_only_open_corpus_v1"


def qasper_eval_config(base_config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Reuse baseline backends and replace only the corpus-facing configuration."""

    config_path = resolve_cli_path(PROJECT_ROOT, base_config_path)
    config = deepcopy(load_config(config_path))
    config["project"]["name"] = "qasper_open_corpus_eval"
    config["paths"]["corpus"] = str((PROJECT_ROOT / "data" / "processed" / "qasper" / "hf_dataset").resolve())
    config["loader"] = {"type": "qasper", "split": "all", "max_documents": None}
    # The baseline pins a concrete revision; QASPER should reuse that cache and
    # must not spend preparation time probing Hugging Face over the network.
    config["chunking"]["local_files_only"] = True
    config["embedding"]["local_files_only"] = True
    # Retrieved units are required for evidence scoring; full prompts are redundant.
    config["logging"]["save_retrieved_chunks"] = True
    config["logging"]["save_prompt"] = False
    config = validate_config(config)

    required = {
        "strict_backends": config["strict_backends"] is True,
        "sentence_transformers": config["embedding"]["backend"] == "sentence_transformers",
        "faiss": config["index"]["backend"] == "faiss",
        "openai": config["generation"]["provider"] == "openai",
    }
    missing = [name for name, enabled in required.items() if not enabled]
    if missing:
        raise ValueError("QASPER full evaluation requires: " + ", ".join(missing))
    configured_key = str(config["generation"].get("api_key") or "").strip()
    environment_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not configured_key and not environment_key:
        raise RuntimeError("QASPER full evaluation requires an OpenAI API key")
    return config, config_path


def qasper_eval_inputs(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_questions: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Derive validation questions and the global paper map from the DatasetDict."""

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
        for question in qasper_questions(article):
            questions.append(
                {
                    **question,
                    "expected_sources": [question["paper_id"]],
                }
            )
            if max_questions is not None and len(questions) >= max_questions:
                return questions, articles_by_id
    return questions, articles_by_id


def _failed_score(question: Mapping[str, Any]) -> dict[str, Any]:
    references = question["references"]
    has_gold_evidence = any(
        not reference.get("unanswerable", False) and bool(reference.get("evidence"))
        for reference in references
    )
    return {
        "qasper_target_paper_hit_at_k": False,
        "qasper_target_paper_rr": 0.0,
        "qasper_answer_f1": 0.0,
        "qasper_target_evidence_recall_at_k": 0.0 if has_gold_evidence else None,
        "qasper_target_evidence_f1_at_k": 0.0,
    }


def _provider_tokens(row: Mapping[str, Any], field: str) -> int:
    generation = row.get("generation") or {}
    usage = generation.get("token_usage") or {}
    reported = usage.get("provider_reported") or {}
    value = reported.get(field, generation.get(field, 0))
    return int(value) if isinstance(value, (int, float)) else 0


def _finalize_qasper_run(
    run_dir: Path,
    questions: Sequence[Mapping[str, Any]],
    articles_by_id: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    effective_top_k: int,
) -> dict[str, Any]:
    rows = list(read_jsonl(run_dir / "results.jsonl"))
    rows_by_id = {row.get("question_id"): row for row in rows}
    if len(rows_by_id) != len(rows) or set(rows_by_id) != {item["question_id"] for item in questions}:
        raise ValueError("QASPER result rows do not exactly match the evaluation questions")

    evidence_units_by_paper = {
        paper_id: [unit.evidence for unit in qasper_unit_records(article)]
        for paper_id, article in articles_by_id.items()
    }
    scores = []
    ordered_rows = []
    for question in questions:
        row = rows_by_id[question["question_id"]]
        if row.get("status") == "success":
            generation = row.get("generation") or {}
            if generation.get("provider") != "openai":
                raise RuntimeError("A successful QASPER row did not use the OpenAI provider")
            score = score_qasper_open_corpus(
                str(generation.get("answer") or ""),
                row.get("retrieval", {}).get("results", []),
                question["references"],
                question["paper_id"],
                evidence_units_by_paper,
            )
        else:
            score = _failed_score(question)
        row.setdefault("metrics", {}).update(score)
        row["evaluation"] = {
            "protocol": EVALUATION_PROTOCOL,
            "target_paper_id": question["paper_id"],
            "reference_count": len(question["references"]),
        }
        scores.append(score)
        ordered_rows.append(row)

    write_results(run_dir / "results.jsonl", ordered_rows)
    generic = summarize_results(ordered_rows)
    summary = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "question_split": "validation",
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
        "total_provider_input_tokens": sum(_provider_tokens(row, "input_tokens") for row in ordered_rows),
        "total_provider_output_tokens": sum(_provider_tokens(row, "output_tokens") for row in ordered_rows),
    }
    write_summary_csv(run_dir / "summary.csv", summary)

    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "command": "run_qasper_eval",
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "question_split": "validation",
            "corpus_splits": list(QASPER_SPLITS),
            "summary": summary,
        }
    )
    write_metadata_json(metadata_path, metadata)
    return summary


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Evaluate the real baseline on global QASPER retrieval.")
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
    dataset_path = resolved_roots(config)["corpus"]
    dataset = load_qasper_dataset(dataset_path)
    questions, articles_by_id = qasper_eval_inputs(dataset, max_questions=args.max_questions)
    manifest = build_index(config)
    if manifest["index"]["backend"] != "faiss":
        raise RuntimeError("QASPER index build did not use FAISS")

    run_id = args.run_id or (
        "qasper_validation_"
        + ("full_" if args.max_questions is None else f"n{args.max_questions}_")
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    eval_argv = ["--config", str(config_path), "--run-id", run_id]
    if args.top_k is not None:
        eval_argv.extend(["--top-k", str(args.top_k)])
    if args.resume:
        eval_argv.append("--resume")
    run_dir = run_eval_main(
        eval_argv,
        config_override=config,
        questions_override=questions,
        questions_source=f"{dataset_path}#validation/qas",
        command_name="run_qasper_eval",
    )
    summary = _finalize_qasper_run(
        run_dir,
        questions,
        articles_by_id,
        manifest,
        args.top_k or config["retrieval"]["top_k"],
    )
    print("\nQASPER open-corpus summary:\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir


if __name__ == "__main__":
    main()
