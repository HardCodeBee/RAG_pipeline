"""Run a small, single-paper QASPER check through the existing RAG pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli_utils import configure_utf8_output, positive_int
from src.components import create_loader
from src.config import load_config, resolve_cli_path
from src.evaluators.qasper_metrics import score_qasper_question, summarize_qasper_scores
from src.index_builder import build_index
from src.loaders.qasper_loader import qasper_evidence_from_hits, qasper_questions
from src.pipeline import NaiveRAGPipeline
from src.provenance import resolved_roots


def run_qasper_smoke(
    config_path: str | Path,
    max_questions: int = 3,
    *,
    text_evidence_only: bool = False,
) -> dict:
    config = load_config(config_path)
    loader_config = config["loader"]
    if loader_config["type"] != "qasper":
        raise ValueError("QASPER smoke requires loader.type=qasper")
    if loader_config["max_documents"] != 1:
        raise ValueError("QASPER smoke requires max_documents=1 to keep retrieval paper-scoped")
    if loader_config["split"] == "test":
        raise ValueError("QASPER smoke must use train or validation, not the held-out test split")

    build_index(config)
    pipeline = NaiveRAGPipeline(config)
    loader = create_loader(config)
    articles = loader.articles(resolved_roots(config)["corpus"])
    if len(articles) != 1:
        raise RuntimeError("QASPER smoke expected exactly one indexed paper")

    article = articles[0]
    questions = qasper_questions(article)[:max_questions]
    if not questions:
        raise RuntimeError("The selected QASPER paper contains no questions")

    rows = []
    scores = []
    for question in questions:
        result = pipeline.query(question["question"], question_id=question["question_id"])
        predicted_evidence = qasper_evidence_from_hits(article, result["retrieval"]["results"])
        score = score_qasper_question(
            result["generation"]["answer"],
            predicted_evidence,
            question["references"],
            text_evidence_only=text_evidence_only,
        )
        scores.append(score)
        rows.append(
            {
                "question_id": question["question_id"],
                "predicted_evidence_count": len(predicted_evidence),
                "answer_f1": score["answer_f1"],
                "evidence_f1": score["evidence_f1"],
            }
        )

    return {
        "status": "complete",
        "mode": "single_paper_smoke",
        "split": loader_config["split"],
        "paper_id": article["id"],
        "questions_scored": len(rows),
        "text_evidence_only": text_evidence_only,
        "metrics": summarize_qasper_scores(scores),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-paper QASPER smoke evaluation.")
    parser.add_argument("--config", default="configs/qasper_smoke.yaml")
    parser.add_argument("--max-questions", type=positive_int, default=3)
    parser.add_argument("--text-evidence-only", action="store_true")
    args = parser.parse_args()
    configure_utf8_output()
    config_path = resolve_cli_path(PROJECT_ROOT, args.config)
    result = run_qasper_smoke(
        config_path,
        max_questions=args.max_questions,
        text_evidence_only=args.text_evidence_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
