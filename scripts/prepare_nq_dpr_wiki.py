"""Command-line entry point for the fixed NQ Open + DPR Wikipedia preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_support import configure_utf8_output, non_negative_int, positive_int
from src.preparers.nq_dpr_wiki import (
    DEFAULT_CALIBRATION_QUESTIONS,
    DEFAULT_EVALUATION_QUESTIONS,
    DEFAULT_HARD_NEGATIVES_PER_QUESTION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_TARGET_PASSAGES,
    prepare_nq_dpr_wiki,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed NQ Open + one-million-passage DPR Wikipedia protocol."
    )
    parser.add_argument("--wikipedia", required=True, type=Path, help="Local DPR psgs_w100.tsv[.gz]")
    parser.add_argument("--questions", required=True, type=Path, help="Local DPR NQ retriever JSON[.gz]")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-passages", type=positive_int, default=DEFAULT_TARGET_PASSAGES)
    parser.add_argument(
        "--calibration-questions",
        type=positive_int,
        default=DEFAULT_CALIBRATION_QUESTIONS,
    )
    parser.add_argument(
        "--evaluation-questions",
        type=positive_int,
        default=DEFAULT_EVALUATION_QUESTIONS,
    )
    parser.add_argument(
        "--hard-negatives-per-question",
        type=non_negative_int,
        default=DEFAULT_HARD_NEGATIVES_PER_QUESTION,
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    configure_utf8_output()
    manifest = prepare_nq_dpr_wiki(
        args.wikipedia,
        args.questions,
        args.output_dir,
        target_passages=args.target_passages,
        calibration_questions=args.calibration_questions,
        evaluation_questions=args.evaluation_questions,
        hard_negatives_per_question=args.hard_negatives_per_question,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
