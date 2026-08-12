"""Prepare selected official BEIR corpus/query packages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_support import configure_utf8_output
from src.preparers.beir_dataset import (
    BEIR_DATASETS,
    DEFAULT_OUTPUT_ROOT,
    canonical_dataset_name,
    prepare_beir_dataset,
)


def _datasets(values: list[str]) -> list[str]:
    if values == ["all"]:
        return list(BEIR_DATASETS)
    if "all" in values:
        raise ValueError("--dataset all cannot be combined with another dataset")
    return list(dict.fromkeys(canonical_dataset_name(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and structure BEIR corpus.jsonl, queries.jsonl, and qrels "
            "without materializing large ID collections in memory."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="BEIR name (repeatable) or 'all' for the seven selected dataset families",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Existing official archive; valid only with one --dataset",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download a missing archive into <output-root>/_archives",
    )
    args = parser.parse_args()
    configure_utf8_output()
    try:
        datasets = _datasets(args.dataset)
    except ValueError as exc:
        parser.error(str(exc))
    if args.archive is not None and len(datasets) != 1:
        parser.error("--archive requires exactly one dataset")

    manifests = {
        dataset: prepare_beir_dataset(
            dataset,
            args.output_root,
            archive_path=args.archive,
            download=args.download,
        )
        for dataset in datasets
    }
    print(json.dumps(manifests, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
