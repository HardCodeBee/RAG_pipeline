"""Prepare a reusable local copy of the official Hugging Face QASPER dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli_utils import configure_utf8_output


DATASET_ID = "allenai/qasper"
PARQUET_REVISION = "refs/convert/parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "qasper" / "hf_dataset"
EXPECTED_SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = {"id", "title", "abstract", "full_text", "qas", "figures_and_tables"}


def prepare_qasper(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as error:
        raise RuntimeError("QASPER caching requires requirements/experiment.txt") from error

    output = Path(output_dir).resolve()
    if output.exists():
        dataset = load_from_disk(str(output))
        loaded_from = "local_disk"
    else:
        dataset = load_dataset(DATASET_ID, revision=PARQUET_REVISION)
        output.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output))
        loaded_from = "huggingface"

    if set(dataset) != set(EXPECTED_SPLITS):
        raise ValueError(f"Unexpected QASPER splits: {sorted(dataset)}")

    split_rows = {}
    for split_name in EXPECTED_SPLITS:
        split = dataset[split_name]
        if split.num_rows <= 0:
            raise ValueError(f"QASPER split is empty: {split_name}")
        missing = REQUIRED_COLUMNS - set(split.column_names)
        if missing:
            raise ValueError(f"QASPER {split_name} is missing columns: {sorted(missing)}")
        split_rows[split_name] = split.num_rows

    return {
        "dataset": DATASET_ID,
        "loaded_from": loaded_from,
        "output_dir": str(output),
        "split_rows": split_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QASPER as a reusable local dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    configure_utf8_output()
    print(json.dumps(prepare_qasper(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
