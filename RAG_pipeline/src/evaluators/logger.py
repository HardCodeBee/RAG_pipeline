from __future__ import annotations

import csv
from pathlib import Path

from src.io_utils import write_jsonl


def write_results(path: str | Path, rows: list[dict]) -> None:
    write_jsonl(path, rows)


def write_summary_csv(path: str | Path, summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

