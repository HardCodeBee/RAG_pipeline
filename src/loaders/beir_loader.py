"""Streaming loader for one prepared BEIR dataset unit."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.persistence.artifact_io import iter_jsonl
from src.preparers.beir_dataset import (
    TEXT_FORMAT,
    canonical_dataset_name,
    validate_beir_unit_directory,
)
from src.records import PageRecord


@dataclass(frozen=True, slots=True)
class BeirQuery:
    query_id: str
    text: str


@dataclass(frozen=True, slots=True)
class BeirQrel:
    query_id: str
    corpus_id: str
    score: float


class BeirCorpusLoader:
    """Load one BEIR corpus row as one page without materializing the corpus."""

    def __init__(self, *, expected_dataset: str | None = None):
        if expected_dataset is not None and (
            not isinstance(expected_dataset, str) or not expected_dataset.strip()
        ):
            raise ValueError("expected_dataset must be a non-empty string or None")
        self.expected_dataset = (
            canonical_dataset_name(expected_dataset)
            if expected_dataset is not None
            else None
        )
        self._validated_root: Path | None = None
        self._manifest: dict[str, Any] | None = None

    def manifest(self, corpus_path: str | Path) -> dict[str, Any]:
        root = Path(corpus_path).resolve()
        if self._validated_root == root and self._manifest is not None:
            return self._manifest
        manifest = validate_beir_unit_directory(root)
        if (
            self.expected_dataset is not None
            and manifest.get("dataset") != self.expected_dataset
        ):
            raise ValueError(
                f"Expected BEIR dataset {self.expected_dataset!r}, "
                f"found {manifest.get('dataset')!r}"
            )
        if manifest.get("text_format") != TEXT_FORMAT:
            raise ValueError(f"Unsupported BEIR text format in {root}")
        self._validated_root = root
        self._manifest = manifest
        return manifest

    def discover(self, corpus_path: str | Path) -> list[Path]:
        root = Path(corpus_path).resolve()
        manifest = self.manifest(root)
        return [
            root / "manifest.json",
            root / manifest["artifacts"]["source_manifest"]["file"],
            root / manifest["artifacts"]["corpus"]["file"],
        ]

    def iter_pages(self, corpus_path: str | Path) -> Iterator[PageRecord]:
        root = Path(corpus_path).resolve()
        manifest = self.manifest(root)
        corpus = root / manifest["artifacts"]["corpus"]["file"]
        expected_rows = manifest["artifacts"]["corpus"]["rows"]
        count = 0
        for line_number, row in enumerate(iter_jsonl(corpus), start=1):
            raw_id = row.get("_id")
            title = row.get("title", "")
            text = row.get("text", "")
            if not isinstance(raw_id, str) or not raw_id:
                raise ValueError(f"BEIR corpus row {line_number} has invalid _id")
            if not isinstance(title, str) or not isinstance(text, str):
                raise ValueError(f"BEIR corpus row {line_number} has invalid title/text")
            has_title = bool(title.strip())
            has_text = bool(text.strip())
            if has_title and has_text:
                combined = f"{title}\n{text}"
            elif has_title:
                combined = title
            else:
                combined = text
            if not combined.strip():
                raise ValueError(f"BEIR corpus row {line_number} has no text")
            count += 1
            yield PageRecord(
                doc_id=raw_id,
                source=title if has_title else f"beir:{manifest['unit']}",
                page=1,
                text=combined,
            )
        if count != expected_rows:
            raise ValueError("BEIR corpus row count differs from its manifest")

    def iter_queries(self, corpus_path: str | Path) -> Iterator[BeirQuery]:
        root = Path(corpus_path).resolve()
        manifest = self.manifest(root)
        path = root / manifest["artifacts"]["queries"]["file"]
        expected_rows = manifest["artifacts"]["queries"]["rows"]
        count = 0
        for line_number, row in enumerate(iter_jsonl(path), start=1):
            raw_id = row.get("_id")
            text = row.get("text")
            if not isinstance(raw_id, str) or not raw_id:
                raise ValueError(f"BEIR query row {line_number} has invalid _id")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"BEIR query row {line_number} has invalid text")
            count += 1
            yield BeirQuery(query_id=raw_id, text=text)
        if count != expected_rows:
            raise ValueError("BEIR query row count differs from its manifest")

    def iter_qrels(
        self,
        corpus_path: str | Path,
        *,
        split: str = "test",
    ) -> Iterator[BeirQrel]:
        root = Path(corpus_path).resolve()
        manifest = self.manifest(root)
        descriptor = manifest["artifacts"]["qrels"].get(split)
        if not isinstance(descriptor, Mapping):
            available = ", ".join(sorted(manifest["artifacts"]["qrels"]))
            raise ValueError(f"Unknown BEIR qrels split {split!r}; available: {available}")
        count = 0
        with (root / descriptor["file"]).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["query-id", "corpus-id", "score"]:
                raise ValueError("BEIR qrels header changed after preparation")
            for row in reader:
                count += 1
                yield BeirQrel(
                    query_id=row["query-id"],
                    corpus_id=row["corpus-id"],
                    score=float(row["score"]),
                )
        if count != descriptor["rows"]:
            raise ValueError("BEIR qrels row count differs from its manifest")
