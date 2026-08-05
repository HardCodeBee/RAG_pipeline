"""Load the frozen DPR Wikipedia passage corpus through existing records."""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.records import PageRecord
from src.persistence.artifact_io import iter_jsonl
from src.persistence.artifact_validation import VerifiedDprCorpus, validate_dpr_corpus_directory
from src.provenance import json_sha256, sha256_file


PROTOCOL = "nq_open_dpr_wiki_1m_gold_preserving_v1"
TEXT_FORMAT = "title_newline_text_v1"
PASSAGE_PREFIX = "dpr_wiki_passage:"
CANONICAL_PASSAGE_COUNT = 1_000_000
CANONICAL_QUESTION_COUNT = 2_000


class DprWikipediaCorpusLoader:
    """Adapt one immutable DPR passage per PageRecord without reading labels."""

    def __init__(
        self,
        expected_protocol: str = PROTOCOL,
        text_format: str = TEXT_FORMAT,
        require_canonical_counts: bool = True,
    ):
        if expected_protocol != PROTOCOL:
            raise ValueError(f"expected_protocol must be {PROTOCOL}")
        if text_format != TEXT_FORMAT:
            raise ValueError(f"text_format must be {TEXT_FORMAT}")
        self.expected_protocol = expected_protocol
        self.text_format = text_format
        self.require_canonical_counts = require_canonical_counts
        self._validated_root: Path | None = None
        self._verified_corpus: VerifiedDprCorpus | None = None

    def manifest(self, corpus_path: str | Path) -> dict[str, Any]:
        root = Path(corpus_path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"DPR Wikipedia corpus directory does not exist: {root}")
        if self._validated_root == root and self._verified_corpus is not None:
            return self._verified_corpus.manifest
        verified = validate_dpr_corpus_directory(
            root,
            expected_protocol=self.expected_protocol,
            text_format=self.text_format,
            require_canonical_counts=self.require_canonical_counts,
            canonical_passage_count=CANONICAL_PASSAGE_COUNT,
            canonical_question_count=CANONICAL_QUESTION_COUNT,
        )
        self._validated_root = root
        self._verified_corpus = verified
        return verified.manifest

    def corpus_inventory(self, corpus_path: str | Path) -> dict[str, Any]:
        """Reuse verified artifact descriptors when deriving corpus identity."""

        root = Path(corpus_path).resolve()
        manifest = self.manifest(root)
        if self._verified_corpus is None:
            raise RuntimeError("DPR Wikipedia corpus was not verified")
        descriptors = manifest["artifacts"]
        descriptor_by_path = {
            self._verified_corpus.files[name]: descriptors[name]
            for name in ("passages", "passage_ids")
        }
        rows = []
        for path in self.discover(root):
            resolved = path.resolve()
            descriptor = descriptor_by_path.get(resolved)
            rows.append(
                {
                    "source": path.name,
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": (
                        path.stat().st_size
                        if descriptor is None
                        else descriptor["size_bytes"]
                    ),
                    "sha256": (
                        sha256_file(path)
                        if descriptor is None
                        else descriptor["sha256"]
                    ),
                }
            )
        return {"documents": rows, "aggregate_sha256": json_sha256(rows)}

    def discover(self, corpus_path: str | Path) -> list[Path]:
        root = Path(corpus_path)
        manifest = self.manifest(root)
        artifacts = manifest["artifacts"]
        paths = [
            root / "manifest.json",
            root / artifacts["passages"]["file"],
            root / artifacts["passage_ids"]["file"],
        ]
        return sorted(paths, key=lambda path: path.relative_to(root).as_posix())

    def iter_pages(self, corpus_path: str | Path) -> Iterator[PageRecord]:
        root = Path(corpus_path)
        manifest = self.manifest(root)
        passages_path = root / manifest["artifacts"]["passages"]["file"]
        ids_path = root / manifest["artifacts"]["passage_ids"]["file"]
        expected_count = manifest["counts"]["final_passages"]
        count = 0
        previous_id = 0

        with ids_path.open("r", encoding="utf-8") as ids_handle:
            ids = (line.strip() for line in ids_handle if line.strip())
            pairs = zip_longest(iter_jsonl(passages_path), ids, fillvalue=None)
            for position, (row, expected_id) in enumerate(pairs, start=1):
                if row is None or expected_id is None:
                    raise ValueError("DPR Wikipedia passages and passage_ids have different lengths")
                if not isinstance(row, Mapping) or set(row) != {"id", "title", "text"}:
                    raise ValueError(
                        "Every DPR Wikipedia corpus row must contain exactly id, title, and text"
                    )
                passage_id = str(row["id"]).strip()
                if not passage_id.isdecimal() or int(passage_id) <= previous_id:
                    raise ValueError("DPR Wikipedia passage ids must be positive and strictly increasing")
                previous_id = int(passage_id)
                if passage_id != expected_id:
                    raise ValueError(f"Passage id list differs from corpus row {position}")
                title = row["title"]
                text = row["text"]
                if not isinstance(title, str) or not title.strip():
                    raise ValueError(f"DPR Wikipedia row {position} has an empty title")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"DPR Wikipedia row {position} has empty text")
                count += 1
                yield PageRecord(
                    doc_id=f"{PASSAGE_PREFIX}{passage_id}",
                    source=title,
                    page=1,
                    text=f"{title}\n{text}",
                )
        if count != expected_count:
            raise ValueError("Loaded DPR Wikipedia row count does not match the manifest")

    def load(self, corpus_path: str | Path) -> list[PageRecord]:
        """Compatibility wrapper; large builds should consume iter_pages()."""

        return list(self.iter_pages(corpus_path))
