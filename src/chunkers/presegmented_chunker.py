"""Preserve pre-segmented retrieval units as exactly one chunk each."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from src.records import ChunkRecord, PageRecord
from src.text.token_counters import validate_token_window


class PresegmentedChunker:
    """Convert each PageRecord to one ChunkRecord without text re-segmentation."""

    def __init__(
        self,
        token_counter,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 0,
        *,
        verify_unique_ids: bool = True,
    ):
        validate_token_window(
            chunk_size_tokens,
            chunk_overlap_tokens,
            size_name="chunk_size_tokens",
            overlap_name="chunk_overlap_tokens",
        )
        if chunk_overlap_tokens != 0:
            raise ValueError("PresegmentedChunker requires chunk_overlap_tokens=0")
        if not isinstance(verify_unique_ids, bool):
            raise TypeError("verify_unique_ids must be a boolean")
        self.token_counter = token_counter
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.verify_unique_ids = verify_unique_ids

    def iter_chunks(
        self,
        records: Iterable[PageRecord | Mapping[str, Any]],
    ) -> Iterator[ChunkRecord]:
        # Multi-million-row prepared corpora validate uniqueness at their disk
        # trust boundary.  Repeating that check with a Python set would retain
        # one string per document and defeat streaming construction.
        seen_chunk_ids: set[str] | None = set() if self.verify_unique_ids else None
        for vector_id, value in enumerate(records):
            record = value if isinstance(value, PageRecord) else PageRecord.from_mapping(value)
            chunk_id = record.doc_id
            if seen_chunk_ids is not None:
                if chunk_id in seen_chunk_ids:
                    raise ValueError(f"Pre-segmented chunk id is duplicated: {chunk_id}")
                seen_chunk_ids.add(chunk_id)
            token_count = self.token_counter.count(record.text)
            if token_count <= 0:
                raise ValueError(f"Pre-segmented chunk has no tokens: {chunk_id}")
            yield ChunkRecord(
                chunk_id=chunk_id,
                vector_id=vector_id,
                doc_id=record.doc_id,
                source=record.source,
                page_start=record.page,
                page_end=record.page,
                text=record.text,
                token_count=token_count,
            )

    def chunk(
        self,
        records: Iterable[PageRecord | Mapping[str, Any]],
    ) -> list[ChunkRecord]:
        """Compatibility wrapper for the current list-based index builder."""

        return list(self.iter_chunks(records))
