"""Preserve pre-segmented retrieval units as exactly one chunk each."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from src.records import ChunkRecord, PageRecord
from src.text.token_counters import validate_token_window


DPR_PASSAGE_PREFIX = "dpr_wiki_passage:"


class PresegmentedChunker:
    """Convert each PageRecord to one ChunkRecord without text re-segmentation."""

    def __init__(
        self,
        token_counter,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 0,
    ):
        validate_token_window(
            chunk_size_tokens,
            chunk_overlap_tokens,
            size_name="chunk_size_tokens",
            overlap_name="chunk_overlap_tokens",
        )
        if chunk_overlap_tokens != 0:
            raise ValueError("PresegmentedChunker requires chunk_overlap_tokens=0")
        self.token_counter = token_counter
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def iter_chunks(
        self,
        records: Iterable[PageRecord | Mapping[str, Any]],
    ) -> Iterator[ChunkRecord]:
        seen_chunk_ids: set[str] = set()
        previous_dpr_passage_id = 0
        for vector_id, value in enumerate(records):
            record = value if isinstance(value, PageRecord) else PageRecord.from_mapping(value)
            chunk_id = record.doc_id
            if chunk_id.startswith(DPR_PASSAGE_PREFIX):
                suffix = chunk_id.removeprefix(DPR_PASSAGE_PREFIX)
                if not suffix.isdecimal() or int(suffix) <= previous_dpr_passage_id:
                    raise ValueError(
                        "DPR pre-segmented chunk ids must be positive and strictly increasing"
                    )
                previous_dpr_passage_id = int(suffix)
            else:
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
