from __future__ import annotations

import re
from collections import defaultdict

from src.io_utils import approx_token_count


SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    if sentences:
        return sentences
    return [text.strip()] if text.strip() else []


def _tail_overlap(sentences: list[dict], overlap_tokens: int) -> list[dict]:
    if overlap_tokens <= 0:
        return []
    selected: list[dict] = []
    total = 0
    for sentence in reversed(sentences):
        selected.append(sentence)
        total += sentence["token_count"]
        if total >= overlap_tokens:
            break
    return list(reversed(selected))


def _make_chunk(doc_id: str, source: str, chunk_index: int, sentences: list[dict]) -> dict:
    text = " ".join(item["text"] for item in sentences).strip()
    page_start = min(item["page"] for item in sentences)
    page_end = max(item["page"] for item in sentences)
    token_count = approx_token_count(text)
    return {
        "chunk_id": f"{doc_id}_p{page_start}_c{chunk_index:04d}",
        "doc_id": doc_id,
        "source": source,
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
        "token_count": token_count,
    }


def chunk_records(
    records: list[dict],
    chunk_size_tokens: int = 300,
    chunk_overlap_tokens: int = 50,
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["doc_id"]].append(record)

    chunks: list[dict] = []
    for doc_id, doc_records in sorted(grouped.items()):
        doc_records = sorted(doc_records, key=lambda row: row["page"])
        source = doc_records[0]["source"]
        sentence_items: list[dict] = []
        for record in doc_records:
            for sentence in split_sentences(record["text"]):
                token_count = approx_token_count(sentence)
                if token_count:
                    sentence_items.append(
                        {
                            "text": sentence,
                            "page": record["page"],
                            "token_count": token_count,
                        }
                    )

        current: list[dict] = []
        current_tokens = 0
        doc_chunk_index = 1

        for sentence in sentence_items:
            would_exceed = current and current_tokens + sentence["token_count"] > chunk_size_tokens
            if would_exceed:
                chunks.append(_make_chunk(doc_id, source, doc_chunk_index, current))
                doc_chunk_index += 1
                current = _tail_overlap(current, chunk_overlap_tokens)
                current_tokens = sum(item["token_count"] for item in current)

            current.append(sentence)
            current_tokens += sentence["token_count"]

            if sentence["token_count"] >= chunk_size_tokens:
                chunks.append(_make_chunk(doc_id, source, doc_chunk_index, current))
                doc_chunk_index += 1
                current = []
                current_tokens = 0

        if current:
            chunks.append(_make_chunk(doc_id, source, doc_chunk_index, current))

    return chunks

