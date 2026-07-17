from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.chunkers.modular_chunker import FixedSentenceChunker
from src.core.records import ChunkRecord, ContextPackage, PageRecord, SearchHit
from src.indexes.faiss_index import FlatIPIndex
from src.prompts.fixed_prompt import build_prompt
from src.retrievers.dense_retriever import DenseRetriever
from src.text.splitters import RegexSentenceSplitter
from src.text.token_counters import RegexTokenCounter


class StaticEmbedder:
    dimension = 2

    def encode_queries(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_chunk_record_rejects_missing_or_invalid_schema_fields() -> None:
    with pytest.raises(KeyError, match="token_count"):
        ChunkRecord.from_mapping(
            {
                "chunk_id": "c",
                "vector_id": 0,
                "doc_id": "doc",
                "source": "paper.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "content",
            }
        )
    with pytest.raises(ValueError, match="vector_id"):
        ChunkRecord("c", True, "doc", "paper.pdf", 1, 1, "content", 1)


def test_fixed_sentence_chunker_is_deterministic_and_respects_budget() -> None:
    counter = RegexTokenCounter()
    chunker = FixedSentenceChunker(RegexSentenceSplitter(), counter, 12, 4)
    pages = [
        PageRecord("doc", "paper.pdf", 1, "One two three four. Five six seven eight."),
        PageRecord("doc", "paper.pdf", 2, "Nine ten eleven twelve. Thirteen fourteen fifteen."),
    ]

    first = chunker.chunk(pages)
    second = chunker.chunk(pages)

    assert first == second
    assert len({item.chunk_id for item in first}) == len(first)
    assert [item.vector_id for item in first] == list(range(len(first)))
    assert all(0 < item.token_count <= 12 for item in first)


def test_fixed_sentence_chunker_preserves_document_page_and_overlap_order() -> None:
    chunker = FixedSentenceChunker(RegexSentenceSplitter(), RegexTokenCounter(), 8, 4)
    pages = [
        PageRecord("doc_b", "b.pdf", 2, "Seven eight nine."),
        {
            "doc_id": "doc_a",
            "source": "a.pdf",
            "page": 1,
            "text": "Alpha beta gamma.",
        },
        PageRecord("doc_b", "b.pdf", 1, "One two three. Four five six."),
    ]

    chunks = chunker.chunk(pages)

    assert [item.to_dict() for item in chunks] == [
        {
            "chunk_id": "doc_a_p1_c0001",
            "vector_id": 0,
            "doc_id": "doc_a",
            "source": "a.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Alpha beta gamma.",
            "token_count": 4,
        },
        {
            "chunk_id": "doc_b_p1_c0001",
            "vector_id": 1,
            "doc_id": "doc_b",
            "source": "b.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "One two three. Four five six.",
            "token_count": 8,
        },
        {
            "chunk_id": "doc_b_p1_c0002",
            "vector_id": 2,
            "doc_id": "doc_b",
            "source": "b.pdf",
            "page_start": 1,
            "page_end": 2,
            "text": "Four five six. Seven eight nine.",
            "token_count": 8,
        },
    ]


def test_flat_ip_round_trip_preserves_explicit_vector_ids(tmp_path) -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.5, 0.5], [-1.0, 0.0]], dtype=np.float32)
    ids = np.asarray([101, 9001, 42], dtype=np.int64)
    index = FlatIPIndex(backend="numpy")
    index.build(embeddings, ids=ids)
    path = tmp_path / "index.npz"
    index.save(path)

    loaded = FlatIPIndex(backend="numpy")
    loaded.load(path)
    hits = loaded.search_hits(np.asarray([[1.0, 0.0]], dtype=np.float32), 3)
    assert [hit.vector_id for hit in hits] == [101, 9001, 42]


def test_dense_retriever_respects_default_and_override_top_k() -> None:
    chunks = [
        {"chunk_id": "c0", "vector_id": 0, "doc_id": "a", "source": "a.pdf", "page_start": 1, "page_end": 1, "text": "A", "token_count": 1},
        {"chunk_id": "c1", "vector_id": 1, "doc_id": "b", "source": "b.pdf", "page_start": 2, "page_end": 2, "text": "B", "token_count": 1},
    ]
    index = FlatIPIndex(backend="numpy")
    index.build(np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32))
    retriever = DenseRetriever(chunks, StaticEmbedder(), index, top_k=2)

    default = retriever.retrieve_trace("query")
    override = retriever.retrieve_trace("query", top_k=1)
    assert [hit.chunk.chunk_id for hit in default.results] == ["c0", "c1"]
    assert [hit.chunk.chunk_id for hit in override.results] == ["c0"]


def test_prompt_is_fixed_and_citation_aware() -> None:
    context = ContextPackage(text="[Chunk 1]\nEvidence.", results=(), token_count=3, truncated=False, builder="test")
    first = build_prompt("Question?", context)
    second = build_prompt("Question?", context)
    assert first == second
    assert "Cite supporting chunks with [Chunk N]" in first.text
    assert first.sha256 == hashlib.sha256(first.text.encode()).hexdigest()


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_dense_retriever_rejects_invalid_top_k(top_k) -> None:
    index = FlatIPIndex(backend="numpy")
    index.build(np.eye(1, 2, dtype=np.float32))
    retriever = DenseRetriever(
        [{"chunk_id": "c", "vector_id": 0, "doc_id": "p", "source": "p.pdf", "page_start": 1, "page_end": 1, "text": "A", "token_count": 1}],
        StaticEmbedder(),
        index,
        top_k=1,
    )
    with pytest.raises(ValueError):
        retriever.retrieve_trace("query", top_k=top_k)
