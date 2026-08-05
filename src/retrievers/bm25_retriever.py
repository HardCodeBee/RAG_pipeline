"""Memory-mapped BM25S retrieval over the canonical chunk-id sequence."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.persistence.artifact_io import close_numpy_memmap
from src.records import ChunkRecord, RetrievalTrace, SearchHit
from src.retrievers.chunk_store import ChunkStore, as_chunk_store


class BM25Retriever:
    """Tokenize a query, search BM25S, and map row ids back to chunks."""

    def __init__(
        self,
        chunks: ChunkStore | Iterable[ChunkRecord | dict],
        model: Any,
        tokenizer: Any,
        *,
        top_k: int = 5,
        search_threads: int = 0,
        sparse_index_id: str | None = None,
    ):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if (
            isinstance(search_threads, bool)
            or not isinstance(search_threads, int)
            or search_threads < 0
        ):
            raise ValueError("search_threads must be a non-negative integer")
        scores = getattr(model, "scores", None)
        if not isinstance(scores, dict) or not isinstance(scores.get("num_docs"), int):
            raise ValueError("BM25 model must be built or loaded before creating a retriever")

        store = as_chunk_store(chunks)
        if len(store) != scores["num_docs"]:
            raise ValueError(
                f"Chunk count {len(store)} does not match BM25 document count "
                f"{scores['num_docs']}"
            )
        self.chunk_store = store
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.search_threads = search_threads
        self.sparse_index_id = sparse_index_id

    @classmethod
    def load(
        cls,
        chunks: ChunkStore | Iterable[ChunkRecord | dict],
        index_dir: str | Path,
        *,
        top_k: int = 5,
        search_threads: int = 0,
        mmap: bool = True,
        sparse_index_id: str | None = None,
    ) -> "BM25Retriever":
        """Load the pinned BM25S arrays and tokenizer vocabulary from disk."""

        try:
            import bm25s
            from bm25s.tokenization import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BM25 retrieval requires bm25s; install requirements/experiment.txt"
            ) from exc

        directory = Path(index_dir).resolve()
        model = bm25s.BM25.load(
            directory,
            load_corpus=False,
            mmap=mmap,
            allow_pickle=False,
        )
        tokenizer = Tokenizer(lower=True, stopwords=[])
        tokenizer.load_vocab(directory)
        tokenizer.load_stopwords(directory)
        return cls(
            chunks,
            model,
            tokenizer,
            top_k=top_k,
            search_threads=search_threads,
            sparse_index_id=sparse_index_id,
        )

    def retrieve_trace(
        self,
        query: str,
        top_k: int | None = None,
        *,
        search_params: Mapping[str, Any] | None = None,
    ) -> RetrievalTrace:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        effective_top_k = self.top_k if top_k is None else top_k
        if (
            isinstance(effective_top_k, bool)
            or not isinstance(effective_top_k, int)
            or effective_top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer")
        if search_params:
            raise ValueError("BM25 retrieval does not accept ANN search parameters")

        if effective_top_k == 0:
            return RetrievalTrace(
                top_k=0,
                results=(),
                timings_ms={
                    "query_tokenization_ms": 0.0,
                    "sparse_search_ms": 0.0,
                    "chunk_mapping_ms": 0.0,
                    "total_ms": 0.0,
                },
            )

        started = time.perf_counter()
        tokenization_started = time.perf_counter()
        query_tokens = self.tokenizer.tokenize(
            [query.strip()],
            update_vocab="never",
            show_progress=False,
        )
        tokenization_ms = (time.perf_counter() - tokenization_started) * 1000

        search_started = time.perf_counter()
        returned = self.model.retrieve(
            query_tokens,
            k=min(effective_top_k, len(self.chunk_store)),
            sorted=True,
            return_as="tuple",
            show_progress=False,
            n_threads=self.search_threads,
        )
        sparse_search_ms = (time.perf_counter() - search_started) * 1000

        documents = returned.documents[0]
        scores = returned.scores[0]
        ranked = sorted(
            (
                (int(vector_id), float(score))
                for vector_id, score in zip(documents, scores)
                if math.isfinite(float(score)) and float(score) > 0.0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        vector_ids = [vector_id for vector_id, _ in ranked]
        if len(vector_ids) != len(set(vector_ids)):
            raise ValueError("BM25 index returned duplicate document ids")

        mapping_started = time.perf_counter()
        try:
            records = self.chunk_store.get_many(vector_ids)
        except KeyError as exc:
            raise ValueError(f"BM25 index returned unknown vector id: {exc.args[0]}") from exc
        results = tuple(
            SearchHit(rank=rank, chunk=chunk, score=score)
            for rank, ((_, score), chunk) in enumerate(zip(ranked, records), start=1)
        )
        mapping_ms = (time.perf_counter() - mapping_started) * 1000

        return RetrievalTrace(
            top_k=effective_top_k,
            results=results,
            timings_ms={
                "query_tokenization_ms": tokenization_ms,
                "sparse_search_ms": sparse_search_ms,
                "chunk_mapping_ms": mapping_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        )

    def close(self) -> None:
        scores = getattr(self.model, "scores", {})
        if isinstance(scores, dict):
            for value in scores.values():
                close_numpy_memmap(value)
