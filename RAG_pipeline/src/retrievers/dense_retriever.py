from __future__ import annotations

import time

from src.embedders.sbert_embedder import TextEmbedder
from src.indexes.faiss_index import FlatIPIndex


class DenseRetriever:
    def __init__(self, chunks: list[dict], embedder: TextEmbedder, index: FlatIPIndex, top_k: int = 5):
        self.chunks = chunks
        self.embedder = embedder
        self.index = index
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[dict], float]:
        started = time.perf_counter()
        query_embedding = self.embedder.encode([query])
        scores, indices = self.index.search(query_embedding, int(top_k or self.top_k))
        latency_ms = (time.perf_counter() - started) * 1000

        results: list[dict] = []
        for rank, (score, index) in enumerate(zip(scores, indices), start=1):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                    "source": chunk["source"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                    "token_count": chunk.get("token_count", 0),
                }
            )
        return results, latency_ms

