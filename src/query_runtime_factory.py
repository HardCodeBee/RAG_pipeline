"""查询期 generator 与 reranker 的显式工厂。"""

from __future__ import annotations

# Query-only generator and reranker assembly.

from typing import Any

from src.generators.answer_generator import LLMGenerator
from src.persistence.artifact_validation import VerifiedBuild
from src.rerankers.cross_encoder import CrossEncoderReranker
from src.rerankers.noop import NoOpReranker
from src.retrievers.bm25_index import resolve_bm25_index
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever


def create_generator(config: dict[str, Any]):
    generation = config["generation"]
    return LLMGenerator(
        provider=generation["provider"],
        model=generation.get("model"),
        temperature=generation.get("temperature", 0.0),
        max_output_tokens=generation["max_output_tokens"],
        timeout_seconds=generation.get("timeout_seconds", 60.0),
        max_retries=generation.get("max_retries", 0),
    )


def create_reranker(config: dict[str, Any]):
    reranker = config["retrieval"]["reranker"]
    if reranker["provider"] == "none":
        return NoOpReranker()
    if reranker["provider"] != "cross_encoder":
        raise ValueError(f"Unsupported reranker: {reranker['provider']}")
    device = reranker["device"]
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoderReranker(
        model_name=reranker["model_name"],
        revision=reranker["revision"],
        device=device,
        batch_size=reranker["batch_size"],
        local_files_only=reranker["local_files_only"],
    )


def create_retriever(
    config: dict[str, Any],
    chunks,
    *,
    embedder=None,
    index=None,
    verified_build: VerifiedBuild | None = None,
):
    """Create exactly the retriever selected by the validated run config."""

    method = config["retrieval"]["method"]
    if method == "dense":
        if embedder is None or index is None:
            raise ValueError("Dense retrieval requires a query embedder and vector index")
        return DenseRetriever(
            chunks,
            embedder,
            index,
            top_k=config["retrieval"]["candidate_k"],
        )
    if method != "bm25":
        raise ValueError(f"Unsupported retriever: {method}")
    if verified_build is None:
        raise ValueError("BM25 retrieval requires a verified source build")
    verified_sparse_index = resolve_bm25_index(config, verified_build)
    return BM25Retriever.load(
        chunks,
        verified_sparse_index.directory,
        top_k=config["retrieval"]["candidate_k"],
        search_threads=config["retrieval"]["search_threads"],
        mmap=config["bm25"]["mmap"],
        sparse_index_id=verified_sparse_index.manifest["sparse_index_id"],
    )
