from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.chunkers.fixed_chunker import chunk_records
from src.config import ensure_parent, resolve_path
from src.embedders.sbert_embedder import TextEmbedder
from src.evaluators.metrics import answer_contains_gold, expected_source_hit
from src.generators.llm_generator import LLMGenerator
from src.indexes.faiss_index import FlatIPIndex
from src.io_utils import read_jsonl, write_jsonl
from src.loaders.pdf_loader import load_pdfs
from src.retrievers.dense_retriever import DenseRetriever


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle) 


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def build_index(config: dict[str, Any]) -> dict[str, Any]:
    corpus_path = resolve_path(config, config["corpus"]["path"])
    chunks_path = resolve_path(config, config["data"]["chunks_path"])
    embeddings_path = resolve_path(config, config["data"]["embeddings_path"])
    index_path = resolve_path(config, config["index"]["save_path"])
    manifest_path = resolve_path(config, config["data"]["manifest_path"])

    started = time.perf_counter()
    page_records = load_pdfs(corpus_path)
    if not page_records:
        raise RuntimeError(f"No PDF text records found in corpus path: {corpus_path}")

    chunk_config = config.get("chunking", {})
    chunks = chunk_records(
        page_records,
        chunk_size_tokens=int(chunk_config.get("chunk_size_tokens", 300)),
        chunk_overlap_tokens=int(chunk_config.get("chunk_overlap_tokens", 50)),
    )
    if not chunks:
        raise RuntimeError("No chunks were produced from the corpus")

    write_jsonl(chunks_path, chunks)

    embedder = TextEmbedder.from_config(config)
    embeddings = embedder.encode([chunk["text"] for chunk in chunks])
    ensure_parent(embeddings_path)
    np.save(embeddings_path, embeddings)

    index = FlatIPIndex(backend=config.get("index", {}).get("backend", "auto"))
    index.build(embeddings)
    index.save(index_path)

    elapsed_ms = (time.perf_counter() - started) * 1000
    manifest = {
        "run_id": _now_id(),
        "corpus_path": str(corpus_path),
        "num_documents": len({record["doc_id"] for record in page_records}),
        "num_pages": len(page_records),
        "num_chunks": len(chunks),
        "embedding": asdict(embedder.info()),
        "index": {
            "backend": index.backend,
            "type": config.get("index", {}).get("type", "flat_ip"),
            "path": str(index_path),
        },
        "chunks_path": str(chunks_path),
        "embeddings_path": str(embeddings_path),
        "build_latency_ms": elapsed_ms,
    }
    _write_manifest(manifest_path, manifest)
    return manifest


class NaiveRAGPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.chunks_path = resolve_path(config, config["data"]["chunks_path"])
        self.embeddings_path = resolve_path(config, config["data"]["embeddings_path"])
        self.index_path = resolve_path(config, config["index"]["save_path"])
        self.manifest_path = resolve_path(config, config["data"]["manifest_path"])
        self.manifest = _read_manifest(self.manifest_path)

        self.chunks = list(read_jsonl(self.chunks_path))
        self.embedder = self._load_embedder()
        self.index = FlatIPIndex(backend=config.get("index", {}).get("backend", "auto"))
        self.index.load(self.index_path, embeddings_path=self.embeddings_path)
        self.retriever = DenseRetriever(
            self.chunks,
            self.embedder,
            self.index,
            top_k=int(config.get("retrieval", {}).get("top_k", 5)),
        )
        self.generator = LLMGenerator.from_config(config)

    def _load_embedder(self) -> TextEmbedder:
        embedding_config = dict(self.config.get("embedding", {}))
        manifest_embedding = self.manifest.get("embedding", {})
        if embedding_config.get("backend", "auto") == "auto" and manifest_embedding.get("backend") == "hashing":
            embedding_config["backend"] = "hashing"
            embedding_config["fallback_dim"] = manifest_embedding.get(
                "dimension",
                embedding_config.get("fallback_dim", 384),
            )

        config = dict(self.config)
        config["embedding"] = embedding_config
        return TextEmbedder.from_config(config)

    def answer_question(
        self,
        question: str,
        question_id: str | None = None,
        gold_answer: str | None = None,
        expected_sources: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        retrieved, retrieval_latency_ms = self.retriever.retrieve(question, top_k=top_k)
        prompt, generation = self.generator.generate(question, retrieved)
        total_latency_ms = (time.perf_counter() - total_started) * 1000

        return {
            "question_id": question_id or f"query_{_now_id()}",
            "question": question,
            "gold_answer": gold_answer,
            "expected_sources": expected_sources or [],
            "retrieval": {
                "top_k": int(top_k or self.config.get("retrieval", {}).get("top_k", 5)),
                "latency_ms": retrieval_latency_ms,
                "results": retrieved,
            },
            "prompt": prompt,
            "generation": {
                "answer": generation.answer,
                "provider": generation.provider,
                "model": generation.model,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "latency_ms": generation.latency_ms,
            },
            "total_latency_ms": total_latency_ms,
            "metrics": {
                "retrieval_expected_source_hit": expected_source_hit(retrieved, expected_sources),
                "answer_contains_gold": answer_contains_gold(generation.answer, gold_answer),
            },
            "notes": "",
        }

