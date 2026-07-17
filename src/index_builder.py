"""构建当前架构的不可变文本块、向量和索引产物目录。"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from src.artifact_io import artifact_descriptor, validate_build_directory, write_manifest
from src.components import create_chunker, create_embedder, create_index, create_loader, create_token_counter
from src.config import validate_config
from src.core.records import ChunkRecord, PageRecord
from src.io_utils import write_jsonl
from src.provenance import (
    build_identity,
    corpus_inventory,
    environment_versions,
    git_state,
    json_sha256,
    PIPELINE_SCHEMA_VERSION,
    recorded_config,
    resolved_roots,
    source_code_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _verify_exact_index(index, embeddings: np.ndarray, vector_ids: np.ndarray) -> None:
    # 用第一条 embedding 当查询，和 NumPy 参考实现比对，确认索引排序正确。
    query = embeddings[0:1]
    top_k = min(10, len(embeddings))
    expected_scores = embeddings @ query[0]
    # 分数相同时用 vector_id 打破平局，保证验证结果稳定。
    expected_positions = np.lexsort((vector_ids, -expected_scores))[:top_k]
    expected_ranked_scores = expected_scores[expected_positions]
    expected_ids = vector_ids[expected_positions].tolist()
    hits = index.search_hits(query, top_k)
    actual_scores = np.asarray([hit.score for hit in hits], dtype=np.float32)
    actual_ids = [hit.vector_id for hit in hits]
    if len(hits) != top_k or len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError("Exact index verification returned invalid vector ids")
    if actual_ids != expected_ids:
        raise RuntimeError("Exact index search does not match the NumPy reference vector ids")
    if not np.allclose(actual_scores, expected_ranked_scores, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Exact index search does not match the NumPy reference scores")


def _overlap_stats(chunks: list[ChunkRecord], token_counter) -> dict[str, Any]:
    # 统计相邻 chunk 实际共享了多少 token，用于 manifest 记录 chunking 质量。
    values = []
    for left, right in zip(chunks, chunks[1:]):
        # 不同文档之间不计算 overlap。
        if left.doc_id != right.doc_id:
            continue
        left_tokens = token_counter.token_sequence(left.text)
        right_tokens = token_counter.token_sequence(right.text)
        limit = min(len(left_tokens), len(right_tokens))
        overlap = 0
        for size in range(1, limit + 1):
            # 找到 left 结尾和 right 开头最长一致 token 序列。
            if left_tokens[-size:] == right_tokens[:size]:
                overlap = size
        values.append(overlap)
    if not values:
        return {"pairs": 0, "min": 0, "mean": 0.0, "median": 0.0, "max": 0, "zero_count": 0}
    return {
        "pairs": len(values),
        "min": min(values),
        "mean": mean(values),
        "median": median(values),
        "max": max(values),
        "zero_count": sum(value == 0 for value in values),
    }


def _load_pages(loader: Any, corpus_path: Path) -> tuple[list[PageRecord], float]:
    load_started = time.perf_counter()
    loaded = loader.load(corpus_path, "pdf")
    # loader 可以返回 PageRecord 或 dict，这里统一成 PageRecord。
    pages = [item if isinstance(item, PageRecord) else PageRecord.from_mapping(item) for item in loaded]
    load_ms = (time.perf_counter() - load_started) * 1000
    if not pages:
        raise RuntimeError("The corpus produced no extractable PDF pages")
    return pages, load_ms


def _validate_chunks(chunks: list[ChunkRecord], token_budget: int) -> None:
    if not chunks:
        raise RuntimeError("No chunks were produced from the corpus")
    # chunk_id 和 vector_id 是后续检索映射的核心键，构建时必须强校验。
    if len({item.chunk_id for item in chunks}) != len(chunks):
        raise RuntimeError("Chunk ids must be unique")
    if [item.vector_id for item in chunks] != list(range(len(chunks))):
        raise RuntimeError("Vector ids must be the zero-based chunk sequence")
    if any(item.token_count <= 0 or item.token_count > token_budget for item in chunks):
        raise RuntimeError("Chunk token counts must be positive and within the configured budget")


def _create_chunks(
    config: dict[str, Any], pages: list[PageRecord]
) -> tuple[Any, list[ChunkRecord], float]:
    token_counter = create_token_counter(config)
    chunker = create_chunker(config, token_counter)
    chunk_started = time.perf_counter()
    produced = chunker.chunk(pages)
    # chunker 输出同样统一成文本块记录，保证后续产物结构稳定。
    chunks = [item if isinstance(item, ChunkRecord) else ChunkRecord.from_mapping(item) for item in produced]
    chunk_ms = (time.perf_counter() - chunk_started) * 1000
    _validate_chunks(chunks, config["chunking"]["chunk_size_tokens"])
    return token_counter, chunks, chunk_ms


def _create_embeddings(
    config: dict[str, Any], chunks: list[ChunkRecord], embeddings_path: Path
) -> tuple[Any, np.ndarray, float]:
    embedding_started = time.perf_counter()
    embedder = create_embedder(config)
    # 文档侧 embedding 必须用 encode_documents()，这样文档前缀才会生效。
    embeddings = np.asarray(embedder.encode_documents([item.text for item in chunks]), dtype=np.float32)
    embedding_ms = (time.perf_counter() - embedding_started) * 1000
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks) or embeddings.shape[1] <= 0:
        raise RuntimeError("Embedding shape does not match the chunk artifact")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embeddings contain non-finite values")
    np.save(embeddings_path, embeddings)
    return embedder, embeddings, embedding_ms


def _build_index_artifact(
    config: dict[str, Any], chunks: list[ChunkRecord], embeddings: np.ndarray, staging: Path
) -> tuple[Any, np.ndarray, Path, float]:
    index_started = time.perf_counter()
    index = create_index(config)
    vector_ids = np.asarray([item.vector_id for item in chunks], dtype=np.int64)
    # 把 ChunkRecord.vector_id 显式写入索引，避免依赖行号隐式约定。
    index.build(embeddings, ids=vector_ids)
    if config["strict_backends"] and index.backend != config["index"]["backend"]:
        raise RuntimeError("Strict build resolved to an unexpected index backend")
    # auto 后端要等实际解析完成后再决定扩展名，避免 FAISS 内容被命名为 .npz。
    index_path = staging / ("index.faiss" if index.backend == "faiss" else "index.npz")
    index.save(index_path)
    index_ms = (time.perf_counter() - index_started) * 1000
    return index, vector_ids, index_path, index_ms


def _verify_saved_index(
    config: dict[str, Any],
    index: Any,
    index_path: Path,
    embeddings_path: Path,
    embeddings: np.ndarray,
    vector_ids: np.ndarray,
    chunk_count: int,
) -> None:
    verified = create_index(config, backend=index.backend, index_type=index.index_type)
    verified.load(index_path, embeddings_path=embeddings_path)
    # 保存后立刻重新加载并查询，提前发现 artifact 写入或 backend 兼容问题。
    if verified.count != chunk_count or verified.dimension != embeddings.shape[1]:
        raise RuntimeError("Reloaded index metadata does not match chunks and embeddings")
    if verified.ids is None or set(verified.ids.tolist()) != set(vector_ids.tolist()):
        raise RuntimeError("Reloaded index vector ids do not match chunks")
    _verify_exact_index(verified, embeddings, vector_ids)


def _ensure_corpus_unchanged(
    loader: Any, corpus_path: Path, expected_corpus: dict[str, Any]
) -> None:
    current_documents = loader.discover(corpus_path, "pdf")
    # 构建期间语料文件如果变化，当前构建身份就不再可信。
    if corpus_inventory(current_documents, corpus_path) != expected_corpus:
        raise RuntimeError("Corpus files changed while the index was being built")


def _manifest_artifacts(
    chunks_path: Path,
    embeddings_path: Path,
    index_path: Path,
    chunks: list[ChunkRecord],
    embeddings: np.ndarray,
) -> dict[str, Any]:
    return {
        "chunks": artifact_descriptor(chunks_path, rows=len(chunks)),
        "embeddings": {
            **artifact_descriptor(embeddings_path, rows=len(chunks)),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
        },
        "index": artifact_descriptor(index_path, rows=len(chunks)),
    }


def _create_manifest(
    *,
    config: dict[str, Any],
    build_id: str,
    build_spec_sha: str,
    spec: dict[str, Any],
    source_sha: str,
    corpus: dict[str, Any],
    documents: list[Path],
    pages: list[PageRecord],
    chunks: list[ChunkRecord],
    token_counter: Any,
    embedder: Any,
    embeddings: np.ndarray,
    index: Any,
    vector_ids: np.ndarray,
    chunks_path: Path,
    embeddings_path: Path,
    index_path: Path,
    timings: dict[str, float],
    started: float,
) -> dict[str, Any]:
    token_counts = [item.token_count for item in chunks]
    embedding_space = embedder.embedding_space("inner_product")
    # manifest 是构建的完整说明书，查询阶段会用它校验产物。
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "status": "complete",
        "build_id": build_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_spec_sha256": build_spec_sha,
        "build_spec": spec,
        "effective_config": recorded_config(config),
        "corpus": {
            **corpus,
            "num_files": len(documents),
            "num_pages": len(pages),
            "num_documents": len({item.doc_id for item in pages}),
        },
        "chunking": {
            **config["chunking"],
            "num_chunks": len(chunks),
            "token_count": {
                "min": min(token_counts),
                "mean": mean(token_counts),
                "max": max(token_counts),
            },
            "realized_overlap_tokens": _overlap_stats(chunks, token_counter),
        },
        "embedding": {
            "space": embedding_space.to_dict(),
            "count": len(chunks),
        },
        "index": {
            "requested_backend": config["index"]["backend"],
            "backend": index.backend,
            "type": index.index_type,
            "count": index.count,
            "dimension": index.dimension,
            "metric": "inner_product",
        },
        "vector_id_sequence_sha256": json_sha256(vector_ids.tolist()),
        "artifacts": _manifest_artifacts(chunks_path, embeddings_path, index_path, chunks, embeddings),
        "source_sha256": source_sha,
        "git": git_state(PROJECT_ROOT),
        "environment": environment_versions(),
        "timings_ms": {
            **timings,
            "total_before_commit": (time.perf_counter() - started) * 1000,
        },
    }


def _commit_build(staging: Path, build_dir: Path, build_id: str, build_spec_sha: str) -> dict[str, Any]:
    if build_dir.exists():
        # 并发构建时，可能另一个进程已经提交了同一个 build。
        existing = validate_build_directory(build_dir, build_id)
        if existing.get("build_spec_sha256") != build_spec_sha:
            raise RuntimeError("Concurrent build produced an incompatible build directory")
        return existing
    # 原子提交：staging 完整后才变成正式 build 目录。
    os.replace(staging, build_dir)
    return validate_build_directory(build_dir, build_id)


def build_index(config: dict[str, Any]) -> dict[str, Any]:
    # build_index 接收已加载或手工构造的配置，先统一校验并补默认值。
    config = validate_config(config)
    if "_base_dir" not in config:
        raise ValueError("config must include _base_dir; use load_config()")
    roots = resolved_roots(config)
    loader = create_loader(config)
    # 先发现一次语料文件，用于构建语料清单和构建身份标识。
    documents = loader.discover(roots["corpus"], "pdf")
    if not documents:
        raise RuntimeError(f"No PDF files found in corpus path: {roots['corpus']}")
    corpus = corpus_inventory(documents, roots["corpus"])
    source_sha = source_code_sha256(PROJECT_ROOT)
    build_id, build_spec_sha, spec = build_identity(config, corpus, source_sha)
    artifacts_root = roots["artifacts_root"]
    build_dir = artifacts_root / build_id
    if build_dir.exists():
        # 同一个 build_spec 对应同一个不可变目录；存在时直接复用。
        manifest = validate_build_directory(build_dir, build_id)
        if manifest.get("build_spec_sha256") != build_spec_sha or manifest.get("build_spec") != spec:
            raise ValueError("Existing build directory does not match the requested build spec")
        return manifest

    artifacts_root.mkdir(parents=True, exist_ok=True)
    # 先写 staging 目录，全部成功后再 os.replace 到最终 build_dir。
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}-", dir=artifacts_root))
    started = time.perf_counter()
    try:
        pages, load_ms = _load_pages(loader, roots["corpus"])
        token_counter, chunks, chunk_ms = _create_chunks(config, pages)

        chunks_path = staging / "chunks.jsonl"
        embeddings_path = staging / "embeddings.npy"
        # chunks.jsonl 是检索结果映射回文本和来源的主产物。
        write_jsonl(chunks_path, [item.to_dict() for item in chunks])

        embedder, embeddings, embedding_ms = _create_embeddings(config, chunks, embeddings_path)
        index, vector_ids, index_path, index_ms = _build_index_artifact(config, chunks, embeddings, staging)
        _verify_saved_index(
            config, index, index_path, embeddings_path, embeddings, vector_ids, len(chunks)
        )
        _ensure_corpus_unchanged(loader, roots["corpus"], corpus)

        timings = {
            "pdf_loading": load_ms,
            "chunking": chunk_ms,
            "embedding": embedding_ms,
            "index_build_and_save": index_ms,
        }
        manifest = _create_manifest(
            config=config,
            build_id=build_id,
            build_spec_sha=build_spec_sha,
            spec=spec,
            source_sha=source_sha,
            corpus=corpus,
            documents=documents,
            pages=pages,
            chunks=chunks,
            token_counter=token_counter,
            embedder=embedder,
            embeddings=embeddings,
            index=index,
            vector_ids=vector_ids,
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            index_path=index_path,
            timings=timings,
            started=started,
        )
        write_manifest(staging / "manifest.json", manifest)
        return _commit_build(staging, build_dir, build_id, build_spec_sha)
    finally:
        # 成功提交后 staging 路径已不存在；失败时清理临时目录。
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
