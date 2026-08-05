"""查询期 RAG 主链路。"""

from __future__ import annotations

# End-to-end query orchestration belongs at the source root.

import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.persistence.artifact_io import close_numpy_memmap, iter_jsonl
from src.persistence.artifact_validation import VerifiedBuild, validate_build_directory
from src.config import validate_config
from src.records import ChunkRecord, EmbeddingSpaceSpec, RetrievalTrace
from src.vector_index_factory import create_index
from src.query_runtime_factory import create_generator, create_reranker, create_retriever
from src.prompts.fixed_prompt import build_prompt
from src.provenance import (
    build_identity,
    json_sha256,
    resolved_roots,
    run_spec,
    source_group_sha256,
    source_snapshot_sha256,
    zero_based_sequence_sha256,
)
from src.retrievers.chunk_store import InMemoryChunkStore, JsonlOffsetChunkStore
from src.rerankers.reranker_contract import RerankTrace
from src.context_builders.ranked_concat_context import build_context
from src.query_plan import QueryPlan, fixed_query_plan
from src.encoded_corpus_factory import (
    create_embedder,
    create_loader,
    create_token_counter,
    discover_corpus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 当没有传入question_id 自动生成唯一查询ID
def _query_id() -> str:
    # query_id 带 UTC 时间和随机后缀，便于日志中区分每次查询。
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"query_{timestamp}_{uuid.uuid4().hex[:8]}"

class NaiveRAGPipeline:
    """加载一个不可变 build，并用显式基线链路回答问题。"""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        verified_build: VerifiedBuild | None = None,
    ):
        # 初始化阶段只加载一个不可变构建，并校验它是否和当前配置匹配。
        self.config = validate_config(config)
        # 要求 config 必须来自 load_config()
        if "_base_dir" not in self.config:
            raise ValueError("config must include _base_dir; use load_config()")

        roots = resolved_roots(self.config)
        
        # 根据 config 创建 loader
        loader = create_loader(self.config)
        # 重新计算构建身份，找到当前配置应该对应的 build_dir。
        if verified_build is None:
            _, corpus = discover_corpus(loader, roots["corpus"])
        else:
            if not isinstance(verified_build, VerifiedBuild):
                raise TypeError("verified_build must be a VerifiedBuild")
            corpus = verified_build.manifest.get("build_spec", {}).get("corpus")
            if not isinstance(corpus, dict):
                raise ValueError("Verified build has no corpus specification")
        build_source_sha = source_group_sha256(PROJECT_ROOT, "build")
        # 重新算出当前配置“应该使用哪个 build”。
        build_id, build_spec_sha, build_spec_value = build_identity(
            self.config, 
            corpus, 
            build_source_sha)
        expected_build_dir = (roots["artifacts_root"] / build_id).resolve()
        self.build_dir = (
            expected_build_dir
            if verified_build is None
            else verified_build.directory.resolve()
        )
        if self.build_dir != expected_build_dir:
            raise ValueError("Verified build directory does not match the active config")
        # query 阶段不重建索引，只接受已经完整校验过的 build 目录。
        # 检查这个 build 目录是否存在、manifest 是否完整、chunk/embedding/index 文件 hash 是否匹配检查这个 build 目录是否存在、
        # manifest 是否完整、chunk/embedding/index 文件 hash 是否匹配
        # 然后返回完整字典
        self.verified_build = (
            validate_build_directory(self.build_dir, build_id)
            if verified_build is None
            else verified_build
        )
        self.manifest = self.verified_build.manifest
        if self.manifest.get("build_spec_sha256") != build_spec_sha:
            raise ValueError("The active build specification does not match the immutable build directory")
        if self.manifest.get("build_spec") != build_spec_value:
            raise ValueError("The active build specification payload differs from the build manifest")

        # 找到并拼出实际文件路径
        artifacts = self.manifest["artifacts"]
        chunks_path = self.build_dir / artifacts["chunks"]["file"]
        embeddings_path = self.build_dir / artifacts["embeddings"]["file"]
        index_path = (
            self.build_dir / artifacts["index"]["file"]
            if self.manifest["index"]["backend"] == "faiss"
            else embeddings_path
        )
        
        offsets_descriptor = artifacts.get("chunk_offsets")
        if offsets_descriptor is None:
            self.chunk_store = InMemoryChunkStore(
                ChunkRecord.from_mapping(row) for row in iter_jsonl(chunks_path)
            )
        else:
            self.chunk_store = JsonlOffsetChunkStore(
                chunks_path,
                self.build_dir / offsets_descriptor["file"],
                expected_rows=artifacts["chunks"]["rows"],
            )
        # Compatibility alias: this is now a lazy store for encoded-corpus builds.
        self.chunks = self.chunk_store
        self._validate_chunks(embeddings_path)

        # 创建 token 计数器
        self.token_counter = create_token_counter(self.config)
        self.embedder = None
        self.index = None
        if self.config["retrieval"]["method"] == "dense":
            # Dense query startup alone loads the embedding model and vector index.
            self.embedder = self._load_query_embedder()
            manifest_index = self.manifest["index"]
            self.index = create_index(
                self.config,
                backend=manifest_index["backend"],
                index_type=manifest_index["type"],
                threads=self.config["retrieval"]["search_threads"]
                or self.config["index"]["faiss_threads"],
            )
            self._expected_index_build_params = self.index.build_params
            self.index.load(index_path)
            self._validate_loaded_index()

        self.retriever = create_retriever(
            self.config,
            self.chunk_store,
            embedder=self.embedder,
            index=self.index,
            verified_build=self.verified_build,
        )
        self.reranker = create_reranker(self.config)
        self.generator = create_generator(self.config)

        # 记录本次run的运行规格
        run_source_sha = source_group_sha256(PROJECT_ROOT, "run")
        source_snapshot_sha = source_snapshot_sha256(PROJECT_ROOT)
        run_spec_value = run_spec(self.config, build_id, run_source_sha)
        # runtime_metadata keeps the minimal run identity written to metadata.json.
        self.runtime_metadata = {
            "build_id": build_id,
            "build_dir": str(self.build_dir.resolve()),
            "build_spec_sha256": build_spec_sha,
            "source_snapshot_sha256": source_snapshot_sha,
            "build_source_sha256": build_source_sha,
            "run_source_sha256": run_source_sha,
            "run_spec": run_spec_value,
            "run_spec_sha256": json_sha256(run_spec_value),
        }
        sparse_index_id = getattr(self.retriever, "sparse_index_id", None)
        if sparse_index_id is not None:
            self.runtime_metadata["sparse_index_id"] = sparse_index_id

    # 确认 chunks.jsonl 和 embeddings.npy 以及 manifest.json 三者完全对得上
    def _validate_chunks(self, embeddings_path: Path) -> None:
        descriptor = self.manifest["artifacts"]["chunks"]
        count = len(self.chunk_store)
        if (
            descriptor.get("rows") != count
            or self.manifest["chunking"].get("num_chunks") != count
        ):
            raise ValueError("Chunk count does not match the build manifest")
        if isinstance(self.chunk_store, InMemoryChunkStore):
            records = self.chunk_store.get_many(list(range(count)))
            chunk_ids = [item.chunk_id for item in records]
            if len(chunk_ids) != len(set(chunk_ids)):
                raise ValueError("Chunk ids must be unique")
        if self.manifest.get("vector_id_sequence_sha256") != zero_based_sequence_sha256(count):
            raise ValueError("Chunk vector ids do not match the build manifest")

        embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
        try:
            embedding_descriptor = self.manifest["artifacts"]["embeddings"]
            if list(embeddings.shape) != embedding_descriptor.get("shape"):
                raise ValueError("Embedding shape does not match the build manifest")
            if embeddings.ndim != 2 or embeddings.shape[0] != count:
                raise ValueError("Embedding rows do not match chunks")
            if str(embeddings.dtype) != embedding_descriptor.get("dtype"):
                raise ValueError("Embedding dtype does not match the build manifest")
            # New encoded-corpus builds validate every batch before commit and
            # validate the immutable artifact hash here. Legacy eager builds
            # retain the historical full finite-value scan.
            if (
                isinstance(self.chunk_store, InMemoryChunkStore)
                and not np.isfinite(embeddings).all()
            ):
                raise ValueError("Embedding artifact contains non-finite values")
        finally:
            close_numpy_memmap(embeddings)
    
    # 保证查询问题生成的 query embedding，和索引里已有的 document embedding 属于同一个向量空间
    # 构建时记录一份向量空间规格，查询时重新生成一份规格，然后逐字段比较
    def _load_query_embedder(self):
        built = self.manifest["embedding"]
        expected_space = EmbeddingSpaceSpec.from_mapping(built["space"])
        # 用构建 manifest 覆盖配置中可能变化的 embedding 字段，确保查询与索引同空间。
        override = {
            "backend": expected_space.backend,
            "model_name": expected_space.model_name,
            "revision": expected_space.revision,
            "normalize": expected_space.normalized,
            "dimension": expected_space.dimension,
            "document_prefix": expected_space.document_prefix,
            "max_sequence_length": expected_space.max_sequence_length,
        }
        embedder = create_embedder(self.config, override=override)
        actual_space = embedder.embedding_space(expected_space.similarity)
        if actual_space != expected_space:
            raise ValueError(
                "Query embedder does not match the build manifest: "
                f"{actual_space.to_dict()} != {expected_space.to_dict()}"
            )
        self.embedding_space_spec = actual_space
        return embedder

    # 索引文件已经 load 到内存之后
    # 确认它和 manifest、chunks、query embedder 都完全匹配
    def _validate_loaded_index(self) -> None:
        
        manifest_index = self.manifest["index"]
        
        # 加载后的索引元数据必须和 manifest 完全一致。
        if self.index.backend != manifest_index["backend"] or self.index.index_type != manifest_index["type"]:
            raise ValueError("Loaded index backend/type does not match the build manifest")
        if self.index.count != len(self.chunks) or self.index.count != manifest_index["count"]:
            raise ValueError("Loaded index count does not match chunks or manifest")
        if self.index.dimension != manifest_index["dimension"]:
            raise ValueError("Loaded index dimension does not match the build manifest")
        if self.index.dimension != self.embedding_space_spec.dimension:
            raise ValueError("Loaded index and embedding dimensions do not match")
        if self.index.build_params != self._expected_index_build_params:
            raise ValueError(
                "Loaded index build parameters do not match the requested build specification"
            )
        manifest_build_params = manifest_index.get("build_params")
        if (
            manifest_build_params is not None
            and self.index.build_params != manifest_build_params
        ):
            raise ValueError("Loaded index build parameters do not match the build manifest")
        
         # 校验集合和 hash，确保 id 序列和 build 阶段一致。
        if self.index.ids is None:
            raise ValueError("Loaded index does not expose vector ids")
        ids = np.asarray(self.index.ids, dtype=np.int64)
        expected_hash = self.manifest["vector_id_sequence_sha256"]
        if (
            ids.shape != (len(self.chunk_store),)
            or not np.array_equal(ids, np.arange(len(self.chunk_store), dtype=np.int64))
            or zero_based_sequence_sha256(len(self.chunk_store)) != expected_hash
        ):
            raise ValueError("Loaded index vector ids do not match chunks or manifest")

    def close(self) -> None:
        """Release query-time file mappings and closeable clients explicitly."""

        for value in (
            self.chunk_store,
            self.retriever,
            self.index,
            getattr(self.generator, "_client", None),
        ):
            close = getattr(value, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "NaiveRAGPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def default_query_plan(self, top_k: int | None = None) -> QueryPlan:
        retrieval = self.config["retrieval"]
        if top_k is None:
            final_k = retrieval["final_k"]
            candidate_k = retrieval["candidate_k"]
        else:
            final_k = top_k
            # A final-output override must not silently turn a configured
            # Top-50 -> Top-5 pipeline into Top-5 -> Top-5.  Preserve the
            # candidate pool, expanding it only when the requested final_k is
            # larger; top_k=0 remains the explicit retrieval gate.
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                candidate_k = top_k  # fixed_query_plan() raises the public error.
            elif top_k == 0:
                candidate_k = 0
            else:
                candidate_k = max(retrieval["candidate_k"], top_k)
        search_params = {
            key: retrieval[key]
            for key in ("nprobe", "ef_search", "max_codes")
            if key in retrieval
        }
        return fixed_query_plan(
            final_k,
            candidate_k=candidate_k,
            search_params=search_params if final_k else {},
        )

    def retrieve_with_plan_details(
        self,
        question: str,
        plan: QueryPlan,
    ) -> tuple[RetrievalTrace, RerankTrace | None]:
        if not isinstance(plan, QueryPlan):
            raise TypeError("plan must be a QueryPlan")
        raw = self.retriever.retrieve_trace(
            question,
            top_k=plan.candidate_k,
            search_params=plan.search_params,
        )
        if not plan.retrieval_enabled:
            return raw, None
        reranked = self.reranker.rerank(
            question,
            raw.results,
            final_k=plan.final_k,
        )
        timings = dict(raw.timings_ms)
        timings["rerank_ms"] = reranked.timing_ms
        timings["total_ms"] = timings.get("total_ms", 0.0) + reranked.timing_ms
        return (
            RetrievalTrace(
                top_k=plan.final_k,
                results=reranked.results,
                timings_ms=timings,
            ),
            reranked.trace,
        )

    def retrieve_with_plan(self, question: str, plan: QueryPlan) -> RetrievalTrace:
        retrieval, _ = self.retrieve_with_plan_details(question, plan)
        return retrieval

    def retrieve_with_details(
        self,
        question: str,
        top_k: int | None = None,
    ) -> tuple[RetrievalTrace, RerankTrace | None]:
        return self.retrieve_with_plan_details(question, self.default_query_plan(top_k))

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalTrace:
        retrieval, _ = self.retrieve_with_details(question, top_k)
        return retrieval

    def query(
        self,
        question: str,
        question_id: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        # 检查问题非空
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        # 记录耗时
        started = time.perf_counter()
        
        # 1. 检索相关 chunks
        retrieval, rerank_trace = self.retrieve_with_details(question, top_k=top_k)

        context_started = time.perf_counter() # 开始记录 context 构造耗时
        # 2. 把检索结果按词元预算拼成提示词上下文。
        context = build_context(
            retrieval.results,
            self.token_counter,
            self.config["context"]["max_tokens"],
        )
        context_latency_ms = (time.perf_counter() - context_started) * 1000 # context 构造耗时
        
        prompt_started = time.perf_counter() # 开始记录 prompt 构造耗时
        # 3. 使用固定模板构造最终 prompt
        prompt = build_prompt(question, context, self.config["prompt"]["version"])
        prompt_latency_ms = (time.perf_counter() - prompt_started) * 1000 # 记录 prompt 构造耗时
        
        # 4. 调用生成器 根据config选择
        generation = self.generator.generate_from_prompt(
            prompt.text, 
            question, 
            context.result_dicts()
            )
        
        # 把生成器返回对象统一转成 dict
        generation_data = asdict(generation)
        # 查询的 query 最终状态应该在顶层统一表示
        # Add only generator configuration to generation metadata.
        generation_data.update(
            {
                "temperature": self.generator.temperature,
                "max_output_tokens": self.generator.max_output_tokens,
            }
        )
        # 处理检索结果 把每个 SearchHit 转成普通 dict 方便写进 result
        retrieved_rows = [item.to_dict() for item in retrieval.results]
        logging = self.config["logging"]
        if not logging["save_retrieved_text"]:
            # 可关闭 chunk 正文保存，只保留来源和 id，减少输出体积。
            retrieved_rows = [{key: value for key, value in row.items() if key != "text"} for row in retrieved_rows]

        # 把查询的完整过程整理成一个结构化 result
        result = {
            "status": "success",
            "question_id": question_id or _query_id(),
            "question": question,
            "identity": {
                "build_id": self.runtime_metadata["build_id"],
                "run_spec_sha256": self.runtime_metadata["run_spec_sha256"],
            },
            "retrieval": {
                "top_k": retrieval.top_k,
                "results": retrieved_rows,
            },
            "context": {
                "builder": context.builder,
                "token_count": context.token_count,
                "num_chunks": len(context.results),
                "chunk_ids": [item.chunk.chunk_id for item in context.results],
                "truncated": context.truncated,
                "build_latency_ms": context_latency_ms,
            },
            "generation": generation_data,
            "prompt": {
                "template": prompt.template,
                "sha256": prompt.sha256,
                "build_latency_ms": prompt_latency_ms,
            },
        }
        if rerank_trace is not None:
            result["retrieval"]["rerank"] = rerank_trace.to_dict()
        # 按 logging 配置决定保留哪些字段
        if logging["save_prompt"]:
            result["prompt"]["text"] = prompt.text
        result["retrieval"]["latency_ms"] = retrieval.latency_ms
        result["retrieval"]["timings_ms"] = dict(retrieval.timings_ms)
        result["total_latency_ms"] = (time.perf_counter() - started) * 1000
        return result
