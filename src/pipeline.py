"""固定基线 RAG 流水线的查询时编排逻辑。"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.artifact_io import validate_build_directory
from src.components import create_embedder, create_generator, create_index, create_token_counter
from src.config import validate_config
from src.core.records import ChunkRecord, EmbeddingSpaceSpec
from src.evaluators.metrics import evaluate_result
from src.io_utils import read_jsonl
from src.prompts.fixed_prompt import build_prompt
from src.provenance import (
    build_identity,
    corpus_inventory,
    json_sha256,
    resolved_roots,
    run_spec,
    source_code_sha256,
)
from src.query.context_builders import build_context
from src.retrievers.dense_retriever import DenseRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _query_id() -> str:
    # query_id 带 UTC 时间和随机后缀，便于日志中区分每次查询。
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"query_{timestamp}_{uuid.uuid4().hex[:8]}"


def _record_dict(value: Any) -> dict[str, Any]:
    # 不同组件可能返回 dataclass、带 to_dict() 的对象或普通 dict，这里统一成 dict。
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result
    if isinstance(value, dict):
        return dict(value)
    raise TypeError("Expected dataclass or mapping metadata")


class NaiveRAGPipeline:
    """加载一个不可变 build，并用显式基线链路回答问题。"""

    def __init__(self, config: dict[str, Any]):
        # 初始化阶段只加载一个不可变构建，并校验它是否和当前配置匹配。
        setup_started = time.perf_counter()
        self.config = validate_config(config)
        if "_base_dir" not in self.config:
            raise ValueError("config must include _base_dir; use load_config()")

        roots = resolved_roots(self.config)
        from src.components import create_loader

        loader = create_loader(self.config)
        # 重新计算构建身份，找到当前配置应该对应的 build_dir。
        documents = loader.discover(roots["corpus"])
        corpus = corpus_inventory(documents, roots["corpus"])
        source_sha = source_code_sha256(PROJECT_ROOT)
        build_id, build_spec_sha, build_spec_value = build_identity(self.config, corpus, source_sha)
        self.build_dir = roots["artifacts_root"] / build_id
        # query 阶段不重建索引，只接受已经完整校验过的 build 目录。
        self.manifest = validate_build_directory(self.build_dir, build_id)
        if self.manifest.get("build_spec_sha256") != build_spec_sha:
            raise ValueError("The active build specification does not match the immutable build directory")
        if self.manifest.get("build_spec") != build_spec_value:
            raise ValueError("The active build specification payload differs from the build manifest")

        artifacts = self.manifest["artifacts"]
        chunks_path = self.build_dir / artifacts["chunks"]["file"]
        embeddings_path = self.build_dir / artifacts["embeddings"]["file"]
        index_path = self.build_dir / artifacts["index"]["file"]
        # chunks.jsonl 是 vector_id -> 文本/来源的映射表。
        self.chunks = [ChunkRecord.from_mapping(row) for row in read_jsonl(chunks_path)]
        self._validate_chunks(embeddings_path)

        self.token_counter = create_token_counter(self.config)
        # 查询 embedder 必须按 manifest 中构建阶段的 embedding 空间加载。
        self.embedder = self._load_query_embedder()
        manifest_index = self.manifest["index"]
        self.index = create_index(
            self.config,
            backend=manifest_index["backend"],
            index_type=manifest_index["type"],
        )
        self.index.load(index_path, embeddings_path=embeddings_path)
        self._validate_loaded_index()

        # retriever 组合 chunks、embedder、index，负责查询向量检索。
        self.retriever = DenseRetriever(
            self.chunks,
            self.embedder,
            self.index,
            top_k=self.config["retrieval"]["top_k"],
        )
        self.generator = create_generator(self.config)

        run_spec_value = run_spec(self.config, build_id, source_sha)
        # runtime_metadata 记录 query 阶段实际使用的构建、模型、索引和生成配置。
        self.runtime_metadata = {
            "build_id": build_id,
            "build_dir": str(self.build_dir.resolve()),
            "build_spec_sha256": build_spec_sha,
            "source_sha256": source_sha,
            "run_spec": run_spec_value,
            "run_spec_sha256": json_sha256(run_spec_value),
            "embedding": {
                "space": self.embedding_space_spec.to_dict(),
                "query_prefix": self.embedder.query_prefix,
            },
            "index": {
                "backend": self.index.backend,
                "type": self.index.index_type,
                "count": self.index.count,
                "dimension": self.index.dimension,
            },
            "generator": {
                "requested_provider": self.generator.provider,
                "requested_model": self.generator.model,
                # 凭据只保留“是否已配置”，密钥值始终只存在于进程内存中。
                "api_key_present": bool(self.generator.api_key),
                "temperature": self.generator.temperature,
                "max_output_tokens": self.generator.max_output_tokens,
            },
        }
        self.runtime_metadata["setup_latency_ms"] = (time.perf_counter() - setup_started) * 1000

    def _validate_chunks(self, embeddings_path: Path) -> None:
        # 文本块产物的行数、vector_id 序列和 embeddings 行数必须完全一致。
        descriptor = self.manifest["artifacts"]["chunks"]
        if descriptor.get("rows") != len(self.chunks) or self.manifest["chunking"].get("num_chunks") != len(
            self.chunks
        ):
            raise ValueError("Chunk count does not match the build manifest")
        chunk_ids = [item.chunk_id for item in self.chunks]
        vector_ids = [item.vector_id for item in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk ids must be unique")
        if vector_ids != list(range(len(self.chunks))):
            raise ValueError("Chunk vector ids must be the zero-based chunk sequence")
        if self.manifest.get("vector_id_sequence_sha256") != json_sha256(vector_ids):
            raise ValueError("Chunk vector ids do not match the build manifest")

        # mmap_mode 避免为了校验 shape/dtype 一次性加载大 embedding 文件。
        embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
        embedding_descriptor = self.manifest["artifacts"]["embeddings"]
        if list(embeddings.shape) != embedding_descriptor.get("shape"):
            raise ValueError("Embedding shape does not match the build manifest")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.chunks):
            raise ValueError("Embedding rows do not match chunks")
        if str(embeddings.dtype) != embedding_descriptor.get("dtype"):
            raise ValueError("Embedding dtype does not match the build manifest")
        if not np.isfinite(embeddings).all():
            raise ValueError("Embedding artifact contains non-finite values")

    def _load_query_embedder(self):
        built = self.manifest["embedding"]
        expected_space = EmbeddingSpaceSpec.from_mapping(built["space"])
        # 用构建 manifest 覆盖配置中可能变化的 embedding 字段，确保查询与索引同空间。
        override = {
            "backend": expected_space.backend,
            "model_name": expected_space.model_name,
            "revision": expected_space.revision,
            "normalize": expected_space.normalized,
            "fallback_dim": expected_space.dimension,
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
        ids = [int(value) for value in self.index.ids.tolist()]
        expected_hash = self.manifest["vector_id_sequence_sha256"]
        # 不只校验集合，也校验 hash，确保 id 序列和 build 阶段一致。
        if set(ids) != {item.vector_id for item in self.chunks} or json_sha256(ids) != expected_hash:
            raise ValueError("Loaded index vector ids do not match chunks or manifest")

    def retrieve(self, question: str, top_k: int | None = None):
        # 对外暴露检索接口，便于单独测试 retrieval，不必走完整 generation。
        return self.retriever.retrieve_trace(question, top_k=top_k)

    def query(
        self,
        question: str,
        question_id: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        started = time.perf_counter()
        # 1. 检索相关 chunks。
        retrieval = self.retrieve(question, top_k=top_k)

        context_started = time.perf_counter()
        # 2. 把检索结果按词元预算拼成提示词上下文。
        context = build_context(
            question,
            retrieval.results,
            self.token_counter,
            self.config["context"]["max_tokens"],
        )
        context_latency_ms = (time.perf_counter() - context_started) * 1000
        prompt_started = time.perf_counter()
        # 3. 使用固定模板构造最终 prompt。
        prompt = build_prompt(question, context, self.config["prompt"]["version"])
        prompt_latency_ms = (time.perf_counter() - prompt_started) * 1000
        # 4. 调用生成器，可能是 OpenAI，也可能是本地抽取式回退。
        generation = self.generator.generate_from_prompt(prompt.text, question, context.result_dicts())

        generation_data = _record_dict(generation)
        # 查询结果只保留一个权威 status，避免顶层与 generation 子结构发生漂移。
        generation_status = generation_data.pop("status", "success")
        # 补充 pipeline 层才知道的 prompt 元数据。
        generation_data.update(
            {
                "temperature": self.generator.temperature,
                "max_output_tokens": self.generator.max_output_tokens,
                "prompt_build_latency_ms": prompt_latency_ms,
                "prompt_sha256": prompt.sha256,
                "prompt_template": prompt.template,
            }
        )
        retrieved_rows = [item.to_dict() for item in retrieval.results]
        logging = self.config["logging"]
        if not logging["save_retrieved_chunks"]:
            # 可关闭 chunk 正文保存，只保留来源和 id，减少输出体积。
            retrieved_rows = [{key: value for key, value in row.items() if key != "text"} for row in retrieved_rows]

        # result 是单次 query 的完整可序列化记录。
        result = {
            "status": generation_status,
            "question_id": question_id or _query_id(),
            "question": question,
            "identity": {
                "build_id": self.runtime_metadata["build_id"],
                "run_spec_sha256": self.runtime_metadata["run_spec_sha256"],
                "source_sha256": self.runtime_metadata["source_sha256"],
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
        }
        if logging["save_prompt"]:
            result["prompt"] = prompt.text
        if logging["save_latency"]:
            # latency 字段较多，可通过 logging 配置关闭。
            result["retrieval"]["latency_ms"] = retrieval.latency_ms
            result["retrieval"]["timings_ms"] = dict(retrieval.timings_ms)
            result["total_latency_ms"] = (time.perf_counter() - started) * 1000
        else:
            result["context"].pop("build_latency_ms", None)
            generation_data.pop("latency_ms", None)
            generation_data.pop("prompt_build_latency_ms", None)
        if not logging["save_token_usage"]:
            # 词元用量可能较冗长，可按需从输出里移除。
            for key in ("input_tokens", "output_tokens", "token_usage"):
                generation_data.pop(key, None)
        return result

    def answer_question(
        self,
        question: str,
        question_id: str | None = None,
        gold_answer: str | None = None,
        expected_sources: list[str] | None = None,
        expected_evidence: list[dict | str] | None = None,
        top_k: int | None = None,
        answerable: bool | None = None,
    ) -> dict[str, Any]:
        # answer_question 在 query() 结果基础上附加人工标签，并计算评估指标。
        result = self.query(question, question_id=question_id, top_k=top_k)
        labels = {
            "gold_answer": gold_answer,
            "expected_sources": expected_sources or [],
            "expected_evidence": expected_evidence or [],
            "answerable": answerable,
        }
        result.update(labels)
        result["metrics"] = evaluate_result(
            result["generation"]["answer"],
            result["retrieval"]["results"],
            **labels,
        )
        return result
