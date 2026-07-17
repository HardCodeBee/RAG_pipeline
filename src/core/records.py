"""本文件定义了流水线各阶段之间传递的数据结构。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


class _AsDictRecord:
    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 页面记录表示“PDF 中一页可检索文本”的标准结构。
# 产生：loader。
# 接受：chunker。
@dataclass(frozen=True, slots=True)
class PageRecord(_AsDictRecord):
    doc_id: str
    source: str
    page: int
    text: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.doc_id, self.source, self.text)):
            raise ValueError("PageRecord text identifiers and text must be non-empty")
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page <= 0:
            raise ValueError("PageRecord.page must be a positive integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PageRecord:
        # 把普通字典转成页面记录，兼容 JSONL 读取结果和测试输入。
        return cls(
            doc_id=value["doc_id"],
            source=value["source"],
            page=value["page"],
            text=value["text"],
        )


# 文本块记录表示“已经切好的检索文本块”。
# 产生：chunker。
# 接受：embedder、retriever、pipeline、artifact 写入逻辑。
@dataclass(frozen=True, slots=True)
class ChunkRecord(_AsDictRecord):
    chunk_id: str
    vector_id: int
    doc_id: str
    source: str
    page_start: int
    page_end: int
    text: str
    token_count: int

    def __post_init__(self) -> None:
        text_fields = (self.chunk_id, self.doc_id, self.source, self.text)
        if any(not isinstance(value, str) or not value.strip() for value in text_fields):
            raise ValueError("ChunkRecord identifiers, source, and text must be non-empty")
        if isinstance(self.vector_id, bool) or not isinstance(self.vector_id, int) or self.vector_id < 0:
            raise ValueError("ChunkRecord.vector_id must be a non-negative integer")
        if isinstance(self.page_start, bool) or not isinstance(self.page_start, int) or self.page_start <= 0:
            raise ValueError("ChunkRecord.page_start must be a positive integer")
        if isinstance(self.page_end, bool) or not isinstance(self.page_end, int) or self.page_end < self.page_start:
            raise ValueError("ChunkRecord.page_end must be an integer no smaller than page_start")
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count <= 0:
            raise ValueError("ChunkRecord.token_count must be a positive integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ChunkRecord:
        return cls(
            chunk_id=value["chunk_id"],
            vector_id=value["vector_id"],
            doc_id=value["doc_id"],
            source=value["source"],
            page_start=value["page_start"],
            page_end=value["page_end"],
            text=value["text"],
            token_count=value["token_count"],
        )


# embedding 空间规格记录向量空间的规格。
# 文本 embedder 生成该规格，再转成字典写入 manifest.json。
# 作用：后续 query 阶段可检查“查询向量”和“索引向量”是否同空间。
@dataclass(frozen=True, slots=True)
class EmbeddingSpaceSpec(_AsDictRecord):
    backend: str
    model_name: str
    revision: str | None
    dimension: int
    normalized: bool
    similarity: str
    document_prefix: str = ""
    max_sequence_length: int | None = None

    def __post_init__(self) -> None:
        names = (self.backend, self.model_name, self.similarity)
        if any(not isinstance(value, str) or not value.strip() for value in names):
            raise ValueError("Embedding space names must be non-empty")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("EmbeddingSpaceSpec.revision must be non-empty or None")
        if not isinstance(self.document_prefix, str):
            raise TypeError("EmbeddingSpaceSpec.document_prefix must be a string")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError("EmbeddingSpaceSpec.dimension must be a positive integer")
        if not isinstance(self.normalized, bool):
            raise TypeError("EmbeddingSpaceSpec.normalized must be a boolean")
        if self.max_sequence_length is not None and (
            isinstance(self.max_sequence_length, bool)
            or not isinstance(self.max_sequence_length, int)
            or self.max_sequence_length <= 0
        ):
            raise ValueError("EmbeddingSpaceSpec.max_sequence_length must be positive or None")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EmbeddingSpaceSpec:
        return cls(
            backend=value["backend"],
            model_name=value["model_name"],
            revision=value.get("revision"),
            dimension=value["dimension"],
            normalized=value["normalized"],
            similarity=value["similarity"],
            document_prefix=value.get("document_prefix", ""),
            max_sequence_length=value.get("max_sequence_length"),
        )


# 向量命中记录是向量索引返回的原始命中结果。
# 此时只有 vector_id 和相似度，还没有映射回 ChunkRecord。
@dataclass(frozen=True, slots=True)
class VectorHit:
    vector_id: int
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.vector_id, bool) or not isinstance(self.vector_id, int) or self.vector_id < 0:
            raise ValueError("VectorHit.vector_id must be a non-negative integer")
        if not math.isfinite(self.score):
            raise ValueError("VectorHit.score must be finite")


# 检索命中记录是 retriever 把向量命中映射回文本块后的结果。
# 查询阶段后续 context 构造器和日志都会使用这个结构。
@dataclass(frozen=True, slots=True)
class SearchHit:
    rank: int
    chunk: ChunkRecord
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("SearchHit.rank must be a positive integer")
        if not math.isfinite(self.score):
            raise ValueError("SearchHit.score must be finite")
        if not isinstance(self.chunk, ChunkRecord):
            raise TypeError("SearchHit.chunk must be a ChunkRecord")

    def to_dict(self) -> dict[str, Any]:
        # 展平成一层 dict，方便日志、评估指标和 JSONL 结果直接消费。
        return {
            "rank": self.rank,
            "chunk_id": self.chunk.chunk_id,
            "vector_id": self.chunk.vector_id,
            "doc_id": self.chunk.doc_id,
            "score": self.score,
            "source": self.chunk.source,
            "page_start": self.chunk.page_start,
            "page_end": self.chunk.page_end,
            "text": self.chunk.text,
            "token_count": self.chunk.token_count,
        }


# 检索轨迹保存一次完整检索过程。
# 不只保存命中的 chunks，也保存 top_k 和分阶段耗时，便于实验复现和诊断。
@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    top_k: int
    results: tuple[SearchHit, ...]
    timings_ms: Mapping[str, float]

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("RetrievalTrace.top_k must be a positive integer")
        if not all(isinstance(result, SearchHit) for result in self.results):
            raise TypeError("RetrievalTrace.results must contain SearchHit values")

    @property
    def latency_ms(self) -> float:
        # 只读属性，用来快速取得总检索耗时。
        return float(self.timings_ms.get("total_ms", 0.0))


# context 包是检索结果和最终 prompt 之间的中间结构。
# 产生：查询 context 构造器。
# 使用：prompt 构造器、pipeline 和生成器。
@dataclass(frozen=True, slots=True)
class ContextPackage:
    text: str
    results: tuple[SearchHit, ...]
    token_count: int
    truncated: bool
    builder: str

    def result_dicts(self) -> list[dict[str, Any]]:
        return [result.to_dict() for result in self.results]


# prompt 包保存最终 prompt 文本及其版本和 hash。
# hash 用于日志记录和实验复现：同一个输入应生成同一个 prompt_sha256。
@dataclass(frozen=True, slots=True)
class PromptPackage:
    text: str
    template: str
    sha256: str
