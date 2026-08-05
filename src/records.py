"""本文件定义了流水线各阶段之间传递的数据结构。"""

from __future__ import annotations

# Stable records passed between component implementations.

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

#  数据结构 都必须可以转为字典
class _AsDictRecord:
    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# PageRecord 表示语料加载器产出的一个可检索文本单元。
# 由 loader 转换语料后生成 PageRecord 
# 由 chunker 接收该 PageRecord
@dataclass(frozen=True, slots=True)
class PageRecord(_AsDictRecord):
    '''
    doc_id: 文档 ID 用来标识这页属于哪一篇文档
    source: 原始来源
    page  : 页码
    text  : 正文文本
    '''
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


# ChunkRecord记录表示“已经切好的检索文本块”。
# 由 chunker 对 PageRecord 切分后产生
# embedder 接受该chunk并编码
@dataclass(frozen=True, slots=True)
class ChunkRecord(_AsDictRecord):
    '''
    chunk_id: 文本块 ID
    vector_id  : 用于ChunkRecord 和向量索引之间的连接ID
    doc_id  : 文档 ID 表示这个 chunk 来自哪篇文档
    page_start/page_end: chunk 覆盖的页面范围
    text    :  chunk 的正文内容
    token_count: chunk 的 token 数
    '''
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


# EmbeddingSpaceSpec 主要用于 记录保存生成向量所需的身份和规格
# 由 embedder 在编码 ChunkRecord 时同步生成
# EmbeddingSpaceSpec 将被写入 manifest.json 用于后续对比
@dataclass(frozen=True, slots=True)
class EmbeddingSpaceSpec(_AsDictRecord):
    '''
    backend: 所使用的模型类型
    model_name: 模型名称
    revision: 模型版本
    dimension: 向量维度
    normalized: 向量是否做了 L2 normalize 归一化
    similarity: 计算相似的方式
    document_prefix: 文档侧 embedding 前缀
    max_sequence_length: 模型最大输入长度
    '''
    
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


# VectorHit 表示对于输入的 query 向量索引返回的原始命中结果
# 是索引层结果 由 index 对 query 搜索后得到 VectorHit
# VectorHit 后续将被映射回 ChunkRecord。
@dataclass(frozen=True, slots=True)
class VectorHit:
    '''
    vector_id : 索引返回的向量 ID 用来之后找回对应的 ChunkRecord
    score     : 索引返回的分数 
        '''
    vector_id: int
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.vector_id, bool) or not isinstance(self.vector_id, int) or self.vector_id < 0:
            raise ValueError("VectorHit.vector_id must be a non-negative integer")
        if not math.isfinite(self.score):
            raise ValueError("VectorHit.score must be finite")

#  SearchHit 是 VectorHit 的下一层
#  是检索层结果 SearchHit 是 retriever 把向量命中 VectorHit 映射回 ChunkRecord 的结果。
#  SearchHit 后续被收集进 RetrievalTrace 由
@dataclass(frozen=True, slots=True)
class SearchHit:
    '''
    rank : 该检索结果的排名 从 1 开始
    chunk: 完整的 ChunkRecord
    score: 从 VectorHit 继承来的相似度分数
    '''
    
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


# RetrievalTrace保存一次完整检索过程。
# 不只保存命中的 chunks，也保存 top_k 和 分阶段耗时，便于实验复现和诊断。
# RetrievalTrace 会被 pipeline 拆开使用
# results 传给 build_context() 构造上下文
# 分阶段耗时 写入日志
@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    '''
    top_k       : top k。
    results     :所有 SearchHit 列表
    timings_ms  :检索过程里的分阶段耗时，单位是毫秒。
    '''
    
    top_k: int
    results: tuple[SearchHit, ...]
    timings_ms: Mapping[str, float]

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("RetrievalTrace.top_k must be a non-negative integer")
        if not all(isinstance(result, SearchHit) for result in self.results):
            raise TypeError("RetrievalTrace.results must contain SearchHit values")

    @property
    def latency_ms(self) -> float:
        # 只读属性，用来快速取得总检索耗时。
        return float(self.timings_ms.get("total_ms", 0.0))


# ContextPackage 是检索结果进入 prompt 之前的上下文包
# 产生：查询 context 构造器。
# 使用：prompt 构造器、pipeline 和生成器。
# ContextPackage 后续会被 prompt构造器 直接拼进最终 prompt
# 其他信息写入日志
@dataclass(frozen=True, slots=True)
class ContextPackage:
    '''
    text        :   已经拼好的上下文字符串 后面直接塞进 prompt 的 Context
    results     :   实际进入 context 的 SearchHit
    token_count :  这个 context 已经使用了多少 token。
    truncated   :   是否因为 context_window_tokens 限制发生过截断。
    builder     :   上下文构造策略名称。
    '''
    text: str
    results: tuple[SearchHit, ...]
    token_count: int
    truncated: bool
    builder: str

    def result_dicts(self) -> list[dict[str, Any]]:
        return [result.to_dict() for result in self.results]


# 由 ContextPackage 进一步生成
# prompt 包保存最终 prompt 文本及其版本和 hash。
# hash 用于日志记录和实验复现：同一个输入应生成同一个 prompt_sha256。
@dataclass(frozen=True, slots=True)
class PromptPackage:
    '''
    text    :最终发送给 generator 的完整 prompt。
    template:prompt 模板
    sha256  :对完整 prompt 文本做 SHA256 后得到的 hash。
    '''
    
    text: str
    template: str
    sha256: str
