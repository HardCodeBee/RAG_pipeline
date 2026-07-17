"""把用户的自然语言问题检索成一组最相关的文本块
接收 chunks已经切好的文本块、embedder已经转成向量的文本、index已经建好的向量索引
把 query 编码成向量
调用索引检索 top-k 向量
把 vector_id 映射回文本块
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from src.core.records import ChunkRecord, RetrievalTrace, SearchHit

"""编码一个查询，检索索引，并把向量 id 映射回文本块。"""
class DenseRetriever:

    def __init__(self, chunks: Iterable[ChunkRecord | dict], embedder, index, top_k: int = 5):

        # 校验 top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        # 校验 index 已经可用
        if index.count <= 0 or index.dimension <= 0:
            raise ValueError("index must be built or loaded before creating a retriever")

        # 统一 chunks 格式
        records = [item if isinstance(item, ChunkRecord)
                   else ChunkRecord.from_mapping(item)
                   for item in chunks]
        # 使用 vector_id 建表，因为索引返回的是 vector_id，而不是文本块列表下标。
        by_vector_id = {item.vector_id: item for item in records}
        # 检查 vector_id 是否唯一
        if len(by_vector_id) != len(records):
            raise ValueError("Chunk vector ids must be unique")
        # 检查 chunk 数量和 index 数量一致
        if len(records) != index.count:
            raise ValueError(f"Chunk count {len(records)} does not match index count {index.count}")
        # 检查 embedder 维度和 index 维度一致 保证点积相似度计算可行
        embedder_dimension = getattr(embedder, "dimension", None)
        if embedder_dimension is not None and embedder_dimension != index.dimension:
            raise ValueError(
                f"Embedder dimension {embedder_dimension} does not match index dimension {index.dimension}"
            )
        # 检查 index 内部 ids 和 chunk 的 vector_id 一致
        if getattr(index, "ids", None) is not None:
            index_ids = {int(value) for value in index.ids.tolist()}
            if index_ids != set(by_vector_id):
                raise ValueError("Index vector ids do not match chunk vector ids")

        self._chunks = by_vector_id
        self.embedder = embedder
        self.index = index
        self.top_k = top_k

    # 输入一个自然语言 query
    # 返回 top-k 个最相关文本块，并附带检索耗时
    def retrieve_trace(self, query: str, top_k: int | None = None) -> RetrievalTrace:
        # query 必须是非空字符串
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        # 调用时传 top_k 可以覆盖默认值，但仍然要重新校验。
        effective_top_k = self.top_k if top_k is None else top_k
        if isinstance(effective_top_k, bool) or not isinstance(effective_top_k, int) or effective_top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        # 查询阶段 embedding query
        started = time.perf_counter()   # 开始总计时
        embedding_started = time.perf_counter() # 开始embedding计时
        # 查询侧必须用 encode_queries()，这样 query_prefix 才会生效。
        query_embedding = self.embedder.encode_queries([query.strip()])
        embedding_latency_ms = (time.perf_counter() - embedding_started) * 1000 # 记录了 embedding 花了多久

        # 对query进行search，获得目标vector_id
        search_started = time.perf_counter()  # 开始search计时
        vector_hits = self.index.search_hits(query_embedding, effective_top_k) # 调用索引检索
        search_latency_ms = (time.perf_counter() - search_started) * 1000 # 记录了 search 花了多久

        # 把 vector_id 映射回 ChunkRecord
        mapping_started = time.perf_counter()
        try:
            # 把 index 返回的 vector_id 映射回 ChunkRecord，并补上 rank。
            results = tuple(
                SearchHit(
                    rank=rank,
                    chunk=self._chunks[hit.vector_id],
                    score=hit.score,
                )
                for rank, hit in enumerate(vector_hits, start=1)
            )
        except KeyError as exc:
            raise ValueError(f"Index returned unknown vector id: {exc.args[0]}") from exc
        mapping_latency_ms = (time.perf_counter() - mapping_started) * 1000

        # 记录阶段耗时
        timings = {
            # 分阶段计时方便定位慢在 embedding、索引检索还是文本块映射。
            "query_embedding_ms": embedding_latency_ms,
            "index_search_ms": search_latency_ms,
            "chunk_mapping_ms": mapping_latency_ms,
            "total_ms": (time.perf_counter() - started) * 1000,
        }
        return RetrievalTrace(top_k=effective_top_k, results=results, timings_ms=timings)
