# 统一导出核心记录类型，调用方不需要知道它们都定义在 records.py。
from src.core.records import (
    ChunkRecord,
    ContextPackage,
    EmbeddingSpaceSpec,
    PageRecord,
    PromptPackage,
    RetrievalTrace,
    SearchHit,
    VectorHit,
)

__all__ = [
    "ChunkRecord",
    "ContextPackage",
    "EmbeddingSpaceSpec",
    "PageRecord",
    "PromptPackage",
    "RetrievalTrace",
    "SearchHit",
    "VectorHit",
]
