"""基线使用的精确内积索引。"""

# 对外只暴露基线使用的平铺内积索引。
from src.indexes.faiss_index import FlatIPIndex

__all__ = ["FlatIPIndex"]
