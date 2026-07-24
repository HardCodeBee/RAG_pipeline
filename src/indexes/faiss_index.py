"""
带 FAISS 和 NumPy 后端的平铺内积向量索引。
构建阶段：把所有 chunk embeddings 建成一个可搜索的向量索引
查询阶段：用 query embedding 在索引里找最相似的 chunks
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.records import VectorHit

# 创建一个空索引对象，并记录用户想使用哪种索引后端
# 有 FAISS 时优先使用 FAISS；
# 没有 FAISS 时退回精确 NumPy 搜索，
class FlatIPIndex:
    def __init__(self, backend: str, index_type: str = "flat_ip"):

        # 校验 backend 是否合法
        if backend not in {"faiss", "numpy"}:
            raise ValueError("backend must be one of: faiss, numpy")
        # 校验索引类型：目前只有flat_ip 基于内积的精确向量索引
        if index_type != "flat_ip":
            raise ValueError("index_type must be flat_ip")

        # requested_backend 是用户想要的后端；
        # backend 是实际解析后的后端。
        self.requested_backend = backend
        self.index_type = index_type
        self.backend = ""
        self.index = None
        self.embeddings: np.ndarray | None = None
        self.ids: np.ndarray | None = None
        self.dimension = 0
        self.count = 0


    # 把一批文本块的 embedding 向量建成一个可以搜索的向量索引
    def build(self, embeddings: np.ndarray, ids: np.ndarray | None = None) -> None:

        #检查和整理输入
        # 统一转 float32 FAISS 和 NumPy 产物都按 float32 存储。
        embeddings = np.asarray(embeddings, dtype=np.float32)
        # embeddings 必须是二维矩阵
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        # 矩阵不能是空的
        if embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
            raise ValueError("embeddings must contain at least one non-empty vector")
        # 所有值都必须是有限数
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")
        # 把数组变成内存连续布局
        # FAISS 通常要求传入连续的 float32 数组
        embeddings = np.ascontiguousarray(embeddings)

        # 处理 ids
        # 如果没有传 ids，就自动生成
        if ids is None:
            # 默认 id 与行号一致；
            # 通常 build_index 会显式传入 ChunkRecord.vector_id。
            ids = np.arange(embeddings.shape[0], dtype=np.int64)
        # 把 ids 转成 NumPy 数组，并统一成 int64
        # FAISS 的 id 通常使用 64 位整数
        ids = np.asarray(ids, dtype=np.int64)
        # 校验 ids：必须是一维数组 长度必须等于 embedding 行数
        if ids.ndim != 1 or ids.shape[0] != embeddings.shape[0]:
            raise ValueError("ids must be a 1D array with one id per embedding")
        # 要求每个 id 唯一
        if len(np.unique(ids)) != len(ids):
            raise ValueError("ids must be unique")
        # 同样把 ids 变成连续内存布局，方便 FAISS 使用
        ids = np.ascontiguousarray(ids)
        # 记录索引的维度和数量
        self.dimension = embeddings.shape[1]
        self.count = embeddings.shape[0]

        # 选择后端并建立索引

        if self.requested_backend == "faiss":
            import faiss

            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap2(base_index)
            self.index.add_with_ids(embeddings, ids)
            self.backend = "faiss"
            self.embeddings = None
            self.ids = ids
        else:
            expected_ids = np.arange(embeddings.shape[0], dtype=np.int64)
            if not np.array_equal(ids, expected_ids):
                raise ValueError("NumPy backend requires zero-based row-aligned vector ids")
            self.backend = "numpy"
            self.embeddings = embeddings
            self.ids = ids
            self.index = None

    # 拿一个 query embedding 去索引里找最相似的 top-k 个向量
    # 该 query 的 top-k 内积分数和向量 id。
    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:

        # top_k 必须是正整数
        # 校验放在 index 层，retriever 即使绕过也不会传入非法值。
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        # 检查索引是否已经可用
        if not self.backend or self.count <= 0 or self.dimension <= 0:
            raise RuntimeError("Index has not been built or loaded")
        #把 query embedding 转成 NumPy 数组，并统一成 float32。
        #兼容 FAISS 和 NumPy 后端
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        # 允许调用者传入单个向量；内部统一成 shape=(1, dim)。
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        # 要求 query embedding必须是：
        # 二维数组 并且 只能包含 1 个 query
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("query_embedding must contain exactly one vector")
        # 检查 query 向量维度必须和索引维度一致
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_embedding.shape[1]} does not match index dimension {self.dimension}"
            )
        # 检查 query 里不能有扭曲的数字例如无穷
        if not np.isfinite(query_embedding).all():
            raise ValueError("query_embedding must contain only finite values")

        # 转成连续内存布局 方便FAISS 底层调用
        query_embedding = np.ascontiguousarray(query_embedding)
        # 如果用户请求的 top_k 超过索引里的向量数量，就自动缩小
        top_k = min(top_k, self.count)

        # 如果当前后端是 FAISS 就走 FAISS 检索
        if self.backend == "faiss":
            # 调用 FAISS 的 search()
            # 返回两个二维数组
            # scores：相似度分数。
            # indices：命中的向量 id。
            scores, indices = self.index.search(query_embedding, top_k)
            valid = indices[0] >= 0  # 过滤无效结果
            selected_scores = scores[0][valid]
            selected_ids = indices[0][valid]
            # 分数相同的情况下按 vector_id 排序，保证 FAISS/NumPy 返回顺序稳定。
            # 第一优先级：分数从高到低
            # 第二优先级：vector_id 从小到大
            order = np.lexsort((selected_ids, -selected_scores))
            return selected_scores[order], selected_ids[order]

        if self.embeddings is None:
            raise RuntimeError("NumPy index has not been built or loaded")
        if self.ids is None:
            raise RuntimeError("NumPy index has no vector ids")
        # NumPy exact search uses the selected canonical embedding matrix.
        # NumPy 检索：手动计算每个文档向量和 query 向量的内积分数
        # 对每个向量打分，然后按分数降序排序。
        scores = self.embeddings @ query_embedding[0]
        top_k = min(top_k, len(scores))
        # np.lexsort 最后一个 key 是主 key：先按 -score，再按 id 打破平分。
        positions = np.lexsort((self.ids, -scores))[:top_k]
        return scores[positions], self.ids[positions]

    # 把搜索得到的底层数组结果转换成统一的 VectorHit 结构，供 retriever 使用。
    def search_hits(self, query_embedding: np.ndarray, top_k: int) -> list[VectorHit]:
        scores, ids = self.search(query_embedding, top_k)
        return [
            VectorHit(vector_id=int(vector_id), score=float(score))
            for score, vector_id in zip(scores, ids)
        ]

    # NumPy search directly uses the canonical embeddings.npy artifact.
    def _read_numpy_embeddings(self, source: Path) -> tuple[np.ndarray, np.ndarray]:
        embeddings = np.load(source, allow_pickle=False).astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
            raise ValueError(f"NumPy index must be a non-empty 2D array: {source}")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"NumPy index contains non-finite values: {source}")
        ids = np.arange(embeddings.shape[0], dtype=np.int64)
        return np.ascontiguousarray(embeddings), np.ascontiguousarray(ids)

    # 构建阶段写入磁盘
    # 把当前已经 build/load 好的索引保存到磁盘
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.backend or self.count <= 0 or self.dimension <= 0:
            raise RuntimeError("Index has not been built or loaded")
        if self.backend == "faiss":
            import faiss

            # FAISS 后端保存原生 index 文件。
            faiss.write_index(self.index, str(path))
            return

        raise RuntimeError("NumPy search uses embeddings.npy directly and has no separate index artifact")

    # 查询阶段从磁盘加载回内存
    # 优先加载 FAISS index.faiss
    # The selected backend determines whether this path is FAISS or embeddings.npy.
    # 加载成功后，FlatIPIndex 就可以继续 search()
    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Index file does not exist: {path}")
        if self.requested_backend == "faiss":
            try:
                import faiss

                loaded_index = faiss.read_index(str(path))
                is_id_map = isinstance(loaded_index, faiss.IndexIDMap2)
                # 接受 IndexIDMap2(IndexFlatIP) 或直接的 IndexFlatIP。
                base_index = faiss.downcast_index(loaded_index.index) if is_id_map else loaded_index
                if not isinstance(base_index, faiss.IndexFlatIP):
                    raise ValueError(
                        f"Expected a FAISS IndexFlatIP, found {type(loaded_index).__name__}"
                    )
                if int(loaded_index.d) <= 0 or int(loaded_index.ntotal) <= 0:
                    raise ValueError("FAISS index must contain at least one non-empty vector")
                self.index = loaded_index
                self.backend = "faiss"
                self.dimension = int(self.index.d)
                self.count = int(self.index.ntotal)
                self.embeddings = None
                self.ids = (
                    # IndexIDMap2 保存了外部 vector_id；普通 FlatIP 只能退回行号。
                    faiss.vector_to_array(loaded_index.id_map).astype(np.int64)
                    if is_id_map
                    else np.arange(self.count, dtype=np.int64)
                )
                return
            except Exception as exc:
                raise RuntimeError(f"Failed to load FAISS flat_ip index: {path}") from exc

        self.embeddings, self.ids = self._read_numpy_embeddings(path)

        self.backend = "numpy"
        self.index = None
        self.dimension = self.embeddings.shape[1]
        self.count = self.embeddings.shape[0]
