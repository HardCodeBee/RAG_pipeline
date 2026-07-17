"""把文本转换成向量 embedding 并保证建索引阶段和查询阶段使用的是同一个向量空间"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

# 使用EmbeddingSpaceSpec描述当前向量空间的规格
# 用于后续manifest 建索引和查询阶段一致性校验
from src.core.records import EmbeddingSpaceSpec


# 简单的正则表达式分词规则
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

# L2 normalize： 把矩阵里的每一行向量都缩放成单位长度
# 消除向量长度影响，只比较语义方向
def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) # 计算每一行的 L2 norm
    # 空向量的 norm 为 0，改成 1 可以避免除零，同时保持空向量仍为全 0。
    norms[norms == 0] = 1.0
    return matrix / norms

# 一个本地哈希版 embedding 后端
class HashingEmbedder:

    def __init__(self, dimension: int = 384, normalize: bool = True):
        # 哈希回退实现的维度和归一化方式仍要固定。
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a boolean")

        self.dimension = dimension
        self.normalize = normalize
        self.model_name = f"hashing-{dimension}"

    # 用哈希方法把文本变成固定维度向量
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                # BLAKE2 能在不同 Python 进程中给出稳定的词元桶。
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little", signed=False)
                index = value % self.dimension
                # 带符号哈希可以减少不同词元落到同一桶时的系统性偏移。
                sign = 1.0 if (value >> 63) == 0 else -1.0
                vectors[row, index] += sign
        if self.normalize:
            vectors = l2_normalize(vectors)
        return vectors.astype(np.float32)

class TextEmbedder:
    """embedding 门面：可使用 SentenceTransformer 也可退回本地 fallback。"""

    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "BAAI/bge-small-en-v1.5",
        revision: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
        fallback_dim: int = 384,
        query_prefix: str = "",
        document_prefix: str = "",
        max_sequence_length: int | None = None,
        local_files_only: bool = False,
    ):
        # 这里集中校验 embedding 后端的所有参数，避免构建和查询阶段各自重复。
        if backend not in {"auto", "hashing", "sentence_transformers"}:
            raise ValueError("backend must be one of: auto, hashing, sentence_transformers")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be a non-empty string or None")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a boolean")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(fallback_dim, bool) or not isinstance(fallback_dim, int) or fallback_dim <= 0:
            raise ValueError("fallback_dim must be a positive integer")
        if not isinstance(query_prefix, str) or not isinstance(document_prefix, str):
            raise TypeError("query_prefix and document_prefix must be strings")
        if max_sequence_length is not None and (
            isinstance(max_sequence_length, bool)
            or not isinstance(max_sequence_length, int)
            or max_sequence_length <= 0
        ):
            raise ValueError("max_sequence_length must be a positive integer or None")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a boolean")

        self.backend = backend
        self.model_name = model_name
        self.revision = revision
        self.resolved_revision = revision
        self.normalize = normalize
        self.batch_size = batch_size
        self.fallback_dim = fallback_dim
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.max_sequence_length = max_sequence_length
        self.local_files_only = local_files_only
        self._model = None
        self._active_backend = ""
        self._dimension = fallback_dim

        # 按配置选择并加载一个已有模型
        # 读取这个模型自己的属性
        if backend in {"auto", "sentence_transformers"}:
            try:
                # 延迟导入可选依赖，让没有安装该依赖的环境也能运行 pipeline。
                from sentence_transformers import SentenceTransformer

                # 准备模型加载参数
                model_kwargs = {"local_files_only": local_files_only}
                if revision is not None:
                    model_kwargs["revision"] = revision
                self._model = SentenceTransformer(model_name, **model_kwargs)

                # 处理最大输入长度
                if max_sequence_length is not None:
                    self._model.max_seq_length = max_sequence_length
                else:
                    # 未显式配置时，记录模型自己的 max_seq_length，写进 manifest。
                    self.max_sequence_length = int(self._model.max_seq_length)

                # 标记当前真实使用的后端
                self._active_backend = "sentence_transformers"

                try:
                    # 尽量记录实际模型提交 hash，增强复现能力。
                    self.resolved_revision = self._model[0].auto_model.config._commit_hash or revision
                except (AttributeError, IndexError, TypeError):
                    self.resolved_revision = revision

                # 读取 embedding 维度
                dimension_getter = getattr(self._model, "get_embedding_dimension", None)
                dimension = (
                    dimension_getter()
                    if callable(dimension_getter)
                    else self._model.get_sentence_embedding_dimension()
                )
                self._dimension = int(dimension or fallback_dim)

                return
            except Exception:
                if backend == "sentence_transformers":
                    # 显式请求该后端时应直接失败，而不是静默降级。
                    raise

        # 当 sentence-transformers 不可用时，auto 会在这里退回本地实现。
        # 回退实现让smoke测试可以在没有模型权重的环境里运行。
        self._model = HashingEmbedder(dimension=fallback_dim, normalize=normalize)
        self._active_backend = "hashing"
        self._dimension = fallback_dim

    @property
    def dimension(self) -> int:
        """返回当前实际后端的向量维度。"""
        return self._dimension

    # 统一的文本向量化入口
    def encode(self, texts: Sequence[str]) -> np.ndarray:

        # 字符串本身也是 Sequence[str]，必须显式拒绝，避免把一个问题按字符编码。
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")

        texts = list(texts)
        # 检查每个元素都是字符串
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Every item in texts must be a string")
        # 处理空输入
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        # 如果当前实际后端是真实模型 直接调用
        if self._active_backend == "sentence_transformers":
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            # 统一转成 NumPy float32
            embeddings = np.asarray(embeddings, dtype=np.float32)
        # 否则回退到HashingEmbedder
        else:
            embeddings = self._model.encode(texts, batch_size=self.batch_size)

        # 后端返回值必须严格匹配预期形状，防止静默不匹配污染索引。
        # 要求必须是二维矩阵，且形状必须严格等于(文本数量, embedding 维度)
        if embeddings.ndim != 2 or embeddings.shape != (len(texts), self._dimension):
            raise RuntimeError(
                f"Embedding backend returned shape {embeddings.shape}; expected {(len(texts), self._dimension)}"
            )
        # 检查有没有非法数值
        if not np.isfinite(embeddings).all():
            raise RuntimeError("Embedding backend returned non-finite values")
        # 检查归一化是否真的生效
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1)
            nonzero = norms > 0
            # 开启归一化时，内积相似度才等价于余弦相似度
            # 目标是语义检索
            if nonzero.any() and not np.allclose(norms[nonzero], 1.0, rtol=1e-4, atol=1e-5):
                raise RuntimeError("Embedding backend did not return normalized vectors")
        return embeddings

    # 给每条文本前面加一个 prefix，然后调用统一的 encode() 做向量化
    def _encode_prefixed(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        # 拒绝传单个字符串
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")

        values = list(texts)
        # 每个元素都必须是字符串
        if not all(isinstance(text, str) for text in values):
            raise TypeError("Every item in texts must be a string")
        # 给每条文本拼前缀，然后交给 encode()
        return self.encode([f"{prefix}{text}" for text in values])

    # 文档前缀
    # 只用于文档侧 embedding，适配 BGE 等双前缀模型。
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode_prefixed(texts, self.document_prefix)

    # 查询前缀
    # 只用于查询侧 embedding，属于运行配置而不是已构建的文档向量空间。
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode_prefixed(texts, self.query_prefix)

    def embedding_space(self, similarity: str = "inner_product") -> EmbeddingSpaceSpec:
        """返回构建 manifest 和查询一致性校验共用的向量空间规格。"""
        if self._active_backend == "sentence_transformers":
            model_name = self.model_name
            revision = self.resolved_revision
        else:
            model_name = self._model.model_name
            revision = None
        return EmbeddingSpaceSpec(
            backend=self._active_backend,
            model_name=model_name,
            revision=revision,
            dimension=self._dimension,
            normalized=self.normalize,
            similarity=similarity,
            document_prefix=self.document_prefix,
            max_sequence_length=self.max_sequence_length,
        )
