"""把文本转换成向量 embedding 并保证建索引阶段和查询阶段使用的是同一个向量空间"""

from __future__ import annotations

# Hashing and SentenceTransformer text embedding implementations.

import hashlib
import re
from typing import Any, Sequence

import numpy as np

# 使用EmbeddingSpaceSpec描述当前向量空间的规格
# 用于后续manifest 建索引和查询阶段一致性校验
from src.records import DocumentEmbeddingInput, EmbeddingSpaceSpec
from src.model_backends.huggingface_snapshot import resolve_hf_snapshot


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
        # The deterministic hashing backend still fixes dimension and normalization.
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a boolean")

        self.dimension = dimension
        self.normalize = normalize
        self.model_name = f"hashing-{dimension}"

    # 用哈希方法把文本变成固定维度向量
    def encode(self, texts: Sequence[str]) -> np.ndarray:
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
    """Explicit hashing or SentenceTransformer embedding backend."""

    def __init__(
        self,
        backend: str,
        model_name: str | None = None,
        revision: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
        dimension: int = 384,
        query_prefix: str = "",
        document_prefix: str = "",
        max_sequence_length: int | None = None,
        local_files_only: bool = False,
        device: str = "auto",
    ):
        if backend not in {"hashing", "sentence_transformers"}:
            raise ValueError("backend must be one of: hashing, sentence_transformers")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a boolean")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
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
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

        self.backend = backend
        self.model_name = model_name
        self.revision = revision
        self.resolved_revision = revision
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.document_input_format = "text"
        self.encoder_family = None
        self.max_sequence_length = max_sequence_length
        self.requested_device = device
        self.device = "cpu"
        self._model = None
        self._active_backend = backend

        if backend == "hashing":
            if model_name not in {None, f"hashing-{dimension}"}:
                raise ValueError("hashing model_name must match the configured dimension")
            if revision is not None:
                raise ValueError("hashing backend does not use a revision")
            if max_sequence_length is not None:
                raise ValueError("hashing backend does not use max_sequence_length")
            self._model = HashingEmbedder(dimension=dimension, normalize=normalize)
            self.model_name = self._model.model_name
            self.resolved_revision = None
            self._dimension = dimension
            return

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty string")

        import torch
        from sentence_transformers import SentenceTransformer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "embedding.device=cuda was requested, but CUDA is not available in the active PyTorch environment"
            )
        resolved_device = None if device == "auto" else device

        snapshot = resolve_hf_snapshot(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        self._model = SentenceTransformer(
            str(snapshot),
            device=resolved_device,
            local_files_only=True,
        )
        self.device = str(self._model.device)
        if max_sequence_length is not None:
            self._model.max_seq_length = max_sequence_length
        else:
            self.max_sequence_length = int(self._model.max_seq_length)
        try:
            self.resolved_revision = self._model[0].auto_model.config._commit_hash or revision
        except (AttributeError, IndexError, TypeError):
            self.resolved_revision = revision
        dimension_getter = getattr(self._model, "get_embedding_dimension", None)
        model_dimension = (
            dimension_getter()
            if callable(dimension_getter)
            else self._model.get_sentence_embedding_dimension()
        )
        if not model_dimension:
            raise RuntimeError("SentenceTransformer did not report an embedding dimension")
        self._dimension = int(model_dimension)

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
        # The explicitly selected hashing backend uses the local implementation.
        else:
            embeddings = self._model.encode(texts)

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


class HFDenseEmbedder:
    """One role-specific Hugging Face encoder for the fixed dense baselines."""

    def __init__(
        self,
        *,
        family: str,
        role: str,
        model_name: str,
        revision: str,
        normalize: bool,
        batch_size: int,
        max_sequence_length: int,
        pooling: str,
        document_input_format: str,
        local_files_only: bool,
        device: str,
    ) -> None:
        if family not in {"dpr", "contriever"}:
            raise ValueError("family must be one of: dpr, contriever")
        if role not in {"document", "query"}:
            raise ValueError("role must be one of: document, query")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty string")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a boolean")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(max_sequence_length, bool)
            or not isinstance(max_sequence_length, int)
            or max_sequence_length <= 0
        ):
            raise ValueError("max_sequence_length must be a positive integer")
        expected_format = "title_text_pair" if family == "dpr" else "title_space_text"
        expected_pooling = "pooler_output" if family == "dpr" else "masked_mean"
        if pooling != expected_pooling:
            raise ValueError(f"{family} pooling must be {expected_pooling}")
        if document_input_format != expected_format:
            raise ValueError(
                f"{family} document_input_format must be {expected_format}"
            )
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

        import torch
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoTokenizer,
            DPRContextEncoder,
            DPRQuestionEncoder,
        )

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "embedding.device=cuda was requested, but CUDA is not available "
                "in the active PyTorch environment"
            )
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        snapshot = resolve_hf_snapshot(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        if family == "dpr":
            expected_architecture = (
                "DPRContextEncoder" if role == "document" else "DPRQuestionEncoder"
            )
            snapshot_config = AutoConfig.from_pretrained(
                str(snapshot),
                local_files_only=True,
            )
            if expected_architecture not in (
                getattr(snapshot_config, "architectures", None) or ()
            ):
                raise ValueError(
                    f"DPR {role} checkpoint must declare {expected_architecture}"
                )
            model_class = (
                DPRContextEncoder if role == "document" else DPRQuestionEncoder
            )
        else:
            model_class = AutoModel

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            local_files_only=True,
            use_fast=True,
        )
        self._model = model_class.from_pretrained(
            str(snapshot),
            local_files_only=True,
        )
        self._model.to(resolved_device)
        self._model.eval()

        config = self._model.config
        projection_dimension = int(getattr(config, "projection_dim", 0) or 0)
        hidden_dimension = int(getattr(config, "hidden_size", 0) or 0)
        dimension = projection_dimension or hidden_dimension
        if dimension <= 0:
            raise RuntimeError("Hugging Face encoder did not report an embedding dimension")

        self.backend = "hf_dense"
        self.encoder_family = family
        self.role = role
        self.model_name = model_name.strip()
        self.revision = revision.strip()
        self.resolved_revision = getattr(config, "_commit_hash", None) or self.revision
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.pooling = pooling
        self.document_input_format = document_input_format
        self.query_prefix = ""
        self.document_prefix = ""
        self.requested_device = device
        self.device = resolved_device
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _strings(texts: Sequence[str]) -> list[str]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        values = list(texts)
        if not all(isinstance(text, str) for text in values):
            raise TypeError("Every item in texts must be a string")
        if any(not text.strip() for text in values):
            raise ValueError("Embedding texts must be non-empty")
        return values

    @staticmethod
    def _documents(
        values: Sequence[DocumentEmbeddingInput],
    ) -> list[DocumentEmbeddingInput]:
        if isinstance(values, (str, bytes)):
            raise TypeError("documents must be a sequence of DocumentEmbeddingInput")
        result = list(values)
        if not all(isinstance(value, DocumentEmbeddingInput) for value in result):
            raise TypeError(
                "Structured HF document encoding requires DocumentEmbeddingInput values"
            )
        return result

    def _model_inputs(self, **tokenizer_inputs: Any) -> dict[str, Any]:
        values = self._tokenizer(
            **tokenizer_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in values.items()}

    def _checked(self, embeddings: Any, expected_rows: int) -> np.ndarray:
        values = embeddings.detach().to(dtype=self._torch.float32).cpu().numpy()
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (expected_rows, self._dimension):
            raise RuntimeError(
                f"Embedding backend returned shape {values.shape}; "
                f"expected {(expected_rows, self._dimension)}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError("Embedding backend returned non-finite values")
        if self.normalize:
            values = l2_normalize(values).astype(np.float32)
        return values

    def _dpr_documents(self, values: Sequence[DocumentEmbeddingInput]) -> np.ndarray:
        result = np.empty((len(values), self._dimension), dtype=np.float32)
        titled = [index for index, value in enumerate(values) if value.title]
        untitled = [index for index, value in enumerate(values) if not value.title]
        for positions in (titled, untitled):
            if not positions:
                continue
            selected = [values[index] for index in positions]
            if selected[0].title:
                inputs = self._model_inputs(
                    text=[value.title for value in selected],
                    text_pair=[value.text for value in selected],
                )
            else:
                inputs = self._model_inputs(text=[value.text for value in selected])
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            result[positions] = self._checked(outputs.pooler_output, len(selected))
        return result

    def _dpr_queries(self, values: Sequence[str]) -> np.ndarray:
        inputs = self._model_inputs(text=list(values))
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        return self._checked(outputs.pooler_output, len(values))

    def _contriever(self, values: Sequence[str]) -> np.ndarray:
        inputs = self._model_inputs(text=list(values))
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self._checked(pooled, len(values))

    def encode_documents(
        self,
        values: Sequence[DocumentEmbeddingInput],
    ) -> np.ndarray:
        if self.role != "document":
            raise RuntimeError("A query-role HF encoder cannot encode documents")
        documents = self._documents(values)
        if not documents:
            return np.empty((0, self._dimension), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            if self.encoder_family == "dpr":
                batches.append(self._dpr_documents(batch))
            else:
                texts = [f"{value.title or ''} {value.text}".strip() for value in batch]
                batches.append(self._contriever(texts))
        return np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.role != "query":
            raise RuntimeError("A document-role HF encoder cannot encode queries")
        values = self._strings(texts)
        if not values:
            return np.empty((0, self._dimension), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start : start + self.batch_size]
            if self.encoder_family == "dpr":
                batches.append(self._dpr_queries(batch))
            else:
                batches.append(self._contriever(batch))
        return np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)

    def embedding_space(self, similarity: str = "inner_product") -> EmbeddingSpaceSpec:
        return EmbeddingSpaceSpec(
            backend=self.backend,
            model_name=self.model_name,
            revision=self.resolved_revision,
            dimension=self._dimension,
            normalized=self.normalize,
            similarity=similarity,
            document_prefix="",
            max_sequence_length=self.max_sequence_length,
            encoder_family=self.encoder_family,
            pooling=self.pooling,
            document_input_format=self.document_input_format,
        )


def create_embedder(
    config: dict[str, Any],
    *,
    role: str = "document",
    override: dict[str, Any] | None = None,
) -> TextEmbedder | HFDenseEmbedder:
    if role not in {"document", "query"}:
        raise ValueError("role must be one of: document, query")
    embedding = {**config["embedding"], **(override or {})}
    if embedding["backend"] == "hf_dense":
        model_name = embedding["model_name"]
        revision = embedding["revision"]
        if role == "query":
            model_name = embedding.get("query_model_name", model_name)
            revision = embedding.get("query_revision", revision)
        return HFDenseEmbedder(
            family=embedding["family"],
            role=role,
            model_name=model_name,
            revision=revision,
            normalize=embedding["normalize"],
            batch_size=embedding.get("batch_size", 32),
            max_sequence_length=embedding["max_sequence_length"],
            pooling=embedding["pooling"],
            document_input_format=embedding["document_input_format"],
            local_files_only=embedding.get("local_files_only", False),
            device=embedding.get("device", "auto"),
        )
    return TextEmbedder(
        backend=embedding["backend"],
        model_name=embedding.get("model_name"),
        revision=embedding.get("revision"),
        normalize=embedding["normalize"],
        batch_size=embedding.get("batch_size", 32),
        dimension=embedding.get("dimension", 384),
        query_prefix=embedding["query_prefix"],
        document_prefix=embedding["document_prefix"],
        max_sequence_length=embedding.get("max_sequence_length"),
        local_files_only=embedding.get("local_files_only", False),
        device=embedding.get("device", "auto"),
    )
