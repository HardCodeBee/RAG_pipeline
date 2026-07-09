from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class EmbedderInfo:
    backend: str
    model_name: str
    dimension: int
    normalized: bool


class HashingEmbedder:
    def __init__(self, dimension: int = 384, normalize: bool = True):
        self.dimension = dimension
        self.normalize = normalize
        self.model_name = f"hashing-{dimension}"

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little", signed=False)
                index = value % self.dimension
                sign = 1.0 if (value >> 63) == 0 else -1.0
                vectors[row, index] += sign
        if self.normalize:
            vectors = l2_normalize(vectors)
        return vectors.astype(np.float32)

    def info(self) -> EmbedderInfo:
        return EmbedderInfo("hashing", self.model_name, self.dimension, self.normalize)


class TextEmbedder:
    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "BAAI/bge-small-en-v1.5",
        normalize: bool = True,
        batch_size: int = 32,
        fallback_dim: int = 384,
    ):
        self.backend = backend
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.fallback_dim = fallback_dim
        self._model = None
        self._active_backend = ""
        self._dimension = fallback_dim

        if backend in {"auto", "sentence_transformers"}:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self._active_backend = "sentence_transformers"
                dimension = self._model.get_sentence_embedding_dimension()
                self._dimension = int(dimension or fallback_dim)
                return
            except Exception:
                if backend == "sentence_transformers":
                    raise

        self._model = HashingEmbedder(dimension=fallback_dim, normalize=normalize)
        self._active_backend = "hashing"
        self._dimension = fallback_dim

    @classmethod
    def from_config(cls, config: dict) -> "TextEmbedder":
        embedding = config.get("embedding", {})
        return cls(
            backend=embedding.get("backend", "auto"),
            model_name=embedding.get("model_name", "BAAI/bge-small-en-v1.5"),
            normalize=bool(embedding.get("normalize", True)),
            batch_size=int(embedding.get("batch_size", 32)),
            fallback_dim=int(embedding.get("fallback_dim", 384)),
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._active_backend == "sentence_transformers":
            embeddings = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)
        return self._model.encode(texts, batch_size=self.batch_size)

    def info(self) -> EmbedderInfo:
        if self._active_backend == "sentence_transformers":
            return EmbedderInfo(
                self._active_backend,
                self.model_name,
                self._dimension,
                self.normalize,
            )
        return self._model.info()

