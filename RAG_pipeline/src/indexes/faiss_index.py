from __future__ import annotations

from pathlib import Path

import numpy as np


class FlatIPIndex:
    def __init__(self, backend: str = "auto"):
        self.requested_backend = backend
        self.backend = ""
        self.index = None
        self.embeddings: np.ndarray | None = None
        self.dimension = 0

    def build(self, embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        self.dimension = embeddings.shape[1]

        if self.requested_backend in {"auto", "faiss"}:
            try:
                import faiss

                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(embeddings)
                self.backend = "faiss"
                self.embeddings = None
                return
            except Exception:
                if self.requested_backend == "faiss":
                    raise

        self.backend = "numpy"
        self.embeddings = embeddings
        self.index = None

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if self.backend == "faiss":
            scores, indices = self.index.search(query_embedding, top_k)
            return scores[0], indices[0]

        if self.embeddings is None:
            raise RuntimeError("NumPy index has not been built or loaded")
        scores = self.embeddings @ query_embedding[0]
        top_k = min(top_k, len(scores))
        indices = np.argsort(-scores)[:top_k]
        return scores[indices], indices.astype(np.int64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss":
            import faiss

            faiss.write_index(self.index, str(path))
            return

        if self.embeddings is None:
            raise RuntimeError("NumPy index has no embeddings to save")
        with path.open("wb") as handle:
            np.savez_compressed(handle, embeddings=self.embeddings)

    def load(self, path: str | Path, embeddings_path: str | Path | None = None) -> None:
        path = Path(path)
        if self.requested_backend in {"auto", "faiss"}:
            try:
                import faiss

                self.index = faiss.read_index(str(path))
                self.backend = "faiss"
                self.dimension = self.index.d
                self.embeddings = None
                return
            except Exception:
                if self.requested_backend == "faiss":
                    raise

        source = Path(embeddings_path) if embeddings_path else path
        try:
            loaded = np.load(source)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                self.embeddings = loaded["embeddings"].astype(np.float32)
            else:
                self.embeddings = loaded.astype(np.float32)
        except Exception:
            with path.open("rb") as handle:
                loaded = np.load(handle)
                self.embeddings = loaded["embeddings"].astype(np.float32)

        self.backend = "numpy"
        self.index = None
        self.dimension = self.embeddings.shape[1]

