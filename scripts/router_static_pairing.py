"""Lookup-only corpus-conditioned query features for a frozen BGE document bank.

This does not run either retriever, encode queries, or load outcome labels.
Callers must supply the actual Dense query vector, not the M6 hidden mean.
The bank must cover the full frozen BM25 vocabulary before it is used.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import sqlite3

import numpy as np

from src.retrievers.sqlite_bm25 import analyze_sqlite_bm25_text


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/static_pairing_bank_v1"
DENSE = ROOT / "artifacts/_encoded_corpora/encoded_corpus_df8e4c53d3e183b4"
DENSE_QUERY_SPACE = "BAAI_bge_small_en_v1_5_5c38ec7c405ec4b44b94cc5a9bb96e735b38267a_default_normalized_max512"
CONTROL_SEED = 2026091882


def fixed_coordinate_control(dimension: int = 384) -> tuple[np.ndarray, np.ndarray]:
    """One fixed signed derangement; no result-dependent seed selection.

    Conjugating a one-step cycle by a random ordering produces no fixed points.
    This preserves vector norms/geometry but not alignment with unrotated q.
    It does not permute document identities.
    """
    rng = np.random.default_rng(CONTROL_SEED)
    order = rng.permutation(dimension)
    permutation = np.empty(dimension, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=dimension)
    return permutation, signs


class StaticPairingBank:
    """Full-vocabulary term-centroid lookup with explicit corpus-OOV status."""

    def __init__(self, query_space: str, directory: Path = BANK):
        if query_space != DENSE_QUERY_SPACE:
            raise ValueError("A query vector in the actual frozen Dense space is required")
        self.directory = Path(directory)
        summary = json.loads((self.directory / "summary.json").read_text(encoding="utf-8"))
        if not summary.get("complete", False):
            raise ValueError("The full corpus bank has not completed")
        with np.load(self.directory / "stats.npz", allow_pickle=False) as source:
            self.stats = {key: source[key].copy() for key in source.files}
        self.centroids = np.load(self.directory / "centroids.npy", mmap_mode="r", allow_pickle=False)
        self.db = sqlite3.connect((self.directory / "vocabulary.sqlite3").as_uri() + "?mode=ro", uri=True)
        self.db.execute("PRAGMA query_only=ON")
        manifest = json.loads((DENSE / "manifest.json").read_text(encoding="utf-8"))
        self.parts = manifest["artifacts"]["embeddings"]["parts"]
        self.starts = np.asarray([part["start_row"] for part in self.parts])
        self.documents = int(manifest["corpus"]["num_documents"])
        self.shards: OrderedDict[int, np.ndarray] = OrderedDict()
        self.permutation, self.signs = fixed_coordinate_control()

    def close(self) -> None:
        self.db.close()
        self.shards.clear()

    def _singleton_vector(self, document_id: int) -> np.ndarray:
        part_id = int(np.searchsorted(self.starts, document_id, side="right") - 1)
        if part_id not in self.shards:
            self.shards[part_id] = np.load(
                DENSE / "embeddings" / self.parts[part_id]["file"], mmap_mode="r", allow_pickle=False,
            )
            if len(self.shards) > 8:
                self.shards.popitem(last=False)
        self.shards.move_to_end(part_id)
        return np.asarray(self.shards[part_id][document_id - self.starts[part_id]], dtype=np.float64)

    def term_vectors(self, terms: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Known terms, aligned term IDs and centroids. Unknown means corpus-OOV."""
        known, ids, centroids = [], [], []
        for term in sorted(set(terms)):
            record = self.db.execute("SELECT term_id, multi_id FROM terms WHERE term=?", (term,)).fetchone()
            if record is None:
                continue
            term_id, multi_id = map(int, record)
            vector = (
                np.asarray(self.centroids[multi_id], dtype=np.float64)
                if multi_id >= 0
                else self._singleton_vector(int(self.stats["singleton_doc_id"][term_id]))
            )
            known.append(term)
            ids.append(term_id)
            centroids.append(vector)
        return known, np.asarray(ids, dtype=np.int64), np.asarray(centroids, dtype=np.float64).reshape(-1, 384)

    def features(self, question: str, dense_query: np.ndarray) -> dict[str, float]:
        """One IDF-weighted centroid, matching summaries and signed control.

        Each term is first normalized by its own BM25 mass. IDF weighting here
        deliberately avoids letting large document frequency dominate the query
        centroid. The resulting alignment is not full-score BM25/Dense covariance.
        A missing query centroid has zero numerical coordinates plus a mask.
        """
        q = np.asarray(dense_query, dtype=np.float64)
        if q.shape != (384,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
            raise ValueError("Expected a finite, normalized 384D actual Dense query vector")
        terms = sorted(set(analyze_sqlite_bm25_text(question)))
        _, ids, vectors = self.term_vectors(terms)
        available = len(ids) > 0
        result = {
            "log_unique_term_count": float(np.log1p(len(terms))),
            "corpus_known_fraction": float(len(ids) / len(terms)) if terms else 0.0,
            "centroid_available": float(available),
            "idf_mean": 0.0,
            "idf_std": 0.0,
            "idf_max": 0.0,
            "log_total_lexical_mass_per_document": 0.0,
            "centered_centroid_squared_norm": 0.0,
            "aligned": 0.0,
            "coordinate_control": 0.0,
        }
        if available:
            idf = self.stats["idf"][ids]
            center = idf @ vectors / idf.sum()
            centered = center - self.stats["corpus_mean"]
            result.update(
                idf_mean=float(idf.mean()), idf_std=float(idf.std()), idf_max=float(idf.max()),
                log_total_lexical_mass_per_document=float(np.log1p(self.stats["mass"][ids].sum() / self.documents)),
                centered_centroid_squared_norm=float(centered @ centered),
                aligned=float(q @ centered),
                coordinate_control=float(q @ (self.signs * centered[self.permutation])),
            )
        return result
