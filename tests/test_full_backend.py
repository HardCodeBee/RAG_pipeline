from __future__ import annotations

import os

import numpy as np
import pytest

from src.embedders.sbert_embedder import TextEmbedder
from src.indexes.faiss_index import FlatIPIndex
from src.text.token_counters import HuggingFaceTokenCounter


if os.environ.get("RUN_FULL_BACKEND_TESTS") != "1":
    pytest.skip("full backend gate is opt-in", allow_module_level=True)

pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")

pytestmark = pytest.mark.full_backend

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def test_pinned_bge_tokenizer_and_embedder_contract() -> None:
    counter = HuggingFaceTokenCounter(MODEL, revision=REVISION, local_files_only=True)
    parts = counter.split("retrieval augmented generation " * 400, 300)

    assert parts
    assert max(counter.count(part) for part in parts) <= 300

    embedder = TextEmbedder(
        backend="sentence_transformers",
        model_name=MODEL,
        revision=REVISION,
        normalize=True,
        batch_size=2,
        local_files_only=True,
    )
    vectors = embedder.encode_documents(["dense retrieval", "retrieval augmented generation"])
    space = embedder.embedding_space("inner_product")

    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert space.revision == REVISION
    assert space.max_sequence_length == 512


def test_faiss_flat_ip_round_trip_with_explicit_ids(tmp_path) -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.5, 0.5], [-1.0, 0.0]], dtype=np.float32)
    vector_ids = np.asarray([101, 9001, 42], dtype=np.int64)
    path = tmp_path / "index.faiss"

    index = FlatIPIndex(backend="faiss")
    index.build(embeddings, ids=vector_ids)
    index.save(path)

    loaded = FlatIPIndex(backend="faiss")
    loaded.load(path)
    hits = loaded.search_hits(np.asarray([[1.0, 0.0]], dtype=np.float32), 3)

    assert loaded.backend == "faiss"
    assert [hit.vector_id for hit in hits] == [101, 9001, 42]
