from __future__ import annotations

from pathlib import Path

import pytest

import src.index_builder as index_builder_module
from src.artifact_io import validate_build_directory
from src.core.records import PageRecord
from src.index_builder import build_index
from src.pipeline import NaiveRAGPipeline


class FakeLoader:
    def __init__(self, document: Path):
        self.document = document

    def discover(self, corpus_path, file_type="pdf"):
        return [self.document]

    def load(self, corpus_path, file_type="pdf"):
        return [
            PageRecord(
                "paper",
                self.document.name,
                1,
                "Dense retrieval finds evidence. Retrieval augmented generation uses that evidence to answer questions.",
            )
        ]


def _config(tmp_path: Path) -> dict:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "paper.pdf").write_bytes(b"identity-only-pdf-bytes")
    return {
        "_base_dir": str(tmp_path),
        "strict_backends": False,
        "project": {"name": "test"},
        "paths": {"corpus": "corpus", "artifacts_root": "artifacts", "outputs_root": "outputs"},
        "loader": {"type": "pypdf", "recursive": False, "empty_page_policy": "skip", "cleaner": "minimal"},
        "chunking": {
            "method": "fixed_sentence",
            "chunk_size_tokens": 20,
            "overlap_budget_tokens": 4,
            "sentence_splitter": "regex",
            "tokenizer": "regex",
            "local_files_only": True,
        },
        "embedding": {
            "backend": "hashing",
            "model_name": "hashing-32",
            "revision": None,
            "normalize": True,
            "batch_size": 2,
            "fallback_dim": 32,
            "query_prefix": "",
            "document_prefix": "",
            "max_sequence_length": None,
            "local_files_only": True,
        },
        "index": {"backend": "numpy", "type": "flat_ip"},
        "retrieval": {"type": "dense", "top_k": 1},
        "context": {"max_tokens": None},
        "prompt": {"version": "fixed_qa_v1"},
        "generation": {
            "provider": "extractive",
            "model": "extractive-fallback",
            "temperature": 0.0,
            "max_output_tokens": 64,
            "timeout_seconds": 10.0,
            "max_retries": 0,
        },
        "logging": {
            "save_retrieved_chunks": True,
            "save_prompt": True,
            "save_latency": True,
            "save_token_usage": True,
        },
    }


def test_build_is_immutable_reusable_and_queryable(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    document = tmp_path / "corpus" / "paper.pdf"
    monkeypatch.setattr(index_builder_module, "create_loader", lambda _: FakeLoader(document))

    first = build_index(config)
    second = build_index(config)
    assert first["build_id"] == second["build_id"]
    assert "schema_version" not in first
    assert "schema_version" not in first["build_spec"]
    assert first["build_spec"]["source_sha256"]
    assert "source_sha256" not in first
    assert "effective_config" not in first
    assert set(first["corpus"]) == {"num_files", "num_pages", "num_documents"}
    assert set(first["chunking"]) == {"num_chunks", "token_count", "realized_overlap_tokens"}
    assert set(first["embedding"]) == {"space"}
    assert "query_prefix" not in first["embedding"]["space"]
    assert first["embedding"]["space"]["dimension"] == 32
    assert set(first["index"]) == {"backend", "type", "count", "dimension"}
    assert "vector_id_sequence_sha256" in first
    assert "vector_ids_sha256" not in first["artifacts"]["chunks"]
    assert "vector_ids_sha256" not in first["artifacts"]["index"]
    assert "rows" not in first["artifacts"]["embeddings"]
    assert "rows" not in first["artifacts"]["index"]
    build_dir = tmp_path / "artifacts" / first["build_id"]
    assert "text_sha256" not in (build_dir / "chunks.jsonl").read_text(encoding="utf-8")
    validate_build_directory(build_dir, first["build_id"])

    # query_prefix 是运行配置；修改它应复用同一构建，而不是污染文档向量空间。
    config["embedding"]["query_prefix"] = "query: "
    config["generation"]["api_key"] = "plaintext-test-key"
    pipeline = NaiveRAGPipeline(config)
    assert set(pipeline.runtime_metadata) == {
        "build_id",
        "build_dir",
        "build_spec_sha256",
        "source_sha256",
        "run_spec",
        "run_spec_sha256",
    }
    assert pipeline.embedder.query_prefix == "query: "
    result = pipeline.query("What does dense retrieval find?", question_id="q")
    assert "schema_version" not in result
    assert "notes" not in result
    assert result["identity"]["build_id"] == first["build_id"]
    assert result["retrieval"]["results"]
    assert result["status"] == "success"
    assert "status" not in result["generation"]


def test_tampered_artifact_is_rejected(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    document = tmp_path / "corpus" / "paper.pdf"
    monkeypatch.setattr(index_builder_module, "create_loader", lambda _: FakeLoader(document))
    manifest = build_index(config)
    build_dir = tmp_path / "artifacts" / manifest["build_id"]
    with (build_dir / "chunks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash|size"):
        validate_build_directory(build_dir)


def test_failed_build_leaves_no_visible_build_directory(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    document = tmp_path / "corpus" / "paper.pdf"
    monkeypatch.setattr(index_builder_module, "create_loader", lambda _: FakeLoader(document))

    class BrokenEmbedder:
        def encode_documents(self, texts):
            raise RuntimeError("synthetic failure")

    monkeypatch.setattr(index_builder_module, "create_embedder", lambda _: BrokenEmbedder())
    with pytest.raises(RuntimeError, match="synthetic failure"):
        build_index(config)
    artifacts = tmp_path / "artifacts"
    assert not list(artifacts.glob("build_*"))
