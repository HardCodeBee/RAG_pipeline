from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.run_eval import validate_resume_compatibility
from src.config import apply_cli_overrides, load_config, resolve_cli_path, validate_config
from src.provenance import (
    PIPELINE_SCHEMA_VERSION,
    build_identity,
    evaluation_spec,
    json_sha256,
    recorded_config,
    run_spec,
    source_code_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def test_active_schema_versions_are_explicit() -> None:
    assert PIPELINE_SCHEMA_VERSION == 6
    assert evaluation_spec("questions", "source")["metrics_version"] == "evidence_and_generation_v3"


def test_active_configs_use_strictness_instead_of_schema_or_profile_flags() -> None:
    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    baseline = load_config(ROOT / "configs" / "baseline.yaml")

    assert "schema_version" not in smoke and "schema_version" not in baseline
    assert "profile" not in smoke and "profile" not in baseline
    assert smoke["strict_backends"] is False
    assert baseline["strict_backends"] is True
    assert baseline["embedding"]["revision"]
    assert baseline["chunking"]["tokenizer_revision"]
    assert baseline["embedding"]["backend"] == "sentence_transformers"
    assert baseline["index"]["backend"] == "faiss"
    assert baseline["generation"]["provider"] == "openai"


def test_unknown_config_and_unpinned_strict_backend_are_rejected() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config["unused_plugin_section"] = {}
    with pytest.raises(ValueError, match="Unknown root"):
        validate_config(config)

    config = load_config(ROOT / "configs" / "baseline.yaml")
    config["embedding"]["revision"] = None
    with pytest.raises(ValueError, match="fixed embedding.revision"):
        validate_config(config)


def test_cli_top_k_changes_run_identity_but_not_build_identity() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    corpus = {"documents": [{"source": "paper.pdf", "sha256": "abc"}], "aggregate_sha256": "abc"}
    build_id, _, _ = build_identity(config, corpus, "build-code")
    overridden = apply_cli_overrides(config, top_k=9)

    assert build_identity(overridden, corpus, "build-code")[0] == build_id
    original_run = run_spec(config, build_id, "runtime-code")
    overridden_run = run_spec(overridden, build_id, "runtime-code")
    assert "profile" not in original_run
    assert json_sha256(original_run) != json_sha256(overridden_run)
    assert overridden_run["retrieval"]["top_k"] == 9


def test_api_key_is_recorded_but_does_not_enter_scientific_identity() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    other = deepcopy(config)
    other["paths"]["outputs_root"] = "somewhere-else"
    other["generation"]["api_key"] = "private-test-key"
    corpus = {"documents": [], "aggregate_sha256": "empty"}
    build_id, _, _ = build_identity(config, corpus, "build-code")

    assert build_identity(other, corpus, "build-code")[0] == build_id
    assert run_spec(config, build_id, "runtime-code") == run_spec(other, build_id, "runtime-code")
    assert recorded_config(other)["generation"]["api_key"] == "private-test-key"


def test_cli_paths_are_resolved_from_project_root() -> None:
    assert resolve_cli_path(ROOT, "data/questions_v1.jsonl") == (ROOT / "data/questions_v1.jsonl").resolve()


def test_source_identity_covers_src_and_scripts_without_a_manual_file_list(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "pipeline.py").write_text("VERSION = 1\n", encoding="utf-8")
    first = source_code_sha256(tmp_path)
    (tmp_path / "scripts" / "run.py").write_text("print('run')\n", encoding="utf-8")
    assert source_code_sha256(tmp_path) != first


def test_resume_requires_exact_scientific_identity() -> None:
    current = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "questions_sha256": "q",
        "build_id": "b",
        "run_spec_sha256": "r",
        "evaluation_spec_sha256": "e",
        "source_sha256": "c",
        "effective_top_k": 5,
    }
    validate_resume_compatibility(current, dict(current))
    with pytest.raises(ValueError, match="schema_version"):
        validate_resume_compatibility(dict(current, schema_version=5), current)
    incompatible = dict(current, effective_top_k=10)
    with pytest.raises(ValueError, match="effective_top_k"):
        validate_resume_compatibility(current, incompatible)
