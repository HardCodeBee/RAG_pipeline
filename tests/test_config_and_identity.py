from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.config import apply_cli_overrides, load_config, resolve_cli_path, validate_config
from src.evaluators.logger import write_metadata_json
from src.evaluators.runner import validate_resume_compatibility
from src.provenance import (
    build_identity,
    evaluation_spec,
    json_sha256,
    run_spec,
    source_group_sha256,
    source_snapshot_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def test_identity_specs_do_not_depend_on_pipeline_schema_constants() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    corpus = {"documents": [], "aggregate_sha256": "empty"}
    build_id, build_spec_sha, build_spec_value = build_identity(config, corpus, "source")
    run_spec_value = run_spec(config, build_id, "source")

    assert "schema_version" not in build_spec_value
    assert "schema_version" not in run_spec_value
    assert "schema_version" not in evaluation_spec("questions", "source")
    assert build_spec_value["build_source_sha256"] == "source"
    assert run_spec_value["run_source_sha256"] == "source"
    assert evaluation_spec("questions", "source")["evaluation_source_sha256"] == "source"
    assert build_spec_sha == json_sha256(build_spec_value)
    assert evaluation_spec("questions", "source")["metrics_version"] == "evidence_and_generation_v3"


def test_active_configs_use_explicit_pinned_backends() -> None:
    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    baseline = load_config(ROOT / "configs" / "baseline.yaml")

    assert "schema_version" not in smoke and "schema_version" not in baseline
    assert "_config_path" not in smoke and "_config_path" not in baseline
    assert "profile" not in smoke and "profile" not in baseline
    assert "strict_backends" not in smoke and "strict_backends" not in baseline
    assert baseline["embedding"]["revision"]
    assert baseline["chunking"]["tokenizer_revision"]
    assert baseline["embedding"]["backend"] == "sentence_transformers"
    assert baseline["index"]["backend"] == "faiss"
    assert baseline["generation"]["provider"] == "openai"


def test_unknown_config_and_unpinned_model_backend_are_rejected() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config["unused_plugin_section"] = {}
    with pytest.raises(ValueError, match="Unknown root"):
        validate_config(config)

    config = load_config(ROOT / "configs" / "baseline.yaml")
    config["embedding"]["revision"] = None
    with pytest.raises(ValueError, match="embedding.revision"):
        validate_config(config)


def test_backend_specific_config_rejects_irrelevant_fields() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config["chunking"]["local_files_only"] = True
    with pytest.raises(ValueError, match="Unknown chunking"):
        validate_config(config)

    config = load_config(ROOT / "configs" / "smoke.yaml")
    config["embedding"]["model_name"] = "unused"
    with pytest.raises(ValueError, match="Unknown embedding"):
        validate_config(config)

    config = load_config(ROOT / "configs" / "smoke.yaml")
    config["generation"]["timeout_seconds"] = 60.0
    with pytest.raises(ValueError, match="Unknown generation"):
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


def test_inline_api_key_is_rejected_and_output_path_does_not_enter_identity() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    other = deepcopy(config)
    other["paths"]["outputs_root"] = "somewhere-else"
    corpus = {"documents": [], "aggregate_sha256": "empty"}
    build_id, _, _ = build_identity(config, corpus, "build-code")

    assert build_identity(other, corpus, "build-code")[0] == build_id
    assert run_spec(config, build_id, "runtime-code") == run_spec(other, build_id, "runtime-code")
    other["generation"]["api_key"] = "private-test-key"
    with pytest.raises(ValueError, match="must not contain an inline secret"):
        validate_config(other)


def test_metadata_writer_removes_nested_credentials_and_redacts_key_shapes(tmp_path) -> None:
    path = tmp_path / "metadata.json"
    sentinel = "sk-abcdefghijklmnopqrstuvwxyz012345"

    write_metadata_json(
        path,
        {
            "effective_config": {"generation": {"api_key": sentinel, "model": "model"}},
            "generator": {"api_key_present": True},
            "error": f"request rejected for {sentinel}",
        },
    )
    raw = path.read_text(encoding="utf-8")
    value = json.loads(raw)

    assert sentinel not in raw
    assert "api_key" not in value["effective_config"]["generation"]
    assert value["generator"]["api_key_present"] is True
    assert "[REDACTED]" in value["error"]


def test_cli_paths_are_resolved_from_project_root() -> None:
    assert resolve_cli_path(ROOT, "data/questions_v1.jsonl") == (ROOT / "data/questions_v1.jsonl").resolve()


def test_source_groups_invalidate_only_their_pipeline_stage(tmp_path) -> None:
    (tmp_path / "src" / "evaluators").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    build_file = tmp_path / "src" / "index_builder.py"
    run_file = tmp_path / "src" / "pipeline.py"
    evaluation_file = tmp_path / "src" / "evaluators" / "metrics.py"
    build_file.write_text("VERSION = 1\n", encoding="utf-8")
    run_file.write_text("VERSION = 1\n", encoding="utf-8")
    evaluation_file.write_text("VERSION = 1\n", encoding="utf-8")

    initial = {
        group: source_group_sha256(tmp_path, group)
        for group in ("build", "run", "evaluation")
    }
    initial_snapshot = source_snapshot_sha256(tmp_path)

    evaluation_file.write_text("VERSION = 2\n", encoding="utf-8")
    assert source_group_sha256(tmp_path, "evaluation") != initial["evaluation"]
    assert source_group_sha256(tmp_path, "build") == initial["build"]
    assert source_group_sha256(tmp_path, "run") == initial["run"]
    assert source_snapshot_sha256(tmp_path) != initial_snapshot

    build_file.write_text("VERSION = 2\n", encoding="utf-8")
    assert source_group_sha256(tmp_path, "build") != initial["build"]
    assert source_group_sha256(tmp_path, "run") == initial["run"]

    run_file.write_text("VERSION = 2\n", encoding="utf-8")
    assert source_group_sha256(tmp_path, "run") != initial["run"]


def test_resume_requires_exact_scientific_identity() -> None:
    current = {
        "questions_sha256": "q",
        "build_id": "b",
        "run_spec_sha256": "r",
        "evaluation_spec_sha256": "e",
        "effective_top_k": 5,
    }
    validate_resume_compatibility(current, dict(current))
    incompatible = dict(current, effective_top_k=10)
    with pytest.raises(ValueError, match="effective_top_k"):
        validate_resume_compatibility(current, incompatible)
