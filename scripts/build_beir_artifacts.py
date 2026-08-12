"""Build reusable dense and SQLite BM25 artifacts for prepared BEIR units."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_support import configure_utf8_output
from scripts.run_beir_suite import (
    DEFAULT_BM25_CONFIG,
    DEFAULT_DENSE_CONFIG,
    PreparedUnit,
    _load_suite_template,
    _unit_config,
    discover_prepared_units,
    select_prepared_units,
)
from src.config import resolve_cli_path
from src.index_builder import build_index
from src.provenance import resolved_roots, sha256_file
from src.retrievers.bm25_index import build_bm25_index


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(value: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _complete_directories(root: Path) -> set[Path]:
    if not root.is_dir():
        return set()
    complete: set[Path] = set()
    for directory in root.iterdir():
        manifest_path = directory / "manifest.json"
        if not directory.is_dir() or not manifest_path.is_file():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(manifest, Mapping) and manifest.get("status") == "complete":
            complete.add(directory.resolve())
    return complete


def _unit_result(
    unit: PreparedUnit,
    verified_build: Any,
    verified_sparse_index: Any,
    *,
    dense_reused: bool,
    sparse_reused: bool,
    elapsed_ms: float,
) -> dict[str, Any]:
    build_manifest = verified_build.manifest
    build_artifacts = build_manifest["artifacts"]
    embeddings = build_artifacts["embeddings"]
    sparse_manifest = verified_sparse_index.manifest
    return {
        "dataset": unit.dataset,
        "unit": unit.unit,
        "prepared_directory": str(unit.directory),
        "prepared_manifest_sha256": unit.manifest_sha256,
        "dense": {
            "build_id": build_manifest["build_id"],
            "directory": str(verified_build.directory),
            "reused_complete_artifact": dense_reused,
            "rows": build_artifacts["chunks"]["rows"],
            "embedding_storage": embeddings.get("storage", "single_npy"),
            "embedding_shape": embeddings.get("shape"),
            "embedding_parts": len(embeddings.get("parts", ())),
        },
        "bm25": {
            "backend": sparse_manifest.get("backend"),
            "sparse_index_id": sparse_manifest["sparse_index_id"],
            "directory": str(verified_sparse_index.directory),
            "reused_complete_artifact": sparse_reused,
            "document_count": sparse_manifest["document_count"],
        },
        "elapsed_ms": elapsed_ms,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or resume sharded dense and SQLite BM25 artifacts for "
            "already-prepared BEIR units."
        )
    )
    parser.add_argument("--data-root", default="data/beir")
    parser.add_argument("--dense-config", default=DEFAULT_DENSE_CONFIG)
    parser.add_argument("--bm25-config", default=DEFAULT_BM25_CONFIG)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Repeat for dataset families/units; default is all-prepared.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the SQLite BM25 builder progress display.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    configure_utf8_output()
    started = time.perf_counter()

    data_root = resolve_cli_path(PROJECT_ROOT, args.data_root)
    dense_path = resolve_cli_path(PROJECT_ROOT, args.dense_config)
    bm25_path = resolve_cli_path(PROJECT_ROOT, args.bm25_config)
    prepared = discover_prepared_units(data_root)
    selected = select_prepared_units(prepared, args.dataset)
    dense_template = _load_suite_template(dense_path, method="dense")
    bm25_template = _load_suite_template(bm25_path, method="bm25")
    dense_artifacts_root = resolved_roots(dense_template)["artifacts_root"]
    bm25_artifacts_root = resolved_roots(bm25_template)["artifacts_root"]
    if dense_artifacts_root != bm25_artifacts_root:
        raise ValueError("Dense and BM25 templates must share artifacts_root")

    completed: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "event": "summary",
        "command": "build_beir_artifacts",
        "status": "running",
        "data_root": str(data_root),
        "dense_template": {
            "path": str(dense_path),
            "sha256": sha256_file(dense_path),
        },
        "bm25_template": {
            "path": str(bm25_path),
            "sha256": sha256_file(bm25_path),
        },
        "selected_units": [unit.unit for unit in selected],
        "completed_units": completed,
        "failed_unit": None,
        "error": None,
    }
    _emit(
        {
            "event": "plan",
            "timestamp_utc": _utc_now(),
            "units": [unit.unit for unit in selected],
            "artifacts_root": str(dense_artifacts_root),
        }
    )

    current_unit: PreparedUnit | None = None
    current_stage: str | None = None
    try:
        for position, unit in enumerate(selected, start=1):
            current_unit = unit
            unit_started = time.perf_counter()
            dense_config = _unit_config(dense_template, unit)
            bm25_config = _unit_config(bm25_template, unit)
            if (
                resolved_roots(dense_config)["artifacts_root"]
                != resolved_roots(bm25_config)["artifacts_root"]
            ):
                raise ValueError(
                    "Dense and BM25 unit configs use different artifact roots: "
                    f"{unit.unit}"
                )

            _emit(
                {
                    "event": "unit_started",
                    "timestamp_utc": _utc_now(),
                    "unit": unit.unit,
                    "dataset": unit.dataset,
                    "position": position,
                    "total_units": len(selected),
                }
            )

            current_stage = "dense"
            complete_builds = _complete_directories(dense_artifacts_root)
            _emit(
                {
                    "event": "stage_started",
                    "timestamp_utc": _utc_now(),
                    "unit": unit.unit,
                    "stage": current_stage,
                }
            )
            verified_build = build_index(dense_config)
            dense_reused = verified_build.directory.resolve() in complete_builds
            _emit(
                {
                    "event": "stage_completed",
                    "timestamp_utc": _utc_now(),
                    "unit": unit.unit,
                    "stage": current_stage,
                    "artifact_id": verified_build.manifest["build_id"],
                    "directory": str(verified_build.directory),
                    "reused_complete_artifact": dense_reused,
                }
            )

            current_stage = "bm25"
            sparse_root = dense_artifacts_root / "_sparse_indexes"
            complete_sparse_indexes = _complete_directories(sparse_root)
            _emit(
                {
                    "event": "stage_started",
                    "timestamp_utc": _utc_now(),
                    "unit": unit.unit,
                    "stage": current_stage,
                }
            )
            verified_sparse_index = build_bm25_index(
                bm25_config,
                verified_build,
                show_progress=not args.no_progress,
            )
            sparse_reused = (
                verified_sparse_index.directory.resolve()
                in complete_sparse_indexes
            )
            _emit(
                {
                    "event": "stage_completed",
                    "timestamp_utc": _utc_now(),
                    "unit": unit.unit,
                    "stage": current_stage,
                    "artifact_id": verified_sparse_index.manifest[
                        "sparse_index_id"
                    ],
                    "directory": str(verified_sparse_index.directory),
                    "reused_complete_artifact": sparse_reused,
                }
            )

            unit_result = _unit_result(
                unit,
                verified_build,
                verified_sparse_index,
                dense_reused=dense_reused,
                sparse_reused=sparse_reused,
                elapsed_ms=(time.perf_counter() - unit_started) * 1000,
            )
            completed.append(unit_result)
            _emit(
                {
                    "event": "unit_completed",
                    "timestamp_utc": _utc_now(),
                    **unit_result,
                }
            )
            current_stage = None
    except Exception as exc:
        error = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "stage": current_stage,
        }
        summary.update(
            {
                "status": "failed",
                "completed_at": _utc_now(),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "completed_count": len(completed),
                "failed_unit": current_unit.unit if current_unit else None,
                "error": error,
            }
        )
        _emit(
            {
                "event": "error",
                "timestamp_utc": _utc_now(),
                "unit": summary["failed_unit"],
                **error,
            }
        )
        _emit(summary)
        raise

    summary.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "completed_count": len(completed),
        }
    )
    _emit(summary)
    return summary


if __name__ == "__main__":
    main()
