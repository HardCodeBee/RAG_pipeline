"""Validate and combine completed four-condition BEIR suite summaries.

This is a read-only consumer of suite directories.  It never opens pipelines,
models, indexes, result rows, or GPU resources.  CQADupStack is represented by
the existing per-suite family macro, so its twelve units contribute one family
to the cross-suite macro average.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_support import configure_utf8_output
from src.provenance import json_sha256, sha256_file


AGGREGATE_SCHEMA_VERSION = 2
SUITE_SCHEMA_VERSION = 1
EXPECTED_EVALUATION_PROTOCOL = (
    "beir_retrieval_only_v2_ignore_identical_pre_candidate"
)
EXPECTED_METRICS_VERSION = "beir_retrieval_v2_trec_linear_ndcg_binary_recall"
EXPECTED_PHYSICAL_CANDIDATE_K = 50
EXPECTED_FINAL_K = 5
EXPECTED_CONDITIONS = (
    "bm25_top5",
    "dense_top5",
    "bm25_top50_bge_top5",
    "dense_top50_bge_top5",
)
_CONDITION_POSITION = {
    condition: position for position, condition in enumerate(EXPECTED_CONDITIONS)
}
_DISCOVERY_PATTERN = "beir_*_four_condition_full_*"
_METRIC_PREFIXES = (
    "ndcg_at_",
    "map_at_",
    "recall_at_",
    "precision_at_",
    "mrr_at_",
    "hit_at_",
    "first_stage_",
    "candidate_pool_",
)


@dataclass(frozen=True, slots=True)
class ValidatedSuite:
    directory: Path
    suite_id: str
    contract: dict[str, Any]
    family_results: tuple[dict[str, Any], ...]
    metadata_artifact: dict[str, Any]
    summary_artifact: dict[str, Any]


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Aggregate source artifact is missing: {path}")
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _strict_positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_empty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _condition_set(value: Any, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) != len(EXPECTED_CONDITIONS)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
        or set(value) != set(EXPECTED_CONDITIONS)
    ):
        raise ValueError(
            f"{label} must contain exactly the four canonical BEIR conditions"
        )
    return EXPECTED_CONDITIONS


def _suite_contract(
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    directory: Path,
) -> dict[str, Any]:
    identity = metadata.get("suite_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"Suite metadata has no identity: {directory}")
    if metadata.get("suite_identity_sha256") != json_sha256(identity):
        raise ValueError(f"Suite identity hash is invalid: {directory}")
    identity_schema = identity.get("schema_version")
    if (
        isinstance(identity_schema, bool)
        or not isinstance(identity_schema, int)
        or identity_schema != SUITE_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported suite identity schema: {directory}")

    evaluation_protocol = _non_empty_string(
        identity.get("evaluation_protocol"),
        label="suite evaluation_protocol",
    )
    metrics_version = _non_empty_string(
        identity.get("metrics_version"),
        label="suite metrics_version",
    )
    physical_candidate_k = _strict_positive_integer(
        identity.get("physical_candidate_k"),
        label="suite physical_candidate_k",
    )
    final_k = _strict_positive_integer(
        identity.get("final_k"),
        label="suite final_k",
    )
    conditions = _condition_set(
        identity.get("conditions"),
        label="suite conditions",
    )
    if evaluation_protocol != EXPECTED_EVALUATION_PROTOCOL:
        raise ValueError(
            f"Suite evaluation_protocol is incompatible: {directory}"
        )
    if metrics_version != EXPECTED_METRICS_VERSION:
        raise ValueError(f"Suite metrics_version is incompatible: {directory}")
    if physical_candidate_k != EXPECTED_PHYSICAL_CANDIDATE_K:
        raise ValueError(f"Suite physical_candidate_k is incompatible: {directory}")
    if final_k != EXPECTED_FINAL_K:
        raise ValueError(f"Suite final_k is incompatible: {directory}")
    if identity.get("max_questions") is not None:
        raise ValueError(f"Only full suites can be aggregated: {directory}")
    if identity.get("split") != "test":
        raise ValueError(f"Only BEIR test suites can be aggregated: {directory}")

    expected_summary_values = {
        "evaluation_protocol": evaluation_protocol,
        "metrics_version": metrics_version,
        "physical_candidate_k": physical_candidate_k,
        "final_k": final_k,
    }
    for key, expected in expected_summary_values.items():
        if summary.get(key) != expected:
            raise ValueError(f"Suite summary {key} differs from metadata: {directory}")
    return {
        **expected_summary_values,
        "conditions": list(conditions),
    }


def _numeric_metrics(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty metric mapping")
    metrics: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} has an invalid metric name")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{label}.{key} must be numeric")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{label}.{key} must be finite")
        metrics[key] = number
    return metrics


def _macro_metric_values(summaries: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not summaries:
        raise ValueError("Cannot compute a macro over no summaries")
    names = set.intersection(
        *(
            {
                key
                for key, value in summary.items()
                if (
                    key.startswith(_METRIC_PREFIXES)
                    or key == "rerank_delta_ndcg"
                )
                and not key.endswith("_valid_count")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
            for summary in summaries
        )
    )
    if not names:
        raise ValueError("Suite summaries share no aggregate retrieval metrics")
    return {
        name: mean(float(summary[name]) for summary in summaries)
        for name in sorted(names)
    }


def _validate_unit_results(
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    directory: Path,
) -> tuple[
    dict[tuple[str, str], list[Mapping[str, Any]]],
    dict[str, set[str]],
]:
    records = summary.get("unit_results")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Suite has no unit results: {directory}")
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    family_units: defaultdict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str, str]] = set()
    conditions_by_unit: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    condition_semantics = {
        "bm25_top5": ("bm25", "none", EXPECTED_FINAL_K),
        "dense_top5": ("dense", "none", EXPECTED_FINAL_K),
        "bm25_top50_bge_top5": (
            "bm25",
            "cross_encoder",
            EXPECTED_PHYSICAL_CANDIDATE_K,
        ),
        "dense_top50_bge_top5": (
            "dense",
            "cross_encoder",
            EXPECTED_PHYSICAL_CANDIDATE_K,
        ),
    }
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Suite unit result {position} is not an object")
        dataset = _non_empty_string(
            record.get("dataset"),
            label=f"unit_results[{position}].dataset",
        )
        unit = _non_empty_string(
            record.get("unit"),
            label=f"unit_results[{position}].unit",
        )
        condition = _non_empty_string(
            record.get("condition"),
            label=f"unit_results[{position}].condition",
        )
        if condition not in condition_semantics:
            raise ValueError(f"Unexpected BEIR condition: {condition}")
        key = (dataset, unit, condition)
        if key in seen:
            raise ValueError(
                f"Duplicate dataset/unit/condition result: "
                f"{dataset}/{unit}/{condition}"
            )
        seen.add(key)

        value = record.get("summary")
        if not isinstance(value, Mapping):
            raise ValueError(f"Unit summary is invalid: {unit}/{condition}")
        method, reranker, candidate_k = condition_semantics[condition]
        expected = {
            "dataset": dataset,
            "unit": unit,
            "evaluation_protocol": contract["evaluation_protocol"],
            "metrics_version": contract["metrics_version"],
            "retrieval_method": method,
            "reranker_provider": reranker,
            "candidate_k": candidate_k,
            "final_k": contract["final_k"],
            "effective_top_k": contract["final_k"],
        }
        for field, expected_value in expected.items():
            if value.get(field) != expected_value:
                raise ValueError(
                    f"Unit summary {field} is incompatible: {unit}/{condition}"
                )
        questions = _strict_positive_integer(
            value.get("num_questions"),
            label=f"{unit}/{condition} num_questions",
        )
        if (
            value.get("num_successful_questions") != questions
            or value.get("num_failed_questions") != 0
        ):
            raise ValueError(f"Unit condition is not fully successful: {unit}/{condition}")
        grouped[(dataset, condition)].append(value)
        family_units[dataset].add(unit)
        conditions_by_unit[(dataset, unit)].add(condition)

    expected_conditions = set(EXPECTED_CONDITIONS)
    for (dataset, unit), conditions in conditions_by_unit.items():
        if conditions != expected_conditions:
            raise ValueError(
                f"Unit does not contain all four conditions: {dataset}/{unit}"
            )
    return dict(grouped), dict(family_units)


def _validate_family_results(
    summary: Mapping[str, Any],
    grouped_units: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    family_units: Mapping[str, set[str]],
    *,
    directory: Path,
) -> tuple[dict[str, Any], ...]:
    records = summary.get("family_results")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Suite has no family results: {directory}")
    expected_keys = {
        (dataset, condition)
        for dataset in family_units
        for condition in EXPECTED_CONDITIONS
    }
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Family result {position} is not an object")
        dataset = _non_empty_string(
            record.get("dataset"),
            label=f"family_results[{position}].dataset",
        )
        condition = _non_empty_string(
            record.get("condition"),
            label=f"family_results[{position}].condition",
        )
        key = (dataset, condition)
        if key in found:
            raise ValueError(f"Duplicate family/condition result: {dataset}/{condition}")
        if key not in expected_keys:
            raise ValueError(f"Unexpected family/condition result: {dataset}/{condition}")
        num_units = _strict_positive_integer(
            record.get("num_units"),
            label=f"{dataset}/{condition} num_units",
        )
        if num_units != len(family_units[dataset]):
            raise ValueError(f"Family unit count is invalid: {dataset}/{condition}")
        if dataset == "cqadupstack" and num_units != 12:
            raise ValueError("CQADupStack must be one family macro over exactly 12 units")
        metrics = _numeric_metrics(
            record.get("metrics"),
            label=f"{dataset}/{condition} metrics",
        )
        recomputed = _macro_metric_values(grouped_units[key])
        if metrics != recomputed:
            raise ValueError(f"Stored family macro is invalid: {dataset}/{condition}")
        found[key] = {
            "dataset": dataset,
            "condition": condition,
            "num_units": num_units,
            "metrics": metrics,
        }
    if set(found) != expected_keys:
        missing = sorted(expected_keys - set(found))
        raise ValueError(f"Suite family results are incomplete: {missing}")
    return tuple(
        found[key]
        for key in sorted(
            found,
            key=lambda item: (item[0], _CONDITION_POSITION[item[1]]),
        )
    )


def validate_suite_directory(path: str | Path) -> ValidatedSuite:
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"BEIR suite directory does not exist: {directory}")
    metadata_path = directory / "metadata.json"
    summary_path = directory / "suite_summary.json"
    metadata = _read_json(metadata_path, label="BEIR suite metadata")
    summary = _read_json(
        summary_path,
        label="BEIR suite summary",
    )
    if metadata.get("status") != "completed" or metadata.get("last_error") is not None:
        raise ValueError(f"BEIR suite is not completed successfully: {directory}")
    if metadata.get("command") != "run_beir_suite":
        raise ValueError(f"BEIR suite command is incompatible: {directory}")
    suite_id = _non_empty_string(metadata.get("run_id"), label="suite run_id")
    if summary.get("suite_id") != suite_id:
        raise ValueError(f"Suite summary identity differs from metadata: {directory}")
    summary_schema = summary.get("schema_version")
    if (
        isinstance(summary_schema, bool)
        or not isinstance(summary_schema, int)
        or summary_schema != SUITE_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported suite summary schema: {directory}")

    contract = _suite_contract(metadata, summary, directory=directory)
    grouped, family_units = _validate_unit_results(
        summary,
        contract,
        directory=directory,
    )
    family_results = _validate_family_results(
        summary,
        grouped,
        family_units,
        directory=directory,
    )
    return ValidatedSuite(
        directory=directory,
        suite_id=suite_id,
        contract=contract,
        family_results=family_results,
        metadata_artifact=_artifact_descriptor(metadata_path),
        summary_artifact=_artifact_descriptor(summary_path),
    )


def discover_completed_suites(
    outputs_root: str | Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    root = Path(outputs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Outputs root does not exist: {root}")
    completed: list[Path] = []
    incomplete: list[Path] = []
    for directory in sorted(root.glob(_DISCOVERY_PATTERN), key=lambda path: path.name):
        if not directory.is_dir():
            continue
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            incomplete.append(directory.resolve())
            continue
        metadata = _read_json(metadata_path, label="discovered BEIR suite metadata")
        if metadata.get("status") == "completed":
            completed.append(directory.resolve())
        else:
            incomplete.append(directory.resolve())
    return tuple(completed), tuple(incomplete)


def aggregate_suites(suite_directories: Sequence[str | Path]) -> dict[str, Any]:
    if not suite_directories:
        raise ValueError("At least one completed BEIR suite is required")
    paths = [Path(path).resolve() for path in suite_directories]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate suite directory input")
    suites = sorted(
        (validate_suite_directory(path) for path in paths),
        key=lambda suite: (suite.suite_id, str(suite.directory)),
    )
    contract = suites[0].contract
    for suite in suites[1:]:
        if suite.contract != contract:
            raise ValueError(
                "Suite protocol, metrics_version, candidate depth, final_k, "
                "or condition set is inconsistent"
            )

    family_sources: dict[str, str] = {}
    combined: list[dict[str, Any]] = []
    for suite in suites:
        suite_families = {record["dataset"] for record in suite.family_results}
        for dataset in suite_families:
            previous = family_sources.get(dataset)
            if previous is not None:
                raise ValueError(
                    f"Duplicate BEIR family {dataset!r} in suites "
                    f"{previous!r} and {suite.suite_id!r}"
                )
            family_sources[dataset] = suite.suite_id
        for record in suite.family_results:
            combined.append(
                {
                    **record,
                    "source_suite_id": suite.suite_id,
                    "source_suite_directory": str(suite.directory),
                }
            )

    combined.sort(
        key=lambda record: (
            record["dataset"],
            _CONDITION_POSITION[record["condition"]],
        )
    )
    condition_metrics: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in combined:
        condition_metrics[record["condition"]].append(record["metrics"])
    family_macro: dict[str, dict[str, Any]] = {}
    for condition in EXPECTED_CONDITIONS:
        metrics = condition_metrics[condition]
        metric_names = {tuple(sorted(value)) for value in metrics}
        if len(metric_names) != 1:
            raise ValueError(
                f"Families expose inconsistent metric sets for condition {condition}"
            )
        family_macro[condition] = {
            "num_families": len(metrics),
            "metrics": _macro_metric_values(metrics),
        }

    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "status": "complete",
        "aggregation": "equal_weight_per_beir_family_v1",
        "contract": contract,
        "num_suites": len(suites),
        "num_families": len(family_sources),
        "source_suites": [
            {
                "suite_id": suite.suite_id,
                "directory": str(suite.directory),
                "families": sorted(
                    {
                        record["dataset"] for record in suite.family_results
                    }
                ),
                "metadata_artifact": suite.metadata_artifact,
                "summary_artifact": suite.summary_artifact,
            }
            for suite in suites
        ],
        "family_results": combined,
        "family_macro": family_macro,
    }


def _csv_bytes(aggregate: Mapping[str, Any]) -> bytes:
    fixed_fields = [
        "scope",
        "dataset",
        "condition",
        "num_units",
        "num_families",
        "source_suite_id",
        "evaluation_protocol",
        "metrics_version",
        "physical_candidate_k",
        "final_k",
    ]
    metric_names = sorted(
        {
            metric
            for record in aggregate["family_results"]
            for metric in record["metrics"]
        }
        | {
            metric
            for record in aggregate["family_macro"].values()
            for metric in record["metrics"]
        }
    )
    contract = aggregate["contract"]
    common = {
        "evaluation_protocol": contract["evaluation_protocol"],
        "metrics_version": contract["metrics_version"],
        "physical_candidate_k": contract["physical_candidate_k"],
        "final_k": contract["final_k"],
    }
    rows: list[dict[str, Any]] = []
    for record in aggregate["family_results"]:
        rows.append(
            {
                "scope": "family_macro",
                "dataset": record["dataset"],
                "condition": record["condition"],
                "num_units": record["num_units"],
                "num_families": "",
                "source_suite_id": record["source_suite_id"],
                **common,
                **record["metrics"],
            }
        )
    for condition in EXPECTED_CONDITIONS:
        record = aggregate["family_macro"][condition]
        rows.append(
            {
                "scope": "all_family_macro",
                "dataset": "",
                "condition": condition,
                "num_units": "",
                "num_families": record["num_families"],
                "source_suite_id": "",
                **common,
                **record["metrics"],
            }
        )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fixed_fields + metric_names)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def write_aggregate_outputs(
    aggregate: Mapping[str, Any],
    *,
    output_json: str | Path,
    output_csv: str | Path,
    source_suites: Sequence[str | Path],
) -> dict[str, Any]:
    json_path = Path(output_json).resolve()
    csv_path = Path(output_csv).resolve()
    if json_path == csv_path:
        raise ValueError("JSON and CSV outputs must use different paths")
    for output in (json_path, csv_path):
        if output.exists() and not output.is_file():
            raise ValueError(f"Aggregate output path is not a file: {output}")
    for source in source_suites:
        suite = Path(source).resolve()
        if _is_within(json_path, suite) or _is_within(csv_path, suite):
            raise ValueError("Aggregate outputs must not be written inside a suite directory")

    csv_payload = _csv_bytes(aggregate)
    completed = dict(aggregate)
    completed["csv_artifact"] = {
        "file": str(csv_path),
        "size_bytes": len(csv_payload),
        "sha256": hashlib.sha256(csv_payload).hexdigest(),
    }
    json_payload = (
        json.dumps(
            completed,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    csv_temporary: Path | None = None
    json_temporary: Path | None = None
    previous_payloads = {
        csv_path: csv_path.read_bytes() if csv_path.is_file() else None,
        json_path: json_path.read_bytes() if json_path.is_file() else None,
    }
    committed: list[Path] = []
    try:
        csv_temporary = _stage_bytes(csv_path, csv_payload)
        json_temporary = _stage_bytes(json_path, json_payload)
        # CSV is committed first; the JSON manifest is the final commit marker
        # and includes the exact CSV hash and size.
        os.replace(csv_temporary, csv_path)
        csv_temporary = None
        committed.append(csv_path)
        os.replace(json_temporary, json_path)
        json_temporary = None
        committed.append(json_path)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for output in reversed(committed):
            previous = previous_payloads[output]
            try:
                if previous is None:
                    output.unlink(missing_ok=True)
                else:
                    restore_temporary = _stage_bytes(output, previous)
                    try:
                        os.replace(restore_temporary, output)
                    finally:
                        restore_temporary.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"{output}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                "Aggregate output commit failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    finally:
        for temporary in (csv_temporary, json_temporary):
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return completed


def summarize_suites(
    suite_directories: Sequence[str | Path],
    *,
    output_json: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    aggregate = aggregate_suites(suite_directories)
    return write_aggregate_outputs(
        aggregate,
        output_json=output_json,
        output_csv=output_csv,
        source_suites=suite_directories,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and combine completed four-condition BEIR suites at the "
            "family-macro level without reading models or result rows."
        )
    )
    parser.add_argument(
        "suite_dirs",
        nargs="*",
        type=Path,
        help="Completed suite directories; omitted means auto-discover.",
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    configure_utf8_output()
    outputs_root = args.outputs_root.resolve()
    skipped: tuple[Path, ...] = ()
    if args.suite_dirs:
        suites = tuple(path.resolve() for path in args.suite_dirs)
    else:
        suites, skipped = discover_completed_suites(outputs_root)
    if not suites:
        raise ValueError("No completed four-condition BEIR suites were found")
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else outputs_root / "beir_four_condition_family_summary.json"
    )
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else outputs_root / "beir_four_condition_family_summary.csv"
    )
    completed = summarize_suites(
        suites,
        output_json=output_json,
        output_csv=output_csv,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_json": str(output_json),
                "output_csv": str(output_csv),
                "num_suites": completed["num_suites"],
                "num_families": completed["num_families"],
                "skipped_incomplete_suites": [str(path) for path in skipped],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return completed


if __name__ == "__main__":
    main()
