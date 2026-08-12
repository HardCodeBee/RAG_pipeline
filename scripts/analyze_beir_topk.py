"""Validate and analyze BEIR Top-k replays across k=5/10/20/50.

The input suites are immutable formal Top-5 runs or identity-bound offline
replays. This script does not retrieve documents or run model inference. It
validates every query prefix, recomputes cross-cutoff summaries, and writes a
small provenance-bound analysis directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_support import configure_utf8_output
from src.provenance import json_sha256, sha256_file


PROTOCOL = "beir_retrieval_only_v2_ignore_identical_pre_candidate"
METRICS_VERSION = "beir_retrieval_v2_trec_linear_ndcg_binary_recall"
METRICS = ("ndcg", "map", "recall", "precision", "mrr", "hit")
BOOTSTRAP_METRICS = ("ndcg", "recall", "hit")
EXPECTED_CUTOFFS = (5, 10, 20, 50)
TRANSITIONS = ((5, 10), (10, 20), (20, 50))
SEMANTICS = ("bm25", "dense", "bm25_bge", "dense_bge")
FAMILY_ORDER = (
    "arguana",
    "fiqa",
    "hotpotqa",
    "nfcorpus",
    "nq",
    "webis-touche2020",
    "cqadupstack",
)
FAMILY_DISPLAY = {
    "arguana": "ArguAna",
    "fiqa": "FiQA-2018",
    "hotpotqa": "HotpotQA",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "webis-touche2020": "Touché-2020",
    "cqadupstack": "CQADupStack",
}
SEMANTIC_DISPLAY = {
    "bm25": "BM25",
    "dense": "Dense (BGE-small)",
    "bm25_bge": "BM25 Top-50 → BGE",
    "dense_bge": "Dense Top-50 → BGE",
}
RANK_BINS = ("1-5", "6-10", "11-20", "21-50", "not_in_top50")
_CONDITION_RE = re.compile(r"^(bm25|dense)_top(\d+)(?:_bge_top(\d+))?$")


@dataclass(frozen=True, slots=True)
class RunInfo:
    cutoff: int
    dataset: str
    unit: str
    method: str
    rerank: bool
    condition: str
    suite_id: str
    directory: Path
    metadata: dict[str, Any]

    @property
    def semantic(self) -> str:
        return f"{self.method}_bge" if self.rerank else self.method


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _parse_condition(condition: str, expected_k: int) -> tuple[str, bool]:
    match = _CONDITION_RE.fullmatch(condition)
    if match is None:
        raise ValueError(f"Unsupported condition name: {condition}")
    method, first_number, rerank_final = match.groups()
    if rerank_final is None:
        if int(first_number) != expected_k:
            raise ValueError(f"Condition cutoff mismatch: {condition}")
        return method, False
    if int(first_number) != 50 or int(rerank_final) != expected_k:
        raise ValueError(f"Rerank condition cutoff mismatch: {condition}")
    return method, True


def _metric(summary: Mapping[str, Any], metric: str, cutoff: int) -> float:
    value = summary[f"{metric}_at_{cutoff}"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {metric}@{cutoff}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {metric}@{cutoff}")
    return number


def _rank_bin(rank: int | None) -> str:
    if rank is None:
        return "not_in_top50"
    if rank <= 5:
        return "1-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    return "21-50"


def _result_ids(row: Mapping[str, Any]) -> list[str]:
    retrieval = row.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise ValueError("Result row has no retrieval object")
    results = retrieval.get("results")
    if not isinstance(results, list):
        raise ValueError("Result row has no retrieval results")
    ids: list[str] = []
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("Invalid retrieval result")
        doc_id = result.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Retrieval result has no doc_id")
        ids.append(doc_id)
    if len(ids) != len(set(ids)):
        raise ValueError("A result row contains duplicate doc IDs")
    return ids


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**32)


def _bootstrap_means(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
    chunk_size: int = 50,
) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] <= 0:
        raise ValueError("Bootstrap values must be a non-empty matrix")
    rng = np.random.default_rng(seed)
    output = np.empty((resamples, values.shape[1]), dtype=np.float64)
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(
            0,
            values.shape[0],
            size=(stop - start, values.shape[0]),
        )
        output[start:stop] = values[indices].mean(axis=1)
    return output


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Top-k config must be a mapping")
    cutoffs = tuple(int(item) for item in value.get("cutoffs", ()))
    if cutoffs != EXPECTED_CUTOFFS:
        raise ValueError(f"Expected cutoffs {EXPECTED_CUTOFFS}, got {cutoffs}")
    suites = value.get("suites")
    if not isinstance(suites, Mapping):
        raise ValueError("Top-k config has no suite mapping")
    for cutoff in EXPECTED_CUTOFFS:
        entries = suites.get(cutoff, suites.get(str(cutoff)))
        if not isinstance(entries, list) or len(entries) != 5:
            raise ValueError(f"Cutoff {cutoff} must name five suites")
    return value


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/beir/topk_selected7_5_10_20_50_v1.yaml",
    )
    args = parser.parse_args(argv)
    configure_utf8_output()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    config = _load_config(config_path)
    output_dir = Path(str(config["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Top-k analysis: {output_dir}")
    output_dir.mkdir(parents=True)

    bootstrap = config.get("bootstrap", {})
    resamples = int(bootstrap.get("resamples", 1000))
    base_seed = int(bootstrap.get("seed", 20260808))
    if resamples <= 0:
        raise ValueError("bootstrap.resamples must be positive")

    suite_provenance: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    cqa_rows: list[dict[str, Any]] = []
    runs: dict[tuple[str, str, str, bool, int], RunInfo] = {}
    source_base_identity: dict[tuple[str, str], str] = {}
    unit_question_counts: dict[tuple[str, str], int] = {}

    suites_config = config["suites"]
    for cutoff in EXPECTED_CUTOFFS:
        suite_entries = suites_config.get(cutoff, suites_config.get(str(cutoff)))
        for suite_entry in suite_entries:
            suite_dir = Path(str(suite_entry))
            if not suite_dir.is_absolute():
                suite_dir = PROJECT_ROOT / suite_dir
            suite_dir = suite_dir.resolve()
            metadata_path = suite_dir / "metadata.json"
            summary_path = suite_dir / "suite_summary.json"
            metadata = _read_json(metadata_path)
            summary = _read_json(summary_path)
            if metadata.get("status") != "completed":
                raise ValueError(f"Suite is not completed: {suite_dir}")
            identity = metadata.get("suite_identity")
            if not isinstance(identity, Mapping):
                raise ValueError(f"Suite has no identity: {suite_dir}")
            if metadata.get("suite_identity_sha256") != json_sha256(identity):
                raise ValueError(f"Suite identity hash mismatch: {suite_dir}")
            if identity.get("final_k") != cutoff or summary.get("final_k") != cutoff:
                raise ValueError(f"Suite cutoff mismatch: {suite_dir}")
            if identity.get("evaluation_protocol") != PROTOCOL:
                raise ValueError(f"Protocol mismatch: {suite_dir}")
            if identity.get("metrics_version") != METRICS_VERSION:
                raise ValueError(f"Metrics version mismatch: {suite_dir}")
            if cutoff > 5:
                if (
                    identity.get("kind") != "beir_offline_topk_reanalysis"
                    or identity.get("retrieval_or_reranking_repeated") is not False
                    or identity.get("source_prefix_replay_validation")
                    != "exact_per_question"
                ):
                    raise ValueError(f"Offline replay identity mismatch: {suite_dir}")
                base_identity = identity.get("source_suite", {}).get(
                    "suite_identity_sha256"
                )
                evidence_class = f"local_offline_replay_top{cutoff}"
            else:
                base_identity = metadata.get("suite_identity_sha256")
                evidence_class = "local_executed_top5"
            if not isinstance(base_identity, str) or not base_identity:
                raise ValueError(f"Missing base suite identity: {suite_dir}")

            suite_provenance.append(
                {
                    "cutoff": cutoff,
                    "evidence_class": evidence_class,
                    "suite_id": metadata["run_id"],
                    "status": metadata["status"],
                    "suite_identity_sha256": metadata["suite_identity_sha256"],
                    "base_suite_identity_sha256": base_identity,
                    "metadata_sha256": sha256_file(metadata_path),
                    "summary_sha256": sha256_file(summary_path),
                    "retrieval_or_reranking_repeated": identity.get(
                        "retrieval_or_reranking_repeated", True
                    ),
                    "source_prefix_replay_validation": identity.get(
                        "source_prefix_replay_validation", "not_applicable"
                    ),
                    "directory": str(suite_dir),
                }
            )

            for result in summary.get("family_results", ()):
                family = result["dataset"]
                condition = result["condition"]
                method, rerank = _parse_condition(condition, cutoff)
                metrics = result["metrics"]
                family_rows.append(
                    {
                        "scope": "family_macro",
                        "evidence_class": evidence_class,
                        "cutoff": cutoff,
                        "family": family,
                        "family_display": FAMILY_DISPLAY[family],
                        "num_units": int(result["num_units"]),
                        "semantic": f"{method}_bge" if rerank else method,
                        "condition": condition,
                        **{metric: _metric(metrics, metric, cutoff) for metric in METRICS},
                    }
                )

            for result in summary.get("unit_results", ()):
                if result["dataset"] != "cqadupstack":
                    continue
                condition = result["condition"]
                method, rerank = _parse_condition(condition, cutoff)
                item = result["summary"]
                cqa_rows.append(
                    {
                        "evidence_class": evidence_class,
                        "cutoff": cutoff,
                        "forum": str(result["unit"]).replace("\\", "/").split("/")[-1],
                        "num_questions": int(item["num_questions"]),
                        "semantic": f"{method}_bge" if rerank else method,
                        "condition": condition,
                        **{metric: _metric(item, metric, cutoff) for metric in METRICS},
                    }
                )

            for run_metadata_path in sorted(suite_dir.glob("units/*/runs/*/metadata.json")):
                run_metadata = _read_json(run_metadata_path)
                if (
                    run_metadata.get("status") != "completed"
                    or int(run_metadata.get("num_failed_rows", 0)) != 0
                    or int(run_metadata.get("effective_top_k", 0)) != cutoff
                ):
                    raise ValueError(f"Invalid run metadata: {run_metadata_path}")
                condition = run_metadata["condition"]
                method, rerank = _parse_condition(condition, cutoff)
                dataset = run_metadata["dataset"]
                unit = run_metadata["unit"]
                key = (dataset, unit, method, rerank, cutoff)
                if key in runs:
                    raise ValueError(f"Duplicate unit condition: {key}")
                run_info = RunInfo(
                    cutoff=cutoff,
                    dataset=dataset,
                    unit=unit,
                    method=method,
                    rerank=rerank,
                    condition=condition,
                    suite_id=metadata["run_id"],
                    directory=run_metadata_path.parent,
                    metadata=run_metadata,
                )
                runs[key] = run_info
                unit_key = (dataset, unit)
                previous_identity = source_base_identity.setdefault(
                    unit_key, base_identity
                )
                if previous_identity != base_identity:
                    raise ValueError(f"Base suite identity changed for {unit_key}")
                count = int(run_metadata["num_question_records"])
                previous_count = unit_question_counts.setdefault(unit_key, count)
                if previous_count != count:
                    raise ValueError(f"Question count changed for {unit_key}")

    if len(family_rows) != 7 * 4 * 4:
        raise ValueError(f"Expected 112 family-condition rows, got {len(family_rows)}")
    if len(cqa_rows) != 12 * 4 * 4:
        raise ValueError(f"Expected 192 CQA unit-condition rows, got {len(cqa_rows)}")
    if len(runs) != 18 * 4 * 4:
        raise ValueError(f"Expected 288 run inputs, got {len(runs)}")
    if sum(unit_question_counts.values()) != 26428:
        raise ValueError("Unexpected unique question count")

    metric_values: defaultdict[
        tuple[str, str, str, int, str], list[float]
    ] = defaultdict(list)
    bin_counts: defaultdict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {rank_bin: 0 for rank_bin in RANK_BINS}
    )
    available_candidate_counts: defaultdict[
        tuple[str, str, str], list[int]
    ] = defaultdict(list)
    prefix_path_checks = 0
    prefix_pair_checks = 0
    monotonic_checks = 0
    top50_set_checks = 0
    top50_metric_invariant_checks = 0
    total_query_method_pairs = 0

    for dataset, unit in sorted(unit_question_counts):
        expected_questions = unit_question_counts[(dataset, unit)]
        for method in ("bm25", "dense"):
            infos = {
                (rerank, cutoff): runs[(dataset, unit, method, rerank, cutoff)]
                for rerank in (False, True)
                for cutoff in EXPECTED_CUTOFFS
            }
            candidate_refs = {
                info.metadata.get("shared_first_stage_cache_ref_sha256")
                for info in infos.values()
            }
            question_hashes = {
                info.metadata.get("questions_sha256") for info in infos.values()
            }
            build_ids = {info.metadata.get("build_id") for info in infos.values()}
            rerank_refs = {
                info.metadata.get("shared_bge_score_cache_ref_sha256")
                for (rerank, _), info in infos.items()
                if rerank
            }
            if (
                len(candidate_refs) != 1
                or len(question_hashes) != 1
                or len(build_ids) != 1
                or len(rerank_refs) != 1
            ):
                raise ValueError(f"Cross-cutoff identity mismatch: {(dataset, unit, method)}")

            ordered_keys = [
                (rerank, cutoff)
                for rerank in (False, True)
                for cutoff in EXPECTED_CUTOFFS
            ]
            with ExitStack() as stack:
                streams = [
                    stack.enter_context(
                        (infos[key].directory / "results.jsonl").open(
                            "r", encoding="utf-8"
                        )
                    )
                    for key in ordered_keys
                ]
                seen_questions = 0
                for line_group in zip_longest(*streams):
                    if any(line is None for line in line_group):
                        raise ValueError(
                            f"Cross-cutoff row count mismatch: {(dataset, unit, method)}"
                        )
                    rows = {
                        key: json.loads(line)
                        for key, line in zip(ordered_keys, line_group, strict=True)
                    }
                    question_ids = {row.get("question_id") for row in rows.values()}
                    if len(question_ids) != 1 or None in question_ids:
                        raise ValueError(
                            f"Cross-cutoff question order mismatch: {(dataset, unit, method)}"
                        )
                    if any(row.get("status") != "success" for row in rows.values()):
                        raise ValueError("A formal result row is not successful")
                    qrels = rows[(False, 50)].get("qrels")
                    if not isinstance(qrels, Mapping):
                        raise ValueError("Result row has no qrels")
                    if any(row.get("qrels") != qrels for row in rows.values()):
                        raise ValueError("Qrels changed across cutoffs/conditions")
                    relevant = {
                        str(doc_id)
                        for doc_id, grade in qrels.items()
                        if float(grade) > 0.0
                    }
                    ids: dict[tuple[bool, int], list[str]] = {}
                    for key, row in rows.items():
                        rerank, cutoff = key
                        ranked = _result_ids(row)
                        ids[key] = ranked
                        semantic = f"{method}_bge" if rerank else method
                        row_metrics = row.get("metrics")
                        if not isinstance(row_metrics, Mapping):
                            raise ValueError("Result row has no metrics")
                        for metric in METRICS:
                            metric_values[(dataset, unit, semantic, cutoff, metric)].append(
                                _metric(row_metrics, metric, cutoff)
                            )

                    for rerank in (False, True):
                        full = ids[(rerank, 50)]
                        if not 0 <= len(full) <= 50:
                            raise ValueError("Top-50 result count must be in [0, 50]")
                        for cutoff in (5, 10, 20):
                            prefix_pair_checks += 1
                            expected_prefix = full[: min(cutoff, len(full))]
                            if ids[(rerank, cutoff)] != expected_prefix:
                                raise ValueError(
                                    f"Non-prefix ranking: {(dataset, unit, method, rerank, cutoff)}"
                                )
                        prefix_path_checks += 1
                        semantic = f"{method}_bge" if rerank else method
                        available_candidate_counts[(dataset, unit, semantic)].append(
                            len(full)
                        )
                        first_rank = next(
                            (
                                position
                                for position, doc_id in enumerate(full, start=1)
                                if doc_id in relevant
                            ),
                            None,
                        )
                        bin_counts[(dataset, unit, semantic)][_rank_bin(first_rank)] += 1
                        for metric in ("recall", "hit"):
                            previous = -math.inf
                            for cutoff in EXPECTED_CUTOFFS:
                                value = metric_values[
                                    (dataset, unit, semantic, cutoff, metric)
                                ][-1]
                                if value + 1e-12 < previous:
                                    raise ValueError(
                                        f"Non-monotonic {metric}: {(dataset, unit, semantic)}"
                                    )
                                previous = value
                            monotonic_checks += 3

                    if set(ids[(False, 50)]) != set(ids[(True, 50)]):
                        raise ValueError(
                            f"Top-50 candidate set changed after reranking: {(dataset, unit, method)}"
                        )
                    top50_set_checks += 1
                    for metric in ("recall", "precision", "hit"):
                        baseline = _metric(rows[(False, 50)]["metrics"], metric, 50)
                        reranked = _metric(rows[(True, 50)]["metrics"], metric, 50)
                        if abs(baseline - reranked) > 1e-12:
                            raise ValueError(
                                f"Top-50 set metric changed: {(dataset, unit, method, metric)}"
                            )
                        top50_metric_invariant_checks += 1
                    seen_questions += 1
                    total_query_method_pairs += 1
                if seen_questions != expected_questions:
                    raise ValueError(
                        f"Unexpected question rows for {(dataset, unit, method)}"
                    )

    family_lookup: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in family_rows:
        key = (int(row["cutoff"]), str(row["family"]), str(row["semantic"]))
        if key in family_lookup:
            raise ValueError(f"Duplicate family result: {key}")
        family_lookup[key] = row
    for cutoff in EXPECTED_CUTOFFS:
        for family in FAMILY_ORDER:
            for semantic in SEMANTICS:
                if (cutoff, family, semantic) not in family_lookup:
                    raise ValueError(f"Missing family result: {(cutoff, family, semantic)}")

    macro_rows: list[dict[str, Any]] = []
    for cutoff in EXPECTED_CUTOFFS:
        for semantic in SEMANTICS:
            members = [family_lookup[(cutoff, family, semantic)] for family in FAMILY_ORDER]
            macro_rows.append(
                {
                    "scope": "seven_family_equal_macro",
                    "evidence_class": "derived_equal_family_macro",
                    "cutoff": cutoff,
                    "semantic": semantic,
                    "condition_display": SEMANTIC_DISPLAY[semantic],
                    **{
                        metric: mean(float(member[metric]) for member in members)
                        for metric in METRICS
                    },
                }
            )
    macro_lookup = {
        (int(row["cutoff"]), str(row["semantic"])): row for row in macro_rows
    }

    marginal_rows: list[dict[str, Any]] = []
    for scope, names in (
        ("family_macro", FAMILY_ORDER),
        ("seven_family_equal_macro", ("seven_family_equal_macro",)),
    ):
        for name in names:
            for semantic in SEMANTICS:
                for from_k, to_k in TRANSITIONS:
                    if scope == "family_macro":
                        before = family_lookup[(from_k, name, semantic)]
                        after = family_lookup[(to_k, name, semantic)]
                    else:
                        before = macro_lookup[(from_k, semantic)]
                        after = macro_lookup[(to_k, semantic)]
                    marginal_rows.append(
                        {
                            "scope": scope,
                            "family": name,
                            "family_display": FAMILY_DISPLAY.get(
                                name, "七-family 等权 macro"
                            ),
                            "semantic": semantic,
                            "from_k": from_k,
                            "to_k": to_k,
                            **{
                                f"delta_{metric}": float(after[metric])
                                - float(before[metric])
                                for metric in METRICS
                            },
                        }
                    )

    reranker_rows: list[dict[str, Any]] = []
    for cutoff in EXPECTED_CUTOFFS:
        for family in (*FAMILY_ORDER, "seven_family_equal_macro"):
            for method in ("bm25", "dense"):
                if family == "seven_family_equal_macro":
                    baseline = macro_lookup[(cutoff, method)]
                    reranked = macro_lookup[(cutoff, f"{method}_bge")]
                else:
                    baseline = family_lookup[(cutoff, family, method)]
                    reranked = family_lookup[(cutoff, family, f"{method}_bge")]
                reranker_rows.append(
                    {
                        "scope": "seven_family_equal_macro"
                        if family == "seven_family_equal_macro"
                        else "family_macro",
                        "cutoff": cutoff,
                        "family": family,
                        "family_display": FAMILY_DISPLAY.get(
                            family, "七-family 等权 macro"
                        ),
                        "first_stage": method,
                        **{
                            f"delta_{metric}": float(reranked[metric])
                            - float(baseline[metric])
                            for metric in METRICS
                        },
                    }
                )

    cqa_sensitivity_rows: list[dict[str, Any]] = []
    for cutoff in EXPECTED_CUTOFFS:
        for semantic in SEMANTICS:
            members = [
                row
                for row in cqa_rows
                if row["cutoff"] == cutoff and row["semantic"] == semantic
            ]
            if len(members) != 12:
                raise ValueError("CQADupStack must have twelve forums")
            total_questions = sum(int(row["num_questions"]) for row in members)
            cqa_sensitivity_rows.append(
                {
                    "cutoff": cutoff,
                    "semantic": semantic,
                    "num_forums": 12,
                    "num_queries": total_questions,
                    **{
                        f"forum_equal_{metric}": mean(
                            float(row[metric]) for row in members
                        )
                        for metric in METRICS
                    },
                    **{
                        f"query_weighted_{metric}": sum(
                            float(row[metric]) * int(row["num_questions"])
                            for row in members
                        )
                        / total_questions
                        for metric in METRICS
                    },
                }
            )

    availability_rows: list[dict[str, Any]] = []
    for (dataset, unit, semantic), values in sorted(
        available_candidate_counts.items()
    ):
        availability_rows.append(
            {
                "scope": "unit",
                "family": dataset,
                "unit": unit,
                "semantic": semantic,
                "num_queries": len(values),
                "minimum_available": min(values),
                "maximum_available": max(values),
                "mean_available": mean(values),
                "underfilled_lt_50": sum(value < 50 for value in values),
                "empty_zero": sum(value == 0 for value in values),
            }
        )

    units_by_family: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for unit_key in sorted(unit_question_counts):
        units_by_family[unit_key[0]].append(unit_key)

    bin_rows: list[dict[str, Any]] = []
    family_bin_values: dict[tuple[str, str, str], float] = {}
    for family in FAMILY_ORDER:
        units = units_by_family[family]
        for semantic in SEMANTICS:
            for dataset, unit in units:
                count = unit_question_counts[(dataset, unit)]
                for rank_bin in RANK_BINS:
                    value = bin_counts[(dataset, unit, semantic)][rank_bin]
                    bin_rows.append(
                        {
                            "scope": "unit",
                            "family": family,
                            "unit": unit,
                            "semantic": semantic,
                            "rank_bin": rank_bin,
                            "count": value,
                            "num_queries": count,
                            "proportion": value / count,
                        }
                    )
            for rank_bin in RANK_BINS:
                proportion = mean(
                    bin_counts[(dataset, unit, semantic)][rank_bin]
                    / unit_question_counts[(dataset, unit)]
                    for dataset, unit in units
                )
                family_bin_values[(family, semantic, rank_bin)] = proportion
                bin_rows.append(
                    {
                        "scope": "family_macro",
                        "family": family,
                        "unit": "",
                        "semantic": semantic,
                        "rank_bin": rank_bin,
                        "count": "",
                        "num_queries": sum(unit_question_counts[item] for item in units),
                        "proportion": proportion,
                    }
                )
    for semantic in SEMANTICS:
        for rank_bin in RANK_BINS:
            bin_rows.append(
                {
                    "scope": "seven_family_equal_macro",
                    "family": "seven_family_equal_macro",
                    "unit": "",
                    "semantic": semantic,
                    "rank_bin": rank_bin,
                    "count": "",
                    "num_queries": 26428,
                    "proportion": mean(
                        family_bin_values[(family, semantic, rank_bin)]
                        for family in FAMILY_ORDER
                    ),
                }
            )

    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_distributions: dict[tuple[str, str], np.ndarray] = {}
    columns = [
        (from_k, to_k, metric)
        for from_k, to_k in TRANSITIONS
        for metric in BOOTSTRAP_METRICS
    ]
    marginal_lookup = {
        (
            str(row["scope"]),
            str(row["family"]),
            str(row["semantic"]),
            int(row["from_k"]),
            int(row["to_k"]),
        ): row
        for row in marginal_rows
    }
    for semantic in SEMANTICS:
        for family in FAMILY_ORDER:
            unit_distributions: list[np.ndarray] = []
            for dataset, unit in units_by_family[family]:
                matrix_columns: list[np.ndarray] = []
                for from_k, to_k, metric in columns:
                    before = np.asarray(
                        metric_values[(dataset, unit, semantic, from_k, metric)],
                        dtype=np.float64,
                    )
                    after = np.asarray(
                        metric_values[(dataset, unit, semantic, to_k, metric)],
                        dtype=np.float64,
                    )
                    if before.shape != after.shape:
                        raise ValueError("Paired metric shape mismatch")
                    matrix_columns.append(after - before)
                matrix = np.column_stack(matrix_columns)
                unit_distributions.append(
                    _bootstrap_means(
                        matrix,
                        resamples=resamples,
                        seed=_stable_seed(base_seed, semantic, dataset, unit),
                    )
                )
            family_distribution = np.mean(unit_distributions, axis=0)
            bootstrap_distributions[(family, semantic)] = family_distribution
            for column_index, (from_k, to_k, metric) in enumerate(columns):
                point = float(
                    marginal_lookup[
                        ("family_macro", family, semantic, from_k, to_k)
                    ][f"delta_{metric}"]
                )
                actual = mean(
                    float(
                        np.mean(
                            np.asarray(
                                metric_values[(dataset, unit, semantic, to_k, metric)]
                            )
                            - np.asarray(
                                metric_values[(dataset, unit, semantic, from_k, metric)]
                            )
                        )
                    )
                    for dataset, unit in units_by_family[family]
                )
                if abs(point - actual) > 1e-10:
                    raise ValueError(
                        f"Query/family aggregate mismatch: {(family, semantic, from_k, to_k, metric)}"
                    )
                samples = family_distribution[:, column_index]
                bootstrap_rows.append(
                    {
                        "scope": "family_macro",
                        "family": family,
                        "family_display": FAMILY_DISPLAY[family],
                        "semantic": semantic,
                        "from_k": from_k,
                        "to_k": to_k,
                        "metric": metric,
                        "point_delta": point,
                        "ci_low_2_5": float(np.percentile(samples, 2.5)),
                        "ci_high_97_5": float(np.percentile(samples, 97.5)),
                        "bootstrap_mean": float(np.mean(samples)),
                        "probability_positive": float(np.mean(samples > 0.0)),
                        "resamples": resamples,
                        "seed": base_seed,
                    }
                )
        macro_distribution = np.mean(
            [bootstrap_distributions[(family, semantic)] for family in FAMILY_ORDER],
            axis=0,
        )
        for column_index, (from_k, to_k, metric) in enumerate(columns):
            point = float(
                marginal_lookup[
                    (
                        "seven_family_equal_macro",
                        "seven_family_equal_macro",
                        semantic,
                        from_k,
                        to_k,
                    )
                ][f"delta_{metric}"]
            )
            samples = macro_distribution[:, column_index]
            bootstrap_rows.append(
                {
                    "scope": "seven_family_equal_macro",
                    "family": "seven_family_equal_macro",
                    "family_display": "七-family 等权 macro",
                    "semantic": semantic,
                    "from_k": from_k,
                    "to_k": to_k,
                    "metric": metric,
                    "point_delta": point,
                    "ci_low_2_5": float(np.percentile(samples, 2.5)),
                    "ci_high_97_5": float(np.percentile(samples, 97.5)),
                    "bootstrap_mean": float(np.mean(samples)),
                    "probability_positive": float(np.mean(samples > 0.0)),
                    "resamples": resamples,
                    "seed": base_seed,
                }
            )

    validation = {
        "status": "passed",
        "formal_source_suites": len(suite_provenance),
        "formal_source_runs": len(runs),
        "unique_test_queries": sum(unit_question_counts.values()),
        "query_condition_rows_per_cutoff": sum(unit_question_counts.values()) * 4,
        "failed_rows": 0,
        "prefix_path_checks": prefix_path_checks,
        "prefix_pair_checks": prefix_pair_checks,
        "recall_hit_monotonic_transition_checks": monotonic_checks,
        "top50_candidate_set_checks": top50_set_checks,
        "top50_set_metric_invariant_checks": top50_metric_invariant_checks,
        "query_method_pairs": total_query_method_pairs,
        "top50_underfilled_query_paths": sum(
            value < 50
            for values in available_candidate_counts.values()
            for value in values
        ),
        "empty_candidate_query_paths": sum(
            value == 0
            for values in available_candidate_counts.values()
            for value in values
        ),
        "minimum_available_candidates": min(
            value
            for values in available_candidate_counts.values()
            for value in values
        ),
        "top50_invariant": (
            "Within each first stage, reranking preserves the complete Top-50 "
            "candidate set and therefore Recall/Precision/Hit@50."
        ),
    }

    family_fields = [
        "scope", "evidence_class", "cutoff", "family", "family_display",
        "num_units", "semantic", "condition", *METRICS,
    ]
    macro_fields = [
        "scope", "evidence_class", "cutoff", "semantic", "condition_display",
        *METRICS,
    ]
    marginal_fields = [
        "scope", "family", "family_display", "semantic", "from_k", "to_k",
        *[f"delta_{metric}" for metric in METRICS],
    ]
    reranker_fields = [
        "scope", "cutoff", "family", "family_display", "first_stage",
        *[f"delta_{metric}" for metric in METRICS],
    ]
    cqa_fields = [
        "evidence_class", "cutoff", "forum", "num_questions", "semantic",
        "condition", *METRICS,
    ]
    cqa_sensitivity_fields = [
        "cutoff", "semantic", "num_forums", "num_queries",
        *[
            field
            for metric in METRICS
            for field in (f"forum_equal_{metric}", f"query_weighted_{metric}")
        ],
    ]
    bootstrap_fields = [
        "scope", "family", "family_display", "semantic", "from_k", "to_k",
        "metric", "point_delta", "ci_low_2_5", "ci_high_97_5",
        "bootstrap_mean", "probability_positive", "resamples", "seed",
    ]
    bin_fields = [
        "scope", "family", "unit", "semantic", "rank_bin", "count",
        "num_queries", "proportion",
    ]
    provenance_fields = list(suite_provenance[0].keys())
    availability_fields = [
        "scope", "family", "unit", "semantic", "num_queries",
        "minimum_available", "maximum_available", "mean_available",
        "underfilled_lt_50", "empty_zero",
    ]

    _write_csv(output_dir / "topk_family_results.csv", family_rows, family_fields)
    _write_csv(output_dir / "topk_macro.csv", macro_rows, macro_fields)
    _write_csv(output_dir / "topk_marginal_gains.csv", marginal_rows, marginal_fields)
    _write_csv(output_dir / "reranker_delta_by_k.csv", reranker_rows, reranker_fields)
    _write_csv(output_dir / "cqadupstack_topk_results.csv", cqa_rows, cqa_fields)
    _write_csv(
        output_dir / "cqadupstack_aggregation_sensitivity.csv",
        cqa_sensitivity_rows,
        cqa_sensitivity_fields,
    )
    _write_csv(output_dir / "first_relevant_rank_bins.csv", bin_rows, bin_fields)
    _write_csv(output_dir / "paired_bootstrap_ci.csv", bootstrap_rows, bootstrap_fields)
    _write_csv(output_dir / "suite_provenance.csv", suite_provenance, provenance_fields)
    _write_csv(
        output_dir / "candidate_availability.csv",
        availability_rows,
        availability_fields,
    )
    _write_json(output_dir / "validation.json", validation)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    colors = {
        "bm25": "#4C78A8",
        "dense": "#F58518",
        "bm25_bge": "#54A24B",
        "dense_bge": "#E45756",
    }
    for metric in ("ndcg", "recall", "precision", "hit"):
        fig, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
        for semantic in SEMANTICS:
            values = [macro_lookup[(cutoff, semantic)][metric] for cutoff in EXPECTED_CUTOFFS]
            axis.plot(
                EXPECTED_CUTOFFS,
                values,
                marker="o",
                linewidth=2,
                label=SEMANTIC_DISPLAY[semantic],
                color=colors[semantic],
            )
        axis.set_xticks(EXPECTED_CUTOFFS)
        axis.set_xlabel("Final Top-k")
        axis.set_ylabel(f"Seven-family equal macro {metric}")
        axis.set_title(f"BEIR Selected-7: {metric} across Top-k")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.savefig(figures_dir / f"topk_macro_{metric}.png", dpi=180)
        plt.close(fig)

    macro_table = []
    for cutoff in EXPECTED_CUTOFFS:
        for semantic in SEMANTICS:
            row = macro_lookup[(cutoff, semantic)]
            macro_table.append(
                [
                    str(cutoff), SEMANTIC_DISPLAY[semantic],
                    f"{row['ndcg']:.6f}", f"{row['map']:.6f}",
                    f"{row['recall']:.6f}", f"{row['precision']:.6f}",
                    f"{row['mrr']:.6f}", f"{row['hit']:.6f}",
                ]
            )
    marginal_macro_table = []
    for row in marginal_rows:
        if row["scope"] != "seven_family_equal_macro":
            continue
        marginal_macro_table.append(
            [
                f"{row['from_k']}→{row['to_k']}",
                SEMANTIC_DISPLAY[str(row["semantic"])],
                f"{row['delta_ndcg']:+.6f}",
                f"{row['delta_recall']:+.6f}",
                f"{row['delta_precision']:+.6f}",
                f"{row['delta_hit']:+.6f}",
            ]
        )
    reranker_ndcg_table = []
    for family in FAMILY_ORDER:
        values: list[str] = []
        for cutoff in EXPECTED_CUTOFFS:
            for method in ("bm25", "dense"):
                row = next(
                    item
                    for item in reranker_rows
                    if item["scope"] == "family_macro"
                    and item["family"] == family
                    and item["cutoff"] == cutoff
                    and item["first_stage"] == method
                )
                values.append(f"{row['delta_ndcg']:+.6f}")
        reranker_ndcg_table.append([FAMILY_DISPLAY[family], *values])
    rank_macro_table = []
    for semantic in SEMANTICS:
        values = []
        for rank_bin in RANK_BINS:
            row = next(
                item
                for item in bin_rows
                if item["scope"] == "seven_family_equal_macro"
                and item["semantic"] == semantic
                and item["rank_bin"] == rank_bin
            )
            values.append(f"{float(row['proportion']):.6f}")
        rank_macro_table.append([SEMANTIC_DISPLAY[semantic], *values])
    bootstrap_macro_table = []
    for row in bootstrap_rows:
        if row["scope"] != "seven_family_equal_macro":
            continue
        bootstrap_macro_table.append(
            [
                f"{row['from_k']}→{row['to_k']}",
                SEMANTIC_DISPLAY[str(row["semantic"])],
                str(row["metric"]),
                f"{row['point_delta']:+.6f}",
                f"[{row['ci_low_2_5']:+.6f}, {row['ci_high_97_5']:+.6f}]",
                f"{row['probability_positive']:.3f}",
            ]
        )

    report = f"""# BEIR Selected-7 Top-k 影响实验

## 范围与口径

- k = 5, 10, 20, 50；四条路径保持一致。
- Top-5 为正式执行；Top-10/20/50 为 identity-bound 离线重放，没有重新执行 retrieval 或 BGE inference。
- 18 units、26,428 个唯一 test queries；每个 cutoff 105,712 个成功 query-condition rows，0 failures。
- CQADupStack 先按 12 forums 等权，再进入七-family 等权 macro。

## 七-family 等权 macro

{_markdown_table(["k", "Path", "nDCG", "MAP", "Recall", "P", "MRR", "Hit"], macro_table)}

## 相邻 k 的边际变化

不同 k 的 nDCG@k/MAP@k 是不同 cutoff 指标；下表只作同一路径的 cutoff 曲线描述。

{_markdown_table(["Transition", "Path", "ΔnDCG", "ΔRecall", "ΔP", "ΔHit"], marginal_macro_table)}

## Reranker 跨域 ΔnDCG

每个 k 下均与匹配的 first-stage 路径比较。

{_markdown_table(["Family", "BM25 Δ@5", "Dense Δ@5", "BM25 Δ@10", "Dense Δ@10", "BM25 Δ@20", "Dense Δ@20", "BM25 Δ@50", "Dense Δ@50"], reranker_ndcg_table)}

## 首个相关文档所在排名区间

下表为七-family 等权比例；`not_in_top50` 是当前缓存的未覆盖部分。

{_markdown_table(["Path", "1-5", "6-10", "11-20", "21-50", "not in Top-50"], rank_macro_table)}

## 相邻 k 边际变化的配对 bootstrap

按 query 配对；CQADupStack 在每个 forum 内 bootstrap 后等权，最终七-family 等权。固定 seed={base_seed}，{resamples} 次重采样。

{_markdown_table(["Transition", "Path", "Metric", "Point Δ", "95% CI", "P(Δ>0)"], bootstrap_macro_table)}

## 验收

- {prefix_path_checks:,} 个 query-path 的完整 5/10/20/50 prefix 已核验；{prefix_pair_checks:,} 个小 cutoff→Top-50 前缀比较全部一致。
- {monotonic_checks:,} 个 Recall/Hit 相邻 cutoff 单调性检查全部通过。
- {top50_set_checks:,} 个 query-method Top-50 候选集合检查全部通过。
- {top50_metric_invariant_checks:,} 个 Recall/Precision/Hit@50 重排不变量检查全部通过。
- 发现 {{underfilled}} 个 query-path 的有效候选少于 50，其中 {{empty}} 个为空，最少为 {{minimum}}；各 unit/path 明细见 `candidate_availability.csv`。所有 cutoff 均按 `min(k, available)` 取前缀，没有补造候选。
- Top-50 下 reranker 保留全部候选，所以 Recall/Precision/Hit 与 first-stage 完全相同；nDCG/MAP/MRR 仍可因排序而变化。

详细数值、CQADupStack 12-forum 结果、聚合敏感性、suite 身份和验证计数均保存在同目录 CSV/JSON 中。离线重放进程时间不是 fresh retrieval/reranking latency。
"""
    report = report.replace(
        "{underfilled}", f"{validation['top50_underfilled_query_paths']:,}"
    ).replace(
        "{empty}", f"{validation['empty_candidate_query_paths']:,}"
    ).replace("{minimum}", str(validation["minimum_available_candidates"]))
    _write_text(output_dir / "report.md", report)

    summary = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "evaluation_protocol": PROTOCOL,
        "metrics_version": METRICS_VERSION,
        "cutoffs": list(EXPECTED_CUTOFFS),
        "family_aggregation": "CQADupStack forum-equal, then seven-family equal",
        "macro_results": macro_rows,
        "marginal_gains": [
            row
            for row in marginal_rows
            if row["scope"] == "seven_family_equal_macro"
        ],
        "reranker_deltas": [
            row
            for row in reranker_rows
            if row["scope"] == "seven_family_equal_macro"
        ],
        "validation": validation,
        "bootstrap": {"resamples": resamples, "seed": base_seed},
    }
    _write_json(output_dir / "summary.json", summary)

    metadata = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "command": "analyze_beir_topk",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config": _sha_descriptor(config_path),
        "source_suites": suite_provenance,
        "validation": validation,
        "retrieval_or_reranking_repeated": False,
        "latency_scope": "not_a_fresh_topk_latency_measurement",
    }
    _write_json(output_dir / "metadata.json", metadata)

    manifest_files = []
    for path in sorted(
        item for item in output_dir.rglob("*") if item.is_file() and item.name != "manifest.json"
    ):
        manifest_files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "hash_algorithm": "sha256",
            "num_files_excluding_manifest": len(manifest_files),
            "files": manifest_files,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved BEIR Top-k analysis: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
