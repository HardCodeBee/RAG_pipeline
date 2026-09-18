"""Extract the predefined Phase 2.11 fields with the existing Phase 2.9 formulas."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.router_phase29_trait_features import (
    CorpusCache,
    build_corpus_cache,
    extract_feature_matrix,
    query_term_universe,
)


def _questions(path: Path, query_ids: np.ndarray) -> list[str]:
    requested = set(map(str, query_ids))
    found: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = str(row.get("_id", row.get("id")))
            if query_id not in requested:
                continue
            if query_id in found:
                raise ValueError(f"Duplicate query-source id: {query_id}")
            found[query_id] = str(row.get("text", row.get("question", "")))
    if set(found) != requested or any(not text.strip() for text in found.values()):
        raise ValueError("Query source does not cover the requested nonempty questions")
    return [found[str(query_id)] for query_id in query_ids]


def _cache_parity(reference: Path, target: Path) -> dict[str, Any]:
    """A different term universe must retain the same corpus and sample statistics."""
    with np.load(reference, allow_pickle=False) as old, np.load(target, allow_pickle=False) as new:
        for name in (
            "sample_vector_ids", "sample_doc_lengths", "sample_rows", "sample_seed",
            "document_count", "total_document_length", "average_document_length",
        ):
            if not np.array_equal(old[name], new[name]):
                raise ValueError(f"Expanded cache changes the frozen corpus/sample: {name}")
        _, old_positions, new_positions = np.intersect1d(
            old["terms"], new["terms"], assume_unique=True, return_indices=True,
        )
        for name in (
            "exact_df", "exact_idf", "exact_cf", "sample_df", "sample_cf",
            "sample_title_df", "sample_body_df", "sample_length_sum",
            "sample_impact_sum", "sample_impact_sq_sum", "sample_impact_max",
        ):
            if not np.array_equal(old[name][old_positions], new[name][new_positions]):
                raise ValueError(f"Expanded cache changes shared term statistics: {name}")
        for offset_name, value_name in (
            ("posting_offsets", "posting_doc_ids"), ("title_offsets", "title_doc_ids"),
        ):
            old_offsets, new_offsets = old[offset_name], new[offset_name]
            old_values, new_values = old[value_name], new[value_name]
            for old_position, new_position in zip(old_positions, new_positions):
                left = old_values[old_offsets[old_position]:old_offsets[old_position + 1]]
                right = new_values[new_offsets[new_position]:new_offsets[new_position + 1]]
                if not np.array_equal(left, right):
                    raise ValueError(f"Expanded cache changes shared term postings: {value_name}")
    return {"shared_terms": len(old_positions), "sample_exact": True,
            "shared_marginals_exact": True, "shared_postings_exact": True}


def extract_new_features(
    data: Any, config: Mapping[str, Any], repo_root: Path, run_dir: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Return 35 columns in config order, preserving existing definitions exactly.

    Only query text and frozen corpus/features are accessed. Discovery feature
    values are read for extraction parity; its labels and outcomes are not read.
    """
    started = time.perf_counter()
    names = list(config["features"])
    if len(names) != 35 or len(set(names)) != 35:
        raise ValueError("The predefined feature list must contain 35 distinct fields")
    paths = {name: repo_root / value for name, value in config["inputs"].items()}
    questions = _questions(paths["query_source"], data.query_ids)
    target_terms = query_term_universe(questions)
    sample = config["corpus_sample"]
    with np.load(paths["old_cache"], allow_pickle=False) as stored:
        old_terms = set(map(str, stored["terms"]))
        if int(stored["sample_rows"][0]) != int(sample["rows"]):
            raise ValueError("Configured sample size differs from the original cache")
        if int(stored["sample_seed"][0]) != int(sample["seed"]):
            raise ValueError("Configured sample seed differs from the original cache")
    missing = set(target_terms) - old_terms
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = paths["old_cache"]
    build_metadata: dict[str, Any] = {}
    if missing:
        cache_path = run_dir / "corpus_cache_9600.npz"
        if not cache_path.exists():
            print(f"Expanding corpus cache: {len(missing)} missing target terms", flush=True)
            build_metadata = build_corpus_cache(
                questions=questions, index_path=paths["bm25_index"],
                chunks_path=paths["corpus_chunks"], offsets_path=paths["corpus_offsets"],
                output_path=cache_path, sample_rows=int(sample["rows"]),
                sample_seed=int(sample["seed"]),
            )
    cache_parity = _cache_parity(paths["old_cache"], cache_path)
    schema = json.loads(paths["feature_schema"].read_text(encoding="utf-8"))
    cache = CorpusCache(cache_path)
    try:
        if not set(target_terms).issubset(cache.term_to_position):
            raise ValueError("Corpus cache still omits required query/morphology terms")
        full_matrix, specs = extract_feature_matrix(
            questions=questions, legacy_lexical=data.lexical, legacy_dense=data.corpus,
            legacy_lexical_names=schema["lexical"], dense_names=schema["dense_corpus"],
            cache=cache,
        )
    finally:
        cache.close()
    spec_by_name = {spec.name: spec for spec in specs}
    all_names = [spec.name for spec in specs]
    matrix = full_matrix[:, [all_names.index(name) for name in names]]
    if matrix.shape != (len(data.query_ids), 35) or not np.isfinite(matrix).all():
        raise ValueError("Extracted 35D matrix has invalid shape or nonfinite values")

    positions = {str(query_id): i for i, query_id in enumerate(data.query_ids)}
    with np.load(paths["discovery_features"], allow_pickle=False) as stored:
        discovery_ids = list(map(str, stored["query_ids"]))
        discovery_names = list(map(str, stored["feature_names"]))
        selected = [i for i, query_id in enumerate(discovery_ids) if query_id in positions]
        reference = stored["matrix"][np.ix_(selected, [discovery_names.index(name) for name in names])]
        reference_groups = stored["group_ids"][selected]
    if len(selected) != 2760:
        raise ValueError(f"Unexpected original/discovery feature overlap: {len(selected)}")
    target_positions = [positions[discovery_ids[i]] for i in selected]
    if not np.array_equal(data.group_ids[target_positions], reference_groups):
        raise ValueError("Original/discovery feature group ids differ")
    observed = matrix[target_positions]
    if not np.array_equal(observed, reference):
        maximum = float(np.max(np.abs(observed - reference)))
        raise ValueError(f"Existing discovery feature values changed: max_abs={maximum}")

    np.savez_compressed(
        run_dir / "new35_features.npz", query_ids=data.query_ids, group_ids=data.group_ids,
        matrix=matrix, feature_names=np.asarray(names, dtype=np.str_),
    )
    metadata = {
        "rows": len(questions), "columns": len(names), "target_terms": len(target_terms),
        "missing_terms_in_old_cache": len(missing), "expanded_cache": bool(missing),
        "cache_file": (Path(config["output_dir"]) / cache_path.name).as_posix()
        if missing else str(config["inputs"]["old_cache"]),
        "cache_build": build_metadata,
        "cache_parity": cache_parity, "discovery_overlap_rows": len(selected),
        "discovery_35d_parity_exact": True, "discovery_max_absolute_difference": 0.0,
        "discovery_group_ids_parity_exact": True,
        "all_values_finite": True, "elapsed_seconds": time.perf_counter() - started,
        "feature_catalog": [asdict(spec_by_name[name]) for name in names],
        "feature_descriptive_statistics": {
            name: {
                "min": float(matrix[:, column].min()),
                "max": float(matrix[:, column].max()),
                "std": float(matrix[:, column].std()),
                "unique_count": int(len(np.unique(matrix[:, column]))),
                "nonzero_ratio": float(np.mean(matrix[:, column] != 0)),
            }
            for column, name in enumerate(names)
        },
        "natural_outcome_rows_read": 0, "official_final_holdout_rows_read": 0,
    }
    return matrix, names, metadata
