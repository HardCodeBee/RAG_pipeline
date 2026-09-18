"""Label-free analysis of the historical sampled-corpus feature encoding.

Run from the repository root with:
    python -B -m scripts.analyze_pre_retrieval_feature_limits

Reads the Phase 2.11 feature-name configuration, but no queries, labels, index,
feature matrices or document vectors. The two posting lists below are synthetic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from scripts.router_phase29_trait_features import _pair_values


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "analysis/hotpotqa_router/phases/phase31/config.yaml"
OUTPUT = ROOT / (
    "outputs/router/hotpotqa_bd_router_v1/runs/"
    "pre_retrieval_feature_reassessment_v1/measurement_limits.json"
)


def probability_no_sample_hit(population: int, sample: int, document_frequency: int) -> float:
    """Hypergeometric zero-hit probability, evaluated without huge binomials."""
    return math.exp(math.fsum(
        math.log1p(-sample / (population - index))
        for index in range(document_frequency)
    ))


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    population = 5_233_329  # Frozen build_65e9037e4c1eed7b, not a new corpus scan.
    sample = int(config["corpus_sample"]["rows"])
    selected_names = set(config["features"])
    pair_name_map = {
        "mean_npmi": "c_pair_npmi_mean",
        "zero_ratio": "c_pair_zero_cooccurrence_ratio",
        "estimable_ratio": "c_pair_estimable_ratio",
    }
    missing = _pair_values([
        np.array([], dtype=np.int64), np.array([0, 1, 2, 3]),
    ], sample_rows=8)
    observed_independent = _pair_values([
        np.array([0, 1, 2, 3]), np.array([0, 1, 4, 5]),
    ], sample_rows=8)
    retained_keys = [key for key, name in pair_name_map.items() if name in selected_names]
    assert retained_keys == ["mean_npmi", "zero_ratio"]
    assert all(missing[key] == observed_independent[key] == 0.0 for key in retained_keys)
    assert missing["estimable_ratio"] == 0.0
    assert observed_independent["estimable_ratio"] == 1.0

    result = {
        "scope": "analytic_sampling_probability_and_synthetic_existing_extractor_example",
        "population_documents": population,
        "sample_documents": sample,
        "sample_fraction": sample / population,
        "sampling_rule": "uniform_without_replacement_fixed_seed_2026083001",
        "probability_no_sample_hit": [
            {"full_corpus_df": df, "probability": probability_no_sample_hit(population, sample, df)}
            for df in (1, 2, 5, 10, 20, 50, 100)
        ],
        "pair_encoding": {
            "selected_pair_features": [pair_name_map[key] for key in retained_keys],
            "estimable_ratio_selected": pair_name_map["estimable_ratio"] in selected_names,
            "missing_term_sample_support": {key: missing[key] for key in pair_name_map},
            "observed_independence_after_smoothing": {
                key: observed_independent[key] for key in pair_name_map
            },
            "selected_pair_coordinates_identical": True,
            "full_35d_collision_claimed": False,
        },
        "interpretation_limits": [
            "Probabilities concern sampling before the frozen sample is drawn, not uncertainty that a fixed index changes on each call.",
            "Exact full-corpus DF and CF remain available; this concerns sample-based cooccurrence and scope measurements.",
            "No estimate of affected real queries or answer-utility impact is made.",
            "Other selected features can partly identify missing sample support.",
        ],
        "query_reads": 0,
        "label_reads": 0,
        "model_fits": 0,
        "retriever_calls": 0,
        "answer_calls": 0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
