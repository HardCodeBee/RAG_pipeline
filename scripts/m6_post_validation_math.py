"""Pure descriptive arithmetic for the fixed M6 post-validation plan.

This module performs no file access, inference, fitting, or hypothesis testing.
The caller must enforce complete, verified evaluation inputs before calling it.
Utilities have shape (N, 2, 3): BM25/Dense, each with repeat IDs 0, 1, 2.
Group labels must be nonempty strings; repeated labels remain the same group.
"""

from __future__ import annotations

import numpy as np


TIE_TOLERANCE = 1e-12
BIN_CUTS = (-0.4296875, 0.0, 0.1611328125)
BIN_NAMES = ("D_far", "D_near", "B_near", "B_far")
SIGN_CATEGORIES = (
    "all_zero",
    "nonnegative_some_positive",
    "nonpositive_some_negative",
    "crosses_zero",
)


def _repeat_summary(values: np.ndarray) -> dict:
    return {
        "values": [float(value) for value in values],
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _sign_counts(categories: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    denominator = int(mask.sum())
    return {
        name: {
            "count": int(np.count_nonzero(category & mask)),
            "fraction": float(np.count_nonzero(category & mask) / denominator),
        }
        for name, category in categories.items()
    }


def compute(scores, groups, utilities) -> dict:
    """Describe the fixed policy without changing scores, bins, or actions.

    ``scores`` and ``groups`` must be one-dimensional and aligned. Scores and
    utilities must contain real integer or floating-point numeric values,
    excluding booleans, complex values, and numeric strings. All four
    fixed score bins must contain a query. Scores must be finite, and every
    utility must be finite and in [0, 1]. Invalid inputs raise ValueError.

    Tolerance affects sign counts only: all utility means and contributions
    retain the original floating-point differences, including near-zero gaps.
    Bin means use each bin's query count; policy contributions use all N.
    All returned values are ordinary JSON-serializable Python types.
    """
    try:
        scores = np.asarray(scores)
        groups = np.asarray(groups, dtype=object)
        utilities = np.asarray(utilities)
    except (TypeError, ValueError) as error:
        raise ValueError("Numeric scores/utilities and one-dimensional group labels required") from error
    if scores.dtype.kind not in "fiu" or utilities.dtype.kind not in "fiu":
        raise ValueError("Scores and utilities require real numeric values, not booleans, strings, or complex values")
    scores = scores.astype(np.float64, copy=False)
    utilities = utilities.astype(np.float64, copy=False)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("A nonempty one-dimensional score array is required")
    n = len(scores)
    if groups.shape != (n,) or any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("Nonempty group labels must align with scores")
    groups = np.asarray(groups, dtype=str)
    if utilities.shape != (n, 2, 3):
        raise ValueError("Utilities must have shape (N, 2, 3), ordered BM25 then Dense")
    if not np.isfinite(scores).all() or not np.isfinite(utilities).all():
        raise ValueError("Scores and utilities must be finite")
    if np.any(utilities < 0) or np.any(utilities > 1):
        raise ValueError("Utilities must be in [0, 1]")

    lower, zero, upper = BIN_CUTS
    masks = (
        scores <= lower,
        (scores > lower) & (scores <= zero),
        (scores > zero) & (scores <= upper),
        scores > upper,
    )
    if any(not mask.any() for mask in masks):
        raise ValueError("Every fixed score bin must contain at least one query")

    switch = scores > 0
    repeated_gap = utilities[:, 0, :] - utilities[:, 1, :]
    gap = repeated_gap.mean(axis=1)
    versus_dense = np.where(switch, gap, 0.0)
    versus_bm25 = np.where(switch, 0.0, -gap)
    bins = []
    for name, mask in zip(BIN_NAMES, masks):
        selected_gap = gap[mask]
        selected_scores = scores[mask]
        score_scale = float(np.max(np.abs(selected_scores)))
        mean_score = 0.0 if score_scale == 0 else float(score_scale * np.mean(selected_scores / score_scale))
        bins.append({
            "name": name,
            "query_count": int(mask.sum()),
            "group_count": int(len(np.unique(groups[mask]))),
            "mean_score": mean_score,
            "mean_gap": float(selected_gap.mean()),
            "gap_counts": {
                "positive": int(np.count_nonzero(selected_gap > TIE_TOLERANCE)),
                "negative": int(np.count_nonzero(selected_gap < -TIE_TOLERANCE)),
                "zero": int(np.count_nonzero(np.abs(selected_gap) <= TIE_TOLERANCE)),
            },
            "contribution": {
                "M6_minus_Dense": float(versus_dense[mask].sum() / n),
                "M6_minus_BM25": float(versus_bm25[mask].sum() / n),
            },
        })

    fraction = float(switch.mean())
    mean_gap = float(gap.mean())
    dense_gain = float(versus_dense.mean())
    random_expectation = fraction * mean_gap
    decomposition = {
        "bm25_fraction": fraction,
        "mean_gap": mean_gap,
        "beneficial_mass": float(np.where(switch, np.maximum(gap, 0.0), 0.0).mean()),
        "harmful_mass": float(np.where(switch, np.maximum(-gap, 0.0), 0.0).mean()),
        "missed_bm25_positive_mass": float(np.where(switch, 0.0, np.maximum(gap, 0.0)).mean()),
        "random_same_count_expected_gain": random_expectation,
        "selection_alignment": dense_gain - random_expectation,
        "M6_minus_Dense": dense_gain,
        "M6_minus_BM25": float(versus_bm25.mean()),
    }

    repeat_dense = np.where(switch[:, None], repeated_gap, 0.0).mean(axis=0)
    repeat_bm25 = np.where(switch[:, None], 0.0, -repeated_gap).mean(axis=0)
    repeats = {
        "ids": [0, 1, 2],
        "M6_minus_Dense": _repeat_summary(repeat_dense),
        "M6_minus_BM25": _repeat_summary(repeat_bm25),
        "bin_mean_gaps": {
            name: _repeat_summary(repeated_gap[mask].mean(axis=0))
            for name, mask in zip(BIN_NAMES, masks)
        },
    }

    positive = (repeated_gap > TIE_TOLERANCE).any(axis=1)
    negative = (repeated_gap < -TIE_TOLERANCE).any(axis=1)
    categories = dict(zip(SIGN_CATEGORIES, (
        ~positive & ~negative,
        positive & ~negative,
        ~positive & negative,
        positive & negative,
    )))
    gap_range = repeated_gap.max(axis=1) - repeated_gap.min(axis=1)
    stability = {
        "global": _sign_counts(categories, np.ones(n, dtype=bool)),
        "by_bin": {
            name: _sign_counts(categories, mask)
            for name, mask in zip(BIN_NAMES, masks)
        },
        "gap_range": {
            "median": float(np.quantile(gap_range, 0.5, method="linear")),
            "p90": float(np.quantile(gap_range, 0.9, method="linear")),
        },
    }
    return {
        "queries": n,
        "groups": int(len(np.unique(groups))),
        "tie_tolerance": TIE_TOLERANCE,
        "bins": bins,
        "within_action_rank_contrasts": {
            "D_near_minus_D_far": bins[1]["mean_gap"] - bins[0]["mean_gap"],
            "B_far_minus_B_near": bins[3]["mean_gap"] - bins[2]["mean_gap"],
        },
        "global_decomposition": decomposition,
        "repeats": repeats,
        "sign_stability": stability,
    }
