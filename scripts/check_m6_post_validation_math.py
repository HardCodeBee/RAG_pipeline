"""Independent scalar reconstruction of the frozen descriptive M6 diagnostics.

This module reads no files. Sums use explicit scalar loops and math.fsum;
the caller owns the complete-evaluation gate before supplying real utilities.
"""
import math
from numbers import Integral, Real

import numpy as np

BIN_NAMES = ("D_far", "D_near", "B_near", "B_far")
CATEGORIES = ("all_zero", "nonnegative_some_positive", "nonpositive_some_negative", "crosses_zero")
TOLERANCE = 1e-12


def _inputs(scores, groups, utilities):
    try:
        score_array, utility_array = map(np.asarray, (scores, utilities))
        group_array = np.asarray(groups, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError("Aligned numeric arrays and group IDs required") from error
    if score_array.ndim != 1 or len(score_array) == 0:
        raise ValueError("Scores must be a nonempty vector")
    n = len(score_array)
    if group_array.shape != (n,) or utility_array.shape != (n, 2, 3):
        raise ValueError("Group or utility shape differs")
    if score_array.dtype.kind not in "fiu" or utility_array.dtype.kind not in "fiu":
        raise ValueError("Scores and utilities must be real numeric values")
    score_values = [float(value) for value in score_array]
    if not all(math.isfinite(value) for value in score_values):
        raise ValueError("Nonfinite score")
    group_values = group_array.tolist()
    if not all(isinstance(value, str) and bool(value) for value in group_values):
        raise ValueError("Groups must be nonempty string IDs")
    values = [[[float(utility_array[i, action, repeat]) for repeat in range(3)]
               for action in range(2)] for i in range(n)]
    if not all(math.isfinite(value) and 0 <= value <= 1
               for item in values for action in item for value in action):
        raise ValueError("Utilities must be finite F1 values between zero and one")
    return score_values, group_values, values


def _bin(score):
    if score <= -0.4296875:
        return 0
    if score <= 0:
        return 1
    if score <= 0.1611328125:
        return 2
    return 3


def _summary(values):
    return {"values": values, "mean": math.fsum(values) / len(values), "min": min(values), "max": max(values)}


def _category(values):
    positive = any(value > TOLERANCE for value in values)
    negative = any(value < -TOLERANCE for value in values)
    if positive and negative:
        return "crosses_zero"
    if positive:
        return "nonnegative_some_positive"
    if negative:
        return "nonpositive_some_negative"
    return "all_zero"


def _fractions(labels, indices):
    return {name: {"count": sum(labels[i] == name for i in indices),
                   "fraction": sum(labels[i] == name for i in indices) / len(indices)}
            for name in CATEGORIES}


def _quantile(values, probability):
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low, high = math.floor(location), math.ceil(location)
    fraction = location - low
    return math.fsum(((1 - fraction) * ordered[low], fraction * ordered[high]))


def _reconstruct(scores, groups, utilities):
    scores, groups, utilities = _inputs(scores, groups, utilities)
    n = len(scores)
    members = [[i for i, score in enumerate(scores) if _bin(score) == b] for b in range(4)]
    if not all(members):
        raise ValueError("All four prespecified score bins must be nonempty")
    gaps = [[utilities[i][0][r] - utilities[i][1][r] for r in range(3)] for i in range(n)]
    mean_gaps = [math.fsum(values) / 3 for values in gaps]
    selected = [score > 0 for score in scores]
    p = sum(selected) / n
    mu = math.fsum(mean_gaps) / n
    dense_gain = math.fsum(mean_gaps[i] for i in range(n) if selected[i]) / n
    bm25_gain = -math.fsum(mean_gaps[i] for i in range(n) if not selected[i]) / n
    bins = []
    for name, indices in zip(BIN_NAMES, members):
        positive = sum(mean_gaps[i] > TOLERANCE for i in indices)
        negative = sum(mean_gaps[i] < -TOLERANCE for i in indices)
        bins.append({"name": name, "query_count": len(indices), "group_count": len({groups[i] for i in indices}),
            "mean_score": math.fsum(scores[i] / len(indices) for i in indices),
            "mean_gap": math.fsum(mean_gaps[i] for i in indices) / len(indices),
            "gap_counts": {"positive": positive, "negative": negative, "zero": len(indices) - positive - negative},
            "contribution": {"M6_minus_Dense": math.fsum(mean_gaps[i] for i in indices if selected[i]) / n,
                             "M6_minus_BM25": -math.fsum(mean_gaps[i] for i in indices if not selected[i]) / n}})
    repeat_dense = [math.fsum(gaps[i][r] for i in range(n) if selected[i]) / n for r in range(3)]
    repeat_bm25 = [-math.fsum(gaps[i][r] for i in range(n) if not selected[i]) / n for r in range(3)]
    labels = [_category(row) for row in gaps]
    ranges = [max(row) - min(row) for row in gaps]
    return {"queries": n, "groups": len(set(groups)), "tie_tolerance": TOLERANCE,
        "bins": bins,
        "within_action_rank_contrasts": {"D_near_minus_D_far": bins[1]["mean_gap"] - bins[0]["mean_gap"],
                                          "B_far_minus_B_near": bins[3]["mean_gap"] - bins[2]["mean_gap"]},
        "global_decomposition": {"bm25_fraction": p, "mean_gap": mu,
            "beneficial_mass": math.fsum(max(mean_gaps[i], 0) for i in range(n) if selected[i]) / n,
            "harmful_mass": math.fsum(max(-mean_gaps[i], 0) for i in range(n) if selected[i]) / n,
            "missed_bm25_positive_mass": math.fsum(max(mean_gaps[i], 0) for i in range(n) if not selected[i]) / n,
            "random_same_count_expected_gain": p * mu, "selection_alignment": dense_gain - p * mu,
            "M6_minus_Dense": dense_gain, "M6_minus_BM25": bm25_gain},
        "repeats": {"ids": [0, 1, 2], "M6_minus_Dense": _summary(repeat_dense),
            "M6_minus_BM25": _summary(repeat_bm25),
            "bin_mean_gaps": {name: _summary([math.fsum(gaps[i][r] for i in indices) / len(indices) for r in range(3)])
                              for name, indices in zip(BIN_NAMES, members)}},
        "sign_stability": {"global": _fractions(labels, list(range(n))),
            "by_bin": {name: _fractions(labels, indices) for name, indices in zip(BIN_NAMES, members)},
            "gap_range": {"median": _quantile(ranges, 0.5), "p90": _quantile(ranges, 0.9)}}}


def verify(scores, groups, utilities, result):
    """Reconstruct every diagnostic scalar; raise ValueError on any mismatch."""
    expected = _reconstruct(scores, groups, utilities)
    count, numeric_count, maximum_error = 0, 0, 0.0

    def compare(actual, wanted, path):
        nonlocal count, numeric_count, maximum_error
        if isinstance(wanted, dict):
            if not isinstance(actual, dict) or set(actual) != set(wanted):
                raise ValueError(f"Diagnostic keys differ at {path}")
            for name in wanted:
                compare(actual[name], wanted[name], path + "." + name)
        elif isinstance(wanted, list):
            if not isinstance(actual, list) or len(actual) != len(wanted):
                raise ValueError(f"Diagnostic list differs at {path}")
            for i, item in enumerate(wanted):
                compare(actual[i], item, f"{path}[{i}]")
        else:
            count += 1
            if isinstance(wanted, str):
                if actual != wanted:
                    raise ValueError(f"Diagnostic label differs at {path}")
            elif isinstance(wanted, int):
                if not isinstance(actual, Integral) or isinstance(actual, bool) or actual != wanted:
                    raise ValueError(f"Diagnostic count differs at {path}")
                numeric_count += 1
            else:
                if not isinstance(actual, Real) or isinstance(actual, bool) or not math.isfinite(actual):
                    raise ValueError(f"Nonfinite or invalid diagnostic at {path}")
                error = abs(float(actual) - wanted)
                maximum_error = max(maximum_error, error)
                numeric_count += 1
                if not math.isclose(actual, wanted, rel_tol=2e-12, abs_tol=2e-15):
                    raise ValueError(f"Diagnostic number differs at {path}")

    compare(result, expected, "result")
    decomposition = result["global_decomposition"]
    p, mu, alignment = (decomposition[name] for name in ("bm25_fraction", "mean_gap", "selection_alignment"))
    checks = {
        "dense_identity": (decomposition["M6_minus_Dense"], math.fsum((p * mu, alignment))),
        "bm25_identity": (decomposition["M6_minus_BM25"], math.fsum((alignment, -(1 - p) * mu))),
        "benefit_minus_harm": (decomposition["M6_minus_Dense"], decomposition["beneficial_mass"] - decomposition["harmful_mass"]),
    }
    for name in ("M6_minus_Dense", "M6_minus_BM25"):
        checks[name + "_bin_sum"] = (decomposition[name], math.fsum(item["contribution"][name] for item in result["bins"]))
        checks[name + "_repeat_mean"] = (decomposition[name], math.fsum(result["repeats"][name]["values"]) / 3)
    for name, (left, right) in checks.items():
        compare(left, right, "identity." + name)
    if sum(item["query_count"] for item in result["bins"]) != expected["queries"]:
        raise ValueError("Four bins do not exhaust query population")
    return {"status": "passed_independent_fsum_diagnostic_reconstruction", "queries": expected["queries"],
        "groups": expected["groups"], "checked_scalar_count": count, "checked_numeric_scalar_count": numeric_count,
        "maximum_absolute_error": maximum_error, "four_score_bins_exhaustive": True,
        "identity_checks": list(checks), "three_repeats_checked": True,
        "tolerance_used_only_for_sign_counts": True, "contribution_denominator": expected["queries"],
        "new_hypothesis_tests": 0, "new_network_calls": 0}
