"""Pure FP64 objective fitting and calibration; no file, model, or network I/O."""
import math

import numpy as np


def _numeric(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu":
            raise ValueError(f"{name} must contain real numeric values")
        result = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name):
    result = _numeric(value, name)
    if result.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return float(result)


def _scores(value, allow_empty=False):
    result = _numeric(value, "scores")
    if result.ndim != 1 or (not allow_empty and result.size == 0):
        raise ValueError("scores must be a nonempty vector")
    return result


def _gap(value, n):
    result = _numeric(value, "gap")
    if result.shape != (n,) or np.any(np.abs(result) > 1):
        raise ValueError("gap must have shape (N,) and lie in [-1, 1]")
    return result


def fit_ridge(x, gap, regularization):
    """Minimize .5 * mean(residual**2) + .5 * lambda * ||coef||**2.

    Centering eliminates the unpenalized intercept only; returned coefficients
    act on the original, untransformed features. All rows, including zero gaps,
    participate. Acceptance requires the original-coordinate gradient <= 1e-10.
    """
    x = _numeric(x, "x")
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("x must be a nonempty two-dimensional array")
    gap = _gap(gap, x.shape[0])
    regularization = _scalar(regularization, "regularization")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    n, dimensions = x.shape
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean_x = np.mean(x, axis=0)
            mean_gap = float(np.mean(gap))
            centered_x = x - mean_x
            centered_gap = gap - mean_gap
            gram = centered_x.T @ centered_x / n
            gram.flat[::dimensions + 1] += regularization
            rhs = centered_x.T @ centered_gap / n
            factor = np.linalg.cholesky(gram)
            coef = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
            intercept = float(mean_gap - mean_x @ coef)
            residual = x @ coef + intercept - gap
            data_loss = float(0.5 * (residual @ residual) / n)
            penalty = float(0.5 * regularization * (coef @ coef))
            loss = data_loss + penalty
            gradient = x.T @ residual / n + regularization * coef
            grad_inf = float(max(np.max(np.abs(gradient)), abs(np.mean(residual))))
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Ridge solution could not be computed in finite FP64") from error
    if not np.isfinite(coef).all() or not all(map(math.isfinite,
            (intercept, data_loss, penalty, loss, grad_inf))):
        raise ValueError("Ridge solution must be finite")
    return {"coef": coef, "intercept": intercept, "data_loss": data_loss,
            "penalty": penalty, "loss": loss, "grad_inf": grad_inf,
            "accepted": bool(grad_inf <= 1e-10)}


def select_threshold(scores, gap, tie_atol=1e-12):
    """Select a strict score threshold by calibration gain over all N rows.

    Candidates are every distinct observed score plus two explicit constant
    policies. No midpoint is used. A compensated suffix sum evaluates candidates
    in O(N log N) time without truncating small gaps. Ties are resolved only
    after finding the global maximum, favoring fewer BM25 decisions and then a
    larger threshold. Explicit all_dense wins its duplicate max-score policy.
    """
    scores = _scores(scores)
    gap = _gap(gap, scores.size)
    tie_atol = _scalar(tie_atol, "tie_atol")
    if tie_atol < 0:
        raise ValueError("tie_atol must be nonnegative")
    n = scores.size
    order = np.argsort(scores, kind="stable")[::-1]
    candidates = [{"mode": "all_dense", "threshold": None,
                   "selected_count": 0, "cal_gain": 0.0}]
    total, compensation, count = 0.0, 0.0, 0
    while count < n:
        threshold = float(scores[order[count]])
        candidates.append({"mode": "threshold", "threshold": threshold,
                           "selected_count": count,
                           "cal_gain": math.fsum((total, compensation)) / n})
        while count < n and scores[order[count]] == threshold:
            value = float(gap[order[count]])
            updated = total + value
            if abs(total) >= abs(value):
                compensation += (total - updated) + value
            else:
                compensation += (value - updated) + total
            total = updated
            count += 1
    candidates.append({"mode": "all_bm25", "threshold": None,
                       "selected_count": int(n),
                       "cal_gain": math.fsum(map(float, gap)) / n})
    best_gain = max(item["cal_gain"] for item in candidates)
    eligible = [item for item in candidates
                if best_gain - item["cal_gain"] <= tie_atol]
    selected = min(eligible, key=lambda item: (
        item["selected_count"], 0 if item["mode"] == "all_dense" else 1,
        -item["threshold"] if item["threshold"] is not None else 0.0))
    return {**selected, "best_gain": best_gain, "candidate_count": len(candidates)}


def apply_threshold(scores, record):
    """Apply the stored policy to finite scores, including an empty vector."""
    scores = _scores(scores, allow_empty=True)
    if not isinstance(record, dict) or "mode" not in record or "threshold" not in record:
        raise ValueError("record must contain mode and threshold")
    mode, threshold = record["mode"], record["threshold"]
    if mode == "threshold":
        return scores > _scalar(threshold, "threshold")
    if mode in ("all_bm25", "all_dense") and threshold is None:
        return np.full(scores.shape, mode == "all_bm25", dtype=bool)
    raise ValueError("Unknown mode or non-null threshold for a constant policy")
