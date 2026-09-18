"""Fixed Fourier maps and the original convex BCE solver; no research-data I/O.

Only the two standalone sibling numerical solver modules are imported. The
caller supplies training rows, labels, frozen geometry, and any torch backend.
"""
import math
from pathlib import Path
import sys

import numpy as np


RESEARCH = Path(__file__).resolve().parents[2] / "work" / "router_research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
import weighted_linear_probe_refined as refined

original = refined.original
BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)
GEOMETRY_SEED = 2026091505


def _x(value, *, allow_empty=False):
    x = np.asarray(value)
    if x.ndim != 2 or x.shape[1] != 384 or x.dtype != np.float32:
        raise ValueError("X must have shape (N, 384) and dtype float32")
    if (not allow_empty and len(x) == 0) or not np.isfinite(x).all():
        raise ValueError("X must be nonempty and finite")
    return x


def _omega(value):
    omega = np.asarray(value)
    if omega.shape != (3, 384, 64) or omega.dtype != np.float64 or not np.isfinite(omega).all():
        raise ValueError("omega must be finite float64 with shape (3, 384, 64)")
    return omega


def _scalar(value, name):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(raw)


def _scale(value):
    scale = _scalar(value, "scale")
    if scale <= 1e-12:
        raise ValueError("scale must be greater than 1e-12")
    return scale


def make_frequencies(seed):
    """Create the data-independent, FP64 frequency bank from one fixed seed."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return np.random.default_rng(seed).standard_normal((3, 384, 64))


def geometry(X_train):
    """Use only supplied training rows to select disjoint pairs and their scale."""
    x = _x(X_train)
    pairs = min(512, len(x) // 2)
    if pairs == 0:
        raise ValueError("At least two training rows are needed for geometry")
    selected = np.random.default_rng(GEOMETRY_SEED).permutation(len(x))[:2 * pairs]
    pair_indices = selected.reshape(pairs, 2)
    differences = x[pair_indices[:, 0]].astype(np.float64) - x[pair_indices[:, 1]].astype(np.float64)
    squared_distances = np.sum(differences * differences, axis=1)
    scale = _scale(math.sqrt(float(np.median(squared_distances))))
    return {"pair_indices": pair_indices, "scale": scale}


def rff(X, omega, s):
    """Return cos64/sin64 per scale, divided by sqrt(192), as 384D FP32."""
    x, omega, scale = _x(X, allow_empty=True), _omega(omega), _scale(s)
    x64 = x.astype(np.float64)
    columns = []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for index, multiplier in enumerate(BANDWIDTH_MULTIPLIERS):
            phase = (x64 @ omega[index]) / (multiplier * scale)
            columns.extend((np.cos(phase), np.sin(phase)))
        features = (np.concatenate(columns, axis=1) / math.sqrt(192)).astype(np.float32)
    if not np.isfinite(features).all():
        raise FloatingPointError("Nonfinite Fourier features")
    return features


def augmented(X, omega, s):
    """Concatenate unchanged original features and RFF, without normalization."""
    x = _x(X, allow_empty=True)
    return np.concatenate((x, rff(x, omega, s)), axis=1)


def fit_design(design768, gap, regularization):
    """Fit a precomputed FP32 design and preserve the sibling solver record.

    The caller must inspect accepted. This wrapper adds no retries, feature
    centering, clipping, or changes to the solver's numerical budget.
    """
    features = np.asarray(design768)
    if (features.ndim != 2 or features.shape[1] != 768 or len(features) == 0
            or features.dtype != np.float32 or not np.isfinite(features).all()):
        raise ValueError("design must be finite nonempty float32 with shape (N, 768)")
    raw_gap = np.asarray(gap)
    if raw_gap.dtype.kind not in "fiu":
        raise ValueError("gap must contain real numeric values")
    gap = raw_gap.astype(np.float64)
    if gap.shape != (len(features),) or not np.isfinite(gap).all() or np.any(np.abs(gap) > 1):
        raise ValueError("gap must be finite with shape (N,) and lie in [-1, 1]")
    regularization = _scalar(regularization, "regularization")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    labels = (gap > 0).astype(np.float64)
    weights = np.where(np.abs(gap) > 1e-12, np.abs(gap), 0.0)
    return refined.fit(features, labels, weights, regularization=regularization)


def fit_augmented(X_train, gap, omega, geometry, regularization):
    """Build the fixed design, then use the same fit_design solver interface."""
    x = _x(X_train)
    if not isinstance(geometry, dict) or "scale" not in geometry:
        raise ValueError("geometry must contain the frozen training scale")
    return fit_design(augmented(x, omega, geometry["scale"]), gap, regularization)


def native_scores(X, omega, s, coef, bias, torch):
    """Two BF16 linear branches, then FP32 addition; batch size remains eight.

    The original 384D branch owns the sole intercept. With theta exactly zero,
    its FP32 logits are preserved by addition of the zero nonlinear branch.
    """
    x, omega, scale = _x(X, allow_empty=True), _omega(omega), _scale(s)
    raw_coef = np.asarray(coef)
    if raw_coef.shape != (768,) or raw_coef.dtype.kind not in "fiu":
        raise ValueError("coef must be a real numeric vector of length 768")
    coef = raw_coef.astype(np.float64)
    bias = _scalar(bias, "bias")
    if not np.isfinite(coef).all():
        raise ValueError("coef must be finite")
    linear_weight = torch.tensor(coef[:384].reshape(1, 384), dtype=torch.float32, device="cuda")
    nonlinear_weight = torch.tensor(coef[384:].reshape(1, 384), dtype=torch.float32, device="cuda")
    intercept = torch.tensor([bias], dtype=torch.float32, device="cuda")
    result = np.empty(len(x), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), 8):
            original_rows = x[start:start + 8]
            linear_rows = torch.from_numpy(original_rows).to("cuda")
            nonlinear_rows = torch.from_numpy(rff(original_rows, omega, scale)).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                linear = torch.nn.functional.linear(linear_rows, linear_weight, intercept).squeeze(1)
                nonlinear = torch.nn.functional.linear(nonlinear_rows, nonlinear_weight).squeeze(1)
            values = linear.float() + nonlinear.float()
            result[start:start + len(original_rows)] = values.cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite native logits")
    return result
