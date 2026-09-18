"""Fixed probe geometry, grouped negative controls, and original weighted BCE.

No research-data I/O or torch import occurs here. Callers provide only the
appropriate training partition to fit_geometry and one isolated partition to
group_donors. The two standalone historical numerical solver modules are reused.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


RESEARCH = Path(__file__).resolve().parents[2] / "work" / "router_research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
import weighted_linear_probe_refined as refined

original = refined.original
STD_THRESHOLD = 1e-12
PROBE_DIMENSIONS = 50
DONOR_TAG = "m6_probe_group_derangement_v1"


def _real_matrix(value, width, *, dtype=None, allow_empty=False):
    x = np.asarray(value)
    if (x.ndim != 2 or x.shape[1] != width or x.dtype.kind not in "fiu"
            or (dtype is not None and x.dtype != dtype)):
        raise ValueError(f"Expected real matrix with {width} columns and dtype {dtype}")
    if (not allow_empty and len(x) == 0) or not np.isfinite(x).all():
        raise ValueError("Matrix must be finite and nonempty unless explicitly allowed")
    return x


def _scalar(value, name):
    v = np.asarray(value)
    if v.ndim != 0 or v.dtype.kind not in "fiu" or not np.isfinite(v):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(v)


def fit_geometry(probe_train):
    """FP64 unweighted mean/population std of every supplied training row.

    No labels, row weights, validation rows, or feature selection are used.
    Both real and permuted arms must share this same unpermuted geometry.
    Columns with population std <= 1e-12 stay zero in every later partition.
    """
    p = _real_matrix(probe_train, PROBE_DIMENSIONS).astype(np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean = np.mean(p, axis=0)
        std = np.std(p, axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("Nonfinite training geometry")
    return {"mean": mean, "std": std, "active": std > STD_THRESHOLD,
            "training_rows": len(p)}


def _geometry(value):
    if not isinstance(value, dict) or not {"mean", "std", "active", "training_rows"} <= value.keys():
        raise ValueError("Incomplete frozen training geometry")
    mean, std, active = (np.asarray(value[k]) for k in ("mean", "std", "active"))
    n = value["training_rows"]
    if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1
            or mean.shape != (50,) or std.shape != (50,) or active.shape != (50,)
            or mean.dtype.kind not in "fiu" or std.dtype.kind not in "fiu"
            or active.dtype != np.bool_ or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std < 0)
            or not np.array_equal(active, std > STD_THRESHOLD)):
        raise ValueError("Invalid frozen training geometry")
    return mean.astype(np.float64), std.astype(np.float64), active


def transform(probe, geometry):
    """Apply frozen train-only z-scores/sqrt(50), returning FP32, all 50 columns.

    Inactive training columns are zero even if they vary in the evaluation data.
    This does not normalize individual rows or a concatenated M6+probe vector.
    """
    p = _real_matrix(probe, PROBE_DIMENSIONS, allow_empty=True).astype(np.float64)
    mean, std, active = _geometry(geometry)
    result = np.zeros(p.shape, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result[:, active] = ((p[:, active] - mean[active]) / std[active]) / math.sqrt(50)
        result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite transformed probe")
    return result


def _ids(value, name):
    a = np.asarray(value)
    if (a.ndim != 1 or len(a) == 0 or a.dtype.kind not in "UO"
            or any(not isinstance(v, str) or not v for v in a.tolist())):
        raise ValueError(f"{name} must contain nonempty string IDs")
    return a.astype(str)


def group_donors(query_ids, group_ids, seed, context):
    """Return local donor indices: new_probe[i] = old_probe[donors[i]].

    Within each group-size bucket, order group IDs by (SHA256 digest, group ID).
    Hash input is exactly the UTF-8 encoding of
      json.dumps(["m6_probe_group_derangement_v1", int(seed), context,
                  int(group_size), group_id],
                 ensure_ascii=False, separators=(",", ":"))
    Recipient group at position i takes donor group at (i+1) modulo bucket size.
    Both blocks are sorted by query_id before pairing their rows. Thus mapping
    is invariant to input row order, bijective, size preserving, and has no
    self-donor group. A bucket with only one group is rejected, with no fallback.

    The caller freezes seed and a context such as fold0/inner1/train, supplies
    one partition only, and leaves M6 features, labels, and weights unpermuted.
    This fixed negative control is not a permutation significance test.
    """
    q, g = _ids(query_ids, "query_ids"), _ids(group_ids, "group_ids")
    if len(q) != len(g) or len(np.unique(q)) != len(q):
        raise ValueError("IDs must align and query IDs must be unique")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0 or not isinstance(context, str) or not context):
        raise ValueError("Seed must be nonnegative integer and context nonempty string")
    blocks = {}
    for i, group in enumerate(g):
        blocks.setdefault(str(group), []).append(i)
    buckets = {}
    for group, rows in blocks.items():
        rows.sort(key=lambda i: q[i])
        buckets.setdefault(len(rows), []).append(group)
    donors = np.full(len(q), -1, dtype=np.int64)
    for size, groups in sorted(buckets.items()):
        if len(groups) < 2:
            raise ValueError(f"Cannot derange group-size bucket {size}: only one group")

        def key(group):
            payload = json.dumps([DONOR_TAG, int(seed), context, int(size), group],
                                 ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(payload).digest(), group

        ordered = sorted(groups, key=key)
        for i, recipient in enumerate(ordered):
            donor = ordered[(i + 1) % len(ordered)]
            donors[blocks[recipient]] = blocks[donor]
    if not np.array_equal(np.sort(donors), np.arange(len(q))) or np.any(g == g[donors]):
        raise RuntimeError("Grouped donor invariants failed")
    return donors


def augmented(x384, probe50_scaled):
    """Concatenate unchanged FP32 M6 and already-transformed FP32 probe."""
    x = _real_matrix(x384, 384, dtype=np.float32, allow_empty=True)
    p = _real_matrix(probe50_scaled, 50, dtype=np.float32, allow_empty=True)
    if len(x) != len(p):
        raise ValueError("M6 and probe rows must align")
    return np.concatenate((x, p), axis=1)


def fit_design(design434, gap, regularization):
    """Original mean weighted BCE + L2; no added retries or solver changes.

    The historical solver ignores the intercept in its L2 penalty. The caller
    must inspect the returned accepted flag and retain any rejected solution.
    """
    x = _real_matrix(design434, 434, dtype=np.float32)
    raw_gap = np.asarray(gap)
    if raw_gap.dtype.kind not in "fiu":
        raise ValueError("Gap must contain real numeric values")
    gap = raw_gap.astype(np.float64)
    if gap.shape != (len(x),) or not np.isfinite(gap).all() or np.any(np.abs(gap) > 1):
        raise ValueError("Gap must be finite, aligned, and lie in [-1, 1]")
    regularization = _scalar(regularization, "regularization")
    if regularization <= 0:
        raise ValueError("Regularization must be positive")
    labels = (gap > 0).astype(np.float64)
    weights = np.where(np.abs(gap) > 1e-12, np.abs(gap), 0.)
    return refined.fit(x, labels, weights, regularization=regularization)


def native_scores(x384, probe50_scaled, coef434, intercept, torch):
    """CUDA BF16 batch8 branches, sole M6 bias, then FP32 branch addition.

    The supplied torch backend may be a CPU adapter for synthetic tests only.
    With probe coefficients exactly zero the original M6 branch is unchanged.
    """
    x = _real_matrix(x384, 384, dtype=np.float32, allow_empty=True)
    p = _real_matrix(probe50_scaled, 50, dtype=np.float32, allow_empty=True)
    if len(x) != len(p):
        raise ValueError("M6 and probe rows must align")
    coef = np.asarray(coef434)
    if coef.shape != (434,) or coef.dtype.kind not in "fiu" or not np.isfinite(coef).all():
        raise ValueError("Coefficients must be a finite real vector of length 434")
    intercept = _scalar(intercept, "intercept")
    beta = torch.tensor(coef[:384].reshape(1, 384), dtype=torch.float32, device="cuda")
    theta = torch.tensor(coef[384:].reshape(1, 50), dtype=torch.float32, device="cuda")
    bias = torch.tensor([intercept], dtype=torch.float32, device="cuda")
    result = np.empty(len(x), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), 8):
            rows = x[start:start + 8]
            linear_rows = torch.from_numpy(rows).to("cuda")
            probe_rows = torch.from_numpy(p[start:start + 8]).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                linear = torch.nn.functional.linear(linear_rows, beta, bias).squeeze(1)
                extra = torch.nn.functional.linear(probe_rows, theta).squeeze(1)
            result[start:start + len(rows)] = (linear.float() + extra.float()).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite native logits")
    return result
