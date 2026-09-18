"""Fit one regularized logit offset per row without tuning or early stopping.

The last axis contains training points. One-dimensional inputs return a float;
two-dimensional inputs return one offset per row. Logits and gaps may broadcast
across rows. The objective averages over points, not over the sum of weights.
"""

import numpy as np
from scipy.special import expit


def _inputs(base_logits, gaps, ridge):
    base = np.asarray(base_logits, dtype=np.float64)
    gap = np.asarray(gaps, dtype=np.float64)
    if base.ndim not in (1, 2) or gap.ndim not in (1, 2):
        raise ValueError('Logits and gaps must be one- or two-dimensional')
    base, gap = np.broadcast_arrays(base, gap)
    if not base.shape[-1]:
        raise ValueError('The training-point axis must be nonempty')
    if not np.isfinite(base).all() or not np.isfinite(gap).all():
        raise ValueError('Logits and gaps must be finite')
    if (np.abs(gap) > 1).any():
        raise ValueError('Gaps must lie in [-1, 1]')
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Ridge must be finite and positive')
    return base, gap, ridge


def _gradient(base, gap, offsets, ridge):
    return np.mean(np.abs(gap) * (expit(base + offsets[..., None]) - (gap > 0)),
                   axis=-1) + ridge * offsets


def _output(value):
    return float(value) if np.ndim(value) == 0 else value


def offset_gradient(base_logits, gaps, offsets, ridge=0.01):
    """Return the offset derivative of the point-mean weighted BCE objective."""
    base, gap, ridge = _inputs(base_logits, gaps, ridge)
    offsets = np.broadcast_to(np.asarray(offsets, dtype=np.float64), base.shape[:-1])
    if not np.isfinite(offsets).all():
        raise ValueError('Offsets must be finite')
    return _output(_gradient(base, gap, offsets, ridge))


def fit_offsets(base_logits, gaps, ridge=0.01):
    """Minimize weighted BCE plus ridge * offset**2 / 2 by 60 bisections.

    Since abs(gap) <= 1, the unregularized gradient has magnitude at most one.
    The strictly increasing regularized gradient therefore has its unique root
    in [-1 / ridge, 1 / ridge]. Exact zero gradients collapse the interval but
    do not change the fixed iteration count.
    """
    base, gap, ridge = _inputs(base_logits, gaps, ridge)
    bound = 1.0 / ridge
    if not np.isfinite(bound):
        raise ValueError('Ridge is too small for a finite float64 bracket')
    lower = np.full(base.shape[:-1], -bound, dtype=np.float64)
    upper = np.full(base.shape[:-1], bound, dtype=np.float64)
    for _ in range(60):
        middle = 0.5 * lower + 0.5 * upper
        gradient = _gradient(base, gap, middle, ridge)
        lower = np.where(gradient <= 0, middle, lower)
        upper = np.where(gradient >= 0, middle, upper)
    return _output(0.5 * lower + 0.5 * upper)


def _self_check():
    from scipy.optimize import minimize_scalar

    tie = fit_offsets([3.0, -2.0, 0.4], [0.0, 0.0, 0.0])
    assert tie == 0.0

    symmetric = fit_offsets(np.zeros((2, 2)), [[0.8, -0.8], [0.35, -0.35]])
    assert symmetric.shape == (2,) and np.array_equal(symmetric, np.zeros(2))

    base = np.array([-1.2, 0.4, 1.1, -0.7])
    gap = np.array([0.9, -0.25, 0.2, -0.7])
    ridge = 0.07
    fitted = fit_offsets(base, gap, ridge)

    def objective(offset):
        logits = base + offset
        return float(np.mean(np.abs(gap) * (np.logaddexp(0.0, logits) - (gap > 0) * logits))
                     + 0.5 * ridge * offset ** 2)

    reference = minimize_scalar(objective, bounds=(-1 / ridge, 1 / ridge),
                                method='bounded', options={'xatol': 1e-13})
    residual = abs(offset_gradient(base, gap, fitted, ridge))
    difference = abs(fitted - reference.x)
    assert reference.success and difference < 1e-7 and residual < 1e-12
    return {'checks_passed': 3, 'all_ties_offset': tie,
            'symmetric_offsets': symmetric.tolist(), 'toy_offset': fitted,
            'toy_scipy_reference': float(reference.x),
            'toy_reference_difference': difference, 'toy_gradient_residual': residual}


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(_self_check(), indent=2))
    else:
        parser.print_help()
