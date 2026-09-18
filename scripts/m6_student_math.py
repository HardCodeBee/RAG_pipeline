"""FP64 weighted soft-label BCE with the historical bounded solver recipe.

No data I/O, target construction, model selection, torch, or encoder dependency.
Only the last-layer coefficients and an unpenalized intercept are optimized.
"""
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits


NUMERICAL_BUDGET = {"maxiter": 5000, "maxfun": 50000, "maxls": 50,
                    "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10}
ACCEPT_GRAD_INF = 1e-8
CPU_THREADS = 2
REFINEMENT = {"max_steps": 8, "max_backtracks": 25, "armijo": 1e-4,
              "roundoff_epsilon_multiplier": 8., "target_gradient_inf": 1e-8}


def _real(value, name, *, allow_bool=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in ("fiub" if allow_bool else "fiu"):
        raise ValueError(f"{name} must contain real numeric values")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _target_masses(target, normalized):
    # Selecting nonzero factors also matches the original binary initialization
    # exactly when all targets are 0 or 1.
    positive = float(np.sum(normalized[target > 0] * target[target > 0]))
    negative = float(np.sum(normalized[target < 1] * (1 - target[target < 1])))
    return positive, negative


def _inputs(x, target, weight, regularization):
    x = _real(x, "X")
    target = _real(target, "target", allow_bool=True)
    weight = _real(weight, "weight")
    lam = _real(regularization, "regularization")
    if lam.ndim != 0 or float(lam) <= 0:
        raise ValueError("regularization must be a finite positive scalar")
    if x.ndim != 2 or len(x) == 0 or target.shape != (len(x),) or weight.shape != (len(x),):
        raise ValueError("Expected nonempty X[N,D], target[N], and weight[N]")
    if np.any((target < 0) | (target > 1)) or np.any(weight < 0):
        raise ValueError("Targets must lie in [0,1] and weights be nonnegative")
    active = weight > 0
    if not active.any():
        raise ValueError("Positive total weight is required")
    x, target, normalized = x[active], target[active], weight[active]
    normalized = normalized / normalized.max()
    normalized = normalized / normalized.sum()
    positive, negative = _target_masses(target, normalized)
    if positive <= 0 or negative <= 0:
        raise ValueError("Both soft class masses must be positive for a finite unpenalized intercept")
    return x, target, normalized, float(lam)


def _objective(parameters, x, target, normalized_weight, regularization):
    beta, intercept = parameters[:-1], parameters[-1]
    logits = x @ beta + intercept
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite soft-BCE logits")
    # This is a mixture of the two hard-label losses, not signed softplus with
    # a fractional sign. Both terms stay stable for extreme finite logits.
    per_row = target * np.logaddexp(0., -logits) + (1 - target) * np.logaddexp(0., logits)
    data_loss = float(normalized_weight @ per_row)
    penalty = .5 * regularization * float(beta @ beta)
    residual = normalized_weight * (expit(logits) - target)
    gradient = np.r_[x.T @ residual + regularization * beta, residual.sum()]
    loss = data_loss + penalty
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite soft-BCE objective or gradient")
    return loss, gradient


def hessian(parameters, x, normalized, regularization):
    logits = x @ parameters[:-1] + parameters[-1]
    curvature = normalized * expit(logits) * expit(-logits)
    augmented = np.column_stack((x, np.ones(len(x))))
    value = augmented.T @ (curvature[:, None] * augmented)
    value[np.arange(x.shape[1]), np.arange(x.shape[1])] += regularization
    if not np.isfinite(value).all():
        raise FloatingPointError("Nonfinite soft-BCE Hessian")
    return value


def refine(parameters, x, target, normalized, regularization):
    """The original maximum-eight-step Newton correction with the soft loss."""
    point = np.asarray(parameters, dtype=np.float64).copy()
    loss, gradient = _objective(point, x, target, normalized, regularization)
    evaluations, trace, status = 1, [], "budget_exhausted"
    for iteration in range(REFINEMENT["max_steps"]):
        before = float(np.max(np.abs(gradient)))
        if before <= ACCEPT_GRAD_INF:
            status = "gradient_accepted"
            break
        matrix = hessian(point, x, normalized, regularization)
        try:
            direction = np.linalg.solve(matrix, gradient)
        except np.linalg.LinAlgError:
            status = "linear_solve_failed"
            break
        slope = float(gradient @ direction)
        if not np.isfinite(direction).all() or not np.isfinite(slope) or slope <= 0:
            status = "invalid_descent_direction"
            break
        rounding = REFINEMENT["roundoff_epsilon_multiplier"] * np.finfo(np.float64).eps * max(1., abs(loss))
        accepted = False
        for backtracks in range(REFINEMENT["max_backtracks"]):
            step = 2. ** (-backtracks)
            candidate = point - step * direction
            try:
                new_loss, new_gradient = _objective(candidate, x, target, normalized, regularization)
            except FloatingPointError:
                evaluations += 1
                continue
            evaluations += 1
            after = float(np.max(np.abs(new_gradient)))
            if after < before and new_loss <= loss - REFINEMENT["armijo"] * step * slope + rounding:
                accepted = True
                break
        if not accepted:
            status = "line_search_failed"
            break
        trace.append({"step": iteration + 1, "backtracks": backtracks, "step_size": step,
            "loss_before": loss, "loss_after": new_loss, "gradient_inf_before": before, "gradient_inf_after": after,
            "direction_inf": float(np.max(np.abs(direction))), "gradient_dot_direction": slope,
            "objective_roundoff_allowance": rounding})
        point, loss, gradient = candidate, new_loss, new_gradient
    final_gradient = float(np.max(np.abs(gradient)))
    if final_gradient <= ACCEPT_GRAD_INF:
        status = "gradient_accepted"
    return point, {"status": status, "iterations": len(trace), "objective_evaluations": evaluations,
                   "trace": trace, "final_gradient_inf": final_gradient}


def fit_soft(X, target, weight, regularization):
    """Return coefficients and the original-style flat diagnostic record.

    Inspect accepted before using a solution. A failed L-BFGS status is retained
    without correction. A successful but insufficiently stationary solution may
    receive only the pre-existing bounded Newton refinement; no budget grows.
    """
    x, target, normalized, regularization = _inputs(X, target, weight, regularization)
    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    positive, negative = _target_masses(target, normalized)
    initial[-1] = np.log(positive) - np.log(negative)
    started = time.perf_counter()
    with threadpool_limits(limits=CPU_THREADS):
        optimized = minimize(_objective, initial, args=(x, target, normalized, regularization),
                             method="L-BFGS-B", jac=True, options=dict(NUMERICAL_BUDGET))
        loss, gradient = _objective(optimized.x, x, target, normalized, regularization)
    if not np.isfinite(optimized.x).all():
        raise FloatingPointError("Nonfinite soft-BCE parameters")
    grad_inf = float(np.max(np.abs(gradient)))
    penalty = .5 * regularization * float(optimized.x[:-1] @ optimized.x[:-1])
    accepted = bool(optimized.success) and grad_inf <= ACCEPT_GRAD_INF
    result = {"coef": optimized.x[:-1].copy(), "intercept": float(optimized.x[-1]),
        "loss": float(loss), "data_loss": float(loss - penalty), "penalty": penalty,
        "grad_inf": grad_inf, "iterations": int(optimized.nit), "evaluations": int(optimized.nfev),
        "optimizer_success": bool(optimized.success), "optimizer_status": int(optimized.status),
        "optimizer_message": str(optimized.message), "accepted": accepted,
        "status": "accepted_stationary_solution" if accepted else "not_accepted",
        "acceptance_gradient_inf": ACCEPT_GRAD_INF, "regularization": regularization,
        "numerical_budget": dict(NUMERICAL_BUDGET), "cpu_threads": CPU_THREADS,
        "positive_weight_rows": len(target), "elapsed_seconds": time.perf_counter() - started}
    result["numerical_refinement"] = {"config": dict(REFINEMENT), "performed": False,
        "reason": "initial_soft_lbfgs_accepted" if accepted else "initial_soft_lbfgs_unsuccessful",
        "iterations": 0, "objective_evaluations": 0, "final_acceptance_objective_evaluations": 0, "trace": []}
    if result["optimizer_success"] and not accepted:
        initial_point = np.r_[result["coef"], result["intercept"]]
        initial_loss, initial_grad_inf = result["loss"], result["grad_inf"]
        with threadpool_limits(limits=CPU_THREADS):
            final, refinement = refine(initial_point, x, target, normalized, regularization)
            loss, gradient = _objective(final, x, target, normalized, regularization)
        penalty = .5 * regularization * float(final[:-1] @ final[:-1])
        grad_inf = float(np.max(np.abs(gradient)))
        accepted = grad_inf <= ACCEPT_GRAD_INF
        result.update(coef=final[:-1].copy(), intercept=float(final[-1]), loss=float(loss),
                      data_loss=float(loss - penalty), penalty=penalty, grad_inf=grad_inf,
                      accepted=bool(accepted), status="accepted_stationary_solution" if accepted else "not_accepted")
        result["numerical_refinement"] = {"config": dict(REFINEMENT), "performed": True,
            "reason": "initial_soft_lbfgs_success_above_gradient_threshold",
            "initial_coef": initial_point[:-1].tolist(), "initial_intercept": float(initial_point[-1]),
            "initial_loss": initial_loss, "initial_gradient_inf": initial_grad_inf,
            "final_acceptance_objective_evaluations": 1, **refinement}
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def predict(model, X):
    """Return untruncated FP64 scores; native BF16 inference belongs to caller."""
    x, coef = _real(X, "X"), _real(model["coef"], "coef")
    bias = _real(model["intercept"], "intercept")
    if x.ndim != 2 or coef.ndim != 1 or x.shape[1] != len(coef) or bias.ndim != 0:
        raise ValueError("Prediction feature and parameter shapes differ")
    with threadpool_limits(limits=CPU_THREADS):
        scores = x @ coef + float(bias)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite soft-BCE predictions")
    return scores
