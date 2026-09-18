"""Independent strict-OOF M6 Student checks; never train or invoke GPU inference.

Saved heads, targets, actual probe designs, native-score bounds, contributions,
and frequency-resampled group intervals are reconstructed without importing
experiment runners, feature/soft-loss helpers, or optimizers.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT.parent / "work" / "router_research"
DEFAULT_OUTPUT = PROJECT / "outputs/router/hotpotqa_bd_router_v1/runs/m6_student_transfer_v1"
ARMS = ("Direct", "Pre", "Probe")
PRIMARY = ("Pre_minus_Direct", "Probe_minus_Pre", "Probe_minus_Direct", "Probe_minus_Dense", "Probe_minus_BM25")
POLICIES = ARMS + ("Dense", "BM25")
TIE_ATOL = 1e-12
GRADIENT_ATOL = 1e-8
SCALAR_ATOL = 2e-12
FEATURE_ATOL = 2e-7
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 2026091509
QUANTILES = (.005, .995)
NUMERICAL_BUDGET = {"maxiter": 5000, "maxfun": 50000, "maxls": 50, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10}
REFINEMENT = {"max_steps": 8, "max_backtracks": 25, "armijo": .0001, "roundoff_epsilon_multiplier": 8.0, "target_gradient_inf": 1e-8}
EXPECTED_FOLDS_SHA256 = "ad2e94f76332f288f798d8bc1b1ff085be9aba64730b84d4906566d37ea76bc6"
PROBE_DIRECTORY = PROJECT / "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_9600_v1/privileged_teacher"
EXPECTED_PROBE_SHA256 = "0794f1664b6b5b01475741dad86dc6690579d160440926456d141c8cbb2f08da"

def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else PROJECT / path).resolve()


def bindings(mapping, label):
    require(isinstance(mapping, dict) and bool(mapping), f"Missing {label} bindings")
    checked = {}
    for filename, expected in mapping.items():
        path = resolve(filename)
        require(isinstance(expected, str) and sha(path) == expected, f"Changed {label}: {path}")
        checked[path] = expected
    return checked


def real_array(value, shape=None, name="array", dtype=None):
    array = np.asarray(value)
    require(array.dtype.kind in "fiu" and np.isfinite(array).all(), f"Invalid {name} values")
    if shape is not None:
        require(array.shape == shape, f"Invalid {name} shape: {array.shape}")
    if dtype is not None:
        require(array.dtype == np.dtype(dtype), f"Invalid {name} dtype: {array.dtype}")
    return array


def group_array(groups, n=None):
    result = np.asarray(groups)
    require(result.ndim == 1 and len(result) > 0, "Group IDs must form a nonempty vector")
    require(n is None or len(result) == n, "Group count differs from row count")
    require(all(isinstance(value, str) and bool(value) for value in result), "Group IDs must be nonempty strings")
    return result


def compare(actual, expected, name, errors, atol=SCALAR_ATOL):
    actual, expected = real_array(actual, name=name), real_array(expected, name=name)
    require(actual.shape == expected.shape, f"Shape differs: {name}")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    error = float(difference.max()) if difference.size else 0.0
    require(error <= atol, f"{name} differs by {error:.17g}; allowed {atol:.17g}")
    errors[name] = max(error, errors.get(name, 0.0))
    return error


def inner_groups(groups, fold):
    groups = group_array(groups)
    ranked = sorted(set(groups), key=lambda group: (
        hashlib.sha256(f"lp_ft_inner_v1|fold={fold}|group={group}".encode()).digest(), group))
    positions = {group: position % 3 for position, group in enumerate(ranked)}
    return np.asarray([positions[group] for group in groups], dtype=np.int64)


def reconstruct_scaler(training_probe):
    """Population mean/std from explicit per-column scalar sums, training only."""
    probe = real_array(training_probe, name="training probe").astype(np.float64)
    require(probe.ndim == 2 and probe.shape[1] == 50 and len(probe) > 0, "Expected nonempty 50D training probe")
    mean, deviation = [], []
    for column in range(50):
        values = list(map(float, probe[:, column]))
        center = math.fsum(values) / len(values)
        variance = math.fsum((value - center) ** 2 for value in values) / len(values)
        mean.append(center)
        deviation.append(math.sqrt(variance))
    mean, deviation = np.asarray(mean), np.asarray(deviation)
    require(np.isfinite(mean).all() and np.isfinite(deviation).all(), "Nonfinite training scaler")
    return {"mean": mean, "std": deviation, "active": deviation > TIE_ATOL}


def standardized_probe(probe, mean, std):
    """Reconstruct the frozen standardized probe block, including its FP32 cast."""
    probe = real_array(probe, name="raw probe").astype(np.float64)
    require(probe.ndim == 2 and probe.shape[1] == 50, "Expected 50D raw probe")
    mean = real_array(mean, (50,), "probe mean").astype(np.float64)
    std = real_array(std, (50,), "probe standard deviations").astype(np.float64)
    require(np.all(std >= 0), "Probe standard deviations must be nonnegative")
    output = np.zeros(probe.shape, dtype=np.float32)
    for column in np.flatnonzero(std > TIE_ATOL):
        output[:, column] = ((probe[:, column] - mean[column]) / std[column] / math.sqrt(50)).astype(np.float32)
    require(np.isfinite(output).all(), "Nonfinite standardized probe")
    return output


def signed_softplus(value):
    value = float(value)
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def policy_summary(utility, action):
    utility = real_array(utility, name="policy utility")
    require(utility.ndim == 2 and utility.shape[1] == 2 and len(utility) > 0,
            "Invalid policy utility shape")
    require(np.all((utility >= 0) & (utility <= 1)), "Utility outside [0,1]")
    action = np.asarray(action)
    require(action.shape == (len(utility),) and action.dtype == bool, "Invalid policy action")
    gap = utility[:, 0] - utility[:, 1]
    indices = np.flatnonzero(action).tolist()
    n = len(gap)
    return {"F1": math.fsum(float(utility[i, 0 if action[i] else 1]) for i in range(n)) / n,
            "bm25_count": len(indices),
            "beneficial_count": sum(bool(gap[i] > TIE_ATOL) for i in indices),
            "harmful_count": sum(bool(gap[i] < -TIE_ATOL) for i in indices),
            "zero_count": sum(bool(abs(gap[i]) <= TIE_ATOL) for i in indices),
            "beneficial_mass": math.fsum(max(float(gap[i]), 0.) for i in indices) / n,
            "harmful_mass": math.fsum(max(-float(gap[i]), 0.) for i in indices) / n}


def means(values):
    values = real_array(values, name="mean values")
    require(values.ndim == 2 and len(values) > 0, "Invalid mean values shape")
    return np.asarray([math.fsum(map(float, values[:, column])) / len(values)
                       for column in range(values.shape[1])])


def expected_probe_names():
    score_names = ("top1", "top2", "top3", "top5", "mean", "std", "range", "margin_1_2", "margin_1_5",
                   "relative_margin_1_2", "relative_margin_1_5", "coefficient_of_variation",
                   "relative_rank_slope", "normalized_softmax_entropy")
    names = [f"probe__{action}_{name}" for action in ("bm25", "dense") for name in score_names]
    names += [f"probe__{action}_{name}" for action in ("bm25", "dense")
              for name in ("context_token_count", "context_truncated", "query_context_jaccard")]
    names += ["probe__" + name for name in ("same_top1_doc", "doc_overlap_count_at_1", "doc_overlap_count_at_3",
              "doc_overlap_count_at_5", "doc_overlap_jaccard_at_3", "doc_overlap_jaccard_at_5",
              "reciprocal_rank_overlap", "bm25_top1_rank_in_dense", "dense_top1_rank_in_bm25")]
    names += ["probe__bm25_minus_dense_" + name for name in ("relative_margin_1_2", "relative_margin_1_5",
              "coefficient_of_variation", "relative_rank_slope", "normalized_softmax_entropy",
              "context_token_count_scaled", "query_context_jaccard")]
    return names


def historical_input_paths():
    paths = [RESEARCH / name for name in (
        "layer_pooling_v1/features.npz", "layer_pooling_v1/predictions.npz",
        "e02_results/fold_indices.npz", "m6_pooled_offset_v1/cal_logits.npz",
        "layer_pooling_v1/completion_record.json", "m6_pooled_offset_v1/completion_record.json")]
    paths += [RESEARCH / "layer_pooling_v1" / f"fold{fold}_M6_{suffix}"
              for fold in range(5) for suffix in ("head.pt", "fit.npz")]
    return paths + [PROBE_DIRECTORY / "teacher_features.npz", PROBE_DIRECTORY / "feature_schema.json",
                    RESEARCH / "e04_protocol.json", RESEARCH / "layer_pooling_v1/results.json"]


def load_original_inputs():
    """Load bound historical inputs only after the full-experiment gate passes."""
    directory = RESEARCH / "layer_pooling_v1"
    historical = {resolve(path): digest for path, digest in read(directory / "completion_record.json")["artifact_sha256"].items()}
    required = [directory / "features.npz", directory / "predictions.npz"]
    required += [directory / f"fold{fold}_M6_{suffix}" for fold in range(5) for suffix in ("head.pt", "fit.npz")]
    for path in required:
        require(path.resolve() in historical and sha(path) == historical[path.resolve()], f"Historical M6 artifact changed: {path}")
    split_path = RESEARCH / "e02_results/fold_indices.npz"
    require(sha(split_path) == EXPECTED_FOLDS_SHA256, "Historical split archive differs")
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        qids, groups, features = (archive[key].copy() for key in ("query_ids", "group_ids", "M6"))
    require(qids.shape == groups.shape == (9600,), "Historical identity shapes differ")
    require(qids.dtype.kind in "US" and groups.dtype.kind in "US", "Historical IDs must be strings")
    require(len(set(qids)) == 9600 and len(set(groups)) == 9559, "Historical identity counts differ")
    real_array(features, (9600, 384), "historical M6 features", np.float32)
    norms = np.sqrt(np.einsum("nd,nd->n", features.astype(float), features.astype(float), optimize=False))
    require(np.max(abs(norms - 1)) < 2e-6, "Historical M6 normalization differs")
    probe_path = PROBE_DIRECTORY / "teacher_features.npz"
    require(sha(probe_path) == EXPECTED_PROBE_SHA256, "Historical probe archive changed")
    e04 = read(RESEARCH / "e04_protocol.json")
    require(e04["teacher_features_sha256"] == EXPECTED_PROBE_SHA256, "E04 probe binding differs")
    schema = read(PROBE_DIRECTORY / "feature_schema.json")
    require(schema["probe"]["dimensions"] == 50 and schema["probe"]["feature_names"] == expected_probe_names(),
            "The original 50 probe fields or their order differ")
    require(schema["probe"]["uses_qrels"] is False and schema["probe"]["uses_generation_outcomes"] is False,
            "Probe schema does not meet the frozen information boundary")
    with np.load(probe_path, allow_pickle=False) as archive:
        require(np.array_equal(archive["query_ids"], qids) and np.array_equal(archive["group_ids"], groups),
                "Probe and M6 identities differ")
        # Do not access the archive's gold or historical Teacher prediction blocks.
        probe = real_array(archive["probe"], (9600, 50), "historical probe", np.float64).copy()
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        require(np.array_equal(qids, archive["query_ids"]) and np.array_equal(groups, archive["group_ids"]),
                "Historical prediction identities differ")
        utility, linear = archive["utility"].copy(), archive["M6"].copy()
    real_array(utility, (9600, 2), "historical utility", np.float64)
    require(np.all((utility >= 0) & (utility <= 1)), "Historical utility range differs")
    real_array(linear, (9600,), "historical M6 scores", np.float64)
    with np.load(split_path, allow_pickle=False) as archive:
        folds = [{part: archive[f"fold{fold}_{part}"].copy() for part in ("fit", "calibration", "test")}
                 for fold in range(5)]
    offset_directory = RESEARCH / "m6_pooled_offset_v1"
    offset_bindings = {resolve(path): digest for path, digest in read(offset_directory / "completion_record.json")["artifact_sha256"].items()}
    offset_path = offset_directory / "cal_logits.npz"
    require(offset_path.resolve() in offset_bindings and sha(offset_path) == offset_bindings[offset_path.resolve()],
            "Historical calibration score binding differs")
    with np.load(offset_path, allow_pickle=False) as archive:
        old_cal = []
        for fold in range(5):
            require(np.array_equal(archive[f"fold{fold}_cal_indices"], folds[fold]["calibration"]),
                    "Historical calibration identities differ")
            old_cal.append(real_array(archive[f"fold{fold}_O"], (1536,), "historical cal scores", np.float32).copy())
    coverage = np.zeros(9600, dtype=np.int64)
    assignments, old_fit, old_coefs, old_intercepts, old_inner = [], [], [], [], []
    # CPU-only deserialization; no model, encoder, CUDA context, or inference.
    import torch
    for fold, parts in enumerate(folds):
        group_sets = []
        for part, size in (("fit", 6144), ("calibration", 1536), ("test", 1920)):
            indices = real_array(parts[part], (size,), f"fold{fold} {part}")
            require(indices.dtype.kind in "iu" and len(set(indices)) == size
                    and np.all((indices >= 0) & (indices < 9600)), "Invalid historical split indices")
            group_sets.append(set(groups[indices]))
        require(not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2]
                     or group_sets[1] & group_sets[2]), "Historical outer group leakage")
        require(np.array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(9600)),
                "Historical fold fails to partition the pool")
        coverage[parts["test"]] += 1
        assigned = inner_groups(groups[parts["fit"]], fold)
        assignments.append(assigned)
        with np.load(directory / f"fold{fold}_M6_fit.npz", allow_pickle=False) as archive:
            require(np.array_equal(archive["fit_indices"], parts["fit"])
                    and np.array_equal(archive["inner_assignment"], assigned), "Historical inner split differs")
            old_fit.append(real_array(archive["final_native_fit_logits"], (6144,), "historical fit scores", np.float32).copy())
            old_inner.append(real_array(archive["inner_validation_native_logits"], (5, 6144),
                                        "historical M6 inner native scores", np.float32)[1].copy())
        head = torch.load(directory / f"fold{fold}_M6_head.pt", map_location="cpu", weights_only=True)
        old_coefs.append(real_array(head["weight"].numpy(), (1, 384), "historical head coefficient", np.float32).reshape(384).copy())
        old_intercepts.append(float(real_array(head["bias"].numpy(), (1,), "historical head bias", np.float32)[0]))
    require(np.all(coverage == 1), "Historical OOF coverage differs")
    return {"query_ids": qids, "group_ids": groups, "features": features, "probe": probe,
            "utility": utility, "gap": utility[:, 0] - utility[:, 1], "L_scores": linear,
            "folds": folds, "inner_assignments": assignments, "L_fit": old_fit, "L_cal": old_cal,
            "L_coef": old_coefs, "L_intercept": old_intercepts, "L_inner": old_inner}


def weights_for_gap(gap):
    gap = real_array(gap, name="gap").astype(np.float64)
    require(gap.ndim == 1 and np.all(np.abs(gap) <= 1), "Invalid gap")
    return np.asarray([abs(float(v)) if abs(float(v)) > TIE_ATOL else 0. for v in gap])


def normalized_active_weights(weights):
    weights = real_array(weights, name="weights").astype(np.float64)
    require(weights.ndim == 1 and len(weights) > 0 and np.all(weights >= 0), "Invalid weights")
    active = np.flatnonzero(weights > 0)
    require(len(active) > 0, "No positive weight")
    scale = max(map(float, weights[active]))
    scaled = [float(weights[i]) / scale for i in active]
    total = math.fsum(scaled)
    normalized = np.asarray([v / total for v in scaled], dtype=np.float64)
    return active, normalized


def sigmoid(scores):
    scores = real_array(scores, name="sigmoid scores").astype(np.float64)
    def scalar(z):
        z = float(z)
        if z >= 0:
            return 1. / (1. + math.exp(-z))
        t = math.exp(z)
        return t / (1. + t)
    return np.asarray([scalar(z) for z in scores.ravel()]).reshape(scores.shape)


def build_targets(gap, teacher_logits, alpha=.5):
    gap = real_array(gap, name="target gap")
    logits = real_array(teacher_logits, (2, len(gap)), "OOF Teacher logits", np.float32)
    require(type(alpha) in (float, int) and 0 <= alpha <= 1, "Invalid alpha")
    y = (gap > 0).astype(np.float64)
    probability = sigmoid(logits.astype(np.float64))
    targets = np.asarray([y, (1 - alpha) * y + alpha * probability[0],
                          (1 - alpha) * y + alpha * probability[1]])
    return probability, targets


def loss_quantities(features, target, weights, coef, intercept, regularization=.001, hessian=False):
    """Scalar softplus and separate einsum gradients, in original coordinates."""
    features = real_array(features, name="objective design")
    require(features.ndim == 2 and len(features) > 0, "Invalid objective design")
    target = real_array(target, (len(features),), "soft target").astype(np.float64)
    require(np.all((target >= 0) & (target <= 1)), "Target outside [0,1]")
    weights = real_array(weights, (len(features),), "objective weights")
    coef = real_array(coef, (features.shape[1],), "coefficient").astype(np.float64)
    require(math.isfinite(float(intercept)) and math.isfinite(regularization) and regularization > 0,
            "Invalid bias or regularization")
    active, normalized = normalized_active_weights(weights)
    x, q = features[active].astype(np.float64), target[active]
    positive = math.fsum(float(w) * float(v) for w, v in zip(normalized, q))
    negative = math.fsum(float(w) * (1. - float(v)) for w, v in zip(normalized, q))
    require(positive > 0 and negative > 0, "Both soft target masses must be positive")
    logits = np.einsum("nd,d->n", x, coef, optimize=False) + float(intercept)
    require(np.isfinite(logits).all(), "Nonfinite objective logits")
    probabilities = sigmoid(logits)
    terms = [float(w) * (float(v) * signed_softplus(-float(z)) +
                        (1. - float(v)) * signed_softplus(float(z)))
             for w, v, z in zip(normalized, q, logits)]
    data_loss = math.fsum(terms)
    penalty = regularization * math.fsum(float(v) ** 2 for v in coef) / 2
    residual = normalized * (probabilities - q)
    gradient = np.r_[np.einsum("nd,n->d", x, residual, optimize=False) + regularization * coef,
                     math.fsum(map(float, residual))]
    require(np.isfinite(gradient).all(), "Nonfinite objective gradient")
    out = dict(data_loss=data_loss, penalty=penalty, loss=data_loss + penalty,
               gradient=gradient, grad_inf=float(np.max(np.abs(gradient))),
               positive_weight_rows=len(active), positive_target_mass=positive,
               negative_target_mass=negative)
    if hessian:
        design = np.column_stack((x, np.ones(len(x))))
        curvature = normalized * probabilities * (1. - probabilities)
        matrix = np.einsum("ni,n,nj->ij", design, curvature, design, optimize=False)
        matrix[np.arange(len(coef)), np.arange(len(coef))] += regularization
        out["hessian"] = matrix
    return out


def weighted_bce(scores, gap):
    scores = real_array(scores, (len(gap),), "hard BCE scores")
    active, weights = normalized_active_weights(weights_for_gap(gap))
    return math.fsum(float(w) * signed_softplus(-float(scores[i]) if gap[i] > 0 else float(scores[i]))
                     for i, w in zip(active, weights))


def bf16_round(value):
    """CPU IEEE nearest-even BF16 rounding, returned as exact FP32 values."""
    values = real_array(value, name="BF16 input").astype(np.float32)
    require(np.isfinite(values).all(), "FP32 head export overflow")
    bits = np.ascontiguousarray(values).view(np.uint32)
    rounded = ((bits.astype(np.uint64) + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    result = rounded.view(np.float32).reshape(values.shape)
    require(np.isfinite(result).all(), "BF16 cast overflow")
    return result


def branch_reference(features, coef, intercept=None):
    """FP64 dot of BF16 operands, with FP32 accumulation and BF16-output bound.

    The forward-error envelope assumes the frozen ordinary FP32 accumulation
    path; it is a consistency test, not exact reproduction of every CUDA kernel.
    gamma_(D+1) is conservative for D rounded products/additions including bias.
    """
    x = bf16_round(features).astype(np.float64)
    beta = bf16_round(coef).astype(np.float64)
    bias = 0. if intercept is None else float(bf16_round(np.asarray(intercept)))
    require(x.ndim == 2 and beta.shape == (x.shape[1],), "Native reference shape mismatch")
    center = np.einsum("nd,d->n", x, beta, optimize=False) + bias
    magnitude = np.einsum("nd,d->n", np.abs(x), np.abs(beta), optimize=False) + abs(bias)
    require(np.isfinite(center).all() and np.isfinite(magnitude).all(), "Native reference overflow")
    u32, ub = 2. ** -24, 2. ** -8
    operations = x.shape[1] + 1
    gamma = operations * u32 / (1. - operations * u32)
    accumulation = gamma * magnitude + operations * 2. ** -149
    bound = ub * np.abs(center) + (1. + ub) * accumulation + 2. ** -134
    return center, bound


def native_score_bound(features, coef, intercept, scores, counters, name, probe=None):
    scores = real_array(scores, (len(features),), name + " native scores", np.float32)
    if probe is None:
        center, bound = branch_reference(features, coef, intercept)
        require(np.array_equal(scores, bf16_round(scores)), name + " single branch scores are not BF16-representable")
    else:
        c0, b0 = branch_reference(features, coef[:384], intercept)
        c1, b1 = branch_reference(probe, coef[384:])
        center = c0 + c1
        bound = b0 + b1 + 2. ** -24 * (np.abs(c0) + np.abs(c1) + b0 + b1) + 2. ** -149
    error = np.abs(scores.astype(np.float64) - center)
    # A tiny FP64 evaluation allowance only; no action or effect tolerance.
    allowance = 16 * np.finfo(np.float64).eps * np.maximum(np.abs(center), 1.)
    require(np.all(error <= bound + allowance), name + " cached scores do not match their head/input envelope")
    counters["score_vectors_checked"] = counters.get("score_vectors_checked", 0) + 1
    counters["score_rows_checked"] = counters.get("score_rows_checked", 0) + len(scores)
    counters["maximum_absolute_error_from_BF16_operand_FP64_center"] = max(counters.get("maximum_absolute_error_from_BF16_operand_FP64_center", 0.), float(error.max(initial=0.)))
    counters["maximum_allowed_absolute_bound"] = max(counters.get("maximum_allowed_absolute_bound", 0.), float((bound + allowance).max(initial=0.)))
    return center, bound


def frequency_bootstrap(values, groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    values = real_array(values, name="bootstrap contrasts")
    require(values.ndim == 2 and len(values) > 0 and type(draws) is int and draws > 0, "Invalid bootstrap dimensions")
    groups = group_array(groups, len(values))
    members = {}
    for i, group in enumerate(groups):
        members.setdefault(str(group), []).append(i)
    names = sorted(members)
    sizes = np.asarray([len(members[g]) for g in names], dtype=np.int64)
    totals = np.asarray([[math.fsum(float(values[i, j]) for i in members[g])
                          for j in range(values.shape[1])] for g in names])
    samples = np.empty((draws, values.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for draw in range(draws):
        chosen = rng.integers(0, len(names), size=len(names))
        count = np.bincount(chosen, minlength=len(names))
        active = np.flatnonzero(count)
        denominator = int(count[active] @ sizes[active])
        samples[draw] = np.einsum("i,ij->j", count[active], totals[active], optimize=False) / denominator
    return samples, np.quantile(samples, QUANTILES, axis=0, method="linear").T


def primary_values(gap, actions):
    a = {}
    for name in ARMS:
        value = np.asarray(actions[name])
        require(value.shape == gap.shape and value.dtype == bool, "Invalid Student actions")
        a[name] = value.astype(np.int64)
    return np.column_stack(((a["Pre"] - a["Direct"]) * gap,
                            (a["Probe"] - a["Pre"]) * gap,
                            (a["Probe"] - a["Direct"]) * gap,
                            a["Probe"] * gap, (a["Probe"] - 1) * gap))


def gates(point, intervals):
    point = real_array(point, (5,), "gate point values")
    intervals = real_array(intervals, (5, 2), "gate intervals")
    transfer = bool(all(point[j] >= .002 and intervals[j, 0] > 0 for j in (1, 2)))
    candidate = transfer and all(point[j] >= .01 and intervals[j, 0] > 0 for j in (3, 4))
    decision = ("PREPARE_FROZEN_STUDENT_INDEPENDENT_CONFIRMATION" if candidate else
                "TRANSFER_INCREMENT_ONLY_NO_QUALIFIED_CANDIDATE" if transfer else
                "END_FIXED_SOFT_BCE_STUDENT_RECIPE_NO_CONFIRMED_TRANSFER")
    return transfer, bool(candidate), decision


def expected_config():
    return dict(queries=9600, groups=9559, folds=5, fit_queries=6144, cal_queries=1536,
        test_queries=1920, inner_folds=3, regularization=.001, alpha=.5,
        student_dimensions=384, probe_dimensions=50, tie_atol=1e-12, std_ddof=0,
        std_atol=1e-12, probe_normalizer="sqrt(50)", gradient_atol=1e-8, batch_size=8,
        formal_solves=45, teacher_heads=30, student_heads=15, pilot_solves=5,
        new_thresholds=0, bootstrap_draws=20000, bootstrap_seed=2026091509,
        interval_quantiles=[.005, .995], minimum_transfer_increment=.002,
        minimum_gain_over_both_fixed=.01, cpu_threads=2, new_encoder_forwards=0,
        new_external_calls=0, runtime_available=False)


def expected_source_paths():
    return [PROJECT / "scripts" / name for name in ("run_m6_student_transfer.py", "m6_student_math.py",
        "m6_probe_math.py", "run_m6_probe_readout.py", "run_m6_objective_readout.py", "m6_objective_math.py",
        "check_m6_student_transfer.py")] + [PROJECT / "analysis/hotpotqa_router/m6_student_transfer_plan_20260915.md",
        RESEARCH / "weighted_linear_probe.py", RESEARCH / "weighted_linear_probe_refined.py",
        PROJECT / "tests/test_m6_student_math.py"]


def expected_artifacts(output):
    fit_files = []
    for f in range(5):
        for k in range(3):
            prefix = f"fold{f}_inner{k}"
            fit_files.append(output / (prefix + "_context.npz"))
            fit_files += [output / (prefix + "_T_" + name + "." + suffix)
                          for name in ("pre", "probe") for suffix in ("npz", "json")]
        fit_files += [output / f"fold{f}_{part}.npz" for part in ("targets", "fit_cal")]
        fit_files += [output / f"fold{f}_S_{name}.{suffix}" for name in ARMS for suffix in ("npz", "json")]
    fit_files.append(output / "solutions.jsonl")
    pilot_files = [output / "pilot_context.npz", output / "pilot_targets.npz"]
    pilot_files += [output / f"pilot_T_{name}.{suffix}" for name in ("pre", "probe") for suffix in ("npz", "json")]
    pilot_files += [output / f"pilot_S_{name}.{suffix}" for name in ARMS for suffix in ("npz", "json")]
    required = fit_files + pilot_files + [output / name for name in ("protocol.json", "pilot.json", "fit_started.json",
        "fit_completion.json", "predictions.npz", "predictions_frozen.json", "bootstrap.npz", "results.json")]
    return fit_files, pilot_files, required


def context_cache(path, train, valid, data, errors):
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == {"train_indices", "valid_indices", "mean", "std", "active", "probe_train", "probe_valid"}, "Wrong context members")
        require(np.array_equal(z["train_indices"], train) and np.array_equal(z["valid_indices"], valid), "Teacher row order differs")
        require(not set(data["group_ids"][train]) & set(data["group_ids"][valid]), "Teacher target-group leakage")
        reconstructed = reconstruct_scaler(data["probe"][train])
        compare(z["mean"], reconstructed["mean"], "Teacher_training_mean", errors, 2e-9)
        compare(z["std"], reconstructed["std"], "Teacher_training_std", errors, 2e-9)
        require(z["active"].dtype == bool and np.array_equal(z["active"], reconstructed["active"]), "Teacher constant columns differ")
        actual = {}
        for name, rows in (("train", train), ("valid", valid)):
            cached = real_array(z["probe_" + name], (len(rows), 50), "actual Teacher probe cache", np.float32).copy()
            expected = standardized_probe(data["probe"][rows], reconstructed["mean"], reconstructed["std"])
            compare(cached, expected, "Teacher_probe_transform", errors, FEATURE_ATOL)
            require(np.all(cached[:, ~reconstructed["active"]] == 0), "Inactive probe columns are not exactly zero")
            actual[name] = cached
    return actual


def checked_model(output, stem, features, gap, target, metadata, errors, soft=False):
    record = read(output / (stem + ".json"))
    require(all(record[key] == value for key, value in metadata.items()), "Wrong solution identity: " + stem)
    require(record["accepted"] is True and record["optimizer_success"] is True and
            record["status"] == "accepted_stationary_solution" and record["acceptance_gradient_inf"] == GRADIENT_ATOL,
            "Unaccepted solution: " + stem)
    require(record["regularization"] == .001 and record["cpu_threads"] == 2 and
            record["numerical_budget"] == NUMERICAL_BUDGET, "Changed numerical budget: " + stem)
    require(0 <= record["iterations"] <= 5000 and record["evaluations"] > 0, "Invalid optimizer counters")
    refinement = record["numerical_refinement"]
    require(refinement["config"] == REFINEMENT and 0 <= refinement["iterations"] <= 8 and
            len(refinement["trace"]) == refinement["iterations"], "Changed refinement budget")
    require(all(0 <= row["backtracks"] < 25 for row in refinement["trace"]), "Refinement backtrack budget exceeded")
    with np.load(output / (stem + ".npz"), allow_pickle=False) as z:
        require(set(z.files) == {"coef", "intercept"}, "Wrong saved model members")
        coef = real_array(z["coef"], (features.shape[1],), "model coefficient", np.float64).copy()
        intercept = float(real_array(z["intercept"], (), "model intercept", np.float64))
    compare(record["intercept"], intercept, "model_intercept", errors)
    if not soft:
        require(np.array_equal(target, (gap > 0).astype(np.float64)), "Hard Teacher/Direct target differs")
    actual = loss_quantities(features, target, weights_for_gap(gap), coef, intercept)
    require(actual["grad_inf"] <= GRADIENT_ATOL, "Independent full-gradient threshold failed: " + stem)
    require(record["positive_weight_rows"] == actual["positive_weight_rows"], "Wrong active training size")
    for name in ("data_loss", "penalty", "loss", "grad_inf"):
        compare(record[name], actual[name], ("soft_" if soft else "hard_") + name, errors,
                1e-10 if name == "grad_inf" else SCALAR_ATOL)
    return {"coef": coef, "intercept": intercept, "gradient_inf": actual["grad_inf"], "record": record,
            "stem": stem, "kind": "soft" if soft else "hard"}


def check_targets(archive, gap, errors):
    logits = real_array(archive["teacher_logits"], (2, len(gap)), "Teacher target logits", np.float32).copy()
    probability, targets = build_targets(gap, logits)
    compare(real_array(archive["teacher_probability"], probability.shape, "Teacher probabilities", np.float64), probability, "Teacher_sigmoid", errors)
    compare(real_array(archive["targets"], targets.shape, "Student targets", np.float64), targets, "Student_target_mix", errors)
    require(np.all((targets >= 0) & (targets <= 1)), "Student target range differs")
    return logits, targets


def check(output_dir, protocol_sha):
    output = Path(output_dir).resolve()
    destination = output / "separate_checks.json"
    require(not destination.exists(), "Preserve existing separate_checks.json")
    fit_files, pilot_files, required = expected_artifacts(output)
    require(all(path.is_file() for path in required), "Wait for the complete formal experiment")
    started = time.perf_counter()
    require(sha(output / "protocol.json") == protocol_sha, "Protocol SHA differs")
    protocol = read(output / "protocol.json")
    require(protocol["status"] == "frozen_before_real_pilot_and_formal_student_fits" and
            protocol["config"] == expected_config(), "Frozen protocol/config differs")
    require(protocol["primary"] == list(PRIMARY) and protocol["policies"] == list(POLICIES), "Primary/policy definitions differ")
    require(protocol["versions"] == {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "threadpoolctl")}, "Software versions changed")
    sources = bindings(protocol["source_sha256"], "source")
    inputs = bindings(protocol["input_sha256"], "input")
    require(set(sources) == {p.resolve() for p in expected_source_paths()}, "Wrong frozen source set")
    require(set(inputs) == {p.resolve() for p in historical_input_paths()}, "Wrong frozen input set")
    snapshots = {str(p): sha(p) for p in required}
    result = read(output / "results.json")
    require(result["status"] == "complete_fixed_student_transfer_pending_separate_check" and result["protocol_sha256"] == protocol_sha, "Student results not complete/bound")
    for key, count in (("formal_solves", 45), ("teacher_heads", 30), ("student_heads", 15), ("pilot_solves", 5),
                       ("new_thresholds", 0), ("new_encoder_forwards", 0), ("new_external_calls", 0)):
        require(type(result[key]) is int and result[key] == count, "Wrong completed count: " + key)
    require(result["runtime_available"] is False and result["core_goal_achieved"] is False and result["scope"] == protocol["scope"], "Wrong development evidence boundary")
    require(set(result["primary"]) == set(PRIMARY) and set(result["policy"]) == set(POLICIES) and len(result["folds"]) == 5, "Result summary structure differs")
    completion = read(output / "fit_completion.json")
    require(completion["status"] == "all_30_teacher_and_15_student_heads_frozen_before_test_prediction" and completion["protocol_sha256"] == protocol_sha, "Wrong pre-test fit completion")
    require(completion["formal_solves"] == 45 and completion["teacher_heads"] == 30 and completion["student_heads"] == 15, "Fit completion counts differ")
    require(set(bindings(completion["artifact_sha256"], "fit completion")) == set(fit_files), "Incomplete fit artifact set")
    require(read(output / "fit_started.json")["protocol_sha256"] == protocol_sha, "Fit-start protocol differs")
    frozen = read(output / "predictions_frozen.json")
    require(frozen["status"] == "all_9600_OOF_student_predictions_before_effects" and frozen["protocol_sha256"] == protocol_sha, "Wrong prediction freeze")
    require(frozen["predictions_sha256"] == snapshots[str(output / "predictions.npz")] and
            frozen["fit_completion_sha256"] == snapshots[str(output / "fit_completion.json")], "Prediction bindings differ")
    require(result["predictions_frozen_sha256"] == snapshots[str(output / "predictions_frozen.json")] and
            result["bootstrap_sha256"] == snapshots[str(output / "bootstrap.npz")], "Final analysis artifact binding differs")
    pilot = read(output / "pilot.json")
    require(pilot["status"] == "passed_five_pilot_fits_and_480_native_controls" and pilot["protocol_sha256"] == protocol_sha, "Wrong pilot binding")
    require(pilot["pilot_solves"] == 5 and pilot["teacher_train_rows"] == 128 and pilot["target_and_student_rows"] == 64 and
            pilot["original_native_control_queries"] == 480 and pilot["new_policy_effects"] is False and
            pilot["new_encoder_forwards"] == pilot["new_external_calls"] == 0, "Wrong pilot scope")
    controls = {f"fold{f}_{part}" for f in range(5) for part in ("fit", "cal", "test")}
    require(set(pilot["original_controls"]) == controls and all(v == 0 for v in pilot["original_controls"].values()), "Original native pilot control failed")
    require(set(bindings(pilot["artifact_sha256"], "pilot")) == set(pilot_files), "Wrong pilot artifact set")
    journal = [json.loads(line) for line in (output / "solutions.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(journal) == 45 and len({row["stem"] for row in journal}) == 45, "Wrong formal solve journal")
    data = load_original_inputs()
    require(all(row["selected_lambdas"]["M6"] == .001 for row in read(RESEARCH / "layer_pooling_v1/results.json")["per_fold"]), "Historical Direct lambda differs")
    qids, groups, x, gap, utility = (data[k] for k in ("query_ids", "group_ids", "features", "gap", "utility"))
    errors, native_counts, all_models, pilot_models, partitions = {}, {}, [], [], []
    cache_rows, target_rows = 0, 0
    with threadpool_limits(limits=2):
        multiplicity = {}
        for g in groups:
            multiplicity[g] = multiplicity.get(g, 0) + 1
        ids = np.asarray([i for i in data["folds"][0]["fit"] if multiplicity[groups[i]] == 1][:192])
        require(len(ids) == 192, "Pilot singleton selection differs")
        pt, pv = ids[:128], ids[128:]
        ctx = context_cache(output / "pilot_context.npz", pt, pv, data, errors)
        with np.load(output / "pilot_targets.npz", allow_pickle=False) as z:
            require(set(z.files) == {"train_indices", "target_indices", "teacher_logits", "teacher_probability", "targets"}, "Pilot targets members differ")
            require(np.array_equal(z["train_indices"], pt) and np.array_equal(z["target_indices"], pv), "Pilot target identities differ")
            logits, targets = check_targets(z, gap[pv], errors)
        for j, name in enumerate(("pre", "probe")):
            design = x[pt] if j == 0 else np.concatenate((x[pt], ctx["train"]), axis=1)
            model = checked_model(output, "pilot_T_" + name, design, gap[pt], (gap[pt] > 0).astype(float),
                dict(role="teacher", teacher=name, context="pilot_context.npz", train_rows=128, target_rows=64), errors)
            pilot_models.append(model)
            native_score_bound(x[pv], model["coef"], model["intercept"], logits[j], native_counts,
                               "pilot_T_" + name, None if j == 0 else ctx["valid"])
        for j, name in enumerate(ARMS):
            pilot_models.append(checked_model(output, "pilot_S_" + name, x[pv], gap[pv], targets[j],
                                dict(role="student", arm=name), errors, soft=j > 0))
        with np.load(output / "predictions.npz", allow_pickle=False) as z:
            require(set(z.files) == {"query_ids", "group_ids", "utility", "arm_scores", "fold_id", *ARMS}, "Prediction members differ")
            require(np.array_equal(z["query_ids"], qids) and np.array_equal(z["group_ids"], groups) and np.array_equal(z["utility"], utility), "Prediction inputs/identities differ")
            scores = real_array(z["arm_scores"], (3, 9600), "Student native test scores", np.float32).copy()
            fold_ids = real_array(z["fold_id"], (9600,), "prediction fold IDs").copy()
            require(fold_ids.dtype.kind in "iu", "Fold IDs not integral")
            actions = {name: scores[j] > 0 for j, name in enumerate(ARMS)}
            for name in ARMS:
                require(z[name].dtype == bool and np.array_equal(z[name], actions[name]), "Student native action differs")
        expected_folds = np.full(9600, -1, dtype=np.int64)
        for f, parts in enumerate(data["folds"]):
            fit, cal, test = (parts[k] for k in ("fit", "calibration", "test"))
            assigned = data["inner_assignments"][f]
            expected_folds[test] = f
            with np.load(output / f"fold{f}_targets.npz", allow_pickle=False) as z:
                require(set(z.files) == {"fit_indices", "inner_assignment", "teacher_logits", "teacher_probability", "targets", "weights"}, "Formal target members differ")
                require(np.array_equal(z["fit_indices"], fit) and np.array_equal(z["inner_assignment"], assigned), "Formal target fold/inner assignment differs")
                require(np.array_equal(z["weights"], weights_for_gap(gap[fit])), "Original utility weights changed")
                logits, targets = check_targets(z, gap[fit], errors)
            require(np.array_equal(logits[0], data["L_inner"][f]),
                    "Pre Teacher OOF logits differ from original M6 lambda-index1 inner native logits")
            visits = np.zeros(len(fit), int)
            for k in range(3):
                train, valid = fit[assigned != k], fit[assigned == k]
                require(not set(groups[train]) & set(groups[np.r_[valid, cal, test]]), "Teacher/outer group leakage")
                prefix = f"fold{f}_inner{k}"
                ctx = context_cache(output / (prefix + "_context.npz"), train, valid, data, errors)
                cache_rows += len(train) + len(valid)
                visits[assigned == k] += 1
                partitions.append(dict(fold=f, inner=k, train_rows=len(train), target_rows=len(valid)))
                for j, name in enumerate(("pre", "probe")):
                    design = x[train] if j == 0 else np.concatenate((x[train], ctx["train"]), axis=1)
                    model = checked_model(output, prefix + "_T_" + name, design, gap[train], (gap[train] > 0).astype(float),
                        dict(role="teacher", teacher=name, context=prefix + "_context.npz", train_rows=len(train), target_rows=len(valid)), errors)
                    all_models.append(model)
                    native_score_bound(x[valid], model["coef"], model["intercept"], logits[j, assigned == k], native_counts,
                                       prefix + "_T_" + name, None if j == 0 else ctx["valid"])
            require(np.all(visits == 1), "OOF Teacher target coverage differs")
            target_rows += len(fit)
            with np.load(output / f"fold{f}_fit_cal.npz", allow_pickle=False) as z:
                require(set(z.files) == {"fit_indices", "cal_indices", "fit_scores", "cal_scores"}, "Student fit/cal members differ")
                require(np.array_equal(z["fit_indices"], fit) and np.array_equal(z["cal_indices"], cal), "Student fit/cal identities differ")
                fs = real_array(z["fit_scores"], (3, 6144), "Student fit scores", np.float32).copy()
                cs = real_array(z["cal_scores"], (3, 1536), "Student cal scores", np.float32).copy()
            record = result["folds"][f]
            require(record["fold"] == f and record["direct_original_replay_exact"] is True, "Fold metadata differs")
            compare(record["teacher_OOF_BCE"], [weighted_bce(row, gap[fit]) for row in logits], "Teacher_OOF_BCE", errors)
            for j, name in enumerate(ARMS):
                model = checked_model(output, f"fold{f}_S_{name}", x[fit], gap[fit], targets[j],
                    dict(role="student", fold=f, arm=name, targets=f"fold{f}_targets.npz"), errors, soft=j > 0)
                all_models.append(model)
                if j == 0:
                    require(np.array_equal(model["coef"].astype(np.float32), data["L_coef"][f]) and
                            np.float32(model["intercept"]) == np.float32(data["L_intercept"][f]), "Direct head export differs from original M6")
                    require(np.array_equal(fs[0], data["L_fit"][f]) and np.array_equal(cs[0], data["L_cal"][f]), "Direct fit/cal native replay differs")
                for part, ix, values in (("fit", fit, fs[j]), ("cal", cal, cs[j]), ("test", test, scores[j, test])):
                    native_score_bound(x[ix], model["coef"], model["intercept"], values, native_counts, f"fold{f}_S_{name}_{part}")
            for part, values, ix in (("fit", fs, fit), ("cal", cs, cal), ("test", scores[:, test], test)):
                compare(record[part + "_BCE"], [weighted_bce(row, gap[ix]) for row in values], "Student_" + part + "_BCE", errors)
        require(np.array_equal(fold_ids, expected_folds), "Student test-fold assignment differs")
        require(np.array_equal(scores[0], data["L_scores"]) and np.array_equal(actions["Direct"], data["L_scores"] > 0) and result["all_direct_original_test_logits_exact"] is True, "Direct test scores/actions differ from original M6")
        require(protocol["identity"] == dict(queries=9600, groups=9559, partitions=partitions), "Frozen Teacher partition inventory differs")
        expected_journal = [dict(stem=m["stem"], **m["record"]) for m in all_models]
        require(journal == expected_journal, "Formal solve journal differs from exact role/order/model records")
        require(len(all_models) == 45 and sum(m["kind"] == "hard" for m in all_models) == 35 and sum(m["kind"] == "soft" for m in all_models) == 10, "Wrong hard/soft formal solve composition")
        values = primary_values(gap, actions)
        point = means(values)
        samples, intervals = frequency_bootstrap(values, groups)
        with np.load(output / "bootstrap.npz", allow_pickle=False) as z:
            require(set(z.files) == {"draws"}, "Unexpected bootstrap archive members")
            compare(real_array(z["draws"], (20000, 5), "stored bootstrap draws", np.float64), samples, "all_bootstrap_draws", errors)
        for j, name in enumerate(PRIMARY):
            compare(result["primary"][name]["mean"], point[j], "primary_mean", errors)
            compare(result["primary"][name]["interval"], intervals[j], "primary_interval", errors)
        actions.update(Dense=np.zeros(9600, bool), BM25=np.ones(9600, bool))
        summaries = {}
        for name in POLICIES:
            summaries[name] = policy_summary(utility, actions[name])
            require(set(result["policy"][name]) == set(summaries[name]), "Policy fields differ")
            for key, value in summaries[name].items():
                if key.endswith("_count"):
                    require(type(result["policy"][name][key]) is int and result["policy"][name][key] == value, "Policy count differs")
                else:
                    compare(result["policy"][name][key], value, "policy_" + key, errors)
        for f, parts in enumerate(data["folds"]):
            observed = result["folds"][f]["primary_means"]
            require(set(observed) == set(PRIMARY), "Fold primary fields differ")
            compare([observed[k] for k in PRIMARY], means(values[parts["test"]]), "fold_primary_mean", errors)
        transfer, candidate, decision = gates(point, intervals)
        require(result["transfer_gate"] is transfer and result["candidate_preparation_gate"] is candidate and result["decision"] == decision, "Prespecified decision differs")
    require(all(sha(resolve(path)) == digest for path, digest in snapshots.items()), "Artifact changed during independent check")
    bindings(protocol["source_sha256"], "final source")
    bindings(protocol["input_sha256"], "final input")
    value = dict(status="passed_independent_student_transfer_checks", created_at_utc=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), protocol_sha256=protocol_sha, results_sha256=snapshots[str(output / "results.json")],
        artifact_sha256=snapshots, source_bindings_checked=len(sources), input_bindings_checked=len(inputs),
        queries=9600, groups=9559, teacher_training_scalers_checked=15, formal_Teacher_probe_cache_rows_checked=cache_rows,
        strict_outer_fit_target_rows_checked=target_rows, Teacher_target_group_exclusions_checked=15,
        formal_hard_Teacher_solutions_checked=30, formal_hard_Direct_solutions_checked=5,
        formal_soft_Student_solutions_checked=10, formal_solutions_checked=45, journal_solutions_checked=45,
        pilot_solutions_checked=5, pilot_hard_solutions_checked=3, pilot_soft_solutions_checked=2,
        Student_heads_checked=15, all_Direct_export_fit_cal_test_exact=True,
        all_pre_Teacher_original_inner_logits_exact=True, pre_Teacher_original_inner_logits_checked=30720,
        maximum_independent_gradient_inf=max(m["gradient_inf"] for m in all_models),
        maximum_pilot_gradient_inf=max(m["gradient_inf"] for m in pilot_models), maximum_absolute_errors=errors,
        native_score_consistency=native_counts,
        native_score_scope="Every saved Teacher target and Student fit/cal/test vector is linked to its own head and actual inputs by CPU nearest-even BF16 operand rounding, FP64 centers, FP32 accumulation gamma bounds, and final BF16 rounding bounds; two-branch probe outputs also allow FP32 addition. This checks numerical consistency under the frozen FP32-accumulation path, not exact GPU replay or every alternative CUDA kernel.",
        independent_native_GPU_replay=False, pilot_Student_native_score_scope="Pilot Student outputs were checked finite by the source-bound runner but not saved; this checker verifies their heads/objectives only. The source-bound 480 historical GPU controls are recorded as exact, not rerun here.",
        projection_scope="Only original M6/probe/identity/utility members loaded; no gold array or old Teacher predictions. All new Students use 384D M6 only.",
        primary={name:dict(mean=float(point[j]), interval=intervals[j].tolist()) for j,name in enumerate(PRIMARY)},
        policy=summaries, transfer_gate=transfer, candidate_preparation_gate=candidate, decision=decision,
        conditional_interval_scope="Fixed OOF predictions and group sampling; excludes refitting, Teacher estimation, historical selection, and new-source uncertainty.",
        runtime_available=False, core_goal_achieved=False, new_model_fits=0, new_paid_calls=0,
        elapsed_seconds=time.perf_counter()-started)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def self_test():
    """Pure synthetic arithmetic only; no model fit or historical input read."""
    passed, errors = [], {}
    def yes(name, condition):
        require(condition, "Synthetic check failed: " + name)
        passed.append(name)
    def rejects(name, function):
        try:
            function()
        except ValueError:
            passed.append(name)
        else:
            raise ValueError("Synthetic rejection failed: " + name)
    rng = np.random.default_rng(2026091509)
    x = rng.normal(size=(17, 4)).astype(np.float32)
    q = rng.uniform(.05, .95, size=17)
    weights = rng.uniform(.1, 2., size=17)
    weights[::6] = 0
    theta = np.asarray([.2, -.3, .1, -.15, .07])
    lam, h = .13, 1e-5
    def objective(point, target=q, ww=weights, xx=x):
        return loss_quantities(xx, target, ww, point[:-1], float(point[-1]), lam, hessian=True)
    actual = objective(theta)
    numeric = np.empty(5)
    numeric_hessian = np.empty((5, 5))
    for j in range(5):
        delta = np.zeros(5); delta[j] = h
        plus, minus = objective(theta + delta), objective(theta - delta)
        numeric[j] = (plus["loss"] - minus["loss"]) / (2 * h)
        numeric_hessian[:, j] = (plus["gradient"] - minus["gradient"]) / (2 * h)
    compare(actual["gradient"], numeric, "synthetic_gradient", errors, 2e-9)
    passed.append("soft_gradient_central_difference")
    compare(actual["hessian"], numeric_hessian, "synthetic_hessian", errors, 2e-9)
    passed.append("soft_hessian_central_difference")
    yes("hessian_bias_not_penalized", abs(actual["hessian"][-1, -1] -
        math.fsum(w * p * (1-p) for w,p in zip(normalized_active_weights(weights)[1],
        sigmoid(np.einsum('nd,d->n',x[weights>0].astype(float),theta[:-1],optimize=False)+theta[-1])))) < 2e-15)
    hard = (q > .5).astype(float)
    ah = objective(theta, hard)
    active, normalized = normalized_active_weights(weights)
    z = np.einsum("nd,d->n", x[active].astype(float), theta[:-1], optimize=False) + theta[-1]
    literal = math.fsum(float(w) * signed_softplus((1 - 2*int(hard[i]))*float(v)) for i,w,v in zip(active,normalized,z))
    yes("binary_target_signed_loss_equivalence", abs(ah["data_loss"]-literal)<2e-15)
    alternate = q[::-1].copy()
    mixed = objective(theta, .5*hard+.5*alternate)
    other = objective(theta, alternate)
    yes("soft_BCE_linear_in_target", abs(mixed["loss"]-.5*(ah["loss"]+other["loss"]))<2e-15)
    changed_x = x.astype(float); changed_x[weights==0] = 1e300
    zero = objective(theta, ww=weights, xx=changed_x)
    yes("zero_weight_rows_removed_before_products", np.array_equal(zero["gradient"],actual["gradient"]) and zero["loss"]==actual["loss"])
    scaled = objective(theta, ww=weights*1e307)
    yes("max_then_sum_weight_normalization", np.max(abs(scaled["gradient"]-actual["gradient"]))<2e-15 and abs(scaled["loss"]-actual["loss"])<2e-15)
    qq = np.asarray([.1,.3,.6,.9]); ww=np.asarray([1.,2.,3.,4.])
    _, nw = normalized_active_weights(ww)
    prior=math.fsum(float(w)*float(t) for w,t in zip(nw,qq))
    intercept=math.log(prior)-math.log1p(-prior)
    stationarity=loss_quantities(np.zeros((4,3)),qq,ww,np.zeros(3),intercept,.001)
    yes("analytic_intercept_only_soft_stationarity", stationarity["grad_inf"]<2e-16)
    logits=np.asarray([[-1000.,-1.,0.,1.,1000.],[3.,-2.,2.,-3.,0.]],np.float32)
    gap=np.asarray([.2,-.3,0.,1e-13,-1e-12])
    probability, targets=build_targets(gap,logits)
    yes("stable_sigmoid_extremes", probability[0,0]==0 and probability[0,-1]==1 and probability[0,2]==.5 and np.isfinite(probability).all())
    yes("three_arm_target_mixing", np.array_equal(targets[0],(gap>0).astype(float)) and np.array_equal(targets[1],.5*(gap>0)+.5*probability[0]) and np.array_equal(targets[2],.5*(gap>0)+.5*probability[1]))
    yes("alpha_zero_hard_degeneracy", np.array_equal(build_targets(gap,logits,0.)[1],np.tile((gap>0).astype(float),(3,1))))
    yes("tie_weight_boundary",np.array_equal(weights_for_gap(gap),np.asarray([.2,.3,0.,0.,0.])))
    rejects("invalid_soft_probability",lambda: objective(theta,np.full(17,1.01)))
    rejects("zero_weight_rejected",lambda: objective(theta,ww=np.zeros(17)))
    rejects("one_sided_soft_mass_rejected",lambda: objective(theta,np.ones(17)))
    rejects("complex_numeric_rejected",lambda: loss_quantities(x.astype(complex),q,weights,theta[:-1],theta[-1]))
    groups=np.asarray(["b","a","a","c","d","d","d"])
    vals=np.asarray([[.2,-.1],[.7,.2],[-.1,.3],[0.,.1],[-.8,.2],[.2,.5],[.1,-.5]])
    samples, intervals=frequency_bootstrap(vals,groups,draws=31,seed=19)
    names=sorted(set(groups)); explicit=[]; generator=np.random.default_rng(19)
    for _ in range(31):
        chosen=generator.integers(0,len(names),size=len(names))
        ids=[i for j in chosen for i,g in enumerate(groups) if g==names[int(j)]]
        explicit.append([math.fsum(float(vals[i,j]) for i in ids)/len(ids) for j in range(2)])
    compare(samples,explicit,"synthetic_bootstrap_draws",errors,2e-15)
    compare(intervals,np.quantile(explicit,QUANTILES,axis=0).T,"synthetic_bootstrap_quantiles",errors,2e-15)
    passed.append("frequency_bootstrap_equals_explicit_unequal_group_draws")
    yes("group_assignment_keeps_repeated_groups_together",len(set(inner_groups(groups,0)[groups=="d"]))==1)
    vals_point=np.asarray([0.,.002,.002,.01,.01]); ci=np.tile([.001,.02],(5,1))
    yes("both_gates_at_point_threshold",gates(vals_point,ci)[:2]==(True,True))
    bad=ci.copy();bad[1,0]=0
    yes("strict_positive_lower_bound",gates(vals_point,bad)[0] is False)
    bad_point=vals_point.copy();bad_point[3]=.009
    yes("absolute_gain_required_for_candidate",gates(bad_point,ci)[:2]==(True,False))
    rounding=bf16_round(np.asarray([1+2**-8,1+3*2**-8,-1-2**-8],np.float32))
    yes("BF16_nearest_even_ties",np.array_equal(rounding,np.asarray([1.,1+4*2**-8,-1.],np.float32)))
    yes("BF16_zero_dimensional_bias",bf16_round(np.asarray(.25)).shape==() and float(bf16_round(np.asarray(.25)))==.25)
    xx=rng.normal(size=(9,384)).astype(np.float32); pp=rng.normal(size=(9,50)).astype(np.float32)
    cc=rng.normal(scale=.1,size=434); bb=.13; counts={}
    c0,b0=branch_reference(xx,cc[:384],bb); c1,b1=branch_reference(pp,cc[384:])
    native_score_bound(xx,cc[:384],bb,bf16_round(c0),counts,"synthetic_single")
    passed.append("single_branch_head_to_score_envelope")
    native_score_bound(xx,cc,bb,(bf16_round(c0)+bf16_round(c1)).astype(np.float32),counts,"synthetic_probe",pp)
    passed.append("two_branch_head_to_score_envelope")
    rejects("mismatched_native_cache_rejected",lambda:native_score_bound(xx,cc[:384],bb,bf16_round(c0)+np.float32(16.),{},"corrupt"))
    utility=np.column_stack((np.asarray([.5,.5+1e-13,.5-.1]),np.full(3,.5)))
    summary=policy_summary(utility,np.ones(3,bool))
    yes("tol_counts_do_not_truncate_contribution",summary['zero_count']==2 and summary['harmful_count']==1 and summary['beneficial_mass']>0)
    return dict(status="passed_pure_synthetic_student_checker_tests",checks=passed,test_count=len(passed),
                maximum_absolute_errors=errors,real_data_reads=0,new_model_fits=0,GPU_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--protocol-sha")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if args.self_test:
        require(args.protocol_sha is None,"Self-test cannot evaluate a frozen run")
        value=self_test()
    else:
        require(isinstance(args.protocol_sha,str) and len(args.protocol_sha)==64,"Supply the frozen --protocol-sha")
        value=check(args.output,args.protocol_sha)
    if args.self_test:
        print(json.dumps(value,ensure_ascii=False,allow_nan=False))
    else:
        print(json.dumps({key:value[key] for key in ("status","formal_solutions_checked","maximum_independent_gradient_inf",
                         "primary","transfer_gate","candidate_preparation_gate","decision","elapsed_seconds")},allow_nan=False))


if __name__ == "__main__":
    main()
