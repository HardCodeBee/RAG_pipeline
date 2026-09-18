"""Independent checks of the matched M6/probe experiment; never fit a model.

This checker does not import the experiment runner, its feature helpers, or the
optimizer. Historical tensor files are deserialized on CPU without inference.
Scalar reconstruction, block-mapping validation, and frequency-based group
bootstrap provide separate numerical paths.
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
DEFAULT_OUTPUT = PROJECT / "outputs/router/hotpotqa_bd_router_v1/runs/m6_probe_readout_v1"
REGULARIZATIONS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
SHUFFLE_SEEDS = (2026091506, 2026091507)
ARMS = ("P", "S0", "S1")
PRIMARY = ("P_minus_L", "P_minus_S0", "P_minus_S1", "P_minus_Dense", "P_minus_BM25")
TIE_ATOL = 1e-12
GRADIENT_ATOL = 1e-8
SCALAR_ATOL = 2e-12
FEATURE_ATOL = 2e-7
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 2026091508
QUANTILES = (.005, .995)
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


def block_derangement(groups, query_ids, seed, context):
    """Rebuild the prescribed SHA ordering, size-bucket cycle, and query order."""
    groups, query_ids = np.asarray(groups), np.asarray(query_ids)
    require(groups.ndim == 1 and query_ids.shape == groups.shape, "Invalid mapping identity shapes")
    if len(groups) == 0:
        return np.empty(0, dtype=np.int64)
    groups = group_array(groups)
    require(all(isinstance(value, str) and bool(value) for value in query_ids)
            and len(set(query_ids)) == len(query_ids), "Invalid mapping query IDs")
    require(type(seed) is int and seed in SHUFFLE_SEEDS and isinstance(context, str) and bool(context),
            "Invalid shuffle seed or context")
    members = {}
    for row, group in enumerate(groups):
        members.setdefault(str(group), []).append(row)
    buckets = {}
    for group, rows in members.items():
        rows.sort(key=lambda row: str(query_ids[row]))
        buckets.setdefault(len(rows), []).append(group)
    donors = np.empty(len(groups), dtype=np.int64)
    for size, names in buckets.items():
        require(len(names) >= 2, "Cannot derange a singleton group-size bucket")
        def key(group):
            payload = json.dumps(["m6_probe_group_derangement_v1", seed, context, int(size), group],
                                 ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(payload).digest(), group
        order = sorted(names, key=key)
        for position, recipient in enumerate(order):
            donor = order[(position + 1) % len(order)]
            for target, source in zip(members[recipient], members[donor]):
                donors[target] = source
    validate_block_derangement(groups, donors)
    return donors


def validate_block_derangement(groups, donor_rows):
    """Check a partition-local row permutation as equal-size whole-group blocks."""
    groups = group_array(groups)
    donors = np.asarray(donor_rows)
    n = len(groups)
    require(donors.shape == (n,) and donors.dtype.kind in "iu", "Donor map must be an integer row vector")
    require(np.array_equal(np.sort(donors), np.arange(n)), "Donor map is not a local row permutation")
    members = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    group_map = {}
    for recipient, rows in members.items():
        donated = donors[rows]
        donor_groups = set(groups[donated])
        require(len(donor_groups) == 1, "A recipient group mixes donor groups")
        donor = next(iter(donor_groups))
        require(donor != recipient, "Self-mapped group")
        require(len(members[donor]) == len(rows), "Donor and recipient group sizes differ")
        require(set(map(int, donated)) == set(members[donor]), "Donor block is incomplete")
        group_map[str(recipient)] = str(donor)
    require(len(set(group_map.values())) == len(group_map), "Donor group used more than once")
    return {"rows": n, "groups": len(group_map), "self_mapped_groups": 0,
            "whole_group_blocks": True, "same_size_blocks": True}


def active_labels_weights(gap):
    gap = real_array(gap, name="utility gap").astype(np.float64)
    require(gap.ndim == 1 and len(gap) > 0 and np.all(np.abs(gap) <= 1), "Expected nonempty gap in [-1,1]")
    active = np.flatnonzero(np.abs(gap) > TIE_ATOL)
    require(len(active) > 0, "Positive total weight is required")
    selected = gap[active]
    total = math.fsum(abs(float(value)) for value in selected)
    weights = np.asarray([abs(float(value)) / total for value in selected])
    return active, selected > 0, weights


def signed_softplus(value):
    value = float(value)
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def weighted_bce(scores, gap):
    scores = real_array(scores, (len(gap),), "BCE scores")
    active, labels, weights = active_labels_weights(gap)
    return math.fsum(float(weight) * signed_softplus(-float(scores[index]) if label else float(scores[index]))
                     for index, label, weight in zip(active, labels, weights))


def bce_quantities(features, gap, coef, intercept, regularization):
    """Independent saved-solution objective and full, unpenalized-bias gradient."""
    features = real_array(features, name="BCE features")
    require(features.ndim == 2 and len(features) == len(gap), "Invalid BCE feature shape")
    coef = real_array(coef, (features.shape[1],), "BCE coefficients").astype(np.float64)
    require(type(intercept) in (float, int) and math.isfinite(intercept), "Invalid intercept")
    require(type(regularization) in (float, int) and math.isfinite(regularization) and regularization > 0,
            "Invalid regularization")
    active, labels, weights = active_labels_weights(gap)
    require(labels.any() and (~labels).any(), "Both labels need positive weight")
    selected = features[active].astype(np.float64)
    logits = np.einsum("nd,d->n", selected, coef, optimize=False) + intercept
    require(np.isfinite(logits).all(), "Nonfinite BCE logits")
    data_loss = math.fsum(float(weight) * signed_softplus(-float(logit) if label else float(logit))
                          for weight, label, logit in zip(weights, labels, logits))
    penalty = regularization * math.fsum(float(value) ** 2 for value in coef) / 2
    residual = np.empty(len(logits), dtype=np.float64)
    for index, (logit, label, weight) in enumerate(zip(logits, labels, weights)):
        value = float(logit)
        if value >= 0:
            probability = 1 / (1 + math.exp(-value))
        else:
            exponent = math.exp(value)
            probability = exponent / (1 + exponent)
        residual[index] = float(weight) * (probability - int(label))
    coef_gradient = np.einsum("nd,n->d", selected, residual, optimize=False) + regularization * coef
    gradient = np.r_[coef_gradient, math.fsum(map(float, residual))]
    require(np.isfinite(gradient).all(), "Nonfinite BCE gradient")
    return {"data_loss": data_loss, "penalty": penalty, "loss": data_loss + penalty,
            "grad_inf": float(np.abs(gradient).max()), "gradient": gradient,
            "positive_weight_rows": len(active)}


def select_lambda(cv_native, gap):
    """Select each arm separately from its five native-score vectors."""
    cv_native = real_array(cv_native, (5, len(gap)), "native inner scores", np.float32)
    losses = [weighted_bce(scores, gap) for scores in cv_native]
    best = min(losses)
    selected = next(index for index, value in enumerate(losses) if value <= best + TIE_ATOL)
    return {"selected_lambda_index": selected, "cv_bce": losses}


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


def contributions(gap, actions):
    gap = real_array(gap, name="contribution gap")
    require(gap.ndim == 1 and len(gap) > 0, "Invalid contribution gap shape")
    arrays = {}
    for name in ("P", "L", "S0", "S1"):
        action = np.asarray(actions[name])
        require(action.shape == gap.shape and action.dtype == bool, f"Invalid {name} actions")
        arrays[name] = action.astype(np.int64)
    p = arrays["P"]
    return np.column_stack(((p - arrays["L"]) * gap, (p - arrays["S0"]) * gap,
                            (p - arrays["S1"]) * gap, p * gap, (p - 1) * gap))


def means(values):
    values = real_array(values, name="mean values")
    require(values.ndim == 2 and len(values) > 0, "Invalid mean values shape")
    return np.asarray([math.fsum(map(float, values[:, column])) / len(values)
                       for column in range(values.shape[1])])


def frequency_intervals(values, groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Same whole-group draws, reconstructed through multiplicity counts."""
    values = real_array(values, name="bootstrap values")
    require(values.ndim == 2 and len(values) > 0, "Invalid bootstrap values")
    groups = group_array(groups, len(values))
    require(type(draws) is int and draws > 0, "Invalid bootstrap draw count")
    members = {}
    for row, group in enumerate(groups):
        members.setdefault(group, []).append(row)
    names = sorted(members)
    totals = np.asarray([[math.fsum(float(values[i, column]) for i in members[group])
                          for column in range(values.shape[1])] for group in names])
    sizes = np.asarray([len(members[group]) for group in names], dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, values.shape[1]), dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(names), size=len(names))
        counts = np.bincount(sampled, minlength=len(names))
        used = np.flatnonzero(counts)
        denominator = int(counts[used] @ sizes[used])
        samples[draw] = np.einsum("i,ij->j", counts[used], totals[used], optimize=False) / denominator
    return np.quantile(samples, QUANTILES, axis=0, method="linear")


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
                    RESEARCH / "e04_protocol.json"]


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
    assignments, old_fit, old_coefs, old_intercepts = [], [], [], []
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
        head = torch.load(directory / f"fold{fold}_M6_head.pt", map_location="cpu", weights_only=True)
        old_coefs.append(real_array(head["weight"].numpy(), (1, 384), "historical head coefficient", np.float32).reshape(384).copy())
        old_intercepts.append(float(real_array(head["bias"].numpy(), (1,), "historical head bias", np.float32)[0]))
    require(np.all(coverage == 1), "Historical OOF coverage differs")
    return {"query_ids": qids, "group_ids": groups, "features": features, "probe": probe,
            "utility": utility, "gap": utility[:, 0] - utility[:, 1], "L_scores": linear,
            "folds": folds, "inner_assignments": assignments, "L_fit": old_fit, "L_cal": old_cal,
            "L_coef": old_coefs, "L_intercept": old_intercepts}


def partition_cache(archive, part, indices, scaler, data, context, errors):
    """Check the P cache and both exact donor maps; return actual stored FP32."""
    cached = real_array(archive[f"probe_{part}"], (len(indices), 50), f"{context} probe", np.float32).copy()
    expected = standardized_probe(data["probe"][indices], scaler["mean"], scaler["std"])
    difference = np.abs(cached.astype(float) - expected.astype(float))
    # At most two final FP32 ulps plus a small absolute tolerance; gradients below
    # always use the actual cache, not these independently rounded values.
    allowance = FEATURE_ATOL + 2 * np.spacing(np.abs(expected)).astype(float)
    require(np.all(difference <= allowance), f"Independent probe transform differs: {context}")
    errors["probe_transform"] = max(errors.get("probe_transform", 0.), float(difference.max()) if difference.size else 0.)
    donors = real_array(archive[f"donors_{part}"], (3, len(indices)), f"{context} donors", np.int64).copy()
    require(np.array_equal(donors[0], np.arange(len(indices))), f"P row identities changed: {context}")
    for arm, seed in enumerate(SHUFFLE_SEEDS, 1):
        expected_map = block_derangement(data["group_ids"][indices], data["query_ids"][indices], seed, context)
        require(np.array_equal(donors[arm], expected_map), f"Exact group derangement differs: {context}/{arm}")
    return cached, donors


def context_cache(path, train, valid, data, train_context, valid_context, errors):
    with np.load(path, allow_pickle=False) as archive:
        require(np.array_equal(archive["train_indices"], train) and np.array_equal(archive["valid_indices"], valid),
                f"Wrong context rows: {train_context}")
        active_rows = train[np.abs(data["gap"][train]) > TIE_ATOL]
        require(np.array_equal(archive["active_indices"], active_rows), f"Wrong active rows: {train_context}")
        expected = reconstruct_scaler(data["probe"][train])
        mean = real_array(archive["mean"], (50,), "training mean", np.float64).copy()
        std = real_array(archive["std"], (50,), "training std", np.float64).copy()
        compare(mean, expected["mean"], "probe_training_mean", errors, atol=1e-10)
        compare(std, expected["std"], "probe_training_std", errors, atol=1e-10)
        active = np.asarray(archive["active"])
        require(active.shape == (50,) and active.dtype == bool and np.array_equal(active, expected["active"]),
                f"Wrong active feature mask: {train_context}")
        require(np.array_equal(active, std > TIE_ATOL), "Saved std and active feature mask disagree")
        stored_scaler = {"mean": mean, "std": std, "active": active}
        probe_train, donors_train = partition_cache(archive, "train", train, expected, data, train_context, errors)
        probe_valid, donors_valid = partition_cache(archive, "valid", valid, expected, data, valid_context, errors)
    return {"train": train, "valid": valid, "active_rows": active_rows, "scaler": stored_scaler,
            "independent_scaler": expected, "probe_train": probe_train, "donors_train": donors_train,
            "probe_valid": probe_valid, "donors_valid": donors_valid}


def checked_solution(record, features, gap, coef, intercept, regularization, errors):
    require(record["accepted"] is True and record["optimizer_success"] is True
            and record["status"] == "accepted_stationary_solution"
            and record["acceptance_gradient_inf"] == GRADIENT_ATOL, "Unaccepted saved solution")
    require(record["regularization"] == regularization, "Saved regularization differs")
    compare(record["intercept"], intercept, "BCE_intercept", errors)
    independent = bce_quantities(features, gap, coef, float(intercept), regularization)
    require(independent["grad_inf"] <= GRADIENT_ATOL, "Independent stationarity acceptance failed")
    require(record["positive_weight_rows"] == independent["positive_weight_rows"], "Positive-weight row count differs")
    for key in ("data_loss", "penalty", "loss", "grad_inf"):
        compare(record[key], independent[key], "BCE_" + key, errors,
                atol=1e-10 if key == "grad_inf" else SCALAR_ATOL)
    return independent["grad_inf"]


def expected_config():
    return {"queries": 9600, "groups": 9559, "folds": 5, "fit_queries": 6144,
        "cal_queries": 1536, "test_queries": 1920, "inner_folds": 3,
        "regularizations": list(REGULARIZATIONS), "arms": list(ARMS), "shuffle_seeds": list(SHUFFLE_SEEDS),
        "linear_dimensions": 384, "probe_dimensions": 50, "dimensions": 434, "std_ddof": 0,
        "std_atol": TIE_ATOL, "probe_normalizer": "sqrt(50)", "tie_atol": TIE_ATOL,
        "gradient_atol": GRADIENT_ATOL, "batch_size": 8, "formal_solves": 240, "pilot_solves": 3,
        "final_heads": 15, "new_thresholds": 0, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED, "interval_quantiles": list(QUANTILES),
        "minimum_diagnostic_increment": .002, "cpu_threads": 2, "new_encoder_forwards": 0,
        "new_external_calls": 0, "runtime_available": False}


def transfer_gate(point_means, intervals):
    point_means = real_array(point_means, (5,), "primary means")
    intervals = real_array(intervals, (2, 5), "primary intervals")
    return bool(np.all(intervals[0, :3] > 0) and np.all(point_means[:3] >= .002))


def check(output_dir):
    """Verify complete artifacts, then exclusively create separate_checks.json."""
    output_dir = Path(output_dir).resolve()
    destination = output_dir / "separate_checks.json"
    require(not destination.exists(), "Preserve existing separate_checks.json")
    fit_files = [output_dir / f"fold{fold}_context{k}.npz" for fold in range(5) for k in range(4)]
    fit_files += [output_dir / f"fold{fold}.{suffix}" for fold in range(5) for suffix in ("npz", "json")]
    fit_files.append(output_dir / "solutions.jsonl")
    pilot_files = [output_dir / "pilot_context.npz"] + [output_dir / f"pilot_{arm}.{suffix}"
                  for arm in ARMS for suffix in ("npz", "json")]
    test_files = [output_dir / f"fold{fold}_test.npz" for fold in range(5)]
    required = fit_files + pilot_files + test_files + [output_dir / name for name in (
        "protocol.json", "pilot.json", "fit_started.json", "fit_completion.json", "predictions.npz",
        "predictions_frozen.json", "results.json")]
    require(all(path.is_file() for path in required), "Wait for the complete formal experiment")
    started = time.perf_counter()
    protocol = read(output_dir / "protocol.json")
    require(protocol["status"] == "frozen_before_real_pilot_and_formal_probe_fits", "Wrong protocol status")
    require(protocol["config"] == expected_config(), "Frozen configuration differs")
    policies = ("P", "L", "S0", "S1", "Dense", "BM25")
    require(protocol["primary"] == list(PRIMARY) and protocol["policies"] == list(policies), "Contrasts or policies differ")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "threadpoolctl")}
    require(protocol["versions"] == versions, "Bound software versions differ")
    sources, inputs = bindings(protocol["source_sha256"], "source"), bindings(protocol["input_sha256"], "input")
    expected_sources = [Path(__file__), PROJECT / "scripts/run_m6_probe_readout.py", PROJECT / "scripts/m6_probe_math.py",
        PROJECT / "analysis/hotpotqa_router/m6_probe_readout_plan_20260915.md", PROJECT / "scripts/run_m6_objective_readout.py",
        PROJECT / "scripts/m6_objective_math.py", RESEARCH / "weighted_linear_probe.py",
        RESEARCH / "weighted_linear_probe_refined.py", PROJECT / "scripts/run_router_phase27_privileged_teacher.py"]
    require(set(sources) == {path.resolve() for path in expected_sources}, "Frozen source set differs")
    require(set(inputs) == {path.resolve() for path in historical_input_paths()}, "Frozen input set differs")
    snapshots = {str(path): sha(path) for path in required}
    protocol_hash = snapshots[str(output_dir / "protocol.json")]
    result = read(output_dir / "results.json")
    require(result["status"] == "complete_fixed_probe_comparison_pending_separate_check"
            and result["protocol_sha256"] == protocol_hash, "Wait for completed bound probe results")
    for key, value in (("formal_solves", 240), ("final_heads", 15), ("new_thresholds", 0),
                       ("new_encoder_forwards", 0), ("new_external_calls", 0)):
        require(type(result[key]) is int and result[key] == value, "Invalid completion count: " + key)
    require(result["runtime_available"] is False and result["candidate_preparation_gate"] is False
            and result["core_goal_achieved"] is False and result["scope"] == protocol["scope"],
            "Post-retrieval diagnostic cannot become a deployed or independently confirmed candidate")
    require(set(result["primary"]) == set(PRIMARY) and set(result["policy"]) == set(policies)
            and len(result["folds"]) == 5, "Result field sets differ")
    fit_completion = read(output_dir / "fit_completion.json")
    require(fit_completion["status"] == "all_240_solves_and_15_heads_fixed_before_new_test_predictions"
            and fit_completion["protocol_sha256"] == protocol_hash, "Wrong pre-test completion record")
    require(fit_completion["formal_solves"] == 240 and fit_completion["final_heads"] == 15
            and fit_completion["new_thresholds"] == 0, "Wrong pre-test completion counts")
    require(set(bindings(fit_completion["artifact_sha256"], "pre-test fit")) == set(fit_files), "Wrong pre-test fit file set")
    require(read(output_dir / "fit_started.json")["protocol_sha256"] == protocol_hash, "Fit-start binding differs")
    frozen = read(output_dir / "predictions_frozen.json")
    require(frozen["status"] == "complete_9600_OOF_actions_before_effects"
            and frozen["protocol_sha256"] == protocol_hash, "Wrong prediction freeze record")
    require(frozen["predictions_sha256"] == snapshots[str(output_dir / "predictions.npz")]
            and frozen["fit_completion_sha256"] == snapshots[str(output_dir / "fit_completion.json")], "Prediction bindings differ")
    require(set(bindings(frozen["test_artifact_sha256"], "test cache")) == set(test_files), "Wrong test cache file set")
    pilot = read(output_dir / "pilot.json")
    require(pilot["status"] == "passed_three_real_pilot_solves_and_480_zero_probe_native_controls"
            and pilot["protocol_sha256"] == protocol_hash, "Wrong pilot record")
    require(pilot["pilot_rows"] == 128 and pilot["pilot_solves"] == 3
            and pilot["original_native_control_queries"] == 480 and pilot["new_policy_effects"] is False,
            "Wrong engineering pilot scope")
    require(len(pilot["original_controls"]) == 15 and all(value == 0 for value in pilot["original_controls"].values()),
            "Zero-probe controls did not pass")
    require(set(bindings(pilot["artifact_sha256"], "pilot")) == set(pilot_files), "Wrong pilot file set")
    journal = [json.loads(line) for line in (output_dir / "solutions.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(journal) == 240, "Expected exactly 240 journal solutions")
    errors, independent_gradients, selections, partition_counts = {}, [], [], {}
    data = load_original_inputs()
    qids, groups, x, gap, utility = (data[key] for key in ("query_ids", "group_ids", "features", "gap", "utility"))
    global_members = {}
    for group in groups:
        global_members[group] = global_members.get(group, 0) + 1
    pilot_train = np.asarray([row for row in data["folds"][0]["fit"] if global_members[groups[row]] == 1][:128])
    require(len(pilot_train) == 128, "Pilot singleton-row scope differs")
    with threadpool_limits(limits=2):
        pilot_context = context_cache(output_dir / "pilot_context.npz", pilot_train, np.empty(0, np.int64),
                                      data, "pilot/train", "pilot/unused", errors)
        pilot_gradients = []
        for arm, name in enumerate(ARMS):
            record = read(output_dir / f"pilot_{name}.json")
            with np.load(output_dir / f"pilot_{name}.npz", allow_pickle=False) as archive:
                coef = real_array(archive["coef"], (434,), "pilot coefficient", np.float64)
                intercept = float(real_array(archive["intercept"], (), "pilot bias", np.float64))
                design = np.concatenate((x[pilot_train], pilot_context["probe_train"][pilot_context["donors_train"][arm]]), axis=1)
                pilot_gradients.append(checked_solution(record, design, gap[pilot_train], coef, intercept, .001, errors))
        compare(pilot["pilot_gradients"], pilot_gradients, "pilot_gradients", errors, atol=1e-10)
        with np.load(output_dir / "predictions.npz", allow_pickle=False) as prediction:
            require(np.array_equal(prediction["query_ids"], qids) and np.array_equal(prediction["group_ids"], groups),
                    "Prediction row identities differ")
            require(np.array_equal(prediction["utility"], utility) and np.array_equal(prediction["L_scores"], data["L_scores"]),
                    "Historical prediction inputs changed")
            arm_scores = real_array(prediction["arm_scores"], (3, 9600), "native arm scores", np.float32).copy()
            expected_fold = np.full(9600, -1, dtype=np.int64)
            actions = {name: arm_scores[arm] > 0 for arm, name in enumerate(ARMS)}
            actions.update(L=data["L_scores"] > 0, Dense=np.zeros(9600, bool), BM25=np.ones(9600, bool))
            for name in ("P", "L", "S0", "S1"):
                require(prediction[name].dtype == bool and np.array_equal(prediction[name], actions[name]),
                        "Native zero-boundary decisions differ: " + name)
            cache_rows = 0
            for fold, parts in enumerate(data["folds"]):
                fit, cal, test = (parts[key] for key in ("fit", "calibration", "test"))
                expected_fold[test] = fold
                assigned = data["inner_assignments"][fold]
                contexts = []
                for k in range(4):
                    train = fit[assigned != k] if k < 3 else fit
                    valid = fit[assigned == k] if k < 3 else np.empty(0, np.int64)
                    train_name = f"fold{fold}/inner{k}/train" if k < 3 else f"fold{fold}/fit"
                    valid_name = f"fold{fold}/inner{k}/valid"
                    ctx = context_cache(output_dir / f"fold{fold}_context{k}.npz", train, valid,
                                        data, train_name, valid_name, errors)
                    contexts.append(ctx)
                    cache_rows += len(train) + len(valid)
                    for name, rows, donors in ((train_name, train, ctx["donors_train"]), (valid_name, valid, ctx["donors_valid"])):
                        if len(rows):
                            partition_counts[name] = partition_identity(groups[rows], donors)
                info = read(output_dir / f"fold{fold}.json")
                require(info["fold"] == fold and len(info["solves"]) == 48, "Wrong fold solution record")
                with np.load(output_dir / f"fold{fold}.npz", allow_pickle=False) as archive:
                    coef = real_array(archive["coef"], (48, 434), "formal coefficient", np.float64).copy()
                    intercept = real_array(archive["intercept"], (48,), "formal bias", np.float64).copy()
                    require(np.array_equal(archive["inner_assignment"], assigned), "Saved inner grouping differs")
                    cv = real_array(archive["cv_native"], (3, 5, 6144), "native inner scores", np.float32)
                    fit_native = real_array(archive["fit_native"], (3, 6144), "native fit scores", np.float32).copy()
                    cal_native = real_array(archive["cal_native"], (3, 1536), "native cal scores", np.float32).copy()
                    selected = [select_lambda(cv[arm], gap[fit]) for arm in range(3)]
                    chosen = [item["selected_lambda_index"] for item in selected]
                    require(info["selected_lambda_index"] == chosen, "Independent per-arm lambda selection differs")
                    compare(info["cv_weighted_BCE"], [item["cv_bce"] for item in selected], "cv_weighted_BCE", errors)
                    selections.append([REGULARIZATIONS[index] for index in chosen])
                    _probe, cal_donors = partition_cache(archive, "cal", cal, contexts[3]["independent_scaler"],
                                                       data, f"fold{fold}/cal", errors)
                    partition_counts[f"fold{fold}/cal"] = partition_identity(groups[cal], cal_donors)
                    cache_rows += len(cal)
                # Use the actual cached FP32 design used by each fit, after the
                # separate scaler/map checks; do not substitute reconstructed rows.
                designs = [[np.concatenate((x[ctx["train"]], ctx["probe_train"][ctx["donors_train"][arm]]), axis=1)
                            for arm in range(3)] for ctx in contexts]
                for index, record in enumerate(info["solves"]):
                    if index < 45:
                        lambda_index, remainder = divmod(index, 9)
                        k, arm = divmod(remainder, 3)
                    else:
                        arm, k = index - 45, 3
                        lambda_index = chosen[arm]
                    ctx = contexts[k]
                    identity = {"arm_index": arm, "arm": ARMS[arm], "lambda_index": lambda_index,
                        "context": k, "inner_split": k if k < 3 else None, "solution_index": index,
                        "training_queries": len(ctx["train"]), "role": "inner" if k < 3 else "refit"}
                    require(all(record[key] == value for key, value in identity.items()), "Saved solution identity differs")
                    row = journal[48 * fold + index]
                    require(row["fold"] == fold and {key: value for key, value in row.items() if key not in ("fold", "coef")} == record,
                            "Journal and fold solver records differ")
                    require(np.array_equal(np.asarray(row["coef"], dtype=np.float64), coef[index]), "Journal coefficient differs")
                    independent_gradients.append(checked_solution(record, designs[k][arm], gap[ctx["train"]], coef[index],
                                                  float(intercept[index]), REGULARIZATIONS[lambda_index], errors))
                for role, native, indices in (("fit", fit_native, fit), ("cal", cal_native, cal)):
                    compare(info[f"{role}_BCE"], [weighted_bce(row, gap[indices]) for row in native], f"{role}_BCE", errors)
                    compare(info[f"L_{role}_BCE"], weighted_bce(data[f"L_{role}"][fold], gap[indices]), f"L_{role}_BCE", errors)
                old_coef = data["L_coef"][fold].astype(np.float64)
                feasible_scores = np.einsum("nd,d->n", x[fit].astype(float), old_coef, optimize=False) + data["L_intercept"][fold]
                feasible_data = weighted_bce(feasible_scores, gap[fit])
                squared_norm = math.fsum(float(value) ** 2 for value in old_coef)
                feasible = [feasible_data + REGULARIZATIONS[index] * squared_norm / 2 for index in chosen]
                compare(info["linear_feasible_objective"], feasible, "linear_feasible_objective", errors)
                require(all(info["solves"][45 + arm]["loss"] <= feasible[arm] + 1e-8 for arm in range(3)),
                        "Fitted objective exceeds the old linear feasible point")
                with np.load(output_dir / f"fold{fold}_test.npz", allow_pickle=False) as archive:
                    require(np.array_equal(archive["test_indices"], test), "Test cache row order differs")
                    _probe, test_donors = partition_cache(archive, "test", test, contexts[3]["independent_scaler"],
                                                        data, f"fold{fold}/test", errors)
                    require(np.array_equal(archive["native_scores"], arm_scores[:, test]), "Test native cache differs")
                    partition_counts[f"fold{fold}/test"] = partition_identity(groups[test], test_donors)
                    cache_rows += len(test)
                reported = result["folds"][fold]
                require(all(reported[key] == value for key, value in info.items()), "Fitted fold record changed in results")
                compare(reported["test_BCE"], [weighted_bce(row[test], gap[test]) for row in arm_scores], "test_BCE", errors)
                compare(reported["L_test_BCE"], weighted_bce(data["L_scores"][test], gap[test]), "L_test_BCE", errors)
                print(json.dumps({"status": "independent_probe_fold_checked", "fold": fold,
                                  "formal_solutions_checked": len(independent_gradients)}), flush=True)
            require(np.array_equal(prediction["fold_id"], expected_fold), "Outer test identities differ")
        identity = protocol["identity"]
        require(identity["queries"] == 9600 and identity["groups"] == 9559 and identity["probe_shape"] == [9600, 50]
                and identity["partitions"] == partition_counts, "Frozen partition metadata differs")
        for name in policies:
            expected = policy_summary(utility, actions[name])
            require(set(result["policy"][name]) == set(expected), "Policy scalar set differs")
            for key, value in expected.items():
                if key.endswith("count"):
                    require(type(result["policy"][name][key]) is int and result["policy"][name][key] == value,
                            f"Policy count differs: {name}.{key}")
                else:
                    compare(result["policy"][name][key], value, "policy_" + key, errors)
        values = contributions(gap, actions)
        main_means, intervals = means(values), frequency_intervals(values, groups)
        checked_primary = {}
        for column, name in enumerate(PRIMARY):
            compare(result["primary"][name]["mean"], main_means[column], "primary_mean", errors)
            compare(result["primary"][name]["interval"], intervals[:, column], "primary_interval", errors)
            checked_primary[name] = {"mean": float(main_means[column]), "interval": intervals[:, column].tolist()}
        for fold, parts in enumerate(data["folds"]):
            fold_means = means(values[parts["test"]])
            for column, name in enumerate(PRIMARY):
                compare(result["folds"][fold]["primary_means"][name], fold_means[column], "fold_primary_mean", errors)
        transfer = transfer_gate(main_means, intervals)
        require(result["transfer_investigation_gate"] is transfer, "Predefined transfer-investigation decision differs")
    bindings(protocol["source_sha256"], "source")
    bindings(protocol["input_sha256"], "input")
    require(all(sha(path) == digest for path, digest in snapshots.items()), "Artifacts changed during checking")
    report = {"status": "passed_independent_probe_readout_checks", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checker_sha256": sha(__file__), "protocol_sha256": protocol_hash,
        "results_sha256": snapshots[str(output_dir / "results.json")], "artifact_sha256": snapshots,
        "source_bindings_checked": len(sources), "input_bindings_checked": len(inputs), "queries": 9600, "groups": 9559,
        "probe_fields_checked": 50, "training_scalers_checked": 20, "partitions_checked": len(partition_counts),
        "formal_group_derangements_checked": 2 * len(partition_counts), "probe_cache_rows_checked": cache_rows,
        "pilot_solutions_checked": 3, "formal_solutions_checked": 240, "journal_solutions_checked": 240,
        "final_heads_checked": 15, "selected_regularizations": selections,
        "maximum_independent_gradient_inf": max(independent_gradients), "maximum_absolute_errors": errors,
        "primary": checked_primary, "transfer_investigation_gate": transfer, "candidate_preparation_gate": False,
        "runtime_available": False, "independent_native_GPU_replay": False,
        "native_score_scope": "Saved FP32 two-branch native scores are checked as numeric inputs; the checker performs no GPU replay. The exact native implementation is source-bound and the real pilot records 480 zero-probe controls.",
        "training_score_scope": "Every actual FP32 probe cache and exact donor map is independently reconstructed. All saved-solution gradients use the actual cached FP32 training design.",
        "linear_feasible_scope": "The five bound historical FP32 heads are deserialized on CPU. Their objective at each selected lambda is a fixed feasible point, not a newly optimized linear baseline.",
        "new_model_fits": 0, "new_paid_calls": 0,
        "scope": "Independent numerical implementation on consumed old9600; no independent-source confirmation, conditional-independence test, or deployable policy.",
        "elapsed_seconds": time.perf_counter() - started}
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(destination),
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


def partition_identity(groups, donors):
    members = {}
    for group in groups:
        members[group] = members.get(group, 0) + 1
    sizes = {}
    for size in members.values():
        sizes[str(size)] = sizes.get(str(size), 0) + 1
    return {"queries": len(groups), "groups": len(members), "group_size_counts": sizes,
            "controls_differing_donors": int(np.sum(donors[1] != donors[2]))}


def self_test():
    """Small synthetic checks of independent math, without fitting or file I/O."""
    errors, tests = {}, []
    rng = np.random.default_rng(909)
    probe = rng.normal(size=(12, 50))
    probe[:, 0] = 3.25
    probe[:, 1] *= 1e-13
    scaler = reconstruct_scaler(probe)
    compare(scaler["mean"], probe.mean(axis=0), "synthetic_scaler_mean", errors)
    compare(scaler["std"], probe.std(axis=0, ddof=0), "synthetic_scaler_std", errors)
    transformed = standardized_probe(probe, scaler["mean"], scaler["std"])
    expected = np.zeros(probe.shape, np.float32)
    active = probe.std(axis=0, ddof=0) > TIE_ATOL
    expected[:, active] = ((probe[:, active] - probe.mean(axis=0)[active]) /
                           probe.std(axis=0, ddof=0)[active] / np.sqrt(50)).astype(np.float32)
    compare(transformed, expected, "synthetic_transform", errors, atol=FEATURE_ATOL)
    heldout = probe[:3].copy()
    heldout[:, :2] = 1e6
    require(np.all(standardized_probe(heldout, scaler["mean"], scaler["std"])[:, :2] == 0),
            "Inactive training columns must remain zero on heldout rows")
    require(standardized_probe(probe[:0], scaler["mean"], scaler["std"]).shape == (0, 50),
            "Empty unused partition shape differs")
    tests.append("training_population_scaler_inactive_heldout_columns_and_empty_partition")

    groups = np.asarray(["g0", "g0", "g1", "g1", "g2", "g3"])
    qids = np.asarray(["b", "a", "c", "d", "e", "f"])
    for seed in SHUFFLE_SEEDS:
        donors = block_derangement(groups, qids, seed, "synthetic/train")
        require(np.array_equal(donors, [3, 2, 1, 0, 5, 4]), "Whole-group cycle/query ordering differs")
        validate_block_derangement(groups, donors)
    for bad in (np.arange(6), np.asarray([2, 4, 0, 1, 5, 3]), np.asarray([2, 3, 0, 1, 5, 5])):
        try:
            validate_block_derangement(groups, bad)
        except ValueError:
            pass
        else:
            raise ValueError("Invalid block map was accepted")
    try:
        block_derangement(["a", "a", "b", "c"], ["q0", "q1", "q2", "q3"], SHUFFLE_SEEDS[0], "synthetic/train")
    except ValueError:
        pass
    else:
        raise ValueError("Singleton size bucket was accepted")
    tests.append("group_derangement_query_order_bijection_self_mapping_and_singleton_refusal")

    x = rng.normal(size=(7, 3))
    gap = np.asarray([.7, -.2, 0., .3, -.8, 1e-13, -1e-12])
    point = np.asarray([.2, -.1, .3, .15])
    regularization = .03
    observed = bce_quantities(x, gap, point[:-1], float(point[-1]), regularization)
    literal_weights = [abs(float(value)) if abs(value) > TIE_ATOL else 0. for value in gap]
    total_weight = math.fsum(literal_weights)
    def literal_objective(parameters):
        values = []
        for row, label_gap, weight in zip(x, gap, literal_weights):
            logit = math.fsum(float(a) * float(b) for a, b in zip(row, parameters[:-1])) + float(parameters[-1])
            signed = -logit if label_gap > 0 else logit
            values.append(weight * math.log1p(math.exp(signed)))
        return math.fsum(values) / total_weight + regularization * math.fsum(float(v) ** 2 for v in parameters[:-1]) / 2
    compare(observed["loss"], literal_objective(point), "synthetic_BCE_loss", errors)
    numerical = np.empty(4)
    for column in range(4):
        left, right = point.copy(), point.copy()
        left[column] -= 1e-6
        right[column] += 1e-6
        numerical[column] = (literal_objective(right) - literal_objective(left)) / 2e-6
    compare(observed["gradient"], numerical, "synthetic_central_difference_gradient", errors, atol=2e-9)
    require(observed["positive_weight_rows"] == 4, "Near-zero gap weighting differs")
    expanded = np.column_stack((x, rng.normal(size=(7, 4))))
    nested = bce_quantities(expanded, gap, np.r_[point[:-1], np.zeros(4)], float(point[-1]), regularization)
    compare(nested["loss"], observed["loss"], "synthetic_linear_nesting", errors)
    compare(nested["penalty"], observed["penalty"], "synthetic_unpenalized_bias", errors)
    tests.append("independent_literal_BCE_finite_difference_gradient_zero_weight_and_linear_nesting")

    cv = np.zeros((5, 7), dtype=np.float32)
    cv[1] = np.where(gap > 0, 1., -1.)
    require(select_lambda(cv, gap)["selected_lambda_index"] == 1, "Independent per-arm selection differs")
    cv[:] = 0
    cv[4] = np.where(gap > 0, 1e-13, -1e-13)
    require(select_lambda(cv, gap)["selected_lambda_index"] == 0, "Global-minimum lambda tie rule differs")
    tests.append("independent_arm_selection_and_global_minimum_tie")

    utility = rng.uniform(0, 1, size=(6, 2))
    utility[0] = (.6, .6 - 5e-13)
    actions = {name: rng.integers(0, 2, 6).astype(bool) for name in ("P", "L", "S0", "S1")}
    actions["P"][0] = True
    values = contributions(utility[:, 0] - utility[:, 1], actions)
    p_quality = np.where(actions["P"], utility[:, 0], utility[:, 1])
    for column, baseline in enumerate(("L", "S0", "S1", "Dense", "BM25")):
        action = actions[baseline] if baseline in actions else np.full(6, baseline == "BM25")
        quality = np.where(action, utility[:, 0], utility[:, 1])
        compare(values[:, column], p_quality - quality, "synthetic_contributions", errors)
    summary = policy_summary(utility, actions["P"])
    require(summary["zero_count"] >= 1, "Tie classification differs")
    compare(summary["F1"], float(p_quality.mean()), "synthetic_policy_F1", errors)
    groups = np.asarray(["a", "a", "b", "c", "c", "c"])
    actual_intervals = frequency_intervals(values, groups, draws=61, seed=501)
    brute_rng = np.random.default_rng(501)
    members = [np.flatnonzero(groups == name) for name in sorted(set(groups))]
    brute_samples = []
    for _ in range(61):
        chosen = brute_rng.integers(0, 3, size=3)
        rows = np.concatenate([members[index] for index in chosen])
        brute_samples.append(np.mean(values[rows], axis=0))
    compare(actual_intervals, np.quantile(brute_samples, QUANTILES, axis=0), "synthetic_frequency_bootstrap", errors)
    tests.append("five_signed_contributions_full_N_counts_and_unequal_group_bootstrap")

    points = np.asarray([.002, .003, .004, -.01, -.02])
    bounds = np.asarray([[.001, .001, .001, -.02, -.03], [.01, .01, .01, .01, .01]])
    require(transfer_gate(points, bounds), "Transfer gate wrongly includes fixed-baseline contrasts")
    for column in range(3):
        changed = bounds.copy()
        changed[0, column] = 0
        require(not transfer_gate(points, changed), "Nonpositive primary lower bound accepted")
        changed_points = points.copy()
        changed_points[column] = .00199
        require(not transfer_gate(changed_points, bounds), "Insufficient primary point gain accepted")
    tests.append("all_three_transfer_conditions_without_deployable_candidate")
    result = {"status": "passed_independent_probe_checker_synthetic_self_test", "checks": tests,
              "maximum_absolute_errors": errors, "new_model_fits": 0, "research_data_reads": 0,
              "GPU_calls": 0, "external_calls": 0}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        check(arguments.output_dir)
