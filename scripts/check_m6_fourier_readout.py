"""Independent numerical checks for the fixed-M6 Fourier readout experiment.

Reconstructs geometric scales and Fourier features, evaluates saved convex BCE
solutions, and checks cached native-score arithmetic. This module never fits a
model, executes GPU inference, or imports the runner or its numerical helpers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits


PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT.parent / "work" / "router_research"
DEFAULT_OUTPUT = PROJECT / "outputs" / "router" / "hotpotqa_bd_router_v1" / "runs" / "m6_fourier_readout_v1"
REGULARIZATIONS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
FOURIER_SEEDS = (2026091502, 2026091503)
GEOMETRY_SEED = 2026091505
BANDWIDTH_MULTIPLIERS = (.5, 1., 2.)
INPUT_DIMENSION = 384
FREQUENCIES_PER_SCALE = 64
OUTPUT_DIMENSION = 768
TIE_ATOL = 1e-12
GRADIENT_ATOL = 1e-8
SCALAR_ATOL = 2e-12
FEATURE_ATOL = 2e-7
BOOTSTRAP_SEED = 2026091504
BOOTSTRAP_DRAWS = 20000
QUANTILES = (.05 / 6, 1 - .05 / 6)
PRIMARY = ("N_minus_L", "N_minus_Dense", "N_minus_BM25")
EXPECTED_FOLDS_SHA256 = "ad2e94f76332f288f798d8bc1b1ff085be9aba64730b84d4906566d37ea76bc6"


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


def compare(actual, expected, name, errors, atol=SCALAR_ATOL):
    actual, expected = real_array(actual, name=name), real_array(expected, name=name)
    require(actual.shape == expected.shape, f"Shape differs: {name}")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    error = float(difference.max()) if difference.size else 0.0
    require(error <= atol, f"{name} differs by {error:.17g}; allowed {atol:.17g}")
    errors[name] = max(error, errors.get(name, 0.0))
    return error


def inner_groups(groups, fold):
    unique = sorted(set(map(str, groups)))
    ranked = sorted(unique, key=lambda group: (
        hashlib.sha256(f"lp_ft_inner_v1|fold={fold}|group={group}".encode()).digest(), group))
    positions = {group: position % 3 for position, group in enumerate(ranked)}
    return np.asarray([positions[str(group)] for group in groups], dtype=np.int64)


def frequencies(seed):
    """Recreate the single, fixed standard-normal array for one seed."""
    require(type(seed) is int and seed >= 0, "Invalid Fourier seed")
    return np.random.default_rng(seed).standard_normal((3, INPUT_DIMENSION, FREQUENCIES_PER_SCALE))


def geometry(features):
    """Determine the training-only scale using explicit pair distances."""
    features = real_array(features, name="geometry features")
    require(features.ndim == 2 and features.shape[1] == INPUT_DIMENSION and len(features) >= 2,
            "Geometry needs at least two 384-dimensional training rows")
    k = min(512, len(features) // 2)
    order = np.random.default_rng(GEOMETRY_SEED).permutation(len(features))[:2 * k]
    pairs = order.reshape(k, 2)
    distances = []
    for left, right in pairs:
        distances.append(math.fsum((float(a) - float(b)) ** 2
                                  for a, b in zip(features[left], features[right])))
    ordered = sorted(distances)
    middle = k // 2
    median = ordered[middle] if k % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    scale = math.sqrt(median)
    require(math.isfinite(scale) and scale > 1e-12, "Training geometry has no valid positive median distance")
    return {"scale": scale, "pair_count": k, "pair_indices": pairs,
            "squared_distances": np.asarray(distances, dtype=np.float64)}


def fourier_transform(features, omega, scale):
    """Independent FP64 einsum construction, followed by one FP32 cast.

    Output order is original M6, then cos64/sin64 for each of the three scales.
    The original coordinates are retained; no centering or joint normalization
    is performed. Native inference and its two-branch rounding are separate.
    """
    features = real_array(features, name="Fourier input", dtype=np.float32)
    require(features.ndim == 2 and features.shape[1] == INPUT_DIMENSION, "Wrong Fourier input width")
    omega = real_array(omega, (3, INPUT_DIMENSION, FREQUENCIES_PER_SCALE), "frequencies", np.float64)
    require(type(scale) in (int, float) and math.isfinite(scale) and scale > 1e-12, "Invalid Fourier scale")
    source = features.astype(np.float64)
    blocks = [source]
    normalizer = math.sqrt(3 * FREQUENCIES_PER_SCALE)
    for index, multiplier in enumerate(BANDWIDTH_MULTIPLIERS):
        angles = np.einsum("nd,dk->nk", source, omega[index], optimize=False) / (multiplier * scale)
        blocks.extend((np.cos(angles) / normalizer, np.sin(angles) / normalizer))
    transformed = np.concatenate(blocks, axis=1).astype(np.float32)
    require(np.isfinite(transformed).all(), "Nonfinite transformed coordinates")
    require(np.array_equal(transformed[:, :INPUT_DIMENSION], features), "Original M6 coordinates changed")
    return transformed


def active_labels_weights(gap):
    gap = real_array(gap, name="utility gap").astype(np.float64)
    require(gap.ndim == 1 and len(gap) > 0, "Expected nonempty utility gap vector")
    active = np.flatnonzero(np.abs(gap) > TIE_ATOL)
    require(len(active) > 0, "Positive total weight is required")
    selected = gap[active]
    weight_sum = math.fsum(abs(float(value)) for value in selected)
    weights = np.asarray([abs(float(value)) / weight_sum for value in selected])
    labels = selected > 0
    return active, labels, weights


def signed_softplus(value):
    value = float(value)
    return max(value, 0.) + math.log1p(math.exp(-abs(value)))


def weighted_bce(scores, gap):
    scores = real_array(scores, (len(gap),), "BCE scores")
    active, labels, weights = active_labels_weights(gap)
    return math.fsum(float(weight) * signed_softplus(-float(scores[i]) if label else float(scores[i]))
                     for i, label, weight in zip(active, labels, weights))


def bce_quantities(features, gap, coef, intercept, regularization):
    """Evaluate a saved solution, independently of the optimization library."""
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
    for i, (logit, label, weight) in enumerate(zip(logits, labels, weights)):
        value = float(logit)
        if value >= 0:
            probability = 1 / (1 + math.exp(-value))
        else:
            exponent = math.exp(value)
            probability = exponent / (1 + exponent)
        residual[i] = float(weight) * (probability - int(label))
    coef_gradient = np.einsum("nd,n->d", selected, residual, optimize=False) + regularization * coef
    gradient = np.r_[coef_gradient, math.fsum(map(float, residual))]
    require(np.isfinite(gradient).all(), "Nonfinite BCE gradient")
    return {"data_loss": data_loss, "penalty": penalty, "loss": data_loss + penalty,
            "grad_inf": float(np.abs(gradient).max()), "gradient": gradient,
            "positive_weight_rows": len(active)}


def select_lambda(cv_native, gap):
    """Select one shared lambda from both native-score arrays, never per seed."""
    cv_native = real_array(cv_native, (2, 5, len(gap)), "native inner scores", np.float32)
    ensemble = (cv_native[0].astype(np.float64) + cv_native[1].astype(np.float64)) / 2
    losses = [weighted_bce(scores, gap) for scores in ensemble]
    best = min(losses)
    selected = next(index for index, value in enumerate(losses) if value <= best + TIE_ATOL)
    return {"selected_lambda_index": selected, "cv_bce": losses, "ensemble_scores": ensemble}


def policy_summary(utility, action):
    utility = real_array(utility, name="policy utility")
    require(utility.ndim == 2 and utility.shape[1] == 2, "Invalid policy utility")
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


def contributions(gap, nonlinear, linear):
    nonlinear, linear = np.asarray(nonlinear), np.asarray(linear)
    require(nonlinear.dtype == bool and linear.dtype == bool, "Actions must be boolean")
    gap = real_array(gap, name="contribution gap")
    require(nonlinear.shape == linear.shape == gap.shape and gap.ndim == 1, "Invalid contribution shapes")
    n, l = nonlinear.astype(np.int64), linear.astype(np.int64)
    return np.column_stack(((n - l) * gap, n * gap, (n - 1) * gap))


def means(values):
    return np.asarray([math.fsum(map(float, values[:, column])) / len(values)
                       for column in range(values.shape[1])])


def frequency_intervals(values, groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Whole-group bootstrap using independent multiplicities and fsum totals."""
    values = real_array(values, name="bootstrap values")
    require(values.ndim == 2 and len(values) == len(groups) and len(values) > 0, "Invalid bootstrap inputs")
    members = {}
    for row, group in enumerate(groups):
        members.setdefault(str(group), []).append(row)
    names = sorted(members)
    totals = np.asarray([[math.fsum(float(values[i, column]) for i in members[group])
                          for column in range(values.shape[1])] for group in names])
    sizes = np.asarray([len(members[group]) for group in names], dtype=np.int64)
    rng = np.random.default_rng(seed)
    sampled_means = np.empty((draws, values.shape[1]), dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(names), size=len(names))
        counts = np.bincount(sampled, minlength=len(names))
        used = np.flatnonzero(counts)
        denominator = int(counts[used] @ sizes[used])
        sampled_means[draw] = np.einsum("i,ij->j", counts[used], totals[used], optimize=False) / denominator
    return np.quantile(sampled_means, QUANTILES, axis=0, method="linear")


def load_original_inputs():
    """Read historical M6 rows/splits and existing scores; no new calibration."""
    directory = RESEARCH / "layer_pooling_v1"
    completion = read(directory / "completion_record.json")
    historical = {resolve(path): digest for path, digest in completion["artifact_sha256"].items()}
    required = [directory / "features.npz", directory / "predictions.npz"]
    required += [directory / f"fold{fold}_M6_{suffix}" for fold in range(5)
                 for suffix in ("head.pt", "fit.npz")]
    for path in required:
        require(path.resolve() in historical and sha(path) == historical[path.resolve()],
                f"Historical M6 artifact differs: {path}")
    split_path = RESEARCH / "e02_results" / "fold_indices.npz"
    require(sha(split_path) == EXPECTED_FOLDS_SHA256, "Historical split archive differs")
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        qids, groups, features = (archive[key].copy() for key in ("query_ids", "group_ids", "M6"))
    require(qids.shape == groups.shape == (9600,), "Wrong historical identity shapes")
    require(qids.dtype.kind in "US" and groups.dtype.kind in "US", "Historical identities must be strings")
    require(len(set(qids)) == 9600 and len(set(groups)) == 9559, "Wrong historical query/group counts")
    real_array(features, (9600, INPUT_DIMENSION), "historical M6 features", np.float32)
    norm = np.sqrt(np.einsum("nd,nd->n", features.astype(float), features.astype(float), optimize=False))
    require(np.max(abs(norm - 1)) < 2e-6, "Historical M6 normalization differs")
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        require(np.array_equal(qids, archive["query_ids"]) and np.array_equal(groups, archive["group_ids"]),
                "Historical feature/prediction identities differ")
        utility, linear = archive["utility"].copy(), archive["M6"].copy()
    real_array(utility, (9600, 2), "historical utility", np.float64)
    require(np.all((0 <= utility) & (utility <= 1)), "Invalid historical utility range")
    real_array(linear, (9600,), "historical M6 native scores", np.float64)
    with np.load(split_path, allow_pickle=False) as archive:
        folds = [{part: archive[f"fold{fold}_{part}"].copy()
                  for part in ("fit", "calibration", "test")} for fold in range(5)]
    offset_directory = RESEARCH / "m6_pooled_offset_v1"
    offset_binding = read(offset_directory / "completion_record.json")["artifact_sha256"]
    offset_binding = {resolve(path): digest for path, digest in offset_binding.items()}
    offset_path = offset_directory / "cal_logits.npz"
    require(offset_path.resolve() in offset_binding and sha(offset_path) == offset_binding[offset_path.resolve()],
            "Historical calibration score binding differs")
    with np.load(offset_path, allow_pickle=False) as archive:
        old_cal = []
        for fold in range(5):
            require(np.array_equal(archive[f"fold{fold}_cal_indices"], folds[fold]["calibration"]),
                    "Historical calibration identities differ")
            old_cal.append(real_array(archive[f"fold{fold}_O"], (1536,), "historical cal scores", np.float32).copy())
    coverage = np.zeros(9600, dtype=np.int64)
    assignments, old_fit, old_coefs, old_intercepts = [], [], [], []
    # CPU-only deserialization of five directly bound historical tensor files;
    # no model, optimizer, training module, CUDA context, or inference is used.
    import torch
    for fold, parts in enumerate(folds):
        group_sets = []
        for part, size in (("fit", 6144), ("calibration", 1536), ("test", 1920)):
            indices = real_array(parts[part], (size,), f"fold{fold} {part}")
            require(indices.dtype.kind in "iu" and len(set(indices)) == size
                    and np.all((0 <= indices) & (indices < 9600)), "Invalid historical split indices")
            group_sets.append(set(groups[indices]))
        require(not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2]
                     or group_sets[1] & group_sets[2]), "Historical outer group leakage")
        require(np.array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(9600)),
                "Historical fold does not partition the pool")
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
    require(np.all(coverage == 1), "Historical OOF test coverage differs")
    return {"query_ids": qids, "group_ids": groups, "features": features,
            "utility": utility, "gap": utility[:, 0] - utility[:, 1],
            "L_scores": linear, "folds": folds, "inner_assignments": assignments,
            "L_fit": old_fit, "L_cal": old_cal, "L_coef": old_coefs, "L_intercept": old_intercepts}


def check(output_dir):
    """Check a completed experiment and exclusively create separate_checks.json."""
    output_dir = Path(output_dir).resolve()
    destination = output_dir / "separate_checks.json"
    require(not destination.exists(), "Preserve existing separate_checks.json")
    fit_files = [output_dir / f"fold{fold}_context{context}.npz"
                 for fold in range(5) for context in range(4)]
    fit_files += [output_dir / f"fold{fold}.{suffix}" for fold in range(5) for suffix in ("json", "npz")]
    fit_files.append(output_dir / "solutions.jsonl")
    pilot_files = [output_dir / f"pilot_seed{seed}.{suffix}" for seed in range(2) for suffix in ("json", "npz")]
    required = fit_files + pilot_files + [output_dir / name for name in (
        "protocol.json", "frequencies.npz", "pilot.json", "fit_completion.json",
        "predictions.npz", "predictions_frozen.json", "results.json")]
    require(all(path.is_file() for path in required), "Wait for the complete formal experiment")
    started = time.perf_counter()
    protocol = read(output_dir / "protocol.json")
    require(protocol["status"] == "frozen_before_real_pilot_and_formal_Fourier_fits", "Wrong protocol status")
    sources = bindings(protocol["source_sha256"], "source")
    inputs = bindings(protocol["input_sha256"], "input")
    expected_sources = [Path(__file__).resolve(), PROJECT / "scripts" / "run_m6_fourier_readout.py",
                        PROJECT / "scripts" / "m6_fourier_math.py",
                        PROJECT / "analysis" / "hotpotqa_router" / "m6_fourier_readout_plan_20260915.md",
                        PROJECT / "scripts" / "run_m6_objective_readout.py",
                        PROJECT / "scripts" / "m6_objective_math.py",
                        RESEARCH / "weighted_linear_probe.py", RESEARCH / "weighted_linear_probe_refined.py"]
    require(set(path.resolve() for path in expected_sources) == set(sources), "Frozen source set differs")
    expected_inputs = [RESEARCH / name for name in (
        "layer_pooling_v1/features.npz", "layer_pooling_v1/predictions.npz",
        "e02_results/fold_indices.npz", "m6_pooled_offset_v1/cal_logits.npz",
        "layer_pooling_v1/completion_record.json", "m6_pooled_offset_v1/completion_record.json")]
    expected_inputs += [RESEARCH / "layer_pooling_v1" / f"fold{fold}_M6_{suffix}"
                        for fold in range(5) for suffix in ("head.pt", "fit.npz")]
    require(set(path.resolve() for path in expected_inputs) == set(inputs), "Frozen input set differs")
    expected_config = {
        "queries": 9600, "groups": 9559, "folds": 5, "fit_queries": 6144,
        "cal_queries": 1536, "test_queries": 1920, "inner_folds": 3,
        "regularizations": list(REGULARIZATIONS), "seeds": list(FOURIER_SEEDS),
        "geometry_seed": GEOMETRY_SEED, "geometry_max_pairs": 512,
        "scales": list(BANDWIDTH_MULTIPLIERS), "frequencies_per_scale": 64,
        "linear_dimensions": 384, "fourier_dimensions": 384, "dimensions": 768,
        "tie_atol": TIE_ATOL, "gradient_atol": GRADIENT_ATOL, "batch_size": 8,
        "formal_solves": 160, "pilot_solves": 2, "new_thresholds": 0,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "interval_quantiles": list(QUANTILES), "minimum_recipe_increment": .002,
        "minimum_gain_over_both_fixed": .01, "cpu_threads": 2,
        "new_encoder_forwards": 0, "new_external_calls": 0,
        "ensemble": "FP64 mean of two FP32 native scores; one shared inner-selected lambda",
    }
    require(protocol["config"] == expected_config, "Frozen configuration differs")
    policy_names = ("N", "L", "S0", "S1", "Dense", "BM25")
    require(protocol["primary"] == list(PRIMARY) and protocol["policies"] == list(policy_names),
            "Frozen contrasts or policies differ")
    snapshots = {str(path): sha(path) for path in required}
    protocol_hash = snapshots[str(output_dir / "protocol.json")]
    require(protocol["frequencies_sha256"] == snapshots[str(output_dir / "frequencies.npz")],
            "Frequency archive binding differs")
    result = read(output_dir / "results.json")
    require(result["status"] == "complete_fixed_Fourier_comparison_pending_separate_check"
            and result["protocol_sha256"] == protocol_hash, "Wait for completed, bound formal evaluation")
    require(result["core_goal_achieved"] is False, "Development output cannot be independent confirmation")
    for key, expected in (("formal_solves", 160), ("final_heads", 10), ("new_thresholds", 0),
                          ("new_encoder_forwards", 0), ("new_external_calls", 0)):
        require(type(result[key]) is int and result[key] == expected, f"Wrong completion count: {key}")
    fit_completion = read(output_dir / "fit_completion.json")
    require(fit_completion["status"] == "all_160_solves_and_10_heads_fixed_before_new_test_predictions"
            and fit_completion["protocol_sha256"] == protocol_hash, "Wrong pre-test head completion record")
    require(fit_completion["formal_solves"] == 160 and fit_completion["final_heads"] == 10
            and fit_completion["new_thresholds"] == 0, "Wrong pre-test completion counts")
    fitted_bindings = bindings(fit_completion["artifact_sha256"], "pre-test fits")
    require(set(fitted_bindings) == set(fit_files), "Pre-test artifact set differs")
    frozen = read(output_dir / "predictions_frozen.json")
    require(frozen["status"] == "complete_9600_OOF_actions_before_effects"
            and frozen["protocol_sha256"] == protocol_hash, "Wrong pre-effect prediction record")
    require(frozen["predictions_sha256"] == snapshots[str(output_dir / "predictions.npz")]
            and frozen["fit_completion_sha256"] == snapshots[str(output_dir / "fit_completion.json")],
            "Prediction freeze bindings differ")
    pilot = read(output_dir / "pilot.json")
    require(pilot["status"] == "passed_two_real_pilot_solves_and_480_zero_extension_native_controls"
            and pilot["protocol_sha256"] == protocol_hash, "Wrong engineering pilot record")
    require(pilot["pilot_solves"] == 2 and pilot["pilot_rows"] == 128
            and pilot["original_native_control_queries"] == 480, "Wrong pilot scope")
    require(set(bindings(pilot["artifact_sha256"], "pilot")) == set(pilot_files), "Wrong pilot artifact set")
    require(len(pilot["original_controls"]) == 15 and all(value == 0 for value in pilot["original_controls"].values()),
            "Native zero-extension controls did not pass")
    with (output_dir / "solutions.jsonl").open(encoding="utf-8") as stream:
        journal = [json.loads(line) for line in stream if line.strip()]
    require(len(journal) == 160, "Expected exactly 160 formal journal solutions")
    errors, independent_gradients, scales, selections = {}, [], [], []
    active_cache_rows, nontraining_transform_rows = 0, 0
    with threadpool_limits(limits=2):
        data = load_original_inputs()
        x, gap, utility, groups = (data[key] for key in ("features", "gap", "utility", "group_ids"))
        with np.load(output_dir / "frequencies.npz", allow_pickle=False) as archive:
            require(set(archive.files) == {"seed0", "seed1"}, "Unexpected frequency archive fields")
            omega = [real_array(archive[f"seed{seed}"], (3, 384, 64), "saved frequencies", np.float64).copy()
                     for seed in range(2)]
        for seed in range(2):
            require(np.array_equal(omega[seed], frequencies(FOURIER_SEEDS[seed])), "Fixed frequencies differ")
        with np.load(output_dir / "predictions.npz", allow_pickle=False) as archive:
            prediction = {key: archive[key].copy() for key in archive.files}
        for key in ("query_ids", "group_ids", "utility", "L_scores"):
            require(np.array_equal(prediction[key], data[key]), f"Frozen original prediction field differs: {key}")
        seed_scores = real_array(prediction["N_seed_scores"], (2, 9600), "native seed scores", np.float32)
        nonlinear_scores = real_array(prediction["N_scores"], (9600,), "ensemble scores", np.float64)
        require(np.array_equal(nonlinear_scores, (seed_scores[0].astype(float) + seed_scores[1].astype(float)) / 2),
                "Final ensemble is not the fixed FP64 mean")
        real_array(prediction["fold_id"], (9600,), "OOF fold index", np.int64)
        actions = {"N": nonlinear_scores > 0, "L": data["L_scores"] > 0,
                   "S0": seed_scores[0] > 0, "S1": seed_scores[1] > 0,
                   "Dense": np.zeros(9600, dtype=bool), "BM25": np.ones(9600, dtype=bool)}
        for name in ("N", "L", "S0", "S1"):
            require(prediction[name].dtype == bool and np.array_equal(prediction[name], actions[name]),
                    f"Actions differ: {name}")
        expected_fold = np.full(9600, -1, dtype=np.int64)
        require(isinstance(result["folds"], list) and len(result["folds"]) == 5, "Expected five result folds")
        for fold, parts in enumerate(data["folds"]):
            fit, cal, test = (parts[key] for key in ("fit", "calibration", "test"))
            expected_fold[test] = fold
            info = read(output_dir / f"fold{fold}.json")
            require(type(info["fold"]) is int and info["fold"] == fold, "Fold record identity differs")
            with np.load(output_dir / f"fold{fold}.npz", allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in archive.files}
            coef = real_array(saved["coef"], (32, 768), "saved coefficients", np.float64)
            intercept = real_array(saved["intercept"], (32,), "saved intercepts", np.float64)
            assignment = data["inner_assignments"][fold]
            require(np.array_equal(saved["inner_assignment"], assignment), "Historical inner assignment changed")
            cv = real_array(saved["cv_native"], (2, 5, 6144), "native CV scores", np.float32)
            fit_native = real_array(saved["fit_native"], (2, 6144), "native fit scores", np.float32)
            cal_native = real_array(saved["cal_native"], (2, 1536), "native cal scores", np.float32)
            choice = select_lambda(cv, gap[fit])
            selected = choice["selected_lambda_index"]
            require(type(info["selected_lambda_index"]) is int and info["selected_lambda_index"] == selected,
                    "Shared-lambda selection differs")
            compare(info["cv_weighted_BCE"], choice["cv_bce"], "cv_weighted_BCE", errors)
            selections.append(REGULARIZATIONS[selected])
            contexts = []
            require(len(info["geometry_scales"]) == 4, "Expected four geometries per fold")
            for context in range(4):
                train = fit[assignment != context] if context < 3 else fit
                active = train[np.abs(gap[train]) > TIE_ATOL]
                expected_geometry = geometry(x[train])
                path = output_dir / f"fold{fold}_context{context}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    cached = {key: archive[key].copy() for key in archive.files}
                require(np.array_equal(cached["train_indices"], train)
                        and np.array_equal(cached["active_indices"], active), "Training/active row identities differ")
                require(np.array_equal(cached["pair_indices"], expected_geometry["pair_indices"]),
                        "Training-only distance pairs differ")
                saved_scale = float(real_array(cached["scale"], (), "saved geometry scale", np.float64))
                compare(saved_scale, expected_geometry["scale"], "training_geometry_scale", errors)
                compare(info["geometry_scales"][context], saved_scale, "recorded_geometry_scale", errors)
                scales.append({"fold": fold, "context": context, "scale": saved_scale,
                               "training_queries": len(train), "active_queries": len(active)})
                design = []
                for seed in range(2):
                    rff = real_array(cached[f"rff_seed{seed}"], (len(active), 384), "actual active RFF cache", np.float32)
                    rebuilt = fourier_transform(x[active], omega[seed], saved_scale)
                    compare(rff, rebuilt[:, 384:], "actual_active_RFF", errors, FEATURE_ATOL)
                    design.append(np.concatenate((x[active], rff), axis=1))
                    active_cache_rows += len(active)
                    # These deterministic transforms are reconstructed fully;
                    # cached native logits remain covered by the separate GPU
                    # path controls and frozen source, not by CPU emulation.
                    validation_sets = [fit[assignment == context]] if context < 3 else [cal, test]
                    for indices in validation_sets:
                        transformed = fourier_transform(x[indices], omega[seed], saved_scale)
                        require(transformed.shape == (len(indices), 768), "Nontraining transform shape differs")
                        nontraining_transform_rows += len(indices)
                contexts.append({"train": train, "active": active, "design": design})
            require(len(info["solves"]) == 32, "Expected 32 solutions per fold")
            for index, record in enumerate(info["solves"]):
                if index < 30:
                    lambda_index, remainder = divmod(index, 6)
                    context, seed = divmod(remainder, 2)
                else:
                    lambda_index, context, seed = selected, 3, index - 30
                ctx = contexts[context]
                expected_identity = {"seed_index": seed, "seed": FOURIER_SEEDS[seed],
                    "lambda_index": lambda_index, "context": context,
                    "inner_split": context if context < 3 else None, "solution_index": index,
                    "training_queries": len(ctx["train"]), "role": "inner" if context < 3 else "refit",
                    "regularization": REGULARIZATIONS[lambda_index], "positive_weight_rows": len(ctx["active"])}
                require(all(record[key] == value for key, value in expected_identity.items()), "Solution identities differ")
                require(record["accepted"] is True and record["optimizer_success"] is True
                        and record["status"] == "accepted_stationary_solution"
                        and record["acceptance_gradient_inf"] == GRADIENT_ATOL, "Unaccepted formal solution")
                journal_record = journal[fold * 32 + index]
                require(journal_record["fold"] == fold and all(journal_record[key] == value for key, value in record.items()),
                        "Journal and fold solver records differ")
                require(np.array_equal(np.asarray(journal_record["coef"], dtype=np.float64), coef[index]),
                        "Journal and saved coefficients differ")
                compare(record["intercept"], intercept[index], "BCE_intercept", errors)
                observed = bce_quantities(ctx["design"][seed], gap[ctx["active"]], coef[index],
                                          float(intercept[index]), REGULARIZATIONS[lambda_index])
                require(observed["grad_inf"] <= GRADIENT_ATOL, f"Independent stationarity failed: fold {fold}, solution {index}")
                independent_gradients.append(observed["grad_inf"])
                for key in ("data_loss", "penalty", "loss", "grad_inf"):
                    compare(record[key], observed[key], f"BCE_{key}", errors,
                            1e-10 if key == "grad_inf" else SCALAR_ATOL)
            for role, native, indices in (("fit", fit_native, fit), ("cal", cal_native, cal)):
                compare(info[f"{role}_BCE_seed"], [weighted_bce(row, gap[indices]) for row in native],
                        f"{role}_BCE_seed", errors)
                ensemble = (native[0].astype(float) + native[1].astype(float)) / 2
                compare(info[f"{role}_BCE_ensemble"], weighted_bce(ensemble, gap[indices]), f"{role}_BCE_ensemble", errors)
                compare(info[f"L_{role}_BCE"], weighted_bce(data[f"L_{role}"][fold], gap[indices]),
                        f"L_{role}_BCE", errors)
            old_coef = data["L_coef"][fold].astype(np.float64)
            feasible_scores = np.einsum("nd,d->n", x[fit].astype(float), old_coef, optimize=False) + data["L_intercept"][fold]
            feasible = weighted_bce(feasible_scores, gap[fit]) + REGULARIZATIONS[selected] * math.fsum(float(v)**2 for v in old_coef) / 2
            compare(info["linear_feasible_objective"], feasible, "linear_feasible_objective", errors)
            require(all(info["solves"][index]["loss"] <= feasible + 1e-8 for index in (30, 31)),
                    "Refit objective exceeds the historical FP32 linear feasible point")
            reported_fold = result["folds"][fold]
            require(all(reported_fold[key] == value for key, value in info.items()), "Result fold changed after fitting")
            compare(reported_fold["test_BCE_seed"], [weighted_bce(row[test], gap[test]) for row in seed_scores],
                    "test_BCE_seed", errors)
            compare(reported_fold["test_BCE_ensemble"], weighted_bce(nonlinear_scores[test], gap[test]),
                    "test_BCE_ensemble", errors)
            compare(reported_fold["L_test_BCE"], weighted_bce(data["L_scores"][test], gap[test]), "L_test_BCE", errors)
        require(np.array_equal(prediction["fold_id"], expected_fold), "Outer test fold identities differ")
        for name in policy_names:
            expected = policy_summary(utility, actions[name])
            for key, value in expected.items():
                if key.endswith("count"):
                    require(type(result["policy"][name][key]) is int and result["policy"][name][key] == value,
                            f"Policy count differs: {name}.{key}")
                else:
                    compare(result["policy"][name][key], value, f"policy_{key}", errors)
        values = contributions(gap, actions["N"], actions["L"])
        main_means = means(values)
        intervals = frequency_intervals(values, groups)
        checked_primary = {}
        for column, name in enumerate(PRIMARY):
            compare(result["primary"][name]["mean"], main_means[column], "primary_mean", errors)
            compare(result["primary"][name]["interval"], intervals[:, column], "primary_interval", errors)
            checked_primary[name] = {"mean": float(main_means[column]), "interval": intervals[:, column].tolist()}
        for fold, parts in enumerate(data["folds"]):
            fold_means = means(values[parts["test"]])
            for column, name in enumerate(PRIMARY):
                compare(result["folds"][fold]["primary_means"][name], fold_means[column], "fold_primary_mean", errors)
        followup = bool(intervals[0, 0] > 0 and main_means[0] >= .002)
        candidate = bool(followup and np.all(intervals[0, 1:] > 0) and np.all(main_means[1:] >= .01))
        require(result["recipe_followup_gate"] is followup and result["candidate_preparation_gate"] is candidate,
                "Predefined decisions differ")
    bindings(protocol["source_sha256"], "source")
    bindings(protocol["input_sha256"], "input")
    require(all(sha(path) == digest for path, digest in snapshots.items()), "Formal artifacts changed during checking")
    report = {
        "status": "passed_independent_fourier_readout_checks", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checker_sha256": sha(__file__), "protocol_sha256": protocol_hash,
        "results_sha256": snapshots[str(output_dir / "results.json")], "artifact_sha256": snapshots,
        "source_bindings_checked": len(sources), "input_bindings_checked": len(inputs),
        "queries": 9600, "groups": 9559, "frequencies_checked": 2, "geometries_checked": 20,
        "formal_solutions_checked": 160, "journal_solutions_checked": 160, "final_heads_checked": 10,
        "active_training_RFF_cache_rows_checked": active_cache_rows,
        "nontraining_transform_rows_reconstructed": nontraining_transform_rows,
        "geometry": scales, "selected_regularizations": selections,
        "maximum_independent_gradient_inf": max(independent_gradients), "maximum_absolute_errors": errors,
        "primary": checked_primary, "recipe_followup_gate": followup, "candidate_preparation_gate": candidate,
        "independent_native_GPU_replay": False,
        "native_score_scope": "Reconstructed all inner-validation/cal/test Fourier transforms, but these nontraining features were not cached for comparison. Native logits are checked as saved numeric inputs; the GPU path has separate source binding and pilot controls. This is not full GPU replay.",
        "training_score_scope": "Compared both seeds' actual FP32 active-row RFF caches against an independent FP64 einsum construction; all original-coordinate BCE gradients use the actual cached FP32 training inputs.",
        "linear_feasible_scope": "Directly bound historical FP32 head tensors, loaded on CPU; a fixed feasible point at the selected lambda, not a newly optimized same-lambda linear baseline.",
        "new_model_fits": 0, "new_paid_calls": 0,
        "scope": "Independent numerical implementation on consumed old9600; no new-sample or independent-team confirmation.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(destination),
                      "maximum_absolute_errors": errors, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    check(arguments.output_dir)
