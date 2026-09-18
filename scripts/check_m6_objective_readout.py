"""Independently check a completed fixed-M6 objective/readout experiment.

Uses the original frozen old-pool inputs, explicit Ridge gradients, exhaustive
calibration thresholds, and frequency-weighted group bootstrap. Does not import
the experiment runner or its numerical module, run inference, or train models.
The only output is a new separate_checks.json; existing checks are preserved.
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
DEFAULT_OUTPUT = PROJECT / "analysis" / "hotpotqa_router" / "m6_objective_readout_v1"
REGULARIZATIONS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
PRIMARY = (
    "R0_minus_L0", "Rt_minus_Lt", "calibration_interaction",
    "Rt_minus_Dense", "Rt_minus_BM25",
)
POLICIES = ("L0", "Lt", "R0", "Rt", "Dense", "BM25")
TIE_ATOL = 1e-12
NUMBER_ATOL = 2e-12
GRADIENT_ATOL = 1e-10
BOOTSTRAP_SEED = 2026091501
BOOTSTRAP_DRAWS = 20000
QUANTILES = (0.005, 0.995)
INPUTS = {
    "features": RESEARCH / "layer_pooling_v1" / "features.npz",
    "old_predictions": RESEARCH / "layer_pooling_v1" / "predictions.npz",
    "folds": RESEARCH / "e02_results" / "fold_indices.npz",
    "old_cal_logits": RESEARCH / "m6_pooled_offset_v1" / "cal_logits.npz",
}
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
    return path if path.is_absolute() else PROJECT / path


def bindings(mapping, label):
    require(isinstance(mapping, dict) and bool(mapping), f"Missing {label} bindings")
    checked = {}
    for filename, expected in mapping.items():
        path = resolve(filename).resolve()
        require(isinstance(expected, str) and sha(path) == expected, f"Changed {label}: {path}")
        checked[path] = expected
    return checked


def compare(actual, expected, name, errors, atol=NUMBER_ATOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, f"Shape differs: {name}")
    require(actual.dtype.kind in "fiu" and expected.dtype.kind in "fiu", f"Non-numeric {name}")
    actual, expected = actual.astype(float), expected.astype(float)
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"Nonfinite {name}")
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    require(error <= atol, f"{name} differs by {error:.17g} (allowed {atol})")
    errors[name] = max(error, errors.get(name, 0.0))
    return error


def finite_array(value, shape, name, dtype=None):
    array = np.asarray(value)
    require(array.shape == shape, f"Invalid {name} shape")
    require(array.dtype.kind in "fiu" and np.isfinite(array).all(), f"Invalid {name} values")
    if dtype is not None:
        require(array.dtype == np.dtype(dtype), f"Invalid {name} dtype: {array.dtype}")
    return array


def inner_groups(groups, fold):
    unique = sorted(set(map(str, groups)))
    ranked = sorted(unique, key=lambda group: (
        hashlib.sha256(f"lp_ft_inner_v1|fold={fold}|group={group}".encode("utf-8")).digest(), group))
    positions = {group: position % 3 for position, group in enumerate(ranked)}
    return np.asarray([positions[str(group)] for group in groups], dtype=np.int64)


def ridge_quantities(features, targets, coef, intercept, regularization):
    """Evaluate the original, uncentered objective without solving it again."""
    features = np.asarray(features, dtype=np.float64)
    residual = np.einsum("ij,j->i", features, coef, optimize=False) + intercept - targets
    n = len(targets)
    data_loss = math.fsum(float(value) ** 2 for value in residual) / (2 * n)
    penalty = regularization * math.fsum(float(value) ** 2 for value in coef) / 2
    gradient = np.einsum("ij,i->j", features, residual, optimize=False) / n + regularization * coef
    intercept_gradient = math.fsum(map(float, residual)) / n
    grad_inf = max(float(np.max(np.abs(gradient))), abs(intercept_gradient))
    require(all(map(math.isfinite, (data_loss, penalty, grad_inf))), "Nonfinite independent Ridge quantities")
    return {"data_loss": data_loss, "penalty": penalty, "loss": data_loss + penalty, "grad_inf": grad_inf}


def threshold_selection(scores, gap):
    """Enumerate actual thresholds using explicit masks and math.fsum."""
    scores, gap = list(map(float, scores)), list(map(float, gap))
    require(len(scores) == len(gap) and len(scores) > 0, "Invalid calibration inputs")
    require(all(map(math.isfinite, scores + gap)), "Nonfinite calibration inputs")
    n = len(scores)
    candidates = [
        {"mode": "all_dense", "threshold": None, "selected_count": 0, "cal_gain": 0.0},
        {"mode": "all_bm25", "threshold": None, "selected_count": n,
         "cal_gain": math.fsum(gap) / n},
    ]
    for threshold in sorted(set(scores)):
        chosen = [i for i, score in enumerate(scores) if score > threshold]
        candidates.append({"mode": "threshold", "threshold": threshold,
                           "selected_count": len(chosen),
                           "cal_gain": math.fsum(gap[i] for i in chosen) / n})
    best = max(item["cal_gain"] for item in candidates)
    eligible = [item for item in candidates if best - item["cal_gain"] <= TIE_ATOL]
    selected = min(eligible, key=lambda item: (
        item["selected_count"], 0 if item["mode"] == "all_dense" else 1,
        -item["threshold"] if item["threshold"] is not None else 0.0))
    return {**selected, "best_gain": best, "candidate_count": len(candidates)}


def threshold_actions(scores, record):
    scores = np.asarray(scores)
    mode = record["mode"]
    if mode in ("all_dense", "all_bm25"):
        require(record["threshold"] is None, "Constant policy must have null threshold")
        return np.full(len(scores), mode == "all_bm25", dtype=bool)
    require(mode == "threshold", "Unknown threshold mode")
    threshold = record["threshold"]
    require(type(threshold) in (float, int) and math.isfinite(threshold), "Invalid finite threshold")
    return scores > threshold


def check_threshold(record, expected, name, errors):
    for key in ("mode", "selected_count", "candidate_count"):
        require(record[key] == expected[key], f"{name}.{key} differs")
    require(type(record["selected_count"]) is int and type(record["candidate_count"]) is int,
            f"{name}: threshold counts must be integers")
    if expected["threshold"] is None:
        require(record["threshold"] is None, f"{name}: endpoint must have null threshold")
    else:
        # Boundaries must be exactly the same stored native score, not a midpoint.
        require(record["threshold"] == expected["threshold"], f"{name}: different native score boundary")
    for key in ("cal_gain", "best_gain"):
        compare(record[key], expected[key], f"threshold_{key}", errors)


def policy_summary(utility, switches):
    gap = utility[:, 0] - utility[:, 1]
    n = len(gap)
    indices = np.flatnonzero(switches).tolist()
    return {
        "F1": math.fsum(float(utility[i, 0 if switches[i] else 1]) for i in range(n)) / n,
        "bm25_count": len(indices),
        "beneficial_count": sum(bool(gap[i] > TIE_ATOL) for i in indices),
        "harmful_count": sum(bool(gap[i] < -TIE_ATOL) for i in indices),
        "zero_count": sum(bool(abs(gap[i]) <= TIE_ATOL) for i in indices),
        "beneficial_mass": math.fsum(max(float(gap[i]), 0.0) for i in indices) / n,
        "harmful_mass": math.fsum(max(-float(gap[i]), 0.0) for i in indices) / n,
    }


def contributions(gap, actions):
    integer = {name: actions[name].astype(np.int64) for name in ("L0", "Lt", "R0", "Rt")}
    return np.column_stack((
        (integer["R0"] - integer["L0"]) * gap,
        (integer["Rt"] - integer["Lt"]) * gap,
        (integer["Rt"] - integer["R0"] - integer["Lt"] + integer["L0"]) * gap,
        integer["Rt"] * gap,
        (integer["Rt"] - 1) * gap,
    ))


def means(values):
    return np.asarray([math.fsum(map(float, values[:, column])) / len(values)
                       for column in range(values.shape[1])])


def frequency_intervals(values, groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    members = {}
    for index, group in enumerate(groups):
        members.setdefault(str(group), []).append(index)
    names = sorted(members)
    totals = np.asarray([[math.fsum(float(values[i, column]) for i in members[group])
                          for column in range(values.shape[1])] for group in names])
    sizes = np.asarray([len(members[group]) for group in names], dtype=np.int64)
    rng = np.random.default_rng(seed)
    sampled_means = np.empty((draws, values.shape[1]), dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(names), size=len(names))
        multiplicities = np.bincount(sampled, minlength=len(names))
        used = np.flatnonzero(multiplicities)
        weights = multiplicities[used]
        denominator = int(weights @ sizes[used])
        sampled_means[draw] = np.einsum("i,ij->j", weights, totals[used], optimize=False) / denominator
    return np.quantile(sampled_means, QUANTILES, axis=0, method="linear")


def load_inputs():
    historical = {}
    for name in ("layer_pooling_v1", "m6_pooled_offset_v1"):
        prior = read(RESEARCH / name / "completion_record.json")["artifact_sha256"]
        historical.update({resolve(path).resolve(): digest for path, digest in prior.items()})
    for name in ("features", "old_predictions", "old_cal_logits"):
        path = INPUTS[name].resolve()
        require(path in historical and sha(path) == historical[path], f"Historical binding differs: {name}")
    require(sha(INPUTS["folds"]) == EXPECTED_FOLDS_SHA256, "Historical split archive differs")
    with np.load(INPUTS["features"], allow_pickle=False) as archive:
        qids, groups, features = (archive[key].copy() for key in ("query_ids", "group_ids", "M6"))
    require(len(qids) == len(set(qids)) == 9600 and len(set(groups)) == 9559, "Wrong original pool")
    finite_array(features, (9600, 384), "original M6 features", np.float32)
    with np.load(INPUTS["old_predictions"], allow_pickle=False) as archive:
        require(np.array_equal(qids, archive["query_ids"]) and np.array_equal(groups, archive["group_ids"]),
                "Original feature/prediction identities differ")
        utility, old_scores = archive["utility"].copy(), archive["M6"].copy()
    finite_array(utility, (9600, 2), "original utility", np.float64)
    require(np.all((0 <= utility) & (utility <= 1)), "Invalid original utilities")
    finite_array(old_scores, (9600,), "original M6 OOF scores")
    with np.load(INPUTS["folds"], allow_pickle=False) as archive:
        folds = [{part: archive[f"fold{fold}_{part}"].copy()
                  for part in ("fit", "calibration", "test")} for fold in range(5)]
    with np.load(INPUTS["old_cal_logits"], allow_pickle=False) as archive:
        old_cal = []
        for fold in range(5):
            require(np.array_equal(folds[fold]["calibration"], archive[f"fold{fold}_cal_indices"]),
                    f"Original cal indices differ at fold {fold}")
            old_cal.append(archive[f"fold{fold}_O"].copy())
    coverage = np.zeros(9600, dtype=np.int64)
    for fold, parts in enumerate(folds):
        sets = []
        for part, size in (("fit", 6144), ("calibration", 1536), ("test", 1920)):
            ix = finite_array(parts[part], (size,), f"fold{fold} {part}")
            require(ix.dtype.kind in "iu" and len(set(ix)) == size and np.all((0 <= ix) & (ix < 9600)),
                    "Invalid original split indices")
            sets.append(set(map(str, groups[ix])))
        require(not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "Group leaks across split")
        require(len(set(np.concatenate(list(parts.values())))) == 9600, "Split does not partition original pool")
        coverage[parts["test"]] += 1
        finite_array(old_cal[fold], (1536,), f"fold{fold} original cal", np.float32)
        original_fit = RESEARCH / "layer_pooling_v1" / f"fold{fold}_M6_fit.npz"
        original_head = RESEARCH / "layer_pooling_v1" / f"fold{fold}_M6_head.pt"
        for path in (original_fit, original_head):
            require(path.resolve() in historical and sha(path) == historical[path.resolve()],
                    f"Historical M6 head/fit binding differs: {path}")
        with np.load(original_fit, allow_pickle=False) as archive:
            require(np.array_equal(parts["fit"], archive["fit_indices"]), "Original M6 fit identities differ")
            require(np.array_equal(inner_groups(groups[parts["fit"]], fold), archive["inner_assignment"]),
                    "Original M6 inner assignment differs")
    require(np.all(coverage == 1), "Original outer test coverage differs")
    return qids, groups, features, utility, old_scores, folds, old_cal


def check(output_dir):
    output_dir = Path(output_dir).resolve()
    destination = output_dir / "separate_checks.json"
    require(not destination.exists(), "Preserve existing separate_checks.json")
    required = [output_dir / name for name in ("protocol.json", "results.json", "predictions.npz",
                                              "fit_completion.json", "predictions_frozen.json")]
    required += [output_dir / f"fold{fold}.{extension}" for fold in range(5) for extension in ("json", "npz")]
    require(all(path.is_file() for path in required), "Wait for all completed results and five fold artifacts")
    started = time.perf_counter()
    protocol = read(output_dir / "protocol.json")
    source_bindings = bindings(protocol["source_sha256"], "source")
    input_bindings = bindings(protocol["input_sha256"], "input")
    expected_sources = [Path(__file__).resolve(), PROJECT / "scripts" / "run_m6_objective_readout.py",
                        PROJECT / "scripts" / "m6_objective_math.py",
                        PROJECT / "analysis" / "hotpotqa_router" / "m6_objective_readout_plan_20260915.md"]
    require(set(path.resolve() for path in expected_sources) == set(source_bindings), "Frozen source set differs")
    expected_inputs = list(INPUTS.values())
    expected_inputs += [RESEARCH / name / "completion_record.json"
                        for name in ("layer_pooling_v1", "m6_pooled_offset_v1")]
    expected_inputs += [RESEARCH / "layer_pooling_v1" / f"fold{fold}_M6_{suffix}"
                        for fold in range(5) for suffix in ("head.pt", "fit.npz")]
    require(set(path.resolve() for path in expected_inputs) == set(input_bindings), "Frozen input set differs")
    config = protocol["config"]
    require(config["regularizations"] == REGULARIZATIONS, "Regularization grid changed")
    require(config["bootstrap_seed"] == BOOTSTRAP_SEED and config["bootstrap_draws"] == BOOTSTRAP_DRAWS,
            "Bootstrap settings changed")
    require(config["interval_quantiles"] == list(QUANTILES), "Interval quantiles changed")
    require(config["tie_atol"] == TIE_ATOL and config["ridge_gradient_atol"] == GRADIENT_ATOL,
            "Numerical tolerances changed")
    require(config["minimum_recipe_increment"] == .002 and config["minimum_gain_over_both_fixed"] == .01,
            "Decision gates changed")
    require(protocol["primary"] == list(PRIMARY) and protocol["policies"] == list(POLICIES),
            "Frozen contrasts or policies differ")
    snapshots = {str(path): sha(path) for path in required}
    result = read(output_dir / "results.json")
    require(result.get("status") == "complete_fixed_development_comparison_pending_separate_check",
            "Wait for completed formal evaluation")
    require(result["protocol_sha256"] == snapshots[str(output_dir / "protocol.json")], "Result protocol differs")
    fit_completion = read(output_dir / "fit_completion.json")
    predictions_frozen = read(output_dir / "predictions_frozen.json")
    require(fit_completion["status"] == "all_five_heads_and_ten_thresholds_fixed_before_new_test_predictions",
            "Fit completion record differs")
    require(fit_completion["protocol_sha256"] == result["protocol_sha256"]
            and fit_completion["ridge_solves"] == 80 and fit_completion["thresholds"] == 10,
            "Fit completion binding or counts differ")
    fit_bindings = bindings(fit_completion["artifact_sha256"], "fit completion")
    expected_fits = {path.resolve() for path in required if path.name.startswith("fold")}
    require(set(fit_bindings) == expected_fits, "Fit completion artifact set differs")
    require(predictions_frozen["status"] == "complete_9600_OOF_actions_before_effects"
            and predictions_frozen["protocol_sha256"] == result["protocol_sha256"],
            "Prediction completion record differs")
    require(predictions_frozen["predictions_sha256"] == snapshots[str(output_dir / "predictions.npz")]
            and predictions_frozen["fit_completion_sha256"] == snapshots[str(output_dir / "fit_completion.json")],
            "Prediction completion artifact bindings differ")
    for key, expected in (("new_Ridge_solves", 80), ("new_thresholds", 10),
                          ("new_encoder_forwards", 0), ("new_external_calls", 0)):
        require(type(result[key]) is int and result[key] == expected, f"Result completion count differs: {key}")
    require(result["core_goal_achieved"] is False, "Development result cannot be a completed independent goal")
    errors = {}
    with threadpool_limits(limits=2):
        qids, groups, features, utility, old_scores, folds, old_cal = load_inputs()
        gap = utility[:, 0] - utility[:, 1]
        with np.load(output_dir / "predictions.npz", allow_pickle=False) as archive:
            predictions = {key: archive[key].copy() for key in archive.files}
        for key, expected in (("query_ids", qids), ("group_ids", groups), ("utility", utility), ("L_scores", old_scores)):
            require(np.array_equal(predictions[key], expected), f"Frozen prediction input differs: {key}")
        finite_array(predictions["R_scores"], (9600,), "Ridge OOF scores", np.float32)
        finite_array(predictions["fold_id"], (9600,), "OOF fold index", np.int64)
        for name in ("L0", "Lt", "R0", "Rt"):
            require(predictions[name].shape == (9600,) and predictions[name].dtype == bool, f"Invalid {name} actions")
        expected_fold = np.full(9600, -1, dtype=np.int64)
        rebuilt = {name: np.zeros(9600, dtype=bool) for name in ("L0", "Lt", "R0", "Rt")}
        threshold_checks = []
        selected_lambdas = []
        for fold, parts in enumerate(folds):
            fit, cal, test = (parts[key] for key in ("fit", "calibration", "test"))
            expected_fold[test] = fold
            info = read(output_dir / f"fold{fold}.json")
            require(type(info["fold"]) is int and info["fold"] == fold, "Fold record identity differs")
            with np.load(output_dir / f"fold{fold}.npz", allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in archive.files}
            coef = finite_array(saved["coef"], (16, 384), "Ridge coefficients", np.float64)
            intercept = finite_array(saved["intercept"], (16,), "Ridge intercepts", np.float64)
            assignment = inner_groups(groups[fit], fold)
            require(np.array_equal(saved["inner_assignment"], assignment), f"Fold {fold} inner group split differs")
            cv_native = finite_array(saved["cv_native"], (5, 6144), "native CV predictions", np.float32)
            fit_native = finite_array(saved["fit_native"], (6144,), "native fit predictions", np.float32)
            cal_native = finite_array(saved["cal_native"], (1536,), "native cal predictions", np.float32)
            mse = [math.fsum((float(score) - float(target)) ** 2
                            for score, target in zip(row, gap[fit])) / len(fit) for row in cv_native]
            compare(info["cv_mse"], mse, "cv_mse", errors)
            best = min(mse)
            selected = next(index for index, value in enumerate(mse) if value <= best + TIE_ATOL)
            require(type(info["selected_lambda_index"]) is int and info["selected_lambda_index"] == selected,
                    f"Fold {fold} lambda selection differs")
            selected_lambdas.append(REGULARIZATIONS[selected])
            require(len(info["solves"]) == 16, "Expected exactly 16 Ridge solve records per fold")
            for solve, record in enumerate(info["solves"]):
                if solve < 15:
                    lambda_index, inner = divmod(solve, 3)
                    training = fit[assignment != inner]
                    validation = fit[assignment == inner]
                    require(not set(groups[training]) & set(groups[validation]), "Inner group leakage")
                else:
                    lambda_index, training = selected, fit
                regularization = REGULARIZATIONS[lambda_index]
                require(record["regularization"] == regularization and record["accepted"] is True,
                        f"Fold {fold} solve {solve} rejected or regularization differs")
                require(record["lambda_index"] == lambda_index
                        and record["inner_split"] == (solve % 3 if solve < 15 else None)
                        and record["training_queries"] == len(training)
                        and record["role"] == ("inner" if solve < 15 else "refit"),
                        f"Fold {fold} solve {solve} identities differ")
                compare(record["intercept"], intercept[solve], "ridge_intercept", errors)
                observed = ridge_quantities(features[training], gap[training], coef[solve], intercept[solve], regularization)
                require(observed["grad_inf"] <= GRADIENT_ATOL, f"Independent gradient failed: fold {fold}, solve {solve}")
                for key in ("data_loss", "penalty", "loss", "grad_inf"):
                    compare(record[key], observed[key], f"ridge_{key}", errors,
                            GRADIENT_ATOL if key == "grad_inf" else NUMBER_ATOL)
            for key, scores in (("L", old_cal[fold]), ("R", cal_native)):
                expected = threshold_selection(scores, gap[cal])
                check_threshold(info["thresholds"][key], expected, f"fold{fold}_{key}", errors)
                threshold_checks.append({"fold": fold, "model": key, **expected})
            rebuilt["L0"][test] = old_scores[test] > 0
            rebuilt["Lt"][test] = threshold_actions(old_scores[test], info["thresholds"]["L"])
            rebuilt["R0"][test] = predictions["R_scores"][test] > 0
            rebuilt["Rt"][test] = threshold_actions(predictions["R_scores"][test], info["thresholds"]["R"])
            # These are diagnostics of saved native scores, not a CPU substitute for BF16 replay.
            native_mse = {
                "fit": math.fsum((float(s) - float(t)) ** 2 for s, t in zip(fit_native, gap[fit])) / len(fit),
                "cal": math.fsum((float(s) - float(t)) ** 2 for s, t in zip(cal_native, gap[cal])) / len(cal),
            }
            for key in native_mse:
                compare(info[f"{key}_mse"], native_mse[key], f"{key}_mse", errors)
        require(np.array_equal(predictions["fold_id"], expected_fold), "Outer fold identities differ")
        for name in rebuilt:
            require(np.array_equal(predictions[name], rebuilt[name]), f"Rebuilt actions differ: {name}")
        rebuilt["Dense"] = np.zeros(9600, dtype=bool)
        rebuilt["BM25"] = np.ones(9600, dtype=bool)
        summaries = {name: policy_summary(utility, rebuilt[name]) for name in POLICIES}
        for name, summary in summaries.items():
            for key, expected in summary.items():
                if key.endswith("count"):
                    require(type(result["policy"][name][key]) is int and result["policy"][name][key] == expected,
                            f"Policy count differs: {name}.{key}")
                else:
                    compare(result["policy"][name][key], expected, f"policy_{key}", errors)
        for name in ("L", "R"):
            gain = math.fsum(float(gap[i]) * (int(rebuilt[name + "t"][i]) - int(rebuilt[name + "0"][i]))
                             for i in range(len(gap))) / len(gap)
            compare(result["descriptive_calibration_gains"][name], gain, "calibration_gain", errors)
        effects = contributions(gap, rebuilt)
        primary_mean = means(effects)
        intervals = frequency_intervals(effects, groups)
        checked_primary = {}
        for column, name in enumerate(PRIMARY):
            compare(result["primary"][name]["mean"], primary_mean[column], "primary_mean", errors)
            compare(result["primary"][name]["interval"], intervals[:, column], "primary_interval", errors)
            checked_primary[name] = {"mean": float(primary_mean[column]), "interval": intervals[:, column].tolist()}
        reported_folds = result["folds"]
        require(isinstance(reported_folds, list) and len(reported_folds) == 5, "Expected five result folds")
        for fold, parts in enumerate(folds):
            expected = means(effects[parts["test"]])
            record = reported_folds[fold]
            original_record = read(output_dir / f"fold{fold}.json")
            require(all(record[key] == value for key, value in original_record.items()), "Result fold changed after fitting")
            test = parts["test"]
            test_mse = math.fsum((float(s) - float(t)) ** 2
                                 for s, t in zip(predictions["R_scores"][test], gap[test])) / len(test)
            compare(record["test_mse"], test_mse, "test_mse", errors)
            for column, name in enumerate(PRIMARY):
                compare(record["primary_means"][name], expected[column], "fold_primary_mean", errors)
        mechanism_support = bool(intervals[0, 1] > 0 and primary_mean[1] >= 0.002)
        candidate_support = bool(mechanism_support and np.all(intervals[0, 3:] > 0)
                                 and np.all(primary_mean[3:] >= 0.01))
        for key, expected in (("recipe_followup_gate", mechanism_support), ("candidate_preparation_gate", candidate_support)):
            require(result[key] is expected, f"Result decision differs: {key}")
    # Refuse to attest an experiment that changed while the check was running.
    bindings(protocol["source_sha256"], "source")
    bindings(protocol["input_sha256"], "input")
    require(all(sha(path) == digest for path, digest in snapshots.items()), "Formal artifacts changed during check")
    report = {
        "status": "passed_independent_objective_readout_checks",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checker_sha256": sha(__file__),
        "protocol_sha256": snapshots[str(output_dir / "protocol.json")],
        "results_sha256": snapshots[str(output_dir / "results.json")],
        "artifact_sha256": snapshots,
        "source_bindings_checked": len(source_bindings), "input_bindings_checked": len(input_bindings),
        "queries": 9600, "groups": 9559, "ridge_solutions_checked": 80,
        "threshold_selections_checked": 10, "selected_regularizations": selected_lambdas,
        "thresholds": threshold_checks, "primary": checked_primary,
        "maximum_absolute_errors": errors,
        "mechanism_support": mechanism_support, "candidate_gate": candidate_support,
        "independent_native_GPU_replay": False,
        "native_score_scope": "Saved native arrays checked for identity, shape, dtype, selection and policy arithmetic; GPU replay is separate.",
        "new_model_fits": 0, "new_paid_calls": 0,
        "scope": "Independent numerical implementation on the same consumed old pool; no new-sample or independent-team confirmation.",
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
