"""Fixed old-pool M6 loss-recipe x decision-threshold development experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

import m6_objective_math as math_core

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT.parent / "work" / "router_research"
OUT = PROJECT / "analysis" / "hotpotqa_router" / "m6_objective_readout_v1"
PLAN = OUT.parent / "m6_objective_readout_plan_20260915.md"
CONFIG = {"queries": 9600, "groups": 9559, "folds": 5,
          "fit_queries": 6144, "cal_queries": 1536, "test_queries": 1920,
          "regularizations": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
          "inner_folds": 3, "tie_atol": 1e-12, "ridge_gradient_atol": 1e-10,
          "batch_size": 8, "formal_ridge_solves": 80, "cal_thresholds": 10,
          "bootstrap_draws": 20000, "bootstrap_seed": 2026091501,
          "interval_quantiles": [.005, .995], "minimum_recipe_increment": .002,
          "minimum_gain_over_both_fixed": .01, "new_encoder_forwards": 0,
          "new_external_calls": 0, "cpu_threads": 2}
PRIMARY = ["R0_minus_L0", "Rt_minus_Lt", "calibration_interaction",
           "Rt_minus_Dense", "Rt_minus_BM25"]
POLICIES = ["L0", "Lt", "R0", "Rt", "Dense", "BM25"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def source_paths():
    return [Path(__file__).resolve(), PROJECT / "scripts" / "m6_objective_math.py",
            PROJECT / "scripts" / "check_m6_objective_readout.py", PLAN]


def input_paths():
    paths = [BASE / name for name in (
        "layer_pooling_v1/features.npz", "layer_pooling_v1/predictions.npz",
        "e02_results/fold_indices.npz", "m6_pooled_offset_v1/cal_logits.npz",
        "layer_pooling_v1/completion_record.json", "m6_pooled_offset_v1/completion_record.json")]
    paths += [BASE / f"layer_pooling_v1/fold{f}_M6_{suffix}" for f in range(5)
              for suffix in ("head.pt", "fit.npz")]
    return paths


def assignment(groups, fold):
    ordered = sorted(set(map(str, groups)), key=lambda g: (
        hashlib.sha256(f"lp_ft_inner_v1|fold={fold}|group={g}".encode()).digest(), g))
    bucket = {g: j % 3 for j, g in enumerate(ordered)}
    return np.array([bucket[str(g)] for g in groups], dtype=np.int64)


def data():
    prior = {}
    for name in ("layer_pooling_v1", "m6_pooled_offset_v1"):
        prior.update(read(BASE / name / "completion_record.json")["artifact_sha256"])
    for path in input_paths():
        if str(path) in prior:
            assert sha(path) == prior[str(path)], f"Prior input changed: {path.name}"
    assert sha(BASE / "e02_results/fold_indices.npz") == "ad2e94f76332f288f798d8bc1b1ff085be9aba64730b84d4906566d37ea76bc6"
    with np.load(BASE / "layer_pooling_v1/features.npz", allow_pickle=False) as z:
        x, qids, groups = (z[k].copy() for k in ("M6", "query_ids", "group_ids"))
    assert x.shape == (9600, 384) and x.dtype == np.float32 and np.isfinite(x).all()
    assert np.max(np.abs(np.linalg.norm(x.astype(float), axis=1) - 1)) < 2e-6
    assert len(set(qids)) == 9600 and len(set(groups)) == 9559
    with np.load(BASE / "layer_pooling_v1/predictions.npz", allow_pickle=False) as z:
        assert np.array_equal(qids, z["query_ids"]) and np.array_equal(groups, z["group_ids"])
        utility, old = z["utility"].copy(), z["M6"].copy()
    assert utility.shape == (9600, 2) and np.isfinite(utility).all()
    assert ((utility >= 0) & (utility <= 1)).all() and old.shape == (9600,) and np.isfinite(old).all()
    gap = utility[:, 0] - utility[:, 1]
    with np.load(BASE / "e02_results/fold_indices.npz", allow_pickle=False) as z:
        folds = [tuple(z[f"fold{f}_{key}"].copy() for key in ("fit", "calibration", "test")) for f in range(5)]
    cal_logits = []
    visits = np.zeros(9600, dtype=int)
    with np.load(BASE / "m6_pooled_offset_v1/cal_logits.npz", allow_pickle=False) as z:
        for f, (fit, cal, test) in enumerate(folds):
            assert [len(fit), len(cal), len(test)] == [6144, 1536, 1920]
            assert np.array_equal(np.sort(np.r_[fit, cal, test]), np.arange(9600))
            assert not (set(groups[fit]) & set(groups[cal]) or set(groups[fit]) & set(groups[test]) or set(groups[cal]) & set(groups[test]))
            visits[test] += 1
            assert np.array_equal(cal, z[f"fold{f}_cal_indices"])
            cal_logits.append(z[f"fold{f}_O"].copy())
            with np.load(BASE / f"layer_pooling_v1/fold{f}_M6_fit.npz", allow_pickle=False) as original:
                assert np.array_equal(fit, original["fit_indices"])
                assert np.array_equal(assignment(groups[fit], f), original["inner_assignment"])
    assert np.all(visits == 1)
    return {"features": x, "query_ids": qids, "group_ids": groups, "utility": utility,
            "gap": gap, "folds": folds, "L_scores": old, "L_cal": cal_logits}


def versions():
    return {name: importlib.metadata.version(name) for name in ("numpy", "torch", "threadpoolctl")}


def bound(binding):
    assert binding and sha(OUT / "protocol.json") == binding
    p = read(OUT / "protocol.json")
    assert p["config"] == CONFIG and p["primary"] == PRIMARY and p["versions"] == versions()
    for key in ("source_sha256", "input_sha256"):
        assert all(sha(path) == value for path, value in p[key].items()), key
    return p, data()


def gpu():
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return torch


def native(x, coef, intercept, torch):
    assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 384
    weight = torch.tensor(np.asarray(coef).reshape(1, 384), dtype=torch.float32, device="cuda")
    bias = torch.tensor([intercept], dtype=torch.float32, device="cuda")
    values = np.empty(len(x), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), 8):
            rows = torch.from_numpy(x[start:start + 8]).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = torch.nn.functional.linear(rows, weight, bias).squeeze(1)
            values[start:start + len(rows)] = logits.float().cpu().numpy()
    assert np.isfinite(values).all()
    return values


def old_replay(d, torch):
    errors = {}
    for f, (fit, cal, test) in enumerate(d["folds"]):
        head = torch.load(BASE / f"layer_pooling_v1/fold{f}_M6_head.pt", map_location="cpu", weights_only=True)
        coef, intercept = head["weight"].numpy().reshape(384), float(head["bias"].item())
        with np.load(BASE / f"layer_pooling_v1/fold{f}_M6_fit.npz", allow_pickle=False) as z:
            expected_fit = z["final_native_fit_logits"][:32]
        for name, ids, expected in (("fit", fit[:32], expected_fit),
                ("cal", cal[:32], d["L_cal"][f][:32]), ("test", test[:32], d["L_scores"][test[:32]])):
            observed = native(d["features"][ids], coef, intercept, torch)
            assert np.array_equal(observed, expected), f"Original native control differs: {f}/{name}"
            errors[f"fold{f}_{name}"] = 0.0
    return errors


def freeze():
    assert not (OUT / "protocol.json").exists()
    d = data()
    OUT.mkdir(exist_ok=True)
    value = {"status": "frozen_before_new_Ridge_fits_and_cal_thresholds", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "config": CONFIG, "primary": PRIMARY, "policies": POLICIES, "versions": versions(),
             "source_sha256": {str(path): sha(path) for path in source_paths()},
             "input_sha256": {str(path): sha(path) for path in input_paths()},
             "scope": "Consumed old9600 development; no independent source confirmation or deployment",
             "new_BEIR_policy_effects": False, "new_external_calls": 0,
             "identity": {"queries": len(d["query_ids"]), "groups": len(set(d["group_ids"])), "feature_shape": list(d["features"].shape)}}
    write(OUT / "protocol.json", value)
    print(json.dumps({"status": value["status"], "protocol_sha256": sha(OUT / "protocol.json")}), flush=True)


def pilot(binding):
    _p, d = bound(binding)
    assert not (OUT / "pilot.json").exists() and not (OUT / "fit_started.json").exists()
    torch = gpu()
    controls = old_replay(d, torch)
    fit = d["folds"][0][0][:128]
    with threadpool_limits(limits=2):
        model = math_core.fit_ridge(d["features"][fit], d["gap"][fit], .001)
    save(OUT / "pilot_model.npz", coef=model["coef"], intercept=np.array(model["intercept"]))
    assert model["accepted"]
    scores = native(d["features"][fit], model["coef"], model["intercept"], torch)
    assert scores.shape == (128,)
    value = {"status": "passed_original_native_replay_and_128row_Ridge_pilot", "protocol_sha256": binding,
             "original_controls": controls, "original_native_control_queries": 480,
             "ridge_solves": 1, "ridge_gradient": model["grad_inf"], "pilot_rows": 128,
             "model_sha256": sha(OUT / "pilot_model.npz"), "new_policy_effects": False,
             "new_encoder_forwards": 0, "new_external_calls": 0}
    write(OUT / "pilot.json", value)
    print(json.dumps(value), flush=True)


def effects(quality):
    return np.column_stack([quality["R0"] - quality["L0"], quality["Rt"] - quality["Lt"],
        (quality["Rt"] - quality["R0"]) - (quality["Lt"] - quality["L0"]),
        quality["Rt"] - quality["Dense"], quality["Rt"] - quality["BM25"]])


def bootstrap(groups, values):
    _, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    totals = np.column_stack([np.bincount(inverse, weights=values[:, j]) for j in range(5)])
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    samples = np.empty((CONFIG["bootstrap_draws"], 5))
    for start in range(0, len(samples), 100):
        take = rng.integers(len(sizes), size=(min(100, len(samples) - start), len(sizes)))
        samples[start:start + len(take)] = totals[take].sum(axis=1) / sizes[take].sum(axis=1)[:, None]
    return np.quantile(samples, CONFIG["interval_quantiles"], axis=0).T


def policy_summary(action, utility, gap):
    return {"F1": float(np.where(action, utility[:, 0], utility[:, 1]).mean()),
        "bm25_count": int(action.sum()), "beneficial_count": int(np.sum(action & (gap > 1e-12))),
        "harmful_count": int(np.sum(action & (gap < -1e-12))),
        "zero_count": int(np.sum(action & (np.abs(gap) <= 1e-12))),
        "beneficial_mass": float(np.where(action, np.maximum(gap, 0), 0).mean()),
        "harmful_mass": float(np.where(action, np.maximum(-gap, 0), 0).mean())}


def run(binding):
    p, d = bound(binding)
    checked = read(OUT / "pilot.json")
    assert checked["status"] == "passed_original_native_replay_and_128row_Ridge_pilot" and checked["protocol_sha256"] == binding
    assert not (OUT / "fit_started.json").exists() and not (OUT / "results.json").exists()
    write(OUT / "fit_started.json", {"protocol_sha256": binding, "started_at_unix": time.time()})
    started = time.perf_counter()
    torch = gpu()
    x, gap = d["features"], d["gap"]
    models, fold_records = [], []
    with (OUT / "solutions.jsonl").open("x", encoding="utf-8", newline="\n") as journal, threadpool_limits(limits=2):
        for f, (fit, cal, _test) in enumerate(d["folds"]):
            buckets = assignment(d["group_ids"][fit], f)
            params, offsets, solves = [], [], []
            cv = np.full((5, len(fit)), np.nan, dtype=np.float32)

            def solve(indices, index, inner):
                model = math_core.fit_ridge(x[indices], gap[indices], CONFIG["regularizations"][index])
                row = {k: value for k, value in model.items() if k != "coef"}
                row.update(regularization=CONFIG["regularizations"][index], lambda_index=index,
                           inner_split=inner, training_queries=len(indices), role="refit" if inner is None else "inner")
                journal.write(json.dumps({"fold": f, "solution_index": len(solves), "coef": model["coef"].tolist(), **row}, allow_nan=False) + "\n")
                journal.flush()
                params.append(model["coef"])
                offsets.append(model["intercept"])
                solves.append(row)
                assert model["accepted"], f"Ridge solution rejected in fold{f}; preserve journal"
                return model

            for index in range(5):
                for inner in range(3):
                    valid = buckets == inner
                    model = solve(fit[~valid], index, inner)
                    cv[index, valid] = native(x[fit[valid]], model["coef"], model["intercept"], torch)
            assert np.isfinite(cv).all()
            cv_mse = np.mean((cv.astype(float) - gap[fit][None, :]) ** 2, axis=1)
            selected = int(np.flatnonzero(cv_mse <= cv_mse.min() + 1e-12)[0])
            model = solve(fit, selected, None)
            fit_native = native(x[fit], model["coef"], model["intercept"], torch)
            cal_native = native(x[cal], model["coef"], model["intercept"], torch)
            thresholds = {"L": math_core.select_threshold(d["L_cal"][f], gap[cal]),
                          "R": math_core.select_threshold(cal_native, gap[cal])}
            record = {"fold": f, "selected_lambda_index": selected, "cv_mse": cv_mse.tolist(),
                      "solves": solves, "thresholds": thresholds,
                      "fit_mse": float(np.mean((fit_native.astype(float) - gap[fit]) ** 2)),
                      "cal_mse": float(np.mean((cal_native.astype(float) - gap[cal]) ** 2))}
            save(OUT / f"fold{f}.npz", coef=np.stack(params), intercept=np.array(offsets),
                 inner_assignment=buckets, cv_native=cv, fit_native=fit_native, cal_native=cal_native)
            write(OUT / f"fold{f}.json", record)
            fold_records.append(record)
            models.append(model)
            print(json.dumps({"status": "fold_fit_and_cal_complete_no_new_test_effects", "fold": f,
                              "completed_ridge_solves": (f + 1) * 16, "selected_lambda": CONFIG["regularizations"][selected]}), flush=True)
    assert len(models) == 5
    paths = [OUT / f"fold{f}.{suffix}" for f in range(5) for suffix in ("npz", "json")]
    write(OUT / "fit_completion.json", {"status": "all_five_heads_and_ten_thresholds_fixed_before_new_test_predictions",
         "protocol_sha256": binding, "ridge_solves": 80, "thresholds": 10,
         "artifact_sha256": {str(path): sha(path) for path in paths}})
    bound(binding)
    r_scores = np.full(9600, np.nan, dtype=np.float32)
    actions = {"L0": d["L_scores"] > 0, "Lt": np.zeros(9600, bool), "R0": np.zeros(9600, bool), "Rt": np.zeros(9600, bool)}
    fold_id = np.full(9600, -1, dtype=np.int64)
    for f, (_fit, _cal, test) in enumerate(d["folds"]):
        model = models[f]
        r_scores[test] = native(x[test], model["coef"], model["intercept"], torch)
        actions["R0"][test] = r_scores[test] > 0
        for name, scores in (("L", d["L_scores"][test]), ("R", r_scores[test])):
            actions[name + "t"][test] = math_core.apply_threshold(scores, fold_records[f]["thresholds"][name])
        fold_id[test] = f
    assert np.isfinite(r_scores).all() and np.all(fold_id >= 0)
    save(OUT / "predictions.npz", query_ids=d["query_ids"], group_ids=d["group_ids"], utility=d["utility"],
         L_scores=d["L_scores"], R_scores=r_scores, fold_id=fold_id, **actions)
    write(OUT / "predictions_frozen.json", {"status": "complete_9600_OOF_actions_before_effects",
         "protocol_sha256": binding, "predictions_sha256": sha(OUT / "predictions.npz"),
         "fit_completion_sha256": sha(OUT / "fit_completion.json")})
    actions.update(Dense=np.zeros(9600, bool), BM25=np.ones(9600, bool))
    quality = {name: np.where(action, d["utility"][:, 0], d["utility"][:, 1]) for name, action in actions.items()}
    values = effects(quality)
    with threadpool_limits(limits=2):
        ci = bootstrap(d["group_ids"], values)
    primary = {name: {"mean": float(values[:, j].mean()), "interval": ci[j].tolist()} for j, name in enumerate(PRIMARY)}
    recipe = primary["Rt_minus_Lt"]["interval"][0] > 0 and primary["Rt_minus_Lt"]["mean"] >= .002
    candidate = recipe and all(primary[f"Rt_minus_{fixed}"]["interval"][0] > 0 and
        primary[f"Rt_minus_{fixed}"]["mean"] >= .01 for fixed in ("Dense", "BM25"))
    for f, (_, _, test) in enumerate(d["folds"]):
        fold_records[f]["primary_means"] = {name: float(values[test, j].mean()) for j, name in enumerate(PRIMARY)}
        fold_records[f]["test_mse"] = float(np.mean((r_scores[test].astype(float) - gap[test]) ** 2))
    result = {"status": "complete_fixed_development_comparison_pending_separate_check", "protocol_sha256": binding,
        "primary": primary, "policy": {name: policy_summary(actions[name], d["utility"], gap) for name in POLICIES},
        "descriptive_calibration_gains": {name: float((quality[name + "t"] - quality[name + "0"]).mean()) for name in ("L", "R")},
        "recipe_followup_gate": bool(recipe), "candidate_preparation_gate": bool(candidate), "folds": fold_records,
        "new_Ridge_solves": 80, "new_thresholds": 10, "new_encoder_forwards": 0, "new_external_calls": 0,
        "elapsed_seconds": time.perf_counter() - started, "core_goal_achieved": False, "scope": p["scope"]}
    bound(binding)
    write(OUT / "results.json", result)
    print(json.dumps({"status": result["status"], "primary": primary, "candidate_preparation_gate": bool(candidate)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "pilot", "run"))
    parser.add_argument("--protocol-sha")
    args = parser.parse_args()
    try:
        if args.mode == "freeze":
            freeze()
        elif args.mode == "pilot":
            pilot(args.protocol_sha)
        else:
            run(args.protocol_sha)
    except Exception:
        failure = OUT / (args.mode + "_failure.json")
        if OUT.exists() and not failure.exists():
            write(failure, {"protocol_sha256": args.protocol_sha, "traceback": traceback.format_exc()})
        raise
