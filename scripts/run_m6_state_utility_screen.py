"""Bounded cached-query readout experiment; unchanged retrieval/generation.

Three fits, one consumed development fold, no encoder or new answer calls.
Existing atomic checkpoint/Adam recovery is reused from the capability screen.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from run_m6_capability_screen import require_execution_allowed, save_arrays, stage_fit, write_json


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "work/router_research"
OUT = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/m6_state_utility_screen_v1"
TEACHER = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_9600_v1/privileged_teacher"
SPEC = {
    "scope": "consumed_fold0_development_only", "fit_queries": 6144, "cal_queries": 1536,
    "features": "frozen_M6_384D_unit_norm_query_only", "arms": ["D", "M", "S"],
    "state_order": ["neither_full", "dense_only_full", "bm25_only_full", "both_full"],
    "state_rule": "2 * (bm25_coverage_at5 == 1) + (dense_coverage_at5 == 1)",
    "soft_target": "[positive_mean3_gap, negative_mean3_gap, 1-abs(mean3_gap)]",
    "optimizer": "Adam", "lr": .003, "betas": [.9, .999], "adam_eps": 1e-8,
    "steps_per_arm": 1000, "batch": "all_fit_rows", "weight_l2": .001, "bias_l2": 0,
    "clip_norm": 1.0, "checkpoint_every_steps": 64, "dtype": "float32", "device": "cuda",
    "initial_weight_std": .02, "initial_prior_eps": 1e-6,
    "initialization": "same M/S parameters; zero gate biases and asymmetric random weights; expert bias from global own-fit soft-target prior",
    "model_seed": 2026091823, "order_seed": 2026091824,
    "empty_state_constant": "own_fit_global_mean_gap", "threshold": 0,
    "minimum_fixed_gain": .01, "minimum_control_gain": .002,
    "simple_candidate_preference": ["D", "C", "M", "S"],
    "simple_candidate_tolerance": .002,
    "numerical_activity_floor": 1e-8, "expert_output_dispersion_floor": 1e-6,
    "new_encoder_forwards": 0, "new_retrieval_calls": 0, "new_answer_calls": 0,
    "outer_test_evaluations": 0, "external_source_evaluations": 0,
}


def soft_targets(gap):
    return torch.stack([gap.clamp_min(0), (-gap).clamp_min(0), 1-gap.abs()], dim=-1)


class StateUtility(nn.Module):
    def __init__(self, dimension, arm, prior):
        super().__init__()
        if arm not in ("D", "M", "S"):
            raise ValueError(arm)
        self.arm = arm
        if arm == "D":
            self.direct = nn.Linear(dimension, 3)
        else:
            self.gate = nn.Linear(dimension, 4)
            self.expert = nn.Linear(dimension, 12)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith("weight"):
                    parameter.normal_(std=SPEC["initial_weight_std"])
                else:
                    parameter.zero_()
            bias = prior.clamp_min(SPEC["initial_prior_eps"]).log()
            bias = bias - bias.mean()
            if arm == "D":
                self.direct.bias.copy_(bias)
            else:
                self.expert.bias.copy_(bias.repeat(4))

    def distributions(self, x):
        if self.arm == "D":
            return F.log_softmax(self.direct(x), dim=-1), None, None
        gate = F.log_softmax(self.gate(x), dim=-1)
        expert = F.log_softmax(self.expert(x).reshape(-1, 4, 3), dim=-1)
        mixed = torch.logsumexp(gate[:, :, None] + expert, dim=1)
        return mixed, gate, expert

    def forward(self, x):
        probability = self.distributions(x)[0].exp()
        return probability[:, 0] - probability[:, 1]


def supervised_loss(model, x, target, state):
    mixed, gate, expert = model.distributions(x)
    if model.arm == "S":
        rows = torch.arange(len(x), device=x.device)
        return (-gate[rows, state] - (target * expert[rows, state]).sum(-1)).mean()
    return -(target * mixed).sum(-1).mean()


def weight_penalty(model):
    return sum(p.square().sum() for name, p in model.named_parameters() if name.endswith("weight")) / 2


def input_paths():
    return [BASE / "layer_pooling_v1/features.npz", BASE / "layer_pooling_v1/predictions.npz",
            BASE / "e02_results/fold_indices.npz", BASE / "m6_pooled_offset_v1/cal_logits.npz",
            TEACHER / "feature_schema.json", TEACHER / "teacher_features.npz"]


def load_data():
    with np.load(BASE / "e02_results/fold_indices.npz", allow_pickle=False) as archive:
        fit, cal = archive["fold0_fit"].copy(), archive["fold0_calibration"].copy()
    assert (len(fit), len(cal)) == (SPEC["fit_queries"], SPEC["cal_queries"])
    selected = np.r_[fit, cal]
    assert len(np.unique(selected)) == len(selected)
    with np.load(BASE / "layer_pooling_v1/features.npz", allow_pickle=False) as archive:
        qids, groups = archive["query_ids"], archive["group_ids"]
        x = archive["M6"][selected].copy()
    assert not set(groups[fit]) & set(groups[cal])
    assert x.shape == (len(selected), 384) and x.dtype == np.float32 and np.isfinite(x).all()
    with np.load(BASE / "layer_pooling_v1/predictions.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["query_ids"], qids) and np.array_equal(archive["group_ids"], groups)
        utility = archive["utility"][selected].copy()
    assert utility.shape == (len(selected), 2) and np.isfinite(utility).all() and ((utility >= 0) & (utility <= 1)).all()
    schema = json.loads((TEACHER / "feature_schema.json").read_text(encoding="utf-8"))["gold"]["feature_names"]
    columns = [schema.index(f"gold__{action}_coverage_at_5") for action in ("bm25", "dense")]
    with np.load(TEACHER / "teacher_features.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["query_ids"], qids) and np.array_equal(archive["group_ids"], groups)
        coverage = archive["gold"][selected][:, columns]
    assert np.isfinite(coverage).all() and ((coverage >= 0) & (coverage <= 1)).all()
    state = 2*(coverage[:, 0] == 1).astype(np.int64) + (coverage[:, 1] == 1).astype(np.int64)
    with np.load(BASE / "m6_pooled_offset_v1/cal_logits.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["fold0_cal_indices"], cal)
        original_cal = archive["fold0_O"].copy()
    assert original_cal.shape == (len(cal),) and np.isfinite(original_cal).all()
    return {"x": x, "utility": utility, "gap": utility[:, 0]-utility[:, 1], "state": state,
            "query_ids": qids[selected], "group_ids": groups[selected], "fit_indices": fit,
            "cal_indices": cal, "M6": original_cal}


def diagnostics(model, x, target, state, initial):
    model.zero_grad(set_to_none=True)
    loss = supervised_loss(model, x, target, state)
    total = loss + SPEC["weight_l2"] * weight_penalty(model)
    total.backward()
    result = {"supervised_loss": float(loss.detach()), "penalized_loss": float(total.detach()),
              "gradient_norm": float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))),
              "parameter_change_max": max(float((p.detach()-initial[name]).abs().max()) for name, p in model.named_parameters())}
    with torch.no_grad():
        if model.arm != "D":
            _, gate, expert = model.distributions(x)
            result.update(gate_weight_change_max=float((model.gate.weight-initial["gate.weight"]).abs().max()),
                          expert_output_dispersion=float(expert.exp().std(dim=1, unbiased=False).mean()),
                          gate_entropy=float(-(gate.exp()*gate).sum(-1).mean()),
                          expert_weight_dispersion=float(model.expert.weight.reshape(4, 3, -1).std(dim=0, unbiased=False).mean()))
    return result


def policy_summary(score, utility):
    action = score > 0
    gap = utility[:, 0]-utility[:, 1]
    return {"F1": float(np.where(action, utility[:, 0], utility[:, 1]).mean()),
            "bm25_count": int(action.sum()), "beneficial_count": int((action & (gap > 1e-12)).sum()),
            "harmful_count": int((action & (gap < -1e-12)).sum()),
            "beneficial_mass": float(np.where(action, np.maximum(gap, 0), 0).mean()),
            "harmful_mass": float(np.where(action, np.maximum(-gap, 0), 0).mean())}


def run():
    require_execution_allowed()
    if not torch.cuda.is_available():
        raise RuntimeError("Current CUDA environment is required; do not silently change device")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    config_path = ROOT / "outputs/router/hotpotqa_bd_router_v1/config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    binding = {"spec": SPEC, "base_config": str(config_path),
               "environment": {key: config[key] for key in ("dataset", "artifacts", "retrieval", "context", "prompt")},
               "generation": {key: config["generation"][key] for key in ("provider", "model", "temperature", "max_output_tokens")},
               "measurement": "existing mean3 normalized-token F1; not semantic Answer Correctness",
               "input_file_metadata": {str(path): {"bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns} for path in input_paths()},
               "code_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                               for name in (Path(__file__).name, "run_m6_capability_screen.py")},
               "torch_version": str(torch.__version__)}
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "protocol.json"
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8"))["binding"] != binding:
            raise ValueError("Existing run has a different recipe or implementation")
    else:
        write_json(protocol, {"binding": binding, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    if (OUT / "results.json").exists():
        print((OUT / "results.json").read_text(encoding="utf-8"))
        return
    data = load_data()
    nfit = SPEC["fit_queries"]
    x = torch.as_tensor(data["x"], device="cuda")
    target = soft_targets(torch.as_tensor(data["gap"], device="cuda", dtype=torch.float32))
    state = torch.as_tensor(data["state"], device="cuda")
    prior = target[:nfit].mean(0).cpu()
    constants = np.array([data["gap"][:nfit][data["state"][:nfit] == value].mean()
                          if np.any(data["state"][:nfit] == value) else data["gap"][:nfit].mean() for value in range(4)])
    scores, training, initial_mixture = {"M6": data["M6"]}, {}, None
    for arm in SPEC["arms"]:
        torch.manual_seed(SPEC["model_seed"])
        model = StateUtility(384, arm, prior).cuda()
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        if arm == "M":
            initial_mixture = {name: p.cpu() for name, p in initial.items()}
        if arm == "S":
            assert all(torch.equal(value.cpu(), initial_mixture[name]) for name, value in initial.items())
        initial_diag = diagnostics(model, x[:nfit], target[:nfit], state[:nfit], initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=SPEC["lr"], betas=tuple(SPEC["betas"]), eps=SPEC["adam_eps"], foreach=False)

        def objective(ids):
            return supervised_loss(model, x[ids], target[ids], state[ids]) + SPEC["weight_l2"] * weight_penalty(model)

        training[arm] = stage_fit(model, optimizer, np.arange(nfit), SPEC["steps_per_arm"], nfit, objective,
                                  OUT / f"model_{arm}.pt", order_seed=SPEC["order_seed"])
        training[arm].update(parameters=sum(p.numel() for p in model.parameters()), initial=initial_diag,
                             final=diagnostics(model, x[:nfit], target[:nfit], state[:nfit], initial))
        with torch.inference_mode():
            scores[arm] = model(x[nfit:]).cpu().numpy()
            if arm == "S":
                mixed, gate, expert = model.distributions(x[nfit:])
                gate_probability = gate.exp().cpu().numpy()
                scores["C"] = gate_probability @ constants
                state_diagnostic = {"fit_counts": np.bincount(data["state"][:nfit], minlength=4).tolist(),
                                    "cal_counts": np.bincount(data["state"][nfit:], minlength=4).tolist(),
                                    "cal_state_accuracy": float((gate.argmax(-1) == state[nfit:]).float().mean()),
                                    "cal_state_cross_entropy": float(-gate[torch.arange(len(gate), device="cuda"), state[nfit:]].mean()),
                                    "own_fit_state_gap_constants": constants.tolist()}
                save_arrays(OUT / "state_calibration_outputs.npz", query_ids=data["query_ids"][nfit:],
                            gate_probability=gate_probability, expert_probability=expert.exp().cpu().numpy())
    assert all(np.isfinite(value).all() for value in scores.values())
    utility = data["utility"][nfit:]
    summaries = {name: policy_summary(value, utility) for name, value in scores.items()}
    quality = {name: value["F1"] for name, value in summaries.items()}
    quality.update(BM25=float(utility[:, 0].mean()), Dense=float(utility[:, 1].mean()))
    numeric = {name: item["final"]["penalized_loss"] < item["initial"]["penalized_loss"]-SPEC["numerical_activity_floor"] for name, item in training.items()}
    mfinal = training["M"]["final"]
    active_control = numeric["M"] and mfinal["gate_weight_change_max"] > SPEC["numerical_activity_floor"] and mfinal["expert_output_dispersion"] > SPEC["expert_output_dispersion_floor"]
    control_gain = min(quality["S"]-quality[name] for name in ("D", "M", "C", "M6"))
    fixed_gain = quality["S"]-max(quality["BM25"], quality["Dense"])
    mechanism_gate = all(numeric.values()) and active_control and control_gain >= SPEC["minimum_control_gain"] and fixed_gain >= SPEC["minimum_fixed_gain"]
    eligible = [name for name in SPEC["simple_candidate_preference"] if numeric["S" if name == "C" else name]
                and quality[name]-quality["M6"] >= SPEC["minimum_control_gain"]
                and quality[name]-max(quality["BM25"], quality["Dense"]) >= SPEC["minimum_fixed_gain"]]
    best = max((quality[name] for name in eligible), default=None)
    candidate = next((name for name in eligible if best-quality[name] <= SPEC["simple_candidate_tolerance"]), None)
    result = {"status": "complete_consumed_calibration_screen", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "fit_queries": nfit, "cal_queries": len(utility), "mean3_answer_f1": quality,
              "policy_summaries": summaries, "state_supervision_development_gate": mechanism_gate,
              "S_minimum_control_gain": control_gain, "S_gain_over_best_fixed": fixed_gain,
              "numerical_objectives_decreased": numeric, "M_numerically_noncollapsed": active_control,
              "candidate_for_further_planning": candidate,
              "decision": "promising_development_only_external_protocol_required" if candidate else "stop_this_recipe_no_seed_epoch_threshold_state_extension",
              "independent_gain_established": False, "training": training, "state_diagnostic_only": state_diagnostic}
    save_arrays(OUT / "calibration_scores.npz", indices=data["cal_indices"], query_ids=data["query_ids"][nfit:], **scores)
    write_json(OUT / "results.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        run()
    else:
        print(json.dumps({"mode": "describe_only", "spec": SPEC}, indent=2))
