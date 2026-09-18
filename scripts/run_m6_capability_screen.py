"""One bounded, old-pool development screen in the existing RAG environment.

Default invocation only describes the proposed run. --execute respects the
registry's user pause before reading research arrays or loading a pretrained
model. There are no retrieval, generation, network, or outer-test operations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
                  USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from router_capability_fusion import CapabilityRouter, UtilityFusion

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "work/router_research"
OUT = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/m6_capability_screen_v1"
MODEL = Path("C:/Users/12442/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
SPEC = {
    "scope": "single_old_fold_development_only_no_independent_confirmation",
    "fold": 0, "depth": 6, "bottleneck": 16,
    "branch_epochs": 2, "branch_batch": 8,
    "fusion_epochs": 10, "fusion_batch": 128, "inference_batch": 8,
    "adapter_lr": 0.0005, "head_lr": 0.001, "fusion_lr": 0.001,
    "adapter_weight_decay": 0.01, "head_l2": 0.001, "fusion_l2": 0.001,
    "clip_norm": 1.0, "checkpoint_every_steps": 64,
    "model_seed": 2026091817, "order_seed": 2026091818, "fusion_seed": 2026091819,
    "branch_tasks": ["utility", "quality", "aux_utility"],
    "readouts": ["U", "RU_static", "RU_conditional", "UU_conditional", "M6_gated"],
    "minimum_control_increment": 0.002, "minimum_fixed_gain": 0.01,
    "new_outer_test_evaluations": 0, "new_generation_calls": 0,
}


def execution_state():
    path = ROOT / "analysis/hotpotqa_router/registry.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["research_continuation"]["experiment_execution"]


def require_execution_allowed():
    if execution_state() == "paused_by_user":
        raise RuntimeError("Experiments remain paused by the user; preparation cannot authorize execution.")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_arrays(path, **values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def stage_fit(module, optimizer, indices, epochs, batch_size, objective, checkpoint, *, order_seed):
    """Resume from a batch boundary with trainable weights AND Adam/RNG state.

    NumPy order is reconstructed from (order_seed + epoch); no global NumPy or
    Python RNG is used. The caller binds the code, recipe and data before reuse.
    """
    parameters = {name: value for name, value in module.named_parameters() if value.requires_grad}
    batches = (len(indices) + batch_size - 1) // batch_size
    total_steps = epochs * batches
    next_step, elapsed, trace = 0, 0.0, []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if list(saved["parameters"]) != list(parameters) or saved["total_steps"] != total_steps:
            raise ValueError("Checkpoint does not match this stage")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(saved["parameters"][name])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"])
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        next_step, elapsed, trace = saved["next_step"], saved["elapsed_seconds"], saved["trace"]
    started = time.monotonic()

    def save(step):
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({
            "parameters": {name: value.detach().cpu().clone() for name, value in parameters.items()},
            "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if any(value.is_cuda for value in parameters.values()) else [],
            "next_step": step, "total_steps": total_steps,
            "elapsed_seconds": elapsed + time.monotonic() - started, "trace": trace,
        }, temporary)
        temporary.replace(checkpoint)

    for epoch in range(next_step // batches, epochs):
        order = np.random.default_rng(order_seed + epoch).permutation(indices)
        first_batch = next_step % batches if epoch == next_step // batches else 0
        for batch in range(first_batch, batches):
            selected = order[batch * batch_size:(batch + 1) * batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(selected)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(list(parameters.values()), SPEC["clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            completed = epoch * batches + batch + 1
            if completed % SPEC["checkpoint_every_steps"] == 0 or completed == total_steps:
                trace.append({"step": completed, "batch_loss": float(loss.detach()), "grad_norm": float(norm)})
                save(completed)
                print(json.dumps({"stage": checkpoint.stem, **trace[-1]}), flush=True)
    return {"steps": total_steps, "elapsed_seconds": elapsed + time.monotonic() - started, "trace": trace}


def load_data():
    import run_m6_objective_readout as previous
    data = previous.data()  # Reuse the established alignment/group-split loader.
    with np.load(BASE / "layer_pooling_v1/tokens.npz", allow_pickle=False) as archive:
        data["tokens"] = {key: archive[key].copy() for key in archive.files}
    path = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_9600_v1/privileged_teacher"
    schema = json.loads((path / "feature_schema.json").read_text(encoding="utf-8"))["gold"]["feature_names"]
    columns = [schema.index(f"gold__{action}_coverage_at_5") for action in ("bm25", "dense")]
    with np.load(path / "teacher_features.npz", allow_pickle=False) as archive:
        if not (np.array_equal(archive["query_ids"], data["query_ids"])
                and np.array_equal(archive["group_ids"], data["group_ids"])):
            raise ValueError("Coverage targets do not align with utility targets")
        data["coverage"] = archive["gold"][:, columns].astype(np.float32)
    if not (np.isfinite(data["coverage"]).all() and ((data["coverage"] >= 0) & (data["coverage"] <= 1)).all()):
        raise ValueError("Coverage targets must lie in [0, 1]")
    return data


def tokens(data, indices):
    return {key: torch.as_tensor(values[indices], device="cuda") for key, values in data["tokens"].items()}


def utility_loss(logits, gaps, normalizer):
    gaps = torch.as_tensor(gaps, dtype=torch.float32, device=logits.device)
    weights = torch.where(gaps.abs() > 1e-12, gaps.abs(), 0.0)
    return (weights * F.binary_cross_entropy_with_logits(logits.float(), (gaps > 0).float(), reduction="none")).mean() / normalizer


def weight_penalty(module):
    return sum(parameter.float().square().sum() for name, parameter in module.named_parameters()
               if parameter.requires_grad and name.endswith("weight")) / 2


def make_model():
    from transformers import AutoModel
    torch.manual_seed(SPEC["model_seed"])
    bert = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=True)
    head = nn.Linear(384, 1)
    head.load_state_dict(torch.load(BASE / f"layer_pooling_v1/fold{SPEC['fold']}_M6_head.pt", weights_only=True))
    return CapabilityRouter(bert, head, depth=SPEC["depth"], bottleneck=SPEC["bottleneck"]).cuda().eval()


def encode_branch(model, data, ids, branch):
    values = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(ids), SPEC["inference_batch"]):
            values.append(model.encode(**tokens(data, ids[start:start + SPEC["inference_batch"]]), branch=branch).float().cpu().numpy())
    return np.concatenate(values)


def learn_branch(task, data, fit, selected, normalizer):
    model = make_model()
    initial_difference = None
    if task == "utility" and not (OUT / f"branch_{task}.pt").exists():
        reference = encode_branch(model, data, fit[:8], branch=None)
        initial_difference = float(np.max(np.abs(reference - data["features"][fit[:8]])))
        if initial_difference > 2e-5:
            raise ValueError("New BERT path disagrees with cached M6 features on the first fit batch")
    branch = "utility" if task == "utility" else "auxiliary"
    model.set_stage(branch)
    head = getattr(model, f"{branch}_head")
    optimizer = torch.optim.AdamW([
        {"params": model.adapters[branch].parameters(), "lr": SPEC["adapter_lr"], "weight_decay": SPEC["adapter_weight_decay"]},
        {"params": head.parameters(), "lr": SPEC["head_lr"], "weight_decay": 0.0},
    ], foreach=False)

    def objective(ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**tokens(data, ids), branch=branch).float()
            if task == "quality":
                targets = torch.as_tensor(data["coverage"][ids], device="cuda")
                supervised = F.binary_cross_entropy_with_logits(logits, targets)
            else:
                if task == "aux_utility":
                    logits = logits[:, 0] - logits[:, 1]
                supervised = utility_loss(logits, data["gap"][ids], normalizer)
        return supervised + SPEC["head_l2"] * weight_penalty(head)

    trace = stage_fit(model, optimizer, fit, SPEC["branch_epochs"], SPEC["branch_batch"], objective,
                      OUT / f"branch_{task}.pt", order_seed=SPEC["order_seed"])
    trace["initial_m6_feature_max_abs_on_eight_queries"] = initial_difference
    model.set_stage("inference")
    features = encode_branch(model, data, selected, branch)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        scores = np.concatenate([head(torch.as_tensor(features[start:start+8], device="cuda")).float().cpu().numpy()
                                 for start in range(0, len(features), 8)])
    save_arrays(OUT / f"features_{task}.npz", indices=selected, query_ids=data["query_ids"][selected], features=features, task_logits=scores)
    del model, optimizer
    torch.cuda.empty_cache()
    return features, scores, trace


def run():
    require_execution_allowed()  # Must precede data reads, output writes and model loading.
    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError("The current CUDA/BF16 environment is required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config_path = ROOT / "outputs/router/hotpotqa_bd_router_v1/config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["retrieval"]["dense"]["revision"] != MODEL.name:
        raise ValueError("Local encoder does not match the current research configuration")
    input_paths = [BASE / f"layer_pooling_v1/{name}" for name in
                   ("features.npz", "predictions.npz", "tokens.npz", f"fold{SPEC['fold']}_M6_head.pt")]
    input_paths += [BASE / "e02_results/fold_indices.npz", MODEL / "model.safetensors", MODEL / "config.json"]
    teacher = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_9600_v1/privileged_teacher"
    input_paths += [teacher / "feature_schema.json", teacher / "teacher_features.npz"]
    binding = {"spec": SPEC, "config_path": str(config_path),
               "environment": {key: config[key] for key in ("dataset", "artifacts", "retrieval", "context", "prompt")},
               "generation": {key: config["generation"][key] for key in ("provider", "model", "temperature", "max_output_tokens")},
               "input_file_metadata": {str(path): {"bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns}
                                       for path in input_paths},
               "code_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                               for name in (Path(__file__).name, "router_capability_fusion.py")}}
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        if json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))["binding"] != binding:
            raise ValueError("Existing run uses a different recipe/configuration/implementation")
    else:
        write_json(OUT / "protocol.json", {"binding": binding, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    if (OUT / "results.json").exists():
        print((OUT / "results.json").read_text(encoding="utf-8"))
        return
    data = load_data()
    fit, cal, _unused_test = data["folds"][SPEC["fold"]]
    selected = np.concatenate([fit, cal])
    normalizer = float(np.where(np.abs(data["gap"][fit]) > 1e-12, np.abs(data["gap"][fit]), 0).mean())
    if normalizer <= 0:
        raise ValueError("Training split has no non-tie utility targets")
    representations, branch_scores, training = {}, {}, {}
    for task in SPEC["branch_tasks"]:
        features_path = OUT / f"features_{task}.npz"
        if features_path.exists():
            with np.load(features_path, allow_pickle=False) as archive:
                if not (np.array_equal(archive["indices"], selected) and np.array_equal(archive["query_ids"], data["query_ids"][selected])):
                    raise ValueError("Cached branch features use different rows")
                representations[task], branch_scores[task] = archive["features"].copy(), archive["task_logits"].copy()
            saved = torch.load(OUT / f"branch_{task}.pt", map_location="cpu", weights_only=True)
            training[task] = {"steps": saved["total_steps"], "elapsed_seconds": saved["elapsed_seconds"],
                              "trace": saved["trace"], "loaded_completed_features": True}
        else:
            representations[task], branch_scores[task], training[task] = learn_branch(task, data, fit, selected, normalizer)
    base = torch.as_tensor(data["features"][selected], device="cuda")
    views = {name: torch.as_tensor(value, device="cuda") for name, value in representations.items()}
    baseline = make_model().baseline_head
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline_scores = torch.cat([baseline(base[start:start+8]).squeeze(-1).float() for start in range(0, len(base), 8)])
    local_fit = np.arange(len(fit))
    scores = {"M6": baseline_scores[len(fit):].cpu().numpy(), "U_direct": branch_scores["utility"][len(fit):, 0]}
    for arm in SPEC["readouts"]:
        torch.manual_seed(SPEC["fusion_seed"])
        mode = "utility_only" if arm == "U" else "static" if arm == "RU_static" else "conditional"
        fusion = UtilityFusion(384, mode).cuda()
        utility = base if arm == "M6_gated" else views["utility"]
        auxiliary = base if arm == "M6_gated" else views["aux_utility"] if arm == "UU_conditional" else views["quality"]
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=SPEC["fusion_lr"], weight_decay=0.0, foreach=False)

        def objective(ids):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = baseline_scores[ids] + fusion(base[ids], utility[ids], auxiliary[ids]).float()
                supervised = utility_loss(logit, data["gap"][fit[ids]], normalizer)
            return supervised + SPEC["fusion_l2"] * weight_penalty(fusion)

        training[arm] = stage_fit(fusion, optimizer, local_fit, SPEC["fusion_epochs"], SPEC["fusion_batch"], objective,
                                  OUT / f"fusion_{arm}.pt", order_seed=SPEC["order_seed"])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            scores[arm] = np.concatenate([
                (baseline_scores[start:start+8] + fusion(base[start:start+8], utility[start:start+8], auxiliary[start:start+8]).float()).cpu().numpy()
                for start in range(len(fit), len(selected), 8)])
    utility = data["utility"][cal]
    quality = {name: float(np.where(value > 0, utility[:, 0], utility[:, 1]).mean()) for name, value in scores.items()}
    quality.update(BM25=float(utility[:, 0].mean()), Dense=float(utility[:, 1].mean()))
    primary = quality["RU_conditional"]
    control_gain = min(primary - quality[name] for name in ("M6", "U_direct", "U", "RU_static", "UU_conditional", "M6_gated"))
    fixed_gain = min(primary - quality[name] for name in ("BM25", "Dense"))
    # A simpler arm can still be useful when the specialized-fusion hypothesis
    # loses. Prefer it within a predeclared .002 development-score tolerance.
    preference = ["M6_gated", "U_direct", "U", "RU_static", "UU_conditional", "RU_conditional"]
    eligible = [name for name in preference
                if quality[name] - quality["M6"] >= SPEC["minimum_control_increment"]
                and quality[name] - max(quality["BM25"], quality["Dense"]) >= SPEC["minimum_fixed_gain"]]
    best = max((quality[name] for name in eligible), default=None)
    candidate = next((name for name in eligible if best - quality[name] <= SPEC["minimum_control_increment"]), None)
    decision = ("promising_development_only_independent_source_required" if candidate
                else "stop_this_recipe_no_seed_epoch_threshold_extension")
    save_arrays(OUT / "calibration_scores.npz", indices=cal, query_ids=data["query_ids"][cal], **scores)
    result = {"status": "complete_consumed_calibration_screen", "fit_queries": len(fit), "cal_queries": len(cal),
              "mean3_answer_f1": quality, "minimum_control_increment": control_gain, "minimum_fixed_gain": fixed_gain,
              "decision": decision, "candidate_for_further_planning": candidate,
              "specialized_conditional_fusion_development_gate": control_gain >= SPEC["minimum_control_increment"] and fixed_gain >= SPEC["minimum_fixed_gain"],
              "independent_gain_established": False, "training": training}
    write_json(OUT / "results.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute/resume only after the existing user pause has been lifted")
    args = parser.parse_args()
    if args.execute:
        run()
    else:
        print(json.dumps({"mode": "describe_only", "execution_state": execution_state(), "spec": SPEC}, indent=2))
