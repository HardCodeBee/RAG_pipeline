"""Independent checks for the M6 adaptation pilot; no training implementation.

Artifact checks will follow the frozen pilot contract. The helpers below use
an independent full-depth hook and explicit double-precision pooling algebra.
Torch is imported only by callers or by the synthetic CPU self-test.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work/router_research'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_output_adaptation_v1'
MODEL = Path('C:/Users/12442/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bf16(value):
    value = np.asarray(value, dtype=np.float32)
    assert np.isfinite(value).all()
    bits = np.ascontiguousarray(value).view(np.uint32)
    rounded = ((bits.astype(np.uint64) + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).astype(np.uint32)
    return rounded.view(np.float32).reshape(value.shape)


def first_head_math(first, trace_row):
    """FP64 loss and an interval for the two BF16 backward casts.

FP32 elementwise backward error is enclosed before residual BF16 rounding.
BF16-input dot products use the ordinary FP32 accumulation error bound; the
result is rounded to BF16 before the FP32 explicit head-penalty gradient.
"""
    x, z, y, w = [np.asarray(first[name]) for name in ('features', 'logits', 'targets', 'weights')]
    assert x.shape == (8, 384) and all(v.dtype == np.float32 for v in (x, z, y, w))
    assert z.shape == y.shape == w.shape == (8,) and np.all((y == 0) | (y == 1))
    assert np.all(w >= 0) and all(np.isfinite(v).all() for v in (x, z, y, w))
    norm = float(first['normalizer']); assert norm > 0 and math.isfinite(norm)
    beta = np.asarray(first['head_weight']); bias = np.asarray(first['head_bias'])
    assert beta.shape == (1, 384) and bias.shape == (1,) and beta.dtype == bias.dtype == np.float32
    supervised = math.fsum(float(wi) * float(np.logaddexp(0., -zi if yi else zi))
                           for zi, yi, wi in zip(z, y, w)) / (8 * norm)
    penalty = .0005 * math.fsum(float(v) ** 2 for v in beta.ravel())
    expected = np.array([supervised, penalty, supervised + penalty])
    loss_error = np.abs(np.asarray(trace_row[:3]) - expected)
    loss_bound = 64 * np.finfo(np.float32).eps * np.maximum(np.abs(expected), 1.)
    assert np.all(loss_error <= loss_bound), 'First-step FP32 loss differs from independent formula'
    probability = np.array([1 / (1 + math.exp(-float(v))) if v >= 0 else
                            math.exp(float(v)) / (1 + math.exp(float(v))) for v in z])
    scale = w.astype(float) / (8 * norm)
    residual = scale * (probability - y)
    r_error = 64 * np.finfo(np.float32).eps * scale
    lower, upper = bf16(residual - r_error).astype(float), bf16(residual + r_error).astype(float)
    xb = bf16(x).astype(float)
    low_products = np.minimum(xb * lower[:, None], xb * upper[:, None])
    high_products = np.maximum(xb * lower[:, None], xb * upper[:, None])
    low = np.sum(low_products, axis=0); high = np.sum(high_products, axis=0)
    magnitude = np.sum(np.maximum(abs(low_products), abs(high_products)), axis=0)
    unit = 2. ** -24; gamma = 8 * unit / (1 - 8 * unit)
    low, high = bf16(low - gamma * magnitude).astype(float), bf16(high + gamma * magnitude).astype(float)
    penalty_gradient = .001 * beta.ravel().astype(float)
    arithmetic = 16 * unit * (np.maximum(abs(low), abs(high)) + abs(penalty_gradient)) + 1e-30
    low += penalty_gradient - arithmetic; high += penalty_gradient + arithmetic
    bmag = math.fsum(np.maximum(abs(lower), abs(upper)))
    bias_low = float(bf16(np.array(math.fsum(lower) - gamma * bmag)))
    bias_high = float(bf16(np.array(math.fsum(upper) + gamma * bmag)))
    observed_weight = np.asarray(first['head_weight_grad'])
    observed_bias = np.asarray(first['head_bias_grad'])
    assert observed_weight.shape == beta.shape and observed_bias.shape == bias.shape
    assert observed_weight.dtype == observed_bias.dtype == np.float32
    assert np.all(observed_weight.ravel() >= low) and np.all(observed_weight.ravel() <= high), 'BF16 head weight gradient outside independent interval'
    assert bias_low <= observed_bias[0] <= bias_high, 'BF16 head bias gradient outside independent interval'
    return dict(loss_max_error=float(loss_error.max()),
        head_weight_gradient_max_interval_width=float(np.max(high - low)),
        head_bias_gradient_interval_width=float(bias_high - bias_low),
        residual_cast='FP32 backward -> BF16; BF16 dot -> BF16 -> FP32; explicit FP32 lambda beta')


def pool_reference(hidden, attention, special, upstream=None):
    """Masked mean/L2 and its analytical pullback, with the CLS fallback."""
    hidden = np.asarray(hidden, dtype=np.float64)
    attention, special = np.asarray(attention), np.asarray(special)
    assert hidden.ndim == 3 and np.isfinite(hidden).all()
    assert attention.shape == special.shape == hidden.shape[:2]
    ordinary = attention.astype(bool) & ~special.astype(bool)
    n, _, dimensions = hidden.shape
    features = np.empty((n, dimensions), np.float64)
    if upstream is not None:
        upstream = np.asarray(upstream, dtype=np.float64)
        assert upstream.shape == features.shape and np.isfinite(upstream).all()
        gradient = np.zeros_like(hidden)
    for i in range(n):
        positions = np.flatnonzero(ordinary[i])
        if len(positions) == 0:
            positions = np.array([0], dtype=np.int64)
        mean = np.array([math.fsum(float(hidden[i, j, k]) for j in positions) / len(positions)
                         for k in range(dimensions)])
        norm = math.sqrt(math.fsum(float(v) ** 2 for v in mean))
        features[i] = mean / max(norm, 1e-12)
        if upstream is not None:
            if norm > 1e-12:
                radial = math.fsum(float(a) * float(b) for a, b in zip(features[i], upstream[i]))
                mean_gradient = (upstream[i] - radial * features[i]) / norm
            else:
                assert norm < 1e-12, 'Derivative at the normalize epsilon boundary is not unique'
                mean_gradient = upstream[i] / 1e-12
            for j in positions:
                gradient[i, j] = mean_gradient / len(positions)
    return (features, gradient) if upstream is not None else features


def full_depth_hook_reference(encoder, head, batch, special, torch):
    """One default 12-block forward; exact FP32 and independent FP64 pool."""
    assert len(encoder.encoder.layer) == 12
    assert not encoder.training and not head.training
    assert all(not module.training for module in encoder.modules())
    captured = []
    handle = encoder.encoder.layer[5].register_forward_hook(
        lambda module, arguments, output: captured.append(output[0].detach()))
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            encoder(**batch, output_hidden_states=False)
            assert len(captured) == 1
            hidden = captured[0]
            mask = batch['attention_mask'].bool() & ~special.bool()
            counts = mask.sum(dim=1)
            mean = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1)
            mean = mean / counts.clamp_min(1).float().unsqueeze(1)
            mean = torch.where((counts == 0).unsqueeze(1), hidden[:, 0].float(), mean)
            feature = torch.nn.functional.normalize(mean, p=2, dim=1)
            assert feature.dtype == torch.float32
            logits = head(feature).flatten().float().cpu().numpy()
            exact = feature.cpu().numpy()
        double = pool_reference(hidden.float().cpu().numpy(),
            batch['attention_mask'].cpu().numpy(), special.cpu().numpy())
        return dict(features=exact, logits=logits, FP64_features=double,
                    FP64_pool_error=float(np.max(abs(double - exact))))
    finally:
        handle.remove()


def expected_config():
    return dict(epochs=4, batch_size=8, max_length=128, steps_per_epoch=768,
        total_steps=3072, warmup_steps=307, encoder_lr=2e-5, head_lr=.001,
        encoder_weight_decay=.01, head_weight_decay=0., head_lambda=.001,
        clip_norm=1., order_seed=2026091517, model_seed=2026091515,
        pilot_steps=8, pilot_fold=0, cal_tie_atol=1e-12, bootstrap_draws=20000,
        bootstrap_seed=2026091519, quantiles=[.05 / 8, 1 - .05 / 8],
        minimum_increment=.002, minimum_fixed_gain=.01)


def require_bindings(mapping):
    assert mapping
    for path, digest in mapping.items():
        assert sha(path) == digest, 'Changed bound file: ' + str(path)


def parameters(encoder, head):
    """The six-block model's order, independent of unused reference blocks."""
    chosen = []
    for name, value in encoder.named_parameters():
        if name.startswith('embeddings.') or any(name.startswith(f'encoder.layer.{j}.') for j in range(6)):
            chosen.append(('encoder.' + name, value))
    return chosen + [('head.' + name, value) for name, value in head.named_parameters()]


def tensor_digest(named):
    digest = hashlib.sha256()
    for name, tensor in named:
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def reference_model(arm, torch):
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=True)
    assert len(encoder.encoder.layer) == 12 and encoder.config.hidden_size == 384
    encoder.pooler = None
    encoder.requires_grad_(False)
    if arm == 'A':
        encoder.embeddings.requires_grad_(True)
        for block in encoder.encoder.layer[:6]:
            block.requires_grad_(True)
    head = torch.nn.Linear(384, 1)
    old = torch.load(BASE / 'layer_pooling_v1/fold0_M6_head.pt', map_location='cpu', weights_only=True)
    assert set(old) == {'weight', 'bias'} and all(v.dtype == torch.float32 for v in old.values())
    head.load_state_dict(old, strict=True)
    encoder.to('cuda').eval(); head.to('cuda').eval()
    assert all(not module.training for module in encoder.modules())
    assert all(not p.requires_grad for block in encoder.encoder.layer[6:] for p in block.parameters())
    return encoder, head


def differentiable_hook_feature(encoder, batch, special, torch):
    captured = []
    handle = encoder.encoder.layer[5].register_forward_hook(
        lambda module, arguments, output: captured.append(output[0]))
    try:
        encoder(**batch, output_hidden_states=False)
        assert len(captured) == 1
        hidden = captured[0]
        mask = batch['attention_mask'].bool() & ~special.bool()
        count = mask.sum(dim=1)
        mean = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1).float().unsqueeze(1)
        mean = torch.where((count == 0).unsqueeze(1), hidden[:, 0].float(), mean)
        result = torch.nn.functional.normalize(mean, p=2, dim=1)
        assert result.dtype == torch.float32
        return result
    finally:
        handle.remove()


def make_batch(tokens, indices, torch):
    batch = {key: torch.from_numpy(value[indices]).to('cuda') for key, value in tokens.items()
             if key != 'special_tokens_mask'}
    special = torch.from_numpy(tokens['special_tokens_mask'][indices]).to('cuda')
    return batch, special


def check_probe(encoder, head, tokens, indices, expected_x, expected_scores, torch):
    errors = []
    for start in range(0, 64, 8):
        batch, special = make_batch(tokens, indices[start:start + 8], torch)
        observed = full_depth_hook_reference(encoder, head, batch, special, torch)
        assert np.array_equal(observed['features'], expected_x[start:start + 8]), 'Independent full12 layer6 features differ'
        assert np.array_equal(observed['logits'], expected_scores[start:start + 8]), 'Independent full12 layer6 logits differ'
        assert observed['FP64_pool_error'] <= 2e-6
        errors.append(observed['FP64_pool_error'])
    return max(errors)


def replay_eight_steps(encoder, head, arm, tokens, fit, gap, saved, record, torch):
    """Fresh optimizer, independently chosen batches and a full12 hook graph."""
    all_named = parameters(encoder, head)
    trainable = [(name, value) for name, value in all_named if value.requires_grad]
    frozen = [(name, value) for name, value in all_named if not value.requires_grad]
    assert [name for name, _ in trainable] == record['trained_names']
    assert len(trainable) == (103 if arm == 'A' else 2)
    assert sum(p.numel() for _, p in trainable) == (22565761 if arm == 'A' else 385)
    groups = []
    encoder_named = [(name[len('encoder.'):], p) for name, p in trainable if name.startswith('encoder.')]
    if arm == 'A':
        for decay in (False, True):
            selected = [p for name, p in encoder_named if (not (name.endswith('bias') or 'LayerNorm.weight' in name)) == decay]
            groups.append(dict(params=selected, lr=2e-5, weight_decay=.01 if decay else 0.))
    groups.append(dict(params=list(head.parameters()), lr=.001, weight_decay=0.))
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, foreach=False)
    assert not optimizer.state
    local_order = np.random.default_rng(2026091517).permutation(6144)[:64]
    order = fit[local_order].astype(np.int64)
    assert np.array_equal(order, saved['training_order'])
    assert record['batch_order_sha256'] == hashlib.sha256(order.tobytes()).hexdigest()
    weights = np.where(abs(gap) > 1e-12, abs(gap), 0.)
    norm = float(weights.mean())
    assert record['fit_mean_weight'] == norm
    encps = [p for _, p in encoder_named]
    headps = list(head.parameters())
    traces = []
    first_gradient_error = 0.
    for step in range(8):
        positions = local_order[step * 8:(step + 1) * 8]
        indices = fit[positions]
        batch, special = make_batch(tokens, indices, torch)
        target = torch.tensor(gap[positions] > 0, dtype=torch.float32, device='cuda')
        weight = torch.tensor(weights[positions], dtype=torch.float32, device='cuda')
        optimizer.zero_grad(set_to_none=True)
        multiplier = (step + 1) / 307
        for j, group in enumerate(optimizer.param_groups):
            group['lr'] = (.001 if j == len(optimizer.param_groups) - 1 else 2e-5) * multiplier
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feature = differentiable_hook_feature(encoder, batch, special, torch)
            if arm == 'A':
                assert feature.requires_grad and feature.grad_fn is not None
            logits = head(feature).flatten().float()
            supervised = (weight * torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')).mean() / norm
        penalty = .0005 * head.weight.float().square().sum()
        loss = supervised + penalty
        loss.backward()
        assert all(p.grad is not None for _, p in trainable)
        assert bool(torch.stack([torch.isfinite(p.grad).all() for _, p in trainable]).all())
        assert all(p.grad is None for _, p in frozen)
        if step == 0:
            actual = dict(indices=indices, features=feature.detach().cpu().numpy(), logits=logits.detach().cpu().numpy(),
                targets=target.cpu().numpy(), weights=weight.cpu().numpy(), normalizer=np.array(norm),
                head_weight=head.weight.detach().cpu().numpy(), head_bias=head.bias.detach().cpu().numpy(),
                head_weight_grad=head.weight.grad.detach().cpu().numpy(), head_bias_grad=head.bias.grad.detach().cpu().numpy())
            for name, value in actual.items():
                assert np.array_equal(value, saved['first_' + name]), 'First-step replay differs: ' + name
            for name, parameter in trainable:
                observed = float(parameter.grad.detach().float().norm())
                difference = abs(observed - record['first_parameter_gradient_norm'][name])
                first_gradient_error = max(first_gradient_error, difference)
                assert difference == 0., 'First-step gradient norm differs: ' + name
        encnorm = float(torch.nn.utils.clip_grad_norm_(encps, 1.)) if encps else 0.
        headnorm = float(torch.nn.utils.clip_grad_norm_(headps, 1.))
        traces.append([float(supervised.detach()), float(penalty.detach()), float(loss.detach()),
                       encnorm, headnorm, multiplier, float(weight.sum() == 0)])
        optimizer.step()
    observed_trace = np.asarray(traces, np.float64)
    assert np.array_equal(observed_trace, saved['step_trace']), 'Eight-step trace replay differs'
    return dict(first_gradient_norm_max_error=first_gradient_error, step_trace_max_error=0.,
        training_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(), optimizer_steps=8)


def load_inputs(torch):
    """Read only original base representations, own-fit gaps, splits and tokens."""
    directory = BASE / 'layer_pooling_v1'
    split = BASE / 'e02_results/fold_indices.npz'
    assert sha(split) == 'ad2e94f76332f288f798d8bc1b1ff085be9aba64730b84d4906566d37ea76bc6'
    historical = {Path(name).resolve(): value for name, value in read(directory / 'completion_record.json')['artifact_sha256'].items()}
    needed = [directory / name for name in ('features.npz', 'predictions.npz', 'tokens.npz')]
    needed += [directory / f'fold{f}_M6_{suffix}' for f in range(5) for suffix in ('head.pt', 'fit.npz')]
    for path in needed:
        assert path.resolve() in historical and sha(path) == historical[path.resolve()]
    with np.load(directory / 'features.npz', allow_pickle=False) as z:
        x, qids, groups = [z[key].copy() for key in ('M6', 'query_ids', 'group_ids')]
    assert x.shape == (9600, 384) and x.dtype == np.float32 and np.isfinite(x).all()
    assert qids.shape == groups.shape == (9600,) and len(set(qids)) == 9600 and len(set(groups)) == 9559
    assert np.max(abs(np.sqrt(np.einsum('nd,nd->n', x.astype(float), x.astype(float))) - 1)) < 2e-6
    with np.load(directory / 'tokens.npz', allow_pickle=False) as z:
        assert set(z.files) == {'input_ids', 'attention_mask', 'token_type_ids', 'special_tokens_mask'}
        tokens = {key: z[key].copy() for key in z.files}
    assert all(value.shape == (9600, 128) and value.dtype == np.int64 for value in tokens.values())
    with np.load(directory / 'predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(qids, z['query_ids']) and np.array_equal(groups, z['group_ids'])
        utility = z['utility'].copy()
    assert utility.shape == (9600, 2) and utility.dtype == np.float64
    assert np.isfinite(utility).all() and np.all((utility >= 0) & (utility <= 1))
    with np.load(split, allow_pickle=False) as z:
        folds = [{part: z[f'fold{f}_{part}'].copy() for part in ('fit', 'calibration', 'test')} for f in range(5)]
    coverage = np.zeros(9600, dtype=np.int64)
    initial_diagnostics = []
    first_gap, first_logits = None, None
    for f, parts in enumerate(folds):
        sets = []
        for part, count in (('fit', 6144), ('calibration', 1536), ('test', 1920)):
            ids = parts[part]
            assert ids.shape == (count,) and ids.dtype.kind in 'iu' and len(set(ids)) == count
            assert np.all((ids >= 0) & (ids < 9600))
            sets.append(set(groups[ids]))
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert np.array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(9600))
        coverage[parts['test']] += 1
        fit = parts['fit']
        gap = utility[fit, 0] - utility[fit, 1]
        w = np.where(abs(gap) > 1e-12, abs(gap), 0.)
        normalized = w / math.fsum(map(float, w))
        payload = torch.load(directory / f'fold{f}_M6_head.pt', map_location='cpu', weights_only=True)
        assert set(payload) == {'weight', 'bias'}
        beta = payload['weight'].numpy().reshape(384); bias = float(payload['bias'].item())
        assert beta.dtype == np.float32
        xf = x[fit].astype(float)
        ideal = np.einsum('nd,d->n', xf, beta.astype(float), optimize=False) + bias
        probability = np.array([1 / (1 + math.exp(-v)) if v >= 0 else math.exp(v) / (1 + math.exp(v)) for v in ideal])
        residual = normalized * (probability - (gap > 0))
        gradient = np.r_[np.einsum('nd,n->d', xf, residual, optimize=False) + .001 * beta.astype(float),
                         math.fsum(map(float, residual))]
        loss = math.fsum(float(wi) * float(np.logaddexp(0., -si if gi > 0 else si))
                         for wi, si, gi in zip(normalized, ideal, gap))
        penalty = .0005 * math.fsum(float(v) ** 2 for v in beta)
        with np.load(directory / f'fold{f}_M6_fit.npz', allow_pickle=False) as z:
            assert np.array_equal(z['fit_indices'], fit)
            logits = z['final_native_fit_logits'].copy()
        assert logits.shape == (6144,) and logits.dtype == np.float32 and np.array_equal(logits, bf16(logits))
        xb, bb, biasb = bf16(x[fit]).astype(float), bf16(beta).astype(float), float(bf16(np.array(bias)))
        center = np.einsum('nd,d->n', xb, bb, optimize=False) + biasb
        magnitude = np.einsum('nd,d->n', abs(xb), abs(bb), optimize=False) + abs(biasb)
        gamma = 385 * 2. ** -24 / (1 - 385 * 2. ** -24)
        accumulation = gamma * magnitude + 385 * 2. ** -149
        bound = accumulation + 2. ** -8 * (abs(center) + accumulation) + 2. ** -133
        error = abs(logits.astype(float) - center)
        assert np.all(error <= bound + 1e-14)
        initial_diagnostics.append(dict(fold=f, ideal_FP64_data_loss=loss, head_penalty=penalty,
            ideal_FP64_loss=loss + penalty, ideal_FP64_gradient_inf=float(abs(gradient).max()),
            cached_native_rows=6144, native_score_bound_passed=True,
            native_score_max_error=float(error.max()), native_score_max_bound=float(bound.max()),
            is_native_BF16_gradient=False, is_pilot_loss_improvement_gate=False))
        if f == 0:
            first_gap, first_logits = gap, logits
    assert np.all(coverage == 1)
    return dict(features=x, tokens=tokens, query_ids=qids, group_ids=groups,
        fit=folds[0]['fit'], fit_gap=first_gap, fit_logits=first_logits,
        initial_head_diagnostics=initial_diagnostics)


def check(protocol_sha):
    started = time.perf_counter()
    destination = OUT / 'pilot_separate_checks.json'
    assert not destination.exists(), 'Preserve an existing independent pilot receipt'
    assert protocol_sha and sha(OUT / 'protocol.json') == protocol_sha
    protocol = read(OUT / 'protocol.json')
    assert protocol['status'] == 'frozen_before_M6_output_adaptation_pilot_and_formal_training'
    assert protocol['config'] == expected_config()
    assert protocol['primary'] == ['A_minus_C', 'A_minus_B', 'A_minus_Dense', 'A_minus_BM25']
    assert protocol['versions'] == {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'transformers', 'scipy')}
    assert protocol['queries'] == 9600
    expected_sources = [ROOT / 'scripts' / name for name in ('run_m6_output_adaptation.py',
        'check_m6_output_adaptation_pilot.py', 'run_m6_output_adaptation_formal.py',
        'run_m6_objective_readout.py', 'm6_objective_math.py')]
    expected_sources.append(ROOT / 'analysis/hotpotqa_router/m6_output_adaptation_plan_20260915.md')
    expected_inputs = [BASE / name for name in ('layer_pooling_v1/features.npz', 'layer_pooling_v1/predictions.npz',
        'e02_results/fold_indices.npz', 'm6_pooled_offset_v1/cal_logits.npz',
        'layer_pooling_v1/completion_record.json', 'm6_pooled_offset_v1/completion_record.json',
        'layer_pooling_v1/tokens.npz')]
    expected_inputs += [BASE / f'layer_pooling_v1/fold{f}_M6_{suffix}' for f in range(5) for suffix in ('head.pt', 'fit.npz')]
    expected_inputs += [MODEL / name for name in ('model.safetensors', 'config.json')]
    assert {Path(path).resolve() for path in protocol['source_sha256']} == {p.resolve() for p in expected_sources}
    assert {Path(path).resolve() for path in protocol['input_sha256']} == {p.resolve() for p in expected_inputs}
    require_bindings(protocol['source_sha256']); require_bindings(protocol['input_sha256'])
    pilot = read(OUT / 'pilot.json')
    assert pilot['status'] == 'complete_M6_output_pilot_pending_independent_check'
    assert pilot['protocol_sha256'] == protocol_sha
    assert pilot['pilot_optimizer_steps'] == 16 and pilot['encoder_query_forwards'] == 12672
    assert pilot['cal_quality_evaluations'] == pilot['test_quality_evaluations'] == pilot['new_api_calls'] == 0
    assert pilot['formal_training_started'] is False
    assert [row['arm'] for row in pilot['arms']] == ['A', 'C']
    expected_artifacts = {OUT / f'pilot_{arm}{suffix}' for arm in ('A', 'C') for suffix in ('.pt', '.npz')}
    assert {Path(path).resolve() for path in pilot['artifact_sha256']} == expected_artifacts
    require_bindings(pilot['artifact_sha256'])
    pilot_start = read(OUT / 'pilot_started.json')
    assert pilot_start['protocol_sha256'] == protocol_sha
    assert datetime.fromisoformat(protocol['created_at_utc']) <= datetime.fromisoformat(pilot_start['started_at_utc'])
    snapshots = dict(pilot['artifact_sha256'])
    snapshots.update({str(OUT / name): sha(OUT / name) for name in ('protocol.json', 'pilot.json', 'pilot_started.json')})
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    data = load_inputs(torch)
    fit, gap = data['fit'], data['fit_gap']
    probe = fit[:64]
    reports = []
    for record in pilot['arms']:
        arm = record['arm']
        with np.load(OUT / f'pilot_{arm}.npz', allow_pickle=False) as archive:
            saved = {name: archive[name].copy() for name in archive.files}
        first_names = ['indices', 'features', 'logits', 'targets', 'weights', 'normalizer',
                       'head_weight', 'head_bias', 'head_weight_grad', 'head_bias_grad']
        assert set(saved) == {'probe_indices', 'initial_features', 'initial_logits', 'final_features', 'final_logits',
                             'training_order', 'step_trace'} | {'first_' + name for name in first_names}
        assert np.array_equal(saved['probe_indices'], probe)
        assert np.array_equal(saved['initial_features'], data['features'][probe])
        assert np.array_equal(saved['initial_logits'], data['fit_logits'][:64])
        for name in ('initial_features', 'final_features'):
            value = saved[name]
            assert value.shape == (64, 384) and value.dtype == np.float32 and np.isfinite(value).all()
            assert np.max(abs(np.linalg.norm(value.astype(float), axis=1) - 1)) < 2e-6
        for name in ('initial_logits', 'final_logits'):
            assert saved[name].shape == (64,) and saved[name].dtype == np.float32 and np.isfinite(saved[name]).all()
        trace = saved['step_trace']
        assert trace.shape == (8, 7) and trace.dtype == np.float64 and np.isfinite(trace).all()
        assert np.array_equal(trace[:, 5], np.arange(1, 9) / 307)
        assert np.all((trace[:, 6] == 0) | (trace[:, 6] == 1))
        assert record['fold'] == 0 and record['epoch'] == 1 and record['optimizer_steps'] == 8
        assert record['all_losses_and_gradients_finite'] is True
        assert record['full_fit_initial_feature_and_score_exact'] is True and record['checkpoint_probe_replay_exact'] is True
        assert record['zero_weight_batches'] == int(trace[:, 6].sum())
        for key, column in (('online_weighted_BCE', 0), ('online_head_penalty', 1), ('online_loss', 2)):
            assert abs(record[key] - math.fsum(map(float, trace[:, column])) / 8) <= 1e-12
        assert np.array_equal(record['mean_encoder_head_gradnorm_before_clip'], trace[:, 3:5].mean(axis=0))
        assert record['first_lr_multiplier'] == 1 / 307 and record['last_lr_multiplier'] == 8 / 307
        local_order = np.random.default_rng(2026091517).permutation(6144)[:64]
        assert np.array_equal(saved['training_order'], fit[local_order])
        assert np.array_equal(saved['first_indices'], fit[local_order[:8]])
        assert np.array_equal(saved['first_features'], data['features'][saved['first_indices']])
        assert np.array_equal(saved['first_logits'], data['fit_logits'][local_order[:8]])
        expected_weights = np.where(abs(gap) > 1e-12, abs(gap), 0.)
        assert np.array_equal(saved['first_targets'], (gap[local_order[:8]] > 0).astype(np.float32))
        assert np.array_equal(saved['first_weights'], expected_weights[local_order[:8]].astype(np.float32))
        assert float(saved['first_normalizer']) == float(expected_weights.mean())
        assert np.array_equal(trace[:, 6], [float(expected_weights[local_order[j:j + 8]].sum() == 0) for j in range(0, 64, 8)])
        head_math = first_head_math({name: saved['first_' + name] for name in first_names}, trace[0])
        encoder, head = reference_model(arm, torch)
        all_named = parameters(encoder, head)
        initial = {name: value.detach().cpu().clone() for name, value in all_named}
        trainable = [(name, value) for name, value in all_named if value.requires_grad]
        frozen = [(name, value) for name, value in all_named if not value.requires_grad]
        assert tensor_digest(all_named) == record['initial_parameters_sha256']
        assert tensor_digest(frozen) == record['frozen_before_sha256'] == record['frozen_after_sha256']
        assert record['trained_parameter_count'] == sum(value.numel() for _, value in trainable)
        initial_error = check_probe(encoder, head, data['tokens'], probe, saved['initial_features'], saved['initial_logits'], torch)
        replay = replay_eight_steps(encoder, head, arm, data['tokens'], fit, gap, saved, record, torch)
        checkpoint = torch.load(OUT / f'pilot_{arm}.pt', map_location='cpu', weights_only=True)
        assert checkpoint['arm'] == arm and checkpoint['fold'] == 0 and checkpoint['epoch'] == -1
        assert checkpoint['protocol_sha256'] == protocol_sha and checkpoint['core_sha256'] == sha(ROOT / 'scripts/run_m6_output_adaptation.py')
        assert set(checkpoint['trained_parameters']) == {name for name, _ in trainable}
        for name, value in trainable:
            archived = checkpoint['trained_parameters'][name]
            assert archived.dtype == torch.float32 and archived.shape == value.shape
            assert torch.equal(archived, value.detach().cpu()), 'Eight-step parameter replay differs: ' + name
        assert tensor_digest(all_named) == record['final_parameters_sha256']
        assert tensor_digest(frozen) == record['frozen_before_sha256']
        changes = {name: float((value.detach().cpu() - initial[name]).abs().max()) for name, value in all_named}
        assert changes == record['parameter_max_changes']
        assert any(changes[name] > 0 for name in changes if name.startswith('head.'))
        if arm == 'A':
            for prefix in ['encoder.embeddings.'] + [f'encoder.encoder.layer.{j}.' for j in range(6)]:
                assert any(changes[name] > 0 for name in changes if name.startswith(prefix))
                assert any(value > 0 for name, value in record['first_parameter_gradient_norm'].items() if name.startswith(prefix))
            assert not np.array_equal(saved['initial_features'], saved['final_features'])
        else:
            assert np.array_equal(saved['initial_features'], saved['final_features'])
            assert all(changes[name] == 0 for name, _ in frozen)
        final_error = check_probe(encoder, head, data['tokens'], probe, saved['final_features'], saved['final_logits'], torch)
        assert record['mean6_probe_max_change'] == float(np.max(abs(saved['final_features'] - saved['initial_features'])))
        reports.append(dict(arm=arm, trained_tensors=len(trainable), trained_parameters=record['trained_parameter_count'],
            initial_and_final_features_logits_exact=True, first_step_and_eight_step_trace_exact=True,
            final_checkpoint_all_trainable_tensors_exact=True, frozen_tensors_unchanged=True,
            FP64_pool_max_error=max(initial_error, final_error), head_math=head_math, replay=replay,
            mean6_probe_max_change=record['mean6_probe_max_change']))
        del encoder, head, trainable, frozen, all_named, initial, checkpoint
        gc.collect(); torch.cuda.empty_cache()
        print(json.dumps(dict(status='independent_pilot_arm_checked', arm=arm, optimizer_steps=8)), flush=True)
    assert pilot['arms'][0]['initial_parameters_sha256'] == pilot['arms'][1]['initial_parameters_sha256']
    assert pilot['arms'][0]['batch_order_sha256'] == pilot['arms'][1]['batch_order_sha256']
    require_bindings(snapshots); require_bindings(protocol['source_sha256']); require_bindings(protocol['input_sha256'])
    receipt = dict(status='passed_independent_M6_output_adaptation_pilot_checks',
        protocol_sha256=protocol_sha, pilot_sha256=sha(OUT / 'pilot.json'), checker_sha256=sha(Path(__file__)),
        artifact_sha256=snapshots, source_sha256=protocol['source_sha256'], input_sha256=protocol['input_sha256'],
        arms=reports, initial_M6_head_ideal_FP64_diagnostics=data['initial_head_diagnostics'],
        independent_encoder_query_forwards=384, replay_optimizer_steps=16,
        independent_full12_layer6_default_forward=True, new_formal_fits=0, new_experimental_conditions=0,
        cal_quality_evaluations=0, test_quality_evaluations=0, new_api_calls=0,
        formal_training_authorized_by_implementation_gate=True,
        elapsed_seconds=time.perf_counter() - started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Implementation pilot replay only; no new heldout policy effect and no ideal-FP64-gradient claim for native BF16 training')
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    return receipt


def self_test():
    """Synthetic differentiability/masking checks; no model/data/GPU access."""
    import torch
    rng = np.random.default_rng(2026091515)
    hidden = rng.normal(size=(3, 6, 7))
    attention = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    special = np.array([[1, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1]])
    upstream = rng.normal(size=(3, 7))
    features, gradient = pool_reference(hidden, attention, special, upstream)
    value = torch.tensor(hidden, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(attention.astype(bool) & ~special.astype(bool))
    count = mask.sum(dim=1)
    mean = (value * mask.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1).unsqueeze(1)
    mean = torch.where((count == 0).unsqueeze(1), value[:, 0], mean)
    actual = torch.nn.functional.normalize(mean, p=2, dim=1)
    (actual * torch.tensor(upstream)).sum().backward()
    forward_error = float(np.max(abs(features - actual.detach().numpy())))
    pullback_error = float(np.max(abs(gradient - value.grad.numpy())))
    assert forward_error < 2e-15 and pullback_error < 2e-14
    finite_difference_error = 0.
    for flat in rng.choice(hidden.size, 25, replace=False):
        perturbation = np.zeros_like(hidden); perturbation.flat[flat] = 1e-6
        plus = pool_reference(hidden + perturbation, attention, special)
        minus = pool_reference(hidden - perturbation, attention, special)
        observed = math.fsum(map(float, ((plus - minus) * upstream).ravel())) / 2e-6
        finite_difference_error = max(finite_difference_error, abs(observed - gradient.flat[flat]))
    assert finite_difference_error < 2e-9
    ordinary = attention.astype(bool) & ~special.astype(bool)
    for row in range(3):
        allowed = ordinary[row].copy()
        if not allowed.any():
            allowed[0] = True
        assert np.count_nonzero(gradient[row, ~allowed]) == 0
    changed = hidden.copy()
    ignored = ~ordinary; ignored[:, 0] = False
    changed[ignored] += 100000
    assert np.array_equal(pool_reference(changed, attention, special), features)
    head_checks = []
    for case in range(5):
        x = rng.normal(size=(8, 384)).astype(np.float32) / math.sqrt(384)
        y = rng.integers(0, 2, 8).astype(np.float32)
        w = rng.random(8).astype(np.float32); w[:case] = 0
        if case == 4:
            w[:] = 0
        norm = .31
        beta = torch.tensor(rng.normal(size=(1, 384)).astype(np.float32), requires_grad=True)
        bias = torch.tensor([[-20., -.2, .3, 20., 0.][case]], requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            score = torch.nn.functional.linear(torch.from_numpy(x), beta, bias).flatten().float()
            data_loss = (torch.tensor(w) * torch.nn.functional.binary_cross_entropy_with_logits(
                score, torch.tensor(y), reduction='none')).mean() / norm
        penalty = .0005 * beta.square().sum()
        loss = data_loss + penalty
        loss.backward()
        saved = dict(features=x, logits=score.detach().numpy(), targets=y, weights=w, normalizer=np.array(norm),
            head_weight=beta.detach().numpy(), head_bias=bias.detach().numpy(),
            head_weight_grad=beta.grad.numpy(), head_bias_grad=bias.grad.numpy())
        head_checks.append(first_head_math(saved, [data_loss.item(), penalty.item(), loss.item()]))
    return dict(status='passed_synthetic_masked_mean_normalize_pullback_checks',
        forward_error=forward_error, analytical_pullback_error=pullback_error,
        finite_difference_error=finite_difference_error, ignored_position_invariance=True,
        zero_token_CLS_fallback=True, BF16_head_gradient_cases=len(head_checks),
        BF16_head_loss_max_error=max(c['loss_max_error'] for c in head_checks),
        real_data_reads=0, model_fits=0, GPU_forwards=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False))
    else:
        receipt = check(args.protocol_sha256)
        print(json.dumps({key: receipt[key] for key in ('status', 'independent_encoder_query_forwards',
            'replay_optimizer_steps', 'elapsed_seconds')}, allow_nan=False))
