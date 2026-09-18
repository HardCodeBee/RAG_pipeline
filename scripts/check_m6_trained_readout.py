"""Independent checkpoint/readout checks; no encoder or head training.

The optional forward helper uses a default encoder forward and a layer hook,
with separate FP64 masked pooling. No new experiment runner is imported.
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
from threadpoolctl import threadpool_limits

import check_m6_student_transfer as independent


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work/router_research'
H_DIRECTORY = BASE / 'lp_ft_v2'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_trained_readout_v1'
NEW_ARMS = ['B12', 'H6', 'H12']
ARMS = ['B6'] + NEW_ARMS
POLICIES = ARMS + ['H_native', 'Dense', 'BM25']
PRIMARY = ['H6_minus_B6', 'H12_minus_B12', 'H6_minus_H12', 'readout_interaction',
           'H6_minus_Dense', 'H6_minus_BM25']
MODEL = Path('C:/Users/12442/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')
QUANTILES = [.05 / 12, 1 - .05 / 12]
BOOTSTRAP_SEED = 2026091513
sha, read = independent.sha, independent.read


def hook_views(encoder, tokens, special_mask, indices, torch, *, device='cuda'):
    """One default forward gives original-precision and independent FP64 views."""
    ids = np.asarray(indices, dtype=np.int64)
    assert ids.ndim == 1 and len(ids) > 0
    assert len(encoder.encoder.layer) == 12
    encoder.eval()
    captured, exact, double = [], [], []

    def receive(module, arguments, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(tensor.detach())

    handle = encoder.encoder.layer[5].register_forward_hook(receive)
    try:
        with torch.inference_mode():
            for start in range(0, len(ids), 8):
                selected = ids[start:start + 8]
                batch = {}
                for name in ('input_ids', 'attention_mask', 'token_type_ids'):
                    value = tokens[name]
                    tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
                    batch[name] = tensor[selected].to(device)
                before = len(captured)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                    output = encoder(**batch, output_hidden_states=False)
                    assert len(captured) == before + 1
                    middle_tensor = captured.pop()
                    final_tensor = output.last_hidden_state
                    special = special_mask if isinstance(special_mask, torch.Tensor) else torch.from_numpy(np.asarray(special_mask))
                    valid_tensor = batch['attention_mask'].bool() & ~special[selected].to(device).bool()
                    count = valid_tensor.sum(dim=1)
                    pooled_tensor = (middle_tensor.float() * valid_tensor.unsqueeze(-1)).sum(dim=1)
                    pooled_tensor = pooled_tensor / count.clamp_min(1).float().unsqueeze(1)
                    pooled_tensor = torch.where((count == 0).unsqueeze(1), middle_tensor[:, 0].float(), pooled_tensor)
                    x6 = torch.nn.functional.normalize(pooled_tensor, p=2, dim=1)
                    x12 = torch.nn.functional.normalize(final_tensor[:, 0], p=2, dim=1)
                    assert x6.dtype == x12.dtype == torch.float32
                    exact.append(torch.stack((x6, x12)).cpu().numpy())
                middle = middle_tensor.float().cpu().numpy().astype(np.float64)
                final = final_tensor.float().cpu().numpy().astype(np.float64)
                assert middle.shape == final.shape and middle.shape[2] == 384
                valid = valid_tensor.cpu().numpy()
                pooled = np.zeros((len(selected), 384), dtype=np.float64)
                for row in range(len(selected)):
                    selected_tokens = middle[row, valid[row]]
                    pooled[row] = selected_tokens.mean(axis=0) if len(selected_tokens) else middle[row, 0]
                pair = []
                for source in (pooled, final[:, 0]):
                    norm = np.sqrt(np.sum(source * source, axis=1))
                    normalized = source / np.maximum(norm[:, None], 1e-12)
                    assert np.isfinite(normalized).all()
                    pair.append(normalized)
                double.append(np.stack(pair))
    finally:
        handle.remove()
    return {'exact': np.concatenate(exact, axis=1), 'fp64': np.concatenate(double, axis=1)}


def primary_values(gap, actions):
    a = {}
    for arm in ARMS:
        action = np.asarray(actions[arm])
        assert action.dtype == bool and action.shape == gap.shape
        a[arm] = action.astype(np.int64)
    return np.column_stack([(a['H6'] - a['B6']) * gap, (a['H12'] - a['B12']) * gap,
        (a['H6'] - a['H12']) * gap,
        ((a['H6'] - a['H12']) - (a['B6'] - a['B12'])) * gap,
        a['H6'] * gap, (a['H6'] - 1) * gap])


def decisions(point, intervals):
    point, intervals = np.asarray(point), np.asarray(intervals)
    assert point.shape == (6,) and intervals.shape == (6, 2)
    recipe = bool(point[0] >= .002 and intervals[0, 0] > 0)
    candidate = recipe and all(point[j] >= .01 and intervals[j, 0] > 0 for j in (4, 5))
    return recipe, bool(candidate)


def compare_tree(actual, expected, name, errors):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (name, set(actual) ^ set(expected))
        for key, value in expected.items():
            compare_tree(actual[key], value, name + '/' + key, errors)
    elif expected is None or isinstance(expected, (bool, str)):
        assert actual == expected, name
    elif isinstance(expected, (int, np.integer)):
        assert isinstance(actual, (int, np.integer)) and actual == expected, name
    else:
        independent.compare(actual, expected, name, errors)


def expected_config():
    return dict(regularization=.001, formal_solves=15, pilot_solves=3, new_encoder_fits=0,
        new_api_calls=0, new_thresholds=0, bootstrap_draws=20000, bootstrap_seed=BOOTSTRAP_SEED,
        quantiles=QUANTILES, minimum_increment=.002, minimum_fixed_gain=.01,
        batch_size=8, max_length=128, gradient_acceptance=1e-8)


def view_array(value, rows, name):
    value = independent.real_array(value, (2, rows, 384), name, np.float32).copy()
    norms = np.sqrt(np.einsum('vnd,vnd->vn', value.astype(float), value.astype(float), optimize=False))
    assert float(np.max(abs(norms - 1))) < 2e-6, name
    return value


def load_base_inputs():
    """Load the bound base views, policies and splits; no privileged probe data."""
    directory = BASE / 'layer_pooling_v1'
    completed = read(directory / 'completion_record.json')['artifact_sha256']
    old_hashes = {Path(name).resolve(): value for name, value in completed.items()}
    required = [directory / name for name in ('features.npz', 'predictions.npz')]
    required += [directory / f'fold{f}_M6_{suffix}' for f in range(5) for suffix in ('head.pt', 'fit.npz')]
    assert all(path.resolve() in old_hashes and sha(path) == old_hashes[path.resolve()] for path in required)
    split_path = BASE / 'e02_results/fold_indices.npz'
    assert sha(split_path) == independent.EXPECTED_FOLDS_SHA256
    with np.load(directory / 'features.npz', allow_pickle=False) as z:
        qids, groups = z['query_ids'].copy(), z['group_ids'].copy()
        assert qids.shape == groups.shape == (9600,) and qids.dtype.kind in 'US' and groups.dtype.kind in 'US'
        assert len(set(qids)) == 9600 and len(set(groups)) == 9559
        views = view_array(np.stack((z['M6'], z['C12'])), 9600, 'historical base views')
    with np.load(directory / 'predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(z['query_ids'], qids) and np.array_equal(z['group_ids'], groups)
        utility = independent.real_array(z['utility'], (9600, 2), 'historical utility', np.float64).copy()
        linear = independent.real_array(z['M6'], (9600,), 'historical M6 scores', np.float64).copy()
        assert np.all((utility >= 0) & (utility <= 1))
    with np.load(split_path, allow_pickle=False) as z:
        folds = [{part: z[f'fold{f}_{part}'].copy() for part in ('fit', 'calibration', 'test')} for f in range(5)]
    offset = BASE / 'm6_pooled_offset_v1'
    cal_path = offset / 'cal_logits.npz'
    offset_hashes = {Path(name).resolve(): value for name, value in read(offset / 'completion_record.json')['artifact_sha256'].items()}
    assert cal_path.resolve() in offset_hashes and sha(cal_path) == offset_hashes[cal_path.resolve()]
    coverage = np.zeros(9600, dtype=np.int64)
    fit_scores, cal_scores, coefs, biases = [], [], [], []
    import torch
    with np.load(cal_path, allow_pickle=False) as calibration:
        for f, parts in enumerate(folds):
            group_sets = []
            for part, size in (('fit', 6144), ('calibration', 1536), ('test', 1920)):
                ids = independent.real_array(parts[part], (size,), 'historical ' + part)
                assert ids.dtype.kind in 'iu' and len(set(ids)) == size
                assert np.all((ids >= 0) & (ids < 9600))
                group_sets.append(set(groups[ids]))
            assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
            assert np.array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(9600))
            coverage[parts['test']] += 1
            assert np.array_equal(calibration[f'fold{f}_cal_indices'], parts['calibration'])
            cal_scores.append(independent.real_array(calibration[f'fold{f}_O'], (1536,), 'M6 cal', np.float32).copy())
            with np.load(directory / f'fold{f}_M6_fit.npz', allow_pickle=False) as z:
                assert np.array_equal(z['fit_indices'], parts['fit'])
                assert np.array_equal(z['inner_assignment'], independent.inner_groups(groups[parts['fit']], f))
                fit_scores.append(independent.real_array(z['final_native_fit_logits'], (6144,), 'M6 fit', np.float32).copy())
            head = torch.load(directory / f'fold{f}_M6_head.pt', map_location='cpu', weights_only=True)
            coefs.append(independent.real_array(head['weight'].numpy(), (1, 384), 'M6 coef', np.float32).reshape(384).copy())
            biases.append(float(independent.real_array(head['bias'].numpy(), (1,), 'M6 bias', np.float32)[0]))
    assert np.all(coverage == 1)
    return dict(query_ids=qids, group_ids=groups, features=views[0], C12=views[1], utility=utility,
        gap=utility[:, 0] - utility[:, 1], L_scores=linear, folds=folds, L_fit=fit_scores,
        L_cal=cal_scores, L_coef=coefs, L_intercept=biases)


def restore_H(fold, torch):
    """Restore only the already-trained tensors into the bound local base model."""
    from transformers import AutoModel
    checkpoint = torch.load(H_DIRECTORY / f'fold{fold}.pt', map_location='cpu', weights_only=True)
    assert checkpoint['fold'] == fold
    assert checkpoint['protocol_sha256'] == sha(H_DIRECTORY / 'protocol.json')
    assert checkpoint['protocol_id'] == read(H_DIRECTORY / 'protocol.json')['id']
    assert checkpoint['runner_sha256'] == sha(BASE / 'run_lp_ft_v2.py')
    parameters = checkpoint['trained_parameters']
    assert len(parameters) == 199 and sum(t.numel() for t in parameters.values()) == 33212545
    assert all(name.startswith(('encoder.', 'answer.')) for name in parameters)
    encoder = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=True)
    replacements = {name[len('encoder.'):]: tensor for name, tensor in parameters.items() if name.startswith('encoder.')}
    assert all(torch.isfinite(tensor).all() for tensor in replacements.values())
    missing = encoder.load_state_dict(replacements, strict=False)
    assert set(missing.missing_keys) == {'pooler.dense.weight', 'pooler.dense.bias'}
    assert not missing.unexpected_keys
    state = encoder.state_dict()
    assert all(torch.equal(state[key], tensor) for key, tensor in replacements.items())
    coef = parameters['answer.weight'].numpy().reshape(384).copy()
    bias = float(parameters['answer.bias'].item())
    encoder.requires_grad_(False).to('cuda').eval()
    return encoder, coef, bias


def native_scores(features, coef, bias, torch):
    """Independent single-branch BF16 head replay; no encoder call or fitting."""
    layer = torch.nn.Linear(384, 1, device='cuda')
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(np.asarray(coef, np.float32).reshape(1, 384)))
        layer.bias.copy_(torch.tensor([bias], dtype=torch.float32))
    layer.requires_grad_(False).eval()
    result = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(features), 8):
            tensor = torch.from_numpy(features[start:start + 8]).to('cuda')
            result.append(layer(tensor).flatten().float().cpu().numpy())
    return np.concatenate(result)


def sample_test(test, query_ids):
    return np.asarray(sorted(map(int, test), key=lambda i: (
        hashlib.sha256(('m6_trained_readout_check_v1|' + str(query_ids[i])).encode('utf-8')).digest(),
        str(query_ids[i])))[:32], dtype=np.int64)


def check_forward(encoder, tokens, ids, cached_views, native, original_coef, original_bias,
                  torch, errors, name):
    replay = hook_views(encoder, tokens, tokens['special_tokens_mask'], ids, torch)
    assert np.array_equal(replay['exact'], cached_views), name + ': original precision views differ'
    independent.compare(replay['fp64'], cached_views, 'FP64_pooling', errors, atol=2e-6)
    observed = native_scores(replay['exact'][1], original_coef, original_bias, torch)
    assert np.array_equal(observed, native), name + ': original H head differs'
    return replay['exact']


def self_test():
    """Synthetic CPU hook, objective derivative, contrasts and resampling only."""
    import torch
    from types import SimpleNamespace

    class Block(torch.nn.Module):
        def __init__(self, number):
            super().__init__()
            self.number = number

        def forward(self, hidden):
            return (hidden + self.number / 17.,)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.layer = torch.nn.ModuleList([Block(i + 1) for i in range(12)])
            self.calls = 0

        def forward(self, input_ids, attention_mask, token_type_ids, output_hidden_states=False):
            assert output_hidden_states is False
            self.calls += 1
            hidden = input_ids.float().unsqueeze(-1) / 31 + torch.arange(384).float()[None, None] / 383
            for block in self.encoder.layer:
                hidden = block(hidden)[0]
            return SimpleNamespace(last_hidden_state=hidden)

    rng = np.random.default_rng(2026091514)
    tokens = {name: rng.integers(0, 20, (16, 9), dtype=np.int64)
              for name in ('input_ids', 'token_type_ids')}
    tokens['attention_mask'] = np.ones((16, 9), np.int64)
    special = np.zeros((16, 9), np.int64); special[:, 0] = 1; special[0] = 1
    tokens['attention_mask'][3, 4:] = 0
    encoder = Encoder()
    replay = hook_views(encoder, tokens, special, np.arange(16), torch, device='cpu')
    assert encoder.calls == 2 and not encoder.encoder.layer[5]._forward_hooks
    np.testing.assert_allclose(replay['exact'], replay['fp64'], atol=2e-7, rtol=0)
    hidden = torch.from_numpy(tokens['input_ids']).float().unsqueeze(-1) / 31 + torch.arange(384).float()[None, None] / 383
    for i in range(12):
        hidden = hidden + (i + 1) / 17.
        if i == 5:
            middle = hidden.clone()
    valid = tokens['attention_mask'].astype(bool) & ~special.astype(bool)
    rows = []
    for i in range(16):
        value = middle[i, valid[i]].mean(dim=0) if valid[i].any() else middle[i, 0]
        rows.append(torch.nn.functional.normalize(value, dim=0).numpy())
    np.testing.assert_allclose(replay['exact'][0], rows, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(replay['exact'][1], torch.nn.functional.normalize(hidden[:, 0], dim=1).numpy())
    split = [hook_views(encoder, tokens, special, np.arange(i, i + 8), torch, device='cpu')['exact'] for i in (0, 8)]
    np.testing.assert_array_equal(replay['exact'], np.concatenate(split, axis=1))
    x = rng.normal(size=(19, 4)); gap = rng.uniform(-1, 1, 19); gap[0] = 0.
    beta, bias = rng.normal(size=4), .29
    q, weights = (gap > 0).astype(float), independent.weights_for_gap(gap)
    quantities = independent.loss_quantities(x, q, weights, beta, bias)
    theta = np.r_[beta, bias]; finite_difference = []
    for j in range(5):
        delta = np.eye(5)[j] * 1e-6
        a, b = theta + delta, theta - delta
        finite_difference.append((independent.loss_quantities(x, q, weights, a[:-1], a[-1])['loss'] -
            independent.loss_quantities(x, q, weights, b[:-1], b[-1])['loss']) / 2e-6)
    gradient_error = float(np.max(abs(np.asarray(finite_difference) - quantities['gradient'])))
    assert gradient_error < 2e-9
    actions = {name: rng.integers(0, 2, 19).astype(bool) for name in ARMS}
    values = primary_values(gap, actions)
    for i in range(19):
        u = {name: float(actions[name][i]) * gap[i] for name in ARMS}
        expected = [u['H6'] - u['B6'], u['H12'] - u['B12'], u['H6'] - u['H12'],
                    u['H6'] - u['H12'] - u['B6'] + u['B12'], u['H6'], u['H6'] - gap[i]]
        np.testing.assert_allclose(values[i], expected, atol=1e-15, rtol=0)
    groups = np.asarray(['g' + str(i // 3) for i in range(19)])
    samples, _ = independent.frequency_bootstrap(values, groups, draws=31, seed=183)
    members = [np.flatnonzero(groups == name) for name in sorted(set(groups))]
    replay_rng = np.random.default_rng(183)
    for j in range(31):
        ids = np.concatenate([members[k] for k in replay_rng.integers(len(members), size=len(members))])
        expected = [math.fsum(values[ids, k]) / len(ids) for k in range(6)]
        np.testing.assert_allclose(samples[j], expected, atol=1e-15, rtol=0)
    points = np.array([.002, -1., -1., -1., .01, .01])
    intervals = np.column_stack((np.full(6, 1e-8), np.ones(6)))
    assert decisions(points, intervals) == (True, True)
    intervals[3, 0] = -1.; assert decisions(points, intervals) == (True, True)
    intervals[0, 0] = 0.; assert decisions(points, intervals) == (False, False)
    return dict(status='passed_synthetic_hook_pooling_gradient_contrast_and_bootstrap_checks',
        FP64_pooling_max_difference=float(np.max(abs(replay['exact'] - replay['fp64']))),
        finite_difference_gradient_error=gradient_error, bootstrap_draws=31,
        real_data_reads=0, new_fits=0, GPU_forwards=0)


def check(output_dir=OUT, protocol_sha=None):
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    destination = output / 'separate_checks.json'
    assert not destination.exists(), 'Preserve an existing check receipt'
    binding = sha(output / 'protocol.json')
    assert protocol_sha and binding == protocol_sha
    protocol = read(output / 'protocol.json')
    assert protocol['status'] == 'frozen_before_new_H_views_and_readouts'
    assert protocol['config'] == expected_config()
    assert protocol['primary'] == PRIMARY and protocol['policies'] == POLICIES
    assert protocol['versions'] == {name: importlib.metadata.version(name)
        for name in ('numpy', 'scipy', 'torch', 'transformers', 'threadpoolctl')}
    sources = independent.bindings(protocol['source_sha256'], 'trained readout source')
    inputs = independent.bindings(protocol['input_sha256'], 'trained readout input')
    required_sources = [Path(__file__), Path(independent.__file__),
        ROOT / 'scripts/run_m6_trained_readout.py', ROOT / 'scripts/m6_probe_math.py',
        ROOT / 'scripts/run_m6_objective_readout.py', ROOT / 'scripts/m6_objective_math.py',
        ROOT / 'analysis/hotpotqa_router/m6_trained_readout_plan_20260915.md',
        BASE / 'weighted_linear_probe.py', BASE / 'weighted_linear_probe_refined.py']
    assert set(sources) == {p.resolve() for p in required_sources}
    required_inputs = [BASE / 'layer_pooling_v1' / name for name in
        ('features.npz', 'predictions.npz', 'tokens.npz', 'protocol.json', 'completion_record.json')]
    required_inputs += [BASE / name for name in ('e02_results/fold_indices.npz',
        'm6_pooled_offset_v1/cal_logits.npz', 'm6_pooled_offset_v1/completion_record.json')]
    required_inputs += [BASE / 'layer_pooling_v1' / f'fold{f}_M6_{suffix}' for f in range(5)
        for suffix in ('head.pt', 'fit.npz')]
    required_inputs += [H_DIRECTORY / name for name in ('protocol.json', 'completion_record.json',
        'separate_checks.json', 'all_endpoints_frozen.json', 'predictions.npz')]
    required_inputs += [H_DIRECTORY / f'fold{f}{suffix}' for f in range(5)
        for suffix in ('.pt', '_fit.npz', '_training.json', '_lp_head.pt', '_lp_predictions.npz')]
    required_inputs += [MODEL / name for name in ('config.json', 'model.safetensors')]
    required_inputs += [BASE / name for name in ('run_lp_ft_v2.py', 'encoder_scope_training.py',
        'run_e11c.py', 'run_e10.py', 'run_e02.py')]
    assert {p.resolve() for p in required_inputs} == set(inputs)
    prior_check = read(H_DIRECTORY / 'separate_checks.json')
    prior_completion = read(H_DIRECTORY / 'completion_record.json')
    assert prior_check['status'] == 'passed_independent_LP_H_F_OOF_training_and_four_comparison_checks'
    assert prior_check['protocol_sha256'] == sha(H_DIRECTORY / 'protocol.json')
    assert prior_check['results_sha256'] == sha(H_DIRECTORY / 'results.json')
    assert prior_completion['checks_sha256'] == sha(H_DIRECTORY / 'separate_checks.json')
    historical = {}
    for mapping in (read(BASE / 'layer_pooling_v1/completion_record.json')['artifact_sha256'],
                    prior_completion['artifact_sha256'], prior_check['artifact_sha256'], prior_check['source_sha256']):
        historical.update({Path(name).resolve(): digest for name, digest in mapping.items()})
    for path, digest in inputs.items():
        if path in historical:
            assert digest == historical[path], 'Historical binding differs: ' + str(path)
    training_rule = read(H_DIRECTORY / 'protocol.json')['training_rule']
    assert training_rule['max_length'] == 128 and training_rule['batch_size'] == 8
    for f in range(5):
        stat = read(H_DIRECTORY / f'fold{f}_training.json')
        assert stat['fold'] == f and stat['stopping_state'] == 'criterion_met'
        assert stat['calibration_quality_evaluations'] == stat['outer_test_quality_evaluations'] == 0
        assert stat['all_losses_and_gradients_finite'] is True
    pilot = read(output / 'pilot.json')
    assert pilot['status'] == 'passed_base_views_H_native_pooling_and_three_head_pilot'
    assert pilot['protocol_sha256'] == binding and pilot['new_policy_effects'] is False
    assert pilot['new_encoder_fits'] == 0 and pilot['new_encoder_query_forwards'] == 192
    assert pilot['base_feature_max_difference'] == pilot['H_native_max_difference'] == 0.
    assert 0 <= pilot['independent_FP64_pool_max_difference'] < 2e-6
    assert pilot['original_native_controls'] == {f'fold{f}_{part}': 0.
        for f in range(5) for part in ('fit', 'cal', 'test')}
    pilot_paths = {output / 'pilot_features.npz'} | {output / ('pilot_' + arm + suffix)
        for arm in NEW_ARMS for suffix in ('.npz', '.json', '_scores.npz')}
    assert set(independent.bindings(pilot['artifact_sha256'], 'readout pilot')) == pilot_paths
    completed = read(output / 'fit_completion.json')
    assert completed['status'] == 'all_15_heads_frozen_before_new_test_views_and_predictions'
    assert completed['protocol_sha256'] == binding and completed['formal_solves'] == 15
    assert completed['encoder_fit_cal_query_forwards'] == 38400
    formal_paths = {output / 'solutions.jsonl'}
    formal_paths |= {output / f'fold{f}_{arm}{suffix}' for f in range(5) for arm in NEW_ARMS
                    for suffix in ('.npz', '.json')}
    formal_paths |= {output / f'fold{f}_{part}.npz' for f in range(5)
                    for part in ('features_fit_cal', 'scores_fit_cal')}
    assert set(independent.bindings(completed['artifact_sha256'], 'readout fit')) == formal_paths
    fit_start = read(output / 'fit_started.json')
    assert fit_start['protocol_sha256'] == binding
    assert datetime.fromisoformat(protocol['created_at_utc']) <= datetime.fromisoformat(fit_start['started_at_utc'])
    assert datetime.fromisoformat(fit_start['started_at_utc']) <= datetime.fromisoformat(completed['completed_at_utc'])
    frozen = read(output / 'predictions_frozen.json')
    assert frozen['status'] == 'all_predictions_before_effects' and frozen['protocol_sha256'] == binding
    assert frozen['predictions_sha256'] == sha(output / 'predictions.npz')
    assert frozen['fit_completion_sha256'] == sha(output / 'fit_completion.json')
    assert frozen['encoder_test_query_forwards'] == 9600
    test_paths = {output / f'fold{f}_features_test.npz' for f in range(5)}
    assert set(independent.bindings(frozen['test_feature_sha256'], 'readout test features')) == test_paths
    result = read(output / 'results.json')
    assert result['status'] == 'complete_trained_readout_pending_separate_check' and result['protocol_sha256'] == binding
    assert result['predictions_frozen_sha256'] == sha(output / 'predictions_frozen.json')
    assert result['bootstrap_sha256'] == sha(output / 'bootstrap.npz')
    for key, expected in dict(formal_solves=15, pilot_solves=3, new_encoder_fits=0,
        new_encoder_query_forwards=48000, pilot_encoder_query_forwards=192, new_api_calls=0).items():
        assert result[key] == expected
    assert result['runtime_deployed'] is False and result['core_goal_achieved'] is False
    assert set(result['primary']) == set(PRIMARY) and set(result['policy']) == set(POLICIES)
    artifact_paths = formal_paths | pilot_paths | test_paths | {output / name for name in
        ('protocol.json', 'pilot.json', 'fit_started.json', 'fit_completion.json', 'predictions.npz',
         'predictions_frozen.json', 'bootstrap.npz', 'results.json')}
    snapshot = {str(path): sha(path) for path in sorted(artifact_paths)}
    data = load_base_inputs()
    gap, utility, groups, qids = [data[key] for key in ('gap', 'utility', 'group_ids', 'query_ids')]
    C12 = data['C12']
    with np.load(BASE / 'layer_pooling_v1/tokens.npz', allow_pickle=False) as z:
        assert set(z.files) == {'input_ids', 'attention_mask', 'token_type_ids', 'special_tokens_mask'}
        tokens = {key: independent.real_array(z[key], (9600, 128), key, np.int64).copy() for key in z.files}
    with np.load(H_DIRECTORY / 'predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(z['query_ids'], qids) and np.array_equal(z['group_ids'], groups)
        assert np.array_equal(z['utility'], utility)
        H_native, C12_prior = z['H_native'].copy(), z['P_native'].copy()
    with np.load(output / 'predictions.npz', allow_pickle=False) as z:
        assert set(z.files) == {'query_ids', 'group_ids', 'utility', 'arm_scores', 'fold_id', 'B6_scores', 'H_native_scores'}
        assert np.array_equal(z['query_ids'], qids) and np.array_equal(z['group_ids'], groups)
        assert np.array_equal(z['utility'], utility) and np.array_equal(z['B6_scores'], data['L_scores'])
        assert np.array_equal(z['H_native_scores'], H_native)
        scores = independent.real_array(z['arm_scores'], (3, 9600), 'formal test scores', np.float32).copy()
        fold_id = independent.real_array(z['fold_id'], (9600,), 'fold ID', np.int64).copy()
    errors, counters, models, journal_expected, fold_records = {}, {}, [], [], []
    gpu_samples = []
    with np.load(output / 'pilot_features.npz', allow_pickle=False) as z:
        assert set(z.files) == {'indices', 'base_views', 'H_views', 'H_native'}
        pilot_ids = z['indices'].copy()
        assert np.array_equal(pilot_ids, data['folds'][0]['fit'][:64])
        base_pilot = view_array(z['base_views'], 64, 'base pilot')
        H_pilot = view_array(z['H_views'], 64, 'H pilot')
        pilot_native = z['H_native'].copy()
        assert np.array_equal(base_pilot[0], data['features'][pilot_ids])
        assert np.array_equal(base_pilot[1], C12[pilot_ids])
    # Imports are local; self-test never loads real data or initializes CUDA.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0',
                      USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):
        pilot_models = []
        for j, (arm, features) in enumerate(zip(NEW_ARMS, (base_pilot[1], H_pilot[0], H_pilot[1]))):
            model = independent.checked_model(output, 'pilot_' + arm, features, gap[pilot_ids],
                (gap[pilot_ids] > 0).astype(float), dict(role='pilot', arm=arm, rows=64), errors)
            models.append(model); pilot_models.append(model)
            with np.load(output / ('pilot_' + arm + '_scores.npz'), allow_pickle=False) as z:
                assert set(z.files) == {'scores'}
                model['saved_scores'] = z['scores'].copy()
            independent.native_score_bound(features, model['coef'], model['intercept'], model['saved_scores'],
                counters, 'pilot_' + arm)
        for f, parts in enumerate(data['folds']):
            fit, cal, test = [parts[key] for key in ('fit', 'calibration', 'test')]
            assert np.array_equal(np.flatnonzero(fold_id == f), np.sort(test))
            for name, ids, values in (('fit', fit, data['L_fit'][f]), ('cal', cal, data['L_cal'][f]),
                                      ('test', test, data['L_scores'][test].astype(np.float32))):
                if name == 'test':
                    assert np.array_equal(values.astype(np.float64), data['L_scores'][test])
                independent.native_score_bound(data['features'][ids], data['L_coef'][f], data['L_intercept'][f],
                    values, counters, f'fold{f}_B6/{name}')
            with np.load(output / f'fold{f}_features_fit_cal.npz', allow_pickle=False) as z:
                assert set(z.files) == {'fit_indices', 'cal_indices', 'fit_features', 'cal_features', 'fit_H_native', 'cal_H_native'}
                assert np.array_equal(z['fit_indices'], fit) and np.array_equal(z['cal_indices'], cal)
                fit_views = view_array(z['fit_features'], 6144, 'fit views')
                cal_views = view_array(z['cal_features'], 1536, 'cal views')
                fit_native, cal_native = z['fit_H_native'].copy(), z['cal_H_native'].copy()
            with np.load(output / f'fold{f}_features_test.npz', allow_pickle=False) as z:
                assert set(z.files) == {'test_indices', 'test_features', 'H_native'}
                assert np.array_equal(z['test_indices'], test)
                test_views = view_array(z['test_features'], 1920, 'test views')
                assert np.array_equal(z['H_native'], H_native[test])
            with np.load(H_DIRECTORY / f'fold{f}_fit.npz', allow_pickle=False) as z:
                assert np.array_equal(z['fit_indices'], fit) and np.array_equal(z['fit_logits'], fit_native)
            with np.load(output / f'fold{f}_scores_fit_cal.npz', allow_pickle=False) as z:
                assert set(z.files) == {'fit_indices', 'cal_indices', 'fit_scores', 'cal_scores'}
                assert np.array_equal(z['fit_indices'], fit) and np.array_equal(z['cal_indices'], cal)
                fit_scores = independent.real_array(z['fit_scores'], (3, 6144), 'fit scores', np.float32).copy()
                cal_scores = independent.real_array(z['cal_scores'], (3, 1536), 'cal scores', np.float32).copy()
            fold_models = []
            for j, arm in enumerate(NEW_ARMS):
                features = C12[fit] if j == 0 else fit_views[j - 1]
                model = independent.checked_model(output, f'fold{f}_{arm}', features, gap[fit],
                    (gap[fit] > 0).astype(float), dict(role='formal', fold=f, arm=arm), errors)
                models.append(model); fold_models.append(model)
                journal_expected.append(dict(stem=model['stem'], **model['record']))
                for name, ids, views, values in (('fit', fit, fit_views, fit_scores[j]),
                        ('cal', cal, cal_views, cal_scores[j]), ('test', test, test_views, scores[j, test])):
                    x = C12[ids] if j == 0 else views[j - 1]
                    independent.native_score_bound(x, model['coef'], model['intercept'], values,
                        counters, f'fold{f}_{arm}/{name}')
            if f > 0:
                prior = torch.load(H_DIRECTORY / f'fold{f}_lp_head.pt', map_location='cpu', weights_only=True)
                assert np.array_equal(fold_models[0]['coef'].astype(np.float32), prior['weight'].numpy().reshape(384))
                assert np.float32(fold_models[0]['intercept']) == prior['bias'].numpy()[0]
                with np.load(H_DIRECTORY / f'fold{f}_lp_predictions.npz', allow_pickle=False) as z:
                    assert np.array_equal(fit_scores[0], z['final_native_fit_logits'])
                assert np.array_equal(scores[0, test], C12_prior[test])
            encoder, original_coef, original_bias = restore_H(f, torch)
            for name, views, values in (('fit', fit_views, fit_native), ('cal', cal_views, cal_native),
                                        ('test', test_views, H_native[test])):
                independent.native_score_bound(views[1], original_coef, original_bias, values,
                    counters, f'fold{f}_H_native/{name}')
            if f == 0:
                assert np.array_equal(H_pilot, fit_views[:, :64]) and np.array_equal(pilot_native, fit_native[:64])
                exact_pilot = check_forward(encoder, tokens, pilot_ids, H_pilot, pilot_native,
                    original_coef, original_bias, torch, errors, 'pilot')
                for j, model in enumerate(pilot_models):
                    features = C12[pilot_ids] if j == 0 else exact_pilot[j - 1]
                    assert np.array_equal(native_scores(features, model['coef'], model['intercept'], torch), model['saved_scores'])
            chosen_test = sample_test(test, qids)
            test_positions = np.asarray([int(np.flatnonzero(test == i)[0]) for i in chosen_test])
            chosen = np.r_[fit[:8], chosen_test]
            cached = np.concatenate((fit_views[:, :8], test_views[:, test_positions]), axis=1)
            original_expected = np.r_[fit_native[:8], H_native[chosen_test]]
            replay = check_forward(encoder, tokens, chosen, cached, original_expected, original_coef,
                original_bias, torch, errors, f'fold{f}')
            for j, model in enumerate(fold_models):
                features = C12[chosen] if j == 0 else replay[j - 1]
                observed = native_scores(features, model['coef'], model['intercept'], torch)
                assert np.array_equal(observed, np.r_[fit_scores[j, :8], scores[j, chosen_test]])
            gpu_samples.append(dict(fold=f, fit_indices=fit[:8].tolist(), test_indices=chosen_test.tolist()))
            fold_records.append(dict(fold=f, fit_BCE=[independent.weighted_bce(s, gap[fit]) for s in fit_scores],
                cal_BCE=[independent.weighted_bce(s, gap[cal]) for s in cal_scores],
                test_BCE=[independent.weighted_bce(s[test], gap[test]) for s in scores],
                base_C12_original_lambda_control=f > 0))
            del encoder, fit_views, cal_views, test_views
            gc.collect(); torch.cuda.empty_cache()
            print(json.dumps(dict(status='independent_fold_checked', fold=f, formal_heads=(f + 1) * 3)), flush=True)
        journal = [json.loads(line) for line in (output / 'solutions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert journal == journal_expected and len(journal) == 15
        actions = {arm: scores[j] > 0 for j, arm in enumerate(NEW_ARMS)}
        actions.update(B6=data['L_scores'] > 0, H_native=H_native > 0,
            Dense=np.zeros(9600, bool), BM25=np.ones(9600, bool))
        values = primary_values(gap, actions)
        draws, _ = independent.frequency_bootstrap(values, groups, draws=20000, seed=BOOTSTRAP_SEED)
        intervals = np.quantile(draws, QUANTILES, axis=0, method='linear').T
        point = independent.means(values)
        with np.load(output / 'bootstrap.npz', allow_pickle=False) as z:
            assert set(z.files) == {'draws'}
            independent.compare(z['draws'], draws, 'bootstrap_draws', errors)
        expected_primary = {name: dict(mean=float(point[j]), interval=intervals[j].tolist())
            for j, name in enumerate(PRIMARY)}
        compare_tree(result['primary'], expected_primary, 'primary', errors)
        compare_tree(result['policy'], {name: independent.policy_summary(utility, actions[name])
            for name in POLICIES}, 'policy', errors)
        for f, parts in enumerate(data['folds']):
            fold_point = independent.means(values[parts['test']])
            fold_records[f]['primary_means'] = {name: float(fold_point[j]) for j, name in enumerate(PRIMARY)}
            compare_tree(result['folds'][f], fold_records[f], f'fold{f}', errors)
        recipe, candidate = decisions(point, intervals)
        assert result['recipe_gate'] is recipe and result['candidate_preparation_gate'] is candidate
        decision = ('PREPARE_H6_INDEPENDENT_CONFIRMATION' if candidate else
            'H6_INTERNAL_INCREMENT_ONLY' if recipe else 'END_FIXED_H_CHECKPOINT_READOUT_NO_CONFIRMED_INCREMENT')
        assert result['decision'] == decision
    independent.bindings(snapshot, 'completed readout artifact after check')
    independent.bindings(protocol['source_sha256'], 'readout source after check')
    independent.bindings(protocol['input_sha256'], 'readout input after check')
    receipt = dict(status='passed_independent_trained_readout_checks', protocol_sha256=binding,
        results_sha256=sha(output / 'results.json'), checker_sha256=sha(Path(__file__)),
        source_sha256=protocol['source_sha256'], input_sha256=protocol['input_sha256'], artifact_sha256=snapshot,
        formal_heads_checked=15, pilot_heads_checked=3, solution_journal_rows=15,
        maximum_independent_gradient_inf=max(model['gradient_inf'] for model in models),
        optimizer_iterations_max=max(model['record']['iterations'] for model in models),
        refinement_used_heads=sum(model['record']['numerical_refinement']['iterations'] > 0 for model in models),
        native_score_checks=counters, original_H_fit_test_exact=True, B12_folds_1_to_4_head_fit_test_exact=True,
        independent_H_encoder_query_forwards=264, pilot_query_indices=pilot_ids.tolist(), GPU_samples=gpu_samples,
        default_forward_hook_original_precision_features_and_scores_exact=True,
        FP64_pooling_tolerance=2e-6, numerical_max_errors=errors, bootstrap_draws=20000,
        bootstrap_seed=BOOTSTRAP_SEED, quantiles=QUANTILES, recipe_gate=recipe,
        candidate_preparation_gate=candidate, new_fits=0, new_external_calls=0,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
        scope='Checks saved development records and fixed GPU samples; no independent-source validation or deployment')
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--protocol-sha')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False))
    else:
        checked = check(args.output_dir, args.protocol_sha)
        print(json.dumps({key: checked[key] for key in ('status', 'formal_heads_checked', 'pilot_heads_checked',
            'maximum_independent_gradient_inf', 'independent_H_encoder_query_forwards', 'elapsed_seconds')}, allow_nan=False))
