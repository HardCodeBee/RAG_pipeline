"""One bounded group-OOF local-offset experiment on consumed 2Wiki development.

Uses saved query features and mean-three answer utilities only. No encoder,
retriever, generator, retained-question access, or hyperparameter search.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/2wiki_development_measurement_v1'
OUT = SOURCE.parent / '2wiki_local_residual_v1'
POLICIES = ('BM25', 'Dense', 'M6', 'F', 'C_global', 'K_neighbor', 'L_local', 'S_shuffled')
REFERENCES = ('Dense', 'BM25', 'C_global', 'K_neighbor', 'S_shuffled')
SPEC = {
    'scope': 'consumed_conditional_2Wiki_development_only_not_confirmation',
    'queries': 304, 'groups': 256, 'outer_folds': 4, 'inner_folds': 3,
    'outer_seed': 2026091850, 'inner_seed': 2026091851,
    'shuffle_seed': 2026091852, 'bootstrap_seed': 2026091854,
    'fold_assignment': 'sort_groups_by_SHA256_seed_context_group_then_round_robin_no_labels',
    'features': 'saved_M6_mean6_384D_FP64_cosine_normalization',
    'base_logits': 'saved_historical_old9600_M6_head_no_refit',
    'neighborhood': '16_nearest_outer_training_queries_cosine_ties_by_query_ID',
    'k': 16,
    'k_rationale': 'single_sqrt_scale_choice_for_roughly_228_outer_training_queries_not_selected_by_labels',
    'ridge': 0.01,
    'target': 'gap=mean3_F1_BM25_minus_Dense; w=abs(gap); y=gap>0; ties_kept_with_zero_weight',
    'offset_objective': 'mean(w*(softplus(base_logit+t)-y*(base_logit+t)))+0.5*ridge*t^2',
    'solver': 'strictly_convex_scalar_60_bisections_in_plus_minus_1_over_ridge',
    'global_C': 'fit_offset_on_outer_train_original_logits; predict_l_query+a_outer',
    'residual_bases': 'l_i+a_inner_OOF_group_disjoint_within_outer_train',
    'local_L': 'fit_offset_on_16_neighbor_inner_OOF_bases; predict_l_query+a_outer+t_query',
    'direct_K': 'mean_of_same_16_neighbor_raw_continuous_gaps',
    'shuffled_S': 'move_complete_base_and_gap_group_blocks_within_equal_size_strata; fixed_random_order_cyclic_shift; unique_size_strata_stay',
    'S_prediction': 'same_global_C_plus_same_local_solver_on_shuffled_neighbor_targets',
    'threshold': 0, 'tie_action': 'Dense',
    'primary_candidate': 'L_local',
    'minimum_gain_over_each_fixed': 0.01, 'minimum_gain_over_each_control': 0.002,
    'gate_comparisons': list(REFERENCES),
    'bootstrap_when': 'only_if_all_five_point_gain_thresholds_pass',
    'bootstrap_draws': 10000, 'CI_per_comparison': 0.99,
    'CI_role': 'conditional_on_fitted_OOF_predictions_consumed_development; does_not_cover_refitting_or_method_selection_uncertainty',
    'family': 'five_L_contrasts_Bonferroni_nominal_95_percent_conditional_family',
    'primary_metric': 'query_weighted_mean3_normalized_token_F1', 'secondary': 'EM_descriptive',
    'failure_rule': 'close_recipe_no_k_distance_ridge_seed_threshold_or_split_extension',
    'passing_role': 'development_candidate_only_prepare_new_independent_protocol',
    'new_encoder_calls': 0, 'new_retrieval_calls': 0, 'new_answer_calls': 0,
    'retained_groups_accessed': 0, 'final_full_sample_model_fit': False,
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def group_folds(groups, folds, seed, context):
    ordered = sorted(set(groups), key=lambda g: (hashlib.sha256(f'{seed}|{context}|{g}'.encode()).digest(), g))
    mapping = {g: i % folds for i, g in enumerate(ordered)}
    return np.array([mapping[g] for g in groups], dtype=np.int64)


def shuffled_group_map(qids, groups, train, seed):
    by_group = {g: np.array(sorted(train[groups[train] == g], key=lambda i: qids[i])) for g in sorted(set(groups[train]))}
    strata = defaultdict(list)
    for g, rows in by_group.items():
        strata[len(rows)].append(g)
    rng = np.random.default_rng(seed)
    mapping = np.full(len(qids), -1, dtype=np.int64)
    for size in sorted(strata):
        names = np.array(strata[size])
        ordered = names[rng.permutation(len(names))]
        sources = np.roll(ordered, 1) if len(names) > 1 else ordered
        for destination, source in zip(ordered, sources):
            mapping[by_group[destination]] = by_group[source]
    assert set(mapping[train]) == set(train)
    return mapping


def prepare():
    continuation = yaml.safe_load((ROOT / 'analysis/hotpotqa_router/registry.yaml').read_text(encoding='utf-8'))['research_continuation']
    if continuation['experiment_execution'] == 'paused_by_user':
        raise RuntimeError('Research is paused by the user')
    if read(SOURCE / 'evaluation.json')['decision'] != 'diagnostic_only_pass':
        raise ValueError('The prerequisite external complementarity decision is not present')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise ValueError('Protocol already exists; do not redraw roles or overwrite')
    with np.load(SOURCE / 'actions.npz', allow_pickle=False) as a:
        qids, gids = a['query_ids'].astype(str), a['group_ids'].astype(str)
    assert len(qids) == SPEC['queries'] and len(set(gids)) == SPEC['groups']
    outer = group_folds(gids, 4, SPEC['outer_seed'], 'outer')
    inner = np.full((4, len(qids)), -1, dtype=np.int64)
    shuffled = np.full_like(inner, -1)
    roles = []
    for fold in range(4):
        train, test = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
        inner[fold, train] = group_folds(gids[train], 3, SPEC['inner_seed'], f'outer_{fold}_inner')
        shuffled[fold] = shuffled_group_map(qids, gids, train, SPEC['shuffle_seed'] + fold)
        unchanged = shuffled[fold, train] == train
        roles.append({'fold': fold, 'train_queries': len(train), 'test_queries': len(test),
                      'train_groups': len(set(gids[train])), 'test_groups': len(set(gids[test])),
                      'inner_train_partition_queries': [int(np.sum(inner[fold, train] == j)) for j in range(3)],
                      'shuffle_unchanged_queries': int(unchanged.sum()),
                      'shuffle_unchanged_groups': len(set(gids[train[unchanged]]))})
    with (OUT / 'roles.npz').open('xb') as stream:
        np.savez_compressed(stream, query_ids=qids, group_ids=gids, outer_fold=outer, inner_fold=inner, shuffled_source_row=shuffled)
    protocol = {'status': 'fixed_before_new_fits', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'spec': SPEC, 'roles': roles,
                'source_actions': str(SOURCE / 'actions.npz'), 'source_scores': str(SOURCE / 'measurement_scores.npz'),
                'unchanged_environment': str(SOURCE / 'execution_contract.json'),
                'label_selection_disclosure': 'hypothesis_chosen_after_aggregate_2Wiki_measurement; development_only'}
    write_new(OUT / 'protocol.json', protocol)
    print(json.dumps({'status': protocol['status'], 'roles': roles, 'new_fits': 0}, ensure_ascii=False))


def run():
    from router_local_offset import fit_offsets, offset_gradient
    protocol = read(OUT / 'protocol.json')
    if protocol['spec'] != SPEC:
        raise ValueError('Code specification differs from the frozen protocol')
    if (OUT / 'predictions.npz').exists() or (OUT / 'results.json').exists():
        raise ValueError('Experiment output exists; inspect rather than refit')
    with np.load(OUT / 'roles.npz', allow_pickle=False) as a:
        roles = {k: a[k] for k in a.files}
    with np.load(SOURCE / 'actions.npz', allow_pickle=False) as a:
        actions = {k: a[k] for k in ('query_ids', 'group_ids', 'M6', 'M6_logits', 'M6_switch', 'F_switch')}
    with np.load(SOURCE / 'measurement_scores.npz', allow_pickle=False) as a:
        for key in ('query_ids', 'group_ids'):
            assert np.array_equal(a[key].astype(str), roles[key])
            assert np.array_equal(actions[key].astype(str), roles[key])
        average = a['outcomes'].mean(axis=2)
    qids, gids, outer, inner = (roles[k] for k in ('query_ids', 'group_ids', 'outer_fold', 'inner_fold'))
    x = actions['M6'].astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    logits = actions['M6_logits'].astype(np.float64)
    gap = average[:, 0, 0] - average[:, 1, 0]
    n, began = len(qids), time.perf_counter()
    scores = np.full((n, 4), np.nan)  # C, K, L, S
    nearest = np.full((n, SPEC['k']), -1, dtype=np.int64)
    inner_bases = np.full((4, n), np.nan)
    calibration = np.empty((4, 4))  # full outer-train, three inner train calibrators
    local_offsets = np.empty((n, 2))
    root_errors, fold_records = [], []
    for fold in range(4):
        train = np.array(sorted(np.flatnonzero(outer != fold), key=lambda i: qids[i]))
        test = np.flatnonzero(outer == fold)
        assert not set(gids[train]) & set(gids[test])
        a_outer = float(fit_offsets(logits[train], gap[train], ridge=SPEC['ridge']))
        calibration[fold, 0] = a_outer
        root_errors.append(abs(float(offset_gradient(logits[train], gap[train], a_outer, ridge=SPEC['ridge']))))
        for j in range(3):
            fit, held = train[inner[fold, train] != j], train[inner[fold, train] == j]
            assert not set(gids[fit]) & set(gids[held])
            a_inner = float(fit_offsets(logits[fit], gap[fit], ridge=SPEC['ridge']))
            calibration[fold, j+1] = a_inner
            inner_bases[fold, held] = logits[held] + a_inner
            root_errors.append(abs(float(offset_gradient(logits[fit], gap[fit], a_inner, ridge=SPEC['ridge']))))
        assert np.isfinite(inner_bases[fold, train]).all()
        similarity = x[test] @ x[train].T
        neighbor_rows = train[np.argsort(-similarity, axis=1, kind='stable')[:, :SPEC['k']]]
        nearest[test] = neighbor_rows
        permuted_rows = roles['shuffled_source_row'][fold, neighbor_rows]
        assert np.all(outer[neighbor_rows] != fold) and np.all(outer[permuted_rows] != fold)
        t_local = fit_offsets(inner_bases[fold, neighbor_rows], gap[neighbor_rows], ridge=SPEC['ridge'])
        t_shuffled = fit_offsets(inner_bases[fold, permuted_rows], gap[permuted_rows], ridge=SPEC['ridge'])
        for rows, offsets in ((neighbor_rows, t_local), (permuted_rows, t_shuffled)):
            root_errors.append(float(np.abs(offset_gradient(inner_bases[fold, rows], gap[rows], offsets, ridge=SPEC['ridge'])).max()))
        scores[test] = np.c_[logits[test] + a_outer, gap[neighbor_rows].mean(axis=1),
                             logits[test] + a_outer + t_local, logits[test] + a_outer + t_shuffled]
        local_offsets[test] = np.c_[t_local, t_shuffled]
        fold_records.append({'fold': fold, 'global_offset': a_outer,
                             'train_nonzero_gaps': int(np.count_nonzero(gap[train])),
                             'fit_queries': len(train), 'held_queries': len(test)})
    fitting_seconds = time.perf_counter() - began
    assert np.isfinite(scores).all() and max(root_errors) < 1e-9
    switches = np.c_[np.ones(n, dtype=bool), np.zeros(n, dtype=bool),
                     actions['M6_switch'], actions['F_switch'], scores > 0]
    utility = np.where(switches[:, :, None], average[:, None, 0], average[:, None, 1])
    point = utility.mean(axis=0)
    target = POLICIES.index('L_local')
    comparisons = {}
    for name in REFERENCES:
        difference = float(point[target, 0] - point[POLICIES.index(name), 0])
        minimum = .01 if name in ('Dense', 'BM25') else .002
        comparisons[name] = {'f1_gain': difference, 'minimum_point_gain': minimum,
                             'point_gate': difference >= minimum, 'CI99': None}
    point_pass = all(row['point_gate'] for row in comparisons.values())
    bootstrap = None
    if point_pass:
        _, group_index = np.unique(gids, return_inverse=True)
        sizes = np.bincount(group_index)
        totals = np.zeros((len(sizes), len(REFERENCES)))
        differences = np.column_stack([utility[:, target, 0] - utility[:, POLICIES.index(name), 0] for name in REFERENCES])
        np.add.at(totals, group_index, differences)
        bootstrap = np.empty((SPEC['bootstrap_draws'], len(REFERENCES)))
        rng = np.random.default_rng(SPEC['bootstrap_seed'])
        for start in range(0, len(bootstrap), 200):
            stop = min(start + 200, len(bootstrap))
            selected = rng.integers(0, len(sizes), size=(stop-start, len(sizes)))
            bootstrap[start:stop] = totals[selected].sum(axis=1) / sizes[selected].sum(axis=1)[:, None]
        for j, name in enumerate(REFERENCES):
            comparisons[name]['CI99'] = np.quantile(bootstrap[:, j], [.005, .995]).tolist()
    passed = point_pass and all(row['CI99'][0] > 0 for row in comparisons.values())
    means = {}
    for j, name in enumerate(POLICIES):
        switch = switches[:, j]
        means[name] = {'f1': float(point[j, 0]), 'em': float(point[j, 1]), 'bm25_count': int(switch.sum()),
                       'f1_gain_over_Dense': float(point[j, 0] - point[1, 0]),
                       'beneficial_switches': int(np.sum(switch & (gap > 0))),
                       'harmful_switches': int(np.sum(switch & (gap < 0))),
                       'tie_switches': int(np.sum(switch & (gap == 0))),
                       'benefit_mass': float(np.mean(switch*np.maximum(gap, 0))),
                       'harm_mass': float(np.mean(switch*np.maximum(-gap, 0)))}
    for row in fold_records:
        held = outer == row['fold']
        row['L_minus_Dense'] = float((utility[held, target, 0]-utility[held, 1, 0]).mean())
        row['L_minus_C'] = float((utility[held, target, 0]-utility[held, POLICIES.index('C_global'), 0]).mean())
    result = {'status': 'complete_consumed_external_group_OOF_development',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol': str(OUT/'protocol.json'),
              'queries': n, 'groups': len(set(gids)), 'policy_means': means, 'L_comparisons': comparisons,
              'all_point_gates': point_pass, 'bootstrap_draws_executed': SPEC['bootstrap_draws'] if point_pass else 0,
              'development_gate': passed,
              'decision': 'development_candidate_prepare_independent_protocol' if passed else 'close_local_offset_recipe_no_parameter_or_split_extension',
              'folds': fold_records, 'global_calibrator_fits': 16, 'local_scalar_solves': 2*n,
              'fitting_and_neighbor_seconds': fitting_seconds, 'maximum_absolute_gradient_at_root': max(root_errors),
              'retained_groups_accessed': 0, 'new_encoder_calls': 0, 'new_answer_calls': 0,
              'new_independent_validation': False, 'objective_achieved': False,
              'limitations': [SPEC['scope'], SPEC['CI_role'], 'one_fixed_neighborhood_and_regularization_recipe_not_all_local_methods',
                              'same_size_unique_group_strata_remain_unshuffled_as_disclosed_in_protocol']}
    saved = {'query_ids': qids, 'group_ids': gids, 'outer_fold': outer,
             'policy_names': np.asarray(POLICIES), 'scores_C_K_L_S': scores,
             'switches': switches, 'query_policy_utilities': utility, 'neighbors': nearest,
             'inner_OOF_bases': inner_bases, 'calibration_offsets': calibration, 'local_offsets_L_S': local_offsets}
    if bootstrap is not None:
        saved['conditional_bootstrap_differences'] = bootstrap
    with (OUT / 'predictions.npz').open('xb') as stream:
        np.savez_compressed(stream, **saved)
    write_new(OUT / 'results.json', result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
