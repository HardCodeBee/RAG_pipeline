"""Quality/call/cost evidence for considering simplification, without promotion.

Offline only. A positive F1 gain is not an adoption requirement, and no numeric
noninferiority margin is invented. Actual charges and equal-rate logical costs
are intentionally separate, especially because exact prefix caches are reused.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np

from .score import ROOT, HERE, OUT, OLD, DATASETS, rows, read, sha, metric, select_case, is_scorable, _stats
from .simplification_snapshot import _p02_efficiency
from .report import budget_snapshot

ARMS = ('union20', 'discount20', 'single_final')
INPUT_RATE_CNY = .8e-6
OUTPUT_RATE_CNY = 2e-6


def utc():
    return datetime.now(timezone.utc).isoformat()


def model_cost(prompt, completion):
    return prompt * INPUT_RATE_CNY + completion * OUTPUT_RATE_CNY


def usage(value):
    calls = value.get('logical_calls', [])
    returned = [call for call in calls if call['status'] == 'success']
    return {'logical_attempts': len(calls), 'logical_calls': len(returned),
            'prompt_tokens': sum((call.get('usage') or {}).get('prompt_tokens', 0) or 0 for call in returned),
            'completion_tokens': sum((call.get('usage') or {}).get('completion_tokens', 0) or 0 for call in returned),
            'unreturned_or_failed_logical_calls': len(calls) - len(returned),
            'returned_call_usage_missing': sum((call.get('usage') or {}).get('prompt_tokens') is None
                                              or (call.get('usage') or {}).get('completion_tokens') is None for call in returned)}


def _usage_total(values):
    values = list(values)
    total = {key: sum(value.get(key, 0) for value in values) for key in [
        'logical_attempts', 'logical_calls', 'prompt_tokens', 'completion_tokens',
        'unreturned_or_failed_logical_calls', 'returned_call_usage_missing']}
    total['selected_case_records'] = len(values)
    total['equal_rate_modeled_generation_cny'] = model_cost(total['prompt_tokens'], total['completion_tokens'])
    total['partial_prefixes_may_be_included'] = True
    return total


def _matched_efficiency(items):
    n = len(items)
    result = {'matched_complete_success_queries': n, 'quality_oracle_selection': False}
    for key in ['logical_calls', 'prompt_tokens', 'completion_tokens']:
        old = sum(item['baseline_usage'][key] for item in items)
        new = sum(item['usage'][key] for item in items)
        result[key] = {'p02_total': old, 'candidate_total': new, 'saved': old - new,
                       'p02_mean': old / n if n else None, 'candidate_mean': new / n if n else None,
                       'relative_change': (new - old) / old if old else None}
    old_cost = model_cost(result['prompt_tokens']['p02_total'], result['completion_tokens']['p02_total'])
    new_cost = model_cost(result['prompt_tokens']['candidate_total'], result['completion_tokens']['candidate_total'])
    result['equal_rate_modeled_generation_cny'] = {
        'p02': old_cost, 'candidate': new_cost, 'saved': old_cost - new_cost,
        'relative_change': (new_cost - old_cost) / old_cost if old_cost else None,
        'actual_paid_cost': False, 'cache_reuse_does_not_reduce_this_logical_cost': True}
    return result


def _paired(items, other_arm, *, replicates=10000):
    available = [item for item in items if item['f1'] is not None and item['other_f1'] is not None]
    delta = np.array([item['f1'] - item['other_f1'] for item in available], dtype=float)
    n = len(available)
    seed = 20261020
    rng = np.random.default_rng(seed)
    resamples = np.zeros(replicates, dtype=float)
    if n:
        for ds in DATASETS:
            values = np.array([item['f1'] - item['other_f1'] for item in available if item['dataset'] == ds])
            if len(values):
                resamples += values[rng.integers(len(values), size=(replicates, len(values)))].sum(axis=1) / n
    return {'comparison': 'single_final minus ' + other_arm,
            'n': n, 'dataset_counts': dict(Counter(item['dataset'] for item in available)),
            'single_final_F1': sum(item['f1'] for item in available) / n if n else None,
            'comparison_F1': sum(item['other_f1'] for item in available) / n if n else None,
            'delta': float(delta.mean()) if n else None,
            'paired_bootstrap95': np.quantile(resamples, [.025, .975]).tolist() if n else None,
            'up': int(sum(delta > 1e-12)), 'tie': int(sum(abs(delta) <= 1e-12)), 'down': int(sum(delta < -1e-12)),
            'bootstrap': 'Paired query bootstrap stratified by dataset, preserving observed dataset counts/weights',
            'replicates': replicates, 'seed': seed,
            'scope': 'Conditional development-cohort uncertainty; not API-generation randomness or proof of noninferiority',
            'numeric_noninferiority_margin': None,
            'all300_comparison_complete': n == 300}


def _actual_charges():
    path = OUT.parent / 'budget.sqlite3'
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=20) as db:
        rows_ = list(db.execute('SELECT id,namespace,reserved,actual,status FROM charges WHERE namespace LIKE ?', (OUT.name + '%',)))
    charges = []
    excluded_fresh = 0
    for rid, namespace, reserved, actual, status in rows_:
        parts = [part for part in namespace.replace('\\', '/').split('/') if part]
        # Exact original300 method namespace, never the later fresh60 subtree.
        if len(parts) == 6 and parts[0] == OUT.name and parts[1] == 'rounds' and parts[3] == 'methods':
            charges.append({'id': rid, 'dataset': parts[4], 'arm': parts[5],
                            'reserved': reserved, 'actual': actual, 'status': status})
        else:
            excluded_fresh += 1

    def summarize(values):
        known = sum(item['actual'] or 0 for item in values)
        reserve = sum(item['reserved'] for item in values if item['actual'] is None)
        return {'new_physical_requests': len(values), 'returned_usage_estimate_cny': known,
                'unreconciled_reserved_cny': reserve, 'occupied_cny': known + reserve,
                'unreconciled_requests': sum(item['actual'] is None for item in values),
                'statuses': dict(Counter(item['status'] for item in values))}
    return {'database': str(path), 'original300_only': summarize(charges),
            'by_arm': {arm: {**summarize([item for item in charges if item['arm'] == arm]),
                             'by_dataset': {ds: summarize([item for item in charges if item['arm'] == arm and item['dataset'] == ds])
                                            for ds in DATASETS}} for arm in ARMS},
            'excluded_other_subtree_charge_rows': excluded_fresh,
            'actual_invoice_verified': False,
            'scope': 'All physical requests in original300 method namespaces, including retry rounds and superseded prefixes; historical seeded receipts and fresh60 requests are not charged here. Unreconciled reservations may include active requests.'}


def _pct(value):
    return '—' if value is None else f'{100 * value:.4f}'


def generate(*, output_dir=None):
    started = utc()
    samples = rows(HERE / 'runtime300.frozen.jsonl')
    references = {item['id']: item for item in rows(HERE / 'references300.local.frozen.jsonl')}
    plan = read(HERE / 'PROTOCOL.frozen.json')
    scorers = {ds: metric(ds) for ds in DATASETS}
    old_e = HERE.parent / 'b0_strategy_round2_v1_20261020'
    old_s = HERE.parent / 'b0_six_single_changes_v1_20261020'
    manifests = {item['id']: item for item in rows(old_e / 'SOURCES.frozen.jsonl')}
    base_sources = {item['id']: item for item in rows(old_s / 'B0_SOURCES.frozen.jsonl')}
    baseline, captured = {}, {}
    # Capture case membership once before reconstructing the more expensive old
    # token receipts; report metrics share exactly this selected-result snapshot.
    for case in samples:
        _, old = select_case(OLD, case['dataset'], 'p02_role_headers', case['qid'])
        assert old and old['status'].startswith('success')
        baseline[case['id']] = old
        for arm in ARMS:
            path, value = select_case(OUT, case['dataset'], arm, case['qid'])
            captured[(case['id'], arm)] = (path, value)
    details, lookup = [], {}
    for case in samples:
        cid, ds = case['id'], case['dataset']
        old_metric = scorers[ds](baseline[cid]['prediction'], references[cid])
        old_usage = _p02_efficiency(case, baseline[cid], manifests, base_sources, plan)
        assert old_usage['logical_calls'] > 0 and old_usage['prompt_tokens'] > 0
        for arm in ARMS:
            path, value = captured[(cid, arm)]
            item = {'id': cid, 'dataset': ds, 'arm': arm,
                    'status': value['status'] if value else 'unstarted',
                    'source': str(path.relative_to(ROOT)) if path else None,
                    'f1': None, 'em': None, 'baseline_f1': old_metric['f1'], 'baseline_em': old_metric['em'],
                    'direct_parent_f1': None, 'baseline_usage': old_usage,
                    'usage': usage(value) if value else None,
                    'complete_success': bool(value and value['status'].startswith('success'))}
            if is_scorable(value):
                item.update({'f1': 0.0, 'em': 0.0} if value['status'] == 'algorithm_failure'
                            else scorers[ds](value['prediction'], references[cid]))
            details.append(item)
            lookup[(cid, arm)] = item
    for item in details:
        parent = {'union20': 'p02_role_headers', 'discount20': 'union20', 'single_final': 'discount20'}[item['arm']]
        item['direct_parent_arm'] = parent
        item['direct_parent_f1'] = item['baseline_f1'] if parent == 'p02_role_headers' else lookup[(item['id'], parent)]['f1']

    def arm_stats(items):
        return {**_stats(items),
                'selected_case_recorded_usage': _usage_total(item['usage'] for item in items if item['usage'] is not None),
                'matched_success_efficiency_vs_p02': _matched_efficiency([item for item in items if item['complete_success']]),
                'frozen_p02_full_planned_usage': _usage_total(item['baseline_usage'] for item in items)}
    summaries = {arm: {**arm_stats([item for item in details if item['arm'] == arm]),
                       'by_dataset': {ds: arm_stats([item for item in details if item['arm'] == arm and item['dataset'] == ds]) for ds in DATASETS}}
                 for arm in ARMS}
    common_ids = {case['id'] for case in samples if all(lookup[(case['id'], arm)]['f1'] is not None for arm in ARMS)}
    common = {'n': len(common_ids), 'dataset_counts': dict(Counter(case['dataset'] for case in samples if case['id'] in common_ids)),
              'arms': {arm: {**_stats([item for item in details if item['arm'] == arm and item['id'] in common_ids]),
                             'by_dataset': {ds: _stats([item for item in details if item['arm'] == arm and item['id'] in common_ids and item['dataset'] == ds]) for ds in DATASETS}}
                       for arm in ARMS}}
    comparisons = {}
    for other_arm in ('p02_role_headers', 'discount20'):
        items = [{**item, 'other_f1': item['baseline_f1'] if other_arm == 'p02_role_headers' else lookup[(item['id'], other_arm)]['f1']}
                 for item in details if item['arm'] == 'single_final']
        comparisons[other_arm] = {**_paired(items, other_arm),
                                  'by_dataset': {ds: _paired([item for item in items if item['dataset'] == ds], other_arm) for ds in DATASETS}}
    direct_efficiency_items = [{**item, 'baseline_usage': lookup[(item['id'], 'discount20')]['usage']}
                              for item in details if item['arm'] == 'single_final' and item['complete_success']
                              and lookup[(item['id'], 'discount20')]['complete_success']]

    def direct_efficiency(items):
        value = _matched_efficiency(items)
        value['comparison_baseline'] = 'discount20'
        for field in ['logical_calls', 'prompt_tokens', 'completion_tokens']:
            for key in ['total', 'mean']:
                value[field]['discount20_' + key] = value[field].pop('p02_' + key)
        value['equal_rate_modeled_generation_cny']['discount20'] = value['equal_rate_modeled_generation_cny'].pop('p02')
        return value
    direct_efficiency_summary = {**direct_efficiency(direct_efficiency_items),
                                'by_dataset': {ds: direct_efficiency([item for item in direct_efficiency_items if item['dataset'] == ds]) for ds in DATASETS}}
    actual = _actual_charges()
    output = Path(output_dir or OUT / 'reports/adoption300')
    output.mkdir(parents=True, exist_ok=True)
    adoption_path = HERE / 'ADOPTION_POLICY.user_update_20261020.json'
    result = {'snapshot_started_utc': started, 'snapshot_finished_utc': utc(),
              'planned': 300, 'dataset_counts': {ds: 100 for ds in DATASETS},
              'native_metric': 'Answer F1/EM: HotpotQA; 2Wiki v1.1 aliases/demonyms; MuSiQue aliases',
              'arms': summaries, 'all_arm_common_same_query': common,
              'single_final_paired_bootstrap': comparisons,
              'matched_success_efficiency_single_final_vs_discount20': direct_efficiency_summary,
              'actual_original300_new_costs': actual, 'project_and_global_budget': budget_snapshot(),
              'frozen_unit_rates_cny_per_token': {'prompt': INPUT_RATE_CNY, 'completion': OUTPUT_RATE_CNY,
                    'source': 'experiments/query_decomposition/decomposition_error_expand60_v1_20261019/run.py returned_usage settlement; unchanged evaluation rates'},
              'adoption_policy': read(adoption_path) if adoption_path.exists() else None,
              'numeric_noninferiority_margin': None, 'auto_promoted': False,
              'limits': ['Positive F1 gain is not required for a useful simplification.',
                         'No statistical nonsignificance is interpreted as equivalence or noninferiority.',
                         'Incomplete infrastructure cases are missing, not zero; algorithm failures count zero.',
                         'Bootstrap describes observed development queries and does not establish unseen generalization.',
                         'Matched full-success efficiency excludes incomplete or algorithm-aborted trajectories; their observed prefix costs remain in selected usage and the actual ledger.',
                         'Logical modeled cost counts cached generations at frozen identical rates; it is not the actual new charge.',
                         'Files were captured once during a running evaluation; later request/cost snapshots can advance during report construction.'],
              'new_API_calls': 0, 'new_index_searches': 0,
              'source_pins': {str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__), HERE / 'runtime300.frozen.jsonl', HERE / 'references300.local.frozen.jsonl']}}
    (output / 'REPORT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (output / 'PAIRED_DETAILS.jsonl').write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in details), encoding='utf8')
    lines = ['取消goal预回答的简化评估：300题当前证据', f'读取起点：{started}', f'报告完成：{result["snapshot_finished_utc"]}',
             '不要求F1提升；不设用户未给定的非劣数值阈值；不据部分结果自动采纳或升级。',
             '下表F1为可计分题均值，未完成保留分母和边界，算法失败计0。', '',
             '配置\t计分/计划\tF1%\t同题p02Δpp\t直接父配置Δpp\t直接父比较n\t同题成功平均逻辑调用\t同题p02平均逻辑调用\t本组新已知费用/预留元']
    for arm, summary in summaries.items():
        eff = summary['matched_success_efficiency_vs_p02']
        cost = actual['by_arm'][arm]
        calls = eff['logical_calls']
        lines.append('\t'.join([arm, f'{summary["scorable"]}/{summary["planned"]}', _pct(summary['F1_available']),
                                 _pct(summary['paired_with_frozen_p02']['delta']), _pct(summary['paired_with_direct_parent']['delta']),
                                 str(summary['paired_with_direct_parent']['n']),
                                 str(calls['candidate_mean']), str(calls['p02_mean']),
                                 f'{cost["returned_usage_estimate_cny"]:.6f}/{cost["unreconciled_reserved_cny"]:.6f}']))
        for ds in DATASETS:
            v = summary['by_dataset'][ds]
            e = v['matched_success_efficiency_vs_p02']
            c = cost['by_dataset'][ds]
            lines.append(f'  {ds}: {v["scorable"]}/{v["planned"]}; F1={_pct(v["F1_available"])}%; p02Δ={_pct(v["paired_with_frozen_p02"]["delta"])}pp; 父Δ={_pct(v["paired_with_direct_parent"]["delta"])}pp; matched-success n={e["matched_complete_success_queries"]}; calls={e["logical_calls"]["candidate_mean"]}/{e["logical_calls"]["p02_mean"]}; modeled元={e["equal_rate_modeled_generation_cny"]["candidate"]:.6f}/{e["equal_rate_modeled_generation_cny"]["p02"]:.6f}; 实际新已知/预留={c["returned_usage_estimate_cny"]:.6f}/{c["unreconciled_reserved_cny"]:.6f}')
        lines.append('  状态：' + json.dumps(summary['statuses'], ensure_ascii=False))
        lines.append('  全计划F1边界：' + json.dumps(summary['F1_all_planned_bounds']))
    lines += ['', '三组共同同题：', json.dumps(common, ensure_ascii=False, indent=2), '',
              'single_final配对差值与bootstrap95%：', json.dumps(comparisons, ensure_ascii=False, indent=2), '',
              '同题完整成功逻辑调用/token/同价模型成本：',
              json.dumps({arm: v['matched_success_efficiency_vs_p02'] for arm, v in summaries.items()}, ensure_ascii=False, indent=2), '',
              'single_final vs discount20同题完整成功效率（隔离取消goal预答）：',
              json.dumps(direct_efficiency_summary, ensure_ascii=False, indent=2), '',
              '实际原300新增请求费用（包括重试/未知，排除fresh60及历史种子费用）：', json.dumps(actual, ensure_ascii=False, indent=2), '',
              '限制：', *result['limits']]
    (output / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    compact = {'utc': result['snapshot_finished_utc'],
               'arms': {arm: {key: value[key] for key in ['planned', 'scorable', 'missing', 'F1_available', 'F1_complete', 'statuses']} for arm, value in summaries.items()},
               'single_final_comparisons': {key: {k: value[k] for k in ['n', 'delta', 'paired_bootstrap95', 'up', 'tie', 'down']} for key, value in comparisons.items()},
               'actual_new_cost': actual['original300_only'], 'output': str(output)}
    print(json.dumps(compact, ensure_ascii=False))
    return result


if __name__ == '__main__':
    generate()
