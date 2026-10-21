"""Offline descriptive snapshot; no selection or non-inferiority claim.

Compares common-query final F1 and matched successful trajectory efficiency.
Calls/tokens include exact cache reuse; actual payments remain ledger based.
"""
from collections import Counter
from datetime import datetime, timezone
import json

from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest
from .score import ROOT, HERE, OUT, OLD, DATASETS, read, rows, metric, select_case, is_scorable
from .offline_audit import source_catalog, stored_request
from .report import budget_snapshot

ARMS = ('union20', 'discount20', 'single_final')
BASELINE = 'p02_role_headers'


def _p02_efficiency(case, record, manifests, base_sources, plan):
    catalog = source_catalog(manifests[case['id']], base_sources[case['id']], record)
    requests = []
    for node in record['nodes']:
        if node.get('rewrite_prompt') is not None:
            requests.append((node['rewrite_prompt'], 256))
        requests.append((node['child_prompt'], 128))
        if node.get('sentence_format_request_id'):
            from .pipeline import b0
            fmt = ('Express the candidate answer below as a concise complete factual sentence that names its subject and relation. '
                   'Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. '
                   'If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n'
                   + 'Question:\n' + node['request'] + '\n\nCandidate answer:\n' + node['initial_answer']
                   + '\n\nContext:\n' + b0.passages(node['reader_packet']))
            requests.append((fmt, 128))
    requests.append((record['final_prompt'], 64))
    usage = Counter()
    for prompt, maximum in requests:
        request = payload(plan['model'], prompt, maximum)
        key = digest(request)
        assert key in catalog, 'Missing exact p02 receipt for efficiency comparison'
        source = catalog[key]
        stored = stored_request(source['source'], source['request_id'])
        assert stored['payload'] == request and stored['status'] == 'success'
        usage.update({key: (stored['result'].get('usage') or {}).get(key, 0) or 0 for key in ['prompt_tokens', 'completion_tokens']})
    return {'logical_calls': len(requests), **dict(usage)}


def _new_efficiency(value):
    calls = value['logical_calls']
    assert all(call['status'] == 'success' for call in calls)
    return {'logical_calls': len(calls), **{key: sum((call.get('usage') or {}).get(key, 0) or 0 for call in calls)
                                            for key in ['prompt_tokens', 'completion_tokens']}}


def _score_stats(items):
    if not items:
        return {'n': 0, 'F1': None, 'p02_F1': None, 'delta': None, 'up': 0, 'tie': 0, 'down': 0}
    delta = [item['f1'] - item['p02_f1'] for item in items]
    return {'n': len(items), 'F1': sum(item['f1'] for item in items) / len(items),
            'p02_F1': sum(item['p02_f1'] for item in items) / len(items),
            'delta': sum(delta) / len(items), 'up': sum(d > 1e-12 for d in delta),
            'tie': sum(abs(d) <= 1e-12 for d in delta), 'down': sum(d < -1e-12 for d in delta)}


def _efficiency(items):
    n = len(items)
    result = {'matched_complete_success_queries': n, 'dataset_counts': dict(Counter(item['dataset'] for item in items))}
    for key in ['logical_calls', 'prompt_tokens', 'completion_tokens']:
        old = sum(item['p02_efficiency'][key] for item in items)
        new = sum(item['single_final_efficiency'][key] for item in items)
        result[key] = {'p02_total': old, 'single_final_total': new, 'saved': old - new,
                       'p02_mean': old / n if n else None, 'single_final_mean': new / n if n else None,
                       'relative_change': (new - old) / old if old else None}
    return result


def snapshot():
    started = datetime.now(timezone.utc).isoformat()
    samples = rows(HERE / 'runtime300.frozen.jsonl')
    refs = {item['id']: item for item in rows(HERE / 'references300.local.frozen.jsonl')}
    plan = read(HERE / 'PROTOCOL.frozen.json')
    native = {ds: metric(ds) for ds in DATASETS}
    old_e = HERE.parent / 'b0_strategy_round2_v1_20261020'
    old_s = HERE.parent / 'b0_six_single_changes_v1_20261020'
    manifests = {item['id']: item for item in rows(old_e / 'SOURCES.frozen.jsonl')}
    bases = {item['id']: item for item in rows(old_s / 'B0_SOURCES.frozen.jsonl')}
    values, old_values, records = {}, {}, {}
    for case in samples:
        _, old = select_case(OLD, case['dataset'], BASELINE, case['qid'])
        assert old and old['status'].startswith('success')
        old_values[case['id']] = old
        for arm in ARMS:
            path, value = select_case(OUT, case['dataset'], arm, case['qid'])
            values[(case['id'], arm)] = value
            if path:
                records[(case['id'], arm)] = str(path.relative_to(ROOT))
    scored = {arm: [] for arm in ARMS}
    common = [case for case in samples if all(is_scorable(values[(case['id'], arm)]) for arm in ARMS)]
    common_ids = {case['id'] for case in common}
    for case in samples:
        old_f1 = native[case['dataset']](old_values[case['id']]['prediction'], refs[case['id']])['f1']
        for arm in ARMS:
            value = values[(case['id'], arm)]
            if not is_scorable(value):
                continue
            f1 = 0.0 if value['status'] == 'algorithm_failure' else native[case['dataset']](value['prediction'], refs[case['id']])['f1']
            scored[arm].append({'id': case['id'], 'dataset': case['dataset'], 'arm': arm,
                                'f1': f1, 'p02_f1': old_f1, 'source': records[(case['id'], arm)]})
    arm_summary = {}
    for arm in ARMS:
        items = scored[arm]
        status = Counter(values[(case['id'], arm)]['status'] if values[(case['id'], arm)] else 'unstarted' for case in samples)
        arm_summary[arm] = {**_score_stats(items), 'planned': 300, 'missing': 300 - len(items), 'statuses': dict(status),
                            'F1_complete': sum(item['f1'] for item in items) / 300 if len(items) == 300 else None,
                            'by_dataset': {ds: {**_score_stats([item for item in items if item['dataset'] == ds]),
                                                'planned': 100,
                                                'statuses': dict(Counter(values[(case['id'], arm)]['status'] if values[(case['id'], arm)] else 'unstarted' for case in samples if case['dataset'] == ds))}
                                           for ds in DATASETS}}
    common_summary = {'n': len(common), 'dataset_counts': dict(Counter(case['dataset'] for case in common)),
                      'arms': {arm: {**_score_stats([item for item in scored[arm] if item['id'] in common_ids]),
                                     'by_dataset': {ds: _score_stats([item for item in scored[arm] if item['id'] in common_ids and item['dataset'] == ds]) for ds in DATASETS}}
                               for arm in ARMS}, 'ids': sorted(common_ids)}
    efficiencies = []
    for case in samples:
        value = values[(case['id'], 'single_final')]
        if value and value['status'].startswith('success'):
            efficiencies.append({'id': case['id'], 'dataset': case['dataset'],
                                 'p02_efficiency': _p02_efficiency(case, old_values[case['id']], manifests, bases, plan),
                                 'single_final_efficiency': _new_efficiency(value)})
    result = {'snapshot_started_utc': started, 'snapshot_finished_utc': datetime.now(timezone.utc).isoformat(),
              'planned': 300, 'planned_datasets': {ds: 100 for ds in DATASETS}, 'arms': arm_summary,
              'all_arm_common_same_query': common_summary,
              'matched_success_efficiency_single_final_vs_p02': {**_efficiency(efficiencies),
                  'by_dataset': {ds: _efficiency([item for item in efficiencies if item['dataset'] == ds]) for ds in DATASETS}},
              'efficiency_details': efficiencies, 'budget': budget_snapshot(),
              'quality_gate': 'No selection or promotion from partial results. Absence of statistical significance is not proof of noninferiority. Simplification need not improve F1, but tolerable regression must be judged on complete comparable evaluation.',
              'partial_scope': 'Each arm first uses its own scorable IDs paired with p02, then all arms share the exact common set. Neither replaces a full300 result.',
              'cost_scope': 'Logical calls/tokens include cache reuse; input tokens can increase with20 documents. Actual incremental charges and unknown reserves remain in budget snapshot.',
              'new_API_calls': 0, 'new_index_searches': 0}
    path = OUT / 'CURRENT300_SIMPLIFICATION_SNAPSHOT.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    lines = [f'简化流程当前300题快照：{result["snapshot_finished_utc"]}',
             '以下均为部分可计分同题描述，尚不据此选择或升级版本；算法失败计0、基础设施缺失不填0。',
             'arm\t计分/300\tF1%\t同题p02%\t差值pp\t升/平/降']
    for arm, value in arm_summary.items():
        if value['n']:
            lines.append(f'{arm}\t{value["n"]}/300\t{100*value["F1"]:.4f}\t{100*value["p02_F1"]:.4f}\t{100*value["delta"]:+.4f}\t{value["up"]}/{value["tie"]}/{value["down"]}')
        lines.append(json.dumps({'arm': arm, 'by_dataset': value['by_dataset']}, ensure_ascii=False))
    lines += ['', '三组共同可计分交集（p02同题）：', json.dumps(common_summary, ensure_ascii=False, indent=2), '',
              'single_final与p02完整成功同题的逻辑调用/token比较：',
              json.dumps(result['matched_success_efficiency_single_final_vs_p02'], ensure_ascii=False, indent=2), '',
              '费用：', json.dumps(result['budget'], ensure_ascii=False, indent=2), '',
              '没有显著差异不能宣称非劣；完成同题评估前不升级版本。']
    (OUT / 'CURRENT300_SIMPLIFICATION_SNAPSHOT.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({key: result[key] for key in ['snapshot_finished_utc', 'arms', 'all_arm_common_same_query', 'matched_success_efficiency_single_final_vs_p02']}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    snapshot()
