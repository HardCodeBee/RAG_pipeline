"""Offline native scoring and receipt/cost audit for a single terminal goal.

References are opened only here, never by the executor. Exact source receipts
prove ordinary QA/rewrite reuse; removed legacy goal/formatter/final calls are
not counted as newly executed logical generations. No API or index is opened.
"""
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
OUT = OUTPUT / HERE.name
ARM = 'terminal_goal'
DATASETS = ('hotpotqa', '2wiki', 'musique')
INPUT_RATE, OUTPUT_RATE = .8e-6, 2e-6


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def receipt(source, request_id):
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=20) as database:
        row = database.execute('SELECT payload,status,result FROM requests WHERE id=?', (request_id,)).fetchone()
    assert row, (source, request_id)
    return {'payload': json.loads(row[0]), 'status': row[1], 'result': json.loads(row[2]) if row[2] else None}


def selected(dataset, qid):
    values = [(path, read(path)) for path in sorted(OUT.glob(f'rounds/r*/cases/{dataset}/{ARM}/{qid}.json'))]
    terminal = [(path, value) for path, value in values if scorable(value)]
    if terminal:
        assert len({(value['status'].startswith('success'), value.get('prediction')) for _, value in terminal}) == 1
        return terminal[0]
    return values[-1] if values else (None, None)


def scorable(value):
    return bool(value and (value.get('status') == 'algorithm_failure'
                          or value.get('status', '').startswith('success') and isinstance(value.get('prediction'), str)))


def usage(calls, *, check=True):
    totals = Counter(logical_attempts=len(calls), logical_generations=0,
                     prompt_tokens=0, completion_tokens=0, unreturned_calls=0, usage_missing=0)
    for call in calls:
        if check:
            stored = receipt(call['source'], call['request_id'])
            assert digest(stored['payload']) == call['payload_sha256']
            assert stored['status'] == call['status']
        else:
            stored = None
        if call['status'] != 'success':
            totals['unreturned_calls'] += 1
            continue
        raw = stored['result'] if stored else None
        if raw is not None:
            assert raw.get('status') == 'success'
            u = raw.get('usage') or {}
            recorded = call.get('usage') or call.get('source_usage')
            if recorded:
                for key in ('prompt_tokens', 'completion_tokens'):
                    if key in recorded:
                        assert recorded[key] == u.get(key)
        else:
            u = call.get('usage') or call.get('source_usage') or {}
        totals['logical_generations'] += 1
        totals['usage_missing'] += any(u.get(key) is None for key in ('prompt_tokens', 'completion_tokens'))
        totals['prompt_tokens'] += u.get('prompt_tokens', 0) or 0
        totals['completion_tokens'] += u.get('completion_tokens', 0) or 0
    result = dict(totals)
    result['equal_rate_generation_cny'] = result['prompt_tokens'] * INPUT_RATE + result['completion_tokens'] * OUTPUT_RATE
    return result


def lineage(value, old):
    goal = old['nodes'][-1]
    assert goal['role'] == 'goal'
    goal_sid = goal['step_id']
    removed_ids = {rid for rid in [goal['answer_request_id'], goal.get('sentence_format_request_id'), old['request_id']] if rid}
    removed_stages = {f'legacy/answer/{goal_sid}', f'legacy/sentence_format/{goal_sid}', 'legacy/final'}
    kept = [call for call in old['logical_calls'] if call['request_id'] not in removed_ids and call['stage'] not in removed_stages]
    actual = value.get('logical_calls', [])
    terminal = [call for call in actual if call['stage'] == 'terminal_goal/answer']
    prior = [call for call in actual if call['stage'] != 'terminal_goal/answer']
    assert len(terminal) <= 1
    assert len(prior) <= len(kept)
    for got, expected in zip(prior, kept):
        for key in ('stage', 'payload_sha256', 'request_id', 'source'):
            assert got[key] == expected[key], ('Changed legacy prefix', key)
        assert got.get('reused') is True or got.get('cache_hit') is True, 'Legacy prefix not labelled as exact reuse'
    assert all(call['request_id'] not in removed_ids for call in actual)
    assert not any(call['stage'] in removed_stages for call in actual)
    assert value['goal_reader_packet'] == goal['reader_packet']
    assert all(doc['block'] in value['goal_prompt'] for doc in goal['reader_packet']['docs'])
    prefix_nodes = value['nodes'][:-1] if value['status'] in ('success', 'algorithm_failure') else value['nodes']
    assert prefix_nodes == old['nodes'][:-1], 'Ordinary QA nodes changed instead of being reused'
    parsed_valid = None
    finish_reason = None
    if terminal and terminal[0]['status'] == 'success':
        stored = receipt(terminal[0]['source'], terminal[0]['request_id'])
        assert stored['payload']['messages'] == [{'role': 'user', 'content': value['goal_prompt']}]
        raw = stored['result']['answer']
        finish_reason = stored['result']['finish_reason']
        assert raw == value['raw_goal_output']
        assert finish_reason == value['finish_reason']
        lines = raw.strip().splitlines() if isinstance(raw, str) else []
        parsed_valid = (finish_reason == 'stop' and len(lines) == 2 and lines[0].startswith('Conclusion:')
                        and lines[1].startswith('Final answer:') and bool(lines[0][len('Conclusion:'):].strip())
                        and bool(lines[1][len('Final answer:'):].strip()))
        assert value['parse_result']['valid'] == parsed_valid
        if parsed_valid:
            assert value['status'] == 'success'
            assert value['prediction'] == lines[1][len('Final answer:'):].strip()
            assert value['conclusion'] == lines[0][len('Conclusion:'):].strip()
        else:
            assert value['status'] == 'algorithm_failure' and value['prediction'] == ''
    if scorable(value):
        assert len(prior) == len(kept) and len(terminal) == 1
    return {'passed': True, 'source_union_step_id': goal_sid,
            'source_logical_calls': len(old['logical_calls']), 'reused_prefix_calls': len(prior),
            'expected_complete_prefix_calls': len(kept), 'terminal_goal_calls_recorded': len(terminal),
            'old_goal_formatter_removed': bool(goal.get('sentence_format_request_id')),
            'expected_complete_new_call_count': len(kept) + 1,
            'all_actual_retained_prefixes_exact': True,
            'terminal_parse_independently_checked': parsed_valid is not None,
            'terminal_format_valid': parsed_valid, 'terminal_finish_reason': finish_reason,
            'full_model_response_observed': len(terminal) == 1 and terminal[0]['status'] == 'success'
                                           and len(prior) == len(kept)}


def _quality(items):
    valid = [item for item in items if item['f1'] is not None]
    total = sum(item['f1'] for item in valid)
    n, m = len(items), len(valid)
    def paired(key):
        delta = [item['f1'] - item[key] for item in valid]
        return {'n': m, 'candidate_F1': total / m if m else None,
                'comparison_F1': sum(item[key] for item in valid) / m if m else None,
                'delta': sum(delta) / m if m else None,
                'up': sum(value > 1e-12 for value in delta),
                'tie': sum(abs(value) <= 1e-12 for value in delta),
                'down': sum(value < -1e-12 for value in delta)}
    return {'planned': n, 'scorable': m, 'missing': n - m,
            'statuses': dict(Counter(item['status'] for item in items)),
            'F1_available': total / m if m else None,
            'F1_complete': total / n if n == m and n else None,
            'EM_available': sum(item['em'] for item in valid) / m if m else None,
            'F1_all_planned_bounds': [total / n, (total + n - m) / n] if n else None,
            'paired_with_union20': paired('union_f1'), 'paired_with_p02': paired('p02_f1')}


def _bootstrap(items, other):
    valid = [item for item in items if item['f1'] is not None]
    n = len(valid)
    rng = np.random.default_rng(20261020)
    sample = np.zeros(10000)
    if n:
        for ds in DATASETS:
            values = np.array([item['f1'] - item[other] for item in valid if item['dataset'] == ds])
            if len(values):
                sample += values[rng.integers(len(values), size=(10000, len(values)))].sum(1) / n
    return {'n': n, 'delta': sum(item['f1'] - item[other] for item in valid) / n if n else None,
            'paired_bootstrap95': np.quantile(sample, [.025, .975]).tolist() if n else None,
            'method': 'Paired query bootstrap stratified by dataset preserving observed counts',
            'dataset_counts': dict(Counter(item['dataset'] for item in valid)), 'seed': 20261020, 'replicates': 10000,
            'not_a_noninferiority_claim': True, 'numeric_noninferiority_margin': None}


def _efficiency(items):
    fully_returned = [item for item in items if item.get('lineage', {}).get('full_model_response_observed')
                      and item.get('new_usage', {}).get('usage_missing') == 0]
    result = {'matched_full_returned_flows': len(fully_returned),
              'algorithm_failures_among_returned': sum(item['status'] == 'algorithm_failure' for item in fully_returned)}
    for key in ('logical_generations', 'prompt_tokens', 'completion_tokens', 'equal_rate_generation_cny'):
        old = sum(item['union_usage'][key] for item in fully_returned)
        new = sum(item['new_usage'][key] for item in fully_returned)
        result[key] = {'union20': old, 'terminal_goal': new, 'saved': old - new,
                       'relative_change': (new - old) / old if old else None}
    result['actual_paid_cost'] = False
    return result


def _charges():
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as database:
        all_rows = list(database.execute('SELECT namespace,reserved,actual,status FROM charges'))
    own = [row for row in all_rows if row[0].startswith(OUT.name)]
    def stats(values):
        known = sum(row[2] or 0 for row in values)
        reserve = sum(row[1] for row in values if row[2] is None)
        return {'new_physical_requests': len(values), 'known_returned_usage_cny': known,
                'unreconciled_reserved_cny': reserve, 'occupied_cny': known + reserve,
                'unreconciled_requests': sum(row[2] is None for row in values),
                'statuses': dict(Counter(row[3] for row in values))}
    def dataset(namespace):
        parts = [part for part in namespace.replace('\\', '/').split('/') if part]
        return parts[parts.index('methods') + 1] if 'methods' in parts else None
    return {'new_namespace': stats(own), 'by_dataset': {ds: stats([row for row in own if dataset(row[0]) == ds]) for ds in DATASETS},
            'global_budget': stats(all_rows), 'global_cap_cny': 300, 'actual_invoice_verified': False,
            'historical_seed_costs_not_charged_again': True}


def generate(*, output_dir=None):
    started = datetime.now(timezone.utc).isoformat()
    runtime = rows(HERE / 'runtime300.frozen.jsonl')
    refs = {item['id']: item for item in rows(HERE / 'references300.local.frozen.jsonl')}
    source_records = read(HERE / 'UNION_SOURCES.frozen.json')
    sources = {item['id']: item for item in source_records}
    assert len(runtime) == len({item['id'] for item in runtime}) == 300
    assert Counter(item['dataset'] for item in runtime) == {ds: 100 for ds in DATASETS}
    native = {ds: metric(ds) for ds in DATASETS}
    from experiments.query_decomposition.p02_flow_simplify_v1_20261020.score import select_case
    p02_root = OUTPUT / 'b0_strategy_round2_v1_20261020'
    details, errors = [], []
    for case in runtime:
        cid, ds = case['id'], case['dataset']
        source = sources[cid]
        assert sha(ROOT / source['path']) == source['sha256']
        old = read(ROOT / source['path'])
        assert old['id'] == cid and old['status'].startswith('success')
        old_metric = native[ds](old['prediction'], refs[cid])
        _, p02 = select_case(p02_root, ds, 'p02_role_headers', case['qid'])
        assert p02 and p02['status'].startswith('success')
        path, value = selected(ds, case['qid'])
        item = {'id': cid, 'dataset': ds, 'status': value['status'] if value else 'unstarted',
                'source': str(path.relative_to(ROOT)) if path else None,
                'f1': None, 'em': None, 'prediction': value.get('prediction') if value else None,
                'union_f1': old_metric['f1'], 'p02_f1': native[ds](p02['prediction'], refs[cid])['f1'],
                'union_usage': usage(old['logical_calls'])}
        if value:
            assert value['id'] == cid and value['question'] == case['question']
            assert value['source_union_case'] == source['path'] and value['source_union_sha256'] == source['sha256']
            if scorable(value):
                item.update({'f1': 0.0, 'em': 0.0} if value['status'] == 'algorithm_failure'
                            else native[ds](value['prediction'], refs[cid]))
            try:
                item['new_usage'] = usage(value.get('logical_calls', []))
                item['lineage'] = lineage(value, old)
            except Exception as exc:
                errors.append({'id': cid, 'error': type(exc).__name__ + ': ' + str(exc)})
        details.append(item)
    result = {'snapshot_started_utc': started, 'snapshot_finished_utc': datetime.now(timezone.utc).isoformat(),
              'arm': ARM, 'overall': _quality(details),
              'by_dataset': {ds: _quality([item for item in details if item['dataset'] == ds]) for ds in DATASETS},
              'paired_bootstrap': {name: {**_bootstrap(details, key),
                    'by_dataset': {ds: _bootstrap([item for item in details if item['dataset'] == ds], key) for ds in DATASETS}}
                                  for name, key in [('union20', 'union_f1'), ('p02', 'p02_f1')]},
              'matched_returned_flow_efficiency': {**_efficiency(details),
                    'by_dataset': {ds: _efficiency([item for item in details if item['dataset'] == ds]) for ds in DATASETS}},
              'recorded_selected_prefix_usage': {key: sum(item.get('new_usage', {}).get(key, 0) for item in details)
                      for key in ('logical_attempts', 'logical_generations', 'prompt_tokens', 'completion_tokens', 'unreturned_calls')},
              'actual_costs': _charges(), 'receipt_lineage_errors': errors,
              'all_saved_lineages_verified': not errors,
              'algorithm_output_diagnostics': [{'id': item['id'], 'dataset': item['dataset'],
                    'finish_reason': captured.get('finish_reason'),
                    'parse_result': captured.get('parse_result'), 'raw_goal_output': captured.get('raw_goal_output')}
                    for item in details if item['status'] == 'algorithm_failure'
                    for _, captured in [selected(item['dataset'], item['id'].split(':', 1)[1])]],
              'frozen_rates_cny_per_token': {'prompt': INPUT_RATE, 'completion': OUTPUT_RATE},
              'rate_source': 'Existing evaluation budget returned_usage settlement; no updated market-price assumption',
              'metric': 'Native final Answer F1/EM; 2Wiki entity aliases/demonyms; MuSiQue aliases',
              'numeric_noninferiority_margin': None, 'auto_promoted': False,
              'notes': ['Positive F1 gain is not required for useful simplification.',
                        'Partial available means do not replace complete300 evaluation; infrastructure missing is not zero.',
                        'Algorithm failures are zero; fully returned malformed outputs remain in efficiency if all usage is known.',
                        'Logical modeled costs count cached generations at frozen equal rates; actual new costs come only from this namespace ledger.',
                        'Bootstrap crossing zero does not prove noninferiority.',
                        'No new decomposition, retrieval or API request is performed by this scorer.'],
              'source_pins': {str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__), HERE / 'runtime300.frozen.jsonl', HERE / 'references300.local.frozen.jsonl', HERE / 'UNION_SOURCES.frozen.json']},
              'new_API_calls': 0, 'new_index_searches': 0}
    output_dir = Path(output_dir or OUT / 'reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'REPORT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (output_dir / 'PAIRED_DETAILS.jsonl').write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in details), encoding='utf8')
    lines = ['终点goal单次回答：独立评分与成本核查', result['snapshot_finished_utc'],
             '范围：同300题，三个数据集各100；F1均为0到1原始比例。',
             '完整结果前不选择或升级；不设未经用户指定的非劣数值阈值。', '',
             json.dumps({key: result[key] for key in ['overall', 'by_dataset', 'paired_bootstrap', 'matched_returned_flow_efficiency', 'recorded_selected_prefix_usage', 'actual_costs', 'receipt_lineage_errors']}, ensure_ascii=False, indent=2), '',
             *result['notes']]
    (output_dir / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'overall': result['overall'], 'by_dataset': result['by_dataset'],
                      'receipt_lineage_errors': errors, 'actual_costs': result['actual_costs']['new_namespace'],
                      'output': str(output_dir)}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    generate(output_dir=args.output_dir)
