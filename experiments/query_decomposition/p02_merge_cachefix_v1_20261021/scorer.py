"""Offline same-input-controlled native scoring, usage and physical ledger audit.

The primary comparison uses two actually saved new-namespace trajectories.
Historical union cases are a labelled diagnostic, never substituted outcomes.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
OLD_OUT = OUTPUT / 'p02_flow_simplify_v1_20261020'
DATASETS = ('hotpotqa', '2wiki', 'musique')
ARMS = ('timing_only', 'union20')
INPUT_RATE, OUTPUT_RATE = .8e-6, 2e-6
_RECEIPTS = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolved(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def scorable(value):
    return bool(value and (value.get('status') == 'algorithm_failure' or
                value.get('status', '').startswith('success') and isinstance(value.get('prediction'), str)))


def selected(dataset, arm, qid, *, root=OUT):
    values = [(path, read(path)) for path in sorted(Path(root).glob(f'rounds/r*/cases/{dataset}/{arm}/{qid}.json'))]
    terminal = [(p, v) for p, v in values if scorable(v)]
    if terminal:
        assert len({(v['status'].startswith('success'), v.get('prediction')) for _, v in terminal}) == 1, 'Conflicting completed outcomes'
        return terminal[0]
    return values[-1] if values else (None, None)


def preload(calls):
    by_source = defaultdict(set)
    for call in calls:
        key = (str(resolved(call['source'])), call['request_id'])
        if key not in _RECEIPTS:
            by_source[key[0]].add(key[1])
    for source, ids in by_source.items():
        ids = sorted(ids)
        with sqlite3.connect(Path(source).as_uri() + '?mode=ro', uri=True, timeout=20) as db:
            for start in range(0, len(ids), 400):
                chunk = ids[start:start + 400]
                for rid, p, status, result in db.execute('SELECT id,payload,status,result FROM requests WHERE id IN (' + ','.join('?' for _ in chunk) + ')', chunk):
                    _RECEIPTS[(source, rid)] = {'payload': json.loads(p), 'status': status,
                        'result': json.loads(result) if result else None}
        assert all((source, rid) in _RECEIPTS for rid in ids), ('Missing source receipt', source)


def receipt(call):
    key = (str(resolved(call['source'])), call['request_id'])
    if key not in _RECEIPTS:
        preload([call])
    return _RECEIPTS[key]


def usage(calls):
    value = Counter(logical_attempts=len(calls), logical_generations=0,
        prompt_tokens=0, completion_tokens=0, unreturned_calls=0, usage_missing=0)
    origins = Counter()
    for call in calls:
        stored = receipt(call)
        assert digest(stored['payload']) == call['payload_sha256'], 'Recorded input hash differs from full SQL payload'
        assert stored['status'] == call['status']
        origins[call.get('cache_origin', 'unspecified')] += 1
        if call['status'] != 'success':
            value['unreturned_calls'] += 1
            continue
        raw = stored['result']
        assert raw['status'] == 'success'
        u = raw.get('usage') or {}
        for key in ('prompt_tokens', 'completion_tokens'):
            if key in (call.get('usage') or {}):
                assert call['usage'][key] == u.get(key), 'Copied token usage differs from SQL receipt'
        assert call.get('finish_reason') == raw.get('finish_reason')
        value['logical_generations'] += 1
        value['usage_missing'] += any(u.get(k) is None for k in ('prompt_tokens', 'completion_tokens'))
        value['prompt_tokens'] += u.get('prompt_tokens', 0) or 0
        value['completion_tokens'] += u.get('completion_tokens', 0) or 0
    result = dict(value)
    result.update(equal_rate_generation_cny=value['prompt_tokens'] * INPUT_RATE + value['completion_tokens'] * OUTPUT_RATE,
                  cache_origin_counts=dict(origins))
    return result


def aggregate_usage(values):
    values = list(values)
    fields = ('logical_attempts', 'logical_generations', 'prompt_tokens', 'completion_tokens',
              'unreturned_calls', 'usage_missing', 'equal_rate_generation_cny')
    return {'selected_case_records': len(values), **{k: sum(v.get(k, 0) for v in values) for k in fields},
            'scope': 'Logical generations of selected records, including cached replays; incomplete prefixes remain labelled.'}


def quality(items):
    n, valid = len(items), [x for x in items if x['f1'] is not None]
    m, total = len(valid), sum(x['f1'] for x in valid)
    return {'planned': n, 'scorable': m, 'missing': n - m,
            'statuses': dict(Counter(x['status'] for x in items)),
            'F1_available': total / m if m else None,
            'EM_available': sum(x['em'] for x in valid) / m if m else None,
            'F1_complete': total / n if n and n == m else None,
            'F1_all_planned_bounds': [total / n, (total + n - m) / n] if n else None,
            'logical_usage': aggregate_usage(x['usage'] for x in items if x.get('usage') is not None)}


def paired(items, comparison_key='baseline_f1'):
    valid = [x for x in items if x['f1'] is not None and x.get(comparison_key) is not None]
    n = len(valid)
    differences = np.array([x['f1'] - x[comparison_key] for x in valid], dtype=float)
    rng, bootstrap = np.random.default_rng(20261021), np.zeros(10000)
    for ds in DATASETS:
        values = np.array([x['f1'] - x[comparison_key] for x in valid if x['dataset'] == ds])
        if len(values):
            bootstrap += values[rng.integers(len(values), size=(10000, len(values)))].sum(1) / n
    return {'n': n, 'dataset_counts': dict(Counter(x['dataset'] for x in valid)),
        'candidate_F1': sum(x['f1'] for x in valid) / n if n else None,
        'comparison_F1': sum(x[comparison_key] for x in valid) / n if n else None,
        'delta': float(differences.mean()) if n else None,
        'up': int(sum(differences > 1e-12)), 'tie': int(sum(abs(differences) <= 1e-12)),
        'down': int(sum(differences < -1e-12)),
        'paired_bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist() if n else None,
        'bootstrap_method': 'Paired query bootstrap stratified by dataset;10000 resamples;seed20261021',
        'numeric_noninferiority_margin': None, 'not_a_noninferiority_claim': True}


def actual_costs():
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True, timeout=20) as db:
        all_rows = list(db.execute('SELECT id,namespace,reserved,actual,status FROM charges'))
    own, excluded = [], []
    for rid, ns, reserved, actual, status in all_rows:
        parts = ns.replace('\\', '/').split('/')
        if parts and parts[0] == OUT.name:
            entry = {'request_id': rid, 'namespace': ns, 'reserved': reserved, 'actual': actual, 'status': status}
            if len(parts) == 6 and parts[1] == 'rounds' and parts[3] == 'methods' and parts[4] in DATASETS and parts[5] in ARMS:
                own.append({**entry, 'dataset': parts[4], 'arm': parts[5]})
            else:
                excluded.append(entry)
    def stats(values):
        known = sum(x['actual'] or 0 for x in values)
        reserve = sum(x['reserved'] for x in values if x['actual'] is None)
        return {'physical_requests': len(values), 'returned_usage_cny': known,
            'unreconciled_reserved_cny': reserve, 'occupied_cny': known + reserve,
            'unreconciled_requests': sum(x['actual'] is None for x in values),
            'statuses': dict(Counter(x['status'] for x in values))}
    global_rows = [{'actual': r[3], 'reserved': r[2], 'status': r[4]} for r in all_rows]
    return {'new_namespace': stats(own), 'by_arm': {arm: {**stats([x for x in own if x['arm'] == arm]),
        'by_dataset': {ds: stats([x for x in own if x['arm'] == arm and x['dataset'] == ds]) for ds in DATASETS}}
        for arm in ARMS}, 'global': stats(global_rows), 'global_cap_cny': 300,
        'unexpected_namespace_charge_rows': excluded, 'actual_invoice_verified': False,
        'historical_cache_usage_not_charged_here': True}


def generate(*, sample='300', output_dir=None):
    started = datetime.now(timezone.utc).isoformat()
    runtime_path = HERE / ('pilot60.runtime.frozen.jsonl' if sample == 'pilot60' else 'runtime300.frozen.jsonl')
    runtime = rows(runtime_path)
    refs_path = HERE / 'references300.local.frozen.jsonl'
    refs = {x['id']: x for x in rows(refs_path)}
    expected_n = 60 if sample == 'pilot60' else 300
    assert len(runtime) == len({x['id'] for x in runtime}) == expected_n
    assert Counter(x['dataset'] for x in runtime) == {ds: expected_n // 3 for ds in DATASETS}
    canonical = {x['id']: x for x in rows(OLD_OUT / 'OFFLINE_AUDIT.details.jsonl')}
    native, captured, old_union, old_p02 = {ds: metric(ds) for ds in DATASETS}, {}, {}, {}
    for case in runtime:
        cid, ds, qid = case['id'], case['dataset'], case['qid']
        old_p02[cid] = read(ROOT / canonical[cid]['p02_source'])
        old_union[cid] = selected(ds, 'union20', qid, root=OLD_OUT)[1]
        for arm in ARMS:
            captured[(cid, arm)] = selected(ds, arm, qid)
    preload([call for _, value in captured.values() if value for call in value.get('logical_calls', [])])
    details, lookup, errors = [], {}, []
    for case in runtime:
        cid, ds = case['id'], case['dataset']
        for arm in ARMS:
            path, value = captured[(cid, arm)]
            item = {'id': cid, 'dataset': ds, 'arm': arm, 'status': value['status'] if value else 'unstarted',
                'source': str(path.relative_to(ROOT)) if path else None,
                'prediction': value.get('prediction') if value else None, 'f1': None, 'em': None,
                'frozen_p02_f1': native[ds](old_p02[cid]['prediction'], refs[cid])['f1'],
                'historical_union_f1': native[ds](old_union[cid]['prediction'], refs[cid])['f1'] if scorable(old_union[cid]) else None,
                'historical_union_prediction': old_union[cid].get('prediction') if old_union[cid] else None,
                'usage': None}
            if value:
                assert value['id'] == cid and value['dataset'] == ds and value['question'] == case['question'] and value['arm'] == arm
                if scorable(value):
                    item.update({'f1': 0.0, 'em': 0.0} if value['status'] == 'algorithm_failure' else native[ds](value['prediction'], refs[cid]))
                try:
                    item['usage'] = usage(value.get('logical_calls', []))
                except Exception as exc:
                    errors.append({'id': cid, 'arm': arm, 'error': type(exc).__name__ + ': ' + str(exc)})
            details.append(item)
            lookup[(cid, arm)] = item
    for item in details:
        item['baseline_f1'] = lookup[(item['id'], 'timing_only')]['f1']
    summaries = {arm: {**quality([x for x in details if x['arm'] == arm]),
        'by_dataset': {ds: quality([x for x in details if x['arm'] == arm and x['dataset'] == ds]) for ds in DATASETS}}
        for arm in ARMS}
    union = [x for x in details if x['arm'] == 'union20']
    pair = {**paired(union), 'by_dataset': {ds: paired([x for x in union if x['dataset'] == ds]) for ds in DATASETS}}
    repaired = [x for x in union if x['f1'] is not None and x['historical_union_f1'] is not None]
    same_p02 = [x for x in details if x['arm'] == 'timing_only' and x['f1'] is not None]
    matched = [x for x in union if x['f1'] is not None and x['baseline_f1'] is not None and x['usage'] is not None
               and lookup[(x['id'], 'timing_only')]['usage'] is not None]
    def efficiency(items):
        result = {'matched_scorable_queries': len(items), 'algorithm_failures_included': True}
        for key in ('logical_generations', 'prompt_tokens', 'completion_tokens', 'equal_rate_generation_cny'):
            baseline = sum(lookup[(x['id'], 'timing_only')]['usage'][key] for x in items)
            candidate = sum(x['usage'][key] for x in items)
            result[key] = {'timing_only': baseline, 'union20': candidate, 'change': candidate - baseline,
                           'relative_change': (candidate - baseline) / baseline if baseline else None}
        return result
    result = {'snapshot_started_utc': started, 'snapshot_finished_utc': datetime.now(timezone.utc).isoformat(),
        'planned': expected_n, 'dataset_counts': dict(Counter(x['dataset'] for x in runtime)), 'arms': summaries,
        'primary_union20_minus_new_timing_only': pair,
        'all_arm_common_same_query': {'n': pair['n'], 'by_dataset': {ds: pair['by_dataset'][ds]['n'] for ds in DATASETS},
            'timing_only_F1': pair['comparison_F1'], 'union20_F1': pair['candidate_F1']},
        'baseline_replay_scoring_check': {'available': len(same_p02), 'score_differences_from_frozen_p02':
            sum(abs(x['f1'] - x['frozen_p02_f1']) > 1e-12 for x in same_p02),
            'prediction_differences': sum(x['prediction'] != old_p02[x['id']]['prediction'] for x in same_p02)},
        'historical_union_cache_repair_diagnostic': {**paired(repaired, 'historical_union_f1'),
            'prediction_changed': sum(x['prediction'] != x['historical_union_prediction'] for x in repaired),
            'not_the_primary_method_comparison': True,
            'by_dataset': {ds: paired([x for x in repaired if x['dataset'] == ds], 'historical_union_f1') for ds in DATASETS}},
        'matched_scorable_flow_efficiency': {**efficiency(matched), 'by_dataset': {ds: efficiency([x for x in matched if x['dataset'] == ds]) for ds in DATASETS}},
        'actual_costs': actual_costs(), 'receipt_usage_errors': errors, 'all_saved_receipt_usage_verified': not errors,
        'frozen_unit_rates_cny_per_token': {'prompt': INPUT_RATE, 'completion': OUTPUT_RATE},
        'metric': 'Native final Answer F1/EM; 2Wiki aliases/demonyms; MuSiQue aliases',
        'algorithm_failure_scored_zero': True, 'infrastructure_missing_not_scored_zero': True,
        'numeric_noninferiority_margin': None, 'auto_promoted': False,
        'new_API_calls': 0, 'new_index_searches': 0,
        'source_pins': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), runtime_path, refs_path]},
        'notes': ['Primary comparison uses actual new timing_only versus new union20 case records on identical questions.',
            'Canonical p02 successful full payloads have priority; controlled replay is not all-fresh inference.',
            'Historical union is reported only to expose the cache repair, never used as a new trajectory.',
            'Logical cold costs count cached generations; physical ledger includes all new retries and unknown liabilities.',
            'Incomplete means are labelled; bootstrap crossing zero does not establish equivalence or noninferiority.',
            'This is an exposed development300 comparison, not an independent generalization test.']}
    output_dir = Path(output_dir or OUT / ('reports' if sample == '300' else 'reports_pilot60'))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'REPORT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (output_dir / 'PAIRED_DETAILS.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in details), encoding='utf8')
    lines = ['Canonical-cache-controlled Top10 versus full union', result['snapshot_finished_utc'],
             'All F1 values below are raw0..1 proportions.', json.dumps(result, ensure_ascii=False, indent=2)]
    (output_dir / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'arms': {a: {k: summaries[a][k] for k in ('planned', 'scorable', 'missing', 'F1_available', 'F1_complete', 'statuses')} for a in ARMS},
        'primary_comparison': pair, 'receipt_usage_errors': errors, 'actual_costs': result['actual_costs']['new_namespace'],
        'output': str(output_dir)}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', choices=['pilot60', '300'], default='300')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    generate(sample=args.sample, output_dir=args.output_dir)
