"""Independent read-only source/receipt audit of the completed adoption report."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
import sqlite3

import numpy as np

from .score import ROOT, HERE, OUT, OLD, DATASETS, rows, read, metric, select_case, sha
from .offline_audit import stored_request
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest

ARMS = ('union20', 'discount20', 'single_final')


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def verify():
    assert (OUT / 'RECOVERY_V003_COMPLETE.json').exists(), 'Recovery still running'
    target = OUT / 'reports/adoption300/REPORT.json'
    report = read(target)
    assert all(report['arms'][arm]['scorable'] == 300 for arm in ARMS), 'Wait for final full300 report'
    samples = rows(HERE / 'runtime300.frozen.jsonl')
    refs = {item['id']: item for item in rows(HERE / 'references300.local.frozen.jsonl')}
    assert len(samples) == len({item['id'] for item in samples}) == 300
    assert Counter(item['dataset'] for item in samples) == {ds: 100 for ds in DATASETS}
    scorers = {ds: metric(ds) for ds in DATASETS}
    old_e = HERE.parent / 'b0_strategy_round2_v1_20261020'
    manifests = {item['id']: item for item in rows(old_e / 'SOURCES.frozen.jsonl')}
    base_calls = {item['id']: item['logical_calls'] for item in rows(HERE.parent / 'b0_six_single_changes_v1_20261020/B0_SOURCES.frozen.jsonl')}
    scores, usage, states = {}, {}, {}
    receipt_count = 0

    def add_usage(calls):
        nonlocal receipt_count
        p = c = 0
        for source in calls:
            stored = stored_request(source['source'], source['request_id'])
            assert stored['status'] == 'success' and stored['result']['status'] == 'success'
            assert digest(stored['payload']) == source['payload_sha256']
            assert stored['payload']['model'] == 'qwen-plus-2025-12-01'
            u = stored['result']['usage']
            assert u['prompt_tokens'] > 0 and u['completion_tokens'] >= 0
            p += u['prompt_tokens']; c += u['completion_tokens']; receipt_count += 1
        return {'logical_calls': len(calls), 'prompt_tokens': p, 'completion_tokens': c}

    for case in samples:
        cid, ds, qid = case['id'], case['dataset'], case['qid']
        _, old = select_case(OLD, ds, 'p02_role_headers', qid)
        assert old['status'].startswith('success')
        scores[(cid, 'p02')] = scorers[ds](old['prediction'], refs[cid])['f1']
        source_rows = list(base_calls[cid])
        manifest = manifests[cid]
        for path in [manifest['control']] + [item['path'] for item in manifest['cache_sources']]:
            source_rows += read(ROOT / path).get('logical_calls', [])
        source_rows += old['logical_calls']
        by_rid = {row['request_id']: row for row in source_rows if row['status'] == 'success'}
        pointers = []
        for node in old['nodes']:
            pointers += [rid for rid in [node.get('rewrite_request_id'), node['answer_request_id'], node.get('sentence_format_request_id')] if rid]
            # Independently tie the pointer to the exact persisted node input.
            for rid, prompt in [(node.get('rewrite_request_id'), node.get('rewrite_prompt')),
                                (node['answer_request_id'], node['child_prompt'])]:
                if rid:
                    source = by_rid[rid]
                    assert stored_request(source['source'], rid)['payload']['messages'] == [{'role': 'user', 'content': prompt}]
        pointers.append(old['request_id'])
        assert stored_request(by_rid[old['request_id']]['source'], old['request_id'])['payload']['messages'] == [{'role': 'user', 'content': old['final_prompt']}]
        usage[(cid, 'p02')] = add_usage([by_rid[rid] for rid in pointers])
        for arm in ARMS:
            _, value = select_case(OUT, ds, arm, qid)
            assert value is not None and (value['status'].startswith('success') or value['status'] == 'algorithm_failure')
            states[(cid, arm)] = value['status']
            scores[(cid, arm)] = 0.0 if value['status'] == 'algorithm_failure' else scorers[ds](value['prediction'], refs[cid])['f1']
            usage[(cid, arm)] = add_usage([call for call in value['logical_calls'] if call['status'] == 'success'])
    for arm in ARMS:
        value = report['arms'][arm]
        assert value['scorable'] == value['planned'] == 300 and value['missing'] == 0
        close(value['F1_complete'], sum(scores[(case['id'], arm)] for case in samples) / 300)
        for ds in DATASETS:
            selected = [case for case in samples if case['dataset'] == ds]
            child = value['by_dataset'][ds]
            assert child['scorable'] == child['planned'] == 100
            close(child['F1_complete'], sum(scores[(case['id'], arm)] for case in selected) / 100)
        success = [case for case in samples if states[(case['id'], arm)].startswith('success')]
        eff = value['matched_success_efficiency_vs_p02']
        assert eff['matched_complete_success_queries'] == len(success)
        for key in ['logical_calls', 'prompt_tokens', 'completion_tokens']:
            close(eff[key]['p02_total'], sum(usage[(case['id'], 'p02')][key] for case in success))
            close(eff[key]['candidate_total'], sum(usage[(case['id'], arm)][key] for case in success))
        modeled = eff['equal_rate_modeled_generation_cny']
        close(modeled['candidate'], eff['prompt_tokens']['candidate_total'] * .8e-6 + eff['completion_tokens']['candidate_total'] * 2e-6)
        close(modeled['p02'], eff['prompt_tokens']['p02_total'] * .8e-6 + eff['completion_tokens']['p02_total'] * 2e-6)
    intervals = {}
    for other, key in [('p02', 'p02_role_headers'), ('discount20', 'discount20')]:
        rng = np.random.default_rng(20261020)
        bootstrap = np.zeros(10000)
        delta = []
        for ds in DATASETS:
            values = np.array([scores[(case['id'], 'single_final')] - scores[(case['id'], other)] for case in samples if case['dataset'] == ds])
            delta.extend(values)
            bootstrap += values[rng.integers(len(values), size=(10000, len(values)))].sum(1) / 300
        expected = np.quantile(bootstrap, [.025, .975])
        item = report['single_final_paired_bootstrap'][key]
        assert item['n'] == 300 and item['dataset_counts'] == {ds: 100 for ds in DATASETS}
        close(item['delta'], float(np.mean(delta)))
        assert np.allclose(item['paired_bootstrap95'], expected, atol=1e-12, rtol=1e-12)
        assert item['up'] == sum(x > 1e-12 for x in delta)
        assert item['down'] == sum(x < -1e-12 for x in delta)
        assert item['up'] + item['tie'] + item['down'] == 300
        intervals[key] = item
    with sqlite3.connect((OUT.parent / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        charges = list(db.execute('SELECT namespace,reserved,actual FROM charges WHERE namespace LIKE ?', (OUT.name + '%',)))
    actual = []
    for namespace, reserved, cost in charges:
        parts = [part for part in namespace.replace('\\', '/').split('/') if part]
        if len(parts) == 6 and parts[:2] == [OUT.name, 'rounds'] and parts[3] == 'methods':
            actual.append((reserved, cost))
    reported = report['actual_original300_new_costs']['original300_only']
    assert reported['new_physical_requests'] == len(actual)
    close(reported['returned_usage_estimate_cny'], sum(cost or 0 for _, cost in actual))
    close(reported['unreconciled_reserved_cny'], sum(reserve for reserve, cost in actual if cost is None))
    totals = {arm: {key: sum(usage[(case['id'], arm)][key] for case in samples)
                    for key in ['logical_calls', 'prompt_tokens', 'completion_tokens']}
              for arm in ['p02', *ARMS]}
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'scorable_method_cases': 900, 'cases_per_dataset_per_arm': 100,
              'native_F1_recomputed_from_actual_case_predictions': True,
              'full_payload_usage_receipts_checked': receipt_count,
              'full_logical_usage': totals,
              'matched_efficiency_and_equal_rate_modeled_cost_verified': True,
              'single_final_bootstrap_recomputed': intervals,
              'actual_original300_costs_match_ledger_and_exclude_fresh': True,
              'numeric_noninferiority_margin': None, 'automatic_adoption_claimed': False,
              'final_report_sha256': sha(target), 'verification_source_sha256': sha(__file__),
              'new_api_calls': 0, 'new_index_searches': 0}
    (OUT / 'reports/adoption300/INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == '__main__':
    verify()
