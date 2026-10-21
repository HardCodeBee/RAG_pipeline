"""Independent offline trajectory, full-payload cache priority and ledger audit.

Re-executes control flow against recorded raw retrieval packets and SQL model
receipts only; it never requests a generation or loads a retrieval index.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt
from experiments.query_decomposition.component_chain_opt_v2_20261019.policy import IntermediateLength
from experiments.query_decomposition.p02_flow_simplify_v1_20261020 import pipeline
from . import cache, scorer
from .scorer import ROOT, HERE, OUT, OUTPUT, OLD_OUT, ARMS, DATASETS, read, rows, sha, digest, resolved


def _packet_db(ds, case, query, resources):
    source = OUT / f'retrieval/{ds}/checkpoint.sqlite3'
    key = digest([case['qid'], query, 10, resources])
    with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True, timeout=20) as db:
        row = db.execute("SELECT value FROM data WHERE kind='retrieval' AND key=?", (key,)).fetchone()
    assert row, ('Missing exact frozen-resource packet', ds, case['id'], query)
    return json.loads(row[0])


def verify_case(case, arm, value, plan, initial_catalog):
    calls, retrievals = value['logical_calls'], value['logical_retrievals']
    call_index, query_index, nodes, selected_counts = 0, 0, [], Counter()
    exposed_counts, route_counts = [], Counter()

    def retrieve(query):
        nonlocal query_index
        expected = retrievals[query_index]
        assert expected['query'] == query and expected['qid'] == case['qid'], 'Downstream query was not reconstructed from actual predecessor outputs'
        packet = _packet_db(case['dataset'], case, query, plan['resources'][case['dataset']])
        assert packet['query'] == query and packet['qid'] == case['qid']
        assert [d['doc_id'] for d in packet['docs']] == expected['doc_ids']
        assert len(packet['docs']) <= 10 and len(packet['docs']) == len({d['doc_id'] for d in packet['docs']})
        query_index += 1
        return packet

    def ask(stage, prompt, maximum, allow_final_truncation=False):
        nonlocal call_index
        wanted = calls[call_index]
        request = payload(plan['model'], prompt, maximum)
        key = digest(request)
        assert stage == wanted['stage'] and key == wanted['payload_sha256']
        sql = scorer.receipt(wanted)
        assert request == sql['payload'], 'Reconstructed prompt/max/settings differ from receipt'
        assert wanted['status'] == sql['status']
        if key in initial_catalog:
            initial = initial_catalog[key]
            assert initial['payload'] == request
            assert wanted['request_id'] == initial['request_id'] and resolved(wanted['source']) == resolved(initial['source']), 'Known successful payload did not use priority-selected receipt'
            selected_counts[initial['cache_tier']] += 1
            assert wanted['reused'] is True
        else:
            selected_counts['new_input'] += 1
        call_index += 1
        return use_receipt({'request_id': wanted['request_id'], **sql}, stage, allow_final_truncation)

    failure = None
    try:
        replay = pipeline.execute(case, retrieve, ask, arm=arm, readout=plan['readout'], record_node=nodes.append)
    except IntermediateLength as exc:
        failure = {'type': type(exc).__name__, 'stage': exc.stage, 'request_id': exc.request_id, 'answer': exc.answer}
        assert value['status'] == 'algorithm_failure'
        assert value['prediction'] == '' and value['failed_stage'] == exc.stage
        assert value['failed_request_id'] == exc.request_id and value['unpropagated_truncated_answer'] == exc.answer
        replay = None
    assert call_index == len(calls) and query_index == len(retrievals), 'Recorded trajectory has unused/stitched callbacks'
    assert nodes == value['nodes'], 'Subproblem nodes differ from sequential reconstruction'
    if replay is not None:
        for key, expected in replay.items():
            assert key in value and value[key] == expected, (case['id'], arm, key)
        assert replay['original_retrieval'] == value['nodes'][-1]['original_retrieval']
        assert all(d['block'] in value['final_prompt'] for d in value['final_reader_packet']['docs'])
    for node in nodes:
        original, rewritten = node['original_retrieval'], node['rewritten_retrieval']
        merged = node['reader_packet']
        expected = pipeline.merge(original, rewritten, arm=arm)
        assert merged == expected
        union_ids = {d['doc_id'] for p in (original, rewritten) if p for d in p['docs']}
        if arm == 'union20':
            assert {d['doc_id'] for d in merged['docs']} == union_ids, 'Full-union candidate was dropped'
        assert all(d['block'] in node['child_prompt'] for d in merged['docs']), 'Selected source block absent from actual child prompt'
        exposed_counts.append(len(merged['docs']))
        route_counts['two_routes' if rewritten else 'single_route'] += 1
    return {'id': case['id'], 'dataset': case['dataset'], 'arm': arm, 'passed': True,
        'logical_calls': len(calls), 'logical_retrievals': len(retrievals), 'complete_nodes': len(nodes),
        'cache_tier_calls': dict(selected_counts), 'reader_doc_counts': exposed_counts,
        'reader_route_counts': dict(route_counts), 'full_union_retained': arm == 'union20',
        'all_recorded_blocks_in_actual_prompts': True, 'failure_independently_reproduced': failure}


def physical_audit(initial_catalog):
    """New physical requests must never overlap any previously successful input."""
    records, duplicate_success = [], []
    success_by_key = defaultdict(list)
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True, timeout=20) as db:
        budget_rows = {r[0]: {'namespace': r[1], 'reserved': r[2], 'actual': r[3], 'status': r[4]}
            for r in db.execute('SELECT id,namespace,reserved,actual,status FROM charges')}
    for source in sorted(OUT.glob('rounds/r*/methods/*/*/checkpoint.sqlite3')):
        namespace = str(source.parent.relative_to(OUTPUT))
        with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True, timeout=20) as db:
            for rid, stage, raw, status, result in db.execute('SELECT id,stage,payload,status,result FROM requests'):
                request = json.loads(raw)
                key = digest(request)
                assert key not in initial_catalog, ('Paid request duplicates pre-existing success', source, rid)
                assert rid in budget_rows, 'Physical request missing budget liability'
                charge = budget_rows[rid]
                assert charge['namespace'] == namespace
                assert rid == digest([namespace, request])
                n_attempts = db.execute('SELECT count(*) FROM attempts WHERE request_id=?', (rid,)).fetchone()[0]
                assert n_attempts == 1, 'Implicit retry occurred within an immutable request'
                raw_result = json.loads(result) if result else None
                if status == 'success':
                    assert raw_result['status'] == 'success'
                    u = raw_result.get('usage') or {}
                    if u:
                        cost = (u.get('prompt_tokens', 0) or 0) * scorer.INPUT_RATE + (u.get('completion_tokens', 0) or 0) * scorer.OUTPUT_RATE
                        assert charge['actual'] is not None and abs(charge['actual'] - cost) < 1e-12
                    success_by_key[key].append({'source': str(source.relative_to(ROOT)), 'request_id': rid})
                elif status in ('unknown', 'in_flight'):
                    assert charge['actual'] is None and charge['reserved'] > 0, 'Unknown request liability was discarded'
                records.append({'source': str(source.relative_to(ROOT)), 'request_id': rid, 'stage': stage,
                    'payload_sha256': key, 'status': status, 'reserved': charge['reserved'], 'actual': charge['actual']})
    for key, successes in success_by_key.items():
        if len(successes) > 1:
            duplicate_success.append({'payload_sha256': key, 'receipts': successes})
    assert not duplicate_success, 'Successful new input was paid again'
    own_charges = {rid for rid, row in budget_rows.items() if row['namespace'].replace('\\', '/').split('/')[0] == OUT.name}
    assert own_charges == {r['request_id'] for r in records}, 'Orphan physical liability/request'
    return {'physical_requests': len(records), 'status_counts': dict(Counter(x['status'] for x in records)),
        'paid_overlap_with_initial_success_catalog': 0, 'duplicate_new_successful_payloads': duplicate_success,
        'all_unknown_liabilities_retained': True, 'all_returned_usage_settlements_exact': True, 'requests': records}


def verify(*, allow_partial=False, report_dir=None):
    from . import run
    plan = read(HERE / 'PROTOCOL.frozen.json')
    run.verify(plan)
    assert read(OUT / 'OFFLINE_AUDIT.json')['passed']
    runtime = rows(HERE / 'runtime300.frozen.jsonl')
    refs = {x['id']: x for x in rows(HERE / 'references300.local.frozen.jsonl')}
    assert len(runtime) == len({x['id'] for x in runtime}) == 300
    assert Counter(x['dataset'] for x in runtime) == {ds: 100 for ds in DATASETS}
    initial = cache.catalog(include_current=False)
    frozen_initial = read(HERE / 'HISTORICAL_SUCCESS_INDEX.frozen.json')
    assert set(initial) == {b['payload_sha256'] for b in frozen_initial['bindings']}
    assert digest(frozen_initial['bindings']) == frozen_initial['bindings_sha256']
    for binding in frozen_initial['bindings']:
        entry = initial[binding['payload_sha256']]
        for key in ('request_id', 'cache_tier', 'result_sha256'):
            assert entry[key] == binding[key], ('Initial successful-input index changed', key)
        assert resolved(entry['source']) == resolved(binding['source'])
    captured = {(c['id'], arm): scorer.selected(c['dataset'], arm, c['qid']) for c in runtime for arm in ARMS}
    for case in runtime:
        for arm in ARMS:
            paths = sorted(OUT.glob(f'rounds/r*/cases/{case["dataset"]}/{arm}/{case["qid"]}.json'))
            terminal = [p for p in paths if scorer.scorable(read(p))]
            assert len(terminal) <= 1, ('Completed method case was rerun', case['id'], arm)
    complete = [(c, a, *captured[(c['id'], a)]) for c in runtime for a in ARMS
                if scorer.scorable(captured[(c['id'], a)][1])]
    if not allow_partial:
        assert len(complete) == 600, 'Incomplete600; final verification must wait'
    scorer.preload([call for _, _, _, value in complete for call in value['logical_calls']])
    native = {ds: scorer.metric(ds) for ds in DATASETS}
    details, quality_rows = [], []
    canonical = {x['id']: x for x in rows(OLD_OUT / 'OFFLINE_AUDIT.details.jsonl')}
    for case, arm, path, value in complete:
        details.append(verify_case(case, arm, value, plan, initial))
        actual_score = {'f1': 0.0, 'em': 0.0} if value['status'] == 'algorithm_failure' else native[case['dataset']](value['prediction'], refs[case['id']])
        quality_rows.append({'id': case['id'], 'dataset': case['dataset'], 'arm': arm,
            'prediction': value['prediction'], 'status': value['status'], **actual_score})
        if arm == 'timing_only':
            old = read(ROOT / canonical[case['id']]['p02_source'])
            for key in ('prediction', 'final_prompt', 'final_context', 'finish_reason'):
                assert value[key] == old[key], ('New baseline did not exactly replay p02', case['id'], key)
            if 'helper' in old:
                assert value['helper'] == old['helper']
            assert len(value['nodes']) == len(old['nodes'])
            for node, previous in zip(value['nodes'], old['nodes']):
                assert all(node.get(key) == expected for key, expected in previous.items()), 'Canonical baseline node content changed'
                # The old frozen executor did not always store this diagnostic
                # copy; the new pipeline derives it from the already checked
                # exact raw child prompt without changing any model input.
                extra = set(node) - set(previous)
                assert extra <= {'raw_B0_child_prompt'}
    report_dir = Path(report_dir or OUT / 'reports')
    report = read(report_dir / 'REPORT.json')
    report_items = {(x['id'], x['arm']): x for x in rows(report_dir / 'PAIRED_DETAILS.jsonl')}
    assert len(report_items) == 600
    for item in quality_rows:
        saved = report_items[(item['id'], item['arm'])]
        assert saved['prediction'] == item['prediction'] and saved['status'] == item['status']
        assert abs(saved['f1'] - item['f1']) < 1e-12 and abs(saved['em'] - item['em']) < 1e-12
    independent_scores = {}
    for arm in ARMS:
        items = [x for x in quality_rows if x['arm'] == arm]
        score = {'n': len(items), 'F1': sum(x['f1'] for x in items) / len(items) if items else None,
                 'by_dataset': {ds: {'n': len([x for x in items if x['dataset'] == ds]),
                     'F1': sum(x['f1'] for x in items if x['dataset'] == ds) / len([x for x in items if x['dataset'] == ds])
                     if any(x['dataset'] == ds for x in items) else None} for ds in DATASETS}}
        assert score['n'] == report['arms'][arm]['scorable']
        if score['n']:
            assert abs(score['F1'] - report['arms'][arm]['F1_available']) < 1e-12
        for ds in DATASETS:
            assert score['by_dataset'][ds]['n'] == report['arms'][arm]['by_dataset'][ds]['scorable']
            if score['by_dataset'][ds]['n']:
                assert abs(score['by_dataset'][ds]['F1'] - report['arms'][arm]['by_dataset'][ds]['F1_available']) < 1e-12
        independent_scores[arm] = score
    cold_usage = {}
    for arm in ARMS:
        groups = {}
        for ds in (None, *DATASETS):
            totals = Counter(logical_attempts=0, logical_generations=0, prompt_tokens=0, completion_tokens=0)
            for case, this_arm, _, value in complete:
                if this_arm != arm or ds is not None and case['dataset'] != ds:
                    continue
                for call in value['logical_calls']:
                    totals['logical_attempts'] += 1
                    sql = scorer.receipt(call)
                    if sql['status'] == 'success':
                        u = sql['result']['usage']
                        assert u.get('prompt_tokens') is not None and u.get('completion_tokens') is not None
                        totals['logical_generations'] += 1
                        totals['prompt_tokens'] += u['prompt_tokens']
                        totals['completion_tokens'] += u['completion_tokens']
            expected = report['arms'][arm]['logical_usage'] if ds is None else report['arms'][arm]['by_dataset'][ds]['logical_usage']
            if not allow_partial:
                for key, amount in totals.items():
                    assert amount == expected[key], ('Cold logical usage differs from SQL', arm, ds, key)
            result_usage = dict(totals)
            result_usage['equal_rate_generation_cny'] = totals['prompt_tokens'] * scorer.INPUT_RATE + totals['completion_tokens'] * scorer.OUTPUT_RATE
            if not allow_partial:
                assert abs(result_usage['equal_rate_generation_cny'] - expected['equal_rate_generation_cny']) < 1e-10
            groups['overall' if ds is None else ds] = result_usage
        cold_usage[arm] = groups
    physical = physical_audit(initial)
    costs = scorer.actual_costs()
    assert costs['global']['occupied_cny'] <= 300 + 1e-9
    assert costs['new_namespace']['physical_requests'] == physical['physical_requests']
    assert costs['by_arm']['timing_only']['physical_requests'] == 0
    for key in ('physical_requests', 'returned_usage_cny', 'unreconciled_reserved_cny'):
        assert abs(costs['new_namespace'][key] - report['actual_costs']['new_namespace'][key]) < 1e-10
    logical = {arm: sum(d['logical_calls'] for d in details if d['arm'] == arm) for arm in ARMS}
    if not allow_partial:
        assert logical['timing_only'] == 1774
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'passed': True, 'partial_snapshot': len(complete) < 600,
        'complete_case_counts': {arm: sum(a == arm for _, a, _, _ in complete) for arm in ARMS},
        'independent_native_scores': independent_scores, 'logical_calls': logical,
        'independent_cold_logical_usage': cold_usage,
        'independent_actual_costs': costs, 'physical_requests_audit': physical,
        'all_saved_model_payloads_and_sequential_trajectories_verified': True,
        'canonical_priority_and_same_input_response_reuse_verified': True,
        'initial_successful_payload_index_verified': len(initial),
        'no_completed_method_cases_rerun': True,
        'allowed_legacy_schema_differences': {
            'goal_node_raw_B0_child_prompt': 'Only additional node diagnostic allowed; independently reconstructed by the exact frozen pipeline before comparing all saved node fields.',
            'optional_old_helper_field': 'Older p02 files may omit helper; exact final_context and final_prompt remain mandatory and include the rendered helper.'},
        'merged_blocks_all_in_actual_prompts': True,
        'no_changed_prefix_stitched_with_old_downstream_case': True,
        'baseline_all1774_exact_when_complete': not allow_partial,
        'report_sha256': sha(report_dir / 'REPORT.json'),
        'source_pins': {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), HERE / 'scorer.py', HERE / 'PROTOCOL.frozen.json')},
        'new_API_calls': 0, 'new_index_searches': 0, 'details': details}
    target = report_dir / ('PARTIAL_VERIFICATION.json' if allow_partial else 'FINAL_VERIFICATION.json')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'passed': True, 'partial_snapshot': result['partial_snapshot'], 'scores': independent_scores,
        'logical_calls': logical, 'physical_requests': physical['physical_requests'],
        'new_cost': costs['new_namespace'], 'output': str(target)}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args()
    verify(allow_partial=args.allow_partial, report_dir=args.report_dir)
