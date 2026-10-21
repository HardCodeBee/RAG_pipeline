"""Network-free receipt replay and fixed-packet merge validation.

Only timing_only may replay old answers as an unchanged complete trajectory.
Changed union20/discount20 inputs are inspected, never assigned old predictions
or scored as newly executed RAG. No API client or index is constructed here.
"""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from experiments.query_decomposition.component_chain_c08_c02_v1_20261019 import pipeline as b0
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest
from experiments.query_decomposition.b0_strategy_round2_v1_20261020.strategy import goal_prompt
from .score import select_case

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1' / HERE.name
OLD_E = HERE.parent / 'b0_strategy_round2_v1_20261020'
OLD_O = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1' / OLD_E.name
SIX_E = HERE.parent / 'b0_six_single_changes_v1_20261020'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def stored_request(source, rid):
    path = ROOT / source
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as database:
        row = database.execute('SELECT payload,status,result FROM requests WHERE id=?', (rid,)).fetchone()
    assert row, (source, rid)
    return {'payload': json.loads(row[0]), 'status': row[1],
            'result': json.loads(row[2]) if row[2] else None}


def source_catalog(manifest, baseline_manifest, p02):
    calls = list(baseline_manifest['logical_calls'])
    files = [(manifest['control'], manifest['control_sha256'])]
    files += [(item['path'], item['sha256']) for item in manifest['cache_sources']]
    for source, expected_sha in files:
        assert sha(ROOT / source) == expected_sha, source
        calls.extend(read(ROOT / source).get('logical_calls', []))
    calls.extend(p02.get('logical_calls', []))
    result = {}
    for call in calls:
        if call['status'] != 'success':
            continue
        result.setdefault(call['payload_sha256'], call)
    return result


def packet_catalog(base):
    packets = [base['original_retrieval']]
    for node in base['nodes']:
        packets.extend(packet for packet in [node['original_retrieval'], node.get('rewritten_retrieval')] if packet)
    result = {}
    for packet in packets:
        key = packet['query']
        if key in result:
            assert result[key] == packet, 'Same query has conflicting frozen packets'
        result[key] = packet
    return result


def _expected_discount(original, rewritten):
    routes = [original]
    if rewritten is not None and rewritten['query'] != original['query']:
        routes.append(rewritten)
    by_id = {}
    for route_index, route in enumerate(routes):
        for rank, doc in enumerate(route['docs'], 1):
            item = by_id.setdefault(doc['doc_id'], {'ranks': {}, 'score': 0.0})
            assert route_index not in item['ranks'], 'Duplicate doc_id within a route'
            item['ranks'][route_index] = rank
            item['score'] += 1 / math.log2(rank + 1)
    return sorted(by_id, key=lambda doc_id: (-by_id[doc_id]['score'], min(by_id[doc_id]['ranks'].values()),
                                            by_id[doc_id]['ranks'].get(0, math.inf), doc_id)), by_id


def audit(*, output=OUT / 'OFFLINE_AUDIT.json', limit=None):
    from . import pipeline
    plan = read(OLD_E / 'PROTOCOL.frozen.json')
    for source, expected in plan['source_pins'].items():
        assert sha(ROOT / source) == expected, source
    assert sha(OLD_E / 'runtime300.frozen.jsonl') == plan['runtime_sha256']
    assert sha(OLD_E / 'SOURCES.frozen.jsonl') == plan['source_manifest_sha256']
    runtime = rows(OLD_E / 'runtime300.frozen.jsonl')
    manifests = {item['id']: item for item in rows(OLD_E / 'SOURCES.frozen.jsonl')}
    baseline_manifests = {item['id']: item for item in rows(SIX_E / 'B0_SOURCES.frozen.jsonl')}
    assert len(runtime) == 300 and len({case['id'] for case in runtime}) == 300
    if limit is not None:
        runtime = runtime[:limit]
    examples = read(ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json')['child']
    totals, details = Counter(), []
    for case in runtime:
        manifest = manifests[case['id']]
        assert sha(ROOT / manifest['base']) == manifest['base_sha256']
        assert manifest['base'] == baseline_manifests[case['id']]['source']
        base = read(ROOT / manifest['base'])
        p02_path, p02 = select_case(OLD_O, case['dataset'], 'p02_role_headers', case['qid'])
        assert p02 and p02['status'].startswith('success'), case['id']
        calls = source_catalog(manifest, baseline_manifests[case['id']], p02)
        packets = packet_catalog(base)
        trace = {'retrievals': [], 'calls': []}

        def retrieve(query):
            assert query in packets, 'Unrecorded query in timing-only replay: ' + query
            trace['retrievals'].append(query)
            return deepcopy(packets[query])

        def ask(stage, prompt, maximum, allow_final_truncation=False):
            request = payload(plan['model'], prompt, maximum)
            key = digest(request)
            assert key in calls, 'Changed API payload cannot be replayed: ' + stage
            source = calls[key]
            stored = stored_request(source['source'], source['request_id'])
            assert stored['payload'] == request and digest(stored['payload']) == key
            row = {**source, **stored}
            trace['calls'].append({'stage': stage, 'payload_sha256': key,
                                   'request_id': row['request_id'], 'source': row['source']})
            return use_receipt(row, stage, allow_final_truncation)

        replay = pipeline.execute(case, retrieve, ask, arm='timing_only', readout=plan['readout'])
        for key in ['prediction', 'final_prompt', 'final_context', 'finish_reason']:
            assert replay[key] == p02[key], (case['id'], key)
        assert len(replay['nodes']) == len(p02['nodes'])
        for node, recorded in zip(replay['nodes'], p02['nodes']):
            for key, value in recorded.items():
                assert node.get(key) == value, (case['id'], node['step_id'], key)
        assert replay['original_retrieval'] == base['original_retrieval']
        assert trace['retrievals'] and trace['retrievals'][-1] in packets
        # Exactly one initial-root retrieval is removed; goal still retrieves it.
        expected_requests = []
        for node in p02['nodes']:
            expected_requests.append(node['request'])
            if node.get('rewritten_retrieval') is not None:
                expected_requests.append(node['rewritten_query'])
        assert Counter(trace['retrievals']) == Counter(expected_requests)
        totals.update(cases=1, timing_nodes=len(replay['nodes']), exact_payload_receipts=len(trace['calls']),
                      timing_logical_retrievals=len(trace['retrievals']))
        case_detail = {'id': case['id'], 'dataset': case['dataset'], 'p02_source': str(p02_path.relative_to(ROOT)),
                       'timing_only_exact': True, 'timing_only_replay_calls': trace['calls'],
                       'fixed_packet_diagnostics': []}
        for node in p02['nodes']:
            original, rewritten = node['original_retrieval'], node.get('rewritten_retrieval')
            original_ids = [doc['doc_id'] for doc in original['docs']]
            raw_ids = set(original_ids)
            if rewritten and rewritten['query'] != original['query']:
                raw_ids.update(doc['doc_id'] for doc in rewritten['docs'])
            expected_interleave, seen = [], set()
            active = [original] + ([rewritten] if rewritten and rewritten['query'] != original['query'] else [])
            for rank in range(max(len(packet['docs']) for packet in active)):
                for packet in active:
                    if rank < len(packet['docs']) and packet['docs'][rank]['doc_id'] not in seen:
                        seen.add(packet['docs'][rank]['doc_id'])
                        expected_interleave.append(packet['docs'][rank]['doc_id'])
            expected_discount, scores = _expected_discount(original, rewritten)
            for arm in ['union20', 'discount20']:
                merged = pipeline.merge(original, rewritten, arm=arm)
                ids = [doc['doc_id'] for doc in merged['docs']]
                assert len(ids) == len(set(ids)) == len(raw_ids) <= 20
                assert set(ids) == raw_ids
                if len(active) == 1:
                    assert ids == original_ids
                else:
                    assert ids == (expected_interleave if arm == 'union20' else expected_discount)
                assert [doc['rank'] for doc in merged['docs']] == list(range(1, len(ids) + 1))
                if arm == 'union20':
                    assert merged['docs'][:10] == node['reader_packet']['docs'], 'union20 must preserve the old first10 exactly'
                if len(active) == 2:
                    fusion = {item['doc_id']: item for item in merged['fusion']}
                    assert set(fusion) == raw_ids
                    for rank, did in enumerate(ids, 1):
                        item = fusion[did]
                        expected_ranks = {('original' if index == 0 else 'rewritten'): value
                                          for index, value in scores[did]['ranks'].items()}
                        assert item['rank'] == rank and item['route_ranks'] == expected_ranks
                        assert abs(item['discount_score'] - scores[did]['score']) <= 1e-15
                originals = {doc['doc_id']: doc for packet in active for doc in packet['docs']}
                for doc in merged['docs']:
                    source = originals[doc['doc_id']]
                    assert doc['text'] == source['text'] and doc['source'] == source['source']
                    assert doc['block'] == f'[Chunk {doc["rank"]} | source={doc["source"]}, pages=1-1]\n{doc["text"]}'
                prompt = b0.legacy_child.child(case['question'], node['step_id'], node['request'],
                            deepcopy(node['prior_complete_answers']), b0.passages(merged), examples,
                            role=node['role'], combine=False)
                if node['role'] == 'goal':
                    prompt = goal_prompt(prompt, 'p02_role_headers')
                assert all(doc['block'] in prompt for doc in merged['docs'])
                assert all(doc['text'] in doc['block'] for doc in merged['docs'])
                totals[arm + '_nodes'] += 1
                totals[arm + '_reader_docs'] += len(ids)
                totals[arm + '_old_reader_docs'] += len(node['reader_packet']['docs'])
                case_detail['fixed_packet_diagnostics'].append({
                    'step_id': node['step_id'], 'role': node['role'], 'arm': arm,
                    'original_count': len(original['docs']), 'rewritten_count': len(rewritten['docs']) if rewritten else 0,
                    'merged_count': len(ids), 'doc_ids': ids,
                    'all_blocks_present': True, 'diagnostic_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'frozen_history_used_for_input_diagnostic_only': True,
                    'new_prediction_generated': False, 'F1_evaluated': False})
        details.append(case_detail)
    result = {'passed': True, 'scope': 'Network-free exact timing-only receipt replay plus fixed-packet reader-input diagnostics',
              'source_runtime_sha256': sha(OLD_E / 'runtime300.frozen.jsonl'),
              'source_manifest_sha256': sha(OLD_E / 'SOURCES.frozen.jsonl'),
              'pipeline_sha256': sha(Path(pipeline.__file__)), 'audit_code_sha256': sha(Path(__file__)),
              'counts': dict(totals), 'by_dataset': dict(Counter(case['dataset'] for case in runtime)),
              'new_api_calls': 0, 'new_index_searches': 0,
              'timing_only_outputs_equal_frozen_p02': True,
              'changed_prompt_predictions_not_replayed_or_scored': True,
              'candidate_set_integrity_and_all_reader_blocks_verified': True,
              'semantic_quality_or_new_final_F1_claimed': False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    output.with_suffix('.details.jsonl').write_text(
        ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in details), encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Explicit network-free validation; default behavior also never calls network.')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, default=OUT / 'OFFLINE_AUDIT.json')
    args = parser.parse_args()
    print(json.dumps(audit(output=args.output, limit=args.limit), ensure_ascii=False))


if __name__ == '__main__':
    main()
