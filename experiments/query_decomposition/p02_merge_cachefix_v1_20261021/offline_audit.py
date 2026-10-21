"""Independent canonical timing-only replay with exact SQLite payload checks.

No provider or retrieval index is opened. Union changes are not assigned old
downstream predictions; actual candidate execution belongs to the new runner.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt
from experiments.query_decomposition.p02_flow_simplify_v1_20261020 import pipeline

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
OUT = OUTPUT / HERE.name
OLD_E = HERE.parent / 'p02_flow_simplify_v1_20261020'
OLD_O = OUTPUT / OLD_E.name
CANONICAL_DETAILS = OLD_O / 'OFFLINE_AUDIT.details.jsonl'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def stored_receipt(source, request_id):
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=20) as db:
        row = db.execute('SELECT payload,status,result FROM requests WHERE id=?', (request_id,)).fetchone()
    assert row, (source, request_id)
    return {'payload': json.loads(row[0]), 'status': row[1], 'result': json.loads(row[2]) if row[2] else None}


def recorded_packets(old):
    root = old.get('original_retrieval')
    if root is None:
        root = read(ROOT / old['source_prefix'])['original_retrieval']
    packets = [root]
    for node in old['nodes']:
        packets += [packet for packet in [node['original_retrieval'], node.get('rewritten_retrieval')] if packet]
    result = {}
    for packet in packets:
        if packet['query'] in result:
            assert result[packet['query']] == packet
        result[packet['query']] = packet
    return result, root


def audit(*, output=OUT / 'OFFLINE_AUDIT.json'):
    from . import cache
    runtime_path = HERE / 'runtime300.frozen.jsonl'
    if not runtime_path.exists():
        runtime_path = OLD_E / 'runtime300.frozen.jsonl'
    samples = rows(runtime_path)
    assert len(samples) == len({case['id'] for case in samples}) == 300
    assert Counter(case['dataset'] for case in samples) == {'hotpotqa': 100, '2wiki': 100, 'musique': 100}
    canonical_rows = {item['id']: item for item in rows(CANONICAL_DETAILS)}
    assert set(canonical_rows) == {case['id'] for case in samples}
    plan = read(OLD_E / 'PROTOCOL.frozen.json')
    catalogs, cache_stats = {}, {}
    for ds in ['hotpotqa', '2wiki', 'musique']:
        catalogs[ds], cache_stats[ds] = cache.catalog_with_audit(ds, include_current=False)
    counts, details = Counter(), []
    for case in samples:
        expected = canonical_rows[case['id']]
        old = read(ROOT / expected['p02_source'])
        packets, root = recorded_packets(old)
        actual_calls, actual_queries = [], []

        def retrieve(query):
            assert query in packets, 'Canonical baseline generated an unrecorded query'
            actual_queries.append(query)
            return deepcopy(packets[query])

        def ask(stage, prompt, maximum, allow_final_truncation=False):
            request = payload(plan['model'], prompt, maximum)
            key = digest(request)
            index = len(actual_calls)
            wanted = expected['timing_only_replay_calls'][index]
            assert stage == wanted['stage'] and key == wanted['payload_sha256']
            assert key in catalogs[case['dataset']], 'Missing canonical successful input'
            entry = catalogs[case['dataset']][key]
            assert entry['request_id'] == wanted['request_id'], 'Canonical receipt priority changed'
            assert Path(entry['source']).resolve() == Path(wanted['source']).resolve(), 'Canonical source priority changed'
            sql = stored_receipt(entry['source'], entry['request_id'])
            assert sql['payload'] == request == entry['payload']
            assert sql['status'] == entry['status'] == 'success'
            assert sql['result'] == entry['result']
            actual_calls.append({'stage': stage, 'payload_sha256': key,
                                 'request_id': entry['request_id'], 'source': entry['source']})
            return use_receipt({**entry, **sql}, stage, allow_final_truncation)

        replay = pipeline.execute(case, retrieve, ask, arm='timing_only', readout=plan['readout'])
        assert len(actual_calls) == len(expected['timing_only_replay_calls'])
        for actual, wanted in zip(actual_calls, expected['timing_only_replay_calls']):
            assert {k: actual[k] for k in ['stage', 'payload_sha256', 'request_id']} == {
                k: wanted[k] for k in ['stage', 'payload_sha256', 'request_id']}
            assert Path(actual['source']).resolve() == Path(wanted['source']).resolve()
        for key in ['prediction', 'final_prompt', 'final_context', 'finish_reason']:
            assert replay[key] == old[key], (case['id'], key)
        assert len(replay['nodes']) == len(old['nodes'])
        for node, previous in zip(replay['nodes'], old['nodes']):
            for key, value in previous.items():
                assert node.get(key) == value, (case['id'], node['step_id'], key)
        assert replay['original_retrieval'] == root
        counts.update(cases=1, nodes=len(replay['nodes']), canonical_full_payload_calls=len(actual_calls),
                      logical_retrievals=len(actual_queries))
        details.append({'id': case['id'], 'dataset': case['dataset'],
                        'source_p02': expected['p02_source'], 'source_p02_sha256': sha(ROOT / expected['p02_source']),
                        'canonical_calls': actual_calls, 'all_prompts_outputs_packets_exact': True})
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'scope': 'Independent all300 timing_only replay from priority-selected canonical successful full payloads',
              'counts': dict(counts), 'cache_catalog_stats': cache_stats,
              'canonical_request_id_source_and_response_priority_verified': True,
              'timing_only_all300_equal_frozen_p02': True,
              'candidate_union_F1_not_fabricated_from_fixed_old_answers': True,
              'new_API_calls': 0, 'new_index_searches': 0,
              'source_pins': {str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__), Path(cache.__file__),
                          Path(pipeline.__file__), runtime_path, CANONICAL_DETAILS]}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    output.with_suffix('.details.jsonl').write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in details), encoding='utf8')
    print(json.dumps({key: result[key] for key in ['passed', 'counts', 'new_API_calls', 'new_index_searches']}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT / 'OFFLINE_AUDIT.json')
    args = parser.parse_args()
    audit(output=args.output)
