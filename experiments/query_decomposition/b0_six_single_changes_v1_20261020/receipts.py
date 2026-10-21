"""Read-only source receipts and exact, baseline-lineage cache priority."""
import json
import sqlite3
from .prepare import ROOT, HERE, OUT, rows, read, sha
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt


def ro(path):
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)


def receipt(record, expected=None):
    with ro(ROOT / record['source']) as db:
        row = db.execute('SELECT payload,status,result FROM requests WHERE id=?', (record['request_id'],)).fetchone()
    assert row, record
    request = json.loads(row[0])
    if expected is not None: assert request == expected, 'Exact payload mismatch'
    return {**record, 'status': row[1], 'result': json.loads(row[2]) if row[2] else None}


def sources():
    return {r['id']: r for r in rows(HERE / 'B0_SOURCES.frozen.jsonl')}


def baseline_case(source):
    path = ROOT / source['source']
    assert sha(path) == source['source_sha256']
    return read(path)


def baseline_packets(value):
    packets = [value['original_retrieval']]
    for n in value['nodes']:
        packets.extend(p for p in [n['original_retrieval'], n.get('rewritten_retrieval')] if p)
    result = {}
    for p in packets:
        query = p['query']
        if query in result: assert result[query] == p
        result[query] = p
    return result


def baseline_calls(source):
    result = {}
    for r in source['logical_calls']:
        k = r['payload_sha256']
        if k in result: assert result[k]['request_id'] == r['request_id']
        result[k] = r
    return result


def replay_callbacks(source, model):
    b = baseline_case(source); packets = baseline_packets(b); calls = baseline_calls(source)
    trace = {'asks': [], 'retrievals': []}
    def retrieve(query):
        if query not in packets: raise AssertionError('Uncached baseline search: ' + query)
        p = packets[query]
        trace['retrievals'].append([query, [d['doc_id'] for d in p['docs']]])
        return p
    def ask(stage, prompt, maximum, allow_final_truncation=False):
        request = payload(model, prompt, maximum); key = digest(request)
        if key not in calls: raise AssertionError('Uncached baseline API payload: ' + stage)
        r = receipt(calls[key], request)
        trace['asks'].append([stage, key, r['request_id']])
        return use_receipt(r, stage, allow_final_truncation)
    return b, retrieve, ask, trace
