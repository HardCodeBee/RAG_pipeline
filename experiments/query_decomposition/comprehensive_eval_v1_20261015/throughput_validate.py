"""Replay real saved calls; verify scheduling parity without paid requests."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib
import heapq
import json
import sqlite3
import time

from .common import ROOT, HERE, RUN, OUTPUT, read, rows, save, sha, log
from .campaign import TARGET, OUR
from .throughput_epoch import EXECUTE
from experiments.query_decomposition.hotpot_formal_eval_v1 import parallel_execution as old


class Store:
    def __init__(self, initial=None):
        self.data = deepcopy(initial or {})
    def get(self, kind, key):
        return self.data.get((kind, key))
    def put(self, kind, key, value):
        if (kind, key) in self.data and self.data[(kind, key)] != value:
            raise AssertionError('Immutable replay state changed')
        self.data[kind, key] = deepcopy(value)
    def event(self, *args, **kwargs): pass
    def export(self, *args, **kwargs): pass


class Retrieval:
    def __init__(self, records):
        self.records = records
        self.groups = []
    def retrieve_many(self, cases, k):
        self.groups.append([(r['qid'], r['query'], k) for r in cases])
        return [deepcopy(self.records[r['qid'], r['query'], k]) for r in cases]
    def retrieve(self, qid, query, k):
        return self.retrieve_many([{'qid': qid, 'query': query}], k)[0]


class API:
    def __init__(self, requests):
        self.requests = requests
        self.calls = []
        self.concurrency = 3
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.simulated_seconds = 0.
        self.next_start = 0.
        self.simulated_payloads = set()
    def ask_one(self, operation, questions):
        stage, prompt, tokens = operation.args
        key = (prompt, tokens)
        if key not in self.requests:
            raise AssertionError('Generated SDK input differs from saved actual call: ' + stage)
        rid, raw = self.requests[key]
        time.sleep(min(raw.get('latency_seconds', 0), 90) / 1000)
        return rid, deepcopy(raw)
    def ask_group(self, operations, questions):
        self.calls.extend((questions[op.index]['qid'], *op.args) for op in operations)
        available = [self.simulated_seconds] * 3
        for op in operations:
            stage, prompt, tokens = op.args
            key = (prompt, tokens)
            if key in self.simulated_payloads:
                continue
            self.simulated_payloads.add(key)
            start = max(heapq.heappop(available), self.next_start)
            self.next_start = start + 1.
            heapq.heappush(available, start + self.requests[key][1].get('latency_seconds', 0))
        self.simulated_seconds = max(available)
        futures = [self.pool.submit(self.ask_one, op, questions) for op in operations]
        return [f.result() for f in futures]


def run_one(directory, module_name, cases, config, initial=None):
    with sqlite3.connect((directory / 'checkpoint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        packets = {}
        for raw, in db.execute("SELECT value FROM data WHERE kind='retrieval'"):
            r = json.loads(raw)
            packets[r['qid'], r['query'], r['k']] = r
        requests = {}
        for rid, stage, payload, raw in db.execute('SELECT id,stage,payload,result FROM requests WHERE result IS NOT NULL'):
            payload = json.loads(payload)
            # Runtime request identity hashes the SDK payload, not the stage.
            key = (payload['messages'][0]['content'], payload['max_tokens'])
            requests[key] = (rid, json.loads(raw))
        reference = {key: json.loads(raw) for key, raw in db.execute("SELECT key,value FROM data WHERE kind='prediction'")}
    module = importlib.import_module(module_name)
    outcomes = []
    for execute in [old._execute, EXECUTE]:
        store, retrieval, api = Store(initial), Retrieval(packets), API(requests)
        started = time.perf_counter()
        try:
            execute(module, cases, retrieval, object(), store, config, api, 20)
        finally:
            api.pool.shutdown(wait=True)
        elapsed = time.perf_counter() - started
        calls = {c['qid']: [x[1:] for x in api.calls if x[0] == c['qid']] for c in cases}
        predictions = {c['qid']: store.get('prediction', c['qid']) for c in cases}
        for qid, value in predictions.items():
            assert value == reference[qid], 'Prediction differs from saved original run'
        outcomes.append((calls, predictions, retrieval.groups, elapsed, api.simulated_seconds))
    assert outcomes[0][:2] == outcomes[1][:2], 'Per-question SDK prompts/results changed'
    return {'cases': len(cases), 'sdk_calls_matched': sum(len(v) for v in outcomes[0][0].values()),
            'predictions_matched': len(cases), 'retrieval_grouping_identical': outcomes[0][2] == outcomes[1][2],
            'old_replay_seconds': outcomes[0][3], 'slot_replay_seconds': outcomes[1][3],
            'old_api_schedule_simulated_seconds': outcomes[0][4],
            'slots_api_schedule_simulated_seconds': outcomes[1][4],
            'simulation_policy': 'saved response latencies; three slots; starts >=1 second apart; no retrieval/retry overhead'}


def main():
    receipt = read(TARGET / 'CAMPAIGN_FROZEN.json')
    settings = read(TARGET / 'hotpotqa/BASELINE_CONFIGS_FROZEN.json')['selected']
    questions = {x['qid']: x for x in rows(HERE / 'inputs_frozen/hotpotqa/formal.questions.jsonl')}
    results = {}
    for mid in ['b00_rag_top10_v1', 'b01_rag_top10_cot3', 'b02_selfask_dense',
                'b03_react_dense', 'b04_ircot_bge', 'b05_iter_retgen_bge', 'b06_decomp_bge']:
        directory = TARGET / 'hotpotqa/formal' / mid / settings[mid]['id']
        with sqlite3.connect((directory / 'checkpoint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
            preds = [json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='prediction'")]
        selected = [questions[r['qid']] for r in preds if r['status'] in {'success', 'algorithm_failure'}][:20]
        results[mid] = run_one(directory, receipt['protocol']['methods'][mid]['module'], selected, settings[mid])
        print(json.dumps({'method': mid, **results[mid]}), flush=True)
    # The method under study also keeps the frozen child/goal/final interfaces.
    source = OUTPUT / 's02_child4_compact_formal200_20261014/formal' / OUR / 'frozen_s02_child4'
    legacy = {r['qid']: r for r in rows(ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/compact_100_200_20261014/inputs_frozen/formal.questions.jsonl')}
    with sqlite3.connect((source / 'checkpoint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        initial = {(kind, key): json.loads(raw) for kind, key, raw in db.execute("SELECT kind,key,value FROM data WHERE kind='decomposition'")}
        predictions = [json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='prediction'")]
    selected = [legacy[r['qid']] for r in predictions if r['status'] == 'success'][:20]
    config = receipt['protocol']['methods'][OUR]['default'].copy()
    config['helper_positions'] = read(OUTPUT / 's02_child4_compact_formal200_20261014/HELPER_POSITIONS_FROZEN.json')['positions']
    results[OUR] = run_one(source, receipt['protocol']['methods'][OUR]['module'], selected, config, initial)
    result = {'passed': True, 'api_submissions': 0, 'execution_source_sha256': sha(HERE / 'throughput_epoch.py'),
              'validation_source_sha256': sha(__file__), 'methods': results,
              'cases': sum(r['cases'] for r in results.values()),
              'sdk_calls_byte_matched': sum(r['sdk_calls_matched'] for r in results.values()),
              'metric_or_gold_labels_used_for_optimization': False}
    save(RUN / 'THROUGHPUT_OFFLINE_PARITY.json', result)
    log('throughput_offline_actual_call_parity_passed', **result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
