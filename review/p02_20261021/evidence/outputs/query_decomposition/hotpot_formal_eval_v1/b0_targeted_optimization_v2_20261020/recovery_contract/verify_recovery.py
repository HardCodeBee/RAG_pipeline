"""Offline fault injection only; never creates a provider client or real Budget."""
import json
from pathlib import Path
import socket
import sqlite3
import sys

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from experiments.query_decomposition.b0_targeted_optimization_v2_20261020 import experiment as e

HERE = Path(__file__).resolve().parent
FROZEN = [e.HERE / 'experiment.py', e.HERE / 'PROTOCOL.frozen.json']
PINNED = {str(p): e.sha(p) for p in FROZEN}
real_out = e.OUT
plan = e.read(e.HERE / 'PROTOCOL.frozen.json')
case = e.rows(e.HERE / 'runtime300.frozen.jsonl')[0]
manifest = {r['id']: r for r in e.rows(e.HERE / 'SOURCES.frozen.jsonl')}[case['id']]
submissions = []
scenario = None
final_counts = {}

def forbidden_network(*args, **kwargs):
    raise AssertionError('Network forbidden during offline recovery contract')

socket.create_connection = forbidden_network

class FakeBudget:
    def __init__(self):
        self.db = sqlite3.connect(':memory:')
    def reserve(self, *args, **kwargs):
        raise AssertionError('No budget reservation allowed')

class FakeOneShot:
    def __init__(self, store, budget, client, model, qid):
        self.store, self.model = store, model
    def ask(self, stage, prompt, maximum, **kwargs):
        req = e.payload(self.model, prompt, maximum)
        rid = e.digest([self.store.namespace, req])
        submissions.append({'scenario': scenario, 'stage': stage, 'namespace': self.store.namespace, 'id': rid})
        status = 'success'
        answer = 'The independent contract produced a complete candidate answer.'
        if stage == 'goal_consistency/goal' and scenario == 'with_formatter':
            answer = 'Alpha'
        elif stage == 'goal_consistency/sentence_format':
            answer = 'The supported candidate answer is Alpha.'
        elif stage == 'targeted/final':
            final_counts[scenario] = final_counts.get(scenario, 0) + 1
            if final_counts[scenario] == 1:
                status, answer = 'unknown', None
            else:
                answer = 'Synthetic final answer'
        result = {'status': status, 'answer': answer, 'finish_reason': 'stop' if status == 'success' else None,
                  'usage': {}, 'http_status': None, 'contract_fake': True}
        self.store.db.execute('INSERT INTO requests VALUES(?,?,?,?,?)',
                              (rid, stage, json.dumps(req), status, json.dumps(result)))
        self.store.db.commit()

e.Budget = FakeBudget
e.OneShot = FakeOneShot
checks = []
for scenario in ['no_formatter', 'with_formatter']:
    e.OUT = HERE / scenario
    assert not (e.OUT / 'rounds').exists(), 'Offline contract has already run; preserve its artifacts'
    before = len(submissions)
    results = []
    for number in [1, 2]:
        round_root = e.OUT / 'rounds' / f'r{number:03d}'
        owner = e.Store(round_root / 'methods' / case['dataset'] / 'g01_goal_consistency', {'synthetic_contract': True})
        e.ERRORS = 0
        try:
            result = e.run_case(case, 'g01_goal_consistency', owner, object(), plan, manifest)
        finally:
            owner.db.close()
        path = round_root / 'cases' / case['dataset'] / 'g01_goal_consistency' / (case['qid'] + '.json')
        assert path.exists(), 'Round output path must match chosen() convention'
        results.append(e.read(path))
    first, second = results
    assert first['status'] == 'incomplete' and first['retryable_infrastructure']
    assert second['status'] == 'success'
    assert second['prediction'] == 'Synthetic final answer'
    actual = submissions[before:]
    expected = ['goal_consistency/goal']
    if scenario == 'with_formatter':
        expected.append('goal_consistency/sentence_format')
    expected.extend(['targeted/final', 'targeted/final'])
    assert [s['stage'] for s in actual] == expected
    prefix = [c for c in second['logical_calls'] if c['stage'] != 'targeted/final']
    assert all(c['reused'] for c in prefix)
    for old, new in zip(first['logical_calls'][:-1], prefix):
        assert old['request_id'] == new['request_id'] and old['source'] == new['source']
    assert not second['logical_calls'][-1]['reused']
    assert e.chosen('g01_goal_consistency', case['id'])['status'] == 'success'
    checks.append({'scenario': scenario, 'passed': True, 'simulated_submissions': len(actual),
                   'second_round_submissions': 1, 'reused_prefix_calls': len(prefix)})

e.OUT = real_out
assert {str(p): e.sha(p) for p in FROZEN} == PINNED
report = {'passed': True, 'checks': checks, 'real_api_submissions': 0, 'real_budget_reservations': 0,
          'synthetic_artifacts_only': True, 'v2_source_and_protocol_unchanged': True,
          'pins': PINNED, 'utc': e.utc()}
(HERE / 'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps(report, ensure_ascii=False))
