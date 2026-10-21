"""Safe successor to old300 selection and the already-running dev90 parent.

--self-test has no provider/SSH calls and does not freeze a live candidate.
Only --run waits for final old300 selection, freezes that exact candidate,
finishes parent/candidate dev90 through the unchanged budgeted runner, and
scores once. A passed gate ends at awaiting_confirmation_remote; this program
never opens confirmation inputs or launches their decomposition/generation.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, digest, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json

HERE = Path(__file__).resolve().parent
OLD = OUTPUT / HERE.name
OUT = OLD / 'fresh_dev90'
STATUS = OUT / 'CONTROLLER_STATUS.json'
EVENTS = OUT / 'CONTROLLER_EVENTS.jsonl'
ARMS = ['g02_evidence_goal', 'p02_role_headers', 's03_g01_plus_f01']


class StopController(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def select_final(selection, results, terminal):
    """Final immutable winner only, never the current leading partial arm."""
    winner = selection.get('selected')
    if winner not in ['parent_g01', *ARMS] or terminal.get('selected') != winner:
        raise StopController('Old300 terminal/selection identities disagree')
    parent = results.get('parent300_F1')
    if not finite(parent):
        raise StopController('Old300 parent full300 score missing')
    eligible = [(parent, 'parent_g01')]
    for arm in ARMS:
        item = results.get('arms', {}).get(arm, {})
        if item.get('scorable') == 300:
            if not finite(item.get('F1')):
                raise StopController('Invalid complete old300 score: ' + arm)
            eligible.append((item['F1'], arm))
    score, expected = max(eligible, key=lambda item: item[0])
    if expected != winner or not finite(selection.get('F1')) or not math.isclose(
            score, selection['F1'], rel_tol=0, abs_tol=1e-12):
        raise StopController('Frozen old300 winner does not match complete300 selection policy')
    if winner != 'parent_g01' and score <= parent:
        raise StopController('Selected candidate has no strict old300 gain')
    return {'selected': winner, 'old300_F1': score, 'parent300_F1': parent,
            'delta_F1': score - parent, 'complete300': True}


def candidate_contract(winner, selection_hash, source_pins):
    return {'selected': winner, 'old300_selection_sha256': selection_hash,
            'source_pins': source_pins, 'frozen_before_dev_candidate_dispatch': True}


def validate_candidate(existing, expected):
    for key in ['selected', 'old300_selection_sha256', 'frozen_before_dev_candidate_dispatch']:
        if existing.get(key) != expected[key]:
            raise StopController('Existing candidate freeze disagrees: ' + key)
    if not isinstance(existing.get('source_pins'), dict):
        raise StopController('Existing candidate has no source pins')
    for path, value in expected['source_pins'].items():
        if existing['source_pins'].get(path) != value:
            raise StopController('Existing candidate source pin disagrees: ' + path)
    for path, value in existing['source_pins'].items():
        if sha(ROOT / path) != value:
            raise StopController('Frozen candidate source changed: ' + path)


class Recorder:
    def __init__(self):
        self.last_by_state = {}

    def emit(self, state, **fields):
        payload = {'state': state, **fields}
        fingerprint = digest(payload)
        if fingerprint == self.last_by_state.get(state):
            return
        self.last_by_state[state] = fingerprint
        record = {'utc': utc(), **payload}
        OUT.mkdir(parents=True, exist_ok=True)
        with EVENTS.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        locked_json(STATUS, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)


@contextmanager
def single_controller():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'CONTROLLER.lock'
    try:
        stream = path.open('x', encoding='utf-8')
    except FileExistsError:
        raise StopController('Controller lock exists; inspect its process/state instead of launching another controller')
    try:
        stream.write(json.dumps({'pid': os.getpid(), 'created_utc': utc(), 'source_sha256': sha(__file__)}))
        stream.flush()
        yield
    finally:
        stream.close()
        path.unlink()


def wait_old(deadline, poll, recorder):
    paths = {name: OLD / filename for name, filename in [
        ('terminal', 'TERMINAL.json'), ('selection', 'SELECTION.frozen.json'),
        ('results', 'FINAL_RESULTS.frozen.json')]}
    while True:
        present = {name: path.exists() for name, path in paths.items()}
        if all(present.values()):
            values = {name: read(path) for name, path in paths.items()}
            choice = select_final(values['selection'], values['results'], values['terminal'])
            recorder.emit('old300_final_selection', **choice)
            return choice
        progress = read(OLD / 'PROGRESS.json') if (OLD / 'PROGRESS.json').exists() else {}
        recorder.emit('waiting_old300_terminal', files_present=present, progress=progress)
        if time.monotonic() >= deadline:
            raise StopController('Timed out waiting for old300 final terminal/selection/results; no candidate chosen')
        time.sleep(min(poll, max(0, deadline - time.monotonic())))


def freeze_candidate(choice, recorder):
    selected = choice['selected']
    path = OUT / 'CANDIDATE.frozen.json'
    pin_paths = [HERE / 'strategy.py', HERE / 'fresh_runner.py', HERE / 'PROTOCOL.frozen.json',
                 HERE / 'fresh_data.py', HERE / 'fresh_confirm.py', Path(__file__),
                 OUT / 'PROTOCOL.frozen.json', OLD / 'FINAL_RESULTS.frozen.json']
    expected = candidate_contract(selected, sha(OLD / 'SELECTION.frozen.json'),
                                  {rel(p): sha(p) for p in pin_paths})
    if path.exists():
        validate_candidate(read(path), expected)
        recorder.emit('candidate_freeze_reused', selected=selected, protocol_sha256=sha(path))
    else:
        dispatched = list((OUT / 'rounds').glob(f'r[0-9][0-9][0-9]/methods/*/{selected}'))
        if selected != 'parent_g01' and dispatched:
            raise StopController('Candidate namespace already exists before its freeze; inspect rather than backdate')
        locked_json(path, expected)
        recorder.emit('candidate_frozen_before_dispatch', selected=selected,
                      protocol_sha256=sha(path), source_pins=expected['source_pins'])
    return path


def arm_snapshot(runner, cases, arm):
    counts = Counter()
    missing = []
    terminal = 0
    for case in cases:
        value = runner.selected(case, arm)
        status = value.get('status', 'unstarted') if value else 'unstarted'
        counts[status] += 1
        if value and isinstance(value.get('prediction'), str) and (
                status.startswith('success') or status == 'algorithm_failure'):
            if any(value.get(k) != case[k] for k in ['id', 'qid', 'dataset', 'question']):
                raise StopController('Completed case identity mismatch: ' + case['id'])
            terminal += 1
        else:
            missing.append({'id': case['id'], 'status': status,
                            'eligible_infrastructure_or_unstarted': runner.eligible(case, arm),
                            'reason': value.get('reason') if value else None})
    return {'arm': arm, 'planned': len(cases), 'scorable': terminal,
            'statuses': dict(counts), 'missing': missing}


def phase(runner, arm, number):
    root = runner.OUT / 'rounds' / f'r{number:03d}'
    directory = root / 'phases' / arm
    namespaces = list((root / 'methods').glob(f'*/{arm}'))
    return {'exists': directory.exists(), 'terminal': (directory / 'TERMINAL.json').exists(),
            'namespace_count': len(namespaces)}


def ensure_arm(runner, plan, cases, arm, deadline, poll, recorder, *, allow_initial):
    """No reopening unknown namespaces; exactly one infrastructure recovery."""
    while True:
        snap = arm_snapshot(runner, cases, arm)
        p0, p1 = phase(runner, arm, 0), phase(runner, arm, 1)
        recorder.emit('dev90_arm_progress', snapshot=snap, r000=p0, r001=p1)
        if snap['scorable'] == len(cases):
            return True, snap
        permanent = [x for x in snap['missing'] if not x['eligible_infrastructure_or_unstarted']]
        if permanent:
            return False, {'reason': 'Permanent/nonretryable missing cases; no payload or method changes',
                           **snap, 'permanent': permanent}
        if p1['terminal']:
            return False, {'reason': 'One immutable infrastructure recovery exhausted', **snap}
        if p1['exists'] or p1['namespace_count']:
            # Could be a live or interrupted process. Never guess that its
            # submissions are absent, and never reopen its owner stores.
            state = 'waiting_existing_r001_phase'
        elif p0['terminal']:
            recorder.emit('dispatch_infrastructure_recovery', arm=arm, round=1,
                          missing_ids=[x['id'] for x in snap['missing']], budget=runner.budget_snapshot())
            runner.run(plan, [arm], round_id=1)
            continue
        elif p0['exists'] or p0['namespace_count']:
            state = 'waiting_existing_r000_phase'
        elif allow_initial:
            recorder.emit('dispatch_frozen_candidate', arm=arm, round=0,
                          cases=len(cases), budget=runner.budget_snapshot())
            runner.run(plan, [arm], round_id=0)
            continue
        else:
            return False, {'reason': 'Expected externally dispatched parent namespace is absent; no blind initial replay', **snap}
        recorder.emit(state, arm=arm, scorable=snap['scorable'], missing=len(snap['missing']))
        if time.monotonic() >= deadline:
            return False, {'reason': 'Timed out awaiting existing phase; namespace requires inspection, not replay', **snap}
        time.sleep(min(poll, max(0, deadline - time.monotonic())))


def score_and_record(selected, recorder):
    from . import fresh_data
    report = fresh_data.score(selected)
    paired = report['paired']
    state = ('awaiting_confirmation_remote' if report['gate']['passed'] else
             'dev90_gate_failed' if paired['n_matched'] == 90 else 'dev90_incomplete_no_confirmation')
    gate_status = {'state': state, 'candidate': selected, 'passed': report['gate']['passed'],
                   'n_complete': paired['n_matched'], 'delta_F1': paired['delta_F1'],
                   'not_for_tuning': True, 'confirmation_started': False,
                   'candidate_protocol_sha256': sha(OUT / 'CANDIDATE.frozen.json')}
    locked_json(OUT / 'CONTROLLER_GATE_STATUS.json', gate_status)
    recorder.emit(state, candidate=selected, parent=report['parent'],
                  candidate_summary=report['candidate_summary'], paired=paired,
                  gate=report['gate'], confirmation_started=False)
    return gate_status


def run(wait_seconds=3600, poll_seconds=20):
    from . import fresh_runner as runner
    recorder = Recorder()
    deadline = time.monotonic() + wait_seconds
    with single_controller():
        recorder.emit('controller_started', source_sha256=sha(__file__),
                      wait_seconds=wait_seconds, poll_seconds=poll_seconds,
                      confirmation_will_not_be_started=True)
        try:
            choice = wait_old(deadline, poll_seconds, recorder)
            if choice['selected'] == 'parent_g01':
                recorder.emit('retained_parent_no_new_candidate', **choice, confirmation_started=False)
                return {'state': 'retained_parent_no_new_candidate', **choice}
            plan = runner.prepare()
            runner.verify(plan)
            freeze_candidate(choice, recorder)
            cases = runner.runtime_cases()
            ok, snapshot = ensure_arm(runner, plan, cases, 'parent_g01', deadline,
                                      poll_seconds, recorder, allow_initial=False)
            if not ok:
                recorder.emit('stopped_parent_incomplete', snapshot=snapshot, confirmation_started=False)
                result = score_and_record(choice['selected'], recorder)
                return {**result, 'stop_reason': snapshot['reason']}
            ok, snapshot = ensure_arm(runner, plan, cases, choice['selected'], deadline,
                                      poll_seconds, recorder, allow_initial=True)
            if not ok:
                recorder.emit('stopped_candidate_incomplete', snapshot=snapshot, confirmation_started=False)
            result = score_and_record(choice['selected'], recorder)
            if not ok:
                result['stop_reason'] = snapshot['reason']
            return result
        except Exception as exc:
            recorder.emit('stopped_needs_inspection', error_type=type(exc).__name__,
                          reason=str(exc), confirmation_started=False)
            raise


def self_test():
    # Deliberately synthetic: no live freeze, sealed inputs, provider, SSH,
    # scoring, references, retrieval or scheduler calls.
    parent = .69
    results = {'parent300_F1': parent, 'arms': {
        'g02_evidence_goal': {'scorable': 100, 'F1': .99},
        'p02_role_headers': {'scorable': 300, 'F1': .71},
        's03_g01_plus_f01': {'scorable': 299, 'F1': .72}}}
    selection = {'selected': 'p02_role_headers', 'F1': .71}
    terminal = {'selected': 'p02_role_headers'}
    choice = select_final(selection, results, terminal)
    assert choice['selected'] == 'p02_role_headers'
    rejected = []
    for name, change in [
        ('partial_leader_not_selected', {'selection': {'selected': 'g02_evidence_goal', 'F1': .99},
                                        'terminal': {'selected': 'g02_evidence_goal'}}),
        ('terminal_mismatch', {'terminal': {'selected': 's03_g01_plus_f01'}}),
        ('wrong_score', {'selection': {'selected': 'p02_role_headers', 'F1': .72}}),
    ]:
        try:
            select_final(change.get('selection', selection), results, change.get('terminal', terminal))
        except StopController:
            rejected.append(name)
        else:
            raise AssertionError('Invalid selection accepted: ' + name)
    tied = deepcopy(results)
    tied['arms']['p02_role_headers']['F1'] = parent
    assert select_final({'selected': 'parent_g01', 'F1': parent}, tied,
                        {'selected': 'parent_g01'})['selected'] == 'parent_g01'
    protocol = candidate_contract('p02_role_headers', 'old-selection-sha', {'synthetic': 'sha'})
    assert protocol['frozen_before_dev_candidate_dispatch'] is True
    assert protocol['selected'] == choice['selected']
    return {'passed': True, 'rejects': rejected, 'tie_retains_parent': True,
            'partial_higher_scores_not_eligible': True, 'candidate_contract_schema': sorted(protocol),
            'current_api_calls': 0, 'current_remote_calls': 0, 'sealed_queries_opened': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    parser.add_argument('--wait-seconds', type=int, default=3600)
    parser.add_argument('--poll-seconds', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 3600 or not 1 <= args.poll_seconds <= 60:
        parser.error('Wait must be1..3600s; poll must be1..60s')
    result = self_test() if args.self_test else run(args.wait_seconds, args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
