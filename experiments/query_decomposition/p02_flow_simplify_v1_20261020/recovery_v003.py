"""Explicit infrastructure-only r002..r004 recovery for the original300.

Never edits the frozen v001/v002 or downstream fresh evaluator. Only --run
dispatches. Dataset-isolated v1.run calls reset the existing circuit and TLS
preflight between datasets; generation payloads/methods/resources stay exact.
The old300 continuation and stopped fresh watcher must both be audited first.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import sqlite3

from . import run as v1
from . import runner_v002 as v2
from . import score
from . import fresh_evaluation as fresh
from .cache_catalog_v002 import augment


HERE = Path(__file__).resolve().parent
OUT = v1.OUT
META = OUT / 'infra_recovery_v003'
PROTOCOL = HERE / 'INFRA_RECOVERY_V003.frozen.json'
DATASETS = ('2wiki', 'musique', 'hotpotqa')
ROUNDS = (2, 3, 4)


def event(name, **fields):
    META.mkdir(parents=True, exist_ok=True)
    with (META / 'EVENTS.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps({'utc': v1.utc(), 'event': name, **fields}, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare():
    parent, cache_revision = v2.prepare_revision()
    protocol = {
        'revision': 'infra_recovery_v003',
        'authorization': 'User explicitly asked to continue filling300, preserve methods and never rerun successful cases',
        'parent_protocol_sha256': v1.sha(HERE / 'PROTOCOL.frozen.json'),
        'cache_v002_protocol_sha256': v1.sha(HERE / 'INFRA_CACHE_V002.frozen.json'),
        'source_pins': {str(Path(path).relative_to(v1.ROOT)): v1.sha(path) for path in (
            Path(__file__), HERE / 'runner_v002.py', HERE / 'cache_catalog_v002.py')},
        'runtime_sha256': v1.sha(HERE / 'runtime300.frozen.jsonl'),
        'old_max_rounds': parent['max_rounds'], 'new_max_rounds': 5,
        'allowed_new_round_ids': list(ROUNDS), 'maximum_additional_rounds': 3,
        'method_changes': 0, 'generation_payload_changes': 0, 'retrieval_resource_changes': 0,
        'cache_policy': cache_revision['cache_priority'],
        'dataset_order': list(DATASETS),
        'dispatch_isolation': 'One unchanged v1.run invocation per dataset per round, sequential; each resets ERRORS and repeats TLSpreflight',
        'reason': 'The old circuit resets per dispatch, not per dataset. Earlier dataset unknowns can defer later datasets without any request submission.',
        'snapshot': 'Freeze ordered missing method-cases and source hashes before each round; dispatch must match this exact mask',
        'eligible': 'Only incomplete retryable_infrastructure cases or genuinely unstarted method-cases; terminal successes/algorithm failures excluded',
        'owner_policy': 'Never reopen an existing method owner, request namespace or execution phase',
        'gates': ['Old CONTINUATION_COMPLETE.json exists',
                  'Old watcher WAITING_FOR_RECOVERY.json and final waiting status exist',
                  'Old AUTHORIZED_FINISH_ACTIVE.lock absent', 'No fresh selection/generation already started'],
        'stop_policy': 'After two consecutive dataset attempts with eligible missing cases and zero new scorable cases, finish the current isolated round and stop further rounds for inspection',
        'historical_streak': 'A zero-gain r001 attempt may count as the first observation, only when actual r001 case files exist',
        'budgets': {'global_cap_cny': 300, 'dispatch_ceiling_cny': 299.9,
                    'shared_project_subcap_cny': 25, 'unknown_reservations_refunded': False},
        'fresh_dispatch': False,
    }
    v1.freeze(PROTOCOL, protocol)
    verify(protocol)
    return parent, protocol


def verify(protocol):
    parent = v1.prepare()
    assert v1.sha(HERE / 'PROTOCOL.frozen.json') == protocol['parent_protocol_sha256']
    assert v1.sha(HERE / 'INFRA_CACHE_V002.frozen.json') == protocol['cache_v002_protocol_sha256']
    assert v1.sha(HERE / 'runtime300.frozen.jsonl') == protocol['runtime_sha256']
    assert parent['max_rounds'] == protocol['old_max_rounds'] == 2
    for path, expected in protocol['source_pins'].items():
        assert v1.sha(v1.ROOT / path) == expected, path


def gate():
    required = [OUT / 'CONTINUATION_COMPLETE.json', OUT / 'WAITING_FOR_RECOVERY.json',
                OUT / 'FINAL_EXECUTION_STATUS.json']
    absent = [str(path.relative_to(OUT)) for path in required if not path.exists()]
    if absent:
        raise RuntimeError('Wait for both old controllers to stop: absent=' + ','.join(absent))
    if (OUT / 'AUTHORIZED_FINISH_ACTIVE.lock').exists():
        raise RuntimeError('Old finish watcher still active; do not race downstream selection')
    status = v1.read(OUT / 'FINAL_EXECUTION_STATUS.json')
    if status['status'] != 'waiting_for_recovery_inspection':
        raise RuntimeError('Old watcher is not in explicit waiting state')
    if fresh.CHOICE.exists() or any(fresh.OUT.glob('phases/*/DISPATCH.frozen.json')):
        raise RuntimeError('Fresh selection/dispatch already exists; inspect before reopening original300')


def provider_records(round_id, dataset):
    """Recorded request rows, separate from cases merely deferred by circuit."""
    counters, classes = Counter(), Counter()
    for path in sorted(OUT.glob(f'rounds/r{round_id:03d}/methods/{dataset}/*/checkpoint.sqlite3')):
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            for status, count in db.execute('SELECT status,count(*) FROM requests GROUP BY status'):
                counters[status] += count
            for status, failure, http, error, count in db.execute(
                "SELECT status,json_extract(result,'$.failure_class'),json_extract(result,'$.http_status'),"
                "json_extract(result,'$.error_code'),count(*) FROM requests "
                "WHERE status!='success' GROUP BY 1,2,3,4"):
                classes[json.dumps({'status': status, 'failure_class': failure,
                                    'http_status': http, 'error_code': error}, sort_keys=True)] += count
    return {'request_status_counts': dict(counters), 'failure_classes': dict(classes),
            'note': 'Request records are not a count of all missing cases or proof the provider processed a request'}


def inspect():
    """No reference answers: inspect actual terminality and retrieval-free logs."""
    gate()
    runtime = v1.rows(HERE / 'runtime300.frozen.jsonl')
    assert len(runtime) == len({row['id'] for row in runtime}) == 300
    missing, blocked, by_dataset = [], [], {}
    for dataset in DATASETS:
        counts, reasons = Counter(), Counter()
        for arm in v1.ARMS:
            for case in runtime:
                if case['dataset'] != dataset:
                    continue
                path, value = score.select_case(OUT, dataset, arm, case['qid'])
                if score.is_scorable(value):
                    counts['scorable'] += 1
                    counts[value['status']] += 1
                    continue
                counts['missing'] += 1
                reason = value.get('reason', '') if value else ''
                if 'Dispatch deferred' in reason:
                    counts['circuit_deferred_cases'] += 1
                reasons[(value or {}).get('error_type', 'unstarted')] += 1
                row = {'id': case['id'], 'qid': case['qid'], 'dataset': dataset, 'arm': arm,
                       'status': value.get('status', 'unstarted') if value else 'unstarted',
                       'retryable_infrastructure': bool(value and value.get('retryable_infrastructure', False)),
                       'error_type': (value or {}).get('error_type'), 'reason': reason,
                       'failed_stage': (value or {}).get('failed_stage'),
                       'request_status': (value or {}).get('request_status'),
                       'http_status': (value or {}).get('http_status'),
                       'source': str(path.relative_to(v1.ROOT)) if path else None,
                       'source_sha256': v1.sha(path) if path else None}
                if value is None or (value['status'] == 'incomplete' and row['retryable_infrastructure']):
                    missing.append(row)
                    counts['eligible_infrastructure'] += 1
                else:
                    blocked.append(row)
                    counts['nonretryable_missing'] += 1
        by_dataset[dataset] = {**dict(counts), 'missing_error_types': dict(reasons)}
    return {'planned': 900, 'scorable': sum(row.get('scorable', 0) for row in by_dataset.values()),
            'eligible_missing': missing, 'nonretryable_missing': blocked,
            'by_dataset': by_dataset, 'budget': v1.budget_snapshot()}


def _historical_streak(current):
    runtime = v1.rows(HERE / 'runtime300.frozen.jsonl')
    result = {}
    for dataset in DATASETS:
        previous_count = 0
        for case in runtime:
            if case['dataset'] == dataset:
                for arm in v1.ARMS:
                    _, value = score.select_case(OUT / 'rounds/r000', dataset, arm, case['qid'])
                    previous_count += score.is_scorable(value)
        attempted_r1 = any(OUT.glob(f'rounds/r001/cases/{dataset}/*/*.json'))
        current_count = current['by_dataset'][dataset].get('scorable', 0)
        eligible = current['by_dataset'][dataset].get('eligible_infrastructure', 0)
        result[dataset] = int(attempted_r1 and eligible > 0 and current_count == previous_count)
    return result


@contextmanager
def _runner_adapter(round_id, protocol):
    old_catalog, old_store = v1.source_catalog, v1.Store
    def catalog(dataset):
        result, stats = augment(old_catalog(dataset), dataset)
        event('cache_v002', round=round_id, **stats)
        return result
    def store(directory, identity):
        path = Path(directory).resolve()
        copied = deepcopy(identity)
        # Existing retrieval-store identity stays untouched. New method/network
        # owners explicitly carry the max-round amendment provenance.
        if path.is_relative_to((OUT / f'rounds/r{round_id:03d}').resolve()) or path.is_relative_to((OUT / 'phases').resolve()):
            copied['infrastructure_revision_protocol_sha256'] = v1.sha(PROTOCOL)
            copied['infrastructure_revision'] = 'infra_recovery_v003'
        return old_store(directory, copied)
    v1.source_catalog, v1.Store = catalog, store
    try:
        yield
    finally:
        v1.source_catalog, v1.Store = old_catalog, old_store


@contextmanager
def _controller():
    META.mkdir(parents=True, exist_ok=True)
    path = META / 'ACTIVE.lock'
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf8') as stream:
            json.dump({'pid': os.getpid(), 'utc': v1.utc()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def _state(value, status, reason=None, *, no_progress=None):
    record = {'utc': v1.utc(), 'status': status, 'reason': reason,
              'actual_completeness': value, 'all900_scorable': value['scorable'] == 900,
              'no_progress_streak': no_progress, 'budget': v1.budget_snapshot(),
              'method_changes': 0, 'unknown_reserves_retained': True, 'fresh_dispatched': False}
    v1.save(META / 'STATUS.json', record)
    if status == 'complete':
        v1.save(OUT / 'RECOVERY_V003_COMPLETE.json', record)
    elif status.startswith('waiting'):
        v1.save(META / 'WAITING_FOR_INSPECTION.json', record)
    return record


def run():
    parent, protocol = prepare()
    gate()
    current = inspect()
    streak = _historical_streak(current)
    extended = deepcopy(parent)
    extended['max_rounds'] = protocol['new_max_rounds']
    assert {key for key in extended if extended[key] != parent[key]} == {'max_rounds'}
    if current['scorable'] == 900:
        return _state(current, 'complete', no_progress=streak)
    for round_id in ROUNDS:
        gate()
        verify(protocol)
        current = inspect()
        pending = current['eligible_missing']
        if not pending:
            return _state(current, 'waiting_nonretryable_missing',
                          'No retryable infrastructure cases remain; do not relabel missing as failures or change methods', no_progress=streak)
        meta = META / f'r{round_id:03d}'
        if meta.exists() or (OUT / f'rounds/r{round_id:03d}').exists():
            raise RuntimeError('Existing recovery round/owner; do not reopen it automatically')
        for dataset in DATASETS:
            for arm in v1.ARMS:
                if (OUT / f'rounds/r{round_id:03d}/methods/{dataset}/{arm}').exists():
                    raise RuntimeError('Method owner already exists')
        mask = [{'id': row['id'], 'arm': row['arm']} for row in pending]
        snapshot = {'round': round_id, 'ordered_missing': pending, 'method_case_mask': mask,
                    'mask_sha256': v1.digest(mask), 'before': current,
                    'infrastructure_protocol_sha256': v1.sha(PROTOCOL),
                    'max_rounds_amendment': {'old': 2, 'new': 5},
                    'historical_provider_records': {dataset: provider_records(1 if round_id == 2 else round_id - 1, dataset)
                                                    for dataset in DATASETS}}
        v1.freeze(meta / 'MISSING.frozen.json', snapshot)
        event('round_snapshot_frozen', round=round_id, eligible=len(mask), mask_sha256=snapshot['mask_sha256'])
        round_rows = []
        with _runner_adapter(round_id, protocol):
            for dataset in DATASETS:
                before = inspect()
                expected = [row for row in pending if row['dataset'] == dataset]
                actual = [row for row in before['eligible_missing'] if row['dataset'] == dataset]
                assert [{'id': r['id'], 'arm': r['arm']} for r in actual] == [
                    {'id': r['id'], 'arm': r['arm']} for r in expected], 'Missing mask changed during sequential recovery'
                if not expected:
                    continue
                event('dataset_dispatch', round=round_id, dataset=dataset,
                      mask_sha256=v1.digest([{'id': r['id'], 'arm': r['arm']} for r in expected]),
                      eligible=len(expected), circuit_reset_by_unchanged_v1_run=True)
                # One independent invocation per dataset. The existing run resets
                # ERRORS and opens/checks TLS; no algorithm/payload edits occur.
                result = v1.run(extended, sample='300', arms=list(v1.ARMS), round_id=round_id, dataset=dataset)
                after = inspect()
                gained = after['by_dataset'][dataset].get('scorable', 0) - before['by_dataset'][dataset].get('scorable', 0)
                streak[dataset] = 0 if gained > 0 else streak[dataset] + 1
                row = {'dataset': dataset, 'new_scorable_cases': gained,
                       'remaining_eligible': after['by_dataset'][dataset].get('eligible_infrastructure', 0),
                       'no_progress_streak': streak[dataset], 'provider_records': provider_records(round_id, dataset),
                       'case_statuses': after['by_dataset'][dataset], 'dispatch_result': result}
                round_rows.append(row)
                event('dataset_terminal', round=round_id, **row)
        current = inspect()
        v1.freeze(meta / 'TERMINAL.json', {'round': round_id, 'datasets': round_rows,
                                         'after': current, 'no_progress_streak': streak})
        if current['scorable'] == 900:
            return _state(current, 'complete', no_progress=streak)
        stagnant = [dataset for dataset in DATASETS if streak[dataset] >= 2 and
                    current['by_dataset'][dataset].get('eligible_infrastructure', 0)]
        if stagnant:
            return _state(current, 'waiting_repeated_no_progress',
                          'Two zero-gain dataset attempts: ' + ','.join(stagnant) + '; inspect recorded provider failures and deferred cases before further rounds',
                          no_progress=streak)
    return _state(current, 'waiting_round_limit', 'Reached explicit r002/r003/r004 limit; no hidden retry', no_progress=streak)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--inspect', action='store_true')
    modes.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        _, protocol = prepare()
        result = {'prepared': True, 'allowed_new_rounds': protocol['allowed_new_round_ids'],
                  'method_changes': 0, 'new_api_calls': 0, 'new_retrieval_calls': 0}
    elif args.inspect:
        prepare()
        result = inspect()
    else:
        with _controller():
            try:
                result = run()
            except Exception as exc:
                record = {'utc': v1.utc(), 'status': 'stopped_after_failure',
                          'error_type': type(exc).__name__, 'reason': str(exc)[:1000],
                          'automatic_retry': False, 'method_changes': 0,
                          'budget': v1.budget_snapshot()}
                v1.save(META / 'STOPPED_AFTER_FAILURE.json', record)
                event('controller_stopped_after_failure', **{key: value for key, value in record.items() if key != 'utc'})
                raise
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.run and not result['all900_scorable']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
