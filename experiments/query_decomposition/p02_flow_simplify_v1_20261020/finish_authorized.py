"""Durable authorized continuation: finish300 gate, then frozen fresh60.

Default waits at most2h for CONTINUATION_COMPLETE.json, polling every5s.
It does not repair the300 campaign, add rounds, tune a method, or retry a
failed subprocess. Only fresh_evaluation's already-authorized r000/r001 are
available. Existing successful cases and all unknown liabilities stay intact.
"""
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import fresh_evaluation as fresh
from . import run as base
from . import score


HERE = Path(__file__).resolve().parent
OUT = base.OUT
MARKER = OUT / 'CONTINUATION_COMPLETE.json'
LOG = OUT / 'AUTHORIZED_FINISH_COMMANDS.jsonl'
FINAL = OUT / 'FINAL_EXECUTION_STATUS.json'
WAITING = OUT / 'WAITING_FOR_RECOVERY.json'
MODULE = 'experiments.query_decomposition.p02_flow_simplify_v1_20261020.fresh_evaluation'


def event(name, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {'utc': base.utc(), 'event': name, **fields}
    with LOG.open('a', encoding='utf8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return row


@contextmanager
def single_controller():
    """A stale lock needs inspection; it is never stolen automatically."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'AUTHORIZED_FINISH_ACTIVE.lock'
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf8') as stream:
            json.dump({'pid': os.getpid(), 'utc': base.utc(), 'script_sha256': base.sha(__file__)}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


class SubprocessFailure(RuntimeError):
    pass


def command(*args):
    cmd = [sys.executable, '-X', 'utf8', '-m', MODULE, *args]
    event('command_started', command=cmd)
    result = subprocess.run(cmd, cwd=base.ROOT, check=False)
    record = event('command_finished', command=cmd, returncode=result.returncode)
    if result.returncode:
        base.save(OUT / 'AUTHORIZED_FINISH_SUBPROCESS_FAILURE.json', record)
        raise SubprocessFailure('Stopping after nonzero returncode; no implicit retry: ' + ' '.join(args))


def complete(summary, planned):
    return all(value['planned'] == planned and value['scorable'] == planned
               and value['F1_complete'] is not None for value in summary['arms'].values())


def compact(summary):
    return {arm: {key: value[key] for key in (
        'planned', 'scorable', 'missing', 'statuses', 'F1_complete', 'F1_available', 'F1_all_planned_bounds')}
        for arm, value in summary['arms'].items()}


def check300():
    # Explicit original300 inputs. No fresh reference or scores enter this gate.
    return score.summarize(
        OUT, HERE / 'runtime300.frozen.jsonl', arms=fresh.CANDIDATES,
        references=HERE / 'references300.local.frozen.jsonl', baseline_root=None,
        output_path=OUT / 'AUTHORIZED_FINISH_300_CHECK.json')


def fresh_state():
    """Offline status/score only, using the already-frozen choice and roster."""
    if not fresh.CHOICE.exists():
        return None
    chosen = base.read(fresh.CHOICE)
    return score.summarize(
        fresh.OUT, fresh.RUNTIME, arms=(fresh.BASELINE, chosen['selected_arm']),
        references=fresh.REFERENCES, baseline_root=fresh.OUT)


def missing_cases(root, samples, arms):
    result = []
    for case in base.rows(samples):
        for arm in arms:
            _, value = score.select_case(root, case['dataset'], arm, case['qid'])
            if score.is_scorable(value):
                continue
            result.append({
                'id': case['id'], 'dataset': case['dataset'], 'arm': arm,
                'status': value.get('status', 'unstarted') if value else 'unstarted',
                'retryable_infrastructure': bool(value and value.get('retryable_infrastructure', False)),
                'error_type': value.get('error_type') if value else None,
                'reason': value.get('reason') if value else None,
                'failed_stage': value.get('failed_stage') if value else None,
                'request_status': value.get('request_status') if value else None,
                'http_status': value.get('http_status') if value else None,
            })
    return result


def save_final(status, *, parent=None, new=None, reason=None, commands_done=()):
    record = {
        'utc': base.utc(), 'status': status, 'reason': reason,
        'completed_commands_this_controller': list(commands_done),
        'development300': compact(parent) if parent else None,
        'all900_development_scorable': complete(parent, 300) if parent else False,
        'fresh60': compact(new) if new else None,
        'all120_fresh_scorable': complete(new, 60) if new else False,
        'fresh_all_generation_success': bool(new) and all(
            value['scorable'] == 60 and all(name.startswith('success') for name in value['statuses'])
            for value in new['arms'].values()),
        'budget': fresh._snapshot(),
        'selection': base.read(fresh.CHOICE) if fresh.CHOICE.exists() else None,
        'limits': {'global_cap_cny': 300, 'shared_project_subcap_cny': 25,
                   'max_method_rounds': 2, 'additional_recovery_rounds_created': 0},
        'limitations': [
            'Fresh60 are new downstream RAG queries but previously exposed decomposer development queries, not an untouched holdout.',
            'MuSiQue20 are primitive-isolated native train heldout queries, not official dev or test.',
            'Algorithm failures are scored zero; infrastructure missing retain their planned denominator and bounds.',
            'Returned-usage price estimates and unresolved reservations are not a verified invoice.',
        ],
    }
    base.save(FINAL, record)
    event('final_status_written', status=status,
          all900_development_scorable=record['all900_development_scorable'],
          all120_fresh_scorable=record['all120_fresh_scorable'], reason=reason)
    return record


def waiting(reason, *, parent=None, new=None, missing=None, commands_done=()):
    base.save(WAITING, {'utc': base.utc(), 'reason': reason, 'missing_cases': missing or [],
                        'automatic_additional_rounds': False, 'new_model_method_changes': False,
                        'preserve_r000_r001_and_successes': True, 'budget': fresh._snapshot()})
    return save_final('waiting_for_recovery_inspection', parent=parent, new=new,
                      reason=reason, commands_done=commands_done)


def execute(wait_seconds=7200):
    if not 0 <= wait_seconds <= 7200:
        raise ValueError('Wait must be bounded to0..7200seconds')
    plan = base.prepare()
    base.verify(plan)
    fresh_plan = fresh.prepare()
    fresh.verify(fresh_plan)
    protocol = {
        'script_sha256': base.sha(__file__), 'parent_protocol_sha256': base.sha(HERE / 'PROTOCOL.frozen.json'),
        'fresh_protocol_sha256': base.sha(fresh.PROTOCOL), 'selection_rule_sha256': base.sha(fresh.RULE),
        'wait_marker': str(MARKER.relative_to(base.ROOT)), 'maximum_wait_seconds': 7200,
        'poll_seconds': 5, 'candidate_rounds': [0, 1], 'extra_parent_recovery': False,
        'extra_fresh_recovery': False,
        'stop_on_any_subprocess_failure': True,
    }
    base.freeze(OUT / 'AUTHORIZED_FINISH_PROTOCOL.frozen.json', protocol)
    parent, new, commands_done = None, None, []
    event('controller_started', pid=os.getpid(), wait_seconds=wait_seconds)
    try:
        deadline = time.monotonic() + wait_seconds
        while not MARKER.exists():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return waiting('Parent continuation marker absent after bounded wait; no dispatch started',
                               commands_done=commands_done)
            time.sleep(min(5, remaining))
        parent = check300()
        if not complete(parent, 300):
            missing = missing_cases(OUT, HERE / 'runtime300.frozen.jsonl', fresh.CANDIDATES)
            return waiting('Development300 still has missing results after the frozen continuation/r001; no extra retries and no fresh selection/dispatch',
                           parent=parent, missing=missing, commands_done=commands_done)

        # Freeze the300-only winner before any new60 generation or scoring.
        command('--select')
        commands_done.append('select_from_complete300')
        command('--run')
        commands_done.append('fresh_r000')
        command('--score')
        commands_done.append('score_fresh_r000')
        new = fresh_state()
        assert new is not None
        chosen = base.read(fresh.CHOICE)
        arms = (fresh.BASELINE, chosen['selected_arm'])
        missing = missing_cases(fresh.OUT, fresh.RUNTIME, arms)
        recovery = [row for row in missing if row['status'] == 'incomplete' and row['retryable_infrastructure']]
        if recovery:
            base.freeze(fresh.OUT / 'AUTHORIZED_RECOVERY_SELECTION.frozen.json', {
                'method_cases': [{'id': row['id'], 'arm': row['arm']} for row in recovery],
                'round': 1, 'successful_cases_excluded': True, 'algorithm_failures_excluded': True,
                'extra_rounds_allowed': False, 'runtime_sha256': base.sha(fresh.RUNTIME),
                'selection_sha256': base.sha(fresh.CHOICE)})
            command('--run', '--round', '1')
            commands_done.append('fresh_r001_infrastructure_only')
            command('--score')
            commands_done.append('score_fresh_r001')
            new = fresh_state()
            missing = missing_cases(fresh.OUT, fresh.RUNTIME, arms)
        if missing:
            return waiting('Fresh60 still has missing results; only frozen r000/r001 were allowed, so further dispatch requires inspection',
                           parent=parent, new=new, missing=missing, commands_done=commands_done)
        return save_final('complete', parent=parent, new=new, commands_done=commands_done)
    except Exception as exc:
        # Offline snapshot only. A failed child process never triggers a retry.
        if fresh.CHOICE.exists():
            try:
                new = fresh_state()
            except Exception as snapshot_error:
                event('snapshot_error', error_type=type(snapshot_error).__name__, reason=str(snapshot_error)[:500])
        save_final('stopped_after_failure', parent=parent, new=new,
                   reason=type(exc).__name__ + ': ' + str(exc)[:700], commands_done=commands_done)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=float, default=7200)
    args = parser.parse_args()
    with single_controller():
        result = execute(args.wait_seconds)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
