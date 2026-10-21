"""Finish authorized infrastructure recovery, then export the 300-query evidence.

Only missing cases can reach recovery_v003. No model/prompt/resource settings
are changed, successful cases remain immutable, and all costs stay in the ledger.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import run as base
from . import score

OUT = base.OUT
PREFIX = 'experiments.query_decomposition.p02_flow_simplify_v1_20261020.'


@contextmanager
def exclusive():
    path = OUT / 'FINISH300_ACTIVE.lock'
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as stream:
            json.dump({'pid': os.getpid(), 'utc': base.utc(), 'source_sha256': base.sha(__file__)}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def command(module, *args):
    cmd = [sys.executable, '-X', 'utf8', '-m', PREFIX + module, *args]
    base.log('finish300_command', command=cmd, source_sha256=base.sha(__file__))
    value = subprocess.run(cmd, cwd=base.ROOT, check=False)
    base.log('finish300_command_returned', command=cmd, returncode=value.returncode)
    if value.returncode:
        raise RuntimeError('Child failed; inspect, no blind restart: ' + module)


def counts():
    result = {arm: 0 for arm in base.ARMS}
    for case in base.rows(base.HERE / 'runtime300.frozen.jsonl'):
        for arm in base.ARMS:
            _, value = score.select_case(OUT, case['dataset'], arm, case['qid'])
            result[arm] += score.is_scorable(value)
    return result


def wait_until(condition, reason, deadline):
    while not condition():
        if time.monotonic() > deadline:
            raise RuntimeError('Bounded wait exhausted: ' + reason)
        time.sleep(5)


def main():
    base.prepare()
    deadline = time.monotonic() + 7200
    with exclusive():
        base.freeze(OUT / 'FINISH300_PROTOCOL.frozen.json', {
            'source_sha256': base.sha(__file__),
            'authorization': 'Continue completing the same300; report per-dataset F1, calls and costs; no gain requirement.',
            'parent_protocol_sha256': base.sha(base.HERE / 'PROTOCOL.frozen.json'),
            'infrastructure_recovery_protocol_sha256': base.sha(base.HERE / 'INFRA_RECOVERY_V003.frozen.json'),
            'generation_changes': 0, 'new_successful_case_reruns': 0,
            'global_cap_cny': 300, 'shared_project_subcap_cny': 25,
            'new_fresh_dispatch': False,
        })
        try:
            wait_until(lambda: (OUT / 'CONTINUATION_COMPLETE.json').exists(), 'r001 continuation', deadline)
            current = counts()
            if any(n != 300 for n in current.values()):
                wait_until(lambda: (OUT / 'WAITING_FOR_RECOVERY.json').exists()
                           and not (OUT / 'AUTHORIZED_FINISH_ACTIVE.lock').exists(),
                           'old watcher to stop before missing-only recovery', deadline)
                command('recovery_v003', '--run')
            command('score', '--sample', '300', '--arms', *base.ARMS)
            command('adoption_report')
            command('diagnostic', '--sample', '300')
            command('report', '--sample', '300')
            final = counts()
            base.save(OUT / 'FINAL300_EXECUTION.json', {
                'utc': base.utc(), 'scorable_by_arm': final,
                'all900_scorable': all(n == 300 for n in final.values()),
                'status': 'complete' if all(n == 300 for n in final.values()) else 'needs_infrastructure_inspection',
                'budget': base.budget_snapshot(), 'positive_F1_gain_required': False,
                'adoption_not_automatically_claimed': True,
            })
        except Exception as exc:
            base.save(OUT / 'FINISH300_FAILURE.json', {
                'utc': base.utc(), 'error_type': type(exc).__name__, 'reason': str(exc)[:700],
                'scorable_by_arm': counts(), 'budget': base.budget_snapshot(),
                'automatic_retry': False,
            })
            raise


if __name__ == '__main__':
    main()
