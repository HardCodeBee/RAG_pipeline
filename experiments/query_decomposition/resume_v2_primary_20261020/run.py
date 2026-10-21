"""Resume frozen v2 under cumulative CNY300, with local process supervision.

No prompts, retrieval rules, sample IDs, generation limits or frozen files change.
This supervisor writes local status files; it does not schedule chat notifications.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
V2 = PARENT / 'comprehensive_v1_20261015/baseline_integrity_repair_v2_20261017'
OUT = PARENT / 'resume_v2_primary_20261020'
MODULE = 'experiments.query_decomposition.resume_v2_primary_20261020.run'
CAP = 300.0
GUARD = 299.9


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
    locked_json(Path(path), value)


@contextmanager
def lock(name):
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def inventory():
    rows = []
    for ds in ['hotpotqa', '2wiki', 'musique']:
        cfgs = read(V2.parent / 'campaign_api_v1' / ds / 'BASELINE_CONFIGS_FROZEN.json')['selected']
        for mid, cfg in cfgs.items():
            directory = (V2 / 'matched50' / ds / 'formal' / mid / 'k10_t5'
                         if mid == 'b05_iter_retgen_bge'
                         else V2 / 'main' / ds / 'formal' / mid / cfg['id'])
            p = directory / 'checkpoint.sqlite3'
            states = {}
            n = valid = 0
            if p.exists():
                with sqlite3.connect(p.resolve().as_uri() + '?mode=ro', uri=True, timeout=5) as db:
                    values = [json.loads(v) for (v,) in db.execute("SELECT value FROM data WHERE kind='prediction'")]
                for v in values:
                    s = v.get('status', 'unknown')
                    states[s] = states.get(s, 0) + 1
                    valid += int(isinstance(v.get('prediction'), str) and
                                 (s.startswith('success') or s == 'algorithm_failure'))
                n = len(values)
            rows.append({'dataset': ds, 'method': mid, 'planned': 500,
                         'processed': n, 'scorable': valid, 'statuses': states})
    with sqlite3.connect((PARENT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True, timeout=5) as db:
        a, b = db.execute('SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
    return {'utc': utc(), 'primary_protocol': 'v2 + matched50 Iter k10/T5',
            'planned': 13500, 'processed': sum(x['processed'] for x in rows),
            'scorable': sum(x['scorable'] for x in rows), 'rows': rows,
            'budget': {'cumulative_cap_cny': CAP, 'submission_guard_cny': GUARD,
                       'known_cny': a, 'unknown_reserves_cny': b,
                       'occupied_cny': a+b, 'remaining_cny': CAP-a-b}}


def install_budget_overlay():
    from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import common, transport
    from experiments.query_decomposition.baseline_origin_binding_fix_20261017 import install
    from experiments.query_decomposition import iter_react_matched50_20261017 as matched
    common.verify()
    runtime = install()
    matched.freeze()  # Verify the historical freeze before installing runtime policy.
    original = transport.Budget

    class AuthorizedBudget(original):
        def __init__(self, path, cap=CAP, interval=2/3):
            super().__init__(path, CAP, interval)

    transport.PROTECTED_CAP = GUARD
    transport.Budget = AuthorizedBudget
    runtime.Budget = AuthorizedBudget
    return common, runtime, matched, AuthorizedBudget


def validate():
    common, runtime, matched, Budget = install_budget_overlay()
    with tempfile.TemporaryDirectory(prefix='budget_test_', dir=OUT) as directory:
        b = Budget(Path(directory) / 'ledger.sqlite3', 200, 2/3)
        assert b.cap == CAP
        b.reserve('seed', 'offline', 299.89)
        blocked = False
        try:
            b.reserve('must_block', 'offline', .02)
        except RuntimeError as e:
            blocked = 'spending cap' in str(e)
        assert blocked and b.cap == CAP
        b.reserve('allowed', 'offline', .005)
        duplicate_blocked = False
        try:
            b.reserve('allowed', 'offline', .005)
        except RuntimeError as e:
            duplicate_blocked = 'silently replayed' in str(e)
        assert duplicate_blocked
        b.db.close()
    common.verify()
    value = {'utc': utc(), 'passed': True, 'api_calls': 0,
             'checks': {'frozen_v2_and_matched50_verified': True,
                        'budget_constructor_uses_current300': True,
                        'atomic_reservation_respects299_9': blocked,
                        'duplicate_liability_still_blocked': duplicate_blocked,
                        'frozen_sources_unchanged': True},
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(OUT / 'VALIDATION.json', value)
    print(json.dumps(value), flush=True)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (V2 / 'STOP_REQUESTED').exists() or (OUT / 'STOP_REQUESTED').exists():
        raise RuntimeError('Stop marker present; do not silently clear it')
    if (OUT / 'PROTOCOL.json').exists():
        value = read(OUT / 'PROTOCOL.json')
        if value['source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise RuntimeError('Continuation source changed; create a new version')
        return
    for name in ['SUPERVISION.json', 'progress.json']:
        p = V2 / name
        if p.exists():
            write(OUT / ('BEFORE_' + name), read(p))
    write(OUT / 'PROTOCOL.json', {
        'utc': utc(), 'authorization': 'User requested resume v2 evaluation and continuous monitoring; cumulative cap previously raised to CNY300.',
        'scope': 'Local v2 primary nine baselines only; no remote resume or change of main method.',
        'cumulative_cap_cny': CAP, 'submission_guard_cny': GUARD,
        'change': 'Runtime budget overlay only; old frozen200/195 receipts retained truthfully.',
        'unchanged': ['samples', 'generator', 'prompts', 'retrieval', 'answer extraction',
                      '32768 generation cap', 'v2 integrity guards', 'matched50 search allowance',
                      'first-complete-response and unknown-liability policy'],
        'worker': 'Original iter_react_matched50 driver, then remaining non-Iter v2 baselines.',
        'monitor': {'local_poll_seconds': 20, 'inventory_seconds': 60,
                    'maximum_transport_recoveries': 2, 'lease_hours': 24,
                    'chat_notifications_scheduled': False},
        'pending': ['uniform two-metric retrieval-overhead instrumentation',
                    'main method and ablations v2 adaptation', 'seven-arm remote expansion adaptation'],
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write(OUT / 'BEFORE_INVENTORY.json', inventory())


def worker():
    with lock('worker.lock'):
        prepare()
        common, runtime, matched, Budget = install_budget_overlay()
        validation = read(OUT / 'VALIDATION.json')
        if not validation['passed'] or validation['source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise RuntimeError('Matching offline validation required')
        write(OUT / 'WORKER.json', {'utc': utc(), 'pid': os.getpid(), 'status': 'starting', 'cap_cny': CAP})
        sys.argv = [sys.argv[0]]
        try:
            matched.main()
        except BaseException as e:
            write(OUT / 'WORKER.json', {'utc': utc(), 'pid': os.getpid(), 'status': 'stopped',
                                       'error': type(e).__name__, 'reason': str(e)})
            raise
        else:
            write(OUT / 'WORKER.json', {'utc': utc(), 'pid': os.getpid(), 'status': 'completed'})


def supervise():
    with lock('supervisor.lock'):
        prepare()
        from experiments.query_decomposition.iter_react_matched50_watch_20261017 import network_stop_only
        deadline = time.monotonic() + 24*3600
        recoveries = 0
        while time.monotonic() < deadline:
            if (OUT/'STOP_REQUESTED').exists() or (V2/'STOP_REQUESTED').exists():
                write(OUT/'SUPERVISION.json', {'utc': utc(), 'status': 'user_stop_no_restart'})
                return
            with (OUT/'worker.stdout.log').open('a', encoding='utf8') as stdout, (OUT/'worker.stderr.log').open('a', encoding='utf8') as stderr:
                child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', '-m', MODULE, '--worker'],
                                         cwd=ROOT, stdout=stdout, stderr=stderr,
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            last_inventory = 0.
            while child.poll() is None:
                if (OUT/'STOP_REQUESTED').exists() and not (V2/'STOP_REQUESTED').exists():
                    write(V2/'STOP_REQUESTED', {'utc': utc(), 'reason': 'Explicit stop marker in continuation directory; drain current work and do not restart.'})
                value = {'utc': utc(), 'pid': os.getpid(), 'worker_pid': child.pid,
                         'status': 'running', 'recoveries': recoveries,
                         'cumulative_cap_cny': CAP, 'submission_guard_cny': GUARD,
                         'continuation_directory': str(OUT), 'chat_notifications_scheduled': False}
                progress = V2/'progress.json'
                if progress.exists():
                    value['generation_progress'] = read(progress)
                write(OUT/'SUPERVISION.json', value)
                write(V2/'SUPERVISION.json', value)
                if time.monotonic() - last_inventory >= 60:
                    try:
                        write(OUT/'PROGRESS.json', inventory())
                    except (sqlite3.Error, OSError) as e:
                        write(OUT/'MONITOR_IO_ERROR.json', {'utc': utc(), 'error': type(e).__name__})
                    last_inventory = time.monotonic()
                time.sleep(20)
            try:
                write(OUT/'PROGRESS.json', inventory())
            except (sqlite3.Error, OSError):
                pass
            progress = read(V2/'progress.json')
            reason = progress.get('reason', '')
            if child.returncode == 0 and progress.get('status') == 'completed':
                state = 'completed'
            elif (recoveries < 2 and reason.startswith('Transport stopped;') and network_stop_only()
                  and not (OUT/'STOP_REQUESTED').exists() and not (V2/'STOP_REQUESTED').exists()):
                recoveries += 1
                time.sleep(20)
                continue
            else:
                state = 'needs_attention_no_auto_restart'
            value = {'utc': utc(), 'pid': os.getpid(), 'status': state,
                     'worker_returncode': child.returncode, 'recoveries': recoveries,
                     'generation_progress': progress, 'continuation_directory': str(OUT)}
            write(OUT/'SUPERVISION.json', value); write(V2/'SUPERVISION.json', value)
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        validate()
    elif args.worker:
        worker()
    else:
        supervise()


if __name__ == '__main__':
    main()
