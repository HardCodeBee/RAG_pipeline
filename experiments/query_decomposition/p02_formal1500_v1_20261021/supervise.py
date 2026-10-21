"""Bounded network recovery for the unchanged formal1500 worker.

Attach to the existing worker first. Never run two paid workers concurrently;
never restart for budget, source-change or algorithm errors. No model call is
used to probe connectivity.
"""
import argparse
import json
import socket
import ssl
import subprocess
import sys
import time

import psutil
from .prepare import HERE, OUT, ROOT, read
from .run import MODULE, budget
from .network import HOST, provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json


def note(state, **fields):
    value = {'utc': utc(), 'state': state, **fields}
    locked_json(OUT/'SUPERVISOR_STATUS.json', value)
    with (OUT/'SUPERVISOR_EVENTS.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps(value, ensure_ascii=False)+'\n')


def reports(native=False):
    # The live worker owns SCORES.*. Refresh that export only after it exits;
    # supplementary comparison files have one writer in this supervisor.
    for suffix in (('.score', '.report_comparison') if native else ('.report_comparison',)):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-m', MODULE.rsplit('.', 1)[0]+suffix],
            cwd=ROOT, capture_output=True, text=True, encoding='utf8')
        if result.returncode:
            note('report_error', report=suffix, error=result.stderr[-1200:])


def preflight():
    store = Store(OUT/'supervisor_network', {'task': HERE.name, 'scope': 'nonpaid TLS only'})
    try:
        with provider_dns(HOST, store):
            with socket.create_connection((HOST, 443), timeout=12) as tcp:
                with ssl.create_default_context().wrap_socket(tcp, server_hostname=HOST):
                    return True
    except (OSError, RuntimeError):
        return False
    finally:
        store.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attach-pid', type=int, required=True)
    args = parser.parse_args()
    process = psutil.Process(args.attach_pid)
    note('attached', worker_pid=process.pid, process_create_time=process.create_time(),
         configuration_changed=False, global_budget_cap_cny=300)
    last_report = 0.
    for recovery in range(4):
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except psutil.TimeoutExpired:
                if time.monotonic()-last_report >= 300:
                    reports(); last_report = time.monotonic()
                note('watching', worker_pid=process.pid, recovery=recovery, budget=budget())
        reports(native=True)
        state = read(OUT/'SCORES.json')
        if state['overall']['scorable'] == 1500:
            note('complete', worker_exit=code, scorable=1500, budget=budget()); return
        if (OUT/'STOP_REQUESTED').exists():
            note('stopped_by_marker', scorable=state['overall']['scorable']); return
        errorpath = OUT/'CONTROLLER_ERROR.json'
        error = read(errorpath) if errorpath.exists() else {}
        if code == 0:
            note('worker_finished_with_missing', scorable=state['overall']['scorable'],
                 missing=state['overall']['missing'], report=str(OUT/'SCORES.json')); return
        network_only = error.get('error_type') in {'OSError', 'ConnectionError', 'gaierror',
            'TimeoutError', 'ConnectionResetError', 'APIConnectionError'}
        if not network_only or recovery == 3:
            note('needs_attention', worker_exit=code, error=error,
                 scorable=state['overall']['scorable']); return
        recovered = False
        for attempt in range(20):
            if (OUT/'STOP_REQUESTED').exists():
                note('stopped_by_marker'); return
            if preflight():
                recovered = True; break
            note('waiting_for_network', recovery=recovery+1, nonpaid_probe=attempt+1,
                 unknown_liabilities_retained=True)
            time.sleep(30)
        if not recovered:
            note('network_unavailable', retries_exhausted=True, paid_probes=0); return
        stdout = (OUT/f'local.supervised{recovery+1:03d}.stdout.log').open('wb')
        stderr = (OUT/f'local.supervised{recovery+1:03d}.stderr.log').open('wb')
        try:
            child = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', '-m', MODULE, '--run'],
                cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        finally:
            stdout.close(); stderr.close()
        process = psutil.Process(child.pid)
        note('network_recovered_worker_restarted', worker_pid=child.pid, recovery=recovery+1,
             configuration_changed=False, successful_cases_rerun=False, budget=budget())


if __name__ == '__main__':
    main()
