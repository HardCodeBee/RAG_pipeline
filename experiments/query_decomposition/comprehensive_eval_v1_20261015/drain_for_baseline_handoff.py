"""Windows-only approved handoff: retire a worker only with zero live requests."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sqlite3
import time

from .common import RUN, save, utc
from .campaign import TARGET


def pending_requests():
    active = []
    # Campaign executes one method block at a time and joins its API workers
    # before switching methods. Inspect its stable, currently active checkpoint.
    progress = json.loads((TARGET / 'progress.json').read_text(encoding='utf8'))
    if progress.get('status') != 'running':
        raise RuntimeError('Active method is not identifiable; leave the worker running')
    path = TARGET / progress['dataset'] / progress['split'] / progress['method'] / progress['config'] / 'checkpoint.sqlite3'
    for path in [path]:
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=2) as db:
            count = db.execute("SELECT count(*) FROM requests WHERE status='in_flight'").fetchone()[0]
        if count:
            active.append({'checkpoint': str(path), 'in_flight': count})
    return active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=300)
    args = parser.parse_args()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    native = ctypes.WinDLL('ntdll')
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    native.NtSuspendProcess.argtypes = [wintypes.HANDLE]
    native.NtSuspendProcess.restype = ctypes.c_long
    native.NtResumeProcess.argtypes = [wintypes.HANDLE]
    native.NtResumeProcess.restype = ctypes.c_long
    handle = kernel.OpenProcess(0x0800 | 0x0001, False, args.worker_pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    deadline = time.monotonic() + args.timeout_seconds
    suspended = False
    try:
        while time.monotonic() < deadline:
            if native.NtSuspendProcess(handle) != 0:
                raise RuntimeError('Cannot inspect the worker at a stable handoff point')
            suspended = True
            active = pending_requests()
            if not active:
                if not kernel.TerminateProcess(handle, 0):
                    raise ctypes.WinError(ctypes.get_last_error())
                suspended = False
                receipt = {'utc': utc(), 'status': 'retired_after_zero_in_flight_requests',
                    'worker_pid': args.worker_pid, 'in_flight_requests': 0,
                    'existing_checkpoints_preserved': True, 'unknown_requests_replayed': False,
                    'user_authorized_brief_shared_worker_handoff': True}
                save(RUN / 'BASELINES_ONLY_DRAIN_RECEIPT.json', receipt)
                print(json.dumps(receipt), flush=True)
                return
            native.NtResumeProcess(handle)
            suspended = False
            print(json.dumps({'utc': utc(), 'status': 'waiting_existing_requests',
                'in_flight_requests': sum(row['in_flight'] for row in active)}), flush=True)
            time.sleep(1)
        raise TimeoutError('No safe handoff point; the existing worker was left running')
    finally:
        if suspended:
            native.NtResumeProcess(handle)
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    main()
