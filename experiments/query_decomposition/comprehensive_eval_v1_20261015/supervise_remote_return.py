"""Transfer supervision without stopping the existing API worker."""
import argparse
import ctypes
from ctypes import wintypes
import os
import subprocess
import sys

from .common import ROOT, HERE, RUN, read, save, sha, log, utc
from . import supervise


def open_process(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel, handle


def wait_process(kernel, handle):
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    while True:
        result = kernel.WaitForSingleObject(handle, 30_000)
        if result == 0:
            return
        if result != 258:
            raise ctypes.WinError(ctypes.get_last_error())


def close_process(kernel, handle):
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-pid', type=int, required=True)
    parser.add_argument('--retiring-supervisor-pid', type=int, required=True)
    args = parser.parse_args()
    epoch = read(RUN / 'REMOTE_RETURN_EXECUTION_EPOCH.json')
    if sha(__file__) != epoch['supervisor_sha256'] or sha(HERE / 'run_execution_v4.py') != epoch['entrypoint_sha256']:
        raise ValueError('Return execution epoch source changed')
    kernel, worker = open_process(args.wait_for_pid)
    _, parent = open_process(args.retiring_supervisor_pid)
    status = {'utc': utc(), 'status': 'handoff_ready_waiting_for_retiring_supervisor',
              'pid': os.getpid(), 'existing_worker_pid': args.wait_for_pid,
              'retiring_supervisor_pid': args.retiring_supervisor_pid,
              'existing_worker_or_api_requests_stopped': False}
    try:
        save(RUN / 'REMOTE_RETURN_HANDOFF_STATUS.json', status)
        wait_process(kernel, parent)
        status.update(utc=utc(), status='waiting_for_existing_baseline_worker')
        save(RUN / 'REMOTE_RETURN_HANDOFF_STATUS.json', status)
        save(RUN / 'SUPERVISOR_STATUS.json', status)
        log('remote_return_supervision_handoff_active', **{k: v for k, v in status.items() if k != 'utc'})
        wait_process(kernel, worker)
    finally:
        close_process(kernel, parent)
        close_process(kernel, worker)
    save(RUN / 'REMOTE_RETURN_HANDOFF_STATUS.json', {**status, 'utc': utc(), 'status': 'existing_worker_exited_resuming_compatible_entrypoint'})
    # The preceding worker retains its current baseline pass and all checkpoints.
    # Reporting is offline, including after its obsolete importer rejects a name.
    with (RUN / 'campaign.out.log').open('a', encoding='utf8') as out, (RUN / 'campaign.err.log').open('a', encoding='utf8') as err:
        subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m',
                        'experiments.query_decomposition.comprehensive_eval_v1_20261015.report'],
                       cwd=ROOT, stdout=out, stderr=err)
    original = subprocess.Popen

    class Launch(original):
        def __init__(self, args, *pargs, **kwargs):
            command = list(args) if isinstance(args, (list, tuple)) else args
            old = 'experiments.query_decomposition.comprehensive_eval_v1_20261015.campaign'
            if isinstance(command, list) and old in command:
                if sha(HERE / 'run_execution_v4.py') != epoch['entrypoint_sha256']:
                    raise ValueError('Return execution entrypoint changed')
                command[command.index(old)] = 'experiments.query_decomposition.comprehensive_eval_v1_20261015.run_execution_v4'
            super().__init__(command, *pargs, **kwargs)

    subprocess.Popen = Launch
    supervise.main()


if __name__ == '__main__':
    main()
