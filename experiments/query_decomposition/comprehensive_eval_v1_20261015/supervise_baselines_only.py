"""Bounded baseline continuation; cannot launch S02 or its controls."""
import os
import subprocess
import sys
import time

from .common import ROOT, RUN, read, save, utc
from .campaign import TARGET, DATASETS
from .run_execution_v10_baselines_only import require_policy


def baselines_complete():
    for dataset in DATASETS:
        selected_path = TARGET / dataset / 'BASELINE_CONFIGS_FROZEN.json'
        if not selected_path.exists():
            return False
        selected = read(selected_path)['selected']
        if len(selected) != 9 or any(not mid.startswith('b') for mid in selected):
            raise ValueError('Expected nine frozen baseline methods')
        for mid, config in selected.items():
            path = TARGET / dataset / 'formal' / mid / config['id'] / 'scores.json'
            if not path.exists() or read(path)['overall']['terminal_records'] != 500:
                return False
    return True


def main():
    require_policy()
    status = RUN / 'BASELINES_ONLY_SUPERVISOR_STATUS.json'
    failures = 0
    with (RUN / 'baseline_only.out.log').open('a', encoding='utf8') as out, (RUN / 'baseline_only.err.log').open('a', encoding='utf8') as err:
        while not (RUN / 'STOP_BASELINES_ONLY_REQUESTED').exists():
            require_policy()
            if baselines_complete():
                save(status, {'utc': utc(), 'status': 'baselines_complete_ours_paused', 'pid': os.getpid(),
                     'full_evaluation_complete': False, 'resume_ours_automatically': False})
                return
            child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', '-m',
                'experiments.query_decomposition.comprehensive_eval_v1_20261015.run_execution_v10_baselines_only'],
                cwd=ROOT, stdout=out, stderr=err)
            save(status, {'utc': utc(), 'status': 'running_baselines_ours_paused',
                 'pid': os.getpid(), 'child_pid': child.pid, 'new_our_method_calls_allowed': False})
            code = child.wait()
            subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m',
                'experiments.query_decomposition.comprehensive_eval_v1_20261015.report'], cwd=ROOT, stdout=out, stderr=err)
            if code:
                failures += 1
                progress = read(TARGET / 'progress.json')
                reason = progress.get('reason', '')
                if failures >= 10 or any(s in reason for s in ['Frozen campaign', 'after freeze',
                        'Authorized API spending cap', 'Protected global budget', 'User-paused']):
                    save(status, {'utc': utc(), 'status': 'need_attention', 'exit_code': code,
                         'reason': reason, 'new_our_method_calls_allowed': False})
                    return
            else:
                failures = 0
            time.sleep(min(60, 5 * max(failures, 1)))


if __name__ == '__main__':
    main()
