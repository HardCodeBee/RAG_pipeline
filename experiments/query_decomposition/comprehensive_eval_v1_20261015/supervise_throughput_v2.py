"""Retain the bounded supervisor; dispatch only the verified execution epoch."""
import subprocess
from .common import HERE, RUN, read, sha
from . import supervise


def main():
    epoch = read(RUN / 'THROUGHPUT_ENTRYPOINT_FROZEN_v2.json')
    for name, expected in epoch['source_sha256'].items():
        if sha(HERE / name) != expected:
            raise ValueError('Throughput entrypoint source changed: ' + name)
    original = subprocess.Popen

    class Launch(original):
        def __init__(self, args, *pargs, **kwargs):
            command = list(args) if isinstance(args, (list, tuple)) else args
            old = 'experiments.query_decomposition.comprehensive_eval_v1_20261015.campaign'
            if isinstance(command, list) and old in command:
                command[command.index(old)] = 'experiments.query_decomposition.comprehensive_eval_v1_20261015.run_execution_v6'
            super().__init__(command, *pargs, **kwargs)

    subprocess.Popen = Launch
    supervise.main()


if __name__ == '__main__': main()
