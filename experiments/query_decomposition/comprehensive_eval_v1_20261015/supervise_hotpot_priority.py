"""Use the original bounded supervisor with a priority-only entrypoint."""
import subprocess
from . import supervise
from .run_execution_v8_hotpot_priority import verify_epoch


def main():
    verify_epoch()
    original = subprocess.Popen

    class Launch(original):
        def __init__(self, args, *pargs, **kwargs):
            command = list(args) if isinstance(args, (list, tuple)) else args
            old = 'experiments.query_decomposition.comprehensive_eval_v1_20261015.campaign'
            if isinstance(command, list) and old in command:
                command[command.index(old)] = 'experiments.query_decomposition.comprehensive_eval_v1_20261015.run_execution_v8_hotpot_priority'
            super().__init__(command, *pargs, **kwargs)

    subprocess.Popen = Launch
    supervise.main()


if __name__ == '__main__':
    main()
