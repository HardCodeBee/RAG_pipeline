"""Run only named offline tests with socket connections disabled."""
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'experiments/query_decomposition/unified_rag_eval_v1_20261021'
attempts = []


def blocked(*args, **kwargs):
    attempts.append('network_connection_attempted')
    raise AssertionError('Offline verification attempted a real network connection')


class Results:
    def __init__(self):
        self.tests = []

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.failed:
            self.tests.append({'test': report.nodeid.split('/')[-1], 'outcome': report.outcome})


if __name__ == '__main__':
    import pytest
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    modules = ['test_runtime.py', 'test_runtime_transport.py', 'test_runtime_reuse.py',
               'test_qa_raw_runtime.py', 'test_runtime_report.py',
               'test_contracts.py', 'test_execution_policy.py']
    collector = Results()
    start = time.perf_counter()
    code = pytest.main(['-q', '--import-mode=importlib', '-p', 'no:cacheprovider',
                       *[str(BASE / n) for n in modules]], plugins=[collector])
    result = {'passed': code == 0 and not attempts, 'pytest_exit_code': int(code),
              'test_count': len({t['test'] for t in collector.tests}),
              'subtest_count': len(collector.tests) - len({t['test'] for t in collector.tests}),
              'tests': collector.tests,
              'network_connection_attempts': len(attempts),
              'new_model_requests': 0, 'GPU_jobs': 0,
              'duration_seconds': time.perf_counter() - start,
              'scope': 'synthetic transport/controller/QA raw, read-only reuse and frozen contract tests; not a live evaluation'}
    (HERE / 'OFFLINE_TESTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    sys.exit(0 if result['passed'] else 1)
