"""Reconnect authorized SSH result collection; never submits a new GPU task."""
import json
import os
import time

import paramiko
from . import remote
from .prepare import OUT, read


def main():
    secret = os.environ.pop('RAG_EVAL_REMOTE_PASSWORD', None)
    if not (OUT/'REMOTE_START.json').exists():
        raise RuntimeError('Existing Slurm receipt required; this helper is continuation only')
    started = read(OUT/'REMOTE_START.json')
    assert str(started['job_id']) == '158008'
    from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import utc
    for attempt in range(12):
        try:
            if secret:
                os.environ['RAG_EVAL_REMOTE_PASSWORD'] = secret
            remote.main()
            return
        except paramiko.AuthenticationException:
            raise  # No guessing credentials or submitting alternative jobs.
        except (paramiko.SSHException, OSError) as exc:
            with (OUT/'REMOTE_RECONNECTS.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps({'utc': utc(), 'attempt': attempt+1,
                    'error_type': type(exc).__name__, 'existing_job': '158008',
                    'new_API_calls': 0, 'new_job_submitted': False})+'\n')
            if attempt == 11:
                raise
            time.sleep(20)
        finally:
            os.environ.pop('RAG_EVAL_REMOTE_PASSWORD', None)


if __name__ == '__main__':
    main()
