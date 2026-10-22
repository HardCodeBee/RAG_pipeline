"""Connection-only recovery; never resend an unresolved/failed request.

The cohort and prompts remain frozen. Previously incomplete cases are held in
the full90 report while successful cases are reused and untouched cases run.
"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021 import run as base
from p02_multigrain_timing_cachefix_v1.core.runtime import Ask


class RecordingClient:
    def __init__(self, client):
        self.delegate = client
        self.last_failure = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **payload):
        self.last_failure = None
        try:
            return self.delegate.chat.completions.create(**payload)
        except Exception as exc:
            chain, cause = [], exc
            while cause is not None and len(chain) < 8:
                chain.append({'type': type(cause).__name__, 'errno': getattr(cause, 'errno', None)})
                cause = cause.__cause__ or cause.__context__
            self.last_failure = chain
            raise

    def close(self):
        self.delegate.close()


class RecordingLLM(base.LimitedLLM):
    def request_once(self, payload):
        result = super().request_once(payload)
        if self.client.last_failure:
            result['failure_causes'] = self.client.last_failure
            result['proved_not_submitted'] = any(
                v['type'] in {'ConnectError', 'ConnectTimeout'} for v in self.client.last_failure)
        return result


class RecoveryAsk(Ask):
    def __init__(self, *args, **kwargs):
        import httpx
        from openai import OpenAI
        import os
        super().__init__(*args, **kwargs)
        timeout = httpx.Timeout(self.settings['timeout_seconds'], connect=10)
        http = httpx.Client(transport=httpx.HTTPTransport(retries=0,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=30)),
            timeout=timeout, trust_env=False)
        api = OpenAI(api_key=os.environ[self.settings['api_key_env']], base_url=self.settings['base_url'],
                     timeout=timeout, max_retries=0, http_client=http)
        self.client = RecordingLLM(self.settings, client=RecordingClient(api))
        self.client.owns_client = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    base.check_pins()
    assert base.read(base.OUT / 'EXTERNAL_PAYLOAD_APPROVED.json')['usage_charges_approved']
    original_rows = base.rows
    held = {base.read(p)['id'] for p in (base.OUT / 'candidate').glob('*/failure_*.json')}
    record = {'utc': base.utc(), 'amendment': 'connection_recovery002',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'changes': 'Explicit single-connection HTTP pool per worker; no environment proxy; connect timeout10, original read timeout60; record typed exception causes.',
        'cohort_unchanged': True, 'prompts_and_generation_payloads_unchanged': True,
        'held_cases': sorted(held), 'unresolved_requests_resent': False,
        'main_evaluation_remains_paused': True}
    path = base.OUT / ('RECOVERY002_' + record['utc'].replace(':', '').replace('.', '') + '.json')
    base.save(path, record)
    def pending_rows(path):
        values = original_rows(path)
        if Path(path).resolve() == (base.COHORT / 'runtime90.frozen.jsonl').resolve():
            return [v for v in values if v['id'] not in held]
        return values
    base.rows = pending_rows
    base.ScopedAsk = RecoveryAsk
    base.emit({'held_without_resubmission': len(held), 'full_scoring_denominator': 90})
    result = base.run()
    base.emit(result)


if __name__ == '__main__':
    main()
