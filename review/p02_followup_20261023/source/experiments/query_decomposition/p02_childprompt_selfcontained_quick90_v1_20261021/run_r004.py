"""Approved one-time recovery in copied checkpoints; original receipts survive.

Requires a separate explicit approval of the listed ambiguous request IDs.
HTTP transport reconnects only before submitting an HTTP request body.
"""
from pathlib import Path
from threading import Lock
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021 import run as base
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021.run_r002 import RecordingClient, RecordingLLM
from p02_multigrain_timing_cachefix_v1.core.runtime import Ask


class RecoveryAsk(Ask):
    def __init__(self, *args, **kwargs):
        import httpx
        from openai import OpenAI
        super().__init__(*args, **kwargs)
        timeout = httpx.Timeout(self.settings['timeout_seconds'], connect=10)
        http = httpx.Client(transport=httpx.HTTPTransport(retries=2,
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
    original = base.OUT
    approval = base.read(original / 'UNKNOWN_RETRY_APPROVED.json')
    approved = set(approval['approved_request_ids'])
    assert approval['maximum_reissues_per_ambiguous_request'] == 1
    target = original / 'approved_recovery004'
    if target.exists():
        raise FileExistsError('Do not reissue the approved recovery a second time')
    target.mkdir()
    for name in ['control90.jsonl', 'OFFLINE_CHECK.json']:
        shutil.copyfile(original / name, target / name)
    retired = []
    for ds in base.DATASETS:
        source_dir, destination = original / 'candidate' / ds, target / 'candidate' / ds
        destination.mkdir(parents=True)
        source = sqlite3.connect((source_dir / 'checkpoint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
        copied = sqlite3.connect(destination / 'checkpoint.sqlite3')
        try:
            source.backup(copied)
            for rid, raw in copied.execute("SELECT id,result FROM requests WHERE status='unknown'").fetchall():
                result = json.loads(raw)
                if rid not in approved and not result.get('proved_not_submitted'):
                    raise RuntimeError('Unapproved ambiguous request remains; no provider submission')
                # Delete only from the new copy. Original unknown receipts and
                # cumulative reservations remain unchanged in their old namespace.
                copied.execute('DELETE FROM request_stages WHERE request_id=?', (rid,))
                copied.execute('DELETE FROM requests WHERE id=?', (rid,))
                retired.append({'dataset': ds, 'old_request_id': rid,
                    'authority': 'explicit_one_time_reissue' if rid in approved else 'typed_connect_phase_proof'})
            copied.commit()
        finally:
            copied.close()
            source.close()
        if (source_dir / 'trace.jsonl').exists():
            shutil.copyfile(source_dir / 'trace.jsonl', destination / 'trace.jsonl')
    assert approved == {v['old_request_id'] for v in retired if v['authority'] == 'explicit_one_time_reissue'}
    record = {'utc': base.utc(), 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_receipts_unchanged': True, 'retired_only_in_new_copies': retired,
        'successful_payloads_preserved_and_reused': True,
        'transport_connect_retries': 2, 'SDK_POST_retries': 0,
        'main_evaluation_remains_paused': True, 'full_cohort_size': 90}
    base.save(target / 'RECOVERY_AUTHORITY.json', record)
    from p02_multigrain_timing_cachefix_v1.core.retrieval import Retriever
    loading, previous_load = Lock(), Retriever.load
    def serialized_load(self):
        with loading:
            return previous_load(self)
    Retriever.load = serialized_load
    base.ScopedAsk = RecoveryAsk
    base.OUT = target
    result = base.run()
    base.emit(result)


if __name__ == '__main__':
    main()
