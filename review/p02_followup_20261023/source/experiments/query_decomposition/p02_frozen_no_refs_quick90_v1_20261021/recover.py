"""Explicitly approved bounded recovery; source receipts and reserves survive."""
from . import run as r
from contextlib import closing
from pathlib import Path
from threading import Lock
import argparse
import hashlib
import json
import shutil
import sqlite3
import time

from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021.run_r002 import RecordingLLM
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.cache import canonical

SUBMIT_LOCK = Lock()
LAST_START = 0.0
APPROVED_429 = set()


class SerialLLM(RecordingLLM):
    def request_once(self, payload):
        global LAST_START
        with SUBMIT_LOCK:
            pause = max(0, LAST_START + 3 - time.monotonic())
            if pause:
                time.sleep(pause)
            LAST_START = time.monotonic()
            return super().request_once(payload)


def retire(db, request_id, authority):
    row = db.execute('SELECT * FROM requests WHERE id=?', (request_id,)).fetchone()
    assert row is not None and row[3] in {'failed', 'unknown'}
    db.execute('CREATE TABLE IF NOT EXISTS retired_requests AS SELECT * FROM requests WHERE 0')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS retired_request_ids ON retired_requests(id)')
    db.execute('CREATE TABLE IF NOT EXISTS retired_stages(request_id TEXT,stage TEXT,authority TEXT)')
    db.execute('INSERT INTO retired_requests VALUES(?,?,?,?,?,?,?)', row)
    db.executemany('INSERT INTO retired_stages VALUES(?,?,?)', [(request_id, stage, authority) for (stage,) in db.execute('SELECT stage FROM request_stages WHERE request_id=?', (request_id,))])
    db.execute('DELETE FROM request_stages WHERE request_id=?', (request_id,))
    db.execute('DELETE FROM requests WHERE id=?', (request_id,))
    db.commit()


class ApprovedAsk(r.RecoveryAsk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        previous = self.client
        self.client = SerialLLM(self.settings, client=previous.client)
        previous.owns_client = False
        self.client.owns_client = True

    def __call__(self, stage, prompt, maximum, allow_final_truncation=False):
        payload = canonical(build_payload(self.settings, prompt, maximum))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        while True:
            try:
                return super().__call__(stage, prompt, maximum, allow_final_truncation=allow_final_truncation)
            except RuntimeError:
                row = self.cache.db.execute('SELECT id,status,result FROM requests WHERE payload=?', (payload,)).fetchone()
                result = json.loads(row[2]) if row and row[2] else {}
                if digest not in APPROVED_429 or not row or row[1] != 'failed' or result.get('http_status') != 429:
                    raise
                attempts = 1 + self.cache.db.execute('SELECT count(*) FROM retired_requests WHERE payload=?', (payload,)).fetchone()[0]
                if attempts >= 5:
                    raise
                retire(self.cache.db, row[0], 'approved_429_maximum5_including_initial')
                self.events({'event': 'approved_HTTP429_recovery', 'stage': stage, 'retired_request_id': row[0], 'attempts_used': attempts, 'maximum': 5})
                # No repeated POST by the SDK. Each next attempt gets its own receipt/reserve.
                time.sleep(min(10 * attempts, 40))


def main():
    global APPROVED_429
    r.check()
    original = r.OUT
    proposal = r.read(original / 'RECOVERY_PROPOSAL.json')
    authority = r.read(original / 'RECOVERY_APPROVED.json')
    assert authority['user_answer'] == '允许上述有限恢复'
    approved = {v['request_id']: v for v in proposal['affected_requests']}
    APPROVED_429 = {v['payload_sha256'] for v in approved.values() if v['http_status'] == 429}
    assert len(APPROVED_429) == 2
    assert [v['request_id'] for v in approved.values() if v['status'] == 'unknown'] == ['6587b6ba8d6241579b569f3e689641db']
    target = original / 'approved_recovery001'
    assert not target.exists(), 'Do not repeat an approved unknown reissue'
    target.mkdir()
    for name in ['control90.jsonl', 'OFFLINE_CHECK.json']:
        shutil.copyfile(original / name, target / name)
    retired = []
    for ds in r.engine.DATASETS:
        source = original / 'candidate' / ds
        destination = target / 'candidate' / ds
        destination.mkdir(parents=True)
        with closing(sqlite3.connect((source / 'checkpoint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as src, closing(sqlite3.connect(destination / 'checkpoint.sqlite3')) as db:
            src.backup(db)
            bad = db.execute("SELECT id,payload,status,result FROM requests WHERE status!='success'").fetchall()
            for rid, payload, status, result in bad:
                assert rid in approved and hashlib.sha256(payload.encode()).hexdigest() == approved[rid]['payload_sha256']
                proof = json.loads(result)
                assert status == 'unknown' or status == 'failed' and proof['http_status'] == 429
                retire(db, rid, 'explicit_one_time_unknown_reissue' if status == 'unknown' else 'approved_429_maximum5_including_initial')
                retired.append(rid)
        shutil.copyfile(source / 'trace.jsonl', destination / 'trace.jsonl')
    assert set(retired) == set(approved)
    receipt = {'utc': r.utc(), 'source_receipts_and_reserves_unchanged': True, 'retired_only_in_new_copies': retired,
        'provider_concurrency': 1, 'minimum_submission_interval_seconds': 3, 'SDK_POST_retries': 0,
        'known429_retry_only_approved_payloads': sorted(APPROVED_429), 'runner_sha256': r.sha(Path(__file__)),
        'main_evaluation_remains_paused': True}
    r.save(target / 'RECOVERY_AUTHORITY.json', receipt)
    r.OUT = target
    r.RecoveryAsk = ApprovedAsk
    r.emit(r.run())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    main()
