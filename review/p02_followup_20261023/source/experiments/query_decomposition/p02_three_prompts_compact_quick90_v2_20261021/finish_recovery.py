"""Complete this quick90 under the user's delegated recovery authority."""
from contextlib import closing
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sqlite3
import time

from . import run as base
from . import automatic_recovery as recovery
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload

SOURCE = base.OUT / 'bounded_recovery002'
TARGET = base.OUT / 'delegated_recovery003'


def claim(payload, old_id, source):
    digest = hashlib.sha256(payload.encode('utf8')).hexdigest()
    with recovery.RETRY_LOCK:
        path = TARGET / 'REISSUES.jsonl'
        previous = base.rows(path) if path.exists() else []
        count = sum(v['payload_sha256'] == digest for v in previous)
        assert count < 4, 'Five actual submissions exhausted; preserve and inspect this stage'
        with path.open('a', encoding='utf8') as h:
            h.write(json.dumps({'utc': base.utc(), 'payload_sha256': digest,
                'old_request_id': old_id, 'old_receipt_source': source,
                'actual_submission_attempt': count + 2,
                'authority': 'User: 任意补跑 不需要我的允许',
                'old_budget_reservation_retained': True}) + '\n')


class PersistentAsk(base.RecoveryAsk):
    def __call__(self, stage, prompt, maximum, allow_final_truncation=False):
        payload = base.canonical(build_payload(self.settings, prompt, maximum))
        for iteration in range(5):
            try:
                return super().__call__(stage, prompt, maximum, allow_final_truncation)
            except RuntimeError:
                row = self.cache.db.execute('SELECT status,result FROM requests WHERE payload=?', (payload,)).fetchone()
                if row is None:
                    raise
                value = json.loads(row[1] or '{}')
                eligible = row[0] == 'unknown' or row[0] == 'failed' and value.get('http_status') in {429, 500, 502, 503, 504}
                if not eligible:
                    raise
                if value.get('response_id') or value.get('raw_response') or value.get('usage'):
                    raise
                if iteration == 4:
                    raise
                rid = self.cache.db.execute('SELECT id FROM requests WHERE payload=?', (payload,)).fetchone()[0]
                archive = TARGET / 'retired_receipts' / (rid + '.sqlite3')
                archive.parent.mkdir(exist_ok=True)
                assert not archive.exists()
                with closing(sqlite3.connect(archive)) as backup:
                    self.cache.db.backup(backup)
                claim(payload, rid, str(archive))
                with self.cache.db:
                    self.cache.db.execute('DELETE FROM request_stages WHERE request_id=?', (rid,))
                    self.cache.db.execute('DELETE FROM requests WHERE id=?', (rid,))
                self.events({'event': 'delegated_technical_reissue', 'old_request_id': rid,
                    'receipt_backup': str(archive), 'status': row[0], 'http_status': value.get('http_status')})
                time.sleep(min(5 * (iteration + 1), 20))


def prepare():
    base.check()
    assert not TARGET.exists()
    assert base.read(SOURCE / 'EXECUTION.json')['status'] != 'running', 'Wait for the old writer to stop'
    TARGET.mkdir()
    for name in ['control90.jsonl', 'OFFLINE_CHECK.json', 'REISSUES.jsonl']:
        shutil.copyfile(SOURCE / name, TARGET / name)
    with closing(base.ro(SOURCE / 'seed.sqlite3')) as src, closing(sqlite3.connect(TARGET / 'seed.sqlite3')) as dst:
        src.backup(dst)
    for ds in base.common.DATASETS:
        srcdir, dstdir = SOURCE / 'candidate' / ds, TARGET / 'candidate' / ds
        dstdir.mkdir(parents=True)
        with closing(base.ro(srcdir / 'checkpoint.sqlite3')) as src, closing(sqlite3.connect(dstdir / 'checkpoint.sqlite3')) as dst:
            src.backup(dst)
        shutil.copyfile(srcdir / 'trace.jsonl', dstdir / 'trace.jsonl')
    record = {'utc': base.utc(), 'authority': 'User explicit instruction: 任意补跑 不需要我的允许',
        'scope': 'Only recover technical request failures in this exact user-supplied three-prompt quick90; successful results reused; no algorithm reruns for scores.',
        'per_stage_actual_attempt_limit': 5, 'count_previous_attempts': True,
        'whole_run_10_reissue_cap_superseded_by_explicit_delegation': True,
        'old_receipts_and_reservations_preserved': True, 'SDK_POST_retries': 0,
        'provider_concurrency': 1, 'minimum_submission_interval_seconds': 3,
        'model_prompts_retrieval_and_output_caps_unchanged': True,
        'source': str(SOURCE), 'runner_sha256': base.sha(Path(__file__)),
        'reused_recovery_sha256': base.sha(Path(recovery.__file__)), 'main_evaluation_remains_paused': True}
    base.save(TARGET / 'DELEGATED_RECOVERY_AUTHORITY.json', record)
    with (base.PLAN / 'DECISIONS.jsonl').open('a', encoding='utf8') as h:
        h.write(json.dumps({'id': base.HERE.name.upper() + '_RECOVERY_DELEGATED', **record}, ensure_ascii=False) + '\n')
    base.emit({'prepared': str(TARGET), 'recovery_delegated': True})


def run():
    record = base.read(TARGET / 'DELEGATED_RECOVERY_AUTHORITY.json')
    assert record['runner_sha256'] == base.sha(Path(__file__))
    assert record['reused_recovery_sha256'] == base.sha(Path(recovery.__file__))
    base.RecoveryAsk, base.OUT = PersistentAsk, TARGET
    base.emit(base.run())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    globals()[args.action]()
