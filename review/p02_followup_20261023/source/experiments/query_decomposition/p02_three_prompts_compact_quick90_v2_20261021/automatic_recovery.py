"""User-approved bounded unknown-request recovery for this bundle only."""
from contextlib import closing
from pathlib import Path
from threading import Lock
import argparse
import hashlib
import json
import shutil
import sqlite3
import time

from . import run as base
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload

TARGET = base.OUT / 'bounded_recovery002'
RETRY_LOCK = Lock()
PENDING = {}


def claim(payload, old_id, source):
    digest = hashlib.sha256(payload.encode('utf8')).hexdigest()
    with RETRY_LOCK:
        path = TARGET / 'REISSUES.jsonl'
        previous = base.rows(path) if path.exists() else []
        assert len(previous) < 10, 'Approved whole-run reissue limit reached'
        assert digest not in {v['payload_sha256'] for v in previous}, 'This exact payload already used its one approved reissue'
        with path.open('a', encoding='utf8') as h:
            h.write(json.dumps({'utc': base.utc(), 'payload_sha256': digest,
                'old_request_id': old_id, 'old_receipt_source': source,
                'one_time_reissue': True, 'old_budget_reservation_retained': True}) + '\n')


class BoundedAsk(base.RecoveryAsk):
    def __call__(self, stage, prompt, maximum, allow_final_truncation=False):
        payload = base.canonical(build_payload(self.settings, prompt, maximum))
        digest = hashlib.sha256(payload.encode('utf8')).hexdigest()
        row = self.cache.db.execute('SELECT id,status FROM requests WHERE payload=?', (payload,)).fetchone()
        if digest in PENDING and row is None:
            old = PENDING[digest]
            claim(payload, old['request_id'], old['source'])
        try:
            return super().__call__(stage, prompt, maximum, allow_final_truncation)
        except RuntimeError:
            row = self.cache.db.execute('SELECT id,status,result FROM requests WHERE payload=?', (payload,)).fetchone()
            if row is None or row[1] != 'unknown':
                raise
            value = json.loads(row[2] or '{}')
            if value.get('response_id') or value.get('raw_response') or value.get('usage'):
                raise RuntimeError('Receipt contains possible completed output; reconcile rather than replay') from None
            archive = TARGET / 'retired_receipts' / (row[0] + '.sqlite3')
            archive.parent.mkdir(exist_ok=True)
            assert not archive.exists()
            with closing(sqlite3.connect(archive)) as backup:
                self.cache.db.backup(backup)
            claim(payload, row[0], str(archive))
            # Exact database backup above and the old cumulative charge survive.
            # Only the new working copy drops the payload uniqueness blocker.
            with self.cache.db:
                self.cache.db.execute('DELETE FROM request_stages WHERE request_id=?', (row[0],))
                self.cache.db.execute('DELETE FROM requests WHERE id=?', (row[0],))
            self.events({'event': 'approved_once_unknown_reissue', 'old_request_id': row[0],
                'receipt_backup': str(archive), 'payload_sha256': digest})
            time.sleep(5)
            return super().__call__(stage, prompt, maximum, allow_final_truncation)


def prepare():
    base.check()
    assert not TARGET.exists(), 'Preserve existing recovery outputs'
    source = base.OUT / 'continuation001'
    TARGET.mkdir()
    for name in ['control90.jsonl', 'OFFLINE_CHECK.json']:
        shutil.copyfile(source / name, TARGET / name)
    with closing(base.ro(source / 'seed.sqlite3')) as src, closing(sqlite3.connect(TARGET / 'seed.sqlite3')) as dst:
        src.backup(dst)
        held = base.read(source / 'CONTINUATION.json')['held_records']
        for record in held:
            dst.execute('DELETE FROM blocked WHERE payload=?', (record['payload'],))
        dst.commit()
    for ds in base.common.DATASETS:
        srcdir, dstdir = source / 'candidate' / ds, TARGET / 'candidate' / ds
        dstdir.mkdir(parents=True)
        with closing(base.ro(srcdir / 'checkpoint.sqlite3')) as src, closing(sqlite3.connect(dstdir / 'checkpoint.sqlite3')) as dst:
            src.backup(dst)
        shutil.copyfile(srcdir / 'trace.jsonl', dstdir / 'trace.jsonl')
    protocol = {'utc': base.utc(), 'authority': 'User explicit reply: 允许上述有限自动恢复',
        'question': '每条最多补发1次，全轮最多补发10次；先前获准那条计入上限；保留所有旧回执与费用预留',
        'maximum_reissues_per_exact_payload': 1, 'maximum_reissues_whole_experiment': 10,
        'only_unknown_connection_failures_without_any_returned_output': True,
        'SDK_POST_retries': 0, 'connect_phase_retries': 2, 'keepalive_connections': 0,
        'provider_concurrency': 1, 'minimum_submission_interval_seconds': 3,
        'retry_delay_seconds': 5, 'successful_results_reused': True,
        'algorithm_failures_not_retried': True, 'source': str(source),
        'pending_previously_retired': [{**r, 'source': str(base.OUT / 'candidate' / r['dataset'] / 'checkpoint.sqlite3')} for r in held],
        'runner_sha256': base.sha(Path(__file__)), 'main_evaluation_remains_paused': True}
    base.save(TARGET / 'BOUNDED_RECOVERY_AUTHORITY.json', protocol)
    offline = base.read(TARGET / 'OFFLINE_CHECK.json')
    offline['seed_sha256'] = base.sha(TARGET / 'seed.sqlite3')
    offline['bounded_recovery_authority'] = str(TARGET / 'BOUNDED_RECOVERY_AUTHORITY.json')
    base.save(TARGET / 'OFFLINE_CHECK.json', offline)
    with (base.PLAN / 'DECISIONS.jsonl').open('a', encoding='utf8') as h:
        h.write(json.dumps({'id': base.HERE.name.upper() + '_BOUNDED_RECOVERY_APPROVED', **protocol}, ensure_ascii=False) + '\n')
    base.emit({'prepared': str(TARGET), 'user_approved_reissue_limit': 10})


def run():
    authority = base.read(TARGET / 'BOUNDED_RECOVERY_AUTHORITY.json')
    assert authority['runner_sha256'] == base.sha(Path(__file__))
    for r in authority['pending_previously_retired']:
        PENDING[hashlib.sha256(r['payload'].encode('utf8')).hexdigest()] = r
    base.RecoveryAsk, base.OUT = BoundedAsk, TARGET
    result = base.run()
    base.emit(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    globals()[args.action]()
