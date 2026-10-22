"""Once per unknown payload, at most three extra reissues, explicitly approved."""
from . import run as r
from . import recover
from contextlib import closing
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sqlite3
import time

CLAIMS = None
HELD = set()
ORIGINAL_ROWS = r.engine.rows


class ExtensionAsk(recover.ApprovedAsk):
    def __call__(self, stage, prompt, maximum, allow_final_truncation=False):
        payload = recover.canonical(recover.build_payload(self.settings, prompt, maximum))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        while True:
            try:
                return super().__call__(stage, prompt, maximum, allow_final_truncation=allow_final_truncation)
            except RuntimeError:
                row = self.cache.db.execute('SELECT id,status,result FROM requests WHERE payload=?', (payload,)).fetchone()
                if not row or row[1] != 'unknown':
                    raise
                # An unknown receipt from an earlier attempt already retired
                # means the one allowed replay of this payload has been used.
                used = self.cache.db.execute("SELECT count(*) FROM retired_requests WHERE payload=? AND status='unknown'", (payload,)).fetchone()[0]
                if used:
                    raise
                with closing(sqlite3.connect(CLAIMS, timeout=30)) as db:
                    db.execute('BEGIN IMMEDIATE')
                    if db.execute('SELECT count(*) FROM claims WHERE payload_hash=?', (digest,)).fetchone()[0] or db.execute('SELECT count(*) FROM claims').fetchone()[0] >= 3:
                        db.rollback()
                        raise
                    db.execute('INSERT INTO claims VALUES(?,?,?,?)', (digest, row[0], stage, r.utc()))
                    db.commit()
                recover.retire(self.cache.db, row[0], 'explicit_extension_once_per_payload_maximum3_additional')
                self.events({'event': 'approved_unknown_reissue', 'stage': stage, 'retired_request_id': row[0], 'payload_sha256': digest, 'maximum_per_payload': 1, 'extension_maximum': 3})
                time.sleep(10)


def filtered_rows(path):
    values = ORIGINAL_ROWS(path)
    if Path(path).resolve() == (r.COHORT / 'runtime90.frozen.jsonl').resolve():
        return [v for v in values if v['id'] not in HELD]
    return values


def unresolved_cases(target):
    held = set()
    for ds in r.engine.DATASETS:
        directory = target / 'candidate' / ds
        with closing(sqlite3.connect((directory / 'checkpoint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as db:
            complete = {key for (key,) in db.execute("SELECT key FROM data WHERE kind='prediction'")}
        held.update(r.read(p)['id'] for p in directory.glob('failure_*.json') if r.read(p)['id'] not in complete)
    return held


def main():
    global CLAIMS
    r.check()
    root = r.OUT
    permission = r.read(root / 'RECOVERY_EXTENSION_APPROVED.json')
    assert permission['user_answer'] == '允许本轮统一限额恢复'
    source = root / 'approved_recovery001'
    target = root / 'approved_recovery002'
    assert not target.exists(), 'Do not recreate or reset the extension allowance'
    target.mkdir()
    for name in ['control90.jsonl', 'OFFLINE_CHECK.json']:
        shutil.copyfile(source / name, target / name)
    for ds in r.engine.DATASETS:
        destination = target / 'candidate' / ds
        destination.mkdir(parents=True)
        with closing(sqlite3.connect((source / 'candidate' / ds / 'checkpoint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as old, closing(sqlite3.connect(destination / 'checkpoint.sqlite3')) as new:
            old.backup(new)
        shutil.copyfile(source / 'candidate' / ds / 'trace.jsonl', destination / 'trace.jsonl')
    CLAIMS = target / 'extension_allowance.sqlite3'
    with closing(sqlite3.connect(CLAIMS)) as db:
        db.execute('CREATE TABLE claims(payload_hash TEXT PRIMARY KEY,old_request_id TEXT,stage TEXT,utc TEXT)')
        db.commit()
    proposal = r.read(root / 'RECOVERY_PROPOSAL.json')
    recover.APPROVED_429 = {v['payload_sha256'] for v in proposal['affected_requests'] if v['http_status'] == 429}
    record = {'utc': r.utc(), 'runner_sha256': r.sha(Path(__file__)), 'approved_maximum_additional_unknown_reissues': 3,
        'maximum_per_unknown_payload': 1, 'previous_unknown_reissue_not_repeated': True,
        'original_and_recovery001_receipts_unchanged': True, 'provider_concurrency': 1, 'minimum_interval_seconds': 3,
        'skip_infrastructure_failures_without_scoring_as_zero': True, 'full_scoring_denominator': 90,
        'prompts_and_generation_payloads_unchanged': True, 'main_evaluation_remains_paused': True}
    r.save(target / 'RECOVERY_AUTHORITY.json', record)
    r.OUT, r.RecoveryAsk = target, ExtensionAsk
    r.engine.rows = filtered_rows
    while True:
        r.engine.STOP.clear()
        try:
            result = r.run()
            break
        except RuntimeError:
            newly_held = unresolved_cases(target) - HELD
            if not newly_held:
                raise
            HELD.update(newly_held)
            r.save(target / 'HELD_WITHOUT_FURTHER_REPLAY.json', {'utc': r.utc(), 'ids': sorted(HELD), 'full_denominator': 90})
    with closing(sqlite3.connect(CLAIMS)) as db:
        used = db.execute('SELECT count(*) FROM claims').fetchone()[0]
    r.emit({'result': result, 'extension_reissues_used': used, 'extension_maximum': 3, 'held': sorted(HELD)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    main()
