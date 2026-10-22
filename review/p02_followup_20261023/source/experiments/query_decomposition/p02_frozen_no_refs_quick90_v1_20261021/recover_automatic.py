"""User-authorized transient recovery without per-request approval, max5 total."""
from . import run as r
from . import recover
from . import recover_extension as extension
from contextlib import closing
from pathlib import Path
import argparse
import json
import shutil
import sqlite3
import time


class AutomaticAsk(r.RecoveryAsk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        previous = self.client
        self.client = recover.SerialLLM(self.settings, client=previous.client)
        previous.owns_client = False
        self.client.owns_client = True

    def __call__(self, stage, prompt, maximum, allow_final_truncation=False):
        payload = recover.canonical(recover.build_payload(self.settings, prompt, maximum))
        while True:
            try:
                return super().__call__(stage, prompt, maximum, allow_final_truncation=allow_final_truncation)
            except RuntimeError:
                row = self.cache.db.execute('SELECT id,status,result FROM requests WHERE payload=?', (payload,)).fetchone()
                result = json.loads(row[2]) if row and row[2] else {}
                transient = row and (row[1] == 'failed' and result.get('http_status') in {408, 429, 500, 502, 503, 504}
                    or row[1] == 'unknown' and result.get('failure_class') in {'APIConnectionError', 'APITimeoutError'})
                if not transient:
                    raise
                attempts = 1 + self.cache.db.execute('SELECT count(*) FROM retired_requests WHERE payload=?', (payload,)).fetchone()[0]
                if attempts >= 5:
                    self.events({'event': 'automatic_recovery_limit_reached', 'stage': stage, 'attempts': attempts, 'maximum': 5})
                    raise
                recover.retire(self.cache.db, row[0], 'user_authorized_automatic_transient_recovery_maximum5_total')
                self.events({'event': 'automatic_transient_recovery', 'stage': stage, 'retired_request_id': row[0],
                    'old_status': row[1], 'attempts_used': attempts, 'maximum': 5})
                time.sleep(min(5 * attempts, 30))


def main():
    r.check()
    root = r.OUT
    authority = r.read(root / 'AUTONOMOUS_RECOVERY_APPROVED.json')
    assert authority['authority'] == 'User: 中断就自己恢复 无需向我批准'
    source = root / 'approved_recovery002'
    assert r.read(source / 'REPORT.json')['status'] != 'completed', 'No need to rerun a completed experiment'
    target = root / 'automatic_recovery003'
    assert not target.exists(), 'Never reset recorded attempts across restarts'
    target.mkdir()
    for name in ['control90.jsonl', 'OFFLINE_CHECK.json', 'ordinary_CONTROL.txt', 'ordinary_CANDIDATE.txt', 'ordinary_CHANGE.diff', 'goal_CONTROL.txt', 'goal_CANDIDATE.txt', 'goal_CHANGE.diff']:
        shutil.copyfile(source / name, target / name)
    for ds in r.engine.DATASETS:
        destination = target / 'candidate' / ds
        destination.mkdir(parents=True)
        with closing(sqlite3.connect((source / 'candidate' / ds / 'checkpoint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as old, closing(sqlite3.connect(destination / 'checkpoint.sqlite3')) as new:
            old.backup(new)
        shutil.copyfile(source / 'candidate' / ds / 'trace.jsonl', destination / 'trace.jsonl')
    record = {'utc': r.utc(), 'runner_sha256': r.sha(Path(__file__)), 'authority': authority['authority'],
        'maximum_total_actual_attempts_per_payload': 5, 'previous_archived_attempts_counted': True,
        'source_receipts_and_reserves_preserved': True, 'provider_concurrency': 1, 'minimum_interval_seconds': 3,
        'algorithm_errors_not_retried': True, 'prompts_and_generation_payloads_unchanged': True,
        'main_evaluation_remains_paused': True}
    r.save(target / 'RECOVERY_AUTHORITY.json', record)
    r.OUT, r.RecoveryAsk = target, AutomaticAsk
    r.engine.rows = extension.filtered_rows
    while True:
        r.engine.STOP.clear()
        try:
            result = r.run()
            break
        except RuntimeError:
            newly_held = extension.unresolved_cases(target) - extension.HELD
            if not newly_held:
                raise
            extension.HELD.update(newly_held)
            r.save(target / 'HELD_WITHOUT_FURTHER_REPLAY.json', {'utc': r.utc(), 'ids': sorted(extension.HELD), 'full_denominator': 90})
    r.emit({'result': result, 'held_after_attempt_limit': sorted(extension.HELD)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    main()
