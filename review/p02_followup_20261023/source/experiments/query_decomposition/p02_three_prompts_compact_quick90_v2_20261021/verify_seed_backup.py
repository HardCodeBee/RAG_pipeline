"""Verify SQLite backup contents before recording its changed physical hash."""
from contextlib import closing
import shutil

from . import run as base
from .finish_recovery import SOURCE, TARGET


def main():
    with closing(base.ro(SOURCE / 'seed.sqlite3')) as db:
        db.execute('ATTACH DATABASE ? AS copied', ((TARGET / 'seed.sqlite3').resolve().as_uri() + '?mode=ro',))
        tables = ['requests', 'retrieval', 'blocked']
        counts = {}
        for table in tables:
            for left, right in [('main', 'copied'), ('copied', 'main')]:
                assert db.execute(f'SELECT count(*) FROM (SELECT * FROM {left}.{table} EXCEPT SELECT * FROM {right}.{table})').fetchone()[0] == 0
            counts[table] = db.execute(f'SELECT count(*) FROM main.{table}').fetchone()[0]
    old = base.read(TARGET / 'OFFLINE_CHECK.json')
    assert old['seed_sha256'] == base.sha(SOURCE / 'seed.sqlite3')
    assert not (TARGET / 'OFFLINE_CHECK.before_seed_verification.json').exists()
    shutil.copyfile(TARGET / 'OFFLINE_CHECK.json', TARGET / 'OFFLINE_CHECK.before_seed_verification.json')
    record = {'utc': base.utc(), 'all_seed_tables_equal_both_directions': True,
        'rows': counts, 'source_sha256': base.sha(SOURCE / 'seed.sqlite3'),
        'backup_sha256': base.sha(TARGET / 'seed.sqlite3'),
        'change': 'SQLite backup physical file hash differs; exact payloads, responses, retrieval and block rows are identical.',
        'new_provider_requests_before_fix': 0}
    base.save(TARGET / 'SEED_BACKUP_VERIFICATION.json', record)
    old['seed_sha256'] = record['backup_sha256']
    old['seed_backup_verification'] = str(TARGET / 'SEED_BACKUP_VERIFICATION.json')
    base.save(TARGET / 'OFFLINE_CHECK.json', old)
    base.emit(record)


if __name__ == '__main__':
    main()
