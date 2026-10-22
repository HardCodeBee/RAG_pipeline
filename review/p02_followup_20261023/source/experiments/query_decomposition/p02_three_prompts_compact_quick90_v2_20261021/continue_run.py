"""Continue independent cases; ambiguous resubmission requires explicit approval."""
from contextlib import closing
from pathlib import Path
import argparse
import json
import shutil
import sqlite3

from . import run as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--source', default='.')
    parser.add_argument('--target', required=True)
    parser.add_argument('--approved-id', action='append', default=[])
    args = parser.parse_args()
    base.check()
    root = base.OUT
    source, target = root / args.source, root / args.target
    assert source.resolve().is_relative_to(root.resolve()) and target.resolve().is_relative_to(root.resolve())
    if args.action == 'prepare':
        assert not target.exists(), 'Preserve previous continuation'
        target.mkdir()
        for name in ['control90.jsonl', 'OFFLINE_CHECK.json']:
            shutil.copyfile(source / name, target / name)
        with closing(base.ro(source / 'seed.sqlite3')) as src, closing(sqlite3.connect(target / 'seed.sqlite3')) as dst:
            src.backup(dst)
        unresolved = []
        # Carry forward an earlier held record even after its receipt was retired
        # only in a new checkpoint copy.
        previous = source / 'CONTINUATION.json'
        if previous.exists():
            unresolved.extend(base.read(previous)['held_records'])
        for ds in base.common.DATASETS:
            srcdir, dstdir = source / 'candidate' / ds, target / 'candidate' / ds
            dstdir.mkdir(parents=True)
            with closing(base.ro(srcdir / 'checkpoint.sqlite3')) as src, closing(sqlite3.connect(dstdir / 'checkpoint.sqlite3')) as dst:
                src.backup(dst)
                for rid, payload, status, raw in dst.execute("SELECT id,payload,status,result FROM requests WHERE status!='success'").fetchall():
                    value = json.loads(raw or '{}')
                    assert status == 'unknown', 'Inspect known failures separately'
                    failed_ids = {json.loads(p.read_text(encoding='utf8'))['id'] for p in srcdir.glob('failure_*.json')}
                    assert len(failed_ids) == 1
                    unresolved.append({'request_id': rid, 'dataset': ds, 'case_id': next(iter(failed_ids)),
                        'payload': payload, 'result': value})
                    dst.execute('DELETE FROM request_stages WHERE request_id=?', (rid,))
                    dst.execute('DELETE FROM requests WHERE id=?', (rid,))
                dst.commit()
            if (srcdir / 'trace.jsonl').exists():
                shutil.copyfile(srcdir / 'trace.jsonl', dstdir / 'trace.jsonl')
        unresolved = list({r['request_id']: r for r in unresolved}.values())
        approved = set(args.approved_id)
        assert approved <= {r['request_id'] for r in unresolved}
        with closing(sqlite3.connect(target / 'seed.sqlite3')) as db:
            for r in unresolved:
                if r['request_id'] in approved:
                    db.execute('DELETE FROM blocked WHERE payload=?', (r['payload'],))
                else:
                    db.execute('INSERT OR IGNORE INTO blocked VALUES(?,?,?)', (r['payload'], r['request_id'], 'unknown'))
            db.commit()
        record = {'utc': base.utc(), 'source': str(source), 'held_records': [r for r in unresolved if r['request_id'] not in approved],
            'explicitly_approved_one_time_reissues': sorted(approved),
            'approval_authority': 'Current side-conversation explicit reply' if approved else None,
            'old_receipts_and_reservations_preserved': True, 'successful_payloads_preserved': True,
            'prompts_model_retrieval_and_caps_unchanged': True, 'main_campaign_resumed': False,
            'runner_sha256': base.sha(Path(__file__))}
        base.save(target / 'CONTINUATION.json', record)
        offline = base.read(target / 'OFFLINE_CHECK.json')
        offline['seed_sha256'] = base.sha(target / 'seed.sqlite3')
        offline['held_request_policy_amendment'] = str(target / 'CONTINUATION.json')
        base.save(target / 'OFFLINE_CHECK.json', offline)
        with (base.PLAN / 'DECISIONS.jsonl').open('a', encoding='utf8') as h:
            h.write(json.dumps({'id': base.HERE.name.upper() + '_' + args.target.upper(),
                'utc': base.utc(), 'status': 'continue_independent_queries_only' if not approved else 'explicit_one_time_reissue_approved',
                'held_case_ids': [r['case_id'] for r in record['held_records']],
                'approved_request_ids': sorted(approved), 'main_evaluation_remains_paused': True}) + '\n')
        base.emit({'prepared': str(target), 'held_cases': len(record['held_records']), 'approved_once': sorted(approved)})
        return
    record = base.read(target / 'CONTINUATION.json')
    assert base.sha(Path(__file__)) == record['runner_sha256']
    assert not args.approved_id, 'Approval must be recorded before execution'
    held = {r['case_id'] for r in record['held_records']}
    original_rows = base.rows
    def active_rows(path):
        values = original_rows(path)
        return [r for r in values if r['id'] not in held] if Path(path).resolve() == (base.COHORT / 'runtime90.frozen.jsonl').resolve() else values
    base.rows, base.OUT = active_rows, target
    base.emit(base.run())


if __name__ == '__main__':
    main()
