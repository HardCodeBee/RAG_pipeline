"""Resume baseline-only v2 with SQLite guards, bounded recovery and CNY300 cap."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

from experiments.query_decomposition.resume_v2_primary_20261020 import network_resume as network
from . import sqlite_guard as guard

base = network.base
PREVIOUS = base.OUT
base.OUT = base.PARENT / 'resume_v2_sqlite_20261021'
base.MODULE = 'experiments.query_decomposition.resume_v2_sqlite_20261021.run'
PAUSE = PREVIOUS / 'OUR_METHOD_PAUSED_20261021.json'
ALLOWED = {'b00_rag_top10_v1', 'b01_rag_top10_cot3', 'b02_selfask_dense',
           'b03_react_dense', 'b04_ircot_bge', 'b05_iter_retgen_bge',
           'b06_decomp_bge', 'b07_hybrid_rerank', 'b08_qd_rerank'}
INSTALLED = None


def signatures():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(guard.__file__)]}


def scope():
    pause = base.read(PAUSE)
    if not pause['active'] or set(pause['continue_methods']) != ALLOWED:
        raise RuntimeError('Baseline-only scope or own-method pause changed')
    for dataset in ['hotpotqa', '2wiki', 'musique']:
        cfg = base.read(base.V2.parent / 'campaign_api_v1' / dataset / 'BASELINE_CONFIGS_FROZEN.json')
        if set(cfg['selected']) != ALLOWED:
            raise RuntimeError('Frozen driver no longer contains exactly nine allowed baselines')


def install():
    global INSTALLED
    if INSTALLED is None:
        INSTALLED = network.install()
        guard.install(INSTALLED[3])
    return INSTALLED


base.install_budget_overlay = install


def prepare():
    base.OUT.mkdir(parents=True, exist_ok=True)
    scope()
    for marker in [base.OUT/'STOP_REQUESTED', base.V2/'STOP_REQUESTED']:
        if marker.exists():
            raise RuntimeError('Stop marker present; no automatic clearing')
    path = base.OUT/'PROTOCOL.json'
    if path.exists():
        if base.read(path)['source_sha256'] != signatures():
            raise RuntimeError('Recovery source changed; create a new version')
        return
    for name in ['SUPERVISION.json', 'PROGRESS.json', 'WORKER.json']:
        old = PREVIOUS/name
        if old.exists():
            base.write(base.OUT/('BEFORE_'+name), base.read(old))
    base.write(base.OUT/'BEFORE_INVENTORY.json', base.inventory())
    base.write(path, {
        'utc': base.utc(), 'source_sha256': signatures(),
        'authorization': 'User requested resume, monitoring and continued progress on 2026-10-21.',
        'scope': 'Nine independent v2 baselines only; own methods, controls and ablations remain paused.',
        'own_method_pause_receipt': str(PAUSE),
        'cumulative_cap_cny': 300, 'submission_guard_cny': 299.9,
        'changes': ['Materialize SQL reads before any later writes.',
                    'Serialize per-file local transactions; retain eight concurrent API slots.',
                    'Retry only safe local busy operations; never retry a generation from this guard.',
                    'Backfill missing token-attempt metadata only from complete saved attempt rows.',
                    'Bounded restart after drained SQLite errors; stop on integrity or budget failure.'],
        'unchanged': ['datasets and sample IDs', 'method prompts and parameters', 'model and API endpoint',
                      'retrieval corpus, index and scoring', 'first-complete-response and unknown-liability policies',
                      'old request, response, prediction and budget records', 'own-method pause'],
        'monitor': {'poll_seconds': 20, 'inventory_seconds': 60, 'lease_hours': 48,
                    'max_network_recoveries': 2, 'max_sqlite_recoveries': 3,
                    'chat_notifications_scheduled': False}})


base.prepare = prepare


def validate():
    base.OUT.mkdir(parents=True, exist_ok=True)
    base.validate()  # Original freeze checks and actual atomic CNY300 budget tests.
    checks = {}
    with tempfile.TemporaryDirectory(prefix='sqlite_validation_', dir=base.OUT) as temp:
        path = Path(temp)/'snapshot.sqlite3'
        a = sqlite3.connect(path)
        a.execute('pragma journal_mode=WAL')
        a.execute('create table items(id integer primary key, value text)')
        a.executemany('insert into items values(?,?)', [(1, 'a'), (2, 'b')])
        a.commit()
        b = sqlite3.connect(path)
        cursor = a.execute('select * from items')
        cursor.fetchone()  # Keep a real active read snapshot, as in the failed loop.
        b.execute("insert into items values(3,'c')"); b.commit()
        try:
            a.execute("insert into items values(4,'d')")
        except sqlite3.OperationalError as error:
            checks['unpatched_busy_snapshot_reproduced'] = getattr(error, 'sqlite_errorcode', None)
        else:
            raise AssertionError('Expected WAL snapshot conflict was not reproduced')
        a.rollback(); cursor.close(); a.close(); b.close()
        a = guard.Connection(sqlite3.connect(path), path)
        b = guard.Connection(sqlite3.connect(path), path)
        cursor = a.execute('select * from items')
        cursor.fetchone()
        b.execute("insert into items values(4,'d')"); b.commit()
        a.execute("insert into items values(5,'e')"); a.commit()
        assert len(cursor.fetchall()) == 2
        assert a.execute('select count(*) from items').fetchone()[0] == 5
        checks['same_snapshot_scenario_fixed_without_repeating_calls'] = True
        a.close(); b.close()
        gate = threading.Barrier(8)
        def writer(worker):
            db = guard.Connection(sqlite3.connect(path), path)
            gate.wait(timeout=10)
            try:
                for i in range(30):
                    old = db.execute('select * from items')
                    old.fetchone()  # An unread Python iterator no longer owns a SQLite cursor.
                    db.execute('begin immediate')
                    db.execute('insert into items values(?,?)', (100+worker*30+i, str(worker)))
                    db.commit()
            finally:
                db.close()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(writer, range(8)))
        with sqlite3.connect(path) as db:
            assert db.execute('select count(*) from items').fetchone()[0] == 245
            assert db.execute('pragma integrity_check').fetchone()[0] == 'ok'
        db.close()  # sqlite3's context manager commits; it does not close on Windows.
        checks['eight_workers_240_committed_transactions_no_missing_rows'] = True

        # Exercise the real immutable Store and v2 Transport for saved success
        # and unknown responses. A client/budget call would fail this test.
        common, runtime, matched, Budget = install()
        from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
        from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.transport import Transport
        store = Store(Path(temp)/'real_store', {'purpose': 'offline immutable replay validation'})
        class Never:
            def __getattr__(self, name):
                raise AssertionError('Offline replay attempted an API or budget action: '+name)
        class Catalog:
            def request(self, *args):
                raise AssertionError('Existing private request must not consult history')
        transport = Transport(store, Never(), Never(), common.MODEL, Catalog())
        transport.qid = 'offline-replay'
        for status in ['success', 'unknown']:
            prompt = 'offline '+status
            payload = {'model': common.MODEL, 'messages': [{'role':'user','content':prompt}],
                       'temperature':0., 'max_tokens':common.MAX_OUTPUT,
                       'extra_body':{'enable_thinking':False}}
            key = common.digest([store.namespace, payload])
            result = {'status':status, 'answer':'alpha' if status=='success' else '',
                      'finish_reason':'stop' if status=='success' else None,
                      'usage':{'prompt_tokens':5,'completion_tokens':1} if status=='success' else None}
            store.db.execute('insert into requests values(?,?,?,?,?)',
                             (key, 'offline', common.canonical(payload), status, common.canonical(result)))
            store.db.commit()
            for _ in range(2):
                actual_key, actual = transport.ask('offline-'+status, prompt, 512)
                assert actual_key == key and actual == result
        store.put('prediction', 'immutable', {'qid':'immutable','prediction':'alpha'})
        try:
            store.put('prediction', 'immutable', {'qid':'immutable','prediction':'beta'})
        except ValueError:
            checks['different_immutable_prediction_still_rejected'] = True
        else:
            raise AssertionError('Immutable Store protection was relaxed')
        assert store.db.execute('select count(*) from attempts').fetchone()[0] == 0
        checks['saved_success_and_unknown_replay_zero_api_attempts'] = True
        store.db.close()
    scope()
    checks['own_method_pause_and_exact_nine_baselines_verified'] = True
    base.write(base.OUT/'SQLITE_VALIDATION.json', {'utc':base.utc(), 'passed':True,
               'source_sha256':signatures(), 'api_calls':0, 'checks':checks})
    print(json.dumps({'passed':True,'api_calls':0,'checks':checks}), flush=True)


def repair_attempt_metadata():
    receipt = base.OUT/'ATTEMPT_METADATA_REPAIR.json'
    if receipt.exists():
        return
    path = base.V2/'matched50/hotpotqa/formal/b05_iter_retgen_bge/k10_t5/checkpoint.sqlite3'
    repaired, unresolved = [], []
    with sqlite3.connect(path) as db:
        db.execute('begin immediate')
        def immutable_digest():
            h = hashlib.sha256()
            for sql in ['select * from requests order by id', 'select * from attempts order by id',
                        "select * from data where kind='prediction' order by key"]:
                for row in db.execute(sql):
                    h.update(json.dumps(row, ensure_ascii=False).encode('utf8'))
            return h.hexdigest()
        before = immutable_digest()
        pending = db.execute("select a.id,a.request_id,a.result,r.stage from attempts a join requests r on r.id=a.request_id "
                    "where a.result is not null and not exists(select 1 from data d where d.kind='token_attempt' and d.key=cast(a.id as text))").fetchall()
        for aid, rid, raw, stage in pending:
            qids = {json.loads(v)['qid'] for v, in db.execute(
                "select value from data where kind in ('evidence','token_call') and json_extract(value,'$.request_id')=?", (rid,)).fetchall()}
            if len(qids) != 1:
                unresolved.append({'attempt_id':aid, 'candidate_qids':sorted(qids)})
                continue
            result = json.loads(raw)
            value = {'qid':next(iter(qids)), 'stage':stage, 'request_id':rid, 'attempt_id':aid,
                     'status':result['status'], 'usage':result.get('usage'),
                     'finish_reason':result.get('finish_reason'),
                     'usage_missing_is_unknown_not_zero':not bool(result.get('usage'))}
            db.execute('insert into data values(?,?,?)', ('token_attempt',str(aid),json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))))
            repaired.append(value)
        assert immutable_digest() == before
        db.commit()
    base.write(receipt, {'utc':base.utc(),'checkpoint':str(path),'api_calls':0,
               'requests_attempts_predictions_unchanged_sha256':before,
               'budget_or_unknown_reserves_modified':False,
               'repaired':repaired,'unresolved':unresolved})


def check_validation():
    v = base.read(base.OUT/'SQLITE_VALIDATION.json')
    if not v['passed'] or v['source_sha256'] != signatures():
        raise RuntimeError('Matching SQLite offline validation required')


def supervise():
    from experiments.query_decomposition.iter_react_matched50_watch_20261017 import network_stop_only
    with base.lock('supervisor.lock'):
        prepare()
        check_validation()
        checks = network.preflight()
        base.write(base.OUT/'NETWORK_PREFLIGHT.json', {'utc':base.utc(),**checks})
        repair_attempt_metadata()
        deadline = time.monotonic()+48*3600
        recoveries = {'network':0,'sqlite':0}
        while time.monotonic()<deadline:
            scope()
            if (base.OUT/'STOP_REQUESTED').exists() or (base.V2/'STOP_REQUESTED').exists():
                base.write(base.OUT/'SUPERVISION.json', {'utc':base.utc(),'status':'user_stop_no_restart'})
                return
            with (base.OUT/'worker.stdout.log').open('a',encoding='utf8') as stdout, (base.OUT/'worker.stderr.log').open('a',encoding='utf8') as stderr:
                child = subprocess.Popen([sys.executable,'-X','utf8','-u','-B','-m',base.MODULE,'--worker'],
                        cwd=base.ROOT,stdout=stdout,stderr=stderr,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            last_inventory = 0.
            while child.poll() is None:
                if (base.OUT/'STOP_REQUESTED').exists() and not (base.V2/'STOP_REQUESTED').exists():
                    base.write(base.V2/'STOP_REQUESTED',{'utc':base.utc(),'reason':'Explicit stop; drain and do not restart.'})
                progress = base.read(base.V2/'progress.json')
                status = {'utc':base.utc(),'pid':__import__('os').getpid(),'worker_pid':child.pid,
                          'status':'running','recoveries':recoveries,'generation_progress':progress,
                          'own_method_evaluations_paused':True,'chat_notifications_scheduled':False,
                          'continuation_directory':str(base.OUT)}
                base.write(base.OUT/'SUPERVISION.json',status)
                base.write(base.V2/'SUPERVISION.json',status)
                if time.monotonic()-last_inventory>=60:
                    try:
                        base.write(base.OUT/'PROGRESS.json',base.inventory())
                    except (sqlite3.Error,OSError) as error:
                        base.write(base.OUT/'MONITOR_IO_ERROR.json',{'utc':base.utc(),'error':type(error).__name__})
                    last_inventory=time.monotonic()
                time.sleep(20)
            base.write(base.OUT/'PROGRESS.json',base.inventory())
            progress=base.read(base.V2/'progress.json')
            reason=progress.get('reason','')
            category=None
            if child.returncode==0 and progress.get('status')=='completed':
                state='completed'
            elif 'database is locked' in reason.casefold() and recoveries['sqlite']<3:
                category='sqlite'
            elif reason.startswith('Transport stopped;') and recoveries['network']<2 and network_stop_only():
                category='network'
            else:
                state='needs_attention_no_auto_restart'
            if category and not (base.OUT/'STOP_REQUESTED').exists() and not (base.V2/'STOP_REQUESTED').exists():
                recoveries[category]+=1
                with (base.OUT/'RECOVERIES.jsonl').open('a',encoding='utf8') as stream:
                    stream.write(json.dumps({'utc':base.utc(),'category':category,'reason':reason,
                                 'recoveries':recoveries,'worker_drained':True,
                                 'completed_private_responses_reused':True,
                                 'old_unknowns_not_resubmitted':True})+'\n')
                time.sleep(10)
                continue
            status={'utc':base.utc(),'status':state,'worker_returncode':child.returncode,
                    'recoveries':recoveries,'generation_progress':progress,'continuation_directory':str(base.OUT)}
            base.write(base.OUT/'SUPERVISION.json',status)
            base.write(base.V2/'SUPERVISION.json',status)
            return
        base.write(base.OUT/'SUPERVISION.json',{'utc':base.utc(),'status':'monitor_lease_expired'})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    base.OUT.mkdir(parents=True,exist_ok=True)
    if args.validate_only:
        validate()
    elif args.worker:
        check_validation()
        base.worker()
    else:
        supervise()


if __name__=='__main__':
    main()
