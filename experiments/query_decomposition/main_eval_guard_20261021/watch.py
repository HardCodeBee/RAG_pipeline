"""Independent, read-only evaluation audit and bounded supervisor recovery.

Only writes its own receipts, or a STOP_REQUESTED on a verified integrity fault.
Never edits a prediction, retries a request, releases money, or kills a worker.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from experiments.query_decomposition.resume_v2_sqlite_20261021 import run as recovery

base = recovery.base
OUT = base.PARENT / 'main_eval_guard_20261021'
MODULE = 'experiments.query_decomposition.main_eval_guard_20261021.watch'


class IntegrityFault(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temp.replace(path)


def event(kind, **fields):
    with (OUT / 'events.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps({'utc': base.utc(), 'event': kind, **fields}) + '\n')


def held(path):
    """File locks remain observable when host process IDs are hidden by sandbox."""
    if not path.exists():
        return False
    with path.open('r+b') as stream:
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno not in {13, 11, 36}:
                raise
            return True
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    return False


@contextmanager
def readonly(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')  # Every checkpoint check uses one consistent snapshot.
        yield db
    finally:
        db.close()


def checkpoints():
    for dataset in ['hotpotqa', '2wiki', 'musique']:
        selected = read(base.V2.parent / 'campaign_api_v1' / dataset / 'BASELINE_CONFIGS_FROZEN.json')['selected']
        for method, config in selected.items():
            track = 'matched50' if method == 'b05_iter_retgen_bge' else 'main'
            if track == 'matched50':
                config = {'id': 'k10_t5', 'k': 10, 'iterations': 5}
            path = base.V2 / track / dataset / 'formal' / method / config['id'] / 'checkpoint.sqlite3'
            if path.exists():
                yield dataset, method, config, track, path


def audit_checkpoint(db, cases, expected):
    errors = []
    def fail(reason):
        if reason not in errors:
            errors.append(reason)
    identity = db.execute("SELECT value FROM data WHERE kind='meta' AND key='identity'").fetchone()
    if not identity or any(json.loads(identity[0]).get(k) != v for k, v in expected.items()):
        fail('checkpoint_identity_mismatch')
    bad_requests = db.execute("SELECT count(*) FROM requests WHERE status='success' AND ("
        "result IS NULL OR json_extract(result,'$.finish_reason') IS NOT 'stop' OR "
        "json_type(result,'$.usage.prompt_tokens') IS NOT 'integer' OR "
        "json_type(result,'$.usage.completion_tokens') IS NOT 'integer' OR "
        "json_extract(result,'$.returned_model') IS NOT ?)", ('qwen-plus-2025-12-01',)).fetchone()[0]
    if bad_requests:
        fail('successful_response_incomplete_or_wrong_model')
    for key, raw in db.execute("SELECT key,value FROM data WHERE kind='prediction'").fetchall():
        row = json.loads(raw)
        if key != row.get('qid') or key not in cases:
            fail('prediction_outside_frozen_cohort')
        if row.get('question') is not None and row['question'] != cases.get(key):
            fail('prediction_question_mismatch')
        if row.get('status') in {'success', 'algorithm_failure'} and not isinstance(row.get('prediction'), str):
            fail('scorable_prediction_not_text')
        if row.get('status') == 'success':
            request = db.execute('SELECT status,result FROM requests WHERE id=?', (row.get('request_id'),)).fetchone()
            if not request or request[0] != 'success':
                fail('success_without_successful_final_request')
        if (expected.get('track') == 'matched50' or expected.get('method') == 'b05_iter_retgen_bge') and row.get('iterations', 0) > 5:
            fail('matched50_search_allowance_exceeded')
    for kind, raw in db.execute("SELECT kind,value FROM data WHERE kind IN "
            "('integrity_violation','retrieval','rerank_token','token_call')"):
        row = json.loads(raw)
        if kind == 'integrity_violation':
            fail('recorded_integrity_violation')
        if kind == 'retrieval' and (row.get('encoder_truncated') or
                sum(row.get('encoded_window_tokens', [])) < row.get('query_tokens', 0)):
            fail('retrieval_input_truncated')
        if kind == 'rerank_token' and row.get('input_truncated'):
            fail('reranker_input_truncated')
        if kind == 'token_call' and row.get('status') == 'success' and (
                row.get('finish_reason') != 'stop' or not row.get('usage')):
            fail('successful_logical_call_incomplete')
    if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
        fail('sqlite_quick_check_failed')
    return errors


def frozen_check():
    # These entry points validate existing receipts only; never call prepare/run.
    from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import common
    try:
        common.verify()
        recovery.scope()
        recovery.check_validation()
    except (common.IntegrityError, ValueError, RuntimeError) as error:
        raise IntegrityFault(str(error)) from error
    for path, receipt, key in [
        (Path(base.__file__), base.OUT/'VALIDATION.json', 'source_sha256'),
        (Path(recovery.network.__file__), recovery.PREVIOUS/'NETWORK_VALIDATION.json', 'source_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != read(receipt)[key]:
            raise IntegrityFault('Previously validated recovery source changed: ' + path.name)
    from experiments.query_decomposition import iter_react_matched50_20261017 as matched
    if common.sha(Path(matched.__file__)) != read(common.OUT/'matched50/VALIDATION.json')['source_sha256']:
        raise IntegrityFault('Matched50 driver changed')
    from experiments.query_decomposition import baseline_origin_binding_fix_20261017 as origin
    if common.sha(Path(origin.__file__)) != read(origin.RECEIPT/'VALIDATION.json')['source_sha256']:
        raise IntegrityFault('Origin binding driver changed')
    trace_receipt=base.PARENT/'resume_v2_trace_20261021/VALIDATION.json'
    if trace_receipt.exists():
        validated=read(trace_receipt)
        if not validated.get('passed'):
            raise IntegrityFault('Trace reporting recovery not validated')
        for name,expected in validated['source_sha256'].items():
            path=base.ROOT/'experiments/query_decomposition/resume_v2_trace_20261021'/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise IntegrityFault('Validated trace reporting recovery source changed')
    return len(read(common.OUT/'CONFIG_FROZEN.json')['source_and_input_sha256'])


def full_audit():
    pins = frozen_check()
    checks, errors = [], []
    for dataset, method, config, track, path in checkpoints():
        cases_path = base.ROOT/'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen'/dataset/'formal.questions.jsonl'
        cases = {r['qid']:r['question'] for r in (json.loads(s) for s in cases_path.read_text(encoding='utf8').splitlines() if s.strip())}
        expected = {'dataset':dataset, 'split':'formal', 'method':method, 'config':config,
                    'track':track, 'model':'qwen-plus-2025-12-01',
                    'input_sha256':hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                    'resource_sha256':hashlib.sha256((base.V2.parent/'campaign_api_v1'/dataset/'RESOURCE_FROZEN.json').read_bytes()).hexdigest(),
                    'token_policy_sha256':hashlib.sha256((base.V2/'CONFIG_FROZEN.json').read_bytes()).hexdigest()}
        with readonly(path) as db:
            found = audit_checkpoint(db, cases, expected)
        scoring = audit_saved_scores(path.parent, dataset)
        found.extend(scoring['errors'])
        checks.append({'dataset':dataset, 'method':method, 'errors':found,
                       'saved_native_scores_checked':scoring['checked']})
        errors.extend({'dataset':dataset, 'method':method, 'reason':e} for e in found)
    recovery_root = base.PARENT/'twowiki_missing_recovery_20261021'
    recovery_plan = recovery_root/'PLAN.frozen.json'
    if recovery_plan.exists():
        plan = read(recovery_plan)
        driver = base.ROOT/'experiments/query_decomposition/twowiki_missing_recovery_20261021/run.py'
        if hashlib.sha256(driver.read_bytes()).hexdigest() != plan['source_sha256']:
            raise IntegrityFault('Authorized 2Wiki recovery driver changed')
        wanted = {(v['method'],v['case']['qid']):v for v in plan['items'] if v['method']!='p02_multigrain_timing_cachefix_v1'}
        cases_path=base.ROOT/'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/2wiki/formal.questions.jsonl'
        cases={v['qid']:v['question'] for v in (json.loads(s) for s in cases_path.read_text(encoding='utf8').splitlines() if s.strip())}
        for path in sorted(recovery_root.glob('rounds/r*/methods/*/*/checkpoint.sqlite3')):
            mid,qid=path.parent.parent.name,path.parent.name
            item=wanted.get((mid,qid))
            if not item:
                errors.append({'method':mid,'reason':'unauthorized_recovery_case'});continue
            expected={'task':recovery_root.name,'dataset':'2wiki','method':mid,'config':item['config'],
                      'round':int(path.parents[3].name[1:]),'qid':qid,'original_source':item['source']}
            with readonly(path) as db:
                found=audit_checkpoint(db,cases,expected)
            checks.append({'dataset':'2wiki','method':mid,'qid':qid,'recovery':True,'errors':found})
            errors.extend({'dataset':'2wiki','method':mid,'reason':e} for e in found)
    result = {'utc':base.utc(), 'passed':not errors, 'frozen_pins_checked':pins,
              'checkpoints':checks, 'errors':errors, 'new_api_calls':0,
              'semantic_answer_correctness':'not established by structural audits'}
    write('AUDIT.json', result)
    return result


@lru_cache(maxsize=3)
def score_inputs(dataset):
    # Gold is loaded solely inside this independent audit process, never sent
    # to a generator or written back to an execution checkpoint.
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    path = base.ROOT/'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen'/dataset/'formal.references.local.jsonl'
    refs = {v['qid']:v for v in (json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip())}
    return refs, metric(dataset)


def audit_saved_scores(directory, dataset):
    path = directory/'scores.jsonl'
    # The evaluator writes this report in place. Defer an active or incomplete
    # export rather than treating its transient state as corrupt evaluation.
    if not path.exists() or time.time()-path.stat().st_mtime < 5:
        return {'checked':0,'errors':[]}
    stamp = path.stat().st_mtime_ns
    try:
        records = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]
    except json.JSONDecodeError:
        return {'checked':0,'errors':[]}
    refs, score = score_inputs(dataset)
    if path.stat().st_mtime_ns != stamp or len(records) != len(refs):
        return {'checked':0,'errors':[]}
    errors = set()
    if {v['qid'] for v in records} != set(refs):
        errors.add('saved_score_cohort_mismatch')
    for row in records:
        if row['qid'] not in refs:
            continue
        valid = isinstance(row.get('prediction'),str) and row.get('status') in {'success','algorithm_failure'}
        expected = score(row['prediction'],refs[row['qid']]) if valid else {'em':None,'f1':None}
        for metric in ['em','f1']:
            actual, native = row.get(metric), expected[metric]
            if (actual is None) != (native is None) or (native is not None and abs(actual-native)>1e-12):
                errors.add('saved_native_metric_or_missing_denominator_mismatch')
    return {'checked':len(records),'errors':sorted(errors)}


def telemetry():
    inventory = base.inventory()
    overlay = base.PARENT/'twowiki_missing_recovery_20261021/RESULTS.json'
    if overlay.exists():
        effective=read(overlay)
        if effective.get('dataset')=='2wiki' and effective.get('formal_per_method')==500:
            for row in inventory['rows']:
                new=effective['runs'].get(row['method']) if row['dataset']=='2wiki' else None
                if new and new['scorable']>=row['scorable']:
                    row.update(processed=new['processed'],scorable=new['scorable'],statuses=new['statuses'],
                               missing_recovery_report=str(overlay))
            inventory['processed']=sum(v['processed'] for v in inventory['rows'])
            inventory['scorable']=sum(v['scorable'] for v in inventory['rows'])
    activity = []
    for dataset, method, _, _, path in checkpoints():
        with readonly(path) as db:
            attempts = db.execute('SELECT count(*),sum(result IS NOT NULL),max(started) FROM attempts').fetchone()
            rows = db.execute('SELECT count(*),sum(result IS NOT NULL) FROM requests').fetchone()
        activity.append([dataset,method,*attempts,*rows])
    return inventory, activity


def decision(supervisor, worker, progress, stopped, occupied, restarts, idle_seconds):
    if stopped:
        return 'stopped_no_restart'
    if occupied >= 299.9:
        return 'budget_guard_no_restart'
    if progress.get('status') == 'completed':
        return 'completed'
    if supervisor and worker:
        return 'idle_needs_attention' if idle_seconds >= 1800 else 'running'
    if worker:
        return 'worker_alive_supervisor_missing'
    if supervisor:
        return 'supervisor_recovering_or_needs_attention'
    if progress.get('status') == 'running' and restarts < 2:
        return 'recover_lost_supervisor'
    return 'needs_attention_no_auto_restart'


def stop_for_integrity(errors):
    # Exclusive creation preserves any user stop and prevents all current dispatch.
    marker = base.V2/'STOP_REQUESTED'
    try:
        with marker.open('x',encoding='utf8') as stream:
            json.dump({'utc':base.utc(), 'reason':'Independent main-evaluation integrity audit failed; drain, inspect AUDIT.json, do not auto-resume.', 'guard':str(OUT), 'errors':errors}, stream)
    except FileExistsError:
        pass
    event('integrity_stop', errors=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'guard.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        state = read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}
        restarts = state.get('supervisor_restarts', 0)
        last_key, last_activity, last_audit, last_inventory = None, time.monotonic(), 0., 0.
        previous = None
        while True:
            try:
                now = time.monotonic()
                if now-last_audit >= 300:
                    audit = full_audit()
                    last_audit = now
                    if not audit['passed']:
                        stop_for_integrity(audit['errors'])
                if now-last_inventory >= 60:
                    inventory, activity = telemetry()
                    # Shared-ledger changes may come from P02. Only this
                    # baseline cohort's own requests/results establish progress.
                    key = json.dumps([inventory['processed'],activity],sort_keys=True)
                    if key != last_key:
                        last_activity, last_key = now, key
                    write('PROGRESS.json', inventory)
                    last_inventory = now
                progress = read(base.V2/'progress.json')
                locks = {k:held(base.OUT/(k+'.lock')) for k in ['supervisor','worker']}
                stopped = any(p.exists() for p in [OUT/'STOP_REQUESTED',base.OUT/'STOP_REQUESTED',base.V2/'STOP_REQUESTED'])
                status = decision(locks['supervisor'],locks['worker'],progress,stopped,
                                  inventory['budget']['occupied_cny'],restarts,now-last_activity)
                if status == 'recover_lost_supervisor' and not args.once:
                    frozen_check()
                    # The existing supervisor/worker singleton locks are a second
                    # barrier if another authorized chat starts recovery concurrently.
                    restarts += 1
                    event('recover_lost_supervisor', recovery_number=restarts)
                    write('STATE.json', {'utc':base.utc(), 'supervisor_restarts':restarts})
                    with (OUT/'recovery.stdout.log').open('a') as stdout, (OUT/'recovery.stderr.log').open('a') as stderr:
                        module=('experiments.query_decomposition.resume_v2_trace_20261021.run'
                                if (base.PARENT/'resume_v2_trace_20261021/VALIDATION.json').exists() else base.MODULE)
                        child = subprocess.Popen([sys.executable,'-X','utf8','-u','-B','-m',module],
                            cwd=base.ROOT,stdout=stdout,stderr=stderr,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    event('supervisor_started',pid=child.pid)
                snapshot = {'utc':base.utc(), 'pid':os.getpid(), 'status':status,
                    'supervisor_restarts':restarts, 'locks_held':locks,
                    'seconds_without_observed_activity':round(now-last_activity),
                    'phase':{k:progress.get(k) for k in ['track','dataset','split','method','status','reason']},
                    'audit_passed':audit['passed'], 'audit_utc':audit['utc'],
                    'processed':inventory['processed'],'scorable':inventory['scorable'],
                    'planned':inventory['planned'],'budget':inventory['budget'],
                    'chat_notifications_scheduled':False, 'poll_seconds':20, 'audit_seconds':300}
                write('STATUS.json', snapshot)
                if status != previous:
                    event('state_changed',status=status)
                    previous = status
                if 'attention' in status or status.startswith(('stopped','budget')):
                    write('ATTENTION.json',snapshot)
                if args.once or status in {'completed','stopped_no_restart'} or (
                        status == 'budget_guard_no_restart' and not any(locks.values())):
                    print(json.dumps(snapshot),flush=True)
                    return
            except IntegrityFault as error:
                write('AUDIT.json', {'utc':base.utc(),'passed':False,'errors':[str(error)],'new_api_calls':0})
                stop_for_integrity([str(error)])
                write('STATUS.json', {'utc':base.utc(),'pid':os.getpid(),'status':'integrity_stop_no_restart','reason':str(error)})
                if args.once:
                    raise
                return
            except (sqlite3.Error, OSError, ValueError, KeyError) as error:
                # A transient read failure is evidence of monitor failure only;
                # do not restart or stop generation based on an incomplete read.
                write('MONITOR_ERROR.json',{'utc':base.utc(),'type':type(error).__name__,'reason':str(error)})
                if args.once:
                    raise
            time.sleep(20)


if __name__ == '__main__':
    main()
