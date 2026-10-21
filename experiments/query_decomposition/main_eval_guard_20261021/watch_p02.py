"""P02-specific monitoring; publish the combined 15000-result dashboard.

Keep frozen P02 rules separate from baseline v2 rules. The existing P02 supervisor
owns worker recovery. This monitor audits records and never submits a worker.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from . import watch as common
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest

ROOT = common.base.ROOT
HERE = ROOT/'experiments/query_decomposition/p02_formal1500_v1_20261021'
RUN = common.base.PARENT/HERE.name
METHOD = 'p02_multigrain_timing_cachefix_v1'


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def verify_plan():
    plan = common.read(HERE/'PROTOCOL.frozen.json')
    errors = []
    for root, pins in [(ROOT,plan['source_pins']), (HERE,plan['input_pins'])]:
        for path, expected in pins.items():
            if hashlib.sha256((root/path).read_bytes()).hexdigest() != expected:
                errors.append('frozen_p02_source_or_input_changed: '+path)
    if plan['method'] != METHOD or plan['planned'] != 1500 or plan['dataset_counts'] != {
            ds:500 for ds in ['hotpotqa','2wiki','musique']}:
        errors.append('p02_formal_cohort_changed')
    if plan['global_cap_cny'] != 300 or plan['dispatch_ceiling_cny'] != 299.9 or plan['local_subcap_cny'] != 20:
        errors.append('p02_budget_contract_changed')
    return plan, errors


def selected_cases():
    selected = {}
    for path in sorted(RUN.glob('rounds/r*/cases/*/*/*.json')):
        case = common.read(path)
        key = case['id']
        previous = selected.get(key)
        if previous and (previous['status'].startswith('success') or previous['status'] == 'algorithm_failure'):
            continue
        selected[key] = case
    return selected


def prefix_errors(call):
    errors = []
    if not call.get('full_payload_match_verified') or call['status'] != 'success':
        errors.append('p02_success_has_failed_or_unmatched_prefix')
    if call.get('finish_reason') != 'stop' and not (
            call.get('finish_reason') == 'length' and call.get('stage') == 'legacy/final'):
        errors.append('p02_intermediate_output_incomplete')
    return errors


def audit():
    plan, errors = verify_plan()
    questions = rows(HERE/'questions1500.frozen.jsonl')
    byid = {v['id']:v for v in questions}
    expected_identity = {
        'api_calls':0,'checkpoint_manifest_sha256':common.read(HERE/'REMOTE_INPUTS.frozen.json')['checkpoint_manifest_sha256'],
        'decode_compat':'explicit_generation_config_v002','do_sample':False,'effective_do_sample':False,
        'effective_num_beams':1,'num_beams':1,'max_new_tokens':1024,'seed':1729,
        'query_sha256':hashlib.sha256((HERE/'queries1500.id_query.frozen.jsonl').read_bytes()).hexdigest(),
        'prompt_sha256':hashlib.sha256((HERE/'v12.txt').read_bytes()).hexdigest()}
    identitypath = RUN/'remote_raw/IDENTITY.json'
    remote_count = 0
    if identitypath.exists():
        if common.read(identitypath) != expected_identity:
            errors.append('remote_decomposer_identity_changed')
        prompt = (HERE/'v12.txt').read_text(encoding='utf8')
        for path in (RUN/'remote_raw').glob('[0-9]*.json'):
            raw = common.read(path)
            index = raw['index']
            q = questions[index-1] if 1 <= index <= 1500 else {}
            if any(raw.get(k) != q.get(v) for k,v in [('id','id'),('dataset','dataset'),('query','question')]) or (
                    raw.get('actual_prompt') != prompt or raw.get('identity_sha256') != digest(expected_identity)):
                errors.append('remote_record_binding_mismatch: '+path.name)
            remote_count += 1
    selected = selected_cases()
    case_checked = 0
    with ExitStack() as stack:
        sources = {}
        for cid, case in selected.items():
            q = byid.get(cid,{})
            if any(case.get(k) != q.get(k) for k in ['id','qid','dataset','question']):
                errors.append('p02_case_outside_frozen_cohort')
            success = case['status'].startswith('success')
            if not success:
                continue
            if case.get('method') != METHOD:
                errors.append('p02_case_method_identity_mismatch')
            calls = case.get('logical_calls',[])
            if not calls or calls[-1].get('request_id') != case.get('request_id'):
                errors.append('p02_final_request_link_mismatch')
            for call in calls:
                errors.extend(prefix_errors(call))
                source = (ROOT/call['source']).resolve()
                if not source.is_relative_to(ROOT):
                    errors.append('p02_request_source_outside_workspace'); continue
                if source not in sources:
                    sources[source] = stack.enter_context(common.readonly(source))
                request = sources[source].execute('SELECT payload,status,result FROM requests WHERE id=?',
                                                  (call['request_id'],)).fetchone()
                if not request or request[1] != 'success':
                    errors.append('p02_success_request_receipt_missing'); continue
                payload, result = json.loads(request[0]), json.loads(request[2])
                if digest(payload) != call['payload_sha256'] or payload.get('model') != plan['model']:
                    errors.append('p02_payload_hash_or_model_mismatch')
                if result.get('answer') is None or result.get('finish_reason') != call.get('finish_reason'):
                    errors.append('p02_response_receipt_mismatch')
                if call is calls[-1] and (result.get('answer') or '').strip() != case.get('prediction'):
                    errors.append('p02_prediction_differs_from_final_response')
            case_checked += 1
    # Verify the stable native score export independently. A concurrent export
    # is deferred; no scoring file is rewritten by this process.
    scorepath = RUN/'SCORES.jsonl'
    score_count = 0
    if scorepath.exists() and time.time()-scorepath.stat().st_mtime > 5:
        stamp = scorepath.stat().st_mtime_ns
        try:
            records = rows(scorepath)
        except json.JSONDecodeError:
            records = []
        if scorepath.stat().st_mtime_ns == stamp and len(records) == 1500:
            if {r['id'] for r in records} != set(byid):
                errors.append('p02_score_cohort_mismatch')
            for row in records:
                refs, metric = common.score_inputs(row['dataset'])
                valid = row['status'].startswith('success') or row['status'] == 'algorithm_failure'
                expected = metric(row.get('prediction') or '',refs[row['qid']]) if valid else {'em':None,'f1':None}
                for key in ['em','f1']:
                    actual, native = row.get(key), expected[key]
                    if (actual is None) != (native is None) or (native is not None and abs(actual-native)>1e-12):
                        errors.append('p02_native_score_or_missingness_mismatch')
            score_count = len(records)
    value = {'utc':common.base.utc(),'method':METHOD,'passed':not errors,'errors':sorted(set(errors)),
             'source_pins_checked':len(plan['source_pins']),'input_pins_checked':len(plan['input_pins']),
             'remote_records_bound_to_queries':remote_count,'successful_cases_receipt_checked':case_checked,
             'saved_native_scores_checked':score_count,'new_api_calls':0,
             'protocol':'frozen P02 revision2; final length allowed; incomplete intermediate is algorithm failure',
             'semantic_answer_correctness':'not established by protocol audits'}
    common.write('P02_AUDIT.json',value)
    return value


def snapshot():
    selected = selected_cases()
    by_dataset = {ds:Counter() for ds in ['hotpotqa','2wiki','musique']}
    for value in selected.values():
        by_dataset[value['dataset']][value['status']] += 1
    scorable = sum(v['status'].startswith('success') or v['status']=='algorithm_failure' for v in selected.values())
    with common.readonly(common.base.PARENT/'budget.sqlite3') as db:
        known, reserved = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
        own = db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?',
                         (HERE.name+'%',)).fetchone()[0]
    remote = common.read(RUN/'REMOTE_PROGRESS.json')
    supervisor = common.read(RUN/'SUPERVISOR_STATUS.json')
    baseline = common.read(common.OUT/'STATUS.json')
    active = common.held(RUN/'controller.lock')
    terminal = (RUN/'STOP_REQUESTED').exists() or supervisor['state'] in {
        'complete','worker_finished_with_missing','needs_attention','network_unavailable','stopped_by_marker'}
    status = 'running' if active else 'needs_attention'
    if terminal:
        status = supervisor['state'] if not active else 'draining_or_stopped'
    return {'utc':common.base.utc(),'status':status,'planned':15000,
            'processed':baseline['processed']+len(selected),'scorable':baseline['scorable']+scorable,
            'baselines':{'planned':13500,'processed':baseline['processed'],'scorable':baseline['scorable'],
                         'status':baseline['status'],'utc':baseline['utc']},
            'p02':{'method':METHOD,'planned':1500,'processed':len(selected),'scorable':scorable,
                   'by_dataset':{ds:dict(c) for ds,c in by_dataset.items()},
                   'controller_lock_held':active,'supervisor':supervisor,
                   'remote_decompositions':remote['completed'],'remote_queue':remote.get('queue'),
                   'remote_receipt_utc':remote['utc'],'experiment_occupied_cny':own,'subcap_cny':20},
            'budget':{'known_cny':known,'unknown_reserved_cny':reserved,'occupied_cny':known+reserved,
                      'cap_cny':300,'remaining_cny':300-known-reserved},
            'chat_notifications_scheduled':False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--once',action='store_true'); args=parser.parse_args()
    common.OUT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (common.OUT/'p02_guard.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        last_audit = 0
        while True:
            try:
                if time.monotonic()-last_audit >= 300:
                    checked = audit();last_audit = time.monotonic()
                    if not checked['passed']:
                        try:
                            with (RUN/'STOP_REQUESTED').open('x',encoding='utf8') as stream:
                                json.dump({'utc':common.base.utc(),'reason':'P02 independent integrity audit failed',
                                           'errors':checked['errors']},stream)
                        except FileExistsError:
                            pass
                value = snapshot();value['p02']['audit_passed']=checked['passed']
                value['p02']['audit_utc']=checked['utc']
                common.write('ALL_METHODS_STATUS.json',value)
                if value['status'] != 'running':
                    common.write('P02_ATTENTION.json',value)
                if args.once:
                    print(json.dumps(value,ensure_ascii=False));return
            except (OSError,ValueError,KeyError,sqlite3.Error) as error:
                common.write('P02_MONITOR_ERROR.json',{'utc':common.base.utc(),'type':type(error).__name__,'reason':str(error)})
                if args.once:
                    raise
            time.sleep(20)


if __name__ == '__main__':
    main()
