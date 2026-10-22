"""Offline integrity, receipt, paired uncertainty and output-format audit."""
from collections import Counter
import json
import math
import re
import numpy as np
from . import run


def main():
    config, examples = run.check()
    before = run.sha(run.OUT / 'checkpoint.sqlite3')
    inputs = {r['id']: r for r in run.rows(run.HERE / 'INPUTS90.frozen.jsonl')}
    with run.ro(run.OUT / 'checkpoint.sqlite3') as db:
        details = [json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='prediction'")]
        requests = {rid: {'payload': json.loads(p), 'status': s, 'result': json.loads(v) if v else {}}
                    for rid,p,s,v in db.execute('SELECT id,payload,status,result FROM requests')}
        stages = dict(db.execute('SELECT stage,request_id FROM request_stages'))
        retired = {rid: {'payload':json.loads(p),'status':s,'result':json.loads(v) if v else {}}
                   for rid,p,s,v in db.execute('SELECT id,payload,status,result FROM retired_requests')}
    assert len(details) == 270 and len({(r['id'],r['arm']) for r in details}) == 270
    assert {r['id'] for r in details} == set(inputs)
    for retired_request in retired.values():
        result = retired_request['result']
        assert result.get('proved_not_submitted') is True and not result.get('usage') and not result.get('response_id')
        assert 'ConnectTimeout' in [c['type'] for c in result['failure_causes']]
    all_requests = {**requests, **retired}
    assert max(Counter(run.digest(r['payload']) for r in all_requests.values()).values()) <= 5
    unknown = {r['payload_sha256'] for r in run.rows(run.HERE / 'EXTERNAL_UNKNOWN.frozen.jsonl')}
    assert not unknown.intersection(run.digest(r['payload']) for r in all_requests.values())
    assert all(r['status'] != 'in_flight' for r in requests.values())
    receipt_counts, formats, failures = Counter(), {}, []
    for arm in run.templates.ARMS:
        formats[arm] = Counter()
    for r in details:
        arm, ident = r['arm'], r['id']
        goal_payload = run.build_payload(config['generation'], run.templates.goal_prompt(inputs[ident], arm), run.GOAL_CAP)
        if r['status'] == 'external_unknown_held':
            assert run.digest(goal_payload) in unknown
            continue
        key = arm + '|' + ident
        gid = stages['goal/' + key]
        g = requests[gid]
        assert g['payload'] == goal_payload
        if r['status'] == 'technical_held':
            assert g['status'] == 'unknown'
            continue
        result = g['result']
        f = formats[arm]
        f['goal_outputs'] += 1
        f['one_paragraph'] += int(not re.search(r'\n\s*\n', result['answer'].strip()))
        f['has_candidate_marker'] += int('Candidate answer:' in result['answer'])
        f['has_unresolved_marker'] += int('Unresolved:' in result['answer'])
        f['has_either_marker'] += int('Candidate answer:' in result['answer'] or 'Unresolved:' in result['answer'])
        f['length_ended'] += int(result['finish_reason'] == 'length')
        if r['status'] == 'algorithm_failure':
            assert result['finish_reason'] == 'length' and result['usage']['completion_tokens'] == run.GOAL_CAP
            assert 'final/' + key not in stages
            failures.append({'id':ident, 'arm':arm, 'finish_reason':result['finish_reason'],
                             'usage':result['usage'], 'goal_text':result['answer'],
                             'returned_usage_CNY':run.cost(result['usage']), 'score':0})
            continue
        assert r['status'] == 'success' and r['goal_text'] == result['answer'].strip()
        final_prompt,_ = run.templates.old.final_prompt(inputs[ident],r['goal_text'],config,examples)
        fid = stages['final/' + key]
        assert requests[fid]['payload'] == run.build_payload(config['generation'],final_prompt,run.FINAL_CAP)
        assert r['prediction'] == requests[fid]['result']['answer'].strip()
        assert (gid,fid) == (r['goal_request_id'],r['final_request_id'])
        for stage,rid in (('goal',gid),('final',fid)):
            assert run.complete_receipt(requests[rid]['result'])
            assert r['usage_by_stage'][stage]['usage'] == requests[rid]['result']['usage']
    for request in requests.values():
        result = request['result']
        receipt_counts[request['status'] + ':' + str(result.get('finish_reason'))] += 1
        if request['status'] != 'success':
            continue
        usage = result['usage']
        assert usage['prompt_tokens'] + usage['completion_tokens'] == usage['total_tokens']
        assert result['returned_model'] == result['requested_model'] == request['payload']['model']
        raw = result['raw_response']
        assert raw['usage'] == usage and raw['choices'][0]['message']['content'] == result['answer']
        assert raw['choices'][0]['finish_reason'] == result['finish_reason']
    namespace = 'p02_prompt_research_v1_20261022/' + str(run.OUT.resolve())
    with run.ro(config['budget']['ledger']) as db:
        charges = list(db.execute('SELECT id,reserved,actual,status FROM charges WHERE namespace=?',(namespace,)))
    paid = reserve = retired_reserve = unresolved_reserve = 0.0
    for rid,reserved,actual,status in charges:
        assert rid in all_requests
        r = all_requests[rid]
        if actual is None:
            assert r['status'] == 'unknown'
            reserve += reserved
            if rid in retired:
                retired_reserve += reserved
            else:
                unresolved_reserve += reserved
        else:
            assert math.isclose(actual,run.cost(r['result']['usage']),abs_tol=1e-12)
            paid += actual
    summary = run.read(run.OUT / 'SUMMARY.json')
    assert math.isclose(paid,summary['fresh_actual_CNY'],abs_tol=1e-10)
    assert math.isclose(reserve,summary['unknown_reserved_CNY'],abs_tol=1e-10)
    scored = {(r['arm'],r['id']):r for r in run.rows(run.OUT / 'DETAILS.scored.local.jsonl')}
    common = run.read(run.OUT / 'REPORT.json')['common_ids']
    pairs, bootstrap = {}, {}
    rng = np.random.default_rng(20261022)
    for a,b in (('G2','I'),('G2','F'),('I','F')):
        deltas = {ds: np.array([scored[(b,i)]['native_metrics']['f1']-scored[(a,i)]['native_metrics']['f1']
                              for i in common if inputs[i]['dataset']==ds]) for ds in run.pilot.DATASETS}
        all_deltas = np.concatenate(list(deltas.values()))
        pairs[a+'_to_'+b] = {'n':len(all_deltas), 'improved':int(np.sum(all_deltas>1e-12)),
                             'harmed':int(np.sum(all_deltas < -1e-12)), 'tied':int(np.sum(abs(all_deltas)<=1e-12)),
                             'F1_delta':float(all_deltas.mean())}
        means = np.mean([d[rng.integers(len(d),size=(10000,len(d)))].mean(axis=1) for d in deltas.values()],axis=0)
        bootstrap[a+'_to_'+b] = {'dataset_equal_mean_F1_delta':float(np.mean([d.mean() for d in deltas.values()])),
                                'percentile_95_interval':np.quantile(means,[.025,.975]).tolist()}
    assert before == run.sha(run.OUT / 'checkpoint.sqlite3')
    result = {'passed':True, 'utc':run.utc(), 'frozen_inputs_and_main_holds_verified':True,
              'checkpoint_unchanged_by_audit':True, 'terminal_conditions':len(details),
              'status_counts':dict(Counter(r['status'] for r in details)),
              'receipt_integrity_counts':dict(receipt_counts), 'logical_outputs_format':formats,
              'length_failure':failures, 'same86_paired_F1':pairs,
              'exploratory_paired_bootstrap':{'seed':20261022,'replicates':10000,
                  'method':'Dataset-stratified paired resampling of the same86; equal dataset weights',
                  'limitation':'Exposed development subset; no correction for prior prompt selection or provider sampling variability',
                  'results':bootstrap},
              'verified_new_returned_usage_CNY':paid,'retained_unknown_reserved_CNY':reserve,
              'retained_reservation_breakdown_CNY':{'unresolved_submission':unresolved_reserve,
                  'known_connect_timeouts_retained_conservatively':retired_reserve},
              'bounded_connect_timeout_recoveries':len(retired),
              'no_new_model_requests_during_audit':True}
    run.save(run.OUT/'OFFLINE_AUDIT.json',result)
    run.save(run.REVIEW/'OFFLINE_AUDIT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='length_failure'},ensure_ascii=True))


if __name__ == '__main__':
    main()
