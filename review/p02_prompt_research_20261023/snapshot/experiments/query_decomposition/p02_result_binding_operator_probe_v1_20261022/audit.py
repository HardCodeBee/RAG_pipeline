"""Offline accounting, intervention contracts and paired uncertainty."""
from collections import Counter
from contextlib import closing
import json
import random
from . import run

def main():
    summary = run.report()
    config, examples = run.check()
    tasks = {r['key']:r for r in run.rows(run.HERE/'TASKS.frozen.jsonl')}
    ordinary = {r['probe_id']:r for r in run.rows(run.HERE/'ORDINARY38.frozen.jsonl')}
    goals = {r['id']:r for r in run.rows(run.HERE/'INPUTS90.frozen.jsonl')}
    external = {r['payload_sha256'] for r in run.rows(run.HERE/'EXTERNAL_UNKNOWN.frozen.jsonl')}
    with closing(run.ro(run.OUT/'checkpoint.sqlite3')) as db:
        results = [json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='prediction'")]
        requests = {rid:dict(status=s,payload=json.loads(p),result=json.loads(r) if r else {}) for rid,s,p,r in db.execute('SELECT id,status,payload,result FROM requests')}
        retired = [dict(status=s,payload=json.loads(p),result=json.loads(r) if r else {}) for s,p,r in db.execute('SELECT status,payload,result FROM retired_requests')]
    verified = 0
    for r in results:
        assert r['key'] in tasks
        record = ordinary[r['id']] if r['suite']=='ordinary' else goals[r['id']]
        prompts = {'ordinary':(run.templates.ordinary_prompt(record,r['arm']),128)} if r['suite']=='ordinary' else {'goal':(run.templates.goal_prompt(record,r['arm']),512)}
        if r.get('goal_text'):
            prompts['final'] = (run.templates.base.final_prompt(record,r['goal_text'],config,examples)[0],64)
        for stage, usage in r['usage_by_stage'].items():
            receipt = requests[usage['request_id']]
            prompt, cap = prompts[stage]
            assert receipt['payload'] == run.build_payload(config['generation'],prompt,cap)
            assert run.digest(receipt['payload']) not in external
            result = {**receipt['result'],'status':receipt['status']}
            assert run.complete_receipt(result)
            assert result['returned_model']==config['generation']['model']
            assert result['usage']==usage['usage']
            assert result['usage']['total_tokens']==result['usage']['prompt_tokens']+result['usage']['completion_tokens']
            returned = r['answer'] if stage=='ordinary' else r['goal_text'] if stage=='goal' else r['prediction']
            assert returned == result['answer'].strip()
            verified += 1
    attempts = Counter(run.digest(r['payload']) for r in list(requests.values())+retired)
    assert max(attempts.values(),default=0)<=5
    recorded_cost=sum(run.cost(r['result']['usage']) for r in list(requests.values())+retired if r['result'].get('usage'))
    assert abs(recorded_cost-summary['new_returned_usage_CNY']) < 1e-8
    held=[dict(payload_sha256=run.digest(r['payload']),status=r['status'],proved_not_submitted=bool(r['result'].get('proved_not_submitted'))) for r in list(requests.values())+retired if r['status'] in ('unknown','in_flight')]
    # Stratify by dataset and preserve pairing; macro gives each dataset equal weight.
    details = {(r['arm'],r['id']):r for r in run.rows(run.OUT/'DETAILS.local.jsonl') if r['suite']=='goal'}
    intervals={}
    for a,b in [('G0','GF'),('G0','GO'),('GF','GO')]:
        ds_deltas={ds:[details[(b,i)]['metrics']['f1']-details[(a,i)]['metrics']['f1'] for i in summary['common_ids'] if goals[i]['dataset']==ds] for ds in ('hotpotqa','2wiki','musique')}
        if not all(ds_deltas.values()):
            continue
        rng=random.Random(20261022)
        means=[]
        for _ in range(10000):
            means.append(sum(sum(rng.choices(v,k=len(v)))/len(v) for v in ds_deltas.values())/3)
        means.sort()
        intervals[a+'_to_'+b]=dict(dataset_equal_F1_delta=sum(sum(v)/len(v) for v in ds_deltas.values())/3,
                                  percentile95=[means[249],means[9749]],replicates=10000,seed=20261022)
    ordinary_present=[i for i in summary['common_ids'] if goals[i]['has_ordinary']]
    sensitivity={a:dict(n=len(ordinary_present),**{k:sum(details[(a,i)]['metrics'][k] for i in ordinary_present)/len(ordinary_present) if ordinary_present else None for k in ('f1','em')}) for a in run.templates.GOAL_ARMS}
    format_stats={}
    for a in run.templates.GOAL_ARMS:
        selected=[r for r in details.values() if r['arm']==a and r.get('goal_text')]
        words=[len(r['goal_text'].split()) for r in selected]
        format_stats[a]=dict(n=len(selected),mean_whitespace_words=sum(words)/len(words) if words else None,
                            maximum_whitespace_words=max(words,default=None),multiple_paragraphs=sum('\n\n' in r['goal_text'] for r in selected))
    result=dict(passed=True,terminal=len(results),successful_logical_stage_receipts_verified=verified,
        maximum_actual_attempts_per_payload=max(attempts.values(),default=0),
        returned_usage_ledger_reconciled_CNY=recorded_cost,unknown_or_retired_connection_receipts=held,
        paired_stratified_bootstrap=intervals,ordinary_present_sensitivity=sensitivity,
        goal_format_statistics=format_stats,main_holds_unchanged=True)
    run.save(run.OUT/'OFFLINE_AUDIT.json',result)
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':
    main()
