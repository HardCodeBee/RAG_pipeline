"""Read-only receipt/interface audit; writes only this trial's audit report."""
from collections import Counter
from contextlib import closing
import json
from . import run


def main():
    config=run.read(run.HERE/'CONFIG.frozen.json');examples=run.read(run.HERE/'FINAL_EXAMPLES.frozen.json')
    tasks={r['key']:r for r in run.rows(run.HERE/'TASKS.frozen.jsonl')}
    inputs={'goal':{r['id']:r for r in run.rows(run.HERE/'INPUTS90.frozen.jsonl')},
            'ordinary':{r['id']:r for r in run.rows(run.HERE/'ORDINARY176.frozen.jsonl')}}
    with closing(run.ro(run.OUT/'checkpoint.sqlite3')) as db:
        outcomes=[json.loads(x) for x, in db.execute("SELECT value FROM data WHERE kind='outcome'")]
        receipts={rid:dict(payload=json.loads(payload),status=status,result=json.loads(raw) if raw else {})
          for rid,payload,status,raw in db.execute('SELECT id,payload,status,result FROM requests')}
        retired=db.execute('SELECT count(*) FROM retired_requests').fetchone()[0]
    errors=[];checked=0;stop_usage=0;algorithm=0
    for o in outcomes:
        try:
            task=tasks[o['key']];r=inputs[o['suite']][o['id']]
            assert all(o[k]==v for k,v in task.items())
            for s in o['stages']:
                rec=receipts[s['request_id']];value=rec['result'];payload=rec['payload']
                assert value.get('returned_model')==config['generation']['model'] or o['status']=='technical_held'
                assert len(payload['messages'])==1 and payload['messages'][0]['role']=='user'
                name=s['stage'].rsplit('/',1)[1]
                if name=='ordinary':prompt=run.ordinary_prompt(r,o['arm']);cap=256
                elif name=='goal':prompt=run.goal_prompt(r,o['arm']);cap=512
                elif name=='final':prompt=run.final_prompt(r,o['goal_text'],o['arm'],config,examples);cap=128
                elif name=='formatter':
                    question=r['current_question'] if o['suite']=='ordinary' else r['question']
                    raw=o['raw_output'] if o['suite']=='ordinary' else o['goal_raw']
                    packet=r['reader_packet'] if o['suite']=='ordinary' else r['goal_reader_packet']
                    assert len(raw.split())<4
                    prompt=run.formatter(question,raw,packet);cap=128
                else:raise AssertionError('Unexpected logical stage')
                assert payload==run.build_payload(config['generation'],prompt,cap)
                if value.get('usage'):
                    u=value['usage'];assert u['total_tokens']==u['prompt_tokens']+u['completion_tokens']
                    assert s['usage']==u
                if o['status']=='success':
                    assert rec['status']=='success' and value['finish_reason']=='stop' and value['answer'].strip()
                    stop_usage+=1
                checked+=1
            if o['status']=='success' and o['suite']=='ordinary' and o['arm']=='O1':
                parsed=run.design.parse('ordinary',o['raw_output'])
                assert parsed=={'Support':o['support'],'Answer':o['answer']}
            if o['status']=='success' and o['suite']=='goal' and o['arm']!='C':
                run.design.parse('goal',o['goal_text'])
                assert o['goal_text']==o['goal_raw']
            if o['status']=='algorithm_failure':algorithm+=1
        except Exception as exc:errors.append(dict(key=o['key'],type=type(exc).__name__,detail=str(exc)[:250]))
    # Paired G/GF must literally consume the same new goal, not separately sampled explanations.
    index={(o['id'],o['arm']):o for o in outcomes if o['suite']=='goal' and o['status']=='success'}
    shared=0
    for ident in inputs['goal']:
        if all((ident,a) in index for a in ('G','GF')):
            x,y=[index[(ident,a)] for a in ('G','GF')]
            if x['goal_text']!=y['goal_text']:errors.append(dict(key=ident,detail='G/GF goal differs'))
            shared+=1
    initial=[t for t in tasks.values() if t['engineering_initial']]
    outcome_index={x['key']:x for x in outcomes}
    missing=[x['key'] for x in initial if x['key'] not in outcome_index]
    result=dict(utc=run.utc(),passed=not errors,conditions_checked=len(outcomes),logical_stages_checked=checked,
      successful_stop_usage_stages=stop_usage,shared_goal_pairs=shared,algorithm_failures=algorithm,
      statuses=dict(Counter(o['status'] for o in outcomes)),retired_request_records=retired,
      initial_engineering_complete=not missing,initial_missing=missing,errors=errors,
      checks='Exact rendered requests/caps, accepted fields, full goal handoff, root material, model/usage/stop, G/GF exact goal reuse',
      not_factual_validation=True,new_API_requests=0)
    run.atomic_json(run.OUT/'RECEIPT_INTERFACE_AUDIT.json',result)
    print(json.dumps(result,ensure_ascii=True))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
