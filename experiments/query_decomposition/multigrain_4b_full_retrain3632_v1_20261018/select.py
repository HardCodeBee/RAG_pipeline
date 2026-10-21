"""Dev-only nominee; latest user requires strict improvement before evaluation."""
import json
from datetime import datetime,timezone
from prepare import HERE,OUT,sha,freeze
from review import audit,summarize

def main():
    states=sorted((OUT/'returned').glob('snapshot_*/results/train/STATE.json'))
    assert states
    state=json.loads(states[-1].read_text(encoding='utf-8'))
    assert state['status']=='completed' and state['global_step']==228 and state['consumed_examples']==7264 and state['base_unchanged']
    policy=json.loads((HERE/'MONITOR_EVALUATION_POLICY.json').read_text(encoding='utf-8'))
    ss={};gates={};eligible=[]
    for step in [114,171,228]:
        name=f'checkpoint-{step:06d}';r=audit('dev60',name);s=summarize(r);assert s['n']==60;ss[name]=s
        d=s['by_dataset'];g={'total_strictly_above_T02':s['passed']>47,'mu_not_lower':d['musique']['passes']>=15,
          'HotWiki_net_loss_at_most1':d['hotpotqa']['passes']+d['2wiki']['passes']>=31,'mu_FS_not_increased':len(s['mu_FS'])<=4}
        gates[name]=g
        if all(g.values()):eligible.append(name)
    eligible.sort(key=lambda n:(-ss[n]['passed'],-min(d['passes'] for d in ss[n]['by_dataset'].values()),int(n.split('-')[1])))
    name=eligible[0] if eligible else None
    result={'recorded_utc':datetime.now(timezone.utc).isoformat(),'selected':name,'gates':gates,'dev_summary':ss,
      'plan_sha256':sha(HERE/'PLAN_FROZEN.json'),'policy_sha256':sha(HERE/'MONITOR_EVALUATION_POLICY.json'),
      'confirmation_used_for_selection':False,'baseline':'frozen T02','pilot_R02_passes':51,
      'api_calls':0,'old_models_retained':True}
    if name:
        markers=sorted((OUT/'returned').glob(f'snapshot_*/results/train/checkpoints/{name}/COMPLETE.json'));assert markers
        marker=json.loads(markers[-1].read_text(encoding='utf-8'));assert marker['complete']
        result.update(checkpoint=f'/home/projects/u7830488/{HERE.name}/results/train/checkpoints/{name}',
          checkpoint_manifest_sha256=marker['manifest_sha256'],strictly_above_pilot_R02=ss[name]['passed']>51)
    freeze(HERE/'NOMINEE_FROZEN.json',result);freeze(OUT/'NOMINEE_FROZEN.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
