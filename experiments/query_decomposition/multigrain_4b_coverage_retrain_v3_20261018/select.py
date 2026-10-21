"""Apply the pre-frozen dev gate, retaining failed/uncertain denominators."""
import json,datetime
from prepare import HERE,OUT,rows,sha,write,freeze
from review import summary,PARENT

def main():
    ss=summary('dev60');assert all(ss[n]['n']==60 for n in [PARENT,'r01_old_data','r02_expanded'])
    locked=json.loads((OUT/'PARENT_DEV60_REVIEW_LOCKED.json').read_text(encoding='utf-8'));assert sha(OUT/'reviews/dev60'/f'{PARENT}.jsonl')==locked['review_sha256']
    b=ss[PARENT];gates={};eligible=[]
    for name in ['r01_old_data','r02_expanded']:
        t=ss[name];control_delta=sum(t['by_dataset'][d]['passes']-b['by_dataset'][d]['passes'] for d in ['hotpotqa','2wiki'])
        gate={'mu_strictly_higher':t['by_dataset']['musique']['passes']>b['by_dataset']['musique']['passes'],
              'overall_not_lower':t['passed']>=b['passed'],'controls_net_loss_at_most1':control_delta>=-1,'mu_FS_not_increased':len(t['mu_FS'])<=len(b['mu_FS'])}
        gates[name]=gate
        if all(gate.values()):eligible.append(name)
    eligible.sort(key=lambda n:(-ss[n]['passed'],-ss[n]['by_dataset']['musique']['passes'],0 if n=='r01_old_data' else 1))
    selected=eligible[0] if eligible else None;record={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gates':gates,'selected':selected,'dev_summary':ss,
        'confirmation_used_for_selection':False,'plan_sha256':sha(HERE/'PLAN_FROZEN.json'),'old_baselines_retained':True}
    if selected:
        # Latest complete snapshot contains immutable training metadata, not weights.
        completes=sorted((OUT/'returned').glob(f'snapshot_*/results/v001/training40/{selected}/checkpoint-final/COMPLETE.json'))
        assert completes;marker=json.loads(completes[-1].read_text(encoding='utf-8'))
        record.update({'checkpoint':f'/home/projects/u7830488/multigrain_4b_coverage_retrain_v3_20261018/results/v001/training40/{selected}/checkpoint-final',
            'checkpoint_manifest_sha256':marker['manifest_sha256'],'confirm_sha256':sha(HERE/'confirm120.questions.frozen.jsonl')})
    freeze(HERE/'NOMINEE_FROZEN.json',record);freeze(OUT/'NOMINEE_FROZEN.json',record);print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
