"""Audit complete full-SFT consumption, dev selection, and conditional held-out QA."""
import collections,json,math,random,statistics
from datetime import datetime,timezone
from prepare import HERE,OUT,PRIOR,ROOT,rows,sha
from review import audit,summarize,raw,OLDOUT,PARENT

def paired(base,candidate):
    assert [r['id'] for r in base]==[r['id'] for r in candidate]
    dd=[int(t['verdict']=='pass')-int(b['verdict']=='pass') for b,t in zip(base,candidate)]
    wins=[b['index'] for b,d in zip(base,dd) if d==1];losses=[b['index'] for b,d in zip(base,dd) if d==-1]
    groups=[[d for b,d in zip(base,dd) if b['dataset']==ds] for ds in ['hotpotqa','2wiki','musique']]
    rng=random.Random(2718)
    boot=sorted(100*sum(sum(rng.choices(g,k=len(g))) for g in groups)/len(base) for _ in range(30000))
    n=len(wins)+len(losses)
    return {'fixes':wins,'regressions':losses,'net_passes':sum(dd),'delta_pp':100*statistics.mean(dd),
      'paired_stratified_bootstrap95_delta_pp':[boot[749],boot[29249]],
      'mcnemar_exact_two_sided_p':min(1.,2*sum(math.comb(n,k) for k in range(min(len(wins),len(losses))+1))/2**n) if n else 1.,
      'uncertainty_limit':'Conditional on same-Codex judgments, one training seed, related Mu primitives. Not independent human gold or population proof.'}

def latest(relative):
    pp=sorted((OUT/'returned').glob('snapshot_*/'+relative));assert pp,relative
    return pp[-1]

def main():
    plan=json.loads((HERE/'PLAN_FROZEN.json').read_text(encoding='utf-8'))
    policy=json.loads((HERE/'MONITOR_EVALUATION_POLICY.json').read_text(encoding='utf-8'))
    c=json.loads((HERE/'config.json').read_text(encoding='utf-8'));s=json.loads((HERE/'NOMINEE_FROZEN.json').read_text(encoding='utf-8'))
    assert sha(HERE/'config.json')==plan['config_sha256'] and sha(HERE/'BATCHES_FROZEN.json')==plan['batches_sha256']
    for n,h in c['data_sha256'].items():assert sha(HERE/n)==h,n
    state=json.loads(latest('results/train/STATE.json').read_text(encoding='utf-8'))
    assert state['status']=='completed' and state['global_step']==228 and state['base_unchanged'] and state['initial_adapter'] is None
    mm=rows(latest('results/train/metrics.jsonl'));bb=json.loads((HERE/'BATCHES_FROZEN.json').read_text(encoding='utf-8'));train=rows(HERE/'DATA3632.frozen.jsonl')
    assert len(mm)==len(bb)==228 and sum(m['actual_examples'] for m in mm)==7264
    for i,(m,b) in enumerate(zip(mm,bb),1):
        assert m['step']==i and m['ids']==[train[x]['id'] for x in b['indices']]
        assert m['actual_examples']==len(b['indices']) and math.isfinite(m['loss']) and math.isfinite(m['gradient_norm'])
    for epoch in [1,2]:
        counts=collections.Counter(q for m in mm if m['epoch']==epoch for q in m['ids'])
        assert len(counts)==3632 and set(counts.values())=={1}
    locked=json.loads((OLDOUT/'PARENT_DEV60_REVIEW_LOCKED.json').read_text(encoding='utf-8'))
    parent_path=OLDOUT/'reviews/dev60'/f'{PARENT}.jsonl';assert sha(parent_path)==locked['review_sha256']
    baseline=rows(parent_path);dev={};pairs={};integrity={}
    for step in [114,171,228]:
        n=f'checkpoint-{step:06d}';rr=audit('dev60',n);assert len(rr)==60
        dev[n]=summarize(rr);pairs[n]=paired(baseline,rr)
        pp=raw('dev60',n)
        source={r['index']:r for p in (OLDOUT/'raw_dev'/PARENT).glob('[0-9][0-9][0-9].json') for r in [json.loads(p.read_text(encoding='utf-8'))]}
        integrity[n]={'n':len(pp),'same_native_input_ids':all(pp[i]['input_token_ids']==source[i]['input_token_ids'] for i in pp),
          'invalid':sum(not r['format_valid'] for r in pp.values()),'truncated':sum(bool(r['truncated']) for r in pp.values()),
          'errors':sum(bool(r.get('generation_error')) for r in pp.values())}
        assert integrity[n]['same_native_input_ids']
    result={'recorded_utc':datetime.now(timezone.utc).isoformat(),'dev60':dev,'paired_dev60_vs_T02':pairs,'nominee':s,
      'actual_training':{'updates':228,'exposures':7264,'unique_per_epoch':3632,'tail_examples_each_epoch':16,
        'target_tokens':sum(m['target_tokens'] for m in mm),'update_seconds':sum(m['seconds'] for m in mm),
        'validation_losses':rows(latest('results/train/validation.jsonl')),'base_unchanged':True,'fresh_LoRA':True},
      'dev_integrity':integrity,'api_calls':0,'review_limit':'Same-Codex F/S/C/U; uncertain is nonpass, not independent human gold.',
      'scope':'Decomposition pass, not downstream RAG Answer F1; Mu is custom primitive-excluded native-train held-out, not official test.'}
    if s['selected']:
        rr=audit('confirm120');assert len(rr)==120
        parent_confirm=OLDOUT/'reviews/confirm120'/f'{PARENT}.jsonl'
        lock=json.loads((OLDOUT/'PARENT_CONFIRM120_REVIEW_LOCKED.json').read_text(encoding='utf-8'));assert sha(parent_confirm)==lock['review_sha256']
        parent=rows(parent_confirm);pilot=rows(OLDOUT/'reviews/confirm120/r02_expanded.jsonl')
        result.update(confirm120=summarize(rr),paired_confirm120_vs_T02=paired(parent,rr),paired_confirm120_vs_pilot_R02=paired(pilot,rr))
        t=result['confirm120'];b=summarize(parent)
        result['confirmation_gate']={'total_strictly_higher':t['passed']>b['passed'],
          'mu_not_lower':t['by_dataset']['musique']['passes']>=b['by_dataset']['musique']['passes'],
          'mu_FS_not_increased':len(t['mu_FS'])<=len(b['mu_FS'])}
        pp=raw('confirm120');orig={r['index']:r for p in (OLDOUT/'raw_confirmation'/PARENT).glob('[0-9][0-9][0-9].json') for r in [json.loads(p.read_text(encoding='utf-8'))]}
        assert len(pp)==120 and all(pp[i]['input_token_ids']==orig[i]['input_token_ids'] for i in pp)
    else:
        result['confirmation']='Not launched: no full-SFT candidate strictly improved dev60 under the frozen gates.'
    (OUT/'FINAL_METRICS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'CURRENT_STATUS.json').write_text(json.dumps({'stage':'completed','job_id':'154361','trained_updates':228,
      'selected':s['selected'],'confirmation_completed':bool(s['selected']),'api_calls':0,'utc':result['recorded_utc']},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'dev60':dev,'selected':s['selected'],'confirm120':result.get('confirm120'),'confirmation':result.get('confirmation')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
