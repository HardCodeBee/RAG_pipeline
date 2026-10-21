"""Accept explicitly reviewed labels, then freeze matched-budget training plans."""
import collections,datetime,json,random,re
from prepare import HERE,ROOT,OUT,PREV,rows,sha,norm,parts,write,freeze

def main():
    inputs=rows(HERE/'NATIVE120_INPUTS.frozen.jsonl');labels={};quarantine={}
    for p in sorted((HERE/'labels').glob('native_*.json')):
        v=json.loads(p.read_text(encoding='utf-8'))
        for k,target in v['targets'].items():assert int(k) not in labels;labels[int(k)]=target
        quarantine.update(v['quarantine_reasons'])
    assert set(labels)==set(range(1,121));assert len(quarantine)==24
    base=rows(HERE/'BASE3496.frozen.jsonl');accepted=[];rejected=[]
    for r in inputs:
        i=r['extension_index'];target=labels[i]
        if target is None:rejected.append({**r,'reason':quarantine[str(i)],'status':'quarantined'});continue
        accepted.append({**r,'target':target,'cohort':'new_native_coverage','status':'accepted_after_query_only_self_review'})
    assert len(accepted)==96
    synth=json.loads((HERE/'labels/synthetic40.json').read_text(encoding='utf-8'))['examples'];assert len(synth)==40
    synthetic=[{'id':f'synthetic:coverage_v3:{i:03d}','query':r['query'],'target':r['target'],'source':{'dataset':'synthetic_coverage','id':f'coverage_v3_{i:03d}','original_split':'authored_training_only','skill':r['skill']},'cohort':'new_synthetic_coverage','status':'accepted_after_query_only_self_review'} for i,r in enumerate(synth,1)]
    for r in accepted+synthetic:
        assert isinstance(r['target'],list) and r['target']
        assert len(set(norm(q) for q in r['target']))==len(r['target'])
        for q in r['target']:
            assert isinstance(q,str) and q.endswith('?') and not re.search(r'#\d+',q)
            assert norm(q)!=norm(r['query']),(r['id'],'exact original-goal duplicate')
    combined=base+accepted+synthetic;assert len(combined)==3632
    assert len({norm(r['query']) for r in combined})==len(combined)
    fresh=rows(HERE/'dev60.questions.frozen.jsonl')+rows(HERE/'confirm120.questions.frozen.jsonl')
    fresh_text={norm(r['query']) for r in fresh};assert not fresh_text&{norm(r['query']) for r in combined}
    for phase in ['dev60','confirm120']:
        mu_eval=[r for r in rows(HERE/(phase+'.questions.frozen.jsonl')) if r['dataset']=='musique']
        eval_parts=set().union(*(parts(r) for r in mu_eval));train_parts=set().union(*(parts(r) for r in combined if r['source']['dataset'].lower()=='musique'))
        assert not eval_parts&train_parts
    freeze(HERE/'DATA3632.frozen.jsonl',combined,True);freeze(HERE/'NATIVE_QUARANTINE24.frozen.jsonl',rejected,True)
    freeze(HERE/'NEW136_ACCEPTED.frozen.jsonl',accepted+synthetic,True)
    prompt=(HERE.parent/'multigrain_4b_sft3500_v1_20261017/remote_v003/v12.txt').read_bytes();assert __import__('hashlib').sha256(prompt).hexdigest()=='508709f217ee58ca53d60ecce6001d5e5b0e6d51a858dd4b95b2d6d311864ee8'
    (HERE/'v12.txt').write_bytes(prompt)
    freeze(HERE/'LABEL_REVIEW_FROZEN.json',{'native_inputs':120,'accepted_native':96,'quarantined_native':24,'accepted_synthetic':40,
        'reviewer':'Current Codex author and second-pass self-review; not independent human gold or independent double review.',
        'review_scope':'All120 query/target or quarantine decisions and all40 explicit fictional examples inspected under original F/S/C/U. Native answers/passages/decomposition labels never consulted.',
        'second_pass_changes':[
          'Native19 remove invented named-after relation; preserve album from the city.',
          'Native72 remove invented company type; retain restaurant.',
          'Native58 apply state to Tuolumne county without additionally forcing neighboring Springfield county into that state.',
          'Native14 and16 quarantine unresolved source relations instead of preserving ambiguous authored targets.',
          'Synthetic18 use a lake source for the upstream river; preserve into/from direction.',
          'Synthetic29 name Ravo Institute to avoid an unspecified largest-institution ranking and redundant identity query.',
          'Synthetic37 request the award-winning school subset rather than paraphrasing the final all-schools Boolean goal.'],
        'fixed_base_policy':'Same8 previous label fixes for both controls;4 previously uncertain rows withheld from both. Original3500 untouched.',
        'label_file_hashes':{p.name:sha(p) for p in sorted((HERE/'labels').glob('*.json'))}})
    mu=[i for i,r in enumerate(base) if r['source']['dataset'].lower()=='musique'];other=[i for i,r in enumerate(base) if r['source']['dataset'].lower()!='musique'];assert len(mu)==505 and len(other)==2991
    real=list(range(len(base),len(base)+len(accepted)));fake=list(range(len(base)+len(accepted),len(combined)))
    rng=random.Random(2718);rng.shuffle(mu);rng.shuffle(other)
    def cycle(pool,n):
        result=[]
        while len(result)<n:
            current=list(pool);rng.shuffle(current);result+=current
        return result[:n]
    real_stream=cycle(real,240);fake_stream=cycle(fake,80);control=[];expanded=[]
    for step in range(40):
        common=other[16*step:16*(step+1)]+mu[8*step:8*(step+1)]
        extra_old=rng.sample([i for i in mu if i not in common],8)
        old=common+extra_old;new=common+real_stream[6*step:6*(step+1)]+fake_stream[2*step:2*(step+1)]
        assert len(old)==len(set(old))==32 and len(new)==len(set(new))==32
        control.append(old);expanded.append(new)
    stats={}
    for name,batches in [('r01_old_data',control),('r02_expanded',expanded)]:
        ids=[i for b in batches for i in b];stats[name]={'exposures':len(ids),'unique_examples':len(set(ids)),
            'by_cohort':dict(collections.Counter(combined[i].get('cohort','old') for i in ids)),
            'old_mu_exposures':sum(i in mu for i in ids),'native_new_exposures':sum(i in real for i in ids),'synthetic_exposures':sum(i in fake for i in ids)}
    freeze(HERE/'PLAN_FROZEN.json',{'parent_checkpoint':'/home/projects/u7830488/multigrain_4b_musique_scope_opt_v2_20261018/results/continuation20_v005/t02_mu50/checkpoint-final',
        'parent_sha256':'281f96415700497ef79d2d47597c82dcddb1352932e1a0cf804d139d9328574c',
        'original165_retained':True,'parent_t02_retained':True,'prompt_sha256':sha(HERE/'v12.txt'),'data_sha256':sha(HERE/'DATA3632.frozen.jsonl'),
        'updates':40,'seed':2718,'lr':2e-5,'warmup_updates':4,'batch_examples':32,'micro_batch':1,'max_supervised_length':1536,
        'optimizer':'Fresh AdamW,betas.9/.999,eps1e-8,wd.01,gradclip1; no optimizer-state-resume claim.',
        'sampling_plans':{'r01_old_data':control,'r02_expanded':expanded},'sampling_counts':stats,
        'comparison':'Same parent, optimizer and40-step budget. Same640 non-Mu plus320 old-Mu exposures; replace remaining320 old-Mu control exposures with240 new-native and80 marked-synthetic exposures. Tests whole expansion package, not separate synthetic/native causal effects.',
        'dev_roster_sha256':sha(HERE/'dev60.questions.frozen.jsonl'),'confirm_roster_sha256':sha(HERE/'confirm120.questions.frozen.jsonl'),
        'screen_gate':'New balanced60 once after final40 only. Require Mu strictly higher than parent; overall not lower than parent; combined Hot+Wiki loss at most1; Mu F/S count not higher. Choose eligible highest total, then Mu, then r01 for tie simplicity. If no candidate qualifies retain parent; do not force confirmation.',
        'confirmation_gate':'One selected candidate vs frozen parent and original165 on new120 once. Require Mu and total strictly higher than parent and Mu F/S not increased; report exact counts/CI and regressions. Never tune on this set.',
        'review':'Original F/S/C/U, same Codex non-independent query/output judgments; lock baseline before candidate; uncertainty nonpass; identical arrays inherit exact baseline.',
        'checkpoint_policy':'Fixed40 end only; original and parent adapter untouched, no selection using old confirmation120 or new confirmation120.',
        'generation':'Native nonthinking,greedy,seed1729,max_new_tokens1024; byte-identical v12; no postprocessor.',
        'api_calls':0})
    summary={'base':3496,'new_native':96,'new_synthetic':40,'total':3632,'base_quarantine':4,'native_candidate_quarantine':24,
        'new_native_prefix':dict(collections.Counter(r['qid'].split('__')[0] for r in accepted)),
        'new_native_max_primitive_reuse':max(collections.Counter(p for r in accepted for p in parts(r)).values()),
        'synthetic_skills':dict(collections.Counter(r['source']['skill'] for r in synthetic)),
        'by_dataset':dict(collections.Counter(r['source']['dataset'].lower() for r in combined)),
        'validation_note':'Custom Mu native-train held-outs with primitive isolation; not official native-test benchmark.',
        'review_status':'Complete query-only self-review, not independent gold.','training_status':'not submitted yet','sampling':stats}
    write(OUT/'DATA_BUILD.json',summary);write(OUT/'CURRENT_STATUS.json',{'phase':'data_accepted_plan_frozen','data_total':3632,'new_native':96,'new_synthetic':40,'training_started':False,'api_calls':0})
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
