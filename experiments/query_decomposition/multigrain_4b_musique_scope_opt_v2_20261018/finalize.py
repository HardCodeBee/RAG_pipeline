"""Audit the completed, bounded iteration without changing locked judgments."""
import collections,datetime,hashlib,json,math,random,statistics
from pathlib import Path
from prepare import HERE,OUT,ROOT,rows,sha,write,freeze,norm,parts
from review import audit,raw

ARMS=['b00_v12','m01_static_mu','m02_adaptive_mu','m03_scope_contrast','t01_original_mix','t02_mu50']
DATASETS=['hotpotqa','2wiki','musique']

def metric(rr):
    return {'n':len(rr),'passes':sum(r['verdict']=='pass' for r in rr),
            'verdicts':dict(collections.Counter(r['verdict'] for r in rr)),
            'dimensions_overlapping':dict(collections.Counter(d for r in rr for d in r['failed_dimensions'])),
            'by_dataset':{d:{'n':sum(r['dataset']==d for r in rr),'passes':sum(r['dataset']==d and r['verdict']=='pass' for r in rr)} for d in DATASETS},
            'mu_FS':[r['index'] for r in rr if r['dataset']=='musique' and set(r['failed_dimensions'])&{'F','S'}]}

def paired(bb,tt,bootstrap=True):
    assert len(bb)==len(tt) and [r['id'] for r in bb]==[r['id'] for r in tt]
    diffs=[int(t['verdict']=='pass')-int(b['verdict']=='pass') for b,t in zip(bb,tt)]
    wins=[b['index'] for b,d in zip(bb,diffs) if d==1];losses=[b['index'] for b,d in zip(bb,diffs) if d==-1]
    n=len(wins)+len(losses)
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(len(wins),len(losses))+1))/2**n) if n else 1.
    result={'fixes':wins,'regressions':losses,'net_passes':sum(diffs),'delta_pp':100*statistics.mean(diffs),'mcnemar_exact_two_sided_p':p}
    if bootstrap:
        # Stratify by dataset; estimates sampling variability conditional on these judgments.
        rng=random.Random(20261018);groups=[[d for r,d in zip(bb,diffs) if r['dataset']==ds] for ds in DATASETS];groups=[g for g in groups if g]
        draws=sorted(100*sum(sum(rng.choices(g,k=len(g))) for g in groups)/len(bb) for _ in range(30000))
        result['paired_stratified_bootstrap95_delta_pp']=[draws[749],draws[29249]]
        result['bootstrap_draws']=30000
    return result

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    baseline=json.loads((HERE/'BASELINE_FROZEN.json').read_text(encoding='utf-8'))
    unchanged={p:sha(ROOT/p)==v for p,v in baseline['frozen_files'].items()};assert all(unchanged.values())
    result={};integrity={};all_calls=0;rosters={}
    for phase,n in [('screen90',90),('confirm120',120),('rare16',16)]:
        pp=HERE/('rare_queries.frozen.jsonl' if phase=='rare16' else phase+'.questions.frozen.jsonl');rosters[phase]=rows(pp)
        names=ARMS if phase=='screen90' else ['b00_v12','t02_mu50'];res={}
        baseline_reviews=audit(phase,'b00_v12');baseline_raw=raw(phase,'b00_v12')
        for arm in names:
            rr=audit(phase,arm);preds=raw(phase,arm);assert len(rr)==len(preds)==n
            res[arm]=metric(rr)
            if arm!='b00_v12':res[arm]['paired']=paired(baseline_reviews,rr,bootstrap=phase=='confirm120')
            if all(v['n'] for v in res[arm]['by_dataset'].values()):res[arm]['macro_rate']=statistics.mean(v['passes']/v['n'] for v in res[arm]['by_dataset'].values())
            counts={k:sum(bool(r[k]) for r in preds.values()) for k in ['generation_error','truncated','explicit_references']}
            counts['format_invalid']=sum(not r['format_valid'] for r in preds.values());assert not any(counts.values())
            assert [preds[i]['id'] for i in range(1,n+1)]==[r['id'] for r in rosters[phase]]
            counts['all_model_payload_prompts_match_frozen_v12']=all(hashlib.sha256(r['actual_prompt'].encode()).hexdigest()==baseline['prompt_sha256'] for r in preds.values())
            if arm in ['b00_v12','t01_original_mix','t02_mu50']:assert counts['all_model_payload_prompts_match_frozen_v12']
            if arm in ['t01_original_mix','t02_mu50']:
                counts['paired_native_input_tokens_identical']=all(preds[i]['input_token_ids']==baseline_raw[i]['input_token_ids'] for i in preds);assert counts['paired_native_input_tokens_identical']
            integrity[phase+'/'+arm]=counts;all_calls+=n
        result[phase]=res
    assert all_calls==812
    duplicate_groups={}
    for phase,rr in rosters.items():
        groups=collections.defaultdict(list)
        for r in rr:groups[norm(r['query'])].append(r['index'])
        duplicate_groups[phase]=[v for v in groups.values() if len(v)>1]
    assert not duplicate_groups['screen90'] and not duplicate_groups['confirm120']
    assert duplicate_groups['rare16']==[[3,5]]
    rare_unique={};seen=set();keep=[]
    for r in rosters['rare16']:
        q=norm(r['query'])
        if q not in seen:keep.append(r['index']);seen.add(q)
    for arm in ['b00_v12','t02_mu50']:rare_unique[arm]=metric([r for r in audit('rare16',arm) if r['index'] in keep])
    write(OUT/'RARE_DUPLICATE_AUDIT.json',{'raw_n':16,'unique_text_n':15,'duplicate_indices':[3,5],
        'duplicate_ids':[r['id'] for r in rosters['rare16'] if r['index'] in [3,5]],'retained_unique_indices':keep,
        'unique_text_results':rare_unique,'policy':'Detected after generation. No replacements or denominator concealment. Report both raw and unique; purposive diagnostic not independent population evidence.'})
    text_sets={phase:{norm(r['query']) for r in rr} for phase,rr in rosters.items()}
    assert not text_sets['screen90']&text_sets['confirm120']
    all_mu={phase:set().union(*(parts(r) for r in rr if r['dataset']=='musique')) for phase,rr in rosters.items()}
    assert not all_mu['screen90']&all_mu['confirm120']
    assert not all_mu['rare16']&(all_mu['screen90']|all_mu['confirm120'])
    sampling=json.loads((HERE/'SAMPLING_FROZEN.json').read_text(encoding='utf-8'))
    known_texts=set();known_primitives=set()
    for src in sampling['exclusion_sources']:
        for r in rows(ROOT/src['path']):
            q=r.get('query',r.get('question',r.get('question_text')))
            if isinstance(q,str):known_texts.add(norm(q))
            known_primitives|=parts(r)
    for phase in rosters:
        assert not text_sets[phase]&known_texts,(phase,'old exposed text overlap')
        assert not all_mu[phase]&known_primitives,(phase,'old Mu primitive overlap')
    confirmation=result['confirm120'];b=confirmation['b00_v12'];t=confirmation['t02_mu50']
    gate={'total_strictly_higher':t['passes']>b['passes'],'mu_strictly_higher':t['by_dataset']['musique']['passes']>b['by_dataset']['musique']['passes'],
          'mu_FS_not_increased':len(t['mu_FS'])<=len(b['mu_FS']),
          'combined_controls_net_loss_at_most1':sum(t['by_dataset'][d]['passes']-b['by_dataset'][d]['passes'] for d in ['hotpotqa','2wiki'])>=-1}
    assert all(gate.values())
    result['review_limits']='Same current Codex authored/reviewed labels; no independent human gold/double review. F/S/C/U failures are not equivalent to wrong downstream QA. CIs are conditional on these judgments; one training seed. Confirmation120 not reused for tuning.'
    result['confirmation_gate']=gate;result['rare_unique_text_results']=rare_unique
    write(OUT/'FINAL_METRICS.json',result)
    freeze(OUT/'CANDIDATE_FROZEN_v2.json',{'candidate':'t02_mu50_end20','parent_checkpoint165_retained':True,
        'checkpoint':'/home/projects/u7830488/multigrain_4b_musique_scope_opt_v2_20261018/results/continuation20_v005/t02_mu50/checkpoint-final',
        'manifest_sha256':'281f96415700497ef79d2d47597c82dcddb1352932e1a0cf804d139d9328574c',
        'prompt':'Unchanged v12','prompt_sha256':baseline['prompt_sha256'],'postprocessor':'none','format':'JSON array of standalone supporting questions, no cross-step references; original query executed separately.',
        'changes':['Separate candidate3500 with8 repaired labels','20 additional updates,lr3e-5,seed1837,freshAdamW,warmup2_then_constant,batch32','Only the new640-example continuation has Mu50% share; no new independent training queries'],
        'sampling':'320 Mu/640 new exposures versus91/640 equal-budget control;738/5260 Mu exposures in parent; candidate cumulative1058/5900.',
        'gate':gate,'confirmation':{'n':120,'baseline_passes':96,'candidate_passes':100,'mu_baseline':29,'mu_candidate':30,'mu_n':40},
        'role':'Separately frozen empirical candidate. Original baseline/default artifacts unchanged. Not proof of90% or statistically established superiority.'})
    failures={}
    bb=raw('confirm120','b00_v12');tt=raw('confirm120','t02_mu50');judgments={r['index']:r for r in audit('confirm120','t02_mu50')}
    for i in [13,19,55,57,88,90,92,99,117]:
        failures[str(i)]={'query':bb[i]['query'],'id':bb[i]['id'],'baseline':bb[i]['parsed'],'candidate':tt[i]['parsed'],'candidate_verdict':judgments[i]['verdict'],'reason':judgments[i]['reason']}
    write(OUT/'CONFIRMATION_CASES.json',failures)
    check={'recorded_utc':now,'old_frozen_files_unchanged':unchanged,'calls':all_calls,'api_calls':0,'payloads':integrity,
           'duplicate_groups':duplicate_groups,'screen_confirm_text_disjoint':True,'screen_confirm_mu_primitives_disjoint':True,
           'rare_mu_primitives_disjoint_from_main':True,'all_phases_text_and_mu_primitives_disjoint_from_old_exclusion_sources':True,
           'n_new_unique_training_queries':0,'extension180_accepted_labels':0,'extension60_accepted_labels':0}
    write(OUT/'FINAL_VERIFICATION.json',check)
    dp=OUT/'DECISIONS.json';dec=json.loads(dp.read_text(encoding='utf-8'))
    additions=[{'id':'E03_CONFIRM_RESULT','change':'Complete baseline96/120 vs t02 100/120; Hot28->31,Wiki39->39,Mu29->30; MuFS9->8. Fixes57,88,90,92,117; regression19. All prespecified gates pass; original baseline unchanged, separately freeze candidate.'},
               {'id':'E04_RARE_RESULT','change':'Targeted rare raw12/16 vs12/16; indices3/5 duplicate text detected after generation. Retain raw records; unique11/15 vs11/15, no replacement. Partial location repair9 still fails. No rare improvement.'},
               {'id':'STOP_BOUNDED_V2','change':'Completed3 prompt candidates+2 equal-budget training candidates and untouched paired confirmation. No edits from confirmation outputs.90% target not reached. Do not keep tuning/retesting this holdout.180/60 future query candidates have0 accepted labels and0 new-data training uses; they are not a completed expansion.'}]
    existing={d['id'] for d in dec['decisions']}
    dec['decisions'] += [d for d in additions if d['id'] not in existing];dec['last_updated_utc']=now;write(dp,dec)
    state={'phase':'completed_bounded_iteration','baseline':'Original checkpoint165/v12 preserved unchanged','candidate':'t02_mu50_end20 separately frozen',
           'screen90':{'baseline':70,'candidate':74,'mu_baseline':33,'mu_candidate':37,'mu_n':50},
           'confirmation120':{'baseline':96,'candidate':100,'mu_baseline':29,'mu_candidate':30,'mu_n':40},
           'rare_diagnostic':{'raw_n':16,'unique_n':15,'both_raw_passes':12,'both_unique_passes':11},
           'goal_near90_achieved':False,'further_inference_or_training_running':False,'api_calls':0,'new_training_queries_used':0,
           'future180_candidate_labels_accepted':0,'future60_candidate_labels_accepted':0,'updated_utc':now}
    write(OUT/'CURRENT_STATUS.json',state);write(OUT/'STATUS_LATEST.json',state)
    print(json.dumps({'confirmation':confirmation,'gate':gate,'rare':result['rare16'],'verification_calls':all_calls},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
