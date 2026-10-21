"""Final paired metrics and actual training-consumption verification."""
import collections,datetime,json,math,random,statistics
from prepare import HERE,OUT,ROOT,PREV,rows,sha,write
from review import summary,audit,raw,PARENT

def pair(bb,tt):
    assert [r['id'] for r in bb]==[r['id'] for r in tt]
    dd=[int(t['verdict']=='pass')-int(b['verdict']=='pass') for b,t in zip(bb,tt)]
    wins=[b['index'] for b,d in zip(bb,dd) if d==1];losses=[b['index'] for b,d in zip(bb,dd) if d==-1];discord=len(wins)+len(losses)
    groups=[[d for r,d in zip(bb,dd) if r['dataset']==ds] for ds in ['hotpotqa','2wiki','musique']];groups=[g for g in groups if g]
    rng=random.Random(2718);boot=sorted(100*sum(sum(rng.choices(g,k=len(g))) for g in groups)/len(bb) for _ in range(30000))
    return {'fixes':wins,'regressions':losses,'net_passes':sum(dd),'delta_pp':100*statistics.mean(dd),'paired_stratified_bootstrap95_delta_pp':[boot[749],boot[29249]],
            'mcnemar_exact_two_sided_p':min(1.,2*sum(math.comb(discord,k) for k in range(min(len(wins),len(losses))+1))/2**discord) if discord else 1.,
            'limits':'Conditional on same-Codex judgments and one training seed. Related Mu rows may share primitives; query bootstrap is a reference, not independent population proof.'}

def main():
    plan=json.loads((HERE/'PLAN_FROZEN.json').read_text(encoding='utf-8'));assert sha(HERE/'DATA3632.frozen.jsonl')==plan['data_sha256'];assert sha(HERE/'v12.txt')==plan['prompt_sha256']
    old=json.loads((PREV/'BASELINE_FROZEN.json').read_text(encoding='utf-8'));assert all(sha(ROOT/p)==h for p,h in old['frozen_files'].items())
    locked=json.loads((OUT/'PARENT_DEV60_REVIEW_LOCKED.json').read_text(encoding='utf-8'));ss=summary('dev60');assert all(ss[k]['n']==60 for k in [PARENT,'r01_old_data','r02_expanded']);assert sha(OUT/'reviews/dev60'/f'{PARENT}.jsonl')==locked['review_sha256']
    arms=['r01_old_data','r02_expanded']+(['r03_expanded20'] if 'r03_expanded20' in ss else [])
    result={'dev60':ss,'paired_dev60':{n:pair(audit('dev60',PARENT),audit('dev60',n)) for n in arms}}
    selected=json.loads((HERE/'NOMINEE_FROZEN.json').read_text(encoding='utf-8'));result['selection40']=selected
    if (HERE/'NOMINEE_LOWDOSE20_FROZEN.json').exists():selected=json.loads((HERE/'NOMINEE_LOWDOSE20_FROZEN.json').read_text(encoding='utf-8'))
    result['selection']=selected
    if selected['selected']:
        conf_names=[PARENT,selected['selected'],'original165']+(['r01_old_data'] if (HERE/'MATCHED_CONTROL_CONFIRM_AMENDMENT.json').exists() else [])
        conf=summary('confirm120');assert all(conf[k]['n']==120 for k in conf_names);result['confirm120']=conf
        result['paired_confirm120']={n:pair(audit('confirm120',PARENT),audit('confirm120',n)) for n in conf_names if n!=PARENT}
        if 'r01_old_data' in conf_names:result['expanded_vs_same40_olddata']=pair(audit('confirm120','r01_old_data'),audit('confirm120',selected['selected']))
        b=conf[PARENT];t=conf[selected['selected']];result['confirmation_gate']={'total_strictly_higher':t['passed']>b['passed'],'mu_strictly_higher':t['by_dataset']['musique']['passes']>b['by_dataset']['musique']['passes'],'mu_FS_not_increased':len(t['mu_FS'])<=len(b['mu_FS'])}
        result['confirmed_candidate']=selected['selected'] if all(result['confirmation_gate'].values()) else None
    else:result['confirmed_candidate']=None
    consumed={};data=rows(HERE/'DATA3632.frozen.jsonl')
    trainings=[(n,b,40,f'v001/training40/{n}') for n,b in plan['sampling_plans'].items()]
    if 'r03_expanded20' in ss:
        low=json.loads((HERE/'PLAN_LOWDOSE20_FROZEN.json').read_text(encoding='utf-8'));trainings.append(('r03_expanded20',low['sampling_plans']['r03_expanded20'],20,'v002/training20/r03_expanded20'))
    for name,batches,updates,subdir in trainings:
        paths=sorted((OUT/'returned').glob(f'snapshot_*/results/{subdir}/metrics.jsonl'));assert paths;mm=rows(paths[-1]);assert len(mm)==updates
        for m,batch in zip(mm,batches):
            assert m['ids']==[data[i]['id'] for i in batch];assert math.isfinite(m['loss']) and math.isfinite(m['gradient_norm'])
        state=json.loads((paths[-1].parent/'STATE.json').read_text(encoding='utf-8'));assert state['status']=='completed' and state['parent_unchanged'] and state['base_unchanged']
        consumed[name]={'updates':updates,'example_exposures':sum(m['examples'] for m in mm),'supervised_tokens':sum(m['target_tokens'] for m in mm),'update_seconds':sum(m['seconds'] for m in mm),
            'first_loss':mm[0]['loss'],'last_loss':mm[-1]['loss'],'parent_unchanged':True,'base_unchanged':True,'checkpoint_sha256':state['checkpoint_manifest_sha256']}
    result['actual_training_consumption']=consumed
    integrity={};calls=0
    for phase in ['dev60','confirm120']:
        names=[PARENT]+arms if phase=='dev60' else (conf_names if selected['selected'] else [])
        pp=raw(phase,PARENT)
        for name in names:
            pred=raw(phase,name);assert len(pred)==(60 if phase=='dev60' else 120);calls+=len(pred)
            integrity[phase+'/'+name]={'n':len(pred),'invalid_format':sum(not r['format_valid'] for r in pred.values()),'truncated':sum(bool(r['truncated']) for r in pred.values()),
                'generation_errors':sum(bool(r['generation_error']) for r in pred.values()),'prompt_exact':all(__import__('hashlib').sha256(r['actual_prompt'].encode()).hexdigest()==plan['prompt_sha256'] for r in pred.values()),
                'native_inputs_identical_to_parent':all(pred[i]['input_token_ids']==pp[i]['input_token_ids'] for i in pred)}
            assert integrity[phase+'/'+name]['prompt_exact'] and integrity[phase+'/'+name]['native_inputs_identical_to_parent']
    result.update({'raw_generation_calls_excluding_historical':calls,'api_calls':0,'integrity':integrity,'original_frozen_files_unchanged':True,
        'dataset_source_limit':'Mu new dev/confirmation are primitive-disjoint custom native-train held-outs, not official test scores.',
        'training_budget_limit':'Matched update count and example exposures; target-token counts and GPU seconds differ and are reported. Native+synthetic effects not separately identified.',
        'review_limit':'Current Codex authored labels and judged outputs; same-context self-review, no independent human gold. Full F/S/C/U pass is not downstream RAG F1.'})
    write(OUT/'FINAL_METRICS.json',result)
    write(OUT/'CURRENT_STATUS.json',{'phase':'completed','candidate':result['confirmed_candidate'],'selection':selected['selected'],'old_baselines_retained':True,'api_calls':0,'training_updates':{n:r['updates'] for n,r in consumed.items()},'dataset_total':3632,'new_native':96,'new_synthetic':40,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'dev60':ss,'confirmation':result.get('confirm120'),'confirmed_candidate':result['confirmed_candidate'],'training':consumed},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
