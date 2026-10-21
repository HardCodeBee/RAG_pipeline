"""Freeze new samples and inspect training exposure before any new inference."""
from pathlib import Path
import collections, hashlib, importlib.util, json, re, statistics, unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'outputs/query_decomposition' / HERE.name
OLD = ROOT / 'outputs/query_decomposition/multigrain_4b_sft3500_v1_20261017'
PREV = HERE.parent / 'multigrain_4b_post_sft_opt_v1_20261018'
TRAIN = OLD / 'data_v003_exact_v12/MASTER_3500.jsonl'
SEED = '20261018-musique-scope-v2-01'

def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(s): return hashlib.sha256((SEED+s).encode()).hexdigest()
def norm(q): return ' '.join(unicodedata.normalize('NFKC',q).casefold().split()).strip(' ?!.')
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def freeze(p,obj,jsonl=False):
    data=(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in obj) if jsonl else json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): assert p.read_bytes()==data,str(p)
    else:p.write_bytes(data)
def parts(r):
    rid=r.get('qid') or r.get('source',{}).get('id') or r.get('id','')
    return set(rid.split('__',1)[1].split('_')) if '__' in rid else set()
def hop(r):
    rid=r.get('source',{}).get('id',r.get('qid',r.get('id','')))
    return next((str(n)+'-hop' for n in [2,3,4] if re.search(r'(?:^|:)'+str(n)+'hop',rid)),r.get('type','unknown'))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'outputs/query_decomposition/multigrain_4b_post_sft_opt_v1_20261018/BASELINE_FROZEN.json'
    control=json.loads(baseline.read_text(encoding='utf-8'))
    for p,h in control['frozen_files'].items(): assert sha(ROOT/p)==h
    freeze(HERE/'BASELINE_FROZEN.json',control)
    train=rows(TRAIN);mu=[r for r in train if r['source']['dataset'].lower()=='musique']
    assert len(train)==3500 and len(mu)==509
    exposure={'train_n':3500,'sources':dict(collections.Counter(r['source']['dataset'] for r in train)),
              'musique_n':509,'musique_fraction':509/3500,
              'musique_hops':dict(collections.Counter(hop(r) for r in mu)),
              'musique_target_items':dict(collections.Counter(len(r['target']) for r in mu)),
              'training_policy':'Each epoch visits all3500 once; no dataset resampling. Selected165 is1.5 epochs, trained220 is2 epochs.',
              'causal_limit':'Counts establish exposure, not that scarcity caused the failures. No new training in this iteration.',
              'score_gap':{'dev':128/150,'formal':236/300,'delta_pp':100*(128/150-236/300),
                           'mu_contribution_pp':(70-54)/3,'mu_fraction_of_total_gap':.8},
              'dev_mu_hops':{'2-hop':{'pass':14,'n':18},'3-hop':{'pass':13,'n':17},'4-hop':{'pass':8,'n':15}},
              'formal_mu_hops':{'2-hop':{'pass':26,'n':35},'3-hop':{'pass':20,'n':35},'4-hop':{'pass':8,'n':30}}}
    write(OUT/'TRAIN_EXPOSURE.json',exposure)
    selected=[];used=set()
    for h in ['2-hop','3-hop','4-hop']:
        pool=sorted([r for r in mu if hop(r)==h],key=lambda r:rank('audit:'+r['id']))
        for r in pool[:20]:selected.append({**r,'audit_cohort':'random20_per_hop'});used.add(r['id'])
    def risk(r):
        q=r['query'].lower()
        return sum(q.count(t) for t in [' where ',' whose ',' and ',' by ',' after ',' before '])+len(q.split())/20
    pool=sorted([r for r in mu if r['id'] not in used],key=lambda r:(-risk(r),rank('risk:'+r['id'])))
    selected.extend({**r,'audit_cohort':'targeted_long_nested20'} for r in pool[:20])
    frozen_audit=[{'audit_index':i,'train_index':r['index'],'id':r['id'],'query':r['query'],'target':r['target'],'hop':hop(r),'cohort':r['audit_cohort']} for i,r in enumerate(selected,1)]
    freeze(HERE/'TRAIN_AUDIT80.frozen.jsonl',frozen_audit,True)
    # Protect prior exposed lists, all trained queries, old dev/test and previous confirmation.
    formal=HERE.parent/'multigrain_4b_frozen_eval300_v1_20261017/questions300.frozen.jsonl'
    oldmeta=json.loads((formal.parent/'EVALUATION_FROZEN.json').read_text(encoding='utf-8'))
    paths=[ROOT/r['path'] for r in oldmeta['exclusion_sources']]+[TRAIN,HERE.parent/'multigrain_4b_sft3500_v1_20261017/dev150.questions.frozen.jsonl',formal,PREV/'confirmation90.questions.frozen.jsonl']
    ids=set();texts=set();mu_parts=set()
    for p in paths:
        for r in rows(p):
            q=r.get('query',r.get('question',r.get('question_text')))
            if isinstance(q,str):texts.add(norm(q))
            for k in ['id','qid']:
                if isinstance(r.get(k),str):ids.update([r[k],r[k].split(':',1)[-1]])
            src=r.get('source',{})
            if isinstance(src,dict) and src.get('id'):ids.add(src['id'])
            mu_parts|=parts(r)
    spec=importlib.util.spec_from_file_location('native_projection',HERE.parent/'multigrain_4b_prompt_train3500_v1_20261017/training_sources.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    # Native test hides decomposition labels; derive hop solely from the public ID.
    test_mu=({'id':r['id'],'query':r['question'],'type':str(int(r['id'][0]))+'-hop'} for r in rows(mod.NATIVE/'musique/native/musique_ans_v1.0_test.jsonl'))
    native=[('hotpotqa','dev',({'id':r['id'],'query':r['question'],'type':r['type']} for r in rows(ROOT/'analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl'))),
            ('2wiki','dev',mod.project('2wiki','dev')),('musique','test',test_mu)]
    pools=collections.defaultdict(list);excluded=collections.Counter();seen=set()
    for ds,split,source in native:
        for r in source:
            key=norm(r['query'])
            if r['id'] in ids or key in texts or key in seen:excluded['known_id_or_text']+=1;continue
            if ds=='musique' and parts(r)&mu_parts:excluded['shared_mu_primitive']+=1;continue
            pools[ds,r['type']].append({'id':ds+':'+r['id'],'qid':r['id'],'dataset':ds,'query':r['query'],'type':r['type'],'source_split':split})
            seen.add(key)
    quotas={
        'screen90':{'hotpotqa':{'bridge':14,'comparison':6},'2wiki':{'comparison':5,'inference':5,'compositional':5,'bridge_comparison':5},'musique':{'2-hop':14,'3-hop':18,'4-hop':18}},
        'confirm120':{'hotpotqa':{'bridge':28,'comparison':12},'2wiki':{'comparison':10,'inference':10,'compositional':10,'bridge_comparison':10},'musique':{'2-hop':14,'3-hop':13,'4-hop':13}},
    }
    chosen_ids=set();first_parts=set();counts={}
    for phase,qs in quotas.items():
        chosen=[]
        for ds,ts in qs.items():
            for typ,n in ts.items():
                pool=sorted(pools[ds,typ],key=lambda r:rank(phase+':'+r['id']))
                eligible=[r for r in pool if r['id'] not in chosen_ids and (ds!='musique' or not parts(r)&first_parts)]
                assert len(eligible)>=n,(phase,ds,typ,len(eligible),n)
                chosen.extend(eligible[:n]);chosen_ids.update(r['id'] for r in eligible[:n])
        chosen.sort(key=lambda r:rank('order:'+phase+r['id']))
        for i,r in enumerate(chosen,1):r['index']=i
        freeze(HERE/f'{phase}.questions.frozen.jsonl',chosen,True)
        if phase=='screen90': first_parts=set().union(*(parts(r) for r in chosen if r['dataset']=='musique'))
        counts[phase]=len(chosen)
    freeze(HERE/'SAMPLING_FROZEN.json',{'seed':SEED,'quotas':quotas,'counts':counts,
        'available':{d+'/'+t:len(v) for (d,t),v in pools.items()},'excluded':dict(excluded),
        'selection_uses_answers_or_outputs':False,'mu_test_projection':'Query/id/hop only; no answers, passages or native decomposition labels supplied to model/reviewer.',
        'split_caution':'New MuSiQue comes from native test, unlike earlier native-dev tests. Paired controls use the same new queries; rates are not a temporal continuation of the previous sample.',
        'primitive_exclusion':'All listed training/exposed Mu primitives; screen and confirmation primitive-disjoint from each other.',
        'exclusion_sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in paths]})
    print(json.dumps({'exposure':exposure,'counts':counts,'available':{d+'/'+t:len(v) for (d,t),v in pools.items()}},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
