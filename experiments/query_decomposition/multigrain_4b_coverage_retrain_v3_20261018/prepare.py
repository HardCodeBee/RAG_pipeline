"""Freeze coverage-driven query inputs and primitive-disjoint new validation."""
from pathlib import Path
import collections,hashlib,importlib.util,json,re,unicodedata

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'outputs/query_decomposition'/HERE.name
PREV=HERE.parent/'multigrain_4b_musique_scope_opt_v2_20261018'
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split()).strip(' ?!.')
def parts(r):
    rid=r.get('qid') or r.get('source',{}).get('id') or r.get('id','')
    return set(rid.split('__',1)[1].split('_')) if '__' in rid else set()
def rank(s):return hashlib.sha256(('20261018-coverage-retrain-v3:'+s).encode()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def freeze(p,v,jsonl=False):
    data=(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in v) if jsonl else json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode();p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data,str(p)
    else:p.write_bytes(data)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((PREV/'SAMPLING_FROZEN.json').read_text(encoding='utf-8'))
    paths=[ROOT/r['path'] for r in meta['exclusion_sources']]+[PREV/'screen90.questions.frozen.jsonl',PREV/'confirm120.questions.frozen.jsonl',PREV/'rare_queries.frozen.jsonl']
    texts=set();blocked=set();ids=set()
    for p in paths:
        for r in rows(p):
            q=r.get('query',r.get('question',r.get('question_text')))
            if isinstance(q,str):texts.add(norm(q))
            blocked|=parts(r)
            for k in ['id','qid']:
                if isinstance(r.get(k),str):ids.add(r[k].split(':',1)[-1])
            if r.get('source',{}).get('id'):ids.add(r['source']['id'])
    candidate180=rows(PREV/'MU_EXPANSION180_QUERY_CANDIDATES.frozen.jsonl');candidate_texts={norm(r['query']) for r in candidate180}
    native=ROOT/'data/multihop_comprehensive_v1_20261015'
    pools=collections.defaultdict(list);seen=set()
    def add(ds,split,r,kind):
        q=r['query'];key=norm(q)
        if key in texts or key in seen or r['id'] in ids or (ds=='musique' and (parts(r)&blocked or key in candidate_texts)):return
        seen.add(key);pools[ds,kind].append({'id':ds+':'+r['id'],'qid':r['id'],'dataset':ds,'query':q,'type':kind,'source_split':split})
    for r in rows(ROOT/'analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl'):add('hotpotqa','dev',{'id':r['id'],'query':r['question']},r['type'])
    spec=importlib.util.spec_from_file_location('native_sources',HERE.parent/'multigrain_4b_prompt_train3500_v1_20261017/training_sources.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for r in mod.project('2wiki','dev'):add('2wiki','dev',r,r['type'])
    native_mu_all=[]
    for r in rows(native/'musique/native/musique_ans_v1.0_train.jsonl'):
        projected={'id':r['id'],'query':r['question']};native_mu_all.append(projected)
        add('musique','train_held_out_query_only',projected,r['id'].split('__')[0])
    quotas={
      'dev60':{'hotpotqa':{'bridge':14,'comparison':6},'2wiki':{'comparison':5,'inference':5,'compositional':5,'bridge_comparison':5},'musique':{'2hop':7,'3hop1':5,'3hop2':2,'4hop1':3,'4hop2':1,'4hop3':2}},
      'confirm120':{'hotpotqa':{'bridge':28,'comparison':12},'2wiki':{'comparison':10,'inference':10,'compositional':10,'bridge_comparison':10},'musique':{'2hop':14,'3hop1':10,'3hop2':4,'4hop1':6,'4hop2':2,'4hop3':4}}}
    # Try deterministic seeds before seeing model outputs; preserve a nonempty diverse training pool.
    for salt in range(200):
        chosen={};previous_pr=set();used=set();ok=True
        for phase,quota in quotas.items():
            cohort=[]
            for ds,q in quota.items():
                for kind,n in sorted(q.items(),key=lambda kv:len(pools[ds,kv[0]])):
                    pool=sorted(pools[ds,kind],key=lambda r:rank(str(salt)+phase+r['id']))
                    take=[r for r in pool if r['id'] not in used and not (ds=='musique' and parts(r)&previous_pr)][:n]
                    if len(take)!=n:ok=False;break
                    cohort+=take;used.update(r['id'] for r in take)
                if not ok:break
            if not ok:break
            previous_pr|=set().union(*(parts(r) for r in cohort if r['dataset']=='musique'));chosen[phase]=cohort
        if not ok:continue
        remaining=[r for r in candidate180 if not parts(r)&previous_pr]
        bins=collections.Counter(r['qid'].split('__')[0] for r in remaining)
        break
    else:raise RuntimeError('Cannot meet declared isolated quotas; do not relax silently.')
    for phase,rr in chosen.items():
        rr.sort(key=lambda r:rank('order:'+phase+r['id']))
        for i,r in enumerate(rr,1):r['index']=i
        freeze(HERE/(phase+'.questions.frozen.jsonl'),rr,True)
    # Prefer rarer branch prefixes, retain all types while respecting protected evaluation families.
    eligible=[];train_seen=set()
    for r in native_mu_all:
        key=norm(r['query']);ps=parts(r)
        if key in texts or key in train_seen or r['id'] in ids or ps&(blocked|previous_pr):continue
        train_seen.add(key);eligible.append({'id':'musique:'+r['id'],'qid':r['id'],'query':r['query'],
            'source':{'dataset':'musique','id':r['id'],'original_split':'train','type':r['id'][0]+'-hop'},
            'weak_nested_scope_cue_count':sum(r['query'].lower().count(w) for w in [' whose ',' where ',' and ',' after ',' before ',' by ']),
            'target':None,'status':'query_candidate_not_labelled_not_accepted'})
    selected=[];usage=collections.Counter()
    for hop,n in [('4',60),('3',40),('2',20)]:
        pp=sorted([r for r in eligible if r['qid'].startswith(hop+'hop')],key=lambda r:(0 if r['qid'].startswith(('3hop2','4hop2')) else 1,-r['weak_nested_scope_cue_count'],rank('new-labels:'+r['id'])))
        chosen_train=[]
        for r in pp:
            ps=parts(r)
            if any(usage[p]>=3 for p in ps):continue
            chosen_train.append(r);usage.update(ps)
            if len(chosen_train)==n:break
        assert len(chosen_train)==n,(hop,len(chosen_train),n)
        selected+=chosen_train
    selected.sort(key=lambda r:rank('label-order:'+r['id']))
    for i,r in enumerate(selected,1):r=dict(r);r['extension_index']=i;selected[i-1]=r
    freeze(HERE/'NATIVE120_INPUTS.frozen.jsonl',selected,True)
    base=rows(PREV/'continuation/MASTER3500_SCOPE8.candidate.jsonl');uncertain=[r for r in rows(PREV/'TRAIN_AUDIT80.review.jsonl') if r['verdict']=='uncertain'];exclude={r['id'] for r in uncertain};clean=[r for r in base if r['id'] not in exclude];assert len(clean)==3496
    freeze(HERE/'QUARANTINE4.frozen.json',uncertain)
    freeze(HERE/'BASE3496.frozen.jsonl',clean,True)
    freeze(HERE/'SAMPLING_FROZEN.json',{'seed':'20261018-coverage-retrain-v3','selection_salt':salt,'quotas':quotas,
        'available_before_selection':{d+'/'+t:len(v) for (d,t),v in pools.items()},'remaining_candidate180_prefixes':dict(bins),
        'native120_prefixes':dict(collections.Counter(r['qid'].split('__')[0] for r in selected)),
        'native120_max_primitive_reuse':max(usage.values()),'native120_source':'Original train query-only, selected after reserving new evaluation. Prior180 are unaccepted candidates, not an immutable training roster.',
        'exclusion_sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in paths],
        'mu_validation_source':'Unused native train query-only; custom primitive-disjoint validation, NOT official native test score.',
        'new_training_vs_validation_mu_primitive_disjoint':True,'dev_confirm_mu_primitive_disjoint':True,
        'no_new_model_predictions_or_answers_used_for_selection':True,
        'native120_status':'Inputs only; no labels accepted yet. Synthetic examples will be separately marked.'})
    print(json.dumps({'salt':salt,'new_native_n':len(selected),'prefix':dict(collections.Counter(r['qid'].split('__')[0] for r in selected)),'base_after_quarantine':len(clean),'validation_counts':{k:len(v) for k,v in chosen.items()}},indent=2))

if __name__=='__main__':main()
