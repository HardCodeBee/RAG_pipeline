"""Immutable v002 training export and separately frozen dev150 (no model/API)."""
import argparse, collections, copy, importlib.util, json, re
from pathlib import Path
from reaudit import HERE,ROOT,PREVIOUS,SOURCE,OUT,rows,sha,norm,write,write_rows,audit
EVAL=HERE.parent/'multigrain_4b_frozen_eval300_v1_20261017'
SEED='20261017-sft3500-dev150-v12-01'
QUOTAS={'hotpotqa':{'bridge':35,'comparison':15},'2wiki':{'comparison':13,'inference':13,'compositional':12,'bridge_comparison':12},'musique':{'2-hop':18,'3-hop':17,'4-hop':15}}

def components(r):
    rid=r.get('qid') or r.get('source',{}).get('id') or r.get('id','')
    return set(rid.split('__',1)[1].split('_')) if '__' in rid else set()

def rank(s):
    import hashlib
    return hashlib.sha256((SEED+s).encode()).hexdigest()

def load_source_module():
    spec=importlib.util.spec_from_file_location('query_only_sources',PREVIOUS/'training_sources.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def reserve():
    protected_parts=set().union(*(components(r) for r in rows(EVAL/'questions300.frozen.jsonl')))
    used={r['index'] for r in rows(SOURCE/'REVIEWED_ALL.jsonl')}
    candidates=[r for r in rows(SOURCE/'LABELING_INPUTS.frozen.jsonl') if r['index'] not in used and r['cohort']=='new_train_pool']
    # Freeze only five reserves, selected before predictions/semantic outcomes.
    selected=[]
    for ds,n in [('2wiki',2),('musique',3)]:
        eligible=[r for r in candidates if r['source']['dataset']==ds and not components(r)&protected_parts]
        selected.extend(sorted(eligible,key=lambda r:rank('reserve:'+r['id']))[:n])
    assert len(selected)==5
    path=HERE/'replacement_candidates.frozen.jsonl'
    if path.exists():assert rows(path)==selected,'Frozen reserves changed'
    else:write_rows(path,selected)
    for r in selected:print(str(r['index'])+' '+r['query'])

def build_train():
    assert audit()['full_second_review_completed']
    old=rows(SOURCE/'MASTER_3500.jsonl');decisions=rows(OUT/'reaudit/DECISIONS.progress.jsonl')
    result=[]
    for src,d in zip(old,decisions):
        assert src['index']==d['index'] and src['query']==d['query']
        if d['status']=='quarantined':continue
        item=copy.deepcopy(src);item['v001_target']=item['target'];item['target']=d['target'];item['second_pass']=d
        result.append(item)
    fixes=json.loads((HERE/'replacement_labels.reviewed.json').read_text(encoding='utf-8'))
    for src in rows(HERE/'replacement_candidates.frozen.jsonl'):
        fix=fixes['labels'][str(src['index'])];item=copy.deepcopy(src)
        item.update(target=fix['target'],status='accepted',review_method=fixes['reviewer'],review_reason=fix['reason'],replacement=True)
        result.append(item)
    assert len(result)==3500 and len({r['id'] for r in result})==3500 and len({norm(r['query']) for r in result})==3500
    prompt=(PREVIOUS/'FINAL_PROMPT.txt').read_text(encoding='utf-8').strip()
    for r in result:
        assert isinstance(r['target'],list) and all(isinstance(q,str) and q.endswith('?') for q in r['target'])
        assert len({norm(q) for q in r['target']})==len(r['target'])
        assert not any(re.search(r'#\d+|UNRESOLVED',q) for q in r['target'])
    dest=OUT/'data_v002';dest.mkdir(exist_ok=True,parents=True)
    def frozen(p,data):
        if p.exists():assert rows(p)==data, str(p)+' immutable'
        else:write_rows(p,data)
    frozen(dest/'MASTER_3500.jsonl',result)
    sft=[{'id':r['id'],'dataset':r['source']['dataset'],'messages':[{'role':'system','content':prompt},{'role':'user','content':r['query']},{'role':'assistant','content':json.dumps(r['target'],ensure_ascii=False,separators=(',',':'))}]} for r in result]
    frozen(dest/'train3500.messages.jsonl',sft)
    write(dest/'TRAINING_DATA_FROZEN.json',{'records':3500,'revision':'v002','source_v001_sha256':sha(SOURCE/'MASTER_3500.jsonl'),'v002_sha256':sha(dest/'MASTER_3500.jsonl'),'messages_sha256':sha(dest/'train3500.messages.jsonl'),'prompt_sha256':sha(PREVIOUS/'FINAL_PROMPT.txt'),'changed_old_records':sum(r['changed'] for r in decisions),'quarantined':sum(r['status']=='quarantined' for r in decisions),'replacement_count':5,'by_dataset':dict(collections.Counter(r['source']['dataset'] for r in result)),'by_cohort':dict(collections.Counter(r['cohort'] for r in result)),'empty_arrays':sum(not r['target'] for r in result),'review_limit':'Same current Codex authored/reviewed, not independent human gold. Changes include canonical simplifications, not all are semantic errors.','api_calls':0})
    print('v002 exported 3500')

def freeze_dev():
    path=HERE/'dev150.questions.frozen.jsonl'
    if path.exists():print('dev150 already frozen');return
    train=rows(OUT/'data_v002/MASTER_3500.jsonl');evaluation=rows(EVAL/'questions300.frozen.jsonl')
    meta=json.loads((EVAL/'EVALUATION_FROZEN.json').read_text(encoding='utf-8'))
    paths=[ROOT/r['path'] for r in meta['exclusion_sources']]
    paths += [OUT/'data_v002/MASTER_3500.jsonl',EVAL/'questions300.frozen.jsonl']
    ids=set();texts=set()
    for p in paths:
        for r in rows(p):
            q=r.get('query',r.get('question',r.get('question_text')))
            if isinstance(q,str):texts.add(norm(q))
            for k in ['qid','id']:
                if isinstance(r.get(k),str):ids.update([r[k],r[k].split(':',1)[-1]])
            s=r.get('source',{})
            if isinstance(s,dict):
                for k in ['id','native_id']:
                    if isinstance(s.get(k),str):ids.update([s[k],s[k].split(':',1)[-1]])
    old=ROOT/'outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer/data'
    for r in json.loads((old/'split_manifest.json').read_text(encoding='utf-8'))['records']:
        if r['project_assignment'] in {'project_dev','project_test','reserved_test_family','reserved_dev_family'}:ids.add(r.get('source',{}).get('id',''))
    screening=ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/multigrain_dual_retrieval_pilot_v1_20261017/MUSIQUE_SCREENING_FROZEN.json'
    ids.update(json.loads(screening.read_text(encoding='utf-8'))['ids'])
    mu_parts=set().union(*(components(r) for r in train+evaluation))
    mod=load_source_module();pools=[];counts=collections.Counter()
    hot=ROOT/'analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl'
    native=[('hotpotqa',({'id':r['id'],'query':r['question'],'type':r['type']} for r in rows(hot))),('2wiki',mod.project('2wiki','dev')),('musique',mod.project('musique','dev'))]
    seen=set()
    for ds,source in native:
        for r in source:
            if r['id'] in ids or norm(r['query']) in texts or norm(r['query']) in seen:counts['known_id_or_text']+=1;continue
            if ds=='musique' and components(r)&mu_parts:counts['shared_train_or_eval300_component']+=1;continue
            pools.append({'id':ds+':'+r['id'],'qid':r['id'],'dataset':ds,'query':r['query'],'type':r['type'],'source_split':'dev'});seen.add(norm(r['query']))
    selected=[];available={}
    for ds,qs in QUOTAS.items():
        for kind,n in qs.items():
            pool=sorted([r for r in pools if r['dataset']==ds and r['type']==kind],key=lambda r:rank(r['id']))
            available[ds+'/'+kind]=len(pool);assert len(pool)>=n,(ds,kind,len(pool),n)
            selected.extend(pool[:n])
    selected.sort(key=lambda r:rank('order:'+r['id']))
    for i,r in enumerate(selected,1):r['index']=i
    write_rows(path,selected)
    write(HERE/'DEV150_FROZEN.json',{'seed':SEED,'quotas':QUOTAS,'selection_uses_predictions_or_answers':False,'role':'development_loss_and_checkpoint_selection_only; never formal_test','questions_sha256':sha(path),'v002_sha256':sha(OUT/'data_v002/MASTER_3500.jsonl'),'excluded_eval300_sha256':sha(EVAL/'questions300.frozen.jsonl'),'mu_primitive_components_excluded':len(mu_parts),'candidate_exclusions':dict(counts),'available':available,'references':'pending_current_Codex_query_only_authoring_and_review'})
    print(json.dumps({'dev150':len(selected),'available':available},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['reserve','train','dev']);args=ap.parse_args()
    {'reserve':reserve,'train':build_train,'dev':freeze_dev}[args.stage]()
