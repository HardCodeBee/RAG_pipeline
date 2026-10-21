"""Reserve fresh native queries before any confirmation inference; no answers/labels."""
from pathlib import Path
import collections, hashlib, importlib.util, json, unicodedata
from prepare import HERE, ROOT, OUT, OLD, DEV, TRAIN, rows, frozen, write

def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split()).strip(' ?!.')
def parts(r):
    q=r.get('qid') or r.get('source',{}).get('id') or r.get('id','')
    return set(q.split('__',1)[1].split('_')) if '__' in q else set()

def main():
    sources=HERE.parent/'multigrain_4b_prompt_train3500_v1_20261017/training_sources.py'
    spec=importlib.util.spec_from_file_location('source_projection',sources);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    test=HERE.parent/'multigrain_4b_frozen_eval300_v1_20261017/questions300.frozen.jsonl'
    meta=json.loads((test.parent/'EVALUATION_FROZEN.json').read_text(encoding='utf-8'))
    paths=[ROOT/x['path'] for x in meta['exclusion_sources']]+[TRAIN,DEV,test]
    ids=set();texts=set()
    for p in paths:
        for r in rows(p):
            q=r.get('query',r.get('question',r.get('question_text')))
            if isinstance(q,str):texts.add(norm(q))
            for key in ['id','qid']:
                if isinstance(r.get(key),str):ids.update([r[key],r[key].split(':',1)[-1]])
            source=r.get('source',{})
            if isinstance(source,dict) and source.get('id'):ids.update([source['id'],source['id'].split(':',1)[-1]])
    mu_parts=set().union(*(parts(r) for p in [TRAIN,DEV,test] for r in rows(p)))
    pools=collections.defaultdict(list);excluded=collections.Counter()
    hot=ROOT/'analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl'
    native=[('hotpotqa',({'id':r['id'],'query':r['question'],'type':r['type']} for r in rows(hot))),('2wiki',mod.project('2wiki','dev')),('musique',mod.project('musique','dev'))]
    seen=set()
    for ds,source in native:
        for r in source:
            key=norm(r['query'])
            if r['id'] in ids or key in texts or key in seen:excluded['known_id_or_text']+=1;continue
            if ds=='musique' and parts(r)&mu_parts:excluded['shared_mu_training_dev_test_primitive']+=1;continue
            pools[ds,r['type']].append({'id':ds+':'+r['id'],'qid':r['id'],'dataset':ds,'query':r['query'],'type':r['type'],'source_split':'dev'})
            seen.add(key)
    available={d+'/'+t:len(a) for (d,t),a in pools.items()}
    desired={'hotpotqa':{'bridge':21,'comparison':9},'2wiki':{'comparison':8,'inference':8,'compositional':7,'bridge_comparison':7},'musique':{'2-hop':12,'3-hop':10,'4-hop':8}}
    quotas={d:dict(q) for d,q in desired.items()}
    # Keep the MuSiQue component exclusion even if fewer four-hop cases remain.
    for typ in ['4-hop','3-hop']:
        missing=max(0,quotas['musique'][typ]-len(pools['musique',typ]))
        quotas['musique'][typ]-=missing;quotas['musique']['2-hop']+=missing
    seed='20261018-post-sft-confirm90-01'
    rank=lambda s:hashlib.sha256((seed+s).encode()).hexdigest()
    selected=[]
    for ds,types in quotas.items():
        for typ,n in types.items():
            pool=sorted(pools[ds,typ],key=lambda r:rank(r['id']));assert len(pool)>=n,(ds,typ,len(pool),n)
            selected.extend(pool[:n])
    selected.sort(key=lambda r:rank('order:'+r['id']))
    for i,r in enumerate(selected,1):r['index']=i
    assert len(selected)==90 and len({r['id'] for r in selected})==90
    frozen(HERE/'confirmation90.questions.frozen.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected).encode())
    record={'seed':seed,'n':90,'desired_quotas':desired,'actual_quotas':quotas,'available':available,'excluded':dict(excluded),'selection_uses_outputs_or_answers':False,'role':'One-time new-query confirmation after dev-only selection; no follow-up tuning from confirmation outputs.','mu_primitive_exclusion':'All3500 train + dev150 + original formal300; retain exclusion even if four-hop quota cannot be filled.','excluded_sources':[str(p.relative_to(ROOT)) for p in paths]}
    frozen(HERE/'CONFIRMATION90_FROZEN.json',(json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps({k:record[k] for k in ['n','actual_quotas','available','excluded']},indent=2))

if __name__=='__main__':main()
