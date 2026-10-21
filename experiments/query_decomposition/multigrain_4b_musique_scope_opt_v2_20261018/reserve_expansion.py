"""Reserve diverse native-training query candidates; no accepted labels/training claim."""
import collections,json,re
from prepare import HERE,ROOT,OUT,TRAIN,rows,parts,norm,rank,freeze,write

def main():
    sampling=json.loads((HERE/'SAMPLING_FROZEN.json').read_text(encoding='utf-8'))
    paths=[ROOT/r['path'] for r in sampling['exclusion_sources']]+[HERE/'screen90.questions.frozen.jsonl',HERE/'confirm120.questions.frozen.jsonl']
    texts=set();ids=set();blocked=set()
    for p in paths:
        for r in rows(p):
            q=r.get('query',r.get('question',r.get('question_text')))
            if isinstance(q,str):texts.add(norm(q))
            rid=r.get('source',{}).get('id') or r.get('qid') or r.get('id','');ids.add(rid.split(':',1)[-1]);blocked|=parts(r)
    source=ROOT/'data/multihop_comprehensive_v1_20261015/musique/native/musique_ans_v1.0_train.jsonl'
    buckets=collections.defaultdict(list)
    for r in rows(source):
        p=parts(r)
        if r['id'] in ids or norm(r['question']) in texts or p&blocked:continue
        cue=sum(r['question'].lower().count(w) for w in [' whose ',' where ',' and ',' after ',' before ',' by '])
        buckets[int(r['id'][0])].append({'id':'musique:'+r['id'],'qid':r['id'],'query':r['question'],'source':{'dataset':'musique','id':r['id'],'original_split':'train','type':str(int(r['id'][0]))+'-hop'},'weak_nested_scope_cue_count':cue,'target':None,'status':'query_candidate_not_labelled_not_accepted'})
    selected=[];used=collections.Counter();deferred=collections.Counter()
    for h,n in [(4,90),(3,60),(2,30)]:
        pool=sorted(buckets[h],key=lambda r:(-r['weak_nested_scope_cue_count'],rank('expand:'+r['id'])))
        chosen=[]
        for r in pool:
            ps=parts(r)
            if any(used[p]>=3 for p in ps):deferred['primitive_reuse_cap3']+=1;continue
            chosen.append(r);used.update(ps)
            if len(chosen)==n:break
        assert len(chosen)==n,(h,len(chosen),n)
        selected+=chosen
    selected.sort(key=lambda r:rank('expand-order:'+r['id']))
    for i,r in enumerate(selected,1):r['index']=i
    freeze(HERE/'MU_EXPANSION180_QUERY_CANDIDATES.frozen.jsonl',selected,True)
    write(OUT/'MU_EXPANSION180_STATUS.json',{'candidates':180,'accepted_labels':0,'training_runs_using_these':0,'quota':{'2-hop':30,'3-hop':60,'4-hop':90},'max_primitive_occurrences':max(used.values()),'eligible_before_cap':{str(k):len(v) for k,v in buckets.items()},'primitive_disjoint_from_all_existing_training_and_evaluations':True,'sampling_caution':'Purposive lexical-cue prioritization with primitive cap, not random benchmark evaluation; scope cues do not certify correct source semantics. Must author and inspect labels before training.','new_current_training':'The separate20-step exposure control uses only original3500 queries, not these candidates.'})
    print(json.dumps({'query_candidates':180,'labels':0,'eligible':{str(k):len(v) for k,v in buckets.items()},'max_primitive_reuse':max(used.values())},indent=2))

if __name__=='__main__':main()
