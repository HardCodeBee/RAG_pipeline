"""Separate, purposive new-query diagnostic for underrepresented native prefixes."""
import collections,json
from prepare import HERE,ROOT,OUT,rows,parts,norm,rank,freeze,write

def main():
    meta=json.loads((HERE/'SAMPLING_FROZEN.json').read_text(encoding='utf-8'))
    paths=[ROOT/x['path'] for x in meta['exclusion_sources']]+[HERE/'screen90.questions.frozen.jsonl',HERE/'confirm120.questions.frozen.jsonl',HERE/'MU_EXPANSION180_QUERY_CANDIDATES.frozen.jsonl']
    text=set();blocked=set();ids=set()
    for p in paths:
        for r in rows(p):
            q=r.get('query',r.get('question',r.get('question_text')))
            if isinstance(q,str):text.add(norm(q))
            rid=r.get('qid') or r.get('source',{}).get('id') or r.get('id','');ids.add(rid.split(':',1)[-1]);blocked|=parts(r)
    pools=collections.defaultdict(list)
    for split in ['test','train']:
        for r in rows(ROOT/f'data/multihop_comprehensive_v1_20261015/musique/native/musique_ans_v1.0_{split}.jsonl'):
            pref=r['id'].split('__')[0]
            if pref not in ['3hop2','4hop2','4hop3']:continue
            if r['id'] in ids or norm(r['question']) in text or parts(r)&blocked:continue
            pools[pref,split].append({'id':'musique:'+r['id'],'qid':r['id'],'dataset':'musique','query':r['question'],'type':pref,'source_split':split})
    selected=[];actual={}
    for pref in ['3hop2','4hop2','4hop3']:
        pool=sorted(pools[pref,'test'],key=lambda r:rank('rare:'+r['id']))+sorted(pools[pref,'train'],key=lambda r:rank('rare:'+r['id']))
        actual[pref]=min(8,len(pool));selected.extend(pool[:actual[pref]])
    selected.sort(key=lambda r:rank('rareorder:'+r['id']))
    for i,r in enumerate(selected,1):r['index']=i
    assert selected
    freeze(HERE/'rare_queries.frozen.jsonl',selected,True)
    freeze(HERE/'RARE_DIAGNOSTIC_FROZEN.json',{'n':len(selected),'desired_quota':{'3hop2':8,'4hop2':8,'4hop3':8},'actual_quota':actual,'unfilled_reason':'No new primitive-disjoint4hop2 candidate remains after all strict exclusions; do not weaken exclusions or replace it with an easier prefix. Main confirmation still includes its previously reserved4hop2 case.','actual_sources':dict(collections.Counter(r['source_split'] for r in selected)),'selection_uses_new_model_outputs':False,'role':'Supplementary purposive diagnostic only; do not pool with balanced120 or call it population accuracy. Same paired baseline/model; old/new training and main evaluations primitive-disjoint.','native_id_prefix_caution':'Use native ID prefixes as labels without assuming an external graph ontology. New main confirmation also contains an observed4hop4 case.','existing_main120_roster_unchanged':True})
    print(json.dumps({'rare_n':len(selected),'actual_quota':actual,'source_split':dict(collections.Counter(r['source_split'] for r in selected)),'available':{str(k):len(v) for k,v in pools.items()}},indent=2))

if __name__=='__main__':main()
