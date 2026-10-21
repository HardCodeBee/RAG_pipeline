"""Freeze query-only samples before any new decomposition or answer result."""
import argparse
from collections import Counter, defaultdict
import hashlib
import re
from .common import ROOT, HERE, RUN, DATA, read, rows, save, sha, log
from experiments.query_decomposition.hotpot_formal_eval_v1.prepare_samples import audit, allocate, group_by, norm, template, Groups

SEED = 2026101501
MANIFEST = ROOT/'outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer/data/split_manifest.json'
EXTRA = [
 ROOT/'experiments/query_decomposition/hotpot_formal_eval_v1/inputs_frozen/dev.questions.jsonl',
 ROOT/'experiments/query_decomposition/ircot_alignment_v1/inputs_frozen/questions500.jsonl',
 ROOT/'experiments/query_decomposition/ircot_alignment_v1/source/dev_subsampled.jsonl',
 ROOT/'experiments/query_decomposition/hotpot_formal_eval_v1/compact_100_200_20261014/inputs_frozen/formal.questions.jsonl',
]

def h(text):return hashlib.sha256(text.encode('utf8')).hexdigest()

def draw(eligible):
    used=set();result={}
    for split,n in [('dev',100),('formal',500)]:
        available=[r for r in eligible if r['group_id'] not in used]
        representatives=[min(v,key=lambda r:h(f'{SEED}:{split}:item:'+r['qid'])) for v in group_by(available).values()]
        quotas=allocate(n,Counter(r['type'] for r in available))
        chosen=[]
        for kind,count in sorted(quotas.items()):
            candidates=sorted([r for r in representatives if r['type']==kind],key=lambda r:h(f'{SEED}:{split}:group:'+r['group_id']))
            if len(candidates)<count:raise ValueError('Insufficient independent groups: '+str((kind,count,len(candidates))))
            chosen.extend(candidates[:count])
        chosen.sort(key=lambda r:h(f'{SEED}:order:{split}:'+r['qid']))
        used.update(r['group_id'] for r in chosen)
        result[split]=chosen
    # Fixed before outcomes; never select ablations from new losses or gains.
    formal=result['formal'];quota=allocate(200,Counter(r['type'] for r in formal));sub=[]
    for kind,n in sorted(quota.items()):
        sub.extend(sorted([r for r in formal if r['type']==kind],key=lambda r:h(f'{SEED}:ablation:'+r['qid']))[:n])
    result['ablation']=sorted(sub,key=lambda r:h(f'{SEED}:ablation-order:'+r['qid']))
    return result

def generic_audit(metadata,dataset):
    manifest=read(MANIFEST); train=rows(MANIFEST.parent/'runtime_qwen_train.jsonl')
    protected=[r for r in manifest['records'] if r['project_assignment'] in {'project_dev','project_test'}]
    groups=Groups();owners={};blocked=defaultdict(set);accessible=[];official=[]
    def add(qid,text,reason=None,kind=None,extra_keys=()):
        index=groups.add()
        keys=[('exact',h(norm(text))),('template',h(template(text))),*extra_keys]
        for key in keys:
            if key in owners:groups.union(index,owners[key])
            else:owners[key]=index
        r={'qid':qid,'question':text.strip(),'type':kind,'index':index}
        accessible.append(r)
        if reason:blocked[index].add(reason)
        return r
    for r in protected:
        index=groups.add();blocked[index].add('project_dev_or_test_metadata')
        for key in [('exact',r['query_sha256']),('template',r['template_sha256']),*[('family',x) for x in r['family_key_hashes']]]:
            if key in owners:groups.union(index,owners[key])
            else:owners[key]=index
    for r in train:add(r['id'],r['query'],'actual_4b_training')
    for path in EXTRA:
        for r in rows(path):add(r.get('qid',r.get('question_id')),r.get('question',r.get('question_text')),'historical_method_development')
    source_ids={p['source']['id'] for p in protected if p['source']['dataset'].casefold().startswith(dataset.casefold())}
    for r in metadata:
        # Exact ID additionally catches source metadata despite punctuation edits.
        item=add(r['qid'],r['question'],'known_source_id' if r['qid'] in source_ids else None,r['type'])
        official.append(item)
    grams=[];postings=defaultdict(list)
    for i,r in enumerate(accessible):
        tokens=re.findall(r'\w+',norm(r['question']));g={tuple(tokens[j:j+3]) for j in range(len(tokens)-2)}
        grams.append(g)
        for gram in g:postings[gram].append(i)
    checked=edges=0
    for i,g in enumerate(grams):
        candidates=set()
        for gram in sorted(g,key=lambda x:(len(postings[x]),x))[:5]:
            if len(postings[gram])<=200:candidates.update(j for j in postings[gram] if j<i)
        for j in candidates:
            old=grams[j]
            if not old or min(len(old),len(g))/max(len(old),len(g))<.82:continue
            checked+=1
            if len(old&g)/len(old|g)>=.82:
                groups.union(accessible[i]['index'],accessible[j]['index']);edges+=1
    reasons=defaultdict(set)
    for i,rs in blocked.items():reasons[groups.find(i)].update(rs)
    eligible=[];excluded=[]
    for r in official:
        root=groups.find(r['index']);rs=reasons[root];item={k:r[k] for k in ['qid','question','type']}
        item['group_id']=dataset+':group:'+str(root)
        (excluded if rs else eligible).append({**item,**({'reasons':sorted(rs)} if rs else {})})
    report={'dataset':dataset,'official':len(official),'eligible':len(eligible),'excluded':len(excluded),
       'overlapping_reason_counts':dict(Counter(x for r in excluded for x in r['reasons'])),
       'manifest_sha256':sha(MANIFEST),'actual_training_sha256':sha(MANIFEST.parent/'runtime_qwen_train.jsonl'),
       'lexical_pairs_checked':checked,'lexical_edges':edges,'sealed_project_test_questions_opened':False,
       'limitations':['Lexical heuristics do not prove semantic or base-model pretraining decontamination.',
                     'Protected project-test text is not read; exact/template/source metadata are used.',
                     'MuSiQue is target-transfer evaluation after excluding known query/family exposure, not a wholly unseen training source.']}
    return eligible,excluded,report

def write_dataset(dataset,selected,references,report,sources):
    out=HERE/'inputs_frozen'/dataset
    if out.exists():raise FileExistsError('Use frozen sample; never redraw '+dataset)
    out.mkdir(parents=True)
    save(out/'EXPOSURE_AUDIT.json',report)
    hashes={};summaries={}
    for split,records in selected.items():
        qs=[{'qid':r['qid'],'question':r['question']} for r in records]
        refs=[{**references[r['qid']], 'group_id':r['group_id'],'type':r['type'],'generation_input_forbidden':True} for r in records]
        for name,items in [(split+'.questions.jsonl',qs),(split+'.references.local.jsonl',refs)]:
            path=out/name;path.write_text(''.join(__import__('json').dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf8');hashes[name]=sha(path)
        positions={r['qid']:1+int(h(f'{SEED}:position:'+r['qid'])[:12],16)%10 for r in records}
        save(out/(split+'.positions.json'),positions);hashes[split+'.positions.json']=sha(out/(split+'.positions.json'))
        summaries[split]={'questions':len(records),'type_counts':dict(Counter(r['type'] for r in records)),'groups':len({r['group_id'] for r in records})}
    save(out/'SAMPLES_FROZEN.json',{'dataset':dataset,'seed':SEED,'files_sha256':hashes,'splits':summaries,
       'selection_used_answers_or_model_outputs':False,'dev_formal_groups_disjoint':True,'ablation_subset_formal':True,
       'sources':sources,'exposure_audit_sha256':sha(out/'EXPOSURE_AUDIT.json')})
    log('new_independent_sample_frozen',dataset=dataset,summaries=summaries,seed=SEED,source_records=sources)
    print(dataset,summaries,flush=True)

def prepare(dataset):
    if (HERE/'inputs_frozen'/dataset/'SAMPLES_FROZEN.json').exists():return
    if dataset=='hotpotqa':
        eligible,excluded,report=audit(EXTRA)
        selected=draw(eligible)  # IDs frozen in memory before gold is opened.
        import pyarrow.parquet as pq
        goldpath=ROOT/'experiments/query_decomposition/hotpot_formal_eval_v1/preparation/offline_gold_source/validation.parquet'
        gold={r['id']:r for r in pq.read_table(goldpath,columns=['id','question','answer','supporting_facts']).to_pylist()}
        qrels=defaultdict(list)
        for line in (ROOT/'data/beir/hotpotqa/qrels/test.tsv').read_text().splitlines()[1:]:
            qid,doc,score=line.split('\t')
            if int(score)>0:qrels[qid].append(doc)
        refs={r['qid']:{'qid':r['qid'],'question':r['question'],'answer':gold[r['qid']]['answer'],'aliases':[],
          'positive_doc_ids':qrels[r['qid']],'supporting_facts':gold[r['qid']]['supporting_facts']} for split in selected.values() for r in split}
        source=[{'path':str(goldpath.relative_to(ROOT)),'sha256':sha(goldpath)}]
    else:
        path=DATA/dataset/'native'/('dev.json' if dataset=='2wiki' else 'musique_ans_v1.0_dev.jsonl')
        native=read(path) if dataset=='2wiki' else rows(path)
        metadata=[{'qid':r.get('_id',r.get('id')),'question':r['question'],
             'type':r['type'] if dataset=='2wiki' else str(len(r['question_decomposition']))+'-hop'} for r in native]
        eligible,excluded,report=generic_audit(metadata,dataset)
        selected=draw(eligible);chosen={r['qid'] for split in selected.values() for r in split};refs={}
        for r in native:
            qid=r.get('_id',r.get('id'))
            if qid not in chosen:continue
            paragraphs=[(p[0],' '.join(p[1])) for p in r['context']] if dataset=='2wiki' else [(p['title'],p['paragraph_text']) for p in r['paragraphs']]
            supports={p[0] for p in r['supporting_facts']} if dataset=='2wiki' else {p['idx'] for p in r['paragraphs'] if p['is_supporting']}
            doc_ids=[h(t+'\n'+text) for idx,(t,text) in enumerate(paragraphs) if (t in supports if dataset=='2wiki' else r['paragraphs'][idx]['idx'] in supports)]
            refs[qid]={'qid':qid,'question':r['question'],'answer':r['answer'],'aliases':r.get('answer_aliases',[]),
                       'answer_id':r.get('answer_id',[]),'positive_doc_ids':sorted(set(doc_ids)),
                       'supporting_facts':r.get('supporting_facts',[])}
        source=[{'path':str(path.relative_to(ROOT)),'sha256':sha(path)}]
    write_dataset(dataset,selected,refs,report,source)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['hotpotqa','2wiki','musique']);args=p.parse_args();prepare(args.dataset)
