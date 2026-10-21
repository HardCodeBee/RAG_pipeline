"""Offline, gold-separated audit of rewrite support-document reader exposure."""
import collections
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/b0_strategy_round2_v1_20261020'
OUT = ROOT / 'outputs/query_decomposition/p02_merge_exposure_audit_20261020/agent_support'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())

def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for sub in ['pilot100_p02_role_headers_r001', 'extension200_p02_role_headers_r001']:
        paths.extend(sorted((SRC/'rounds'/sub/'cases').rglob('*.json')))
    cases = [read(p) for p in paths]
    assert len(cases) == len({c['id'] for c in cases}) == 300
    selected = {ds: {c['qid'] for c in cases if c['dataset']==ds} for ds in ['hotpotqa','2wiki','musique']}
    refs = {}
    import pyarrow.parquet as pq
    goldpath = ROOT/'experiments/query_decomposition/hotpot_formal_eval_v1/preparation/offline_gold_source/validation.parquet'
    hot = {r['id']:r for r in pq.read_table(goldpath, columns=['id','question','answer','supporting_facts']).to_pylist() if r['id'] in selected['hotpotqa']}
    qrels = collections.defaultdict(set)
    for line in (ROOT/'data/beir/hotpotqa/qrels/test.tsv').read_text(encoding='utf8').splitlines()[1:]:
        qid, did, rel = line.split('\t')
        if qid in selected['hotpotqa'] and int(rel)>0:qrels[qid].add(did)
    for qid,r in hot.items():
        sf=r['supporting_facts']; titles=sf['title'] if isinstance(sf,dict) else [x[0] for x in sf]
        refs['hotpotqa:'+qid]={'ids':qrels[qid], 'titles':set(map(norm,titles)), 'answer':r['answer'], 'question':r['question']}
    native=ROOT/'data/multihop_comprehensive_v1_20261015'
    for r in read(native/'2wiki/native/dev.json'):
        qid=r['_id']
        if qid not in selected['2wiki']:continue
        titles={s[0] for s in r['supporting_facts']}
        ids={sha(t+'\n'+' '.join(ss)) for t,ss in r['context'] if t in titles}
        refs['2wiki:'+qid]={'ids':ids,'titles':set(map(norm,titles)), 'answer':r['answer'], 'question':r['question']}
    with (native/'musique/native/musique_ans_v1.0_dev.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line);qid=r['id']
            if qid not in selected['musique']:continue
            docs=[p for p in r['paragraphs'] if p['is_supporting']]
            refs['musique:'+qid]={'ids':{sha(p['title']+'\n'+p['paragraph_text']) for p in docs}, 'titles':{norm(p['title']) for p in docs}, 'answer':r['answer'], 'question':r['question']}
    assert len(refs)==300
    nodes=[];case_details=[];verify=[]
    for path,c in zip(paths,cases):
        assert c['status'].startswith('success')
        ref=refs[c['id']];assert ref['question'].strip()==c['question'].strip()
        parent=read(ROOT/c['parent_source']);base=read(ROOT/c['source_prefix'])
        assert c['nodes']==parent['nodes'], c['id']
        # GOAL_RULE may change child prompts and answers, so only require base retrieval packets.
        assert len(c['nodes'])==len(base['nodes'])
        for cn,bn in zip(c['nodes'],base['nodes']):
            for key in ['original_retrieval','rewritten_retrieval','reader_packet','rewritten_query']:
                assert cn.get(key)==bn.get(key),(c['id'],cn['step_id'],key)
        union_reader={d['doc_id'] for n in c['nodes'] for d in n['reader_packet']['docs']}
        union_titles={norm(d['source']) for n in c['nodes'] for d in n['reader_packet']['docs']}
        root_ids={d['doc_id'] for d in base['original_retrieval']['docs']}
        root_titles={norm(d['source']) for d in base['original_retrieval']['docs']}
        case_nodes=[]
        for n in c['nodes']:
            rw=n.get('rewritten_retrieval')
            if not rw:continue
            od=n['original_retrieval']['docs'];rd=rw['docs'];md=n['reader_packet']['docs']
            mi={d['doc_id'] for d in md};mt={norm(d['source']) for d in md}
            oi={d['doc_id'] for d in od};ot={norm(d['source']) for d in od}
            rgi=[d for d in rd if d['doc_id'] in ref['ids']]
            rgt=[d for d in rd if norm(d['source']) in ref['titles']]
            il=[d for d in rgi if d['doc_id'] not in mi]
            tl=[d for d in rgt if norm(d['source']) not in mt]
            lost=[]
            for d in il:
                lost.append({'doc_id':d['doc_id'],'source':d['source'],'rewrite_rank':d['rank'],
                    'same_title_still_in_reader':norm(d['source']) in mt,
                    'exact_doc_present_in_original_top10':d['doc_id'] in oi,
                    'same_title_present_in_original_top10':norm(d['source']) in ot,
                    'exact_doc_read_by_other_node':d['doc_id'] in union_reader,
                    'same_title_read_by_other_node':norm(d['source']) in union_titles,
                    'exact_doc_in_final_original_top10':d['doc_id'] in root_ids,
                    'same_title_in_final_original_top10':norm(d['source']) in root_titles,
                    'text':d['text']})
            val={'id':c['id'],'dataset':c['dataset'],'question':c['question'],'step_id':n['step_id'],'role':n['role'],
                 'request':n['request'],'rewritten_query':n['rewritten_query'],'answer':n['answer'],
                 'rewrite_support_exact_count':len(rgi),'rewrite_support_title_count':len(rgt),
                 'lost_exact_count':len(il),'lost_title_count':len(tl),
                 'lost_exact':lost,
                 'lost_title_docs':[{'doc_id':d['doc_id'],'source':d['source'],'rewrite_rank':d['rank'],'text':d['text']} for d in tl],
                 'original_reader_doc_ids':[d['doc_id'] for d in md if d['doc_id'] in oi],
                 'rewrite_reader_doc_ids':[d['doc_id'] for d in md if d['doc_id'] in {x['doc_id'] for x in rd}],
                 'lost_rewrite_any_count':sum(d['doc_id'] not in mi for d in rd),
                 'rewrite_support_exact_kept_via_original_count':sum(d['doc_id'] in mi and d['doc_id'] in oi for d in rgi),
                 'reader_contains_all_rewrite_support_titles':all(norm(d['source']) in mt for d in rgt),
                 'source_case':str(path.relative_to(ROOT))}
            nodes.append(val);case_nodes.append(val)
        case_details.append({'id':c['id'],'dataset':c['dataset'],'question':c['question'],'prediction':c['prediction'],
            'gold_answer':ref['answer'],'nodes_total':len(c['nodes']),'rewrite_nodes':len(case_nodes),
            'rew_support_exact_any':any(n['rewrite_support_exact_count'] for n in case_nodes),
            'rew_support_title_any':any(n['rewrite_support_title_count'] for n in case_nodes),
            'lost_exact_any':any(n['lost_exact_count'] for n in case_nodes),
            'lost_title_any':any(n['lost_title_count'] for n in case_nodes),
            'lost_exact_no_same_title_any':any(any(not d['same_title_still_in_reader'] for d in n['lost_exact']) for n in case_nodes),
            'lost_exact_globally_unread_any':any(any(not d['exact_doc_read_by_other_node'] and not d['exact_doc_in_final_original_top10'] for d in n['lost_exact']) for n in case_nodes),
            'lost_title_globally_unread_any':any(any(not d['same_title_read_by_other_node'] and not d['same_title_in_final_original_top10'] for d in n['lost_exact']) for n in case_nodes)})
    def summ(ds=None):
        nn=[n for n in nodes if ds is None or n['dataset']==ds];cc=[c for c in case_details if ds is None or c['dataset']==ds]
        lost=[d for n in nn for d in n['lost_exact']]
        out={'cases_total':len(cc),'nodes_total':sum(c['nodes_total'] for c in cc),'rewrite_nodes':len(nn),
             'rewrite_nodes_with_any_dropped_doc':sum(n['lost_rewrite_any_count']>0 for n in nn),
             'rewrite_nodes_with_gold_exact':sum(n['rewrite_support_exact_count']>0 for n in nn),
             'rewrite_nodes_with_gold_title':sum(n['rewrite_support_title_count']>0 for n in nn),
             'nodes_lost_gold_exact':sum(n['lost_exact_count']>0 for n in nn),
             'nodes_lost_gold_title':sum(n['lost_title_count']>0 for n in nn),
             'gold_exact_occurrences_retrieved':sum(n['rewrite_support_exact_count'] for n in nn),
             'gold_exact_occurrences_lost':len(lost),
             'gold_exact_lost_same_title_still_in_reader':sum(d['same_title_still_in_reader'] for d in lost),
             'gold_exact_lost_same_title_absent_from_reader':sum(not d['same_title_still_in_reader'] for d in lost),
             'gold_exact_lost_present_in_original_top10':sum(d['exact_doc_present_in_original_top10'] for d in lost),
             'gold_exact_kept_and_also_in_original_top10':sum(n['rewrite_support_exact_kept_via_original_count'] for n in nn),
             'gold_exact_lost_global_unread':sum(not d['exact_doc_read_by_other_node'] and not d['exact_doc_in_final_original_top10'] for d in lost),
             'gold_exact_lost_global_title_unread':sum(not d['same_title_read_by_other_node'] and not d['same_title_in_final_original_top10'] for d in lost),
             'lost_exact_rank_histogram':dict(collections.Counter(d['rewrite_rank'] for d in lost)),
             'case_counts':{k:sum(c[k] for c in cc) for k in ['rew_support_exact_any','rew_support_title_any','lost_exact_any','lost_title_any','lost_exact_no_same_title_any','lost_exact_globally_unread_any','lost_title_globally_unread_any']}}
        return out
    summary={'scope':'Frozen development300,100/dataset; all p02 successful. No new model/retrieval requests.',
             'related_definition':'Dataset annotated original-query supporting documents, exact corpus doc IDs; title-based sensitivity also provided. These labels are offline diagnostics only. Not exhaustive semantic usefulness or per-child relevance.',
             'matching':'Hotpot BEIR positive qrels; 2Wiki sha256(title + newline + context joined sentence text), supporting titles; Mu sha256(title + newline + paragraph_text), is_supporting. Title sensitivity NFKC casefold whitespace.',
             'verified_p02_nodes_equal_g01_all300':True,'verified_all_original_rewritten_reader_packets_equal_B0_all300':True,
             'all':summ(),'by_dataset':{d:summ(d) for d in selected},'new_API_calls':0,'new_index_searches':0}
    for name,value in [('summary.json',summary),('node_details.json',nodes),('case_details.json',case_details)]:
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (OUT/'SUMMARY.txt').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
