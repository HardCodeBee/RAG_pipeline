"""QD+Rerank compatibility: standalone subqueries, union, original-query rerank."""
import json
import re
from ..hybrid_rerank.method import Reranker, PROMPT

DECOMPOSE = '''Decompose the question into at most {max_queries} distinct standalone search questions whose evidence helps answer it.
Each search question must name its entities and preserve relevant conditions. Do not use numbered references such as #1, invent facts, or answer the question.
Return only a JSON array of question strings. If decomposition is unnecessary, return an empty array.

Question:
{question}
'''

def parse(text,max_queries):
    value=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip()))
    if not isinstance(value,list) or len(value)>max_queries or any(not isinstance(q,str) or not q.strip() or re.search(r'#\d+',q) for q in value):
        raise ValueError('Invalid standalone subqueries')
    return list(dict.fromkeys(q.strip() for q in value))

def run(cases,retrieval,transport,store,config):
    reranker=None
    for case in cases:
        qid=case['qid'];transport.qid=qid
        if store.get('prediction',qid) is not None:continue
        record=store.get('standalone_queries',qid)
        if record is None:
            key,raw=transport.ask('decompose',DECOMPOSE.format(max_queries=config['max_queries'],question=case['question']),512)
            if raw['status']!='success':
                store.put('prediction',qid,{'qid':qid,'question':case['question'],'prediction':None,'status':raw['status'],'request_id':key,'failed_stage':'decompose'});continue
            error=None
            try:subqueries=parse(raw.get('answer',''),config['max_queries'])
            except (ValueError,TypeError) as exc:subqueries=[];error=str(exc)
            record={'qid':qid,'queries':list(dict.fromkeys([case['question'],*subqueries])),'raw':raw['answer'],'parse_error':error}
            store.put('standalone_queries',qid,record)
        packet=store.get('hybrid',qid)
        if packet is None:
            found=retrieval.retrieve_many([{'qid':qid,'query':q} for q in record['queries']],10)
            union={}
            for p in found:
                for d in p['docs']:union.setdefault(d['doc_id'],{**d,'rrf_score':0})
            if reranker is None:reranker=Reranker(store)
            docs,seconds=reranker.rerank(case['question'],list(union.values()),10,qid=qid)
            packet={'qid':qid,'docs':docs,'union_doc_ids':sorted(union),'rerank_seconds':seconds}
            store.put('hybrid',qid,packet)
        key,raw=transport.ask('final',PROMPT.format(question=case['question'],context='\n\n'.join(d['block'] for d in packet['docs'])),64)
        store.put('prediction',qid,{'qid':qid,'question':case['question'],'prediction':raw.get('answer') if raw['status']=='success' else None,
            'status':raw['status'],'request_id':key,'finish_reason':raw.get('finish_reason'),'final_doc_ids':[d['doc_id'] for d in packet['docs']]})
        store.export()
