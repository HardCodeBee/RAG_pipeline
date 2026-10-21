"""Real cached-query parity; no generation or paid API requests."""
import json
import sqlite3
import time
from .common import ROOT,RUN,read,save,sha,log
from .campaign import TARGET
from .qd_batch_epoch import batch_retrieve
from .retrieval import Retrieval
from .methods.qd_rerank import method as qd

class Cache:
    def __init__(self):self.data={}
    def get(self,kind,key):return self.data.get((kind,key))
    def put(self,kind,key,value):
        old=self.get(kind,key)
        if old is not None and old!=value:raise ValueError('Immutable validation cache differs')
        self.data[(kind,key)]=value
    def event(self,*args,**kwargs):pass
    def export(self,*args,**kwargs):pass

def main():
    progress=RUN/'QD_BATCH_PARITY_PROGRESS.json';started=time.perf_counter();picked=[]
    root=TARGET/'hotpotqa/dev/b08_qd_rerank'
    for config,start in [('standalone3',0),('standalone5',4)]:
        db=sqlite3.connect((root/config/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
        values=[(key,json.loads(raw)) for key,raw in db.execute("SELECT key,value FROM data WHERE kind=?",('prediction',))]
        for qid,prediction in values[start:start+4]:
            if prediction.get('prediction') is None:continue
            record=json.loads(db.execute('SELECT value FROM data WHERE kind=? AND key=?',('standalone_queries',qid)).fetchone()[0])
            packets={r['query']:r for (raw,) in db.execute('SELECT value FROM data WHERE kind=? AND json_extract(value,?)=?',('retrieval','$.qid',qid)) for r in [json.loads(raw)]}
            hybrid=json.loads(db.execute('SELECT value FROM data WHERE kind=? AND key=?',('hybrid',qid)).fetchone()[0])
            payload=json.loads(db.execute('SELECT payload FROM requests WHERE id=?',(prediction['request_id'],)).fetchone()[0])
            picked.append({'qid':qid,'question':prediction['question'],'queries':record['queries'],'packets':packets,'hybrid':hybrid,'final_prompt':payload['messages'][0]['content']})
        db.close()
    if len(picked)!=8 or len({r['qid'] for r in picked})!=8:raise ValueError('Need eight distinct saved questions')
    cache=Cache();retrieval=Retrieval(cache,resources=read(ROOT/'experiments/query_decomposition/query_form_ablation_v1/resources.json'))
    queries=[{'qid':r['qid'],'query':q} for r in picked for q in r['queries']]
    save(progress,{'status':'batch_real_corpus_search','questions':8,'queries':len(queries)})
    found=batch_retrieve(retrieval,queries,10);elapsed=time.perf_counter()-started;max_diff=0.
    references={(r['qid'],q):r['packets'][q] for r in picked for q in r['queries']}
    for case,packet in zip(queries,found):
        old=references[(case['qid'],case['query'])]
        assert [d['doc_id'] for d in packet['docs']]==[d['doc_id'] for d in old['docs']],'Top10 document order changed'
        assert [d['block'] for d in packet['docs']]==[d['block'] for d in old['docs']],'Prompt passages changed'
        max_diff=max(max_diff,max(abs(a['score']-b['score']) for a,b in zip(packet['docs'],old['docs'])))
    save(progress,{'status':'rerank_parity','search_seconds_including_load':elapsed,'retrieval_cases_matched':len(queries)})
    reranker=qd.Reranker();prompt_matches=0
    for case in picked:
        union={}
        for query in case['queries']:
            packet=next(p for c,p in zip(queries,found) if c['qid']==case['qid'] and c['query']==query)
            for doc in packet['docs']:union.setdefault(doc['doc_id'],{**doc,'rrf_score':0})
        docs,seconds=reranker.rerank(case['question'],list(union.values()),10)
        assert [d['doc_id'] for d in docs]==[d['doc_id'] for d in case['hybrid']['docs']],'Rerank document order changed'
        prompt=qd.PROMPT.format(question=case['question'],context='\n\n'.join(d['block'] for d in docs))
        assert prompt==case['final_prompt'],'Final SDK prompt changed'
        prompt_matches+=1
    result={'passed':True,'api_submissions':0,'questions':len(picked),'retrieval_queries':len(queries),
      'top10_document_order_and_text_matches':len(queries),'reranked_top10_order_matches':8,'final_sdk_prompts_byte_identical':prompt_matches,
      'max_dense_score_absolute_difference':max_diff,'batch_search_seconds_including_initial_load':elapsed,
      'total_validation_seconds':time.perf_counter()-started,'source_sha256':sha(__file__),
      'execution_source_sha256':sha(__import__('pathlib').Path(__file__).with_name('qd_batch_epoch.py')),
      'original_algorithm_prompts_inputs_model_and_parameters_changed':False}
    save(RUN/'QD_BATCH_PARITY.json',result);save(progress,{'status':'completed',**result});log('qd_batch_real_corpus_parity_passed',**result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
