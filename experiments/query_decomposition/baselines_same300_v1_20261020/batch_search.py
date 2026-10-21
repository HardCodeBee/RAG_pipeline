"""Share corpus scans; retain each query's original encoder/FAISS arithmetic."""
from concurrent.futures import Future
from queue import Queue, Empty
from threading import Thread
import time
import numpy as np
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import WorkerStore


def exact_packets(retriever, jobs, k=10):
    """One original encode per query; original search kernel with batch_size=1."""
    retriever.load()
    import faiss
    faiss.omp_set_num_threads(4)
    tokenizer= retriever.encoder._model[0].tokenizer
    vectors=np.ascontiguousarray(np.concatenate([retriever.encoder.encode_queries([j['query']]) for j in jobs],axis=0),dtype=np.float32)
    counts=[len(tokenizer(j['query'],add_special_tokens=True,truncation=False)['input_ids']) for j in jobs]
    scores,ids=retriever.index.search_many(vectors,k+1,query_batch_size=1)
    packets=[]
    for job,values,indices,count in zip(jobs,scores.tolist(),ids.tolist(),counts):
        docs=[]
        for score,chunk in zip(values,retriever.chunks.get_many(indices)):
            if chunk.doc_id==job['qid']:continue
            rank=len(docs)+1
            docs.append({'rank':rank,'score':score,'doc_id':chunk.doc_id,'vector_id':chunk.vector_id,
                         'source':chunk.source,'text':chunk.text,
                         'block':f'[Chunk {rank} | source={chunk.source}, pages={chunk.page_start}-{chunk.page_end}]\n{chunk.text}'})
            if len(docs)==k:break
        assert len(docs)==k
        packets.append({'qid':job['qid'],'query':job['query'],'k':k,'docs':docs,'query_tokens':count,
                        'encoder_truncated':count>512,'batch_queries':len(jobs),'execution':'single-query encoding and FAISS kernels; shared read-only corpus blocks'})
    return packets


class BatchSearch:
    def __init__(self, owner, retriever, max_batch=20):
        self.owner,self.retriever,self.max_batch=owner,retriever,max_batch
        self.queue=Queue();self.thread=Thread(target=self._work,daemon=True);self.thread.start()
    def get(self,qid,query,k):
        future=Future();self.queue.put((qid,query,k,future));return future.result()
    def close(self):
        self.queue.put(None);self.thread.join()
    def _work(self):
        store=WorkerStore(self.owner);self.retriever.store=store
        try:
            while True:
                first=self.queue.get()
                if first is None:break
                group=[first];deadline=time.monotonic()+.15
                while len(group)<self.max_batch:
                    try:
                        value=self.queue.get(timeout=max(.001,deadline-time.monotonic()))
                    except Empty:break
                    if value is None:raise RuntimeError('BatchSearch closed while requests pending')
                    group.append(value)
                    if time.monotonic()>deadline:break
                missing=[]
                for qid,query,k,future in group:
                    key=digest([qid,query,k,self.retriever.resources]);packet=store.get('retrieval',key)
                    if packet is not None:future.set_result((packet,True,0.))
                    else:missing.append((qid,query,k,future,key))
                if not missing:continue
                started=time.perf_counter()
                try:
                    assert {v[2] for v in missing}=={10}
                    packets=exact_packets(self.retriever,[{'qid':v[0],'query':v[1]} for v in missing])
                    elapsed=time.perf_counter()-started
                    for value,packet in zip(missing,packets):
                        packet['batch_seconds']=elapsed
                        store.put('retrieval',value[4],packet)
                        value[3].set_result((packet,False,elapsed/len(missing)))
                    store.event('exact_shared_scan',queries=len(missing),seconds=elapsed,query_batch_size=1,single_query_encoding=True)
                except BaseException as exc:
                    for value in missing:
                        if not value[3].done():value[3].set_exception(exc)
        finally:store.db.close()


def validate():
    """Compare fresh batched exact scores/IDs/blocks against real frozen hits."""
    import sqlite3,json
    from .common import ROOT,OUT,HERE,read,prepare,source_databases,sha,save,log
    from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
    plan=prepare();install();resource=plan['resources']['hotpotqa']
    source=OUT/'attempts/r000/retrieval/hotpotqa/checkpoint.sqlite3'
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as db:
        values=[json.loads(v) for (v,) in db.execute("SELECT value FROM data WHERE kind='retrieval'")]
    # These observations were actually computed by the original single-query path.
    values=[v for v in values if v.get('batch_queries')==1 and v.get('k')==10][-5:]
    assert len(values)==5
    store=Store(OUT/'offline_batch_validation',{'task':HERE.name,'api_calls':0})
    retriever=Retrieval(store,resources=resource)
    started=time.perf_counter()
    result=exact_packets(retriever,[{'qid':v['qid'],'query':v['query']} for v in values])
    assert all(a['docs']==b['docs'] and a['query_tokens']==b['query_tokens'] for a,b in zip(values,result))
    record={'passed':True,'queries':5,'docs_scores_ids_and_blocks_bit_identical':True,'single_query_encoding':True,
            'original_faiss_query_batch_size':1,'new_faiss_query_batch_size':1,'seconds':time.perf_counter()-started,
            'new_api_calls':0,'source_sha256':sha(__file__),'tested_queries':[{'qid':v['qid'],'query':v['query']} for v in values]}
    save(OUT/'EXACT_BATCH_PARITY.frozen.json',record);log('offline_batch_parity',**record)
    store.db.close()
    if retriever.chunks is not None:retriever.chunks.close()
    print(json.dumps(record))


if __name__=='__main__':validate()
