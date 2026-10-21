"""Execution-only QD batching; original algorithm files and private state retained."""
from collections import OrderedDict
import json
import time
import numpy as np
from .common import ROOT,HERE,RUN,read,save,sha,log,utc
from . import campaign
from .campaign import TARGET
from .execution import run_parallel
from .methods.qd_rerank import method as qd
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest

class Decompose:
    @staticmethod
    def run(cases,retrieval,transport,store,config):
        for case in cases:
            qid=case['qid'];transport.qid=qid
            if store.get('prediction',qid) is not None or store.get('standalone_queries',qid) is not None:continue
            key,raw=transport.ask('decompose',qd.DECOMPOSE.format(max_queries=config['max_queries'],question=case['question']),512)
            if raw['status']!='success':
                store.put('prediction',qid,{'qid':qid,'question':case['question'],'prediction':None,'status':raw['status'],'request_id':key,'failed_stage':'decompose'});continue
            error=None
            try:subqueries=qd.parse(raw.get('answer',''),config['max_queries'])
            except (ValueError,TypeError) as exc:subqueries=[];error=str(exc)
            store.put('standalone_queries',qid,{'qid':qid,'queries':list(dict.fromkeys([case['question'],*subqueries])),'raw':raw['answer'],'parse_error':error})

class Final:
    @staticmethod
    def run(cases,retrieval,transport,store,config):
        for case in cases:
            qid=case['qid'];transport.qid=qid
            if store.get('prediction',qid) is not None:continue
            packet=store.get('hybrid',qid)
            key,raw=transport.ask('final',qd.PROMPT.format(question=case['question'],context='\n\n'.join(d['block'] for d in packet['docs'])),64)
            store.put('prediction',qid,{'qid':qid,'question':case['question'],'prediction':raw.get('answer') if raw['status']=='success' else None,
              'status':raw['status'],'request_id':key,'finish_reason':raw.get('finish_reason'),'final_doc_ids':[d['doc_id'] for d in packet['docs']]})
            store.export()

def batch_retrieve(retrieval,cases,k):
    """Same per-question encoder groups and direct FAISS kernels; one disk scan."""
    store=retrieval.store;resources=retrieval.resources
    keys=[digest([r['qid'],r['query'],k,resources]) for r in cases]
    pending=[(r,key) for r,key in zip(cases,keys) if store.get('retrieval',key) is None]
    if not pending:return [store.get('retrieval',key) for key in keys]
    retrieval.load();groups=OrderedDict()
    for row,key in pending:groups.setdefault(row['qid'],[]).append((row,key))
    ordered=[item for group in groups.values() for item in group]
    vectors=[];counts=[];tokenizer=retrieval.encoder._model[0].tokenizer
    started=time.perf_counter()
    for group in groups.values():
        text=[r['query'] for r,key in group]
        counts.extend(len(tokenizer(q,add_special_tokens=True,truncation=False)['input_ids']) for q in text)
        vectors.append(np.ascontiguousarray(retrieval.encoder.encode_queries(text),dtype=np.float32))
    vectors=np.ascontiguousarray(np.concatenate(vectors),dtype=np.float32)
    scores,ids=retrieval.index.search_many(vectors,k+1,query_batch_size=1)
    elapsed=time.perf_counter()-started
    for (case,key),values,indices,count in zip(ordered,scores.tolist(),ids.tolist(),counts):
        docs=[]
        for score,chunk in zip(values,retrieval.chunks.get_many(indices)):
            if chunk.doc_id==case['qid']:continue
            rank=len(docs)+1
            docs.append({'rank':rank,'score':score,'doc_id':chunk.doc_id,'vector_id':chunk.vector_id,'source':chunk.source,'text':chunk.text,
                'block':f'[Chunk {rank} | source={chunk.source}, pages={chunk.page_start}-{chunk.page_end}]\n{chunk.text}'})
            if len(docs)==k:break
        if len(docs)!=k:raise ValueError('Incomplete batched retrieval')
        store.put('retrieval',key,{'qid':case['qid'],'query':case['query'],'k':k,'docs':docs,'query_tokens':count,
           'encoder_truncated':count>512,'batch_queries':len(ordered),'batch_seconds':elapsed,
           'encoder_grouping':'original per-question standalone-query group','faiss_query_batch_size':1})
        store.event('retrieval',qid=case['qid'],query=case['query'],k=k,doc_ids=[d['doc_id'] for d in docs])
    store.event('qd_batch_dense_search',questions=len(groups),queries=len(ordered),seconds=elapsed)
    store.export()
    return [store.get('retrieval',key) for key in keys]

def run(cases,retrieval,transport,store,config):
    epoch=read(TARGET/'QD_BATCH_EXECUTION_EPOCH.json')
    store.put('execution_epoch',epoch['id'],epoch)
    run_parallel(Decompose,cases,retrieval,transport,store,config)
    pending=[c for c in cases if store.get('prediction',c['qid']) is None and store.get('hybrid',c['qid']) is None]
    queries=[{'qid':c['qid'],'query':q} for c in pending for q in store.get('standalone_queries',c['qid'])['queries']]
    batch_retrieve(retrieval,queries,10)
    reranker=qd.Reranker() if pending else None
    for case in pending:
        union={}
        for query in store.get('standalone_queries',case['qid'])['queries']:
            packet=retrieval.retrieve(case['qid'],query,10)
            for d in packet['docs']:union.setdefault(d['doc_id'],{**d,'rrf_score':0})
        docs,seconds=reranker.rerank(case['question'],list(union.values()),10)
        store.put('hybrid',case['qid'],{'qid':case['qid'],'docs':docs,'union_doc_ids':sorted(union),'rerank_seconds':seconds})
    run_parallel(Final,cases,retrieval,transport,store,config)

def install():
    receipt_path=TARGET/'CAMPAIGN_FROZEN.json';receipt=read(receipt_path)
    campaign.verify(receipt)
    check=read(RUN/'QD_BATCH_PARITY.json')
    if not check['passed'] or check['api_submissions']!=0:raise ValueError('QD batching parity did not pass')
    epoch={'id':'qd_batch_epoch_v2_20261015','source_sha256':sha(__file__),
       'original_campaign_sha256':sha(receipt_path),'parity_sha256':sha(RUN/'QD_BATCH_PARITY.json'),
       'original_algorithm_sources_changed':False,'prompts_model_samples_parameters_changed':False,
       'changes':['overlap independent decomposition calls','same per-question BGE encoder groups; one corpus scan','serial unchanged rerank','overlap independent final calls'],
       'existing_private_checkpoints_and_successful_responses_preserved':True}
    epoch_path=TARGET/'QD_BATCH_EXECUTION_EPOCH.json'
    if epoch_path.exists() and read(epoch_path)!=epoch:raise ValueError('Execution epoch changed')
    if not epoch_path.exists():save(epoch_path,epoch);log('execution_only_qd_batch_epoch_activated',**epoch)
    def frozen():
        campaign.verify(receipt)
        if sha(__file__)!=epoch['source_sha256']:raise ValueError('QD execution source changed')
        options=__import__('src.generators.api_options',fromlist=['get_api_option']).get_api_option(2)
        if options['model']!=receipt['generator'] or options['base_url']!=receipt['base_url']:raise ValueError('Frozen API identity changed')
        return receipt,options
    campaign.freeze=frozen
    qd.run=run
    old_execute=campaign.Campaign.execute
    def execute(self,dataset,split,mid,config,cases):
        if sha(__file__)!=epoch['source_sha256']:raise ValueError('QD execution source changed')
        result=old_execute(self,dataset,split,mid,config,cases)
        for key,store in self.stores.items():
            if store.get('execution_epoch',epoch['id']) is None:store.put('execution_epoch',epoch['id'],epoch)
        return result
    campaign.Campaign.execute=execute

if __name__=='__main__':
    install();campaign.main()
