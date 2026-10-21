"""Exact-success reuse for ablations; unchanged bounded API infrastructure."""
import json
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_resume import ResumeTransport
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelAPI,ParallelBudget,WorkerStore
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_execution import _execute

class ExactTransport(ResumeTransport):
    def ask(self,stage,prompt,max_tokens):
        key,result=super().ask(stage,prompt,max_tokens)
        # ResumeTransport preserves the exact returned response. For this new
        # campaign, also bind its evidence to the passages actually shown now.
        shown={}
        for (raw,) in self.store.db.execute("SELECT value FROM data WHERE kind IN ('retrieval','sparse_retrieval','hybrid') AND json_extract(value,'$.qid')=?",(self.qid,)):
            packet=json.loads(raw)
            for doc in packet.get('docs',[]):
                if doc.get('text') and doc['text'] in prompt:shown[doc['doc_id']]=doc
        self.store.put('evidence',digest([self.qid,key,stage]),{'qid':self.qid,'stage':stage,'request_id':key,'doc_ids':sorted(shown),
            'source_match':'exact raw corpus text present in this call; generated answers are not source evidence'})
        return key,result

class API(ParallelAPI):
    def _ask(self,qid,args,kwargs):
        stage,prompt,max_tokens=args
        if kwargs:raise ValueError('Unexpected transport arguments')
        original=self.transport
        key=digest([original.store.namespace,{'model':original.model,'messages':[{'role':'user','content':prompt}],
          'temperature':0.,'max_tokens':max_tokens,'extra_body':{'enable_thinking':False}}])
        from threading import Lock
        with self.keys_lock:mutex=self.keys.setdefault(key,Lock())
        with mutex:
            store=WorkerStore(original.store);budget=ParallelBudget(original.budget.path,original.budget.cap)
            try:
                transport=ExactTransport(store,original.client,budget,original.model);transport.qid=qid
                return transport.ask(stage,prompt,max_tokens)
            finally:store.db.close();budget.db.close()

def run_parallel(module,cases,retrieval,transport,store,config):
    api=API(transport,3)
    try:return _execute(module,cases,retrieval,transport,store,config,api,20)
    finally:api.close()
