"""Offline regression checks and real pinned-model token-coverage probes."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import numpy as np
from .common import *
from .catalog import Catalog
from .transport import Transport, API, Budget
from .answers import extract
from .retrieval import encode_complete, windows
from .reranking import Reranker
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store


def new_store(name):
    return Store(OUT/'offline_checks'/(name+'_'+uuid.uuid4().hex[:8]),{'offline_check':name})


class FakeBudget:
    def __init__(self):
        self.path=OUT/'offline_checks/fake_budget.sqlite3'
        self.cap=200
        self.reserves=[]
    def wait_for_start(self):pass
    def reserve(self,*a):self.reserves.append(a)
    def settle(self,*a):pass
    def summary(self):return {'offline_no_api':True}


class FakeClient:
    def __init__(self,finish='stop',usage=True):
        self.calls=[];self.finish=finish;self.has_usage=usage
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**payload):
        self.calls.append(payload)
        usage={'prompt_tokens':7,'completion_tokens':2,'total_tokens':9} if self.has_usage else {}
        raw={'usage':usage}
        return SimpleNamespace(model=MODEL,model_dump=lambda **kw:raw,
            choices=[SimpleNamespace(finish_reason=self.finish,message=SimpleNamespace(content='Final answer: Rome',refusal=None))])


def run():
    checks={}
    assert extract('Reason one.\nReason two unfinished')==''
    assert extract('Reason one.\n**Final answer:** Rome')=='Rome'
    assert extract('Rome')=='Rome'
    assert extract('Reason one.\nThe answer is: Rome')=='Rome'
    checks['no_raw_multiline_answer_fallback']=True
    source=new_store('catalog_source')
    def payload(prompt,cap):return {'model':MODEL,'messages':[{'role':'user','content':prompt}],
        'temperature':0.,'max_tokens':cap,'extra_body':{'enable_thinking':False}}
    for rid, prompt, finish in [('good','complete','stop'),('cut','incomplete','length')]:
        result={'status':'success','finish_reason':finish,'answer':'Final answer: Rome',
            'usage':{'prompt_tokens':4,'completion_tokens':2}}
        source.db.execute('insert into requests values(?,?,?,?,?)',(rid,'reader',canonical(payload(prompt,512)),'success',canonical(result)))
        source.put('evidence',rid,{'qid':'q1','stage':'reader','request_id':rid,'doc_ids':[]})
    source.db.commit()
    catalog=Catalog(source.directory/'checkpoint.sqlite3')
    assert catalog.request('q1','reader',payload('complete',32768))
    assert not catalog.request('q1','reader',payload('incomplete',32768))
    assert not catalog.request('q2','reader',payload('complete',32768))
    assert not catalog.request('q1','reader',payload('different question',32768))
    checks['complete_only_same_method_question_stage_payload_migration']=True
    store=new_store('transport')
    client=FakeClient()
    t=Transport(store,client,FakeBudget(),MODEL,catalog);t.qid='q1'
    key,result=t.ask('reader','complete',512)
    assert not client.calls
    actual=json.loads(store.db.execute('select payload from requests where id=?',(key,)).fetchone()[0])
    assert actual['max_tokens']==512
    key,result=t.ask('reader','A'*120000,64)
    assert client.calls[-1]['max_tokens']==32768
    assert len(client.calls[-1]['messages'][0]['content'])==120000
    t.ask('reader','A'*120000,64)
    assert len(client.calls)==1
    assert store.db.execute("select count(*) from data where kind='token_attempt'").fetchone()[0]==1
    checks['raised_cap_full_input_truthful_legacy_cap_and_new_usage_once']=True
    for finish, usage in [('length',True),('stop',False)]:
        bad=new_store('blocked')
        fake=FakeClient(finish,usage)
        tt=Transport(bad,fake,FakeBudget(),MODEL,catalog);tt.qid='q3'
        try:tt.ask('reader','never reused '+finish+str(usage),512)
        except IntegrityError:pass
        else:raise AssertionError('Incomplete or unmetered output was accepted')
        assert bad.db.execute("select count(*) from data where kind='integrity_violation'").fetchone()[0]==1
        assert bad.db.execute("select count(*) from data where kind='prediction'").fetchone()[0]==0
        bad.db.close()
    checks['capped_and_unmetered_outputs_blocked_before_scoring']=True
    try:t.ask('reader','B'*1000000,512)
    except IntegrityError:pass
    else:raise AssertionError('Oversized input was sent')
    assert len(client.calls)==1
    checks['context_overflow_rejected_without_cropping_or_call']=True
    parallel_store=new_store('parallel_attribution')
    fake=FakeClient()
    tt=Transport(parallel_store,fake,FakeBudget(),MODEL,catalog)
    api=API(tt,2,2/3)
    try:
        futures=[api.pool.submit(api._ask,q,('reader','Parallel '+q,512),{}) for q in ['qa','qb']]
        for future in futures:future.result()
    finally:api.close()
    recorded=[json.loads(v) for v, in parallel_store.db.execute("select value from data where kind='token_attempt'")]
    assert sorted(v['qid'] for v in recorded)==['qa','qb']
    assert len(fake.calls)==2
    parallel_store.db.close()
    checks['concurrent_attempt_tokens_attributed_to_correct_query']=True
    assert usage_cost(300000,100)==300000*4.8e-6+100*48e-6
    checks['tiered_pricing_large_input_reservations']=True
    for mid,name in METHODS.items():
        parent=('experiments.query_decomposition.comprehensive_eval_v1_20261015.methods.' if name=='qd_rerank' else
                'experiments.query_decomposition.hotpot_formal_eval_v1.methods.')+name+'.method'
        old=importlib.import_module(parent)
        new=importlib.import_module(__package__+'.methods.'+name+'.method')
        for constant in ['INSTRUCTIONS','PROMPT','READ','REASON','OUTER','INNER','CHILD','READER','DECOMPOSE']:
            if hasattr(old,constant):assert getattr(old,constant)==getattr(new,constant),(mid,constant)
        original=Path(old.__file__).with_name('examples.json')
        if original.exists():assert original.read_bytes()==Path(new.__file__).with_name('examples.json').read_bytes()
    checks['all_nine_prompts_and_demonstrations_unchanged']=True
    # Actual20-case replay validates retrieve_many compatibility and scheduling.
    from .retrieval import Retrieval
    from .executor import execute
    native=read(PARENT/'2wiki/BASELINE_CONFIGS_FROZEN.json')['selected']['b00_rag_top10_v1']
    source_path=PARENT/'2wiki/dev/b00_rag_top10_v1'/native['id']/'checkpoint.sqlite3'
    replay_catalog=Catalog(source_path)
    cases=rows(INPUTS/'2wiki/dev.questions.jsonl')[:20]
    resource=read(PARENT/'2wiki/RESOURCE_FROZEN.json')['resources']
    method=importlib.import_module(__package__+'.methods.rag_top10_v1.method')
    snapshots=[]
    for parallel in [False,True]:
        replay_store=new_store('real_replay_parallel' if parallel else 'real_replay_serial')
        replay_client=FakeClient()
        tr=Transport(replay_store,replay_client,FakeBudget(),MODEL,replay_catalog)
        rt=Retrieval(replay_store,resource,replay_catalog)
        if parallel:
            api=API(tr,8,2/3)
            try:execute(method,cases,rt,tr,replay_store,native,api,20)
            finally:api.close()
        else:method.run(cases,rt,tr,replay_store,native)
        assert not replay_client.calls,'A supposedly exact historical replay needed a new response'
        snapshots.append({key:json.loads(value) for key,value in replay_store.db.execute("select key,value from data where kind='prediction'")})
        replay_store.db.close()
    # Namespaced request IDs differ but prompts/predictions/documents must agree.
    for values in snapshots:
        for value in values.values():value.pop('request_id',None)
    assert snapshots[0]==snapshots[1]
    checks['twenty_real_questions_serial_async_retrieve_many_parity']=True
    from src.embedders.text_embedder import TextEmbedder
    spec=read(PARENT/'2wiki/RESOURCE_FROZEN.json')['resources']['dense']['bge']
    encoder=TextEmbedder(backend='sentence_transformers',model_name=spec['query_model'],revision=spec['query_revision'],
        normalize=True,batch_size=16,query_prefix='',document_prefix='',max_sequence_length=512,local_files_only=True,device='cuda')
    short=['Who directed Alien (1979)?','When was Manhattan Center built?']
    original=encoder.encode_queries(short)
    actual, meta=encode_complete(encoder,short)
    assert np.array_equal(original,actual)
    long=['evidence '*1500+'The final question asks for Manhattan Center construction year.']
    vector, meta=encode_complete(encoder,long)
    assert meta[0]['window_count']>1 and not meta[0]['encoder_truncated']
    assert meta[0]['processed_content_tokens']==meta[0]['content_tokens']
    assert max(meta[0]['encoded_window_tokens'])<=512
    assert np.isclose(np.linalg.norm(vector[0]),1.,atol=1e-5)
    checks['real_bge_short_exact_parity_full_overflow_token_coverage']=meta[0]
    rerank_store=new_store('reranker')
    rr=Reranker(rerank_store)
    docs=[{'doc_id':'1','text':'Paris is the capital of France.','rrf_score':1},
          {'doc_id':'2','text':'The monument stands in Rome.','rrf_score':.5}]
    ranked,_=rr.rerank('What is the capital of France?',docs,2,qid='short')
    raw=rr.model.predict([('What is the capital of France?',d['text']) for d in docs],batch_size=4,show_progress_bar=False)
    assert {d['doc_id']:d['rerank_score'] for d in ranked}=={d['doc_id']:float(x) for d,x in zip(docs,raw)}
    rr.rerank('What is the capital of France?', [{'doc_id':'3','text':'A long passage. '*800+'Paris is the capital of France.','rrf_score':1}],1,qid='long')
    row=json.loads(rerank_store.db.execute("select value from data where kind='rerank_token' and json_extract(value,'$.qid')='long'").fetchone()[0])
    assert row['windows']>1 and not row['input_truncated']
    checks['real_reranker_short_exact_parity_full_overflow_pair_coverage']=row
    for s in [source,store,rerank_store]:s.db.close()
    result={'utc':utc(),'passed':True,'checks':checks,'api_calls':0,'tests_use_fake_client_or_local_pinned_weights':True,
        'validated_source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in HERE.rglob('*.py')}}
    save(OUT/'OFFLINE_VALIDATION.json',result)
    journal('offline_validation_completed',passed=True,api_calls=0,checks=list(checks))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:
        run()
    except BaseException as exc:
        journal('offline_validation_failed',error=type(exc).__name__,reason=str(exc),api_calls=0)
        raise
