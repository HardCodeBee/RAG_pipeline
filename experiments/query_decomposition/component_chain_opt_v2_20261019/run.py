"""Isolated factorial / repair / fresh confirmation, one-shot within global200."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import ExitStack
import hashlib,json,os,socket,sqlite3,ssl
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
import httpx
from openai import OpenAI
from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store,digest,utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget,WorkerStore,locked_json
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.decomposition_error_expand60_v1_20261019.run import OneShot
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
from .prepare import ROOT,HERE,OUT,RUN,OUTPUT,PREVIOUS_E,PREVIOUS_O,CAMPAIGN,read,rows,sha,save,log
from .policy import execute,IntermediateLength

RETRIEVAL_LOCK=Lock();KEYS_LOCK=Lock();KEYS={};CIRCUIT_LOCK=Lock();ERRORS=0


class RecordedFailure(RuntimeError):
    def __init__(self,stage,rid,status,source):
        super().__init__('Recorded unsuccessful request; no replay: '+status)
        self.stage,self.request_id,self.request_status,self.source=stage,rid,status,source


class Budget(ParallelBudget):
    def __init__(self,ceiling):
        super().__init__(OUTPUT/'budget.sqlite3',200);self.ceiling=ceiling
    def reserve(self,rid,namespace,amount):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total=self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
            local=self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?',
                                 (str(RUN.relative_to(OUTPUT))+'%',)).fetchone()[0]
            if total+amount>self.ceiling or local+amount>1.85:
                raise RuntimeError('Budget dispatch ceiling reached; no submission')
            if self.db.execute('SELECT 1 FROM charges WHERE id=?',(rid,)).fetchone():
                raise RuntimeError('Existing liability cannot be submitted again')
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')",(rid,namespace,amount));self.db.commit()
        except BaseException:self.db.rollback();raise


def catalog(dataset):
    path=PREVIOUS_O/'run_v001/api'/dataset/'checkpoint.sqlite3'
    result={}
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        for rid,payload,status,value in db.execute('SELECT id,payload,status,result FROM requests'):
            key=digest(json.loads(payload));r={'request_id':rid,'status':status,'result':json.loads(value) if value else None,
                                             'source':str(path.relative_to(ROOT))}
            if key not in result or status=='success':result[key]=r
    return result


def payload(model,prompt,maximum):
    return {'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0.,
            'max_tokens':maximum,'extra_body':{'enable_thinking':False}}


def use_receipt(row,stage,allow_final_truncation):
    if row['status']!='success':
        raise RecordedFailure(stage,row['request_id'],row['status'],row['source'])
    r=row['result'];finish=r.get('finish_reason')
    if finish=='length' and not allow_final_truncation:
        raise IntermediateLength(stage,row['request_id'],r['answer'])
    if finish not in ('stop','length') or r.get('status')!='success':
        raise RecordedFailure(stage,row['request_id'],'invalid_finish_or_model',row['source'])
    return row['request_id'],r['answer'].strip(),finish


def seed(owner,dataset,cases,resources):
    """All imports finish before retriever threads start; immutable exact keys only."""
    old=PREVIOUS_O/'run_v001/api'/dataset/'checkpoint.sqlite3'
    count=0
    with sqlite3.connect(old.resolve().as_uri()+'?mode=ro',uri=True) as db:
        for key,value in db.execute("SELECT key,value FROM data WHERE kind='retrieval'"):
            p=json.loads(value)
            assert key==digest([p['qid'],p['query'],10,resources])
            owner.put('retrieval',key,p);count+=1
    source=CAMPAIGN/dataset/'formal/b00_rag_top10_v1/fixed/checkpoint.sqlite3'
    if source.exists():
        with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as db:
            for c in cases:
                key=digest([c['qid'],c['question'],10,resources])
                if owner.get('retrieval',key):continue
                value=db.execute("SELECT value FROM data WHERE kind='retrieval' AND key=?",(key,)).fetchone()
                if value:owner.put('retrieval',key,json.loads(value[0]));count+=1
    return count


def identity(plan):
    previous=read(PREVIOUS_E/'SELECTION.frozen.json')
    assert sha(PREVIOUS_E/'runtime60.frozen.jsonl')==plan['development']['runtime_sha256']
    assert sha(PREVIOUS_O/'CASE_RESULTS.jsonl')==plan['development']['source_scores_sha256']
    chain=ROOT/'experiments/query_decomposition/component_chain_c08_c02_v1_20261019'
    assert sha(chain/'PROTOCOL.frozen.json')==plan['component_protocol_sha256']
    for path,expected in read(chain/'PROTOCOL.frozen.json')['component_pins'].items():assert sha(ROOT/path)==expected,path
    receipt={'protocol_sha256':sha(HERE/'PROTOCOL.frozen.json'),'driver_sha256':sha(Path(__file__)),
             'policy_sha256':sha(HERE/'policy.py'),'frozen_pipeline_sha256':sha(chain/'pipeline.py'),
             'previous_selection_sha256':sha(PREVIOUS_E/'SELECTION.frozen.json'),'resources':previous['resources']}
    target=RUN/'EXECUTION_IDENTITY.json'
    if target.exists():assert read(target)==receipt,'Source changed: start a new run'
    else:save(target,receipt)
    return receipt


def runtime_confirmation():
    plan=read(HERE/'PROTOCOL.frozen.json');directory=OUT/'remote/r02'
    ii=read(directory/'IDENTITY.json');qq=rows(HERE/'confirmation18.query_only.frozen.jsonl')
    assert ii['checkpoint_manifest_sha256']==plan['decomposer']['checkpoint_manifest_sha256']
    assert ii['query_sha256']==plan['confirmation']['query_sha256']==sha(HERE/'confirmation18.query_only.frozen.jsonl')
    assert ii['decode_compat']=='explicit_generation_config_v002'
    assert ii['effective_do_sample'] is False and ii['effective_num_beams']==1
    rr=[];bad=[]
    for i,q in enumerate(qq,1):
        r=read(directory/f'{i:03d}.json');assert r['id']==q['id'] and r['query']==q['query'] and r['index']==i
        try:
            assert not r.get('generation_error') and not r.get('truncated')
            ss=json.loads(r['raw_output']);assert isinstance(ss,list) and all(isinstance(x,str) and x.strip() for x in ss)
        except Exception:ss=[];bad.append(q['id'])
        rr.append({'id':q['id'],'qid':q['qid'],'dataset':q['dataset'],'question':q['query'],'supporting_questions':ss})
    value=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr)
    path=HERE/'confirmation18.runtime.frozen.jsonl'
    if path.exists():assert path.read_text(encoding='utf-8')==value
    else:path.write_text(value,encoding='utf-8')
    save(OUT/'REMOTE_VALIDATION.json',{'n':18,'identity':ii,'runtime_sha256':sha(path),'parse_failure_original_goal_fallback':bad})
    return rr


def run_case(case,conditions,phase,owner,retrieval,client,model,ceiling,old_catalog):
    global ERRORS
    store=WorkerStore(owner);budget=Budget(ceiling);transport=OneShot(store,budget,client,model,case['qid'])
    def retrieve(query):
        with RETRIEVAL_LOCK:
            previous,retrieval.store=retrieval.store,store
            try:return retrieval.retrieve(case['qid'],query,10)
            finally:retrieval.store=previous
    def ask(stage,prompt,maximum,**kwargs):
        global ERRORS
        packet=payload(model,prompt,maximum);key=digest(packet);rid=digest([store.namespace,packet])
        with KEYS_LOCK:mutex=KEYS.setdefault(key,Lock())
        with mutex:
            if key in old_catalog:
                r=old_catalog[key]
                store.event('exact_source_payload_reuse',qid=case['qid'],stage=stage,payload_sha256=key,
                            source=r['source'],request_id=r['request_id'],source_status=r['status'])
                store.put('cache_origin',digest([case['qid'],phase,stage,key]),
                          {'qid':case['qid'],'phase':phase,'stage':stage,'payload_sha256':key,**r})
                return use_receipt(r,stage,kwargs.get('allow_final_truncation',False))
            try:
                with CIRCUIT_LOCK:
                    if ERRORS>=3:raise RuntimeError('Three connection unknowns: dispatch stopped')
                answer=transport.ask(stage,prompt,maximum,**kwargs)
                with CIRCUIT_LOCK:ERRORS=0
                return answer
            except Exception:
                row=store.db.execute('SELECT status,result FROM requests WHERE id=?',(rid,)).fetchone()
                if row:
                    if row[0] in ('unknown','in_flight'):
                        with CIRCUIT_LOCK:ERRORS+=1
                    data={'request_id':rid,'status':row[0],'result':json.loads(row[1]) if row[1] else None,
                          'source':str((store.directory/'checkpoint.sqlite3').relative_to(ROOT))}
                    return use_receipt(data,stage,kwargs.get('allow_final_truncation',False))
                raise
    order=sorted(conditions,key=lambda c:hashlib.sha256((case['id']+c).encode()).hexdigest())
    statuses={}
    try:
        for condition in order:
            target=RUN/'cases'/phase/case['dataset']/condition/(case['qid']+'.json')
            if target.exists():statuses[condition]=read(target)['status'];continue
            completed=[]
            try:output=execute(case,retrieve,ask,condition,record_node=completed.append)
            except Exception as exc:
                output={**case,'condition':condition,'status':'incomplete','nodes':completed,
                        'error_type':type(exc).__name__,'reason':str(exc)[:400]}
                if isinstance(exc,IntermediateLength):
                    output.update(status='algorithm_failure',failed_stage=exc.stage,failed_request_id=exc.request_id,
                                  unpropagated_truncated_answer=exc.answer,failure_policy='Known intermediate length: F1/EM0')
                elif isinstance(exc,RecordedFailure):
                    output.update(failed_stage=exc.stage,failed_request_id=exc.request_id,request_status=exc.request_status,
                                  failure_source=exc.source)
            output.update(phase=phase,execution_order=order,finished_utc=utc())
            locked_json(target,output);statuses[condition]=output['status']
        return {'id':case['id'],'conditions':statuses}
    finally:store.db.close();budget.db.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['A','B','C'],required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    plan=read(HERE/'PROTOCOL.frozen.json');ident=identity(plan)
    cases=runtime_confirmation() if args.phase=='C' else rows(PREVIOUS_E/'runtime60.frozen.jsonl')
    conditions=plan['phase_'+args.phase]['conditions']
    if args.phase=='C':
        selected=read(HERE/'SELECTED.frozen.json');conditions=['legacy',selected['condition']]
        assert sha(HERE/'SELECTED.frozen.json')==read(OUT/'SELECTION_RECEIPT.json')['selected_sha256']
    if args.prepare_only:print(json.dumps({'phase':args.phase,'cases':len(cases),'conditions':conditions,'new_calls':0}));return
    option=get_api_option(2);assert option['model']==plan['model'];install()
    owners={};retrievers={};network=Store(RUN/'network',{'task':'component_opt_v2_transport'})
    try:
        catalogs={ds:catalog(ds) for ds in plan['resources']}
        for ds,resource in plan['resources'].items():
            owner=Store(RUN/'api'/ds,{'task':'component_opt_v2','execution_identity':ident,'dataset':ds})
            owners[ds]=owner;count=seed(owner,ds,cases,resource);retrievers[ds]=Retrieval(owner,resources=resource)
            print(json.dumps({'dataset':ds,'exact_retrieval_packets':count,'old_unique_payloads':len(catalogs[ds])}),flush=True)
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'))
            host=urlsplit(option['base_url']).hostname;stack.enter_context(provider_dns(host,network))
            with socket.create_connection((host,443),timeout=15) as sock:
                with ssl.create_default_context().wrap_socket(sock,server_hostname=host):network.event('TLS_preflight_pass',paid_request=False)
            http=stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0,
                limits=httpx.Limits(max_connections=3,max_keepalive_connections=3,keepalive_expiry=15)),timeout=90))
            api=stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'],base_url=option['base_url'],
                         timeout=90,max_retries=0,http_client=http));client=StreamingClient(api)
            by_ds={ds:[c for c in cases if c['dataset']==ds] for ds in plan['resources']}
            order=[cc[i] for i in range(max(map(len,by_ds.values()))) for cc in by_ds.values() if i<len(cc)]
            completed=[]
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs=[pool.submit(run_case,c,conditions,args.phase,owners[c['dataset']],retrievers[c['dataset']],
                           client,plan['model'],plan['phase_'+args.phase]['global_dispatch_ceiling_cny'],catalogs[c['dataset']]) for c in order]
                for future in as_completed(jobs):
                    r=future.result();completed.append(r)
                    locked_json(RUN/f'PROGRESS_{args.phase}.json',{'utc':utc(),'planned':len(cases),'completed':len(completed),'results':completed})
                    print(json.dumps({'phase':args.phase,'completed':len(completed),**r}),flush=True)
            locked_json(RUN/f'TERMINAL_{args.phase}.json',{'utc':utc(),'planned':len(cases),'attempted':len(completed),'conditions':conditions})
    finally:
        for r in retrievers.values():
            if r.chunks is not None:r.chunks.close()
        for s in owners.values():s.export(full=True);s.db.close()
        network.export(full=True);network.db.close()


if __name__=='__main__':main()
