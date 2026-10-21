"""Independent prompt interventions; own stores, immutable recovery rounds."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
import gc, json, os, socket, sqlite3, ssl, time
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
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload,use_receipt,RecordedFailure
from experiments.query_decomposition.component_chain_opt_v2_20261019.policy import IntermediateLength
from experiments.query_decomposition.complete_pipeline_comparison_v1_20261020 import result_sources300 as parent
from .prepare import ROOT,OUTPUT,HERE,OUT,prepare,rows,read,sha,save,log
from .pipeline import ARMS,execute
from .receipts import ro,receipt,sources,baseline_case,baseline_calls,baseline_packets
from .report import report

RETRIEVAL_LOCK=Lock(); CATALOG_LOCK=Lock(); KEY_LOCK=Lock(); KEYS={}; CIRCUIT=Lock(); ERRORS=0


class Budget(ParallelBudget):
    def __init__(self): super().__init__(OUTPUT/'budget.sqlite3',300)
    def reserve(self,rid,namespace,amount):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total=self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
            local=self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?',
                                  (str(OUT.relative_to(OUTPUT))+'%',)).fetchone()[0]
            if total+amount>299.9 or local+amount>15: raise RuntimeError('Frozen cumulative300/experiment15 ceiling; no submission')
            if self.db.execute('SELECT 1 FROM charges WHERE id=?',(rid,)).fetchone(): raise RuntimeError('Existing liability cannot be resubmitted')
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')",(rid,namespace,amount)); self.db.commit()
        except BaseException: self.db.rollback(); raise


def identity(plan):
    for path,h in plan['source_pins'].items(): assert sha(ROOT/path)==h,path
    assert sha(HERE/'runtime300.frozen.jsonl')==plan['runtime_sha256']
    assert sha(HERE/'B0_SOURCES.frozen.jsonl')==plan['baseline_sources_sha256']
    value={'protocol_sha256':sha(HERE/'PROTOCOL.frozen.json'),
           'own_sources':{p.name:sha(p) for p in HERE.glob('*.py')}}
    save(HERE/'EXECUTION_IDENTITY.frozen.json',value)
    assert read(OUT/'OFFLINE_VALIDATION.frozen.json')['passed']
    return value


def catalog():
    result={}
    paths=[]
    for root in parent.attempt_roots(): paths.extend(sorted((root/'methods').glob('*/*/checkpoint.sqlite3')))
    paths.extend(sorted((OUT/'rounds').glob('r[0-9][0-9][0-9]/methods/*/*/checkpoint.sqlite3')))
    for p in paths:
        with ro(p) as db:
            for rid,raw in db.execute("SELECT id,payload FROM requests WHERE status='success'"):
                result.setdefault(digest(json.loads(raw)),{'request_id':rid,'source':str(p.relative_to(ROOT)),'status':'success'})
    return result


def all_paths(case,arm):
    return [p/'cases'/case['dataset']/arm/(case['qid']+'.summary.json') for p in sorted((OUT/'rounds').glob('r[0-9][0-9][0-9]'))]


def eligible(case,arm):
    last=None
    for p in all_paths(case,arm):
        if p.exists():
            last=read(p)
            if last['status'].startswith('success') or last['status']=='algorithm_failure': return False
    if last is None: return True
    return last.get('retryable_infrastructure',False)


def run_case(case,arm,round_root,owner,retr_owner,retriever,client,model,cached,source):
    target=round_root/'cases'/case['dataset']/arm/(case['qid']+'.json')
    assert not target.exists(), 'Never replay an existing attempt namespace'
    store=WorkerStore(owner); retrstore=WorkerStore(retr_owner); budget=Budget()
    transport=OneShot(store,budget,client,model,case['qid'])
    baseline=baseline_calls(source); completed=[]; started=time.perf_counter()
    def retrieve(query):
        with RETRIEVAL_LOCK:
            previous,retriever.store=retriever.store,retrstore
            key=digest([case['qid'],query,10,retriever.resources]); hit=retrstore.get('retrieval',key) is not None
            try: packet=retriever.retrieve(case['qid'],query,10)
            finally: retriever.store=previous
            store.event('logical_retrieval',qid=case['qid'],query=query,cache_hit=hit,doc_ids=[d['doc_id'] for d in packet['docs']])
            return packet
    def ask(stage,prompt,maximum,allow_final_truncation=False):
        global ERRORS
        request=payload(model,prompt,maximum); key=digest(request); rid=digest([store.namespace,request])
        with KEY_LOCK: mutex=KEYS.setdefault(key,Lock())
        with mutex:
            # Baseline case's own receipt always wins over cross-arm caches.
            with CATALOG_LOCK: old=baseline.get(key) or cached.get(key)
            r=receipt(old,request) if old else None; reused=r is not None
            if r is None:
                row=store.db.execute('SELECT status,result FROM requests WHERE id=?',(rid,)).fetchone()
                if row:
                    r={'request_id':rid,'status':row[0],'result':json.loads(row[1]) if row[1] else None,
                       'source':str((store.directory/'checkpoint.sqlite3').relative_to(ROOT))}; reused=True
            if r is None:
                with CIRCUIT:
                    if ERRORS>=3: raise RuntimeError('Three consecutive connection unknowns; dispatch deferred')
                try: transport.ask(stage,prompt,maximum,allow_final_truncation=True)
                except Exception:
                    if not store.db.execute('SELECT 1 FROM requests WHERE id=?',(rid,)).fetchone(): raise
                row=store.db.execute('SELECT status,result FROM requests WHERE id=?',(rid,)).fetchone()
                r={'request_id':rid,'status':row[0],'result':json.loads(row[1]) if row[1] else None,
                   'source':str((store.directory/'checkpoint.sqlite3').relative_to(ROOT))}
                with CIRCUIT: ERRORS=ERRORS+1 if r['status'] in ['unknown','in_flight'] else 0
                if r['status']=='success':
                    with CATALOG_LOCK: cached.setdefault(key,{k:v for k,v in r.items() if k!='result'})
            value=r['result'] or {}
            store.put('logical_call',digest([case['qid'],stage,key]),
                      {'qid':case['qid'],'stage':stage,'payload_sha256':key,'request_id':r['request_id'],'source':r['source'],
                       'status':r['status'],'reused':reused,'usage':value.get('usage',{}),'finish_reason':value.get('finish_reason'),
                       'latency_seconds':value.get('latency_seconds'),'http_status':value.get('http_status')})
            return use_receipt(r,stage,allow_final_truncation)
    try:
        result=execute(case,retrieve,ask,arm=arm,record_node=completed.append)
    except Exception as exc:
        result={**case,'arm':arm,'status':'incomplete','prediction':None,'nodes':completed,
                'error_type':type(exc).__name__,'reason':str(exc)[:400],'retryable_infrastructure':False}
        if isinstance(exc,IntermediateLength):
            result.update(status='algorithm_failure',prediction='',failed_stage=exc.stage,failed_request_id=exc.request_id,
                          unpropagated_truncated_answer=exc.answer)
        elif isinstance(exc,RecordedFailure):
            rr=receipt({'source':exc.source,'request_id':exc.request_id})['result'] or {}
            result.update(failed_stage=exc.stage,failed_request_id=exc.request_id,request_status=exc.request_status,
                          http_status=rr.get('http_status'),retryable_infrastructure=exc.request_status in ['unknown','in_flight'] or rr.get('http_status') in [408,409,425,429,500,502,503,504])
        elif 'dispatch deferred' in str(exc): result['retryable_infrastructure']=True
    finally:
        calls=[json.loads(v) for v, in store.db.execute("SELECT value FROM data WHERE kind='logical_call' AND json_extract(value,'$.qid')=?",(case['qid'],))]
        store.db.close(); budget.db.close(); retrstore.db.close()
    result.update(finished_utc=utc(),actual_seconds=time.perf_counter()-started,baseline_source=source['source'],logical_calls=calls)
    locked_json(target,result)
    summary={k:result.get(k) for k in ['id','qid','dataset','arm','status','prediction','retryable_infrastructure','reason','request_status','http_status']}
    summary.update(source=str(target.relative_to(ROOT)),logical_calls=len(calls),new_calls=sum(not c['reused'] for c in calls),
                   tokens={k:sum(c['usage'].get(k,0) for c in calls) for k in ['prompt_tokens','completion_tokens']})
    locked_json(target.with_name(case['qid']+'.summary.json'),summary)
    return summary


def seed_retrieval(owner,ds,resource,cases,src):
    count=0
    for c in cases:
        for query,p in baseline_packets(baseline_case(src[c['id']])).items():
            key=digest([c['qid'],query,10,resource])
            if owner.get('retrieval',key) is None: owner.put('retrieval',key,p); count+=1
    wanted={c['qid'] for c in cases}
    for root in parent.attempt_roots():
        path=root/'retrieval'/ds/'checkpoint.sqlite3'
        if not path.exists(): continue
        with ro(path) as db:
            for key,raw in db.execute("SELECT key,value FROM data WHERE kind='retrieval'"):
                p=json.loads(raw)
                if p['qid'] not in wanted or key!=digest([p['qid'],p['query'],10,resource]): continue
                if owner.get('retrieval',key) is None: owner.put('retrieval',key,p); count+=1
    return count


def main():
    global ERRORS
    plan=prepare(); ident=identity(plan); cases=rows(HERE/'runtime300.frozen.jsonl'); src=sources()
    option=get_api_option(2); assert option['model']==plan['model']; install()
    if report()['all_complete']: return
    for round_id in range(1,plan['execution']['max_rounds']+1):
        root=OUT/'rounds'/f'r{round_id:03d}'
        if root.exists():
            if (root/'TERMINAL.json').exists(): continue
            raise RuntimeError('Existing unterminated round requires inspection; no silent resume')
        pending=[(c,a) for a in ['C_final_readout_P1','A_child_prompt_C08','B_rewrite_prompt_c02'] for c in cases if eligible(c,a)]
        if not pending: break
        ERRORS=0; cached=catalog(); network=Store(root/'network',{'task':HERE.name,'round':round_id,'identity':ident})
        log('start_independent_round',round=round_id,pending=len(pending),success_payloads=len(cached))
        print(json.dumps({'round':round_id,'pending':len(pending),'cached_payloads':len(cached)}),flush=True)
        try:
            with ExitStack() as stack:
                stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'))
                host=urlsplit(option['base_url']).hostname; stack.enter_context(provider_dns(host,network))
                with socket.create_connection((host,443),timeout=15) as sock:
                    with ssl.create_default_context().wrap_socket(sock,server_hostname=host): network.event('TLS_preflight',paid_request=False)
                http=stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0,
                      limits=httpx.Limits(max_connections=3,max_keepalive_connections=3,keepalive_expiry=15)),timeout=90))
                api=stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'],base_url=option['base_url'],timeout=90,max_retries=0,http_client=http))
                client=StreamingClient(api)
                done=0
                # Run final-only first: it needs no new searches or upstream calls.
                for arm in ['C_final_readout_P1','A_child_prompt_C08','B_rewrite_prompt_c02']:
                    for ds in ['2wiki','musique','hotpotqa']:
                        cc=[c for c,a in pending if a==arm and c['dataset']==ds]
                        if not cc: continue
                        own=Store(root/'methods'/ds/arm,{'task':HERE.name,'arm':arm,'dataset':ds,'identity':ident})
                        retr_owner=Store(OUT/'retrieval'/ds,{'task':HERE.name,'dataset':ds,'resources':plan['resources'][ds]})
                        seeded=seed_retrieval(retr_owner,ds,plan['resources'][ds],[c for c in cases if c['dataset']==ds],src)
                        retriever=Retrieval(retr_owner,resources=plan['resources'][ds])
                        print(json.dumps({'round':round_id,'arm':arm,'dataset':ds,'planned':len(cc),'seeded_packets':seeded}),flush=True)
                        try:
                            with ThreadPoolExecutor(max_workers=plan['execution']['max_workers']) as pool:
                                jobs=[pool.submit(run_case,c,arm,root,own,retr_owner,retriever,client,plan['model'],cached,src[c['id']]) for c in cc]
                                for f in as_completed(jobs):
                                    r=f.result(); done+=1
                                    locked_json(OUT/'PROGRESS.json',{'utc':utc(),'round':round_id,'completed_this_round':done,'planned_this_round':len(pending),'latest':r})
                                    print(json.dumps({'round':round_id,'completed':done,**r},ensure_ascii=False),flush=True)
                                    if done%10==0: report()
                        finally:
                            if retriever.chunks is not None: retriever.chunks.close()
                            own.export(full=True); own.db.close(); retr_owner.export(); retr_owner.db.close()
                            del retriever; gc.collect()
                        report()
            locked_json(root/'TERMINAL.json',{'utc':utc(),'attempted':done,'planned':len(pending)})
        finally:
            network.export(full=True); network.db.close()
        current=report()
        if current['all_complete']: break
        if round_id<plan['execution']['max_rounds']: time.sleep(min(15*round_id,45))
    current=report(); identity(plan)
    locked_json(OUT/'TERMINAL.json',{'utc':utc(),'complete':current['all_complete'],'results':current})
    log('finish_six_independent_changes',complete=current['all_complete'],budget=current['budget'])
    print(json.dumps({'terminal':True,'all_complete':current['all_complete'],'budget':current['budget']}),flush=True)


if __name__=='__main__': main()
