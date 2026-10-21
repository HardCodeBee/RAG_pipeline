"""Ramp real long-output requests; retain formal model/payload and global cap."""
from contextlib import ExitStack
from types import SimpleNamespace
import json
import sqlite3
import statistics
import time
from urllib.parse import urlsplit
import httpx
from openai import OpenAI
from .common import RUN, OUTPUT, read, save, sha, log, utc
from .campaign import TARGET
from .execution import ExactTransport
from .scan_api_epoch import CapacityAPI
from .credential_latency_probe import secret
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns


def main():
    frozen=read(TARGET/'CAMPAIGN_FROZEN.json');cases=[]
    path=TARGET/'hotpotqa/formal/b01_rag_top10_cot3/fixed/checkpoint.sqlite3'
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        for raw, in db.execute("SELECT value FROM data WHERE kind='prediction'"):
            r=json.loads(raw)
            if r['status']!='success':continue
            payload=json.loads(db.execute('SELECT payload FROM requests WHERE id=?',(r['request_id'],)).fetchone()[0])
            assert payload['model']==frozen['generator'] and payload['max_tokens']==512
            cases.append({'qid':r['qid'],'prompt':payload['messages'][0]['content']})
            if len(cases)==12:break
    results=[]
    output=RUN/'API_CAPACITY_PROBE.json'
    for slots,interval in [(3,1.),(6,1.),(8,2/3)]:
        label=f'slots{slots}_interval{interval:.6f}'
        store=Store(RUN/'api_capacity_probe_v1'/label,{'purpose':'execution capacity diagnostics only',
            'slots':slots,'start_interval':interval,'model':frozen['generator'],'base_url':frozen['base_url'],
            'qids':[r['qid'] for r in cases]})
        if store.db.execute('SELECT count(*) FROM requests').fetchone()[0]:
            raise ValueError('A partially used capacity diagnostic must not be timed as fresh')
        budget=ParallelBudget(OUTPUT/'budget.sqlite3',200)
        try:
            with ExitStack() as stack:
                network=stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=2,
                    limits=httpx.Limits(max_connections=slots,max_keepalive_connections=slots,keepalive_expiry=10)),timeout=60))
                client=stack.enter_context(OpenAI(api_key=secret('api_code_qwen.txt'),base_url=frozen['base_url'],
                    max_retries=0,timeout=60,http_client=network))
                transport=ExactTransport(store,StreamingClient(client),budget,frozen['generator'])
                api=CapacityAPI(transport,slots,interval)
                operations=[SimpleNamespace(index=i,args=('capacity_probe',c['prompt'],512),kwargs={}) for i,c in enumerate(cases)]
                started=time.perf_counter();exception=None;values=[]
                with provider_dns(urlsplit(frozen['base_url']).hostname,store):
                    try:values=api.ask_group(operations,cases)
                    except Exception as exc:exception=type(exc).__name__
                    finally:api.close()
                seconds=time.perf_counter()-started
            attempts=[json.loads(r[0]) for r in store.db.execute('SELECT result FROM attempts WHERE result IS NOT NULL')]
            success=[r for _,r in values if r['status']=='success' and r.get('returned_model')==frozen['generator']]
            rejected=sum(r.get('http_status')==429 for r in attempts)
            unknown=sum(r['status']=='unknown' for r in attempts)
            r={'api_slots':slots,'start_interval':interval,'planned_calls':12,'successful_same_model':len(success),
               'wall_seconds':seconds,'questions_per_minute':len(success)*60/seconds,
               'http429_attempts':rejected,'network_unknown_attempts':unknown,'exception_type':exception,
               'median_request_seconds':statistics.median([r['latency_seconds'] for r in success]) if success else None,
               'accepted':len(success)==12 and rejected==0 and unknown==0}
            results.append(r);store.export(full=True)
            save(RUN/'API_CAPACITY_PROBE_PROGRESS.json',{'stage':'measured','candidates':results})
            print(json.dumps(r),flush=True)
            if rejected or exception:break
        finally:
            budget.db.close();store.db.close()
    control=results[0]
    valid=[r for r in results if r['accepted']]
    chosen=min(valid,key=lambda r:r['wall_seconds']) if valid else control
    if chosen['wall_seconds']>control['wall_seconds']*.9:chosen=control
    result={'utc':utc(),'passed':bool(valid),'source_sha256':sha(__import__(__package__+'.scan_api_epoch',fromlist=['__file__']).__file__),
        'validation_source_sha256':sha(__file__),'candidates':results,
        'selected':{k:chosen[k] for k in ['api_slots','start_interval']},
        'observed_wall_speed_ratio':control['wall_seconds']/chosen['wall_seconds'],
        'formal_answers_replaced':False,'f1_used_for_selection':False,'samples_are_latency_diagnostics_only':True,
        'generator':frozen['generator'],'base_url':frozen['base_url'],'budget_cap_cny':200,
        'official_rate_limit_reference':'https://help.aliyun.com/zh/model-studio/rate-limit',
        'request_start_ceiling_rpm':90,'independent_second_account_quota_assumed':False}
    save(output,result);log('api_capacity_ramp_completed',**result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
