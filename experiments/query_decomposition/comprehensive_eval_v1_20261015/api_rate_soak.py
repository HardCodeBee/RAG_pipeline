"""One rolling minute at the proposed rate, isolated from formal answers."""
from contextlib import ExitStack
from types import SimpleNamespace
import json
import sqlite3
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
    with sqlite3.connect((TARGET/'hotpotqa/formal/b00_rag_top10_v1/fixed/checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        for raw, in db.execute("SELECT value FROM data WHERE kind='prediction'"):
            r=json.loads(raw)
            if r['status']!='success':continue
            p=json.loads(db.execute('SELECT payload FROM requests WHERE id=?',(r['request_id'],)).fetchone()[0])
            cases.append({'qid':r['qid'],'prompt':p['messages'][0]['content']})
            if len(cases)==96:break
    store=Store(RUN/'api_rate_soak_v1',{'purpose':'bounded rate stability only','planned':96,
        'slots':8,'interval':2/3,'generator':frozen['generator'],'base_url':frozen['base_url']})
    if store.db.execute('SELECT count(*) FROM requests').fetchone()[0]:
        raise ValueError('Do not time a reused diagnostic as fresh')
    budget=ParallelBudget(OUTPUT/'budget.sqlite3',200)
    try:
        with ExitStack() as stack:
            network=stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=2,
                limits=httpx.Limits(max_connections=8,max_keepalive_connections=8,keepalive_expiry=10)),timeout=60))
            client=stack.enter_context(OpenAI(api_key=secret('api_code_qwen.txt'),base_url=frozen['base_url'],
                timeout=60,max_retries=0,http_client=network))
            api=CapacityAPI(ExactTransport(store,StreamingClient(client),budget,frozen['generator']),8,2/3)
            operations=[SimpleNamespace(index=i,args=('rate_soak',r['prompt'],64),kwargs={}) for i,r in enumerate(cases)]
            started=time.perf_counter();error=None;values=[]
            with provider_dns(urlsplit(frozen['base_url']).hostname,store):
                try:values=api.ask_group(operations,cases)
                except Exception as exc:error=type(exc).__name__
                finally:api.close()
            seconds=time.perf_counter()-started
        attempts=[(t,json.loads(raw)) for t,raw in store.db.execute('SELECT started,result FROM attempts WHERE result IS NOT NULL')]
        success=sum(r['status']=='success' and r.get('returned_model')==frozen['generator'] for _,r in values)
        throttled=sum(r.get('http_status')==429 for t,r in attempts)
        unknown=sum(r['status']=='unknown' for t,r in attempts)
        from datetime import datetime
        times=sorted(datetime.fromisoformat(t).timestamp() for t,r in attempts)
        peak=max(sum(a<=b<a+60 for b in times) for a in times) if times else 0
        result={'utc':utc(),'passed':success==96 and throttled==0 and error is None and unknown<=2,
            'planned':96,'same_model_success':success,'seconds':seconds,'completed_per_minute':success*60/seconds,
            'http429_attempts':throttled,'network_unknown_attempts':unknown,'exception_type':error,
            'highest_observed_starts_in_rolling60s':peak,'api_slots':8,'start_interval':2/3,
            'source_sha256':sha(__file__),'execution_source_sha256':sha(__import__(__package__+'.scan_api_epoch',fromlist=['__file__']).__file__),
            'formal_answers_replaced':False,'scoring_labels_used':False,'budget_cap_cny':200}
        save(RUN/'API_RATE_SOAK.json',result);store.export(full=True)
        log('api_proposed_rate_stability_checked',**result);print(json.dumps(result),flush=True)
    finally:budget.db.close();store.db.close()


if __name__=='__main__':main()
