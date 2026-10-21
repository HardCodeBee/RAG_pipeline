"""Verify idle connection reuse across real retrieval-length pauses."""
from contextlib import ExitStack
import json
import sqlite3
import time
from urllib.parse import urlsplit
import httpx
from openai import OpenAI
from .common import RUN, OUTPUT, read, save, log, sha, utc
from .campaign import TARGET
from .execution import ExactTransport
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
            if len(cases)==3:break
    store=Store(RUN/'connection_reuse_probe_v1',{'purpose':'idle connection reuse diagnostic','expiry':60,'qids':[c['qid'] for c in cases]})
    budget=ParallelBudget(OUTPUT/'budget.sqlite3',200);connects=[]
    def trace(name, info):
        # Do not retain headers, bodies or credential information.
        if name.endswith('connect_tcp.started'):connects.append(time.monotonic())
    class Traced(httpx.HTTPTransport):
        def handle_request(self, request):
            request.extensions['trace']=trace
            return super().handle_request(request)
    try:
        with ExitStack() as stack:
            network=stack.enter_context(httpx.Client(transport=Traced(retries=2,
                limits=httpx.Limits(max_connections=8,max_keepalive_connections=8,keepalive_expiry=60)),timeout=60))
            client=stack.enter_context(OpenAI(api_key=secret('api_code_qwen.txt'),base_url=frozen['base_url'],
                timeout=60,max_retries=0,http_client=network))
            transport=ExactTransport(store,StreamingClient(client),budget,frozen['generator']);result=[]
            with provider_dns(urlsplit(frozen['base_url']).hostname,store):
                for gap,case in zip([0,15,35],cases):
                    if gap:time.sleep(gap)
                    count=len(connects);transport.qid=case['qid']
                    _,raw=transport.ask('idle_connection_probe',case['prompt'],64)
                    result.append({'idle_seconds':gap,'status':raw['status'],'returned_model':raw.get('returned_model'),
                        'new_tcp_connections':len(connects)-count,'latency_seconds':raw.get('latency_seconds')})
        output={'utc':utc(),'passed':all(r['status']=='success' and r['returned_model']==frozen['generator'] for r in result)
                and all(r['new_tcp_connections']==0 for r in result[1:]),'requests':result,'selected_expiry':60,
                'request_body_or_tls_verification_changed':False,'trace_retains_only_tcp_connect_counts':True,
                'source_sha256':sha(__file__),'formal_answers_replaced':False}
        save(RUN/'CONNECTION_REUSE_PROBE.json',output);log('idle_provider_connection_reuse_measured',**output)
        store.export(full=True);print(json.dumps(output),flush=True)
    finally:budget.db.close();store.db.close()


if __name__=='__main__':main()
