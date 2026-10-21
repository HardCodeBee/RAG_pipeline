"""Explicitly authorized recovery of missing formal-2Wiki results only.

Frozen algorithms/prompts remain unchanged. Complete predictions (including
algorithm failures) are protected. Every retry has a fresh immutable namespace;
old requests and liabilities are never reset. Scoring is offline and separate.
"""
from collections import Counter
from contextlib import ExitStack
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import time

from experiments.query_decomposition.main_eval_guard_20261021 import watch as guard
from experiments.query_decomposition.main_eval_guard_20261021 import watch_p02
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store, digest, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json

ROOT = guard.base.ROOT
OUTPUT = guard.base.PARENT
OUT = OUTPUT/'twowiki_missing_recovery_20261021'
ORIGINAL = guard.base.V2.parent/'campaign_api_v1'
V2 = guard.base.V2
INPUT = ROOT/'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/2wiki'
P02 = watch_p02.METHOD
P02_OUT = watch_p02.RUN
P02_HERE = watch_p02.HERE
ALLOWED = guard.recovery.ALLOWED | {P02}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def save(name, value):
    locked_json(OUT/name, value)


def event(kind, **fields):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'EVENTS.jsonl').open('a',encoding='utf8') as stream:
        stream.write(json.dumps({'utc':utc(),'event':kind,**fields},ensure_ascii=False)+'\n')


def scorable(value):
    return bool(value and isinstance(value.get('prediction'),str) and
                (value['status'].startswith('success') or value['status']=='algorithm_failure'))


def baseline_sources():
    config = read(ORIGINAL/'2wiki/BASELINE_CONFIGS_FROZEN.json')['selected']
    for method,cfg in config.items():
        track = 'matched50' if method=='b05_iter_retgen_bge' else 'main'
        if track=='matched50':
            cfg={'id':'k10_t5','k':10,'iterations':5}
        yield method,cfg,V2/track/'2wiki/formal'/method/cfg['id']/'checkpoint.sqlite3'


def predictions(path):
    with guard.readonly(path) as db:
        return {qid:json.loads(raw) for qid,raw in db.execute("SELECT key,value FROM data WHERE kind='prediction'")}


def baseline_selected(method,source):
    current = predictions(source)
    for path in sorted(OUT.glob(f'rounds/r*/methods/{method}/*/checkpoint.sqlite3')):
        for qid,value in predictions(path).items():
            if not scorable(current.get(qid)):
                current[qid]=value
    return current


def protected_digest(method, source=None):
    values = (predictions(source) if source else
              {v['qid']:v for v in watch_p02.selected_cases().values() if v['dataset']=='2wiki'})
    values = {qid:v for qid,v in values.items() if scorable(v)}
    return {'count':len(values),'sha256':digest(values)}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    guard.frozen_check()
    p02_plan,errors=watch_p02.verify_plan()
    assert not errors, errors
    path=OUT/'PLAN.frozen.json'
    if path.exists():
        plan=read(path)
        assert plan['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return plan
    questions={v['qid']:v['question'] for v in rows(INPUT/'formal.questions.jsonl')}
    items,protected=[],{}
    for method,config,source in baseline_sources():
        existing=predictions(source)
        protected[method]=protected_digest(method,source)
        for qid,question in questions.items():
            old=existing.get(qid)
            if not scorable(old):
                items.append({'method':method,'case':{'qid':qid,'question':question},
                              'config':config,'source':str(source.relative_to(ROOT)),
                              'old_status':old.get('status') if old else 'unstarted'})
    selected={v['qid']:v for v in watch_p02.selected_cases().values() if v['dataset']=='2wiki'}
    protected[P02]=protected_digest(P02)
    for q in rows(P02_HERE/'questions1500.frozen.jsonl'):
        if q['dataset']=='2wiki' and not scorable(selected.get(q['qid'])):
            old=selected.get(q['qid'],{})
            items.append({'method':P02,'case':{'qid':q['qid'],'question':q['question']},
                          'old_status':old.get('status','unstarted')})
    assert set(protected)==ALLOWED
    plan={'utc':utc(),'authorization':'User: prioritize attempting recovery to500 scorable for every formal2Wiki method, then update evaluation.',
          'dataset':'2wiki','formal_per_method':500,'methods':sorted(ALLOWED),
          'primary_iter':'matched50 k10/T5','items':items,'protected':protected,
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'baseline_config_sha256':hashlib.sha256((V2/'CONFIG_FROZEN.json').read_bytes()).hexdigest(),
          'p02_protocol_sha256':hashlib.sha256((P02_HERE/'PROTOCOL.frozen.json').read_bytes()).hexdigest(),
          'policy':'First scorable recovery for originally missing only; no score-based selection. Same failed payload may be explicitly retried in new namespaces; old unknown reserves retained.',
          'limits':{'global_cap_cny':300,'dispatch_ceiling_cny':299.9,'p02_existing_subcap_cny':20,
                    'passes':2,'content_rejection_and_http400_attempts':1},
          'new_api_calls_in_preparation':0}
    save('PLAN.frozen.json',plan)
    return plan


class SuccessCatalog:
    """Same-method/same-qid/same-stage successes; never import cached failure."""
    def __init__(self, paths):
        from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.catalog import Catalog
        self.path=paths[0]
        self.catalogs=[Catalog(path) for path in paths if path.exists()]
        self.retrievals,self.hybrid={},{}
        for catalog in self.catalogs:
            for key,value in catalog.retrievals.items():self.retrievals.setdefault(key,value)
            for key,value in catalog.hybrid.items():self.hybrid.setdefault(key,value)

    def request(self,qid,stage,payload):
        for catalog in self.catalogs:
            found=catalog.request(qid,stage,payload)
            if found and found['status']=='success':return found
        return None


def retryable(value):
    return value and (value['status'] in {'unknown','in_flight'} or value.get('retryable_infrastructure'))


def run_baseline(item,number,client,budget_class,handles):
    from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import run as runtime
    method,case=item['method'],item['case'];source=ROOT/item['source']
    if scorable(baseline_selected(method,source).get(case['qid'])):
        return {'status':'already_scorable','qid':case['qid']}
    directory=OUT/f'rounds/r{number:03d}/methods/{method}/{case["qid"]}'
    if (directory/'checkpoint.sqlite3').exists():
        return predictions(directory/'checkpoint.sqlite3').get(case['qid'],{'status':'unknown'})
    owner=Store(directory,{'task':OUT.name,'dataset':'2wiki','method':method,'config':item['config'],
                          'round':number,'qid':case['qid'],'original_source':item['source']})
    budget=budget_class(OUTPUT/'budget.sqlite3',300,2/3)
    old_paths=[source]+sorted(OUT.glob(f'rounds/r*/methods/{method}/{case["qid"]}/checkpoint.sqlite3'))
    old_paths=[p for p in old_paths if p!=directory/'checkpoint.sqlite3']
    catalog=SuccessCatalog(old_paths)
    resource=read(ORIGINAL/'2wiki/RESOURCE_FROZEN.json')['resources']
    retrieval=runtime.Retrieval(owner,resource,catalog)
    if handles:
        retrieval.index,retrieval.encoder,retrieval.chunks,retrieval.sparse=handles[0]
    transport=runtime.Transport(owner,client,budget,'qwen-plus-2025-12-01',catalog)
    module=importlib.import_module('experiments.query_decomposition.baseline_integrity_repair_v2_20261017.methods.'+
                                  runtime.METHODS[method]+'.method')
    try:
        try:
            module.run([case],retrieval,transport,owner,item['config'])
        except RuntimeError as error:
            if 'Transport stopped;' not in str(error):raise
            row=owner.db.execute('SELECT id,status FROM requests ORDER BY rowid DESC LIMIT 1').fetchone()
            if owner.get('prediction',case['qid']) is None:
                owner.put('prediction',case['qid'],{**case,'prediction':None,
                    'status':row[1] if row else 'unknown','request_id':row[0] if row else None,
                    'recovery_error':type(error).__name__})
        value=owner.get('prediction',case['qid'])
        assert value is not None
        owner.export(full=True)
        return value
    finally:
        if retrieval.index is not None:
            handles[:]=[(retrieval.index,retrieval.encoder,retrieval.chunks,retrieval.sparse)]
        owner.db.close();budget.db.close()


def run_p02(item,number,client,p02_plan,adapters,catalogs):
    from experiments.query_decomposition.p02_formal1500_v1_20261021 import run as p02
    from experiments.query_decomposition.p02_merge_cachefix_v1_20261021 import run as transport
    qid=item['case']['qid']
    latest={v['qid']:v for v in watch_p02.selected_cases().values() if v['dataset']=='2wiki'}
    if scorable(latest.get(qid)):
        return {'status':'already_scorable','qid':qid}
    # Separate authorized operational rounds. Original r000-r003 remain intact.
    round_root=P02_OUT/f'rounds/r{100+number:03d}'
    result_path=round_root/f'cases/2wiki/{P02}/{qid}.json'
    if result_path.exists():return read(result_path)
    case=read(P02_OUT/f'runtime/2wiki/{qid}.json')
    p02.bind_transport()
    if not adapters:
        owner=Store(OUT/'p02_retrieval/2wiki',{'task':OUT.name,'dataset':'2wiki',
                    'resources_sha256':digest(p02_plan['resources']['2wiki'])})
        seeds=[P02_OUT/'retrieval/2wiki/checkpoint.sqlite3']
        adapters.extend([owner,p02.Adapter(owner,p02_plan['resources']['2wiki'],seed_databases=seeds)])
    owner=Store(round_root/f'methods/2wiki/{P02}',{'task':P02_HERE.name,
        'protocol_sha256':hashlib.sha256((P02_HERE/'PROTOCOL.frozen.json').read_bytes()).hexdigest(),
        'round':100+number,'dataset':'2wiki'})
    try:
        catalog=p02.load_catalog() if not catalogs else catalogs[0]
        # A failed payload is retained in its database, never imported as a
        # successful prefix that would prevent an authorized recovery attempt.
        catalog={k:v for k,v in catalog.items() if v['status']=='success'}
        transport.run_case(case,P02,owner,adapters[1],client,catalog,p02_plan,round_root)
        catalogs[:]=[catalog]
        return read(result_path)
    finally:
        owner.export();owner.db.close()


def score(plan):
    # Offline-only. Never call this from inside a method or feed scores to one.
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    refs={v['qid']:v for v in rows(INPUT/'formal.references.local.jsonl')}
    scorer=metric('2wiki');report={};details=[]
    source_map={m:(cfg,p) for m,cfg,p in baseline_sources()}
    p02={v['qid']:v for v in watch_p02.selected_cases().values() if v['dataset']=='2wiki'}
    for method in sorted(ALLOWED):
        if method==P02:values=p02
        else:values=baseline_selected(method,source_map[method][1])
        vv=[]
        for qid,ref in refs.items():
            v=values.get(qid,{'status':'unstarted'})
            s=scorer(v.get('prediction') or '',ref) if scorable(v) else {'f1':None,'em':None}
            vv.append({'method':method,'qid':qid,'status':v['status'],**s})
        valid=[v for v in vv if v['f1'] is not None];n=len(valid);total=sum(v['f1'] for v in valid)
        report[method]={'planned':500,'processed':len(values),'scorable':n,'missing':500-n,
             'F1':total/n if n else None,'EM':sum(v['em'] for v in valid)/n if n else None,
             'F1_all500_bounds':[total/500,(total+500-n)/500],
             'statuses':dict(Counter(v['status'] for v in vv))}
        details.extend(vv)
    preserved=[]
    for method,protected in plan['protected'].items():
        if method==P02:
            # Protected case objects keep exact original rows; verify those
            # qids individually, since newly recovered cases increase the total.
            old={v['qid']:v for path in sorted(P02_OUT.glob('rounds/r00*/cases/2wiki/*/*.json'))
                 for v in [read(path)] if scorable(v)}
            observed={'count':len(old),'sha256':digest(old)}
        else:observed=protected_digest(method,source_map[method][1])
        assert observed==protected, 'Previously scorable results changed: '+method
        preserved.append(method)
    with guard.readonly(OUTPUT/'budget.sqlite3') as db:
        known,reserved=db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
    result={'utc':utc(),'dataset':'2wiki','formal_per_method':500,'runs':report,
            'previously_scorable_results_preserved':preserved,'budget':{'known_cny':known,'reserved_cny':reserved,
            'occupied_cny':known+reserved,'cap_cny':300},'score_only_new_api_calls':0,
            'selection':'Original scorable results retained; first scorable recovery for originally missing only.'}
    save('RESULTS.json',result)
    (OUT/'SCORES.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in details),encoding='utf8')
    lines=['2Wiki formal500: authorized missing-only recovery; native Answer F1/EM','']
    for method,v in report.items():
        f='--' if v['F1'] is None else f"{v['F1']*100:.2f} / {v['EM']*100:.2f}"
        lines.append(f"{method}: n={v['scorable']}/500; F1/EM={f}; missing={v['missing']}")
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return result


def run(plan):
    import httpx
    from openai import OpenAI
    import os,msvcrt
    from scripts.cli_support import temporary_api_key
    from src.generators.api_options import get_api_option
    from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
    common,runtime,matched,Budget=guard.recovery.install()
    install()
    p02_plan,errors=watch_p02.verify_plan();assert not errors,errors
    handles,adapters,catalogs=[],[],[]
    with (OUT/'worker.lock').open('a+b') as lock,ExitStack() as stack:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        network=Store(OUT/'network',{'task':OUT.name,'scope':'2wiki missing only'})
        stack.callback(network.db.close)
        stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'))
        option=get_api_option(2)
        assert option['model']==p02_plan['model'] and option['base_url']==p02_plan['base_url']
        stack.enter_context(guard.recovery.network.provider_dns(guard.recovery.network.HOST,network))
        guard.recovery.network.preflight()
        http=stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0),timeout=90))
        api=stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'],base_url=option['base_url'],
                                     timeout=90,max_retries=0,http_client=http))
        client=StreamingClient(api)
        try:
            remaining=plan['items']
            for number in range(plan['limits']['passes']):
                again=[];catalogs.clear()
                for index,item in enumerate(remaining):
                    if (OUT/'STOP_REQUESTED').exists():return
                    guard.frozen_check()
                    save('STATUS.json',{'utc':utc(),'status':'running','round':number,'method':item['method'],
                         'qid':item['case']['qid'],'index':index+1,'round_items':len(remaining)})
                    event('case_recovery_started',round=number,method=item['method'],qid=item['case']['qid'],old_status=item['old_status'])
                    result=(run_p02(item,number,client,p02_plan,adapters,catalogs) if item['method']==P02 else
                            run_baseline(item,number,client,Budget,handles))
                    event('case_recovery_finished',round=number,method=item['method'],qid=item['case']['qid'],status=result['status'])
                    print(json.dumps({'round':number,'method':item['method'],'qid':item['case']['qid'],'status':result['status']}),flush=True)
                    if retryable(result):again.append(item)
                score(plan)
                remaining=again
                if not remaining:break
            result=score(plan)
            save('STATUS.json',{'utc':utc(),'status':'completed_attempts','remaining_missing':{
                m:v['missing'] for m,v in result['runs'].items() if v['missing']},
                'all_methods_scorable500':all(v['scorable']==500 for v in result['runs'].values())})
        finally:
            if adapters:
                adapters[1].close();adapters[0].db.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--score',action='store_true')
    args=parser.parse_args();plan=prepare()
    if args.score:
        value=score(plan);print(json.dumps({m:v['scorable'] for m,v in value['runs'].items()}));return
    if args.run:
        try:run(plan)
        except BaseException as error:
            save('STATUS.json',{'utc':utc(),'status':'stopped_for_audit','error':type(error).__name__,'reason':str(error)[:1200]})
            raise
    else:print(json.dumps({'missing_method_questions':len(plan['items']),'by_method':dict(Counter(v['method'] for v in plan['items'])),'new_api_calls':0}))


if __name__=='__main__':main()
