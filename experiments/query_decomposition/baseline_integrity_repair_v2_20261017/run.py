"""Dev integrity gate, then resume isolated fixed-configuration formal runs."""
import argparse
from contextlib import ExitStack
import gc
import importlib
import json
import os
from urllib.parse import urlsplit
import httpx
from openai import OpenAI
from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install as install_offsets
from .common import *
from .catalog import Catalog
from .transport import Transport, API, Budget
from .retrieval import Retrieval
from .executor import execute
from .report import score_store


class Runner:
    def __init__(self, receipt, client, budget):
        self.receipt, self.client, self.budget = receipt, client, budget
        self.handles = {}

    def run_method(self, track, dataset, split, mid, config, size):
        verify()
        if (OUT/'STOP_REQUESTED').exists():
            raise IntegrityError('User requested stop')
        cases=rows(INPUTS/dataset/(split+'.questions.jsonl'))[:size]
        if any(set(c)!={'qid','question'} for c in cases):
            raise IntegrityError('Gold-free execution input contract failed')
        source=PARENT/dataset/split/mid/self.receipt['native_selected_configs_preserved'][dataset][mid]['id']/'checkpoint.sqlite3'
        if track=='matched25':
            name='iter' if mid=='b05_iter_retgen_bge' else 'react'
            source=RUN/'baseline_repair_budget_v1/2wiki'/split/name/'checkpoint.sqlite3'
        directory=OUT/track/dataset/split/mid/config['id']
        identity={'dataset':dataset,'split':split,'method':mid,'track':track,'config':config,
            'token_policy_sha256':sha(OUT/'CONFIG_FROZEN.json'),
            'input_sha256':sha(INPUTS/dataset/(split+'.questions.jsonl')),
            'resource_sha256':sha(PARENT/dataset/'RESOURCE_FROZEN.json'),
            'model':MODEL,'source_checkpoint':str(source)}
        store=Store(directory,identity)
        catalog=Catalog(source)
        resource=read(PARENT/dataset/'RESOURCE_FROZEN.json')['resources']
        retrieval=Retrieval(store,resource,catalog)
        if dataset in self.handles:
            retrieval.index,retrieval.encoder,retrieval.chunks,retrieval.sparse=self.handles[dataset]
        module=importlib.import_module(__package__+'.methods.'+METHODS[mid]+'.method')
        try:
            remaining=[c for c in cases if store.get('prediction',c['qid']) is None]
            for c in remaining:
                if c['qid'] in catalog.hybrid:
                    store.put('legacy_hybrid',c['qid'],catalog.hybrid[c['qid']])
            if remaining:
                transport=Transport(store,self.client,self.budget,MODEL,catalog)
                serial=mid in {'b07_hybrid_rerank','b08_qd_rerank'}
                batch_size=len(remaining) if serial else 60
                for start in range(0,len(remaining),batch_size):
                    if (OUT/'STOP_REQUESTED').exists():
                        raise IntegrityError('User stop requested')
                    save(OUT/'progress.json',{'utc':utc(),'status':'running','track':track,
                        'dataset':dataset,'split':split,'method':mid,'config':config,
                        'remaining_in_method':len(remaining)-start,'budget':self.budget.summary()})
                    block=remaining[start:start+batch_size]
                    with provider_dns(urlsplit(self.receipt['base_url']).hostname,store):
                        if serial:
                            module.run(block,retrieval,transport,store,config)
                        else:
                            api=API(transport,8,2/3)
                            try:
                                execute(module,block,retrieval,transport,store,config,api,60)
                            finally:
                                api.close()
                    scored,audit=score_store(dataset,split,directory)
                    print(json.dumps({'track':track,'dataset':dataset,'split':split,'method':mid,
                        'overall':scored['overall'],'token_integrity_passed':audit['passed'],
                        'max_input':audit['max_actual_prompt_tokens'],'max_output':audit['max_actual_completion_tokens'],
                        'budget':self.budget.summary()},ensure_ascii=False),flush=True)
            return score_store(dataset,split,directory)
        finally:
            if retrieval.index is not None:
                self.handles[dataset]=(retrieval.index,retrieval.encoder,retrieval.chunks,retrieval.sparse)
            store.export(full=True)
            store.db.close()
            del catalog
            gc.collect()

    def matched(self, split, size):
        results={}
        for mid in ['b05_iter_retgen_bge','b03_react_dense']:
            config={'id':'k5_t5' if mid=='b05_iter_retgen_bge' else 'k5_s5','k':5,
                    **({'iterations':5} if mid=='b05_iter_retgen_bge' else {'max_search':5})}
            results[mid]=self.run_method('matched25','2wiki',split,mid,config,size)
        return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','dev','formal','matched25'],default='all')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    from .prepare import main as prepare
    receipt=prepare()
    if args.prepare_only:
        print('Frozen; no API calls')
        return
    validation=read(OUT/'OFFLINE_VALIDATION.json')
    if not validation['passed']:
        raise IntegrityError('Offline validation required')
    for path,expected in validation['validated_source_sha256'].items():
        if sha(ROOT/path)!=expected:
            raise IntegrityError('Code differs from offline validation: '+path)
    install_offsets()
    options=get_api_option(2)
    if options['model']!=MODEL or options['base_url']!=receipt['base_url']:
        raise IntegrityError('Generator endpoint changed')
    budget=Budget(OUTPUT/'budget.sqlite3',200,2/3)
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'))
            network=stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=2,
                limits=httpx.Limits(max_connections=8,max_keepalive_connections=8,keepalive_expiry=10)),timeout=60))
            client=stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'],base_url=receipt['base_url'],
                max_retries=0,timeout=60,http_client=network))
            runner=Runner(receipt,StreamingClient(client),budget)
            if args.stage in {'all','dev'}:
                checks=[]
                for dataset in DATASETS:
                    for mid,config in receipt['native_selected_configs_preserved'][dataset].items():
                        scored,audit=runner.run_method('main',dataset,'dev',mid,config,20)
                        if scored['overall']['terminal_records']<20 or not audit['passed']:
                            raise IntegrityError('Development integrity gate incomplete')
                        checks.append({'dataset':dataset,'method':mid,'terminal':20,'token_integrity_passed':True})
                for mid,(scored,audit) in runner.matched('dev',20).items():
                    if scored['overall']['terminal_records']<20 or not audit['passed']:
                        raise IntegrityError('Matched development gate incomplete')
                    checks.append({'dataset':'2wiki','track':'matched25','method':mid,'terminal':20,'token_integrity_passed':True})
                once(OUT/'DEV_GATE.json',{'passed':True,'checks':checks,'f1_used_for_selection':False,
                    'gate':'all29 method/dataset configurations have20 terminal records and zero token integrity errors'})
                journal('development_integrity_gate_passed',configurations=len(checks),formal_selection=False)
            if args.stage in {'all','formal','matched25'}:
                if not read(OUT/'DEV_GATE.json')['passed']:
                    raise IntegrityError('Development gate required before formal requests')
            if args.stage in {'all','formal'}:
                for dataset in DATASETS:
                    for mid,config in receipt['native_selected_configs_preserved'][dataset].items():
                        runner.run_method('main',dataset,'formal',mid,config,500)
                    if dataset=='2wiki':
                        runner.matched('formal',500)
                    from .report import main as report
                    report()
            if args.stage=='matched25':
                runner.matched('formal',500)
            from .report import main as report
            report()
            save(OUT/'progress.json',{'utc':utc(),'status':'completed','stage':args.stage,'budget':budget.summary()})
            journal('stage_completed',stage=args.stage,budget=budget.summary())
    except BaseException as exc:
        save(OUT/'progress.json',{'utc':utc(),'status':'stopped_for_audit','error':type(exc).__name__,'reason':str(exc),'budget':budget.summary()})
        journal('stage_stopped',error=type(exc).__name__,reason=str(exc))
        raise
    finally:
        budget.db.close()


if __name__=='__main__':
    main()
