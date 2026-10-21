"""Budgeted multi-dataset run, immutable methods, native scoring, no test tuning."""
import argparse
from contextlib import ExitStack
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit
import httpx
from openai import OpenAI
from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store,utc
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget
from experiments.query_decomposition.hotpot_formal_eval_v1.reuse_sparse_validation import seed as seed_sparse
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful.mechanics import structural_errors
from .common import ROOT,HERE,RUN,OUTPUT,DATA,read,rows,save,once,sha,log,audit_prior
from .retrieval import Retrieval
from .execution import ExactTransport,run_parallel
from . import scoring
from .methods.ablations.support import seed_exact_responses

TARGET=RUN/'campaign_api_v1'
OLD_PLAN=ROOT/'experiments/query_decomposition/hotpot_formal_eval_v1'
OLD_PREFIX='experiments.query_decomposition.hotpot_formal_eval_v1.methods.'
NEW_PREFIX='experiments.query_decomposition.comprehensive_eval_v1_20261015.methods.'
DATASETS=['hotpotqa','2wiki','musique']
SERIAL={'b07_hybrid_rerank','b08_qd_rerank'}
OUR='m00_s02_child4_faithful'
CONTROL='c01_same_final_reader'
ABLATIONS={'a02_short_qa':'short_qa','a03_no_goal_retrieval':'no_goal_retrieval','a04_no_final_constraints':'no_final_constraints'}

def protocol():
    previous=read(OUTPUT/'baseline_compact100_200_20261014/BASELINE_CONFIGS_FROZEN.json')['selected']
    definitions=[m for m in read(OLD_PLAN/'compact_100_200_20261014/PLAN_CONFIG.json')['methods'] if not m['id'].startswith('m00')]
    buddies={'b02_selfask_dense':'k10_s5','b03_react_dense':'k5_s5','b04_ircot_bge':'k8',
      'b05_iter_retgen_bge':'t3','b06_decomp_bge':'k6_cot','b07_hybrid_rerank':'k50_rrf60'}
    methods={m['id']:{'module':OLD_PREFIX+m['directory']+'.method','default':previous[m['id']]} for m in definitions}
    target_grids={}
    for m in definitions:
        target_grids[m['id']]=[previous[m['id']]]
        if m['id'] in buddies:
            target_grids[m['id']].append(next(c for c in m['dev_configs'] if c['id']==buddies[m['id']]))
    methods['b08_qd_rerank']={'module':NEW_PREFIX+'qd_rerank.method','default':{'id':'standalone5','max_queries':5}}
    target_grids['b08_qd_rerank']=[{'id':'standalone3','max_queries':3},{'id':'standalone5','max_queries':5}]
    methods[OUR]={'module':OLD_PREFIX+'s02_child4_faithful.method','default':{'id':'frozen_s02_child4','original_k':10,'child_k':10,'goal_k':10,
       'short_child_tokens':64,'contextual_child_tokens':128,'goal_tokens':128,'final_tokens':64}}
    methods[CONTROL]={'module':NEW_PREFIX+'same_reader_top10.method','default':{'id':'matched_final_reader','k':10,'output_tokens':64}}
    for mid,name in ABLATIONS.items():methods[mid]={'module':NEW_PREFIX+name+'.method','default':{'id':'fixed_packets'}}
    return {'methods':methods,'development_grids':target_grids,
      'target_dev_selection':'two preregistered candidates per adaptable family on separate dev100; no formal-score reads',
      'hotpot_existing_eight_parameters':'reuse prior dev100 selected configurations unchanged; tune only new QD+Rerank on new dev100',
      'same_reader_control':'exact full-method final prompt/examples/output; frozen v1 fallback matched per case; original Top10 only',
      'four_ablations':'A1 reuses C1; A2/A3/A4 private source-packet copies, fixed formal200 subset; legacy200 post-hoc separately',
      'protected_budget_remaining_cny':5,'global_cap_cny':200,'statistical_primary':'same-question native Answer F1',
      'primary_contrasts':[OUR+' vs '+m for m in ['b00_rag_top10_v1','b03_react_dense','b08_qd_rerank',CONTROL]],
      'holm_family':'four primary contrasts within each dataset; ablations separately; paired stratified group bootstrap10000',
      'missingness':'planned denominators preserved; paired common cohort and all-planned difference bounds',
      'stages_deferred':['no_retrieval_control','hallucination_audit','second_generator','robustness_expansion'],
      'remote_scope':'only query decomposition, manual transfer','new_training':False}

def freeze():
    audit=audit_prior();cfg=protocol();options=get_api_option(2)
    expected=read(OUTPUT/'baseline_compact100_200_20261014/CAMPAIGN_FROZEN.json')
    if options['model']!=expected['generator'] or options['base_url']!=expected['api_base_url']:raise ValueError('Shared generator/endpoint changed')
    paths=list(HERE.rglob('*.py'))+list(HERE.glob('*.txt'))+[HERE/'PLAN.json']
    paths+=list((HERE/'inputs_frozen').rglob('*.json'))+list((HERE/'inputs_frozen').rglob('*.jsonl'))
    paths+=list(DATA.glob('*/official/**/*.py'))+[DATA/'2wiki/native/id_aliases.json',DATA/'SOURCES_FROZEN.json']
    pins={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    receipt={'protocol':cfg,'generator':options['model'],'base_url':options['base_url'],'source_sha256':pins,
             'prior_frozen_checks':audit,'local_api_budget_root':str(OUTPUT/'budget.sqlite3'),'sample_seed':2026101501}
    path=TARGET/'CAMPAIGN_FROZEN.json'
    if not path.exists():
        save(path,receipt)
        for name in pins:
            dest=TARGET/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
        log('comprehensive_api_protocol_frozen',receipt=str(path),files=len(pins),protocol=cfg,generator=options['model'])
    elif read(path)!=receipt:raise ValueError('Campaign sources/protocol changed after freeze')
    return receipt,options

def verify(receipt):
    for name,expected in receipt['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen campaign file changed: '+name)
    audit_prior()

def questions(dataset,split):
    path=HERE/'inputs_frozen'/dataset/(split+'.questions.jsonl');rs=rows(path)
    if any(set(r)!={'qid','question'} for r in rs) or len({r['qid'] for r in rs})!=len(rs):raise ValueError('Gold-free runtime contract violated')
    return path,rs

def source_directory(dataset):return TARGET/dataset/'formal'/OUR/'frozen_s02_child4'

def attach_inputs(store,cases,dataset,split,mid,config):
    if mid==OUR:
        path=RUN/'remote1500'/(dataset+'.decompositions.jsonl')
        if not path.exists():raise FileNotFoundError('Await manual remote query decomposition')
        rs={r['id']:r for r in rows(path)}
        if set(rs)!={r['qid'] for r in questions(dataset,'formal')[1]}:raise ValueError('Remote dataset IDs differ')
        for case in cases:
            r=rs[case['qid']]
            if r['query']!=case['question']:raise ValueError('Remote query text differs')
            target=None if r.get('generation_error') or r.get('parse_error') else r['parsed_target']
            store.put('decomposition',case['qid'],{'qid':case['qid'],'qdmr':target,'raw':r,'structural_errors':structural_errors(target)})
    if mid in ABLATIONS or mid==CONTROL:
        seed_exact_responses(config['source_directory'],store,cases)

def quarantine_unknown(store):
    unknown={r[0] for r in store.db.execute("SELECT id FROM requests WHERE status IN ('unknown','in_flight')")}
    count=0
    for (value,) in store.db.execute("SELECT value FROM data WHERE kind='evidence'").fetchall():
        r=json.loads(value)
        if r['request_id'] in unknown and store.get('prediction',r['qid']) is None:
            store.put('prediction',r['qid'],{'qid':r['qid'],'prediction':None,'status':'unknown','request_id':r['request_id'],
              'failed_stage':r['stage'],'reason':'unresolved_submission_retained_not_replayed'})
            count+=1
    if unknown and not count and any(store.get('prediction',json.loads(v)['qid']) is None for (v,) in store.db.execute("SELECT value FROM data WHERE kind='evidence'").fetchall()):
        raise RuntimeError('An unknown request lacks safe question attribution')
    return count

class Campaign:
    def __init__(self,receipt,options,client):
        self.receipt=receipt;self.options=options;self.client=client;self.resources={};self.handles={};self.stores={}
        self.budget=ParallelBudget(OUTPUT/'budget.sqlite3',200)
    def resource(self,dataset):
        if dataset not in self.resources:
            path=ROOT/'experiments/query_decomposition/query_form_ablation_v1/resources.json' if dataset=='hotpotqa' else HERE/'resources_frozen'/(dataset+'.json')
            resource=read(path)
            normalization=None
            if dataset!='hotpotqa':
                # The builder inherits the shared encoder specification. Its
                # parent Hotpot build labels are not native corpus identities.
                dense=resource['dense']['bge']
                dense['build_id']='comprehensive_'+dataset+'_global_bge_v1'
                dense['embeddings_manifest_sha256']=dense['manifest_sha256']
                normalization='replace_unused_parent_dense_build_labels_with_native_identity; paths/vectors/model unchanged'
            once(TARGET/dataset/'RESOURCE_FROZEN.json',{'authority':str(path),'sha256':sha(path),'resources':resource,'metadata_normalization':normalization})
            if dataset=='hotpotqa':seed_sparse(resource)
            self.resources[dataset]=resource
        return self.resources[dataset]
    def progress(self,**fields):
        save(TARGET/'progress.json',{'utc':utc(),'budget':self.budget.summary(),**fields})
    def execute(self,dataset,split,mid,config,cases):
        verify(self.receipt);resource=self.resource(dataset)
        cfg=json.loads(json.dumps(config));question_path,_=questions(dataset,split)
        if mid==OUR:cfg['helper_positions']=read(HERE/'inputs_frozen'/dataset/'formal.positions.json')
        if mid in ABLATIONS or mid==CONTROL:
            cfg['source_directory']=str(source_directory(dataset))
            cfg['helper_positions']=read(HERE/'inputs_frozen'/dataset/'formal.positions.json')
        key=(dataset,split,mid,cfg['id']);directory=TARGET/dataset/split/mid/cfg['id']
        module=importlib.import_module(self.receipt['protocol']['methods'][mid]['module'])
        if key not in self.stores:
            identity={'dataset':dataset,'split':split,'method':mid,'config':cfg,'questions_sha256':sha(question_path),
              'campaign_sha256':sha(TARGET/'CAMPAIGN_FROZEN.json'),'resources_sha256':sha(TARGET/dataset/'RESOURCE_FROZEN.json'),
              'generator':self.options['model'],'base_url':self.options['base_url']}
            store=Store(directory,identity);quarantine_unknown(store)
            self.stores[key]=store
        store=self.stores[key]
        attach_inputs(store,cases,dataset,split,mid,cfg)
        remaining=[c for c in cases if store.get('prediction',c['qid']) is None]
        if remaining:
            liability=self.budget.summary();total=liability['returned_usage_list_price_cny']+liability['unreconciled_reserved_cny']
            if total>=195:raise RuntimeError('Protected global budget reserve reached; preserve unstarted denominators')
            self.progress(status='running',dataset=dataset,split=split,method=mid,config=cfg['id'],block_questions=len(remaining))
            print(json.dumps({'start':dataset,'split':split,'method':mid,'config':cfg['id'],'questions':len(remaining)}),flush=True)
            retrieval=Retrieval(store,resources=resource)
            if dataset in self.handles:retrieval.index,retrieval.encoder,retrieval.chunks,retrieval.sparse=self.handles[dataset]
            transport=ExactTransport(store,self.client,self.budget,self.options['model'])
            with provider_dns(urlsplit(self.options['base_url']).hostname,store):
                if mid not in SERIAL:run_parallel(module,remaining,retrieval,transport,store,cfg)
                else:module.run(remaining,retrieval,transport,store,cfg)
            if retrieval.index is not None:self.handles[dataset]=(retrieval.index,retrieval.encoder,retrieval.chunks,retrieval.sparse)
        store.export(full=True)
        ref=HERE/'inputs_frozen'/dataset/((split if split=='ablation' else split)+'.references.local.jsonl')
        report=scoring.evaluate(dataset,directory,ref)
        print(json.dumps({'done':dataset,'split':split,'method':mid,'config':cfg['id'],'overall':report['overall'],'cost_cny':report['returned_usage_cny']}),flush=True)
    def selection(self,dataset):
        path=TARGET/dataset/'BASELINE_CONFIGS_FROZEN.json'
        if path.exists():return read(path)['selected']
        methods=self.receipt['protocol']['methods'];grids=self.receipt['protocol']['development_grids']
        _,cases=questions(dataset,'dev')
        selected={mid:m['default'] for mid,m in methods.items() if mid.startswith('b') and dataset=='hotpotqa'}
        active={'b08_qd_rerank':grids['b08_qd_rerank']} if dataset=='hotpotqa' else grids
        for block in range(0,100,20):
            for mid,configs in active.items():
                for cfg in configs:self.execute(dataset,'dev',mid,cfg,cases[block:block+20])
        diagnostics={}
        for mid,configs in active.items():
            records=[]
            for cfg in configs:
                directory=TARGET/dataset/'dev'/mid/cfg['id'];results={r['qid']:r for r in rows(directory/'scores.jsonl')}
                records.append((cfg,results,read(directory/'scores.json')))
            common=set.intersection(*({q for q,r in result.items() if r['f1'] is not None} for cfg,result,report in records))
            if len(common)<90:raise RuntimeError('Less than90/100 same-family development availability')
            scores=[{'config':cfg,'f1':sum(result[q]['f1'] for q in common)/len(common),'cost':report['logical_cold_cost_cny'],
                     'calls':report['logical_calls']} for cfg,result,report in records]
            chosen=min(scores,key=lambda r:(-r['f1'],r['cost'],r['calls'],r['config']['id']))
            selected[mid]=chosen['config'];diagnostics[mid]={'common':len(common),'missing_ids':sorted({c['qid'] for c in cases}-common),'candidates':scores}
        value={'selected':selected,'dataset':dataset,'dev_questions':100,'selection_formal_scores_read':False,
               'within_family_common_cohorts':diagnostics,'our_prompt_parameters_selected':False}
        once(path,value);log('dataset_baseline_parameters_frozen',**value)
        return selected
    def baseline_main(self,dataset):
        selected=self.selection(dataset);_,cases=questions(dataset,'formal')
        for block in range(0,500,20):
            ordered=list(selected);offset=(block//20)%len(ordered);ordered=ordered[offset:]+ordered[:offset]
            for mid in ordered:self.execute(dataset,'formal',mid,selected[mid],cases[block:block+20])
        return selected
    def our_main_and_controls(self,dataset):
        _,cases=questions(dataset,'formal');methods=self.receipt['protocol']['methods']
        for block in range(0,500,20):self.execute(dataset,'formal',OUR,methods[OUR]['default'],cases[block:block+20])
        for block in range(0,500,20):self.execute(dataset,'formal',CONTROL,methods[CONTROL]['default'],cases[block:block+20])
        _,subset=questions(dataset,'ablation')
        for block in range(0,200,20):
            for mid in ABLATIONS:self.execute(dataset,'ablation',mid,methods[mid]['default'],subset[block:block+20])
    def close(self):
        for store in self.stores.values():store.export(full=True);store.db.close()
        self.budget.db.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--stage',choices=['baselines','ours','all'],default='all');args=parser.parse_args()
    receipt,options=freeze()
    if args.freeze_only:print(json.dumps({'frozen':True,'source_files':len(receipt['source_sha256'])}));return
    from .accept_return import main as accept_return
    with ExitStack() as stack:
        stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'))
        network=stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=2,limits=httpx.Limits(max_connections=3,max_keepalive_connections=3,keepalive_expiry=10)),timeout=60))
        client=stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'],base_url=options['base_url'],timeout=60,max_retries=0,http_client=network))
        campaign=Campaign(receipt,options,StreamingClient(client))
        try:
            ready=[d for d in DATASETS if d=='hotpotqa' or (HERE/'resources_frozen'/(d+'.json')).exists()]
            if args.stage in {'baselines','all'}:
                for dataset in DATASETS:
                    if dataset=='hotpotqa' or (HERE/'resources_frozen'/(dataset+'.json')).exists():campaign.baseline_main(dataset)
                from .legacy_ablations import run as legacy_diagnostics
                legacy_diagnostics(campaign)
            ready=[d for d in DATASETS if d=='hotpotqa' or (HERE/'resources_frozen'/(d+'.json')).exists()]
            remote=accept_return()
            if args.stage in {'ours','all'} and remote.get('passed'):
                for dataset in ready:
                    if not (TARGET/dataset/'BASELINE_CONFIGS_FROZEN.json').exists():continue
                    expected={c['qid'] for c in questions(dataset,'formal')[1]}
                    selected=read(TARGET/dataset/'BASELINE_CONFIGS_FROZEN.json')['selected']
                    complete=True
                    for mid,cfg in selected.items():
                        p=TARGET/dataset/'formal'/mid/cfg['id']/'scores.json'
                        if not p.exists() or read(p)['overall']['terminal_records']!=len(expected):complete=False
                    if complete:campaign.our_main_and_controls(dataset)
            campaign.progress(status='pass_completed_or_waiting_for_dependencies',ready_datasets=ready,remote=remote)
        except BaseException as exc:
            campaign.progress(status='stopped_for_audit',error_type=type(exc).__name__,reason=str(exc));raise
        finally:campaign.close()

if __name__=='__main__':main()
