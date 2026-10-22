"""Scoped quick90: jointly replace the three exact user-supplied prompts."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, closing
from copy import deepcopy
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
import argparse
import ast
import difflib
import hashlib
import importlib
import json
import os
import socket
import sqlite3
import ssl
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021 import run as common
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021.run_r004 import RecoveryAsk
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.runtime import Budget, check_resources
from . import prompts

PACKAGE, COHORT, PLAN = common.PACKAGE, common.COHORT, common.PLAN_DIR
OUT = common.OUT.parent / HERE.name
read, rows, save, sha, emit = common.read, common.rows, common.save, common.sha, common.emit


def ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def module():
    return importlib.import_module('experiments.query_decomposition.' + HERE.name + '.pipeline')


def prepare():
    assert not (HERE / 'PROTOCOL.frozen.json').exists(), 'Preserve the frozen protocol'
    plan = read(PLAN / 'PLAN.json')
    assert plan['execution_approval']['approved'] is False
    (PLAN / 'PLAN.txt').read_text(encoding='utf8')
    (PLAN / 'DECISIONS.jsonl').read_text(encoding='utf8')
    source = (PACKAGE / 'core/pipeline.py').read_text(encoding='utf8')
    changed = source.replace('from . import METHOD, ROOT',
        "from p02_multigrain_timing_cachefix_v1.core import ROOT\nMETHOD = 'p02_three_prompts_user_v1'", 1)
    changed = changed.replace('from . import child_prompts, rewrite_prompts, final_prompts',
        'from p02_multigrain_timing_cachefix_v1.core import rewrite_prompts\nfrom . import prompts as child_prompts, prompts as final_prompts', 1)
    old = '''            if is_goal:
                prompt = (
                    prompt[: -len("Answer:")] + child_prompts.GOAL_RULE + "\\n\\nAnswer:"
                )
'''
    assert changed.count(old) == 1
    changed = changed.replace(old, '', 1)
    assert changed.count('context = block + "\\n\\n" + passages(root)') == 1
    changed = changed.replace('context = block + "\\n\\n" + passages(root)',
        'context = block + "\\n\\nRetrieved passages:\\n" + passages(root)', 1)
    ast.parse(changed)
    (HERE / 'pipeline.py').write_text(changed, encoding='utf8')
    (HERE / 'PIPELINE_CHANGE.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True), changed.splitlines(True), fromfile='frozen_P02', tofile=HERE.name)), encoding='utf8')
    cases = rows(COHORT / 'runtime90.frozen.jsonl')
    assert len(cases) == len({c['id'] for c in cases}) == 90
    assert Counter(c['dataset'] for c in cases) == {d: 30 for d in common.DATASETS}
    holds = [PLAN / 'STOP_REQUESTED', ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED', ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED']
    protocol = {
        'id': HERE.name, 'utc': utc(), 'authority': 'User: 在90题快速测试集上 我们先整体评估三段',
        'scope': 'Joint evaluation of the three exact templates supplied in this side conversation; no suggested edits mixed in.',
        'control': 'Frozen original native P02; exact successful recorded-input replay. Not the earlier selfcontained/minimal prompt variants.',
        'candidate': 'Simultaneous ordinary + goal + final prompt substitution, including their supplied examples and formatting.',
        'planned': 90, 'per_dataset': 30, 'cohort_role': 'Existing exposed development quick90, not independent confirmation',
        'fixed': 'Frozen R02 decompositions; original native model, rewriter, BGE, dual-route merge, history propagation, formatter and 256/128/64 output caps.',
        'downstream': 'Candidate ordinary answers drive their own later rewrite and retrieval; do not freeze changed downstream packets.',
        'cache': 'Only identical successful full payloads; original package first, then completed previous quick90 sources in fixed order. Unknown records are blocked unless an exact successful receipt exists.',
        'failure_policy': 'Existing connect-only retries2, SDK POST retries0. No automatic ambiguous-request resubmission, algorithm rerun for score, sample replacement or candidate activation.',
        'measurement': 'Native final F1/EM, paired gains/ties/losses, complete-flow input/output tokens, estimated generation expense and explicit missingness. No new timing claims.',
        'destination': read(PACKAGE / 'config.json')['generation']['base_url'],
        'budget': 'Unchanged native P02 cumulative ledger and cap; preserve all historical liabilities. No main campaign resume.',
        'main_evaluation_remains_paused': True,
        'inputs': {str((COHORT / n).relative_to(ROOT)): sha(COHORT / n) for n in ['runtime90.frozen.jsonl','references90.local.frozen.jsonl','COHORT.frozen.json','SOURCE_BINDINGS.frozen.jsonl']},
        'pins': {str(p.relative_to(ROOT)): sha(p) for p in [HERE / 'run.py', HERE / 'prompts.py', HERE / 'pipeline.py', HERE / 'ordinary.frozen.txt', HERE / 'goal.frozen.txt', HERE / 'final.frozen.txt', PACKAGE / 'config.json', PACKAGE / 'core/pipeline.py', PACKAGE / 'core/child_prompts.py', PACKAGE / 'core/final_prompts.py', common.HERE / 'run.py', common.HERE / 'run_r004.py', common.HERE / 'run_r002.py']},
        'hold_pins': {str(p.relative_to(ROOT)): sha(p) for p in holds if p.exists()},
    }
    save(HERE / 'PROTOCOL.frozen.json', protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'AUTHORIZATION.json', protocol)
    # Append the scoped authority to the existing agreement without changing its
    # shared PLAN.json or any main-campaign pause/control file.
    with (PLAN / 'DECISIONS.jsonl').open('a', encoding='utf8') as handle:
        handle.write(json.dumps({'id': HERE.name.upper(), 'utc': utc(), 'authority': protocol['authority'],
            'scope': protocol['scope'], 'protocol': str((HERE / 'PROTOCOL.frozen.json').relative_to(ROOT)),
            'status': 'approved_scoped_joint_three_prompt_quick90', 'main_evaluation_remains_paused': True}, ensure_ascii=False) + '\n')
    return {'prepared': True, 'planned': 90}


def check():
    p = read(HERE / 'PROTOCOL.frozen.json')
    for group in ['inputs','pins','hold_pins']:
        for name, wanted in p[group].items():
            assert sha(ROOT / name) == wanted, name
    assert read(PLAN / 'PLAN.json')['execution_approval']['approved'] is False
    return p


def seed():
    path = OUT / 'seed.sqlite3'
    assert not path.exists()
    with closing(sqlite3.connect(path)) as db, closing(ro(PACKAGE / 'data/replay.sqlite3')) as original:
        db.executescript('CREATE TABLE requests(payload TEXT PRIMARY KEY,request_id TEXT,result TEXT,origin TEXT); CREATE TABLE retrieval(dataset TEXT,qid TEXT,query TEXT,packet TEXT,origin TEXT,PRIMARY KEY(dataset,qid,query)); CREATE TABLE blocked(payload TEXT PRIMARY KEY,request_id TEXT,status TEXT);')
        db.executemany('INSERT INTO requests VALUES(?,?,?,?)', [(*v,'original_P02') for v in original.execute('SELECT payload,request_id,result FROM requests')])
        db.executemany('INSERT INTO retrieval VALUES(?,?,?,?,?)', [(*v,'original_P02') for v in original.execute('SELECT dataset,qid,query,packet FROM retrieval')])
        prior_dirs = [common.OUT / 'approved_recovery004',
            common.OUT.parent / 'p02_prompt_minimal_ablation_quick90_v1_20261021/A_no_only',
            common.OUT.parent / 'p02_prompt_minimal_ablation_quick90_v1_20261021/B_no_resolve', common.OUT]
        for directory in prior_dirs:
            for ds in common.DATASETS:
                cache = directory / 'candidate' / ds / 'checkpoint.sqlite3'
                if not cache.exists():
                    continue
                with closing(ro(cache)) as source:
                    for payload,rid,status,result in source.execute('SELECT payload,id,status,result FROM requests'):
                        if status == 'success':
                            parsed = json.loads(result)
                            if parsed.get('usage'):
                                db.execute('INSERT OR IGNORE INTO requests VALUES(?,?,?,?)',(payload,rid,result,str(cache.relative_to(ROOT))))
                        else:
                            db.execute('INSERT OR IGNORE INTO blocked VALUES(?,?,?)',(payload,rid,status))
                    for key,value in source.execute("SELECT key,value FROM data WHERE kind='retrieval'"):
                        dataset,qid,query,k = json.loads(key)
                        assert dataset == ds and k == 10
                        db.execute('INSERT OR IGNORE INTO retrieval VALUES(?,?,?,?,?)',(ds,qid,query,value,str(cache.relative_to(ROOT))))
        db.execute('DELETE FROM blocked WHERE payload IN (SELECT payload FROM requests)')
        db.commit()


def offline():
    check()
    seed()
    config = read(PACKAGE / 'config.json')
    controls = []
    changed = Counter()
    with closing(ro(OUT / 'seed.sqlite3')) as db:
        for case in rows(COHORT / 'runtime90.frozen.jsonl'):
            control = common.baseline_replay(case, db, config)
            controls.append(control)
            nodes = control['trajectory']['nodes']
            for node in nodes:
                actual = prompts.child(case['question'],node['step_id'],node['request'],node['prior_complete_answers'],common.baseline.passages(node['reader_packet']),[],role=node['role'])
                assert actual != node['child_prompt'] and actual.endswith('Answer:')
                assert 'Resolve each #N' not in actual
                assert actual.count('Example ') == (0 if node['role']=='goal' else 4)
                if node['role']=='goal':
                    assert 'Current subquestion #' not in actual
                    assert actual == prompts.GOAL.format_map({'ORIGINAL_QUESTION':case['question'],'PRIOR_QA_BLOCK':prompts.prior_block(node['prior_complete_answers']),'GOAL_RETRIEVED_PASSAGES':common.baseline.passages(node['reader_packet'])})
                changed[node['role']] += 1
            packet = control['trajectory']['original_retrieval']
            final,context,block = module().render_final(case['question'],nodes,packet,config['readout'],[])
            values = {'ORIGINAL_QUESTION':case['question'], 'INTERMEDIATE_QA_BLOCK':prompts.prior_block(nodes[:-1]) if nodes[:-1] else '',
                      'GOAL_STEP_ID':nodes[-1]['step_id'],'GOAL_ANSWER':nodes[-1]['answer'],
                      'ORIGINAL_QUERY_RETRIEVED_PASSAGES':common.baseline.passages(packet)}
            if nodes[:-1]:
                assert final == prompts.FINAL.format_map(values)
            assert final.count('Example ') == 5 and final.endswith('Final short answer:')
            assert context == block + '\n\nRetrieved passages:\n' + common.baseline.passages(packet)
            changed['final'] += 1
    (OUT / 'control90.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in controls),encoding='utf8')
    assert dict(changed) == {'ordinary':176,'goal':90,'final':90}
    result = {'passed':True,'controls_replayed':90,'changed_prompts_on_same_input':dict(changed),
        'exact_user_template_rendering_verified':True,'seed_sha256':sha(OUT/'seed.sqlite3'),'new_API_requests':0}
    save(OUT / 'OFFLINE_CHECK.json',result)
    return result


def run_dataset(dataset):
    from p02_multigrain_timing_cachefix_v1.core.retrieval import Retriever
    from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
    config = read(PACKAGE / 'config.json')
    directory = OUT / 'candidate' / dataset
    current_id = None
    def event(value):
        with (directory / 'trace.jsonl').open('a',encoding='utf8') as handle:
            handle.write(json.dumps({'utc':utc(),'id':current_id,**value},ensure_ascii=False)+'\n')
    with run_lock(directory), ExitStack() as stack:
        saved = stack.enter_context(closing(ro(OUT / 'seed.sqlite3')))
        cache = stack.enter_context(Cache(directory / 'checkpoint.sqlite3'))
        cache.put('meta','identity',{'protocol_sha256':sha(HERE/'PROTOCOL.frozen.json'),'dataset':dataset})
        budget = stack.enter_context(closing(Budget(config,directory,config['budget']['cumulative_cap_cny'])))
        live = stack.enter_context(closing(RecoveryAsk(cache,saved,config,budget,event)))
        retriever = Retriever(config['resources'][dataset],device='cuda')
        stack.callback(retriever.close)
        pipeline = module().Pipeline(config)
        for index,case in enumerate([common.baseline.safe_case(c) for c in rows(COHORT/'runtime90.frozen.jsonl') if c['dataset']==dataset],1):
            if common.STOP.is_set():
                break
            current_id = case['id']
            if cache.get('prediction',current_id) is not None:
                continue
            def retrieve(query):
                key = canonical([dataset,case['qid'],query,10])
                packet = cache.get('retrieval',key)
                if packet is None:
                    row = saved.execute('SELECT packet,origin FROM retrieval WHERE dataset=? AND qid=? AND query=?',(dataset,case['qid'],query)).fetchone()
                    packet,origin = (json.loads(row[0]),row[1]) if row else (retriever(case['qid'],query),'new_index_search')
                    cache.put('retrieval',key,packet)
                    event({'event':'retrieval','query':query,'source':origin})
                return deepcopy(packet)
            def ask(stage,prompt,maximum,allow_final_truncation=False):
                payload = canonical(build_payload(config['generation'],prompt,maximum))
                blocked = saved.execute('SELECT request_id FROM blocked WHERE payload=?',(payload,)).fetchone()
                if blocked:
                    raise RuntimeError('Existing unresolved payload is held; no resubmission')
                return live(stage,prompt,maximum,allow_final_truncation=allow_final_truncation)
            try:
                result = pipeline.execute(case,retrieve,ask,record_node=lambda n:cache.put('node',current_id+'/'+str(n['step_id']),n))
                cache.put('prediction',current_id,result)
                emit({'dataset':dataset,'completed':index,'planned':30,'status':result['status']})
            except Exception as exc:
                failure = {'id':current_id,'dataset':dataset,'status':'incomplete','error_class':type(exc).__name__,'error_summary':str(exc)[:200]}
                save(directory/('failure_'+hashlib.sha256(current_id.encode()).hexdigest()[:16]+'.json'),failure)
                event({'event':'case_incomplete',**failure})
                receipts = cache.requests()
                if any(v['status']!='success' for v in receipts):
                    common.STOP.set()
                    raise RuntimeError('Unresolved provider receipt; stopped without resubmission') from None
                if isinstance(exc,RuntimeError) and any(s in str(exc) for s in ['Incomplete','incomplete','Final generation']):
                    cache.put('prediction',current_id,{**case,'status':'algorithm_failure','prediction':'','error_class':type(exc).__name__})
                    emit({'dataset':dataset,'completed':index,'planned':30,'status':'algorithm_failure'})
                else:
                    common.STOP.set()
                    raise RuntimeError('Unexpected scoped execution error; inspect saved failure') from None


def report():
    old_here,old_out = common.HERE,common.OUT
    try:
        common.HERE,common.OUT = HERE,OUT
        result = common.report()
        result['comparison'] = 'Frozen original P02 versus all three user-supplied prompts jointly'
        result['candidate_not_activated'] = True
        save(OUT/'REPORT.json',result)
        return result
    finally:
        common.HERE,common.OUT = old_here,old_out


def run():
    check()
    assert read(OUT/'OFFLINE_CHECK.json')['passed']
    assert sha(OUT/'seed.sqlite3') == read(OUT/'OFFLINE_CHECK.json')['seed_sha256']
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    from p02_multigrain_timing_cachefix_v1.core.retrieval import Retriever
    config = read(PACKAGE/'config.json')
    check_resources(config,rows(COHORT/'runtime90.frozen.jsonl'))
    loading,old_load = Lock(),Retriever.load
    def serialized_load(self):
        with loading:
            return old_load(self)
    Retriever.load = serialized_load
    class NetworkEvents:
        def event(self,event,**fields):
            with (OUT/'network.jsonl').open('a',encoding='utf8') as handle:
                handle.write(json.dumps({'utc':utc(),'event':event,**fields})+'\n')
    host = urlsplit(config['generation']['base_url']).hostname
    with temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'),provider_dns(host,NetworkEvents()):
        with socket.create_connection((host,443),timeout=15) as sock:
            with ssl.create_default_context().wrap_socket(sock,server_hostname=host):
                NetworkEvents().event('TLS_preflight',paid_request=False)
        save(OUT/'EXECUTION.json',{'utc':utc(),'status':'running','scope':'joint three-prompt quick90 only','main_campaign_resumed':False})
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                for future in as_completed([executor.submit(run_dataset,ds) for ds in common.DATASETS]):
                    future.result()
        finally:
            common.STOP.set()
            result = report()
    check()
    with (PLAN/'DECISIONS.jsonl').open('a',encoding='utf8') as handle:
        handle.write(json.dumps({'id':HERE.name.upper()+'_COMPLETED','utc':utc(),'status':result['status'],
            'report':str((OUT/'REPORT.json').relative_to(ROOT)),'planned':90,'completed':result['metrics']['overall']['complete'],
            'main_evaluation_remains_paused':True,'candidate_not_activated':True},ensure_ascii=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','offline','run','report'])
    args = parser.parse_args()
    emit(globals()[args.action]())
