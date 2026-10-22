"""Bounded frozen-prefix follow-up, strict prior-receipt replay, no main resume."""
from collections import Counter
from contextlib import ExitStack,closing
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import hashlib
import json
import sqlite3
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from . import templates
from experiments.query_decomposition.p02_goal_target_transfer_new60_v1_20261022 import run as source
from experiments.query_decomposition.p02_goal_target_transfer_new60_v1_20261022.analyze import paired_ci

read,rows,save,jsonl,sha,ro,digest,cost=source.read,source.rows,source.save,source.jsonl,source.sha,source.ro,source.digest,source.cost
Cache,run_lock,utc,build_payload=source.Cache,source.run_lock,source.utc,source.build_payload
prior=source.prev;PLAN=source.PLAN;DATASETS=source.DATASETS
OUT=ROOT/'outputs/query_decomposition'/HERE.name
REVIEW=PLAN/'reviews/20261022_prompt_binding_goal_synthesis'/HERE.name
INDEX=PLAN/'reviews/20261022_prompt_binding_goal_synthesis/side_goal_research'
DECISION='P02_GOAL_SUFFICIENCY_RETENTION60_SIDE_AUTONOMOUS_20261022'


def prepare():
    if (HERE/'PROTOCOL.frozen.json').exists():raise FileExistsError('Preserve frozen protocol')
    config,examples=source.check()
    completion=read(source.OUT/'COMPLETION.json');assert completion['conditions']==240
    for name,h in completion['artifacts'].items():assert sha(source.OUT/name)==h,name
    for name in ('INPUTS60.frozen.jsonl','REFERENCES60.local.frozen.jsonl','HISTORY60.local.frozen.jsonl','SOURCE_BINDINGS.frozen.jsonl'):
        (HERE/name).write_bytes((source.HERE/name).read_bytes())
    save(HERE/'GENERATION.frozen.json',config['generation']);save(HERE/'FINAL_EXAMPLES.frozen.json',examples)
    inputs=rows(HERE/'INPUTS60.frozen.jsonl');assert len(inputs)==60
    # Snapshot complete source receipts into a separate seed file. No source DB mutation.
    seeds=[]
    with closing(ro(source.OUT/'checkpoint.sqlite3')) as db:
        for rid,payload,status,raw in db.execute('SELECT id,payload,status,result FROM requests'):
            result=json.loads(raw) if raw else {};result['status']=status
            if prior.complete_receipt(result) and result.get('returned_model')==config['generation']['model']:
                seeds.append(dict(request_id=rid,payload=json.loads(payload),result=result,
                    source=str((source.OUT/'checkpoint.sqlite3').relative_to(ROOT))))
    jsonl(HERE/'SEED_RECEIPTS.frozen.jsonl',seeds)
    oldresults=[r for r in rows(source.OUT/'DETAILS.local.jsonl') if r['arm'] in ('G0','GF')]
    assert len(oldresults)==120;jsonl(HERE/'CONTROLS.frozen.jsonl',oldresults)
    unknown=list(rows(source.HERE/'EXTERNAL_UNKNOWN.frozen.jsonl'))
    # Current snapshots guard new unknown payloads, including independently running studies.
    for path in sorted((ROOT/'outputs/query_decomposition').glob('p02*/**/checkpoint.sqlite3')):
        if path.is_relative_to(OUT) or 'formal1500' in str(path):continue
        with closing(ro(path)) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='requests'").fetchone():continue
            cols={x[1] for x in db.execute('PRAGMA table_info(requests)')}
            if not {'id','payload','status'}<=cols:continue
            for rid,p,s in db.execute("SELECT id,payload,status FROM requests WHERE status IN ('unknown','in_flight')"):
                unknown.append(dict(request_id=rid,payload_sha256=digest(json.loads(p)),status=s,source=str(path.relative_to(ROOT))))
    unique={(x['request_id'],x['payload_sha256']):x for x in unknown}
    jsonl(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl',list(unique.values()))
    tasks=[]
    for r in inputs:
        offset=int(hashlib.sha256(('retention60|'+r['id']).encode()).hexdigest(),16)%5
        arms=templates.ARMS[offset:]+templates.ARMS[:offset]
        for a in arms:tasks.append(dict(id=r['id'],dataset=r['dataset'],arm=a,key=a+'|'+r['id']))
    jsonl(HERE/'TASKS.frozen.jsonl',tasks)
    for name,text in [('A',templates.ALTERNATIVE),('P',templates.PRESERVATION),('R',templates.READOUT)]:
        (HERE/(name+'.frozen.txt')).write_text(text+'\n',encoding='utf8')
    files=list(HERE.glob('*.frozen.*'))+[HERE/'run.py',HERE/'templates.py',HERE/'analyze.py',
        source.HERE/'run.py',source.HERE/'templates.py',source.HERE/'analyze.py',source.HERE/'PROTOCOL.frozen.json',
        prior.HERE/'run.py',prior.HERE/'templates.py',prior.old.HERE/'run.py',prior.old.HERE/'resume.py',
        HERE.parent/'p02_prompt_binding_probe_v1_20261022/shared_runtime.py',
        HERE.parent/'p02_goal_synthesis_probe_v1_20261022/templates.py',ROOT/'src/p02_multigrain_timing_cachefix_v1/config.json']
    files += [ROOT/'src/p02_multigrain_timing_cachefix_v1/core'/n for n in ('cache.py','llm.py','pipeline.py','final_prompts.py')]
    holds=[PLAN/'STOP_REQUESTED',ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED',
           ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED']
    protocol=dict(id=HERE.name,utc=utc(),authority='做好记录 继续实验; prior autonomous small-study and existing Qwen API expense approval',
        endpoint=config['generation']['base_url'],model=config['generation']['model'],cases=60,conditions=300,
        cohort='Same exposed60 as prior target-transfer round; known cases informed additions; development diagnosis only',
        arms={'G0':'Strict replay of prior factual-sentence control','A':'G0 plus alternative-route sufficiency paragraph',
              'GF':'Strict replay of prior short-synthesis control','P':'GF plus candidate-answer preservation paragraph',
              'R':'Identical replayedGF goal, only final gains whole-paragraph/readout instruction'},
        comparisons=['G0->A','GF->P','GF->R'],combined_variant=False,extra_demonstrations=False,
        fixed='FrozenR02/v12 decomposition, ordinary QA, all goal/root retrieved packets, base examples, model and decoding',
        caps=dict(goal=512,final=64),formatter=False,
        cache='All complete source payloads may be reused exactly with source IDs/model/usage; controls never freshly rerolled; actual new expense separated',
        temporal_limitation='Frozen control receipts, not fresh simultaneous replicas; no independent holdout or full-RAG claim',
        retries='SDK0, durable max5 actual known technical attempts; typed connect-before-send proof only; prior or current unknown held, no automatic reissue',
        expense='Existing local provider gate and shared ledger, no GPU/retrieval allocation; compare logical stage costs, never dilute them with replay cost',
        analysis=dict(native_F1_EM=True,paired_bootstrap=10000,seed=20261022,stratified_by_dataset=True,confidence=.95,
                      multiple_comparison_adjustment=False,exploratory=True,review_all_changed_finals=True,
                      technical_missing='excluded with explicit denominator',algorithm_failure='zero without reroll'),
        stop_after='All300 terminal conditions or explicit technical holds; no outcome-adaptive prompt/sample changes',
        main_evaluation_resumed=False,candidate_promoted=False,publication=False,
        pins={str(p.relative_to(ROOT)):sha(p) for p in files},hold_pins={str(p.relative_to(ROOT)):sha(p) for p in holds})
    save(HERE/'PROTOCOL.frozen.json',protocol);save(REVIEW/'PROTOCOL.json',protocol)
    decision=dict(id=DECISION,approved=True,utc=utc(),authority=protocol['authority'],experiment=HERE.name,
        protocol_sha256=sha(HERE/'PROTOCOL.frozen.json'),cases=60,conditions=300,main_evaluation_resumed=False,unknown_reissue=False)
    with (PLAN/'DECISIONS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(decision,ensure_ascii=False)+'\n')
    INDEX.mkdir(parents=True,exist_ok=True)
    with (INDEX/'ROUNDS.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(dict(utc=utc(),status='prior_round_recorded',experiment=source.HERE.name,
            completion=str((source.OUT/'COMPLETION.json').relative_to(ROOT)),report=str((source.OUT/'INTERPRETATION.zh.md').relative_to(ROOT)),
            F1=dict(G0=67.47049247049247,T=66.97049247049247,GF=68.26190476190476,GF_SPLIT=68.5),promotion=False),ensure_ascii=False)+'\n')
        f.write(json.dumps(decision,ensure_ascii=False)+'\n')
    return dry_run()


def check():
    p=read(HERE/'PROTOCOL.frozen.json')
    for group in ('pins','hold_pins'):
        for path,h in p[group].items():assert sha(ROOT/path)==h,path
    assert read(PLAN/'PLAN.json')['execution_approval']['approved'] is False
    assert any(x.get('id')==DECISION and x.get('approved') for x in rows(PLAN/'DECISIONS.jsonl'))
    config,examples=prior.old.pilot.settings()
    assert config['generation']==read(HERE/'GENERATION.frozen.json') and examples==read(HERE/'FINAL_EXAMPLES.frozen.json')
    return config,examples


def seed_db():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE requests(payload TEXT PRIMARY KEY,request_id TEXT,result TEXT)')
    for x in rows(HERE/'SEED_RECEIPTS.frozen.jsonl'):
        assert prior.complete_receipt(x['result']) and x['result']['returned_model']==x['payload']['model']
        db.execute('INSERT INTO requests VALUES(?,?,?)',(prior.canonical(x['payload']),x['request_id'],prior.canonical(x['result'])))
    db.commit();return db


def dry_run():
    config,examples=check();inputs=rows(HERE/'INPUTS60.frozen.jsonl')
    seed={digest(x['payload']):x for x in rows(HERE/'SEED_RECEIPTS.frozen.jsonl')}
    unknown={x['payload_sha256'] for x in rows(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl')}
    controls={(x['arm'],x['id']):x for x in rows(HERE/'CONTROLS.frozen.jsonl')}
    for r in inputs:
        assert templates.goal_prompt(r,'A').replace('\n\n'+templates.ALTERNATIVE,'',1)==templates.goal_prompt(r,'G0')
        assert templates.goal_prompt(r,'P').replace('\n'+templates.PRESERVATION+'\n','',1)==templates.goal_prompt(r,'GF')
        assert templates.goal_prompt(r,'R')==templates.goal_prompt(r,'GF')
        f,c=templates.final_prompt(r,'FIXED GOAL',config,examples,'GF')
        fr,cr=templates.final_prompt(r,'FIXED GOAL',config,examples,'R')
        assert fr.replace('\n'+templates.READOUT+'\n','',1)==f and cr==c
        for a in templates.ARMS:
            d=digest(build_payload(config['generation'],templates.goal_prompt(r,a),512));assert d not in unknown
        for a in ('G0','GF'):
            control=controls[(a,r['id'])]
            for prompt,cap,answer in [(templates.goal_prompt(r,a),512,control['goal_text']),
                    (templates.final_prompt(r,control['goal_text'],config,examples,a)[0],64,control['prediction'])]:
                d=digest(build_payload(config['generation'],prompt,cap))
                assert seed[d]['result']['answer'].strip()==answer
    result=dict(passed=True,cases=60,conditions=300,strict_control_replay_conditions=120,
        independent_minimal_interventions=True,actual_task_and_examples_preserved=True,new_API_requests=0,main_evaluation_paused=True)
    save(OUT/'DRYRUN.json',result);return result


def execute(limit=None):
    config,examples=check()
    from scripts.cli_support import temporary_api_key
    inputs={x['id']:x for x in rows(HERE/'INPUTS60.frozen.jsonl')}
    blocked={x['payload_sha256'] for x in rows(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl')};current={};count=0
    def event(value):
        with (OUT/'trace.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(utc=utc(),**current,**value),ensure_ascii=False)+'\n')
    with run_lock(OUT),ExitStack() as stack:
        cache=stack.enter_context(Cache(OUT/'checkpoint.sqlite3'));seed=stack.enter_context(closing(seed_db()))
        cache.put('meta','protocol',dict(sha256=sha(HERE/'PROTOCOL.frozen.json')))
        budget=stack.enter_context(closing(prior.NoCapBudget(config,OUT)))
        stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'))
        stack.enter_context(prior.connection_route(urlsplit(config['generation']['base_url']).hostname,event))
        live=stack.enter_context(closing(prior.ConnectionEvidenceAsk(cache,seed,config,budget,event,
            recover_unknown=False,maximum_attempts=5,stop=lambda:(OUT/'STOP_REQUESTED').exists())))
        def ask(stage,prompt,cap):
            if digest(build_payload(config['generation'],prompt,cap)) in blocked:raise RuntimeError('Prior unknown payload held')
            rid,answer,finish=live(stage,prompt,cap)
            receipt=json.loads(cache.db.execute('SELECT result FROM requests WHERE id=?',(rid,)).fetchone()[0])
            assert receipt.get('returned_model')==config['generation']['model'],'Returned model mismatch'
            u=receipt['usage'];assert u['prompt_tokens']+u['completion_tokens']==u['total_tokens']
            return rid,answer,finish
        for task in rows(HERE/'TASKS.frozen.jsonl'):
            if cache.get('prediction',task['key']) is not None:continue
            if (OUT/'STOP_REQUESTED').exists() or (limit is not None and count>=limit):return dict(status='local_limit',new_conditions=count)
            current.clear();current.update(task);r=inputs[task['id']];used={};texts={};stages=[]
            try:
                goalarm='GF' if task['arm']=='R' else task['arm']
                stage='goal/'+goalarm+'|'+task['id'];stages.append(stage)
                rid,goal,_=ask(stage,templates.goal_prompt(r,goalarm),512);used['goal']=rid;texts['goal_text']=goal
                final,_=templates.final_prompt(r,goal,config,examples,task['arm'])
                stage='final/'+task['key'];stages.append(stage)
                rid,prediction,_=ask(stage,final,64);used['final']=rid;texts['prediction']=prediction
                result=dict(**task,status='success',**texts)
            except Exception as exc:
                sql='SELECT q.status,q.result FROM requests q JOIN request_stages s ON s.request_id=q.id WHERE s.stage IN ('+','.join('?' for _ in stages)+') ORDER BY q.rowid DESC LIMIT 1'
                terminal=cache.db.execute(sql,stages).fetchone();receipt=json.loads(terminal[1]) if terminal and terminal[1] else {}
                algorithm=bool(terminal and terminal[0]=='success' and receipt.get('usage') and
                    (receipt.get('finish_reason') in ('length','content_filter') or receipt.get('refusal') or not (receipt.get('answer') or '').strip()))
                result=dict(**task,status='algorithm_failure' if algorithm else 'technical_held',**texts,
                    error_type=type(exc).__name__,error=str(exc)[:300]);result.setdefault('prediction','' if algorithm else None)
                if terminal and terminal[0] in ('unknown','in_flight'):event(dict(event='ambiguous_request_held',automatic_reissue=False))
            result['usage_by_stage']={}
            for stage,rid in used.items():
                receipt=json.loads(cache.db.execute('SELECT result FROM requests WHERE id=?',(rid,)).fetchone()[0])
                result['usage_by_stage'][stage]=dict(request_id=rid,usage=receipt['usage'])
            cache.put('prediction',task['key'],result);event(dict(event='condition_terminal',status=result['status']));count+=1
            print(json.dumps(dict(**task,status=result['status'],new_conditions=count)),flush=True)
    return dict(status='bounded_run_finished',new_conditions=count)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','check','run']);p.add_argument('--limit',type=int);a=p.parse_args()
    print(json.dumps(prepare() if a.mode=='prepare' else dry_run() if a.mode=='check' else execute(a.limit),ensure_ascii=True),flush=True)
