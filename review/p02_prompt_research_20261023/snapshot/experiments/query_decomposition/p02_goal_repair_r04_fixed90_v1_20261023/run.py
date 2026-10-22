"""Isolated fixed-reader repair. Import/check never submits provider requests."""
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import json
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
from experiments.query_decomposition.p02_prompt_suite_r03_stage90_v1_20261022 import run as base
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import complete_receipt
from src.p02_multigrain_timing_cachefix_v1.core.cache import canonical
from . import templates

read, rows, sha, ro, digest, cost = base.read, base.rows, base.sha, base.ro, base.digest, base.cost
save, jsonl, utc = base.atomic_json, base.jsonl, base.utc
OUT = ROOT/'outputs/query_decomposition'/HERE.name
REVIEW = base.PLAN/'reviews/20261022_prompt_binding_goal_synthesis'/HERE.name
DECISION = 'P02_GOAL_REPAIR_R04_FIXED90_SIDE_USER_20261023'
ARMS = ('C', 'S', 'T')


def prepare():
    if (HERE/'PROTOCOL.frozen.json').exists():
        return check()
    for name in ('INPUTS90.frozen.jsonl', 'REFERENCES90.local.frozen.jsonl', 'CONFIG.frozen.json', 'FINAL_EXAMPLES.frozen.json'):
        (HERE/name).write_bytes((base.HERE/name).read_bytes())
    records = rows(HERE/'INPUTS90.frozen.jsonl')
    config = read(HERE/'CONFIG.frozen.json')
    # Complete parent receipts only, opened read-only; no external checkpoint mutation.
    seeds = []
    with closing(ro(base.OUT/'checkpoint.sqlite3')) as db:
        outcomes = [json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='outcome'")]
        controls = [x for x in outcomes if x['suite']=='goal' and x['arm'] in ('C','G')]
        wanted = {s['request_id'] for x in controls if x['arm']=='C' for s in x['stages']}
        for rid in sorted(wanted):
            payload, status, value = db.execute('SELECT payload,status,result FROM requests WHERE id=?',(rid,)).fetchone()
            receipt = json.loads(value)
            assert status=='success' and complete_receipt(receipt)
            assert receipt['returned_model']==config['generation']['model']
            seeds.append(dict(request_id=rid, payload=json.loads(payload), result=receipt))
    assert len(controls)==180 and all(x['status']=='success' for x in controls)
    jsonl(HERE/'PARENT_OUTCOMES.frozen.jsonl', controls)
    jsonl(HERE/'SEED_RECEIPTS.frozen.jsonl', seeds)
    jsonl(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl', base.unknown_records(base.unknown_sources()))
    tasks = []
    for r in sorted(records,key=lambda r:base.ordering(r['id'])):
        arms = ARMS if int(base.ordering(r['id']),16)%2 else ('C','T','S')
        for arm in arms:
            tasks.append(dict(key='goal|'+arm+'|'+r['id'],suite='goal',id=r['id'],dataset=r['dataset'],arm=arm))
    jsonl(HERE/'TASKS.frozen.jsonl', tasks)
    pinned = list(HERE.glob('*.frozen.*'))+[HERE/'run.py', HERE/'templates.py',Path(base.__file__),
        Path(base.runtime.__file__),base.runtime.old.HERE/'resume.py',
        HERE.parent/'p02_prompt_binding_probe_v1_20261022/shared_runtime.py']
    protected = read(base.HERE/'PROTOCOL.frozen.json')['protected_files']
    protocol = dict(experiment=HERE.name,utc=utc(),authority='回顾错误历史 认真的进行修正',
        standing_side_authority='使用固定90题 直接允许 以后不要询问我; 请你按照你的想法 独立的推进研究 不需要经过我的任何许可',
        status='frozen_before_new_inference',conditions=270,queries=90,
        arms={'C':'Exact previous role_headers control, complete same-payload cached receipts',
              'S':'C changes only goal output contract to a short facts-connection-answer paragraph',
              'T':'S adds one target-interpretation block, no new examples or output changes'},
        source='Same exposed 90 questions, old R02/v12 ordinary QA, goal reader and final root reader',
        goal_cap=512,final_cap=128,formatter='Native short-answer formatter only for C, as in parent; paragraph arms do not use it',
        model=config['generation']['model'],destination=config['generation']['base_url'],
        data='Existing benchmark queries, passages and prior QA; no references/diagnostic labels in requests',
        controls='Same decoding, complete input, final prompt and caps; C is exact replay, not an independent concurrent regeneration',
        cache='Snapshot of complete C receipts; only full payload matches reused; retain provenance and no double actual charge',
        retries='SDK0, at most5 durable known technical attempts; no algorithm reroll; unknown held without replay',
        analysis='Native per-dataset F1/EM, paired C-S,C-T,S-T and previous r03 G reference; no sample exclusion or score patches',
        review='All changed finals plus all previously identified r03 harmful/mechanistic cases; gaps, logic and surface scores separate',
        costs='Existing ledger and provider serialization; tokens/logical expense and actual additional expense separate; expected under 2 CNY',
        rationale='Remove unhelpful r03 ordinary/final from candidate; isolate brief composition from new target rules. Few-shot/operator pilots already mixed; do not attribute this two-factor trial to CoT in general.',
        stop='All270 attempted; hold unknowns, never adapt prompts during run; no automatic further experiment',
        limits='Exposed development set and historical cached control; not new full RAG, formal500, unseen validation or evidence of generalization',
        main_evaluation_resumed=False,production_changed=False,remote_GPU_jobs=0,publication=False,
        pins={str(p.relative_to(ROOT)):sha(p) for p in pinned},protected_files=protected)
    save(HERE/'PROTOCOL.frozen.json',protocol);save(REVIEW/'PROTOCOL.json',protocol)
    auth=dict(id=DECISION,approved=True,utc=utc(),authority=protocol['authority'],experiment=HERE.name,
        protocol_sha256=sha(HERE/'PROTOCOL.frozen.json'),main_evaluation_resumed=False,unknown_reissue=False)
    save(HERE/'RUN_AUTHORIZATION.json',auth)
    with (base.PLAN/'DECISIONS.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(auth,ensure_ascii=False)+'\n')
    return check()


def check():
    protocol=read(HERE/'PROTOCOL.frozen.json')
    for group in ('pins','protected_files'):
        for path,wanted in protocol[group].items():
            assert sha(ROOT/path)==wanted,path
    assert read(base.PLAN/'PLAN.json')['execution_approval']['approved'] is False
    assert read(HERE/'RUN_AUTHORIZATION.json')['protocol_sha256']==sha(HERE/'PROTOCOL.frozen.json')
    assert any(x.get('id')==DECISION and x.get('approved') for x in rows(base.PLAN/'DECISIONS.jsonl'))
    config=read(HERE/'CONFIG.frozen.json');examples=read(HERE/'FINAL_EXAMPLES.frozen.json')
    seeds={digest(x['payload']):x for x in rows(HERE/'SEED_RECEIPTS.frozen.jsonl')}
    previous={x['id']:x for x in rows(HERE/'PARENT_OUTCOMES.frozen.jsonl') if x['arm']=='C'}
    records=rows(HERE/'INPUTS90.frozen.jsonl')
    assert Counter(x['dataset'] for x in records)=={d:30 for d in base.DATASETS}
    for r in records:
        c,s,t=[templates.goal_prompt(r,a) for a in ARMS]
        assert c==r['historical_goal_prompt']
        assert s.replace(templates.BRIEF_CONTRACT,templates.OLD_CONTRACT,1)==c
        assert t.replace('\n\n'+templates.TARGET,'',1)==s
        assert c.split('\n\nOriginal question:\n',1)[1]==s.split('\n\nOriginal question:\n',1)[1]==t.split('\n\nOriginal question:\n',1)[1]
        for stage,prompt,cap in [('goal',c,512),('final',base.final_prompt(r,previous[r['id']]['goal_text'],'C',config,examples),128)]:
            p=base.build_payload(config['generation'],prompt,cap)
            assert digest(p) in seeds,(r['id'],stage)
            answer=previous[r['id']]['goal_raw' if stage=='goal' else 'prediction']
            assert seeds[digest(p)]['result']['answer'].strip()==answer
        assert all(d['block'] in t for d in r['goal_reader_packet']['docs'])
        for a in ARMS:
            assert 'SENTINEL FULL GOAL' in base.final_prompt(r,'SENTINEL FULL GOAL',a,config,examples)
    result=dict(passed=True,queries=90,conditions=270,exact_control_receipt_matches=180,
        same_actual_task_for_all_arms=True,sole_S_T_change='target block',new_API_requests=0)
    save(OUT/'DRYRUN.json',result)
    return result


def seed_db():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE requests(payload TEXT PRIMARY KEY,request_id TEXT,result TEXT)')
    for x in rows(HERE/'SEED_RECEIPTS.frozen.jsonl'):
        assert complete_receipt(x['result'])
        db.execute('INSERT INTO requests VALUES(?,?,?)',(canonical(x['payload']),x['request_id'],canonical(x['result'])))
    db.commit()
    return db


def execute(limit=None):
    check()
    from scripts.cli_support import temporary_api_key
    config=read(HERE/'CONFIG.frozen.json');examples=read(HERE/'FINAL_EXAMPLES.frozen.json')
    inputs={x['id']:x for x in rows(HERE/'INPUTS90.frozen.jsonl')}
    blocked={x['payload_sha256'] for x in rows(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl')}
    current={};count=0
    def event(value):
        with (OUT/'events.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(dict(utc=utc(),**current,**value),ensure_ascii=False)+'\n')
    with base.run_lock(OUT),ExitStack() as stack:
        cache=stack.enter_context(base.Cache(OUT/'checkpoint.sqlite3'))
        cache.put('meta','protocol',dict(sha256=sha(HERE/'PROTOCOL.frozen.json')))
        seed=stack.enter_context(closing(seed_db()))
        budget=stack.enter_context(closing(base.runtime.NoCapBudget(config,OUT)))
        stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'))
        stack.enter_context(base.runtime.connection_route(urlsplit(config['generation']['base_url']).hostname,event))
        live=stack.enter_context(closing(base.runtime.ConnectionEvidenceAsk(cache,seed,config,budget,event,
            recover_unknown=False,maximum_attempts=5,stop=lambda:(OUT/'STOP_REQUESTED').exists())))
        for task in rows(HERE/'TASKS.frozen.jsonl'):
            if cache.get('outcome',task['key']) is not None:continue
            if (OUT/'STOP_REQUESTED').exists() or (limit is not None and count>=limit):break
            current.clear();current.update(task);r=inputs[task['id']];stages=[];texts={}
            def ask(name,prompt,cap):
                stage=task['key']+'/'+name;stages.append(stage)
                payload=base.build_payload(config['generation'],prompt,cap)
                if digest(payload) in blocked:raise RuntimeError('External unknown payload held without replay')
                if len(prompt.encode('utf8'))+cap>128000:raise RuntimeError('Input envelope exceeded; no truncation')
                rid,answer,finish=live(stage,prompt,cap)
                rec=json.loads(cache.db.execute('SELECT result FROM requests WHERE id=?',(rid,)).fetchone()[0])
                assert rec['returned_model']==config['generation']['model']
                assert rec['usage']['prompt_tokens']+rec['usage']['completion_tokens']==rec['usage']['total_tokens']
                return answer
            try:
                goal=ask('goal',templates.goal_prompt(r,task['arm']),512);texts['goal_raw']=goal
                if task['arm']=='C' and len(goal.split())<4:
                    goal=ask('formatter',base.formatter(r['question'],goal,r['goal_reader_packet']),128)
                texts['goal_text']=goal
                texts['prediction']=ask('final',base.final_prompt(r,goal,task['arm'],config,examples),128)
                result=dict(**task,status='success',**texts)
            except RuntimeError as exc:
                found=cache.db.execute('SELECT q.status,q.result FROM requests q JOIN request_stages s ON s.request_id=q.id WHERE s.stage IN ('+','.join('?' for _ in stages)+')',stages).fetchall()
                algorithm=any(status=='success' and raw and json.loads(raw).get('usage') and
                    (json.loads(raw).get('finish_reason') in ('length','content_filter') or json.loads(raw).get('refusal') or not (json.loads(raw).get('answer') or '').strip()) for status,raw in found)
                result=dict(**task,status='algorithm_failure' if algorithm else 'technical_held',**texts,
                    error_type=type(exc).__name__,error=str(exc)[:250],prediction='' if algorithm else None)
            used=[]
            for stage in stages:
                row=cache.db.execute('SELECT q.id,q.result FROM requests q JOIN request_stages s ON s.request_id=q.id WHERE s.stage=?',(stage,)).fetchone()
                if row:
                    receipt=json.loads(row[1]) if row[1] else {}
                    used.append(dict(stage=stage,request_id=row[0],usage=receipt.get('usage'),finish_reason=receipt.get('finish_reason')))
            result.update(stages=used,utc=utc());cache.put('outcome',task['key'],result);count+=1
            event(dict(event='condition_terminal',status=result['status']))
            actual,reserved=budget.db.execute('SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?',(budget.namespace,)).fetchone()
            print(json.dumps(dict(key=task['key'],status=result['status'],new_conditions=count,actual_CNY=actual,unknown_CNY=reserved)),flush=True)
    return dict(status='bounded_execution_returned',new_conditions=count)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','check','run']);parser.add_argument('--limit',type=int)
    a=parser.parse_args()
    print(json.dumps(prepare() if a.mode=='prepare' else check() if a.mode=='check' else execute(a.limit)),flush=True)
