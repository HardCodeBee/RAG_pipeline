"""Isolated r03 fixed-reader diagnosis. Paid execution only in explicit run mode."""
from collections import Counter
from contextlib import ExitStack,closing
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import hashlib
import json
import sqlite3
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.query_decomposition.p02_prompt_suite_design_v1_20261022 import design_tools as design
from experiments.query_decomposition.p02_result_binding_operator_probe_v1_20261022 import run as runtime
from src.p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline,passages,render_final
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import atomic_json

read,rows,sha,ro,digest,cost=runtime.read,runtime.rows,runtime.sha,runtime.ro,runtime.digest,runtime.cost
Cache,run_lock,utc,build_payload=runtime.Cache,runtime.run_lock,runtime.utc,runtime.build_payload
OUT=ROOT/'outputs/query_decomposition'/HERE.name
PLAN=HERE.parent/'unified_rag_eval_v1_20261021'
REVIEW=PLAN/'reviews/20261022_prompt_binding_goal_synthesis'/HERE.name
SOURCE=runtime.HERE
DESIGN=design.HERE
DECISION='P02_PROMPT_SUITE_R03_STAGE90_USER_TRIAL_20261022'
DATASETS=('hotpotqa','2wiki','musique')
GOAL_ARMS=('C','G','GF');ORDINARY_ARMS=('O0','O1')


def jsonl(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values),encoding='utf8')


def history(nodes):
    return [dict(step_id=n['step_id'],question=n['request'],answer=n['answer']) for n in nodes]


def packet_blocks(packet):return [d['block'] for d in packet['docs']]


def ordering(ident):return hashlib.sha256(('p02-r03-stage90|20261022|'+ident).encode()).hexdigest()


def unknown_sources():
    parent=ROOT/'outputs/query_decomposition'
    paths=set(parent.glob('p02*/checkpoint.sqlite3'))
    paths.update(parent.glob('p02*/*/checkpoint.sqlite3'))
    return sorted(p for p in paths if p!=OUT/'checkpoint.sqlite3')


def unknown_records(paths):
    result=[]
    for path in paths:
        with closing(ro(path)) as db:
            if not db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='requests'").fetchone():continue
            for rid,payload,status in db.execute("SELECT id,payload,status FROM requests WHERE status IN ('unknown','in_flight')"):
                result.append(dict(source=str(path.relative_to(ROOT)),request_id=rid,status=status,
                                   payload_sha256=digest(json.loads(payload)),automatic_reissue=False))
    return result


def ordinary_prompt(r,arm):
    if arm=='O0':return r['control_prompt']
    return design.render('r03','ordinary',dict(original_question=r['original_question'],
      history=history(r['history']),current_question=r['current_question'],
      retrieval_question=r['accepted_rewrite'],passages=packet_blocks(r['reader_packet'])))


def goal_prompt(r,arm):
    if arm=='C':return r['historical_goal_prompt']
    return design.render('r03','goal',dict(original_question=r['question'],history=history(r['ordinary_nodes']),
      retrieval_question=r['goal_rewritten_query'],passages=packet_blocks(r['goal_reader_packet'])))


def final_prompt(r,goal,arm,config,examples):
    if arm=='GF':
        return design.render('r03','final',dict(original_question=r['question'],history=history(r['ordinary_nodes']),
          synthesis=design.parse('goal',goal),passages=packet_blocks(r['final_root_packet'])))
    nodes=deepcopy(r['ordinary_nodes'])+[dict(step_id=r['goal_step_id'],request=r['question'],answer=goal)]
    return render_final(r['question'],nodes,r['final_root_packet'],config['readout'],examples)[0]


def formatter(question,answer,packet):
    return ('Express the candidate answer below as a concise complete factual sentence that names its subject and relation. '
      'Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. '
      'If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n'
      +'Question:\n'+question+'\n\nCandidate answer:\n'+answer+'\n\nContext:\n'+passages(packet))


def prepare():
    if (HERE/'PROTOCOL.frozen.json').exists():return check()
    assert read(DESIGN/'SESSION.json')['status']=='design_complete_not_model_validated'
    config,examples=runtime.old.pilot.settings()
    goals=rows(SOURCE/'INPUTS90.frozen.jsonl')
    historical={r['id']:r for r in rows(SOURCE/'HISTORY90.local.frozen.jsonl')}
    assert len(goals)==90 and Counter(r['dataset'] for r in goals)=={d:30 for d in DATASETS}
    ordinary=[];bindings=[]
    for r in goals:
        path=ROOT/historical[r['id']]['source'];assert sha(path)==r['source_sha256'],r['id']
        raw=read(path);nodes=raw['nodes'];wanted=r['ordinary_nodes']
        assert [dict(step_id=n['step_id'],request=n['request'],answer=n['answer']) for n in nodes[:-1]]==wanted
        for n in nodes[:-1]:
            ordinary.append(dict(id=r['id']+'/ordinary/'+str(n['step_id']),case_id=r['id'],qid=r['qid'],dataset=r['dataset'],
              step_id=n['step_id'],original_question=r['question'],current_question=n['request'],history=n['prior_complete_answers'],
              reader_packet=n['reader_packet'],accepted_rewrite=n['rewritten_query'],control_prompt=n['child_prompt'],
              historical_initial_answer=n['initial_answer'],historical_accepted_answer=n['answer']))
        bindings.append(dict(id=r['id'],source=str(path.relative_to(ROOT)),sha256=sha(path)))
    assert len(ordinary)==176
    jsonl(HERE/'INPUTS90.frozen.jsonl',goals);jsonl(HERE/'ORDINARY176.frozen.jsonl',ordinary)
    jsonl(HERE/'HISTORY90.local.frozen.jsonl',list(historical.values()))
    jsonl(HERE/'REFERENCES90.local.frozen.jsonl',rows(SOURCE/'REFERENCES90.local.frozen.jsonl'))
    jsonl(HERE/'SOURCE_BINDINGS.frozen.jsonl',bindings)
    atomic_json(HERE/'CONFIG.frozen.json',config);atomic_json(HERE/'FINAL_EXAMPLES.frozen.json',examples)
    initial=[x['id'] for ds in DATASETS for x in sorted((r for r in goals if r['dataset']==ds),key=lambda r:ordering(r['id']))[:2]]
    arranged=sorted(goals,key=lambda r:(r['id'] not in initial,ordering(r['id'])))
    tasks=[]
    for suite,records,arms in [('goal',arranged,GOAL_ARMS),('ordinary',sorted(ordinary,key=lambda r:ordering(r['id'])),ORDINARY_ARMS)]:
        for r in records:
            offset=int(ordering(r['id']),16)%len(arms)
            for arm in arms[offset:]+arms[:offset]:
                tasks.append(dict(key=suite+'|'+arm+'|'+r['id'],suite=suite,arm=arm,id=r['id'],dataset=r['dataset'],engineering_initial=r['id'] in initial))
    assert len(tasks)==622
    jsonl(HERE/'TASKS.frozen.jsonl',tasks)
    external=unknown_records(unknown_sources());jsonl(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl',external)
    protected=read(DESIGN/'SESSION.json')['source_core_sha256']
    pins={str(p.relative_to(ROOT)):sha(p) for p in list(HERE.glob('*.frozen.*'))+[Path(__file__),DESIGN/'design_tools.py',
      *list((DESIGN/'revisions/r03').glob('*.txt')),DESIGN/'revisions/r03/examples.json',
      runtime.old.HERE/'resume.py',SOURCE/'run.py',
      HERE.parent/'p02_prompt_binding_probe_v1_20261022/shared_runtime.py',
      ROOT/'scripts/cli_support.py',ROOT/'src/p02_multigrain_timing_cachefix_v1/core/llm.py',
      ROOT/'src/p02_multigrain_timing_cachefix_v1/core/cache.py']}
    p=dict(experiment=HERE.name,utc=utc(),authority='进行试验尝试效果',
      standing_side_authorization='使用固定90题 直接允许 以后不要询问我; 请你按照你的想法 独立的推进研究 不需要经过我的任何许可',
      scope='First fixed-reader stage diagnosis for r03; NOT full dynamic RAG. No new retrieval, rewrite inference or decomposer training.',
      candidate='Exact r03 design, no adaptive prompt edits after observing outcomes',goal_queries=90,ordinary_nodes=176,
      planned_conditions=622,goal_arms={'C':'old goal and old final','G':'new r03 goal and old final','GF':'same new r03 goal and new r03 final'},
      ordinary_arms={'O0':'old ordinary and native short-answer formatter','O1':'new r03 Support+Answer, no formatter'},
      generation_caps={'ordinary':256,'goal':512,'final':128,'formatter':128},
      controls='Same model/decoding/caps; complete stop plus usage; old final length exception not accepted. History and packets frozen for every node.',
      final_input='Original-query root only; full goal text; no additional goal raw passages',
      model=config['generation']['model'],destination=config['generation']['base_url'],
      data='Existing benchmark questions, frozen retrieved documents and prior QA, synthetic demonstrations; no reference answers, labels or ledger in requests',
      cache='Fresh namespace; no historical success seeds; strict exact-payload current-run reuse only. G and GF deliberately share the same goal output.',
      retry='At most5 durable known technical attempts, no SDK retry, unknown remains held, algorithm/format failures not rerolled',
      scoring='Native F1/EM for goal-final flows, paired by query; ordinary answers retained for factual and information-sufficiency review, never use root gold to grade child answers',
      costs='Shared provider gate and durable existing ledger; logical stage tokens/cost and new actual expense separately; retain unknown reservations',
      engineering_initial_ids=initial,engineering_gate='First18 goal conditions validate structural handoffs only, not F1 selection',
      expected_prompt_size='Input UTF8bytes + output cap guarded <=128000 conservative units, no silent truncation; not a model token count',
      limits='Repeatedly exposed development90. Stage effects conditional on old upstream and packets; no claim of whole-suite or dynamic retrieval improvement.',
      main_evaluation_resumed=False,remote_GPU_jobs=0,publication=False,pins=pins,protected_files=protected)
    atomic_json(HERE/'PROTOCOL.frozen.json',p);atomic_json(REVIEW/'PROTOCOL.json',p)
    auth=dict(id=DECISION,approved=True,utc=utc(),authority=p['authority'],experiment=HERE.name,
      protocol_sha256=sha(HERE/'PROTOCOL.frozen.json'),conditions=622,main_evaluation_resumed=False,unknown_reissue=False)
    atomic_json(HERE/'RUN_AUTHORIZATION.json',auth)
    with (PLAN/'DECISIONS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(auth,ensure_ascii=False)+'\n')
    return check()


def check():
    p=read(HERE/'PROTOCOL.frozen.json')
    for group in ('pins','protected_files'):
        for path,wanted in p[group].items():assert sha(ROOT/path)==wanted,path
    auth=read(HERE/'RUN_AUTHORIZATION.json');assert auth['approved'] and auth['protocol_sha256']==sha(HERE/'PROTOCOL.frozen.json')
    assert any(x.get('id')==DECISION and x.get('protocol_sha256')==auth['protocol_sha256'] for x in rows(PLAN/'DECISIONS.jsonl'))
    config=read(HERE/'CONFIG.frozen.json');examples=read(HERE/'FINAL_EXAMPLES.frozen.json')
    assert config['generation']==read(SOURCE/'GENERATION.frozen.json')
    for r in rows(HERE/'INPUTS90.frozen.jsonl'):
        assert goal_prompt(r,'G')==goal_prompt(r,'GF')
        assert all(n['answer'] in goal_prompt(r,'GF') for n in r['ordinary_nodes'])
        fake='Facts: TEST FACT\nLink: TEST CONNECTION\nCandidate: TEST ANSWER'
        for arm in GOAL_ARMS:
            prompt=final_prompt(r,fake,arm,config,examples)
            assert fake in prompt
            assert all(d['block'] in prompt for d in r['final_root_packet']['docs'])
    for r in rows(HERE/'ORDINARY176.frozen.jsonl'):
        for arm in ORDINARY_ARMS:
            prompt=ordinary_prompt(r,arm);assert r['current_question'] in prompt
            assert all(d['block'] in prompt for d in r['reader_packet']['docs'])
    result=dict(status='prepared_and_checked',queries=90,ordinary_nodes=176,conditions=622,API_requests=0,main_evaluation_paused=True)
    atomic_json(OUT/'DRYRUN.json',result);return result


class FormatFailure(ValueError):pass


def run(limit=None):
    check();config=read(HERE/'CONFIG.frozen.json');examples=read(HERE/'FINAL_EXAMPLES.frozen.json')
    from scripts.cli_support import temporary_api_key
    inputs={'goal':{r['id']:r for r in rows(HERE/'INPUTS90.frozen.jsonl')},
            'ordinary':{r['id']:r for r in rows(HERE/'ORDINARY176.frozen.jsonl')}}
    tasks=rows(HERE/'TASKS.frozen.jsonl');current={};count=0
    blocked={r['payload_sha256'] for r in rows(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl')}
    def event(value):
        with (OUT/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(utc=utc(),**current,**value),ensure_ascii=False)+'\n')
    with run_lock(OUT),ExitStack() as stack:
        cache=stack.enter_context(Cache(OUT/'checkpoint.sqlite3'))
        cache.put('meta','protocol',dict(sha256=sha(HERE/'PROTOCOL.frozen.json')))
        budget=stack.enter_context(closing(runtime.NoCapBudget(config,OUT)))
        stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'))
        stack.enter_context(runtime.connection_route(urlsplit(config['generation']['base_url']).hostname,event))
        live=stack.enter_context(closing(runtime.ConnectionEvidenceAsk(cache,None,config,budget,event,
          recover_unknown=False,maximum_attempts=5,stop=lambda:(OUT/'STOP_REQUESTED').exists())))
        for task in tasks:
            if cache.get('outcome',task['key']) is not None:continue
            if (OUT/'STOP_REQUESTED').exists() or (limit is not None and count>=limit):break
            current.clear();current.update(task);r=inputs[task['suite']][task['id']];stages=[];texts={}
            def ask(name,prompt,cap):
                stage=task['key']+'/'+name;stages.append(stage)
                payload=build_payload(config['generation'],prompt,cap)
                if digest(payload) in blocked:raise RuntimeError('Prior unknown payload held without replay')
                if len(prompt.encode('utf8'))+cap>128000:raise RuntimeError('Conservative input envelope exceeded; no truncation')
                rid,answer,finish=live(stage,prompt,cap)
                receipt=json.loads(cache.db.execute('SELECT result FROM requests WHERE id=?',(rid,)).fetchone()[0])
                assert receipt.get('returned_model')==config['generation']['model']
                u=receipt['usage'];assert u['prompt_tokens']+u['completion_tokens']==u['total_tokens']
                return answer
            try:
                if task['suite']=='ordinary':
                    raw=ask('ordinary',ordinary_prompt(r,task['arm']),256);texts['raw_output']=raw
                    if task['arm']=='O1':
                        try:parsed=design.parse('ordinary',raw)
                        except ValueError as e:raise FormatFailure(str(e))
                        texts.update(support=parsed['Support'],answer=parsed['Answer'])
                    else:
                        texts.update(support=None,answer=raw)
                        if len(raw.split())<4:texts['answer']=ask('formatter',formatter(r['current_question'],raw,r['reader_packet']),128)
                else:
                    raw=ask('goal',goal_prompt(r,task['arm']),512);texts['goal_raw']=raw
                    if task['arm']=='C':
                        if len(raw.split())<4:raw=ask('formatter',formatter(r['question'],raw,r['goal_reader_packet']),128)
                    else:
                        try:design.parse('goal',raw)
                        except ValueError as e:raise FormatFailure(str(e))
                    texts['goal_text']=raw
                    texts['prediction']=ask('final',final_prompt(r,raw,task['arm'],config,examples),128)
                result=dict(**task,status='success',**texts)
            except (RuntimeError,FormatFailure) as exc:
                request_rows=cache.db.execute('SELECT q.status,q.result FROM requests q JOIN request_stages s ON s.request_id=q.id WHERE s.stage IN ('+','.join('?' for _ in stages)+')',stages).fetchall() if stages else []
                algorithm=isinstance(exc,FormatFailure) or any(s=='success' and raw and json.loads(raw).get('usage') and
                  (json.loads(raw).get('finish_reason') in ('length','content_filter') or json.loads(raw).get('refusal') or not (json.loads(raw).get('answer') or '').strip()) for s,raw in request_rows)
                result=dict(**task,status='algorithm_failure' if algorithm else 'technical_held',**texts,error_type=type(exc).__name__,error=str(exc)[:250])
                if task['suite']=='goal':result.setdefault('prediction','' if algorithm else None)
            used=[]
            for stage in stages:
                row=cache.db.execute('SELECT q.id,q.result FROM requests q JOIN request_stages s ON s.request_id=q.id WHERE s.stage=?',(stage,)).fetchone()
                if row:
                    rec=json.loads(row[1]) if row[1] else {}
                    used.append(dict(stage=stage,request_id=row[0],usage=rec.get('usage'),finish_reason=rec.get('finish_reason')))
            result['stages']=used;result['utc']=utc();cache.put('outcome',task['key'],result);count+=1
            event(dict(event='condition_terminal',status=result['status']))
            summary=Counter(x['status'] for x in cache.all('outcome').values())
            actual,reserved=budget.db.execute('SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?',(budget.namespace,)).fetchone()
            atomic_json(OUT/'STATUS.json',dict(utc=utc(),planned_conditions=622,terminal=sum(summary.values()),statuses=dict(summary),
              actual_CNY=actual,unresolved_reserved_CNY=reserved,last_task=task,main_evaluation_paused=True))
            print(json.dumps(dict(key=task['key'],status=result['status'],terminal=sum(summary.values()),actual_CNY=actual)),flush=True)
    return dict(status='bounded_execution_returned',new_conditions=count)


def report():
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    scorers={ds:metric(ds) for ds in DATASETS};refs={r['id']:r for r in rows(HERE/'REFERENCES90.local.frozen.jsonl')}
    with closing(ro(OUT/'checkpoint.sqlite3')) as db:outcomes=[json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='outcome'")]
    for r in outcomes:
        if r['suite']=='goal':r['metrics']=scorers[r['dataset']](r['prediction'],refs[r['id']]) if r['status'] in ('success','algorithm_failure') else dict(f1=None,em=None)
        us=[x['usage'] for x in r['stages'] if x.get('usage')]
        r['input_tokens']=sum(x['prompt_tokens'] for x in us);r['output_tokens']=sum(x['completion_tokens'] for x in us)
        r['logical_stage_CNY']=sum(cost(x) for x in us)
    goals=[r for r in outcomes if r['suite']=='goal'];index={(r['arm'],r['id']):r for r in goals}
    data=rows(HERE/'INPUTS90.frozen.jsonl');summary={};paired={}
    for ds in ('all',)+DATASETS:
        ids=[r['id'] for r in data if ds=='all' or r['dataset']==ds];summary[ds]={};paired[ds]={}
        for arm in GOAL_ARMS:
            available=[index[(arm,i)] for i in ids if (arm,i) in index];valid=[r for r in available if r['metrics']['f1'] is not None]
            summary[ds][arm]=dict(planned=len(ids),terminal=len(available),scorable=len(valid),
              F1=sum(r['metrics']['f1'] for r in valid)/len(ids) if len(valid)==len(ids) else None,
              EM=sum(r['metrics']['em'] for r in valid)/len(ids) if len(valid)==len(ids) else None,
              available_only_F1=sum(r['metrics']['f1'] for r in valid)/len(valid) if valid else None,
              mean_input_tokens=sum(r['input_tokens'] for r in valid)/len(valid) if valid else None,
              mean_output_tokens=sum(r['output_tokens'] for r in valid)/len(valid) if valid else None,
              mean_goal_final_CNY=sum(r['logical_stage_CNY'] for r in valid)/len(valid) if valid else None,
              statuses=dict(Counter(r['status'] for r in available)))
        for a,b in (('C','G'),('C','GF'),('G','GF')):
            pairs=[(index[(a,i)],index[(b,i)]) for i in ids if all((arm,i) in index and index[(arm,i)]['metrics']['f1'] is not None for arm in (a,b))]
            diffs=[y['metrics']['f1']-x['metrics']['f1'] for x,y in pairs]
            paired[ds][a+'->'+b]=dict(n=len(pairs),F1_delta=sum(diffs)/len(diffs) if diffs else None,
              improved=sum(v>1e-12 for v in diffs),harmed=sum(v< -1e-12 for v in diffs),tied=sum(abs(v)<=1e-12 for v in diffs))
    config=read(HERE/'CONFIG.frozen.json');namespace='p02_prompt_research_v1_20261022/'+str(OUT.resolve())
    with closing(ro(config['budget']['ledger'])) as db:
        actual,reserved=db.execute('SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?',(namespace,)).fetchone()
    result=dict(utc=utc(),status='all_conditions_attempted' if len(outcomes)==622 else 'partial',
      planned_conditions=622,terminal_conditions=len(outcomes),by_dataset_arm=summary,paired=paired,
      ordinary_conditions=sum(r['suite']=='ordinary' for r in outcomes),
      ordinary_statuses=dict(Counter(r['status'] for r in outcomes if r['suite']=='ordinary')),
      actual_CNY=actual,unresolved_reserved_CNY=reserved,not_dynamic_full_RAG=True,main_evaluation_paused=True)
    jsonl(OUT/'DETAILS.local.jsonl',outcomes);atomic_json(OUT/'REPORT.json',result);atomic_json(REVIEW/'RESULTS.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','check','run','report']);p.add_argument('--limit',type=int);a=p.parse_args()
    value=prepare() if a.mode=='prepare' else check() if a.mode=='check' else run(a.limit) if a.mode=='run' else report()
    print(json.dumps(value,ensure_ascii=True),flush=True)
