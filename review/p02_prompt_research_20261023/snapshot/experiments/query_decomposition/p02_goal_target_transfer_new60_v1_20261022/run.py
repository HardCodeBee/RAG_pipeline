"""Isolated disjoint-cohort goal/final study; no live retrieval or main resume."""
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import hashlib
import json
import math
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
from . import templates
from experiments.query_decomposition.p02_result_binding_operator_probe_v1_20261022 import run as prev
from src.p02_multigrain_timing_cachefix_v1.core.pipeline import passages

read, rows, save, jsonl, sha, ro, digest, cost = prev.read, prev.rows, prev.save, prev.jsonl, prev.sha, prev.ro, prev.digest, prev.cost
Cache, run_lock, utc, build_payload = prev.Cache, prev.run_lock, prev.utc, prev.build_payload
OUT = ROOT/'outputs/query_decomposition'/HERE.name
PLAN = prev.PLAN
REVIEW = PLAN/'reviews/20261022_prompt_binding_goal_synthesis'/HERE.name
DECISION = 'P02_GOAL_TARGET_TRANSFER_NEW60_SIDE_AUTONOMOUS_20261022'
DATASETS = ('hotpotqa','2wiki','musique')
SEED = 'p02-goal-target-transfer-new60-20261022-v1'


def norm(value):
    return ' '.join(re.findall(r'\w+', value.casefold()))


def order(value):
    return hashlib.sha256((SEED+'|'+value).encode()).hexdigest()


def exclusions():
    excluded, inventory = set(), []
    for directory in sorted(HERE.parent.glob('p02*')):
        if directory == HERE or not directory.is_dir() or not any(w in directory.name for w in
                ('probe','fewshot','binding','quick90','shared_context','balanced','local_cot')):
            continue
        for path in sorted(directory.rglob('*.jsonl')):
            if not any(w in path.name.lower() for w in ('input','probe','runtime90','questions','selection','ordinary38')):
                continue
            if any(w in str(path.relative_to(directory)).lower() for w in ('review','annotation','label','history','reference')):
                continue
            ids = set()
            for row in rows(path):
                for k in ('id','case_id','probe_id'):
                    value = row.get(k,'')
                    if isinstance(value,str) and value.startswith(tuple(d+':' for d in DATASETS)):
                        ids.add(value.split('/')[0]); break
            if ids:
                excluded.update(ids)
                inventory.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),ids=sorted(ids)))
    return excluded, inventory


def freeze_cohort():
    path = HERE/'COHORT.frozen.json'
    if path.exists():
        return rows(HERE/'RUNTIME60.frozen.jsonl')
    source = ROOT/'src/p02_multigrain_timing_cachefix_v1/data/runtime300.jsonl'
    runtime = rows(source)
    excluded, inventory = exclusions()
    excluded_texts = {norm(r['question']) for r in runtime if r['id'] in excluded}
    eligible = [r for r in runtime if r['id'] not in excluded and norm(r['question']) not in excluded_texts
                and bool(r['supporting_questions'])]
    # Deduplicate normalized question text before sampling, never by outcome.
    unique = {}
    for r in sorted(eligible,key=lambda r:order(r['id'])):
        unique.setdefault(norm(r['question']),r)
    eligible = list(unique.values())
    selected, allocation = [], {}
    for ds in DATASETS:
        groups = defaultdict(list)
        for r in eligible:
            if r['dataset']==ds: groups[r['type']].append(r)
        total = sum(map(len,groups.values())); assert total >=20, (ds,total)
        quotas = {k:math.floor(20*len(v)/total) for k,v in groups.items()}
        remaining = 20-sum(quotas.values())
        for k in sorted(groups,key=lambda k:(-(20*len(groups[k])/total-quotas[k]),order(ds+'|'+k)))[:remaining]:
            quotas[k]+=1
        allocation[ds]=dict(eligible=total,quotas=quotas)
        for k,n in quotas.items():
            selected.extend(sorted(groups[k],key=lambda r:order(r['id']))[:n])
    selected.sort(key=lambda r:order(r['id']))
    assert len(selected)==len({r['id'] for r in selected})==len({norm(r['question']) for r in selected})==60
    assert not {r['id'] for r in selected}&excluded
    jsonl(HERE/'RUNTIME60.frozen.jsonl',selected)
    save(path,dict(seed=SEED,selection='Proportional dataset type quotas, largest remainder; SHA256 ties/order;20 per dataset',
        source=str(source.relative_to(ROOT)),source_sha256=sha(source),selected_ids=[r['id'] for r in selected],
        excluded_ids=sorted(excluded),exclusion_files=inventory,allocation=allocation,
        nonempty_decomposition_required=True,normalized_question_deduplication=True,
        predictions_and_references_used=False,
        interpretation='Disjoint from inventoried prompt cohorts; previously evaluated runtime300, not wholly unseen or independent test'))
    return selected


def prepare():
    if (HERE/'PROTOCOL.frozen.json').exists(): raise FileExistsError('Preserve frozen experiment')
    selected=freeze_cohort()  # Written before reading predictions/references.
    config,examples=prev.old.pilot.settings()
    provenance={r['id']:r for r in read(ROOT/'src/p02_multigrain_timing_cachefix_v1/SOURCE_PROVENANCE.json')['sources']}
    refs={r['id']:r for r in rows(HERE.parent/'p02_flow_simplify_v1_20261020/references300.local.frozen.jsonl')}
    inputs,history,bindings=[],[],[]
    for q in selected:
        p=ROOT/provenance[q['id']]['source']; assert sha(p)==provenance[q['id']]['sha256']
        trace=read(p); rootpath=ROOT/trace['source_prefix']; original=read(rootpath)
        assert trace['id']==original['id']==q['id'] and trace['question']==original['question']==q['question']
        assert original['supporting_questions']==q['supporting_questions']
        assert trace['status']=='success' and trace['finish_reason']=='stop'
        goal=trace['nodes'][-1]; assert goal['role']=='goal' and goal['request']==q['question']
        assert all(n['role']=='ordinary' for n in trace['nodes'][:-1])
        assert [n['request'] for n in trace['nodes'][:-1]]==q['supporting_questions']
        assert trace['final_context'].endswith(passages(original['original_retrieval']))
        record={**{k:q[k] for k in ('id','qid','dataset','question','type')},
            'ordinary_nodes':[{k:n[k] for k in ('step_id','request','answer')} for n in trace['nodes'][:-1]],
            'has_ordinary':True,'goal_step_id':goal['step_id'],'goal_rewritten_query':goal['rewritten_query'],
            'goal_reader_packet':goal['reader_packet'],'goal_original_packet':goal['original_retrieval'],
            'final_root_packet':original['original_retrieval'],'historical_goal_prompt':goal['child_prompt'],
            'source_sha256':sha(p)}
        assert not set(record)&{'reference','gold','prediction','answer','f1','em'}
        inputs.append(record)
        history.append(dict(id=q['id'],dataset=q['dataset'],prediction=trace['prediction'],source=str(p.relative_to(ROOT))))
        bindings.append(dict(id=q['id'],source=str(p.relative_to(ROOT)),sha256=sha(p),
                             root_source=str(rootpath.relative_to(ROOT)),root_sha256=sha(rootpath)))
    jsonl(HERE/'INPUTS60.frozen.jsonl',inputs); jsonl(HERE/'HISTORY60.local.frozen.jsonl',history)
    jsonl(HERE/'REFERENCES60.local.frozen.jsonl',[refs[q['id']] for q in selected])
    jsonl(HERE/'SOURCE_BINDINGS.frozen.jsonl',bindings)
    save(HERE/'GENERATION.frozen.json',config['generation']);save(HERE/'FINAL_EXAMPLES.frozen.json',examples)
    unknown=[]
    # Read-only snapshot across research checkpoints; never seed successes from another experiment.
    for p in sorted((ROOT/'outputs/query_decomposition').glob('p02*/**/checkpoint.sqlite3')):
        if 'formal1500' in str(p) or p.is_relative_to(OUT): continue
        with closing(ro(p)) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='requests'").fetchone(): continue
            cols={r[1] for r in db.execute('PRAGMA table_info(requests)')}
            if not {'id','payload','status'}<=cols: continue
            for rid,payload,status in db.execute("SELECT id,payload,status FROM requests WHERE status IN ('unknown','in_flight')"):
                unknown.append(dict(request_id=rid,payload_sha256=digest(json.loads(payload)),status=status,source=str(p.relative_to(ROOT))))
    jsonl(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl',unknown)
    tasks=[]
    for r in inputs:
        pairs=[('G0','T'),('GF','GF_SPLIT')]
        if int(order('pair|'+r['id']),16)%2:pairs.reverse()
        for pair in pairs:
            # GF and GF_SPLIT both use a single common lazy goal generation.
            if int(order('order|'+r['id']+'|'+pair[0]),16)%2:pair=pair[::-1]
            for a in pair:tasks.append(dict(id=r['id'],dataset=r['dataset'],arm=a,key=a+'|'+r['id']))
    jsonl(HERE/'TASKS.frozen.jsonl',tasks)
    pinned=list(HERE.glob('*.frozen.*'))+[HERE/'run.py',HERE/'templates.py',Path(prev.__file__),Path(templates.prior.__file__),
        prev.old.HERE/'run.py',prev.old.HERE/'resume.py',Path(templates.prior.base.__file__),
        HERE.parent/'p02_prompt_binding_probe_v1_20261022/shared_runtime.py',ROOT/'src/p02_multigrain_timing_cachefix_v1/config.json']
    pinned += [ROOT/'src/p02_multigrain_timing_cachefix_v1/core'/n for n in ('cache.py','llm.py','pipeline.py','final_prompts.py')]
    holds=[PLAN/'STOP_REQUESTED',ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED',
           ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED']
    protocol=dict(id=HERE.name,utc=utc(),authority='持续推进; 请你按照你的想法 独立的推进研究 不需要经过我的任何许可',
        external_API_authority='Explicit existing Qwen API small-experiment approval; future scoped small studies no repeated permission',
        endpoint=config['generation']['base_url'],model=config['generation']['model'],cases=60,conditions=240,
        data='Public benchmark questions and frozen retrieved/prior QA; fictional demonstrations; no references or private ledger payloads',
        arms={'G0':'Exact historical goal fact-sentence prompt','T':'G0 plus target/scope instruction and four paired fictional examples',
              'GF':'Previously frozen compact facts/conclusion synthesis prompt','GF_SPLIT':'Exactly GF goal text, with body/closing-sentence candidate headings inserted; no added facts or extra goal generation'},
        primary_comparisons=['G0->T','GF->GF_SPLIT'],secondary='G0->GF descriptive transfer comparison',
        goal_cap=512,final_cap=64,formatter=False,seed=None,upstream='Frozen decomposition, ordinary QA, goal retrieval and final root',
        sampling='Metadata-only fixed60,20 per dataset, disjoint IDs/normalized text from inventoried prompt cohorts; nonempty decomposition',
        inference='Goal/final intervention on frozen prefix, not new complete RAG or pristine holdout',
        cache='Only current-run exact full payload complete receipts; GF goal shared by both final presentations; no quality rerolls',
        retry='SDK0, durable max5 known technical attempts; unknown held without automatic reissue; prior unknown digests blocked',
        cost='Shared existing budget ledger and provider gate; actual new usage distinct from logical per-arm goal+final usage; no full-flow cost/timing claim',
        split='Deterministic last-sentence boundary with abbreviation guards, lossless text and order; no suitable boundary means raw no-op; report effective changes',
        stop_after='240 terminal outcomes including explicit technical holds; no adaptive prompt or sample changes',
        main_evaluation_resumed=False,candidate_promotion=False,publication=False,
        pins={str(p.relative_to(ROOT)):sha(p) for p in pinned},hold_pins={str(p.relative_to(ROOT)):sha(p) for p in holds})
    save(HERE/'PROTOCOL.frozen.json',protocol);save(REVIEW/'PROTOCOL.json',protocol)
    with (PLAN/'DECISIONS.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(dict(id=DECISION,utc=utc(),approved=True,authority=protocol['authority'],experiment=HERE.name,
            protocol_sha256=sha(HERE/'PROTOCOL.frozen.json'),cases=60,conditions=240,main_evaluation_resumed=False,
            prior_unknown_reissue=False),ensure_ascii=False)+'\n')
    return dry_run()


def check():
    p=read(HERE/'PROTOCOL.frozen.json')
    for group in ('pins','hold_pins'):
        for path,h in p[group].items():assert sha(ROOT/path)==h,path
    assert read(PLAN/'PLAN.json')['execution_approval']['approved'] is False
    assert any(r.get('id')==DECISION and r.get('approved') for r in rows(PLAN/'DECISIONS.jsonl'))
    config,examples=prev.old.pilot.settings()
    assert config['generation']==read(HERE/'GENERATION.frozen.json')
    assert examples==read(HERE/'FINAL_EXAMPLES.frozen.json')
    return config,examples


def dry_run():
    config,examples=check();inputs=rows(HERE/'INPUTS60.frozen.jsonl')
    cohort=read(HERE/'COHORT.frozen.json');assert not set(cohort['selected_ids'])&set(cohort['excluded_ids'])
    blocked={r['payload_sha256'] for r in rows(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl')}
    for r in inputs:
        p=templates.goal_prompt(r,'G0');t=templates.goal_prompt(r,'T')
        assert t.replace('\n\n'+templates.TARGET_GUIDANCE,'',1)==p
        assert t.split('\n\nOriginal question:\n',1)[1]==p.split('\n\nOriginal question:\n',1)[1]
        assert templates.goal_prompt(r,'GF')==templates.goal_prompt(r,'GF_SPLIT')
        for a in ('G0','T','GF'):
            prompt=templates.goal_prompt(r,a)
            assert r['question'] in prompt and passages(r['goal_reader_packet']) in prompt
            assert digest(build_payload(config['generation'],prompt,512)) not in blocked
        raw='A fictional fact is established. The answer is Vale.'
        split,meta=templates.split_presentation(raw)
        assert meta['body']+meta['closing']==raw
        assert split.replace(templates.BODY_HEADER,'',1).replace(templates.CLOSE_HEADER,'',1)==raw
        for a in templates.ARMS:
            p,c,_=templates.final_prompt(r,raw,config,examples,a)
            assert passages(r['final_root_packet']) in c and r['question'] in p
    for raw in ('A single sentence.', 'Dr. Vale attended Elun University.'):
        assert templates.split_presentation(raw)[0]==raw
    result=dict(passed=True,cases=60,conditions=240,new_API_requests=0,disjoint_prompt_cohorts=True,
        G0_T_actual_task_identical=True,GF_goal_shared=True,split_lossless=True,main_evaluation_paused=True)
    save(OUT/'DRYRUN.json',result);return result


def execute(limit=None):
    config,examples=check()
    from scripts.cli_support import temporary_api_key
    inputs={r['id']:r for r in rows(HERE/'INPUTS60.frozen.jsonl')}
    blocked={r['payload_sha256'] for r in rows(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl')}; current={}
    def event(value):
        with (OUT/'trace.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(utc=utc(),**current,**value),ensure_ascii=False)+'\n')
    count=0
    with run_lock(OUT),ExitStack() as stack:
        cache=stack.enter_context(Cache(OUT/'checkpoint.sqlite3'))
        cache.put('meta','protocol',dict(sha256=sha(HERE/'PROTOCOL.frozen.json')))
        budget=stack.enter_context(closing(prev.NoCapBudget(config,OUT)))
        stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'))
        stack.enter_context(prev.connection_route(urlsplit(config['generation']['base_url']).hostname,event))
        live=stack.enter_context(closing(prev.ConnectionEvidenceAsk(cache,None,config,budget,event,
            recover_unknown=False,maximum_attempts=5,stop=lambda:(OUT/'STOP_REQUESTED').exists())))
        def ask(stage,prompt,cap):
            if digest(build_payload(config['generation'],prompt,cap)) in blocked:raise RuntimeError('Prior unknown payload held')
            rid,answer,finish=live(stage,prompt,cap)
            receipt=json.loads(cache.db.execute('SELECT result FROM requests WHERE id=?',(rid,)).fetchone()[0])
            assert receipt.get('returned_model')==config['generation']['model'],'Returned model mismatch'
            u=receipt['usage'];assert u['prompt_tokens']+u['completion_tokens']==u['total_tokens'],'Usage mismatch'
            return rid,answer,finish
        for task in rows(HERE/'TASKS.frozen.jsonl'):
            if cache.get('prediction',task['key']) is not None:continue
            if (OUT/'STOP_REQUESTED').exists() or (limit is not None and count>=limit):return dict(status='local_limit',new_conditions=count)
            current.clear();current.update(task);r=inputs[task['id']];used={};texts={};stages=[]
            try:
                goalarm='GF' if task['arm']=='GF_SPLIT' else task['arm']
                stage='goal/'+goalarm+'|'+task['id'];stages.append(stage)
                rid,goal,_=ask(stage,templates.goal_prompt(r,goalarm),512)
                used['goal']=rid;texts['goal_text']=goal
                prompt,_,meta=templates.final_prompt(r,goal,config,examples,task['arm'])
                texts['presentation']=meta
                stage='final/'+task['key'];stages.append(stage)
                rid,prediction,_=ask(stage,prompt,64);used['final']=rid;texts['prediction']=prediction
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


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','check','run']);p.add_argument('--limit',type=int);a=p.parse_args()
    result=prepare() if a.mode=='prepare' else dry_run() if a.mode=='check' else execute(a.limit)
    print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
