"""Bounded isolated experiments; explicit prepare/run/report modes."""
from collections import Counter
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import hashlib
import ipaddress
import json
import socket
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from . import templates
from experiments.query_decomposition.p02_goal_fewshot_fixed90_v1_20261022 import run as old
from experiments.query_decomposition.p02_goal_fewshot_fixed90_v1_20261022.resume import ConnectionEvidenceAsk
from src.p02_multigrain_timing_cachefix_v1.core.cache import Cache, run_lock, utc, canonical
from src.p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import NoCapBudget, complete_receipt

read, rows, save, jsonl, sha, ro, digest, cost = old.read, old.rows, old.save, old.jsonl, old.sha, old.ro, old.digest, old.cost
OUT = ROOT / "outputs/query_decomposition" / HERE.name
PLAN = old.PLAN
REVIEW = PLAN / "reviews/20261022_prompt_binding_goal_synthesis" / HERE.name
DECISION = "P02_RESULT_BINDING_OPERATOR_SIDE_AUTONOMOUS_20261022"
SOURCE_O = HERE.parent / "p02_reference_binding_priority_probe_v1_20261022/INPUTS.frozen.jsonl"

def prepare():
    if (HERE / "PROTOCOL.frozen.json").exists():
        raise FileExistsError("Preserve frozen experiment")
    config, examples = old.pilot.settings()
    ordinary = rows(SOURCE_O)
    goals = rows(old.HERE / "INPUTS90.frozen.jsonl")
    assert len(ordinary) == 38 and len({x['probe_id'] for x in ordinary}) == 38
    assert len(goals) == 90 and Counter(r['dataset'] for r in goals) == {'hotpotqa': 30, '2wiki': 30, 'musique': 30}
    for r in ordinary:
        assert sha(ROOT / r['source']) == r['source_sha256']
        templates.ordinary_prompt(r, "B")
    jsonl(HERE / "ORDINARY38.frozen.jsonl", ordinary)
    for name in ("INPUTS90.frozen.jsonl", "HISTORY90.local.frozen.jsonl"):
        jsonl(HERE / name, rows(old.HERE / name))
    jsonl(HERE / "REFERENCES90.local.frozen.jsonl", rows(old.pilot.COHORT / "references90.local.frozen.jsonl"))
    save(HERE / "GENERATION.frozen.json", config['generation'])
    save(HERE / "FINAL_EXAMPLES.frozen.json", examples)
    save(HERE / "DEMOS.frozen.json", templates.DEMOS)
    (HERE / "GOAL_INSTRUCTION.frozen.txt").write_text(templates.INSTRUCTION, encoding='utf8')
    unknown = []
    prior_dirs = ['p02_goal_synthesis_probe_v1_20261022/pilot15',
                  'p02_goal_fewshot_synthesis_v1_20261022',
                  'p02_goal_fewshot_fixed90_v1_20261022',
                  'p02_reference_binding_priority_probe_v1_20261022',
                  'p02_ordinary_fewshot_probe_v1_20261022']
    for directory in prior_dirs:
        path = ROOT / 'outputs/query_decomposition' / directory / 'checkpoint.sqlite3'
        if not path.exists():
            continue
        with closing(ro(path)) as db:
            for rid, payload, status in db.execute("SELECT id,payload,status FROM requests WHERE status IN ('unknown','in_flight')"):
                unknown.append(dict(request_id=rid, payload_sha256=digest(json.loads(payload)), status=status,
                                    source=str(path.relative_to(ROOT)), automatic_reissue=False))
    jsonl(HERE / "EXTERNAL_UNKNOWN.frozen.jsonl", unknown)
    tasks = []
    for suite, records, arms in [('ordinary', ordinary, templates.ORDINARY_ARMS), ('goal', goals, templates.GOAL_ARMS)]:
        for record in records:
            ident = record['probe_id'] if suite == 'ordinary' else record['id']
            offset = int(hashlib.sha256(('20261022|' + ident).encode()).hexdigest(), 16) % len(arms)
            for arm in arms[offset:] + arms[:offset]:
                tasks.append(dict(suite=suite, id=ident, dataset=record['dataset'], arm=arm, key=f'{suite}|{arm}|{ident}'))
    jsonl(HERE / "TASKS.frozen.jsonl", tasks)
    pinned = list(HERE.glob('*.frozen.*')) + [HERE/'run.py', HERE/'templates.py', SOURCE_O]
    # Resolve imported implementation paths explicitly; do not pin changing live checkpoints.
    pinned = [p for p in pinned if p.is_file()]
    pinned += [old.HERE/'resume.py', old.HERE/'run.py',
               HERE.parent/'p02_prompt_binding_probe_v1_20261022/shared_runtime.py',
               HERE.parent/'p02_goal_synthesis_probe_v1_20261022/templates.py',
               HERE.parent/'p02_ordinary_fewshot_probe_v1_20261022/run.py',
               ROOT/'src/p02_multigrain_timing_cachefix_v1/config.json']
    pinned += [ROOT/'src/p02_multigrain_timing_cachefix_v1/core'/n for n in
               ('llm.py','cache.py','final_prompts.py','pipeline.py')]
    holds = [PLAN/'STOP_REQUESTED', ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED',
             ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED']
    protocol = dict(id=HERE.name, utc=utc(), authority='请你按照你的想法 独立的推进研究 不需要经过我的任何许可',
        prior_external_authority='使用固定90题 直接允许 以后不要询问我',
        destination=config['generation']['base_url'], model=config['generation']['model'],
        data='Existing public benchmark questions, frozen retrieved passages/prior QA, synthetic demonstration facts and prompt instructions; no labels, references or ledger in payloads',
        scope='Two exposed component development experiments. No main evaluation, live retrieval, decomposer changes or publication.',
        ordinary=dict(nodes=38, unique_queries=len({r['case_id'] for r in ordinary}), arms=['D','B'],
                      D='Existing E2 descriptive demonstrations', B='Same D plus two demonstration-only result extraction/binding notes',
                      actual_task_unchanged=True, cap=128, formatter=False,
                      interpretation='Raw ordinary answer intervention only; correctness and necessary-information review, no downstream F1 claim'),
        goal=dict(cases=90, arms=['G0','GF','GO'],
                  G0='Historical fact-sentence prompt at common512 cap and neutral common final wrapper',
                  GF='Compact synthesis instruction plus six matched fact-and-conclusion demonstrations',
                  GO='Same GF instruction/facts/conclusions plus six essential-operation sentences',
                  cap=512, final_cap=64, formatter=False, upstream='Frozen R02/v12, ordinary QA, goal rewrite and reader, final root'),
        cache='Fresh interleaved controls with seed=None; exact current-run duplicates may reuse strict complete receipts; no quality rerolls',
        retry='SDK/HTTP retry0; at most5 durable known technical attempts; typed proved pre-transmission failures recoverable; unknown held without replay',
        unknown='Existing unknown payload digests guarded; each newly ambiguous request held, other unrelated cases may continue',
        cost='Existing ledger/gate; logical input/output tokens and returned usage costs; unknown reserves retained',
        limitations='Exposed and partly stress-selected data; ordinary has 8 stress plus30 existing nodes. GF-GO also changes output demonstration length; no pure reasoning or symbol-necessity claim.',
        stop_after='All346 conditions or documented technical holds; no adaptive prompt changes during this experiment',
        main_evaluation_resumed=False, publication=False,
        pins={str(p.relative_to(ROOT)):sha(p) for p in pinned},
        hold_pins={str(p.relative_to(ROOT)):sha(p) for p in holds if p.exists()})
    save(HERE/'PROTOCOL.frozen.json', protocol)
    save(REVIEW/'PROTOCOL.json', protocol)
    with (PLAN/'DECISIONS.jsonl').open('a', encoding='utf8') as handle:
        handle.write(json.dumps(dict(id=DECISION, utc=utc(), authority=protocol['authority'],
            prior_external_authority=protocol['prior_external_authority'], approved=True,
            experiment=HERE.name, protocol_sha256=sha(HERE/'PROTOCOL.frozen.json'),
            main_evaluation_resumed=False, unknown_reissue=False), ensure_ascii=False)+'\n')
    return dry_run()

def check():
    p = read(HERE/'PROTOCOL.frozen.json')
    for group in ('pins','hold_pins'):
        for path, expected in p[group].items():
            assert sha(ROOT/path) == expected, path
    assert read(PLAN/'PLAN.json')['execution_approval']['approved'] is False
    assert any(d.get('id') == DECISION and d.get('approved') for d in rows(PLAN/'DECISIONS.jsonl'))
    config, examples = old.pilot.settings()
    assert config['generation'] == read(HERE/'GENERATION.frozen.json')
    assert examples == read(HERE/'FINAL_EXAMPLES.frozen.json')
    return config, examples

def dry_run():
    config, examples = check()
    for record in rows(HERE/'ORDINARY38.frozen.jsonl'):
        d, b = [templates.ordinary_prompt(record,a) for a in ('D','B')]
        stripped = b
        for _, note in templates.NOTES:
            stripped = stripped.replace('\n\n'+note, '', 1)
        assert stripped == d
        assert d.split('\n\nActual task:\n',1)[1] == record['control_prompt'].split('\n\nActual task:\n',1)[1]
    gf, go = [templates.demonstration_text(a) for a in ('GF','GO')]
    for demo in templates.DEMOS:
        go = go.replace(demo['operation']+' ', '', 1)
    assert go == gf
    for record in rows(HERE/'INPUTS90.frozen.jsonl'):
        f, o = [templates.goal_prompt(record,a) for a in ('GF','GO')]
        assert f.split('\n\nActual task:\n')[1] == o.split('\n\nActual task:\n')[1]
        assert all(n['request'] in f and n['answer'] in f for n in record['ordinary_nodes'])
        assert templates.goal_prompt(record,'G0') == record['historical_goal_prompt']
        final, _ = templates.base.final_prompt(record, 'FIXED CANDIDATE', config, examples)
        assert 'FIXED CANDIDATE' in final
    result = dict(passed=True, ordinary_nodes=38, goal_queries=90, conditions=346,
                  ordinary_only_example_notes_change=True, goal_only_demonstration_operation_sentences_change=True,
                  new_API_requests=0, main_evaluation_paused=True)
    save(OUT/'DRYRUN.json', result)
    return result

@contextmanager
def connection_route(host, events):
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    class Logger:
        def event(self, name, **values):
            events(dict(event=name, **values))
    with provider_dns(host, Logger()):
        original = socket.getaddrinfo
        addresses = {r[4][0] for r in original(host,443,type=socket.SOCK_STREAM)}
        preferred = '47.94.20.201'
        selected = preferred if preferred in addresses and ipaddress.ip_address(preferred).is_global else None
        def resolve(name, port, *args, **kwargs):
            decoded = name.decode('ascii') if isinstance(name,bytes) else name
            return original(selected if selected and decoded == host else name,port,*args,**kwargs)
        socket.getaddrinfo = resolve
        events(dict(event='process_only_dns_route',host=host,selected=selected,original_URL_Host_SNI_TLS=True))
        try:
            yield
        finally:
            socket.getaddrinfo = original

def execute(limit=None):
    config, examples = check()
    from scripts.cli_support import temporary_api_key
    records = {'ordinary':{r['probe_id']:r for r in rows(HERE/'ORDINARY38.frozen.jsonl')},
               'goal':{r['id']:r for r in rows(HERE/'INPUTS90.frozen.jsonl')}}
    unknown = {x['payload_sha256'] for x in rows(HERE/'EXTERNAL_UNKNOWN.frozen.jsonl')}
    current = {}
    def event(value):
        with (OUT/'trace.jsonl').open('a',encoding='utf8') as handle:
            handle.write(json.dumps(dict(utc=utc(),**current,**value),ensure_ascii=False)+'\n')
    count = 0
    with run_lock(OUT), ExitStack() as stack:
        cache = stack.enter_context(Cache(OUT/'checkpoint.sqlite3'))
        cache.put('meta','protocol',dict(sha256=sha(HERE/'PROTOCOL.frozen.json')))
        budget = stack.enter_context(closing(NoCapBudget(config,OUT)))
        stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'))
        stack.enter_context(connection_route(urlsplit(config['generation']['base_url']).hostname,event))
        live = stack.enter_context(closing(ConnectionEvidenceAsk(cache,None,config,budget,event,
            recover_unknown=False,maximum_attempts=5,stop=lambda:(OUT/'STOP_REQUESTED').exists())))
        def ask(stage,prompt,cap):
            if digest(build_payload(config['generation'],prompt,cap)) in unknown:
                raise RuntimeError('Prior unresolved payload held without resubmission')
            output = live(stage,prompt,cap)
            receipt = json.loads(cache.db.execute('SELECT result FROM requests WHERE id=?',(output[0],)).fetchone()[0])
            assert receipt.get('returned_model') == config['generation']['model'], 'Returned model mismatch; do not reuse or reroll'
            usage = receipt['usage']
            assert usage['prompt_tokens'] + usage['completion_tokens'] == usage['total_tokens'], 'Returned usage mismatch'
            return output
        for task in rows(HERE/'TASKS.frozen.jsonl'):
            if cache.get('prediction',task['key']) is not None:
                continue
            if (OUT/'STOP_REQUESTED').exists() or (limit is not None and count >= limit):
                return dict(status='local_limit',new_conditions=count)
            current.clear(); current.update(task)
            record = records[task['suite']][task['id']]
            used, texts = {}, {}
            stages = []
            try:
                stage = task['suite']+'/'+task['key']; stages.append(stage)
                if task['suite'] == 'ordinary':
                    rid, answer, _ = ask(stage,templates.ordinary_prompt(record,task['arm']),128)
                    used['ordinary'] = rid; texts['answer'] = answer
                else:
                    rid, goal, _ = ask(stage,templates.goal_prompt(record,task['arm']),512)
                    used['goal'] = rid; texts['goal_text'] = goal
                    final, _ = templates.base.final_prompt(record,goal,config,examples)
                    stage = 'final/'+task['key']; stages.append(stage)
                    rid, prediction, _ = ask(stage,final,64)
                    used['final'] = rid; texts['prediction'] = prediction
                result = dict(**task,status='success',**texts)
            except Exception as exc:
                sql = 'SELECT q.status,q.result FROM requests q JOIN request_stages s ON s.request_id=q.id WHERE s.stage IN ('+','.join('?' for _ in stages)+') ORDER BY q.rowid DESC LIMIT 1'
                terminal = cache.db.execute(sql,stages).fetchone()
                receipt = json.loads(terminal[1]) if terminal and terminal[1] else {}
                algorithm = bool(terminal and terminal[0]=='success' and receipt.get('usage') and
                    (receipt.get('finish_reason') in ('length','content_filter') or receipt.get('refusal') or not (receipt.get('answer') or '').strip()))
                result = dict(**task,status='algorithm_failure' if algorithm else 'technical_held',
                    **texts,error_type=type(exc).__name__,error=str(exc)[:300])
                result.setdefault('prediction','' if algorithm else None)
                if terminal and terminal[0] in ('unknown','in_flight'):
                    # The request itself stays unknown; other payloads may proceed.
                    event(dict(event='ambiguous_request_held',automatic_reissue=False))
            usage_by_stage = {}
            for stage, rid in used.items():
                raw = cache.db.execute('SELECT result FROM requests WHERE id=?',(rid,)).fetchone()[0]
                receipt = json.loads(raw)
                usage_by_stage[stage] = dict(request_id=rid,usage=receipt['usage'])
            result['usage_by_stage'] = usage_by_stage
            cache.put('prediction',task['key'],result)
            event(dict(event='condition_terminal',status=result['status']))
            count += 1
            print(json.dumps(dict(suite=task['suite'],arm=task['arm'],id=task['id'],status=result['status'],new_conditions=count)),flush=True)
    return dict(status='bounded_run_finished',new_conditions=count)

def report():
    config, _ = check()
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    refs = {r['id']:r for r in rows(HERE/'REFERENCES90.local.frozen.jsonl')}
    inputs = {r['id']:r for r in rows(HERE/'INPUTS90.frozen.jsonl')}
    history = {r['id']:r for r in rows(HERE/'HISTORY90.local.frozen.jsonl')}
    scorers = {d:metric(d) for d in ('hotpotqa','2wiki','musique')}
    with closing(ro(OUT/'checkpoint.sqlite3')) as db:
        results = [json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='prediction'")]
        receipt_status = dict(db.execute('SELECT status,count(*) FROM requests GROUP BY status'))
    goals = [r for r in results if r['suite']=='goal']
    for r in goals:
        valid = r['status'] in ('success','algorithm_failure')
        r['metrics'] = scorers[r['dataset']](r.get('prediction',''),refs[r['id']]) if valid else None
    index = {(r['arm'],r['id']):r for r in goals}
    common = [i for i in inputs if all(index.get((a,i),{}).get('metrics') is not None for a in templates.GOAL_ARMS)]
    table = []
    for ds in ('all','hotpotqa','2wiki','musique'):
        ids = [i for i in common if ds=='all' or inputs[i]['dataset']==ds]
        scores = {}
        for arm in ('historical_native_P02',)+templates.GOAL_ARMS:
            values = [scorers[inputs[i]['dataset']](history[i]['prediction'],refs[i]) if arm=='historical_native_P02'
                      else index[(arm,i)]['metrics'] for i in ids]
            scores[arm] = {k:sum(x[k] for x in values)/len(ids) if ids else None for k in ('f1','em')}
        table.append(dict(dataset=ds,n=len(ids),scores=scores))
    paired = {}
    for a,b in (('G0','GF'),('G0','GO'),('GF','GO')):
        counts = Counter(); differences = []
        for i in common:
            d = index[(b,i)]['metrics']['f1']-index[(a,i)]['metrics']['f1']
            differences.append(d); counts['improved' if d>1e-12 else 'harmed' if d < -1e-12 else 'tied']+=1
        paired[a+'_to_'+b] = dict(n=len(common),**counts,mean_F1_delta=sum(differences)/len(common) if common else None)
    namespace = 'p02_prompt_research_v1_20261022/'+str(OUT.resolve())
    with closing(ro(config['budget']['ledger'])) as db:
        actual,reserve = db.execute('SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?',(namespace,)).fetchone()
    successful = [i for i in common if all(index[(a,i)]['status']=='success' for a in templates.GOAL_ARMS)]
    costs = {}
    for a in templates.GOAL_ARMS:
        usages = [[x['usage'] for x in index[(a,i)]['usage_by_stage'].values()] for i in successful]
        costs[a] = dict(n=len(successful),**{k:sum(sum(u[k] for u in us) for us in usages)/len(usages) if usages else None for k in ('prompt_tokens','completion_tokens')},
                        goal_final_CNY=sum(sum(cost(u) for u in us) for us in usages)/len(usages) if usages else None)
    ordinary = [r for r in results if r['suite']=='ordinary']
    ordinary_index = {(r['arm'],r['id']):r for r in ordinary}
    oi = rows(HERE/'ORDINARY38.frozen.jsonl')
    ordinary_changed = [dict(probe_id=r['probe_id'],case_id=r['case_id'],cohort=r['cohort'],current_question=r['current_question'],
                           history=r['history'],variants={a:ordinary_index.get((a,r['probe_id'])) for a in ('D','B')}) for r in oi
                        if all((a,r['probe_id']) in ordinary_index for a in ('D','B')) and
                        ordinary_index[('D',r['probe_id'])].get('answer') != ordinary_index[('B',r['probe_id'])].get('answer')]
    summary = dict(terminal_conditions=len(results),planned_conditions=346,receipt_statuses=receipt_status,
        statuses={s:dict(Counter(r['status'] for r in results if r['suite']==s)) for s in ('ordinary','goal')},
        same_three_arm_n=len(common),common_ids=common,table=table,paired=paired,costs=costs,
        ordinary_different_text_pairs=len(ordinary_changed),new_returned_usage_CNY=actual,unresolved_reserved_CNY=reserve,
        main_evaluation_paused=True,scope='Frozen-prefix exposed development intervention, not complete-RAG validation')
    save(OUT/'SUMMARY.json',summary); save(REVIEW/'RESULTS.json',summary)
    jsonl(OUT/'DETAILS.local.jsonl',results)
    jsonl(OUT/'ORDINARY_CHANGED.local.jsonl',ordinary_changed)
    jsonl(OUT/'GOAL_CHANGED.local.jsonl',[dict(id=i,question=inputs[i]['question'],reference=refs[i]['answer'],
          variants={a:index[(a,i)] for a in templates.GOAL_ARMS}) for i in common
          if len({index[(a,i)]['prediction'] for a in templates.GOAL_ARMS})>1])
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','check','run','report'])
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    answer={'prepare':prepare,'check':dry_run,'report':report}.get(args.mode)
    print(json.dumps(answer() if answer else execute(args.limit),ensure_ascii=True),flush=True)

if __name__=='__main__':
    main()
