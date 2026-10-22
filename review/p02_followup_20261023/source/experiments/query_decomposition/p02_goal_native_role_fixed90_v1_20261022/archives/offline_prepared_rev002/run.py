"""Prepare/verify native goal128 ROLE; only separately authorized run can submit."""
from collections import Counter
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from p02_multigrain_timing_cachefix_v1.core.cache import canonical, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case, passages
from experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.shared_runtime import (
    StrictAsk, NoCapBudget, complete_receipt, atomic_json, INPUT_RATE, OUTPUT_RATE,
)
from . import templates

PACKAGE = ROOT / 'src/p02_multigrain_timing_cachefix_v1'
W = HERE.parent / 'p02_final_target_check_fixed90_v1_20261022'
BASE = HERE.parent / 'p02_noformatter_quick90_v1_20261021'
PLAN = HERE.parent / 'unified_rag_eval_v1_20261021'
OUT = ROOT / 'outputs/query_decomposition' / HERE.name
INPUTS = HERE / 'INPUTS90.frozen.jsonl'
HISTORY = HERE / 'HISTORICAL_STAGE_RECEIPTS90.local.frozen.jsonl'
CONTRACTS = HERE / 'CONTRACTS180.frozen.jsonl'
ORDER = HERE / 'ORDER180.frozen.jsonl'
PROTOCOL = HERE / 'PROTOCOL.frozen.json'
SEED = 20261022
DATASETS = ('hotpotqa', '2wiki', 'musique')
SCORERS = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\', '/')


def jsonl(path, values):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values), encoding='utf8')


def ro(path, immutable=False):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro'+('&immutable=1' if immutable else ''), uri=True)


def usage_ok(receipt):
    u = receipt.get('usage') or {}
    return (all(type(u.get(k)) is int and u[k] >= 0 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
            and u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens'])


def integrity(receipt, model):
    return (receipt.get('status') == 'success' and usage_ok(receipt)
            and receipt.get('requested_model') == receipt.get('returned_model') == model)


def normal(receipt, model):
    return integrity(receipt, model) and complete_receipt(receipt)


def algorithm_failure(receipt, model):
    if not integrity(receipt, model):
        return None
    if receipt.get('finish_reason') in ('length', 'content_filter'):
        return 'terminal_' + receipt['finish_reason']
    if receipt.get('refusal'):
        return 'terminal_refusal'
    if isinstance(receipt.get('answer'), str) and not receipt['answer'].strip():
        return 'terminal_empty'
    return None


def any_returned(receipt):
    # Even isolated refusal/finish_reason is returned evidence, not a blind retry.
    if any(receipt.get(k) is not None for k in ('response_id', 'raw_response', 'usage', 'returned_model', 'refusal', 'finish_reason')):
        return True
    value = receipt.get('answer') or receipt.get('content') or ''
    return bool(value.strip()) if isinstance(value, str) else bool(value)


class SubmissionAwareStrictAsk(StrictAsk):
    """Add return-evidence/ledger crash-window guard without editing shared code."""
    def _retire(self, rid, stage, payload, attempts, reason):
        row = self.cache.db.execute('SELECT result FROM requests WHERE id=?', (rid,)).fetchone()
        receipt = json.loads(row[0]) if row and row[0] else {}
        charge = self.budget.db.execute('SELECT actual,status FROM charges WHERE id=?', (rid,)).fetchone()
        if any_returned(receipt) or (charge and (charge[0] is not None or charge[1] == 'returned_usage')):
            raise RuntimeError('Held returned evidence in request or ledger; reconcile without resubmission')
        return super()._retire(rid, stage, payload, attempts, reason)


def cost(usage):
    return usage['prompt_tokens']*INPUT_RATE + usage['completion_tokens']*OUTPUT_RATE


def old_success(receipt, payload):
    # Old source receipts omit requested_model, so bind it by exact old payload.
    return (receipt.get('status') == 'success' and receipt.get('returned_model') == payload['model']
            and usage_ok(receipt) and receipt.get('finish_reason') == 'stop'
            and isinstance(receipt.get('answer'), str) and bool(receipt['answer'].strip())
            and not receipt.get('refusal'))


def stage_receipt(db, request_id, expected_payload=None):
    row = db.execute('SELECT payload,request_id,result FROM requests WHERE request_id=?', (request_id,)).fetchone()
    assert row and row[1] == request_id
    payload, result = json.loads(row[0]), json.loads(row[2])
    if expected_payload is not None:
        assert row[0] == canonical(expected_payload) and payload == expected_payload
    assert old_success(result, payload)
    return dict(request_id=request_id, payload=payload, result=result)


def assemble():
    """Read only W sources/native replay. Never open reference/gold here."""
    config = read(PACKAGE / 'config.json')
    examples = read(PACKAGE / 'prompts/final_examples.json')
    wi = rows(W / 'INPUTS90.frozen.jsonl')
    proofs = {r['id']: r for r in rows(W / 'SOURCE_PROOF90.frozen.jsonl')}
    public = read(W / 'PUBLIC_PAYLOAD_PROOF.json')
    controls = {r['id']: r for r in rows(ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/p02_frozen_no_refs_quick90_v1_20261021/control90.jsonl')}
    input_rows, history, audits = [], [], []
    with closing(ro(PACKAGE / 'data/replay.sqlite3', True)) as db:
        for w in wi:
            assert sha(ROOT / w['source']) == w['source_sha256'] == proofs[w['id']]['source_sha256']
            source = read(ROOT / w['source'])
            t = controls[w['id']]['trajectory']
            goal = source['nodes'][-1]
            ordinary = source['nodes'][:-1]
            assert goal['role'] == 'goal' and all(n['role'] == 'ordinary' for n in ordinary)
            assert goal['request'] == source['question']
            assert [n['request'] for n in ordinary] == t['supporting_questions']
            assert ordinary == t['nodes'][:-1]
            goal_payload = build_payload(config['generation'], goal['child_prompt'], 128)
            receipt = stage_receipt(db, goal['answer_request_id'], goal_payload)
            assert receipt['result']['answer'].strip() == goal['initial_answer']
            goal_inputs = {k: deepcopy(goal[k]) for k in ('step_id', 'request', 'role', 'prior_complete_answers',
                'excluded_prior_step_ids', 'original_retrieval', 'rewritten_query', 'rewritten_retrieval', 'reader_packet',
                'rewrite_prompt', 'raw_rewrite', 'rewrite_request_id', 'rewrite_rejection', 'child_prompt')}
            record = dict(id=w['id'], dataset=w['dataset'], qid=w['qid'], question=source['question'],
                supporting_questions=t['supporting_questions'], source=w['source'], source_sha256=w['source_sha256'],
                ordinary_nodes=deepcopy(ordinary), goal_inputs=goal_inputs, original_goal_payload=goal_payload,
                final_root_packet=deepcopy(t['original_retrieval']), upstream_state_sha256=w['upstream_state_sha256'])
            formatter = None
            should_format = templates.needs_formatter(goal['initial_answer'])
            assert should_format == bool(goal['sentence_format_request_id'])
            if should_format:
                fp = build_payload(config['generation'], templates.formatter_prompt(record, goal['initial_answer']), 128)
                formatter = stage_receipt(db, goal['sentence_format_request_id'], fp)
                assert formatter['result']['answer'].strip() == goal['answer']
            else:
                assert goal['initial_answer'] == goal['answer']
            final_text, final_context, helper = templates.final_prompt(record, goal['answer'], config, examples)
            final_payload = build_payload(config['generation'], final_text, 64)
            assert final_payload == w['historical_final_payload']
            assert final_text == source['final_prompt'] == t['final_prompt'] and final_context == source['final_context'] == t['final_context'] and helper == t['helper']
            final = stage_receipt(db, source['request_id'], final_payload)
            assert final['result'] == w['historical_final_result'] and final['result']['answer'].strip() == source['prediction']
            excluded = {goal['answer_request_id'], source['request_id']}
            if formatter:
                excluded.add(formatter['request_id'])
            prefix = [r for r in controls[w['id']]['calls'] if r['request_id'] not in excluded]
            assert len({r['request_id'] for r in prefix}) == len(prefix)
            assert all(usage_ok(r) for r in prefix)
            record['frozen_prefix_usage'] = {k: sum(r['usage'][k] for r in prefix) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            record['frozen_prefix_request_ids'] = [r['request_id'] for r in prefix]
            record['historical_stage_usage'] = dict(goal=receipt['result']['usage'], formatter=formatter['result']['usage'] if formatter else None, final=final['result']['usage'])
            assert not excluded.intersection(record['frozen_prefix_request_ids'])
            for k in ('prompt_tokens','completion_tokens','total_tokens'):
                # W.upstream_usage excludes only final, so explicitly subtract
                # the historical goal (+ actual goal formatter if any) once.
                assert record['frozen_prefix_usage'][k]+receipt['result']['usage'][k]+(formatter['result']['usage'][k] if formatter else 0)==w['upstream_usage'][k]
            for role,value in [('goal',receipt),('final',final)]+([('formatter',formatter)] if formatter else []):
                matches=[r for r in controls[w['id']]['calls'] if r['request_id']==value['request_id']]
                assert len(matches)==1 and matches[0]['usage']==value['result']['usage']
            assert not {'gold', 'reference', 'answer', 'prediction', 'f1', 'em'} & set(record)
            input_rows.append(record)
            history.append(dict(id=w['id'], dataset=w['dataset'], source=w['source'], source_sha256=w['source_sha256'], goal=receipt, formatter=formatter, final=final))
            audits.append(dict(id=w['id'], dataset=w['dataset'], source=w['source'], source_sha256=w['source_sha256'],
                original_goal_request_id=receipt['request_id'], original_goal_payload_sha256=digest(goal_payload), original_goal_cap=128,
                original_goal_prompt_exact=True, original_goal_stop_model_three_usage=True,
                historical_formatter_triggered=should_format, actual_formatter_payload_exact=bool(formatter) if formatter else 'not_triggered',
                formatter_branch_preserved=True, original_final_request_id=final['request_id'], original_final_payload_sha256=digest(final_payload),
                original_final_renderer_exact=True, original_final_stop_model_three_usage=True,
                historical_missing_requested_model_bound_by_exact_payload=True, public_container=proofs[w['id']]['public_container']))
            audits[-1]['pre_goal_usage_exact_W_upstream_minus_old_goal_and_goal_formatter']=True
    assert len(input_rows) == 90 and Counter(r['dataset'] for r in input_rows) == {ds: 30 for ds in DATASETS}
    return input_rows, history, audits, public


def task_order(data):
    offset = int(hashlib.sha256(str(SEED).encode()).hexdigest()[:8], 16) % 2
    return [dict(index=i*2+j, id=r['id'], dataset=r['dataset'], arm=arm, task_id=r['id']+'/'+arm)
            for i, r in enumerate(data) for j, arm in enumerate(templates.ARMS[(i+offset)%2:]+templates.ARMS[:(i+offset)%2])]


def contracts(data):
    config, examples = read(PACKAGE / 'config.json'), read(PACKAGE / 'prompts/final_examples.json')
    result = []
    for r in data:
        for arm in templates.ARMS:
            payload = templates.goal_payload(r, arm)
            assert payload == build_payload(config['generation'], payload['messages'][0]['content'], 128)
            stripped = deepcopy(payload)
            if arm == 'L1':
                assert stripped['messages'][0]['content'].count(templates.ROLE) == 1
                stripped['messages'][0]['content'] = stripped['messages'][0]['content'].replace('\n'+templates.ROLE+templates.MARKER, templates.MARKER, 1)
            assert stripped == r['original_goal_payload']
            fp, context, helper = templates.final_prompt(r, 'SYNTHETIC SAME FACTUAL SENTENCE.', config, examples)
            result.append(dict(id=r['id'], dataset=r['dataset'], arm=arm, goal_payload_sha256=digest(payload),
                original_goal_cap=128, formatter_cap=128, final_cap=64, goal_control_exact=(arm=='L0'),
                treatment_only_exact_ROLE=(arm=='L1'), final_same_content_payload_sha256=digest(build_payload(config['generation'], fp, 64)),
                formatter_two_arms_identical=True, ordinary_goal_rewrite_reader_root_frozen=True))
    assert len(result) == len({r['goal_payload_sha256'] for r in result}) == 180
    assert all(len({c['final_same_content_payload_sha256'] for c in result if c['id']==r['id']})==1 for r in data)
    return result


def core_formatter_contract(record, historical):
    """Exercise native Pipeline's real <4-word branch; only synthetic responses."""
    config = read(PACKAGE / 'config.json')
    case = safe_case(record)
    captured = {}
    with closing(ro(PACKAGE / 'data/replay.sqlite3', True)) as seed:
        def retrieve(query):
            row = seed.execute('SELECT packet FROM retrieval WHERE dataset=? AND qid=? AND query=?', (case['dataset'], case['qid'], query)).fetchone()
            assert row, 'No new retrieval in offline contract'
            return json.loads(row[0])

        def ask(stage, prompt, maximum, allow_final_truncation=False):
            sid = record['goal_inputs']['step_id']
            if stage == f'legacy/answer/{sid}':
                assert prompt == record['original_goal_payload']['messages'][0]['content'] and maximum == 128
                return 'synthetic:goal', 'Paris', 'stop'
            if stage == f'legacy/sentence_format/{sid}':
                assert maximum == 128 and prompt == templates.formatter_prompt(record, 'Paris')
                captured['formatter'] = digest(build_payload(config['generation'], prompt, maximum))
                return 'synthetic:formatter', 'Paris is a city in France.', 'stop'
            if stage == 'legacy/final':
                expected = templates.final_prompt(record, 'Paris is a city in France.', config, read(PACKAGE/'prompts/final_examples.json'))[0]
                assert prompt == expected and maximum == 64
                captured['final'] = digest(build_payload(config['generation'], prompt, maximum))
                return 'synthetic:final', 'Paris', 'stop'
            payload = build_payload(config['generation'], prompt, maximum)
            row = seed.execute('SELECT request_id,result FROM requests WHERE payload=?', (canonical(payload),)).fetchone()
            assert row, 'Native unchanged prefix must replay exactly'
            receipt = json.loads(row[1])
            assert old_success(receipt, payload)
            return row[0], receipt['answer'].strip(), 'stop'

        Pipeline(config).execute(case, retrieve, ask)
    assert set(captured) == {'formatter', 'final'}
    assert templates.needs_formatter('one two three') and not templates.needs_formatter('one two three four')
    return dict(passed=True, source_case_id=record['id'], synthetic_only_not_quality_data=True, exact_native_branch_payloads=captured)


def prepare():
    if PROTOCOL.exists():
        return dry_run()
    OUT.mkdir(parents=True, exist_ok=True)
    data, history, audit, public = assemble()
    jsonl(INPUTS, data)
    jsonl(HISTORY, history)
    jsonl(HERE/'SOURCE_AUDIT90.frozen.jsonl', audit)
    jsonl(CONTRACTS, contracts(data))
    jsonl(ORDER, task_order(data))
    (HERE/'ROLE.frozen.txt').write_text(templates.ROLE+'\n', encoding='utf8')
    (HERE/'references90.local.frozen.jsonl').write_bytes((BASE/'references90.local.frozen.jsonl').read_bytes())
    (HERE/'PUBLIC_PAYLOAD_PROOF.json').write_bytes((W/'PUBLIC_PAYLOAD_PROOF.json').read_bytes())
    for arm in templates.ARMS:
        (HERE/(arm+'_EXAMPLE_PROMPT.txt')).write_text(templates.goal_payload(data[0], arm)['messages'][0]['content'], encoding='utf8')
    pipeline_fixture = core_formatter_contract(data[0], history[0])
    atomic_json(HERE/'FORMATTER_CORE_BRANCH_CONTRACT.frozen.json', pipeline_fixture)
    files = [p for p in HERE.iterdir() if p.is_file() and p.name != 'PROTOCOL.frozen.json']
    files += [W/'PROTOCOL.frozen.json', W/'INPUTS90.frozen.jsonl', W/'SOURCE_PROOF90.frozen.jsonl',
              HERE.parent/'p02_goal_fact_role_probe_v1_20261022/templates.py',
              HERE.parent/'p02_goal_scope_role_probe_v1_20261022/templates.py',
              HERE.parent/'p02_prompt_binding_probe_v1_20261022/shared_runtime.py',
              PACKAGE/'config.json', PACKAGE/'core/pipeline.py', PACKAGE/'core/cache.py', PACKAGE/'core/llm.py',
              PACKAGE/'core/child_prompts.py', PACKAGE/'core/rewrite_prompts.py', PACKAGE/'core/final_prompts.py',
              PACKAGE/'core/runtime.py', PACKAGE/'prompts/child_examples.json', PACKAGE/'prompts/final_examples.json',
              ROOT/'scripts/cli_support.py', HERE.parent/'p02_formal1500_v1_20261021/network.py',
              HERE.parent/'comprehensive_eval_v1_20261015/scoring.py']
    pins = {rel(p): sha(p) for p in files}
    pins.update({r['source']: r['source_sha256'] for r in data})
    protocol = dict(id=HERE.name, utc=utc(), status='offline_prepared_not_authorized', cases=90, conditions=180,
        per_dataset={ds:30 for ds in DATASETS}, arms=list(templates.ARMS), goal_cap=128, formatter_cap=128, final_cap=64,
        source='Exactly W90 source order and real historical R02/v12 D; repeatedly exposed development, not formal1500 decode cohort or manual corrected60',
        empty_decomposition_cases=sum(not r['supporting_questions'] for r in data),
        only_treatment='L1 append exact ROLE at original goal main instruction end; L0 fresh exact original complete goal payload',
        ROLE=templates.ROLE, insertion_marker=templates.MARKER, uses_T512_or_G2_output=False,
        formatter='Original stripped-answer whitespace wordcount<4, exact original formatter128 preserved in botharms; original question and original goal reader only',
        final='Original core.render_final and helper/examples/readout64, no synthesis wrapper',
        fixed='All D, ordinaryQA, goal rewrite/both route packets/merged reader and final original root; no new retrieval or GPU',
        order='W90 case order; seed20261022 cyclic arm starts,45 each; no result-based selection',
        historical_goal_formatter_final_used_as_fresh_seed=False, seed=None,
        terminal='Complete matching requested/returned model and all3 int usage fields excludingbool+sum; length/filter/refusal/empty terminal0 no reroll, suppress later stages; malformed/mismatch/technicalunknown held',
        request_policy=dict(SDK_retries=0,HTTP_retries=0,maximum_actual_attempts=5,across_restarts=True,shared_gate=True,
            unknown_reservations_preserved=True,ledger_returned_evidence_guard=True,isolated_refusal_finish_reason_hold=True),
        maximum_stage_payloads_before_technical_retries=540,
        cost='No fixed cap; all actual returned usage/unknown liabilities separate; logical fullflow frozen prefix plus newgoal/formatter/final; old expense not billed twice',
        expense_alert='Same-stage single3x or recent20 mean1.5x paired historical; newly-triggered formatter separately logged without fake zero historical comparator',
        scoring='Local native refs only in report; factory once/dataset; F1/EM full90 only if allqualityterminal, bounds otherwise; oldP02 historical separately',
        public_provenance='Copied W90 already verified publicsource proof with exactsourcehashes; no fresh web verification',
        authorization='Root decides after locked T review; separate RUN_AUTHORIZATION binding this protocolhash+decisionID required; no auth created byprepare',
        model=read(PACKAGE/'config.json')['generation']['model'],
        main_evaluation_remains_paused=True, new_API_requests_at_prepare=0, pins=pins)
    atomic_json(PROTOCOL, protocol)
    return dry_run()


def check():
    p = read(PROTOCOL)
    assert all(sha(ROOT/k) == v for k,v in p['pins'].items())
    assert read(PLAN/'PLAN.json')['execution_approval']['approved'] is False
    return p


def dry_run():
    check()
    from experiments.query_decomposition.p02_goal_scope_role_probe_v1_20261022.templates import ROLE
    assert templates.ROLE == ROLE
    data, history, audit, public = assemble()
    assert data == rows(INPUTS) and history == rows(HISTORY) and audit == rows(HERE/'SOURCE_AUDIT90.frozen.jsonl')
    assert public == read(HERE/'PUBLIC_PAYLOAD_PROOF.json')
    assert contracts(data) == rows(CONTRACTS) and task_order(data) == rows(ORDER)
    fixture = core_formatter_contract(data[0], history[0])
    assert fixture == read(HERE/'FORMATTER_CORE_BRANCH_CONTRACT.frozen.json')
    model = read(PACKAGE/'config.json')['generation']['model']
    valid = dict(status='success',requested_model=model,returned_model=model,usage=dict(prompt_tokens=20,completion_tokens=5,total_tokens=25),answer='text',finish_reason='stop')
    assert normal(valid,model)
    for mutation, reason in [({'finish_reason':'length'},'terminal_length'),({'finish_reason':'content_filter'},'terminal_content_filter'),({'refusal':'declined'},'terminal_refusal'),({'answer':''},'terminal_empty')]:
        assert algorithm_failure({**valid,**mutation},model) == reason
    for invalid in ({**valid,'usage':None},{**valid,'returned_model':'other'},{**valid,'usage':dict(prompt_tokens=True,completion_tokens=5,total_tokens=6)}, {**valid,'usage':dict(prompt_tokens=20,completion_tokens=5,total_tokens=24)}):
        assert not normal(invalid,model) and algorithm_failure({**invalid,'finish_reason':'length'},model) is None
    assert not any_returned(dict(status='unknown',error='Connection error'))
    for k,value in [('refusal','declined'),('finish_reason','length'),('usage',{}),('raw_response',{}),('returned_model','model'),('response_id','id'),('answer','content')]:
        assert any_returned(dict(status='unknown',**{k:value}))
    assert not (OUT/'checkpoint.sqlite3').exists() or (HERE/'RUN_AUTHORIZATION.json').exists()
    result = dict(utc=utc(),passed=True,cases=90,goal_source_payloads_exact=90,goal128_verified=90,
        source_final_renderer_exact=90, historical_formatter_count=sum(r['formatter'] is not None for r in history),
        original_formatter_core_branch_contract=True, conditions=180, exact_ROLE_only_contracts=90,
        all_stage_usage_models_stop_verified=True, strict_terminal_and_return_guard_contracts=True,
        first_arm_counts=dict(Counter(r['arm'] for r in task_order(data) if r['index']%2==0)),
        gold_loaded=False,new_API_requests=0,new_retrieval_or_GPU=0,checkpoint_created_by_dryrun=False,
        execution_authorized=(HERE/'RUN_AUTHORIZATION.json').exists(),protocol_sha256=sha(PROTOCOL))
    atomic_json(OUT/'DRYRUN.json', result)
    return result


def event(value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'events.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps({'utc':utc(),**value},ensure_ascii=False)+'\n')


def active(cache, prompt, cap, config):
    key=canonical(build_payload(config['generation'],prompt,cap))
    r=cache.db.execute('SELECT id,status,result FROM requests WHERE payload=?',(key,)).fetchone()
    return (r[0],{**(json.loads(r[2]) if r[2] else {}),'status':r[1]}) if r else (None,{})


def obtain(cache,ask,record,task,stage,prompt,cap,config):
    key=task['task_id']+'/'+stage
    saved=cache.get('stage',key)
    if saved:
        rid,result=active(cache,prompt,cap,config)
        assert rid==saved['request_id'] and normal(result,config['generation']['model'])
        assert saved['payload_sha256']==digest(build_payload(config['generation'],prompt,cap))
        return saved,None
    try:
        rid,answer,finish=ask('native_goal/'+key,prompt,cap)
        rid,result=active(cache,prompt,cap,config)
        if not normal(result,config['generation']['model']):
            raise RuntimeError('Held incomplete matching-model/three-usage receipt')
        value=dict(stage=stage,request_id=rid,answer=answer,finish_reason=finish,usage=result['usage'],
                   payload_sha256=digest(build_payload(config['generation'],prompt,cap)))
        cache.put('stage',key,value)
        return value,None
    except (RuntimeError,InterruptedError,ValueError,TypeError,AttributeError) as exc:
        rid,result=active(cache,prompt,cap,config)
        reason=algorithm_failure(result,config['generation']['model'])
        if reason:
            value=dict(stage=stage,request_id=rid,answer=result.get('answer',''),finish_reason=result.get('finish_reason'),
                usage=result['usage'],payload_sha256=digest(build_payload(config['generation'],prompt,cap)),terminal_reason=reason)
            return value,reason
        raise RuntimeError('Held technical stage '+stage+'; '+type(exc).__name__) from None


def execute_condition(cache,ask,record,task,config,examples):
    """Injected execution used by both paid runner and isolated fake contracts."""
    stages=[]
    payload=templates.goal_payload(record,task['arm'])
    goal,terminal=obtain(cache,ask,record,task,'goal',payload['messages'][0]['content'],128,config)
    stages.append(goal)
    formatted=goal['answer'].strip()
    trigger=not terminal and templates.needs_formatter(formatted)
    stage='goal'
    if trigger:
        stage='formatter'
        fmt,terminal=obtain(cache,ask,record,task,stage,templates.formatter_prompt(record,formatted),128,config)
        stages.append(fmt)
        formatted=fmt['answer'].strip()
    if not terminal:
        stage='final'
        prompt,context,helper=templates.final_prompt(record,formatted,config,examples)
        final,terminal=obtain(cache,ask,record,task,stage,prompt,64,config)
        stages.append(final)
    return {**task,'status':'algorithm_failure' if terminal else 'success',
        'answer':'' if terminal else stages[-1]['answer'],'goal_initial_answer':goal['answer'],
        'goal_answer_used':formatted if stage=='final' or not terminal else None,'formatter_triggered':trigger,
        'stages':stages,'terminal_reason':terminal,'terminal_stage':stage if terminal else None,'utc':utc()}


def run():
    check()
    authority=read(HERE/'RUN_AUTHORIZATION.json')
    assert authority.get('approved') is True and authority.get('experiment')==HERE.name
    assert authority.get('protocol_sha256')==sha(PROTOCOL)
    assert any(d.get('id')==authority.get('decision_id') for d in rows(PLAN/'DECISIONS.jsonl'))
    from p02_multigrain_timing_cachefix_v1.core.cache import Cache,run_lock
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    config,examples=read(PACKAGE/'config.json'),read(PACKAGE/'prompts/final_examples.json')
    data={r['id']:r for r in rows(INPUTS)}
    class NetworkEvents:
        def event(self,name,**fields):event(dict(event=name,**fields))
    with run_lock(OUT),Cache(OUT/'checkpoint.sqlite3') as cache,closing(NoCapBudget(config,OUT)) as budget, \
        temporary_api_key('api_code_qwen.txt',allowed_root=ROOT,provider='dashscope'), \
        provider_dns(urlsplit(config['generation']['base_url']).hostname,NetworkEvents()):
        identity=dict(protocol_sha256=sha(PROTOCOL),inputs_sha256=sha(INPUTS),order_sha256=sha(ORDER))
        old=cache.get('meta','identity')
        if old is None:
            assert cache.db.execute('SELECT count(*) FROM requests').fetchone()[0]==0 and not cache.all('outcome')
            cache.put('meta','identity',identity)
        else:assert old==identity
        stop=lambda:(OUT/'STOP_REQUESTED').exists()
        recent={s:[] for s in ('goal','formatter','final')}
        for oldoutcome in sorted(cache.all('outcome').values(),key=lambda r:r['utc']):
            for s in oldoutcome['stages']:
                prior=data[oldoutcome['id']]['historical_stage_usage'][s['stage']]
                if prior:recent[s['stage']].append((cost(prior),cost(s['usage'])))
        with closing(SubmissionAwareStrictAsk(cache,None,config,budget,event,maximum_attempts=5,recover_unknown=True,stop=stop)) as ask:
            for task in rows(ORDER):
                if cache.get('outcome',task['task_id']) is not None:continue
                if stop():break
                record=data[task['id']]
                try:
                    outcome=execute_condition(cache,ask,record,task,config,examples)
                    cache.put('outcome',task['task_id'],outcome)
                    for s in outcome['stages']:
                        oldusage=record['historical_stage_usage'][s['stage']]
                        if oldusage:
                            reference,actual=cost(oldusage),cost(s['usage'])
                            recent[s['stage']].append((reference,actual))
                            reasons=[]
                            if reference and actual>3*reference:reasons.append('single_stage_above3x_paired_historical')
                            recent_stage=recent[s['stage']]
                            if len(recent_stage)>=20 and sum(a for _,a in recent_stage[-20:])>1.5*sum(b for b,_ in recent_stage[-20:]):reasons.append('recent20_stage_above1.5x_paired_historical')
                            if reasons:event(dict(event='expense_alert',task_id=task['task_id'],stage=s['stage'],reasons=reasons))
                        elif s['stage']=='formatter':event(dict(event='new_formatter_without_historical_pair',task_id=task['task_id'],cost_CNY=cost(s['usage'])))
                    event(dict(event='condition_terminal',task_id=task['task_id'],status=outcome['status']))
                    print(json.dumps(dict(event='condition_terminal',task_id=task['task_id'],status=outcome['status']),ensure_ascii=True),flush=True)
                except RuntimeError as exc:
                    hold={**task,'status':'held_technical','reason':str(exc),'utc':utc()}
                    atomic_json(OUT/'held'/(digest(task['task_id'])+'.json'),hold)
                    event(dict(event='held_technical',**hold))
                    continue
    return report()


def report():
    check()
    if not (OUT/'checkpoint.sqlite3').exists():
        return dict(status='prepared_not_started',planned_conditions=180,new_API_requests=0)
    with closing(ro(OUT/'checkpoint.sqlite3')) as db:
        outcomes=[json.loads(r[0]) for r in db.execute("SELECT value FROM data WHERE kind='outcome'")]
        request_states=dict(db.execute('SELECT status,count(*) FROM requests GROUP BY status'))
        retired_states=dict(db.execute('SELECT status,count(*) FROM retired_requests GROUP BY status'))
    expected={r['task_id'] for r in rows(ORDER)}
    assert len(outcomes)==len({r['task_id'] for r in outcomes}) and all(r['task_id'] in expected for r in outcomes)
    global SCORERS
    if SCORERS is None:
        from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
        SCORERS={ds:metric(ds) for ds in DATASETS}
    refs={r['id']:r for r in rows(HERE/'references90.local.frozen.jsonl')}
    data={r['id']:r for r in rows(INPUTS)}
    details=[]
    for outcome in outcomes:
        s=SCORERS[outcome['dataset']](outcome['answer'],refs[outcome['id']]) if outcome['status']=='success' else dict(f1=0.,em=0.)
        u={k:data[outcome['id']]['frozen_prefix_usage'][k]+sum(r['usage'][k] for r in outcome['stages']) for k in ('prompt_tokens','completion_tokens','total_tokens')}
        details.append({**outcome,**s,'full_flow_logical_usage':u,'full_flow_logical_estimated_CNY':cost(u),
                        'new_stages_attributed_CNY':sum(cost(r['usage']) for r in outcome['stages'])})
    def summarize(arm,ds=None,ordinary_only=False):
        planned={i for i,item in data.items() if(ds is None or item['dataset']==ds) and(not ordinary_only or bool(item['supporting_questions']))}
        r=[s for s in details if s['arm']==arm and s['id'] in planned]
        n=len(planned);m=n-len(r);f=sum(s['f1'] for s in r);e=sum(s['em'] for s in r)
        return dict(planned=n,quality_terminal=len(r),held_or_unstarted=m,algorithm_failures=sum(s['status']=='algorithm_failure' for s in r),
            F1=f/n if not m else None,EM=e/n if not m else None,F1_fixed_denominator_bounds=[f/n,(f+m)/n],EM_fixed_denominator_bounds=[e/n,(e+m)/n],
            formatter_triggered=sum(s['formatter_triggered'] for s in r),full_flow_logical_prompt_tokens=sum(s['full_flow_logical_usage']['prompt_tokens'] for s in r),
            full_flow_logical_completion_tokens=sum(s['full_flow_logical_usage']['completion_tokens'] for s in r),full_flow_logical_estimated_CNY=sum(s['full_flow_logical_estimated_CNY'] for s in r))
    idx={(r['id'],r['arm']):r for r in details}
    pairs=[dict(id=i,dataset=data[i]['dataset'],L0=idx[i,'L0']['f1'],L1=idx[i,'L1']['f1'],delta_F1=idx[i,'L1']['f1']-idx[i,'L0']['f1'],
                delta_EM=idx[i,'L1']['em']-idx[i,'L0']['em']) for i in data if(i,'L0')in idx and(i,'L1')in idx]
    with closing(ro(read(PACKAGE/'config.json')['budget']['ledger'])) as ledger:
        actual,reserve=ledger.execute("SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace=?",('p02_prompt_research_v1_20261022/'+str(OUT.resolve()),)).fetchone()
    historic=[SCORERS[h['dataset']](h['final']['result']['answer'].strip(),refs[h['id']]) for h in rows(HISTORY)]
    completed={r['task_id'] for r in outcomes}
    holds=[read(p) for p in (OUT/'held').glob('*.json')] if (OUT/'held').exists() else []
    held_ids=sorted({r['task_id'] for r in holds if r['task_id'] not in completed})
    value=dict(utc=utc(),status='completed' if len(outcomes)==180 else 'partial',scope='Frozen-upstream native-goal128 complete-RAG final quality development contrast; not fresh upstream/timing',
        planned_conditions=180,quality_terminal=len(outcomes),held_technical_tasks=len(held_ids),held_task_ids=held_ids,
        not_started_tasks=180-len(outcomes)-len(held_ids),arms={a:summarize(a) for a in templates.ARMS},by_dataset={ds:{a:summarize(a,ds) for a in templates.ARMS} for ds in DATASETS},
        ordinary_present_sensitivity=dict(planned_cases=sum(bool(r['supporting_questions']) for r in data.values()),
            arms={a:summarize(a,ordinary_only=True) for a in templates.ARMS},
            by_dataset={ds:{a:summarize(a,ds,ordinary_only=True) for a in templates.ARMS} for ds in DATASETS},
            selection='Predeclared D-nonempty84, main90 including6goal-only is retained; no score-based selection'),
        historical_P02_reference_only=dict(F1=sum(x['f1'] for x in historic)/90,EM=sum(x['em'] for x in historic)/90),
        paired=dict(complete_pairs=len(pairs),delta_F1=sum(p['delta_F1'] for p in pairs)/90 if len(pairs)==90 else None,delta_EM=sum(p['delta_EM'] for p in pairs)/90 if len(pairs)==90 else None,
            improved=sum(p['delta_F1']>0 for p in pairs),tied=sum(p['delta_F1']==0 for p in pairs),harmed=sum(p['delta_F1']<0 for p in pairs)),
        actual_new_returned_usage_CNY=actual,unresolved_reserved_CNY=reserve,request_statuses=request_states,retired_request_statuses=retired_states,
        scorer_factory_calls_once_per_dataset=3,main_evaluation_remains_paused=True)
    jsonl(OUT/'DETAILS.jsonl',details);jsonl(OUT/'PAIRED.jsonl',pairs);atomic_json(OUT/'REPORT.json',value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','dry-run','report','run'))
    args=parser.parse_args()
    print(json.dumps({'prepare':prepare,'dry-run':dry_run,'report':report,'run':run}[args.action](),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
