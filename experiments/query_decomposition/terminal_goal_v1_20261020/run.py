"""Frozen-union300 terminal-goal experiment; only --run makes paid calls.

Reuse ordinary QA, goal rewrite and goal retrieval exactly. Replace the old
goal answer with one conclusion+short-answer call, and remove legacy/final.
References are copied byte-for-byte for the separate offline scorer only.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import socket
import sqlite3
import ssl
from threading import Lock
import time
from urllib.parse import urlsplit

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, Store, digest, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget, WorkerStore
from experiments.query_decomposition.p02_flow_simplify_v1_20261020 import run as old
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
from . import candidate

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
ARM = 'terminal_goal'
DATASETS = ('2wiki', 'musique', 'hotpotqa')
ERROR_LOCK = Lock()
ERRORS = 0
read, rows, sha, save, freeze = old.read, old.rows, old.sha, old.save, old.freeze


def event(name, **fields):
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / 'DECISIONS.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps({'utc': utc(), 'event': name, **fields}, ensure_ascii=False) + '\n')


def budget_snapshot():
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        known, reserved = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
        own = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ?',
                         (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()
    return {'global_known_cny': known, 'global_reserved_cny': reserved,
            'global_occupied_cny': known + reserved, 'experiment_known_cny': own[0],
            'experiment_reserved_cny': own[1], 'experiment_submissions': own[2],
            'global_cap_cny': 300, 'actual_bill_verified': False}


def prepare():
    protocol = HERE / 'PROTOCOL.frozen.json'
    if protocol.exists():
        plan = read(protocol)
        verify(plan)
        return plan
    old_plan = old.prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['runtime300.frozen.jsonl', 'references300.local.frozen.jsonl']:
        source, target = old.HERE / name, HERE / name
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            target.write_bytes(source.read_bytes())
    runtime = rows(HERE / 'runtime300.frozen.jsonl')
    assert len(runtime) == len({c['id'] for c in runtime}) == 300
    assert Counter(c['dataset'] for c in runtime) == {'hotpotqa': 100, '2wiki': 100, 'musique': 100}
    sources = {}
    for p in sorted(old.OUT.glob('rounds/r*/cases/*/union20/*.json')):
        value = read(p)
        if value['status'].startswith('success'):
            sources.setdefault(value['id'], p)
    assert set(sources) == {c['id'] for c in runtime}
    manifest = [{'id': c['id'], 'path': str(sources[c['id']].relative_to(ROOT)), 'sha256': sha(sources[c['id']])} for c in runtime]
    freeze(HERE / 'UNION_SOURCES.frozen.json', manifest)
    checks, packet_sizes = [], Counter()
    for record in manifest:
        value = read(ROOT / record['path'])
        goal = value['nodes'][-1]
        assert goal['role'] == 'goal' and goal['request'] == value['question']
        assert goal['child_prompt'].endswith('Answer:')
        rendered = candidate.render_goal(goal['child_prompt'])
        assert rendered != goal['child_prompt']
        assert len(goal['reader_packet']['docs']) <= 20
        assert len({d['doc_id'] for d in goal['reader_packet']['docs']}) == len(goal['reader_packet']['docs'])
        packet_sizes[len(goal['reader_packet']['docs'])] += 1
        checks.append({'id': value['id'], 'original_goal_prompt_sha256': digest(goal['child_prompt']),
                       'new_goal_prompt_sha256': digest(rendered), 'reader_packet_sha256': digest(goal['reader_packet']),
                       'ordinary_nodes': len(value['nodes']) - 1})
    pins = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), HERE / 'candidate.py']}
    plan = {'task': HERE.name, 'arm': ARM, 'model': old_plan['model'], 'temperature': 0,
            'enable_thinking': False, 'max_tokens': 128, 'workers': 3, 'max_rounds': 4,
            'primary_control': 'union20', 'planned': 300, 'dataset_counts': dict(Counter(c['dataset'] for c in runtime)),
            'parent_protocol_sha256': sha(old.HERE / 'PROTOCOL.frozen.json'),
            'source_pins': pins,
            'input_hashes': {n: sha(HERE / n) for n in ['runtime300.frozen.jsonl', 'references300.local.frozen.jsonl', 'UNION_SOURCES.frozen.json']},
            'changes': ['Replace terminal goal output contract with complete Conclusion plus Final answer in one call',
                        'Mechanically extract Final answer; preserve Conclusion and raw output',
                        'Delete legacy/final and terminal-goal sentence_format; ordinary formatting unchanged'],
            'fixed': ['All ordinary nodes and their API receipts', 'Goal rewrite', 'Goal original-first full deduplicated union<=20',
                      'Original question and prior QA', 'Goal factual-evidence instructions and GOAL_RULE',
                      'Model/temperature/thinking/max128 and retrieval resources'],
            'parsing_policy': 'candidate.parse_output only; stop finish required; invalid/empty/two-line contract failure scores zero, no semantic repair',
            'infrastructure_policy': 'Keep unknown liabilities; no success rerun; only missing infrastructure cases in a new immutable round; per-dataset circuit',
            'adoption_policy': 'User accepts simplification without F1 gain; report paired losses and uncertainty rather than invent a noninferiority margin',
            'global_cap_cny': 300, 'dispatch_ceiling_cny': 299.9, 'shared_project_subcap_cny': 25,
            'cost_unit': {'prompt_token_cny': 0.0000008, 'completion_token_cny': 0.000002},
            'scope': 'Previously exposed development300; controlled downstream ablation, not fresh independent test'}
    freeze(protocol, plan)
    freeze(OUT / 'OFFLINE_AUDIT.json', {'passed': True, 'cases': 300, 'checks': checks,
                                      'packet_size_counts': dict(packet_sizes), 'no_new_retrieval': True})
    freeze(OUT / 'BUDGET_START.json', budget_snapshot())
    event('protocol_frozen', authorization='User 可以 following inverse goal simplification discussion',
          protocol_sha256=sha(protocol), samples=300, new_paid_stage='terminal_goal/answer')
    verify(plan)
    return plan


def verify(plan):
    assert sha(old.HERE / 'PROTOCOL.frozen.json') == plan['parent_protocol_sha256']
    old.verify(read(old.HERE / 'PROTOCOL.frozen.json'))
    for p, h in plan['source_pins'].items():
        assert sha(ROOT / p) == h, p
    for p, h in plan['input_hashes'].items():
        assert sha(HERE / p) == h, p
    for record in read(HERE / 'UNION_SOURCES.frozen.json'):
        assert sha(ROOT / record['path']) == record['sha256']


def selected(case):
    last = None
    for path in sorted(OUT.glob(f'rounds/r*/cases/{case["dataset"]}/{ARM}/{case["qid"]}.json')):
        value = read(path)
        last = value
        if value['status'] in ['success', 'algorithm_failure']:
            return value
    return last


class Budget(ParallelBudget):
    def __init__(self):
        super().__init__(OUTPUT / 'budget.sqlite3', 300)

    def reserve(self, rid, namespace, amount):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
            shared = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ? OR namespace LIKE ?',
                (str(OUT.relative_to(OUTPUT)) + '%', str(old.OUT.relative_to(OUTPUT)) + '%')).fetchone()[0]
            if total + amount > 299.9 or shared + amount > 25:
                raise RuntimeError('Global300/shared-project25 ceiling; request not submitted')
            if self.db.execute('SELECT 1 FROM charges WHERE id=?', (rid,)).fetchone():
                raise RuntimeError('Existing liability cannot be submitted again')
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (rid, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


def run_case(case, source, owner, client, plan, round_root):
    from experiments.query_decomposition.decomposition_error_expand60_v1_20261019.run import OneShot
    global ERRORS
    store, budget = WorkerStore(owner), Budget()
    started = time.perf_counter()
    old_case = read(ROOT / source['path'])
    goal = deepcopy(old_case['nodes'][-1])
    sid = goal['step_id']
    excluded = {f'legacy/answer/{sid}', f'legacy/sentence_format/{sid}', 'legacy/final'}
    calls = [dict(deepcopy(c), reused=True, reused_upstream=True) for c in old_case['logical_calls'] if c['stage'] not in excluded]
    prompt = candidate.render_goal(goal['child_prompt'])
    request = payload(plan['model'], prompt, plan['max_tokens'])
    rid = digest([store.namespace, request])
    base = {**case, 'arm': ARM, 'source_union_case': source['path'], 'source_union_sha256': source['sha256'],
            'nodes': deepcopy(old_case['nodes'][:-1]), 'goal_prompt': prompt,
            'goal_reader_packet': goal['reader_packet'], 'goal_inputs_reused': True,
            'source_goal_step_id': sid, 'terminal_goal_request_id': rid,
            'removed_stages': sorted(excluded), 'final_overall_call_removed': True,
            'logical_retrievals': deepcopy(old_case.get('logical_retrievals', []))}
    try:
        with ERROR_LOCK:
            if ERRORS >= 3:
                raise RuntimeError('Dispatch deferred after three consecutive unknown provider outcomes')
        transport = OneShot(store, budget, client, plan['model'], case['qid'])
        try:
            transport.ask('terminal_goal/answer', prompt, plan['max_tokens'], allow_final_truncation=True)
        except Exception:
            if not store.db.execute('SELECT 1 FROM requests WHERE id=?', (rid,)).fetchone():
                raise
        row = store.db.execute('SELECT status,result FROM requests WHERE id=?', (rid,)).fetchone()
        raw = json.loads(row[1]) if row[1] else {}
        with ERROR_LOCK:
            ERRORS = ERRORS + 1 if row[0] in ['unknown', 'in_flight'] else 0
        calls.append({'qid': case['qid'], 'stage': 'terminal_goal/answer', 'payload_sha256': digest(request),
                      'request_id': rid, 'source': str((store.directory / 'checkpoint.sqlite3').relative_to(ROOT)),
                      'status': row[0], 'reused': False, 'usage': raw.get('usage', {}),
                      'finish_reason': raw.get('finish_reason'), 'http_status': raw.get('http_status')})
        if row[0] != 'success':
            base.update(status='incomplete', prediction=None, raw_goal_output=raw.get('answer'),
                        request_status=row[0], http_status=raw.get('http_status'),
                        reason=raw.get('failure_class', row[0]),
                        retryable_infrastructure=row[0] in ['unknown', 'in_flight'] or raw.get('http_status') in [408,409,425,429,500,502,503,504])
        else:
            parsed = candidate.parse_output(raw['answer'], raw.get('finish_reason'))
            base.update(status='success' if parsed['valid'] else 'algorithm_failure',
                        prediction=parsed['prediction'] if parsed['valid'] else '',
                        conclusion=parsed['conclusion'], parse_result=parsed,
                        raw_goal_output=raw['answer'], finish_reason=raw.get('finish_reason'),
                        retryable_infrastructure=False)
            goal.update(answer=parsed['conclusion'], initial_answer=raw['answer'],
                        child_prompt=prompt, answer_request_id=rid, sentence_format_request_id=None,
                        terminal_short_answer=base['prediction'])
            base['nodes'].append(goal)
    except Exception as exc:
        base.update(status='incomplete', prediction=None, reason=str(exc)[:500], error_type=type(exc).__name__,
                    retryable_infrastructure='Dispatch deferred' in str(exc))
    finally:
        store.db.close()
        budget.db.close()
    base.update(logical_calls=calls, finished_utc=utc(), actual_seconds=time.perf_counter() - started)
    path = round_root / f'cases/{case["dataset"]}/{ARM}/{case["qid"]}.json'
    assert not path.exists()
    save(path, base)
    return {k: base.get(k) for k in ['id', 'dataset', 'status', 'prediction', 'reason']}


def inspect():
    counts = {}
    for ds in DATASETS:
        vals = [selected(c) for c in rows(HERE / 'runtime300.frozen.jsonl') if c['dataset'] == ds]
        counts[ds] = dict(Counter((v or {}).get('status', 'unstarted') for v in vals))
    return {'utc': utc(), 'by_dataset': counts, 'scorable': sum(v.get('success',0)+v.get('algorithm_failure',0) for v in counts.values()), 'budget': budget_snapshot()}


def run_dataset(plan, round_id, dataset):
    from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
    from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
    from scripts.cli_support import temporary_api_key
    from src.generators.api_options import get_api_option
    import httpx
    from openai import OpenAI
    global ERRORS
    verify(plan)
    runtime = rows(HERE / 'runtime300.frozen.jsonl')
    pending = []
    for c in runtime:
        if c['dataset'] != dataset:
            continue
        prev = selected(c)
        if prev is None or (prev['status'] == 'incomplete' and prev.get('retryable_infrastructure')):
            pending.append(c)
    if not pending:
        return
    root = OUT / f'rounds/r{round_id:03d}'
    phase = OUT / f'phases/r{round_id:03d}_{dataset}'
    assert not phase.exists(), 'Immutable phase exists; use next round'
    phase.mkdir(parents=True)
    freeze(phase / 'DISPATCH.frozen.json', {'utc': utc(), 'ids': [c['id'] for c in pending], 'budget': budget_snapshot(), 'protocol_sha256': sha(HERE / 'PROTOCOL.frozen.json')})
    identity = {'task': HERE.name, 'protocol_sha256': sha(HERE / 'PROTOCOL.frozen.json'), 'round': round_id, 'dataset': dataset}
    network = Store(phase / 'network', identity)
    option = get_api_option(2)
    assert option['model'] == plan['model']
    ERRORS = 0
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key('api_code_qwen.txt', allowed_root=ROOT, provider='dashscope'))
            host = urlsplit(option['base_url']).hostname
            stack.enter_context(provider_dns(host, network))
            with socket.create_connection((host,443), timeout=15) as sock:
                with ssl.create_default_context().wrap_socket(sock, server_hostname=host):
                    network.event('TLS_preflight', paid_request=False)
            http = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0,
                limits=httpx.Limits(max_connections=3,max_keepalive_connections=3,keepalive_expiry=15)),timeout=90))
            api = stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'],base_url=option['base_url'],timeout=90,max_retries=0,http_client=http))
            client = StreamingClient(api)
            owner = Store(root / f'methods/{dataset}/{ARM}', identity)
            manifest = {s['id']:s for s in read(HERE / 'UNION_SOURCES.frozen.json')}
            try:
                with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                    futures = [pool.submit(run_case,c,manifest[c['id']],owner,client,plan,root) for c in pending]
                    for future in as_completed(futures):
                        value = future.result()
                        save(OUT / 'PROGRESS.json', inspect())
                        print(json.dumps(value,ensure_ascii=False),flush=True)
            finally:
                owner.export(full=True)
                owner.db.close()
        save(phase / 'COMPLETE.json', inspect())
    finally:
        network.export(full=True)
        network.db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--inspect',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--start-round',type=int,default=0)
    args=parser.parse_args()
    plan=prepare()
    if args.prepare or args.inspect:
        print(json.dumps(inspect(),ensure_ascii=False),flush=True)
        return
    lock=OUT / 'ACTIVE.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        with os.fdopen(descriptor,'w') as stream:
            stream.write(str(os.getpid()))
        for r in range(args.start_round,plan['max_rounds']):
            before=inspect()['scorable']
            for ds in DATASETS:
                run_dataset(plan,r,ds)
            current=inspect()
            event('round_complete',round=r,**current)
            if current['scorable']==300:
                save(OUT / 'EXECUTION_COMPLETE.json',current)
                break
            if current['scorable']==before and r>args.start_round:
                save(OUT / 'WAITING_FOR_INSPECTION.json',current)
                break
        print(json.dumps(inspect(),ensure_ascii=False),flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
