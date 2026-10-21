"""Budgeted actual RAG for isolated p02 simplifications. No gold in execution.

Only --run opens a provider client. Each infrastructure retry uses a new round;
successful method cases and exact successful payloads are retained unchanged.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import argparse
import gc
import hashlib
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
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget, WorkerStore, locked_json
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt, RecordedFailure
from experiments.query_decomposition.component_chain_opt_v2_20261019.policy import IntermediateLength
from . import pipeline

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
OLD = OUTPUT / 'b0_strategy_round2_v1_20261020'
ARMS = ['union20', 'discount20', 'single_final']
CACHE_LOCK, KEY_LOCK, CIRCUIT_LOCK = Lock(), Lock(), Lock()
KEYS = {}
ERRORS = 0


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    locked_json(path, value)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, 'Frozen state differs: ' + str(path)
    else:
        save(path, value)


def log(event, **fields):
    with (HERE / 'DECISIONS.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps({'utc': utc(), 'event': event, **fields}, ensure_ascii=False) + '\n')


def budget_snapshot():
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        all_cost = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
        own = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ?',
                         (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()
    return {'global_known_cny': all_cost[0], 'global_reserved_cny': all_cost[1], 'global_occupied_cny': sum(all_cost),
            'experiment_known_cny': own[0], 'experiment_reserved_cny': own[1], 'experiment_submissions': own[2],
            'global_cap_cny': 300, 'actual_bill_verified': False}


def baseline_paths():
    paths = sorted(OLD.glob('rounds/*/cases/*/p02_role_headers/*.json'))
    mapping = {}
    for p in paths:
        c = read(p)
        if c['status'].startswith('success'):
            mapping.setdefault(c['id'], p)
    assert len(mapping) == 300
    return mapping


def prepare():
    path = HERE / 'PROTOCOL.frozen.json'
    if path.exists():
        plan = read(path)
        verify(plan)
        return plan
    OUT.mkdir(parents=True, exist_ok=True)
    old_plan = read(ROOT / 'experiments/query_decomposition/b0_strategy_round2_v1_20261020/PROTOCOL.frozen.json')
    runtime_source = ROOT / 'experiments/query_decomposition/b0_strategy_round2_v1_20261020/runtime300.frozen.jsonl'
    reference_source = ROOT / 'experiments/query_decomposition/b0_six_single_changes_v1_20261020/references300.local.frozen.jsonl'
    runtime = rows(runtime_source)
    assert len(runtime) == len({c['id'] for c in runtime}) == 300
    assert Counter(c['dataset'] for c in runtime) == {'hotpotqa': 100, '2wiki': 100, 'musique': 100}
    for c in runtime:
        pipeline.safe_case(c)
    # Reference bytes are copied for the separate offline scorer, not parsed by generation.
    for name, source in [('runtime300.frozen.jsonl', runtime_source), ('references300.local.frozen.jsonl', reference_source)]:
        target = HERE / name
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            target.write_bytes(source.read_bytes())
    pilot = []
    for ds in ['hotpotqa', '2wiki', 'musique']:
        values = sorted((c for c in runtime if c['dataset'] == ds),
                        key=lambda c: hashlib.sha256(('p02-flow-simplify-pilot60|' + c['id']).encode()).hexdigest())
        pilot.extend(values[:20])
    pilot_path = HERE / 'pilot60.runtime.frozen.jsonl'
    text = ''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in pilot)
    if pilot_path.exists():
        assert pilot_path.read_text(encoding='utf8') == text
    else:
        pilot_path.write_text(text, encoding='utf8', newline='\n')
    sources = [{'id': cid, 'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for cid, p in baseline_paths().items()]
    freeze(HERE / 'BASELINE_SOURCES.frozen.json', sources)
    resources = read(ROOT / 'experiments/query_decomposition/b0_six_single_changes_v1_20261020/PROTOCOL.frozen.json')['resources']
    pins = dict(old_plan['source_pins'])
    new_files = [HERE / n for n in ['run.py', 'pipeline.py', 'retrieval_adapter.py', 'offline_audit.py', 'score.py', 'PLAN.proposed.txt']]
    for p in new_files:
        pins[str(p.relative_to(ROOT))] = sha(p)
    plan = {'task': HERE.name, 'baseline': 'p02_role_headers', 'baseline_F1_300': .7072417314845074,
            'model': old_plan['model'], 'temperature': 0, 'enable_thinking': False, 'resources': resources,
            'readout': old_plan['readout'], 'arms': ARMS, 'pilot': 60, 'full': 300,
            'pilot_selection': '20/dataset; SHA256(p02-flow-simplify-pilot60|id); outcome-blind; no quality gate to skip full300',
            'fixed': ['R02/v12 decomposition', 'legacy child/rewrite prompts', 'p02 final instructions/examples/P1', 'retriever/corpus/model/settings'],
            'changes': {'timing_only': 'Delay root retrieval; exact prompt/output replay, no paid calls',
                        'union20': 'Original-first interleaving, ID-deduplicated full union, <=20; goal and final retained',
                        'discount20': 'Same full union; equal sum 1/log2(rank+1) ordering',
                        'single_final': 'No appended-goal answer; final reader gets ordinary QA plus full goal dual-route raw passages'},
            'global_cap_cny': 300, 'dispatch_ceiling_cny': 299.9, 'local_subcap_cny': 25,
            'workers': 3, 'max_rounds': 2, 'source_pins': pins,
            'input_hashes': {n: sha(HERE / n) for n in ['runtime300.frozen.jsonl', 'pilot60.runtime.frozen.jsonl', 'references300.local.frozen.jsonl', 'BASELINE_SOURCES.frozen.json']},
            'cache': 'Exact successful full payload only; frozen receipt first, no score selection. Unknown liabilities retained.',
            'recovery': 'One new immutable infrastructure round; no successful/algorithm-failed case rerun',
            'scope': 'Exposed development300; all scores are downstream native Answer F1. No new training or remote decomposition.'}
    freeze(path, plan)
    freeze(OUT / 'BUDGET_START.json', budget_snapshot())
    log('execution_protocol_prepared', authorization='User: 可以 开始执行; annotation: 那就全部保留试一下', arms=ARMS,
        sample=300, pilot=60, frozen_baseline_unchanged=True)
    return plan


def verify(plan):
    for p, h in plan['source_pins'].items():
        assert sha(ROOT / p) == h, p
    for p, h in plan['input_hashes'].items():
        assert sha(HERE / p) == h, p


def source_catalog(dataset):
    """Read only successful receipts; never pick an answer based on its score."""
    db_ids = defaultdict(set)
    for record in read(HERE / 'BASELINE_SOURCES.frozen.json'):
        if not record['id'].startswith(dataset + ':'):
            continue
        c = read(ROOT / record['path'])
        cases = [c] + [read(ROOT / c[k]) for k in ['source_prefix', 'parent_source']]
        for value in cases:
            for call in value.get('logical_calls', []):
                source = call.get('source')
                rid = call.get('request_id')
                if source and rid:
                    db_ids[(ROOT / source).resolve()].add(rid)
    result = {}
    # Exact receipts from prior candidate runs are valid prefix caches as well.
    for p in sorted(OUT.glob(f'rounds/r*/methods/{dataset}/*/checkpoint.sqlite3')):
        db_ids.setdefault(p.resolve(), None)
    for p, ids in db_ids.items():
        with sqlite3.connect(p.as_uri() + '?mode=ro', uri=True) as db:
            if ids is None:
                entries = db.execute("SELECT id,payload,status,result FROM requests WHERE status='success'")
                groups = [list(entries)]
            else:
                values = sorted(ids)
                groups = [list(db.execute('SELECT id,payload,status,result FROM requests WHERE id IN (' + ','.join('?' for _ in values[i:i+500]) + ')', values[i:i+500])) for i in range(0, len(values), 500)]
            for group in groups:
                for rid, raw_payload, status, raw_result in group:
                    if status != 'success' or not raw_result:
                        continue
                    request, value = json.loads(raw_payload), json.loads(raw_result)
                    if value.get('status') != 'success':
                        continue
                    result.setdefault(digest(request), {'request_id': rid, 'status': status, 'result': value,
                                                       'source': str(p.relative_to(ROOT))})
    return result


def selected(case, arm):
    last = None
    for p in sorted(OUT.glob(f'rounds/r*/cases/{case["dataset"]}/{arm}/{case["qid"]}.json')):
        value = read(p)
        last = value
        if value['status'].startswith('success') or value['status'] == 'algorithm_failure':
            return value
    return last


class Budget(ParallelBudget):
    def __init__(self):
        super().__init__(OUTPUT / 'budget.sqlite3', 300)

    def reserve(self, rid, namespace, amount):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
            own = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?', (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()[0]
            if total + amount > 299.9 or own + amount > 25:
                raise RuntimeError('Cumulative300 or local25 dispatch ceiling; no submission')
            if self.db.execute('SELECT 1 FROM charges WHERE id=?', (rid,)).fetchone():
                raise RuntimeError('Existing liability cannot be submitted again')
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (rid, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


def run_case(case, arm, owner, adapter, client, catalog, plan, round_root):
    from experiments.query_decomposition.decomposition_error_expand60_v1_20261019.run import OneShot
    global ERRORS
    target = round_root / f'cases/{case["dataset"]}/{arm}/{case["qid"]}.json'
    assert not target.exists()
    store, budget = WorkerStore(owner), Budget()
    transport = OneShot(store, budget, client, plan['model'], case['qid'])
    completed, calls, retrievals = [], [], []
    started = time.perf_counter()

    def get(query):
        packet, hit, elapsed = adapter.get(case['qid'], query, 10)
        row = {'qid': case['qid'], 'query': query, 'cache_hit': hit, 'seconds': elapsed,
               'doc_ids': [d['doc_id'] for d in packet['docs']]}
        retrievals.append(row)
        store.event('logical_retrieval', **row)
        return packet

    def ask(stage, prompt, maximum, allow_final_truncation=False):
        global ERRORS
        request = payload(plan['model'], prompt, maximum)
        key = digest(request)
        rid = digest([store.namespace, request])
        with KEY_LOCK:
            mutex = KEYS.setdefault(key, Lock())
        with mutex:
            with CACHE_LOCK:
                value = deepcopy(catalog.get(key))
            reused = value is not None
            if value is None:
                own = store.db.execute('SELECT status,result FROM requests WHERE id=?', (rid,)).fetchone()
                reused = own is not None
                if own is None:
                    with CIRCUIT_LOCK:
                        if ERRORS >= 3:
                            raise RuntimeError('Dispatch deferred after three consecutive unknown provider outcomes')
                    try:
                        transport.ask(stage, prompt, maximum, allow_final_truncation=True)
                    except Exception:
                        if not store.db.execute('SELECT 1 FROM requests WHERE id=?', (rid,)).fetchone():
                            raise
                    own = store.db.execute('SELECT status,result FROM requests WHERE id=?', (rid,)).fetchone()
                    with CIRCUIT_LOCK:
                        ERRORS = ERRORS + 1 if own[0] in ['unknown', 'in_flight'] else 0
                value = {'request_id': rid, 'source': str((store.directory / 'checkpoint.sqlite3').relative_to(ROOT)),
                         'status': own[0], 'result': json.loads(own[1]) if own[1] else None}
                # A failed prefix is not silently resubmitted by another candidate in this execution.
                with CACHE_LOCK:
                    catalog.setdefault(key, deepcopy(value))
            raw = value.get('result') or {}
            call = {'qid': case['qid'], 'stage': stage, 'payload_sha256': key, 'request_id': value['request_id'],
                    'source': value['source'], 'status': value['status'], 'reused': reused,
                    'usage': raw.get('usage', {}), 'finish_reason': raw.get('finish_reason'), 'http_status': raw.get('http_status')}
            calls.append(call)
            store.put('logical_call', digest([case['qid'], stage, key]), call)
            return use_receipt(value, stage, allow_final_truncation)

    try:
        result = pipeline.execute(case, get, ask, arm=arm, readout=plan['readout'], record_node=completed.append)
    except Exception as exc:
        result = {**pipeline.safe_case(case), 'arm': arm, 'status': 'incomplete', 'prediction': None, 'nodes': completed,
                  'error_type': type(exc).__name__, 'reason': str(exc)[:500], 'retryable_infrastructure': False}
        if isinstance(exc, IntermediateLength):
            result.update(status='algorithm_failure', prediction='', failed_stage=exc.stage,
                          failed_request_id=exc.request_id, unpropagated_truncated_answer=exc.answer)
        elif isinstance(exc, RecordedFailure):
            last = calls[-1] if calls else {}
            result.update(failed_stage=exc.stage, failed_request_id=exc.request_id, request_status=exc.request_status,
                          http_status=last.get('http_status'), retryable_infrastructure=exc.request_status in ['unknown', 'in_flight']
                          or last.get('http_status') in [408, 409, 425, 429, 500, 502, 503, 504])
        elif 'Dispatch deferred' in str(exc):
            result['retryable_infrastructure'] = True
    finally:
        store.db.close()
        budget.db.close()
    result.update(logical_calls=calls, logical_retrievals=retrievals, finished_utc=utc(),
                  actual_seconds=time.perf_counter() - started)
    save(target, result)
    return {k: result.get(k) for k in ['id', 'dataset', 'arm', 'status', 'prediction', 'retryable_infrastructure', 'reason']}


def run(plan, *, sample='pilot60', arms=None, round_id=0, dataset=None, limit=None):
    from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
    from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
    from scripts.cli_support import temporary_api_key
    from src.generators.api_options import get_api_option
    from .retrieval_adapter import Adapter
    import httpx
    from openai import OpenAI
    global ERRORS
    verify(plan)
    audit = read(OUT / 'OFFLINE_AUDIT.json')
    assert audit['passed'], 'Offline audit must pass before API execution'
    assert round_id in range(plan['max_rounds'])
    arms = ARMS if arms is None else arms
    assert arms and len(set(arms)) == len(arms) and all(a in ARMS for a in arms)
    cases = rows(HERE / ('pilot60.runtime.frozen.jsonl' if sample == 'pilot60' else 'runtime300.frozen.jsonl'))
    if dataset:
        cases = [c for c in cases if c['dataset'] == dataset]
    if limit:
        cases = cases[:limit]
    pending = []
    for c in cases:
        for arm in arms:
            old = selected(c, arm)
            if old is None or (old['status'] == 'incomplete' and old.get('retryable_infrastructure', False)):
                target = OUT / f'rounds/r{round_id:03d}/cases/{c["dataset"]}/{arm}/{c["qid"]}.json'
                if target.exists():
                    raise RuntimeError('Immutable attempt exists; use a new recovery round')
                pending.append((c, arm))
    if not pending:
        return {'pending': 0, 'budget': budget_snapshot()}
    phase = OUT / f'phases/{sample}_r{round_id:03d}_{dataset or "all"}_{"_".join(arms)}'
    if phase.exists():
        raise RuntimeError('Execution phase already exists; inspect it rather than reopening')
    phase.mkdir(parents=True)
    save(phase / 'DISPATCH.frozen.json', {'utc': utc(), 'pending': len(pending), 'sample': sample, 'arms': arms,
                                         'round': round_id, 'dataset': dataset, 'budget_before': budget_snapshot()})
    round_root = OUT / f'rounds/r{round_id:03d}'
    ERRORS = 0
    completed_n = 0
    option = get_api_option(2)
    assert option['model'] == plan['model']
    identity = {'task': HERE.name, 'protocol_sha256': sha(HERE / 'PROTOCOL.frozen.json'), 'round': round_id}
    network = Store(phase / 'network', identity)
    try:
        with ExitStack() as stack:
            stack.enter_context(temporary_api_key('api_code_qwen.txt', allowed_root=ROOT, provider='dashscope'))
            host = urlsplit(option['base_url']).hostname
            stack.enter_context(provider_dns(host, network))
            with socket.create_connection((host, 443), timeout=15) as sock:
                with ssl.create_default_context().wrap_socket(sock, server_hostname=host):
                    network.event('TLS_preflight', paid_request=False)
            http = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0,
                limits=httpx.Limits(max_connections=3, max_keepalive_connections=3, keepalive_expiry=15)), timeout=90))
            api = stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'], base_url=option['base_url'],
                                            timeout=90, max_retries=0, http_client=http))
            client = StreamingClient(api)
            manifest = read(HERE / 'BASELINE_SOURCES.frozen.json')
            for ds in ['2wiki', 'musique', 'hotpotqa']:
                jobs = [(c, a) for c, a in pending if c['dataset'] == ds]
                if not jobs:
                    continue
                catalog = source_catalog(ds)
                retrieval_owner = Store(OUT / f'retrieval/{ds}', {'task': HERE.name, 'resources': plan['resources'][ds]})
                seed_paths = []
                for s in manifest:
                    if s['id'].startswith(ds + ':'):
                        c = read(ROOT / s['path'])
                        seed_paths.extend([ROOT / s['path'], ROOT / c['source_prefix']])
                adapter = Adapter(retrieval_owner, plan['resources'][ds], seed_cases=seed_paths,
                                  source_resources=plan['resources'][ds])
                try:
                    for arm in arms:
                        work = [c for c, a in jobs if a == arm]
                        if not work:
                            continue
                        # Owners are created once per phase before any workers; never recover in-flight in a worker.
                        owner = Store(round_root / f'methods/{ds}/{arm}', {**identity, 'arm': arm, 'dataset': ds})
                        try:
                            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                                futures = [pool.submit(run_case, c, arm, owner, adapter, client, catalog, plan, round_root) for c in work]
                                for future in as_completed(futures):
                                    value = future.result()
                                    completed_n += 1
                                    save(OUT / 'PROGRESS.json', {'utc': utc(), 'sample': sample, 'completed_this_dispatch': completed_n,
                                        'planned_this_dispatch': len(pending), 'latest': value, 'budget': budget_snapshot()})
                                    print(json.dumps(value, ensure_ascii=False), flush=True)
                        finally:
                            owner.export(full=True)
                            owner.db.close()
                finally:
                    adapter.close()
                    retrieval_owner.export()
                    retrieval_owner.db.close()
                    gc.collect()
        record = {'utc': utc(), 'attempted': completed_n, 'planned': len(pending), 'budget': budget_snapshot()}
        save(phase / 'TERMINAL.json', record)
        verify(plan)
        log('dispatch_completed', sample=sample, round=round_id, arms=arms, completed=completed_n, budget=record['budget'])
        return record
    finally:
        network.export(full=True)
        network.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--sample', choices=['pilot60', '300'], default='pilot60')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--dataset', choices=['hotpotqa', '2wiki', 'musique'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    plan = prepare()
    value = plan if args.prepare else run(plan, sample=args.sample, arms=args.arms, round_id=args.round,
                                         dataset=args.dataset, limit=args.limit)
    print(json.dumps(value if not args.prepare else {'prepared': True, 'arms': plan['arms'], 'sample': 300,
                                                   'pilot': 60, 'budget': budget_snapshot()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
