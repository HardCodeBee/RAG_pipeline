"""Canonical-cache repair and controlled Top10 versus full-union comparison. No gold in execution.

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
from experiments.query_decomposition.p02_flow_simplify_v1_20261020 import pipeline
from experiments.query_decomposition.p02_flow_simplify_v1_20261020 import run as historical
from . import cache

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
OLD = OUTPUT / 'b0_strategy_round2_v1_20261020'
ARMS = ['timing_only', 'union20']
CACHE_LOCK, KEY_LOCK, CIRCUIT_LOCK = Lock(), Lock(), Lock()
KEYS = {}
ERRORS = 0
HISTORICAL_KEYS = set()


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
    parent = historical.prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['runtime300.frozen.jsonl', 'pilot60.runtime.frozen.jsonl', 'references300.local.frozen.jsonl', 'BASELINE_SOURCES.frozen.json']:
        source, target = historical.HERE / name, HERE / name
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            target.write_bytes(source.read_bytes())
    inputs = rows(HERE / 'runtime300.frozen.jsonl')
    assert len(inputs) == len({v['id'] for v in inputs}) == 300
    assert Counter(v['dataset'] for v in inputs) == {'hotpotqa':100, '2wiki':100, 'musique':100}
    for value in inputs:
        pipeline.safe_case(value)
    _, initial_cache = cache.catalog_with_audit(include_current=False)
    freeze(HERE / 'HISTORICAL_SUCCESS_INDEX.frozen.json', {
        'priority':initial_cache['priority'],
        'canonical_cases':initial_cache['canonical_cases'],
        'canonical_logical_calls':initial_cache['canonical_logical_calls'],
        'canonical_unique_payloads':initial_cache['canonical_unique_payloads'],
        'canonical_internal_response_conflicts':initial_cache['canonical_internal_response_conflicts'],
        'bindings_sha256':initial_cache['selected_receipt_bindings_sha256'],
        'bindings':initial_cache['selected_receipt_bindings']})
    pins = dict(parent['source_pins'])
    for name in ['run.py', 'cache.py', 'adapter.py', 'preflight.py', 'execute_all.py']:
        source = HERE / name
        pins[str(source.relative_to(ROOT))] = sha(source)
    plan = {'task':HERE.name, 'model':parent['model'], 'temperature':0, 'enable_thinking':False,
            'resources':parent['resources'], 'readout':parent['readout'], 'arms':ARMS,
            'full':300, 'pilot':60, 'workers':3, 'max_rounds':5,
            'parent_protocol_sha256':sha(historical.HERE / 'PROTOCOL.frozen.json'),
            'source_pins':pins,
            'input_hashes':{n:sha(HERE/n) for n in ['runtime300.frozen.jsonl','pilot60.runtime.frozen.jsonl','references300.local.frozen.jsonl','BASELINE_SOURCES.frozen.json','HISTORICAL_SUCCESS_INDEX.frozen.json']},
            'only_method_difference':'Original-first interleaving ID-deduplicated reader Top10 versus complete union<=20 at ordinary and goal readers',
            'fixed':['Same300 questions and R02/v12 decomposition','Root retrieval delayed until goal in both arms','All legacy child/rewrite/goal/final prompts and examples','Same original-query Top10 and helper final routing','Same model/temperature/thinking/max_tokens','Same corpus/retriever/ranking arithmetic','Goal pre-answer AND final answer retained in both arms'],
            'cache_policy':'Canonical frozen p02/B0 full-payload successes take precedence globally, then deterministic historical exact successes, then new exact successes; no score-based choice or whole-case splice',
            'recovery_policy':'Only incomplete infrastructure cases in a fresh immutable round; never rerun terminal successes/algorithm failures; retain unknown liabilities',
            'global_cap_cny':300, 'dispatch_ceiling_cny':299.9, 'local_subcap_cny':8,
            'primary_metric':'Complete paired native final Answer F1 over300; per-dataset100; no cleanup of native predictions',
            'adoption_status':'User adopted timing_only only; union20 stays experimental regardless of result until decision recorded',
            'scope':'Controlled canonical-cache recomputation of exposed development300; unchanged successful calls reused, not all fresh inference'}
    freeze(path, plan)
    freeze(OUT / 'BUDGET_START.json', budget_snapshot())
    log('canonical_cache_protocol_frozen', authorization='User requested fix exact-input success reuse and repeat unchanged Top10 versus full union',
        parent_protocol_sha256=plan['parent_protocol_sha256'], arms=ARMS, budget=budget_snapshot())
    verify(plan)
    return plan


def verify(plan):
    assert sha(historical.HERE / 'PROTOCOL.frozen.json') == plan['parent_protocol_sha256']
    for p, h in plan['source_pins'].items():
        assert sha(ROOT / p) == h, p
    for p, h in plan['input_hashes'].items():
        assert sha(HERE / p) == h, p


def source_catalog(dataset):
    global HISTORICAL_KEYS
    values, stats = cache.catalog_with_audit(dataset)
    frozen = read(HERE / 'HISTORICAL_SUCCESS_INDEX.frozen.json')
    for binding in frozen['bindings']:
        got = values[binding['payload_sha256']]
        for key in ('request_id','source','cache_tier','result_sha256'):
            assert got[key] == binding[key], ('Frozen cache source changed',key)
    HISTORICAL_KEYS = {binding['payload_sha256'] for binding in frozen['bindings']}
    log('canonical_catalog_loaded',dataset=dataset,entries=len(values),
        canonical_logical_calls=stats['canonical_logical_calls'],
        bindings_sha256=stats['selected_receipt_bindings_sha256'],
        blocked_different_response_count=len(stats['blocked_different_response_receipts']))
    return values


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
            if total + amount > 299.9 or own + amount > 8:
                raise RuntimeError('Cumulative300 or local8 dispatch ceiling; no submission')
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
            if value is not None and 'payload' in value:
                assert value['payload'] == request, 'Exact input cache mismatch'
            if value is None:
                assert key not in HISTORICAL_KEYS, 'Known successful complete input must never be submitted again'
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
                         'status': own[0], 'result': json.loads(own[1]) if own[1] else None,
                         'payload': request, 'cache_origin': 'current_execution'}
                # A failed prefix is not silently resubmitted by another candidate in this execution.
                with CACHE_LOCK:
                    catalog.setdefault(key, deepcopy(value))
            raw = value.get('result') or {}
            call = {'qid': case['qid'], 'stage': stage, 'payload_sha256': key, 'request_id': value['request_id'],
                    'source': value['source'], 'status': value['status'], 'reused': reused,
                    'cache_origin': value.get('cache_tier',value.get('cache_origin','historical_exact')),
                    'full_payload_match_verified': True,
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


def run(plan, *, sample='300', arms=None, round_id=0, dataset=None, limit=None):
    from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
    from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
    from scripts.cli_support import temporary_api_key
    from src.generators.api_options import get_api_option
    from .adapter import Adapter
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
    parser.add_argument('--sample', choices=['pilot60', '300'], default='300')
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
