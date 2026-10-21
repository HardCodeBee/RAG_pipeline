"""Fresh dev90 full RAG; no reference answers enter this runner.

Offline contract/replay check (no provider/retriever calls)::
    python -m experiments.query_decomposition.b0_strategy_round2_v1_20261020.fresh_runner --offline

After R02 has produced runtime_dev90.frozen.jsonl, freeze execution inputs::
    python -m experiments.query_decomposition.b0_strategy_round2_v1_20261020.fresh_runner --prepare

Only an explicit --run opens network clients. Run the parent first, then one
selected candidate, e.g. --run --round 0 --arms parent_g01 p02_role_headers.
A new, previously uncreated arm may be added to the same round later; an
existing method namespace may only recover in a new immutable round.
An infrastructure recovery uses a new immutable --round 1, retaining successful
payloads and all old unknown liabilities; successful/algorithm-failed cases
are never rerun. The runner deliberately has no reference/scoring reader.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import ssl
from threading import BoundedSemaphore, Lock
import time
from urllib.parse import urlsplit

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, digest, utc
from experiments.query_decomposition.component_chain_c08_c02_v1_20261019 import pipeline as b0

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name / 'fresh_dev90'
RUNTIME = HERE / 'fresh_validation/runtime_dev90.frozen.jsonl'
SIX = ROOT / 'experiments/query_decomposition/b0_six_single_changes_v1_20261020/PROTOCOL.frozen.json'
API_SLOTS = BoundedSemaphore(3)
CACHE_LOCK = Lock()
KEY_LOCK = Lock()
RETRIEVAL_LOCK = Lock()
CIRCUIT_LOCK = Lock()
KEYS = {}
ERRORS = 0


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strategy():
    return importlib.import_module('experiments.query_decomposition.b0_strategy_round2_v1_20261020.strategy')


def p1_readout():
    path = ROOT / 'experiments/query_decomposition/component_chain_opt_v2_20261019/PROTOCOL.frozen.json'
    return read(path)['phase_B']['P1_readout']


def safe_case(case):
    """Project a runtime row to the public query/R02 interface only."""
    forbidden = {'answer', 'answers', 'gold', 'gold_answer', 'supporting_facts',
                 'gold_documents', 'question_decomposition', 'prediction', 'f1'}
    if forbidden.intersection(case):
        raise ValueError('Reference/result fields are forbidden in fresh runtime')
    required = ['id', 'qid', 'dataset', 'question', 'supporting_questions']
    if any(key not in case for key in required):
        raise ValueError('Expected query-only R02 runtime fields: ' + ','.join(required))
    result = {key: deepcopy(case[key]) for key in required}
    if result['dataset'] not in {'hotpotqa', '2wiki', 'musique'}:
        raise ValueError('Unexpected dataset')
    if result['id'] != result['dataset'] + ':' + result['qid']:
        raise ValueError('Dataset/query identity mismatch')
    if not isinstance(result['question'], str) or not result['question'].strip():
        raise ValueError('Empty original question')
    if not isinstance(result['supporting_questions'], list) or not all(
            isinstance(q, str) and q.strip() for q in result['supporting_questions']):
        raise ValueError('R02 supporting_questions must be an ordered list of strings')
    if 'query' in case and case['query'] != result['question']:
        raise ValueError('Original query changed while importing R02')
    for key in ['type', 'hops']:
        if key in case:
            result[key] = case[key]
    return result


def execute_fresh(case, retrieve, ask, *, arm='parent_g01', readout=None,
                  record_node=None, strategy_module=None):
    """Wrap the immutable legacy executor without monkeypatching globals.

    The injected ask has the legacy (stage,prompt,max_tokens,**kwargs) interface.
    Retrieval, ordinary child answers, sentence formatter, and old rewrite
    stages are byte-for-byte delegated. Only the last child prompt and final
    render are transformed by strategy.py. Actual transformed prompts are
    reflected in callbacks and returned artifacts, including failed prefixes.
    """
    case = safe_case(case)
    module = strategy_module or strategy()
    if arm not in ['parent_g01', *module.ARMS]:
        raise ValueError('Unknown fresh arm: ' + arm)
    readout = p1_readout() if readout is None else readout
    goal_sid = len(case['supporting_questions']) + 1
    root_packet = None
    nodes = []
    goal_inputs = {}
    final_inputs = {}

    def get(query):
        nonlocal root_packet
        packet = retrieve(query)
        if root_packet is None:
            assert query == case['question'], 'Legacy executor root retrieval order differs'
            root_packet = deepcopy(packet)
        return packet

    def submit(stage, prompt, maximum, **kwargs):
        actual = prompt
        if stage == f'legacy/answer/{goal_sid}':
            assert maximum == 128
            actual = module.goal_prompt(prompt, arm)
            goal_inputs.update(raw_B0_goal_prompt=prompt, actual_goal_prompt=actual)
        elif stage == 'legacy/final':
            assert maximum == 64 and root_packet is not None
            assert len(nodes) == goal_sid and nodes[-1]['role'] == 'goal'
            actual, context = module.render_final(case['question'], nodes, root_packet, readout, arm)
            final_inputs.update(raw_B0_final_prompt=prompt, actual_final_prompt=actual, context=context)
        elif stage.startswith('legacy/rewrite/'):
            assert maximum == 256
        else:
            assert stage.startswith(('legacy/answer/', 'legacy/sentence_format/')) and maximum == 128
        return ask(stage, actual, maximum, **kwargs)

    def remember(node):
        copied = deepcopy(node)
        if copied['role'] == 'goal':
            assert copied['step_id'] == goal_sid
            assert copied['child_prompt'] == goal_inputs['raw_B0_goal_prompt']
            copied['raw_B0_child_prompt'] = copied['child_prompt']
            copied['child_prompt'] = goal_inputs['actual_goal_prompt']
        nodes.append(copied)
        if record_node is not None:
            record_node(deepcopy(copied))

    result = b0.execute(case, get, submit, variant='legacy', record_node=remember)
    assert final_inputs and result['final_prompt'] == final_inputs['raw_B0_final_prompt']
    result.update(arm=arm, nodes=nodes, final_prompt=final_inputs['actual_final_prompt'],
                  raw_B0_final_prompt=final_inputs['raw_B0_final_prompt'],
                  final_context=final_inputs['context'],
                  goal_prompt_change=goal_inputs, upstream_executor='frozen legacy b0.execute')
    # Render-only metadata; do not infer the role helper by splitting passage
    # text, as a real retrieved passage can contain arbitrary marker strings.
    passages = b0.passages(root_packet)
    context = final_inputs['context']
    assert context.endswith(passages)
    result['helper'] = context[:-len(passages)].removesuffix('\n\n') if passages else context
    assert result['history_all'] == [{k: n[k] for k in ['step_id', 'request', 'answer']} for n in nodes]
    return result


def runtime_cases():
    cases = [safe_case(c) for c in rows(RUNTIME)]
    assert len(cases) == len({c['id'] for c in cases}) == 90
    assert Counter(c['dataset'] for c in cases) == {'hotpotqa': 30, '2wiki': 30, 'musique': 30}
    return cases


def prepare():
    """Freeze source/runtime identities; deliberately never reads references."""
    from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
    if not RUNTIME.exists():
        raise FileNotFoundError('Awaiting R02 query-only runtime: ' + str(RUNTIME))
    cases = runtime_cases()
    module = strategy()
    paths = [Path(__file__), Path(module.__file__), Path(b0.__file__), SIX,
             Path(b0.legacy_child.__file__), Path(b0.legacy_rewrite.__file__),
             Path(b0.final_prompts.__file__),
             ROOT / 'experiments/query_decomposition/multigrain_prompt_v2_20261017/prompts.py',
             ROOT / 'experiments/query_decomposition/multigrain_prompt_v3_20261017/final_examples.json',
             ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json',
             ROOT / 'experiments/query_decomposition/baselines_same300_v1_20261020/batch_search.py',
             ROOT / 'experiments/query_decomposition/component_chain_opt_v2_20261019/PROTOCOL.frozen.json']
    plan = {'task': HERE.name, 'cohort': 'fresh dev90, 30 per dataset; not sealed150',
            'runtime': str(RUNTIME.relative_to(ROOT)), 'runtime_sha256': sha(RUNTIME),
            'cases': len(cases), 'arms_available': ['parent_g01', *module.ARMS],
            'model': read(SIX)['model'], 'temperature': 0, 'enable_thinking': False,
            'resources': read(SIX)['resources'], 'max_workers': 3, 'max_rounds': 2,
            'limits': {'rewrite': 256, 'child_and_goal': 128, 'final': 64},
            'readout': p1_readout(), 'global_cap_cny': 300, 'dispatch_ceiling_cny': 299.9,
            'local_subcap_cny': 10,
            'source_pins': {str(p.relative_to(ROOT)): sha(p) for p in paths},
            'cache': 'First successful exact full payload only; no score-based receipt selection.',
            'recovery': 'One immutable infrastructure-only recovery round; retain all unknown liabilities.',
            'hotpot_batch': 'Existing exact_packets; one encode per query and FAISS query_batch_size=1.',
            'scoring': 'Separate offline scorer; this runner does not read reference answers.'}
    assert plan['model'] == 'qwen-plus-2025-12-01'
    target = OUT / 'PROTOCOL.frozen.json'
    if target.exists():
        assert read(target) == plan, 'Frozen execution identity changed; create a new version'
    else:
        locked_json(target, plan)
    return plan


def verify(plan):
    assert sha(RUNTIME) == plan['runtime_sha256']
    for p, expected in plan['source_pins'].items():
        assert sha(ROOT / p) == expected, p


def offline():
    """Three genuine historical trajectory replays plus all-arm fake callbacks."""
    from experiments.query_decomposition.b0_six_single_changes_v1_20261020.receipts import receipt, baseline_packets, sources
    from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt
    from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
    prior = ROOT / 'experiments/query_decomposition/b0_targeted_optimization_v2_20261020'
    prior_out = OUTPUT / prior.name
    recorded_sources = sources()
    selected = []
    for ds in ['hotpotqa', '2wiki', 'musique']:
        entry = next(r for r in rows(prior / 'SOURCES.frozen.jsonl') if r['id'].startswith(ds + ':'))
        assert sha(ROOT / entry['base']) == entry['base_sha256']
        base = read(ROOT / entry['base'])
        known = next(read(p) for p in sorted((prior_out / 'rounds').glob(
            f'*/cases/{ds}/g01_goal_consistency/{base["qid"]}.json')) if read(p)['status'].startswith('success'))
        case = safe_case({k: base[k] for k in ['id', 'qid', 'dataset', 'question', 'supporting_questions']})
        packets = baseline_packets(base)
        prefix_calls = recorded_sources[entry['id']]['logical_calls']
        cache = {r['payload_sha256']: r for r in prefix_calls}
        cache.update({r['payload_sha256']: r for r in known['logical_calls']})
        asks = []
        def get(query):
            assert query in packets, 'Unexpected retrieval in frozen replay'
            return deepcopy(packets[query])
        def ask(stage, prompt, maximum, allow_final_truncation=False):
            request = payload('qwen-plus-2025-12-01', prompt, maximum)
            key = digest(request)
            assert key in cache, 'Changed supposedly frozen payload: ' + stage
            asks.append((stage, prompt, maximum, allow_final_truncation))
            return use_receipt(receipt(cache[key], request), stage, allow_final_truncation)
        value = execute_fresh(case, get, ask)
        assert value['prediction'] == known['prediction']
        assert value['final_prompt'] == known['final_prompt']
        assert value['final_context'] == known['final_context']
        assert value['nodes'][:-1] == base['nodes'][:-1]
        assert value['nodes'][-1]['child_prompt'] == known['nodes'][-1]['child_prompt']
        # For each candidate, replay fixed stage answers without asserting a
        # quality result; validate only wrapper plumbing and actual prompt logs.
        stage_answers = {}
        for r in [*prefix_calls, *known['logical_calls']]:
            stage_answers[r['stage']] = receipt(r)
        contracts = []
        for arm in strategy().ARMS:
            observed = []
            def fake(stage, prompt, maximum, allow_final_truncation=False):
                row = stage_answers.get(stage)
                if row is None:
                    # Historical g01 labels its own last stages differently;
                    # the exact matching frozen B0 receipt supplies a valid
                    # synthetic returned sentence for this contract only.
                    source_stage = next(s for s in stage_answers if s.endswith(stage.removeprefix('legacy/')))
                    row = stage_answers[source_stage]
                observed.append((stage, prompt, maximum))
                return use_receipt(row, stage, allow_final_truncation)
            completed = []
            test = execute_fresh(case, get, fake, arm=arm, record_node=completed.append)
            assert test['nodes'] == completed
            assert test['nodes'][:-1] == base['nodes'][:-1]
            assert observed[-1][1] == test['final_prompt']
            goal_stage = f'legacy/answer/{len(case["supporting_questions"]) + 1}'
            assert next(p for s, p, _ in observed if s == goal_stage) == test['nodes'][-1]['child_prompt']
            contracts.append(arm)
        selected.append({'id': case['id'], 'parent_exact_recorded_payloads': len(asks),
                         'parent_prediction_context_and_goal_exact': True, 'candidate_contracts': contracts})
    # Native duplicate handling is unchanged: the wrapper has no retrieval
    # implementation, and BatchSearch uses each query's original encoder/kernel.
    parity_path = OUTPUT / 'baselines_same300_v1_20261020/EXACT_BATCH_PARITY.frozen.json'
    parity = read(parity_path)
    batch_path = ROOT / 'experiments/query_decomposition/baselines_same300_v1_20261020/batch_search.py'
    assert parity['passed'] and parity['docs_scores_ids_and_blocks_bit_identical']
    assert parity['single_query_encoding'] and parity['new_faiss_query_batch_size'] == 1
    assert sha(batch_path) == parity['source_sha256']
    record = {'passed': True, 'checks': selected, 'historical_replay_not_fresh_inference': True,
              'retrieval_batch_prior_parity_verified': True, 'batch_source_sha256': sha(batch_path),
              'new_api_calls': 0, 'new_retrieval_calls': 0, 'runner_sha256': sha(__file__),
              'strategy_sha256': sha(Path(strategy().__file__))}
    locked_json(OUT / 'OFFLINE_CONTRACT.json', record)
    return record


def source_catalog():
    """Exact payloads from fresh runs only, including valid length receipts."""
    result = {}
    for path in sorted((OUT / 'rounds').glob('r[0-9][0-9][0-9]/methods/*/*/checkpoint.sqlite3')):
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            for rid, raw, rr in db.execute("SELECT id,payload,result FROM requests WHERE status='success' ORDER BY rowid"):
                value = json.loads(rr) if rr else {}
                if value.get('status') != 'success' or value.get('finish_reason') not in {'stop', 'length'}:
                    continue
                result.setdefault(digest(json.loads(raw)), {'request_id': rid, 'source': str(path.relative_to(ROOT)),
                                                          'status': 'success', 'result': value})
    return result


def selected(case, arm):
    last = None
    for path in sorted((OUT / 'rounds').glob(f'r[0-9][0-9][0-9]/cases/{case["dataset"]}/{arm}/{case["qid"]}.json')):
        last = read(path)
        if last['status'].startswith('success') or last['status'] == 'algorithm_failure':
            return last
    return last


def eligible(case, arm):
    value = selected(case, arm)
    return value is None or (value['status'] == 'incomplete' and value.get('retryable_infrastructure', False))


def budget_snapshot():
    path = OUTPUT / 'budget.sqlite3'
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        global_row = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
        own = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ?',
                         (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()
    return {'global_occupied_cny': sum(global_row), 'global_known_cny': global_row[0],
            'global_reserved_cny': global_row[1], 'fresh_known_cny': own[0],
            'fresh_reserved_cny': own[1], 'fresh_submissions': own[2], 'cap_cny': 300}


def run_case(case, arm, round_root, owner, retrieval_owner, retriever, search_service,
             client, plan, cached, budget_class):
    from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import WorkerStore, locked_json
    from experiments.query_decomposition.decomposition_error_expand60_v1_20261019.run import OneShot
    from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt, RecordedFailure
    from experiments.query_decomposition.component_chain_opt_v2_20261019.policy import IntermediateLength
    global ERRORS
    target = round_root / 'cases' / case['dataset'] / arm / (case['qid'] + '.json')
    assert not target.exists()
    store, retrstore, budget = WorkerStore(owner), WorkerStore(retrieval_owner), budget_class()
    transport = OneShot(store, budget, client, plan['model'], case['qid'])
    completed = []
    calls = []
    started = time.perf_counter()

    def get(query):
        if search_service is not None:
            packet, hit, elapsed = search_service.get(case['qid'], query, 10)
        else:
            with RETRIEVAL_LOCK:
                old, retriever.store = retriever.store, retrstore
                key = digest([case['qid'], query, 10, retriever.resources])
                hit = retrstore.get('retrieval', key) is not None
                begun = time.perf_counter()
                try:
                    packet = retriever.retrieve(case['qid'], query, 10)
                finally:
                    retriever.store = old
                elapsed = time.perf_counter() - begun
        store.event('logical_retrieval', qid=case['qid'], query=query, cache_hit=hit,
                    seconds=elapsed, doc_ids=[d['doc_id'] for d in packet['docs']])
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
                value = cached.get(key)
            reused = value is not None
            if value is not None:
                value = deepcopy(value)
            else:
                own = store.db.execute('SELECT status,result FROM requests WHERE id=?', (rid,)).fetchone()
                if own:
                    reused = True
                else:
                    with API_SLOTS:
                        with CIRCUIT_LOCK:
                            if ERRORS >= 3:
                                raise RuntimeError('Dispatch deferred after three consecutive unknowns')
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
                if value['status'] == 'success':
                    with CACHE_LOCK:
                        cached.setdefault(key, deepcopy(value))
            raw = value['result'] or {}
            call = {'qid': case['qid'], 'stage': stage, 'payload_sha256': key,
                    'request_id': value['request_id'], 'source': value['source'],
                    'status': value['status'], 'reused': reused, 'usage': raw.get('usage', {}),
                    'finish_reason': raw.get('finish_reason'), 'http_status': raw.get('http_status')}
            store.put('logical_call', digest([case['qid'], stage, key]), call)
            calls.append(call)
            return use_receipt(value, stage, allow_final_truncation)

    try:
        result = execute_fresh(case, get, ask, arm=arm, readout=plan['readout'], record_node=completed.append)
    except Exception as exc:
        result = {**case, 'arm': arm, 'status': 'incomplete', 'prediction': None, 'nodes': completed,
                  'error_type': type(exc).__name__, 'reason': str(exc)[:400], 'retryable_infrastructure': False}
        if isinstance(exc, IntermediateLength):
            result.update(status='algorithm_failure', prediction='', failed_stage=exc.stage,
                          failed_request_id=exc.request_id, unpropagated_truncated_answer=exc.answer)
        elif isinstance(exc, RecordedFailure):
            last = calls[-1] if calls else {}
            result.update(failed_stage=exc.stage, failed_request_id=exc.request_id,
                          request_status=exc.request_status, http_status=last.get('http_status'),
                          retryable_infrastructure=exc.request_status in ['unknown', 'in_flight'] or
                          last.get('http_status') in [408, 409, 425, 429, 500, 502, 503, 504])
        elif 'Dispatch deferred' in str(exc):
            result['retryable_infrastructure'] = True
    finally:
        store.db.close()
        retrstore.db.close()
        budget.db.close()
    result.update(logical_calls=calls, finished_utc=utc(), actual_seconds=time.perf_counter() - started)
    locked_json(target, result)
    return {k: result.get(k) for k in ['id', 'dataset', 'arm', 'status', 'prediction', 'retryable_infrastructure', 'reason']}


def run(plan, arms, round_id=0, dataset=None, limit=None):
    """Explicit network execution; max3 workers, immutable attempts."""
    from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store
    from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget, locked_json
    from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
    from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.offset_compatibility import install
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.retrieval import Retrieval
    from experiments.query_decomposition.baselines_same300_v1_20261020.batch_search import BatchSearch
    from scripts.cli_support import temporary_api_key
    from src.generators.api_options import get_api_option
    import httpx
    from openai import OpenAI
    global ERRORS
    verify(plan)
    contract = read(OUT / 'OFFLINE_CONTRACT.json')
    assert contract['passed'] and contract['runner_sha256'] == sha(__file__)
    assert contract['strategy_sha256'] == sha(Path(strategy().__file__))
    assert round_id in range(plan['max_rounds'])
    assert arms and len(arms) == len(set(arms)) and all(a in plan['arms_available'] for a in arms)
    if 'parent_g01' in arms:
        assert arms[0] == 'parent_g01', 'Run the parent before new candidate arms'
    cases = [c for c in runtime_cases() if dataset is None or c['dataset'] == dataset]
    if limit is not None:
        assert limit > 0
        cases = cases[:limit]
    pending = [(c, a) for a in arms for c in cases if eligible(c, a)]
    if not pending:
        return {'complete': True, 'pending': 0, 'budget': budget_snapshot()}
    round_root = OUT / 'rounds' / f'r{round_id:03d}'
    # A later selected candidate can be appended without using up the one
    # recovery round. Only entirely new method namespaces may be added; never
    # reopen an owner that could contain an unknown or successful prefix.
    for c, arm in pending:
        if (round_root / 'methods' / c['dataset'] / arm).exists():
            raise RuntimeError('Method attempt already exists; use a new immutable recovery round')
    phase_root = round_root / 'phases' / '__'.join(arms)
    if phase_root.exists():
        raise RuntimeError('Execution phase already exists; inspect it instead of reopening it')
    cached = source_catalog()
    option = get_api_option(2)
    assert option['model'] == plan['model']
    install()
    class Budget(ParallelBudget):
        def __init__(self):
            super().__init__(OUTPUT / 'budget.sqlite3', plan['global_cap_cny'])
        def reserve(self, rid, namespace, amount):
            self.db.execute('BEGIN IMMEDIATE')
            try:
                total = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
                own = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?',
                                      (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()[0]
                if total + amount > plan['dispatch_ceiling_cny'] or own + amount > plan['local_subcap_cny']:
                    raise RuntimeError('Cumulative300/fresh10 budget dispatch limit; no submission')
                if self.db.execute('SELECT 1 FROM charges WHERE id=?', (rid,)).fetchone():
                    raise RuntimeError('Existing liability cannot be submitted again')
                self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (rid, namespace, amount))
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
    identity = {'task': HERE.name, 'protocol_sha256': sha(OUT / 'PROTOCOL.frozen.json'), 'round': round_id}
    network = Store(phase_root / 'network', identity)
    ERRORS = 0
    done = 0
    locked_json(phase_root / 'DISPATCH.frozen.json', {'utc': utc(), 'arms': arms, 'dataset': dataset,
                'limit': limit, 'planned': len(pending), 'budget_before': budget_snapshot()})
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
            for ds in ['2wiki', 'musique', 'hotpotqa']:
                ds_jobs = [(c, a) for c, a in pending if c['dataset'] == ds]
                if not ds_jobs:
                    continue
                retrieval_owner = Store(OUT / 'retrieval' / ds, {
                    'task': HERE.name,
                    'protocol_sha256': identity['protocol_sha256'],
                    'dataset': ds,
                    'resources': plan['resources'][ds],
                })
                retriever = Retrieval(retrieval_owner, resources=plan['resources'][ds])
                service = BatchSearch(retrieval_owner, retriever) if ds == 'hotpotqa' else None
                try:
                    # Parent completed first: later candidates reuse exact
                    # ordinary/rewrite prefixes, not new sampled answers.
                    for arm in arms:
                        work = [c for c, a in ds_jobs if a == arm]
                        if not work:
                            continue
                        if arm != 'parent_g01':
                            # Do not implicitly use a new candidate as a retry
                            # vehicle for an unfinished parent's shared prefix.
                            unfinished = [c['id'] for c in work if selected(c, 'parent_g01') is None or
                                          not (selected(c, 'parent_g01')['status'].startswith('success') or
                                               selected(c, 'parent_g01')['status'] == 'algorithm_failure')]
                            if unfinished:
                                raise RuntimeError('Complete parent infrastructure recovery before candidate: ' + ','.join(unfinished[:3]))
                        owner = Store(round_root / 'methods' / ds / arm, {**identity, 'arm': arm, 'dataset': ds})
                        try:
                            with ThreadPoolExecutor(max_workers=plan['max_workers']) as pool:
                                jobs = [pool.submit(run_case, c, arm, round_root, owner, retrieval_owner,
                                                    retriever, service, client, plan, cached, Budget) for c in work]
                                for future in as_completed(jobs):
                                    summary = future.result()
                                    done += 1
                                    locked_json(OUT / 'PROGRESS.json', {'utc': utc(), 'completed_this_round': done,
                                                'planned_this_round': len(pending), 'latest': summary, 'budget': budget_snapshot()})
                                    print(json.dumps(summary, ensure_ascii=False), flush=True)
                        finally:
                            owner.export(full=True)
                            owner.db.close()
                finally:
                    if service is not None:
                        service.close()
                    if retriever.chunks is not None:
                        retriever.chunks.close()
                    retrieval_owner.export()
                    retrieval_owner.db.close()
                    del retriever
                    gc.collect()
        record = {'utc': utc(), 'attempted': done, 'planned': len(pending), 'budget': budget_snapshot()}
        locked_json(phase_root / 'TERMINAL.json', record)
        verify(plan)
        return record
    finally:
        network.export(full=True)
        network.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--offline', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--arms', nargs='+', default=['parent_g01'])
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--dataset', choices=['hotpotqa', '2wiki', 'musique'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.offline:
        value = offline()
    elif args.prepare:
        value = prepare()
    else:
        value = run(prepare(), args.arms, args.round, args.dataset, args.limit)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
