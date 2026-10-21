"""Formal all-1500 P02 execution. Gold is opened only by the separate scorer.

The standalone method supplies every prompt and all algorithm decisions. This
module supplies storage, exact caches, a budgeted transport and bounded retries.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import socket
import sqlite3
import ssl
import subprocess
import sys
import time

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store, OUTPUT, digest, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget, locked_json
from experiments.query_decomposition.p02_merge_cachefix_v1_20261021 import run as transport
from experiments.query_decomposition.p02_merge_cachefix_v1_20261021 import cache
from experiments.query_decomposition.p02_merge_cachefix_v1_20261021.adapter import Adapter as CacheFirst
from experiments.query_decomposition.p02_flow_simplify_v1_20261020.retrieval_adapter import RetrievalAdapter
from src.p02_multigrain_timing_cachefix_v1.core import pipeline
from .prepare import ROOT, HERE, OUT, read, rows, sha

METHOD = 'p02_multigrain_timing_cachefix_v1'
PACKAGE = ROOT/'src'/METHOD
DATASETS = ('hotpotqa', '2wiki', 'musique')
MODULE = 'experiments.query_decomposition.' + HERE.name + '.run'
INGESTED = {}


def save(path, value):
    locked_json(path, value)


def freeze(path, value):
    if Path(path).exists():
        assert read(path) == value, 'Frozen record changed: ' + str(path)
    else:
        save(path, value)


def event(kind, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'DECISIONS.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps({'utc': utc(), 'event': kind, **fields}, ensure_ascii=False)+'\n')


class Budget(ParallelBudget):
    def __init__(self):
        super().__init__(OUTPUT/'budget.sqlite3', 300)

    def reserve(self, rid, namespace, amount):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
            own = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?',
                                  (str(OUT.relative_to(OUTPUT))+'%',)).fetchone()[0]
            if total + amount > 299.9 or own + amount > 20:
                raise RuntimeError('Budget dispatch deferred: global300/local20 ceiling; no submission')
            if self.db.execute('SELECT 1 FROM charges WHERE id=?', (rid,)).fetchone():
                raise RuntimeError('Existing request liability cannot be submitted again')
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (rid, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


class Method:
    safe_case = staticmethod(pipeline.safe_case)

    @staticmethod
    def execute(case, retrieve, ask, *, arm, readout, record_node):
        assert arm == METHOD
        assert readout == read(PACKAGE/'config.json')['readout']
        return pipeline.execute(case, retrieve, ask, record_node=record_node)


class Adapter(CacheFirst):
    """Keep the tested cache-first get without seeding the unrelated dev300."""
    def __init__(self, owner, resources, seed_databases=()):
        self.offline_only = False
        RetrievalAdapter.__init__(self, owner, resources, device='cuda', max_batch=20,
                                  seed_databases=seed_databases)


def bind_transport():
    # Process-local adapters only. None of the reused modules are edited.
    transport.OUT = OUT
    transport.Budget = Budget
    transport.pipeline = Method
    transport.ERRORS = 0


def budget():
    bind = transport.OUT
    transport.OUT = OUT
    try:
        return transport.budget_snapshot()
    finally:
        transport.OUT = bind


def prepare():
    path = HERE/'PROTOCOL.frozen.json'
    if path.exists():
        plan = read(path)
        verify(plan)
        return plan
    config = read(PACKAGE/'config.json')
    replay = read(OUT/'PRE_RUN_REPLAY300.json')
    assert replay['passed'] and replay['counts']['cases'] == 300
    assert replay['all_prompts_outputs_and_packets_exact'] and replay['new_API_calls'] == 0
    amendment = read(HERE/'AMENDMENT001_ALL1500.user.frozen.json')
    assert amendment['exclude_or_replace_any_query'] is False
    qq = rows(HERE/'questions1500.frozen.jsonl')
    assert len(qq) == len({q['id'] for q in qq}) == 1500
    assert Counter(q['dataset'] for q in qq) == {ds: 500 for ds in DATASETS}
    _, audit = cache.catalog_with_audit(include_current=True)
    freeze(HERE/'SUCCESS_BINDINGS.frozen.json', {
        'priority': audit['priority'], 'bindings_sha256': audit['selected_receipt_bindings_sha256'],
        'bindings': audit['selected_receipt_bindings'],
        'reads_references_or_scores': False})
    files = [HERE/n for n in ('run.py', 'network.py', 'score.py', 'prepare.py')]
    files += list((PACKAGE/'core').glob('*.py')) + list((PACKAGE/'prompts').glob('*'))
    files += [PACKAGE/'config.json', PACKAGE/'FROZEN.json']
    files += [Path(transport.__file__), Path(cache.__file__)]
    from experiments.query_decomposition.p02_flow_simplify_v1_20261020 import retrieval_adapter
    from experiments.query_decomposition.decomposition_error_expand60_v1_20261019 import run as one_shot
    files += [Path(retrieval_adapter.__file__), Path(one_shot.__file__)]
    # The cleanup removed descriptive resource fields. Preserve the original
    # full resource identity for exact retrieval caches and the unchanged backend.
    old_resource_plan = HERE.parent/'p02_merge_cachefix_v1_20261021/PROTOCOL.frozen.json'
    resources = read(old_resource_plan)['resources']
    for ds in DATASETS:
        for section, current in [('corpus', config['resources'][ds]['corpus']),
                                 ('dense', config['resources'][ds]['dense']['bge'])]:
            original = resources[ds]['corpus'] if section == 'corpus' else resources[ds]['dense']['bge']
            assert all(original[k] == v for k, v in current.items()), 'Active retrieval settings changed'
    files += [old_resource_plan]
    plan = {
        'method': METHOD, 'planned': 1500, 'dataset_counts': {ds: 500 for ds in DATASETS},
        'cohort': 'All original formal1500; no exclusions or replacements',
        'model': config['generation']['model'], 'base_url': config['generation']['base_url'],
        'resources': resources, 'readout': config['readout'],
        'workers': 3, 'max_rounds': 4, 'global_cap_cny': 300,
        'dispatch_ceiling_cny': 299.9, 'local_subcap_cny': 20,
        'source_pins': {p.relative_to(ROOT).as_posix(): sha(p) for p in files if p.is_file()},
        'input_pins': {n: sha(HERE/n) for n in ('questions1500.frozen.jsonl',
            'references1500.local.frozen.jsonl', 'v12.txt', 'queries1500.id_query.frozen.jsonl',
            'REMOTE_INPUTS.frozen.json', 'AMENDMENT001_ALL1500.user.frozen.json', 'SUCCESS_BINDINGS.frozen.json')},
        'original_inputs_manifest_sha256': sha(HERE/'INPUTS.frozen.json'),
        'packaging_revision': 2,
        'packaging_revision_note': 'Concurrent standalone cleanup completed before execution; fresh 300-case exact prompt/output/packet replay passed. Initial package-manifest pin superseded by this explicit revision; no algorithm change.',
        'frozen_method_settings': ['R02/v12', 'delay root retrieval until goal',
            'legacy rewrite256', 'dual original-first interleave deduplicated reader10',
            'legacy complete child answer128', 'goal preanswer retained',
            'p02 role headers', 'final4shot/P1 short answer64', 'original BGE512 exact full-index search'],
        'scoring': 'Separate native final Answer F1/EM; algorithm failure=0, infrastructure missing kept missing, all1500 denominator and bounds retained',
        'recovery': 'Only retryable infrastructure failures in fresh immutable rounds; never rerun successful complete cases; reuse every exact successful full payload; unknown liabilities retained',
        'scope': 'P02 formal1500 only; do not resume other paused methods/ablations or alter baseline configuration',
        'new_training': False,
    }
    freeze(path, plan)
    freeze(OUT/'BUDGET_START.json', budget())
    event('formal_protocol_frozen', planned=1500, per_dataset=500, packaging_revision=2,
          cache_entries=audit['unique_success_payloads'], new_API_calls=0)
    verify(plan)
    return plan


def verify(plan):
    for p, h in plan['source_pins'].items():
        assert sha(ROOT/p) == h, 'Execution source changed: '+p
    for n, h in plan['input_pins'].items():
        assert sha(HERE/n) == h, 'Frozen input changed: '+n


def ingest():
    """Validate query-only GPU receipts; missing outputs are never fallback cases."""
    qq = rows(HERE/'questions1500.frozen.jsonl')
    identity_file = OUT/'remote_raw/IDENTITY.json'
    if not identity_file.exists():
        return [], []
    ii = read(identity_file)
    expected = {'decode_compat': 'explicit_generation_config_v002',
        'effective_do_sample': False, 'effective_num_beams': 1,
        'checkpoint_manifest_sha256': read(HERE/'REMOTE_INPUTS.frozen.json')['checkpoint_manifest_sha256'],
        'query_sha256': sha(HERE/'queries1500.id_query.frozen.jsonl'),
        'prompt_sha256': sha(HERE/'v12.txt'), 'do_sample': False, 'num_beams': 1,
        'max_new_tokens': 1024, 'seed': 1729, 'api_calls': 0}
    assert ii == expected, 'Unexpected remote decomposer identity'
    runtime, invalid = [], []
    prompt = (HERE/'v12.txt').read_text(encoding='utf8')
    for index, q in enumerate(qq, 1):
        rawpath = OUT/f'remote_raw/{index:03d}.json'
        if not rawpath.exists():
            continue
        signature = (rawpath.stat().st_size, rawpath.stat().st_mtime_ns)
        if index in INGESTED:
            prior = INGESTED[index]
            assert prior['signature'] == signature, 'Previously ingested decomposition changed'
            runtime.append(prior['case'])
            if prior['invalid']:
                invalid.append(prior['invalid'])
            continue
        raw = read(rawpath)
        assert raw['index'] == index and raw['id'] == q['id'] and raw['dataset'] == q['dataset']
        assert raw['query'] == q['question'] and raw['actual_prompt'] == prompt
        assert raw['identity_sha256'] == digest(ii), 'Remote identity binding differs'
        reason = None
        try:
            assert not raw.get('generation_error') and not raw.get('truncated')
            ss = json.loads(raw['raw_output'])
            assert isinstance(ss, list) and all(isinstance(s, str) and s.strip() for s in ss)
        except (AssertionError, ValueError, TypeError):
            ss, reason = [], 'invalid_array_or_incomplete_generation; original-goal-only fallback'
            invalid.append({'id': q['id'], 'reason': reason, 'raw_file': str(rawpath.relative_to(ROOT))})
        case = pipeline.safe_case({**q, 'supporting_questions': ss})
        freeze(OUT/f'runtime/{q["dataset"]}/{q["qid"]}.json', case)
        INGESTED[index] = {'signature': signature, 'case': case, 'invalid': invalid[-1] if reason else None}
        runtime.append(case)
    save(OUT/'INGEST_STATUS.json', {'utc': utc(), 'ready': len(runtime), 'planned': 1500,
        'by_dataset': dict(Counter(c['dataset'] for c in runtime)), 'mechanical_fallbacks': invalid,
        'excluded_queries': 0, 'semantic_repair': False, 'remote_identity': ii})
    return runtime, invalid


def selected(case):
    last = None
    for path in sorted(OUT.glob(f'rounds/r*/cases/{case["dataset"]}/{METHOD}/{case["qid"]}.json')):
        value = read(path)
        value['_selected_source'] = str(path.relative_to(ROOT))
        last = value
        if value['status'].startswith('success') or value['status'] == 'algorithm_failure':
            return value
    return last


def load_catalog():
    catalog, _ = cache.catalog_with_audit(include_current=True)
    initial = read(HERE/'SUCCESS_BINDINGS.frozen.json')['bindings']
    for row in initial:
        current = catalog[row['payload_sha256']]
        for key in ('request_id', 'source', 'result_sha256'):
            assert row[key] == current[key], 'Initial success receipt binding changed'
    transport.HISTORICAL_KEYS = {r['payload_sha256'] for r in initial}
    # Fresh rounds reuse successful local prefixes; failed/unknown requests are
    # not erased or declared free. Only explicitly authorized new-round retries.
    for path in sorted(OUT.glob('rounds/r*/methods/*/*/checkpoint.sqlite3')):
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
            for rid, rawpayload, status, rawresult in db.execute("SELECT id,payload,status,result FROM requests WHERE status='success' ORDER BY rowid"):
                request, result = json.loads(rawpayload), json.loads(rawresult)
                if result.get('status') == 'success' and result.get('finish_reason') in ('stop', 'length'):
                    catalog.setdefault(digest(request), {'payload': request, 'request_id': rid,
                        'status': status, 'result': result, 'source': str(path.relative_to(ROOT)),
                        'cache_tier': 'formal1500_prior_round'})
    return catalog


def record_progress(state, active=0):
    all_questions = rows(HERE/'questions1500.frozen.jsonl')
    counts = {ds: Counter() for ds in DATASETS}
    for q in all_questions:
        old = selected(q)
        counts[q['dataset']][old['status'] if old else 'not_completed'] += 1
    remote = read(OUT/'REMOTE_PROGRESS.json') if (OUT/'REMOTE_PROGRESS.json').exists() else {}
    value = {'utc': utc(), 'state': state, 'planned': 1500, 'active_cases': active,
        'by_dataset': {ds: dict(v) for ds, v in counts.items()},
        'decompositions_returned': remote.get('completed', 0), 'budget': budget()}
    save(OUT/'PROGRESS.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def run(plan):
    from .network import provider_dns
    from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
    from scripts.cli_support import temporary_api_key
    from src.generators.api_options import get_api_option
    import httpx
    from openai import OpenAI
    bind_transport()
    verify(plan)
    lockpath = OUT/'controller.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    # A Windows byte-range lock is released by the OS on process exit.
    import msvcrt
    with lockpath.open('a+b') as lock, ExitStack() as stack:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        option = get_api_option(2)
        assert option['model'] == plan['model'] and option['base_url'] == plan['base_url']
        identity = {'task': HERE.name, 'protocol_sha256': sha(HERE/'PROTOCOL.frozen.json')}
        netstore = Store(OUT/'network', identity)
        stack.callback(netstore.db.close)
        stack.enter_context(temporary_api_key('api_code_qwen.txt', allowed_root=ROOT, provider='dashscope'))
        host = option['base_url'].split('/')[2]
        stack.enter_context(provider_dns(host, netstore))
        with socket.create_connection((host, 443), timeout=15) as tcp:
            with ssl.create_default_context().wrap_socket(tcp, server_hostname=host):
                netstore.event('TLS_preflight', paid_request=False)
        http = stack.enter_context(httpx.Client(transport=httpx.HTTPTransport(retries=0,
            limits=httpx.Limits(max_connections=3, max_keepalive_connections=3, keepalive_expiry=15))))
        api = stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'], base_url=plan['base_url'],
            timeout=90, max_retries=0, http_client=http))
        client = StreamingClient(api)
        adapters = {}
        for ds in DATASETS:
            owner = Store(OUT/f'retrieval/{ds}', {**identity, 'dataset': ds})
            stack.callback(owner.db.close)
            old = OUTPUT/f'comprehensive_v1_20261015/campaign_api_v1/{ds}/formal/b00_rag_top10_v1/fixed/checkpoint.sqlite3'
            adapters[ds] = Adapter(owner, plan['resources'][ds], seed_databases=[old] if old.exists() else [])
            stack.callback(adapters[ds].close)
        catalog = load_catalog()
        for round_id in range(plan['max_rounds']):
            round_root = OUT/f'rounds/r{round_id:03d}'
            owners = {}
            for ds in DATASETS:
                owners[ds] = Store(round_root/f'methods/{ds}/{METHOD}', {**identity, 'round': round_id, 'dataset': ds})
            transport.ERRORS = 0
            try:
                event('round_started', round=round_id, budget=budget())
                running, considered, last_report, idle_since = {}, set(), 0., None
                with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                    while True:
                        runtime, _ = ingest()
                        now = time.monotonic()
                        if (OUT/'STOP_REQUESTED').exists():
                            event('local_stop_observed', round=round_id)
                            break
                        # Interleave ready datasets without changing their frozen membership.
                        ordered = sorted(runtime, key=lambda c: (int.from_bytes(c['id'].encode()[:2], 'big'), c['id']))
                        for case in ordered:
                            if len(running) >= plan['workers'] or transport.ERRORS >= 3:
                                break
                            if case['id'] in considered:
                                continue
                            old = selected(case)
                            terminal = old and (old['status'].startswith('success') or old['status'] == 'algorithm_failure')
                            target = round_root/f'cases/{case["dataset"]}/{METHOD}/{case["qid"]}.json'
                            if terminal or target.exists() or (old and not old.get('retryable_infrastructure')):
                                considered.add(case['id']); continue
                            verify(plan)
                            future = pool.submit(transport.run_case, case, METHOD, owners[case['dataset']],
                                adapters[case['dataset']], client, catalog, plan, round_root)
                            running[future] = case['id']; considered.add(case['id'])
                        if running:
                            done, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
                            for future in done:
                                cid = running.pop(future)
                                result = future.result()
                                # Pre-submit guards are infrastructure deferrals, not algorithm failures.
                                if (result.get('reason') or '').startswith('Budget dispatch deferred'):
                                    raise RuntimeError('Authorized budget dispatch ceiling reached')
                                event('case_finished', round=round_id, **result)
                            idle_since = None
                        elif transport.ERRORS >= 3:
                            event('provider_circuit_open', round=round_id, unknowns=transport.ERRORS)
                            break
                        elif len(runtime) == 1500:
                            break
                        else:
                            idle_since = idle_since or now
                            if now-idle_since > 1800:
                                raise RuntimeError('No new remote decomposition for30min; inspect remote progress')
                            time.sleep(5)
                        if now-last_report >= 60:
                            record_progress('evaluating', len(running))
                            subprocess.run([sys.executable, '-X', 'utf8', '-m', MODULE.rsplit('.', 1)[0]+'.score'],
                                cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
                            last_report = now
                    for future, cid in list(running.items()):
                        result = future.result()
                        event('case_finished', round=round_id, **result)
            finally:
                for owner in owners.values():
                    owner.export(); owner.db.close()
            record_progress('round_finished')
            runtime, _ = ingest()
            remaining = [c for c in runtime if not selected(c) or selected(c).get('retryable_infrastructure')]
            if (OUT/'STOP_REQUESTED').exists() or not remaining:
                break
            if round_id+1 < plan['max_rounds']:
                # One network-only check before a new immutable recovery round.
                time.sleep(20)
                with socket.create_connection((host, 443), timeout=15) as tcp:
                    with ssl.create_default_context().wrap_socket(tcp, server_hostname=host):
                        netstore.event('recovery_TLS_preflight', round=round_id+1, paid_request=False)
                catalog = load_catalog()
        record_progress('execution_finished')
        subprocess.run([sys.executable, '-X', 'utf8', '-m', MODULE.rsplit('.', 1)[0]+'.score'], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--ingest-only', action='store_true')
    args = parser.parse_args()
    if args.ingest_only:
        ready, bad = ingest(); print(json.dumps({'ready': len(ready), 'mechanical_fallbacks': len(bad)})); return
    plan = prepare()
    if args.run:
        try:
            run(plan)
        except BaseException as exc:
            save(OUT/'CONTROLLER_ERROR.json', {'utc': utc(), 'error_type': type(exc).__name__,
                'reason': str(exc)[:1500], 'successful_results_retained': True})
            raise
    else:
        print(json.dumps({'prepared': True, 'planned': 1500, 'new_API_calls': 0}))


if __name__ == '__main__':
    main()
