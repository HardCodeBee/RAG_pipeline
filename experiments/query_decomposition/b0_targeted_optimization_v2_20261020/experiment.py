"""Two isolated B0+P1 answer-layer experiments. Frozen upstream replay only.

No new decomposition, retrieval, model training, or changes to ancestor methods.
References are loaded only by summarize(), never by candidate rendering/calls.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import argparse
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

import httpx
import numpy as np
from openai import OpenAI

from scripts.cli_support import temporary_api_key
from src.generators.api_options import get_api_option
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, Store, digest, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget, WorkerStore, locked_json
from experiments.query_decomposition.hotpot_formal_eval_v1.common.provider_dns import provider_dns
from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingClient
from experiments.query_decomposition.decomposition_error_expand60_v1_20261019.run import OneShot
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt, RecordedFailure
from experiments.query_decomposition.component_chain_opt_v2_20261019.policy import IntermediateLength
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.receipts import receipt, baseline_case, sources
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.prepare import read, rows, sha, save, save_rows
from experiments.query_decomposition.component_chain_c08_c02_v1_20261019 import pipeline as b0
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
PREV_E = HERE.parent / 'b0_six_single_changes_v1_20261020'
PREV_O = OUTPUT / PREV_E.name
ARMS = ['g01_goal_consistency', 'f01_final_consistency']
FINAL_CHECK = ('Check the original question\'s comparison or combination against the supporting facts or values; '
               'do not copy an auxiliary conclusion that contradicts those facts or values.\n')
LOCK = Lock()
KEY_LOCK = Lock()
KEYS = {}
ERRORS = 0


def log(event, **fields):
    with (HERE / 'DECISIONS.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps({'utc': utc(), 'event': event, **fields}, ensure_ascii=False) + '\n')


def budget_snapshot():
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        total = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
        local = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ?', (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()
    return {'global_known_cny': total[0], 'global_reserved_cny': total[1], 'global_occupied_cny': sum(total),
            'new_known_cny': local[0], 'new_reserved_cny': local[1], 'new_submissions': local[2], 'cap_cny': 300}


def pilot_ids(cases):
    out = []
    for ds, count in [('hotpotqa', 33), ('2wiki', 33), ('musique', 34)]:
        cc = sorted((c for c in cases if c['dataset'] == ds),
                    key=lambda c: hashlib.sha256(('b0-targeted-20261020|' + c['id']).encode()).hexdigest())
        out.extend(c['id'] for c in cc[:count])
    assert len(out) == len(set(out)) == 100
    return out


def altered_goal_prompt(prompt):
    assert prompt.endswith('Answer:')
    return prompt[:-len('Answer:')] + b0.legacy_child.GOAL_RULE + '\n\nAnswer:'


def final_prompt(case, nodes, original, readout, arm):
    history = [{k: n[k] for k in ['step_id', 'request', 'answer']} for n in nodes]
    helper = b0.legacy_child.render_helper(history, None, 'Answer') if history else ''
    context = helper + '\n\n' + b0.passages(original) if helper else b0.passages(original)
    examples = read(ROOT / 'experiments/query_decomposition/multigrain_prompt_v3_20261017/final_examples.json')
    prompt = b0.final_prompts.final(case['question'], context, examples)
    assert prompt.endswith(b0.final_prompts.FINAL_READOUT)
    suffix = (FINAL_CHECK if arm == 'f01_final_consistency' else '') + readout
    return prompt[:-len(b0.final_prompts.FINAL_READOUT)] + suffix, context


def prepare():
    if (HERE / 'PROTOCOL.frozen.json').exists():
        plan = read(HERE / 'PROTOCOL.frozen.json')
        verify(plan)
        return plan
    HERE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    previous = read(PREV_E / 'PROTOCOL.frozen.json')
    cases = rows(PREV_E / 'runtime300.frozen.jsonl')
    assert len(cases) == 300 and all('answer' not in c for c in cases)
    manifest = []
    details = {r['id']: r for r in rows(PREV_O / 'PAIRED_DETAILS_LATEST.jsonl') if r['arm'] == 'C_final_readout_P1'}
    assert len(details) == 300
    readout = read(ROOT / 'experiments/query_decomposition/component_chain_opt_v2_20261019/PROTOCOL.frozen.json')['phase_B']['P1_readout']
    baseline_sources = sources()
    nodes = 0
    for c in cases:
        src = baseline_sources[c['id']]
        base = baseline_case(src)
        control_path = ROOT / details[c['id']]['source']
        control = read(control_path)
        assert control['status'].startswith('success')
        assert base['nodes'] == control['nodes']
        assert base['final_context'] == control['final_context']
        prompt, context = final_prompt(c, base['nodes'], base['original_retrieval'], readout, 'parent')
        assert prompt == control['final_prompt'] and context == base['final_context']
        goal = base['nodes'][-1]
        assert goal['role'] == 'goal' and goal['request'] == c['question']
        changed = altered_goal_prompt(goal['child_prompt'])
        assert changed.replace(b0.legacy_child.GOAL_RULE + '\n\nAnswer:', 'Answer:') == goal['child_prompt']
        nodes += len(base['nodes'])
        manifest.append({'id': c['id'], 'base': src['source'], 'base_sha256': src['source_sha256'],
                         'control': str(control_path.relative_to(ROOT)), 'control_sha256': sha(control_path)})
    save_rows(HERE / 'runtime300.frozen.jsonl', cases)
    save_rows(HERE / 'SOURCES.frozen.jsonl', manifest)
    save_rows(HERE / 'pilot100.ids.frozen.jsonl', [{'id': i} for i in pilot_ids(cases)])
    pins = {str(p.relative_to(ROOT)): sha(p) for p in [
        Path(__file__), PREV_E / 'PROTOCOL.frozen.json', PREV_O / 'FINAL_RESULTS.frozen.json',
        PREV_O / 'FINAL_AUDIT.frozen.json',
        ROOT / 'experiments/query_decomposition/component_chain_c08_c02_v1_20261019/pipeline.py',
        ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/prompts.py',
        ROOT / 'experiments/query_decomposition/multigrain_prompt_v3_20261017/prompts.py',
        ROOT / 'experiments/query_decomposition/multigrain_prompt_v3_20261017/final_examples.json',
    ]}
    plan = {'utc': utc(), 'parent': 'B0 + P1 readout (completed independent C arm)',
            'arms': ARMS, 'model': previous['model'], 'temperature': 0, 'enable_thinking': False,
            'readout': readout, 'g01_added_rule': b0.legacy_child.GOAL_RULE, 'f01_added_rule': FINAL_CHECK,
            'single_factor': {'g01_goal_consistency': 'Only insert existing GOAL_RULE before the goal Answer marker; retain P1 final readout.',
                              'f01_final_consistency': 'Only prepend FINAL_CHECK to P1 final readout; all B0 nodes retained.'},
            'pilot_n': 100, 'maximum_n_per_arm': 300, 'pilot_sample': '33 HotpotQA/33 2Wiki/34 MuSiQue; deterministic query-ID hash, not outcome selection',
            'gate': 'Expand an arm from pilot100 to full300 only if all100 scorable and paired mean native F1 strictly exceeds the P1 parent. No claim of independent confirmation.',
            'selection': 'Among complete300 arms and parent choose higher mean native F1, prefer fewer edits on exact tie; preserve B0 and S02 unchanged. Describe result as development selection, not significance or generalization.',
            'sample_role': 'Exposed existing diagnostic300; pilot is not fresh holdout.',
            'fixed': ['R02 decomposition', 'all ordinary child prompts/answers', 'all rewrite outputs', 'all query strings and retrieved packets',
                      'original-first interleaving/dedup Top10', 'full original-query Top10', 'all auxiliary QA included',
                      'model and generation limits (goal128/final64)', 'no external factual knowledge added by program'],
            'upstream_replay_scope': 'Full frozen B0 prefix including actual retrieval and ordinary QA; no changed output is used to retrieve because only last goal/final are varied.',
            'cache': 'Exact full payload only; same-case completed P1 control receipts first. First successful receipt, no score-based selection.',
            'retries': 'At most one new immutable recovery round for infrastructure failures; preserve unknown liabilities and successful prefixes. Algorithm failure is retained, not retried.',
            'budget': {'cumulative_cap_cny': 300, 'dispatch_ceiling_cny': 299.9, 'local_subcap_cny': 3},
            'runtime_sha256': sha(HERE / 'runtime300.frozen.jsonl'), 'source_manifest_sha256': sha(HERE / 'SOURCES.frozen.jsonl'),
            'pilot_sha256': sha(HERE / 'pilot100.ids.frozen.jsonl'), 'source_pins': pins}
    save(HERE / 'PROTOCOL.frozen.json', plan)
    save(OUT / 'OFFLINE_VALIDATION.frozen.json', {'passed': True, 'cases': 300, 'prefix_nodes': nodes,
         'parent_final_prompts_rebuilt': 300, 'all_goal_inputs_preserved': True, 'new_api_calls': 0})
    save(OUT / 'BUDGET_START.json', budget_snapshot())
    log('prepare_two_single_factor_candidates', pilot=100, maximum_per_arm=300, preserved_upstream_nodes=nodes,
        global_cap_cny=300, local_subcap_cny=3, code_sha256=sha(Path(__file__)))
    return plan


def verify(plan):
    for p, h in plan['source_pins'].items():
        assert sha(ROOT / p) == h, p
    assert sha(HERE / 'runtime300.frozen.jsonl') == plan['runtime_sha256']
    assert sha(HERE / 'SOURCES.frozen.jsonl') == plan['source_manifest_sha256']
    assert sha(HERE / 'pilot100.ids.frozen.jsonl') == plan['pilot_sha256']


class Budget(ParallelBudget):
    def __init__(self):
        super().__init__(OUTPUT / 'budget.sqlite3', 300)

    def reserve(self, rid, namespace, amount):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
            local = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?', (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()[0]
            if total + amount > 299.9 or local + amount > 3:
                raise RuntimeError('Global300/local3 dispatch ceiling; no submission')
            if self.db.execute('SELECT 1 FROM charges WHERE id=?', (rid,)).fetchone():
                raise RuntimeError('Existing liability cannot be resubmitted')
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (rid, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


def chosen(arm, cid):
    ds, qid = cid.split(':', 1)
    last = None
    for p in sorted((OUT / 'rounds').glob(f'*/cases/{ds}/{arm}/{qid}.json')):
        last = read(p)
        if isinstance(last.get('prediction'), str) and (last['status'].startswith('success') or last['status'] == 'algorithm_failure'):
            return last
    return last


def run_case(case, arm, owner, client, plan, manifest):
    global ERRORS
    store = WorkerStore(owner)
    budget = Budget()
    base = read(ROOT / manifest['base'])
    control = read(ROOT / manifest['control'])
    assert sha(ROOT / manifest['base']) == manifest['base_sha256']
    assert sha(ROOT / manifest['control']) == manifest['control_sha256']
    cached = {r['payload_sha256']: r for r in control['logical_calls'] if r['status'] == 'success'}
    # Preserve a successful goal/formatter prefix when a later final request
    # failed in an earlier immutable round. Never resend successful work.
    current_round = owner.directory.parents[2]
    for old_round in sorted((OUT / 'rounds').iterdir()):
        if old_round == current_round:
            continue
        old_db = old_round / 'methods' / case['dataset'] / arm / 'checkpoint.sqlite3'
        if not old_db.exists():
            continue
        with sqlite3.connect(old_db.resolve().as_uri() + '?mode=ro', uri=True) as db:
            previous = db.execute("SELECT value FROM data WHERE kind='logical_call' AND json_extract(value,'$.qid')=? AND json_extract(value,'$.status')='success'", (case['qid'],)).fetchall()
        for (raw,) in previous:
            row = json.loads(raw)
            cached.setdefault(row['payload_sha256'], row)
    transport = OneShot(store, budget, client, plan['model'], case['qid'])
    calls = []
    started = time.perf_counter()

    def ask(stage, prompt, maximum, allow_final_truncation=False):
        global ERRORS
        req = payload(plan['model'], prompt, maximum)
        key = digest(req)
        with KEY_LOCK:
            mutex = KEYS.setdefault((store.namespace, key), Lock())
        with mutex:
            old = cached.get(key)
            r = receipt(old, req) if old else None
            reused = r is not None
            if r is None:
                rid = digest([store.namespace, req])
                row = store.db.execute('SELECT status,result FROM requests WHERE id=?', (rid,)).fetchone()
                if row:
                    reused = True
                else:
                    with LOCK:
                        if ERRORS >= 3:
                            raise RuntimeError('Dispatch deferred after three consecutive unknowns')
                    try:
                        transport.ask(stage, prompt, maximum, allow_final_truncation=True)
                    except Exception:
                        if not store.db.execute('SELECT 1 FROM requests WHERE id=?', (rid,)).fetchone():
                            raise
                    row = store.db.execute('SELECT status,result FROM requests WHERE id=?', (rid,)).fetchone()
                    with LOCK:
                        ERRORS = ERRORS + 1 if row[0] in ['unknown', 'in_flight'] else 0
                r = {'request_id': rid, 'source': str((store.directory / 'checkpoint.sqlite3').relative_to(ROOT)),
                     'status': row[0], 'result': json.loads(row[1]) if row[1] else None}
            value = r['result'] or {}
            call = {'qid': case['qid'], 'stage': stage, 'payload_sha256': key, 'request_id': r['request_id'],
                    'source': r['source'], 'status': r['status'], 'reused': reused,
                    'usage': value.get('usage', {}), 'finish_reason': value.get('finish_reason')}
            store.put('logical_call', digest([case['qid'], stage, key]), call)
            calls.append(call)
            return use_receipt(r, stage, allow_final_truncation)

    nodes = deepcopy(base['nodes'])
    try:
        if arm == 'g01_goal_consistency':
            goal = nodes[-1]
            prompt = altered_goal_prompt(goal['child_prompt'])
            rid, answer, finish = ask('goal_consistency/goal', prompt, 128)
            if not answer:
                raise RuntimeError('Empty goal answer; no propagation')
            initial_answer, formatter_id = answer, None
            if len(answer.split()) < 4:
                format_prompt = ('Express the candidate answer below as a concise complete factual sentence that names its subject and relation. '
                                 'Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. '
                                 'If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n'
                                 + 'Question:\n' + goal['request'] + '\n\nCandidate answer:\n' + answer
                                 + '\n\nContext:\n' + b0.passages(goal['reader_packet']))
                formatter_id, answer, finish = ask('goal_consistency/sentence_format', format_prompt, 128)
            goal.update(answer=answer, child_prompt=prompt, answer_request_id=rid,
                        initial_answer=initial_answer, sentence_format_request_id=formatter_id)
        prompt, context = final_prompt(case, nodes, base['original_retrieval'], plan['readout'], arm)
        rid, prediction, finish = ask('targeted/final', prompt, 64, True)
        result = {'status': 'success' if finish == 'stop' else 'success_truncated', 'prediction': prediction,
                  'nodes': nodes, 'final_context': context, 'final_prompt': prompt,
                  'request_id': rid, 'finish_reason': finish}
    except Exception as exc:
        result = {'status': 'incomplete', 'prediction': None, 'reason': str(exc)[:400],
                  'error_type': type(exc).__name__, 'retryable_infrastructure': False}
        if isinstance(exc, IntermediateLength) or 'Empty goal answer' in str(exc):
            result.update(status='algorithm_failure', prediction='')
        elif isinstance(exc, RecordedFailure):
            raw = receipt({'source': exc.source, 'request_id': exc.request_id})['result'] or {}
            result.update(request_status=exc.request_status, http_status=raw.get('http_status'),
                          retryable_infrastructure=exc.request_status in ['unknown', 'in_flight'] or raw.get('http_status') in [408, 409, 425, 429, 500, 502, 503, 504])
        elif 'Dispatch deferred' in str(exc):
            result['retryable_infrastructure'] = True
    finally:
        store.db.close()
        budget.db.close()
    result.update(id=case['id'], qid=case['qid'], dataset=case['dataset'], question=case['question'],
                  arm=arm, source_prefix=manifest['base'], parent_source=manifest['control'], logical_calls=calls,
                  elapsed_seconds=time.perf_counter() - started, utc=utc())
    target = owner.directory.parents[2] / 'cases' / case['dataset'] / arm / (case['qid'] + '.json')
    assert not target.exists()
    locked_json(target, result)
    return {k: result.get(k) for k in ['id', 'arm', 'status', 'prediction', 'reason']}


def summarize(plan):
    cases = rows(HERE / 'runtime300.frozen.jsonl')
    manifest = {r['id']: r for r in rows(HERE / 'SOURCES.frozen.jsonl')}
    refs = {r['id']: r for r in rows(PREV_E / 'references300.local.frozen.jsonl')}
    scorers = {ds: metric(ds) for ds in ['hotpotqa', '2wiki', 'musique']}
    pilot = {r['id'] for r in rows(HERE / 'pilot100.ids.frozen.jsonl')}
    parent_scores, baseline_scores = {}, {}
    details, result = [], {}
    for c in cases:
        src = manifest[c['id']]
        pp = read(ROOT / src['control'])
        bb = read(ROOT / src['base'])
        parent_scores[c['id']] = scorers[c['dataset']](pp['prediction'], refs[c['id']])['f1']
        baseline_scores[c['id']] = scorers[c['dataset']](bb['prediction'], refs[c['id']])['f1']
    for arm in ARMS:
        values, states = [], Counter()
        for c in cases:
            x = chosen(arm, c['id'])
            states[x['status'] if x else 'unstarted'] += 1
            if not x or not isinstance(x.get('prediction'), str):
                continue
            if not x['status'].startswith('success') and x['status'] != 'algorithm_failure':
                continue
            f1 = scorers[c['dataset']](x['prediction'], refs[c['id']])['f1']
            item = {'id': c['id'], 'dataset': c['dataset'], 'arm': arm, 'prediction': x['prediction'],
                    'f1': f1, 'parent_f1': parent_scores[c['id']], 'B0_f1': baseline_scores[c['id']],
                    'delta_parent': f1 - parent_scores[c['id']], 'pilot': c['id'] in pilot}
            values.append(item)
            details.append(item)
        pilot_values = [v for v in values if v['pilot']]
        pilot_delta = sum(v['delta_parent'] for v in pilot_values) / len(pilot_values) if pilot_values else None
        rng = np.random.default_rng(20261020)
        bootstrap = np.zeros(5000)
        if values:
            for ds in scorers:
                dd = np.array([v['delta_parent'] for v in values if v['dataset'] == ds])
                if len(dd):
                    bootstrap += dd[rng.integers(len(dd), size=(5000, len(dd)))].sum(1) / len(values)
        result[arm] = {'scorable': len(values), 'statuses': dict(states),
                       'F1': sum(v['f1'] for v in values) / len(values) if values else None,
                       'same_parent_F1': sum(v['parent_f1'] for v in values) / len(values) if values else None,
                       'delta_parent': sum(v['delta_parent'] for v in values) / len(values) if values else None,
                       'paired95': np.quantile(bootstrap, [.025, .975]).tolist() if values else None,
                       'up': sum(v['delta_parent'] > 1e-12 for v in values), 'tie': sum(abs(v['delta_parent']) <= 1e-12 for v in values),
                       'down': sum(v['delta_parent'] < -1e-12 for v in values),
                       'pilot_scorable': len(pilot_values), 'pilot_delta': pilot_delta,
                       'expand': len(pilot_values) == 100 and pilot_delta > 1e-12,
                       'by_dataset': {ds: {'n': len(vv := [v for v in values if v['dataset'] == ds]),
                                          'F1': sum(v['f1'] for v in vv) / len(vv) if vv else None,
                                          'delta_parent': sum(v['delta_parent'] for v in vv) / len(vv) if vv else None}
                                      for ds in scorers}}
    value = {'utc': utc(), 'parent300_F1': sum(parent_scores.values()) / 300,
             'B0_300_F1': sum(baseline_scores.values()) / 300, 'arms': result,
             'budget': budget_snapshot(), 'sample_role': plan['sample_role']}
    locked_json(OUT / 'RESULTS_LATEST.json', value)
    (OUT / 'PAIRED_DETAILS.jsonl').write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in details), encoding='utf8')
    return value


def run_phase(plan, cases, arm, phase, manifests):
    global ERRORS
    option = get_api_option(2)
    assert option['model'] == plan['model']
    for recovery in range(2):
        pending = []
        for c in cases:
            x = chosen(arm, c['id'])
            if x is None or x.get('retryable_infrastructure'):
                pending.append(c)
        if not pending:
            break
        root = OUT / 'rounds' / f'{phase}_{arm}_r{recovery + 1:03d}'
        if root.exists():
            raise RuntimeError('Existing round requires explicit inspection; no silent resume')
        ERRORS = 0
        network = Store(root / 'network', {'phase': phase, 'arm': arm, 'recovery': recovery})
        owners = {ds: Store(root / 'methods' / ds / arm, {'arm': arm, 'model': plan['model'], 'protocol_sha256': sha(HERE / 'PROTOCOL.frozen.json')})
                  for ds in ['hotpotqa', '2wiki', 'musique']}
        log('begin_round', phase=phase, arm=arm, recovery=recovery, pending=len(pending), unknown_liabilities_retained=True)
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
                api = stack.enter_context(OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'], base_url=option['base_url'], timeout=90, max_retries=0, http_client=http))
                client = StreamingClient(api)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    jobs = [pool.submit(run_case, c, arm, owners[c['dataset']], client, plan, manifests[c['id']]) for c in pending]
                    for done, job in enumerate(as_completed(jobs), 1):
                        r = job.result()
                        print(json.dumps({'phase': phase, 'done': done, 'n': len(pending), **r}, ensure_ascii=False), flush=True)
                        locked_json(OUT / 'PROGRESS.json', {'utc': utc(), 'phase': phase, 'arm': arm, 'done': done, 'planned': len(pending), 'latest': r})
                locked_json(root / 'TERMINAL.json', {'utc': utc(), 'attempted': len(pending)})
        finally:
            for owner in owners.values():
                owner.export(full=True)
                owner.db.close()
            network.export(full=True)
            network.db.close()
        summarize(plan)
        if recovery == 0:
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    if not args.run:
        print(json.dumps({'prepared': True, 'validation': read(OUT / 'OFFLINE_VALIDATION.frozen.json'), 'new_api_calls': 0}, ensure_ascii=False))
        return
    cases = rows(HERE / 'runtime300.frozen.jsonl')
    manifests = {r['id']: r for r in rows(HERE / 'SOURCES.frozen.jsonl')}
    pilot = {r['id'] for r in rows(HERE / 'pilot100.ids.frozen.jsonl')}
    for arm in ARMS:
        run_phase(plan, [c for c in cases if c['id'] in pilot], arm, 'pilot100', manifests)
        current = summarize(plan)
        save(OUT / (arm + '.PILOT.frozen.json'), current['arms'][arm])
        if current['arms'][arm]['expand']:
            log('pilot_gate', arm=arm, expand=True, pilot_delta=current['arms'][arm]['pilot_delta'])
            run_phase(plan, [c for c in cases if c['id'] not in pilot], arm, 'extension200', manifests)
        else:
            log('pilot_gate', arm=arm, expand=False, pilot_delta=current['arms'][arm]['pilot_delta'], reason='No positive complete100 delta; retain parent')
    current = summarize(plan)
    verify(plan)
    for r in manifests.values():
        assert sha(ROOT / r['base']) == r['base_sha256']
        assert sha(ROOT / r['control']) == r['control_sha256']
    save(OUT / 'FINAL_RESULTS.frozen.json', current)
    parent_score = current['parent300_F1']
    eligible = [(parent_score, 'B0_P1_parent')]
    eligible += [(a['F1'], arm) for arm, a in current['arms'].items() if a['scorable'] == 300]
    score, winner = max(eligible, key=lambda x: x[0])
    save(OUT / 'SELECTION.frozen.json', {'selected': winner, 'F1': score, 'scope': 'Exposed300 development selection',
         'B0_and_S02_unchanged': True, 'not_independent_confirmation': True})
    save(OUT / 'TERMINAL.json', {'utc': utc(), 'selected': winner, 'budget': current['budget'], 'original_sources_unchanged': True})
    log('complete', selected=winner, final_F1=score, budget=current['budget'])
    print(json.dumps({'terminal': True, 'selected': winner, 'F1': score, 'budget': current['budget']}), flush=True)


if __name__ == '__main__':
    main()
