"""Independent quick90 runner; --prepare/--offline never submit requests.

Only --run opens the provider. This does not clear or inspect a campaign hold
as execution authorization: its authority is the user's explicit quick90 task.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, closing
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock
import argparse
import ast
import difflib
import hashlib
import importlib
import json
import os
import socket
import sqlite3
import ssl
import sys
import time
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
PACKAGE = ROOT / 'src/p02_multigrain_timing_cachefix_v1'
COHORT = HERE.parent / 'p02_noformatter_quick90_v1_20261021'
OUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1' / HERE.name
PLAN_DIR = HERE.parent / 'unified_rag_eval_v1_20261021'
DECISION = 'P02_CHILDPROMPT_SELFCONTAINED_QUICK90_V1_20261021'
DATASETS = ('hotpotqa', '2wiki', 'musique')
MODULE = 'experiments.query_decomposition.' + HERE.name
SLOTS = BoundedSemaphore(3)
PRINT_LOCK = Lock()
STOP = Event()

from p02_multigrain_timing_cachefix_v1.core import pipeline as baseline
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.llm import LLM, build_payload
from p02_multigrain_timing_cachefix_v1.core.runtime import Ask, Budget, check_resources


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(temporary, path)


def emit(value):
    with PRINT_LOCK:
        print(json.dumps(value, ensure_ascii=True), flush=True)


def variant():
    return importlib.import_module(MODULE + '.pipeline')


def prepare():
    if (HERE / 'PROTOCOL.frozen.json').exists():
        raise FileExistsError('Preserve this frozen protocol')
    # Read the current agreement; this scoped experiment leaves campaign holds intact.
    main_plan = read(PLAN_DIR / 'PLAN.json')
    assert main_plan['execution_approval']['approved'] is False
    (PLAN_DIR / 'PLAN.txt').read_text(encoding='utf8')
    (PLAN_DIR / 'DECISIONS.jsonl').read_text(encoding='utf8')
    candidate = ROOT / 'outputs/p02/prompt_review_20261021/ordinary_child_prompt_selfcontained_v1.candidate.txt'
    (HERE / 'ordinary_child_prompt.frozen.txt').write_bytes(candidate.read_bytes())
    source = (PACKAGE / 'core/pipeline.py').read_text(encoding='utf8')
    changed = source.replace('from . import METHOD, ROOT',
        "from p02_multigrain_timing_cachefix_v1.core import ROOT\nMETHOD = 'p02_ordinary_child_selfcontained_v1'", 1)
    changed = changed.replace('from . import child_prompts, rewrite_prompts, final_prompts',
        'from p02_multigrain_timing_cachefix_v1.core import rewrite_prompts, final_prompts\nfrom . import child_prompts', 1)
    ast.parse(changed)
    (HERE / 'pipeline.py').write_text(changed, encoding='utf8')
    (HERE / 'PIPELINE_CHANGE.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True), fromfile='frozen_P02', tofile='child_prompt_candidate')), encoding='utf8')
    inputs = {str((COHORT / name).relative_to(ROOT)): sha(COHORT / name) for name in
              ['runtime90.frozen.jsonl', 'references90.local.frozen.jsonl', 'COHORT.frozen.json', 'SOURCE_BINDINGS.frozen.jsonl']}
    cases = rows(COHORT / 'runtime90.frozen.jsonl')
    assert len(cases) == len({c['id'] for c in cases}) == 90
    assert Counter(c['dataset'] for c in cases) == {d: 30 for d in DATASETS}
    holds = [ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/STOP_REQUESTED',
             PLAN_DIR / 'STOP_REQUESTED',
             ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/STOP_REQUESTED']
    protocol = {
        'id': HERE.name, 'utc': utc(), 'authority': 'User: 在快速90题集快速评测该子问题prompt',
        'scope': 'Existing quick90 development sample; paired complete native-P02 flows; ordinary-child prompt only.',
        'planned': 90, 'per_dataset': 30, 'inputs': inputs,
        'candidate_sha256': sha(HERE / 'ordinary_child_prompt.frozen.txt'),
        'original_package_pins': {str(p.relative_to(ROOT)): sha(p) for p in PACKAGE.rglob('*.py')},
        'fixed_config_sha256': sha(PACKAGE / 'config.json'),
        'code_pins': {p.name: sha(p) for p in [Path(__file__), HERE / 'pipeline.py', HERE / 'child_prompts.py']},
        'hold_pins': {str(p.relative_to(ROOT)): sha(p) for p in holds if p.exists()},
        'change': 'Use the exact user-reviewed self-contained template on ordinary nodes only; unchanged goal, rewriter, formatter, merge, final, model and output limits.',
        'control': 'Read-only exact-payload replay from the frozen P02 package; retain per-query receipt bindings.',
        'candidate_execution': 'Fresh inference where full payload differs; regenerate downstream from candidate history and retrieve changed queries; reuse only exact successful complete payloads.',
        'cache': 'Frozen package first, then this experiment only; no stage-name or answer-only reuse; unknown requests are never resent automatically.',
        'model': read(PACKAGE / 'config.json')['generation'],
        'budget': 'Existing native-P02 ledger and cumulative cap remain unchanged; no campaign resume or main-budget reconfiguration.',
        'device': 'cuda for frozen BGE encoder; exact CPU FAISS search, four threads; not a timing benchmark.',
        'failure_policy': 'Keep known algorithm errors separate from infrastructure/unknown requests; no retries or answer selection for F1.',
        'metrics': 'Existing native Answer-F1/EM; paired improved/tied/harmed, full 90 denominator and missingness; full-flow input/output tokens and estimated cost, new spend separately.',
        'not_independent_confirmation': True, 'main_evaluation_remains_paused': True,
    }
    save(HERE / 'PROTOCOL.frozen.json', protocol)
    # Append only the scoped authorization; keep global execution approval false.
    entry = {'decision_id': DECISION, 'scope': protocol['scope'],
             'protocol': str((HERE / 'PROTOCOL.frozen.json').relative_to(ROOT)),
             'campaign_resume_authorized': False, 'new_inference': True}
    main_plan.setdefault('standalone_diagnostics', []).append(entry)
    save(PLAN_DIR / 'PLAN.json', main_plan)
    with (PLAN_DIR / 'DECISIONS.jsonl').open('a', encoding='utf8') as handle:
        handle.write(json.dumps({'id': DECISION, 'utc': utc(), 'status': 'approved_scoped_quick_test',
            'authority': protocol['authority'], **entry, 'main_evaluation_remains_paused': True}, ensure_ascii=False) + '\n')
    return {'prepared': True, 'planned': 90, 'protocol': str(HERE / 'PROTOCOL.frozen.json')}


def check_pins():
    protocol = read(HERE / 'PROTOCOL.frozen.json')
    assert sha(HERE / 'ordinary_child_prompt.frozen.txt') == protocol['candidate_sha256']
    assert sha(PACKAGE / 'config.json') == protocol['fixed_config_sha256']
    for group in ['inputs', 'original_package_pins', 'hold_pins']:
        for path, expected in protocol[group].items():
            assert sha(ROOT / path) == expected, path
    for name, expected in protocol['code_pins'].items():
        assert sha(HERE / name) == expected, name
    assert read(PLAN_DIR / 'PLAN.json')['execution_approval']['approved'] is False
    return protocol


def baseline_replay(case, seed, config):
    calls, queries = [], []
    def retrieve(query):
        row = seed.execute('SELECT packet FROM retrieval WHERE dataset=? AND qid=? AND query=?',
                           (case['dataset'], case['qid'], query)).fetchone()
        if row is None:
            raise RuntimeError('Missing frozen control retrieval')
        queries.append(query)
        return json.loads(row[0])
    def ask(stage, prompt, maximum, allow_final_truncation=False):
        payload = build_payload(config['generation'], prompt, maximum)
        row = seed.execute('SELECT request_id,result FROM requests WHERE payload=?', (canonical(payload),)).fetchone()
        if row is None:
            raise RuntimeError('Missing frozen control payload')
        result = json.loads(row[1])
        assert result['status'] == 'success' and result['finish_reason'] == 'stop'
        assert result.get('usage')
        calls.append({'stage': stage, 'request_id': row[0], 'usage': result['usage'], 'source': 'frozen_replay'})
        return row[0], result['answer'].strip(), result['finish_reason']
    answer = baseline.Pipeline(config).execute(baseline.safe_case(case), retrieve, ask)
    return {'id': case['id'], 'dataset': case['dataset'], 'prediction': answer['prediction'],
            'status': answer['status'], 'calls': calls, 'trajectory': answer}


def offline():
    check_pins()
    config = read(PACKAGE / 'config.json')
    cases = rows(COHORT / 'runtime90.frozen.jsonl')
    bindings = {v['id']: v for v in rows(COHORT / 'SOURCE_BINDINGS.frozen.jsonl')}
    OUT.mkdir(parents=True, exist_ok=True)
    originals, checks = [], []
    with closing(sqlite3.connect((PACKAGE / 'data/replay.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as seed:
        for case in cases:
            original = baseline_replay(case, seed, config)
            binding = bindings[case['id']]
            assert original['calls'][-1]['request_id'] == binding['final_request_id']
            assert sha(ROOT / binding['source']) == binding['source_sha256']
            nodes = original['trajectory']['nodes']
            checked = 0
            for node in nodes:
                packet = baseline.passages(node['reader_packet'])
                prompt = variant().child_prompts.child(case['question'], node['step_id'], node['request'],
                    node['prior_complete_answers'], packet, [], role=node['role'])
                if node['role'] == 'goal':
                    prompt = prompt[:-len('Answer:')] + variant().child_prompts.GOAL_RULE + '\n\nAnswer:'
                    assert prompt == node['child_prompt']
                else:
                    assert prompt != node['child_prompt']
                    assert 'Resolve each #N' not in prompt
                    assert prompt.count('Example ') == 4
                    assert prompt.endswith('Answer:')
                    checked += 1
            checks.append({'id': case['id'], 'ordinary_prompts_changed': checked, 'goal_prompt_unchanged_on_identical_input': True})
            originals.append(original)
    (OUT / 'control90.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in originals), encoding='utf8')
    report = {'passed': True, 'control_replayed': len(originals), 'ordinary_prompts_changed': sum(v['ordinary_prompts_changed'] for v in checks),
              'all_goal_prompts_equal_on_identical_inputs': True, 'new_API_requests': 0, 'new_index_searches': 0, 'checks': checks}
    save(OUT / 'OFFLINE_CHECK.json', report)
    return {k: v for k, v in report.items() if k != 'checks'}


class LimitedLLM(LLM):
    def request_once(self, payload):
        with SLOTS:
            return super().request_once(payload)


class ScopedAsk(Ask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = LimitedLLM(self.settings)


class NetworkEvents:
    def event(self, event, **fields):
        with (OUT / 'network.jsonl').open('a', encoding='utf8') as handle:
            handle.write(json.dumps({'utc': utc(), 'event': event, **fields}) + '\n')


def run_dataset(dataset):
    from p02_multigrain_timing_cachefix_v1.core.retrieval import Retriever
    config = read(PACKAGE / 'config.json')
    cases = [baseline.safe_case(c) for c in rows(COHORT / 'runtime90.frozen.jsonl') if c['dataset'] == dataset]
    directory = OUT / 'candidate' / dataset
    directory.mkdir(parents=True, exist_ok=True)
    current_id = None
    def event(value):
        value = {'id': current_id, **value}
        with (directory / 'trace.jsonl').open('a', encoding='utf8') as handle:
            handle.write(json.dumps({'utc': utc(), **value}, ensure_ascii=False) + '\n')
    with run_lock(directory), ExitStack() as stack:
        seed = stack.enter_context(closing(sqlite3.connect((PACKAGE / 'data/replay.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)))
        cache = stack.enter_context(Cache(directory / 'checkpoint.sqlite3'))
        cache.put('meta', 'identity', {'protocol_sha256': sha(HERE / 'PROTOCOL.frozen.json'), 'dataset': dataset})
        budget = stack.enter_context(closing(Budget(config, directory, config['budget']['cumulative_cap_cny'])))
        ask = stack.enter_context(closing(ScopedAsk(cache, seed, config, budget, event)))
        retriever = Retriever(config['resources'][dataset], device='cuda')
        stack.callback(retriever.close)
        pipeline = variant().Pipeline(config)
        for index, case in enumerate(cases, 1):
            if STOP.is_set():
                break
            current_id = case['id']
            if cache.get('prediction', case['id']) is not None:
                continue
            def retrieve(query):
                key = canonical([dataset, case['qid'], query, 10])
                packet = cache.get('retrieval', key)
                if packet is None:
                    saved = seed.execute('SELECT packet FROM retrieval WHERE dataset=? AND qid=? AND query=?',
                                         (dataset, case['qid'], query)).fetchone()
                    if saved:
                        packet = json.loads(saved[0])
                        event({'event': 'retrieval', 'id': case['id'], 'query': query, 'source': 'frozen_replay'})
                    else:
                        packet = retriever(case['qid'], query)
                        event({'event': 'retrieval', 'id': case['id'], 'query': query, 'source': 'new_index_search'})
                    cache.put('retrieval', key, packet)
                return deepcopy(packet)
            try:
                result = pipeline.execute(case, retrieve, ask,
                    record_node=lambda node: cache.put('node', case['id'] + '/' + str(node['step_id']), node))
                cache.put('prediction', case['id'], result)
                emit({'dataset': dataset, 'completed': index, 'planned': 30, 'id': case['id'], 'status': result['status']})
            except Exception as exc:
                # Persist without dumping exception bodies, which may include provider data.
                failure = {'id': case['id'], 'dataset': dataset, 'status': 'incomplete', 'error_class': type(exc).__name__}
                save(directory / ('failure_' + hashlib.sha256(case['id'].encode()).hexdigest()[:16] + '.json'), failure)
                event({'event': 'case_incomplete', **failure})
                emit(failure)
                # Unknown and provider/config failures require reconciliation before further dispatch.
                recent = cache.requests()
                bad = [r for r in recent if r['status'] not in {'success'}]
                if bad or isinstance(exc, (ValueError, FileNotFoundError)):
                    STOP.set()
                    raise RuntimeError('Scoped test stopped; inspect saved sanitized failure and receipts') from None
                if isinstance(exc, RuntimeError) and ('Incomplete' in str(exc) or 'incomplete' in str(exc) or 'Final generation' in str(exc)):
                    terminal = {**case, 'status': 'algorithm_failure', 'prediction': '', 'error_class': type(exc).__name__}
                    cache.put('prediction', case['id'], terminal)
                    event({'event': 'algorithm_failure', 'id': case['id']})
                else:
                    STOP.set()
                    raise RuntimeError('Scoped test stopped on unexpected execution error') from None
        (directory / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
            for r in cache.all('prediction').values()), encoding='utf8')
    return dataset


def run():
    check_pins()
    assert read(OUT / 'OFFLINE_CHECK.json')['passed']
    config = read(PACKAGE / 'config.json')
    check_resources(config, rows(COHORT / 'runtime90.frozen.jsonl'))
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    with temporary_api_key('api_code_qwen.txt', allowed_root=ROOT, provider='dashscope'), provider_dns(
            urlsplit(config['generation']['base_url']).hostname, NetworkEvents()):
        host = urlsplit(config['generation']['base_url']).hostname
        with socket.create_connection((host, 443), timeout=15) as sock:
            with ssl.create_default_context().wrap_socket(sock, server_hostname=host):
                NetworkEvents().event('TLS_preflight', paid_request=False)
        save(OUT / 'EXECUTION.json', {'utc': utc(), 'status': 'running', 'scope': 'quick90 only', 'main_campaign_resumed': False})
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(run_dataset, ds): ds for ds in DATASETS}
                for future in as_completed(futures):
                    future.result()
        except Exception:
            STOP.set()
            report()
            raise
    return report()


def report():
    # Gold/reference data is read only here, after method execution.
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    controls = {r['id']: r for r in rows(OUT / 'control90.jsonl')}
    references = {r['id']: r for r in rows(COHORT / 'references90.local.frozen.jsonl')}
    details, usage = [], {}
    for ds in DATASETS:
        score = metric(ds)
        directory = OUT / 'candidate' / ds
        predictions = {}
        if (directory / 'checkpoint.sqlite3').exists():
            with closing(sqlite3.connect((directory / 'checkpoint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as db:
                predictions = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM data WHERE kind='prediction'")}
            trace = rows(directory / 'trace.jsonl') if (directory / 'trace.jsonl').exists() else []
            model_calls = [v for v in trace if v['event'] == 'model_call']
            complete_ids = {cid for cid, value in predictions.items() if value['status'] in {'success', 'success_truncated'}}
            logical = list({(v.get('id'), v['stage']): v for v in model_calls if v.get('id') in complete_ids}.values())
            control_calls = [call for r in controls.values() if r['dataset'] == ds for call in r['calls']]
            usage[ds] = {
                'new_returned_input_tokens': sum((v.get('usage') or {}).get('prompt_tokens', 0) for v in model_calls if not v['reused']),
                'new_returned_output_tokens': sum((v.get('usage') or {}).get('completion_tokens', 0) for v in model_calls if not v['reused']),
                'new_index_searches': sum(v['event'] == 'retrieval' and v.get('source') == 'new_index_search' for v in trace),
                'baseline_input_tokens_per_query': sum(v['usage']['prompt_tokens'] for v in control_calls) / 30,
                'baseline_output_tokens_per_query': sum(v['usage']['completion_tokens'] for v in control_calls) / 30,
                'candidate_complete_flow_denominator': len(complete_ids),
                'candidate_input_tokens_per_complete_query': sum((v.get('usage') or {}).get('prompt_tokens', 0) for v in logical) / len(complete_ids) if complete_ids else None,
                'candidate_output_tokens_per_complete_query': sum((v.get('usage') or {}).get('completion_tokens', 0) for v in logical) / len(complete_ids) if complete_ids else None,
                'candidate_statuses': dict(Counter(v['status'] for v in predictions.values())),
            }
        for cid, control in controls.items():
            if control['dataset'] != ds:
                continue
            candidate = predictions.get(cid)
            base_score = score(control['prediction'], references[cid])
            candidate_score = score(candidate['prediction'], references[cid]) if candidate else None
            details.append({'id': cid, 'dataset': ds, 'baseline_prediction': control['prediction'],
                'candidate_prediction': candidate['prediction'] if candidate else None,
                'baseline': base_score, 'candidate': candidate_score,
                'delta_F1': candidate_score['f1'] - base_score['f1'] if candidate_score else None})
    summary = {}
    for label in (*DATASETS, 'overall'):
        group = [r for r in details if label == 'overall' or r['dataset'] == label]
        paired = [r for r in group if r['candidate'] is not None]
        summary[label] = {'planned': len(group), 'complete': len(paired), 'missing': len(group) - len(paired),
            'baseline_F1': sum(r['baseline']['f1'] for r in group) / len(group),
            'candidate_F1': sum(r['candidate']['f1'] for r in paired) / len(group) if len(paired) == len(group) else None,
            'baseline_EM': sum(r['baseline']['em'] for r in group) / len(group),
            'candidate_EM': sum(r['candidate']['em'] for r in paired) / len(group) if len(paired) == len(group) else None,
            'improved': sum(r['delta_F1'] > 1e-12 for r in paired),
            'tied': sum(abs(r['delta_F1']) <= 1e-12 for r in paired),
            'harmed': sum(r['delta_F1'] < -1e-12 for r in paired),
        }
        if paired:
            summary[label]['paired_baseline_F1'] = sum(r['baseline']['f1'] for r in paired) / len(paired)
            summary[label]['paired_candidate_F1'] = sum(r['candidate']['f1'] for r in paired) / len(paired)
    with closing(sqlite3.connect(Path(read(PACKAGE / 'config.json')['budget']['ledger']).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        known, reserved = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace LIKE ?',
            ('%' + HERE.name + '%',)).fetchone()
    result = {'id': HERE.name, 'utc': utc(), 'status': 'completed' if summary['overall']['complete'] == 90 else 'incomplete',
        'scope': 'Existing exposed development quick90; control exact replay, candidate changed-payload inference; not independent confirmation.',
        'metrics': summary, 'new_usage': usage, 'new_estimated_CNY': known, 'unknown_reserved_CNY': reserved,
        'main_evaluation_remains_paused': True}
    save(OUT / 'REPORT.json', result)
    (OUT / 'PAIRED_DETAILS.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in details), encoding='utf8')
    save(OUT / 'EXECUTION.json', {'utc': utc(), 'status': result['status'], 'completed': summary['overall']['complete'], 'planned': 90})
    return result


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--offline', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        result = prepare()
    elif args.offline:
        result = offline()
    elif args.run:
        result = run()
    else:
        result = report()
    emit(result)


if __name__ == '__main__':
    main()
