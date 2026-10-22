"""Fixed90, two minimal ablations. No main-campaign execution or unknown retry."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, closing
from copy import deepcopy
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021 import run as parent
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021.run_r004 import RecoveryAsk
from p02_multigrain_timing_cachefix_v1.core.cache import Cache, canonical, run_lock, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from p02_multigrain_timing_cachefix_v1.core.runtime import Budget, check_resources
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021 import pipeline as parent_pipeline
from . import child_prompts

PACKAGE, COHORT, PLAN = parent.PACKAGE, parent.COHORT, parent.PLAN_DIR
CONTROL = parent.OUT / 'approved_recovery004'
OUT = parent.OUT.parent / HERE.name
ARMS = ('A_no_only', 'B_no_resolve')
read, rows, save, sha, emit = parent.read, parent.rows, parent.save, parent.sha, parent.emit


def ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def predictions(directory):
    result = {}
    for ds in parent.DATASETS:
        path = directory / 'candidate' / ds / 'checkpoint.sqlite3'
        if path.exists():
            with closing(ro(path)) as db:
                result.update({k: json.loads(v) for k, v in db.execute("SELECT key,value FROM data WHERE kind='prediction'")})
    return result


def prepare():
    parent.check_pins()
    assert not (HERE / 'PROTOCOL.frozen.json').exists()
    main = read(PLAN / 'PLAN.json')
    assert main['execution_approval']['approved'] is False
    assert read(CONTROL / 'REPORT.json')['status'] == 'completed'
    assert len(predictions(CONTROL)) == 90
    OUT.mkdir(parents=True, exist_ok=True)
    # Same pipeline control flow, independent prompt wrapper; historical files untouched.
    source = (parent.HERE / 'pipeline.py').read_text(encoding='utf8')
    source = source.replace("METHOD = 'p02_ordinary_child_selfcontained_v1'", "METHOD = 'p02_prompt_minimal_ablation_quick90_v1'")
    ast.parse(source)
    (HERE / 'pipeline.py').write_text(source, encoding='utf8')
    template = (parent.HERE / 'ordinary_child_prompt.frozen.txt').read_text(encoding='utf8')
    assert template.count(child_prompts.ONLY) == 1
    changed = template.replace(child_prompts.ONLY, 'using the retrieved passages', 1)
    (HERE / 'A_ordinary_prompt.frozen.txt').write_text(changed, encoding='utf8')
    (HERE / 'A_CHANGE.diff').write_text(''.join(difflib.unified_diff(template.splitlines(True), changed.splitlines(True), fromfile='revised_child_control', tofile='A_no_only')), encoding='utf8')
    (HERE / 'B_CHANGE.diff').write_text('-' + child_prompts.RESOLVE + '\n+\n', encoding='utf8')
    protocol = {
        'id': HERE.name, 'utc': utc(),
        'authority': 'User: 我们还是先采用你这里的最小对照 之后我们再修正goal提示词并尝试 开始实验对照',
        'annotation': '1', 'cohort': 'Same fixed quick90, 30 per dataset, no resampling',
        'control': str(CONTROL.relative_to(ROOT)), 'planned_per_arm': 90,
        'A_no_only': 'Delete only the first-line word only in the revised ordinary-child prompt; full downstream native flow.',
        'B_no_resolve': 'Keep revised ordinary prompts and answers. Freeze all upstream payloads/answers and goal retrieval packets. Delete only Resolve each #N from its numbered prior question and answer. from goal instructions. Regenerate goal/formatter/final only as exact payloads require.',
        'preserved': 'Legacy goal wording, repeated original query, examples, formatter, rewrite, retrieval, scoring, generation settings and token caps otherwise unchanged. Proposed goal rewrite is deferred.',
        'cache': 'Exact successful full-payload reuse from completed revised control first, original frozen package second; no cross-arm inference reuse. Fresh requests only for missing complete payloads.',
        'failure_policy': 'No SDK POST retry, unknown request replay or score-based rerun; unchanged approved HTTP connect-before-send retries2. Preserve failed receipts and reserves.',
        'destination': read(PACKAGE / 'config.json')['generation']['base_url'],
        'payload_and_charges': 'Same question/model-generated-QA/retrieved-passage payload classes and existing Qwen endpoint, user-approved two contrasts on the fixed90; usage billed under unchanged native ledger.',
        'metrics': 'Native final Answer-F1/EM, paired changes, complete-flow input/output usage and expense. Missingness and unknown liabilities explicit. No new timing claims.',
        'pins': {str(p.relative_to(ROOT)): sha(p) for p in [
            HERE / 'run.py', HERE / 'child_prompts.py', HERE / 'pipeline.py', HERE / 'A_ordinary_prompt.frozen.txt',
            parent.HERE / 'run_r004.py', parent.HERE / 'run_r002.py',
            *[CONTROL / 'candidate' / ds / 'checkpoint.sqlite3' for ds in parent.DATASETS]]},
        'main_evaluation_remains_paused': True, 'not_independent_confirmation': True,
    }
    save(HERE / 'PROTOCOL.frozen.json', protocol)
    entry = {'decision_id': HERE.name.upper(), 'scope': 'Two explicitly selected minimal prompt contrasts on the existing fixed quick90',
        'protocol': str((HERE / 'PROTOCOL.frozen.json').relative_to(ROOT)), 'campaign_resume_authorized': False, 'new_inference': True}
    main.setdefault('standalone_diagnostics', []).append(entry)
    save(PLAN / 'PLAN.json', main)
    with (PLAN / 'DECISIONS.jsonl').open('a', encoding='utf8') as handle:
        handle.write(json.dumps({'id': entry['decision_id'], 'utc': utc(), 'status': 'approved_scoped_minimal_ablations',
            'authority': protocol['authority'], **entry, 'destination': protocol['destination'], 'main_evaluation_remains_paused': True}, ensure_ascii=False) + '\n')
    return {'prepared': True, 'arms': list(ARMS), 'planned_each': 90}


def check():
    parent.check_pins()
    p = read(HERE / 'PROTOCOL.frozen.json')
    for path, expected in p['pins'].items():
        assert sha(ROOT / path) == expected, path
    return p


def seed_control():
    """An immutable union, without provider calls or ambiguous requests."""
    path = OUT / 'seed.sqlite3'
    assert not path.exists()
    with closing(sqlite3.connect(path)) as db, closing(ro(PACKAGE / 'data/replay.sqlite3')) as source:
        db.executescript('CREATE TABLE requests(payload TEXT PRIMARY KEY,request_id TEXT,result TEXT,origin TEXT); CREATE TABLE retrieval(dataset TEXT,qid TEXT,query TEXT,packet TEXT,origin TEXT,PRIMARY KEY(dataset,qid,query));')
        db.executemany('INSERT INTO requests VALUES(?,?,?,?)', [(*r, 'original_P02') for r in source.execute('SELECT payload,request_id,result FROM requests')])
        db.executemany('INSERT INTO retrieval VALUES(?,?,?,?,?)', [(*r, 'original_P02') for r in source.execute('SELECT dataset,qid,query,packet FROM retrieval')])
        for ds in parent.DATASETS:
            with closing(ro(CONTROL / 'candidate' / ds / 'checkpoint.sqlite3')) as newer:
                db.executemany('INSERT OR REPLACE INTO requests VALUES(?,?,?,?)', [(*r, 'completed_revised_child_control/' + ds) for r in newer.execute("SELECT payload,id,result FROM requests WHERE status='success'")])
                for key, value in newer.execute("SELECT key,value FROM data WHERE kind='retrieval'"):
                    dataset, qid, query, k = json.loads(key)
                    assert dataset == ds and k == 10
                    db.execute('INSERT OR REPLACE INTO retrieval VALUES(?,?,?,?,?)', (ds, qid, query, value, 'completed_revised_child_control/' + ds))
        db.commit()


def offline():
    check()
    seed_control()
    config, original = read(PACKAGE / 'config.json'), predictions(CONTROL)
    controls = []
    counts = {a: {'ordinary_changed': 0, 'goal_changed': 0} for a in ARMS}
    with closing(ro(OUT / 'seed.sqlite3')) as db:
        for case in rows(COHORT / 'runtime90.frozen.jsonl'):
            calls = []
            def retrieve(query):
                row = db.execute('SELECT packet FROM retrieval WHERE dataset=? AND qid=? AND query=?', (case['dataset'], case['qid'], query)).fetchone()
                assert row is not None
                return json.loads(row[0])
            def ask(stage, prompt, maximum, allow_final_truncation=False):
                row = db.execute('SELECT request_id,result,origin FROM requests WHERE payload=?', (canonical(build_payload(config['generation'], prompt, maximum)),)).fetchone()
                assert row is not None, (case['id'], stage)
                result = json.loads(row[1])
                assert result['status'] == 'success' and result['usage']
                calls.append({'stage': stage, 'request_id': row[0], 'usage': result['usage'], 'source': row[2]})
                return row[0], result['answer'].strip(), result['finish_reason']
            replay = parent_pipeline.Pipeline(config).execute(parent.baseline.safe_case(case), retrieve, ask)
            stored = original[case['id']]
            assert replay['prediction'] == stored['prediction'] and replay['final_prompt'] == stored['final_prompt']
            for old, node in zip(stored['nodes'], replay['nodes']):
                for field in ['child_prompt', 'answer', 'reader_packet', 'prior_complete_answers', 'rewritten_query']:
                    assert old[field] == node[field], (case['id'], field)
                for arm in ARMS:
                    child_prompts.ARM = arm
                    rendered = child_prompts.child(case['question'], node['step_id'], node['request'], node['prior_complete_answers'], parent.baseline.passages(node['reader_packet']), [], role=node['role'])
                    if node['role'] == 'goal':
                        rendered = rendered[:-len('Answer:')] + child_prompts.GOAL_RULE + '\n\nAnswer:'
                    expected = node['child_prompt']
                    if arm == 'A_no_only' and node['role'] == 'ordinary':
                        expected = expected.replace(child_prompts.ONLY, 'using the retrieved passages', 1)
                    if arm == 'B_no_resolve' and node['role'] == 'goal':
                        expected = expected.replace(child_prompts.RESOLVE, '', 1)
                    assert rendered == expected
                    if rendered != node['child_prompt']:
                        counts[arm][node['role'] + '_changed'] += 1
            controls.append({'id': case['id'], 'dataset': case['dataset'], 'prediction': stored['prediction'], 'status': stored['status'], 'calls': calls, 'trajectory': stored})
    assert counts == {'A_no_only': {'ordinary_changed': 176, 'goal_changed': 0}, 'B_no_resolve': {'ordinary_changed': 0, 'goal_changed': 90}}
    for arm in ARMS:
        directory = OUT / arm
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'control90.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in controls), encoding='utf8')
    receipt = {'passed': True, 'revised_control_replayed': 90, 'exact_prompt_deltas': counts, 'seed_sha256': sha(OUT / 'seed.sqlite3'), 'new_API_requests': 0, 'main_evaluation_remains_paused': True}
    save(OUT / 'OFFLINE_CHECK.json', receipt)
    return receipt


def run_dataset(arm, dataset):
    from p02_multigrain_timing_cachefix_v1.core.retrieval import Retriever
    config = read(PACKAGE / 'config.json')
    cases = [parent.baseline.safe_case(c) for c in rows(COHORT / 'runtime90.frozen.jsonl') if c['dataset'] == dataset]
    controls = {r['id']: r for r in rows(OUT / arm / 'control90.jsonl')}
    directory = OUT / arm / 'candidate' / dataset
    directory.mkdir(parents=True, exist_ok=True)
    current_id = None
    def event(value):
        with (directory / 'trace.jsonl').open('a', encoding='utf8') as handle:
            handle.write(json.dumps({'utc': utc(), 'id': current_id, **value}, ensure_ascii=False) + '\n')
    variant = importlib.import_module('experiments.query_decomposition.' + HERE.name + '.pipeline')
    with run_lock(directory), ExitStack() as stack:
        seed = stack.enter_context(closing(ro(OUT / 'seed.sqlite3')))
        cache = stack.enter_context(Cache(directory / 'checkpoint.sqlite3'))
        cache.put('meta', 'identity', {'protocol_sha256': sha(HERE / 'PROTOCOL.frozen.json'), 'arm': arm, 'dataset': dataset})
        budget = stack.enter_context(closing(Budget(config, directory, config['budget']['cumulative_cap_cny'])))
        live_ask = stack.enter_context(closing(RecoveryAsk(cache, seed, config, budget, event)))
        retriever = Retriever(config['resources'][dataset], device='cuda') if arm == 'A_no_only' else None
        if retriever:
            stack.callback(retriever.close)
        pipeline = variant.Pipeline(config)
        for index, case in enumerate(cases, 1):
            if parent.STOP.is_set():
                break
            current_id = case['id']
            if cache.get('prediction', current_id) is not None:
                continue
            control = controls[current_id]['trajectory']
            goal_sid = len(case['supporting_questions']) + 1
            fixed_retrieval = {}
            for n in control['nodes']:
                for key in ['original_retrieval', 'rewritten_retrieval']:
                    if n[key] is not None:
                        fixed_retrieval[n[key]['query']] = n[key]
            def retrieve(query):
                if arm == 'B_no_resolve':
                    assert query in fixed_retrieval, 'B upstream retrieval changed'
                    event({'event': 'retrieval', 'query': query, 'source': 'fixed_revised_control'})
                    return deepcopy(fixed_retrieval[query])
                key = canonical([dataset, case['qid'], query, 10])
                packet = cache.get('retrieval', key)
                if packet is None:
                    row = seed.execute('SELECT packet,origin FROM retrieval WHERE dataset=? AND qid=? AND query=?', (dataset, case['qid'], query)).fetchone()
                    packet, origin = (json.loads(row[0]), row[1]) if row else (retriever(case['qid'], query), 'new_index_search')
                    event({'event': 'retrieval', 'query': query, 'source': origin})
                    cache.put('retrieval', key, packet)
                return deepcopy(packet)
            def ask(stage, prompt, maximum, allow_final_truncation=False):
                if arm == 'B_no_resolve':
                    if stage == f'legacy/answer/{goal_sid}':
                        assert prompt == control['nodes'][-1]['child_prompt'].replace(child_prompts.RESOLVE, '', 1)
                    elif stage != 'legacy/final' and stage != f'legacy/sentence_format/{goal_sid}':
                        # Prefix answers/rewrite outputs must come from this exact frozen case.
                        call = next(c for c in controls[current_id]['calls'] if c['stage'] == stage)
                        expected = seed.execute('SELECT payload FROM requests WHERE request_id=?', (call['request_id'],)).fetchone()[0]
                        assert canonical(build_payload(config['generation'], prompt, maximum)) == expected, (current_id, stage)
                return live_ask(stage, prompt, maximum, allow_final_truncation=allow_final_truncation)
            def record(node):
                if arm == 'B_no_resolve':
                    old = control['nodes'][node['step_id'] - 1]
                    for field in ['request', 'prior_complete_answers', 'original_retrieval', 'rewritten_retrieval', 'reader_packet', 'rewrite_prompt', 'raw_rewrite']:
                        assert node[field] == old[field], (current_id, field)
                    if node['role'] == 'ordinary':
                        assert node['answer'] == old['answer'] and node['child_prompt'] == old['child_prompt']
                cache.put('node', current_id + '/' + str(node['step_id']), node)
            try:
                result = pipeline.execute(case, retrieve, ask, record_node=record)
                cache.put('prediction', current_id, result)
                emit({'arm': arm, 'dataset': dataset, 'completed': index, 'planned': 30, 'status': result['status']})
            except Exception as exc:
                failure = {'id': current_id, 'arm': arm, 'error_class': type(exc).__name__, 'error_summary': str(exc)[:240], 'status': 'incomplete'}
                save(directory / ('failure_' + hashlib.sha256(current_id.encode()).hexdigest()[:16] + '.json'), failure)
                event({'event': 'case_incomplete', **failure})
                emit(failure)
                parent.STOP.set()
                raise RuntimeError('Ablation stopped; preserve receipts, no automatic replay') from None
        (directory / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in cache.all('prediction').values()), encoding='utf8')


class NetworkEvents:
    def event(self, event, **fields):
        with (OUT / 'network.jsonl').open('a', encoding='utf8') as handle:
            handle.write(json.dumps({'utc': utc(), 'event': event, **fields}) + '\n')


def report():
    results = {}
    original_here, original_out = parent.HERE, parent.OUT
    try:
        parent.HERE = HERE
        for arm in ARMS:
            parent.OUT = OUT / arm
            value = parent.report()
            with closing(ro(read(PACKAGE / 'config.json')['budget']['ledger'])) as db:
                known, reserved = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace LIKE ?', ('%' + str((OUT / arm).resolve()) + '%',)).fetchone()
            value.update({'arm': arm, 'new_estimated_CNY': known, 'unknown_reserved_CNY': reserved, 'control': 'completed revised ordinary-child prompt quick90'})
            save(OUT / arm / 'REPORT.json', value)
            results[arm] = value
    finally:
        parent.HERE, parent.OUT = original_here, original_out
    combined = {'id': HERE.name, 'utc': utc(), 'status': 'completed' if all(r['status'] == 'completed' for r in results.values()) else 'incomplete',
        'planned_per_arm': 90, 'arms': results, 'new_estimated_CNY': sum(v['new_estimated_CNY'] for v in results.values()),
        'unknown_reserved_CNY': sum(v['unknown_reserved_CNY'] for v in results.values()), 'main_evaluation_remains_paused': True, 'candidate_not_activated': True}
    save(OUT / 'REPORT.json', combined)
    return combined


def run():
    check()
    offline_check = read(OUT / 'OFFLINE_CHECK.json')
    assert offline_check['passed'] and sha(OUT / 'seed.sqlite3') == offline_check['seed_sha256']
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    from p02_multigrain_timing_cachefix_v1.core.retrieval import Retriever
    config = read(PACKAGE / 'config.json')
    check_resources(config, rows(COHORT / 'runtime90.frozen.jsonl'))
    loading, old_load = Lock(), Retriever.load
    def serialized_load(self):
        with loading:
            return old_load(self)
    Retriever.load = serialized_load
    host = urlsplit(config['generation']['base_url']).hostname
    with temporary_api_key('api_code_qwen.txt', allowed_root=ROOT, provider='dashscope'), provider_dns(host, NetworkEvents()):
        with socket.create_connection((host, 443), timeout=15) as sock:
            with ssl.create_default_context().wrap_socket(sock, server_hostname=host):
                NetworkEvents().event('TLS_preflight', paid_request=False)
        try:
            # Fixed-material B first, then the full downstream A; no results enter the seed.
            for arm in reversed(ARMS):
                child_prompts.ARM = arm
                save(OUT / 'EXECUTION.json', {'utc': utc(), 'status': 'running', 'arm': arm, 'main_campaign_resumed': False})
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(run_dataset, arm, ds) for ds in parent.DATASETS]
                    for future in as_completed(futures):
                        future.result()
                report()
        except Exception:
            parent.STOP.set()
            report()
            raise
    return report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'offline', 'run', 'report'])
    args = parser.parse_args()
    emit(globals()[args.action]())
