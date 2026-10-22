"""One approved fixed90 contrast, starting from original P02, never revised QA."""
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Lock
import argparse
import difflib
import hashlib
import importlib
import json
import re
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021 import run as engine
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021.run_r004 import RecoveryAsk
from p02_multigrain_timing_cachefix_v1.core.cache import utc
from . import child_prompts

PACKAGE, COHORT, PLAN = engine.PACKAGE, engine.COHORT, engine.PLAN_DIR
OUT = engine.OUT.parent / HERE.name
read, rows, sha, save, emit = engine.read, engine.rows, engine.sha, engine.save, engine.emit
PARENT_PROTOCOL = engine.HERE / 'PROTOCOL.frozen.json'
ENGINE_PINS = [engine.HERE / name for name in ['run.py', 'run_r004.py', 'run_r002.py']]
AUTHORITY = 'User: 我们尝试一下在冻结版本的基础上，同时删除子回答和goal回答prompt的 #N，其他尽量与冻结版本保持不变，再次快速评估一下；范围确认：统一移除全部问题编号与结果引用（推荐）'


def check():
    protocol = read(HERE / 'PROTOCOL.frozen.json')
    for path, expected in protocol['pins'].items():
        assert sha(ROOT / path) == expected, path
    assert read(PLAN / 'PLAN.json')['execution_approval']['approved'] is False
    return protocol


def prepare():
    # Before rebinding engine globals, verify prior frozen source/cohort pins.
    engine.check_pins()
    assert not (HERE / 'PROTOCOL.frozen.json').exists()
    source = (PACKAGE / 'core/pipeline.py').read_text(encoding='utf8')
    changed = source.replace('from . import METHOD, ROOT',
        "from p02_multigrain_timing_cachefix_v1.core import ROOT\nMETHOD = 'p02_frozen_no_refs_quick90_v1'", 1)
    changed = changed.replace('from . import child_prompts, rewrite_prompts, final_prompts',
        'from p02_multigrain_timing_cachefix_v1.core import rewrite_prompts, final_prompts\nfrom . import child_prompts', 1)
    (HERE / 'pipeline.py').write_text(changed, encoding='utf8')
    (HERE / 'PIPELINE_CHANGE.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True), changed.splitlines(True), fromfile='original_frozen_P02', tofile='independent_reference_removal_variant')), encoding='utf8')
    original = read(PARENT_PROTOCOL)
    pins = {}
    for group in ['inputs', 'original_package_pins', 'hold_pins']:
        pins.update(original[group])
    paths = [HERE / 'run.py', HERE / 'child_prompts.py', HERE / 'pipeline.py', *ENGINE_PINS,
        PACKAGE / 'config.json', PACKAGE / 'prompts/child_examples.json', PACKAGE / 'prompts/final_examples.json',
        PACKAGE / 'data/replay.sqlite3']
    pins.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    protocol = {'id': HERE.name, 'utc': utc(), 'authority': AUTHORITY,
        'control': 'Original frozen p02_multigrain_timing_cachefix_v1; full fixed90 F1 0.7668887868810469. Not the damaged revised child prompt.',
        'sample': 'Same already exposed quick90; 30 each HotpotQA/2Wiki/MuSiQue; no resampling',
        'planned': 90, 'changes': [
            'Delete only the sentence Resolve each #N from its numbered prior question and answer. from both ordinary and goal instructions.',
            'Remove #<step_id> prefixes from previous-QA labels, and #<step_id> from Current subquestion headers, in both roles and worked examples.',
            'Replace the two #1 result references in ordinary worked-example questions by their descriptive entity phrases.'],
        'preserved': 'All other original rules, wording, empty-history wording, examples including original questions/answers/passages, duplicated goal question, GOAL_RULE, full history order, rewrite/final prompts, formatter, model, limits, decomposition, retrieval and merge remain frozen.',
        'execution': 'Full downstream original pipeline. Changed answers can change subsequent rewrites, retrieval and final outputs. Exact successful full-payload reuse only from original frozen P02 and this run, not earlier revised candidates.',
        'destination': read(PACKAGE / 'config.json')['generation']['base_url'],
        'external_data': 'User-approved same fixed90 Qwen evaluation: questions, model-generated QA, retrieved passages and worked examples; billed existing endpoint.',
        'retry': 'Preserve established transport connect-before-send retries2, SDK POST retries0; no automatic failed/unknown request replay or score-based resampling.',
        'metrics': 'Unchanged native final Answer-F1/EM; paired improved/tied/harmed; full90/missingness; full-flow input/output tokens and estimated expense; new charges separate. Not a timing benchmark.',
        'main_evaluation_remains_paused': True, 'pins': pins}
    save(HERE / 'PROTOCOL.frozen.json', protocol)
    main = read(PLAN / 'PLAN.json')
    assert main['execution_approval']['approved'] is False
    entry = {'decision_id': HERE.name.upper(), 'scope': protocol['control'] + ' Remove references in both answer roles only.',
        'protocol': str((HERE / 'PROTOCOL.frozen.json').relative_to(ROOT)), 'campaign_resume_authorized': False, 'new_inference': True}
    main.setdefault('standalone_diagnostics', []).append(entry)
    save(PLAN / 'PLAN.json', main)
    with (PLAN / 'DECISIONS.jsonl').open('a', encoding='utf8') as handle:
        handle.write(json.dumps({'id': entry['decision_id'], 'utc': utc(), 'status': 'approved_scoped_quick90',
            'authority': AUTHORITY, **entry, 'destination': protocol['destination'], 'main_evaluation_remains_paused': True}, ensure_ascii=False) + '\n')
    OUT.mkdir(parents=True, exist_ok=True)
    return {'prepared': True, 'planned': 90, 'control': 'original frozen P02', 'remove_labels_and_result_references': True}


def expected_edit(prompt):
    assert prompt.count(child_prompts.RESOLVE) == 1
    prompt = prompt.replace(child_prompts.RESOLVE, '', 1)
    return child_prompts.edit_example(prompt)


def offline():
    check()
    assert not (OUT / 'OFFLINE_CHECK.json').exists()
    config = read(PACKAGE / 'config.json')
    controls, changed_counts = [], {'ordinary': 0, 'goal': 0}
    samples = {}
    candidate = importlib.import_module('experiments.query_decomposition.' + HERE.name + '.pipeline')
    pipeline = candidate.Pipeline(config)
    bindings = {v['id']: v for v in rows(COHORT / 'SOURCE_BINDINGS.frozen.jsonl')}
    with closing(sqlite3.connect((PACKAGE / 'data/replay.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as seed:
        for case in rows(COHORT / 'runtime90.frozen.jsonl'):
            control = engine.baseline_replay(case, seed, config)
            binding = bindings[case['id']]
            assert control['calls'][-1]['request_id'] == binding['final_request_id']
            assert sha(ROOT / binding['source']) == binding['source_sha256']
            for node in control['trajectory']['nodes']:
                assert not re.search(r'#\d+', node['request'])
                prompt = child_prompts.child(case['question'], node['step_id'], node['request'], node['prior_complete_answers'],
                    engine.baseline.passages(node['reader_packet']), pipeline.child_examples, role=node['role'])
                if node['role'] == 'goal':
                    prompt = prompt[:-len('Answer:')] + child_prompts.GOAL_RULE + '\n\nAnswer:'
                assert prompt == expected_edit(node['child_prompt']), (case['id'], node['step_id'])
                assert 'Resolve each #N' not in prompt
                assert prompt != node['child_prompt']
                # Data text is untouched. Only prompt labels/rules/example questions change.
                assert engine.baseline.passages(node['reader_packet']) in prompt
                changed_counts[node['role']] += 1
                samples.setdefault(node['role'], (node['child_prompt'], prompt))
            controls.append(control)
    assert changed_counts == {'ordinary': 176, 'goal': 90}
    (OUT / 'control90.jsonl').write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in controls), encoding='utf8')
    for role, (before, after) in samples.items():
        (OUT / (role + '_CONTROL.txt')).write_text(before, encoding='utf8')
        (OUT / (role + '_CANDIDATE.txt')).write_text(after, encoding='utf8')
        (OUT / (role + '_CHANGE.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='original_frozen_P02_' + role, tofile='remove_references_' + role)), encoding='utf8')
    result = {'passed': True, 'control_replayed': 90, 'changed_prompts': changed_counts,
        'only_approved_reference_edits_verified_on_identical_inputs': True, 'new_API_requests': 0,
        'main_evaluation_remains_paused': True}
    save(OUT / 'OFFLINE_CHECK.json', result)
    return result


@contextmanager
def activate():
    names = ['HERE', 'OUT', 'MODULE', 'ScopedAsk', 'check_pins']
    previous = {n: getattr(engine, n) for n in names}
    engine.HERE, engine.OUT = HERE, OUT
    engine.MODULE = 'experiments.query_decomposition.' + HERE.name
    engine.ScopedAsk, engine.check_pins = RecoveryAsk, check
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)


def report():
    with activate():
        result = engine.report()
    result['control'] = 'original frozen P02 (not revised prompt)'
    result['candidate_not_activated'] = True
    save(OUT / 'REPORT.json', result)
    return result


def run():
    check()
    assert read(OUT / 'OFFLINE_CHECK.json')['passed']
    from p02_multigrain_timing_cachefix_v1.core.retrieval import Retriever
    lock, previous_load = Lock(), Retriever.load
    def serialized_load(self):
        with lock:
            return previous_load(self)
    Retriever.load = serialized_load
    with activate():
        engine.run()
    return report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'offline', 'run', 'report'])
    args = parser.parse_args()
    emit(globals()[args.action]())
