"""Final-only successor validation on the SAME development90, not confirmation.

--prepare returns ready:false until the final old300 winner and all90 p02
parent outcomes are available. Only an explicit --run can submit API calls.
The unique old300 winner is locked BEFORE inspecting fresh predictions or F1.
This module never reads round2 dev90 score/gate files or any sealed150 input.
Existing g01-versus-p02 validation/confirmation artifacts remain untouched.
"""
from collections import Counter
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import ParallelBudget, locked_json
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from . import experiment as development
from . import strategy

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'fresh90_successor'
OUT = OUTPUT / HERE.name / 'fresh90_successor'
DEV = OUTPUT / 'b0_strategy_round2_v1_20261020/fresh_dev90'
DEV_INPUT = HERE.parent / 'b0_strategy_round2_v1_20261020/fresh_validation'
RUNTIME_SOURCE = DEV_INPUT / 'runtime_dev90.frozen.jsonl'
REFERENCE_SOURCE = DEV_INPUT / 'reference_dev90.local.frozen.jsonl'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def freeze(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, 'Frozen artifact changed: ' + str(path)
    else:
        locked_json(path, value)


def log(event, **fields):
    INPUT.mkdir(parents=True, exist_ok=True)
    with (INPUT / 'DECISIONS.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'utc': utc(), 'event': event, **fields}, ensure_ascii=False) + '\n')


def winner(selection, results, terminal):
    selected = selection.get('selected')
    assert selected in ['parent_p02', *strategy.ARMS]
    assert terminal.get('selected') == selected
    parent = results['parent300_F1']
    eligible = [(parent, 'parent_p02')]
    eligible += [(v['F1'], arm) for arm, v in results['arms'].items()
                 if arm in strategy.ARMS and v['scorable'] == 300 and v['F1'] > parent + 1e-12]
    score, expected = max(eligible, key=lambda item: item[0])
    assert selected == expected and math.isclose(selection['F1'], score, rel_tol=0, abs_tol=1e-12)
    return selected


def metrics_files():
    base = ROOT / 'data/multihop_comprehensive_v1_20261015'
    return [ROOT / 'experiments/query_decomposition/comprehensive_eval_v1_20261015/scoring.py',
            ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/common/scoring.py',
            base / '2wiki/official/2wikimultihop_evaluate_v1.1.py',
            base / '2wiki/native/id_aliases.json', base / 'musique/official/metrics/answer.py']


def lock_old300():
    paths = [development.OUT / name for name in ['SELECTION.frozen.json', 'FINAL_RESULTS.frozen.json', 'TERMINAL.json']]
    missing = [rel(p) for p in paths if not p.exists()]
    if missing:
        return None, {'ready': False, 'reason': 'Awaiting final old300 successor selection', 'missing': missing}
    selection, results, terminal = map(read, paths)
    selected = winner(selection, results, terminal)
    if selected == 'parent_p02':
        return None, {'ready': False, 'reason': 'Old300 retained p02; no fresh successor is selected', 'selected': selected}
    protocol = read(HERE / 'PROTOCOL.frozen.json')
    development.verify(protocol)
    pins = [Path(__file__), HERE / 'strategy.py', HERE / 'experiment.py', HERE / 'PROTOCOL.frozen.json',
            HERE.parent / 'b0_strategy_round2_v1_20261020/experiment.py',
            DEV / 'PROTOCOL.frozen.json', DEV / 'CANDIDATE.frozen.json', *paths, *metrics_files()]
    missing_pins = [rel(p) for p in pins if not p.exists()]
    if missing_pins:
        return None, {'ready': False, 'reason': 'Awaiting frozen execution/scoring prerequisites',
                      'selected': selected, 'missing': missing_pins}
    expected = {'selected': selected, 'parent': 'p02_role_headers',
                'old300_selection_sha256': sha(paths[0]),
                'old300_results_sha256': sha(paths[1]),
                'source_pins': {rel(p): sha(p) for p in pins},
                'fixed_before_successor_fresh_outcome_inspection': True,
                'sample_role': 'Same development90 reused for a fixed successor; not a fresh confirmation.',
                'existing_round2_candidate_and_sealed_plan_unchanged': True}
    target = OUT / 'CANDIDATE.frozen.json'
    is_new = not target.exists()
    freeze(target, expected)
    if is_new:
        log('lock_unique_old300_successor_before_fresh_inspection', selected=selected,
            candidate_protocol_sha256=sha(target), no_fresh_scores_or_gate_read=True)
    return expected, None


def parent_record(case):
    latest = None
    for path in sorted((DEV / 'rounds').glob(f'r[0-9][0-9][0-9]/cases/{case["dataset"]}/p02_role_headers/{case["qid"]}.json')):
        value = read(path)
        latest = (path, value)
        # Type/status checks only here; the final answer text is never used to
        # select the successor. Selection has already been locked by old300.
        if isinstance(value.get('prediction'), str) and (
                value['status'].startswith('success') or value['status'] == 'algorithm_failure'):
            return latest
    return latest


def classify_parent(case, value):
    if value is None:
        return 'infrastructure_missing'
    assert all(value.get(k) == case[k] for k in ['id', 'qid', 'dataset', 'question'])
    status = value.get('status', '')
    if status == 'algorithm_failure' and isinstance(value.get('prediction'), str):
        # No assumption that nodes/root/goal exist; no corrective final call.
        return 'inherit_algorithm_failure'
    if status.startswith('success') and isinstance(value.get('prediction'), str):
        assert isinstance(value.get('nodes'), list) and value['nodes']
        assert value['nodes'][-1]['role'] == 'goal'
        assert value['nodes'][-1]['request'] == case['question']
        assert isinstance(value.get('original_retrieval'), dict)
        return 'execute_final_only'
    return 'infrastructure_missing'


def prepare():
    locked, blocked = lock_old300()
    if blocked:
        return blocked
    prerequisites = [RUNTIME_SOURCE, REFERENCE_SOURCE, DEV / 'REFERENCES.frozen.json']
    missing = [rel(p) for p in prerequisites if not p.exists()]
    if missing:
        return {'ready': False, 'selected': locked['selected'],
                'reason': 'Awaiting frozen dev90 runtime/references', 'missing': missing,
                'candidate_already_locked': True, 'new_api_calls': 0}
    runtime = rows(RUNTIME_SOURCE)
    from experiments.query_decomposition.b0_strategy_round2_v1_20261020.fresh_runner import safe_case
    cases = [safe_case(c) for c in runtime]
    assert len(cases) == len({c['id'] for c in cases}) == 90
    assert Counter(c['dataset'] for c in cases) == {'hotpotqa': 30, '2wiki': 30, 'musique': 30}
    previous = read(DEV / 'PROTOCOL.frozen.json')
    assert sha(RUNTIME_SOURCE) == previous['runtime_sha256']
    assert read(DEV / 'CANDIDATE.frozen.json')['selected'] == 'p02_role_headers'
    assert sha(REFERENCE_SOURCE) == read(DEV / 'REFERENCES.frozen.json')['references_sha256']
    missing, manifest, inherited = [], [], []
    for case in cases:
        pair = parent_record(case)
        path, parent = pair if pair else (None, None)
        kind = classify_parent(case, parent)
        if kind == 'infrastructure_missing':
            missing.append({'id': case['id'], 'status': parent.get('status') if parent else 'unstarted'})
            continue
        if kind == 'execute_final_only':
            prompt, context = strategy.render_final(case['question'], parent['nodes'], parent['original_retrieval'],
                                                    previous['readout'], 'parent_p02')
            assert prompt == parent['final_prompt'] and context == parent['final_context']
        else:
            inherited.append(case['id'])
        manifest.append({'id': case['id'], 'base': rel(path), 'base_sha256': sha(path),
                         'control': rel(path), 'control_sha256': sha(path), 'cache_sources': [],
                         'inherit_algorithm_failure': kind == 'inherit_algorithm_failure'})
    if missing:
        return {'ready': False, 'selected': locked['selected'], 'reason': 'Awaiting complete p02 parent90',
                'parent_scorable': 90 - len(missing), 'missing': missing,
                'candidate_already_locked': True, 'new_api_calls': 0}
    INPUT.mkdir(parents=True, exist_ok=True)
    runtime_target = INPUT / 'runtime90.frozen.jsonl'
    source_target = INPUT / 'SOURCES.frozen.jsonl'
    for target, values in [(runtime_target, cases), (source_target, manifest)]:
        text = ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values)
        if target.exists():
            assert target.read_text(encoding='utf-8') == text
        else:
            target.write_text(text, encoding='utf-8')
    plan = {'selected': locked['selected'], 'parent': 'p02_role_headers', 'cases': 90,
            'arms': [locked['selected']], 'model': previous['model'], 'temperature': 0,
            'enable_thinking': False, 'readout': previous['readout'], 'final_max_tokens': 64,
            'sample_role': locked['sample_role'],
            'runtime_source': rel(RUNTIME_SOURCE), 'runtime_source_sha256': sha(RUNTIME_SOURCE),
            'runtime_sha256': sha(runtime_target), 'manifest_sha256': sha(source_target),
            'reference_source': rel(REFERENCE_SOURCE), 'reference_sha256': sha(REFERENCE_SOURCE),
            'candidate_protocol_sha256': sha(OUT / 'CANDIDATE.frozen.json'),
            'source_pins': locked['source_pins'],
            'inherited_algorithm_failure_ids': inherited,
            'budget': {'cumulative_cap_cny': 300, 'dispatch_ceiling_cny': 299.9, 'local_subcap_cny': 2},
            'fixed': 'Original questions, R02 plans, all QA including goal, packets and root Top10 unchanged; final-only intervention.',
            'retries': 'One immutable infrastructure recovery, no successful or algorithm-failed case rerun.',
            'confirmation': False, 'sealed150_opened': False}
    freeze(INPUT / 'PROTOCOL.frozen.json', plan)
    return {'ready': True, 'selected': locked['selected'], 'cases': 90,
            'inherited_algorithm_failures': len(inherited), 'plan': plan, 'new_api_calls': 0}


def verify(plan):
    for name, expected in plan['source_pins'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(RUNTIME_SOURCE) == plan['runtime_source_sha256']
    assert sha(REFERENCE_SOURCE) == plan['reference_sha256']
    assert sha(INPUT / 'runtime90.frozen.jsonl') == plan['runtime_sha256']
    assert sha(INPUT / 'SOURCES.frozen.jsonl') == plan['manifest_sha256']
    assert sha(OUT / 'CANDIDATE.frozen.json') == plan['candidate_protocol_sha256']
    for source in rows(INPUT / 'SOURCES.frozen.jsonl'):
        assert sha(ROOT / source['base']) == source['base_sha256']


def chosen(arm, cid):
    ds, qid = cid.split(':', 1)
    last = None
    for path in sorted((OUT / 'rounds').glob(f'*/cases/{ds}/{arm}/{qid}.json')):
        last = read(path)
        if isinstance(last.get('prediction'), str) and (
                last['status'].startswith('success') or last['status'] == 'algorithm_failure'):
            return last
    return last


def budget_snapshot():
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        all_cost = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
        own = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ?', (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()
    return {'global_known_cny': all_cost[0], 'global_reserved_cny': all_cost[1],
            'global_occupied_cny': sum(all_cost), 'new_known_cny': own[0], 'new_reserved_cny': own[1],
            'new_submissions': own[2], 'cap_cny': 300}


class Budget(ParallelBudget):
    def __init__(self):
        super().__init__(OUTPUT / 'budget.sqlite3', 300)

    def reserve(self, rid, namespace, amount):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges').fetchone()[0]
            own = self.db.execute('SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace LIKE ?', (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()[0]
            if total + amount > 299.9 or own + amount > 2:
                raise RuntimeError('Global300/local2 dispatch ceiling; no submission')
            if self.db.execute('SELECT 1 FROM charges WHERE id=?', (rid,)).fetchone():
                raise RuntimeError('Existing liability cannot be submitted again')
            self.db.execute("INSERT INTO charges VALUES(?,?,?,NULL,'reserved')", (rid, namespace, amount))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise


def backend(plan):
    bound = development.backend()
    bound.HERE, bound.OUT, bound.ARMS = INPUT, OUT, [plan['selected']]
    bound.final_prompt = lambda case, nodes, original, readout, arm: strategy.render_final(case['question'], nodes, original, readout, arm)
    bound.log, bound.budget_snapshot, bound.chosen, bound.summarize = log, budget_snapshot, chosen, summarize
    bound.Budget = Budget
    return bound


def inherit_failures(plan, cases, manifest):
    count = 0
    for case in cases:
        source = manifest[case['id']]
        if not source['inherit_algorithm_failure'] or chosen(plan['selected'], case['id']) is not None:
            continue
        parent = read(ROOT / source['control'])
        assert classify_parent(case, parent) == 'inherit_algorithm_failure'
        value = {k: deepcopy(parent[k]) for k in ['id', 'qid', 'dataset', 'question', 'status', 'prediction']}
        value.update(arm=plan['selected'], parent_source=source['control'], parent_sha256=source['control_sha256'],
                     inherited_algorithm_failure=True, no_new_calls=True, logical_calls=[],
                     reason='Inherited frozen p02 algorithm failure; a final-only intervention does not repair the shared prefix.')
        target = OUT / 'rounds' / 'inherited_parent_failures' / 'cases' / case['dataset'] / plan['selected'] / (case['qid'] + '.json')
        freeze(target, value)
        count += 1
    return count


def summarize(plan):
    """Scoring happens only after the unique candidate freeze; no reselection."""
    verify(plan)
    cases = rows(INPUT / 'runtime90.frozen.jsonl')
    manifest = {r['id']: r for r in rows(INPUT / 'SOURCES.frozen.jsonl')}
    refs = {r['id']: r for r in rows(REFERENCE_SOURCE)}
    scorers = {ds: metric(ds) for ds in ['hotpotqa', '2wiki', 'musique']}
    values, states = [], Counter()
    for case in cases:
        parent = read(ROOT / manifest[case['id']]['control'])
        candidate = chosen(plan['selected'], case['id'])
        state = candidate['status'] if candidate else 'unstarted'
        states[state] += 1
        valid = candidate and isinstance(candidate.get('prediction'), str) and (
            state.startswith('success') or state == 'algorithm_failure')
        ref = refs[case['id']]
        assert ref['question'] == case['question']
        p = ({'em': 0., 'f1': 0.} if parent['status'] == 'algorithm_failure' else
             scorers[case['dataset']](parent['prediction'], ref))
        c = ({'em': 0., 'f1': 0.} if valid and state == 'algorithm_failure' else
             scorers[case['dataset']](candidate['prediction'], ref) if valid else {'em': None, 'f1': None})
        values.append({'id': case['id'], 'dataset': case['dataset'], 'parent_prediction': parent['prediction'],
                       'candidate_prediction': candidate['prediction'] if candidate else None,
                       'status': state, 'parent_f1': p['f1'], 'candidate_f1': c['f1'],
                       'parent_em': p['em'], 'candidate_em': c['em'],
                       'delta': c['f1'] - p['f1'] if valid else None})
    def group(data):
        matched = [v for v in data if v['delta'] is not None]
        n, m = len(data), len(matched)
        return {'planned': n, 'scorable': m, 'missing': n - m,
                'parent_F1_all': sum(v['parent_f1'] for v in data) / n,
                'candidate_F1_full': sum(v['candidate_f1'] for v in matched) / n if m == n else None,
                'same_parent_F1_available': sum(v['parent_f1'] for v in matched) / m if m else None,
                'candidate_F1_available': sum(v['candidate_f1'] for v in matched) / m if m else None,
                'delta_available': sum(v['delta'] for v in matched) / m if m else None,
                'up': sum(v['delta'] > 1e-12 for v in matched),
                'tie': sum(abs(v['delta']) <= 1e-12 for v in matched),
                'down': sum(v['delta'] < -1e-12 for v in matched)}
    matched = [v for v in values if v['delta'] is not None]
    rng, boot = np.random.default_rng(20261020), np.zeros(5000)
    for ds in scorers:
        delta = np.array([v['delta'] for v in matched if v['dataset'] == ds])
        if len(delta):
            boot += delta[rng.integers(len(delta), size=(5000, len(delta)))].sum(1) / len(matched)
    report = {'utc': utc(), 'selected_from_old300_before_successor_scoring': plan['selected'],
              'sample_role': plan['sample_role'], 'overall': group(values),
              'paired95_descriptive': np.quantile(boot, [.025, .975]).tolist() if matched else None,
              'bootstrap': '5000 paired resamples within dataset, seed20261020; descriptive development interval',
              'by_dataset': {ds: group([v for v in values if v['dataset'] == ds]) for ds in scorers},
              'statuses': dict(states), 'inherited_algorithm_failures': len(plan['inherited_algorithm_failure_ids']),
              'budget': budget_snapshot(),
              'cost_scope': 'Incremental final-stage new submissions plus unreconciled reserves; frozen upstream work is reused, not repriced as fresh full-RAG execution.',
              'sealed150_opened': False, 'existing_round2_plan_unchanged': True}
    locked_json(OUT / 'RESULTS_LATEST.json', report)
    target = OUT / 'PAIRED_DETAILS.jsonl'
    tmp = target.with_suffix('.jsonl.tmp')
    tmp.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values), encoding='utf-8')
    tmp.replace(target)
    return report


def run():
    prepared = prepare()
    if not prepared['ready']:
        return prepared
    plan = prepared['plan']
    verify(plan)
    if (OUT / 'TERMINAL.json').exists():
        terminal = read(OUT / 'TERMINAL.json')
        assert terminal['selected'] == plan['selected']
        # Do not overwrite the existing immutable summary with a new UTC or
        # a later unrelated global budget snapshot, and do not start round3.
        return {'already_terminal': True, 'terminal': terminal,
                'results': read(OUT / 'RESULTS_LATEST.json'), 'new_api_calls': 0}
    cases = rows(INPUT / 'runtime90.frozen.jsonl')
    manifest = {r['id']: r for r in rows(INPUT / 'SOURCES.frozen.jsonl')}
    inherited = inherit_failures(plan, cases, manifest)
    log('start_fixed_successor_final_only', selected=plan['selected'], inherited_failures=inherited,
        cases=90, candidate_protocol_sha256=plan['candidate_protocol_sha256'], budget=budget_snapshot())
    engine = backend(plan)
    engine.run_phase(plan, cases, plan['selected'], 'fresh90', manifest)
    report = summarize(plan)
    verify(plan)
    if report['overall']['scorable'] == 90:
        freeze(OUT / 'FINAL_RESULTS.frozen.json', report)
    freeze(OUT / 'TERMINAL.json', {'selected': plan['selected'], 'complete90': report['overall']['scorable'] == 90,
                                 'sample_role': plan['sample_role'], 'budget': report['budget'],
                                 'round2_candidate_not_replaced': True, 'confirmation_started': False})
    return report


def self_test():
    # Pure admission/schema tests. No fresh90 parent predictions or references.
    example = {'id': 'musique:fixture', 'qid': 'fixture', 'dataset': 'musique', 'question': 'Fixture question?'}
    failed = {**example, 'status': 'algorithm_failure', 'prediction': ''}
    assert classify_parent(example, failed) == 'inherit_algorithm_failure'
    assert classify_parent(example, {**failed, 'nodes': [{'role': 'ordinary'}]}) == 'inherit_algorithm_failure'
    assert classify_parent(example, None) == 'infrastructure_missing'
    assert classify_parent(example, {**example, 'status': 'incomplete', 'prediction': None}) == 'infrastructure_missing'
    try:
        classify_parent(example, {**example, 'status': 'success', 'prediction': 'Fixture'})
    except AssertionError:
        pass
    else:
        raise AssertionError('Success without actual root/nodes was accepted')
    result = {'parent300_F1': .70, 'arms': {'q03_question_last': {'scorable': 299, 'F1': .99},
                                         'e03_goal_evidence': {'scorable': 300, 'F1': .71}}}
    assert winner({'selected': 'e03_goal_evidence', 'F1': .71}, result,
                  {'selected': 'e03_goal_evidence'}) == 'e03_goal_evidence'
    tied = deepcopy(result)
    tied['arms']['e03_goal_evidence']['F1'] = .70
    assert winner({'selected': 'parent_p02', 'F1': .70}, tied, {'selected': 'parent_p02'}) == 'parent_p02'
    # Replay one OLD300 all-duplicate e03 prompt from a real p02 receipt.
    # This checks the no-new-information exact cache path, not fresh inference.
    from experiments.query_decomposition.b0_six_single_changes_v1_20261020.receipts import receipt
    from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
    old_cases = rows(HERE / 'runtime300.frozen.jsonl')
    sources = {r['id']: r for r in rows(HERE / 'SOURCES.frozen.jsonl')}
    replay = None
    for case in old_cases:
        src = sources[case['id']]
        parent, base = read(ROOT / src['control']), read(ROOT / src['base'])
        if strategy.goal_extra_document(parent['nodes'], base['original_retrieval']) is not None:
            continue
        prompt, context = strategy.render_final(case['question'], parent['nodes'], base['original_retrieval'],
                                                read(HERE / 'PROTOCOL.frozen.json')['readout'], 'e03_goal_evidence')
        assert prompt == parent['final_prompt'] and context == parent['final_context']
        req = payload('qwen-plus-2025-12-01', prompt, 64)
        from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest
        call = next(v for v in parent['logical_calls'] if v['payload_sha256'] == digest(req))
        raw = receipt(call, req)
        assert raw['status'] == 'success' and raw['result']['answer'].strip() == parent['prediction']
        replay = case['id']
        break
    assert replay is not None
    before = (development.HERE, development.OUT)
    engine = backend({'selected': 'e03_goal_evidence'})
    assert engine.HERE == INPUT and engine.OUT == OUT and engine.ARMS == ['e03_goal_evidence']
    assert engine.run_case.__globals__ is engine.__dict__
    assert engine.run_phase.__globals__ is engine.__dict__
    assert engine.Budget is Budget and engine.summarize is summarize and engine.chosen is chosen
    assert (development.HERE, development.OUT) == before
    return {'passed': True, 'algorithm_failure_without_root_or_nodes_inherited': True,
            'success_missing_schema_rejected': True, 'private_binding_and_local2_budget': True,
            'infrastructure_missing_blocks_prepare': True, 'partial_candidate_not_selected': True,
            'old300_exact_parent_payload_replay': replay, 'fresh90_outcomes_opened': False,
            'new_api_calls': 0, 'new_retrieval_calls': 0, 'new_remote_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    result = self_test() if args.self_test else run() if args.run else prepare()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
