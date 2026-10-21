"""Two final-reader interventions on frozen p02; old300 development only.

The round2 dev90 and its sealed150 plan are separate and remain unchanged.
No training, new decomposition, retrieval or child answering is performed.
"""
from collections import Counter
import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.prepare import read, rows, sha, save, save_rows
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
PREV_E = HERE.parent / 'b0_strategy_round2_v1_20261020'
PREV_O = OUTPUT / PREV_E.name
SIX_E = HERE.parent / 'b0_six_single_changes_v1_20261020'


def strategies():
    from . import strategy
    return strategy


def log(event, **fields):
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / 'DECISIONS.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps({'utc': utc(), 'event': event, **fields}, ensure_ascii=False) + '\n')


def budget_snapshot():
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        total = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
        own = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0),count(*) FROM charges WHERE namespace LIKE ?',
                         (str(OUT.relative_to(OUTPUT)) + '%',)).fetchone()
    return {'global_known_cny': total[0], 'global_reserved_cny': total[1], 'global_occupied_cny': sum(total),
            'new_known_cny': own[0], 'new_reserved_cny': own[1], 'new_submissions': own[2], 'cap_cny': 300}


def completed_parent(cid):
    ds, qid = cid.split(':', 1)
    for path in sorted((PREV_O / 'rounds').glob(f'*/cases/{ds}/p02_role_headers/{qid}.json')):
        value = read(path)
        if value['status'].startswith('success') and isinstance(value.get('prediction'), str):
            return path, value
    raise AssertionError('Missing completed p02: ' + cid)


def verify(plan):
    for path, value in plan['source_pins'].items():
        assert sha(ROOT / path) == value, path
    for name in ['runtime300.frozen.jsonl', 'SOURCES.frozen.jsonl']:
        assert sha(HERE / name) == plan['input_hashes'][name]


def prepare():
    target = HERE / 'PROTOCOL.frozen.json'
    if target.exists():
        plan = read(target)
        verify(plan)
        return plan
    module = strategies()
    previous = read(PREV_E / 'PROTOCOL.frozen.json')
    assert read(PREV_O / 'SELECTION.frozen.json')['selected'] == 'p02_role_headers'
    # Verify the frozen predecessor without modifying its preparation or outputs.
    from experiments.query_decomposition.b0_strategy_round2_v1_20261020.experiment import verify as verify_parent
    verify_parent(previous)
    cases = rows(PREV_E / 'runtime300.frozen.jsonl')
    assert len(cases) == 300 and Counter(c['dataset'] for c in cases) == {'hotpotqa': 100, '2wiki': 100, 'musique': 100}
    assert all(not {'answer', 'gold', 'aliases', 'prediction', 'f1'}.intersection(c) for c in cases)
    prior = {r['id']: r for r in rows(PREV_E / 'SOURCES.frozen.jsonl')}
    manifest = []
    for case in cases:
        src = prior[case['id']]
        base = read(ROOT / src['base'])
        assert sha(ROOT / src['base']) == src['base_sha256']
        path, parent = completed_parent(case['id'])
        prompt, context = module.render_final(case['question'], parent['nodes'], base['original_retrieval'], previous['readout'], 'parent_p02')
        assert prompt == parent['final_prompt'] and context == parent['final_context']
        # Candidate renderers consume public questions, actual nodes and packets only.
        for arm in module.ARMS:
            changed, changed_context = module.render_final(case['question'], parent['nodes'], base['original_retrieval'], previous['readout'], arm)
            assert isinstance(changed, str) and isinstance(changed_context, str)
        manifest.append({'id': case['id'], 'base': src['base'], 'base_sha256': src['base_sha256'],
                         'control': str(path.relative_to(ROOT)), 'control_sha256': sha(path), 'cache_sources': []})
    OUT.mkdir(parents=True, exist_ok=True)
    save_rows(HERE / 'runtime300.frozen.jsonl', cases)
    save_rows(HERE / 'SOURCES.frozen.jsonl', manifest)
    audit = read(HERE / 'OFFLINE_AUDIT.json')
    assert audit['passed'] and audit['counts']['parent_prompts_and_contexts_exact'] == 300
    assert audit['source_pins'][str(Path(module.__file__).relative_to(ROOT))] == sha(Path(module.__file__))
    save(HERE / 'OFFLINE_AUDIT.frozen.json', audit)
    paths = [Path(__file__), Path(module.__file__), PREV_E / 'experiment.py', PREV_E / 'strategy.py',
             HERE / 'OFFLINE_AUDIT.py', HERE / 'OFFLINE_AUDIT.frozen.json',
             PREV_E / 'PROTOCOL.frozen.json', PREV_O / 'SELECTION.frozen.json', PREV_O / 'FINAL_RESULTS.frozen.json',
             SIX_E / 'receipts.py',
             HERE.parent / 'component_chain_c08_c02_v1_20261019/pipeline.py',
             HERE.parent / 'multigrain_prompt_v3_20261017/prompts.py',
             HERE.parent / 'multigrain_prompt_v3_20261017/final_examples.json']
    plan = {'utc': utc(), 'parent': 'frozen p02_role_headers', 'arms': module.ARMS,
            'model': previous['model'], 'temperature': 0, 'enable_thinking': False, 'readout': previous['readout'],
            'sample_role': 'Exposed diagnostic300, full300 for both arms; no outcome-based pilot gate.',
            'n_per_arm': 300,
            'single_factor': {
                'q03_question_last': 'Relocate the actual Original question block after Context, before final readout; no text/information added or removed.',
                'e03_goal_evidence': 'Keep every p02 QA and original Top10; append exactly the first goal-reader passage with doc_id absent from original Top10. No new retrieval. This adds evidence, not merely formatting.'},
            'selection': 'Highest complete300 native mean F1 against p02; keep p02 on tie or incomplete candidate. Development selection only.',
            'fixed': ['R02', 'ordinary and goal QA', 'all rewrite outputs', 'retrieval packets', 'p02 QA headers',
                      'root Top10', 'all auxiliary QA', 'model', 'examples', 'P1 readout', 'final64'],
            'fresh_data_boundary': 'Do not read/use round2 dev90 scores or sealed150 to choose these interventions. Existing p02 dev90 validation and its confirmation candidate remain unchanged. Any winning successor needs its own frozen validation protocol.',
            'cache': 'Exact complete payload only; first successful p02 receipt or successful prior infrastructure attempt. No successful request rerun.',
            'retries': 'One immutable infrastructure recovery; retain unknown liability; no algorithm retry.',
            'budget': {'cumulative_cap_cny': 300, 'dispatch_ceiling_cny': 299.9, 'local_subcap_cny': 5},
            'input_hashes': {p.name: sha(p) for p in [HERE / 'runtime300.frozen.jsonl', HERE / 'SOURCES.frozen.jsonl']},
            'source_pins': {**previous['source_pins'], **{str(p.relative_to(ROOT)): sha(p) for p in paths}}}
    save(target, plan)
    save(OUT / 'OFFLINE_VALIDATION.frozen.json', {'passed': True, 'parent_prompts_rebuilt': 300,
                                               'new_API_calls': 0, 'new_retrieval_calls': 0})
    save(OUT / 'BUDGET_START.frozen.json', budget_snapshot())
    log('freeze_two_single_factors', source_pins=plan['source_pins'], n_per_arm=300, previous_candidates_preserved=True,
        original_fresh90_plan_unchanged=True, sealed150_opened=False, budget=plan['budget'])
    return plan


def chosen(arm, cid):
    ds, qid = cid.split(':', 1)
    latest = None
    for path in sorted((OUT / 'rounds').glob(f'*/cases/{ds}/{arm}/{qid}.json')):
        latest = read(path)
        if isinstance(latest.get('prediction'), str) and (latest['status'].startswith('success') or latest['status'] == 'algorithm_failure'):
            return latest
    return latest


def summarize(plan):
    cases = rows(HERE / 'runtime300.frozen.jsonl')
    manifest = {v['id']: v for v in rows(HERE / 'SOURCES.frozen.jsonl')}
    refs = {v['id']: v for v in rows(SIX_E / 'references300.local.frozen.jsonl')}
    scorers = {ds: metric(ds) for ds in ['hotpotqa', '2wiki', 'musique']}
    parents = {c['id']: scorers[c['dataset']](read(ROOT / manifest[c['id']]['control'])['prediction'], refs[c['id']]) for c in cases}
    all_details, arms = [], {}
    for arm in plan['arms']:
        values, states = [], Counter()
        for case in cases:
            value = chosen(arm, case['id'])
            states[value['status'] if value else 'unstarted'] += 1
            if not value or not isinstance(value.get('prediction'), str) or not (value['status'].startswith('success') or value['status'] == 'algorithm_failure'):
                continue
            score = scorers[case['dataset']](value['prediction'], refs[case['id']])
            parent = parents[case['id']]
            row = {'id': case['id'], 'dataset': case['dataset'], 'arm': arm,
                   'prediction': value['prediction'], 'f1': score['f1'], 'em': score['em'],
                   'parent_f1': parent['f1'], 'delta_parent': score['f1'] - parent['f1']}
            values.append(row)
            all_details.append(row)
        rng = np.random.default_rng(20261020)
        boot = np.zeros(5000)
        if values:
            for ds in scorers:
                dd = np.array([v['delta_parent'] for v in values if v['dataset'] == ds])
                if len(dd):
                    boot += dd[rng.integers(len(dd), size=(5000, len(dd)))].sum(1) / len(values)
        arms[arm] = {'scorable': len(values), 'statuses': dict(states),
                     'F1': np.mean([v['f1'] for v in values]).item() if values else None,
                     'same_parent_F1': np.mean([v['parent_f1'] for v in values]).item() if values else None,
                     'delta_parent': np.mean([v['delta_parent'] for v in values]).item() if values else None,
                     'paired95_descriptive': np.quantile(boot, [.025, .975]).tolist() if values else None,
                     'up': sum(v['delta_parent'] > 1e-12 for v in values),
                     'tie': sum(abs(v['delta_parent']) <= 1e-12 for v in values),
                     'down': sum(v['delta_parent'] < -1e-12 for v in values),
                     'by_dataset': {ds: {'n': len(vv := [v for v in values if v['dataset'] == ds]),
                                        'F1': np.mean([v['f1'] for v in vv]).item() if vv else None,
                                        'delta_parent': np.mean([v['delta_parent'] for v in vv]).item() if vv else None}
                                    for ds in scorers}}
    result = {'utc': utc(), 'parent300_F1': np.mean([v['f1'] for v in parents.values()]).item(),
              'arms': arms, 'budget': budget_snapshot(), 'sample_role': plan['sample_role']}
    locked_json(OUT / 'RESULTS_LATEST.json', result)
    (OUT / 'PAIRED_DETAILS.jsonl').write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in all_details), encoding='utf8')
    return result


def backend():
    """Private binding of the already audited transport; no predecessor edits."""
    path = PREV_E / 'experiment.py'
    spec = importlib.util.spec_from_file_location('experiments.query_decomposition.b0_strategy_round2_v1_20261020._round3_transport', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE, module.OUT, module.ARMS = HERE, OUT, strategies().ARMS
    module.final_prompt = lambda case, nodes, original, readout, arm: strategies().render_final(case['question'], nodes, original, readout, arm)
    module.log, module.budget_snapshot, module.summarize, module.chosen = log, budget_snapshot, summarize, chosen
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    if not args.run:
        print(json.dumps({'prepared': True, 'parent': plan['parent'], 'n_per_arm': 300, 'new_API_calls': 0}))
        return
    engine = backend()
    cases = rows(HERE / 'runtime300.frozen.jsonl')
    manifest = {r['id']: r for r in rows(HERE / 'SOURCES.frozen.jsonl')}
    for arm in plan['arms']:
        engine.run_phase(plan, cases, arm, 'full300', manifest)
    current = summarize(plan)
    verify(plan)
    for source in manifest.values():
        for field in ['base', 'control']:
            assert sha(ROOT / source[field]) == source[field + '_sha256']
    save(OUT / 'FINAL_RESULTS.frozen.json', current)
    eligible = [(current['parent300_F1'], 'parent_p02')]
    eligible += [(v['F1'], arm) for arm, v in current['arms'].items()
                 if v['scorable'] == 300 and v['F1'] > current['parent300_F1'] + 1e-12]
    score, selected = max(eligible, key=lambda x: x[0])
    save(OUT / 'SELECTION.frozen.json', {'selected': selected, 'F1': score,
                                       'scope': 'Exposed300 development only', 'prior_candidates_and_fresh90_unchanged': True})
    save(OUT / 'TERMINAL.json', {'utc': utc(), 'selected': selected, 'budget': budget_snapshot()})
    log('complete', selected=selected, F1=score, budget=budget_snapshot())
    print(json.dumps({'completed': True, 'selected': selected, 'F1': score, 'budget': budget_snapshot()}), flush=True)


if __name__ == '__main__':
    main()
