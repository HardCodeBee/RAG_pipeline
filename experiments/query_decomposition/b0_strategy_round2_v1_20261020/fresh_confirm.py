"""One frozen candidate versus g01 on sealed150; no tuning or scoring here.

Safe readiness check, without opening confirmation queries::
    python -m experiments.query_decomposition.b0_strategy_round2_v1_20261020.fresh_confirm --check

Only after old300 selection, complete positive dev90 gate, and pre-dispatch
candidate freeze all validate, --prepare may open runtime_confirm150. An
explicit --run --round 0 runs BOTH locked arms on ALL150 queries; --round 1
is the sole infrastructure recovery. There are no arm/limit/dataset overrides.

This module loads the unchanged fresh_runner.py into a private module binding.
Its generation/retrieval driver and execute_fresh bytecode are reused intact;
only cohort paths, cohort validator, and prerequisite verifier are bound here.
The imported dev90 module and every frozen source file remain unchanged.
"""
from collections import Counter
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name / 'fresh_confirm150'
RUNTIME = HERE / 'fresh_validation/runtime_confirm150.frozen.jsonl'
SEALED_QUERY_ONLY = HERE / 'fresh_validation/sealed150.query_only.frozen.jsonl'
COHORT_MANIFEST = HERE / 'fresh_validation/MANIFEST.frozen.json'
FRESH_SOURCE = HERE / 'fresh_runner.py'
OLD_OUT = OUTPUT / HERE.name
DEV_OUT = OLD_OUT / 'fresh_dev90'
OLD_SELECTION = OLD_OUT / 'SELECTION.frozen.json'
DEV_GATE = DEV_OUT / 'GATE.frozen.json'
CANDIDATE = DEV_OUT / 'CANDIDATE.frozen.json'


class GateClosed(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GateClosed('Invalid finite number: ' + name)
    return value


def validate_values(selection, results, gate, candidate, selection_sha256, candidate_sha256):
    """Pure, fail-closed gate validation; never reads confirmation queries."""
    selected = selection.get('selected')
    if selected == 'parent_g01':
        raise GateClosed('Old300 retained g01; no new candidate confirmation is authorized')
    if selected not in {'g02_evidence_goal', 'p02_role_headers', 's03_g01_plus_f01'}:
        raise GateClosed('Unknown selected candidate')
    arm = results.get('arms', {}).get(selected, {})
    if arm.get('scorable') != 300:
        raise GateClosed('Old300 selected candidate is not complete on300')
    parent_f1 = number(results.get('parent300_F1'), 'old300 parent F1')
    candidate_f1 = number(arm.get('F1'), 'old300 candidate F1')
    if candidate_f1 <= parent_f1:
        raise GateClosed('Old300 candidate did not strictly improve over parent')
    if not math.isclose(number(selection.get('F1'), 'selection F1'), candidate_f1, abs_tol=1e-12, rel_tol=0):
        raise GateClosed('Old300 selection/result score mismatch')
    if candidate.get('selected') != selected:
        raise GateClosed('Candidate freeze does not match old300 selection')
    if candidate.get('old300_selection_sha256') != selection_sha256:
        raise GateClosed('Old300 selection hash changed after candidate freeze')
    if candidate.get('frozen_before_dev_candidate_dispatch') is not True:
        raise GateClosed('Candidate was not declared frozen before dev dispatch')
    if not isinstance(candidate.get('source_pins'), dict) or not candidate['source_pins']:
        raise GateClosed('Missing frozen candidate source pins')
    if gate.get('passed') is not True or gate.get('not_for_tuning') is not True:
        raise GateClosed('Dev90 gate is closed or permits tuning')
    if gate.get('candidate') != selected:
        raise GateClosed('Dev90 gate names a different candidate')
    if type(gate.get('n_complete')) is not int or gate['n_complete'] != 90:
        raise GateClosed('Dev90 is not complete on all90 paired queries')
    if number(gate.get('delta_F1'), 'dev90 delta_F1') <= 0:
        raise GateClosed('Dev90 candidate did not strictly improve over parent')
    if gate.get('candidate_protocol_sha256') != candidate_sha256:
        raise GateClosed('Candidate freeze hash does not match dev90 gate')
    return selected


def relative(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise GateClosed('Prerequisite/source must be inside the workspace: ' + str(path))
    return str(path.relative_to(ROOT.resolve()))


def prerequisites(old_selection=OLD_SELECTION, dev_gate=DEV_GATE, candidate_protocol=CANDIDATE):
    """Verify identities and source pins before any sealed runtime is opened."""
    old_selection, dev_gate, candidate_protocol = map(Path, [old_selection, dev_gate, candidate_protocol])
    old_results = old_selection.parent / 'FINAL_RESULTS.frozen.json'
    dev_protocol = DEV_OUT / 'PROTOCOL.frozen.json'
    dev_contract = DEV_OUT / 'OFFLINE_CONTRACT.json'
    paths = [old_selection, old_results, dev_gate, candidate_protocol, dev_protocol, dev_contract]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise GateClosed('Awaiting frozen prerequisites: ' + ', '.join(missing))
    selection, results = read(old_selection), read(old_results)
    gate, candidate = read(dev_gate), read(candidate_protocol)
    selected = validate_values(selection, results, gate, candidate, sha(old_selection), sha(candidate_protocol))
    source_pins = candidate['source_pins']
    required_pins = [HERE / 'strategy.py', FRESH_SOURCE]
    resolved_pins = {}
    for name, expected in source_pins.items():
        path = (ROOT / name).resolve()
        relative(path)
        if not path.exists() or sha(path) != expected:
            raise GateClosed('Candidate source changed: ' + name)
        resolved_pins[path] = expected
    if any(p.resolve() not in resolved_pins for p in required_pins):
        raise GateClosed('Candidate must pin strategy.py and fresh_runner.py')
    base = read(dev_protocol)
    if base.get('model') != 'qwen-plus-2025-12-01' or base.get('temperature') != 0 or base.get('enable_thinking') is not False:
        raise GateClosed('Dev90 generation contract differs')
    if base.get('limits') != {'rewrite': 256, 'child_and_goal': 128, 'final': 64}:
        raise GateClosed('Dev90 generation limits differ')
    if base.get('global_cap_cny') != 300 or base.get('max_workers') != 3:
        raise GateClosed('Dev90 budget/concurrency contract differs')
    if selected not in base.get('arms_available', []):
        raise GateClosed('Selected candidate was not available in the frozen dev protocol')
    for name, expected in base['source_pins'].items():
        path = ROOT / name
        relative(path)
        if sha(path) != expected:
            raise GateClosed('Dev90 frozen source changed: ' + name)
    contract = read(dev_contract)
    if contract.get('passed') is not True or contract.get('runner_sha256') != sha(FRESH_SOURCE):
        raise GateClosed('Frozen dev90 offline contract no longer matches driver')
    if contract.get('strategy_sha256') != sha(HERE / 'strategy.py'):
        raise GateClosed('Frozen dev90 offline contract no longer matches strategy')
    return {'selected': selected, 'base': base,
            'paths': {'old_selection': old_selection, 'old_results': old_results,
                      'dev_gate': dev_gate, 'candidate_protocol': candidate_protocol,
                      'dev_protocol': dev_protocol, 'dev_contract': dev_contract},
            'hashes': {relative(p): sha(p) for p in paths}}


def private_driver():
    """Independent globals; no monkeypatch to the live dev90 module."""
    from experiments.query_decomposition.b0_strategy_round2_v1_20261020 import fresh_runner as original
    spec = importlib.util.spec_from_file_location('_b0_sealed150_private_driver', FRESH_SOURCE)
    bound = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bound)
    assert bound is not original and bound.__dict__ is not original.__dict__
    assert bound.execute_fresh.__code__.co_code == original.execute_fresh.__code__.co_code
    original_out, original_runtime = original.OUT, original.RUNTIME
    bound.OUT, bound.RUNTIME = OUT, RUNTIME
    bound.runtime_cases = lambda: confirmation_cases(bound.safe_case)
    assert original.OUT == original_out and original.RUNTIME == original_runtime
    return bound


def confirmation_cases(project):
    # This is called only after prerequisites() succeeded. No answers/qrels or
    # native supporting paragraphs are ever read by the generation runner.
    if not RUNTIME.exists():
        raise GateClosed('Awaiting R02 runtime_confirm150.frozen.jsonl')
    cases = [project(json.loads(line)) for line in RUNTIME.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(cases) != 150 or len({c['id'] for c in cases}) != 150:
        raise GateClosed('Confirmation runtime must contain exactly150 unique queries')
    if Counter(c['dataset'] for c in cases) != {'hotpotqa': 50, '2wiki': 50, 'musique': 50}:
        raise GateClosed('Confirmation runtime must contain50 queries per dataset')
    manifest = read(COHORT_MANIFEST)
    if sha(SEALED_QUERY_ONLY) != manifest['files'][SEALED_QUERY_ONLY.name]['sha256']:
        raise GateClosed('Metadata-selected sealed150 cohort changed')
    selected = {row['id']: row for row in (json.loads(line) for line in
                SEALED_QUERY_ONLY.read_text(encoding='utf-8').splitlines() if line.strip())}
    if {c['id'] for c in cases} != set(selected):
        raise GateClosed('Confirmation runtime differs from the metadata-selected sealed150 IDs')
    for c in cases:
        metadata = selected[c['id']]
        if c['question'] != metadata['query'] or c['dataset'] != metadata['dataset'] or c['qid'] != metadata['qid']:
            raise GateClosed('Confirmation original query changed while importing R02')
    return cases


def prepare(old_selection=OLD_SELECTION, dev_gate=DEV_GATE, candidate_protocol=CANDIDATE):
    from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
    checked = prerequisites(old_selection, dev_gate, candidate_protocol)
    bound = private_driver()
    cases = confirmation_cases(bound.safe_case)
    plan = deepcopy(checked['base'])
    plan.update(cohort='sealed confirmation150, 50 per dataset; no adaptive edits',
                runtime=relative(RUNTIME), runtime_sha256=sha(RUNTIME), cases=len(cases),
                arms_available=['parent_g01', checked['selected']], selected=checked['selected'],
                confirmation=True, not_for_tuning=True,
                selection_policy='Exactly the old300 winner that strictly improved on complete dev90; no reselection.',
                frozen_prerequisites=checked['hashes'],
                prerequisite_paths={k: relative(v) for k, v in checked['paths'].items()})
    plan['source_pins'].update(checked['hashes'])
    plan['source_pins'][relative(Path(__file__))] = sha(__file__)
    plan['source_pins'][relative(SEALED_QUERY_ONLY)] = sha(SEALED_QUERY_ONLY)
    plan['source_pins'][relative(COHORT_MANIFEST)] = sha(COHORT_MANIFEST)
    target = OUT / 'PROTOCOL.frozen.json'
    if target.exists():
        if read(target) != plan:
            raise GateClosed('Confirmation frozen protocol changed; no overwrite')
    else:
        locked_json(target, plan)
    contract = {'passed': True, 'runner_sha256': sha(FRESH_SOURCE),
                'strategy_sha256': sha(HERE / 'strategy.py'),
                'wrapper_sha256': sha(__file__),
                'frozen_dev90_contract_sha256': checked['hashes'][relative(checked['paths']['dev_contract'])],
                'same_execute_fresh_bytecode': True, 'private_cohort_binding': True,
                'arms': plan['arms_available'], 'confirmation_queries': len(cases),
                'no_reference_answers_read': True, 'new_api_calls': 0, 'new_retrieval_calls': 0}
    old_contract = OUT / 'OFFLINE_CONTRACT.json'
    if old_contract.exists() and read(old_contract) != contract:
        raise GateClosed('Confirmation offline identity changed; no overwrite')
    locked_json(old_contract, contract)
    return plan, bound


def verify(plan):
    p = plan['prerequisite_paths']
    checked = prerequisites(ROOT / p['old_selection'], ROOT / p['dev_gate'], ROOT / p['candidate_protocol'])
    if checked['selected'] != plan['selected'] or checked['hashes'] != plan['frozen_prerequisites']:
        raise GateClosed('Confirmation prerequisites changed')
    if sha(RUNTIME) != plan['runtime_sha256']:
        raise GateClosed('Confirmation R02 runtime changed')
    if plan['arms_available'] != ['parent_g01', checked['selected']]:
        raise GateClosed('Only the locked parent/candidate pair is permitted')
    for name, expected in plan['source_pins'].items():
        if sha(ROOT / name) != expected:
            raise GateClosed('Confirmation source changed: ' + name)


def check(old_selection=OLD_SELECTION, dev_gate=DEV_GATE, candidate_protocol=CANDIDATE):
    """Readiness only; importantly does not open RUNTIME even if gates pass."""
    try:
        value = prerequisites(old_selection, dev_gate, candidate_protocol)
        return {'ready': RUNTIME.exists(), 'gates_passed': True, 'selected': value['selected'],
                'confirmation_runtime_present': RUNTIME.exists(),
                'confirmation_queries_opened': False, 'new_api_calls': 0, 'new_remote_calls': 0}
    except (GateClosed, FileNotFoundError, KeyError, ValueError) as exc:
        return {'ready': False, 'gates_passed': False, 'reason': str(exc), 'confirmation_queries_opened': False,
                'new_api_calls': 0, 'new_remote_calls': 0}


def self_test():
    """Synthetic gate rejection tests; does not read sealed query contents."""
    selected = 'p02_role_headers'
    selection = {'selected': selected, 'F1': .71}
    results = {'parent300_F1': .70, 'arms': {selected: {'scorable': 300, 'F1': .71}}}
    candidate = {'selected': selected, 'old300_selection_sha256': 'selection-sha',
                 'source_pins': {'synthetic-source': 'sha'}, 'frozen_before_dev_candidate_dispatch': True}
    gate = {'passed': True, 'candidate': selected, 'n_complete': 90, 'delta_F1': .01,
            'candidate_protocol_sha256': 'candidate-sha', 'not_for_tuning': True}
    assert validate_values(selection, results, gate, candidate, 'selection-sha', 'candidate-sha') == selected
    rejected = []
    cases = [
        ('parent_selected', 'selection', 'selected', 'parent_g01'),
        ('incomplete_old300', 'arm', 'scorable', 299),
        ('no_old300_gain', 'arm', 'F1', .70),
        ('selection_score_mismatch', 'selection', 'F1', .72),
        ('candidate_changed', 'candidate', 'selected', 'g02_evidence_goal'),
        ('old_selection_hash_changed', 'candidate', 'old300_selection_sha256', 'other'),
        ('not_frozen_before_dev', 'candidate', 'frozen_before_dev_candidate_dispatch', False),
        ('dev_gate_closed', 'gate', 'passed', False),
        ('different_dev_candidate', 'gate', 'candidate', 'g02_evidence_goal'),
        ('incomplete_dev90', 'gate', 'n_complete', 89),
        ('zero_dev_gain', 'gate', 'delta_F1', 0),
        ('nonfinite_dev_gain', 'gate', 'delta_F1', float('nan')),
        ('candidate_hash_changed', 'gate', 'candidate_protocol_sha256', 'other'),
        ('permits_tuning', 'gate', 'not_for_tuning', False),
    ]
    for label, obj, key, val in cases:
        ss, rr, gg, cc = map(deepcopy, [selection, results, gate, candidate])
        table = {'selection': ss, 'arm': rr['arms'][selected], 'gate': gg, 'candidate': cc}
        table[obj][key] = val
        try:
            validate_values(ss, rr, gg, cc, 'selection-sha', 'candidate-sha')
        except GateClosed:
            rejected.append(label)
        else:
            raise AssertionError('Failed to reject synthetic case: ' + label)
    bound = private_driver()
    assert bound.OUT == OUT and bound.RUNTIME == RUNTIME
    return {'passed': True, 'positive_synthetic_gate': True, 'rejected': rejected,
            'frozen_driver_private_binding': True, 'confirmation_queries_opened': False,
            'new_api_calls': 0, 'new_remote_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--round', type=int, choices=[0, 1], default=0)
    parser.add_argument('--old-selection', type=Path, default=OLD_SELECTION)
    parser.add_argument('--dev-gate', type=Path, default=DEV_GATE)
    parser.add_argument('--candidate-protocol', type=Path, default=CANDIDATE)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.check:
        result = check(args.old_selection, args.dev_gate, args.candidate_protocol)
    else:
        plan, bound = prepare(args.old_selection, args.dev_gate, args.candidate_protocol)
        verify(plan)
        if args.prepare:
            result = {'prepared': True, 'arms': plan['arms_available'], 'cases': plan['cases'],
                      'not_for_tuning': True, 'new_api_calls': 0, 'new_remote_calls': 0}
        else:
            bound.verify = verify
            # Both frozen arms/all150 only; no caller-controlled arm, cohort
            # slicing, dataset filtering, prompt change, or tuning interface.
            result = bound.run(plan, plan['arms_available'], args.round)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
