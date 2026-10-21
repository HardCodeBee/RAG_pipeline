"""Run the fixed150 confirmation once, with one infrastructure recovery per arm.

This controller changes execution staging only. Candidate, prompts, complete
cohort and generation/retrieval contracts are checked by fresh_confirm.py.
"""
import argparse
import json
import os
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
from . import confirm_data, fresh_confirm

OUT = fresh_confirm.OUT


def log(event, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'CONTROLLER_EVENTS.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps({'utc': utc(), 'event': event, **fields}, ensure_ascii=False) + '\n')
    locked_json(OUT / 'CONTROLLER_STATUS.json', {'utc': utc(), 'state': event, **fields})


def terminal(value):
    return value is not None and (value['status'].startswith('success') or value['status'] == 'algorithm_failure')


def execute():
    # Gate/candidate hashes are checked before opening any confirmation input.
    checked = fresh_confirm.prerequisites()
    assert checked['selected'] == 'p02_role_headers'
    confirm_data.ingest()
    confirm_data.references()
    plan, bound = fresh_confirm.prepare()
    fresh_confirm.verify(plan)
    bound.verify = fresh_confirm.verify
    cases = bound.runtime_cases()
    assert len(cases) == 150 and plan['arms_available'] == ['parent_g01', 'p02_role_headers']
    dispatch = OUT / 'CONTROLLER_DISPATCH.frozen.json'
    if dispatch.exists():
        raise RuntimeError('Confirmation controller was already dispatched; inspect existing work, do not reopen it')
    # Exclusive creation prevents concurrent controllers from submitting the
    # same cohort. Keep the claim after completion or interruption for review.
    claim = OUT / 'CONTROLLER.claim.json'
    with claim.open('x', encoding='utf8') as stream:
        json.dump({'utc': utc(), 'pid': os.getpid(),
                   'controller_sha256': fresh_confirm.sha(Path(__file__))}, stream)
    locked_json(dispatch, {'utc': utc(), 'cases': 150, 'arms': plan['arms_available'],
                           'controller_sha256': fresh_confirm.sha(Path(__file__)),
                           'execution_protocol_sha256': fresh_confirm.sha(OUT / 'PROTOCOL.frozen.json'),
                           'execution_staging': 'Complete parent150, then fixed p02; one immutable infrastructure-only recovery per arm',
                           'no_tuning': True, 'budget_before': bound.budget_snapshot()})
    for arm in plan['arms_available']:
        for round_id in [0, 1]:
            pending = [case for case in cases if not terminal(bound.selected(case, arm))]
            if not pending:
                break
            if round_id == 1 and not all((value := bound.selected(case, arm)) is not None and value.get('retryable_infrastructure') for case in pending):
                log('stopped_permanent_missing', arm=arm, missing=len(pending), candidate_unchanged=True)
                return confirm_data.score()
            log('dispatch_fixed_arm', arm=arm, round=round_id, pending=len(pending), budget=bound.budget_snapshot())
            bound.run(plan, [arm], round_id)
        missing = [case['id'] for case in cases if not terminal(bound.selected(case, arm))]
        if missing:
            log('stopped_infrastructure_missing', arm=arm, missing_ids=missing, candidate_unchanged=True)
            return confirm_data.score()
    result = confirm_data.score()
    assert result['complete_150_pairs']
    fresh_confirm.verify(plan)
    log('complete_confirmation', candidate=checked['selected'], complete_pairs=150,
        parent=result['parent'], candidate_summary=result['candidate_summary'], paired=result['paired'],
        budget=bound.budget_snapshot(), no_tuning=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(fresh_confirm.check(), ensure_ascii=False))
        return
    try:
        result = execute()
    except Exception as error:
        log('stopped_for_inspection', error_type=type(error).__name__, reason=str(error)[:500],
            candidate_unchanged=True, successful_requests_preserved=True)
        raise
    print(json.dumps({'complete': result['complete_150_pairs'], 'paired': result['paired']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
