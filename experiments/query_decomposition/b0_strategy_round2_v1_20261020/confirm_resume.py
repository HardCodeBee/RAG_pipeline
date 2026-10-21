"""Resume stopped confirmation without changing any model or method input.

One explicit same-payload HTTP400 diagnostic is an operational exception to
the original automatic-retry list. Keep it, all failed attempts and liabilities
separate. Unresolved cases remain missing in the original150 scoring cohort.
"""
import argparse
import json
import os
from pathlib import Path
import re
from threading import Lock
from types import SimpleNamespace

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, digest, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
from . import confirm_data, fresh_confirm

OUT = fresh_confirm.OUT
RESUME = OUT / 'continuation_v001'
MU = 'musique:3hop2__607269_467331_162182'
HOT = 'hotpotqa:5ab3b2c4554299233954ff8a'
LOCK = Lock()


def terminal(record):
    return record is not None and (record['status'].startswith('success') or record['status'] == 'algorithm_failure')


def event(name, **fields):
    RESUME.mkdir(parents=True, exist_ok=True)
    row = {'utc': utc(), 'event': name, **fields}
    with LOCK:
        with (RESUME / 'EVENTS.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        locked_json(RESUME / 'STATUS.json', row)


def redact(value):
    text = str(value)
    for name in ['DASHSCOPE_API_KEY', 'OPENAI_API_KEY']:
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, '[redacted]')
    return re.sub(r'sk-[A-Za-z0-9_-]+', '[redacted]', text)[:1600]


def configure():
    checked = fresh_confirm.prerequisites()
    assert checked['selected'] == 'p02_role_headers'
    plan = fresh_confirm.read(OUT / 'PROTOCOL.frozen.json')
    fresh_confirm.verify(plan)
    assert fresh_confirm.read(OUT / 'CONTROLLER_STATUS.json')['state'] == 'stopped_permanent_missing'
    bound = fresh_confirm.private_driver()
    bound.verify = fresh_confirm.verify
    cases = bound.runtime_cases()
    assert len(cases) == 150
    incomplete = {c['id']: bound.selected(c, 'parent_g01') for c in cases if not terminal(bound.selected(c, 'parent_g01'))}
    assert set(incomplete) == {MU, HOT}
    assert incomplete[MU]['http_status'] == 400 and incomplete[MU]['retryable_infrastructure'] is False
    assert incomplete[HOT]['retryable_infrastructure'] is True
    assert all(bound.selected(c, 'p02_role_headers') is None for c in cases)
    assert not (OUT / 'rounds/r001').exists()
    first_payload = {}
    for cid, failed in incomplete.items():
        call = next(v for v in failed['logical_calls'] if v['request_id'] == failed['failed_request_id'])
        first_payload[cid] = call['payload_sha256']
    protection = {str(p.relative_to(ROOT)): fresh_confirm.sha(p) for p in (OUT/'rounds/r000/cases').glob('*/*/*.json')}
    assert len(protection) == 150
    return plan, bound, cases, first_payload, protection


def audit_proxy(client, cid, allowed_first, first_seen, arm, seen_key=None):
    class AuditedCompletions:
        def create(self, **request):
            hashed = digest(request)
            if arm == 'p02_role_headers':
                assert hashed == allowed_first, 'Candidate may only submit its frozen final payload'
            with LOCK:
                key = seen_key or cid
                if key not in first_seen:
                    assert hashed == allowed_first, 'First unpaid continuation payload differs from the frozen failed/final request'
                    first_seen.add(key)
            try:
                return client.chat.completions.create(**request)
            except Exception as error:
                body = getattr(error, 'body', None)
                if isinstance(body, dict):
                    body = body.get('error', body)
                safe_body = {k: redact(body[k]) for k in ['code', 'type', 'message', 'param'] if k in body} if isinstance(body, dict) else None
                event('provider_error', id=cid, arm=arm, payload_sha256=hashed,
                      error_type=type(error).__name__, http_status=getattr(error, 'status_code', None),
                      provider_code=redact(getattr(error, 'code', None)), message=redact(error), provider_error=safe_body)
                raise
    return SimpleNamespace(chat=SimpleNamespace(completions=AuditedCompletions()))


def execute():
    plan, bound, cases, expected, protection = configure()
    RESUME.mkdir(parents=True, exist_ok=True)
    with (RESUME/'CLAIM.json').open('x', encoding='utf8') as stream:
        json.dump({'pid': os.getpid(), 'utc': utc()}, stream)
    sidecar = {'utc': utc(), 'source_sha256': fresh_confirm.sha(Path(__file__)),
               'original_protocol_sha256': fresh_confirm.sha(OUT/'PROTOCOL.frozen.json'),
               'original_cohort': 150, 'candidate': 'p02_role_headers',
               'frozen_source_pins': plan['source_pins'], 'protected_original_cases': protection,
               'parent_recovery_round': 1, 'recovery_ids': [MU, HOT], 'first_payload_sha256': expected,
               'diagnostic_exception': 'Exactly one unchanged-payload retry of the previous Mu HTTP400, outside the original auto-retry list; retain original failure.',
               'missing_policy': 'Keep full150 scoring. If a parent remains missing, run candidate only on parent-terminal cases and expose paired subset and full-denominator bounds; do not replace missingness with algorithm failure.',
               'successful_calls': 'Exact complete payload cache only; never rerun success.',
               'model_prompt_retrieval_and_metric_changed': False}
    locked_json(RESUME/'PROTOCOL.frozen.json', sidecar)
    locked_json(OUT/'CURRENT_EXECUTION.json', {'controller': str(Path(__file__).relative_to(ROOT)), 'status': str((RESUME/'STATUS.json').relative_to(ROOT))})
    original_case, original_eligible, all_cases = bound.run_case, bound.eligible, bound.runtime_cases
    first_seen = set()

    def audited_case(case, arm, round_root, owner, retrieval_owner, retriever, search_service, client, execution_plan, cached, budget_class):
        if arm == 'parent_g01':
            allowed = expected[case['id']]
            seen_key = case['id']
        else:
            parent = bound.selected(case, 'parent_g01')
            assert terminal(parent)
            if parent['status'] == 'algorithm_failure':
                # The original executor must fail from its exact cached prefix;
                # it is not authorized to issue a new paid request for repair.
                allowed = 'no-new-paid-request-for-algorithm-failure'
            else:
                prompt, _ = bound.strategy().render_final(case['question'], parent['nodes'], parent['original_retrieval'], plan['readout'], arm)
                allowed = digest(payload(plan['model'], prompt, 64))
            seen_key = arm+':'+case['id']+':'+str(round_root.name)
        proxy = audit_proxy(client, case['id'], allowed, first_seen, arm, seen_key)
        return original_case(case, arm, round_root, owner, retrieval_owner, retriever, search_service, proxy, execution_plan, cached, budget_class)

    bound.run_case = audited_case
    bound.eligible = lambda case, arm: original_eligible(case, arm) or (arm == 'parent_g01' and case['id'] == MU)
    event('recover_two_parent_requests', original_budget=bound.budget_snapshot(), diagnostic_exception=True)
    try:
        bound.run(plan, ['parent_g01'], 1)
    finally:
        bound.eligible = original_eligible
    available = [c for c in cases if terminal(bound.selected(c, 'parent_g01'))]
    unresolved = [c['id'] for c in cases if c not in available]
    event('parent_recovery_finished', complete_parent=len(available), missing_parent=unresolved)
    # Scheduling filter only: runtime, protocol and references still contain150.
    bound.runtime_cases = lambda: available
    for round_id in [0, 1]:
        pending = [c for c in available if bound.eligible(c, 'p02_role_headers')]
        if not pending:
            break
        event('run_fixed_p02', round=round_id, planned=len(pending), parent_missing_retained=unresolved)
        bound.run(plan, ['p02_role_headers'], round_id)
    bound.runtime_cases = all_cases
    fresh_confirm.verify(plan)
    for name, expected_sha in protection.items():
        assert fresh_confirm.sha(ROOT/name) == expected_sha
    result = confirm_data.score()
    event('complete' if result['complete_150_pairs'] else 'partial_missing_retained',
          complete_150_pairs=result['complete_150_pairs'], parent=result['parent'], candidate=result['candidate_summary'], paired=result['paired'], budget=bound.budget_snapshot())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        _, _, cases, failed, protected = configure()
        print(json.dumps({'ready': True, 'cases': len(cases), 'retry_ids': list(failed), 'protected_cases': len(protected), 'new_api_calls': 0}))
        return
    try:
        result = execute()
    except Exception as error:
        event('stopped_for_inspection', error_type=type(error).__name__, reason=redact(error), successful_requests_preserved=True)
        raise
    print(json.dumps({'complete_150_pairs': result['complete_150_pairs'], 'paired': result['paired']}), flush=True)


if __name__ == '__main__':
    main()
