"""Thin budget-bound adapter around the existing answer collection engine.

--prepare is local only and writes a proposal, never an authorized budget.
--freeze-authorized-budget must only be used after explicit user authorization.
--run requires that separate authorization record before any network operation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'work/router_research'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/2wiki_development_measurement_v1'
ANSWERS = OUT / 'answers_v1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
import collect_validation_answers as common
import pool_answer_collection as engine

PRICING = {
    'checked_on': '2026-09-18', 'currency': 'USD',
    'source': 'https://developers.openai.com/api/docs/pricing',
    'model_source': 'https://developers.openai.com/api/docs/models/gpt-4.1-mini',
    'model': 'gpt-4.1-mini-2025-04-14', 'service': 'standard_direct_API',
    'input_price_usd_per_million': .4, 'output_price_usd_per_million': 1.6,
    'cached_input_discount_assumed': False,
}
SCHEDULE = (6, 128, 512, 512, 512, 512)
PROPOSED_CEILING_USD = 2.0


def load_prepared():
    manifest_path = OUT / 'contexts_manifest.json'
    manifest = common.read_json(manifest_path)
    assert manifest['status'] == 'complete' and manifest['queries'] == 304
    assert manifest['query_actions'] == 608 and manifest['required_successful_generations'] == 1824
    freeze = OUT / 'actions_freeze.json'
    contract_path = OUT / 'execution_contract.json'
    context_path = OUT / manifest['contexts']['path']
    assert common.sha256(freeze) == manifest['actions_freeze_sha256']
    assert common.sha256(contract_path) == manifest['execution_contract_sha256']
    assert common.sha256(context_path) == manifest['contexts']['sha256']
    assert common.sha256(OUT / 'questions.jsonl') == manifest['questions_sha256']
    contract = common.read_json(contract_path)
    g = contract['generation']
    assert g['model'] == PRICING['model'] and g['temperature'] == 0
    assert g['max_output_tokens'] == 64 and g['repeats_per_query_action'] == 3
    assert g['max_retries'] == 0 and g['maximum_attempts_per_outcome'] == 6
    rows = [json.loads(line) for line in context_path.read_text(encoding='utf-8').splitlines()]
    expected_qids = {json.loads(line)['query_id'] for line in (OUT / 'questions.jsonl').read_text(encoding='utf-8').splitlines()}
    assert len(rows) == 608 and {(r['query_id'], r['action']) for r in rows} == {(q,a) for q in expected_qids for a in ('bm25','dense')}
    order = sorted(expected_qids, key=lambda q: hashlib.sha256(('2wiki_measurement_2026091843|'+q).encode()).digest())
    by_key = {(r['query_id'], r['action']): r for r in rows}
    rows = [by_key[q,a] for q in order for a in ('bm25','dense')]
    for row in rows:
        assert row['status'] == 'context_ready' and isinstance(row['reference_answers'], list)
        assert hashlib.sha256(row['prompt'].encode('utf-8')).hexdigest() == row['prompt_sha256']
        assert row['provider_input_tokens_reserved'] >= row['tiktoken_prompt_tokens'] > 0
    binding = {'actions_freeze_sha256': manifest['actions_freeze_sha256'],
               'execution_contract_sha256': manifest['execution_contract_sha256'],
               'contexts_manifest_sha256': common.sha256(manifest_path),
               'contexts_jsonl_sha256': manifest['contexts']['sha256'],
               'analysis_protocol_sha256': common.sha256(OUT / 'analysis_protocol.json'),
               'queries': 304, 'required_successful_generations': 1824,
               'worker_count': 4, 'attempt_limit_per_outcome': 6,
               'collector_sha256': common.sha256(Path(__file__)),
               'engine_sha256': common.sha256(Path(engine.__file__)),
               'scheduling': 'fixed_query_hash_then_repeat0_1_2_then_BM25_Dense; max4_inflight'}
    return rows, contract, binding


def prepare():
    rows, contract, binding = load_prepared()
    plan = {'status': 'prepared_local_only_not_authorized',
            'binding': binding, 'pricing': PRICING,
            'generation_attempt_schedule': list(SCHEDULE),
            'failure_stop': 'first6_zero_failures; next128_at_most2; later_batches_at_most10percent; any_engine_hard_stop',
            'readiness': 'reuse_m6_ready_clients_v2_four_clients_model_metadata_only_before_each_batch',
            'no_partial_policy_effects': True, 'new_answer_requests': 0,
            'network_requires': 'separate_budget.json_explicit_user_approval_bound_to_this_plan'}
    plan_path = OUT / 'collection_plan.json'
    if plan_path.exists():
        assert common.read_json(plan_path) == plan, 'Collection plan changed; inspect instead of overwriting'
    else:
        common.write_json(plan_path, plan)
    prompt_tokens = 3*sum(r['tiktoken_prompt_tokens'] for r in rows)
    reserved_tokens = 3*sum(r['provider_input_tokens_reserved'] for r in rows)
    out_tokens = 1824*64
    proposal = {'status': 'proposed_not_authorized', 'currency': 'USD', 'model': PRICING['model'],
                'pricing': PRICING, 'binding': binding, 'collection_plan_sha256': common.sha256(plan_path),
                'sample_queries': 304, 'successful_answers_required': 1824,
                'proposed_hard_budget_usd': PROPOSED_CEILING_USD,
                'prompt_tokens_all_three_repeats': prompt_tokens,
                'output_tokens_max_without_retry': out_tokens,
                'local_token_estimate_no_retry_max_output_usd': (prompt_tokens*.4+out_tokens*1.6)/1e6,
                'all_input_reservations_one_attempt_each_usd': (reserved_tokens*.4+out_tokens*1.6)/1e6,
                'maximum_attempts_per_outcome': 6, 'maximum_global_attempts': 1824*6,
                'budget_enforcement': 'existing_integer_micro_USD_reservation_before_every_attempt; unknown_usage_remains_reserved; stop_at_cap',
                'completion_not_guaranteed_by_ceiling': True,
                'estimate_note': 'Local token estimate excludes provider framing and uses max64 outputs; billing from actual provider usage. No cached discount assumed.'}
    common.write_json(OUT / 'budget_proposal.json', proposal)
    print(json.dumps(proposal, ensure_ascii=False), flush=True)
    return proposal


def freeze_budget(amount, approval_note):
    if not approval_note.strip():
        raise ValueError('Record the actual explicit user approval; autonomous continuation alone is insufficient')
    proposal = prepare()
    if not 0 < amount <= proposal['proposed_hard_budget_usd']:
        raise ValueError('Budget must fit the concrete reviewed proposal')
    path = OUT / 'budget.json'
    if path.exists():
        raise ValueError('An authorization already exists; do not replace a budget or reset its ledger')
    binding = proposal['binding']
    common.write_json(path, {'status': 'frozen', 'currency': 'USD', 'model': PRICING['model'],
                            **{k: binding[k] for k in ('actions_freeze_sha256','execution_contract_sha256','contexts_manifest_sha256')},
                            'input_price_usd_per_million': .4, 'output_price_usd_per_million': 1.6,
                            'hard_budget_usd': amount, 'maximum_global_attempts': 1824*6,
                            'maximum_attempts_per_outcome': 6, 'maximum_workers': 4,
                            'authorization_source': 'explicit_user_approval', 'approval_note': approval_note,
                            'authorized_at_utc': datetime.now(timezone.utc).isoformat(),
                            'collection_plan_sha256': proposal['collection_plan_sha256']})


def state():
    path = ANSWERS / 'attempts.sqlite3'
    if not path.exists():
        return {'attempts': 0, 'successes': 0, 'charged_or_reserved_micro_usd': 0}
    with sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True) as conn:
        attempts, charged = conn.execute('SELECT COUNT(*),COALESCE(SUM(charged_micro_usd),0) FROM attempts').fetchone()
        successes = conn.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]
    return {'attempts': attempts, 'successes': successes, 'charged_or_reserved_micro_usd': charged}


def stop_reason(index, before, after, reason):
    if reason == 'complete' and after['successes'] == 1824:
        return 'complete'
    if reason != 'invocation_attempt_limit':
        return 'collector_stopped_'+str(reason)
    attempts, successes = after['attempts']-before['attempts'], after['successes']-before['successes']
    if attempts != SCHEDULE[index] or not 0 <= successes <= attempts:
        return 'invalid_batch_accounting'
    failures = attempts-successes
    if (index == 0 and failures) or (index == 1 and failures > 2) or (index >= 2 and failures*10 > attempts):
        return 'predefined_failure_rate_stop'
    return None


def run():
    # All financial authorization checks precede network setup or readiness GETs.
    rows, contract, binding = load_prepared()
    plan_path = OUT / 'collection_plan.json'
    plan = common.read_json(plan_path)
    assert plan['binding'] == binding
    budget = common.load_budget(OUT / 'budget.json', binding, PRICING['model'])
    assert budget['authorization_source'] == 'explicit_user_approval' and budget['approval_note'].strip()
    assert budget['collection_plan_sha256'] == common.sha256(plan_path)
    assert budget['hard_budget_usd'] <= PROPOSED_CEILING_USD and budget['maximum_workers'] == 4
    if not os.environ.get('OPENAI_API_KEY','').strip() or os.environ.get('OPENAI_BASE_URL','').strip():
        raise ValueError('Existing credential and the standard official endpoint are required')
    import m6_ready_clients_v2 as readiness
    ANSWERS.mkdir(parents=True, exist_ok=True)
    control = ANSWERS / 'controller'
    control.mkdir(exist_ok=True)
    with common.exclusive_run(control):
        terminal = control / 'completion.json'
        if terminal.exists():
            print(terminal.read_text(encoding='utf-8'), flush=True)
            return
        reason = 'schedule_exhausted'
        for index, maximum in enumerate(SCHEDULE):
            receipt_path = control / f'batch_{index+1:02d}.json'
            if receipt_path.exists():
                receipt = common.read_json(receipt_path)
                if receipt['stop_reason'] is not None:
                    reason = receipt['stop_reason']; break
                continue
            before = state()
            ready_path = control / f'readiness_{index+1:02d}.json'
            clients, _proof = readiness.prepare_clients(engine, contract, ready_path, common.sha256(plan_path))
            if not clients:
                reason = 'readiness_failed_before_generation_reservation'
                common.write_json(receipt_path, {'before': before, 'after': state(), 'stop_reason': reason})
                break
            try:
                factory = readiness.make_provider_factory(engine, contract, clients, common.sha256(plan_path), common.sha256(ready_path))
                manifest = engine.collect(rows, contract, binding, budget, ANSWERS, factory,
                                          workers=4, max_new_attempts=maximum)
            finally:
                readiness.close_clients(clients)
            after = state()
            reason = stop_reason(index, before, after, manifest['stop_reason'])
            common.write_json(receipt_path, {'before': before, 'after': after, 'stop_reason': reason,
                                            'collector_stop_reason': manifest['stop_reason'], 'partial_policy_effects': False})
            if reason is not None:
                break
        common.write_json(terminal, {'status': 'complete' if reason == 'complete' else 'stopped',
                                     'stop_reason': reason, 'ledger': state(), 'partial_policy_effects': False})
        print(terminal.read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--freeze-authorized-budget', type=float)
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--approval-note', default='')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.freeze_authorized_budget is not None:
        freeze_budget(args.freeze_authorized_budget, args.approval_note)
    else:
        run()
