"""Cache-only execution to the first genuinely unavailable RAG input/output.

No API client or index backend is opened. A missing payload stops execution;
its answer is never substituted from a different historical prompt. A missing
retrieval expression is logged without loading FAISS or an encoder.
"""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
import time

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, Store, digest, utc
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt
from experiments.query_decomposition.component_chain_opt_v2_20261019.policy import IntermediateLength
from experiments.query_decomposition.p02_flow_simplify_v1_20261020 import pipeline, score
from . import cache
from .adapter import Adapter, RetrievalMiss


HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
ARMS = ('timing_only', 'union20')


class PayloadMiss(RuntimeError):
    def __init__(self, stage, request):
        self.stage, self.payload = stage, request
        self.payload_sha256 = digest(request)
        super().__init__('No exact successful full-payload receipt; offline execution stops before any answer is generated')


def _remaining(case, first_stage):
    """Structural suffix bounds; actual future cache availability is unknown."""
    goal_sid = len(case['supporting_questions']) + 1
    profile = []
    for sid in range(1, goal_sid + 1):
        if sid > 1:
            profile.append((f'legacy/rewrite/{sid}', 256, False))
        profile.append((f'legacy/answer/{sid}', 128, False))
        profile.append((f'legacy/sentence_format/{sid}', 128, True))
    profile.append(('legacy/final', 64, False))
    index = next(i for i, row in enumerate(profile) if row[0] == first_stage)
    suffix = profile[index:]
    return {'remaining_logical_calls_min': sum(not optional or stage == first_stage for stage, _, optional in suffix),
            'remaining_logical_calls_max': len(suffix),
            'remaining_output_token_allowances_max': sum(maximum for _, maximum, _ in suffix),
            'meaning': 'Counts remaining stage opportunities including the first miss, not a prediction that every one needs API; later exact receipts may still exist'}


def execute_cached(case, arm, adapter, catalog, readout, model):
    calls, retrieved, nodes = [], [], []
    first_miss = None
    started = time.perf_counter()

    def get(query):
        packet, hit, elapsed = adapter.get(case['qid'], query, 10)
        retrieved.append({'qid': case['qid'], 'query': query, 'cache_hit': hit,
                          'doc_ids': [doc['doc_id'] for doc in packet['docs']], 'seconds': elapsed})
        return packet

    def ask(stage, prompt, maximum, allow_final_truncation=False):
        request = payload(model, prompt, maximum)
        key = digest(request)
        value = catalog.get(key)
        if value is None:
            raise PayloadMiss(stage, request)
        if value['payload'] != request or value['payload_sha256'] != key:
            raise AssertionError('Cache entry does not match the complete requested payload')
        calls.append({'qid': case['qid'], 'stage': stage, 'payload_sha256': key, 'request_id': value['request_id'],
                      'source': value['source'], 'cache_tier': value['cache_tier'],
                      'cache_origin': value['cache_tier'], 'status': value['status'],
                      'full_payload_match_verified': True,
                      'finish_reason': value['result'].get('finish_reason'),
                      'usage': value['result'].get('usage', {}), 'reused': True})
        return use_receipt(value, stage, allow_final_truncation)

    result = None
    try:
        result = pipeline.execute(case, get, ask, arm=arm, readout=readout, record_node=nodes.append)
        status = 'complete_from_exact_cache'
    except PayloadMiss as exc:
        status = 'needs_api'
        first_miss = {'kind': 'payload', 'stage': exc.stage, 'payload_sha256': exc.payload_sha256,
                      'payload': exc.payload, **_remaining(case, exc.stage),
                      'first_request_reservation_cny': (len(exc.payload['messages'][0]['content'].encode('utf8')) + 256) * .8e-6
                                                       + exc.payload['max_tokens'] * 2e-6}
    except RetrievalMiss as exc:
        status = 'needs_full_retrieval'
        sid = len(nodes) + 1
        previous_stage = calls[-1]['stage'] if calls else None
        next_stage = f'legacy/answer/{sid}' if sid == 1 or previous_stage == f'legacy/rewrite/{sid}' else f'legacy/rewrite/{sid}'
        first_miss = {'kind': 'retrieval', 'query': exc.query, 'qid': exc.qid, 'k': exc.k,
                      'resource_sha256': exc.resource_sha256, 'next_model_stage_if_retrieval_succeeds': next_stage,
                      **_remaining(case, next_stage),
                      'api_need_not_yet_proven': True}
    except IntermediateLength as exc:
        status = 'cached_algorithm_failure'
        result = {**case, 'arm': arm, 'status': 'algorithm_failure', 'prediction': '',
                  'nodes': deepcopy(nodes), 'failed_stage': exc.stage,
                  'failed_request_id': exc.request_id,
                  'unpropagated_truncated_answer': exc.answer,
                  'retryable_infrastructure': False, 'api_attempted': False}
    except Exception as exc:
        status = 'cached_prefix_failure'
        first_miss = {'kind': 'cached_or_contract_failure', 'error_type': type(exc).__name__, 'reason': str(exc)}
    row = {'id': case['id'], 'qid': case['qid'], 'dataset': case['dataset'], 'arm': arm,
           'status': status, 'cached_calls': calls, 'cached_retrievals': retrieved,
           'completed_nodes': len(nodes), 'first_miss': first_miss,
           'prediction': result['prediction'] if result else None,
           'new_api_calls': 0, 'new_full_index_searches': 0}
    if result is None:
        if status == 'cached_prefix_failure':
            raise RuntimeError('Cached execution contract failed; do not dispatch or replace outputs: ' + json.dumps(row))
        result = {**case, 'arm': arm, 'status': 'incomplete', 'prediction': None,
                  'nodes': deepcopy(nodes), 'history_all': [{key: node[key] for key in ('step_id', 'request', 'answer')} for node in nodes],
                  'retryable_infrastructure': True, 'pending_new_input': True,
                  'error_type': 'CacheMiss', 'reason': 'Offline cache boundary, not an API failure; resume the exact pipeline in a new immutable round',
                  'first_cache_miss': first_miss, 'api_attempted': False}
    result.update(logical_calls=deepcopy(calls), logical_retrievals=deepcopy(retrieved),
                  finished_utc=utc(), actual_seconds=time.perf_counter() - started,
                  execution_mode='exact_full_payload_cache_recompute',
                  materialized_by='preflight.py', new_api_calls=0, new_retrieval_searches=0)
    return row, result


def run(output=None, *, materialize=False):
    from . import run as execution
    plan = execution.prepare()
    cases = execution.rows(HERE / 'runtime300.frozen.jsonl')
    assert len(cases) == len({case['id'] for case in cases}) == 300
    catalog = cache.catalog('hotpotqa')  # Globally mandatory canonical tier.
    details, datasets = [], {}
    complete_dir = OUT / 'preflight_complete'
    for dataset in ('2wiki', 'musique', 'hotpotqa'):
        owner = Store(OUT / f'retrieval/{dataset}', {'task': HERE.name, 'resources': plan['resources'][dataset]})
        adapter = None
        try:
            adapter = Adapter(owner, plan['resources'][dataset], offline_only=True)
            tested = []
            for case in cases:
                if case['dataset'] != dataset:
                    continue
                for arm in ARMS:
                    row, result = execute_cached(pipeline.safe_case(case), arm, adapter,
                                                 catalog, plan['readout'], plan['model'])
                    details.append(row)
                    tested.append(row)
                    if row['status'] == 'complete_from_exact_cache':
                        # Separate diagnostics, never injected as whole-case
                        # results into the real rounds/scorer namespace.
                        target = complete_dir / dataset / arm / (case['qid'] + '.json')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        execution.save(target, result)
                    if materialize:
                        target = OUT / f'rounds/r000/cases/{dataset}/{arm}/{case["qid"]}.json'
                        if target.exists():
                            raise RuntimeError('Immutable cached recomputation already exists: ' + str(target))
                        execution.save(target, result)
            datasets[dataset] = {
                'method_cases': len(tested), 'statuses': dict(Counter(row['status'] for row in tested)),
                'retrieval_cache_entries': owner.db.execute("SELECT count(*) FROM data WHERE kind='retrieval'").fetchone()[0],
                'seed_summary': deepcopy(adapter.seed_summary),
                'backend_started': adapter.retriever is not None or adapter.service is not None,
            }
            assert not datasets[dataset]['backend_started']
        finally:
            if adapter is not None:
                adapter.close()
            owner.export()
            owner.db.close()

    arms = {}
    for arm in ARMS:
        values = [row for row in details if row['arm'] == arm]
        misses = [row for row in values if row['status'] == 'needs_api']
        retrieval_misses = [row for row in values if row['status'] == 'needs_full_retrieval']
        incomplete = misses + retrieval_misses
        proof_min = len({row['first_miss']['payload_sha256'] for row in misses})
        bounds = sum(row['first_miss']['remaining_logical_calls_max'] for row in incomplete)
        arms[arm] = {
            'planned': len(values), 'statuses': dict(Counter(row['status'] for row in values)),
            'needs_api_queries': len(misses), 'needs_api_ids': [row['id'] for row in misses],
            'first_miss_stage_counts': dict(Counter(row['first_miss']['stage'] for row in misses)),
            'needs_full_retrieval_queries': len(retrieval_misses),
            'proven_unique_new_api_calls_min': proof_min,
            'remaining_api_calls_structural_max': bounds,
            'bounds_note': 'The lower bound counts unique first-miss payloads only. The maximum assumes all remaining stages miss cache and every future short answer needs formatting. Actual later outputs/cache hits can reduce it.',
            'by_dataset': {dataset: dict(Counter(row['status'] for row in values if row['dataset'] == dataset))
                           for dataset in ('hotpotqa', '2wiki', 'musique')},
        }
    baseline_exact = []
    for row in details:
        if row['arm'] != 'timing_only' or row['status'] != 'complete_from_exact_cache':
            continue
        case = next(case for case in cases if case['id'] == row['id'])
        _, original = score.select_case(execution.OLD, case['dataset'], 'p02_role_headers', case['qid'])
        assert original is not None
        baseline_exact.append(row['prediction'] == original['prediction'])
    report = {
        'utc': utc(), 'task': HERE.name, 'planned_queries': 300, 'planned_method_cases': 600,
        'arms': arms, 'dataset_retrieval_seed': datasets,
        'canonical_timing_predictions_exact': all(baseline_exact) and len(baseline_exact) == 300,
        'baseline_compared': len(baseline_exact),
        'source_sha256': execution.sha(Path(__file__)), 'adapter_sha256': execution.sha(HERE / 'adapter.py'),
        'cache_sha256': execution.sha(HERE / 'cache.py'), 'runtime_sha256': execution.sha(HERE / 'runtime300.frozen.jsonl'),
        'no_upstream_answer_substitution_after_miss': True,
        'materialized_method_cases': len(details) if materialize else 0,
        'materialized_root': str((OUT / 'rounds/r000/cases').relative_to(ROOT)) if materialize else None,
        'no_new_api_calls': True, 'no_full_index_searches': True, 'no_encoder_or_faiss_loaded': True,
        'reference_answers_read': False, 'budget_before_any_dispatch': execution.budget_snapshot(),
        'details_file': str((OUT / 'PREFLIGHT.details.jsonl').relative_to(ROOT)),
    }
    execution.save(output or OUT / 'PREFLIGHT.json', report)
    (OUT / 'PREFLIGHT.details.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in details), encoding='utf8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--materialize', action='store_true', help='Write exact pipeline recomputations and explicit CacheMiss boundaries to immutable r000 cases')
    args = parser.parse_args()
    result = run(args.output, materialize=args.materialize)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
