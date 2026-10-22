"""Read existing checkpoints and write a planning review; never dispatch work."""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

HERE = Path(__file__).resolve().parent
PLAN = HERE.parents[1]
ROOT = PLAN.parents[2]
OUTPUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
V2 = OUTPUT / 'comprehensive_v1_20261015/baseline_integrity_repair_v2_20261017'
ORIGINAL = V2.parent / 'campaign_api_v1'
DATASETS = ('hotpotqa', '2wiki', 'musique')
P02 = 'p02_multigrain_timing_cachefix_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def scorable(value):
    return isinstance(value.get('prediction'), str) and (
        value.get('status', '').startswith('success') or value.get('status') == 'algorithm_failure')


def predictions(path):
    if not path.exists():
        return {}
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        return {qid: json.loads(raw) for qid, raw in db.execute(
            "SELECT key,value FROM data WHERE kind='prediction'")}


def main():
    now = datetime.now(timezone.utc).isoformat()
    old = read(PLAN / 'reviews/20261021T051154Z/REUSE_AUDIT.json')
    refs = read(PLAN / 'execution_details/COST_REFERENCE.frozen.json')['groups']
    current_plan = read(PLAN / 'PLAN.json')
    inventory, selected_configs = [], {}
    for dataset in DATASETS:
        selected = read(ORIGINAL / dataset / 'BASELINE_CONFIGS_FROZEN.json')['selected']
        selected_configs[dataset] = selected
        for method, config in selected.items():
            track = 'main'
            if method == 'b05_iter_retgen_bge':
                track, config = 'matched50', dict(id='k10_t5', k=10, iterations=5)
            path = V2 / track / dataset / 'formal' / method / config['id'] / 'checkpoint.sqlite3'
            values = predictions(path)
            overlays = []
            if dataset == '2wiki':
                overlays = sorted((OUTPUT / 'twowiki_missing_recovery_20261021').glob(
                    f'rounds/r*/methods/{method}/*/checkpoint.sqlite3'))
                for overlay in overlays:
                    for qid, value in predictions(overlay).items():
                        if not scorable(values.get(qid, {})):
                            values[qid] = value
            frozen_questions = {
                r['qid']: r['question'] for r in (
                    json.loads(line) for line in (ROOT /
                        'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen' /
                        dataset / 'formal.questions.jsonl').read_text(encoding='utf8').splitlines() if line.strip())}
            assert len(frozen_questions) == 500
            assert set(values) <= set(frozen_questions)
            assert all(v.get('question', frozen_questions[q]) == frozen_questions[q] for q, v in values.items())
            inventory.append(dict(dataset=dataset, method=method, config=config,
                processed=len(values), scorable=sum(map(scorable, values.values())),
                statuses=dict(Counter(v['status'] for v in values.values())),
                source=str(path.relative_to(ROOT)), recovery_sources=[str(p.relative_to(ROOT)) for p in overlays]))

    selected_p02 = {}
    for path in sorted((OUTPUT / 'p02_formal1500_v1_20261021').glob('rounds/r*/cases/*/*/*.json')):
        value = read(path)
        if not scorable(selected_p02.get(value['id'], {})):
            selected_p02[value['id']] = {k: value[k] for k in ('id', 'qid', 'dataset', 'status', 'prediction')}
    p02_by_dataset = {ds: dict(Counter(v['status'] for v in selected_p02.values()
        if v['dataset'] == ds)) for ds in DATASETS}
    p02_complete = {ds: sum(v['dataset'] == ds and scorable(v) for v in selected_p02.values()) for ds in DATASETS}
    counts = {(r['dataset'], r['method']): r['scorable'] for r in inventory}
    counts.update({(ds, P02): n for ds, n in p02_complete.items()})

    ledger = OUTPUT / 'budget.sqlite3'
    with closing(sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        known, reserve = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()

    changed = {'b02_selfask_dense', 'b03_react_dense', 'b04_ircot_bge', 'b05_iter_retgen_bge', 'b06_decomp_bge'}
    projections = []
    for r in refs:
        ds, method, mu = r['dataset'], r['method'], r['mean_cny']
        n = counts.get((ds, method), 0)
        affected = method in changed or (method == 'b08_qd_rerank' and ds != 'hotpotqa')
        # Budget assumption only: do not credit affected-cell prefix reuse yet.
        equivalent = 500 if affected else 500 - n
        factor = 1.0
        if method == 'b02_selfask_dense':
            factor = 5 / selected_configs[ds][method]['max_search']
        elif method == 'b04_ircot_bge':
            factor = 10 / selected_configs[ds][method]['max_reason_sentences']
        elif method == 'b05_iter_retgen_bge':
            factor = {'hotpotqa': 4, '2wiki': 6, 'musique': 7}[ds] / 5
        projections.append(dict(dataset=ds, method=method, old_complete=n,
            historical_mean_CNY=mu, historical_basis=r['basis'], historical_n=r['n'],
            prospective_config_changed=affected, full_flow_equivalents_for_budget=equivalent,
            quality_proxy_CNY=equivalent * mu, timing50_proxy_CNY=50 * mu,
            round_count_sensitivity_factor=factor,
            quality_sensitivity_CNY=equivalent * mu * factor,
            timing50_sensitivity_CNY=50 * mu * factor))

    quality = sum(r['quality_proxy_CNY'] for r in projections)
    timing = sum(r['timing50_proxy_CNY'] for r in projections)
    quality_sensitivity = sum(r['quality_sensitivity_CNY'] for r in projections)
    timing_sensitivity = sum(r['timing50_sensitivity_CNY'] for r in projections)
    policy_proposals = {
        'b00_rag_top10_v1': dict(k=10, source='existing local reference control'),
        'b01_rag_top10_cot3': dict(k=10, demonstrations=3, source='existing local reference control'),
        'b02_selfask_dense': dict(k=5, max_search=5, source='Iter-RetGen section4.4, follow-up baseline setting'),
        'b03_react_dense': dict(k=5, max_search=5, source='Iter-RetGen section4.4, Dense search adaptation; not original Search/Lookup environment'),
        'b04_ircot_bge': dict(k=6, max_reason_sentences=10, max_document_pool=15, distractors_M=2, reader='cot',
            source='author base_configs + default controller; not paper-selected best', approval='pending explicit selection'),
        'b05_iter_retgen_bge': dict(k=5, iterations=dict(hotpotqa=4, **{'2wiki':6}, musique=7),
            source='highest published per-dataset F1 in original Table2, consistent with existing approved policy; not a universal iteration standard'),
        'b06_decomp_bge': dict(k=2, max_steps=8, reader='cot',
            source='author ODQA base configuration, outer and inner controller max_steps8; not paper-selected best', approval='pending explicit selection'),
        'b07_hybrid_rerank': dict(per_retriever_k=dict(hotpotqa=20, **{'2wiki':50}, musique=20), rrf_constant=60, final_k=10,
            source='existing local extra-resource reference control; no unique official best claimed'),
        'b08_qd_rerank': dict(k=10, max_generated_subqueries=5, retain_original=True, final_k=10,
            source='original paper sections4.5/5.3 and Algorithm1; applying to 2Wiki/MuSiQue is a dataset adaptation'),
        P02: dict(source='frozen original P02; no prompt-research candidate promotion', execution_variant='p02_shared_baseline_v2')}
    budget = dict(currency='CNY', public_price_source='https://help.aliyun.com/zh/model-studio/qwen-plus',
        shared_ledger_known_CNY=known, shared_ledger_unknown_reserved_CNY=reserve,
        shared_ledger_occupied_CNY=known + reserve, actual_provider_bill_verified=False,
        main_campaign_known_at_pause_CNY=old['budget']['known_cny'],
        main_campaign_unknown_at_pause_CNY=old['budget']['unknown_reserved_cny'],
        quality_historical_mean_proxy_CNY=quality, timing50_historical_mean_proxy_CNY=timing,
        total_historical_mean_proxy_CNY=quality + timing,
        quality_round_count_sensitivity_CNY=quality_sensitivity,
        timing50_round_count_sensitivity_CNY=timing_sensitivity,
        total_round_count_sensitivity_CNY=quality_sensitivity + timing_sensitivity,
        fifty_percent_spend_sensitivity_CNY=[1.5 * (quality + timing), 1.5 * (quality_sensitivity + timing_sensitivity)],
        fixed_cumulative_or_stage_budget_cap=None, new_spending_this_review_CNY=0,
        limitations=[
            'Conditional on the proposed configuration map; not approved or frozen runtime parameters.',
            'Changed method/dataset cells are costed as full fresh-flow equivalents without crediting same-payload prefix reuse; actual execution must still obey strict C1 reuse.',
            'Historical means describe old configurations; new retrieval k and adaptive behavior can change prompt lengths and actual stages.',
            'Round-count sensitivity scales full-flow means by new/old step limits, not measured mean stage counts; it is neither a statistical interval nor an upper bound.',
            'Missing MuSiQue same-method observations use labelled cross-dataset proxies.',
            'No 32K-token ceiling charge is treated as actual output usage; output is charged by provider-returned usage.',
            'No GPU/cluster CNY rate is available; report requested resources and actual allocation time separately rather than inventing cash cost.',
            'Overlapping quality and timing execution is not charged twice; separate timing repeats still have real new usage.',
            'Unknown liabilities are retained; they are not confirmed costs or permission to replay.',
            'Prior 37.15-CNY estimate applies to old revision005 configurations, not the current parameter proposal.'
        ])
    dataset_progress = [{
        'dataset': ds, 'planned': 5000,
        'old_baseline_scorable': sum(r['scorable'] for r in inventory if r['dataset'] == ds),
        'old_p02_scorable': p02_complete[ds],
        'new_protocol_completed': 'not migrated / not measured'
    } for ds in DATASETS]
    value = dict(utc=now, status='paused_by_user_preexecution_review',
        latest_authority='同时给出一下预算 先确认一下评估计划是否完整 先不要执行',
        planned_quality_method_queries=15000, planned_timing_flows=1500,
        existing_quality_counts=dataset_progress, baseline_inventory=inventory,
        p02_by_dataset=p02_by_dataset,
        old_primary_scorable=sum(counts.values()),
        old_counts_are_not_new_configuration_completion=True,
        proposed_configuration_map=policy_proposals, runtime_changes_adopted=False,
        completeness=dict(design='mostly specified; parameter choices and executable integration incomplete', ready_to_dispatch=False,
            pending=[
                'Confirm exact IRCoT and DecomP source mapping; defaults are not best-F1 configurations.',
                'Freeze the complete ten-method dataset/config manifest and unchanged common Qwen/BGE generation/resource identities.',
                'Recalculate strict same-stage whole-case/prefix reuse after config freeze.',
                'Integrate and validate no-cap ledger, persistent five-attempt policy, unknown holds and cost alerts in the actual new runner.',
                'Capture first nonempty content TTFT and local/GPU synchronized component time; old first-chunk fields are ineligible.',
                'Restore authorized remote SSH identity and verify resource/precision/actual GPU allocation before remote timing.',
                'Separate quality and timing outcomes; preserve first valid result, failures, and fixed500 denominators.',
                'Predefine paired bootstrap confidence intervals on complete same-query cohorts; point estimates do not by themselves establish a gain.'
            ]),
        preflight=dict(local_evaluation_processes_found=0, source_v2_verification_passed=True,
            frozen_provider_non_generation_GET_status=401, authenticated_provider_generation_tested=False,
            remote_passwordless_SSH='authentication_failed', new_API_generation_requests=0, new_GPU_jobs=0,
            stop_files_cleared=0), budget=budget, cost_projection_rows=projections,
        explicitly_deferred=['same_final_prompt_control','mechanism_ablations','equal_budget_extension',
            'new_models_retrievers_extension','S02_rerun','prompt_candidate_promotion'])
    assert len(inventory) == 27 and len(refs) == 30
    assert value['old_primary_scorable'] == 11314, value['old_primary_scorable']
    assert set(current_plan['baselines']['methods']) | {P02} == set(policy_proposals)
    write(HERE / 'REVIEW.json', value)
    print(json.dumps(dict(status=value['status'], counts=dataset_progress,
        scorable=value['old_primary_scorable'], budget=budget), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
