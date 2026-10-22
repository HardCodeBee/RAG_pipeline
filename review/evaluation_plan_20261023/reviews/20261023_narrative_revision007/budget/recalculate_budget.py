"""Offline planning arithmetic only; source JSON/SQLite are read-only.

Run with Python from any directory. Writes only sibling BUDGET.json,
QA_RAW_PROXY_ROWS.jsonl and BUDGET.md. No model, network or GPU imports.
"""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = ROOT / 'experiments/query_decomposition/unified_rag_eval_v1_20261021'
OUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
DATASETS = ('hotpotqa', '2wiki', 'musique')
P02 = 'p02_multigrain_timing_cachefix_v1'
METHODS = (P02, 'b00_rag_top10_v1', 'b01_rag_top10_cot3',
           'b08_qd_rerank', 'b02_selfask_dense', 'b06_decomp_bge')
INPUT_RATE, OUTPUT_RATE = 0.8 / 1_000_000, 2 / 1_000_000
OUTPUT_ALLOWANCE = 32768
PROMPT_SOURCE = HERE.parent / 'qa_raw/PROMPT.txt'


def escape_field(value):
    # Cost-only concrete escaping; no execution renderer is implemented here.
    return str(value).replace('\\', '\\\\').replace('\r', '\\r').replace(
        '\n', '\\n').replace('|', '\\|').replace(']', '\\]')


def passage(label, doc):
    return f'[Passage {label} | doc_id={escape_field(doc["doc_id"])} | source={escape_field(doc["source"])}]\n{doc["text"]}'


def neutral_final_proxy(case, auxiliary):
    return PROMPT_SOURCE.read_text(encoding='utf-8-sig').format(
        question=case['question'], root_passages='\n\n'.join(
            passage(f'R{i}', d) for i, d in enumerate(case['original_retrieval']['docs'], 1)),
        auxiliary_material=auxiliary)


def qa_helper_proxy(case):
    return '\n\n'.join(f'Step {n["step_id"]} ({n["role"]})\nQuestion: {n["request"]}\nAnswer: {n["answer"]}'
                        for n in case['nodes'])


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rel(path):
    return path.relative_to(ROOT).as_posix()


def scorable(v):
    return isinstance(v.get('prediction'), str) and (
        v.get('status', '').startswith('success') or v.get('status') == 'algorithm_failure')


def predictions(path):
    if not path.exists():
        return {}
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        return {k: json.loads(v) for k, v in db.execute(
            "SELECT key,value FROM data WHERE kind='prediction'")}


def checkpoint_mean_check(path, reference):
    if not path.exists():
        return {'source_exists': False, 'mean_recomputed': False,
                'basis': reference['basis'], 'reference_n': reference['n']}
    originals = predictions(path)
    complete = {q for q, v in originals.items() if scorable(v)}
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        calls = [json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='token_call'")]
    used = [c for c in calls if c.get('qid') in complete]
    u = usage_cost(used)
    actual = u['known_cny'] / len(complete)
    assert len(complete) == reference['n'], (path, len(complete), reference['n'])
    assert abs(actual - reference['mean_cny']) < 1e-12, (path, actual, reference['mean_cny'])
    return {'source_exists': True, 'mean_recomputed': True, 'n': len(complete),
            'mean_cny': actual, 'usage_missing_in_selected_token_call_rows': u['unpriced_calls'],
            'original_database_excludes_later_recovery_overlays_for_this_mean': True}


def usage_cost(calls):
    known = [c for c in calls if isinstance(c.get('usage', {}).get('prompt_tokens'), int)
             and isinstance(c.get('usage', {}).get('completion_tokens'), int)]
    return dict(input_tokens=sum(c['usage']['prompt_tokens'] for c in known),
                output_tokens=sum(c['usage']['completion_tokens'] for c in known),
                unpriced_calls=len(calls) - len(known),
                known_cny=sum(c['usage']['prompt_tokens'] * INPUT_RATE +
                              c['usage']['completion_tokens'] * OUTPUT_RATE for c in known))


def raw_helper_proxy(case):
    """Cost proxy, not an adopted final prompt. Preserve question/source mapping.

    All final-visible helper nodes contribute their actual reader packets.
    Deduplicate by doc_id against the unchanged original-query Top10 and
    earlier helper documents; retain first occurrence order and complete text.
    """
    assert case['nodes'] and case['nodes'][-1]['role'] == 'goal'
    roots = case['original_retrieval']['docs']
    labels = {d['doc_id']: f'R{i}' for i, d in enumerate(roots, 1)}
    contents = {d['doc_id']: (d['source'], d['text']) for d in roots}
    groups, extra = [], []
    for n in case['nodes']:
        refs = []
        for d in n['reader_packet']['docs']:
            content = (d['source'], d['text'])
            assert d['doc_id'] not in contents or contents[d['doc_id']] == content, case['id']
            contents[d['doc_id']] = content
            if d['doc_id'] not in labels:
                label = f'A{len(extra) + 1}'
                labels[d['doc_id']] = label
                extra.append(passage(label, d))
            refs.append(labels[d['doc_id']])
        groups.append(f'Step {n["step_id"]} ({n["role"]})\nQuestion: {n["request"]}\n'
                      + 'Passages: ' + ', '.join(refs))
    return '\n\n'.join(groups + extra), len(extra), len(case['nodes'])


def main():
    old = read(PLAN / 'reviews/20261023_preexecution_plan_budget/REVIEW.json')
    refs = {(r['dataset'], r['method']): r for r in
            read(PLAN / 'execution_details/COST_REFERENCE.frozen.json')['groups']}
    inventory, mean_checks = [], []
    for r in old['baseline_inventory']:
        if r['method'] not in METHODS:
            continue
        mean_checks.append(dict(dataset=r['dataset'], method=r['method'], source=r['source'],
            **checkpoint_mean_check(ROOT / r['source'], refs[r['dataset'], r['method']])))
        vals = predictions(ROOT / r['source'])
        for path in r['recovery_sources']:
            for qid, v in predictions(ROOT / path).items():
                if not scorable(vals.get(qid, {})):
                    vals[qid] = v
        qs = {v['qid']: v['question'] for v in (json.loads(s) for s in
              (ROOT / 'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen'
               / r['dataset'] / 'formal.questions.jsonl').read_text(encoding='utf8').splitlines() if s.strip())}
        assert len(qs) == 500 and set(vals) <= set(qs)
        assert all(v.get('question', qs[q]) == qs[q] for q, v in vals.items())
        count = sum(map(scorable, vals.values()))
        assert count == r['scorable'], (r['dataset'], r['method'], count)
        inventory.append({**r, 'scorable': count,
                          'statuses': dict(Counter(v['status'] for v in vals.values()))})

    selected = {}
    for path in sorted((OUT / 'p02_formal1500_v1_20261021').glob('rounds/r*/cases/*/*/*.json')):
        v = read(path)
        if not scorable(selected.get(v['id'], ({}, None))[0]):
            selected[v['id']] = (v, path)
    formal = {ds: [(v, p) for v, p in selected.values() if v['dataset'] == ds and scorable(v)]
              for ds in DATASETS}
    assert {ds: len(v) for ds, v in formal.items()} == {'hotpotqa': 283, '2wiki': 498, 'musique': 0}
    mu_dev = [(read(p), p) for p in sorted((OUT / 'p02_merge_cachefix_v1_20261021'
              / 'rounds/r000/cases/musique/timing_only').glob('*.json'))]
    assert len(mu_dev) == 100 and all(scorable(v) for v, _ in mu_dev)
    assert len({v['id'] for v, _ in mu_dev}) == 100
    mu_usage = [usage_cost(v['logical_calls']) for v, _ in mu_dev]
    assert not any(u['unpriced_calls'] for u in mu_usage)
    mu_mean = mean(u['known_cny'] for u in mu_usage)
    assert abs(mu_mean - 0.008257268) < 1e-10

    rows = []
    for ds in DATASETS:
        for method in METHODS:
            ref = refs[ds, method]
            if method == P02:
                count, old_config = len(formal[ds]), 'native frozen P02; new shared execution layer pending'
            else:
                inv = next(r for r in inventory if (r['dataset'], r['method']) == (ds, method))
                count, old_config = inv['scorable'], inv['config']
            changed = method in {'b02_selfask_dense', 'b06_decomp_bge'} or (
                method == 'b08_qd_rerank' and ds != 'hotpotqa')
            candidate = 0 if changed else count
            fresh = 500 - candidate
            mu, basis, n = ref['mean_cny'], ref['basis'], ref['n']
            if method == P02 and ds == 'musique':
                mu, basis, n = mu_mean, 'same_dataset_native_P02_dev100_logical_usage_proxy', 100
            factor = 5 / old_config['max_search'] if method == 'b02_selfask_dense' else 1.0
            rows.append(dict(dataset=ds, method=method, old_config=old_config,
                old_scorable_count=count, configuration_change_requires_recompute=changed,
                old_changed_configuration_cases=count if changed else 0,
                whole_case_reuse_candidates=candidate, strict_reuse_verified=False,
                conditional_fresh_flow_equivalents=fresh, historical_mean_cny=mu,
                historical_basis=basis, historical_n=n,
                quality_proxy_cny=fresh * mu, step_limit_sensitivity_factor=factor,
                quality_step_sensitivity_cny=fresh * mu * factor,
                cold_all500_proxy_cny=500 * mu,
                cold_all500_step_sensitivity_cny=500 * mu * factor,
                optional_timing50_proxy_cny=50 * mu,
                optional_timing50_step_sensitivity_cny=50 * mu * factor))

    proxy_rows, proxy_summaries = [], []
    for ds in DATASETS:
        cases = formal[ds] if ds != 'musique' else mu_dev
        current = []
        for v, path in cases:
            finals = [c for c in v['logical_calls'] if c['stage'].split('/')[-1] == 'final'
                      and isinstance(c.get('usage', {}).get('prompt_tokens'), int)]
            assert len(finals) == 1, (v['id'], len(finals))
            final = finals[0]
            assert final.get('finish_reason') == 'stop'
            prompt, helper = v['final_prompt'], qa_helper_proxy(v)
            raw_helper, extra_n, node_n = raw_helper_proxy(v)
            qa = neutral_final_proxy(v, helper)
            raw = neutral_final_proxy(v, raw_helper)
            p, o = final['usage']['prompt_tokens'], final['usage']['completion_tokens']
            # Remote qwen-plus tokenization is not available offline. Calibrate
            # per case against the actual billed QA prompt, including overhead.
            ratio = p / len(prompt)
            qp = len(qa) * ratio
            rp = len(raw) * ratio
            item = dict(dataset=ds, qid=v['qid'], source=rel(path),
                proxy_basis='formal historical native P02' if ds != 'musique' else 'same-method native P02 dev100',
                historical_qa_input_tokens_actual=p, qa_output_tokens_actual=o,
                historical_qa_prompt_characters=len(prompt),
                qa_prompt_characters=len(qa), raw_prompt_characters=len(raw),
                actual_prompt_tokens_per_character=ratio,
                qa_input_tokens_estimated=qp,
                raw_input_tokens_estimated=rp, raw_extra_unique_documents=extra_n,
                visible_helper_nodes=node_n,
                qa_final_cost_cny=qp * INPUT_RATE + o * OUTPUT_RATE,
                raw_final_cost_proxy_cny=rp * INPUT_RATE + o * OUTPUT_RATE,
                qa_raw_pair_proxy_cny=(qp + rp) * INPUT_RATE + 2 * o * OUTPUT_RATE,
                qa_raw_pair_50pct_input_mean_output64_scenario_cny=1.5 * (qp + rp) * INPUT_RATE
                                                        + 2 * 64 * OUTPUT_RATE)
            current.append(item)
        proxy_rows.extend(current)
        proxy_summaries.append(dict(dataset=ds, source_cases=len(current), planned_queries=100,
            source_role='historical formal completion sample' if ds != 'musique' else 'same-method dev100 proxy; not formal',
            historical_qa_mean_input_tokens_actual=mean(r['historical_qa_input_tokens_actual'] for r in current),
            qa_mean_input_tokens_estimated=mean(r['qa_input_tokens_estimated'] for r in current),
            raw_mean_input_tokens_estimated=mean(r['raw_input_tokens_estimated'] for r in current),
            raw_max_input_tokens_estimated=max(r['raw_input_tokens_estimated'] for r in current),
            mean_final_output_tokens=mean(r['qa_output_tokens_actual'] for r in current),
            raw_mean_extra_unique_documents=mean(r['raw_extra_unique_documents'] for r in current),
            qa100_final_cost_cny=100 * mean(r['qa_final_cost_cny'] for r in current),
            raw100_final_cost_proxy_cny=100 * mean(r['raw_final_cost_proxy_cny'] for r in current),
            pair100_proxy_cny=100 * mean(r['qa_raw_pair_proxy_cny'] for r in current),
            pair100_input50pct_mean_output64_scenario_cny=100 * mean(
                r['qa_raw_pair_50pct_input_mean_output64_scenario_cny'] for r in current)))

    with closing(sqlite3.connect((OUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        known, reserve = db.execute('SELECT coalesce(sum(actual),0), '
            'coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
    totals = {key: sum(r[key] for r in rows) for key in [
        'old_scorable_count', 'whole_case_reuse_candidates', 'old_changed_configuration_cases',
        'conditional_fresh_flow_equivalents', 'quality_proxy_cny', 'quality_step_sensitivity_cny',
        'cold_all500_proxy_cny', 'cold_all500_step_sensitivity_cny',
        'optional_timing50_proxy_cny', 'optional_timing50_step_sensitivity_cny']}
    totals['qa_raw_600_final_conditions_proxy_cny'] = sum(r['pair100_proxy_cny'] for r in proxy_summaries)
    totals['qa_raw_input50pct_mean_output64_scenario_cny'] = sum(
        r['pair100_input50pct_mean_output64_scenario_cny'] for r in proxy_summaries)
    totals['necessary_proxy_cny'] = totals['quality_proxy_cny'] + totals['qa_raw_600_final_conditions_proxy_cny']
    totals['necessary_step_sensitivity_cny'] = totals['quality_step_sensitivity_cny'] + totals['qa_raw_600_final_conditions_proxy_cny']
    totals['necessary_50pct_main_plus_qa_raw_input50pct_mean_output64_scenario_cny'] = (
        1.5 * totals['quality_step_sensitivity_cny'] + totals['qa_raw_input50pct_mean_output64_scenario_cny'])
    totals['no_reuse_credited_proxy_cny'] = totals['cold_all500_proxy_cny'] + totals['qa_raw_600_final_conditions_proxy_cny']
    totals['no_reuse_credited_step_sensitivity_cny'] = totals['cold_all500_step_sensitivity_cny'] + totals['qa_raw_600_final_conditions_proxy_cny']
    totals['no_reuse_50pct_main_plus_qa_raw_input50pct_mean_output64_scenario_cny'] = (
        1.5 * totals['cold_all500_step_sensitivity_cny'] + totals['qa_raw_input50pct_mean_output64_scenario_cny'])
    result = dict(utc=datetime.now(timezone.utc).isoformat(), status='plan_only_no_evaluation',
        scope=dict(main_methods=list(METHODS), main_quality_flows=9000,
                   qa_raw_queries=300, qa_raw_final_conditions=600, optional_timing_flows=900,
                   qa_raw_output_token_allowance=OUTPUT_ALLOWANCE,
                   qa_raw_final_prompt=rel(PROMPT_SOURCE),
                   qa_raw_rendering='exact new template plus cost-only rendering consistent with proposed schema; not an execution renderer'),
        unit_prices=dict(currency='CNY', input_per_million=0.8, output_per_million=2,
                         basis='existing frozen project planning rates; no live price re-verification'),
        totals=totals, main_rows=rows, baseline_inventory=inventory,
        baseline_checkpoint_cost_mean_checks=mean_checks,
        p02_formal_statuses={ds: dict(Counter(v['status'] for v, _ in selected.values()
                                            if v['dataset'] == ds)) for ds in DATASETS},
        p02_musique_dev100_usage=dict(n=100, mean_complete_flow_cny=mu_mean,
            input_tokens=sum(x['input_tokens'] for x in mu_usage),
            output_tokens=sum(x['output_tokens'] for x in mu_usage),
            source='outputs/query_decomposition/hotpot_formal_eval_v1/p02_merge_cachefix_v1_20261021/rounds/r000/cases/musique/timing_only'),
        qa_raw_proxy_summaries=proxy_summaries,
        ledger_snapshot=dict(source='outputs/query_decomposition/hotpot_formal_eval_v1/budget.sqlite3',
            known_cny=known, unknown_reserved_cny=reserve, occupied_cny=known + reserve,
            scope='this shared ledger only, not all later side experiments nor verified provider invoice',
            historical_not_incremental=True),
        strict_reuse_audit_completed=False, strict_reuse_verified_case_count=0,
        new_api_generation_requests=0, new_gpu_jobs=0,
        limitations=[
            'Scorable old cases are candidates only. New shared-layer complete-payload/resource/receipt equivalence remains unverified.',
            'Changed SelfAsk and DecomP cells plus 2Wiki/MuSiQue QD are budgeted as full 500-flow equivalents; zero changed-prefix reuse credit.',
            'Top10/CoT and Hotpot QD candidate credit is conditional; no-reuse-credit scenario is also given, not an instruction to replay valid responses.',
            'Cost means describe old parameter choices; k reductions and adaptive stopping can alter cost nonlinearly.',
            'SelfAsk 5/old_max_search scales the full-flow cost, not measured actual stages; neither sensitivity scenario is a bound or confidence interval.',
            'P02 MuSiQue uses native timing_only dev100 usage; missing MuSiQue SelfAsk/DecomP/QD use labelled cross-dataset same-method references.',
            'QA/raw estimates use all available historical native P02 records rather than only the planned stratified main500 sample. The common new template is read exactly, but old-prefix cost renders are not validated new shared-layer inputs.',
            'Raw helper cost proxy preserves question/source grouping and all actual reader documents, de-duplicates by doc_id against root and earlier helper docs, and never truncates.',
            'Both shared pre-frozen zero-shot prompt token estimates use each historical case actual QA prompt_tokens/character ratio; this is not exact provider tokenization. Output length is provisionally the same observed short-answer mean.',
            'The final output allowance is 32768, not 64. The 64-token mean-output sensitivity is an illustrative spend scenario, not a frozen cap or a worst-case limit.',
            'QA/raw budgets all 600 final generations and zero incremental upstream generations; upstream full logical cost still belongs in reported method cost.',
            'The QA/raw common shared pre-frozen zero-shot interface differs from the old and main native final; both 300-result arms are budgeted as new final conditions.',
            'The 50-percent expense/input scenario is a planning sensitivity, not a cap, upper bound, or provider-price guarantee. Five-attempt recovery is not five full-flow repeats.',
            'Unknown liabilities remain unresolved and are not zero; no unknown request may be replayed based on this estimate.',
            'GPU/cluster time has no verified cash rate and is excluded from CNY totals; optional timing repeats are separate from necessary budget.',
            'No new inference, GPU job, evaluation process, sample selection or runtime configuration change is performed by this script.'
        ])
    (HERE / 'BUDGET.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (HERE / 'QA_RAW_PROXY_ROWS.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in proxy_rows), encoding='utf8')
    lines = ['# Revision007 六方法评估：费用和复用候选盘点', '',
        '仅规划；本脚本未执行模型推理、API、GPU或修改冻结配置。金额为人民币，沿用项目冻结估费率（输入0.8元/百万token、输出2元/百万token），不是本轮核验的供应商账单。', '',
        '## 主表：6方法 × 3数据集 × 500题 = 9000流程', '',
        '| 数据集 | 方法 | 旧可评分 | 参数变化旧记录 | 整题复用候选 | 条件性新增流程当量 | 旧均价估费 | 轮次敏感性 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["dataset"]} | {r["method"]} | {r["old_scorable_count"]} | '
                     f'{r["old_changed_configuration_cases"]} | {r["whole_case_reuse_candidates"]} | '
                     f'{r["conditional_fresh_flow_equivalents"]} | {r["quality_proxy_cny"]:.4f} | '
                     f'{r["quality_step_sensitivity_cny"]:.4f} |')
    lines += ['', f'合计旧可评分{totals["old_scorable_count"]}条，其中{totals["old_changed_configuration_cases"]}条属于参数变化，'
        f'{totals["whole_case_reuse_candidates"]}条仅是复用候选；严格迁移已验证为0条，检查尚未完成。条件性新增{totals["conditional_fresh_flow_equivalents"]}流程当量，'
        f'估费{totals["quality_proxy_cny"]:.4f}元；SelfAsk上限变化敏感性为{totals["quality_step_sensitivity_cny"]:.4f}元。', '',
        'SelfAsk三集k5/max5、DecomP三集k2/outer与inner8/CoT、QD的2Wiki和MuSiQue参数变化均不抵扣旧前缀。Top10、Top10+CoT与Hotpot QD保持原设定，但仍须完整输入/资源/回执检查。P02原生结果不能直接标成新统一执行层完成。', '',
        f'不预先抵扣任何整题复用时，主表估费{totals["cold_all500_proxy_cny"]:.4f}元，轮次敏感性{totals["cold_all500_step_sensitivity_cny"]:.4f}元；这只是复用失败情景，不授权重跑已有效请求。', '',
        '## 新增 P02 QA/raw：固定300题、600个最终生成条件', '',
        '预算读取qa_raw/PROMPT.txt精确共同zero-shot模板，并依该方案渲染旧原生P02轨迹作长度代理：root为R1..、完整doc_id/source/text；QA保留全部ordinary/goal的Question与实际传播Answer；Raw取这些节点实际reader原文、保留Question与R/A关联，稳定docID去重，不重排、不截断。资源namespace固定在各case内，同ID内容冲突即停止估算。这里的离线渲染仅服务预算，不是新统一层执行器或600个合格输入。', '',
        '| 数据集 | 代理来源数 | QA估计平均输入token | Raw估计平均输入token | Raw最大估计输入token | 200个final条件估费 | 输入+50%且平均输出64情景 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in proxy_summaries:
        lines.append(f'| {r["dataset"]} | {r["source_cases"]} | {r["qa_mean_input_tokens_estimated"]:.1f} | '
                     f'{r["raw_mean_input_tokens_estimated"]:.1f} | {r["raw_max_input_tokens_estimated"]:.1f} | '
                     f'{r["pair100_proxy_cny"]:.4f} | {r["pair100_input50pct_mean_output64_scenario_cny"]:.4f} |')
    lines += ['', 'Hotpot/2Wiki使用旧原生P02正式完整轨迹，MuSiQue使用同版本dev100 timing_only控制；未使用S02。长度按各题实际QA输入token/字符校准Raw，不是精确Qwen分词。计划中的分层300题和新统一层最终prompt尚未实际执行，代理样本不冒充该300题。', '',
        f'新增QA/raw合计{totals["qa_raw_600_final_conditions_proxy_cny"]:.4f}元；输入+50%、平均输出假设64token的敏感性情景为{totals["qa_raw_input50pct_mean_output64_scenario_cny"]:.4f}元。600个final条件不等于固定600次API提交，失败尝试与同payload缓存另记。实际允许输出上限是32768，64只是此情景的平均用量假设，既非配置上限也非最坏费用。两臂使用共同、预先冻结的zero-shot提示，移除旧四例；新接口可能改变输出长度，旧stop短答案均值只是估费代理。所有上游由主表承担，新增费用不重复计入；完整方法成本仍须包含上游并排除未使用的主表原final。', '',
        '## 预算汇总与历史负债', '',
        f'- 必要新增费用（有条件的旧整题复用）：{totals["necessary_proxy_cny"]:.4f}元；SelfAsk轮次敏感性{totals["necessary_step_sensitivity_cny"]:.4f}元。',
        f'- 主表轮次敏感性再增加50%，加上QA/raw输入+50%/平均输出64：{totals["necessary_50pct_main_plus_qa_raw_input50pct_mean_output64_scenario_cny"]:.4f}元。此项不是上界或预算硬帽。',
        f'- 不抵扣旧整题复用时必要费用：{totals["no_reuse_credited_proxy_cny"]:.4f}元，轮次敏感性{totals["no_reuse_credited_step_sensitivity_cny"]:.4f}元。',
        f'- 不抵扣整题复用，再对主表轮次敏感性增加50%，加上QA/raw输入+50%/平均输出64：{totals["no_reuse_50pct_main_plus_qa_raw_input50pct_mean_output64_scenario_cny"]:.4f}元。同样是审慎规划情景，不是费用上界或预算硬帽。',
        f'- 可选计时900完整流程：{totals["optional_timing50_proxy_cny"]:.4f}元；轮次敏感性{totals["optional_timing50_step_sensitivity_cny"]:.4f}元。不计入必要费用。',
        f'- 当前共享账本历史已知{known:.7f}元，未知预留{reserve:.7f}元，合计占用{known + reserve:.7f}元；不是新增费用、不是全部旁支账本，也不是已核实供应商账单。',
        '- GPU/集群没有已核实现金费率，不编造人民币金额。技术失败最多5次是失败逻辑阶段累计尝试规则，不是预算乘5；未知请求须先核回执。', '',
        '## 复算', '',
        '`python analysis/narrative_eval_revision007_20261023/budget/recalculate_budget.py`', '',
        '脚本只读源JSON和SQLite（mode=ro并PRAGMA query_only），只写本目录三个结果文件。公式、逐单元参数、来源和881条QA/raw长度代理记录随本目录保留。']
    (HERE / 'BUDGET.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'totals': totals, 'qa_raw': proxy_summaries,
                      'shared_ledger': result['ledger_snapshot']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
