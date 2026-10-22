"""Compare the completed second bundle with both preserved quick90 controls."""
from collections import Counter
import random

from . import run, prompts


def main():
    current = run.read(run.OUT / 'FINAL_SUMMARY.json')
    previous_out = run.common.OUT.parent / 'p02_three_prompts_quick90_v1_20261021/approved_unknown_recovery003'
    previous = run.read(previous_out / 'FINAL_SUMMARY.json')
    assert current['status'] == previous['status'] == 'completed'
    now = {r['id']: r for r in run.rows(run.OUT / 'PAIRED_DETAILS.jsonl')}
    before = {r['id']: r for r in run.rows(previous_out / 'PAIRED_DETAILS.jsonl')}
    assert set(now) == set(before) and len(now) == 90
    for cid in now:
        assert now[cid]['baseline'] == before[cid]['baseline']
    groups = [[now[cid]['candidate']['f1'] - before[cid]['candidate']['f1']
               for cid in now if now[cid]['dataset'] == ds] for ds in run.common.DATASETS]
    rng = random.Random(20261021)
    boot = sorted(sum(sum(rng.choices(g, k=len(g))) for g in groups) / 90 for _ in range(10000))
    deltas = [d for group in groups for d in group]
    comparison = {
        'planned': 90, 'completed': 90, 'same_question_ids': True,
        'previous_bundle_primary_original_caps_only': True,
        'previous_single_query_cap512_diagnostic_excluded': True,
        'F1_delta_vs_previous': sum(deltas) / 90,
        'paired95_F1_delta_vs_previous': [boot[249], boot[9749]],
        'improved_vs_previous': sum(d > 1e-12 for d in deltas),
        'tied_vs_previous': sum(abs(d) <= 1e-12 for d in deltas),
        'harmed_vs_previous': sum(d < -1e-12 for d in deltas),
        'template_characters': {k: len(v) for k, v in [('ordinary', prompts.ORDINARY), ('goal', prompts.GOAL), ('final', prompts.FINAL)]},
        'metrics': {},
    }
    lines = ['三段 prompt 第二套整体 quick90 结果', '',
        '同一批开发90题，各数据集30题；三段均按用户提供原文替换。模型、分解、改写、检索、formatter及256/128/64输出上限沿用原版。',
        '冻结原版为精确回放；两套候选按自身前置回答继续改写与检索，仅成功且完整payload相同才复用。上一套单题goal512诊断不混入该表。', '',
        '| 数据集 | 题数 | 冻结原版F1 | 上一套F1 | 本套F1 | 本套减原版 | 本套减上一套 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for ds in (*run.common.DATASETS, 'overall'):
        m, old = current['metrics'][ds], previous['metrics'][ds]
        comparison['metrics'][ds] = {'baseline_F1': m['baseline_F1'], 'previous_F1': old['candidate_F1'],
            'current_F1': m['candidate_F1'], 'baseline_EM': m['baseline_EM'],
            'previous_EM': old['candidate_EM'], 'current_EM': m['candidate_EM']}
        lines.append(f"| {ds} | {m['complete']} | {100*m['baseline_F1']:.2f} | {100*old['candidate_F1']:.2f} | {100*m['candidate_F1']:.2f} | {100*(m['candidate_F1']-m['baseline_F1']):+.2f} | {100*(m['candidate_F1']-old['candidate_F1']):+.2f} |")
    m = current['metrics']['overall']
    lines += ['', f"总体EM：原版 {100*m['baseline_EM']:.2f}%，上一套 {100*previous['metrics']['overall']['candidate_EM']:.2f}%，本套 {100*m['candidate_EM']:.2f}%。",
        f"相对原版F1升/平/降：{m['improved']}/{m['tied']}/{m['harmed']}；相对上一套：{comparison['improved_vs_previous']}/{comparison['tied_vs_previous']}/{comparison['harmed_vs_previous']}。",
        f"相对原版F1差值95%分层配对bootstrap区间：{[round(100*x,4) for x in current['stratified_paired_bootstrap95_F1']]} 个百分点；相对上一套：{[round(100*x,4) for x in comparison['paired95_F1_delta_vs_previous']]} 个百分点。",
        f"终态：{current['terminal_statuses']}；length响应：{len(current['length_responses'])}。算法失败按既定规则保留在90题分母中计0。",
        f"实际prompt逐项核对：{current['exact_actual_prompts_verified']}；离线额外核对全部6个无普通子问题案例。",
        f"本轮新增估算生成费用 {current['new_estimated_CNY']:.6f} 元，未结算预留 {current['unknown_reserved_CNY']:.6f} 元。逻辑tokens和旧费用不重复计作新增费用。",
        '该90题已反复用于开发，本次为三段联合效果，不能判断单段贡献，也不能作为独立确认。未激活候选；其他暂停实验保持暂停。']
    run.save(run.OUT / 'BUNDLE_COMPARISON.json', comparison)
    (run.OUT / 'BUNDLE_COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    run.emit(comparison)


if __name__ == '__main__':
    main()
