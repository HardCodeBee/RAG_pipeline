"""Publish only local experiment report and append its outcome to the agreement."""
import contextlib
import io
import json
from . import run, summarize, audit


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        summarize.main()
        audit.main()
    summary = run.read(run.OUT/'SUMMARY.json')
    checked = run.read(run.OUT/'OFFLINE_AUDIT.json')
    inputs = {r['id']:r for r in run.rows(run.HERE/'INPUTS90.frozen.jsonl')}
    changed = {r['id']:r for r in run.rows(run.OUT/'CHANGED_FINAL_CASES.local.jsonl')}
    notes = [
        ('2wiki:654dbe680bb011ebab90acde48001122', '保留前序关系事实：few-shot 的正例',
         'G2和I均弃答；F使用前序QA中“父亲的父亲＝Hermann Georg”得到正确答案。当前材料未重述这条关系，不应自动视为反证。F同时有冗长、循环和弱证据归因，因此最终正确不等于整段解释均可靠。'),
        ('2wiki:079947de0baf11ebab90acde48001122', '答案修复，同时出现错误的引文归因',
         'G2因当前检索未再次确认而弃答；I/F正确使用前序QA中的公公身份。然而两者又把Maximilian自身父母的传记误写成支持他是Franz父亲的证据。正确性收益不能直接解释为完整证据链得到验证。'),
        ('musique:2hop__289517_139339', '原题条件被擅自收紧：相近示例仍未防止回退',
         '问题通过After Hours指认演员，并未要求提名来自这部电影。I正确回答Griffin Dunne的Academy Award for Best Live Action Short Film提名；F看到同一条事实仍附加“因出演After Hours获提名”的条件并弃答。实际F请求含全部四个示例，包括直接针对不同电影提名的Example 2；返回stop，非缺模板或截断。'),
        ('hotpotqa:5ab64cfa55429953192ad2b7', '过度要求完整字段而弃答',
         'F声称寿命比较必需双方出生、死亡日期，因Momaday死亡日期缺失而弃答。G2/I给出数据集正确答案。比较原则上可以通过下界判定，不必总知道死亡日期；本例应依据冻结材料的时间和生存描述判断。G2/I部分解释也粗糙，不能以答案满分证明推理均正确。'),
        ('hotpotqa:5ab8c3c65542991b5579effd', '数值表达造成原生分数下降',
         'G2回答Two而I/F回答2。三者表达同一数量，原生Hotpot计分却从F1/EM=1降为0。保留原生评分，同时把这项变化与事实性退步区分。'),
        ('hotpotqa:5a71474b5542994082a3e74f', '额外日期造成答案匹配损失',
         '三组指向同一场补选；I/F额外给出日期，原生F1由1下降到0.7143/0.6667，EM下降。它说明需要控制最终短答案粒度，不足以证明组合事实错误。'),
        ('musique:4hop3__275416_24325_156850_10557', '512-token硬上限截断',
         'F的goal输出达到512 tokens、finish_reason=length，未进入final；按预先协议保留为算法失败，F1/EM计0。G2回答Latin（F1=0.6667），I回答Walloon（0）。不通过增加长度或重试改写这一轮结果。'),
    ]
    records = []
    for ident,title,note in notes:
        row = {**changed[ident], 'review_title':title,'review_note':note,
               'frozen_goal_reader_packet':inputs[ident]['goal_reader_packet']}
        if ident == 'musique:4hop3__275416_24325_156850_10557':
            row['length_ended_raw_receipt'] = checked['length_failure'][0]
        if ident == 'musique:2hop__289517_139339':
            row['four_examples_present_in_verified_exact_payload'] = True
        records.append(row)
    run.jsonl(run.OUT/'CASE_REVIEW.local.jsonl', records)
    lines = ['# 固定90题 goal 提示词实验：案例复核', '',
             '以下案例用于解释实测差异，不构成对所有输出逐句事实准确率的估计。完整实际goal文本、最终回答、原生分数和冻结检索材料见CASE_REVIEW.local.jsonl。', '']
    for row in records:
        lines.extend(['## '+row['review_title'], '', '`'+row['id']+'`', '',
                      '原题：'+row['question'], '', row['review_note'], '',
                      '| 版本 | 实际最终回答 | F1 / EM |','|---|---|---:|'])
        for arm,v in row['variants'].items():
            m = v['native_metrics']
            lines.append(f"| {arm} | {v['prediction'] or '[算法失败，无final输出]'} | {m['f1']:.4f} / {m['em']:.0f} |")
        lines.append('')
    (run.OUT/'CASE_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    macro = summary['dataset_equal_mean']
    costs = summary['costs_same_cases']
    report_lines = ['', '## Interpretation and verification', '',
        'G2 is the old synthesis paragraph. I is the revised instruction package without demonstrations. F inserts the four frozen fictional demonstrations into I. The actual frozen templates are stored with the experiment.', '',
        f"Dataset-equal F1 changes: I minus G2 = {(macro['I']['F1']-macro['G2']['F1'])*100:.4f} points; F minus G2 = {(macro['F']['F1']-macro['G2']['F1'])*100:.4f}; F minus I = {(macro['F']['F1']-macro['I']['F1'])*100:.4f}.", '',
        'On the same86 cases, I improves7/hurts7/ties72 vs G2. F improves6/hurts8/ties72 vs G2; vs I it improves5/hurts7/ties74. The pairwise-only I/F set in SUMMARY.json additionally contains one case and is clearly labelled n87.', '',
        'All three synthesis variants remain below matched historical native P02. That historical comparison also changes the goal/final interface and output allowance, so it is not a pure test of reasoning text.', '',
        'The demonstrated gain is dataset-dependent: F helps 2Wiki but hurts HotpotQA and MuSiQue. Some losses are native answer-format mismatches; others are genuine added conditions or unnecessary abstention. See [actual cases](CASE_REVIEW.md) and full local case evidence.', '',
        'All I87 and F87 returned goals use a single paragraph and include at least one requested marker. Four F outputs and one I output contain both marker forms; format compliance does not establish reasoning correctness. One F output reaches the512-token cap; it remains a scored failure.', '',
        'Exploratory paired bootstrap, resampling within each dataset with equal dataset weights, seed20261022 and10000 replicates; all intervals below include zero. These exposed development cases do not establish a stable improvement.', '',
        '| Contrast | Macro F1 delta (points) | Exploratory95% interval |','|---|---:|---:|']
    for key,value in checked['exploratory_paired_bootstrap']['results'].items():
        lo,hi=value['percentile_95_interval']
        report_lines.append(f"| {key} | {value['dataset_equal_mean_F1_delta']*100:.2f} | [{lo*100:.2f}, {hi*100:.2f}] |")
    report_lines += ['',
        f"On the same85 completed-flow cases, I goal+final expense is{(costs['I']['mean_goal_final_CNY']/costs['G2']['mean_goal_final_CNY']-1)*100:.2f}% above G2; F is{(costs['F']['mean_goal_final_CNY']/costs['G2']['mean_goal_final_CNY']-1)*100:.2f}% above G2 and{(costs['F']['mean_goal_final_CNY']/costs['I']['mean_goal_final_CNY']-1)*100:.2f}% above I. Full-flow estimates add identical reused upstream usage, not new upstream spending.", '',
        'Offline audit passed: full frozen payload identity, model identity, finish reasons, raw usage totals, predictions, same-case native scoring, per-receipt ledger expense, frozen inputs and main hold hashes. Two typed pre-transmission connection timeouts were recovered within the predeclared5-attempt rule; their reservations remain conservatively retained. Seven unresolved condition outcomes across4 questions were not reissued.', '',
        'Next research direction, not executed here: prioritize preserving intermediate relation facts, avoiding added question conditions and excessive abstention, and isolating final answer brevity from goal synthesis. Do not infer that longer explanations or more examples alone improve F1. No variant is promoted by this side experiment.']
    with (run.OUT/'REPORT.md').open('a',encoding='utf8') as handle:
        handle.write('\n'.join(report_lines)+'\n')
    result = {'id':'P02_GOAL_FEWSHOT_FIXED90_COMPLETED_20261022','utc':run.utc(),
              'experiment':run.HERE.name,'terminal_conditions':270,'same_three_arm_scorable':86,
              'status_counts':checked['status_counts'],'dataset_equal_mean':macro,
              'fresh_estimated_expense_CNY':summary['fresh_actual_CNY'],
              'main_evaluation_resumed':False,'method_promoted':False,'published':False,
              'report':str((run.OUT/'REPORT.md').relative_to(run.ROOT)),
              'report_sha256':run.sha(run.OUT/'REPORT.md'),'offline_audit_passed':True}
    if not any(r.get('id')==result['id'] for r in run.rows(run.PLAN/'DECISIONS.jsonl')):
        with (run.PLAN/'DECISIONS.jsonl').open('a',encoding='utf8') as handle:
            handle.write(json.dumps(result,ensure_ascii=False)+'\n')
    run.save(run.REVIEW/'COMPLETED.json',result)
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':
    main()
