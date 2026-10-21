"""Offline final accounting/audit after the isolated generator terminates."""
import json
import sqlite3
import time
from pathlib import Path
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.prepare import ROOT,HERE,OUT,OUTPUT,read,rows,sha,save,log
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.pipeline import ARMS
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.receipts import sources,baseline_case
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.report import report,selected
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.run import identity
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json


def finalize():
    # No submissions, model calls, scheduler edits or parent writes here.
    while not (OUT/'TERMINAL.json').exists(): time.sleep(20)
    identity(read(HERE/'PROTOCOL.frozen.json'))
    results=report()
    prefix=str(OUT.relative_to(OUTPUT)).replace('\\','/')+'/'
    with sqlite3.connect((OUTPUT/'budget.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        charges=[r for r in db.execute('SELECT namespace,reserved,actual,status FROM charges') if r[0].replace('\\','/').startswith(prefix)]
    costs={}
    for arm in ARMS:
        cc=[r for r in charges if r[0].replace('\\','/').split('/')[-1]==arm]
        costs[arm]={'new_returned_usage_cny':sum(r[2] or 0 for r in cc),'new_reserved_cny':sum(r[1] for r in cc if r[2] is None),'new_submissions':len(cc)}
    results['budget']['by_arm']=costs
    src=sources(); checked=0; upstream_calls=0
    for case in rows(HERE/'runtime300.frozen.jsonl'):
        baseline=baseline_case(src[case['id']])
        result=selected('C_final_readout_P1',case)
        if not result or not result['status'].startswith('success'): continue
        actual=read(ROOT/result['source'])
        assert actual['history_all']==baseline['history_all']
        assert actual['original_retrieval']==baseline['original_retrieval']
        assert actual['final_context']==baseline['final_context']
        old={r['payload_sha256'] for r in src[case['id']]['logical_calls'] if r['stage']!='legacy/final'}
        for call in actual['logical_calls']:
            if call['stage']=='legacy/final': continue
            assert call['reused'] and call['payload_sha256'] in old
            upstream_calls+=1
        checked+=1
    audit={'utc':utc(),'frozen_sources_intact':True,'baseline_cases_hash_checked':len(src),
           'P1_successes_full_upstream_equivalence_checked':checked,'P1_successful_reused_upstream_calls_checked':upstream_calls,
           'D_E_F_full_replay':read(OUT/'OFFLINE_VALIDATION.frozen.json'),
           'cost_display_correction':'Windows namespace separators normalized for grouping. Generation, F1, total spending and caps unchanged.',
           'parent_method_or_result_writes':False,'additional_api_calls':0}
    save(OUT/'FINAL_AUDIT.frozen.json',audit)
    save(OUT/'FINAL_RESULTS.frozen.json',results)
    locked_json(OUT/'RESULTS_LATEST.json',results)
    lines=['# B0六项独立改动：最终报告','','同一批300题，每数据集100题，均为已用开发诊断样本。每组只修改一项。', '',
           '| 改动 | 可计分 | 原始B0 F1 | 本组F1 | 差值（百分点） | 升/平/降 | 新增费用 |',
           '|---|---:|---:|---:|---:|---|---:|']
    for arm,a in results['arms'].items():
        if a['scorable']:
            lines.append(f"| {arm} | {a['scorable']}/300 | {100*a['B0_same_queries_F1']:.2f}% | {100*a['F1']:.2f}% | {100*a['delta_F1']:+.2f} | {a['up']}/{a['tie']}/{a['down']} | ¥{costs[arm]['new_returned_usage_cny']:.4f} |")
        else: lines.append(f'| {arm} | 0/300 | — | — | — | — | — |')
    lines+=['','## 每数据集与同题B0对照','', '| 改动 | 数据集 | 可计分 | B0 F1 | 本组F1 | 差值（百分点） |','|---|---|---:|---:|---:|---:|']
    for arm,a in results['arms'].items():
        for ds,b in a['by_dataset'].items():
            if b['scorable']: lines.append(f"| {arm} | {ds} | {b['scorable']} | {100*b['B0_F1']:.2f}% | {100*b['F1']:.2f}% | {100*b['delta_F1']:+.2f} |")
    lines+=['','## 判断边界','',
            'D/E/F分别完整回放300题均未触发。输入、检索顺序和API载荷逐项一致，因此复用B0答案，不重复调用。该结论只说明它们单独用于本批B0轨迹没有影响，不能说明与C08配合时没有价值。',
            'C08提示词实验剥离了其渲染器内嵌的运行时缺答过滤，保留B0的短答案整理，避免同时修改两个因素。',
            '评价保持数据集原生F1。删去冗余文字、16/Sixteen、直引号/弯引号均可能改变F1，分数上升不能直接解释为事实准确率或检索覆盖提高。',
            '置信区间为5000次按数据集分层配对bootstrap；不使用分数挑选重试答案。旧未知请求及费用预留保留。',
            '本次没有自动替换主线程的任何冻结配置。','', '## 提示词改动的配对区间','']
    for arm in list(ARMS)[:3]:
        a=results['arms'][arm]; ci=a['paired_stratified_bootstrap95']
        if ci: lines.append(f"- {arm}: {100*a['delta_F1']:+.2f}个百分点；95%区间 [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]。")
    details=rows(OUT/'PAIRED_DETAILS_LATEST.jsonl')
    lines+=['','## 变化实例','','下列为按F1差值选出的实例，不能由文字相似性直接判定语义正误。']
    for arm in list(ARMS)[:3]:
        vv=[r for r in details if r['arm']==arm]
        for direction,examples in [('上升',sorted([r for r in vv if r['delta']>0],key=lambda r:-r['delta'])[:3]),('下降',sorted([r for r in vv if r['delta']<0],key=lambda r:r['delta'])[:3])]:
            for r in examples:
                lines+=['',f"### {arm} / {direction} / {r['id']}",f"问题：{r['question']}",f"标准答案：{r['gold']}",f"B0：{r['B0_prediction']}（F1={r['B0_f1']:.4f}）",f"本组：{r['prediction']}（F1={r['f1']:.4f}）"]
    b=results['budget']
    lines+=['','## 开销','',f"本次新增已返回用量估算 ¥{b['experiment_returned_cny']:.4f}；未核销预留 ¥{b['experiment_reserve_cny']:.4f}；新增提交 {b['experiment_submissions']} 次。实际账单未核对。",f"累计账本占用 ¥{b['global_returned_cny']+b['global_reserve_cny']:.4f}/300。",'',
            '按组费用依据原始账本核对，已修正运行中报告的Windows路径分组显示问题；总开销和实验调用没有改变。']
    (OUT/'REPORT_FINAL.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
    (OUT/'REPORT_LATEST.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
    log('offline_final_audit_and_accounting',all_complete=results['all_complete'],costs=costs,api_calls=0)
    print(json.dumps({'completed':results['all_complete'],'audit':str(OUT/'FINAL_AUDIT.frozen.json'),'report':str(OUT/'REPORT_FINAL.md')},ensure_ascii=False),flush=True)


if __name__=='__main__': finalize()
