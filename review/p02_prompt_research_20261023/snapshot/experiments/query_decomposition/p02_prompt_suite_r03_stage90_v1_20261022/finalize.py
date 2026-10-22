"""Finish the isolated trial's local reports. No generation or publication."""
from collections import Counter
from contextlib import closing
from . import run,analyze,audit,diagnostics
import json


def main():
    with closing(run.ro(run.OUT/'checkpoint.sqlite3')) as db:
        outcomes=[json.loads(x) for x, in db.execute("SELECT value FROM data WHERE kind='outcome'")]
    assert len(outcomes)==622, 'Finish the frozen condition list first'
    assert not any(r['status']=='technical_held' for r in outcomes), 'Report held outcomes separately before finalization'
    run.check()
    audit.main()
    results=analyze.analyze()
    diagnostics.main()
    diag=run.read(run.OUT/'DIAGNOSTICS.json')
    assert results['unresolved_reserved_CNY']==0
    assert diag['requests']['statuses']=={'success':sum(diag['requests']['statuses'].values())}
    assert not diag['changed_case_review']['missing']
    assert abs(diag['requests']['unique_receipt_cost_CNY']-results['actual_CNY'])<1e-8
    sample_plan=run.read(run.HERE/'QUALITATIVE_REVIEW_PLAN.frozen.json')
    sample_ids=set(sum(sample_plan['ordinary_sample'].values(),[]))
    notes=run.read(run.OUT/'ORDINARY_SAMPLE_REVIEW.local.json')
    assert set(notes['cases'])==sample_ids
    nodes={r['id']:r for r in run.rows(run.HERE/'ORDINARY176.frozen.jsonl')}
    index={(r['id'],r['arm']):r for r in outcomes}
    sample=[]
    for ident in sorted(sample_ids):
        sample.append(dict(id=ident,input=nodes[ident],review=notes['cases'][ident],
          arms={a:index[(ident,a)] for a in run.ORDINARY_ARMS}))
    run.jsonl(run.OUT/'ORDINARY_SAMPLE18.local.jsonl',sample)
    failure_rows=[]
    with closing(run.ro(run.OUT/'checkpoint.sqlite3')) as db:
        for r in outcomes:
            if r['status']=='success':continue
            receipts=[]
            for stage in r['stages']:
                raw=db.execute('SELECT result FROM requests WHERE id=?',(stage['request_id'],)).fetchone()
                rec=json.loads(raw[0]) if raw and raw[0] else {}
                receipts.append(dict(stage=stage['stage'],request_id=stage['request_id'],
                  answer=rec.get('answer'),finish_reason=rec.get('finish_reason'),usage=rec.get('usage')))
            failure_rows.append(dict(outcome=r,receipts=receipts))
    run.jsonl(run.OUT/'FAILURES.local.jsonl',failure_rows)
    ordinary=results['ordinary'];old,new=[ordinary[a] for a in run.ORDINARY_ARMS]
    ratio=new['mean_logical_CNY_all_billed']/old['mean_logical_CNY_all_billed']
    text=(run.OUT/'GOAL90_FINDINGS.zh.md').read_text(encoding='utf8')
    text=text.replace('# r03：固定90题的goal/final阶段结果','# r03提示词阶段试验：完整报告（90题＋176对子回答）',1)
    text=text.replace('普通子问题的176对试验另行完成后补充整体试验报告。其新回答本轮没有向后传播；是否改善动态检索、后续绑定和最终F1仍未由本轮验证。主评估保持暂停，生产提示词不变。',
      '普通子问题的176对试验已完成，见下文。其新回答本轮没有向后传播；是否改善动态检索、后续绑定和最终F1仍未由本轮验证。主评估保持暂停，生产提示词不变。')
    text+='\n\n## 普通子回答176对\n\n'
    text+='所有176个节点均完成两臂尝试；沿用同一冻结历史、当前子问题、改写文本和reader包。O0使用旧指令，O1使用r03的Support/Answer，后者按设计只允许Answer向后传递，本轮没有实际传播到后续节点。\n\n'
    text+='|指标|O0旧版|O1新版|\n|---|---:|---:|\n'
    text+=f"|完成并通过接口要求|{old['statuses'].get('success',0)}/176|{new['statuses'].get('success',0)}/176|\n"
    text+=f"|算法失败，保留不重采样|{old['statuses'].get('algorithm_failure',0)}|{new['statuses'].get('algorithm_failure',0)}|\n"
    for title,key,precision in [('平均输入tokens','mean_input_tokens_all_billed',2),('平均输出tokens','mean_output_tokens_all_billed',2),('平均估算费用，元','mean_logical_CNY_all_billed',6)]:
        text+=f"|{title}|{old[key]:.{precision}f}|{new[key]:.{precision}f}|\n"
    text+=f'\n费用和tokens包含已计费的失败输出；O1单节点费用为O0的{ratio:.3f}倍。接口成功不表示事实正确。'
    text+=f"全部节点中有{results['changed_ordinary_answer_nodes']}个的Answer文本或成功状态发生变化；这个数量包含标点、语序和补充关系描述，不能等同于改善题数。\n\n"
    text+='固定抽查18个节点（每集6个，普通推理开始前按hash选定）已逐条由本助手对照本地材料复核，记录在ORDINARY_SAMPLE18.local.jsonl和ORDINARY_SAMPLE_REVIEW.local.json。没有调用外部评审模型，也没有拿原query标准答案给子问题打分。各例的判断范围和材料冲突均保留；不据此报告176节点总体事实准确率。\n\n'
    categories=dict(Counter(r['category'] for r in notes['cases'].values()))
    assert all(c.startswith('same_') for c in categories)
    unresolved=categories.get('same_answer_unresolved_source_conflict',0)
    text+=f'18个预选节点的新旧主要答案均相同，措辞和补充信息有所不同。其中{18-unresolved}例的主要局部事实与冻结材料相符；{unresolved}例是Sunrise Avenue成立年份的材料冲突，新版Support没有解决冲突。此样本未观察到主要答案修复，也不能证明全部176条都没有改善。同一主要答案不代表Support的每个额外断言都经过了独立事实核验。\n\n'
    text+='另外保留3个事后发现的典型案例，明确不计入预选样本频率：问Fernand Rivers个人职业却提前做两人的职业交集；问国家却加入原问题所问的总统；Support过长使Le Reve后续子问题的Answer触及256 tokens硬上限而截断。前两条展示局部任务越界，但其句子仍含所问的局部事实，不能仅因越界就判为事实错误。最后一条按冻结stop＋usage规则计为算法失败，原始响应保存在FAILURES.local.jsonl。\n\n'
    text+='## 执行与复核\n\n'
    text+=f"共622条条件均已尝试完毕，状态为{dict(Counter(r['status'] for r in outcomes))}。本轮总实际估算支出**{results['actual_CNY']:.6f}元**，无待结算未知预留。以上金额按冻结配置的输入/输出单价估算，非供应商正式账单。\n\n"
    text+='最终请求与接口审计通过：实际请求与冻结渲染一致，G/GF共用90条同一goal输出，完整三字段没有丢失，成功条件均正常stop并有usage。所有冻结输入、生成代码、模板、生产核心文件及主评暂停标记的散列仍匹配。13:24 UTC曾因STATUS原子替换遇到Windows文件占用退出；270条goal结果已先入库，从断点继续后未重复已完成请求，详见EXECUTION_NOTE.md。\n\n'
    text+='本轮没有新rewrite推理、动态检索、分解器训练、完整RAG传播或正式计时补测；没有恢复主评估，没有修改生产提示词，也没有推送或发布。保留全部成功、退步和失败记录。\n\n'
    text+='## 本地记录索引\n\n'
    for name,description in [('ANALYSIS.json','完整分集原生分数、配对差值、tokens和费用'),('RECEIPT_INTERFACE_AUDIT.json','请求与阶段接口复核'),('CHANGED_FINAL_CASES.md','28个最终文本变化案例的完整goal/final输出'),('CHANGED_FINAL_CASES.local.jsonl','同批案例的冻结材料'),('CASE_REVIEW.local.json','逐例归因观察和边界'),('CHANGED_ORDINARY_NODES.local.jsonl','所有变化的普通子回答'),('ORDINARY_SAMPLE18.local.jsonl','预选18个节点的完整输入、输出及复核'),('EXTRA_ORDINARY_CASES.local.json','事后补充案例，单独标记'),('FAILURES.local.jsonl','失败条件及原始截断输出'),('checkpoint.sqlite3','完整模型请求和响应断点')]:
        text+=f'- `{name}`：{description}。\n'
    (run.OUT/'REPORT.zh.md').write_text(text,encoding='utf8')
    (run.REVIEW/'REPORT.md').write_text(text+'\n\n本地完整材料目录：`'+str(run.OUT)+'`。\n',encoding='utf8')
    completion=dict(utc=run.utc(),status='completed_with_algorithm_failures' if failure_rows else 'completed',
      conditions=622,goal_queries=90,ordinary_pairs=176,statuses=dict(Counter(r['status'] for r in outcomes)),
      actual_estimated_CNY=results['actual_CNY'],unresolved_reserved_CNY=0,
      all_goal_changed_cases_reviewed=len(run.read(run.OUT/'CASE_REVIEW.local.json')['cases']),
      fixed_ordinary_sample_reviewed=len(sample),protected_pins_unchanged=True,
      promoted=False,new_rewrite_tested=False,dynamic_full_RAG_tested=False,main_evaluation_resumed=False,
      report=str((run.OUT/'REPORT.zh.md').relative_to(run.ROOT)))
    run.atomic_json(run.OUT/'COMPLETION.json',completion)
    run.atomic_json(run.REVIEW/'COMPLETION.json',completion)
    print(json.dumps(completion,ensure_ascii=True))


if __name__=='__main__':main()
