"""Finalize only after every frozen condition and required case review exists."""
from collections import Counter
from . import run as r
from .analyze import analyze


def finalize():
    x=analyze()
    assert x['attempted']==270 and set(x['statuses'])<= {'success','algorithm_failure'}
    assert x['unknown_reserved_CNY']==0
    assert (r.OUT/'QUALITATIVE_SUMMARY.zh.md').exists()
    review=r.read(r.OUT/'CASE_REVIEW.local.json')
    cases=r.rows(r.OUT/'REVIEW_CASES.local.jsonl')
    wanted={q['id'] for q in cases}
    assert wanted<=set(review['cases']),sorted(wanted-set(review['cases']))
    details=r.rows(r.OUT/'DETAILS.local.jsonl')
    line=['# r04：历史错误修正后的90题阶段验证','',
        '固定90题，每集30题；C严格复用上一轮相同请求的完整回执，S/T为本轮新输出。分解、普通QA、改写和检索材料均冻结，旧final不变。G_r03是上一轮r03 goal＋旧final的历史参照。所有数据均为反复暴露的开发题，不能当作独立测试或新完整RAG成绩。','',
        '## 改了什么','',
        '- S：只把父版本goal的一句事实句契约改为简短的事实—必要连接—答案段落。',
        '- T：S加一个目标理解段，区分识别线索与被问关系，限制凭空添加条件和纠错。',
        '- 撤回r03普通QA和final作为候选组合的默认选择。本轮没有新生成普通QA或执行动态传播；生产文件此前未被r03覆盖，本轮也未改。',
        '- 两组没有新增示例，避免把few-shot效果混入本轮；T−S评估整个规则段，不能分离其中单句的效应。','',
        '## 完整结果','',
        '|数据集|C旧对照 F1/EM|旧r03 G F1/EM|S简短组合 F1/EM|T增加目标约束 F1/EM|',
        '|---|---:|---:|---:|---:|']
    for ds in ('all',)+r.base.DATASETS:
        row=x['by_dataset_arm'][ds]
        line.append('|'+ds+'|'+'|'.join(f"{row[a]['F1']*100:.2f}/{row[a]['EM']*100:.2f}" for a in ('C','G_r03','S','T'))+'|')
    line+=['','|配对对照|F1差（百分点）|改善/下降/不变|描述性95%区间（百分点）|','|---|---:|---:|---|']
    for pair,v in x['paired']['all'].items():
        low,high=v['descriptive95_CI']
        line.append(f"|{pair}|{v['F1_delta']*100:+.2f}|{v['improved']}/{v['harmed']}/{v['tied']}|[{low*100:+.2f}, {high*100:+.2f}]|")
    line+=['','分层配对bootstrap10000次，seed20261023；不校正历次开发选择或多重比较，也没有估计新生成的重复采样波动。C是严格历史回放，不是新同期重复对照。','',
        '## 逐例复核','',
        f"复核集合共{len(cases)}题，覆盖全部{x['new_final_changed_cases']}个C/S/T最终文本变化案例及28个上一轮已记录问题。完整goal、final与材料保存在REVIEW_CASES.local.jsonl，逐例观察在CASE_REVIEW.local.json；没有把无分数变化的逻辑错误略去。",'',
        '详见本目录QUALITATIVE_SUMMARY.zh.md。原生分数完全保留，不做数字、标点或实体别名补丁。个别正确答案仍附带无依据解释，因此恢复F1不自动视为恢复正确推理。','',
        '## 开销与完整性','',
        '|条件|输入tokens/题|输出tokens/题|goal词数/题|goal＋final逻辑估算元/题|','|---|---:|---:|---:|---:|']
    for a in ('C','G_r03','S','T'):
        v=x['by_dataset_arm']['all'][a]
        line.append(f"|{a}|{v['mean_input_tokens']:.1f}|{v['mean_output_tokens']:.1f}|{v['mean_goal_words']:.1f}|{v['mean_logical_CNY']:.6f}|")
    ratios={a:x['by_dataset_arm']['all'][a]['mean_logical_CNY']/x['by_dataset_arm']['all']['C']['mean_logical_CNY'] for a in ('S','T')}
    line+=['',f"本轮新增实际usage估算费用 **{x['actual_new_CNY']:.6f}元**；未结未知预留0。S/T逻辑费用分别为C的{ratios['S']:.3f}/{ratios['T']:.3f}倍。历史回放费用不重复计入新增支出。金额按冻结单价估算，不是正式供应商账单，也不是新完整流程费用或耗时。",'',
        f"270/270条件均已完成尝试，终态为{x['statuses']}。所有请求与冻结渲染一致，{x['receipt_audit']['exact_payloads_checked']}个阶段payload完成逐条复核，成功回执均stop并有完整usage；C90题输出与父版本逐字相同。计费总额与唯一付费回执核对一致，任何算法失败都保留计零而不重跑。",'',
        '初次受限网络尝试在DNS/连接准备阶段失败，未提交模型请求；随后使用已授权的联网执行，前6条件通过接口核验后完成余下条件。新运行用SQLite和追加事件作为进度来源，避免旧STATUS原子替换文件占用造成的中断。','',
        '## 结论与范围','']
    c=x['by_dataset_arm']['all']['C']['F1'];best=max(('S','T'),key=lambda a:x['by_dataset_arm']['all'][a]['F1'])
    if x['by_dataset_arm']['all'][best]['F1']<=c:
        line+=['两种修正版仍没有超过本轮旧对照，保留旧提示词，不将S/T晋级完整流程或正式评估。部分r03退步被恢复，不等于完成了提升P02的目标。']
    else:
        line+=['存在高于旧对照的开发集点估计，但当前只完成冻结reader的阶段验证。还需判断配对得失、解释可靠性和完整流程传播后再考虑采用；不自动替换生产或恢复正式评估。']
    line+=['','更早实验的复核与本轮约束理由见实验目录ERROR_REVIEW.zh.md；普通子回答的新增四例语义复核见ORDINARY_REVIEW_ADDENDUM.zh.md。下一步建议仅作为研究方向写入QUALITATIVE_SUMMARY，不自动追加新模型请求。',
        '主评估、正式计时、基线配置、远端GPU作业及发布均未恢复。所有原始实验、暂停标记和受保护核心文件散列仍匹配。','']
    (r.OUT/'REPORT.zh.md').write_text('\n'.join(line),encoding='utf8')
    completion=dict(experiment=r.HERE.name,utc=r.utc(),status='complete_reviewed_not_promoted',conditions=270,
        cases_reviewed=len(cases),actual_new_CNY=x['actual_new_CNY'],unknown_reserved_CNY=0,protocol_sha256=r.sha(r.HERE/'PROTOCOL.frozen.json'),
        source_checks='passed',main_evaluation_resumed=False,production_changed=False,
        report_sha256=r.sha(r.OUT/'REPORT.zh.md'))
    r.save(r.OUT/'COMPLETION.json',completion);r.save(r.REVIEW/'COMPLETION.json',completion)
    for name in ('ERROR_REVIEW.zh.md','ORDINARY_REVIEW_ADDENDUM.zh.md'):
        (r.REVIEW/name).write_bytes((r.HERE/name).read_bytes())
    (r.REVIEW/'REPORT.zh.md').write_bytes((r.OUT/'REPORT.zh.md').read_bytes())
    decision=dict(id=r.DECISION+'_COMPLETED',**completion)
    if not any(x.get('id')==decision['id'] for x in r.rows(r.base.PLAN/'DECISIONS.jsonl')):
        with (r.base.PLAN/'DECISIONS.jsonl').open('a',encoding='utf8') as f:
            import json
            f.write(json.dumps(decision,ensure_ascii=False)+'\n')
    return completion


if __name__=='__main__':
    import json
    print(json.dumps(finalize(),ensure_ascii=False))
