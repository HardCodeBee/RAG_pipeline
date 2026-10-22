"""Summarize the completed quick90 without model calls or source mutations."""
from collections import Counter
from contextlib import closing
from pathlib import Path
import json
import random
import sqlite3

from . import run, prompts


def main():
    run.check()
    report = run.read(run.OUT / 'REPORT.json')
    assert report['status'] == 'completed'
    details = run.rows(run.OUT / 'PAIRED_DETAILS.jsonl')
    controls = {r['id']:r for r in run.rows(run.OUT/'control90.jsonl')}
    baseline_tokens = Counter()
    candidate_tokens = Counter()
    length_events, statuses, changed = [], Counter(), []
    audit = Counter()
    for ds in run.common.DATASETS:
        directory = run.OUT/'candidate'/ds
        with closing(run.ro(directory/'checkpoint.sqlite3')) as db:
            results = {cid:json.loads(raw) for cid,raw in db.execute("SELECT key,value FROM data WHERE kind='prediction'")}
            assert len(results) == 30
            for stage,rid,raw in db.execute("SELECT stage,id,result FROM requests WHERE status='success'"):
                value = json.loads(raw)
                if value.get('finish_reason') == 'length':
                    length_events.append({'dataset':ds,'stage':stage,'request_id':rid,'output_tokens':value.get('usage',{}).get('completion_tokens')})
        logical = {(r['id'],r['stage']):r for r in run.rows(directory/'trace.jsonl') if r['event']=='model_call'}
        for value in logical.values():
            u = value.get('usage') or {}
            candidate_tokens.update(input=u.get('prompt_tokens',0),output=u.get('completion_tokens',0))
        for cid,result in results.items():
            statuses[result['status']] += 1
            for call in controls[cid]['calls']:
                u = call['usage']
                baseline_tokens.update(input=u['prompt_tokens'],output=u['completion_tokens'])
            if result['status'] not in {'success','success_truncated'}:
                continue
            for node in result['nodes']:
                expected = prompts.child(result['question'],node['step_id'],node['request'],node['prior_complete_answers'],run.common.baseline.passages(node['reader_packet']),[],role=node['role'])
                assert expected == node['child_prompt']
                audit[node['role']] += 1
            expected,context,block = run.module().render_final(result['question'],result['nodes'],result['original_retrieval'],'',[])
            assert expected == result['final_prompt'] and context == result['final_context']
            audit['final'] += 1
    rng = random.Random(20261021)
    groups = [[d['delta_F1'] for d in details if d['dataset']==ds] for ds in run.common.DATASETS]
    boot = sorted(sum(sum(rng.choices(g,k=len(g))) for g in groups)/90 for _ in range(10000))
    bounds = [boot[249],boot[9749]]
    for d in details:
        if abs(d['delta_F1']) > 1e-12:
            changed.append({**d,'question':controls[d['id']]['trajectory']['question']})
    cost = lambda t:t['input']*0.8e-6+t['output']*2e-6
    summary = {
        **report,'stratified_paired_bootstrap95_F1':bounds,'bootstrap_replicates':10000,
        'terminal_statuses':dict(statuses),'length_responses':length_events,
        'exact_actual_prompts_verified':dict(audit),
        'baseline_complete90_logical_tokens':dict(baseline_tokens),
        'candidate_observed90_logical_tokens_including_failed_prefixes':dict(candidate_tokens),
        'baseline_complete90_estimated_generation_CNY':cost(baseline_tokens),
        'candidate_observed90_estimated_generation_CNY':cost(candidate_tokens),
        'cost_scope':'Returned hosted-generation usage at the unchanged historical native rates; candidate includes failed prefixes; frozen local decomposition excluded. Cached usage is not new payment.',
        'exact_Cannot_be_determined':{
            arm:sum((d[arm+'_prediction'] or '').strip().rstrip('.').lower()=='cannot be determined' for d in details)
            for arm in ['baseline','candidate']},
        'interpretation':'Joint prompt-bundle development result, not individual-rule attribution or independent confirmation',
    }
    run.save(run.OUT/'FINAL_SUMMARY.json',summary)
    run.save(run.OUT/'CHANGED_CASES.json',sorted(changed,key=lambda d:d['delta_F1']))
    lines=['P02 三段 prompt 整体 quick90 对照','',
        '固定开发90题，各数据集30题。对照为冻结原版P02；候选同时使用用户提供的三段原文。',
        '对照为精确缓存回放；候选按实际变化的前置回答执行后续改写与检索，仅完整成功payload相同才复用。',
        '原有改写、双路合并、formatter、模型与256/128/64输出上限不变。','',
        '| 数据集 | 已评估/计划 | 原版F1 | 新版F1 | 差值百分点 | 原版EM | 新版EM | 升/平/降 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for ds in (*run.common.DATASETS,'overall'):
        m=report['metrics'][ds]
        lines.append(f"| {ds} | {m['complete']}/{m['planned']} | {100*m['baseline_F1']:.4f} | {100*m['candidate_F1']:.4f} | {100*(m['candidate_F1']-m['baseline_F1']):+.4f} | {100*m['baseline_EM']:.4f} | {100*m['candidate_EM']:.4f} | {m['improved']}/{m['tied']}/{m['harmed']} |")
    lines += ['',f'分层配对bootstrap95%差值区间：[{100*bounds[0]:+.4f}, {100*bounds[1]:+.4f}] 个百分点。',
        '终态：'+json.dumps(dict(statuses),ensure_ascii=False),
        f"length响应：{len(length_events)}；算法失败保留在90题分母中计0，最终length按原版规则保留原文评分。",
        f"原版完整90题逻辑生成tokens：{dict(baseline_tokens)}；候选90题观测tokens（含失败前缀）：{dict(candidate_tokens)}。",
        f"同价估算生成费用：原版 {cost(baseline_tokens):.6f} 元；候选 {cost(candidate_tokens):.6f} 元。缓存用量不等于新增付费。",
        f"本轮新增估算API费用 {report['new_estimated_CNY']:.6f} 元，未知预留 {report['unknown_reserved_CNY']:.6f} 元。",
        '本轮只评估三段整体，不归因于某条规则；该90题已经用于开发，不能作为独立泛化确认。',
        '冻结原版、已有实验结果和主评暂停状态保持原样；未激活候选。']
    (run.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    run.emit({'status':'completed','overall':report['metrics']['overall'],'terminal_statuses':dict(statuses),
        'length_responses':len(length_events),'new_estimated_CNY':report['new_estimated_CNY'],
        'paired95_F1':bounds,'report':str(run.OUT/'REPORT.md')})


if __name__ == '__main__':
    main()
