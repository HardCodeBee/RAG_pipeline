"""Native-score analysis and exact changed-case export. Does not alter inference."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
from . import run


def interval(groups,settings):
    rng=np.random.default_rng(settings['seed']);rep=settings['replicates']
    if not groups:return None
    accum=np.zeros(rep);total=0
    for values in groups:
        x=np.asarray(values,float)
        accum+=x[rng.integers(0,len(x),size=(rep,len(x)))].sum(axis=1);total+=len(x)
    values=accum/total
    return [float(x) for x in np.quantile(values,[.025,.975])]


def analyze():
    results=run.report();plan=run.read(run.HERE/'ANALYSIS.frozen.json')
    details=run.rows(run.OUT/'DETAILS.local.jsonl')
    goals=[r for r in details if r['suite']=='goal'];index={(x['id'],x['arm']):x for x in goals}
    inputs=run.rows(run.HERE/'INPUTS90.frozen.jsonl');references={r['id']:r for r in run.rows(run.HERE/'REFERENCES90.local.frozen.jsonl')}
    for ds in ('all',)+run.DATASETS:
        for a,b in (('C','G'),('C','GF'),('G','GF')):
            groups=[]
            for dataset in run.DATASETS if ds=='all' else (ds,):
                ids=[r['id'] for r in inputs if r['dataset']==dataset]
                if not all((i,arm) in index and index[(i,arm)]['metrics']['f1'] is not None for i in ids for arm in (a,b)):
                    groups=[];break
                groups.append([index[(i,b)]['metrics']['f1']-index[(i,a)]['metrics']['f1'] for i in ids])
            results['paired'][ds][a+'->'+b]['descriptive_paired95_CI']=interval(groups,plan['paired_bootstrap'])
    changed=[]
    for r in inputs:
        found={a:index[(r['id'],a)] for a in run.GOAL_ARMS if (r['id'],a) in index}
        if len(found)!=3:continue
        if len({x.get('prediction') for x in found.values()})>1:
            changed.append(dict(id=r['id'],dataset=r['dataset'],question=r['question'],
              reference=references[r['id']],ordinary_nodes=r['ordinary_nodes'],
              goal_reader_packet=r['goal_reader_packet'],root_packet=r['final_root_packet'],
              arms={a:{k:x.get(k) for k in ('status','prediction','goal_raw','goal_text','metrics','input_tokens','output_tokens','logical_stage_CNY')} for a,x in found.items()}))
    ordinary=[x for x in details if x['suite']=='ordinary'];oi={(x['id'],x['arm']):x for x in ordinary}
    nodes=run.rows(run.HERE/'ORDINARY176.frozen.jsonl');ordinary_changes=[]
    for r in nodes:
        if not all((r['id'],arm) in oi for arm in run.ORDINARY_ARMS):continue
        a,b=[oi[(r['id'],arm)] for arm in run.ORDINARY_ARMS]
        if a.get('answer')!=b.get('answer') or a['status']!=b['status']:
            ordinary_changes.append(dict(id=r['id'],dataset=r['dataset'],case_id=r['case_id'],
              current_question=r['current_question'],original_question=r['original_question'],
              accepted_rewrite=r['accepted_rewrite'],history=r['history'],reader_packet=r['reader_packet'],
              O0={k:a.get(k) for k in ('status','raw_output','answer')},
              O1={k:b.get(k) for k in ('status','raw_output','support','answer')}))
    results['ordinary']={}
    for arm in run.ORDINARY_ARMS:
        values=[x for x in ordinary if x['arm']==arm];success=[x for x in values if x['status']=='success']
        billed=[x for x in values if x['input_tokens'] or x['output_tokens']]
        results['ordinary'][arm]=dict(planned=176,terminal=len(values),statuses=dict(Counter(x['status'] for x in values)),
          mean_input_tokens=sum(x['input_tokens'] for x in success)/len(success) if success else None,
          mean_output_tokens=sum(x['output_tokens'] for x in success)/len(success) if success else None,
          mean_logical_CNY=sum(x['logical_stage_CNY'] for x in success)/len(success) if success else None,
          billed_conditions_including_failures=len(billed),
          mean_input_tokens_all_billed=sum(x['input_tokens'] for x in billed)/len(billed) if billed else None,
          mean_output_tokens_all_billed=sum(x['output_tokens'] for x in billed)/len(billed) if billed else None,
          mean_logical_CNY_all_billed=sum(x['logical_stage_CNY'] for x in billed)/len(billed) if billed else None,
          manual_factual_accuracy_not_estimated=True)
    results['changed_final_text_cases']=len(changed);results['changed_ordinary_answer_nodes']=len(ordinary_changes)
    # Give expense context using the same old prefix, explicitly not a new full run.
    results['cost_comparison']={}
    for arm in ('G','GF'):
        paired=[(index[(r['id'],'C')],index[(r['id'],arm)],r) for r in inputs if all((r['id'],a) in index and index[(r['id'],a)]['status']=='success' for a in ('C',arm))]
        if not paired:continue
        c=sum(a['logical_stage_CNY'] for a,b,r in paired);v=sum(b['logical_stage_CNY'] for a,b,r in paired)
        pc=sum(run.cost(r['frozen_prefix_usage']) for a,b,r in paired)
        results['cost_comparison'][arm]=dict(paired=len(paired),goal_final_cost_ratio_vs_C=v/c,
          frozen_prefix_plus_new_stage_ratio_vs_C=(pc+v)/(pc+c),
          note='Only goal/final regenerated; prefix costs copied for accounting illustration, not newly measured full-flow costs')
    run.jsonl(run.OUT/'CHANGED_FINAL_CASES.local.jsonl',changed)
    run.jsonl(run.OUT/'CHANGED_ORDINARY_NODES.local.jsonl',ordinary_changes)
    run.atomic_json(run.OUT/'ANALYSIS.json',results);run.atomic_json(run.REVIEW/'RESULTS.json',results)
    # Compact complete texts are useful for human review; retrieval stays in JSONL.
    lines=['# 所有最终文本变化：阶段试验，事后复核','',
      '当前显示的是已经完成的三臂配对。没有把答案正确性当作选择或重试条件；标准答案仅在分析阶段读入。原始检索包见同目录JSONL。','']
    for case in changed:
        lines+=['## '+case['id'],'',case['question'],'']
        for arm,x in case['arms'].items():
            lines+=['### '+arm,'','Goal：',x.get('goal_text') or '(无成功goal输出)','',
              'Final：'+str(x.get('prediction')),'',json.dumps(x['metrics'],ensure_ascii=False),'']
    (run.OUT/'CHANGED_FINAL_CASES.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status=results['status'],terminal=results['terminal_conditions'],
      goal=results['by_dataset_arm']['all'],ordinary=results['ordinary'],cost=results['cost_comparison'],
      actual_CNY=results['actual_CNY'],unknown_reserved_CNY=results['unresolved_reserved_CNY']),ensure_ascii=True))
    return results


if __name__=='__main__':analyze()
