"""Native paired answer evaluation of original24, new36 and combined60."""

import json
import random
import sqlite3

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.multigrain_prompt_v2_20261017.react_comparison import logical_usage
from .prepare import ROOT, OUTPUT, HERE, OUT, ORIGINAL_OUT, sha, rows, load, write


def avg(values):
    return sum(values)/len(values) if values else None


def paired(records,left,right):
    valid=[r for r in records if all(r['methods'][m].get('f1') is not None for m in (left,right))]
    delta=[r['methods'][left]['f1']-r['methods'][right]['f1'] for r in valid]
    by_ds={ds:[d for r,d in zip(valid,delta) if r['dataset']==ds] for ds in sorted({r['dataset'] for r in valid})}
    samples=[];rng=random.Random(20261017)
    if valid:
        for _ in range(10000):
            samples.append(sum(sum(rng.choices(v,k=len(v))) for v in by_ds.values())/len(valid))
        samples.sort()
    return {'pairs':len(valid),'planned':len(records),'mean_difference':avg(delta),
        'left_mean_f1':avg([r['methods'][left]['f1'] for r in valid]),
        'right_mean_f1':avg([r['methods'][right]['f1'] for r in valid]),'gains':sum(d>0 for d in delta),
        'ties':sum(d==0 for d in delta),'losses':sum(d<0 for d in delta),
        'gain_qids':[r['qid'] for r,d in zip(valid,delta) if d>0],'loss_qids':[r['qid'] for r,d in zip(valid,delta) if d<0],
        'paired_bootstrap95_conditional_on_sample':[samples[249],samples[9749]] if samples else None}


def summary(records):
    methods={}
    for name in ('single','dual','react'):
        valid=[r['methods'][name] for r in records if r['methods'][name].get('f1') is not None]
        known=[r for r in valid if r.get('complete_usage')]
        total=sum(r['f1'] for r in valid);n=len(records);missing=n-len(valid)
        methods[name]={'planned':n,'available':len(valid),'missing':missing,'mean_f1_available':avg([r['f1'] for r in valid]),
            'mean_em_available':avg([r['em'] for r in valid]),'f1_all_planned_bounds':[total/n,(total+missing)/n] if n else None,
            'usage_available':len(known),'mean_logical_api_calls':avg([r['logical_calls'] for r in known]),
            'mean_logical_standalone_cny':avg([r['logical_standalone_list_price_cny'] for r in known])}
    common=[r for r in records if all(r['methods'][m].get('f1') is not None for m in ('single','dual','react'))]
    return {'planned':len(records),'methods':methods,
        'common_all_methods':{'available':len(common),
            'mean_f1':{m:avg([r['methods'][m]['f1'] for r in common]) for m in ('single','dual','react')},
            'mean_em':{m:avg([r['methods'][m]['em'] for r in common]) for m in ('single','dual','react')}},
        'dual_minus_single':paired(records,'dual','single'),'dual_minus_react':paired(records,'dual','react'),
        'all_planned_mean_difference_bounds':{m:[methods['dual']['f1_all_planned_bounds'][0]-methods[m]['f1_all_planned_bounds'][1],
            methods['dual']['f1_all_planned_bounds'][1]-methods[m]['f1_all_planned_bounds'][0]] for m in ('single','react')} if records else {}}


def main():
    cases=rows(HERE/'cases.frozen.jsonl');protocol=load(HERE/'protocol.json')
    baseline={(r['dataset'],r['qid']):r for r in load(HERE/'react_predictions.frozen.json')}
    terminal_path=OUT/'KNOWN_TERMINAL_FAILURES.json'
    terminal={(r['dataset'],r['qid'],r['condition']):r for r in load(terminal_path)} if terminal_path.exists() else {}
    details=[]
    for ds in ('hotpotqa','2wiki','musique'):
        ref_path=ROOT/f'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{ds}/formal.references.local.jsonl'
        refs={r['qid']:r for r in rows(ref_path)};score=metric(ds)
        for c in [r for r in cases if r['dataset']==ds]:
            row={k:c[k] for k in ('qid','dataset','type','question','evaluation_group')}
            row['gold_answer']=refs[c['qid']]['answer'];row['methods']={}
            for name,label in (('single','single_original'),('dual','dual_original_rewritten')):
                p=OUT/'cases'/ds/label/(c['qid']+'.json')
                if not p.exists():
                    row['methods'][name]={'status':'incomplete','f1':None,'em':None,
                        'terminal_failure':terminal.get((ds,c['qid'],label))};continue
                r=load(p);ids=[]
                for node in r['nodes']:
                    ids += [node[k] for k in ('rewrite_request_id','answer_request_id','sentence_format_request_id') if node.get(k)]
                ids.append(r['request_id'])
                row['methods'][name]={'status':r['status'],'prediction':r['prediction'],**score(r['prediction'],refs[c['qid']]),
                    **logical_usage(ids,(OUT if (OUT/'api'/ds/'checkpoint.sqlite3').exists() else ORIGINAL_OUT)/'api'/ds/'checkpoint.sqlite3'),
                    'changed_rewrites':sum(n.get('rewritten_retrieval') is not None for n in r['nodes'])}
                # Goal equals original query; full original Top10 must remain in final context.
                assert r['nodes'][-1]['request']==c['question']
                assert all(d['block'] in r['final_context'] for d in r['nodes'][-1]['original_retrieval']['docs'])
            b=baseline[(ds,c['qid'])];r=b['record']
            row['methods']['react']={'status':r['status'],'prediction':r['prediction'],**score(r['prediction'],refs[c['qid']]),
                **logical_usage(b['logical_request_ids'],ROOT/b['source']/'checkpoint.sqlite3')}
            details.append(row)
    all_summary=summary(details);complete=all(v['missing']==0 for v in all_summary['methods'].values())
    missing=[{'dataset':r['dataset'],'qid':r['qid'],'method':m,'known_terminal_failure':v.get('terminal_failure') is not None}
        for r in details for m,v in r['methods'].items() if v.get('f1') is None]
    groups={g:summary([r for r in details if r['evaluation_group']==g]) for g in ('original24','new36')}
    original=load(ORIGINAL_OUT/'RESULTS.json')['overall']['conditions']
    reproduction={m:abs(groups['original24']['methods'][m]['mean_f1_available']-original[label]['mean_f1_available'])<1e-12 for m,label in (('single','single_original'),('dual','dual_original_rewritten'))}
    assert all(reproduction.values())
    with sqlite3.connect((OUTPUT/'budget.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        q='SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges'
        actual,reserved=db.execute(q+' WHERE namespace LIKE ?',(str(OUT.relative_to(OUTPUT))+'%',)).fetchone()
        global_actual,global_reserved=db.execute(q).fetchone()
    result={'complete':complete,'all_runnable_generation_completed':all(r['known_terminal_failure'] for r in missing),
        'missing_outputs':missing,'planned':60,'overall':all_summary,'by_evaluation_group':groups,
        'by_dataset':{ds:summary([r for r in details if r['dataset']==ds]) for ds in ('hotpotqa','2wiki','musique')},
        'new36_by_dataset':{ds:summary([r for r in details if r['dataset']==ds and r['evaluation_group']=='new36']) for ds in ('hotpotqa','2wiki','musique')},
        'original24_reproduced':reproduction,'known_terminal_failures':list(terminal.values()),
        'new_api_usage_list_price_cny':actual,'unresolved_reservations_cny':reserved,
        'global_api_usage_list_price_cny':global_actual,'global_reservations_cny':global_reserved,
        'frozen_checks':{p:sha(ROOT/p)==v for p,v in protocol['frozen_files'].items()},
        'limits':['manual query-only reference plans under ideal-correctness assumption, not automatic4B evaluation',
            'original24 purposive; new36 sampled by type and existing ReAct availability, not answer score; neither combined60 nor bootstrap establishes full-benchmark SOTA',
            'bootstrap resamples paired differences within datasets; intervals describe this fixed cohort, not representative population uncertainty',
            'ReAct is frozen local shared-resource adaptation k10,max_search5; prompts and total search/call budgets differ']}
    write(OUT/'RESULTS.json',result)
    (OUT/'CASE_RESULTS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details),encoding='utf-8')
    pct=lambda x:'未完成' if x is None else f'{100*x:.2f}'
    text=['首轮双路收益60题扩样复验','原24题保留原输出，新增36题，每数据集20题；首轮配置全部冻结，未混入后续优化。',
        '原24题是成功/失败定向选取；新36题按类型和ReAct可用性抽样，不按答案分数选题。新36单独报告。',
        '下表F1统一使用三组共同有效题；完整计划分母、各组缺失及全计划F1范围保留在RESULTS.json。',
        '范围\t计划题数\t共同有效\t单路F1%\t双路F1%\tReActF1%\t双路相对单路上升/持平/下降']
    for name,value in [('原24',groups['original24']),('新36',groups['new36']),('全部60',all_summary),*result['by_dataset'].items()]:
        p=value['dual_minus_single']
        text.append('\t'.join([name,str(value['planned']),str(value['common_all_methods']['available']),*[pct(value['common_all_methods']['mean_f1'][m]) for m in ('single','dual','react')],f"{p['gains']}/{p['ties']}/{p['losses']}"]))
    text += ['', '全部计划答案齐全='+str(complete),'全部可运行题已完成='+str(result['all_runnable_generation_completed']),
        '缺失记录='+json.dumps(missing,ensure_ascii=False),'原24完全复现='+str(reproduction),'新增用量估算CNY='+str(actual),'未结预留CNY='+str(reserved),
        '', '配对比较及条件置信区间：',json.dumps({name:value['dual_minus_single'] for name,value in [('original24',groups['original24']),('new36',groups['new36']),('all60',all_summary)]},ensure_ascii=False),
        '', '限制：']+result['limits']+['','逐题：']
    for r in details:
        text += ['',r['dataset']+'/'+r['qid']+' / '+r['evaluation_group'],r['question'],'Gold（仅离线）: '+str(r['gold_answer'])]
        text += [m+': '+json.dumps(r['methods'][m],ensure_ascii=False) for m in ('single','dual','react')]
    (OUT/'REPORT.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'complete':complete,'overall':all_summary,'original24':groups['original24'],'new36':groups['new36'],'cost':actual},ensure_ascii=False))


if __name__=='__main__':
    main()
