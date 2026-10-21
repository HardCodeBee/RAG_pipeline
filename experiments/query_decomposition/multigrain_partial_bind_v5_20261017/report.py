"""Offline native scoring; preserve denominators, regressions and resource costs."""

import json
import sqlite3

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import rows, load, write, CAMPAIGN
from experiments.query_decomposition.multigrain_prompt_v2_20261017.report import paired as observed_pairs, mean
from experiments.query_decomposition.multigrain_prompt_v2_20261017.react_comparison import logical_usage
from .prepare import HERE, OUT, ROOT, OUTPUT, sha


def paired(records, left, right):
    result = observed_pairs(records, left, right)
    total = sum(r['conditions'][left]['f1']-r['conditions'][right]['f1'] for r in records if all(r['conditions'][c].get('f1') is not None for c in (left,right)))
    missing = len(records)-result['pairs']
    result['all_planned_difference_bounds'] = [(total-missing)/len(records),(total+missing)/len(records)] if records else None
    return result


def usage(result, checkpoint):
    ids = []
    for n in result['nodes']:
        ids += [n[k] for k in ('rewrite_request_id','answer_request_id','sentence_format_request_id','reread_request_id') if n.get(k)]
    return logical_usage(ids+[result['request_id']], checkpoint)


def summarize(records, labels):
    conditions = {}
    for label in labels:
        valid = [r['conditions'][label] for r in records if r['conditions'][label].get('f1') is not None]
        conditions[label] = {'available': len(valid), 'missing': len(records)-len(valid),
            'mean_f1_available': mean([r['f1'] for r in valid]), 'f1_all_planned_bounds': [sum(r['f1'] for r in valid)/len(records),(sum(r['f1'] for r in valid)+len(records)-len(valid))/len(records)] if records else None, 'mean_em_available': mean([r['em'] for r in valid]),
            'mean_logical_calls': mean([r['logical_calls'] for r in valid if r.get('complete_usage')]),
            'mean_standalone_list_price_cny': mean([r['logical_standalone_list_price_cny'] for r in valid if r.get('complete_usage')]),
            'reread_nodes': sum(r['reread_nodes'] for r in valid)}
    pairs = {'new_minus_single':paired(records,'reuse_new','reuse_single'), 'previous_minus_single':paired(records,'reuse_previous','reuse_single'), 'new_minus_previous':paired(records,'reuse_new','reuse_previous'), 'support_new_minus_single':paired(records,'support_new','support_single'), 'support_new_minus_previous':paired(records,'support_new','support_previous'), 'final_change_new':paired(records,'support_new','reuse_new'), 'final_change_previous':paired(records,'support_previous','reuse_previous'), 'final_change_single':paired(records,'support_single','reuse_single')}
    return {'planned': len(records), 'conditions': conditions, 'pairs': pairs}


def main():
    protocol = load(HERE/'protocol.json')
    labels = list(protocol['conditions'])
    cases = rows(HERE/'cases.frozen.jsonl')
    details, react_details = [], []
    for ds in ('hotpotqa','2wiki','musique'):
        refs = {r['qid']: r for r in rows(ROOT/f'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{ds}/formal.references.local.jsonl')}
        scorer = metric(ds)
        react = CAMPAIGN/ds/'formal/b03_react_dense/k10_s5'
        ids = {}
        for e in rows(react/'trace.jsonl'):
            if e.get('event') == 'logical_api_call':
                ids.setdefault(e.get('qid'), {})[(e['stage'],e['request_id'])] = e['request_id']
        with sqlite3.connect((react/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
            for case in [c for c in cases if c['dataset'] == ds]:
                row = {k: case[k] for k in ('qid','dataset','question','evaluation_group')}
                row['gold_answer'] = refs[case['qid']]['answer']
                row['conditions'] = {}
                for label in labels:
                    path = OUT/'cases'/ds/label/(case['qid']+'.json')
                    if not path.exists():
                        row['conditions'][label] = {'status': 'incomplete','f1':None,'em':None}
                        continue
                    r = load(path)
                    row['conditions'][label] = {'status':r['status'], 'prediction':r['prediction'], **scorer(r['prediction'],refs[case['qid']]),
                        **usage(r,OUT/'api'/ds/'checkpoint.sqlite3'),
                        'reread_nodes':sum(bool(n.get('reread_request_id')) for n in r['nodes']),
                        'partial_first_pass':sum(n.get('first_pass_state',n['answer_state']) == 'partial' for n in r['nodes'])}
                raw = db.execute("SELECT value FROM data WHERE kind='prediction' AND key=?",(case['qid'],)).fetchone()
                react_prediction = json.loads(raw[0]) if raw else None
                valid = react_prediction and react_prediction.get('status') in ('success','algorithm_failure')
                rival = {'status':react_prediction.get('status') if react_prediction else 'missing','f1':None,'em':None}
                if valid:
                    rival.update({'prediction':react_prediction['prediction'], **scorer(react_prediction['prediction'],refs[case['qid']]),
                        **logical_usage(list(ids.get(case['qid'],{}).values()),react/'checkpoint.sqlite3')})
                react_details.append({**row,'react':rival})
                details.append(row)
    overall = summarize(details,labels)
    complete = all(v['missing']==0 for v in overall['conditions'].values())
    pair = overall['pairs']['support_new_minus_single']
    passed = complete and pair['mean_difference'] > 0 and pair['losses']==0
    with sqlite3.connect((OUTPUT/'budget.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        query = 'SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges'
        actual,reserved = db.execute(query+' WHERE namespace LIKE ?',(str(OUT.relative_to(OUTPUT))+'%',)).fetchone()
        total,total_reserved = db.execute(query).fetchone()
    result = {'complete':complete,'passing_gate_met':passed,'scope':protocol['scope'],'overall':overall,
        'by_evaluation_group':{g:summarize([r for r in details if r['evaluation_group']==g],labels) for g in ('fresh','regression','competitor_gap')},
        'by_dataset':{ds:summarize([r for r in details if r['dataset']==ds],labels) for ds in ('hotpotqa','2wiki','musique')},
        'new_api_usage_list_price_cny':actual,'unresolved_reservations_cny':reserved,'global_api_usage_list_price_cny':total,'global_reservations_cny':total_reserved,
        'frozen_hash_checks':{p:sha(ROOT/p)==v for p,v in protocol['frozen_files'].items()}}
    write(OUT/'RESULTS.json',result)
    (OUT/'CASE_RESULTS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details),encoding='utf-8')
    losses = set(pair['loss_qids'])
    write(OUT/'REGRESSIONS_FOR_NEXT_ROUND.json',[{**c,'evaluation_group':'regression'} for c in cases if c['qid'] in losses])
    def react_summary(records):
        valid = [r for r in records if r['react']['f1'] is not None and r['conditions']['support_new']['f1'] is not None]
        diffs = [r['conditions']['support_new']['f1']-r['react']['f1'] for r in valid]
        methods = {label:overall['conditions'][label] for label in ('reuse_new','reuse_single')} if records is react_details else {}
        return {'planned':len(records),'pairs':len(valid),'our_f1':mean([r['conditions']['support_new']['f1'] for r in valid]), 'our_em':mean([r['conditions']['support_new']['em'] for r in valid]), 'react_em':mean([r['react']['em'] for r in valid]),
            'react_f1':mean([r['react']['f1'] for r in valid]),'mean_difference':mean(diffs),
            'gains':sum(d>0 for d in diffs),'ties':sum(d==0 for d in diffs),'losses':sum(d<0 for d in diffs),
            'loss_qids':[r['qid'] for r,d in zip(valid,diffs) if d<0],
            'our_mean_calls':mean([r['conditions']['support_new']['logical_calls'] for r in valid]),
            'react_mean_calls':mean([r['react']['logical_calls'] for r in valid if r['react'].get('complete_usage')]),
            'our_mean_standalone_cny':mean([r['conditions']['support_new']['logical_standalone_list_price_cny'] for r in valid]),
            'react_mean_standalone_cny':mean([r['react']['logical_standalone_list_price_cny'] for r in valid if r['react'].get('complete_usage')])}
    comparison = {'our_condition':'support_new','single_condition':'support_single','overall':react_summary(react_details),'by_dataset':{ds:react_summary([r for r in react_details if r['dataset']==ds]) for ds in ('hotpotqa','2wiki','musique')},
        'by_evaluation_group':{g:react_summary([r for r in react_details if r['evaluation_group']==g]) for g in ('fresh','regression','competitor_gap')},
        'limit':'manual query-only reference plans versus frozen adapted ReAct k10 s5; shared corpora/BGE/model, different prompts/search budgets; diagnostic, not official SOTA or automatic-method dominance'}
    write(OUT/'REACT_COMPARISON.json',comparison)
    (OUT/'REACT_CASE_RESULTS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in react_details),encoding='utf-8')
    pct = lambda n:'未完成' if n is None else f'{100*n:.2f}'
    text = ['第4轮：已知部分绑定和辅助缺省冲突的独立对照',f'完成={complete}；零下降门槛={passed}',
        '12新题+前轮全部下降题+对ReAct缺口诊断题；人工query-only参考分解，非独立泛化结论。',
        '三种检索路乘两种最终prompt；子回答与一次已有材料重读固定v4，分别隔离两项改动，其他条件一致。',
        '方案\t全部F1%\t新题F1%\t平均API调用\t重读节点数']
    for label in labels:
        s=overall['conditions'][label]
        text.append(f"{label}\t{pct(s['mean_f1_available'])}\t{pct(result['by_evaluation_group']['fresh']['conditions'][label]['mean_f1_available'])}\t{s['mean_logical_calls']}\t{s['reread_nodes']}")
    text += ['',json.dumps(overall['pairs'],ensure_ascii=False),'', '与ReAct：'+json.dumps(comparison,ensure_ascii=False),
        '',f'新增用量估算¥{actual}；未结预留¥{reserved}；总已返回用量¥{total}；总预留¥{total_reserved}',
        '逻辑成本包含所有第一遍和重读调用，含复用成功请求；与新增实际提交用量分开。','', '逐题：']
    for r in details:
        text += ['',r['dataset']+'/'+r['qid']+' / '+r['evaluation_group'],r['question'],'Gold（仅离线）: '+str(r['gold_answer'])]
        text += [label+': '+str(r['conditions'][label]) for label in labels]
    (OUT/'REPORT.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'complete':complete,'gate':passed,'conditions':overall['conditions'],'pairs':overall['pairs'],'react':comparison['overall'],'new_cost':actual},ensure_ascii=False))


if __name__ == '__main__':
    main()
