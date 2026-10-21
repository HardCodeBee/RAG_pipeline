"""Per-question actual/retained tokens, cutoff invariants and native F1."""
from collections import Counter, defaultdict
import json
import sqlite3
from pathlib import Path
from .common import OUT, ROOT, PARENT, INPUTS, OUTPUT, read, rows, save, canonical, utc, usage_cost
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring_safe_trace import evaluate


def audit_tokens(directory):
    with sqlite3.connect((directory/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro', uri=True) as db:
        records = [(kind, key, json.loads(value)) for kind, key, value in db.execute(
            "select kind,key,value from data where kind in ('prediction','token_call','token_attempt','retrieval','rerank_token','integrity_violation')")]
        requests = {rid: (status, json.loads(result) if result else None) for rid, status, result in db.execute('select id,status,result from requests')}
    byqid, errors = defaultdict(lambda: defaultdict(list)), []
    for kind, key, row in records:
        byqid[row['qid']][kind].append(row)
        if kind=='integrity_violation':
            errors.append({'qid': row['qid'], 'reason': row['reason']})
        elif kind=='retrieval' and (row.get('encoder_truncated') or sum(row.get('encoded_window_tokens',[]))<row['query_tokens']):
            errors.append({'qid': row['qid'], 'reason': 'retrieval_input_not_fully_encoded'})
        elif kind=='rerank_token' and row.get('input_truncated'):
            errors.append({'qid': row['qid'], 'reason': 'rerank_input_truncated'})
        elif kind=='token_call' and row['status']=='success' and (row['finish_reason']!='stop' or not row['usage']):
            errors.append({'qid': row['qid'], 'reason': 'incomplete_successful_call'})
        elif kind=='prediction' and row['status']=='success':
            linked = requests.get(row.get('request_id'))
            if linked is None or linked[0]!='success' or linked[1].get('finish_reason')!='stop':
                errors.append({'qid': row['qid'], 'reason': 'final_answer_without_complete_request'})
    details = []
    for qid, data in sorted(byqid.items()):
        calls, attempts = data['token_call'], data['token_attempt']
        good = [c for c in calls if c.get('usage')]
        billed = [a for a in attempts if a.get('usage')]
        details.append({'qid': qid,
            'question': data['prediction'][0].get('question') if data['prediction'] else None,
            'logical_generation_calls': len(calls),
            'logical_prompt_tokens': sum(c['usage'].get('prompt_tokens',0) for c in good),
            'logical_completion_tokens': sum(c['usage'].get('completion_tokens',0) for c in good),
            'logical_total_tokens': sum(c['usage'].get('prompt_tokens',0)+c['usage'].get('completion_tokens',0) for c in good),
            'logical_cold_cost_estimate_cny': sum(usage_cost(c['usage'].get('prompt_tokens',0),c['usage'].get('completion_tokens',0)) for c in good),
            'new_returned_prompt_tokens': sum(c['usage'].get('prompt_tokens',0) for c in billed),
            'new_returned_completion_tokens': sum(c['usage'].get('completion_tokens',0) for c in billed),
            'new_returned_total_tokens': sum(c['usage'].get('prompt_tokens',0)+c['usage'].get('completion_tokens',0) for c in billed),
            'new_attempts_with_missing_usage': sum(not bool(a.get('usage')) for a in attempts),
            'calls_with_missing_usage': sum(not bool(c.get('usage')) for c in calls),
            'retained_historical_calls': sum(c['response_origin']=='complete_historical_prefix' for c in calls),
            'max_actual_prompt_tokens': max((c['usage'].get('prompt_tokens',0) for c in good),default=0),
            'max_actual_completion_tokens': max((c['usage'].get('completion_tokens',0) for c in good),default=0),
            'generation_calls': calls, 'new_submission_attempts': attempts,
            'dense_queries': [{k:v for k,v in r.items() if k not in {'docs'}} for r in data['retrieval']],
            'rerank_pairs': data['rerank_token'],
            'prediction_status': data['prediction'][0]['status'] if data['prediction'] else 'unstarted'})
    with (directory/'per_query_tokens.jsonl').open('w',encoding='utf8') as f:
        for item in details:
            f.write(canonical(item)+'\n')
    result = {'passed': not errors, 'errors': errors, 'questions_with_records': len(details),
        'logical_prompt_tokens': sum(d['logical_prompt_tokens'] for d in details),
        'logical_completion_tokens': sum(d['logical_completion_tokens'] for d in details),
        'logical_cold_cost_estimate_cny': sum(d['logical_cold_cost_estimate_cny'] for d in details),
        'new_returned_prompt_tokens': sum(d['new_returned_prompt_tokens'] for d in details),
        'new_returned_completion_tokens': sum(d['new_returned_completion_tokens'] for d in details),
        'retained_historical_calls': sum(d['retained_historical_calls'] for d in details),
        'max_actual_prompt_tokens': max((d['max_actual_prompt_tokens'] for d in details),default=0),
        'max_actual_completion_tokens': max((d['max_actual_completion_tokens'] for d in details),default=0),
        'note': 'Provider usage is authoritative; byte preflight bounds are not reported as measured tokens. Logical totals include retained/cached responses; new returned totals count actual new attempts once. Missing usage is unknown.'}
    save(directory/'TOKEN_AUDIT.json', result)
    return result


def score_store(dataset, split, directory):
    scored = evaluate(dataset, directory, INPUTS/dataset/(split+'.references.local.jsonl'))
    audit = audit_tokens(directory)
    if not audit['passed']:
        raise RuntimeError('Token integrity audit failed; no normal summary accepted')
    scored['logical_cold_cost_cny']=audit['logical_cold_cost_estimate_cny']
    scored['cost_pricing']='frozen Beijing tiered public list prices; actual bill not verified'
    save(directory/'scores.json',scored)
    return scored, audit


def main():
    lines = ['Token完整性修复版；原结果保留。'+utc(),
        '历史完整响应在输入一致且仅输出上限不同的情况下保留；不是全部重新抽样生成。',
        '主评估保留各数据集原选定检索参数；matched25另作等最大检索预算对照。', '']
    summaries = []
    for track in ['main','matched25']:
        for dataset in ['2wiki','hotpotqa','musique']:
            for directory in sorted((OUT/track/dataset/'formal').glob('*/*')):
                if not (directory/'checkpoint.sqlite3').exists():
                    continue
                scored, audit = score_store(dataset,'formal',directory)
                status = scored['overall']; mid = directory.parent.name
                summaries.append({'track':track,'dataset':dataset,'method':mid,'config':directory.name,
                    'overall':status,'tokens':audit,'directory':str(directory)})
                n=status['available'];f1=status['f1_available']
                lines.append(f"{track} {dataset} {mid}：结束{status['terminal_records']}/500 可评分{n} "
                    f"F1={100*f1:.2f}%" if f1 is not None else f'{track} {dataset} {mid}: no scorable answers')
                if track=='main':
                    selected=read(PARENT/dataset/'BASELINE_CONFIGS_FROZEN.json')['selected'][mid]['id']
                    oldpath=PARENT/dataset/'formal'/mid/selected/'scores.jsonl'
                    if oldpath.exists():
                        old={x['qid']:x for x in rows(oldpath)}
                        new={x['qid']:x for x in rows(directory/'scores.jsonl')}
                        pairs=[(old[q]['f1'],x['f1']) for q,x in new.items() if q in old and old[q]['f1'] is not None and x['f1'] is not None]
                        if pairs:
                            ds=[b-a for a,b in pairs]
                            summaries[-1]['old_new_common']={'common':len(pairs),
                                'old_f1':sum(a for a,b in pairs)/len(pairs),
                                'new_f1':sum(b for a,b in pairs)/len(pairs),'delta':sum(ds)/len(ds),
                                'gain':sum(d>1e-12 for d in ds),'tie':sum(abs(d)<=1e-12 for d in ds),'loss':sum(d < -1e-12 for d in ds)}
    save(OUT/'RESULTS.json', {'utc':utc(),'runs':summaries,'hallucination_judge_called':False})
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
