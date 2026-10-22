"""Offline fixed analysis; source controls and new charges remain distinct."""
from collections import Counter,defaultdict
from contextlib import closing
import argparse
import json
import statistics
from . import run as r


def snapshot():
    config,examples=r.check()
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    inputs={x['id']:x for x in r.rows(r.HERE/'INPUTS60.frozen.jsonl')}
    refs={x['id']:x for x in r.rows(r.HERE/'REFERENCES60.local.frozen.jsonl')}
    with closing(r.ro(r.OUT/'checkpoint.sqlite3')) as db:
        results=[json.loads(x) for x, in db.execute("SELECT value FROM data WHERE kind='prediction'")]
        receipts={i:dict(payload=json.loads(p),status=s,result=json.loads(v) if v else {})
                  for i,p,s,v in db.execute('SELECT id,payload,status,result FROM requests')}
        retired=[dict(id=i,payload=json.loads(p),status=s,result=json.loads(v) if v else {})
                 for i,p,s,v in db.execute('SELECT id,payload,status,result FROM retired_requests')]
        stages=set(db.execute('SELECT request_id,stage FROM request_stages'))
    scorers={ds:metric(ds) for ds in r.DATASETS}
    for x in results:x['metrics']=scorers[x['dataset']](x.get('prediction',''),refs[x['id']]) if x['status'] in ('success','algorithm_failure') else None
    index={(x['arm'],x['id']):x for x in results}
    common=[i for i in inputs if all(index.get((a,i),{}).get('metrics') is not None for a in r.templates.ARMS)]
    table=[]
    for ds in ('all',)+r.DATASETS:
        ids=[i for i in common if ds=='all' or inputs[i]['dataset']==ds]
        scores={a:{k:statistics.mean(index[(a,i)]['metrics'][k] for i in ids) if ids else None for k in ('f1','em')} for a in r.templates.ARMS}
        table.append(dict(dataset=ds,n=len(ids),scores=scores))
    paired={};changed=[]
    for a,b in r.templates.PAIRS:
        ids=[i for i in inputs if all(index.get((v,i),{}).get('metrics') is not None for v in (a,b))]
        differences=[];groups=defaultdict(list);counts=Counter()
        for i in ids:
            delta=index[(b,i)]['metrics']['f1']-index[(a,i)]['metrics']['f1']
            differences.append(delta);groups[inputs[i]['dataset']].append(delta)
            direction='improved' if delta>1e-12 else 'harmed' if delta < -1e-12 else 'tied';counts[direction]+=1
            if index[(a,i)]['prediction']!=index[(b,i)]['prediction']:
                changed.append(dict(pair=a+'->'+b,id=i,dataset=inputs[i]['dataset'],question=inputs[i]['question'],
                    reference=refs[i]['answer'],F1_delta=delta,direction=direction,
                    ordinary_nodes=inputs[i]['ordinary_nodes'],variants={arm:index[(arm,i)] for arm in (a,b)}))
        paired[a+'->'+b]=dict(n=len(ids),**{k:counts[k] for k in ('improved','harmed','tied')},
            F1_delta=statistics.mean(differences) if differences else None,
            paired_bootstrap95=r.paired_ci(list(groups.values())),seed=20261022,samples=10000,
            exploratory_unadjusted=True)
    costs={}
    for a in r.templates.ARMS:
        vals=[x for x in results if x['arm']==a and x['status']=='success']
        usages=[{k:sum(u['usage'][k] for u in x['usage_by_stage'].values()) for k in ('prompt_tokens','completion_tokens')} for x in vals]
        costs[a]=dict(n=len(vals),**{k:statistics.mean(u[k] for u in usages) if usages else None for k in ('prompt_tokens','completion_tokens')},
            goal_final_CNY=statistics.mean(r.cost(u) for u in usages) if usages else None,
            goal_words_mean=statistics.mean(len(x['goal_text'].split()) for x in vals) if vals else None)
    namespace='p02_prompt_research_v1_20261022/'+str(r.OUT.resolve())
    with closing(r.ro(config['budget']['ledger'])) as db:
        charges={i:dict(actual=a,reserved=rs,status=s) for i,a,rs,s in db.execute('SELECT id,actual,reserved,status FROM charges WHERE namespace=?',(namespace,))}
    summary=dict(planned_conditions=300,terminal_conditions=len(results),statuses=dict(Counter(x['status'] for x in results)),
        same_five_arm_n=len(common),common_ids=common,table=table,paired=paired,costs=costs,
        receipt_statuses=dict(Counter(x['status'] for x in receipts.values())),
        actual_new_returned_usage_CNY=sum(x['actual'] or 0 for x in charges.values()),
        unresolved_reserved_CNY=sum(x['reserved'] for x in charges.values() if x['actual'] is None),
        changed_final_pairs=len(changed),scope='Exposed60, frozen-prefix intervention; strict replayed controls; no independent/full-RAG claim',
        main_evaluation_paused=True,candidate_promoted=False)
    r.save(r.OUT/'SUMMARY.json',summary);r.save(r.REVIEW/'RESULTS.json',summary)
    r.jsonl(r.OUT/'DETAILS.local.jsonl',results);r.jsonl(r.OUT/'CHANGED.local.jsonl',changed)
    return summary,results,receipts,retired,stages,charges,inputs,config,examples


def audit(data):
    summary,results,receipts,retired,stages,charges,inputs,config,examples=data
    assert summary['terminal_conditions']==300
    seeds={r.digest(x['payload']):x for x in r.rows(r.HERE/'SEED_RECEIPTS.frozen.jsonl')}
    ids=set(receipts)|{x['id'] for x in retired};assert set(charges)<=ids
    attempts=Counter(r.digest(x['payload']) for rid,x in receipts.items() if rid in charges)
    attempts.update(r.digest(x['payload']) for x in retired if x['id'] in charges)
    assert not attempts or max(attempts.values())<=5
    blocked={x['payload_sha256'] for x in r.rows(r.HERE/'EXTERNAL_UNKNOWN.frozen.jsonl')}
    reused=0;verified=0
    for rid,x in receipts.items():
        d=r.digest(x['payload']);assert d not in blocked
        z=x['result'];u=z.get('usage')
        if rid not in charges:
            assert d in seeds and z==seeds[d]['result'],'Uncharged request without exact source receipt'
            reused+=1
        elif u:
            assert abs(r.cost(u)-charges[rid]['actual'])<1e-10
        if u:
            assert u['prompt_tokens']+u['completion_tokens']==u['total_tokens']
            assert z['returned_model']==config['generation']['model']
    index={(x['arm'],x['id']):x for x in results}
    controls={(x['arm'],x['id']):x for x in r.rows(r.HERE/'CONTROLS.frozen.jsonl')}
    for x in results:
        if x['status']!='success':continue
        item=inputs[x['id']];goalarm='GF' if x['arm']=='R' else x['arm']
        expected={'goal':(r.templates.goal_prompt(item,goalarm),512,'goal/'+goalarm+'|'+x['id'],'goal_text'),
            'final':(r.templates.final_prompt(item,x['goal_text'],config,examples,x['arm'])[0],64,'final/'+x['key'],'prediction')}
        for kind,(prompt,cap,stage,key) in expected.items():
            rid=x['usage_by_stage'][kind]['request_id'];z=receipts[rid]
            assert z['payload']==r.build_payload(config['generation'],prompt,cap)
            assert z['status']=='success' and z['result']['finish_reason']=='stop' and (rid,stage) in stages
            assert z['result']['answer'].strip()==x[key] and z['result']['usage']==x['usage_by_stage'][kind]['usage']
            verified+=1
        if x['arm'] in ('G0','GF'):
            control=controls[(x['arm'],x['id'])]
            assert (x['goal_text'],x['prediction'])==(control['goal_text'],control['prediction'])
        if x['arm']=='R':
            gf=index[('GF',x['id'])]
            assert x['goal_text']==gf['goal_text'] and x['usage_by_stage']['goal']['request_id']==gf['usage_by_stage']['goal']['request_id']
    for binding in r.rows(r.HERE/'SOURCE_BINDINGS.frozen.jsonl'):
        assert r.sha(r.ROOT/binding['source'])==binding['sha256'] and r.sha(r.ROOT/binding['root_source'])==binding['root_sha256']
    result=dict(passed=True,verified_logical_receipts=verified,source_reused_receipts=reused,
        maximum_actual_attempts=max(attempts.values()) if attempts else 0,strict_controls_identical=True,
        R_goal_identical_to_GF=True,ledger_reconciled=True,main_evaluation_paused=True)
    r.save(r.OUT/'OFFLINE_AUDIT.json',result);return result


def report(s):
    def f(x):return '—' if x is None else f'{x*100:.2f}'
    lines=['# goal充分性与信息保留：固定60题开发实验','',
        '沿用上一轮已暴露60题；G0/GF严格复用完整同payload回执，非新同时期重复对照。分解、普通QA及检索固定，只测goal→final，不属于完整RAG正式评估。','',
        'A仅在G0上补充允许证据充分的替代求解路径；P仅在GF上补充候选答案保留约束；R复用同一GF文本，仅改final指令。三个增量没有新增few-shot或组合执行。','',
        '| 数据集 | n | G0 F1/EM | A F1/EM | GF F1/EM | P F1/EM | R F1/EM |','|---|---:|---:|---:|---:|---:|---:|']
    for row in s['table']:
        lines.append('| '+row['dataset']+' | '+str(row['n'])+' | '+' | '.join(f(row['scores'][a]['f1'])+'/'+f(row['scores'][a]['em']) for a in r.templates.ARMS)+' |')
    lines+=['','| 与自身对照比较 | n | F1变化/百分点 | 提升/退步/不变 | 95%配对区间/百分点 |','|---|---:|---:|---:|---:|']
    for name,p in s['paired'].items():
        ci=p['paired_bootstrap95']
        lines.append(f"| {name} | {p['n']} | {f(p['F1_delta'])} | {p['improved']}/{p['harmed']}/{p['tied']} | {f(ci[0])} 至 {f(ci[1])} |")
    lines+=['','按数据集保持样本量的配对bootstrap，seed20261022，10000次；探索性区间未作多重比较校正。题目已用于形成修改动机，不能称独立确认。','',
        '| 条件 | 输入tokens/题 | 输出tokens/题 | goal+final逻辑费用/题（元） |','|---|---:|---:|---:|']
    for a,c in s['costs'].items():
        if c['n']:lines.append(f"| {a} | {c['prompt_tokens']:.1f} | {c['completion_tokens']:.1f} | {c['goal_final_CNY']:.6f} |")
    lines += ['',f"已结束{s['terminal_conditions']}/300个条件，状态{s['statuses']}。新增返回usage费用{s['actual_new_returned_usage_CNY']:.6f}元；未结预留{s['unresolved_reserved_CNY']:.6f}元。",
        '逻辑费用包含复用的goal/final使用量；本次实际费用仅记新付费回执，不将缓存支出混入方法效率或完整流程费用。','',
        '逐题差异见CHANGED.local.jsonl。主评估保持暂停，生产模板不晋级；本轮停止于固定300个条件。']
    text='\n'.join(lines)+'\n';(r.OUT/'REPORT.md').write_text(text,encoding='utf8');(r.REVIEW/'REPORT.md').write_text(text,encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');a=p.parse_args()
    data=snapshot();report(data[0]);answer=data[0]
    if a.audit:answer['audit']=audit(data)
    print(json.dumps(answer,ensure_ascii=True))
