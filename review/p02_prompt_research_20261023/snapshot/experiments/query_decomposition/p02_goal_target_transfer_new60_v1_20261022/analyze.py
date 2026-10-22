"""Offline aggregation and request-level integrity checks; never submit models."""
from collections import Counter,defaultdict
from contextlib import closing
import argparse
import json
import random
import statistics
from . import run as r


def paired_ci(groups):
    if not groups:return None
    rng=random.Random(20261022)
    values=[]
    for _ in range(10000):
        # Preserve each dataset's paired cohort size.
        values.append(statistics.mean(sum((rng.choices(g,k=len(g)) for g in groups),[])))
    values.sort()
    return [values[249],values[9749]]


def snapshot():
    config,examples=r.check()
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    inputs={x['id']:x for x in r.rows(r.HERE/'INPUTS60.frozen.jsonl')}
    refs={x['id']:x for x in r.rows(r.HERE/'REFERENCES60.local.frozen.jsonl')}
    history={x['id']:x for x in r.rows(r.HERE/'HISTORY60.local.frozen.jsonl')}
    with closing(r.ro(r.OUT/'checkpoint.sqlite3')) as db:
        results=[json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='prediction'")]
        receipts={rid:dict(payload=json.loads(p),status=s,result=json.loads(v) if v else {})
                  for rid,p,s,v in db.execute('SELECT id,payload,status,result FROM requests')}
        retired=[dict(id=rid,payload=json.loads(p),status=s,result=json.loads(v) if v else {})
                 for rid,p,s,v in db.execute('SELECT id,payload,status,result FROM retired_requests')]
        stages=set(db.execute('SELECT request_id,stage FROM request_stages'))
    index={(x['arm'],x['id']):x for x in results}
    scorers={d:metric(d) for d in r.DATASETS}
    for x in results:
        x['metrics']=scorers[x['dataset']](x.get('prediction',''),refs[x['id']]) if x['status'] in ('success','algorithm_failure') else None
    common=[i for i in inputs if all(index.get((a,i),{}).get('metrics') is not None for a in r.templates.ARMS)]
    table=[]
    for ds in ('all',)+r.DATASETS:
        ids=[i for i in common if ds=='all' or inputs[i]['dataset']==ds]
        scores={}
        for a in ('historical_native_P02',)+r.templates.ARMS:
            vals=[scorers[inputs[i]['dataset']](history[i]['prediction'],refs[i]) if a=='historical_native_P02'
                  else index[(a,i)]['metrics'] for i in ids]
            scores[a]={k:statistics.mean(v[k] for v in vals) if vals else None for k in ('f1','em')}
        table.append(dict(dataset=ds,n=len(ids),scores=scores))
    paired={}; changes=[]
    for a,b in (('G0','T'),('GF','GF_SPLIT'),('G0','GF')):
        ids=[i for i in inputs if all(index.get((arm,i),{}).get('metrics') is not None for arm in (a,b))]
        delta=[];groups=defaultdict(list);counts=Counter()
        for i in ids:
            diff=index[(b,i)]['metrics']['f1']-index[(a,i)]['metrics']['f1'];delta.append(diff);groups[inputs[i]['dataset']].append(diff)
            direction='improved' if diff>1e-12 else 'harmed' if diff < -1e-12 else 'tied';counts[direction]+=1
            if direction!='tied':changes.append(dict(comparison=a+'->'+b,direction=direction,id=i,question=inputs[i]['question'],
                reference=refs[i]['answer'],F1_delta=diff,ordinary_nodes=inputs[i]['ordinary_nodes'],
                variants={arm:index[(arm,i)] for arm in (a,b)}))
        paired[a+'->'+b]=dict(n=len(ids),**{k:counts[k] for k in ('improved','harmed','tied')},
            F1_delta=statistics.mean(delta) if delta else None,paired_dataset_stratified_bootstrap95=paired_ci(list(groups.values())),
            bootstrap_samples=10000,bootstrap_seed=20261022,exploratory_unadjusted=True)
    costs={}
    for a in r.templates.ARMS:
        vals=[x for x in results if x['arm']==a and x['status']=='success']
        usages=[{k:sum(s['usage'][k] for s in x['usage_by_stage'].values()) for k in ('prompt_tokens','completion_tokens')} for x in vals]
        costs[a]=dict(n=len(vals),**{k:statistics.mean(u[k] for u in usages) if usages else None for k in ('prompt_tokens','completion_tokens')},
            goal_final_CNY=statistics.mean(r.cost(u) for u in usages) if usages else None,
            goal_words_mean=statistics.mean(len(x['goal_text'].split()) for x in vals) if vals else None)
    namespace='p02_prompt_research_v1_20261022/'+str(r.OUT.resolve())
    with closing(r.ro(config['budget']['ledger'])) as db:
        charges={i:dict(actual=a,reserved=rs,status=s) for i,a,rs,s in db.execute('SELECT id,actual,reserved,status FROM charges WHERE namespace=?',(namespace,))}
    summary=dict(planned_conditions=240,terminal_conditions=len(results),statuses=dict(Counter(x['status'] for x in results)),
        receipt_statuses=dict(Counter(x['status'] for x in receipts.values())),same_four_arm_n=len(common),common_ids=common,
        table=table,paired=paired,costs=costs,
        split_effective_changed=sum(x.get('presentation',{}).get('changed',False) for x in results if x['arm']=='GF_SPLIT'),
        new_returned_usage_CNY=sum(x['actual'] or 0 for x in charges.values()),
        unresolved_reserved_CNY=sum(x['reserved'] for x in charges.values() if x['actual'] is None),
        main_evaluation_paused=True,candidate_promoted=False,
        scope='New disjoint prompt-development cohort; frozen-prefix goal-final intervention, historically evaluated runtime300, not complete-RAG/test result')
    r.save(r.OUT/'SUMMARY.json',summary);r.save(r.REVIEW/'RESULTS.json',summary)
    r.jsonl(r.OUT/'DETAILS.local.jsonl',results);r.jsonl(r.OUT/'CHANGED.local.jsonl',changes)
    return summary,results,receipts,retired,stages,charges,inputs,config,examples


def audit(data):
    summary,results,receipts,retired,stages,charges,inputs,config,examples=data
    assert summary['terminal_conditions']==240,'Run incomplete; snapshot only'
    assert set(charges)==set(receipts)|{x['id'] for x in retired},'Ledger-request identity mismatch'
    prior_unknown={x['payload_sha256'] for x in r.rows(r.HERE/'EXTERNAL_UNKNOWN.frozen.jsonl')}
    attempts=Counter(r.digest(x['payload']) for x in receipts.values())
    attempts.update(r.digest(x['payload']) for x in retired)
    assert max(attempts.values())<=5
    verified=0
    for rid,x in receipts.items():
        assert r.digest(x['payload']) not in prior_unknown
        z=x['result'];u=z.get('usage')
        if u:
            assert u['prompt_tokens']+u['completion_tokens']==u['total_tokens']
            assert z['returned_model']==config['generation']['model']
            assert abs(r.cost(u)-charges[rid]['actual'])<1e-10
        if x['status']=='success':assert u,'Success without returned usage'
    index={(x['arm'],x['id']):x for x in results}
    for x in results:
        if x['status']!='success':continue
        record=inputs[x['id']];goalarm='GF' if x['arm']=='GF_SPLIT' else x['arm']
        final,context,meta=r.templates.final_prompt(record,x['goal_text'],config,examples,x['arm'])
        expected={'goal':(r.templates.goal_prompt(record,goalarm),512,'goal/'+goalarm+'|'+x['id']),
                  'final':(final,64,'final/'+x['key'])}
        for kind,(prompt,cap,stage) in expected.items():
            rid=x['usage_by_stage'][kind]['request_id'];receipt=receipts[rid]
            assert receipt['payload']==r.build_payload(config['generation'],prompt,cap)
            assert receipt['status']=='success' and receipt['result']['finish_reason']=='stop'
            assert (rid,stage) in stages
            assert receipt['result']['usage']==x['usage_by_stage'][kind]['usage']
            assert receipt['result']['answer'].strip()==x['goal_text' if kind=='goal' else 'prediction']
            verified+=1
        if x['arm']=='GF_SPLIT':
            other=index[('GF',x['id'])]
            assert other['goal_text']==x['goal_text'] and other['usage_by_stage']['goal']['request_id']==x['usage_by_stage']['goal']['request_id']
            if meta['changed']:assert meta['body']+meta['closing']==x['goal_text']
    cohort=r.read(r.HERE/'COHORT.frozen.json')
    assert not set(cohort['selected_ids'])&set(cohort['excluded_ids'])
    for binding in r.rows(r.HERE/'SOURCE_BINDINGS.frozen.jsonl'):
        assert r.sha(r.ROOT/binding['source'])==binding['sha256']
        assert r.sha(r.ROOT/binding['root_source'])==binding['root_sha256']
    result=dict(passed=True,verified_logical_receipts=verified,maximum_actual_attempts=max(attempts.values()),
        paired_GF_goal_identity=True,lossless_split=True,ledger_reconciled=True,source_hashes_verified=True,
        old_unknown_reissued=False,main_evaluation_paused=True)
    r.save(r.OUT/'OFFLINE_AUDIT.json',result);return result


def report(summary):
    lines=['# 新60题：goal问法约束与综合文本传递','',
        '本轮使用另外60题（每集20题），按题型比例与固定哈希顺序选取，排除已登记提示词实验的题目ID及规范化题面。名单在读取参考答案与历史预测前冻结。来源仍是已有runtime300，不能称作完全未见测试集。','',
        '冻结分解、普通子问答、goal改写及检索、final原始检索；只新测goal→final。G0为原goal事实句，T为G0加问法约束及四个虚构对照示例，GF为此前短综合模板，GF_SPLIT复用完全相同GF文本，仅插入正文/末句候选标题。goal上限512、final64，原生产配置不变。','',
        '| 数据集 | n | 历史P02 F1/EM | G0 F1/EM | T F1/EM | GF F1/EM | GF_SPLIT F1/EM |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def f(v):return '—' if v is None else f'{v*100:.2f}'
    for row in summary['table']:
        values=[f(row['scores'][a]['f1'])+'/'+f(row['scores'][a]['em']) for a in ('historical_native_P02',)+r.templates.ARMS]
        lines.append('| '+row['dataset']+' | '+str(row['n'])+' | '+' | '.join(values)+' |')
    lines+=['','历史P02为非同期原生输出参考；因goal上限和final候选包装不同，实验的因果对照使用G0。','',
        '| 配对对照 | n | F1变化（百分点） | 提升/退步/不变 | 95%配对bootstrap区间（百分点） |',
        '|---|---:|---:|---:|---:|']
    for name,p in summary['paired'].items():
        ci=p['paired_dataset_stratified_bootstrap95']
        lines.append(f"| {name} | {p['n']} | {f(p['F1_delta'])} | {p['improved']}/{p['harmed']}/{p['tied']} | {f(ci[0])} 至 {f(ci[1])} |")
    lines+=['','区间为分数据集保持样本量的配对重采样（固定seed、10000次），为探索性、未经多重比较校正的区间。小样本不能据此宣布通用收益或直接替换生产模板。','',
        '| 条件 | 输入tokens/query | 输出tokens/query | goal+final逻辑费用/题（元） | goal平均词数 |',
        '|---|---:|---:|---:|---:|']
    for a,c in summary['costs'].items():
        if c['n']:lines.append(f"| {a} | {c['prompt_tokens']:.1f} | {c['completion_tokens']:.1f} | {c['goal_final_CNY']:.6f} | {c['goal_words_mean']:.1f} |")
    lines+=['',f"已结束 {summary['terminal_conditions']}/240 个条件；状态：{summary['statuses']}。GF_SPLIT实际插入标题{summary['split_effective_changed']}题；其他题按预设保留原文。",
        f"本次新增返回usage费用：{summary['new_returned_usage_CNY']:.6f}元；未结预留：{summary['unresolved_reserved_CNY']:.6f}元。共享GF生成在本次实际费用中只记一次，条件逻辑费用各记其使用量。上述开销不包含被冻结的上游阶段。",
        '', '主评估继续暂停；没有更新生产模板、重新检索、分解器推理、GPU任务或远端发布。']
    text='\n'.join(lines)+'\n';(r.OUT/'REPORT.md').write_text(text,encoding='utf8');(r.REVIEW/'REPORT.md').write_text(text,encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    data=snapshot();report(data[0]);out=data[0]
    if args.audit:out['audit']=audit(data)
    print(json.dumps(out,ensure_ascii=True))
