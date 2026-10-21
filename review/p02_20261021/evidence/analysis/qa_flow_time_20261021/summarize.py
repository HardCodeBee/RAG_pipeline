"""Offline timing summary. All experiment databases are opened read-only.

No model/API imports. No mutation of sources, results, queues or running jobs.
Only writes a new timestamped report beneath this analysis directory.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import statistics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1'
CAMPAIGN = OUTPUT/'comprehensive_v1_20261015'
INPUTS = ROOT/'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen'
DATASETS = ['hotpotqa', '2wiki', 'musique']
NAMES = {
    'b00_rag_top10_v1':'Top-10 RAG', 'b01_rag_top10_cot3':'Top-10 + CoT',
    'b02_selfask_dense':'Self-Ask', 'b03_react_dense':'ReAct',
    'b04_ircot_bge':'IRCoT', 'b05_iter_retgen_bge':'Iter-RetGen',
    'b06_decomp_bge':'Decomp', 'b07_hybrid_rerank':'Hybrid + Rerank',
    'b08_qd_rerank':'QD + Rerank', 'p02_multigrain_timing_cachefix_v1':'P02',
    's04_scope_plus_evidence':'S04', 'm00_s02_child4_faithful':'S02 (historical)',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def lines(path):
    with Path(path).open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def number(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value) and value >= 0


def stats(values):
    values = [v for v in values if number(v)]
    return {'n':len(values), 'mean_seconds':statistics.mean(values) if values else None,
            'median_seconds':statistics.median(values) if values else None,
            'total_seconds':sum(values) if values else None}


@contextmanager
def ro(path):
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=5)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        yield db
    finally:
        db.close()


def is_final(pred):
    return pred.get('status','').startswith('success') and isinstance(pred.get('prediction'),str)


def extract_latency(db):
    return {rid:seconds for rid,seconds in db.execute(
        "SELECT id,json_extract(result,'$.latency_seconds') FROM requests WHERE result IS NOT NULL")}


def baseline(version,dataset,method,config,path):
    base = {'version':version,'dataset':dataset,'method':method,'name':NAMES[method],
            'config':config,'planned':500,'source':str(path.relative_to(ROOT)),
            'snapshot_utc':datetime.now(timezone.utc).isoformat()}
    if not path.exists():
        return {**base,'processed':0,'final_answers':0,'timing_class':'NO_FORMAL_RESULT',
                'mean_seconds':None,'timed_queries':0},[]
    with ro(path) as db:
        preds={qid:json.loads(raw) for qid,raw in db.execute("SELECT key,value FROM data WHERE kind='prediction'")}
        latency=extract_latency(db)
        binding_kind='token_call' if version=='primary_v2' else 'evidence'
        bindings=defaultdict(set)
        for qid,stage,rid in db.execute("SELECT json_extract(value,'$.qid'),json_extract(value,'$.stage'),"
                "json_extract(value,'$.request_id') FROM data WHERE kind=?",(binding_kind,)):
            bindings[qid].add((stage,rid))
        dense,sparse,rerank=defaultdict(list),defaultdict(list),defaultdict(list)
        legacy_rerank={qid:seconds for qid,seconds in db.execute(
            "SELECT json_extract(value,'$.qid'),json_extract(value,'$.rerank_seconds') "
            "FROM data WHERE kind='legacy_hybrid'")}
        reused_rerank_qids=set()
        for kind,qid,bsec,bn,sec,total_ms,rsec in db.execute(
                "SELECT kind,json_extract(value,'$.qid'),json_extract(value,'$.batch_seconds'),"
                "json_extract(value,'$.batch_queries'),json_extract(value,'$.seconds'),"
                "json_extract(value,'$.timings_ms.total_ms'),json_extract(value,'$.rerank_seconds') "
                "FROM data WHERE kind IN ('retrieval','sparse_retrieval','hybrid')"):
            if kind=='retrieval':
                dense[qid].append(bsec/bn if number(bsec) and number(bn) and bn>0 else (sec if number(sec) else None))
            elif kind=='sparse_retrieval':
                sparse[qid].append(total_ms/1000 if number(total_ms) else None)
            else:
                if rsec == 0 and qid in legacy_rerank:
                    # v2 records zero for a reused old ranking. Preserve the
                    # old measured reranking work, never call that work free.
                    rsec=legacy_rerank[qid] if number(legacy_rerank[qid]) and legacy_rerank[qid]>0 else None
                    reused_rerank_qids.add(qid)
                rerank[qid].append(rsec if number(rsec) else None)
    expected={v['qid'] for v in lines(INPUTS/dataset/'formal.questions.jsonl')}
    assert set(preds).issubset(expected),str(path)
    details=[]
    for qid,pred in preds.items():
        if not is_final(pred):
            continue
        linked=bindings[qid]
        times=[latency.get(rid) for stage,rid in linked]
        known=[x for x in times if number(x)]
        row={**{k:base[k] for k in ['version','dataset','method']},'qid':qid,
             'status':pred['status'],'observed_total_seconds':None,
             'llm_seconds':sum(known) if known else None,
             'llm_calls':len(linked),'llm_calls_with_timing':len(known),
             'dense_records':len(dense[qid]),'sparse_records':len(sparse[qid]),
             'rerank_records':len(rerank[qid])}
        row['rerank_uses_historical_measurement']=qid in reused_rerank_qids
        for name,values in [('dense_seconds',dense[qid]),('sparse_seconds',sparse[qid]),('rerank_seconds',rerank[qid])]:
            observed=[v for v in values if number(v)]
            row[name]=sum(observed) if observed else None
            row[name+'_missing_records']=len(values)-len(observed)
        components=[row[k] for k in ['llm_seconds','dense_seconds','sparse_seconds','rerank_seconds'] if row[k] is not None]
        row['known_stage_sum_seconds']=sum(components) if components else None
        details.append(row)
    combined=stats([v['known_stage_sum_seconds'] for v in details])
    component_stats={k:stats([v[k] for v in details]) for k in ['llm_seconds','dense_seconds','sparse_seconds','rerank_seconds']}
    return {**base,'processed':len(preds),'statuses':dict(Counter(v['status'] for v in preds.values())),
            'final_answers':len(details),'timing_class':'PARTIAL_COMPONENT_SUM' if combined['n'] else 'NO_TIMING',
            'mean_seconds':combined['mean_seconds'],'timed_queries':combined['n'],
            'components':component_stats,'llm_calls_with_timing':sum(v['llm_calls_with_timing'] for v in details),
            'llm_calls_without_timing':sum(v['llm_calls']-v['llm_calls_with_timing'] for v in details),
            'queries_reusing_historical_rerank_time':sum(v['rerank_uses_historical_measurement'] for v in details),
            'missing_parts':['per-query entry and final-return boundary','API pool/rate-slot waiting',
                'retry backoff and failed earlier attempts','prompt construction and local orchestration',
                'unlogged repeated retrieval-cache lookups; document fetch/merge not uniformly timed'],
            'caveats':['Stored historical response latencies remain historical measurements, not live replay latency.',
                'Dense batch time is divided by batch_queries; this is amortized work, not the full wait of each query.',
                'Different record types are summed as available; missing components are explicitly unmeasured, not claimed zero.',
                'Only final-answer-producing queries are averaged; algorithm/infrastructure failures are counted separately.']},details


def p02():
    output=OUTPUT/'p02_formal1500_v1_20261021'
    selected={}
    for path in sorted(output.glob('rounds/r*/cases/*/*/*.json')):
        value=read(path);key=value['id']
        if key in selected and (is_final(selected[key][0]) or selected[key][0]['status']=='algorithm_failure'):
            continue
        selected[key]=(value,path)
    remote={}
    for path in (output/'remote_raw').glob('[0-9]*.json'):
        value=read(path);remote[value['id']]=value.get('elapsed_seconds')
    summaries,details=[],[]
    for dataset in DATASETS:
        cases=[(v,p) for v,p in selected.values() if v['dataset']==dataset]
        measured=[]
        for case,path in cases:
            if not is_final(case):
                continue
            local=case.get('actual_seconds');decomp=remote.get(case['id'])
            row={'version':'primary_frozen_p02','dataset':dataset,'method':'p02_multigrain_timing_cachefix_v1',
                 'qid':case['qid'],'source':str(path.relative_to(ROOT)),
                 'observed_local_seconds':local if number(local) else None,
                 'remote_decomposition_seconds':decomp if number(decomp) else None,
                 'staged_total_seconds':local+decomp if number(local) and number(decomp) else None,
                 'round':path.parts[path.parts.index('rounds')+1],
                 'reused_llm_calls':sum(bool(v.get('reused')) for v in case.get('logical_calls',[]))}
            measured.append(row);details.append(row)
        staged=stats([v['staged_total_seconds'] for v in measured])
        summaries.append({'version':'primary_frozen_p02','dataset':dataset,
            'method':'p02_multigrain_timing_cachefix_v1','name':'P02','planned':500,
            'processed':len(cases),'final_answers':len(measured),
            'statuses':dict(Counter(v['status'] for v,p in cases)),
            'timing_class':'MEASURED_STAGES_SUM' if staged['n'] else 'NO_FORMAL_RESULT',
            'mean_seconds':staged['mean_seconds'],'timed_queries':staged['n'],
            'observed_local':stats([v['observed_local_seconds'] for v in measured]),
            'remote_decomposition':stats([v['remote_decomposition_seconds'] for v in measured]),
            'selected_success_rounds':dict(Counter(v['round'] for v in measured)),
            'queries_reusing_llm_calls':sum(v['reused_llm_calls']>0 for v in measured),
            'caveats':['Local actual_seconds measures the selected successful attempt, including its actual queue/cache behavior.',
                'Remote inference is separately measured; adding it does not yield a continuous end-to-end wall-clock measurement.',
                'Inter-machine transfer, remote job queue, startup and pauses between recovery rounds are not included.',
                'Earlier failed attempts are retained but not added to the selected-success mean.']})
    return summaries,details


def block_history():
    grouped=defaultdict(lambda: {'seconds':0.,'new_terminal_records':0,'blocks':0})
    for row in lines(ROOT/'experiments/query_decomposition/comprehensive_eval_v1_20261015/changes.jsonl'):
        if row.get('event')!='execution_epoch_block_timing' or row.get('split')!='formal':
            continue
        key=(row['dataset'],row['method'])
        grouped[key]['seconds']+=row['seconds']
        grouped[key]['new_terminal_records']+=row['new_terminal_records'];grouped[key]['blocks']+=1
    return [{'dataset':ds,'method':mid,**value,
             'amortized_evaluation_seconds':value['seconds']/value['new_terminal_records'],
             'scope':'Original campaign block runtime; includes parallel execution, exports and offline scoring. Not query response latency.'}
            for (ds,mid),value in sorted(grouped.items()) if value['new_terminal_records']>0]


def table(rows):
    result=['方法 | HotpotQA | 2Wiki | MuSiQue','单位：秒/query；括号为有最终回答且有计时的题数。P=部分组件计时；S=阶段实测相加。']
    for mid in [*NAMES]:
        matches={r['dataset']:r for r in rows if r['method']==mid}
        if not matches:
            continue
        cells=[]
        for ds in DATASETS:
            row=matches.get(ds,{})
            seconds=row.get('mean_seconds')
            tag='S' if row.get('timing_class')=='MEASURED_STAGES_SUM' else 'P'
            cells.append(f"{seconds:.2f} [{tag}; n={row['timed_queries']}]" if seconds is not None else
                         ('无正式结果' if row.get('timing_class')=='NO_FORMAL_RESULT' else '无计时数据'))
        result.append(NAMES[mid]+' | '+' | '.join(cells))
    return result


def main():
    now=datetime.now(timezone.utc)
    destination=HERE/'reports'/now.strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True,exist_ok=False)
    primary,history,details=[],[],[]
    for dataset in DATASETS:
        configs=read(CAMPAIGN/'campaign_api_v1'/dataset/'BASELINE_CONFIGS_FROZEN.json')['selected']
        for method,config in configs.items():
            for version in ['original_campaign','primary_v2']:
                path=CAMPAIGN/'campaign_api_v1'/dataset/'formal'/method/config['id']/'checkpoint.sqlite3'
                actual_config=config
                if version=='primary_v2':
                    parent=CAMPAIGN/'baseline_integrity_repair_v2_20261017'
                    path=parent/'main'/dataset/'formal'/method/config['id']/'checkpoint.sqlite3'
                    if method=='b05_iter_retgen_bge':
                        actual_config={'id':'k10_t5','k':10,'iterations':5}
                        path=parent/'matched50'/dataset/'formal'/method/'k10_t5/checkpoint.sqlite3'
                summary,records=baseline(version,dataset,method,actual_config,path)
                (primary if version=='primary_v2' else history).append(summary);details.extend(records)
        original=CAMPAIGN/'campaign_api_v1'/dataset/'formal/m00_s02_child4_faithful/frozen_s02_child4/checkpoint.sqlite3'
        if original.exists():
            summary,records=baseline('original_campaign',dataset,'m00_s02_child4_faithful',{'id':'frozen_s02_child4'},original)
            history.append(summary);details.extend(records)
        primary.append({'version':'s04_formal','method':'s04_scope_plus_evidence','name':'S04',
                        'dataset':dataset,'planned':500,'final_answers':0,'timed_queries':0,
                        'mean_seconds':None,'timing_class':'NO_FORMAL_RESULT',
                        'note':'Only development300 replay is located; not substituted for formal500.'})
    pp,records=p02();primary.extend(pp);details.extend(records)
    assert all(v.get('mean_seconds') is None or number(v['mean_seconds']) for v in primary+history)
    assert all(v.get('timed_queries',0)<=v.get('final_answers',0)<=500 for v in primary+history)
    result={'snapshot_utc':now.isoformat(),'target_metric':'Original query entry to final answer completion; seconds per query.',
        'methodology':{
            'PARTIAL_COMPONENT_SUM':'Sum available successful-call latency, dense batch_seconds/batch_queries, sparse total_ms/1000 and rerank_seconds by query, then average final-answer queries with timing. Unobserved stages are marked missing, not asserted zero.',
            'MEASURED_STAGES_SUM':'P02 selected successful local actual_seconds plus that query remote elapsed_seconds; a reconstruction from measured stages, not one continuous timer.',
            'NO_FORMAL_RESULT':'No current-version formal final-answer result; no timing substitution from old version/development.',
            'NO_TIMING':'A formal result exists but no usable timing was found.',
            'denominator':'Final answer produced, regardless of answer correctness. Algorithm failures, infrastructure failures and unstarted queries remain separate.'},
        'primary':primary,'historical_original_separate':history,'historical_block_runtime_separate':block_history(),
        'new_api_calls':0,'new_retrievals':0,'frozen_sources_modified':False,
        'limitations':['None of the partial or staged reconstructions is a fully comparable controlled end-to-end latency benchmark.',
            'Original and repaired version results are kept separate; native Iter T2/T3 is not substituted for matched50 T5.',
            'Live databases are individually read under consistent read-only snapshots; cross-method snapshot times can differ.',
            'Timing from caches retains its actual measured provenance; replay is not presented as fresh inference.']}
    (destination/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (destination/'per_query.jsonl').open('w',encoding='utf8') as stream:
        for row in details:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    text=['逐题问答流程时间汇总（已有记录；不新增推理）',now.isoformat(),
        '目标：原始query进入流程至最终回答完成。完整连续计时不足时，按已有计时重建并显式标记。',
        'P：已记录组件耗时之和；缺少调度/排队/部分本地操作，不能称完整实测。',
        'S：P02本地成功执行实测 + 远端分解实测；不含跨机排队/传输和旧失败轮次，不能称连续实测。',
        '仅统计实际产出最终回答的题，不按答对与否筛选。失败与未启动题不伪造耗时。','',
        '当前正式主评估口径（v2；Iter k10/T5；P02独立冻结协议）',*table(primary),'',
        '历史原始版（独立列示，不填入修复版缺失单元）',*table(history),'',
        '各项组件计时、覆盖题数、缺失项目和批次运行时间见summary.json；逐题数值见per_query.jsonl。',
        '0新增API调用，0新增检索，不修改冻结源和运行进程。']
    (destination/'REPORT.txt').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({'report':str(destination/'REPORT.txt'),'summary':str(destination/'summary.json'),
                      'primary_rows':len(primary),'historical_rows':len(history),
                      'per_query_records':len(details),'new_api_calls':0},ensure_ascii=False),flush=True)
    print('\n'.join(table(primary)),flush=True)


if __name__=='__main__':
    main()
