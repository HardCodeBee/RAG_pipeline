"""Read-only token accounting; write new analysis artifacts, never run inference."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
CAMPAIGN = OUT / 'comprehensive_v1_20261015'
P02 = OUT / 'p02_formal1500_v1_20261021'
DATASETS = ('hotpotqa', '2wiki', 'musique')
NAMES = {
    'b00_rag_top10_v1': 'Top-10 RAG', 'b01_rag_top10_cot3': 'Top-10 + CoT',
    'b02_selfask_dense': 'Self-Ask', 'b03_react_dense': 'ReAct',
    'b04_ircot_bge': 'IRCoT', 'b05_iter_retgen_bge': 'Iter-RetGen',
    'b06_decomp_bge': 'Decomp', 'b07_hybrid_rerank': 'Hybrid + Rerank',
    'b08_qd_rerank': 'QD + Rerank', 'p02': 'P02',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def valid_usage(value):
    return isinstance(value, dict) and all(
        isinstance(value.get(k), int) and not isinstance(value[k], bool) and value[k] >= 0
        for k in ('prompt_tokens', 'completion_tokens'))


def success(pred):
    return pred.get('status', '').startswith('success') and isinstance(pred.get('prediction'), str)


def aggregate(rows, version, dataset, method, **extra):
    complete = [r for r in rows if r['complete_usage']]
    return dict(version=version, dataset=dataset, method=method, name=NAMES[method],
                completed_queries=len(rows), complete_usage_queries=len(complete),
                incomplete_usage_queries=len(rows)-len(complete), planned_queries=500,
                mean_input_tokens=mean(r['input_tokens'] for r in complete) if complete else None,
                mean_output_tokens=mean(r['output_tokens'] for r in complete) if complete else None,
                mean_calls=mean(r['calls'] for r in complete) if complete else None,
                **extra)


def baseline(dataset, method, config, path):
    if not path.exists():
        return aggregate([], 'formal_v2', dataset, method, source=str(path.relative_to(ROOT)),
                         config=config, state='no_current_version_results'), []
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=5)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        preds = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM data WHERE kind='prediction'")}
        calls = defaultdict(dict)
        for raw, in db.execute("SELECT value FROM data WHERE kind='token_call'"):
            c = json.loads(raw)
            calls[c['qid']][c['stage'], c['request_id']] = c
        request_usage = {rid: json.loads(raw).get('usage') for rid, raw in db.execute(
            'SELECT id,result FROM requests WHERE result IS NOT NULL')}
        details = []
        usage_receipt_mismatches = 0
        for qid, p in preds.items():
            if not success(p):
                continue
            cc = list(calls[qid].values())
            returned = [c for c in cc if c.get('status') == 'success']
            stages = Counter(c['stage'] for c in returned)
            ambiguous = any(v > 1 for v in stages.values())
            good = [c for c in returned if valid_usage(c.get('usage'))]
            mismatches = sum(c.get('usage') != request_usage.get(c['request_id']) for c in good)
            usage_receipt_mismatches += mismatches
            final_linked = p.get('request_id') in {c['request_id'] for c in returned}
            complete = bool(returned) and len(good) == len(returned) and not ambiguous and not mismatches and final_linked
            details.append(dict(version='formal_v2', dataset=dataset, method=method, qid=qid,
                status=p['status'], complete_usage=complete,
                input_tokens=sum(c['usage']['prompt_tokens'] for c in good),
                output_tokens=sum(c['usage']['completion_tokens'] for c in good), calls=len(returned),
                final_request_linked=final_linked, ambiguous_successful_stage=ambiguous,
                usage_receipt_mismatches=mismatches, unknown_or_failed_calls=len(cc)-len(returned),
                input_components={'qa_llm': sum(c['usage']['prompt_tokens'] for c in good)},
                output_components={'qa_llm': sum(c['usage']['completion_tokens'] for c in good)}))
        statuses = dict(Counter(p.get('status') for p in preds.values()))
    finally:
        db.close()
    summary = aggregate(details, 'formal_v2', dataset, method, config=config,
        source=str(path.relative_to(ROOT)), processed_queries=len(preds), statuses=statuses,
        state='available' if details else 'no_completed_answers',
        usage_receipt_mismatches=usage_receipt_mismatches)
    return summary, details


def p02():
    score_path = P02 / 'SCORES.jsonl'
    score_bytes = score_path.read_bytes()
    scores = [json.loads(x) for x in score_bytes.decode('utf-8-sig').splitlines() if x.strip()]
    needed = {x['id'] for x in scores if success(x)}
    remote = {}
    for path in sorted((P02/'remote_raw').glob('[0-9]*.json')):
        c = read(path)
        if c.get('id') in needed:
            assert c['id'] not in remote, 'Duplicate remote decomposition ID'
            remote[c['id']] = c
    details = []
    for s in scores:
        if not success(s):
            continue
        c = read(ROOT/s['source'])
        assert c['id'] == s['id'] and success(c)
        cc = c.get('logical_calls', [])
        good = [x for x in cc if x.get('status') == 'success' and valid_usage(x.get('usage'))]
        stages = Counter(x['stage'] for x in good)
        dc = remote.get(s['id'], {})
        remote_ok = all(isinstance(dc.get(k), int) and dc[k] >= 0 for k in ('input_tokens', 'output_tokens'))
        remote_ok = remote_ok and dc.get('generation_error') is None and not dc.get('truncated')
        qa_in = sum(x['usage']['prompt_tokens'] for x in good)
        qa_out = sum(x['usage']['completion_tokens'] for x in good)
        final_linked = c.get('request_id') in {x['request_id'] for x in good}
        complete = bool(cc) and len(good) == len(cc) and not any(v>1 for v in stages.values()) and remote_ok and final_linked
        details.append(dict(version='frozen_p02_formal', dataset=s['dataset'], method='p02', qid=s['qid'],
            status=s['status'], source=s['source'], complete_usage=complete,
            input_tokens=qa_in + (dc['input_tokens'] if remote_ok else 0),
            output_tokens=qa_out + (dc['output_tokens'] if remote_ok else 0), calls=len(good)+int(remote_ok),
            qa_calls=len(good), remote_usage_present=remote_ok, final_request_linked=final_linked,
            reused_qa_calls=sum(bool(x.get('reused')) for x in cc),
            input_components={'qa_llm': qa_in, 'decomposer_4b': dc.get('input_tokens') if remote_ok else None},
            output_components={'qa_llm': qa_out, 'decomposer_4b': dc.get('output_tokens') if remote_ok else None}))
    summaries = []
    for ds in DATASETS:
        dsrows = [r for r in details if r['dataset'] == ds]
        ok = [r for r in dsrows if r['complete_usage']]
        summary = aggregate(dsrows, 'frozen_p02_formal', ds, 'p02',
            source=str(score_path.relative_to(ROOT)), score_snapshot_sha256=hashlib.sha256(score_bytes).hexdigest(),
            statuses=dict(Counter(x['status'] for x in scores if x['dataset'] == ds)))
        summary['mean_components'] = {direction: {component: mean(r[direction+'_components'][component] for r in ok)
            if ok else None for component in ('qa_llm', 'decomposer_4b')} for direction in ('input', 'output')}
        summaries.append(summary)
    return summaries, details


def paired(details):
    index = {(r['dataset'], r['method'], r['qid']): r for r in details if r['complete_usage']}
    result = []
    for ds in DATASETS:
        ours = {q: r for (d, m, q), r in index.items() if d == ds and m == 'p02'}
        for m in NAMES:
            if m == 'p02':
                continue
            matches = [(a, index[ds,m,q]) for q,a in ours.items() if (ds,m,q) in index]
            row = dict(dataset=ds, baseline=m, name=NAMES[m], common_completed_queries=len(matches))
            for direction in ('input', 'output'):
                a = mean(x[direction+'_tokens'] for x,y in matches) if matches else None
                b = mean(y[direction+'_tokens'] for x,y in matches) if matches else None
                row[direction] = dict(p02_mean=a, baseline_mean=b, p02_over_baseline=a/b if b else None)
            result.append(row)
    return result


def main():
    started = datetime.now(timezone.utc)
    primary, details = [], []
    for ds in DATASETS:
        configs = read(CAMPAIGN/'campaign_api_v1'/ds/'BASELINE_CONFIGS_FROZEN.json')['selected']
        for m in NAMES:
            if m == 'p02':
                continue
            config = configs[m]
            path = CAMPAIGN/'baseline_integrity_repair_v2_20261017/main'/ds/'formal'/m/config['id']/'checkpoint.sqlite3'
            if m == 'b05_iter_retgen_bge':
                config = {'id': 'k10_t5', 'k': 10, 'iterations': 5}
                path = CAMPAIGN/'baseline_integrity_repair_v2_20261017/matched50'/ds/'formal'/m/'k10_t5/checkpoint.sqlite3'
            summary, rows = baseline(ds, m, config, path)
            primary.append(summary)
            details.extend(rows)
        print('baseline read complete: '+ds, flush=True)
    pp, rows = p02()
    primary.extend(pp)
    details.extend(rows)
    pairs = paired(details)
    result = dict(snapshot_started_utc=started.isoformat(), snapshot_finished_utc=datetime.now(timezone.utc).isoformat(),
        primary=primary, paired_common_completed_queries=pairs,
        method='Sum recorded successful generation calls per completed query; average complete-usage queries. P02 includes its remote 4B decomposition.',
        accounting='Cache reuse retains original receipt usage; incomplete/unknown earlier attempts are not added to selected successful-flow means. No missing usage is treated as zero.',
        exclusions='Embedding and reranker encoder tokenization, failed-attempt overhead, unreturned usage, and future queries are outside these generation-token means.',
        limitations=['P02 decomposer and API reader use different tokenizers; component counts are separate and their sum is operational accounting, not normalized compute or cost.',
            'Each live baseline database is read in a read-only transaction; P02 uses the saved score snapshot and immutable selected case sources. This is not one global instantaneous snapshot.',
            'No old-version scores or usages fill missing formal-v2 cells. Query mix differs in unpaired means; use paired rows for comparisons.',
            'Complete usage means all recorded successful calls have token receipts and a linked final answer; previously failed physical submissions are separate expense.'],
        new_API_calls=0, new_retrieval_calls=0, experiment_sources_modified=False)
    dest = Path(__file__).resolve().parent/'reports'/started.strftime('%Y%m%dT%H%M%S%fZ')
    dest.mkdir(parents=True, exist_ok=False)
    (dest/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (dest/'per_query.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in details), encoding='utf-8')
    report = ['# P02 与九种基线：平均每个 query 的生成 token', '',
        '统计正式评估当前版本；Iter-RetGen 使用 k10/T5。仅统计已完成最终回答且用量完整的题，不按正确性筛选。',
        '每题累加改写、子问题回答、推理、最终回答等所有成功调用；P02 另加入远端4B分解。缓存计入原用量。',
        '输入/输出均为 token/query；n 为有完整用量的题数，每组原计划500题。— 表示无该版本可用完成记录。', '']
    for ds in DATASETS:
        report += ['## '+ds, '', '| 方法 | 平均输入 | 平均输出 | n |', '|---|---:|---:|---:|']
        for m in NAMES:
            r = next(r for r in primary if r['dataset']==ds and r['method']==m)
            a,b = r['mean_input_tokens'],r['mean_output_tokens']
            report.append(f"| {NAMES[m]} | {a:,.2f} | {b:,.2f} | {r['complete_usage_queries']} |" if a is not None else f'| {NAMES[m]} | — | — | 0 |')
        r = next(r for r in pp if r['dataset']==ds)
        if r['complete_usage_queries']:
            comp = r['mean_components']
            report += ['', f"P02 分项：问答模型输入 {comp['input']['qa_llm']:.2f}、输出 {comp['output']['qa_llm']:.2f}；4B分解输入 {comp['input']['decomposer_4b']:.2f}、输出 {comp['output']['decomposer_4b']:.2f}。"]
        report += ['', '| 与 P02 比较的基线 | 同题 n | P02/基线 输入倍数 | P02/基线 输出倍数 |', '|---|---:|---:|---:|']
        for r in pairs:
            if r['dataset']==ds and r['common_completed_queries']:
                report.append(f"| {r['name']} | {r['common_completed_queries']} | {r['input']['p02_over_baseline']:.3f} | {r['output']['p02_over_baseline']:.3f} |")
        report.append('')
    report += ['## 统计边界', '', '两个模型的 tokenizer 不同，4B与API模型用量分项保留；总token不能直接解释为费用或算力倍数。',
        '失败与重试额外消耗、请求结果未知的消耗、检索嵌入和重排序编码token不包含在本表。',
        '不同方法完成题数不同，优先使用逐方法与P02的同题比较。MuSiQue若P02尚无完成结果，则不作P02比较。',
        '只读来源，0新增API，0新增检索；逐题来源及用量覆盖详见 per_query.jsonl 和 summary.json。', '']
    (dest/'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'destination': str(dest), 'rows':len(primary), 'query_rows':len(details)}, ensure_ascii=False), flush=True)
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
