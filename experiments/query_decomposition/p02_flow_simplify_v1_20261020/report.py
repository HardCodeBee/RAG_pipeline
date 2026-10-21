"""Live, offline reports for actual p02 flow-simplification execution.

Reads immutable completed case files and read-only SQLite snapshots; does not
construct an API client or a retriever. Infrastructure missingness and recorded
algorithm failures remain distinct throughout the report.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest
from . import pipeline
from .offline_audit import stored_request
from .score import ROOT, HERE, OUT, OLD, DATASETS, read, rows, sha, summarize, select_case, is_scorable

ARMS = ('union20', 'discount20', 'single_final')
OUTPUT_ROOT = OUT.parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=20)


def _arm_from_namespace(namespace):
    parts = [part for part in namespace.replace('\\', '/').split('/') if part]
    if 'methods' in parts:
        i = parts.index('methods')
        if len(parts) > i + 2:
            return parts[i + 2], parts[i + 1]
    return 'other', None


def budget_snapshot():
    """Actual ledger reservations/returned-usage estimates, not invoice claims."""
    path = OUTPUT_ROOT / 'budget.sqlite3'
    with ro(path) as db:
        db.execute('BEGIN')
        records = list(db.execute('SELECT id,namespace,reserved,actual,status FROM charges'))
    prefix = OUT.name
    own = [row for row in records if row[1].startswith(prefix)]

    def stats(items):
        known = sum(row[3] or 0 for row in items)
        reserve = sum(row[2] for row in items if row[3] is None)
        return {'submissions': len(items), 'known_returned_usage_cny': known,
                'unknown_reserved_cny': reserve, 'occupied_cny': known + reserve,
                'unreconciled_requests': sum(row[3] is None for row in items),
                'statuses': dict(Counter(row[4] for row in items))}

    by_arm = {arm: stats([row for row in own if _arm_from_namespace(row[1])[0] == arm]) for arm in ARMS}
    by_dataset = {ds: stats([row for row in own if _arm_from_namespace(row[1])[1] == ds]) for ds in DATASETS}
    return {'database': str(path), 'experiment_namespace_prefix': prefix,
            'experiment': stats(own), 'by_arm': by_arm, 'by_dataset': by_dataset,
            'global': stats(records), 'global_cap_cny': 300,
            'actual_bill_verified': False,
            'meaning': 'Known costs are returned-usage estimates recorded by the budget ledger; unknown liabilities remain reserved. Cache replays do not add ledger charges.'}


def request_snapshot():
    """Includes in-progress method stores before a full query case is saved."""
    by_arm = {arm: Counter() for arm in ARMS}
    by_dataset = {ds: Counter() for ds in DATASETS}
    by_arm_status = {arm: Counter() for arm in ARMS}
    source_count = 0
    for path in sorted(OUT.glob('rounds/r*/methods/*/*/checkpoint.sqlite3')):
        arm, ds = path.parent.name, path.parent.parent.name
        if arm not in ARMS or ds not in DATASETS:
            continue
        source_count += 1
        with ro(path) as db:
            values = list(db.execute('SELECT id,status,result FROM requests'))
            logical = list(db.execute("SELECT value FROM data WHERE kind='logical_call'"))
        for _, status, raw in values:
            by_arm_status[arm][status] += 1
            for group in [by_arm[arm], by_dataset[ds]]:
                group['new_request_rows'] += 1
            value = json.loads(raw) if raw else {}
            usage = value.get('usage') or {}
            for group in [by_arm[arm], by_dataset[ds]]:
                if status == 'success':
                    group['successful_new_request_rows'] += 1
                group['known_new_request_prompt_tokens'] += usage.get('prompt_tokens', 0) or 0
                group['known_new_request_completion_tokens'] += usage.get('completion_tokens', 0) or 0
        for (raw,) in logical:
            value = json.loads(raw)
            for group in [by_arm[arm], by_dataset[ds]]:
                group['recorded_logical_call_events'] += 1
                if value.get('reused'):
                    group['recorded_reused_call_events'] += 1
    return {'method_stores': source_count,
            'by_arm': {arm: {**dict(value), 'request_statuses': dict(by_arm_status[arm])} for arm, value in by_arm.items()},
            'by_dataset': {ds: dict(value) for ds, value in by_dataset.items()},
            'scope': 'All task method stores, including currently running cases and both immutable retry rounds; these are experiment-wide totals rather than a pilot-only payment allocation.'}


def _recorded_packets(value):
    values = []
    if value.get('original_retrieval'):
        values.append(value['original_retrieval'])
    for node in value.get('nodes', []):
        values += [packet for packet in [node['original_retrieval'], node.get('rewritten_retrieval')] if packet]
    final = value.get('final_retrieval')
    if final:
        values += [packet for packet in [final['original_retrieval'], final.get('rewritten_retrieval')] if packet]
    result = {}
    for packet in values:
        if packet['query'] in result:
            assert result[packet['query']] == packet, 'Conflicting actual packets for the same query'
        result[packet['query']] = packet
    return result


def audit_case(case, arm, path, value, plan):
    """Rebuild successful complete trajectories from their actual receipts.

    Exact payload matching checks that no old downstream answer is silently
    attached to a changed upstream question, history or document context.
    """
    result = {'id': case['id'], 'arm': arm, 'status': value['status'],
              'source': str(path.relative_to(ROOT)), 'case_sha256': sha(path),
              'completed_nodes': len(value.get('nodes', [])),
              'full_receipt_replay_verified': False, 'errors': []}
    try:
        blocks = 0
        history = []
        for node in value.get('nodes', []):
            assert node['prior_complete_answers'] == history, 'Reader history differs from actual preceding answers'
            merged = pipeline.merge(node['original_retrieval'], node.get('rewritten_retrieval'), arm=arm)
            assert merged == node['reader_packet'], 'Actual merged packet differs from frozen merge rule'
            for doc in merged['docs']:
                assert doc['block'] in node['child_prompt'], 'Merged candidate block missing from actual child prompt'
                blocks += 1
            history.append({key: node[key] for key in ['step_id', 'request', 'answer']})
        if value.get('final_prompt'):
            final_packet = value['final_reader_packet']
            if arm == 'single_final':
                final = value['final_retrieval']
                assert final['prior_complete_answers'] == history
                assert pipeline.merge(final['original_retrieval'], final.get('rewritten_retrieval'), arm=arm) == final_packet
                assert all(node['role'] == 'ordinary' for node in value['nodes'])
                assert 'Candidate conclusion for the Original question' not in value['helper']
            for doc in final_packet['docs']:
                assert doc['block'] in value['final_context'] and doc['block'] in value['final_prompt']
                blocks += 1
        result['actual_reader_blocks_verified'] = blocks
        result['all_saved_completed_reader_blocks_present'] = True
        receipt_map = {}
        for call in value.get('logical_calls', []):
            stored = stored_request(call['source'], call['request_id'])
            assert digest(stored['payload']) == call['payload_sha256'], 'Logical-call hash differs from stored full payload'
            assert stored['status'] == call['status'], 'Immutable call receipt status differs'
            if call['status'] == 'success':
                receipt_map.setdefault(call['payload_sha256'], {**call, **stored})
        result['saved_logical_call_payloads_checked'] = len(value.get('logical_calls', []))
        if value['status'].startswith('success'):
            packets = _recorded_packets(value)
            seen_calls, seen_queries = [], []

            def retrieve(query):
                assert query in packets, 'Reconstructed history caused a different retrieval query'
                seen_queries.append(query)
                return deepcopy(packets[query])

            def ask(stage, prompt, maximum, allow_final_truncation=False):
                request = payload(plan['model'], prompt, maximum)
                key = digest(request)
                assert key in receipt_map, 'No exact successful actual payload for reconstructed stage ' + stage
                receipt = receipt_map[key]
                assert receipt['payload'] == request
                seen_calls.append((stage, key, receipt['request_id']))
                return use_receipt(receipt, stage, allow_final_truncation)

            rebuilt = pipeline.execute(case, retrieve, ask, arm=arm, readout=plan['readout'])
            for key, item in rebuilt.items():
                assert value.get(key) == item, 'Actual/reconstructed field mismatch: ' + key
            assert len(seen_calls) == len(value['logical_calls']), 'Different number of actual reconstructed calls'
            actual_calls = [(call['stage'], call['payload_sha256'], call['request_id']) for call in value['logical_calls']]
            assert seen_calls == actual_calls, 'Different chronological actual call lineage'
            assert seen_queries == [item['query'] for item in value['logical_retrievals']], 'Different chronological actual query lineage'
            result.update(full_receipt_replay_verified=True, exact_replayed_calls=len(seen_calls),
                          exact_replayed_retrievals=len(seen_queries),
                          no_unsupported_old_downstream_answer_substitution=True)
        else:
            result['partial_record_note'] = 'Only saved completed nodes and recorded payload hashes checked; a full generated final trajectory is not claimed.'
    except Exception as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
    return result


def _f1(value):
    return '—' if value is None else f'{value * 100:.4f}'


def _signed(value):
    return '—' if value is None else f'{value * 100:+.4f}'


def report(*, sample='pilot60', output_dir=None):
    started_utc = utc()
    sample_path = HERE / ('pilot60.runtime.frozen.jsonl' if sample == 'pilot60' else 'runtime300.frozen.jsonl')
    samples, plan = rows(sample_path), read(HERE / 'PROTOCOL.frozen.json')
    scores = summarize(OUT, sample_path, arms=ARMS, references=HERE / 'references300.local.frozen.jsonl', baseline_root=OLD)
    budget, requests = budget_snapshot(), request_snapshot()
    audits = []
    for case in samples:
        for arm in ARMS:
            path, value = select_case(OUT, case['dataset'], arm, case['qid'])
            if value:
                audits.append(audit_case(case, arm, path, value, plan))
    counts = Counter()
    for item in audits:
        counts['case_records_checked'] += 1
        counts['full_successful_trajectories_verified'] += item['full_receipt_replay_verified']
        counts['actual_reader_blocks_verified'] += item.get('actual_reader_blocks_verified', 0)
        counts['saved_logical_call_payloads_checked'] += item.get('saved_logical_call_payloads_checked', 0)
        counts['records_with_errors'] += bool(item['errors'])
    existing_score_path = OUT / ('SCORES_' + sample + '_LATEST.json')
    value = {'utc': utc(), 'snapshot_started_utc': started_utc,
             'live_snapshot_note': 'Case files, costs and active request stores are read sequentially while execution may continue; their snapshots can differ slightly during this compilation interval.',
             'sample': sample, 'scores': scores, 'budget': budget, 'requests': requests,
             'existing_score_snapshot': {'path': str(existing_score_path), 'sha256': sha(existing_score_path)} if existing_score_path.exists() else None,
             'reader_and_lineage_audit': {'counts': dict(counts),
                  'all_saved_records_checked_without_error': not any(item['errors'] for item in audits),
                  'all_planned_cases_finished': all(scores['arms'][arm]['missing'] == 0 for arm in ARMS),
                  'details': audits},
             'new_api_calls_from_reporting': 0, 'new_index_searches_from_reporting': 0,
             'report_code_sha256': sha(Path(__file__)),
             'scope': 'Development evaluation. Partial available-only means are not complete planned-cohort results. Algorithm failures score zero; infrastructure missingness retains bounds.'}
    lines = [f'p02流程简化：{sample}实时报告', f'UTC: {value["utc"]}',
             f'快照读取起点：{started_utc}；执行进行中时，各项顺序读取的快照时间可能略有不同。',
             '最终Answer F1单位为%；部分完成成绩仅为可计分题均值，非完整分母成绩。',
             '运行初期可计分题可能集中在一个数据集；须看各数据集完成量，不以部分总体均值判定方法优劣。',
             '算法失败计0；基础设施未完成不填0，单独保留缺失及全分母边界。', '',
             '配置\t可计分/计划\t总体F1(可计分)\t完整F1\tHotpotQA\t2Wiki\tMuSiQue\t相对p02同题Δ\t升/平/降\t相对直接父配置同题Δ\t父比较n']
    for arm in ARMS:
        item = scores['arms'][arm]
        paired, parent = item['paired_with_frozen_p02'], item['paired_with_direct_parent']
        ds = item['by_dataset']
        lines.append('\t'.join([arm, f'{item["scorable"]}/{item["planned"]}', _f1(item['F1_available']), _f1(item['F1_complete']),
                                 *[_f1(ds[key]['F1_available']) for key in DATASETS], _signed(paired['delta']),
                                 f'{paired["up"]}/{paired["tie"]}/{paired["down"]}', _signed(parent['delta']), str(parent['n'])]))
        lines += [f'  状态：{json.dumps(item["statuses"], ensure_ascii=False)}',
                  f'  全计划F1边界：{[_f1(number) for number in item["F1_all_planned_bounds"]]}',
                  f'  所选题记录的成功逻辑调用/输入/输出tokens：{json.dumps(item["logical_usage_on_selected_case_records"], ensure_ascii=False)}']
        for key in DATASETS:
            d = ds[key]
            p, b = d['paired_with_direct_parent'], d['paired_with_frozen_p02']
            lines.append(f'  {key}: {d["scorable"]}/{d["planned"]}; F1={_f1(d["F1_available"])}; p02 Δ={_signed(b["delta"])} n={b["n"]} 升/平/降={b["up"]}/{b["tie"]}/{b["down"]}; 父配置 Δ={_signed(p["delta"])} n={p["n"]} 升/平/降={p["up"]}/{p["tie"]}/{p["down"]}')
    lines += ['', '本实验实际新增请求费用（预算账本返回用量估计，未核对账单；缓存不重复计费）：',
              json.dumps(budget, ensure_ascii=False, indent=2), '',
              '未核销预留包含正在请求中的费用，不等同于全部请求已异常丢失。',
              '正在运行及已结束方法SQLite的请求/token汇总（全任务含重试）：',
              json.dumps(requests, ensure_ascii=False, indent=2), '',
              '实际材料曝光及回执链路核查：', json.dumps(dict(counts), ensure_ascii=False),
              '成功记录按真实前置答案、真实检索包及全payload回执完整重执行；未完成记录只核查已保存节点，不假称完整通过。',
              f'有错误的记录数：{counts["records_with_errors"]}',
              '本报告未新增API调用或索引检索。']
    if any(item['errors'] for item in audits):
        lines += ['', '核查错误：']
        lines += [json.dumps(item, ensure_ascii=False) for item in audits if item['errors']]
    output_dir = Path(output_dir or OUT / 'reports' / sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Report paths are independent of the immutable runner or scorer outputs.
    (output_dir / 'REPORT.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (output_dir / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    return value, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', choices=['pilot60', '300'], default='pilot60')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    value, output_dir = report(sample=args.sample, output_dir=args.output_dir)
    compact = {'sample': args.sample,
               'arms': {arm: {key: value['scores']['arms'][arm][key] for key in ['planned', 'scorable', 'missing', 'F1_available', 'F1_complete', 'statuses']} for arm in ARMS},
               'budget': value['budget']['experiment'],
               'audit': value['reader_and_lineage_audit']['counts'],
               'output_directory': str(output_dir)}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == '__main__':
    main()
