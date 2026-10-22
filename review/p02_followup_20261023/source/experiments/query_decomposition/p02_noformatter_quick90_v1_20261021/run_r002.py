"""Run both complete pipelines using read-only exact historical records only."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import socket
import sqlite3
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'experiments.query_decomposition.p02_noformatter_quick90_v1_20261021'

from .prepare import HERE, ROOT, PACKAGE, PLAN_DIR, OUT, DATASETS, METHOD, DECISION_ID, read, rows, sha, write
from src.p02_multigrain_timing_cachefix_v1.core import pipeline as baseline
from src.p02_multigrain_timing_cachefix_v1.core.cache import canonical
from src.p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from . import pipeline as noformatter


def no_network(*args, **kwargs):
    raise RuntimeError('This experiment permits recorded cache replay only; network access is disabled.')


def check_pins(protocol):
    for group in ['inputs', 'source_pins']:
        for name, wanted in protocol[group].items():
            assert sha(HERE / name) == wanted, (group, name)
    for group in ['original_package_pins', 'hold_pins']:
        for name, wanted in protocol[group].items():
            assert sha(ROOT / name) == wanted, (group, name)
    assert read(PLAN_DIR / 'PLAN.json')['execution_approval']['approved'] is False
    amendment = read(HERE / 'EXECUTION_AMENDMENT001.frozen.json')
    assert sha(Path(__file__)) == amendment['new_runner_sha256']


def boundary_contract(config):
    case = {'id': 'hotpotqa:synthetic_formatter_boundary', 'qid': 'synthetic_formatter_boundary',
            'dataset': 'hotpotqa', 'question': 'Where was Alice born?', 'supporting_questions': []}
    packet = {'qid': case['qid'], 'query': case['question'], 'k': 10,
              'docs': [{'doc_id': str(i), 'source': 'synthetic', 'rank': i,
                        'text': 'Alice was born in Paris.', 'block': 'Alice was born in Paris.'} for i in range(1, 11)]}
    results = {}
    for label, module in [('original', baseline), ('noformatter', noformatter)]:
        stages = []
        def ask(stage, prompt, maximum, allow_final_truncation=False):
            stages.append(stage)
            answer = 'Alice was born in Paris.' if '/sentence_format/' in stage else 'Paris'
            return 'synthetic:' + stage, answer, 'stop'
        actual = module.Pipeline(config).execute(baseline.safe_case(case), lambda query: deepcopy(packet), ask)
        results[label] = {'formatter_triggers': sum('/sentence_format/' in s for s in stages),
                          'intermediate_answer': actual['nodes'][0]['answer']}
    assert results['original']['formatter_triggers'] == 1
    assert results['noformatter']['formatter_triggers'] == 0
    assert results['noformatter']['intermediate_answer'] == 'Paris'
    return {'passed': True, 'scope': 'synthetic code-branch contract only; excluded from quality results', **results}


def replay(module, case, db, config, binding):
    trace, queries = [], []
    def retrieve(query):
        row = db.execute('SELECT packet FROM retrieval WHERE dataset=? AND qid=? AND query=?',
                         (case['dataset'], case['qid'], query)).fetchone()
        if row is None:
            raise RuntimeError('New retrieval is forbidden in this replay: ' + case['id'])
        packet = json.loads(row[0]); queries.append(query)
        return packet

    def ask(stage, prompt, maximum, allow_final_truncation=False):
        request = build_payload(config['generation'], prompt, maximum)
        key = canonical(request)
        row = db.execute('SELECT request_id,result FROM requests WHERE payload=?', (key,)).fetchone()
        if row is None:
            raise RuntimeError('New payload requires separate inference approval: ' + case['id'] + ':' + stage)
        rid, stored = row; result = json.loads(stored)
        assert result['status'] == 'success' and result['finish_reason'] == 'stop'
        usage = result.get('usage')
        assert usage and all(k in usage for k in ['prompt_tokens', 'completion_tokens'])
        if stage == 'legacy/final':
            assert rid == binding['final_request_id'], ('Final receipt query binding mismatch', case['id'])
        trace.append({'stage': stage, 'request_id': rid,
                      'payload_sha256': hashlib.sha256(key.encode('utf-8')).hexdigest(), 'usage': usage})
        return rid, result['answer'].strip(), result['finish_reason']

    actual = module.Pipeline(config).execute(baseline.safe_case(case), retrieve, ask)
    usage = {'input': sum(v['usage']['prompt_tokens'] for v in trace),
             'output': sum(v['usage']['completion_tokens'] for v in trace)}
    return actual, trace, queries, usage


def ledger_snapshot():
    path = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/budget.sqlite3'
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        n, actual, reserve = db.execute('SELECT count(*),sum(actual),sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END) FROM charges').fetchone()
    return {'rows': n, 'known_CNY': actual, 'unknown_reserved_CNY': reserve}


def main():
    socket.socket.connect = no_network
    socket.create_connection = no_network
    socket.getaddrinfo = no_network
    protocol = read(HERE / 'PROTOCOL.frozen.json')
    check_pins(protocol)
    if (OUT / 'REPORT.json').exists():
        raise FileExistsError('Preserve completed quick90; do not overwrite prior results.')
    before = ledger_snapshot()
    cases = rows(HERE / 'runtime90.frozen.jsonl')
    assert len(cases) == 90 and Counter(v['dataset'] for v in cases) == {ds: 30 for ds in DATASETS}
    expected = {v['id']: v for v in rows(PACKAGE / 'data/expected300.jsonl')}
    bindings = {v['id']: v for v in rows(HERE / 'SOURCE_BINDINGS.frozen.jsonl')}
    config = read(PACKAGE / 'config.json')
    contract = boundary_contract(config)
    records, trajectories = [], []
    counts = Counter()
    with sqlite3.connect((PACKAGE / 'data/replay.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        for case in cases:
            binding = bindings[case['id']]
            assert sha(ROOT / binding['source']) == binding['source_sha256']
            source = expected[case['id']]
            original, calls_a, queries_a, usage_a = replay(baseline, case, db, config, binding)
            variant, calls_b, queries_b, usage_b = replay(noformatter, case, db, config, binding)
            for key in ['prediction', 'final_prompt', 'final_context', 'finish_reason', 'original_retrieval']:
                assert original[key] == source[key], (case['id'], 'original_record_mismatch', key)
                assert variant[key] == original[key], (case['id'], 'variant_difference', key)
            assert len(original['nodes']) == len(source['nodes'])
            for actual_node, recorded_node in zip(original['nodes'], source['nodes']):
                for key, value in recorded_node.items():
                    assert actual_node.get(key) == value, (case['id'], 'recorded_node_field_mismatch', key)
            assert variant['nodes'] == original['nodes'], (case['id'], 'variant_nodes_differ')
            assert calls_a == calls_b and queries_a == queries_b and usage_a == usage_b
            assert all(not n.get('sentence_format_request_id') for n in original['nodes'])
            assert not any('/sentence_format/' in v['stage'] for v in calls_b)
            counts['queries'] += 1; counts['nodes'] += len(original['nodes'])
            records.append({'id': case['id'], 'qid': case['qid'], 'dataset': case['dataset'], 'type': case['type'],
                            'baseline_prediction': original['prediction'], 'variant_prediction': variant['prediction'],
                            'complete_flow_exact': True, 'formatter_triggered': False,
                            'hosted_LLM_logical_tokens': usage_a, 'bound_receipts': calls_a,
                            'retrieval_queries': queries_a, 'canonical_source': binding['source']})
            trajectories += [{'id': case['id'], 'arm': 'original', 'trajectory': original},
                             {'id': case['id'], 'arm': 'noformatter', 'trajectory': variant}]

    # Native metrics and gold are opened only after both methods finish all flows.
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    references = {v['id']: v for v in rows(HERE / 'references90.local.frozen.jsonl')}
    scorers = {ds: metric(ds) for ds in DATASETS}
    for item in records:
        score = scorers[item['dataset']]
        ref = references[item['id']]
        a = score(item['baseline_prediction'], ref); b = score(item['variant_prediction'], ref)
        assert a == b
        assert abs(a['f1'] - expected[item['id']]['native_answer_f1']) < 1e-12
        item.update(baseline_metrics=a, variant_metrics=b)
    groups = {}
    for ds in DATASETS:
        vv = [v for v in records if v['dataset'] == ds]
        groups[ds] = {'planned': 30, 'complete': len(vv),
            **{arm: {'F1': sum(v[arm + '_metrics']['f1'] for v in vv) / len(vv),
                     'EM': sum(v[arm + '_metrics']['em'] for v in vv) / len(vv)}
               for arm in ['baseline', 'variant']},
            'delta_F1': 0.0, 'delta_EM': 0.0,
            'hosted_LLM_logical_input_tokens_per_query': sum(v['hosted_LLM_logical_tokens']['input'] for v in vv) / len(vv),
            'hosted_LLM_logical_output_tokens_per_query': sum(v['hosted_LLM_logical_tokens']['output'] for v in vv) / len(vv)}
    after = ledger_snapshot()
    check_pins(protocol)
    assert before == after, ('Global billing changed during an offline-only test; investigate concurrency', before, after)
    report = {'id': HERE.name, 'utc': datetime.now(timezone.utc).isoformat(), 'passed': True, 'status': 'completed',
        'execution': protocol['execution'], 'scope': 'Previously exposed quick90 development subset; old P02 protocol, not shared-layer formal evaluation or fresh inference.',
        'cohort_sha256': sha(HERE / 'COHORT.frozen.json'), 'by_dataset': groups,
        'overall': {arm: {'F1': sum(v[arm + '_metrics']['f1'] for v in records) / 90,
                          'EM': sum(v[arm + '_metrics']['em'] for v in records) / 90}
                    for arm in ['baseline', 'variant']},
        'exact_flow_matches': counts['queries'], 'node_records_verified': counts['nodes'],
        'old_formatter_triggers': 0, 'variant_formatter_requests': 0, 'synthetic_boundary_contract': contract,
        'new_API_requests': 0, 'new_GPU_jobs': 0, 'new_actual_API_expense_CNY': 0,
        'hosted_LLM_logical_tokens_delta': {'input': 0, 'output': 0},
        'cost_scope': protocol['cost_scope'], 'fresh_component_processing_time': None, 'fresh_client_TTFT': None,
        'ledger_before': before, 'ledger_after': after, 'original_package_hashes_unchanged': True,
        'all_hold_files_unchanged': True, 'main_evaluation_remains_paused': True,
        'interpretation': 'Removing the dormant formatter leaves all 90 recorded complete trajectories identical. This is exact replay equivalence, not fresh generation or evidence of quality/latency improvement.'}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'REPORT.json', report)
    (OUT / 'DETAILS.jsonl').write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in records), encoding='utf-8')
    with gzip.open(OUT / 'TRAJECTORIES.jsonl.gz', 'wt', encoding='utf-8') as f:
        for v in trajectories:
            f.write(json.dumps(v, ensure_ascii=False) + '\n')
    lines = ['# P02 去formatter：开发快速90题对照', '',
        '执行方式：完整流程严格同输入缓存回放，不是新推理。各数据集固定30题；原生Answer F1/EM，百分制。', '',
        '| 数据集 | 原版F1 | 去formatter F1 | 原版EM | 去formatter EM |',
        '| --- | ---: | ---: | ---: | ---: |']
    for ds, g in groups.items():
        lines.append(f"| {ds} | {100*g['baseline']['F1']:.4f} | {100*g['variant']['F1']:.4f} | {100*g['baseline']['EM']:.4f} | {100*g['variant']['EM']:.4f} |")
    lines += ['', '90/90题的检索、完整生成payload、回执、中间回答、最终prompt与答案完全一致。',
              f"核对{counts['nodes']}个节点，原版formatter触发0次；合成短答案检查确认变体已经移除该分支。",
              '输入/输出tokens差异为0；新增API费用0元；没有开展新鲜计时或GPU推理。',
              '结论只支持这些固定历史轨迹上的等价，不证明新生成短答案场景下去除formatter也无影响。',
              '冻结原包保持原样，其他正式评估和计时补测继续暂停。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (PLAN_DIR / 'DECISIONS.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'id': DECISION_ID + '_COMPLETED', 'utc': report['utc'],
            'status': 'completed_scoped_quick90_exact_replay', 'planned': 90, 'completed': 90,
            'receipt': str((OUT / 'REPORT.json').relative_to(ROOT)).replace('\\', '/'),
            'new_API_requests': 0, 'new_GPU_jobs': 0, 'main_evaluation_remains_paused': True}, ensure_ascii=False) + '\n')
    print(json.dumps({'passed': True, 'by_dataset': groups, 'node_records_verified': counts['nodes'],
                      'new_API_requests': 0, 'main_evaluation_remains_paused': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
