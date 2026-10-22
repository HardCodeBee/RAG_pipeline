"""Preserve r001 and correct a legacy-export metadata comparison in r002."""
from datetime import datetime, timezone
from pathlib import Path
import json

from .prepare import HERE, ROOT, OUT, PLAN_DIR, DECISION_ID, read, write, sha


def main():
    amendment = HERE / 'EXECUTION_AMENDMENT001.frozen.json'
    if amendment.exists():
        raise FileExistsError(amendment)
    source = (HERE / 'run.py').read_text(encoding='utf-8-sig')
    old = '''            for key in ['prediction', 'final_prompt', 'final_context', 'finish_reason', 'original_retrieval', 'nodes']:
                assert original[key] == source[key], (case['id'], 'original_record_mismatch', key)
                assert variant[key] == original[key], (case['id'], 'variant_difference', key)
'''
    new = '''            for key in ['prediction', 'final_prompt', 'final_context', 'finish_reason', 'original_retrieval']:
                assert original[key] == source[key], (case['id'], 'original_record_mismatch', key)
                assert variant[key] == original[key], (case['id'], 'variant_difference', key)
            assert len(original['nodes']) == len(source['nodes'])
            for actual_node, recorded_node in zip(original['nodes'], source['nodes']):
                for key, value in recorded_node.items():
                    assert actual_node.get(key) == value, (case['id'], 'recorded_node_field_mismatch', key)
            assert variant['nodes'] == original['nodes'], (case['id'], 'variant_nodes_differ')
'''
    assert source.count(old) == 1
    updated = source.replace(old, new, 1)
    needle = "    assert read(PLAN_DIR / 'PLAN.json')['execution_approval']['approved'] is False\n"
    replacement = needle + "    amendment = read(HERE / 'EXECUTION_AMENDMENT001.frozen.json')\n" + \
        "    assert sha(Path(__file__)) == amendment['new_runner_sha256']\n"
    assert updated.count(needle) == 1
    updated = updated.replace(needle, replacement, 1)
    (HERE / 'run_r002.py').write_text(updated, encoding='utf-8')
    write(amendment, {'id': DECISION_ID + '_EXECUTION_AMENDMENT001',
        'utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'The historical expected export omits raw_B0_child_prompt for goal nodes. All recorded fields match. Exact dict equality incorrectly rejected this additional audit-only field.',
        'change': 'Use the frozen P02 replay rule: verify every stored node field, then compare the two newly executed node dictionaries in full.',
        'original_runner': 'run.py', 'original_runner_sha256': sha(HERE / 'run.py'),
        'new_runner': 'run_r002.py', 'new_runner_sha256': sha(HERE / 'run_r002.py'),
        'cohort_unchanged': True, 'method_unchanged': True, 'cache_policy_unchanged': True,
        'new_API_requests': 0, 'main_evaluation_remains_paused': True})
    failure = OUT / 'attempts/r001/FAILURE.json'
    failure.parent.mkdir(parents=True, exist_ok=True)
    write(failure, {'status': 'validation_stopped_before_scoring',
        'case': 'hotpotqa:5addcfd65542997dc790703e',
        'error': 'original_record_mismatch:nodes', 'cause': 'Additional goal raw_B0_child_prompt metadata absent from historical export.',
        'all_stored_node_fields_equal': True, 'no_quality_results_published': True,
        'new_API_requests': 0, 'new_GPU_jobs': 0,
        'original_runner_sha256': sha(HERE / 'run.py'),
        'correction': str(amendment.relative_to(ROOT)).replace('\\', '/')})
    registry = read(HERE / 'experiment.json')
    for name in ['run_r002.py', 'amend_execution.py']:
        registry['files'].append({'path': str((HERE / name).relative_to(ROOT)).replace('\\', '/'),
            'role': 'Preserved execution amendment and corrected exact replay validator', 'sha256': sha(HERE / name)})
    registry['commands']['run'] = str((HERE / 'run_r002.py').relative_to(ROOT)).replace('\\', '/')
    registry['evidence_paths'].append(str(amendment.relative_to(ROOT)).replace('\\', '/'))
    write(HERE / 'experiment.json', registry)
    p = HERE / 'README.md'
    text = p.read_text(encoding='utf-8-sig').replace('p02_noformatter_quick90_v1_20261021.run\n',
                                               'p02_noformatter_quick90_v1_20261021.run_r002\n')
    text += '\n执行修正001：保留首轮run.py及停止记录；r002按冻结原包的回放规则核对历史节点已存字段，同时严格比较两版新执行的全部节点字段。仅纠正历史导出省略审计字段造成的误报，题目与算法不变。\n'
    p.write_text(text, encoding='utf-8')
    with (PLAN_DIR / 'DECISIONS.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'id': DECISION_ID + '_EXECUTION_AMENDMENT001',
            'utc': datetime.now(timezone.utc).isoformat(), 'status': 'offline_validation_implementation_fix',
            'scope': 'Same approved quick90, same algorithms and receipts; preserve failed runner and use existing frozen-package field comparison semantics.',
            'new_evaluation_decision': False, 'receipt': str(amendment.relative_to(ROOT)).replace('\\', '/'),
            'new_API_requests': 0, 'main_evaluation_remains_paused': True}, ensure_ascii=False) + '\n')
    print('Preserved r001; created pinned r002 validator. Cohort, method, prompts and pauses unchanged.')


if __name__ == '__main__':
    main()
