"""Complete B0 trajectory equivalence and inactive-factor replay, no API."""
from collections import Counter
from copy import deepcopy
from .prepare import HERE, OUT, prepare, rows, save, save_rows, log, sha
from .pipeline import execute, c08_prompt_without_filter, c08, ARMS, b0
from .receipts import sources, replay_callbacks
from experiments.query_decomposition.component_chain_opt_v2_20261019.policy import IntermediateLength


def validate():
    plan = prepare(); src = sources(); results = []; counts = Counter()
    # Prompt-only A must preserve prior QA even where C08's original renderer
    # embeds factor D. Demonstrations/instructions remain unchanged.
    h = [{'step_id': 1, 'request': 'Where is X?', 'answer': 'X is in Y.'}]
    assert c08_prompt_without_filter('Q?', 'q?', h, 'p') == c08.child('Q?', 'q?', h, 'p', [])
    h.append({'step_id': 2, 'request': 'Who founded Y?', 'answer': c08.NO_ANSWER})
    assert c08.NO_ANSWER in c08_prompt_without_filter('Q?', 'q?', h, 'p').split('Actual task:')[1]
    assert c08.usable_history(h) == h[:1]
    assert all(not c08.is_no_answer(a) for a in ['0', 'None', 'No', 'X is Y; its founder is unknown.'])
    for case in rows(HERE / 'runtime300.frozen.jsonl'):
        b, retrieve, ask, reference = replay_callbacks(src[case['id']], plan['model'])
        restored = execute(case, retrieve, ask, arm='baseline_replay')
        for k in ['prediction', 'final_prompt', 'final_context', 'helper', 'history_all', 'history_usable', 'original_retrieval']:
            assert restored[k] == b[k], (case['id'], k)
        assert len(restored['nodes']) == len(b['nodes'])
        for a, z in zip(restored['nodes'], b['nodes']):
            assert all(a[k] == v for k, v in z.items()), (case['id'], 'node')
        counts['B0_cases'] += 1; counts['B0_nodes'] += len(b['nodes'])
        counts['B0_logical_calls'] += len(reference['asks'])
        for arm in list(ARMS)[3:]:
            _, r, a, trace = replay_callbacks(src[case['id']], plan['model'])
            output = execute(case, r, a, arm=arm)
            assert trace == reference and output['factor_events'] == [] and output['fallback'] is None
            for k in ['prediction', 'final_prompt', 'final_context', 'helper', 'history_all', 'history_usable']:
                assert output[k] == restored[k], (case['id'], arm, k)
            results.append({'id': case['id'], 'qid': case['qid'], 'dataset': case['dataset'], 'arm': arm,
                            'status': output['status'], 'prediction': output['prediction'],
                            'source': src[case['id']]['source'], 'source_sha256': src[case['id']]['source_sha256'],
                            'evidence': 'All search queries/document order and full API payload sequence identical to B0 in complete offline replay.',
                            'factor_triggered': False, 'new_api_calls': 0})
            counts[arm] += 1
    # R1 implementation is checked synthetically; do not count this as an
    # observed gain. Only child known-length fallback, no semantic retry.
    case = rows(HERE / 'runtime300.frozen.jsonl')[0]
    _, retrieve, ask, trace = replay_callbacks(src[case['id']], plan['model'])
    seen = []
    def truncated(stage, prompt, maximum, **kw):
        seen.append(stage)
        if '/answer/' in stage: raise IntermediateLength(stage, 'synthetic-only', 'truncated')
        assert stage == 'legacy/final'
        # Demonstrations contain helper headings even when actual QA is empty.
        # Check the actual final readout and resulting history instead.
        assert prompt.endswith(b0.final_prompts.FINAL_READOUT)
        return 'synthetic-final', 'synthetic-only', 'stop'
    result = execute(case, retrieve, truncated, arm='F_child_length_fallback')
    assert result['fallback'] and not result['nodes'] and len(seen) == 2
    save_rows(OUT / 'OFFLINE_INACTIVE_ARMS.frozen.jsonl', results)
    report = {'passed': True, 'counts': dict(counts), 'new_api_calls': 0,
              'C08_runtime_filter_removed_from_prompt_arm': True,
              'synthetic_contract_checks_not_runtime_benefits': True,
              'protocol_sha256': sha(HERE / 'PROTOCOL.frozen.json')}
    save(OUT / 'OFFLINE_VALIDATION.frozen.json', report)
    log('complete_offline_replay', **report)
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(validate()))
