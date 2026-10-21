"""Reconstruct full-method token/call costs separately from marginal reruns."""
from collections import Counter
import json
from . import experiment as e


def aggregate(records):
    usage = Counter()
    stages = Counter()
    for r in records:
        for k in ['prompt_tokens', 'completion_tokens']:
            usage[k] += r.get('usage', {}).get(k, 0)
        stage = r['stage']
        stages['final' if stage.endswith('/final') else 'sentence_format' if 'sentence_format' in stage
               else 'rewrite' if '/rewrite/' in stage else 'answer'] += 1
    return {'logical_generation_calls': len(records), 'by_stage': dict(stages), 'tokens': dict(usage),
            'list_rate_generation_cny': usage['prompt_tokens'] * .8e-6 + usage['completion_tokens'] * 2e-6}


def main():
    plan = e.read(e.HERE / 'PROTOCOL.frozen.json')
    e.verify(plan)
    manifests = e.rows(e.HERE / 'SOURCES.frozen.jsonl')
    parent_calls = []
    all_calls = {a: [] for a in e.ARMS}
    counts = Counter()
    for r in manifests:
        parent = e.read(e.ROOT / r['control'])
        parent_calls.extend(parent['logical_calls'])
        for arm in e.ARMS:
            x = e.chosen(arm, r['id'])
            if not x or not x['status'].startswith('success'):
                continue
            counts[arm] += 1
            sid = len(parent['nodes'])
            excluded = {'legacy/final'}
            if arm == 'g01_goal_consistency':
                excluded |= {f'legacy/answer/{sid}', f'legacy/sentence_format/{sid}'}
            kept = [c for c in parent['logical_calls'] if c['stage'] not in excluded]
            all_calls[arm].extend(kept + x['logical_calls'])
    assert len(parent_calls) == 1774
    result = {'scope': 'Full300 successful-method logical token cost at recorded list rates, as if generated once; includes replayed frozen upstream. Not marginal API fees or cold latency.',
              'parent': aggregate(parent_calls),
              'arms': {a: {'n': counts[a], **aggregate(rr)} for a, rr in all_calls.items() if counts[a] == 300},
              'marginal_budget': e.budget_snapshot(), 'new_api_calls_for_this_report': 0,
              'excluded': ['retrieval hardware cost', 'remote decomposition GPU monetary price', 'provider invoice reconciliation']}
    e.save(e.OUT / 'STANDALONE_COSTS.frozen.json', result)
    e.log('reconstruct_full_method_costs', new_api_calls=0, parent_calls=len(parent_calls),
          arms={a: c['logical_generation_calls'] for a, c in result['arms'].items()})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
