"""All300 actual p02 prompts/receipts; only example role headings may change.

No fresh90/confirmation/reference data, provider, retrieval, training or SSH.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, digest
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.receipts import receipt
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
from .strategy import ACTUAL_MARKER, ARMS, EXAMPLE_PATH, aligned_examples, p02, render_final

HERE = Path(__file__).resolve().parent
PARENT_E = HERE.parent / 'b0_strategy_round2_v1_20261020'
PARENT_O = OUTPUT / PARENT_E.name
ORDINARY = 'Intermediate candidate results\n\n'
GOAL = 'Candidate conclusion for the Original question\n\n'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_path(dataset, qid):
    for path in sorted((PARENT_O / 'rounds').glob(f'*/cases/{dataset}/p02_role_headers/{qid}.json')):
        value = read(path)
        if value['status'].startswith('success') and isinstance(value.get('prediction'), str):
            return path, value
    raise AssertionError('Missing completed p02 case: ' + dataset + ':' + qid)


def run():
    protocol = read(PARENT_E / 'PROTOCOL.frozen.json')
    cases = rows(PARENT_E / 'runtime300.frozen.jsonl')
    source = {v['id']: v for v in rows(PARENT_E / 'SOURCES.frozen.jsonl')}
    assert len(cases) == len(source) == 300
    assert Counter(c['dataset'] for c in cases) == {'hotpotqa': 100, '2wiki': 100, 'musique': 100}
    assert all(not {'answer', 'answers', 'gold', 'prediction', 'supporting_facts', 'f1'}.intersection(c) for c in cases)
    examples = read(EXAMPLE_PATH)
    example_hash = sha(EXAMPLE_PATH)
    aligned = aligned_examples(examples)
    examples_review = []
    for old, new in zip(examples, aligned):
        assert new['context'].replace(ORDINARY, '').replace(GOAL, '') == old['context']
        old_docs = old['context'].split('Auxiliary question-answer pairs', 1)[0]
        assert new['context'].split('Auxiliary question-answer pairs', 1)[0] == old_docs
        assert {k: v for k, v in old.items() if k != 'context'} == {k: v for k, v in new.items() if k != 'context'}
        assert new['context'].count(ORDINARY) == int(len(old['qas']) > 1)
        assert new['context'].count(GOAL) == 1
        examples_review.append({'source_qid': old['source_qid'], 'QA_count': len(old['qas']),
                                'ordinary_count': len(old['qas']) - 1, 'goal_count': 1,
                                'last_request_equals_question': True, 'docs_first_unchanged': True,
                                'all_QA_and_final_example_answers_unchanged': True})
    assert sum(r['ordinary_count'] > 0 for r in examples_review) == 3
    assert sum(r['goal_count'] for r in examples_review) == 4
    counts, details, hashes = Counter(), [], {}
    for case in cases:
        src = source[case['id']]
        base_path = ROOT / src['base']
        assert sha(base_path) == src['base_sha256']
        base = read(base_path)
        path, parent = parent_path(case['dataset'], case['qid'])
        hashes[str(path.relative_to(ROOT))] = sha(path)
        assert parent['id'] == case['id'] and parent['question'] == case['question']
        prompt, context = render_final(case['question'], parent['nodes'], base['original_retrieval'], protocol['readout'], 'parent_p02')
        assert prompt == parent['final_prompt'] and context == parent['final_context']
        req = payload(protocol['model'], prompt, 64)
        call = next(row for row in parent['logical_calls'] if row['payload_sha256'] == digest(req))
        actual = receipt(call, req)
        assert actual['status'] == 'success' and actual['result']['status'] == 'success'
        assert actual['result']['answer'].strip() == parent['prediction']
        counts['parent_prompts_and_contexts_exact'] += 1
        counts['parent_actual_final_receipts_exact'] += 1
        changed, changed_context = render_final(case['question'], parent['nodes'], base['original_retrieval'], protocol['readout'], ARMS[0])
        assert changed_context == context
        before_prefix, before_actual = prompt.split(ACTUAL_MARKER, 1)
        after_prefix, after_actual = changed.split(ACTUAL_MARKER, 1)
        assert before_actual == after_actual
        assert before_prefix.split('\nWorked examples:', 1)[0] == after_prefix.split('\nWorked examples:', 1)[0]
        assert after_prefix.replace(ORDINARY, '').replace(GOAL, '') == before_prefix
        assert after_prefix.count(ORDINARY) - before_prefix.count(ORDINARY) == 3
        assert after_prefix.count(GOAL) - before_prefix.count(GOAL) == 4
        assert Counter(changed) == Counter(prompt) + Counter(ORDINARY * 3 + GOAL * 4)
        assert changed.endswith(protocol['readout'])
        counts['actual_task_and_P1_readout_byte_exact'] += 1
        counts['only_three_ordinary_four_goal_example_headers_added'] += 1
        counts['original_top10_docs_retained'] += 10
        counts['all_actual_auxiliary_QAs_retained'] += len(parent['nodes'])
        details.append({'id': case['id'], 'parent_source': str(path.relative_to(ROOT)),
                        'parent_sha256': sha(path), 'parent_payload_sha256': digest(req),
                        'candidate_payload_sha256': digest(payload(protocol['model'], changed, 64)),
                        'actual_task_sha256': hashlib.sha256(before_actual.encode('utf-8')).hexdigest(),
                        'added_header_characters': len(changed) - len(prompt)})
    assert sha(EXAMPLE_PATH) == example_hash
    for path, value in hashes.items():
        assert sha(ROOT / path) == value
    report = {'passed': True, 'arms': ARMS, 'parent': 'frozen p02 actual300', 'counts': dict(counts),
              'example_review': examples_review,
              'change': 'Only worked-example QA role headings; three intermediate headings and four goal-candidate headings.',
              'preserved': 'All4 few-shot tasks, docs-first layout, QA contents/order, target answers, instructions, actual task/context and P1 readout.',
              'scope': 'Example role-heading alignment, not full layout alignment; examples remain docs-first while actual remains helper-first.',
              'interpretation': 'Adds role-format cues and input tokens but no facts, rules, examples, retrieval or model calls.',
              'reference_boundary': 'No fresh90/confirmation predictions or scores, gold references or qrels were opened.',
              'source_pins': {str(p.relative_to(ROOT)): sha(p) for p in [
                  Path(__file__), HERE / 'strategy.py', PARENT_E / 'strategy.py', PARENT_E / 'PROTOCOL.frozen.json',
                  PARENT_E / 'runtime300.frozen.jsonl', PARENT_E / 'SOURCES.frozen.jsonl', EXAMPLE_PATH]},
              'protected_parent_cases_unchanged': len(hashes),
              'new_api_calls': 0, 'new_retrieval_calls': 0, 'new_remote_calls': 0, 'cases': details}
    (HERE / 'OFFLINE_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    run()
