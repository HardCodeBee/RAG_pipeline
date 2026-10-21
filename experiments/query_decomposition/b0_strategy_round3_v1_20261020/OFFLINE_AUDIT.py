"""Rebuild all300 p02 prompts/receipts; audit reorder and goal-evidence arms.

Reads old300 runtime/source manifests and round2 rounds only. It never opens
fresh_dev90, confirmation inputs/results, score references, qrels, or native
gold. This checks execution identity/input changes, not answer quality.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, digest
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.receipts import receipt
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
from .strategy import ARMS, GOAL_EVIDENCE_HEADER, actual_tail, goal_extra_document, render_final, p02

HERE = Path(__file__).resolve().parent
PREV_E = HERE.parent / 'b0_strategy_round2_v1_20261020'
PREV_O = OUTPUT / PREV_E.name


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_case(dataset, qid):
    for path in sorted((PREV_O / 'rounds').glob(f'*/cases/{dataset}/p02_role_headers/{qid}.json')):
        value = read(path)
        if isinstance(value.get('prediction'), str) and value['status'].startswith('success'):
            return path, value
    raise AssertionError('Missing completed old300 p02 case: ' + dataset + ':' + qid)


def run():
    protocol = read(PREV_E / 'PROTOCOL.frozen.json')
    assert protocol['model'] == 'qwen-plus-2025-12-01'
    runtime = rows(PREV_E / 'runtime300.frozen.jsonl')
    sources = {row['id']: row for row in rows(PREV_E / 'SOURCES.frozen.jsonl')}
    assert len(runtime) == len(sources) == 300
    assert Counter(c['dataset'] for c in runtime) == {'hotpotqa': 100, '2wiki': 100, 'musique': 100}
    assert all(not {'answer', 'answers', 'gold', 'prediction', 'supporting_facts', 'qrels'} & set(c) for c in runtime)
    readout = protocol['readout']
    model = protocol['model']
    counts = Counter()
    extra_by_dataset = Counter()
    extra_characters = []
    cases = []
    protected = {}
    for c in runtime:
        source = sources[c['id']]
        base_path = ROOT / source['base']
        assert sha(base_path) == source['base_sha256']
        base = read(base_path)
        path, parent = parent_case(c['dataset'], c['qid'])
        protected[str(path.relative_to(ROOT))] = sha(path)
        assert parent['id'] == c['id'] and parent['question'] == c['question']
        assert parent['nodes'][:-1] == base['nodes'][:-1]
        assert parent['nodes'][-1]['role'] == 'goal'
        assert parent['nodes'][-1]['request'] == c['question']
        packet = base['original_retrieval']
        assert packet['query'] == c['question'] and len(packet['docs']) == 10
        assert len({doc['doc_id'] for doc in packet['docs']}) == 10
        rebuilt, context = render_final(c['question'], parent['nodes'], packet, readout, 'parent_p02')
        assert rebuilt == parent['final_prompt'] and context == parent['final_context']
        counts['parent_prompts_and_contexts_exact'] += 1
        request = payload(model, rebuilt, 64)
        calls = [row for row in parent['logical_calls'] if row.get('payload_sha256') == digest(request)]
        assert calls and all(row['status'] == 'success' for row in calls)
        final = receipt(calls[0], request)
        assert final['status'] == 'success' and final['result']['status'] == 'success'
        assert final['result']['answer'].strip() == parent['prediction']
        assert final['result']['finish_reason'] == parent['finish_reason']
        assert request['temperature'] == 0 and request['extra_body'] == {'enable_thinking': False}
        assert request['max_tokens'] == 64
        counts['actual_final_receipts_payload_and_prediction_exact'] += 1
        # The retained goal packet was not merely an unexposed candidate
        # retrieval: validate it against the goal's actual raw API request.
        control_path = ROOT / source['control']
        assert sha(control_path) == source['control_sha256']
        control = read(control_path)
        assert parent['nodes'] == control['nodes']
        goal = parent['nodes'][-1]
        goal_packet = goal['reader_packet']
        assert p02.b0.passages(goal_packet) in goal['child_prompt']
        goal_request = payload(model, goal['child_prompt'], 128)
        goal_calls = [row for row in control['logical_calls'] if row.get('payload_sha256') == digest(goal_request)]
        assert goal_calls
        goal_result = receipt(goal_calls[0], goal_request)
        assert goal_result['status'] == 'success' and goal_result['result']['status'] == 'success'
        assert goal_result['result']['answer'].strip() == goal['initial_answer']
        counts['actual_goal_reader_packet_and_raw_receipt_exact'] += 1
        helper = p02.helper(parent['nodes'], 'p02_role_headers')
        passages = p02.b0.passages(packet)
        assert context == helper + '\n\n' + passages
        tail = actual_tail(c['question'], context, readout)
        prefix = rebuilt[:-len(tail)]
        assert rebuilt.endswith(tail) and 'Worked examples:' in prefix
        detail = {'id': c['id'], 'parent_case': str(path.relative_to(ROOT)),
                  'parent_sha256': sha(path), 'parent_payload_sha256': digest(request), 'arms': {}}
        for arm in ARMS:
            changed, changed_context = render_final(c['question'], parent['nodes'], packet, readout, arm)
            assert changed.startswith(prefix)
            assert changed.endswith(readout)
            if arm == 'q03_question_last':
                assert len(changed) == len(rebuilt) and Counter(changed) == Counter(rebuilt)
                assert changed_context == context
                expected_tail = ('Actual task:\n\nContext:\n' + context +
                                 '\n\nOriginal question:\n' + c['question'] + '\n\n' + readout)
                assert changed == prefix + expected_tail
                assert changed.replace(expected_tail, tail, 1) == rebuilt
                counts['q03_only_actual_question_block_reordered'] += 1
            else:
                extra = goal_extra_document(parent['nodes'], packet)
                independent = next((doc for doc in sorted(goal_packet['docs'], key=lambda doc: doc['rank'])
                                    if doc['doc_id'] not in {d['doc_id'] for d in packet['docs']}), None)
                assert extra == independent
                if extra is None:
                    assert changed == rebuilt and changed_context == context
                    assert digest(payload(model, changed, 64)) == digest(request)
                    counts['e03_all_goal_docs_duplicate_exact_parent_reuse'] += 1
                    addition = ''
                else:
                    assert extra in goal_packet['docs']
                    assert extra['block'] in goal['child_prompt']
                    addition = GOAL_EVIDENCE_HEADER + extra['block']
                    assert changed_context == context + addition
                    assert changed == prefix + actual_tail(c['question'], changed_context, readout)
                    assert len(changed) - len(rebuilt) == len(addition)
                    assert Counter(changed) == Counter(rebuilt) + Counter(addition)
                    counts['e03_one_first_rank_new_goal_doc_appended'] += 1
                    extra_by_dataset[c['dataset']] += 1
                    extra_characters.append(len(addition))
                    detail['additional_goal_passage'] = {k: extra[k] for k in ['rank', 'doc_id', 'source']}
                    detail['additional_goal_passage']['block_sha256'] = hashlib.sha256(extra['block'].encode('utf-8')).hexdigest()
                counts['e03_parent_QA_original_top10_and_readout_retained'] += 1
            # Whole blocks are identical, so all request/answer strings, role
            # headings, numbered QAs and ten numbered documents are retained
            # in their original internal order. No score/gold is an input.
            detail['arms'][arm] = {'prompt_chars': len(changed),
                                   'prompt_sha256': hashlib.sha256(changed.encode('utf-8')).hexdigest(),
                                   'worked_examples_and_instructions_exact': True,
                                   'full_question_QA_passages_readout_preserved': True,
                                   'adds_existing_goal_retrieval_information': arm == 'e03_goal_evidence' and bool(addition)}
        counts['query_only_runtime_cases'] += 1
        counts['original_top10_documents_preserved'] += 10
        counts['all_auxiliary_QA_preserved'] += len(parent['nodes'])
        cases.append(detail)
    for path, expected in protected.items():
        assert sha(ROOT / path) == expected
    report = {'passed': True, 'scope': 'All300 completed p02 development cases; actual frozen final receipts',
              'model': model, 'arms': ARMS, 'counts': dict(counts),
              'reference_boundary': 'No gold/reference, fresh90 or sealed150 file was opened; first-new-doc selection uses only existing doc IDs and goal ranks.',
              'e03_interpretation': 'Adds already-retrieved source information, not merely formatting; no extra retrieval or generation call.',
              'e03_added_by_dataset': dict(extra_by_dataset),
              'e03_added_characters': {'n': len(extra_characters), 'mean': sum(extra_characters) / len(extra_characters) if extra_characters else 0,
                                       'minimum': min(extra_characters) if extra_characters else 0,
                                       'maximum': max(extra_characters) if extra_characters else 0},
              'not_executed_alternatives': ['e03_evidence_first'],
              'new_api_calls': 0, 'new_retrieval_calls': 0, 'new_remote_calls': 0,
              'prompt_quality_or_f1_validated': False,
              'source_pins': {str(p.relative_to(ROOT)): sha(p) for p in [
                  Path(__file__), HERE / 'strategy.py', PREV_E / 'strategy.py',
                  PREV_E / 'PROTOCOL.frozen.json', PREV_E / 'runtime300.frozen.jsonl',
                  PREV_E / 'SOURCES.frozen.jsonl']},
              'protected_parent_cases_unchanged': len(protected), 'cases': cases}
    target = HERE / 'OFFLINE_AUDIT.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    run()
