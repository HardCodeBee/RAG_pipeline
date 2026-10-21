"""All300 original p02 receipts; only scoped rule/evidence input changes.

No fresh90/confirmation outcome, reference answer, retrieval or provider call.
The interaction's extra passage is chosen by the frozen round3 implementation.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, digest
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.receipts import receipt
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
from .strategy import ARMS, NEW_SCOPE_RULE, OLD_SCOPE_RULE, evidence, p02, render_final

HERE = Path(__file__).resolve().parent
PARENT_E = HERE.parent / 'b0_strategy_round2_v1_20261020'
PARENT_O = OUTPUT / PARENT_E.name
EVIDENCE_E = HERE.parent / 'b0_strategy_round3_v1_20261020'


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
    source = {row['id']: row for row in rows(PARENT_E / 'SOURCES.frozen.jsonl')}
    assert len(cases) == len(source) == 300
    assert Counter(c['dataset'] for c in cases) == {'hotpotqa': 100, '2wiki': 100, 'musique': 100}
    assert all(not {'answer', 'answers', 'gold', 'prediction', 'supporting_facts', 'f1'}.intersection(c) for c in cases)
    readout, model = protocol['readout'], protocol['model']
    counts, details, hashes = Counter(), [], {}
    for case in cases:
        src = source[case['id']]
        base_path = ROOT / src['base']
        assert sha(base_path) == src['base_sha256']
        base = read(base_path)
        path, parent = parent_path(case['dataset'], case['qid'])
        hashes[str(path.relative_to(ROOT))] = sha(path)
        assert parent['id'] == case['id'] and parent['question'] == case['question']
        packet = base['original_retrieval']
        assert packet['query'] == case['question'] and len(packet['docs']) == 10
        nodes_before = json.dumps(parent['nodes'], sort_keys=True, ensure_ascii=False)
        prompt, context = render_final(case['question'], parent['nodes'], packet, readout, 'parent_p02')
        assert prompt == parent['final_prompt'] and context == parent['final_context']
        request = payload(model, prompt, 64)
        call = next(row for row in parent['logical_calls'] if row['payload_sha256'] == digest(request))
        actual = receipt(call, request)
        assert actual['status'] == 'success' and actual['result']['status'] == 'success'
        assert actual['result']['answer'].strip() == parent['prediction']
        counts['actual_parent_prompts_contexts_and_receipts_exact'] += 1
        ordinary, ordinary_context = render_final(case['question'], parent['nodes'], packet, readout, ARMS[0])
        assert ordinary_context == context
        assert ordinary.replace(NEW_SCOPE_RULE, OLD_SCOPE_RULE, 1) == prompt
        assert ordinary == prompt.replace(OLD_SCOPE_RULE, NEW_SCOPE_RULE, 1)
        assert Counter(ordinary) + Counter(OLD_SCOPE_RULE) == Counter(prompt) + Counter(NEW_SCOPE_RULE)
        counts['r04_exactly_one_instruction_sentence_replaced'] += 1
        old_evidence, old_evidence_context = evidence.render_final(
            case['question'], parent['nodes'], packet, readout, 'e03_goal_evidence')
        combined, combined_context = render_final(case['question'], parent['nodes'], packet, readout, ARMS[1])
        assert combined_context == old_evidence_context
        assert combined == old_evidence.replace(OLD_SCOPE_RULE, NEW_SCOPE_RULE, 1)
        assert combined.replace(NEW_SCOPE_RULE, OLD_SCOPE_RULE, 1) == old_evidence
        counts['s04_only_scope_replacement_on_frozen_e03_input'] += 1
        extra = evidence.goal_extra_document(parent['nodes'], packet)
        if extra is None:
            assert combined == ordinary and combined_context == ordinary_context
            assert payload(model, combined, 64) == payload(model, ordinary, 64)
            counts['s04_duplicate_only_exact_cross_arm_payload_reuse'] += 1
        else:
            assert extra in parent['nodes'][-1]['reader_packet']['docs']
            assert extra['doc_id'] not in {d['doc_id'] for d in packet['docs']}
            assert combined_context == context + evidence.GOAL_EVIDENCE_HEADER + extra['block']
            counts['s04_identical_frozen_first_new_goal_passage'] += 1
        for arm, changed in [(ARMS[0], ordinary), (ARMS[1], combined)]:
            assert changed.count(NEW_SCOPE_RULE) == 1 and OLD_SCOPE_RULE not in changed
            assert changed.endswith(readout)
            assert changed.split('\nWorked examples:', 1)[1] == (
                prompt if arm == ARMS[0] else old_evidence).split('\nWorked examples:', 1)[1]
        assert json.dumps(parent['nodes'], sort_keys=True, ensure_ascii=False) == nodes_before
        details.append({'id': case['id'], 'parent_source': str(path.relative_to(ROOT)),
                        'parent_sha256': sha(path), 'parent_payload_sha256': digest(request),
                        'r04_payload_sha256': digest(payload(model, ordinary, 64)),
                        's04_payload_sha256': digest(payload(model, combined, 64)),
                        'goal_evidence_doc_id': extra['doc_id'] if extra else None,
                        'cross_arm_payload_identical': combined == ordinary})
        counts['original_top10_docs_retained'] += 10
        counts['all_auxiliary_QAs_retained'] += len(parent['nodes'])
    for path, value in hashes.items():
        assert sha(ROOT / path) == value
    # Compatibility with the root driver's frozen-parent preparation gate.
    counts['parent_prompts_and_contexts_exact'] = counts['actual_parent_prompts_contexts_and_receipts_exact']
    report = {'passed': True, 'arms': ARMS, 'parent': 'original p02 actual300',
              'counts': dict(counts), 'scope_replacement': {'old': OLD_SCOPE_RULE, 'new': NEW_SCOPE_RULE},
              'design_interpretation': {
                  'r04': 'One instruction replacement for answer target/conditions; no new factual information.',
                  's04': 'Interaction of that replacement and frozen e03 evidence addition; not a single-factor improvement over p02.',
                  'risk': 'Reference binding is explicitly permitted, but whether an auxiliary fact changes the intended target/relation/conditions still requires model interpretation; this is not a semantic verifier.',
                  'comparators': ['p02', 'completed e03_goal_evidence', 'r04_scope_contract']},
              'cross_arm_cache_requirement': 'When the whole final payload is identical, reuse one first successful receipt; do not pay twice for the86 duplicate-only cases.',
              'reference_boundary': 'No fresh90/confirmation outcomes, reference answers or qrels opened; chooser is the unchanged round3 rank/doc_id rule.',
              'model': model, 'max_tokens': 64, 'new_api_calls': 0, 'new_retrieval_calls': 0, 'new_remote_calls': 0,
              'source_pins': {str(p.relative_to(ROOT)): sha(p) for p in [
                  Path(__file__), HERE / 'strategy.py', PARENT_E / 'strategy.py',
                  PARENT_E / 'PROTOCOL.frozen.json', PARENT_E / 'runtime300.frozen.jsonl',
                  PARENT_E / 'SOURCES.frozen.jsonl', EVIDENCE_E / 'strategy.py', EVIDENCE_E / 'PROTOCOL.frozen.json']},
              'protected_parent_cases_unchanged': len(hashes), 'cases': details}
    (HERE / 'OFFLINE_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    run()
