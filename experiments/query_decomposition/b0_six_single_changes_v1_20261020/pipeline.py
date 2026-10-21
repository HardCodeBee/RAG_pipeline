"""Six one-factor interventions on B0. No writes to parent method modules."""
from copy import deepcopy
import json
from pathlib import Path

from experiments.query_decomposition.component_chain_c08_c02_v1_20261019 import pipeline as b0
from experiments.query_decomposition.child_answer_accuracy_v1_20261017 import candidate_c08 as c08
from experiments.query_decomposition.rewrite_api_baseline_c02_v1_20261019 import prompts as c02
from experiments.query_decomposition.component_chain_opt_v2_20261019.policy import IntermediateLength

ROOT = Path(__file__).resolve().parents[3]
ARMS = {
    'A_child_prompt_C08': 'child_prompt',
    'B_rewrite_prompt_c02': 'rewrite_prompt',
    'C_final_readout_P1': 'final_readout',
    'D_drop_whole_abstentions': 'filter_missing',
    'E_remove_sentence_formatter': 'remove_formatter',
    'F_child_length_fallback': 'length_fallback',
}


def c08_prompt_without_filter(original, question, history, context):
    """Keep C08 text/examples, but keep B0's complete prior-QA input policy.

    Calling c08.child directly would additionally filter prior QA inside its
    renderer, contaminating the prompt-only intervention with factor D.
    """
    def prior(records):
        return '\n\n'.join('Question: ' + r['request'] + '\nAnswer: ' + r['answer'] for r in records) or 'No prior answers.'
    shots = c08.load(c08.HERE / 'c08_examples.frozen.json')
    # Demonstrations remain byte-identical to the selected C08 demonstrations.
    demonstrations = '\n\n'.join(
        f"Example {i+1}\nCurrent question: {r['request']}\nPrevious question-answer pairs:\n{prior(c08.usable_history(r['dependencies']))}\nRetrieved passages:\n{r['context']}\nAnswer: {r['answer']}"
        for i, r in enumerate(shots))
    return (c08.INSTRUCTION + '\nWorked examples:\n' + demonstrations +
            '\n\nActual task:\nOriginal question:\n' + original +
            '\n\nPrevious question-answer pairs:\n' + prior(history) +
            '\n\nCurrent question:\n' + question + '\n\nRetrieved passages:\n' + context + '\n\nAnswer:')


def execute(case, retrieve, ask, *, arm, record_node=None):
    assert arm in ARMS or arm == 'baseline_replay'
    factor = ARMS.get(arm)
    examples = json.loads((ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json').read_text(encoding='utf-8'))
    final_examples = json.loads((ROOT / 'experiments/query_decomposition/multigrain_prompt_v3_20261017/final_examples.json').read_text(encoding='utf-8'))
    history, nodes, events = [], [], []
    root = retrieve(case['question'])
    assert len(root['docs']) == len({d['doc_id'] for d in root['docs']}) == 10
    requests = list(case['supporting_questions']) + [case['question']]
    fallback = None
    try:
        for sid, question in enumerate(requests, 1):
            available = c08.usable_history(history) if factor == 'filter_missing' else deepcopy(history)
            if len(available) != len(history): events.append({'kind': 'filtered_prior_QA', 'step_id': sid, 'excluded': [r['step_id'] for r in history if r['step_id'] not in {a['step_id'] for a in available}]})
            original = retrieve(question)
            rewrite, rewrite_id, raw_rewrite, rejection, rewrite_prompt = question, None, None, None, None
            if available:
                rewrite_prompt = (c02 if factor == 'rewrite_prompt' else b0.legacy_rewrite).rewrite(question, available)
                rewrite_id, raw_rewrite, finish = ask(f'legacy/rewrite/{sid}', rewrite_prompt, 256)
                if finish != 'stop': raise RuntimeError('Incomplete rewrite is not propagated')
                candidate = raw_rewrite.strip()
                if candidate and candidate.endswith('?') and '\n' not in candidate: rewrite = candidate
                else: rejection = 'not_one_complete_question; original retained'
            rewritten = retrieve(rewrite) if rewrite != question else None
            packet = b0.merge(original, rewritten)
            is_goal = sid == len(requests)
            if factor == 'child_prompt':
                child_prompt = c08_prompt_without_filter(case['question'], question, available, b0.passages(packet))
            else:
                child_prompt = b0.legacy_child.child(case['question'], sid, question, available, b0.passages(packet), examples['child'],
                                                    role='goal' if is_goal else 'ordinary', combine=False)
            answer_id, answer, finish = ask(f'legacy/answer/{sid}', child_prompt, 128)
            if finish != 'stop': raise RuntimeError('Incomplete child answer is not propagated')
            answer = answer.strip()
            if not answer: raise RuntimeError('Empty child answer is not propagated')
            initial_answer, format_id = answer, None
            if len(answer.split()) < 4 and factor != 'remove_formatter':
                format_prompt = ('Express the candidate answer below as a concise complete factual sentence that names its subject and relation. '
                                 'Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. '
                                 'If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n'
                                 + 'Question:\n' + question + '\n\nCandidate answer:\n' + answer + '\n\nContext:\n' + b0.passages(packet))
                format_id, answer, finish = ask(f'legacy/sentence_format/{sid}', format_prompt, 128)
                if finish != 'stop': raise RuntimeError('Incomplete legacy sentence formatting is not propagated')
                answer = answer.strip()
            elif len(answer.split()) < 4:
                events.append({'kind': 'formatter_omitted', 'step_id': sid})
            record = {'step_id': sid, 'request': question, 'answer': answer}
            node = {**record, 'role': 'goal' if is_goal else 'ordinary',
                    'prior_complete_answers': deepcopy(available),
                    'excluded_prior_step_ids': [r['step_id'] for r in history if r['step_id'] not in {h['step_id'] for h in available}],
                    'original_retrieval': original, 'rewritten_query': rewrite, 'rewritten_retrieval': rewritten,
                    'reader_packet': packet, 'rewrite_prompt': rewrite_prompt, 'raw_rewrite': raw_rewrite,
                    'rewrite_request_id': rewrite_id, 'rewrite_rejection': rejection,
                    'child_prompt': child_prompt, 'answer_request_id': answer_id,
                    'initial_answer': initial_answer, 'sentence_format_request_id': format_id,
                    'is_whole_no_answer': None}
            history.append(record); nodes.append(node)
            if record_node: record_node(deepcopy(node))
    except IntermediateLength as failure:
        if factor != 'length_fallback' or '/answer/' not in failure.stage: raise
        fallback = {'reason': 'child_length', 'failed_stage': failure.stage, 'failed_request_id': failure.request_id,
                    'unpropagated_truncated_answer': failure.answer, 'remaining_nodes_skipped': True}
        events.append({'kind': 'child_length_fallback', 'stage': failure.stage})
    usable = c08.auxiliary_pairs(history) if factor == 'filter_missing' else history
    if len(usable) != len(history): events.append({'kind': 'filtered_final_QA', 'excluded': [r['step_id'] for r in history if r['step_id'] not in {a['step_id'] for a in usable}]})
    helper = b0.legacy_child.render_helper(usable, None, 'Answer') if usable else ''
    context = helper + '\n\n' + b0.passages(root) if helper else b0.passages(root)
    final_prompt = b0.final_prompts.final(case['question'], context, final_examples)
    if factor == 'final_readout':
        suffix = json.loads((ROOT / 'experiments/query_decomposition/component_chain_opt_v2_20261019/PROTOCOL.frozen.json').read_text(encoding='utf-8'))['phase_B']['P1_readout']
        assert final_prompt.endswith(b0.final_prompts.FINAL_READOUT)
        final_prompt = final_prompt[:-len(b0.final_prompts.FINAL_READOUT)] + suffix
    final_id, prediction, finish = ask('legacy/final', final_prompt, 64, allow_final_truncation=True)
    if finish not in {'stop', 'length'}: raise RuntimeError('Final generation not successful')
    status = 'success' if finish == 'stop' else 'success_truncated'
    if fallback: status = 'success_fallback' if finish == 'stop' else 'success_fallback_truncated'
    return {**case, 'variant': 'legacy', 'arm': arm, 'single_factor': factor, 'status': status,
            'nodes': nodes, 'history_all': history, 'history_usable': usable, 'factor_events': events,
            'helper': helper, 'helper_position': 1 if helper else None, 'original_retrieval': root,
            'final_context': context, 'final_prompt': final_prompt, 'prediction': prediction.strip(),
            'request_id': final_id, 'finish_reason': finish, 'fallback': fallback}
