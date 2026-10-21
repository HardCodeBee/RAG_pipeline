"""Compose selected C08/c02 components without editing either frozen component.

Retrieval and generation are injected so interface validation has no network,
and later paid execution must use the existing budgeted, one-shot transport.
"""
from copy import deepcopy
from pathlib import Path
import json

from experiments.query_decomposition.child_answer_accuracy_v1_20261017 import candidate_c08 as c08
from experiments.query_decomposition.rewrite_api_baseline_c02_v1_20261019 import prompts as c02
from experiments.query_decomposition.multigrain_prompt_v3_20261017 import prompts as final_prompts
from experiments.query_decomposition.hotpot_formal_eval_v1.methods.s02_child4_faithful import prompts as legacy_child
from experiments.query_decomposition.multigrain_prompt_v2_20261017 import prompts as legacy_rewrite

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = {
    'legacy': (False, False),
    'c08_only': (True, False),
    'c02_only': (False, True),
    'c08_c02': (True, True),
}


def passages(packet):
    return '\n\n'.join(d['block'] for d in packet['docs'])


def merge(original, rewritten):
    """Exactly the frozen original-first interleaving; M1 is not used."""
    if rewritten is None:
        return deepcopy(original)
    docs, seen = [], set()
    for rank in range(10):
        for route in (original, rewritten):
            if rank < len(route['docs']):
                d = route['docs'][rank]
                if d['doc_id'] not in seen:
                    seen.add(d['doc_id']); docs.append(d)
    docs = [{**d, 'rank': i, 'block': f'[Chunk {i} | source={d["source"]}, pages=1-1]\n{d["text"]}'}
            for i, d in enumerate(docs[:10], 1)]
    assert len(docs) == len({d['doc_id'] for d in docs}) == 10
    return {'qid': original['qid'], 'query': original['query'], 'k': 10,
            'docs': docs, 'merge_policy': 'interleave_original_first_top10'}


def execute(case, retrieve, ask, *, variant='c08_c02', record_node=None):
    """ask(stage,prompt,limit,allow_final_truncation=...) -> (id,text,finish).

    ``case`` supplies only qid/question/supporting_questions/dataset. Semantic
    validity of decomposition is not inferred here. Every original question is
    still independently retrievable even when all prerequisites abstain.
    """
    child_selected, rewrite_selected = VARIANTS[variant]
    examples = json.loads((ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json').read_text(encoding='utf-8'))
    final_examples = json.loads((ROOT / 'experiments/query_decomposition/multigrain_prompt_v3_20261017/final_examples.json').read_text(encoding='utf-8'))
    history, nodes = [], []
    root = retrieve(case['question'])
    assert len(root['docs']) == len({d['doc_id'] for d in root['docs']}) == 10
    requests = list(case['supporting_questions']) + [case['question']]
    for sid, question in enumerate(requests, 1):
        available = c08.usable_history(history) if child_selected else deepcopy(history)
        original = retrieve(question)
        rewrite, rewrite_id, raw_rewrite, rejection = question, None, None, None
        rewrite_prompt = None
        if available:
            rewrite_prompt = (c02 if rewrite_selected else legacy_rewrite).rewrite(question, available)
            rewrite_id, raw_rewrite, finish = ask(f'{variant}/rewrite/{sid}', rewrite_prompt, 256)
            if finish != 'stop':
                raise RuntimeError('Incomplete rewrite is not propagated')
            candidate = raw_rewrite.strip()
            if candidate and candidate.endswith('?') and '\n' not in candidate:
                rewrite = candidate
            else:
                rejection = 'not_one_complete_question; original retained'
        rewritten = retrieve(rewrite) if rewrite != question else None
        packet = merge(original, rewritten)
        is_goal = sid == len(requests)
        if child_selected:
            child_prompt = c08.child(case['question'], question, available, passages(packet), [])
        else:
            child_prompt = legacy_child.child(case['question'], sid, question, available, passages(packet), examples['child'],
                                               role='goal' if is_goal else 'ordinary', combine=False)
        answer_id, answer, finish = ask(f'{variant}/answer/{sid}', child_prompt, 128)
        if finish != 'stop':
            raise RuntimeError('Incomplete child answer is not propagated')
        answer = answer.strip()
        if not answer:
            raise RuntimeError('Empty child answer is not propagated')
        initial_answer, format_id = answer, None
        # C08 already emits complete factual sentences or its exact sentinel.
        # Its selected output must not pass through the legacy extra formatter.
        if not child_selected and len(answer.split()) < 4:
            format_prompt = ('Express the candidate answer below as a concise complete factual sentence that names its subject and relation. '
                             'Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. '
                             'If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n'
                             + 'Question:\n' + question + '\n\nCandidate answer:\n' + answer + '\n\nContext:\n' + passages(packet))
            format_id, answer, finish = ask(f'{variant}/sentence_format/{sid}', format_prompt, 128)
            if finish != 'stop':
                raise RuntimeError('Incomplete legacy sentence formatting is not propagated')
            answer = answer.strip()
        record = {'step_id': sid, 'request': question, 'answer': answer}
        node = {**record, 'role': 'goal' if is_goal else 'ordinary',
                'prior_complete_answers': deepcopy(available),
                'excluded_prior_step_ids': [r['step_id'] for r in history if r['step_id'] not in {h['step_id'] for h in available}],
                'original_retrieval': original, 'rewritten_query': rewrite, 'rewritten_retrieval': rewritten,
                'reader_packet': packet, 'rewrite_prompt': rewrite_prompt, 'raw_rewrite': raw_rewrite,
                'rewrite_request_id': rewrite_id, 'rewrite_rejection': rejection,
                'child_prompt': child_prompt, 'answer_request_id': answer_id,
                'initial_answer': initial_answer, 'sentence_format_request_id': format_id,
                'is_whole_no_answer': c08.is_no_answer(answer) if child_selected else None}
        history.append(record); nodes.append(node)
        if record_node:
            record_node(deepcopy(node))
    usable = c08.auxiliary_pairs(history) if child_selected else history
    helper = legacy_child.render_helper(usable, None, 'Answer') if usable else ''
    context = helper + '\n\n' + passages(root) if helper else passages(root)
    final_prompt = final_prompts.final(case['question'], context, final_examples)
    final_id, prediction, finish = ask(variant + '/final', final_prompt, 64, allow_final_truncation=True)
    if finish not in {'stop', 'length'}:
        raise RuntimeError('Final generation not successful')
    return {**case, 'variant': variant, 'status': 'success' if finish == 'stop' else 'success_truncated',
            'nodes': nodes, 'history_all': history, 'history_usable': usable,
            'helper': helper, 'helper_position': 1 if helper else None,
            'original_retrieval': root, 'final_context': context, 'final_prompt': final_prompt,
            'prediction': prediction.strip(), 'request_id': final_id, 'finish_reason': finish}
