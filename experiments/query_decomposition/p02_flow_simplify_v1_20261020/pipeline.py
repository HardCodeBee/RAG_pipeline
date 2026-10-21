"""Isolated p02 flow simplifications; all retrieval/generation is injected.

No transport, scorer, references, model, or retriever is opened by this module.
The exact frozen legacy prompt renderers are imported rather than rewritten.
"""
from copy import deepcopy
from math import log2, inf
import json

from experiments.query_decomposition.component_chain_c08_c02_v1_20261019 import pipeline as b0
from experiments.query_decomposition.b0_strategy_round2_v1_20261020 import strategy as p02
from experiments.query_decomposition.b0_strategy_round2_v1_20261020.fresh_runner import safe_case


ARMS = ('timing_only', 'union20', 'discount20', 'single_final')


def default_readout():
    path = b0.ROOT / 'experiments/query_decomposition/component_chain_opt_v2_20261019/PROTOCOL.frozen.json'
    return json.loads(path.read_text(encoding='utf-8'))['phase_B']['P1_readout']


def _validate_route(packet):
    docs = packet['docs']
    if len(docs) > 10 or len(docs) != len({d['doc_id'] for d in docs}):
        raise ValueError('Each injected route must contain at most 10 unique document IDs')


def merge(original, rewritten, *, arm='union20'):
    """Merge a question's own two Top-10 routes, with no cross-step pooling.

    ``timing_only`` delegates the old merge byte-for-byte. Other arms retain
    the full ID-deduplicated union; only discount20/single_final change order.
    union20 keeps the first interleaved occurrence's representation exactly
    as the old merge did. Discount fusion keeps the original representation
    when both routes have the same ID; route-specific ranks are separate.
    A repeated query is one route, never a second source of fusion credit.
    """
    if arm not in ARMS:
        raise ValueError('Unknown arm: ' + arm)
    _validate_route(original)
    if rewritten is not None:
        _validate_route(rewritten)
        if original['qid'] != rewritten['qid']:
            raise ValueError('Cannot merge packets from different query IDs')
        if original['query'] == rewritten['query']:
            rewritten = None
    if arm == 'timing_only':
        return b0.merge(original, rewritten)
    if rewritten is None:
        return deepcopy(original)

    docs_by_id, ranks = {}, {}
    for route_name, route in (('original', original), ('rewritten', rewritten)):
        for rank, doc in enumerate(route['docs'], 1):
            did = doc['doc_id']
            docs_by_id.setdefault(did, deepcopy(doc))
            ranks.setdefault(did, {})[route_name] = rank
    scores = {did: sum(1 / log2(rank + 1) for rank in value.values())
              for did, value in ranks.items()}
    if arm == 'union20':
        selected, seen = [], set()
        for index in range(max(len(original['docs']), len(rewritten['docs']))):
            for route in (original, rewritten):
                if index < len(route['docs']):
                    did = route['docs'][index]['doc_id']
                    if did not in seen:
                        seen.add(did)
                        selected.append(did)
                        docs_by_id[did] = deepcopy(route['docs'][index])
        policy = 'interleave_original_first_all_id_unique'
    else:
        selected = sorted(docs_by_id, key=lambda did: (
            -scores[did], min(ranks[did].values()), ranks[did].get('original', inf), did))
        policy = 'discount_rank_sum_all_id_unique'

    docs = []
    for rank, did in enumerate(selected, 1):
        doc = deepcopy(docs_by_id[did])
        doc.update(rank=rank, block=f'[Chunk {rank} | source={doc["source"]}, pages=1-1]\n{doc["text"]}')
        docs.append(doc)
    assert len(docs) == len(docs_by_id) == len({d['doc_id'] for d in docs}) <= 20
    return {
        'qid': original['qid'], 'query': original['query'], 'k': len(docs),
        'docs': docs, 'merge_policy': policy,
        'fusion': [{'doc_id': did, 'rank': rank, 'route_ranks': ranks[did],
                    'discount_score': scores[did]} for rank, did in enumerate(selected, 1)],
        'route_queries': {'original': original['query'], 'rewritten': rewritten['query']},
    }


def helper(nodes, *, single_final=False):
    if not single_final:
        return p02.helper(nodes, 'p02_role_headers')
    if any(node['role'] != 'ordinary' for node in nodes):
        raise ValueError('single_final helper must not contain an appended goal answer')
    if not nodes:
        return ''
    pairs = '\n\n'.join(f"#{n['step_id']} Question: {n['request']}\nAnswer: {n['answer']}"
                        for n in nodes)
    return 'Auxiliary question-answer pairs\n\nIntermediate candidate results\n\n' + pairs


def render_final(question, nodes, packet, readout, *, single_final=False):
    block = helper(nodes, single_final=single_final)
    context = block + '\n\n' + b0.passages(packet) if block else b0.passages(packet)
    examples = json.loads((b0.ROOT / 'experiments/query_decomposition/multigrain_prompt_v3_20261017/final_examples.json').read_text(encoding='utf-8'))
    raw = b0.final_prompts.final(question, context, examples)
    assert raw.endswith(b0.final_prompts.FINAL_READOUT)
    actual = raw[:-len(b0.final_prompts.FINAL_READOUT)] + readout
    return actual, context, block, raw


def execute(case, retrieve, ask, *, arm='timing_only', readout=None, record_node=None):
    """Execute one query-only case with immutable legacy stage payloads.

    ``retrieve(query)`` returns a Top-10 packet. ``ask(stage,prompt,limit,
    allow_final_truncation=...)`` returns ``(request_id,text,finish_reason)``.
    Exact successful payload reuse and all budget/transport decisions belong
    to the caller. This function never substitutes recorded downstream work
    after a changed upstream answer.
    """
    case = safe_case(case)
    if arm not in ARMS:
        raise ValueError('Unknown arm: ' + arm)
    readout = default_readout() if readout is None else readout
    examples = json.loads((b0.ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json').read_text(encoding='utf-8'))
    single_final = arm == 'single_final'
    filtered = []
    supporting = []
    for source_sid, question in enumerate(case['supporting_questions'], 1):
        # Exact text equality only. No semantic-equivalence or truth judgment.
        if single_final and question == case['question']:
            filtered.append({'original_supporting_step_id': source_sid, 'request': question,
                             'reason': 'exact_original_query_duplicate'})
        else:
            supporting.append((source_sid, question))
    requests = [question for _, question in supporting]
    if not single_final:
        requests.append(case['question'])
    nodes, history = [], []
    root = None
    goal_inputs = {}

    def retrieve_current(question, sid):
        available = deepcopy(history)
        original = retrieve(question)
        _validate_route(original)
        rewrite, rewrite_id, raw_rewrite, rejection = question, None, None, None
        rewrite_prompt = None
        if available:
            rewrite_prompt = b0.legacy_rewrite.rewrite(question, available)
            rewrite_id, raw_rewrite, finish = ask(f'legacy/rewrite/{sid}', rewrite_prompt, 256)
            if finish != 'stop':
                raise RuntimeError('Incomplete rewrite is not propagated')
            candidate = raw_rewrite.strip()
            if candidate and candidate.endswith('?') and '\n' not in candidate:
                rewrite = candidate
            else:
                rejection = 'not_one_complete_question; original retained'
        rewritten = retrieve(rewrite) if rewrite != question else None
        return {
            'prior_complete_answers': available, 'excluded_prior_step_ids': [],
            'original_retrieval': original, 'rewritten_query': rewrite,
            'rewritten_retrieval': rewritten, 'reader_packet': merge(original, rewritten, arm=arm),
            'rewrite_prompt': rewrite_prompt, 'raw_rewrite': raw_rewrite,
            'rewrite_request_id': rewrite_id, 'rewrite_rejection': rejection,
        }

    for sid, question in enumerate(requests, 1):
        is_goal = not single_final and sid == len(requests)
        inputs = retrieve_current(question, sid)
        if is_goal:
            # Only a retrieval timing change: the same original-query Top-10
            # supplies the final root context, without the redundant early get.
            root = deepcopy(inputs['original_retrieval'])
        raw_child = b0.legacy_child.child(
            case['question'], sid, question, inputs['prior_complete_answers'],
            b0.passages(inputs['reader_packet']), examples['child'],
            role='goal' if is_goal else 'ordinary', combine=False)
        prompt = p02.goal_prompt(raw_child, 'p02_role_headers') if is_goal else raw_child
        if is_goal:
            goal_inputs.update(raw_B0_goal_prompt=raw_child, actual_goal_prompt=prompt)
        answer_id, answer, finish = ask(f'legacy/answer/{sid}', prompt, 128)
        if finish != 'stop':
            raise RuntimeError('Incomplete child answer is not propagated')
        answer = answer.strip()
        if not answer:
            raise RuntimeError('Empty child answer is not propagated')
        initial_answer, format_id, format_prompt = answer, None, None
        if len(answer.split()) < 4:
            format_prompt = (
                'Express the candidate answer below as a concise complete factual sentence that names its subject and relation. '
                'Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. '
                'If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n'
                + 'Question:\n' + question + '\n\nCandidate answer:\n' + answer
                + '\n\nContext:\n' + b0.passages(inputs['reader_packet']))
            format_id, answer, finish = ask(f'legacy/sentence_format/{sid}', format_prompt, 128)
            if finish != 'stop':
                raise RuntimeError('Incomplete legacy sentence formatting is not propagated')
            answer = answer.strip()
        record = {'step_id': sid, 'request': question, 'answer': answer}
        node = {**record, 'role': 'goal' if is_goal else 'ordinary', **inputs,
                'child_prompt': prompt, 'answer_request_id': answer_id,
                'initial_answer': initial_answer, 'sentence_format_request_id': format_id,
                'is_whole_no_answer': None}
        if is_goal:
            node['raw_B0_child_prompt'] = raw_child
        if single_final:
            node['original_supporting_step_id'] = supporting[sid - 1][0]
        history.append(record)
        nodes.append(node)
        if record_node:
            record_node(deepcopy(node))

    final_retrieval = None
    if single_final:
        # This executes only retrieval/rewrite, never a goal factual-answer call.
        final_retrieval = retrieve_current(case['question'], len(nodes) + 1)
        root = deepcopy(final_retrieval['original_retrieval'])
        final_packet = final_retrieval['reader_packet']
    else:
        assert root is not None and nodes[-1]['role'] == 'goal'
        final_packet = root
    final_prompt, context, block, raw_final = render_final(
        case['question'], nodes, final_packet, readout, single_final=single_final)
    final_id, prediction, finish = ask('legacy/final', final_prompt, 64, allow_final_truncation=True)
    if finish not in {'stop', 'length'}:
        raise RuntimeError('Final generation not successful')
    return {
        **case, 'variant': 'legacy', 'arm': arm,
        'status': 'success' if finish == 'stop' else 'success_truncated',
        'nodes': nodes, 'history_all': history, 'history_usable': deepcopy(history),
        'helper': block, 'helper_position': 1 if block else None,
        'original_retrieval': root, 'final_reader_packet': final_packet,
        'final_retrieval': final_retrieval,
        'final_context': context, 'final_prompt': final_prompt,
        'raw_B0_final_prompt': raw_final, 'goal_prompt_change': goal_inputs,
        'filtered_supporting_questions': filtered,
        'prediction': prediction.strip(), 'request_id': final_id, 'finish_reason': finish,
        'upstream_executor': 'isolated p02 flow simplification with frozen legacy prompt renderers',
    }
