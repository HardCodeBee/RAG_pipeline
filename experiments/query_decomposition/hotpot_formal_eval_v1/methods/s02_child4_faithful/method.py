"""Frozen s02 interfaces on private, newly retrieved shared-BGE packets."""
import json
from pathlib import Path

from . import goal_plan, mechanics, prompts


HERE = Path(__file__).resolve().parent


def context(packet):
    return '\n\n'.join(d['block'] for d in packet['docs'])


def dependency_records(case, sid, nodes):
    request = case['qdmr']['steps'][sid - 1]['request']
    result = []
    for ref in mechanics.references(request):
        node = nodes.get(ref)
        answer = node['answer'] if mechanics.returned(node) else 'No usable answer was returned for this prior step.'
        result.append({'step_id': ref, 'request': case['qdmr']['steps'][ref - 1]['request'], 'answer': answer})
    return result


def save_result(case, sid, kind, plan, transport, store, *, short=False):
    key = f"{case['qid']}/{sid}"
    cached = store.get(kind, key)
    if cached is not None:
        return cached
    store.put(kind + '_input', key, plan)
    transport.qid = case['qid']
    request_id, raw = transport.ask(kind + '_' + str(sid), plan['prompt'], plan['max_tokens'])
    node = {'qid': case['qid'], 'step_id': sid, 'request': case['qdmr']['steps'][sid - 1]['request'],
            'request_id': request_id, 'status': raw['status'], 'answer': raw.get('answer', ''),
            'finish_reason': raw.get('finish_reason'), 'refusal': raw.get('refusal')}
    if short:
        phrases = json.loads((HERE / 'refusal_phrases.json').read_text(encoding='utf8'))
        node['pure_refusal'] = mechanics.pure_refusal(node['answer'], phrases) if node['answer'] else False
        # Historical short binding allowed a successful nonempty truncated answer.
        node['available'] = raw['status'] == 'success' and bool(node['answer'].strip()) and not node['refusal'] and not node['pure_refusal']
    else:
        node['available'] = mechanics.returned(node)
    store.put(kind, key, node)
    if raw['status'] != 'success':
        # Provider failures remain infrastructure missingness, not scored zeros.
        store.put('prediction', case['qid'], {'qid': case['qid'], 'question': case['question'],
                  'prediction': None, 'status': raw['status'], 'request_id': request_id,
                  'failed_stage': kind + '_' + str(sid), 'finish_reason': raw.get('finish_reason')})
    return node


def run(cases, retrieval, transport, store, config):
    examples = json.loads((HERE / 'examples.json').read_text(encoding='utf8'))
    originals = retrieval.retrieve_many([{'qid': c['qid'], 'query': c['question']} for c in cases], 10)
    packets = {c['qid']: p for c, p in zip(cases, originals)}
    active = []
    for case in cases:
        if store.get('prediction', case['qid']) is not None:
            continue
        record = store.get('decomposition', case['qid'])
        if record is None:
            raise ValueError('Missing audited frozen remote decomposition')
        qdmr = record['qdmr']
        errors = mechanics.structural_errors(qdmr)
        if not errors and len(qdmr['steps']) > 1:
            active.append({**case, 'qdmr': qdmr})
    max_steps = max((len(c['qdmr']['steps']) for c in active), default=0)
    # First pass: short answers bind retrieval; all resulting packets remain fixed.
    for sid in range(1, max_steps + 1):
        jobs = []
        for case in active:
            if sid >= case['qdmr']['goal'] or store.get('prediction', case['qid']) is not None:
                continue
            key = f"{case['qid']}/{sid}"
            if store.get('short_node', key) is not None:
                continue
            previous = {i: store.get('short_node', f"{case['qid']}/{i}") for i in range(1, sid)}
            answers = {i: n['answer'] for i, n in previous.items() if n and n.get('available')}
            request = case['qdmr']['steps'][sid - 1]['request']
            bound, error = mechanics.bind(request, sid, answers)
            if error:
                store.put('short_node', key, {'qid': case['qid'], 'step_id': sid, 'request': request,
                          'status': 'skipped', 'answer': '', 'available': False, 'reason': error})
            else:
                jobs.append((case, bound))
        found = retrieval.retrieve_many([{'qid': c['qid'], 'query': q} for c, q in jobs], 10) if jobs else []
        for (case, bound), packet in zip(jobs, found):
            plan = {'qid': case['qid'], 'step_id': sid, 'binding_query': bound, 'packet': packet,
                    'prompt': prompts.V1.format(question=bound, context=context(packet)), 'max_tokens': 64}
            save_result(case, sid, 'short_node', plan, transport, store, short=True)
    goal_jobs = []
    for case in active:
        qid = case['qid']
        if store.get('prediction', qid) is not None:
            continue
        if store.get('goal_plan', qid) is None:
            shorts = [store.get('short_node', f'{qid}/{i}') for i in range(1, case['qdmr']['goal'])]
            plan = goal_plan.plan_goal(qid, {'qid': qid, 'qdmr': case['qdmr']}, [n for n in shorts if n])
            store.put('goal_plan', qid, plan)
        plan = store.get('goal_plan', qid)
        if plan['action'] == 'retrieve' and store.get('goal_packet', qid) is None:
            goal_jobs.append((case, plan['query']))
        elif plan['action'] != 'retrieve':
            store.put('goal_packet', qid, packets[qid])
    found = retrieval.retrieve_many([{'qid': c['qid'], 'query': q} for c, q in goal_jobs], 10) if goal_jobs else []
    for (case, _), packet in zip(goal_jobs, found):
        store.put('goal_packet', case['qid'], packet)
    # Early goal support follows the inherited p02a packet construction. It is
    # inactive on the current formal200, but never silently moves the goal.
    for case in active:
        if case['qdmr']['goal'] == len(case['qdmr']['steps']) or store.get('prediction', case['qid']) is not None:
            continue
        warm = {}
        for sid in range(1, len(case['qdmr']['steps']) + 1):
            qid, request = case['qid'], case['qdmr']['steps'][sid - 1]['request']
            if sid < case['qdmr']['goal']:
                source = store.get('short_node_input', f'{qid}/{sid}')
                packet = source['packet'] if source else None
            elif sid == case['qdmr']['goal']:
                packet = store.get('goal_packet', qid)
            else:
                source = store.get('warm_node_input', f'{qid}/{sid}')
                if source:
                    packet = source['packet']
                else:
                    bound, error = mechanics.bind(request, sid, {i: n['answer'] for i, n in warm.items() if mechanics.returned(n)})
                    packet = retrieval.retrieve(qid, bound, 10) if not error else None
            if packet is None:
                warm[sid] = {'qid': qid, 'step_id': sid, 'request': request, 'status': 'skipped', 'answer': '', 'available': False}
                continue
            deps = dependency_records(case, sid, warm)
            prompt = prompts.child_prompt(case['question'], sid, request, deps, context(packet))
            if sid == case['qdmr']['goal'] and store.get('goal_plan', qid)['action'] == 'combine':
                prompt = prompt[:-len('Answer:')] + prompts.GOAL_RULE + '\n\nAnswer:'
            plan = {'qid': qid, 'step_id': sid, 'packet': packet, 'dependencies': deps, 'prompt': prompt, 'max_tokens': 128}
            warm[sid] = save_result(case, sid, 'warm_node', plan, transport, store)
            if store.get('prediction', qid) is not None:
                break
    # Second pass: raw #N questions + prior contextual QA on fixed packets.
    for sid in range(1, max_steps + 1):
        for case in active:
            qid = case['qid']
            if sid > len(case['qdmr']['steps']) or store.get('prediction', qid) is not None:
                continue
            key = f'{qid}/{sid}'
            if store.get('contextual_node', key) is not None:
                continue
            goal = case['qdmr']['goal']
            if sid == goal:
                packet = store.get('goal_packet', qid)
            else:
                source = store.get('short_node_input' if sid < goal else 'warm_node_input', key)
                packet = source['packet'] if source else None
            request = case['qdmr']['steps'][sid - 1]['request']
            if packet is None:
                store.put('contextual_node', key, {'qid': qid, 'step_id': sid, 'request': request,
                          'status': 'skipped', 'answer': '', 'available': False, 'reason': 'no_first_pass_packet'})
                continue
            previous = {i: store.get('contextual_node', f'{qid}/{i}') for i in range(1, sid)}
            deps = dependency_records(case, sid, previous)
            role = 'goal' if sid == goal else 'ordinary'
            plan = {'qid': qid, 'step_id': sid, 'packet': packet, 'dependencies': deps, 'role': role,
                    'original_goal': goal, 'retrieval_refreshed': False,
                    'prompt': prompts.child(case['question'], sid, request, deps, context(packet), examples['child'],
                        role=role, combine=role == 'goal' and store.get('goal_plan', qid)['action'] == 'combine'), 'max_tokens': 128}
            save_result(case, sid, 'contextual_node', plan, transport, store)
    for case in cases:
        qid = case['qid']
        if store.get('prediction', qid) is not None:
            continue
        record = store.get('decomposition', qid)
        target = record['qdmr']
        errors = mechanics.structural_errors(target)
        nodes = []
        if not errors and len(target['steps']) > 1:
            for step in target['steps']:
                node = store.get('contextual_node', f"{qid}/{step['id']}")
                if mechanics.returned(node):
                    nodes.append({'step_id': step['id'], 'request': step['request'], 'answer': node['answer']})
        helper = prompts.render_helper(nodes, None, 'Answer') if nodes else ''
        blocks = [d['block'] for d in packets[qid]['docs']]
        position = config['helper_positions'][qid] if helper else None
        if helper:
            blocks.insert(position - 1, helper)
        final_context = '\n\n'.join(blocks)
        plan = {'qid': qid, 'context': final_context, 'helper': helper, 'position': position,
                'helper_step_ids': [n['step_id'] for n in nodes], 'fallback': not helper,
                'original_doc_ids': [d['doc_id'] for d in packets[qid]['docs']], 'structural_errors': errors}
        store.put('context', qid, plan)
        prompt = prompts.final(case['question'], final_context, examples['final']) if helper else prompts.V1.format(question=case['question'], context=final_context)
        transport.qid = qid
        request_id, raw = transport.ask('final', prompt, 64)
        store.put('prediction', qid, {'qid': qid, 'question': case['question'], 'prediction': raw.get('answer') if raw['status'] == 'success' else None,
                  'status': raw['status'], 'request_id': request_id, 'finish_reason': raw.get('finish_reason'),
                  'fallback': not helper, 'fallback_reason': errors or ('no_usable_generated_helper' if not helper else None),
                  'position': position, 'final_doc_ids': plan['original_doc_ids'], 'helper_step_ids': plan['helper_step_ids']})
        store.export()
