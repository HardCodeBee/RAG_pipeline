"""Online Decomp with nested retrieve-ODQA and title-bound readers.

Task syntax is a strict JSON adaptation. Modular execution and answer-conditioned
planning are retained; BGE replaces Elasticsearch/BM25.
"""
import json
from pathlib import Path
import re

OUTER = ('Produce only the next task, using the actual completed task results. '
         'Do not generate future tasks or invent results. Valid JSON tasks are '
         '{"task":"retrieve_odqa","question":"standalone question"}, '
         '{"task":"multihop_titleqa","titles":["retrieved title"],"question":"original question"}, '
         'or {"task":"EOQ"} to return the last completed answer.')
INNER = ('Solve the single-hop Question using modular tools. Produce only the next JSON task: '
         '{"task":"retrieve","question":"search question"}, '
         '{"task":"singlehop_titleqa","titles":["retrieved title"],"question":"single-hop question"}, '
         'or {"task":"EOQ"} to return the last title-QA answer. '
         'Use actual completed results; do not invent observations or answer the question yourself.')
CHILD = 'Answer this single-hop Question using only the passages. Return only the short answer. Do not add citations or explanation.'
READER = 'Answer the original Question using these passages. End with Final answer: followed by the short answer.'


from ...answers import extract


def parse(text, scope):
    cleaned = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', text).strip()
    try:
        obj = json.loads(cleaned)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    if obj == {'task': 'EOQ'}:
        return obj
    retrieval_task = 'retrieve_odqa' if scope == 'outer' else 'retrieve'
    qa_task = 'multihop_titleqa' if scope == 'outer' else 'singlehop_titleqa'
    if obj.get('task') == retrieval_task and set(obj) == {'task', 'question'} and isinstance(obj['question'], str) and obj['question'].strip():
        return obj
    if (obj.get('task') == qa_task and set(obj) == {'task', 'question', 'titles'}
        and isinstance(obj['question'], str) and obj['question'].strip()
        and isinstance(obj['titles'], list) and obj['titles']
        and all(isinstance(t, str) and t.strip() for t in obj['titles'])):
        return obj
    return None


def prompt(instructions, demonstrations, question, history):
    return instructions + '\n\n' + demonstrations + '\n\nQC: ' + question + '\n' + '\n'.join(history) + '\nQS:'


class ProtocolFailure(Exception):
    def __init__(self, reason, status='algorithm_failure', request_id=None):
        self.reason, self.status, self.request_id = reason, status, request_id


def request(transport, stage, text, tokens):
    key, result = transport.ask(stage, text, tokens)
    if result['status'] != 'success':
        raise ProtocolFailure('transport_failure', result['status'], key)
    return key, result


def select_titles(task, documents):
    available = {doc['source']: doc for doc in documents.values()}
    if any(title not in available for title in task['titles']):
        raise ProtocolFailure('title_not_in_retrieved_results')
    return [available[title] for title in dict.fromkeys(task['titles'])]


def record_task(store, qid, stage, key, task, answer, history):
    store.put('task', f'{qid}/{stage}', {'request_id': key, 'task': task, 'result': answer})
    history.extend(['QS: ' + json.dumps(task, ensure_ascii=False), 'A: ' + json.dumps(answer, ensure_ascii=False)])


def retrieve_odqa(case, question, step, retrieval, transport, store, config, demos, all_documents):
    history, local_documents, last, last_key = [], {}, None, None
    for inner_step in range(1, config['max_steps'] + 1):
        stage = f'handler_{step}_{inner_step}'
        key, result = request(transport, stage, prompt(INNER, demos, question, history), 512)
        task = parse(result['answer'], 'inner')
        if task is None:
            raise ProtocolFailure('invalid_handler_task', request_id=key)
        if task['task'] == 'EOQ':
            if last is None:
                raise ProtocolFailure('handler_EOQ_without_answer', request_id=key)
            return last, last_key
        if task['task'] == 'retrieve':
            packet = retrieval.retrieve(case['qid'], task['question'], config['k'])
            for doc in packet['docs']:
                local_documents.setdefault(doc['doc_id'], doc)
                all_documents.setdefault(doc['doc_id'], doc)
            answer = [d['source'] for d in packet['docs']]
        else:
            selected = select_titles(task, local_documents)
            context = '\n\n'.join(d['block'] for d in selected)
            last_key, qa = request(transport, stage + '_titleqa', CHILD + '\n\nQuestion: ' + task['question'] + '\nContext:\n' + context + '\nAnswer:', 64)
            last = {'titles': list(dict.fromkeys(task['titles'])), 'answer': [qa['answer']]}
            answer = last
        record_task(store, case['qid'], stage, key, task, answer, history)
    raise ProtocolFailure('handler_task_limit_without_EOQ')


def run(cases, retrieval, transport, store, config):
    demos = json.loads(Path(__file__).with_name('examples.json').read_text(encoding='utf8'))['examples']
    outer_demo = '\n\n'.join(e['decomp'] for e in demos)
    inner_demo = '\n\n'.join(e['retrieve_odqa'] for e in demos)
    reader_demo = '\n\n'.join(e['cot'] for e in demos)
    for case in cases:
        transport.qid = case['qid']
        if store.get('prediction', case['qid']) is not None:
            continue
        history, documents, last, last_key, final_docs = [], {}, None, None, []
        prediction = {'qid': case['qid'], 'question': case['question'], 'prediction': '', 'status': 'algorithm_failure', 'reason': 'outer_task_limit_without_EOQ'}
        try:
            for step in range(1, config['max_steps'] + 1):
                stage = f'decompose_{step}'
                key, result = request(transport, stage, prompt(OUTER, outer_demo, case['question'], history), 512)
                task = parse(result['answer'], 'outer')
                if task is None:
                    raise ProtocolFailure('invalid_outer_task', request_id=key)
                if task['task'] == 'EOQ':
                    if last is None:
                        raise ProtocolFailure('outer_EOQ_without_answer', request_id=key)
                    prediction.update(prediction=last['answer'][0], status='success', reason=None, request_id=last_key)
                    break
                if task['task'] == 'retrieve_odqa':
                    last, last_key = retrieve_odqa(case, task['question'], step, retrieval, transport, store, config, inner_demo, documents)
                else:
                    selected = select_titles(task, documents)
                    context = '\n\n'.join(d['block'] for d in selected)
                    text = READER + '\n\n' + reader_demo + '\n\nRetrieved passages:\n' + context + '\nQuestion: ' + task['question'] + '\nAnswer:'
                    if config['reader'] == 'direct':
                        text = 'Answer the Question using only these passages. Return only the short answer.\n\nContext:\n' + context + '\nQuestion: ' + task['question'] + '\nAnswer:'
                    last_key, qa = request(transport, 'multihop_reader_' + str(step), text, 64 if config['reader'] == 'direct' else 512)
                    last = {'titles': task['titles'], 'answer': [extract(qa['answer'])]}
                    final_docs = [d['doc_id'] for d in selected]
                    prediction['finish_reason'] = qa.get('finish_reason')
                record_task(store, case['qid'], stage, key, task, last, history)
        except ProtocolFailure as exc:
            prediction.update(prediction='' if exc.status == 'algorithm_failure' else None,
                              status=exc.status, reason=exc.reason, request_id=exc.request_id)
        if prediction.get('status')=='success' and prediction.get('prediction')=='':
            prediction.update(status='algorithm_failure',reason='missing_final_answer')
        prediction.update(final_doc_ids=final_docs, all_retrieved_doc_ids=list(documents), completed_tasks=history)
        store.put('prediction', case['qid'], prediction)
        store.export()
