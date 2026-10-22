"""Frozen P02 QA/raw packet construction and guarded final execution.

This module has no model, network client, retrieval constructor or scheduling
side effect. Prefix eligibility is established by replay against durable stage
receipts. The replay deliberately stops before the native P02 final request.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import common
from src.p02_multigrain_timing_cachefix_v1.core import pipeline

HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE / 'reviews/20261023_narrative_revision007/qa_raw'
METHOD = 'p02_multigrain_timing_cachefix_v1'
VARIANT = 'p02_shared_baseline_v2'
SCHEMA = 'p02_shared_prefix_v1'
ARMS = ('QA', 'Raw')
FINAL_METHOD = 'p02_qa_raw_final_interface'


class PrefixBlocked(ValueError):
    """The fixed question remains pending; a different question cannot replace it."""


def fingerprint(value):
    return common.digest(value)


def payload(prompt):
    return {'model': common.MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0., 'max_tokens': common.MAX_OUTPUT,
            'extra_body': {'enable_thinking': False}}


def _usage(value):
    u = value.get('usage')
    if not isinstance(u, dict) or any(type(u.get(k)) is not int or u[k] < 0
                                     for k in ('prompt_tokens', 'completion_tokens')):
        return None
    return u


def _fail(condition, reason):
    if not condition:
        raise PrefixBlocked(reason)


def _receipt_ok(receipt, expected, persisted, case, namespace):
    source = receipt.get('source', {})
    identity = {'dataset': case['dataset'], 'qid': case['qid'], 'method': METHOD,
                'execution_variant': VARIANT, 'resource_namespace': namespace}
    _fail(all(source.get(k) == v for k, v in identity.items()), 'receipt_source_identity_mismatch')
    _fail(source.get('kind') in {'shared_layer', 'c1'}, 'unvalidated_prefix_source')
    _fail(isinstance(persisted, dict), 'missing_durable_receipt')
    for k in ('dataset', 'qid', 'method'):
        _fail(persisted.get(k) == identity[k], 'durable_receipt_identity_mismatch:' + k)
    _fail(persisted.get('stage') == receipt['stage'], 'durable_receipt_stage_mismatch')
    _fail(persisted.get('request_id') == receipt['request_id'], 'durable_request_id_mismatch')
    if source['kind'] == 'shared_layer':
        _fail(persisted.get('origin') == 'new_runtime', 'shared_layer_durable_origin_mismatch')
    else:
        _fail(persisted.get('origin') == 'c1_historical', 'c1_durable_origin_missing')
        proof = persisted.get('provenance', {}).get('c1_validation', {})
        _fail(all(proof.get(k) == v for k, v in {
            'dataset': case['dataset'], 'qid': case['qid'], 'method': METHOD,
            'stage': receipt['stage']}.items()), 'c1_durable_stage_binding_missing')
        _fail(all(proof.get(k) is True for k in (
            'full_messages_settings_equal_except_nonbinding_output_cap',
            'complete_stop_usage_model_verified', 'resources_verified')) and
            type(proof.get('current_retrieval_packets_verified')) is int and
            proof['current_retrieval_packets_verified'] > 0 and
            isinstance(proof.get('source_meta_sha256'), str) and
            len(proof['source_meta_sha256']) == 64, 'c1_durable_resource_validation_missing')
    actual, result = receipt.get('payload'), receipt.get('result')
    _fail(isinstance(actual, dict) and isinstance(result, dict), 'missing_payload_or_result')
    _fail(persisted.get('payload') == actual and persisted.get('result') == result,
          'durable_receipt_payload_or_result_mismatch')
    _fail(persisted.get('status', result.get('status')) == 'success', 'durable_receipt_not_success')
    _fail(result.get('status') == 'success' and result.get('finish_reason') == 'stop', 'upstream_incomplete_response')
    usage = _usage(result)
    _fail(usage is not None, 'upstream_missing_usage')
    _fail(result.get('returned_model') == common.MODEL, 'upstream_model_identity_mismatch')
    _fail(isinstance(result.get('answer'), str) and bool(result['answer'].strip()), 'upstream_empty_answer')
    _fail(usage['prompt_tokens'] <= common.MAX_INPUT and
          usage['prompt_tokens'] + common.MAX_OUTPUT <= common.CONTEXT, 'upstream_input_overflow')
    if actual != expected:
        _fail(source.get('kind') == 'c1', 'shared_layer_payload_changed')
        _fail({k: v for k, v in actual.items() if k != 'max_tokens'} ==
              {k: v for k, v in expected.items() if k != 'max_tokens'}, 'c1_payload_changed')
        cap = actual.get('max_tokens')
        _fail(type(cap) is int and 0 < cap <= common.MAX_OUTPUT and
              usage['completion_tokens'] < cap, 'historical_output_cap_not_proven_nonbinding')
    _fail(usage['completion_tokens'] <= actual['max_tokens'], 'upstream_usage_exceeds_output_cap')
    return usage


def validate_prefix(prefix, *, expected_case, expected_namespace,
                    expected_prefix_sha256, receipt_resolver):
    """Replay saved upstream; resolver must read authoritative saved receipts.

    ``receipt_resolver(receipt)`` returns an independently loaded dict containing
    request_id, dataset, method, qid, stage, payload, status and result. A caller
    must obtain expected_prefix_sha256 from the main run's unique binding, not
    from the proposed prefix itself. This function neither migrates old records
    nor establishes their provenance merely from a ``trusted`` boolean.
    """
    case = pipeline.safe_case(expected_case)
    _fail(isinstance(prefix, dict) and prefix.get('schema') == SCHEMA, 'prefix_schema_mismatch')
    _fail(callable(receipt_resolver), 'durable_receipt_resolver_required')
    _fail(prefix.get('case') == case, 'frozen_question_or_decomposition_mismatch')
    _fail(prefix.get('method') == METHOD and prefix.get('execution_variant') == VARIANT, 'prefix_method_mismatch')
    _fail(isinstance(expected_namespace, str) and bool(expected_namespace) and
          prefix.get('resource_namespace') == expected_namespace, 'retrieval_resource_binding_mismatch')
    _fail(bool(expected_prefix_sha256) and fingerprint(prefix) == expected_prefix_sha256,
          'main_prefix_binding_mismatch')
    nodes = prefix.get('nodes')
    _fail(isinstance(nodes, list) and len(nodes) == len(case['supporting_questions']) + 1,
          'incomplete_ordinary_goal_prefix')
    by_query = {}
    for i, node in enumerate(nodes, 1):
        _fail(node.get('step_id') == i, 'non_contiguous_steps')
        goal = i == len(nodes)
        _fail(node.get('role') == ('goal' if goal else 'ordinary'), 'goal_must_be_unique_and_last')
        _fail(node.get('request') == (case['question'] if goal else case['supporting_questions'][i-1]),
              'node_question_binding_mismatch')
        for name in ('original_retrieval', 'rewritten_retrieval'):
            packet = node.get(name)
            if packet is None and name == 'rewritten_retrieval':
                continue
            _fail(isinstance(packet, dict), 'missing_retrieval_packet')
            _fail(packet.get('qid') == case['qid'] and packet.get('k') == 10, 'retrieval_identity_mismatch')
            docs = packet.get('docs', [])
            _fail(len(docs) == 10 and len({d.get('doc_id') for d in docs}) == 10, 'retrieval_top10_contract')
            _fail(packet.get('encoder_truncated') is False, 'retrieval_completeness_unproven')
            _fail(type(packet.get('query_tokens')) is int and packet['query_tokens'] >= 0,
                  'retrieval_token_metadata_missing')
            if packet.get('query_tokens', 0) > 512:
                _fail(packet.get('processed_content_tokens') == packet.get('content_tokens') and
                      type(packet.get('content_tokens')) is int and packet.get('window_count', 0) > 1 and
                      bool(packet.get('encoded_window_tokens')) and
                      all(0 < n <= 512 for n in packet['encoded_window_tokens']),
                      'retrieval_complete_windows_missing')
            query = packet.get('query')
            _fail(isinstance(query, str), 'retrieval_query_missing')
            if query in by_query:
                _fail(by_query[query] == packet, 'same_query_different_retrieval')
            by_query[query] = packet
    _fail(prefix.get('original_retrieval') == nodes[-1].get('original_retrieval'), 'root_goal_retrieval_mismatch')
    calls = prefix.get('receipts', [])
    by_stage = {}
    for call in calls:
        _fail(isinstance(call, dict) and isinstance(call.get('stage'), str), 'receipt_stage_missing')
        _fail(call['stage'] != 'legacy/final', 'native_final_must_not_enter_prefix')
        _fail(call['stage'] not in by_stage, 'duplicate_logical_stage_receipt')
        by_stage[call['stage']] = call
    used, replayed, usages = [], [], []

    class PrefixComplete(Exception):
        pass

    def retrieve(query):
        _fail(query in by_query, 'replay_retrieval_missing')
        return deepcopy(by_query[query])

    def ask(stage, prompt, maximum, allow_final_truncation=False):
        if stage == 'legacy/final':
            raise PrefixComplete()
        _fail(stage in by_stage, 'missing_dependency_receipt:' + stage)
        call = by_stage[stage]
        result = call['result']
        try:
            persisted = receipt_resolver(call)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise PrefixBlocked('durable_receipt_unavailable:' + stage) from exc
        usages.append(_receipt_ok(call, payload(prompt), persisted, case, expected_namespace))
        used.append(stage)
        return call['request_id'], result['answer'], result['finish_reason']

    try:
        pipeline.execute(case, retrieve, ask, record_node=replayed.append)
    except PrefixComplete:
        pass
    except PrefixBlocked:
        raise
    except (KeyError, TypeError, ValueError, AssertionError, RuntimeError) as exc:
        raise PrefixBlocked('prefix_replay_failed:' + str(exc)) from exc
    else:
        raise PrefixBlocked('native_final_boundary_not_reached')
    _fail(set(used) == set(by_stage), 'unconsumed_dependency_receipts')
    _fail(replayed == nodes, 'saved_nodes_do_not_match_receipt_replay')
    return {'prefix_sha256': expected_prefix_sha256,
            'ordinary_nodes': len(nodes)-1, 'goal_nodes': 1,
            'logical_prompt_tokens': sum(u['prompt_tokens'] for u in usages),
            'logical_completion_tokens': sum(u['completion_tokens'] for u in usages),
            'logical_cost_cny': sum(common.usage_cost(u['prompt_tokens'], u['completion_tokens']) for u in usages),
            'native_final_required': False,
            'source_request_ids': [by_stage[s]['request_id'] for s in used]}


def escape_metadata(value):
    """Lossless metadata escaping v1: backslash, CR, LF, tab, pipe and brackets."""
    _fail(isinstance(value, str), 'document_metadata_must_be_string')
    return (value.replace('\\', '\\\\').replace('\r', '\\r').replace('\n', '\\n')
            .replace('\t', '\\t').replace('|', '\\|').replace('[', '\\[').replace(']', '\\]'))


def build_pair(prefix, *, expected_case, expected_namespace, expected_prefix_sha256,
               receipt_resolver, token_counter=None):
    upstream = validate_prefix(prefix, expected_case=expected_case,
                               expected_namespace=expected_namespace,
                               expected_prefix_sha256=expected_prefix_sha256,
                               receipt_resolver=receipt_resolver)
    registry, root_blocks, auxiliary_blocks, references = {}, [], [], []

    def register(doc, *, root=False):
        _fail(all(isinstance(doc.get(k), str) and bool(doc[k]) for k in ('doc_id', 'source', 'text')),
              'reader_document_fields_missing')
        key = (expected_namespace, doc['doc_id'])
        content = (doc['source'], doc['text'])
        if key in registry:
            label, previous = registry[key]
            _fail(previous == content, 'same_document_id_different_source_or_text')
            return label
        target = root_blocks if root else auxiliary_blocks
        label = ('R' if root else 'A') + str(len(target) + 1)
        registry[key] = (label, content)
        target.append(f'[Passage {label} | doc_id={escape_metadata(doc["doc_id"])} | source={escape_metadata(doc["source"])}]\n{doc["text"]}')
        return label

    root_docs = prefix['original_retrieval']['docs']
    for doc in root_docs:
        register(doc, root=True)
    qa_rows, raw_rows = [], []
    for node in prefix['nodes']:
        docs = node.get('reader_packet', {}).get('docs')
        _fail(isinstance(docs, list) and len(docs) == 10, 'actual_reader_top10_missing')
        labels = [register(doc) for doc in docs]
        title = f'Step {node["step_id"]} ({node["role"]})\nQuestion: {node["request"]}'
        qa_rows.append(title + '\nAnswer: ' + node['answer'])
        raw_rows.append(title + '\nPassages: ' + ', '.join(labels))
        references.append({'step_id': node['step_id'], 'question': node['request'], 'labels': labels,
                           'doc_ids': [doc['doc_id'] for doc in docs]})
    root_text = '\n\n'.join(root_blocks)
    auxiliary = {'QA': '\n\n'.join(qa_rows),
                 'Raw': '\n\n'.join(raw_rows + auxiliary_blocks)}
    template_bytes = (PROTOCOL_DIR / 'PROMPT.txt').read_bytes()
    template = template_bytes.decode('utf8')
    arms = {}
    for arm in ARMS:
        prompt = template.format(question=expected_case['question'], root_passages=root_text,
                                 auxiliary_material=auxiliary[arm])
        upper = len(prompt.encode('utf8')) + 256
        count = token_counter(prompt) if token_counter else None
        _fail(count is None or type(count) is int and count >= 0, 'invalid_model_token_count')
        safe = upper <= common.MAX_INPUT and upper + common.MAX_OUTPUT <= common.CONTEXT
        if count is not None:
            safe = safe and count <= common.MAX_INPUT and count + common.MAX_OUTPUT <= common.CONTEXT
        arms[arm] = {'prompt': prompt, 'payload': payload(prompt),
                     'payload_sha256': fingerprint(payload(prompt)),
                     'input_check': {'model_tokens': count, 'model_tokens_source': 'provided_model_tokenizer' if count is not None else 'unavailable',
                                     'prompt_utf8_bytes_plus_256': upper, 'passed': safe,
                                     'max_input': common.MAX_INPUT, 'max_output': common.MAX_OUTPUT,
                                     'context': common.CONTEXT, 'truncated': False}}
    case = prefix['case']
    first = sha256(f'20261023:{case["dataset"]}:qa_raw_arm_order:{case["qid"]}'.encode()).digest()[0] % 2
    pair = {'schema': 'p02_qa_raw_pair_v1', 'dataset': case['dataset'], 'qid': case['qid'],
            'question': case['question'], 'resource_namespace': expected_namespace,
            'prefix_sha256': expected_prefix_sha256, 'upstream': upstream,
            'prompt_template_sha256': sha256(template_bytes).hexdigest(),
            'metadata_escaping': 'backslash_cr_lf_tab_pipe_brackets_v1',
            'root_passages': root_text, 'reader_associations': references,
            'arms': arms, 'arm_order': list(ARMS if first == 0 else reversed(ARMS)),
            'status': 'ready' if all(a['input_check']['passed'] for a in arms.values()) else 'input_overflow',
            'native_final_required': False, 'new_upstream_cost_cny': 0}
    pair['pair_sha256'] = fingerprint(pair)
    return pair


def prepare_manifest(cohort, prefixes, cases, *, expected_namespace, bindings,
                     receipt_resolver, token_counter=None):
    """Keep every fixed ID, including missing upstream and invalid prefixes."""
    rows, seen = [], set()
    for dataset, group in cohort['datasets'].items():
        for query in group['queries']:
            qid = query['qid']
            key = dataset + ':' + qid
            _fail(key not in seen, 'duplicate_fixed_cohort_id')
            seen.add(key)
            row = {'dataset': dataset, 'qid': qid, 'status': 'pending_upstream',
                   'QA': {'status': 'pending_upstream'}, 'Raw': {'status': 'pending_upstream'}}
            prefix = prefixes.get(key)
            if prefix is not None:
                try:
                    pair = build_pair(prefix, expected_case=cases[key], expected_namespace=expected_namespace,
                                      expected_prefix_sha256=bindings.get(key), receipt_resolver=receipt_resolver,
                                      token_counter=token_counter)
                    row.update(status=pair['status'], pair=pair)
                    for arm in ARMS:
                        row[arm]['status'] = pair['status']
                except (PrefixBlocked, KeyError) as exc:
                    row.update(status='upstream_not_evaluable', reason=str(exc))
                    for arm in ARMS:
                        row[arm]['status'] = 'upstream_not_evaluable'
            rows.append(row)
    _fail(len(rows) == cohort['planned_queries'], 'fixed_cohort_count_mismatch')
    return {'schema': 'p02_qa_raw_manifest_v1', 'planned_queries': len(rows),
            'planned_final_conditions': len(rows)*2, 'rows': rows,
            'execution_authorized': False, 'question_replacements': 0}


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temp.replace(path)


def _settle(result, pair, arm, rid, score):
    usage = _usage(result)
    cost = common.usage_cost(usage['prompt_tokens'], usage['completion_tokens']) if usage else None
    value = {'status': 'technical_held', 'request_id': rid, 'receipt': deepcopy(result),
             'usage': usage, 'final_logical_cost_cny': cost,
             'upstream_logical_cost_cny': pair['upstream']['logical_cost_cny'],
             'arm_logical_cost_cny': pair['upstream']['logical_cost_cny'] + cost if cost is not None else None,
             'actual_new_expenditure_source': 'runtime_ledger_unique_final_attempts',
             'new_upstream_cost_cny': 0, 'prediction': None, 'score': None}
    status = result.get('status')
    if usage is None:
        value['reason'] = 'missing_usage_or_receipt'
    elif result.get('returned_model') != common.MODEL:
        value['reason'] = 'model_identity_mismatch'
    elif usage['prompt_tokens'] > common.MAX_INPUT or usage['prompt_tokens'] + common.MAX_OUTPUT > common.CONTEXT or usage['completion_tokens'] > common.MAX_OUTPUT:
        value['reason'] = 'usage_context_integrity_violation'
    elif status == 'success' and result.get('finish_reason') == 'stop' and isinstance(result.get('answer'), str) and result['answer'].strip():
        value.update(status='success', prediction=result['answer'])
        if score:
            value['score'] = score(pair['dataset'], pair['qid'], result['answer'])
    elif status in {'success', 'algorithm_failure', 'integrity_failure'} and (
            result.get('finish_reason') == 'length' and usage['completion_tokens'] == common.MAX_OUTPUT or
            result.get('finish_reason') == 'stop' and isinstance(result.get('answer'), str) and
            (status == 'algorithm_failure' or not result['answer'].strip())):
        value.update(status='algorithm_failure', reason='length_or_final_format_failure', score={'f1': 0., 'em': 0.})
    else:
        value['reason'] = result.get('reason') or status or 'unknown_receipt'
    return value


def _recovered_result(receipt, pair, arm, rid):
    _fail(receipt.get('payload') == pair['arms'][arm]['payload'], 'recovered_final_payload_changed')
    _fail(receipt.get('request_id') == rid and receipt.get('method') == FINAL_METHOD and
          receipt.get('dataset') == pair['dataset'] and receipt.get('qid') == pair['qid'] and
          receipt.get('stage') == f'qa_raw/{arm}/final', 'recovered_final_identity_changed')
    return receipt.get('result') or {'status': receipt.get('status', 'unknown')}


def run_pair(pair, transport, output_dir, *, guard, receipt_lookup=None, score=None):
    """Execute only the two final arms through the caller's guarded transport.

    Unknown/in-flight/missing-usage requests are never resubmitted here. A saved
    held arm can advance only via an independently recovered durable receipt.
    Successful or algorithm-failure arms remain immutable across restarts.
    """
    if not callable(guard):
        raise ValueError('Execution approval guard is required')
    _fail(fingerprint({k: v for k, v in pair.items() if k != 'pair_sha256'}) == pair.get('pair_sha256'),
          'prepared_pair_was_changed')
    _fail(pair.get('status') == 'ready', 'pair_input_not_ready')
    _fail(getattr(transport, 'dataset', None) == pair['dataset'] and
          getattr(transport, 'qid', None) == pair['qid'] and
          getattr(transport, 'method', None) == FINAL_METHOD, 'final_transport_question_binding_mismatch')
    path = Path(output_dir) / (fingerprint([pair['dataset'], pair['qid']]) + '.json')
    state = json.loads(path.read_text(encoding='utf8')) if path.exists() else {
        'schema': 'p02_qa_raw_result_v1', 'dataset': pair['dataset'], 'qid': pair['qid'],
        'prefix_sha256': pair['prefix_sha256'], 'pair_sha256': pair['pair_sha256'],
        'arms': {arm: {'status': 'pending'} for arm in ARMS}}
    _fail(state.get('pair_sha256') == pair['pair_sha256'], 'saved_pair_binding_changed')
    for arm in pair['arm_order']:
        previous = state['arms'][arm]
        if previous['status'] in {'success', 'algorithm_failure'}:
            continue
        retryable = previous.get('receipt', {}).get('status') == 'failed'
        if previous['status'] in {'technical_held', 'dispatching'} and not retryable:
            if receipt_lookup and previous.get('request_id'):
                recovered = receipt_lookup(previous['request_id'])
                if recovered:
                    state['arms'][arm] = _settle(_recovered_result(recovered, pair, arm, previous['request_id']),
                                                 pair, arm, previous['request_id'], score)
                    _save(path, state)
            continue
        # The guard is checked immediately before each actual transport path;
        # a pause between arms preserves the completed first arm.
        for _ in range(5):
            if guard() is False:
                raise common.IntegrityError('Execution guard denied submission')
            stage, prompt = f'qa_raw/{arm}/final', pair['arms'][arm]['prompt']
            rid = transport.request_id_for(stage, prompt, common.MAX_OUTPUT)
            state['arms'][arm] = {'status': 'dispatching', 'request_id': rid,
                                  'payload_sha256': pair['arms'][arm]['payload_sha256']}
            _save(path, state)
            try:
                actual_rid, result = transport.ask(stage, prompt, common.MAX_OUTPUT)
                _fail(actual_rid == rid, 'transport_changed_prebound_request_id')
            except common.IntegrityError as exc:
                recovered = receipt_lookup(rid) if receipt_lookup else None
                result = _recovered_result(recovered, pair, arm, rid) if recovered else {'status': 'unknown'}
                state['arms'][arm] = _settle(result, pair, arm, rid, score)
                state['arms'][arm]['hold_reason'] = str(exc)
                _save(path, state)
                break
            state['arms'][arm] = _settle(result, pair, arm, rid, score)
            _save(path, state)
            if result.get('status') != 'failed':
                break
    state['complete_pair'] = all(a['status'] in {'success', 'algorithm_failure'} for a in state['arms'].values())
    _save(path, state)
    return state
