"""Read-only independent audit of final terminal-goal results and reporting."""
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import sqlite3

from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1'
OUT = OUTPUT / HERE.name
DATASETS = ('hotpotqa', '2wiki', 'musique')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


@lru_cache(maxsize=None)
def receipt(source, rid):
    with sqlite3.connect((ROOT / source).resolve().as_uri() + '?mode=ro', uri=True) as db:
        row = db.execute('SELECT payload,status,result FROM requests WHERE id=?', (rid,)).fetchone()
    assert row, (source, rid)
    return {'payload': json.loads(row[0]), 'status': row[1], 'result': json.loads(row[2]) if row[2] else None}


def close(left, right):
    assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (left, right)


def independent_parse(raw, finish):
    lines = raw.strip().splitlines() if isinstance(raw, str) else []
    valid = (finish == 'stop' and len(lines) == 2 and lines[0].startswith('Conclusion:')
             and lines[1].startswith('Final answer:') and bool(lines[0][len('Conclusion:'):].strip())
             and bool(lines[1][len('Final answer:'):].strip()))
    return (valid, lines[1][len('Final answer:'):].strip() if valid else '',
            lines[0][len('Conclusion:'):].strip() if lines and lines[0].startswith('Conclusion:') else None)


def verify():
    import hashlib
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report_path = OUT / 'reports/REPORT.json'
    report = read(report_path)
    assert report['overall']['scorable'] == report['overall']['planned'] == 300
    assert report['overall']['missing'] == 0
    runtime = rows(HERE / 'runtime300.frozen.jsonl')
    refs = {item['id']: item for item in rows(HERE / 'references300.local.frozen.jsonl')}
    sources = {item['id']: item for item in read(HERE / 'UNION_SOURCES.frozen.json')}
    plan = read(HERE / 'PROTOCOL.frozen.json')
    assert len(runtime) == len({item['id'] for item in runtime}) == 300
    assert Counter(item['dataset'] for item in runtime) == {ds: 100 for ds in DATASETS}
    scorers = {ds: metric(ds) for ds in DATASETS}
    f1, em, states, tokens = {}, {}, Counter(), Counter()
    old_call_count, removed_call_count, receipt_checks, docs_checked = 0, 0, 0, 0
    failed_output_details = []
    for case in runtime:
        cid, ds, qid = case['id'], case['dataset'], case['qid']
        source = sources[cid]
        assert sha(ROOT / source['path']) == source['sha256']
        old = read(ROOT / source['path'])
        files = sorted(OUT.glob(f'rounds/r*/cases/{ds}/terminal_goal/{qid}.json'))
        candidates = [(path, read(path)) for path in files]
        terminal = [(path, value) for path, value in candidates if value['status'] in ('success', 'algorithm_failure')]
        assert terminal, 'Missing terminal case: ' + cid
        path, value = terminal[0]
        assert len(terminal) == 1 and not any(p > path for p, _ in candidates), 'Completed case was rerun: ' + cid
        assert value['id'] == cid and value['question'] == case['question']
        assert value['source_union_case'] == source['path'] and value['source_union_sha256'] == source['sha256']
        goal = old['nodes'][-1]
        assert goal['role'] == 'goal' and value['source_goal_step_id'] == goal['step_id']
        assert value['nodes'][:-1] == old['nodes'][:-1]
        assert value['goal_reader_packet'] == goal['reader_packet']
        assert len(value['goal_reader_packet']['docs']) <= 20
        for doc in value['goal_reader_packet']['docs']:
            assert doc['block'] in value['goal_prompt']
            docs_checked += 1
        previous_instruction = 'Return one concise factual sentence preserving the entity, relation and necessary qualifiers.'
        new_instruction = 'The Conclusion line must be one concise factual sentence preserving the entity, relation and necessary qualifiers.'
        assert goal['child_prompt'].endswith('Answer:')
        # Check that source body and GOAL_RULE remain intact, independent of
        # calling the candidate's renderer or mechanical parser.
        expected_prefix = goal['child_prompt'][:-len('Answer:')].replace(previous_instruction, new_instruction, 1)
        assert value['goal_prompt'].startswith(expected_prefix)
        sid = goal['step_id']
        remove = {f'legacy/answer/{sid}', f'legacy/sentence_format/{sid}', 'legacy/final'}
        old_kept = [call for call in old['logical_calls'] if call['stage'] not in remove]
        actual = value['logical_calls']
        assert len(actual) == len(old_kept) + 1
        for call, before in zip(actual[:-1], old_kept):
            assert call.get('reused') is True
            for key in ('stage', 'payload_sha256', 'request_id', 'source'):
                assert call[key] == before[key]
        assert actual[-1]['stage'] == 'terminal_goal/answer'
        assert actual[-1]['request_id'] == value['terminal_goal_request_id']
        assert not any(call['stage'] in remove for call in actual)
        old_call_count += len(old['logical_calls'])
        removed_call_count += len(old['logical_calls']) - len(old_kept)
        for call in actual:
            raw = receipt(call['source'], call['request_id'])
            assert raw['status'] == call['status'] == 'success'
            assert raw['result']['status'] == 'success'
            assert digest(raw['payload']) == call['payload_sha256']
            u = raw['result']['usage']
            assert u['prompt_tokens'] > 0 and u['completion_tokens'] >= 0
            for key in ('prompt_tokens', 'completion_tokens'):
                assert call['usage'][key] == u[key]
                tokens[key] += u[key]
            receipt_checks += 1
        terminal_receipt = receipt(actual[-1]['source'], actual[-1]['request_id'])
        request, response = terminal_receipt['payload'], terminal_receipt['result']
        assert request['messages'] == [{'role': 'user', 'content': value['goal_prompt']}]
        assert request['model'] == plan['model'] and request['temperature'] == 0
        assert request['extra_body']['enable_thinking'] is False and request['max_tokens'] == plan['max_tokens']
        assert response['answer'] == value['raw_goal_output'] and response['finish_reason'] == value['finish_reason']
        valid, prediction, conclusion = independent_parse(response['answer'], response['finish_reason'])
        assert value['parse_result']['valid'] == valid
        assert value['prediction'] == prediction
        assert value['status'] == ('success' if valid else 'algorithm_failure')
        assert value.get('conclusion') == conclusion
        states[value['status']] += 1
        score = scorers[ds](prediction, refs[cid]) if valid else {'f1': 0.0, 'em': 0.0}
        f1[cid], em[cid] = score['f1'], score['em']
        if not valid:
            failed_output_details.append({'id': cid, 'finish_reason': response['finish_reason'],
                                          'raw': response['answer'], 'scored_f1': 0.0})
    assert receipt_checks == old_call_count - removed_call_count + len(runtime)
    assert report['recorded_selected_prefix_usage']['logical_generations'] == receipt_checks
    for key in ('prompt_tokens', 'completion_tokens'):
        assert report['recorded_selected_prefix_usage'][key] == tokens[key]
    close(report['overall']['F1_complete'], sum(f1.values()) / 300)
    close(report['overall']['EM_available'], sum(em.values()) / 300)
    for ds in DATASETS:
        ids = [case['id'] for case in runtime if case['dataset'] == ds]
        assert report['by_dataset'][ds]['scorable'] == report['by_dataset'][ds]['planned'] == 100
        close(report['by_dataset'][ds]['F1_complete'], sum(f1[cid] for cid in ids) / 100)
    modeled_cost = tokens['prompt_tokens'] * .8e-6 + tokens['completion_tokens'] * 2e-6
    close(report['matched_returned_flow_efficiency']['equal_rate_generation_cny']['terminal_goal'], modeled_cost)
    all_physical = []
    for path in sorted(OUT.glob('rounds/r*/methods/*/terminal_goal/checkpoint.sqlite3')):
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            for rid, stage, raw_payload, status in db.execute('SELECT id,stage,payload,status FROM requests'):
                assert stage == 'terminal_goal/answer', 'Legacy physical request was resubmitted'
                all_physical.append((rid, status, digest(json.loads(raw_payload))))
    returned = [entry for entry in all_physical if entry[1] == 'success']
    assert len(returned) == 300
    assert len({entry[2] for entry in returned}) == 300, 'Successful payload was submitted twice'
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        charges = list(db.execute('SELECT namespace,reserved,actual FROM charges'))
    own = [row for row in charges if row[0].startswith(OUT.name)]
    close(report['actual_costs']['new_namespace']['known_returned_usage_cny'], sum(row[2] or 0 for row in own))
    close(report['actual_costs']['new_namespace']['unreconciled_reserved_cny'], sum(row[1] for row in own if row[2] is None))
    global_occupied = sum(row[2] if row[2] is not None else row[1] for row in charges)
    assert global_occupied <= 300
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'native_predictions_independently_parsed_and_scored': 300,
              'per_dataset_n': {ds: 100 for ds in DATASETS}, 'statuses': dict(states),
              'final_F1': sum(f1.values()) / 300,
              'by_dataset_F1': {ds: sum(f1[case['id']] for case in runtime if case['dataset'] == ds) / 100 for ds in DATASETS},
              'full_payload_usage_receipts_checked': receipt_checks,
              'source_union_logical_calls': old_call_count,
              'removed_legacy_goal_formatter_final_calls': removed_call_count,
              'new_terminal_goal_calls': 300, 'new_logical_calls': receipt_checks,
              'prompt_tokens': tokens['prompt_tokens'], 'completion_tokens': tokens['completion_tokens'],
              'equal_rate_modeled_generation_cny': modeled_cost,
              'goal_document_blocks_preserved': docs_checked,
              'ordinary_QA_nodes_and_all_prefix_receipts_unchanged': True,
              'successful_cases_or_payloads_rerun': False,
              'only_terminal_goal_new_physical_requests': True,
              'actual_new_costs_match_namespace_budget': True,
              'global_budget_occupied_cny': global_occupied, 'global_budget_cap_cny': 300,
              'algorithm_failure_raw_outputs_scored_zero': failed_output_details,
              'numeric_noninferiority_margin': None, 'automatic_adoption_claimed': False,
              'report_sha256': sha(report_path), 'verification_source_sha256': sha(__file__),
              'new_API_calls': 0, 'new_index_searches': 0}
    output = OUT / 'reports/FINAL_VERIFICATION.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({key: value for key, value in result.items() if key != 'algorithm_failure_raw_outputs_scored_zero'}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    verify()
