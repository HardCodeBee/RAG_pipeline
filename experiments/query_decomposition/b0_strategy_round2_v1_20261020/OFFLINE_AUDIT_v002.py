"""Read-only execution audit. Never calls prepare(), run(), API or retrieval."""
from collections import Counter
import inspect
import json
from pathlib import Path
import sqlite3

from . import experiment as e
from . import strategy as s
from experiments.query_decomposition.b0_six_single_changes_v1_20261020.receipts import receipt
from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest, utc


def main():
    e.verify(e.read(e.HERE / 'PROTOCOL.frozen.json'))
    cases = e.rows(e.PREV_E / 'runtime300.frozen.jsonl')
    readout = e.read(e.PREV_E / 'PROTOCOL.frozen.json')['readout']
    source = e.sources()
    counters, failures = Counter(), []
    source_before = {str(p): e.sha(p) for p in [Path(e.__file__), Path(s.__file__)]}
    for case in cases:
        try:
            base = e.baseline_case(source[case['id']])
            pp, parent = e.parent_case_path(case['id'], 'g01_goal_consistency')
            fp, final_sibling = e.parent_case_path(case['id'], 'f01_final_consistency')
            assert base['nodes'][:-1] == parent['nodes'][:-1]
            assert e.goal_prompt(base['nodes'][-1]['child_prompt'], 'parent_g01') == parent['nodes'][-1]['child_prompt']
            assert parent['nodes'][-1]['role'] == 'goal'
            assert parent['nodes'][-1]['request'] == case['question']
            assert len(base['nodes']) == len(parent['nodes'])
            counters['identical_ordinary_nodes'] += 1
            # Ancestor is the literal completed P1 control, not its score-table row.
            ancestor_path = e.ROOT / parent['parent_source']
            ancestor = e.read(ancestor_path)
            assert ancestor['nodes'] == base['nodes']
            assert ancestor['final_context'] == base['final_context']
            counters['ancestor_source_resolves_to_P1_with_B0_nodes'] += 1
            parent_prompt, parent_context = e.final_prompt(case, parent['nodes'], base['original_retrieval'], readout, 'parent_g01')
            assert parent_prompt == parent['final_prompt']
            assert parent_context == parent['final_context']
            counters['rebuilt_parent_final_exact'] += 1
            expected = payload(e.read(e.PREV_E / 'PROTOCOL.frozen.json')['model'], parent_prompt, 64)
            calls = [c for c in parent['logical_calls'] if c['payload_sha256'] == digest(expected)]
            assert calls and all(receipt(c, expected)['status'] == 'success' for c in calls)
            counters['actual_parent_final_receipt_payload_exact'] += 1
            # Reconstruct root original evidence from baseline source. No new retrieval.
            assert e.b0.passages(base['original_retrieval']) in parent_context
            counters['original_retrieval_from_original_B0_source'] += 1
            changed = e.altered_goal_prompt(parent['nodes'][-1]['child_prompt'])
            assert changed.replace(s.GOAL_EVIDENCE_RULE, e.b0.legacy_child.GOAL_RULE, 1) == parent['nodes'][-1]['child_prompt']
            assert s.GOAL_EVIDENCE_RULE not in parent['nodes'][-1]['child_prompt']
            assert changed == s.goal_prompt(base['nodes'][-1]['child_prompt'], 'g02_evidence_goal')
            counters['g02_only_replaces_goal_rule'] += 1
            p02_prompt, p02_context = e.final_prompt(case, parent['nodes'], base['original_retrieval'], readout, 'p02_role_headers')
            cleaned_context = p02_context.replace('Intermediate candidate results\n\n', '').replace('Candidate conclusion for the Original question\n\n', '')
            cleaned_prompt = p02_prompt.replace('Intermediate candidate results\n\n', '').replace('Candidate conclusion for the Original question\n\n', '')
            assert cleaned_context == parent_context
            assert cleaned_prompt == parent_prompt
            counters['p02_only_adds_two_role_headers'] += 1
            s03_prompt, s03_context = e.final_prompt(case, parent['nodes'], base['original_retrieval'], readout, 's03_g01_plus_f01')
            assert s03_context == parent_context
            assert s03_prompt == parent_prompt[:-len(readout)] + s.FINAL_CHECK + readout
            counters['s03_only_adds_existing_final_check'] += 1
            # Cache candidates are actual same-case lineage and exact receipts.
            for value in [parent, ancestor, final_sibling]:
                assert value['qid'] == case['qid'] and value['dataset'] == case['dataset']
                for call in value['logical_calls']:
                    if call['status'] != 'success': continue
                    r = receipt(call)
                    assert r['status'] == 'success'
                    with sqlite3.connect((e.ROOT / call['source']).resolve().as_uri() + '?mode=ro', uri=True) as db:
                        body = json.loads(db.execute('SELECT payload FROM requests WHERE id=?', (call['request_id'],)).fetchone()[0])
                    assert digest(body) == call['payload_sha256']
                    counters['successful_cached_payload_hash_verified'] += 1
            counters['all_cases_passed'] += 1
        except Exception as exc:
            failures.append({'id': case['id'], 'type': type(exc).__name__, 'reason': str(exc)})
    text = inspect.getsource(e.run_case)
    static = {
        'run_case_uses_deepcopy_g01_nodes': "nodes = deepcopy(control['nodes'])" in text,
        'run_case_uses_original_base_retrieval': "base['original_retrieval']" in text,
        'retry_reads_old_same_dataset_arm_DB': "old_round / 'methods' / case['dataset'] / arm / 'checkpoint.sqlite3'" in text,
        'retry_filters_same_qid_and_success': "json_extract(value,'$.qid')=?" in text and "json_extract(value,'$.status')='success'" in text,
        'cache_receipt_checks_expected_full_payload': 'receipt(old, req)' in text,
        'new_round_preserves_unknown_reserves': 'budget.db.close()' in text and 'DELETE FROM charges' not in text,
    }
    for name, before in source_before.items(): assert e.sha(Path(name)) == before, 'Concurrent source edit: rerun audit'
    report = {'utc': utc(), 'cases': len(cases), 'counts': dict(counters), 'failures': failures,
              'static_checks': static, 'source_sha256': source_before,
              'new_API_calls': 0, 'new_retrieval_calls': 0, 'prepare_invoked': False,
              'scope': 'Existing300 real parent/prefix/receipt reconstruction plus static retry review. Does not validate live API recovery.'}
    path = Path(__file__).with_name('OFFLINE_AUDIT.v002.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    assert not failures and all(static.values())


if __name__ == '__main__': main()
