"""Verify candidate pins; optionally replay all 300 exact recorded trajectories.

No API client, remote connection, embedding model, or retrieval index is opened.
The replay is a consistency check, not a fresh quality evaluation.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1' / HERE.name


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(*, replay=False):
    plan = read(HERE / 'CANDIDATE.frozen.json')
    for name, expected in plan['file_pins'].items():
        assert sha(ROOT / name) == expected, ('Frozen source changed', name)
    assert plan['candidate_id'] == 'p02_multigrain_timing_cachefix_v1'
    assert plan['method']['arm'] == 'timing_only'
    assert plan['method']['goal_preanswer'] and plan['method']['final_overall_answer']
    assert plan['method']['reader_merge']['keep'] == 10
    result = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'candidate_id': plan['candidate_id'], 'passed': True,
        'pinned_files_verified': len(plan['file_pins']),
        'new_API_calls': 0, 'new_index_searches': 0, 'new_remote_calls': 0,
        'replay_requested': replay,
        'source_only_verification_is_not_new_evaluation': True,
    }
    if replay:
        from .pipeline import execute
        from experiments.query_decomposition.p02_merge_cachefix_v1_20261021 import cache
        from experiments.query_decomposition.p02_merge_cachefix_v1_20261021 import offline_audit as old
        from experiments.query_decomposition.component_chain_opt_v2_20261019.run import payload, use_receipt
        from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest

        samples = old.rows(ROOT / plan['evaluation']['runtime'])
        expected = {v['id']: v for v in old.rows(old.CANONICAL_DETAILS)}
        assert len(samples) == len(expected) == 300
        assert Counter(v['dataset'] for v in samples) == {'hotpotqa':100,'2wiki':100,'musique':100}
        catalogs = {ds: cache.catalog_with_audit(ds, include_current=False)[0]
                    for ds in ['hotpotqa', '2wiki', 'musique']}
        counts, details = Counter(), []
        for case in samples:
            wanted = expected[case['id']]
            source = old.read(ROOT / wanted['p02_source'])
            packets, root = old.recorded_packets(source)
            calls, queries = [], []

            def retrieve(query):
                assert query in packets, 'New query requires a separate evaluation'
                queries.append(query)
                return deepcopy(packets[query])

            def ask(stage, prompt, maximum, allow_final_truncation=False):
                request = payload(plan['generation']['model'], prompt, maximum)
                key = digest(request)
                call = wanted['timing_only_replay_calls'][len(calls)]
                assert (stage, key) == (call['stage'], call['payload_sha256'])
                entry = catalogs[case['dataset']][key]
                assert entry['request_id'] == call['request_id']
                assert Path(entry['source']).resolve() == Path(call['source']).resolve()
                sql = old.stored_receipt(entry['source'], entry['request_id'])
                assert request == entry['payload'] == sql['payload']
                assert sql['status'] == entry['status'] == 'success'
                assert sql['result'] == entry['result']
                calls.append({'stage':stage,'payload_sha256':key})
                return use_receipt({**entry, **sql}, stage, allow_final_truncation)

            actual = execute(case, retrieve, ask)
            assert len(calls) == len(wanted['timing_only_replay_calls'])
            for key in ['prediction', 'final_prompt', 'final_context', 'finish_reason']:
                assert actual[key] == source[key], (case['id'], key)
            assert len(actual['nodes']) == len(source['nodes'])
            for node, previous in zip(actual['nodes'], source['nodes']):
                for key, value in previous.items():
                    assert node.get(key) == value, (case['id'], node['step_id'], key)
            assert actual['original_retrieval'] == root
            assert actual['final_reader_packet'] == root
            first_question = case['supporting_questions'][0] if case['supporting_questions'] else case['question']
            assert queries[0] == first_question
            counts.update(cases=1, nodes=len(actual['nodes']), full_payload_calls=len(calls),
                          logical_retrievals=len(queries))
            details.append({'id':case['id'],'dataset':case['dataset'],
                            'canonical_source':wanted['p02_source'],
                            'all_prompts_outputs_packets_exact':True,
                            'calls':calls})
        result.update(replay_counts=dict(counts),
                      all300_equal_frozen_p02=True,
                      complete_payload_success_priority_verified=True,
                      not_fresh_inference=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'REPLAY300.details.jsonl').write_text(
            ''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in details),encoding='utf8')
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / ('VERIFICATION_REPLAY300.json' if replay else 'VERIFICATION_PINS.json')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    verify(replay=args.replay)

