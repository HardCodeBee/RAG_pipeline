"""Close experiment-local records after validated execution and stage inspection."""
from . import run as r
from contextlib import closing
import json
import re


def main():
    r.check()
    report = r.read(r.OUT / 'REPORT.json')
    assert report['status'] == 'completed' and report['unknown_reserved_CNY'] == 0
    old = {v['id']: v['trajectory'] for v in r.rows(r.CONTROL / 'control90.jsonl')}
    revised = r.predictions(r.CONTROL)
    variants = {'old_P02': old, 'revised_control': revised,
        **{a: r.predictions(r.OUT / a) for a in r.ARMS}}
    for name, values in variants.items():
        assert len(values) == 90 and set(values) == set(old)
        assert all(v['status'] == 'success' for v in values.values())
    b = variants['B_no_resolve']
    checked_ordinary, checked_goals = 0, 0
    for cid, control in revised.items():
        for before, after in zip(control['nodes'], b[cid]['nodes']):
            for key in ['prior_complete_answers', 'original_retrieval', 'rewritten_retrieval', 'reader_packet', 'rewritten_query']:
                assert before[key] == after[key], (cid, key)
            if before['role'] == 'ordinary':
                assert before['answer'] == after['answer'] and before['child_prompt'] == after['child_prompt']
                checked_ordinary += 1
            else:
                assert after['child_prompt'] == before['child_prompt'].replace(r.child_prompts.RESOLVE, '', 1)
                checked_goals += 1
    assert (checked_ordinary, checked_goals) == (176, 90)
    selected = {
        'musique:4hop3__129456_29339_508306_70744': [7],
        'musique:2hop__104095_40501': [1, 2],
        'musique:3hop1__106042_64399_53006': [1, 5],
        'hotpotqa:5a90bb4e55429916514e7536': [1, 2],
        '2wiki:923b15ba0bb011ebab90acde48001122': [1, 2, 3],
        '2wiki:6c0c4c1a0bb011ebab90acde48001122': [2, 3],
        'hotpotqa:5a84d2ea5542992a431d1aa8': [3],
        'hotpotqa:5a7a59d15542996c55b2dd67': [1, 2],
        'musique:4hop3__373866_5189_38229_86687': [4],
        'hotpotqa:5a70f9335542994082a3e46a': [1, 2, 3],
        'musique:3hop2__326964_265410_7713': [5],
        'musique:4hop3__275416_24325_156850_10557': [4, 5],
    }
    fields = ['step_id', 'role', 'request', 'answer', 'prior_complete_answers', 'rewritten_query', 'reader_packet', 'original_retrieval', 'rewritten_retrieval']
    evidence = {}
    for cid, steps in selected.items():
        evidence[cid] = {'question': old[cid]['question'], 'variants': {}}
        for name, values in variants.items():
            evidence[cid]['variants'][name] = [{k: values[cid]['nodes'][sid - 1][k] for k in fields} for sid in steps]
    # Verify the material loss against every recorded trajectory, not only A.
    germany = 'musique:4hop3__373866_5189_38229_86687'
    for values in variants.values():
        node = values[germany]['nodes'][-1]
        direct = node['rewritten_retrieval']['docs'][8]
        assert direct['source'] == 'German reunification' and 'In May 1990' in direct['text']
        assert direct['doc_id'] not in {d['doc_id'] for d in node['reader_packet']['docs']}
        nanjing = values['musique:3hop2__326964_265410_7713']['nodes'][-1]
        assert any('capital city of Yangzhou for about 400 years' in d['text'] for d in nanjing['reader_packet']['docs'])
    r.save(r.OUT / 'STAGE_AUDIT_EVIDENCE.json', evidence)
    ref_bearing = {role: sum(bool(re.search(r'#\d+', n['request'])) for v in old.values() for n in v['nodes'] if n['role'] == role) for role in ['ordinary', 'goal']}
    assert ref_bearing == {'ordinary': 0, 'goal': 0}
    result = {'utc': r.utc(), 'status': 'completed', 'planned_per_arm': 90, 'completed_per_arm': 90,
        'protocol_sha256': r.sha(r.HERE / 'PROTOCOL.frozen.json'),
        'B_identical_ordinary_nodes_checked': checked_ordinary, 'B_goal_input_packets_checked': checked_goals,
        'current_questions_with_result_references': ref_bearing,
        'pins_verified': True, 'new_estimated_CNY': report['new_estimated_CNY'], 'unknown_reserved_CNY': report['unknown_reserved_CNY'],
        'main_evaluation_remains_paused': True, 'candidate_not_activated': True,
        'goal_full_rewrite_deferred': True,
        'files': {name: str((r.OUT / name).relative_to(r.ROOT)) for name in ['REPORT.json', 'STAGE_COMPARISON.json', 'CHANGE_REVIEW.json', 'STAGE_AUDIT.md', 'STAGE_AUDIT_EVIDENCE.json']}}
    r.save(r.OUT / 'EXECUTION.json', result)
    r.save(r.OUT / 'FINAL_DELIVERY.json', result)
    entry_id = r.HERE.name.upper() + '_COMPLETED'
    decisions = r.rows(r.PLAN / 'DECISIONS.jsonl')
    assert not any(v.get('id') == entry_id for v in decisions)
    with (r.PLAN / 'DECISIONS.jsonl').open('a', encoding='utf8') as handle:
        handle.write(json.dumps({'id': entry_id, **result,
            'metrics': {a: report['arms'][a]['metrics']['overall'] for a in r.ARMS},
            'scope': 'Two approved fixed90 minimal prompt contrasts, plus offline child/goal retrieval-and-answer inspection requested by user; no additional API evaluation'}, ensure_ascii=False) + '\n')
    r.emit({k: result[k] for k in ['status', 'completed_per_arm', 'B_identical_ordinary_nodes_checked', 'B_goal_input_packets_checked', 'new_estimated_CNY', 'unknown_reserved_CNY', 'main_evaluation_remains_paused']})


if __name__ == '__main__':
    main()
