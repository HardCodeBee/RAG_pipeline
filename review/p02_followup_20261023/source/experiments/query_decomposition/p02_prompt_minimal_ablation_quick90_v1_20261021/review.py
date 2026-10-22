"""Offline paired inspection after both arms; no extra model or judge calls."""
from . import run as r
import json
import re


def main():
    report = r.read(r.OUT / 'REPORT.json')
    assert report['status'] == 'completed'
    controls = r.predictions(r.CONTROL)
    refs = {v['id']: v for v in r.rows(r.COHORT / 'references90.local.frozen.jsonl')}
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    scorers = {dataset: metric(dataset) for dataset in r.parent.DATASETS}
    reviews, summaries = {}, {}
    # A text marker to locate explicit uncertainty for inspection, not a new correctness metric.
    uncertain = re.compile(r'cannot be determined|can(?:not|\'t) determine|unable to determine|insufficient (?:information|evidence)|not enough (?:information|evidence)|does not (?:provide|specify|state|establish)|do not (?:provide|specify|state|establish)|unknown from', re.I)
    for arm in r.ARMS:
        values = r.predictions(r.OUT / arm)
        changed, touched, text_missing_control, text_missing_arm = [], [], [], []
        for cid, control in controls.items():
            candidate = values[cid]
            score = scorers[control['dataset']]
            before, after = score(control['prediction'], refs[cid]), score(candidate['prediction'], refs[cid])
            differences = []
            for a, b in zip(control['nodes'], candidate['nodes']):
                same_input = all(a[k] == b[k] for k in ['request', 'prior_complete_answers', 'reader_packet'])
                if a['role'] == 'ordinary':
                    if uncertain.search(a['answer']):
                        text_missing_control.append({'id': cid, 'step': a['step_id'], 'answer': a['answer']})
                    if uncertain.search(b['answer']):
                        text_missing_arm.append({'id': cid, 'step': b['step_id'], 'answer': b['answer']})
                if a['answer'] != b['answer'] or a['rewritten_query'] != b['rewritten_query'] or not same_input:
                    differences.append({'step_id': a['step_id'], 'role': a['role'], 'question': a['request'],
                        'same_input_excluding_prompt': same_input, 'control_answer': a['answer'], 'arm_answer': b['answer'],
                        'control_rewrite': a['rewritten_query'], 'arm_rewrite': b['rewritten_query']})
            row = {'id': cid, 'question': control['question'], 'reference': refs[cid],
                'control_prediction': control['prediction'], 'arm_prediction': candidate['prediction'],
                'control_score': before, 'arm_score': after, 'delta_F1': after['f1'] - before['f1'], 'node_differences': differences}
            if differences or control['prediction'] != candidate['prediction']:
                touched.append(row)
            if abs(row['delta_F1']) > 1e-12:
                changed.append(row)
        reviews[arm] = {'score_changed': changed, 'text_or_trajectory_changed': touched,
            'explicit_uncertainty_text_control': text_missing_control, 'explicit_uncertainty_text_arm': text_missing_arm}
        summaries[arm] = {'score_changed': len(changed), 'text_or_trajectory_changed': len(touched),
            'ordinary_answers_changed': sum(a['answer'] != b['answer'] for cid in controls for a, b in zip(controls[cid]['nodes'][:-1], values[cid]['nodes'][:-1])),
            'goal_answers_changed': sum(controls[cid]['nodes'][-1]['answer'] != values[cid]['nodes'][-1]['answer'] for cid in controls),
            'explicit_uncertainty_text_control': len(text_missing_control), 'explicit_uncertainty_text_arm': len(text_missing_arm),
            'uncertainty_marker_is_not_a_correctness_metric': True}
    r.save(r.OUT / 'CHANGE_REVIEW.json', {'summary': summaries, 'arms': reviews})
    r.emit({'summary': summaries, 'score_changed': {a: reviews[a]['score_changed'] for a in r.ARMS}})


if __name__ == '__main__':
    main()
