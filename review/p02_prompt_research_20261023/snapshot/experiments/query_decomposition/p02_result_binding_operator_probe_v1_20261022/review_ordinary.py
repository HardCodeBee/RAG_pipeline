"""Reuse only identical prior statements under identical actual-task context."""
from contextlib import closing
import json
from . import run

def main():
    inputs=run.rows(run.HERE/'ORDINARY38.frozen.jsonl')
    with closing(run.ro(run.OUT/'checkpoint.sqlite3')) as db:
        results=[json.loads(v) for v, in db.execute("SELECT value FROM data WHERE kind='prediction'")]
    results=[r for r in results if r['suite']=='ordinary']
    existing=[]
    old_name='p02_ordinary_fewshot_probe_v1_20261022'
    old_inputs={r['probe_id']:r for r in run.rows(run.HERE.parent/old_name/'INPUTS.frozen.jsonl')}
    label_path=run.ROOT/'outputs/query_decomposition'/old_name/'root_blind_review/SEMANTIC_JOIN_POSTSUPPLEMENT.json'
    labels={(r['case_id'],r['arm']):r for r in run.read(label_path)['records']}
    with closing(run.ro(label_path.parents[1]/'checkpoint.sqlite3')) as db:
        for raw, in db.execute("SELECT value FROM data WHERE kind='outcome'"):
            r=json.loads(raw); label=labels.get((r['probe_id'],r['arm']))
            if label and r['status']=='success':
                existing.append(dict(actual=old_inputs[r['probe_id']]['control_prompt'].split('\n\nActual task:\n',1)[1],
                    answer=r['answer'],label=label,source=str(label_path.relative_to(run.ROOT)),source_sha256=run.sha(label_path)))
    old_name='p02_reference_binding_priority_probe_v1_20261022'
    label_path=run.ROOT/'outputs/query_decomposition'/old_name/'root_blind_review/initial8_20261022/SEMANTIC_JOIN.json'
    labels={(r['case_id'],r['arm']):r for r in run.read(label_path)['records']}
    with closing(run.ro(label_path.parents[2]/'checkpoint.sqlite3')) as db:
        for raw, in db.execute("SELECT value FROM data WHERE kind='outcome'"):
            r=json.loads(raw); label=labels.get((r['probe_id'],r['arm']))
            if label and r['status']=='success' and '\n\nActual task:\n' in r['actual_child_prompt']:
                existing.append(dict(actual=r['actual_child_prompt'].split('\n\nActual task:\n',1)[1],
                    answer=r['answer'],label=label,source=str(label_path.relative_to(run.ROOT)),source_sha256=run.sha(label_path)))
    records, pending=[],[]
    indexed={r['probe_id']:r for r in inputs}
    for r in results:
        source=indexed[r['id']]
        actual=source['control_prompt'].split('\n\nActual task:\n',1)[1]
        matches=[x for x in existing if x['actual']==actual and x['answer']==r.get('answer')]
        item=dict(probe_id=r['id'],arm=r['arm'],cohort=source['cohort'],answer=r.get('answer'),
                  current_question=source['current_question'],source_sha256=source['source_sha256'])
        if matches:
            labels={(x['label']['factual_correct'],x['label']['addresses_actual_question'],x['label']['necessary_information_retained']) for x in matches}
            assert len(labels)==1, 'Conflicting historical review; resolve explicitly'
            selected=matches[0]; label=selected['label']
            item.update(factual_correct=label['factual_correct'],addresses_actual_question=label['addresses_actual_question'],
                necessary_information_retained=label['necessary_information_retained'],
                reason=label['reason_with_input_evidence'],review='Exact answer and actual-task context reuse of prior informed local review',
                prior_label_source=selected['source'],prior_label_sha256=selected['source_sha256'])
        else:
            item.update(factual_correct=None,addresses_actual_question=None,necessary_information_retained=None,review='needs local review')
            pending.append(dict(**item,history=source['history'],original_question=source['original_question'],reader_packet=source['reader_packet']))
        records.append(item)
    run.jsonl(run.OUT/'ORDINARY_REVIEW.draft.local.jsonl',records)
    run.jsonl(run.OUT/'ORDINARY_REVIEW_PENDING.local.jsonl',pending)
    print(json.dumps(dict(records=len(records),exact_context_and_answer_reused=len(records)-len(pending),new_review=len(pending)),ensure_ascii=True))
    for r in pending:
        print(json.dumps({k:r[k] for k in ('probe_id','arm','current_question','answer','history')},ensure_ascii=True))

if __name__=='__main__':
    main()
