"""Keep the supplied final template's empty intermediate heading verbatim."""
from contextlib import closing
from pathlib import Path
from threading import Lock
import json
import shutil
import sqlite3
import time

from . import run, prompts, analyze


def main():
    run.check()
    original = run.OUT
    source = original/'rate_limit_recovery001'
    assert run.read(source/'REPORT.json')['metrics']['overall']['complete'] == 84
    held = []
    for ds in run.common.DATASETS:
        trace = run.rows(source/'candidate'/ds/'trace.jsonl')
        with closing(run.ro(source/'candidate'/ds/'checkpoint.sqlite3')) as db:
            for rid,stage,status,raw in db.execute("SELECT id,stage,status,result FROM requests WHERE status!='success'"):
                value=json.loads(raw or '{}')
                assert status == 'unknown' and not value.get('proved_not_submitted')
                event=next(t for t in reversed(trace) if t.get('request_id')==rid)
                held.append({'id':event['id'],'dataset':ds,'request_id':rid,'stage':stage,
                             'failure_class':value.get('failure_class'),'reconciliation':value.get('reconciliation')})
    assert len(held) == 1
    target = original/'exact_template_recovery002'
    if target.exists():
        raise FileExistsError('Keep previous exact-template results')
    pipeline = run.module()
    def exact_final(question,nodes,root,readout,examples):
        block = pipeline.helper(nodes)
        if len(nodes) == 1:
            marker = 'Auxiliary question-answer pairs\n\n'
            assert block.startswith(marker)
            block = marker + 'Intermediate candidate results\n\n\n\n' + block[len(marker):]
        context = block + '\n\nRetrieved passages:\n' + pipeline.passages(root)
        prompt = prompts.final(question,context,examples,readout)
        values = {'ORIGINAL_QUESTION':question,
            'INTERMEDIATE_QA_BLOCK':prompts.prior_block(nodes[:-1]) if nodes[:-1] else '',
            'GOAL_STEP_ID':nodes[-1]['step_id'],'GOAL_ANSWER':nodes[-1]['answer'],
            'ORIGINAL_QUERY_RETRIEVED_PASSAGES':pipeline.passages(root)}
        assert prompt == prompts.FINAL.format_map(values)
        return prompt,context,block
    pipeline.render_final = exact_final
    target.mkdir()
    for name in ['control90.jsonl','OFFLINE_CHECK.json','seed.sqlite3']:
        shutil.copyfile(source/name,target/name)
    repaired = []
    for ds in run.common.DATASETS:
        previous, destination = source/'candidate'/ds,target/'candidate'/ds
        destination.mkdir(parents=True)
        with closing(run.ro(previous/'checkpoint.sqlite3')) as old, closing(sqlite3.connect(destination/'checkpoint.sqlite3')) as db:
            old.backup(db)
            for cid,raw in db.execute("SELECT key,value FROM data WHERE kind='prediction'").fetchall():
                result=json.loads(raw)
                if result['supporting_questions'] or result['status'] not in {'success','success_truncated'}:
                    continue
                assert result['final_context'].startswith('Auxiliary question-answer pairs\n\nCandidate conclusion')
                expected,context,block=exact_final(result['question'],result['nodes'],result['original_retrieval'],'',[])
                repaired.append({'id':cid,'old_final_prompt_sha256':run.hashlib.sha256(result['final_prompt'].encode()).hexdigest(),
                    'new_final_prompt_sha256':run.hashlib.sha256(expected.encode()).hexdigest(),
                    'change':'Keep the empty Intermediate candidate results heading exactly as supplied; all QA and passages unchanged.'})
                db.execute("DELETE FROM data WHERE kind='prediction' AND key=?",(cid,))
            db.commit()
        shutil.copyfile(previous/'trace.jsonl',destination/'trace.jsonl')
    assert len(repaired) == 5
    record={'utc':run.utc(),'scope':'Correct the adapter omission of an empty heading; no prompt wording or algorithm change.',
        'affected_final_calls':repaired,'successful_upstream_requests_retained':True,
        'original_results_preserved':True,'failed_truncated_goal_unchanged':True,
        'selection_uses_only_empty_supporting_questions_not_scores':True,
        'held_unknown_requests_not_resent':held,
        'main_evaluation_remains_paused':True,'runner_sha256':run.sha(Path(__file__))}
    run.save(target/'RENDERING_REPAIR.json',record)
    with (run.PLAN/'DECISIONS.jsonl').open('a',encoding='utf8') as handle:
        handle.write(json.dumps({'id':run.HERE.name.upper()+'_EXACT_EMPTY_GROUP_REPAIR',**record},ensure_ascii=False)+'\n')
    mutex,last=Lock(),[0.0]
    previous_request=run.common.LimitedLLM.request_once
    def paced(self,payload):
        with mutex:
            pause=2.0-(time.monotonic()-last[0])
            if pause>0:
                time.sleep(pause)
            last[0]=time.monotonic()
            return previous_request(self,payload)
    run.common.LimitedLLM.request_once=paced
    original_rows=run.rows
    def pending_rows(path):
        values=original_rows(path)
        if Path(path).resolve()==(run.COHORT/'runtime90.frozen.jsonl').resolve():
            return [v for v in values if v['id'] not in {h['id'] for h in held}]
        return values
    run.rows=pending_rows
    run.OUT=target
    run.emit({'exact_template_repair_finals':len(repaired),'held_unknown':len(held),'other_completed_cases_preserved':79})
    result=run.run()
    if result['status']=='completed':
        analyze.main()
    run.save(target/'PENDING_REQUEST.json',{'held_unknown_requests':held,'automatic_reissue':False,
        'completed':result['metrics']['overall']['complete'],'planned':90,
        'all_other_authorized_work_completed':result['metrics']['overall']['complete']==89})
    run.save(original/'FINAL_DELIVERY.json',{'status':result['status'],'report':str((target/'REPORT.json').relative_to(run.ROOT)),
        'pending_request':str((target/'PENDING_REQUEST.json').relative_to(run.ROOT)),
        'all_three_supplied_templates_preserved':True,'main_evaluation_remains_paused':True,'candidate_not_activated':True})
    run.emit({'status':result['status'],'completed':result['metrics']['overall']['complete'],'held_unknown':len(held)})


if __name__ == '__main__':
    main()
