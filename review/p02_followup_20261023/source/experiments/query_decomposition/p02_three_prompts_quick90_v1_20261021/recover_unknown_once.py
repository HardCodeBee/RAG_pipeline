"""One explicitly user-approved reissue; all original unknown evidence survives."""
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
    original=run.OUT
    source=original/'exact_template_recovery002'
    proposal=run.read(source/'UNKNOWN_RETRY_PROPOSAL.json')
    approval=run.read(original/'UNKNOWN_RETRY_APPROVED.json')
    approved=set(approval['approved_request_ids'])
    assert approved=={v['request_id'] for v in proposal['held_requests']}
    assert approval['maximum_reissues_per_ambiguous_request']==1
    assert run.read(source/'REPORT.json')['metrics']['overall']['complete']==89
    target=original/'approved_unknown_recovery003'
    if target.exists():
        raise FileExistsError('Approved reissue may only be started once')
    target.mkdir()
    for name in ['control90.jsonl','OFFLINE_CHECK.json','seed.sqlite3']:
        shutil.copyfile(source/name,target/name)
    retired=[]
    for ds in run.common.DATASETS:
        previous,destination=source/'candidate'/ds,target/'candidate'/ds
        destination.mkdir(parents=True)
        with closing(run.ro(previous/'checkpoint.sqlite3')) as old,closing(sqlite3.connect(destination/'checkpoint.sqlite3')) as db:
            old.backup(db)
            for rid,payload,status in db.execute("SELECT id,payload,status FROM requests WHERE status!='success'").fetchall():
                assert rid in approved and status=='unknown'
                expected=next(v for v in proposal['held_requests'] if v['request_id']==rid)
                assert run.hashlib.sha256(payload.encode()).hexdigest()==expected['payload_sha256']
                db.execute('DELETE FROM request_stages WHERE request_id=?',(rid,))
                db.execute('DELETE FROM requests WHERE id=?',(rid,))
                retired.append(rid)
            db.commit()
        shutil.copyfile(previous/'trace.jsonl',destination/'trace.jsonl')
    assert set(retired)==approved
    pipeline=run.module()
    def exact_final(question,nodes,root,readout,examples):
        block=pipeline.helper(nodes)
        if len(nodes)==1:
            marker='Auxiliary question-answer pairs\n\n'
            block=marker+'Intermediate candidate results\n\n\n\n'+block[len(marker):]
        context=block+'\n\nRetrieved passages:\n'+pipeline.passages(root)
        prompt=prompts.final(question,context,examples,readout)
        values={'ORIGINAL_QUESTION':question,'INTERMEDIATE_QA_BLOCK':prompts.prior_block(nodes[:-1]) if nodes[:-1] else '',
                'GOAL_STEP_ID':nodes[-1]['step_id'],'GOAL_ANSWER':nodes[-1]['answer'],
                'ORIGINAL_QUERY_RETRIEVED_PASSAGES':pipeline.passages(root)}
        assert prompt==prompts.FINAL.format_map(values)
        return prompt,context,block
    pipeline.render_final=exact_final
    mutex,last=Lock(),[0.0]
    previous_request=run.common.LimitedLLM.request_once
    def paced(self,payload):
        with mutex:
            delay=2.0-(time.monotonic()-last[0])
            if delay>0:time.sleep(delay)
            last[0]=time.monotonic()
            return previous_request(self,payload)
    run.common.LimitedLLM.request_once=paced
    run.save(target/'RECOVERY_AUTHORITY.json',{'utc':run.utc(),'approval':approval,'retired_only_in_new_copy':retired,
        'original_unknown_receipt_and_reserve_preserved':True,'other89_results_preserved':True,
        'runner_sha256':run.sha(Path(__file__)),'main_evaluation_remains_paused':True})
    run.OUT=target
    run.emit({'approved_unknown_reissues':1,'other_terminal_cases_preserved':89})
    result=run.run()
    if result['status']=='completed':
        analyze.main()
    run.save(original/'FINAL_DELIVERY.json',{'status':result['status'],'evaluated':result['metrics']['overall']['complete'],'planned':90,
        'report':str((target/('REPORT.md' if result['status']=='completed' else 'REPORT.json')).relative_to(run.ROOT)),
        'summary':str((target/'FINAL_SUMMARY.json').relative_to(run.ROOT)),
        'all_three_supplied_templates_preserved':True,'main_evaluation_remains_paused':True,'candidate_not_activated':True})


if __name__=='__main__':
    main()
