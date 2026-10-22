"""Recover only explicitly returned HTTP429 failures; retain original receipts."""
from contextlib import closing
from pathlib import Path
from threading import Lock
import json
import shutil
import sqlite3
import time

from . import run


def main():
    run.check()
    original = run.OUT
    target = original / 'rate_limit_recovery001'
    if target.exists():
        raise FileExistsError('Preserve the existing recovery; no repeated reissue')
    rejected = []
    for ds in run.common.DATASETS:
        with closing(run.ro(original/'candidate'/ds/'checkpoint.sqlite3')) as db:
            for rid,stage,status,raw in db.execute("SELECT id,stage,status,result FROM requests WHERE status!='success'"):
                result = json.loads(raw or '{}')
                if status != 'failed' or result.get('http_status') != 429 or result.get('reconciliation') != 'HTTP response received':
                    raise RuntimeError('Only confirmed HTTP429 rejection may enter this recovery')
                rejected.append({'dataset':ds,'request_id':rid,'stage':stage,'http_status':429})
    assert len(rejected) == 2
    target.mkdir()
    for name in ['control90.jsonl','OFFLINE_CHECK.json','seed.sqlite3']:
        shutil.copyfile(original/name,target/name)
    for ds in run.common.DATASETS:
        source_dir, destination = original/'candidate'/ds,target/'candidate'/ds
        destination.mkdir(parents=True)
        with closing(run.ro(source_dir/'checkpoint.sqlite3')) as source, closing(sqlite3.connect(destination/'checkpoint.sqlite3')) as copied:
            source.backup(copied)
            for entry in [e for e in rejected if e['dataset']==ds]:
                rid = entry['request_id']
                copied.execute('DELETE FROM request_stages WHERE request_id=?',(rid,))
                copied.execute('DELETE FROM requests WHERE id=?',(rid,))
            copied.commit()
        shutil.copyfile(source_dir/'trace.jsonl',destination/'trace.jsonl')
    record = {'utc':run.utc(),'authority':'User requested completing the joint three-prompt quick90; operational recovery of confirmed rate-limit rejections only.',
        'rejected_requests':rejected,'original_receipts_and_reserves_preserved':True,
        'successful_payloads_preserved':True,'unknown_requests_reissued':False,
        'maximum_additional_attempts_for_these_rejections':1,'SDK_POST_retries':0,
        'process_API_concurrency':1,'minimum_new_request_start_interval_seconds':2.0,
        'prompts_cohort_generation_payloads_unchanged':True,'main_evaluation_remains_paused':True,
        'runner_sha256':run.sha(Path(__file__))}
    run.save(target/'RECOVERY.json',record)
    with (run.PLAN/'DECISIONS.jsonl').open('a',encoding='utf8') as handle:
        handle.write(json.dumps({'id':run.HERE.name.upper()+'_HTTP429_RECOVERY001',**record},ensure_ascii=False)+'\n')
    # The shared generation layer is changed only in this process. Unchanged
    # successful payloads remain cached, including complete and partial cases.
    mutex, last_start = Lock(), [0.0]
    previous = run.common.LimitedLLM.request_once
    def paced(self,payload):
        with mutex:
            delay = 2.0-(time.monotonic()-last_start[0])
            if delay > 0:
                time.sleep(delay)
            last_start[0] = time.monotonic()
            return previous(self,payload)
    run.common.LimitedLLM.request_once = paced
    run.OUT = target
    run.emit({'recovery':'HTTP429_only','preserved_terminal_cases':76,'remaining':14,'request_concurrency':1})
    result = run.run()
    run.emit({'status':result['status'],'overall':result['metrics']['overall'],'new_estimated_CNY':result['new_estimated_CNY']})


if __name__ == '__main__':
    main()
