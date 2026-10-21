"""Verify only this approved plan, unchanged frozen sources and offline checks."""
import sys
from pathlib import Path
if __package__ in {None,''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
import sqlite3
import unittest
from datetime import datetime,timezone
from experiments.query_decomposition.unified_rag_eval_v1_20261021.prepare_plan import ROOT,HERE,read,write,sha


def main():
    plan=read(HERE/'PLAN.json')
    assert plan['primary_method']['id']=='p02_multigrain_timing_cachefix_v1'
    assert plan['reference_route']['historical_or_unified_rerun']=='historical full-flow scores only; no rerun'
    assert plan['execution_approval']['approved'] is False and (HERE/'STOP_REQUESTED').exists()
    assert plan['metrics']['no_LLM_call_count_metric'] is True
    assert plan['planned_core_method_query_results']==15000
    assert all(sha(ROOT/p)==h for p,h in read(HERE/'PROTECTED_FILES_BEFORE.json').items())
    archive=read(ROOT/'archives/rag_evaluation_history_20261021/INDEX.json')
    assert archive['status']=='archived_verified' and len(archive['operations'])==50
    for op in archive['operations']:
        source,target=Path(op['source']),Path(op['target'])
        assert source.resolve()==target.resolve() and target.stat().st_ino==op['before']['directory_file_id']
    with sqlite3.connect((ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/budget.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        submissions=db.execute('SELECT count(*) FROM charges').fetchone()[0]
    assert submissions==read(ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1/pause_all_20261021/PAUSE_VERIFIED.json')['submissions']
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'experiments.query_decomposition.unified_rag_eval_v1_20261021.test_contracts',
        'experiments.query_decomposition.unified_rag_eval_v1_20261021.test_execution_policy'])
    test=unittest.TextTestRunner(verbosity=1).run(suite)
    value={'utc':datetime.now(timezone.utc).isoformat(),'plan_revision':plan['revision'],'status':'paused_pending_user_execution_approval','passed':test.wasSuccessful(),
           'offline_tests':test.testsRun,'protected_frozen_files_unchanged':len(read(HERE/'PROTECTED_FILES_BEFORE.json')),
           'physical_archive_directories':len(archive['operations']),'API_submissions_unchanged':True,
           'new_API_requests':0,'new_GPU_jobs':0,'whole_new_evaluation_runner_ready':False,
           'limitation':'Offline policy helpers, shared callbacks and frozen timing IDs/remote target profile are prepared; actual remote allocation/resources, migration and complete guarded paid runner remain unverified or unimplemented. Explicit resume is still required.'}
    write(HERE/'VERIFICATION.json',value)
    print(json.dumps(value,ensure_ascii=False,indent=2))
    return 0 if value['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
