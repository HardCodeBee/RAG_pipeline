"""Record completed actions without presenting submission as completed training."""
from datetime import datetime,timezone
import json
from pathlib import Path
from reaudit import HERE,ROOT,OUT,PREVIOUS,SOURCE,write,sha

def main():
    status=json.loads((OUT/'SCHEDULER_STATUS.json').read_text())
    submission=json.loads((OUT/'JOB_SUBMISSION.json').read_text())
    job=submission['job_id'];package='remote_v003' if submission.get('revision')=='v003_exact_v12' else 'remote'
    records=[
        {'id':'D001','change':'Full second query/target-only review of3500; preserve v001, revise36, quarantine5, author/review5 reserves, export immutable v0023500.','purpose':'Fix relation/qualifier/coverage defects without pretending all textual changes are semantic errors.','evidence':'reaudit/DECISIONS.progress.jsonl','sha256':sha(OUT/'reaudit/DECISIONS.progress.jsonl')},
        {'id':'D002','change':'Freeze separate dev150 and current-Codex references; freeze train-only diagnostic150.','purpose':'Observe validation loss/semantic generalization and select checkpoint without test300.','evidence':'data_v002/VALIDATION.json','sha256':sha(OUT/'data_v002/VALIDATION.json')},
        {'id':'E001','change':'Finish original4B/no-adapter/v12 independent300 generation and F/S/C/U review.','purpose':'Establish untouched same-question before/after comparison; no tuning from these cases.','evidence':'baseline300/SUMMARY.json','sha256':sha(OUT/'baseline300/SUMMARY.json')},
        {'id':'T001','change':'New adapter/optimizer on original Qwen3-4B; use approved LoRA32/64/.05, lr1e-4, fixed330 schedule, min2/max3 epochs.','purpose':'Use independent dev evidence for plateau/overfit control instead of inferring convergence from training loss.','evidence':package+'/config.json','sha256':sha(HERE/package/'config.json')},
        {'id':'T002','change':'New3500 sampler uses109 batches32 plus final12; actual-target-token gradient normalization.','purpose':'Avoid replaying20 or dropping12 when3500 is not divisible by32; old2368 had no such tail.','evidence':'CPU_NUMERICAL_CHECKS.json','sha256':sha(OUT/'CPU_NUMERICAL_CHECKS.json')},
        {'id':'V001','change':'Before submission, align array-format statistics to frozen baseline; explicit references remain a separate diagnostic/S failure.','purpose':'Prevent changing the comparison metric by counting references as JSON format errors.','evidence':'remote/code/contract.py','sha256':sha(HERE/'remote/code/contract.py')},
        {'id':'V002','change':'Check native supervision masks for3800 records, synthetic tail gradients, and real AdamW/scheduler CPU serialization at55->56.','purpose':'Verify mathematical/serialization invariants before actual4B GPU smoke.','evidence':'LOCAL_CONTRACT_CHECKS.json','sha256':sha(OUT/'LOCAL_CONTRACT_CHECKS.json')},
        {'id':'S001','change':'Submit original bundle153977, then hold/cancel it before any GPU work after exact-prompt audit found a stripped final newline. Submit corrected v003 job'+job+'.','purpose':'Execute approved training with byte-exact v12, without old adapter or smoke weights; preserve both versions.','evidence':'JOB_SUBMISSION.json','sha256':sha(OUT/'JOB_SUBMISSION.json')},
        {'id':'E002','change':'Prepare dev-only selection, selected-checkpoint eval300 generation, small return export and paired/cluster-sensitivity comparison scripts.','purpose':'Require real dev/test outputs before selecting or claiming accuracy gains.','execution':'not yet run; awaiting GPU training and current-Codex semantic review'}
    ]
    if (OUT/'EXACT_PROMPT_REVISION.json').exists():records.append({'id':'V003','change':'Preserve frozen v12 final newline; no label/hyperparameter change. All300 actual baseline input-token sequences match exactly;3800 loss encodings rechecked.','purpose':'Make actual training/after-evaluation system payload identical to the frozen before-model baseline.','evidence':'EXACT_PROMPT_REVISION.json','sha256':sha(OUT/'EXACT_PROMPT_REVISION.json')})
    write(OUT/'CHANGELOG.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),'changes':records,'latest_remote_snapshot':status,'api_calls':0})
    write(OUT/'CURRENT_RUN.json',{'job_id':job,'remote_package':submission['remote_package'],'local_package':str(HERE/package),'local_data':str(OUT/'data_v003_exact_v12'),'config_sha256':submission['config_sha256'],'superseded_job':'153977_cancelled_before_start','latest_status':status,'api_calls':0})
    (OUT/'STATUS.txt').write_text('Submitted Slurm'+job+'. Latest recorded remote check: '+status['checked_utc']+'\n'+status['queue'].replace('\\n','\n')+'\nSmoke verified: '+str(status['smoke_verified'])+'\nTrain metrics present: '+str(status['train_metrics_present'])+'\nFull second data review/dev150/baseline300 and exact-prompt token parity completed; actual GPU smoke/SFT/after-training300 pending.\n',encoding='utf-8')
    base=ROOT/'outputs/query_decomposition/multigrain_4b_frozen_eval300_v1_20261017'
    summary=json.loads((OUT/'baseline300/SUMMARY.json').read_text())
    write(base/'SEMANTIC_REVIEW_COMPLETED.json',{'generation_job':'153972','generation_completed':300,'semantic_review_complete':True,'review_summary':str((OUT/'baseline300/SUMMARY.json').relative_to(ROOT)),'review_cases':str((OUT/'baseline300/REVIEW.progress.jsonl').relative_to(ROOT)),'passed':157,'failed':139,'uncertain':4,'denominator':300,'pass_rate':157/300,'review_limit':summary['provenance'],'old_generation_return_unchanged':True})
    (base/'STATUS.txt').write_text('Original4B/v12 eval300 generation and current-Codex F/S/C/U review completed.\n300 format-valid, no generation error or truncation.\n157pass/139fail/4uncertain;157/300=52.33%.\nSee SEMANTIC_REVIEW_COMPLETED.json for full review paths and scope.\nNo adapter or new API was used.\n',encoding='utf-8')
    print(json.dumps({'recorded_changes':len(records),'gpu_training_started':status['train_metrics_present'],'job_id':job,'api_calls':0}))
if __name__=='__main__':main()
