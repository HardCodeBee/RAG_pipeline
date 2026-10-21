"""Remote lifecycle for query decomposition only, isolated from existing jobs."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);text=json.dumps(v,ensure_ascii=False,indent=2)+'\n'
    if p.exists():assert p.read_text()==text,'Immutable remote receipt differs'
    else:p.write_text(text)
def command(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=40)
    if r.returncode:raise RuntimeError('Remote command failed: '+r.stderr[-1200:])
    return r.stdout.strip()


def progress():
    receipt=HERE/'SUBMISSION.json';job=read(receipt)['job_id'] if receipt.exists() else None
    queue=command(['squeue','-h','-j',str(job),'-o','%T|%M|%R']) if job else ''
    account=command(['sacct','-X','-n','-P','-j',str(job),'--format=JobIDRaw,State,Elapsed,ExitCode']) if job and not queue else ''
    files=sorted((HERE/'results').glob('[0-9]*.json'))
    state=read(HERE/'results/STATE.json') if (HERE/'results/STATE.json').exists() else None
    identity=read(HERE/'results/IDENTITY.json') if (HERE/'results/IDENTITY.json').exists() else None
    return {'job_id':job,'queue':queue,'accounting':account,'completed':len(files),'planned':1500,
            'state':state,'identity':identity,'files':[p.name for p in files]}


def stage():
    meta=read(HERE/'REMOTE_INPUTS.frozen.json');assert str(HERE)==meta['remote_directory']
    for name,h in meta['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
    q=[json.loads(s) for s in (HERE/'queries1500.id_query.frozen.jsonl').read_text().splitlines() if s.strip()]
    assert len(q)==1500 and all(set(v)=={'id','query'} for v in q)
    with (HERE/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (HERE/'SUBMISSION.json').exists():return {'submission':'existing_receipt_no_resubmit',**progress()}
        # Reconcile an uncertain submission by its exact isolated working dir.
        for job in command(['squeue','-h','-u','u7830488','-o','%i']).splitlines():
            desc=command(['scontrol','show','job','-o',job.strip()])
            fields=dict(v.split('=',1) for v in desc.split() if '=' in v)
            if fields.get('WorkDir')==str(HERE):
                save(HERE/'SUBMISSION.json',{'job_id':job.strip(),'scope':'query_decomposition_only'})
                return {'submission':'reconciled_existing_job_no_resubmit',**progress()}
        if (HERE/'SUBMISSION_PENDING.json').exists():raise RuntimeError('Uncertain previous submission without queue match; preserve and reconcile, never blindly resubmit')
        save(HERE/'SUBMISSION_PENDING.json',{'scope':'query_decomposition_only','planned':1500})
        job=command(['sbatch','--parsable',str(HERE/'infer.slurm')]).split(';')[0]
        assert job.isdigit()
        save(HERE/'SUBMISSION.json',{'job_id':job,'scope':'query_decomposition_only'})
        return {'submission':'submitted_once',**progress()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','progress']);args=parser.parse_args()
    print(json.dumps(stage() if args.action=='stage' else progress(),ensure_ascii=False))
