"""Read-only CPU status/snapshot; never load model weights or use CUDA."""
import argparse,json,subprocess,time,zipfile
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--job',required=True);ap.add_argument('--snapshot');a=ap.parse_args()
status={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'job_id':a.job}
status['queue']=subprocess.run(['squeue','-j',a.job,'-h','-o','%i %T %M %R'],capture_output=True,text=True).stdout.strip()
status['accounting']=subprocess.run(['sacct','-j',a.job,'-n','-X','-o','JobID,State,Elapsed,ExitCode','--parsable2'],capture_output=True,text=True).stdout.strip()
for mode in ['smoke','train']:
    p=a.root/'results'/mode
    if (p/'STATE.json').exists():status[mode]=json.loads((p/'STATE.json').read_text())
    if (p/'metrics.jsonl').exists():
        mm=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines() if s.strip()]
        status[mode+'_last_update']={k:v for k,v in mm[-1].items() if k!='ids'} if mm else None
    status[mode+'_dev_counts']={d.name:len(list(d.glob('[0-9][0-9][0-9].json'))) for d in (p/'dev60_predictions').glob('*') if d.is_dir()}
for suffix in ['out','err']:
    f=a.root/f'train-{a.job}.{suffix}'
    if f.exists():status[suffix+'_tail']=f.read_text(errors='replace')[-3500:]
if a.snapshot:
    dest=a.root/(a.snapshot+'.zip');assert not dest.exists()
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('STATUS.json',json.dumps(status,indent=2)+'\n')
        for p in (a.root/'results').rglob('*'):
            if p.is_file() and p.suffix in ['.json','.jsonl','.txt'] and p.stat().st_size<=2000000:
                z.write(p,str(p.relative_to(a.root)))
        for n in ['SUBMISSION.json','PLAN_FROZEN.json','FREEZE_TIME.json']:
            z.write(a.root/n,n)
    status['snapshot']=str(dest)
print(json.dumps(status))
