"""Small read-only result snapshot; never packages model weight files."""
from pathlib import Path
import argparse,datetime,json,subprocess,zipfile

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--job',required=True);ap.add_argument('--number',type=int,required=True);a=ap.parse_args()
    result=a.root/'results';states={};trials={}
    for p in result.rglob('STATE.json'):
        try:states[p.relative_to(a.root).as_posix()]=json.loads(p.read_text())
        except (OSError,ValueError):pass
    for name in ['r01_old_data','r02_expanded']:
        p=result/'v001/training40'/name/'metrics.jsonl'
        if p.exists():
            mm=[json.loads(s) for s in p.read_text().splitlines() if s.strip()];trials[name]={'updates':len(mm),'last':mm[-1] if mm else None}
            if trials[name]['last']:trials[name]['last']={k:v for k,v in trials[name]['last'].items() if k!='ids'}
    status={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue':subprocess.check_output(['squeue','-j',a.job,'-o','%i %T %M %N'],text=True),
        'states':states,'trials':trials,'error_tail':(a.root/f'train-{a.job}.err').read_text()[-1200:] if (a.root/f'train-{a.job}.err').exists() else ''}
    (a.root/'STATUS_LATEST.json').write_text(json.dumps(status,indent=2));path=a.root/f'snapshot_{a.number:03d}.zip';assert not path.exists()
    skipped=[]
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in result.rglob('*'):
            if not p.is_file() or p.suffix not in ['.json','.jsonl','.txt'] or p.stat().st_size>2_000_000:continue
            data=p.read_bytes()
            if p.suffix=='.json':
                try:json.loads(data)
                except ValueError:skipped.append(str(p));continue
            z.writestr(p.relative_to(a.root).as_posix(),data)
        z.write(a.root/'STATUS_LATEST.json','STATUS_LATEST.json')
    print(json.dumps({'snapshot':str(path),'queue':status['queue'],'states':states,'trials':trials,'transient_skips':skipped}))

if __name__=='__main__':main()
