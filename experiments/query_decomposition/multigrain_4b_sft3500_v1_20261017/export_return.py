"""Small completed-run return: predictions/logs/checkpoint identities, no weights."""
import argparse,hashlib,json,zipfile
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--archive',type=Path,required=True);a=ap.parse_args()
    import fcntl
    with (a.package/'results/train/.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root=a.package/'results';state=json.loads((root/'train/STATE.json').read_text());assert state['status'] in ['completed','early_stopped','interrupted']
        selected=[]
        for p in root.rglob('*'):
            if not p.is_file() or p.is_symlink():continue
            if p.suffix=='.json' or p.name in ['metrics.jsonl','validation.jsonl']:
                selected.append(p)
        files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
        manifest={'files':files,'status':state,'scope':'Predictions and execution evidence; adapter/training-state binaries remain on remote for verified selection/resume.','api_calls':0}
        with zipfile.ZipFile(a.archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in selected:z.write(p,p.relative_to(root).as_posix())
            z.writestr('RETURN_MANIFEST.json',json.dumps(manifest,indent=2))
        print(json.dumps({'archive':str(a.archive),'members':len(selected)+1,'bytes':a.archive.stat().st_size,'status':state},indent=2))
if __name__=='__main__':main()
