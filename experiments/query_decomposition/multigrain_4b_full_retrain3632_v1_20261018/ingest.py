"""Import full-retrain snapshots with immutable predictions and checkpoints."""
import argparse,json,shutil,zipfile
from prepare import HERE,OUT

ap=argparse.ArgumentParser();ap.add_argument('snapshot');a=ap.parse_args()
zpath=OUT/a.snapshot;dest=OUT/'returned'/zpath.stem;dest.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(zpath) as z:
    for info in z.infolist():assert dest.resolve() in (dest/info.filename).resolve().parents
    z.extractall(dest)
for src in (dest/'results/train/dev60_predictions').glob('checkpoint-*'):
    target=OUT/'raw_dev'/src.name;target.mkdir(parents=True,exist_ok=True)
    for p in src.glob('*.json'):
        q=target/p.name
        if q.exists():assert q.read_bytes()==p.read_bytes(),str(q)
        else:shutil.copyfile(p,q)
if (dest/'results/confirmation').exists():
    target=OUT/'raw_confirmation';target.mkdir(parents=True,exist_ok=True)
    for p in (dest/'results/confirmation').glob('*.json'):
        q=target/p.name
        if q.exists() and p.name not in ['STATE.json']:assert q.read_bytes()==p.read_bytes(),str(q)
        else:shutil.copyfile(p,q)
status=json.loads((dest/'STATUS.json').read_text())
(OUT/'STATUS_LATEST.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':{k:v for k,v in status.items() if k not in ['out_tail','err_tail']},
  'dev_files':{p.name:len(list(p.glob('[0-9][0-9][0-9].json'))) for p in (OUT/'raw_dev').glob('*')}}))
