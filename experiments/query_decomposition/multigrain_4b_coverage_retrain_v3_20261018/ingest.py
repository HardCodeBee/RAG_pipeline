"""Import immutable predictions and separate mutable progress metadata."""
import argparse,json,shutil,zipfile
from prepare import OUT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshot');a=ap.parse_args();zpath=OUT/a.snapshot;dest=OUT/'returned'/zpath.stem;dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():assert dest.resolve() in (dest/info.filename).resolve().parents
        z.extractall(dest)
    sources=[(dest/'results/v001/parent_dev60',OUT/'raw_dev/p00_parent_t02')]
    for name in ['r01_old_data','r02_expanded']:sources.append((dest/'results/v001/training40'/name/'predictions',OUT/'raw_dev'/name))
    sources.append((dest/'results/v002/training20/r03_expanded20/predictions',OUT/'raw_dev/r03_expanded20'))
    sources.append((dest/'results/historical16/r02_expanded',OUT/'raw_historical/r02_expanded'))
    for src,target in sources:
        target.mkdir(parents=True,exist_ok=True)
        if not src.exists():continue
        for p in src.glob('[0-9][0-9][0-9].json'):
            q=target/p.name
            if q.exists():assert q.read_bytes()==p.read_bytes()
            else:shutil.copyfile(p,q)
    # Confirmation imports use the same paired parent name and method directories.
    conf=dest/'results/confirmation'
    if conf.exists():
        for src in conf.iterdir():
            if not src.is_dir():continue
            target=OUT/'raw_confirmation'/src.name;target.mkdir(parents=True,exist_ok=True)
            for p in src.glob('[0-9][0-9][0-9].json'):
                q=target/p.name
                if q.exists():assert q.read_bytes()==p.read_bytes()
                else:shutil.copyfile(p,q)
    if (dest/'STATUS_LATEST.json').exists():shutil.copyfile(dest/'STATUS_LATEST.json',OUT/'STATUS_LATEST.json')
    print(json.dumps({p.name:len(list(p.glob('[0-9][0-9][0-9].json'))) for p in (OUT/'raw_dev').iterdir() if p.is_dir()}))

if __name__=='__main__':main()
