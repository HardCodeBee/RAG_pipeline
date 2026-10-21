"""Acquire official public native data and evaluators, with bounded recovery."""
import time
import zipfile
import requests
from .common import DATA, RUN, sha, save, log

ASSETS = {
 '2wiki/data_ids.zip':'https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1',
 'musique/musique_v1.0.zip':'https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t',
 '2wiki/official/2wikimultihop_evaluate_v1.1.py':'https://raw.githubusercontent.com/Alab-NII/2wikimultihop/main/2wikimultihop_evaluate_v1.1.py',
 'musique/official/evaluate_v1.0.py':'https://raw.githubusercontent.com/stonybrooknlp/musique/main/evaluate_v1.0.py',
 'musique/official/metrics/answer.py':'https://raw.githubusercontent.com/stonybrooknlp/musique/main/metrics/answer.py',
 'musique/official/metrics/support.py':'https://raw.githubusercontent.com/stonybrooknlp/musique/main/metrics/support.py',
 'musique/official/metrics/__init__.py':'https://raw.githubusercontent.com/stonybrooknlp/musique/main/metrics/__init__.py',
 'musique/official/metrics/metric.py':'https://raw.githubusercontent.com/stonybrooknlp/musique/main/metrics/metric.py',
 'musique/official/metrics/group_answer_sufficiency.py':'https://raw.githubusercontent.com/stonybrooknlp/musique/main/metrics/group_answer_sufficiency.py',
 'musique/official/metrics/group_support_sufficiency.py':'https://raw.githubusercontent.com/stonybrooknlp/musique/main/metrics/group_support_sufficiency.py',
}

def fetch(path, url):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.partial')
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30,60)) as r:
                r.raise_for_status()
                with partial.open('wb') as f:
                    for part in r.iter_content(1024*1024):
                        f.write(part)
            if path.suffix=='.zip' and not zipfile.is_zipfile(partial):
                raise ValueError('Response is not a ZIP archive: '+str(path))
            partial.replace(path)
            return
        except Exception:
            if attempt==2: raise
            time.sleep(5)

def extract(dataset, archive):
    allowed = {'2wiki':{'train.json','dev.json','test.json','id_aliases.json'},
               'musique':{'musique_'+kind+'_v1.0_'+split+'.jsonl' for kind in ['ans','full'] for split in ['train','dev','test']}}
    selected=[]
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            base=name.replace('\\','/').split('/')[-1]
            if base not in allowed[dataset]:continue
            dest=DATA/dataset/'native'/base
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():continue
            with z.open(name) as source, dest.with_suffix(dest.suffix+'.partial').open('wb') as target:
                while part:=source.read(1024*1024):target.write(part)
            dest.with_suffix(dest.suffix+'.partial').replace(dest)
            selected.append(base)
    if not allowed[dataset].issubset({p.name for p in (DATA/dataset/'native').iterdir()}):
        raise ValueError('Native archive lacks required split/alias files')
    return selected

def main():
    evidence=[]
    try:
        for relative,url in ASSETS.items():
            save(RUN/'data_progress.json',{'status':'downloading','asset':relative})
            fetch(DATA/relative,url)
            evidence.append({'path':relative,'url':url,'sha256':sha(DATA/relative),'bytes':(DATA/relative).stat().st_size})
            print(relative, (DATA/relative).stat().st_size,flush=True)
            if relative.endswith('.zip'):
                extract(relative.split('/')[0], DATA/relative)
        save(DATA/'SOURCES_FROZEN.json',{'assets':evidence, 'runtime_gold_access':False})
        save(RUN/'data_progress.json',{'status':'official_data_downloaded','assets':len(evidence)})
        log('official_datasets_downloaded',assets=evidence,api_calls=0)
    except BaseException as exc:
        save(RUN/'data_progress.json',{'status':'stopped','type':type(exc).__name__,'reason':str(exc),'completed_assets':evidence})
        raise

if __name__=='__main__':main()
