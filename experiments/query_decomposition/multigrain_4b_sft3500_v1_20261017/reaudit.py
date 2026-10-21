"""Second-pass query/target-only review. Never auto-accepts undisplayed rows."""
import argparse,hashlib,json,re,unicodedata
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREVIOUS=HERE.parent/'multigrain_4b_prompt_train3500_v1_20261017'
SOURCE=ROOT/'outputs/query_decomposition'/PREVIOUS.name/'training'
OUT=ROOT/'outputs/query_decomposition'/HERE.name
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split()).strip(' ?!.？！')
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_rows(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x),encoding='utf-8')
def audit():
    original=rows(SOURCE/'MASTER_3500.jsonl');decisions={}
    for p in sorted((HERE/'second_review').glob('batch_*.json')):
        d=json.loads(p.read_text(encoding='utf-8'));lo,hi=d['reviewed_master_lines'];assert not set(range(lo,hi+1))&set(decisions)
        assert all(lo<=int(k)<=hi for k in d.get('revisions',{}))
        assert all(lo<=int(k)<=hi for k in d.get('quarantine',{}))
        for line in range(lo,hi+1):
            r=original[line-1];change=d.get('revisions',{}).get(str(line));reason=d.get('quarantine',{}).get(str(line))
            decisions[line]={'master_line':line,'index':r['index'],'id':r['id'],'query':r['query'],'previous_target':r['target'],'target':change['target'] if change else r['target'],'status':'quarantined' if reason else 'accepted','changed':bool(change),'reason':reason or (change['reason'] if change else 'Second inspection found no definite F/S/C/U defect; original ambiguity and useful complementary granularities allowed.'),'review_batch':p.name,'batch_sha256':sha(p),'review_method':'Current Codex second query/target-only pass; no prior reasons supplied; same reviewer, not independent human gold.'}
    write_rows(OUT/'reaudit/DECISIONS.progress.jsonl',sorted(decisions.values(),key=lambda r:r['master_line']))
    report={'source_records':3500,'reviewed':len(decisions),'remaining':3500-len(decisions),'changed':sum(r['changed'] for r in decisions.values()),'quarantined':sum(r['status']=='quarantined' for r in decisions.values()),'source_sha256':sha(SOURCE/'MASTER_3500.jsonl'),'full_second_review_completed':len(decisions)==3500,'training_started':False,'api_calls':0}
    write(OUT/'reaudit/PROGRESS.json',report);return report
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int);ap.add_argument('--end',type=int);ap.add_argument('--audit',action='store_true');args=ap.parse_args()
    if args.audit:print(json.dumps(audit(),indent=2));return
    for line,r in enumerate(rows(SOURCE/'MASTER_3500.jsonl'),1):
        if args.start<=line<=args.end:print(f'{line} [{r["index"]}] '+r['query']+' => '+json.dumps(r['target'],ensure_ascii=False))
if __name__=='__main__':main()
