"""After optimization is closed, audit the original sealed test300 once."""
import argparse,json,collections
from reaudit import HERE,OUT,rows,write,write_rows,sha

PRED=rows(OUT/'final300/predictions.jsonl')
BASE={r['id']:r for r in rows(OUT/'baseline300/REVIEW.progress.jsonl')}

def audit():
    d={}
    for path in sorted((HERE/'final300_review').glob('batch_*.json')):
        batch=json.loads(path.read_text(encoding='utf-8'));lo,hi=batch['reviewed_indices']
        for i in range(lo,hi+1):
            assert i not in d
            pred=PRED[i-1];b=BASE[pred['id']];assert pred['index']==i and pred['query']==b['query']
            same=pred['parsed']==json.loads(b['raw_output']);f=batch.get('exceptions',{}).get(str(i))
            if same:
                assert f is None
                verdict,dims,reason=b['verdict'],b['failed_dimensions'],'Identical question strings: preserve original frozen verdict. '+b['reason']
            else:verdict,dims,reason=(f['verdict'],f['dimensions'],f['reason']) if f else ('pass',[],'Current Codex inspected changed output under frozen F/S/C/U; original descriptions and useful prerequisites retained.')
            if verdict=='pass':assert pred['format_valid'] and not pred['truncated'] and not pred['generation_error'] and not pred['explicit_references']
            d[i]={'index':i,'id':pred['id'],'dataset':pred['dataset'],'query':pred['query'],'raw_output':pred['raw_output'],'verdict':verdict,'failed_dimensions':dims,'reason':reason,'review_mode':'inherit_identical_questions' if same else 'inspected_changed_output','review_batch':path.name,'review_batch_sha256':sha(path),'reviewer':batch['reviewer']}
    a=sorted(d.values(),key=lambda r:r['index']);write_rows(OUT/'final300/REVIEW.progress.jsonl',a)
    s={'reviewed':len(a),'planned':300,'verdicts':dict(collections.Counter(x['verdict'] for x in a)),'by_dataset':{ds:dict(collections.Counter(x['verdict'] for x in a if x['dataset']==ds)) for ds in ['hotpotqa','2wiki','musique']},'scope':'Original model-versus-SFT test300; no optimization or checkpoint selection after reading these outputs.'}
    write(OUT/'final300/REVIEW_SUMMARY.json',s);print(json.dumps(s,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int);ap.add_argument('--end',type=int);ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if a.audit:audit()
    else:
        for p in PRED[a.start-1:a.end]:
            b=BASE[p['id']]
            print(str(p['index'])+' SAME '+b['verdict'] if p['parsed']==json.loads(b['raw_output']) else f"{p['index']} [{p['dataset']}] {p['query']} => {p['raw_output']}")
