"""Review changed question strings; preserve literal baseline verdicts when unchanged."""
import argparse, collections, json
from prepare import HERE, OUT, rows, sha, write, frozen

BASE = {r['index']:r for r in rows(OUT/'control/checkpoint-000165.review.frozen.jsonl')}
NAMES = ['c01_adjacent_bridge','c02_full_description']

def equivalent(pred, base):
    return pred.get('format_valid') and not pred.get('truncated') and not pred.get('generation_error') and not pred.get('explicit_references') and pred.get('parsed') == json.loads(base['raw_output'])

def display(name, start, end):
    for i in range(start,end+1):
        p=OUT/'raw'/name/f'{i:03d}.json'
        if not p.exists():print('MISSING',name,i);continue
        r=json.loads(p.read_text(encoding='utf-8'));b=BASE[i]
        assert r['id']==b['id'] and r['query']==b['query']
        print(f"{i} [{b['dataset']}] {b['query']}\nBASE({b['verdict']}) {b['raw_output']}")
        print('NEW SAME QUESTIONS' if equivalent(r,b) else 'NEW '+str(r['raw_output']))

def assemble():
    table=[]
    for name in NAMES:
        records={}
        for path in sorted((HERE/'reviews'/name).glob('batch_*.json')):
            batch=json.loads(path.read_text(encoding='utf-8'));lo,hi=batch['reviewed_indices']
            assert all(lo<=int(x)<=hi for x in batch.get('exceptions',{}))
            for i in range(lo,hi+1):
                assert i not in records
                p=OUT/'raw'/name/f'{i:03d}.json';r=json.loads(p.read_text(encoding='utf-8'));b=BASE[i]
                assert r['id']==b['id'] and r['query']==b['query']
                same=equivalent(r,b)
                if same:
                    assert str(i) not in batch.get('exceptions',{}),'Cannot change frozen verdict for identical question strings'
                    verdict,dimensions,reason=b['verdict'],b['failed_dimensions'],'Identical question strings to frozen baseline; preserve verdict: '+b['reason']
                else:
                    f=batch.get('exceptions',{}).get(str(i))
                    verdict=f['verdict'] if f else 'pass';dimensions=f['dimensions'] if f else [];reason=f['reason'] if f else 'Current Codex inspected changed questions under frozen F/S/C/U; no definite defect.'
                    if verdict=='pass':assert r['format_valid'] and not r['truncated'] and not r['generation_error'] and not r['explicit_references']
                records[i]={'index':i,'id':r['id'],'dataset':r['dataset'],'query':r['query'],'raw_output':r['raw_output'],'verdict':verdict,'failed_dimensions':dimensions,'reason':reason,'review_mode':'identical_question_strings_inherit_baseline' if same else 'inspected_changed_output','review_batch':path.name,'review_batch_sha256':sha(path)}
        a=sorted(records.values(),key=lambda r:r['index'])
        dest=OUT/'reviews'/(name+'.jsonl');dest.parent.mkdir(exist_ok=True,parents=True)
        dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in a),encoding='utf-8')
        x={'variant':name,'reviewed':len(a),'planned':150,'passed':sum(x['verdict']=='pass' for x in a),'by_dataset':{d:dict(collections.Counter(x['verdict'] for x in a if x['dataset']==d)) for d in ['hotpotqa','2wiki','musique']},'fixes':[x['index'] for x in a if x['verdict']=='pass' and BASE[x['index']]['verdict']!='pass'],'regressions':[x['index'] for x in a if x['verdict']!='pass' and BASE[x['index']]['verdict']=='pass'],'review_sha256':sha(dest)}
        table.append(x)
    write(OUT/'reviews/SUMMARY.progress.json',{'baseline':{'passed':128,'planned':150},'candidates':table})
    print(json.dumps(table,ensure_ascii=False,indent=2))
    if all(x['reviewed']==150 for x in table):
        eligible=[x for x in table if x['passed']>128]
        key=lambda x:(x['passed'],min(x['by_dataset'][d].get('pass',0)/50 for d in ['hotpotqa','2wiki','musique']),-len((HERE/'prompts'/(x['variant']+'.txt')).read_bytes()))
        winner=max(eligible,key=key)['variant'] if eligible else 'b00_v12'
        frozen(OUT/'WINNER_DEV_FROZEN.json',(json.dumps({'winner':winner,'baseline_passed':128,'denominator':150,'all_candidates':table,'selection_uses_formal300_or_confirmation90':False,'criterion':'Higher full-denominator pass count; then worst-dataset rate; then shorter prompt. Otherwise retain b00.','review_limit':'Current Codex query-only nonindependent F/S/C/U, not RAG F1.'},ensure_ascii=False,indent=2)+'\n').encode())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=NAMES);ap.add_argument('--start',type=int);ap.add_argument('--end',type=int);ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if a.audit:assemble()
    else:display(a.variant,a.start,a.end)
