"""One-time fresh-query paired review; no prompt selection or editing from these cases."""
import argparse,collections,json,random
from prepare import HERE,OUT,rows,write,sha

def get(name,i):return json.loads((OUT/'raw_confirmation'/name/f'{i:03d}.json').read_text(encoding='utf-8'))

def show(lo,hi):
    meta=rows(HERE/'confirmation90.questions.frozen.jsonl')
    for i in range(lo,hi+1):
        b=get('b00_v12',i);c=get('c02_full_description',i)
        assert b['id']==c['id']==meta[i-1]['id'] and b['query']==c['query']==meta[i-1]['query']
        print(f"{i} [{b['dataset']}/{meta[i-1]['type']}] {b['query']}\nBASE {b['raw_output']}")
        print('C02 SAME' if b['parsed']==c['parsed'] else 'C02 '+str(c['raw_output']))

def audit():
    decisions={n:{} for n in ['b00_v12','c02_full_description']}
    for p in sorted((HERE/'confirmation_reviews').glob('batch_*.json')):
        batch=json.loads(p.read_text(encoding='utf-8'));lo,hi=batch['reviewed_indices']
        for i in range(lo,hi+1):
            b=get('b00_v12',i);c=get('c02_full_description',i)
            assert b['id']==c['id']
            base_finding=batch.get('baseline_exceptions',{}).get(str(i))
            for name,pred in [('b00_v12',b),('c02_full_description',c)]:
                assert i not in decisions[name]
                if name=='b00_v12' or b['parsed']==c['parsed']:f=base_finding
                else:f=batch.get('candidate_exceptions',{}).get(str(i))
                verdict=f['verdict'] if f else 'pass';dims=f['dimensions'] if f else [];reason=f['reason'] if f else 'Inspected full question set under frozen F/S/C/U; useful multiple granularities allowed.'
                if verdict=='pass':assert pred['format_valid'] and not pred['truncated'] and not pred['generation_error'] and not pred['explicit_references']
                decisions[name][i]={'index':i,'id':pred['id'],'dataset':pred['dataset'],'query':pred['query'],'raw_output':pred['raw_output'],'verdict':verdict,'failed_dimensions':dims,'reason':reason,'review_batch':p.name,'review_batch_sha256':sha(p),'reviewer':batch['reviewer']}
    summary={}
    for name,items in decisions.items():
        a=sorted(items.values(),key=lambda r:r['index']);path=OUT/'confirmation_reviews'/(name+'.jsonl');path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in a),encoding='utf-8')
        summary[name]={'reviewed':len(a),'planned':90,'passed':sum(x['verdict']=='pass' for x in a),'verdicts':dict(collections.Counter(x['verdict'] for x in a)),'by_dataset':{d:dict(collections.Counter(x['verdict'] for x in a if x['dataset']==d)) for d in ['hotpotqa','2wiki','musique']}}
    if all(s['reviewed']==90 for s in summary.values()):
        b,c=decisions['b00_v12'],decisions['c02_full_description']
        fixes=[i for i in b if b[i]['verdict']!='pass' and c[i]['verdict']=='pass'];reg=[i for i in b if b[i]['verdict']=='pass' and c[i]['verdict']!='pass']
        summary['paired']={'fixes':fixes,'regressions':reg,'delta':(len(fixes)-len(reg))/90}
        rng=random.Random(20261018);blocks={d:[int(c[i]['verdict']=='pass')-int(b[i]['verdict']=='pass') for i in b if b[i]['dataset']==d] for d in ['hotpotqa','2wiki','musique']}
        boot=sorted(sum(rng.choice(z) for z in blocks.values() for _ in range(30))/90 for _ in range(10000))
        summary['paired']['bootstrap95']=[boot[249],boot[9749]]
        summary['decision']='c02_confirmed_point_gain_candidate' if summary['c02_full_description']['passed']>summary['b00_v12']['passed'] else 'retain_b00_no_confirmed_gain'
        summary['scope']='One-time90, current Codex query-only nonindependent review. Development150 remains separate; no RAG F1 inference.'
    write(OUT/'confirmation_reviews/SUMMARY.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int);ap.add_argument('--end',type=int);ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if a.audit:audit()
    else:show(a.start,a.end)
