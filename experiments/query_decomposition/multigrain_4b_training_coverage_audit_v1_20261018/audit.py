"""Read-only coverage audit. Native answers/paragraphs are never projected or saved."""
from pathlib import Path
import collections,hashlib,itertools,json,re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'outputs/query_decomposition'/HERE.name
MASTER=ROOT/'outputs/query_decomposition/multigrain_4b_sft3500_v1_20261017/data_v003_exact_v12/MASTER_3500.jsonl'
NATIVE=ROOT/'data/multihop_comprehensive_v1_20261015/musique/native'

def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

CUES={
    'explicit_north_of':r'\bnorth of\b',
    'explicit_north_of_then_and':r'\bnorth of\b.*\band\b',
    'explicit_birthplace_or_born':r'\bbirthplace\b|\bborn\b|\bbirth (?:city|town|place)\b',
    'explicit_within':r'\bwithin\b',
    'explicit_same_state_province_country':r'\bsame (?:state|province|country)\b',
    'explicit_after_which':r'\bafter which\b',
    'explicit_before':r'\bbefore\b',
    'explicit_led_to':r'\bled to\b',
    'explicit_mouth':r'\bmouth\b',
    'explicit_watercourse':r'\bwatercourse\b',
    'explicit_family_and_title':r'(?=.*\bfamily\b)(?=.*\btitle\b)',
    'explicit_same_title':r'\bsame title\b',
    'explicit_territorial_qualifier_after_birth':r'\b(?:birthplace|born|birth city)\b.*\b(?:state|province)\b',
    'explicit_school':r'\bschool\b',
    'explicit_density':r'\bdensity\b',
    'explicit_possessive_belief':r"(?:'s|’s) belief\b",
}

def main():
    alltrain=list(rows(MASTER));mu=[r for r in alltrain if r['source']['dataset'].lower()=='musique'];assert len(mu)==509
    projections={};native_prefix={}
    for split in ['train','dev','test']:
        selected={};prefix=collections.Counter()
        for r in rows(NATIVE/f'musique_ans_v1.0_{split}.jsonl'):
            p={'id':r['id'],'query':r['question']}
            selected[r['id']]=p;prefix[r['id'].split('__',1)[0]]+=1
        projections[split]=selected;native_prefix[split]=prefix
    matched=[]
    for r in mu:
        source=r['source'];p=projections[source['original_split']][source['id']]
        assert p['query']==r['query'],r['id']
        matched.append({**p,'training_id':r['id'],'source_split':source['original_split'],'target':r['target']})
    sp=collections.Counter(r['id'].split('__',1)[0] for r in matched)
    cues={}
    for name,pattern in CUES.items():
        rx=re.compile(pattern,re.I)
        cues[name]={key:sum(bool(rx.search(r['query'])) for r in rr) for key,rr in [('our509',matched),('all3500',alltrain)]+[(s,list(p.values())) for s,p in projections.items()]}
    primitive_counts=collections.Counter(x for r in matched for x in r['id'].split('__',1)[1].split('_'))
    prefix_primitive={}
    family_concentration={}
    for pref in sp:
        subset=[r for r in matched if r['id'].startswith(pref+'__')]
        pp=collections.Counter(x for r in subset for x in r['id'].split('__',1)[1].split('_'))
        prefix_primitive[pref]={'records':sp[pref],'unique_primitives':len(pp),'primitive_occurrences':sum(pp.values()),'max_primitive_occurrences':max(pp.values())}
        if pref.startswith('4hop'):
            triples=collections.Counter(t for r in subset for t in itertools.combinations(sorted(r['id'].split('__',1)[1].split('_')),3))
            top=triples.most_common(1)[0][0]
            native_subset=[r for r in projections['train'].values() if r['id'].startswith(pref+'__')]
            native_n=sum(set(top)<=set(r['id'].split('__',1)[1].split('_')) for r in native_subset)
            family_concentration[pref]={'top_shared_three_primitives':top,'our_family_records':triples[top],
                'our_prefix_records':len(subset),'native_train_family_records':native_n,'native_train_prefix_records':len(native_subset),
                'our_family_queries':[r['query'] for r in subset if set(top)<=set(r['id'].split('__',1)[1].split('_'))]}
    native_primitive_counts={s:len({x for r in p.values() for x in r['id'].split('__',1)[1].split('_')}) for s,p in projections.items()}
    native_train_primitives={x for r in projections['train'].values() for x in r['id'].split('__',1)[1].split('_')}
    result={'master_sha256':hashlib.sha256(MASTER.read_bytes()).hexdigest(),'mu_n':len(mu),
        'source_splits':dict(collections.Counter(r['source_split'] for r in matched)),
        'source_cohorts':dict(collections.Counter(r['cohort'] for r in mu)),
        'prefix_counts':{'our509':dict(sp),**{k:dict(v) for k,v in native_prefix.items()}},
        'four_hop_family_concentration':family_concentration,
        'primitive_counts':{'unique':len(primitive_counts),'occurrences':sum(primitive_counts.values()),'native_unique_by_split':native_primitive_counts,
            'native_train_primitives_seen_in_our509':len(set(primitive_counts)&native_train_primitives),'top20':primitive_counts.most_common(20),'by_prefix':prefix_primitive},
        'weak_wording_cues':cues,
        'interpretation_limits':[
            'ID prefix coverage concerns native categories only, not semantic faithfulness or label quality. No exact DAG coverage inferred from question text.',
            'Keyword counts are exact lexical diagnostics, not semantic coverage estimates. Synonyms can cover a missing cue; the same cue may express a different relation.',
            'Primitive IDs represent underlying question reuse; covering every ID is neither required nor desirable for held-out generalization.',
            '11 dev-origin designated hard cases are training exposure, not independent evaluation; current held-outs exclude trained queries and their Mu primitive overlaps.',
            'Native answer/passages were not projected or saved; no model inference, training, prompt change, or locked review changes in this audit.']}
    write('COVERAGE_AUDIT.json',result)
    chosen=[]
    for r in matched:
        qs=r['query'];hits=[name for name,pattern in CUES.items() if re.search(pattern,qs,re.I)]
        if 'explicit_family_and_title' in hits or 'explicit_mouth' in hits or 'explicit_density' in hits or ('explicit_after_which' in hits):
            chosen.append({'id':r['training_id'],'query':qs,'target':r['target'],'weak_cues':hits})
    write('TARGETED_TRAIN_EXAMPLES.json',chosen)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
