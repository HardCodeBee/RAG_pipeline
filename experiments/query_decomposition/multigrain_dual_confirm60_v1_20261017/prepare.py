"""Expand the original24 without changing method configuration or old outputs."""

from collections import Counter
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.screen_musique import rows, load, write, CAMPAIGN
from experiments.query_decomposition.multigrain_dual_retrieval_pilot_v1_20261017.run import sha

HERE = Path(__file__).resolve().parent
OUT = OUTPUT/HERE.name
ORIGINAL = ROOT/'experiments/query_decomposition/multigrain_dual_retrieval_pilot_v1_20261017'
ORIGINAL_OUT = OUTPUT/ORIGINAL.name
S02 = ROOT/'experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful'
POOL = OUTPUT/'multigrain_60_expansion_preparation_20261017/QUERY_ONLY_ELIGIBLE_POOL.json'

PLANS = {
 'hotpotqa': {
  '5ab94d9a554299743d22eae3': ['Which Nigerian rapper featured Deyemi Okanlawon in a music video?'],
  '5a7a33205542996a35c1712f': ['Which two American colonies had border skirmishes and raids from 1701 to 1765?'],
  '5a8b058055429971feec4612': ['Which shopping center is next to Holland Park?'],
  '5aba3bb855429955dce3ede7': ["To which scientist was Chien-Shiung Wu compared because of her expertise in experimental physics?"],
  '5a8661535542994775f60758': ['With which academic discipline is Hafiz Rashid Khan associated?'],
  '5ae54c085542992663a4f1c4': ['Which deer species is common in the forests of Mara and Mondrem?'],
  '5a8db42f5542994ba4e3dd11': ['What subject does Race to Nowhere document?', 'What subject does SlamNation document?'],
  '5a7f96a95542995d8a8ddead': ['Of which public university system is Universidad de Oriente a part?', 'Of which public university system is California State University, Dominguez Hills a part?'],
  '5a8e0cb85542995085b373b2': ['When was The Celluloid Closet released?', 'When was Station to Station released?'],
  '5ac41f1d5542991943173924': ['Where is Ohio State University located?', 'Where is California State University, Dominguez Hills located?'],
  '5adea0c5554299728e26c776': ['Is Helen Dunmore of West Indian descent?', 'Is M. P. Shiel of West Indian descent?'],
  '5adf2fa35542993344016c11': ['Of which bands has Jonny Craig been a member?', 'Of which bands has Pete Doherty been a member?'],
 },
 '2wiki': {
  'fbe7c55e085811ebbd5bac1f6bf848b6': ['Who directed Agent For H.A.R.M.?', 'When was the director of Agent For H.A.R.M. born?', 'Who directed The Hard Ride?', 'When was the director of The Hard Ride born?'],
  '3f894264086411ebbd5eac1f6bf848b6': ['Who directed A Day With The Devil?', 'When did the director of A Day With The Devil die?', 'Who directed 5 År – Som Vi Så Dem?', 'When did the director of 5 År – Som Vi Så Dem die?'],
  '8531b4a0089a11ebbd77ac1f6bf848b6': ['Who directed Sujon Sokhi?', 'When was the director of Sujon Sokhi born?', 'Who directed Two Thousand Maniacs!?', 'When was the director of Two Thousand Maniacs! born?'],
  '581696180bdb11eba7f7acde48001122': ['Who directed Manoranjan?'],
  '75990ae60bda11eba7f7acde48001122': ["Who is the father of John Annesley, 4Th Earl Of Anglesey?"],
  '512a07dc0bda11eba7f7acde48001122': ["Who is the father of Alfonso Xi Of Castile?"],
  'f96dfaba0bae11ebab90acde48001122': [],
  '039006b80bb011ebab90acde48001122': ["Who is the father of Charles James Irwin Grant, 6Th Baron De Longueuil?"],
  '1195a6f00bb011ebab90acde48001122': ["Who are the children of Victoria Sackville-West, Baroness Sackville?"],
  '757e321e094d11ebbdaeac1f6bf848b6': ['When was All The Women I Am released?', 'When was Fuse Of Love released?'],
  '3e33d262089311ebbd74ac1f6bf848b6': ['When was The Breach (Film) released?', 'When was The Twentieth Century Tramp; Or, Happy Hooligan And His Airship released?'],
  'c61541df084b11ebbd56ac1f6bf848b6': ["What was Alton W. Knappenberger's nationality?", "What was Heriberto Seda's nationality?"],
 },
 'musique': {
  '2hop__439614_349102': ["Who is Xiao Liwu's mother?"],
  '2hop__574419_270105': ['In which county is Van Hook Township located?'],
  '2hop__2150_78851': ['In which city does the Amateur Boxing Golden Gloves take place?'],
  '2hop__423972_540249': [],
  '2hop__143915_68489': ['Who performed Sings America?'],
  '3hop1__119402_102039_158266': ['Who discovered C/2014 Q2 (Lovejoy)?', 'Of which country is the discoverer of C/2014 Q2 (Lovejoy) a citizen?'],
  '3hop2__159979_30587_44003': ['Which ocean is adjacent to Miami?', 'In which ocean basin did Tropical Storm Allison end up?'],
  '3hop1__139862_2053_52946': ['Which network broadcast Highway to Heaven?', 'Which major broadcaster based in New York is the other one alongside ABC and the network that broadcast Highway to Heaven?'],
  '3hop1__280834_547811_80702': ['Who composed Bajazet?', 'Where was the composer of Bajazet born?'],
  '3hop1__458916_665330_51423': ['Who performed Spirit If..?', 'In which city did the performers of Spirit If.. form their band?'],
  '3hop1__376309_160713_77246': ['In which country is Garlat?', 'What is the majority religion in the area of British India that became India when the country containing Garlat was created?'],
  '4hop1__606685_508773_85832_745702': ['Where was Sergio Villanueva born?', 'On which continent is the birthplace of Sergio Villanueva?', 'Which navigator explored the east coast of the continent where Sergio Villanueva was born?'],
 },
}


def main():
    if (HERE/'protocol.json').exists():
        raise RuntimeError('Expansion already frozen')
    old=rows(ORIGINAL/'cases.frozen.jsonl')
    assert len(old)==24
    used={r['qid'] for p in (ROOT/'experiments/query_decomposition').glob('multigrain*/cases.frozen.jsonl') for r in rows(p)}
    pool=load(POOL)
    cases=[{**r,'evaluation_group':'original24'} for r in old]
    for ds,plans in PLANS.items():
        eligible={r['qid']:r for r in pool[ds]}
        assert len(plans)==12
        for qid,middle in plans.items():
            assert qid not in used and qid in eligible
            q=eligible[qid]
            steps=[{'id':i+1,'request':s} for i,s in enumerate(middle+[q['question']])]
            cases.append({'dataset':ds,'qid':qid,'question':q['question'],'type':q['type'],
                'evaluation_group':'new36','qdmr':{'steps':steps,'goal':len(steps)}})
    assert Counter(r['dataset'] for r in cases)=={'hotpotqa':20,'2wiki':20,'musique':20}
    assert len({r['qid'] for r in cases})==60
    (HERE/'cases.frozen.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases),encoding='utf-8')
    snapshot=[]
    for ds in ('hotpotqa','2wiki','musique'):
        source=CAMPAIGN/ds/'formal/b03_react_dense/k10_s5'
        calls={}
        for e in rows(source/'trace.jsonl'):
            if e.get('event')=='logical_api_call':calls.setdefault(e.get('qid'),{})[(e['stage'],e['request_id'])]=e['request_id']
        with sqlite3.connect((source/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
            for c in [r for r in cases if r['dataset']==ds]:
                raw=db.execute("SELECT value FROM data WHERE kind='prediction' AND key=?",(c['qid'],)).fetchone()
                r=json.loads(raw[0]);assert r['status'] in ('success','algorithm_failure') and r['question']==c['question']
                snapshot.append({'dataset':ds,'qid':c['qid'],'record':r,'logical_request_ids':list(calls.get(c['qid'],{}).values()),'source':str(source.relative_to(ROOT))})
    write(HERE/'react_predictions.frozen.json',snapshot)
    old_protocol=load(ORIGINAL/'protocol.json')
    files=[HERE/'cases.frozen.jsonl',HERE/'react_predictions.frozen.json',ORIGINAL/'protocol.json',ORIGINAL/'cases.frozen.jsonl',ORIGINAL/'run.py',S02/'prompts.py',S02/'examples.json',POOL]
    old_output_hashes={}
    for c in old:
        for label in ('single_original','dual_original_rewritten'):
            p=ORIGINAL_OUT/'cases'/c['dataset']/label/(c['qid']+'.json')
            old_output_hashes[str(p.relative_to(ROOT))]=sha(p)
            destination=OUT/'cases'/c['dataset']/label/p.name
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(p.read_bytes())
    protocol={**old_protocol,'status':'confirmed_original24_plus_new36','planned_questions':60,
        'sampling':'retain exact original24 outputs; add12 previously unused IDs per dataset in frozen hash order, stratified by query type, no answer-score filtering',
        'scope':'diagnostic expansion, fixed manually authored query-only reference plans; original24 was purposive, combined60 is not a fully random benchmark',
        'type_counts':{ds:dict(Counter(r['type'] for r in cases if r['dataset']==ds)) for ds in ('hotpotqa','2wiki','musique')},
        'frozen_files':{str(p.relative_to(ROOT)):sha(p) for p in files},'original_output_hashes':old_output_hashes,
        'controls':'same original child/final prompts, rewrite instruction with original query, full-sentence propagation, goal execution, original-first interleave10, originalTop10+helper, model and frozen positions. No v2-v5 optimization included.',
        'budget':'global200CNY, protected cutoff195CNY, new expansion5CNY subcap; old24 are reused without new provider calls'}
    write(HERE/'protocol.json',protocol)
    (HERE/'DECISIONS.txt').write_text(
        '2026-10-17 user explicitly chose original24 retained +36 new,20per dataset, to replicate original dual-retrieval gain.\n'
        'Use exact original v1 configuration; none of the later rewrite, child status, reread, final-example or auxiliary-authority changes enters this expansion.\n'
        'Old48 final outputs copied byte-for-byte; preserve their complete API receipts and count standalone logical costs. No re-generation of old24.\n'
        'New36 query IDs exclude every earlier multigrain attempt, including interrupted preparations. Hash seed dual-confirmation-60-v1; select first6bridge+6comparison Hotpot, first3per type2Wiki, first5two-hop+6three-hop+1four-hop Mu. No score screening or semantic exclusion after seeing answers.\n'
        'These quotas preserve original type proportions: Hotpot10/10,2Wiki5each,Mu8two-hop/10three-hop/2four-hop (integer rounding of original3/4/1). ReAct availability, not performance, is an eligibility filter; snapshot its60predictions before execution.\n'
        'New plans use only question text, retain self-contained requests and the original query as goal. When query already asks a direct relation, allow goal alone rather than inventing an extra dependency. This remains an assumed ideal-plan diagnostic, not independently human-verified oracle annotation or trained4B evaluation.\n'
        'Report old24, new36 and combined60 separately, plus each dataset; retain all missingness and per-query gains/ties/losses. Do not call the purposive+new mixture an unbiased full benchmark or use gold for runtime output choice.\n'
        'Execution caches only exact successful payloads with source receipts. Batch precompute fixed questions for efficiency; same corpus/model/search/merge and per-arm history. Missing optional later-version reread handling is irrelevant because original v1 has no reread.\n',encoding='utf-8')
    print(json.dumps({'questions':60,'copied_old_outputs':48,'new_questions':36,'types':protocol['type_counts']},ensure_ascii=False))


if __name__=='__main__':
    main()
