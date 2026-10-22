"""Informed local adjudication; no external evaluator or model generation."""
from collections import Counter
import json
from . import run

REVIEWS = {
    '923b15ba0bb011ebab90acde48001122':(True,True,False,
        'All ten current passages were inspected: Araripe Junior rank1 has only occupation and lifespan; other ranks concern unrelated people. Abstention is reasonable under the agreed rule. His father is missing from this answer, whereas the other arm supplies the previously verified father. Correctness is not the same as useful information.'),
    'a6fd9c120baf11ebab90acde48001122':(False,False,False,
        'History and reader ranks1/2 establish William -> Robert I -> Richard II. Rank7 establishes Richard is William\'s son. The output attaches Richard II as grandfather of that son, shifting one generation despite naming the locally requested entity. Full-statement rule makes both outputs wrong.'),
    '5a893fc855429951533612f0':(True,True,False,
        'All ten passages were inspected. Rank2 ends after identifying the Cretan Hound as a hunting dog from Crete; other passages concern other breeds, packs or generic hounds. The other breed\'s preceding answer cannot decide this breed\'s pack behavior. Reasonable abstention, with no useful yes/no result established.'),
    '5ab4e48655429942dd415f9c':(True,True,True,
        'Rank1 names Slieve Commedagh and Slieve Donard as the river\'s sources; ranks2/8 give767m and850m respectively. Both arms return the same true850m height for Donard. D additionally calls it the tallest source mountain, which the two heights support. B removes that qualifier; no factual or requested-height change.'),
    '3hop1__106042_64399_53006':(True,True,True,
        'Both outputs interpret the malformed city question as asking Christianity\'s origin city and name Jerusalem in Judea. Prior saved fact verification in ordinary_fewshot/root_blind_review/batch002/SUPPLEMENTAL_FACT_VERIFICATION.local.json quotes the Jerusalem Patriarchate historical account establishing the first church in Jerusalem. This is reuse of existing verification, not a new web lookup. The qualification distinguishes city from region.'),
    '4hop3__129456_29339_508306_70744':(True,True,True,
        'Reader rank2, Fort Hill (Clemson, South Carolina), explicitly locates Fort Hill on the Clemson University campus in Clemson. The current question and prior Fort Hill=South Carolina identify that property, not Fort Hill High School in Maryland. Clemson University is correct despite the unrelated earlier Kansas answer.'),
    '02f574780baf11ebab90acde48001122':(True,True,True,
        'History and rank1 identify Liu E\'s spouse as Liu Cong (Han Zhao); rank2 explicitly calls Liu Yuan his father. Liu Yin is Liu E\'s own father. D preserves the spouse/father composition; B\'s inherited identical-text review correctly flags the wrong parent binding.'),
}

def main():
    records=run.rows(run.OUT/'ORDINARY_REVIEW.draft.local.jsonl')
    assert len(records)==76
    for r in records:
        if r['review']!='needs local review':continue
        matches=[values for key,values in REVIEWS.items() if key in r['probe_id']]
        assert len(matches)==1
        f,a,n,reason=matches[0]
        r.update(factual_correct=f,addresses_actual_question=a,necessary_information_retained=n,
                 reason=reason,review='New informed local full-statement review; method labels visible; not independent human gold')
    for r in records:
        f,a=r['factual_correct'],r['addresses_actual_question']
        r['stage_correct']=False if f is False or a is False else True if f is True and a is True else None
    summaries={}
    for cohort in ('all','stress8','fewshot30'):
        summaries[cohort]={}
        for arm in ('D','B'):
            selected=[r for r in records if r['arm']==arm and (cohort=='all' or r['cohort']==cohort)]
            summaries[cohort][arm]=dict(n=len(selected),correct=sum(r['stage_correct'] is True for r in selected),
                wrong=sum(r['stage_correct'] is False for r in selected),pending=sum(r['stage_correct'] is None for r in selected),
                necessary_information_retained=dict(Counter(str(r['necessary_information_retained']) for r in selected)))
    ids=sorted({r['probe_id'] for r in records});index={(r['arm'],r['probe_id']):r for r in records}
    changes=[];counts=Counter();info_counts=Counter()
    for ident in ids:
        d,b=index[('D',ident)],index[('B',ident)]
        if d['stage_correct'] is not None and b['stage_correct'] is not None:
            change=int(b['stage_correct'])-int(d['stage_correct']);counts['improved' if change>0 else 'harmed' if change<0 else 'tied']+=1
        for_info=[r['necessary_information_retained'] for r in (d,b)]
        if all(x is not None for x in for_info):
            change=int(for_info[1])-int(for_info[0]);info_counts['improved' if change>0 else 'harmed' if change<0 else 'tied']+=1
        if d['answer']!=b['answer']:
            changes.append(dict(probe_id=ident,D=d,B=b))
    summary=dict(nodes=38,unique_queries=35,counts=summaries,paired_stage_correct=dict(counts),
        paired_necessary_information=dict(info_counts),different_text_pairs=len(changes),
        provenance='65 exact-answer/actual-context prior local reviews reused,11 newly reviewed outputs. Prior reviews and current reviewer are informed, not independent human labels.',
        pending='Two arm outputs of the same historical unresolved factual item remain pending; excluded from binary paired denominators.',
        scope='Ordinary fixed-reader component only; no end-to-end F1 inference')
    run.jsonl(run.OUT/'ORDINARY_REVIEW.local.jsonl',records)
    run.jsonl(run.OUT/'ORDINARY_REVIEW_CHANGES.local.jsonl',changes)
    run.save(run.OUT/'ORDINARY_REVIEW_SUMMARY.json',summary)
    print(json.dumps(summary,ensure_ascii=True))
    for r in changes:
        print(json.dumps(dict(id=r['probe_id'],question=r['D']['current_question'],
            D=r['D']['answer'],B=r['B']['answer'],correct=[r[a]['stage_correct'] for a in ('D','B')],
            info=[r[a]['necessary_information_retained'] for a in ('D','B')]),ensure_ascii=True))

if __name__=='__main__':main()
