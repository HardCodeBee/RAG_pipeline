"""Package local qualitative notes and their exact inputs, without inference."""
from collections import Counter
import json
from . import run as r


def main():
    r.check()
    assert r.read(r.HERE/'ANALYSIS.frozen.json')['analysis_sha256']==r.sha(r.HERE/'analyze.py')
    inputs={x['id']:x for x in r.rows(r.HERE/'INPUTS60.frozen.jsonl')}
    refs={x['id']:x for x in r.rows(r.HERE/'REFERENCES60.local.frozen.jsonl')}
    results=r.rows(r.OUT/'DETAILS.local.jsonl');index={(x['arm'],x['id']):x for x in results}
    primary=r.rows(r.OUT/'PRIMARY_CASE_REVIEW.local.jsonl')
    changed={(a+'->'+b,i) for a,b in [('G0','T'),('GF','GF_SPLIT')] for i in inputs
             if index[(a,i)]['prediction']!=index[(b,i)]['prediction']}
    assert {(x['pair'],x['id']) for x in primary}==changed
    descriptions={
        '2wiki:4f81c5260bdc11eba7f7acde48001122':('typographic_only','Same spouse, curly versus ASCII apostrophe; no entity correction.'),
        'musique:3hop1__346812_15840_36002':('partial_fact_recovery','GF recovers correct16-bit architecture already present as16 - bit in BOTH saved goal and final packets. G0 did not use it. GF also retains developer-support material and final remains below full native score; this is improved use of available evidence, not new retrieval.'),
        'hotpotqa:5a7e14505542995ed0d16685':('alternative_solution_despite_abstaining_subanswers','Ordinary QA abstains on exact longitude. Reader passages identify NewYork versus southwesternOhio; GF uses geographical knowledge to compare them and obtains native answer. This does not require fabricated exact coordinates.'),
        'musique:2hop__85993_512508':('candidate_answer_selection_discards_reference_project','Both TubeAlloys and ManhattanProject are supported. GF paragraph retains both but privileges TubeAlloys, then final dropsManhattan. Native mismatch is not proof TubeAlloys participation is false.'),
        'hotpotqa:5abb23035542992ccd8e7f22':('qualifier_addition','Addingnear keeps the same location but lowers native overlap; split arm reverses this surface change.'),
        'hotpotqa:5a8c7595554299585d9e36b6':('unrequested_ranking_and_answer_selection','Reader andGF body containChiefOfProtocol. GF invents a highest-ranking selection criterion absent from question and choosesambassadorship; final loses reference role. Ambassadorship itself is supported; causal diagnosis concerns unrequested ranking and information selection.'),
        'musique:2hop__132909_33564':('era_taxonomy_change','Both know1958. GF suppliesColdWarEra while reference specifies1950s-1970s; this changes answer framing, not the company or founding year.'),
        'musique:2hop__85463_66122':('date_coarsening_and_native_tokenization','GF turnslateMay1951 intomid-1951, losing native token overlap with10July1951. Neither establishes native exact date; do not count old partial score as a correct event date.'),
        'musique:4hop1__94201_642284_131926_13165':('unsupported_target_to_abstention','GF drops unsupportedPacificOcean and declines; native answer not recovered. G0 wrong treaty earned wordTreaty overlap.'),
        'musique:2hop__25396_212301':('partial_relation_correction_with_residual_generation_error','GF correctsWilliamYatesPeel from son tobrother, but still misreadsSidneyPeel as son rather than grandson. The savedSidney paragraph identifiesArthur as youngest son of PMRobertPeel. FinalWilliamPeel is a supported son, but differs from nativeArthur; the score gain does not certify the entire synthesis.'),
        '2wiki:15b9210e0baf11ebab90acde48001122':('title_omission','Same paternal grandfather; removingCount yields exact native match. No changed relative or relation depth.'),
    }
    secondary=[]
    for x in r.rows(r.OUT/'CHANGED.local.jsonl'):
        if x['comparison']!='G0->GF':continue
        category,note=descriptions[x['id']]
        secondary.append(dict(pair='G0->GF',id=x['id'],category=category,note=note))
    assert len(secondary)==11
    r.jsonl(r.OUT/'SECONDARY_CASE_REVIEW.local.jsonl',secondary)
    notes=primary+secondary;caseids={x['id'] for x in notes}
    native_path=r.ROOT/'data/multihop_comprehensive_v1_20261015/musique/native/musique_ans_v1.0_dev.jsonl'
    native={}
    for line in native_path.open(encoding='utf8'):
        x=json.loads(line)
        if 'musique:'+x['id'] in caseids:
            native['musique:'+x['id']]={k:v for k,v in x.items() if k!='paragraphs'}
            native['musique:'+x['id']]['supporting_paragraphs']=[p for p in x['paragraphs'] if p.get('is_supporting')]
    bundle=[dict(id=i,input=inputs[i],reference=refs[i],variants={a:index[(a,i)] for a in r.templates.ARMS},
        notes=[x for x in notes if x['id']==i],native_diagnostic_only=native.get(i)) for i in sorted(caseids)]
    r.jsonl(r.OUT/'CASE_EVIDENCE.local.jsonl',bundle)
    summary=dict(primary_changed_final_pairs=len(primary),primary_cases=len({x['id'] for x in primary}),
        primary_categories=dict(Counter(x['category'] for x in primary)),secondary_scored_changed_pairs=len(secondary),
        evidence_cases=len(bundle),review='Informed local Codex inspection; not blinded or independent human labels; no external judge',
        native_gold_usage='Offline post-generation explanation only; never in model input',
        native_source=str(native_path.relative_to(r.ROOT)),native_sha256=r.sha(native_path),
        all_primary_changed_predictions_reviewed=True,all_secondary_F1_changes_reviewed=True)
    r.save(r.OUT/'QUALITATIVE_REVIEW.json',summary)
    lines=['# 本轮差异案例与完整goal输出','',
        '覆盖两个主要对照所有最终文本变化，以及G0→GF全部F1变化。以下是知晓分数后的本地证据复核，不是独立盲评；原生分数不因解释而修改。完整检索包及数据集诊断证据见CASE_EVIDENCE.local.jsonl。','']
    for item in bundle:
        lines += ['## '+item['id'],'',item['input']['question'],'',
                  '参考答案：'+str(item['reference']['answer']),'']
        for n in item['notes']:lines += [n['pair']+' / '+n['category']+': '+n['note'],'']
        for a,v in item['variants'].items():
            lines += ['### '+a,'','Goal完整输出：','',v['goal_text'],'','Final：'+v['prediction'],
                      '',f"F1={v['metrics']['f1']:.6f}; EM={v['metrics']['em']:.6f}",'']
    text='\n'.join(lines)+'\n'
    (r.OUT/'CASE_NOTES.md').write_text(text,encoding='utf8');(r.REVIEW/'CASE_NOTES.md').write_text(text,encoding='utf8')
    print(json.dumps(summary,ensure_ascii=True))


if __name__=='__main__':main()
