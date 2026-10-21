"""Freeze a one-factor rewrite comparison on new cases and carried losses."""

import json
from pathlib import Path

from experiments.query_decomposition.multigrain_evidence_v4r1_20261017.prepare import ROOT, OUTPUT, OLD_OUT, V2, V3, S02, sha, rows, load, write
from . import prompts

HERE = Path(__file__).resolve().parent
OUT = OUTPUT/HERE.name
PRIOR = ROOT/'experiments/query_decomposition/multigrain_evidence_v4r1_20261017'
PRIOR_OUT = OUTPUT/PRIOR.name

PLANS = {
 'hotpotqa': {
  '5ae24abc5542994d89d5b3c3': ['Which company owns the thanKing Horse pub restaurant chain?'],
  '5ab8cd3d5542991b5579f009': ["In which rock band did Peter Gifford play from 1980 to 1987?"],
  '5a7f39d055429934daa2fd2a': ['Is The Million Dollar Duck a comedy film?', 'Is Don Quixote a comedy film?'],
  '5a8caf1d554299585d9e3720': ['Where was the rock group Dig formed?', 'Where was Thinking Fellers Union Local 282 formed?'],
 },
 '2wiki': {
  '5e735d77092011ebbdaeac1f6bf848b6': ['When was The Artist And The Model released?', 'When was Pájaro Loco released?'],
  'e2f26a20089111ebbd73ac1f6bf848b6': ['Who directed An Old, Old Tale?', 'When did the director of An Old, Old Tale die?', 'Who directed Reefer Madness?', 'When did the director of Reefer Madness die?'],
  '28bbe1600bdc11eba7f7acde48001122': ['Who is the keyboardist of Veslefrekk?'],
  '4a9c5bde0baf11ebab90acde48001122': ["Who is the father of Louis Viii, Landgrave Of Hesse-Darmstadt?"],
 },
 'musique': {
  '2hop__96177_62765': ['Who starred in Dark Hazard?'],
  '2hop__543853_124498': ['Who performed Labo M?'],
  '3hop2__49541_140875_51068': ['Which station aired High Feather?', 'Which program on the station that aired High Feather depicts the Queen of England in 1890?'],
  '4hop2__71753_623626_70784_61381': ['In what region is Israel located?', 'In which body of water is Kharg Island located?', 'Which region lies north of the region where Israel is located and the body of water containing Kharg Island?'],
 },
}


def main():
    if (HERE/'protocol.json').exists():
        raise RuntimeError('Already frozen')
    assert load(PRIOR_OUT/'RESULTS.json')['complete']
    used = {r['qid'] for p in (ROOT/'experiments/query_decomposition').glob('multigrain*/cases.frozen.jsonl') for r in rows(p)}
    cases = []
    for ds, plans in PLANS.items():
        base=ROOT/f'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen/{ds}'
        qs={r['qid']:r for r in rows(base/'formal.questions.jsonl')}
        refs={r['qid']:r for r in rows(base/'formal.references.local.jsonl')}
        for qid, middle in plans.items():
            assert qid not in used
            q=qs[qid]['question'];steps=[{'id':i+1,'request':s} for i,s in enumerate(middle+[q])]
            cases.append({'qid':qid,'dataset':ds,'question':q,'type':refs[qid]['type'],'evaluation_group':'fresh','qdmr':{'steps':steps,'goal':len(steps)}})
    carried=load(PRIOR_OUT/'REGRESSIONS_FOR_NEXT_ROUND.json')
    cases += carried
    prior_cases=rows(PRIOR/'cases.frozen.jsonl')
    for qid in ('bc7073a80baf11ebab90acde48001122','5a8eb63455429917b4a5bdb7'):
        if qid not in {r['qid'] for r in cases}:
            cases.append({**next(r for r in prior_cases if r['qid']==qid),'evaluation_group':'competitor_gap'})
    assert len(cases)==len({r['qid'] for r in cases})
    (HERE/'cases.frozen.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases),encoding='utf-8')
    (HERE/'child_examples.json').write_bytes((PRIOR/'child_examples.json').read_bytes())
    examples=load(V2/'final_examples.json')
    first=examples[0]['context']
    examples[0]['context']=first.split('\n\n[Chunk 2')[0]+'\n\nAuxiliary question-answer pairs'+first.split('Auxiliary question-answer pairs',1)[1]
    examples[0]['fixture_origin']='Same real Spanish Valley/Colorado River QA fixture. Omit only the Colorado River passage from final context to teach a fact supplied by subquery QA. Not a runtime model output.'
    assert 'source=Colorado River' not in examples[0]['context'] and '1450' in examples[0]['context']
    write(HERE/'final_examples.json',examples)
    files=[HERE/'cases.frozen.jsonl',HERE/'prompts.py',HERE/'child_examples.json',HERE/'final_examples.json',PRIOR/'prompts.py',V2/'prompts.py',V2/'final_examples.json',PRIOR_OUT/'CASE_RESULTS.jsonl']
    protocol={'scope':'12 fresh query IDs + all preceding dual-vs-single losses + separately reported ReAct-gap cases; manual query-only plans, diagnostic development',
        'conditions':{prefix+'_'+branch:{'rewrite':rewrite,'final':'v2' if prefix=='reuse' else 'aux_absence_v5'} for prefix in ('reuse','support') for branch,rewrite in (('new','partial_binding_v5'),('previous','v2'),('single','none'))},
        'model':'qwen-plus-2025-12-01','rewrite_tokens':256,'child_tokens':128,'final_tokens':64,'reuse_added_document_cap':5,
        'global_cap_cny':200,'round_subcap_cny':5,
        'frozen_files':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'passing_gate':'all planned complete; support_new mean F1 above support_single, zero individual losses; also report all isolated factors, previous dual and ReAct',
        'two_factors':'rewrite v2/partial-binding/none crossed with final v2/auxiliary-absence distinction. Each upstream branch feeds both final prompts with identical context; original path, child prompt, once-only reread, corpus, retriever, merge Top10 and placement fixed'}
    write(HERE/'protocol.json',protocol)
    (HERE/'REWRITE_TEMPLATE.txt').write_text(prompts.rewrite('{current_question}',[{'request':'{prior_question}','answer':'{prior_complete_sentence}'}]),encoding='utf-8')
    (HERE/'DECISIONS.txt').write_text(
        '2026-10-17 round4 frozen before generation\n'
        'Observed gap: v4 retains Anurutha father Inthasom, but exact-scope v2 rewriting leaves parents-brothers query unchanged. Existing-material reread cannot fetch missing siblings.\n'
        'Change only optional rewrite contract: allow known subset as an additional retrieval path; preserve requested relation, never invent missing members, never assert subset is the complete set. No verification duty added to rewriter.\n'
        'Keep v2 three shots, add one authored partial-set shot based on Clianthus red-flower fixture, with no regression/test names or answers. This is a teaching scenario, not an actual dataset query/model output.\n'
        'Three upstream arms: previous rewrite, partial-binding rewrite, no rewrite. All use frozen v4 child and once-only existing-material reread, unchanged original-first interleave Top10, and both final-prompt controls. No new queries beyond each fixed node original and optional rewrite; no ReAct loop.\n'
        'Before freezing, add a second independently controlled factor: final auxiliary-absence handling. v4 Nigel Noble goal correctly states the documentary-short award, but final returns Cannot be determined. This is a final-reader failure, not missing child retrieval.\n'
        'Final revision adds one sentence: auxiliary may supply facts from other retrievals; absence in shown original passages is not contradiction; related entity/different event facts do not conflict. Modify only first existing shot by omitting its Colorado River source paragraph while retaining its correct full-sentence auxiliary length. Other3 shots remain unchanged. No test entities or gold answers enter teaching examples.\n'
        'Full 3x2 design isolates rewrite-only, final-only and their combination. Child trajectories and final contexts identical across the two final prompts. No new model, new judge, training, or additional search loop.\n'
        'Execution efficiency: batch precompute fixed original node queries using the same frozen exact BGE search; all arms share these packets. Rewritten queries still computed from each arm history. This does not add query expressions or change merge policy.\n'
        'Sampling uses hash seed partial-bind-v5-round4, query-only screening and strata (Hotpot2bridge2comparison;2Wiki4types;Mu2twohop+1threehop+1fourhop). All12 fresh IDs exclude earlier rounds including interrupted attempts. Additional targeted old cases separate from fresh. No model-score-based output selection.\n',encoding='utf-8')
    OUT.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'cases':len(cases),'fresh':12,'regressions':len(carried),'conditions':6}))


if __name__=='__main__':
    main()
