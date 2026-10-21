"""Freeze isolated edits and an untouched query-only confirmation cohort."""
from datetime import datetime, timezone
import hashlib
import json

from experiments.query_decomposition.component_chain_eval60_v1_20261019.prepare import (
    ROOT, OUTPUT, INPUTS, CAMPAIGN, HERE as PREVIOUS_E, OUT as PREVIOUS_O, read, rows, sha, save, normalize)
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = OUTPUT / HERE.name
RUN = OUT / 'run_v001'
SEED = 'component-opt-fresh18-20261019-v1'
QUOTAS = {'hotpotqa':{'bridge':4,'comparison':2},
          '2wiki':{'comparison':2,'inference':1,'compositional':1,'bridge_comparison':2},
          'musique':{'2-hop':2,'3-hop':2,'4-hop':2}}
READOUT_P1 = '''Final response requirements:
Return only the requested answer: a name, value, category or brief phrase.
Preserve the answer's complete identifying name and any qualifier required by the Original question.
Exclude facts describing that answer, such as relatives, locations, events or years, unless they are what the question asks for.
For a supported yes/no answer, return exactly Yes or No.
Do not restate the question, copy a full auxiliary sentence, or add explanation or citations.
Final short answer:'''


def log(action, **fields):
    with (HERE/'DECISIONS.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'action':action,**fields},ensure_ascii=False)+'\n')


def main():
    if (HERE/'PROTOCOL.frozen.json').exists():
        print('Frozen optimization protocol preserved.');return
    previous = read(PREVIOUS_E/'SELECTION.frozen.json')
    old_queries = rows(PREVIOUS_E/'queries60.query_only.frozen.jsonl')
    training_file = ROOT/'experiments/query_decomposition/multigrain_4b_coverage_retrain_v3_20261018/DATA3632.frozen.jsonl'
    excluded_texts = {normalize(r['query']) for r in rows(training_file)} | {normalize(r['query']) for r in old_queries}
    child = rows(ROOT/'experiments/query_decomposition/child_answer_accuracy_v1_20261017/round10/cases.inputs.jsonl')
    rewrites = rows(OUTPUT/'rewrite_short_answer_ablation_v1_20261017/inputs.frozen.jsonl')
    excluded_ids = {r['qid'] for r in child+rewrites+old_queries}
    excluded_ids.update(r.get('source_qid') for r in read(ROOT/'experiments/query_decomposition/child_answer_accuracy_v1_20261017/c08_examples.frozen.json'))
    chosen, sources = [], {}
    for ds, quota in QUOTAS.items():
        qfile, rfile = INPUTS/ds/'formal.questions.jsonl', INPUTS/ds/'formal.references.local.jsonl'
        types = {r['qid']:r['type'] for r in rows(rfile)}
        pool = [r for r in rows(qfile) if r['qid'] not in excluded_ids and normalize(r['question']) not in excluded_texts]
        for kind,n in quota.items():
            ordered = sorted((q for q in pool if types[q['qid']]==kind),
                             key=lambda q:hashlib.sha256((SEED+'|'+ds+'|'+q['qid']).encode()).hexdigest())
            assert len(ordered)>=n
            chosen += [{'id':ds+':'+q['qid'],'qid':q['qid'],'dataset':ds,'query':q['question'],'type':kind} for q in ordered[:n]]
        sources[str(qfile.relative_to(ROOT))]=sha(qfile)
        sources[str(rfile.relative_to(ROOT))]=sha(rfile)
    assert len(chosen)==18 and len({normalize(q['query']) for q in chosen})==18
    qpath=HERE/'confirmation18.query_only.frozen.jsonl'
    qpath.write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in chosen),encoding='utf-8')
    protocol = {'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'model':previous['model'],'resources':previous['resources'],'decomposer':previous['decomposer'],
        'development':{'n':60,'runtime':str((PREVIOUS_E/'runtime60.frozen.jsonl').relative_to(ROOT)),
                       'runtime_sha256':sha(PREVIOUS_E/'runtime60.frozen.jsonl'),
                       'source_scores_sha256':sha(PREVIOUS_O/'CASE_RESULTS.jsonl')},
        'confirmation':{'n':18,'seed':SEED,'quotas':QUOTAS,'query_sha256':sha(qpath),
                        'selection':'Hash/type before new generation, without score or baseline-availability selection.',
                        'zero_exact_overlap_with_development_training_and_component_parent_queries':True,
                        'scope':'New to this pipeline optimization; drawn from the existing formal question pools. Not an untouched decomposer benchmark.'},
        'phase_A':{'conditions':['c08_only','c02_only'],'global_dispatch_ceiling_cny':198.90,
                   'change':'Complete the two missing factorial arms; frozen pipeline, prompts and failure policy unchanged.'},
        'phase_B':{'conditions':['r1','p1','r1p1'],'global_dispatch_ceiling_cny':199.20,
                   'R1':'Only on known child-answer length termination: discard incomplete answer, keep completed QA and originalTop10, go directly to final original-query answer. Normal goal path unchanged. Provider failures remain missing.',
                   'P1':'Only replace final v3 readout. Preserve all preceding final instructions/examples/context and all upstream generation.',
                   'P1_readout':READOUT_P1,
                   'selection':'Compare each against frozen C08+c02 by paired native F1; require >=40 paired queries and positive mean delta for provisional selection. Report missingness and all planned bounds. Prefer fewer edits on exact tie; do not use confirmation scores to choose.'},
        'phase_C':{'conditions':['legacy','selected_before_confirmation'],'global_dispatch_ceiling_cny':199.95,
                   'change':'Freeze one candidate after development; run original B0 and selected candidate on18 fresh queries using the same actual R02 outputs. No candidate changes after confirmation starts.'},
        'generation':previous['generation'],'global_cap_cny':200,'local_cap_cny':1.85,
        'cache_policy':'Reuse only exact-payload successful receipts; known failed/unknown requests are not resubmitted. Reuse exact qid/query/k/resources retrieval packets. Every cache origin retained.',
        'metrics':'Native Answer F1/EM, known algorithm failures0, provider/budget missing not0, paired denominators and bounds. No gold in generation.',
        'source_files_sha256':sources,
        'component_protocol_sha256':previous['chain_protocol_sha256'],
        'no_new_training':True,'remote_scope':'Query decomposition only; query-only18 sent to frozen R02/v12.'}
    save(HERE/'PROTOCOL.frozen.json',protocol)
    save(OUT/'STATUS.json',{'status':'prepared','new_API_calls':0,'confirmation_frozen':18})
    log('freeze_before_new_inference',protocol_sha256=sha(HERE/'PROTOCOL.frozen.json'),
        reason='Separate component effects, child-length recovery and final-readout edit; reserve fresh confirmation and original200CNY cap.')
    print(json.dumps({'confirmation':18,'quotas':QUOTAS,'query_sha256':sha(qpath)},ensure_ascii=False))


if __name__=='__main__':main()
