"""Freeze the existing three-dataset formal cohort and query-only R02 handoff."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import unicodedata
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
E=ROOT/'experiments/query_decomposition'
OUT=ROOT/'outputs/query_decomposition/hotpot_formal_eval_v1'/HERE.name
REMOTE='/home/projects/u7830488/'+HERE.name
BASE='/home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package_v003'
CHECKPOINT='/home/projects/u7830488/multigrain_4b_coverage_retrain_v3_20261018/results/v001/training40/r02_expanded/checkpoint-final'
PYTHON='/home/projects/u7830488/demand_parser_route2_training_v1/20261009T191901Z/route2_question_decomposer/environment/venv/bin/python'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def write(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    value=text.encode('utf8') if isinstance(text,str) else text
    if p.exists():assert p.read_bytes()==value,('Frozen file differs',str(p))
    else:p.write_bytes(value)
def save(p,v):write(p,json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    package=ROOT/'src/p02_multigrain_timing_cachefix_v1'
    method=read(package/'FROZEN.json')
    for name,h in method['files'].items():assert sha(package/name)==h,name
    source=E/'comprehensive_eval_v1_20261015/inputs_frozen'
    cohorts={};queries=[];references=[];pins={}
    for ds in ['hotpotqa','2wiki','musique']:
        frozen=read(source/ds/'SAMPLES_FROZEN.json')
        for name in ['formal.questions.jsonl','formal.references.local.jsonl']:
            assert sha(source/ds/name)==frozen['files_sha256'][name]
            pins[(source/ds/name).relative_to(ROOT).as_posix()]=sha(source/ds/name)
        qq=rows(source/ds/'formal.questions.jsonl');rr=rows(source/ds/'formal.references.local.jsonl')
        assert len(qq)==len(rr)==500
        ref={r['qid']:r for r in rr};assert set(ref)=={q['qid'] for q in qq}
        for q in qq:
            cid=ds+':'+q['qid']
            queries.append({'id':cid,'qid':q['qid'],'dataset':ds,'question':q['question']})
            references.append({**ref[q['qid']],'id':cid,'dataset':ds})
        cohorts[ds]=frozen['splits']['formal']
    exclusions={}
    for name,p in [('development300',E/'p02_merge_cachefix_v1_20261021/runtime300.frozen.jsonl'),
        ('development90',E/'b0_strategy_round2_v1_20261020/fresh_validation/dev90.query_only.frozen.jsonl'),
        ('confirmation150',E/'b0_strategy_round2_v1_20261020/fresh_validation/sealed150.query_only.frozen.jsonl')]:
        vv=rows(p);ids={v['id'] for v in vv};texts={norm(v.get('question',v.get('query',''))) for v in vv}
        overlap=[q['id'] for q in queries if q['id'] in ids]
        text_overlap=[q['id'] for q in queries if norm(q['question']) in texts]
        exclusions[name]={'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),
                          'id_overlap':overlap,'normalized_question_overlap':text_overlap}
        assert not overlap,('Formal/development ID overlap',name)
    training=E/'multigrain_4b_coverage_retrain_v3_20261018/DATA3632.frozen.jsonl'
    train=rows(training);bytext={}
    for r in train:bytext.setdefault(norm(r['query']),[]).append(r['id'])
    duplicate=[{'id':q['id'],'dataset':q['dataset'],'question':q['question'],
                'training_ids':bytext[norm(q['question'])]} for q in queries if norm(q['question']) in bytext]
    omitted={v['id'] for v in duplicate}
    clean=[q['id'] for q in queries if q['id'] not in omitted]
    for info in exclusions.values():
        # Text duplicates, if present, are disclosed rather than silently hidden.
        omitted.update(info['normalized_question_overlap'])
    clean=[q['id'] for q in queries if q['id'] not in omitted]
    write(HERE/'questions1500.frozen.jsonl',''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in queries))
    write(HERE/'references1500.local.frozen.jsonl',''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in references))
    write(HERE/'queries1500.id_query.frozen.jsonl',''.join(json.dumps({'id':q['id'],'query':q['question']},ensure_ascii=False)+'\n' for q in queries))
    save(HERE/'EXPOSURE_AUDIT.frozen.json',{
        'formal_n':1500,'development_overlaps':exclusions,'training_source':training.relative_to(ROOT).as_posix(),
        'training_sha256':sha(training),'training_normalized_question_duplicates':duplicate,
        'training_duplicates_by_dataset':dict(Counter(v['dataset'] for v in duplicate)),
        'primary_clean_n':len(clean),'primary_clean_by_dataset':dict(Counter(i.split(':',1)[0] for i in clean)),
        'primary_clean_ids':clean,'excluded_from_clean_ids':sorted(omitted),
        'policy':'Keep all1500 planned method results and baseline overlap; primary clean analysis excludes question-text training/development overlap. Also report the complete fixed1500 cohort. No answer/F1-based exclusion.',
        'training_targets_or_standard_answers_uploaded':False})
    infer=E/'b0_strategy_round2_v1_20261020/fresh_validation/remote_infer_id_query_v001.py'
    write(HERE/'remote_infer.py',infer.read_bytes())
    write(HERE/'v12.txt',(package/'prompts/decomposition_v12.txt').read_bytes())
    slurm=(f'#!/usr/bin/env bash\n#SBATCH --job-name=p02-formal1500\n#SBATCH --account=db4ai\n'
        f'#SBATCH --partition=db4ai\n#SBATCH --qos=db4ai\n#SBATCH --gres=gpu:a6000:1\n'
        f'#SBATCH --cpus-per-task=4\n#SBATCH --mem=32G\n#SBATCH --time=03:00:00\n'
        f'#SBATCH --chdir={REMOTE}\n#SBATCH --output={REMOTE}/infer-%j.out\n#SBATCH --error={REMOTE}/infer-%j.err\n'
        'set -euo pipefail\nexport HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1\n'
        f'exec bash {BASE}/runtime/run_stage.slurm {PYTHON} {REMOTE}/remote_infer.py '
        f'--base-package {BASE} --checkpoint {CHECKPOINT} '
        '--checkpoint-sha 728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92 '
        f'--queries {REMOTE}/queries1500.id_query.frozen.jsonl --queries-sha {sha(HERE/"queries1500.id_query.frozen.jsonl")} '
        f'--prompt {REMOTE}/v12.txt --output {REMOTE}/results\n')
    write(HERE/'infer.slurm',slurm)
    names=['queries1500.id_query.frozen.jsonl','remote_infer.py','v12.txt','infer.slurm','remote_worker.py']
    save(HERE/'REMOTE_INPUTS.frozen.json',{'remote_directory':REMOTE,'count':1500,
        'scope':'Only id/query, fixed prompt/inference code; no answers, support labels, local F1, RAG execution or training.',
        'files':{n:sha(HERE/n) for n in names},'checkpoint':CHECKPOINT,
        'checkpoint_manifest_sha256':'728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92',
        'model_decode':'Existing frozen R02/v12 explicit greedy compat_v002 wrapper, single query, seed1729, max1024; not a new model.',
        'source_infer_sha256':sha(infer)})
    archive=HERE/'p02_formal1500_remote.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for name in names+['REMOTE_INPUTS.frozen.json']:z.write(HERE/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read(n)==(HERE/n).read_bytes() for n in names+['REMOTE_INPUTS.frozen.json'])
    OUT.mkdir(parents=True,exist_ok=True)
    save(HERE/'INPUTS.frozen.json',{'method':'p02_multigrain_timing_cachefix_v1',
        'package':'src/p02_multigrain_timing_cachefix_v1','method_manifest_sha256':sha(package/'FROZEN.json'),
        'planned':1500,'dataset_counts':{ds:500 for ds in cohorts},'strata':cohorts,
        'input_pins':{n:sha(HERE/n) for n in ['questions1500.frozen.jsonl','references1500.local.frozen.jsonl','EXPOSURE_AUDIT.frozen.json','REMOTE_INPUTS.frozen.json']},
        'source_pins':pins,'remote_archive_sha256':sha(archive),
        'primary_metrics':'Native final Answer F1/EM per dataset, clean cohort and all fixed1500; API missingness separate from algorithm failures.',
        'method_settings_unchanged':['R02/v12','dual interleave Top10','legacy complete child answers','goal preanswer','p02 role headers','final4shot/P1','max_tokens256/128/64','BGE512 exact original index arithmetic'],
        'do_not_apply_baseline_v2_changes_to_frozen_p02':True,
        'authorization':'User explicitly requested formal evaluation of p02_multigrain_timing_cachefix_v1; resumes this method only after earlier our-method pause. Other method/control/ablation pauses remain.',
        'budget':{'global_cap_cny':300,'dispatch_ceiling_cny':299.9,'local_soft_cap_cny':20},
        'new_API_calls_in_preparation':0})
    print(json.dumps({'prepared':1500,'clean_primary':len(clean),'training_question_duplicates':len(duplicate),
                      'remote_package':str(archive),'new_API_calls':0},ensure_ascii=False))


if __name__=='__main__':main()
