"""User-requested cap expansion on the single truncated goal, isolated from90."""
from contextlib import ExitStack, closing
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit
import json
import sqlite3

from . import run,prompts

CASE_ID='musique:2hop__22431_16335'
GOAL_CAP=512


def main():
    run.check()
    original=run.OUT
    source=original/'approved_unknown_recovery003'
    assert run.read(source/'REPORT.json')['status']=='completed'
    out=original/'goal_cap512_single'
    if out.exists():
        raise FileExistsError('Preserve the single-query cap diagnostic')
    out.mkdir()
    case=next(c for c in run.rows(run.COHORT/'runtime90.frozen.jsonl') if c['id']==CASE_ID)
    case=run.common.baseline.safe_case(case)
    assert case['supporting_questions']==[]
    config=run.read(run.PACKAGE/'config.json')
    protocol={'utc':run.utc(),'authority':'User annotation on 输出用满了128-token上限: 扩展一下输出上限 完整回答这个问题',
        'case_id':CASE_ID,'change':'Only this query goal max_tokens128->512; preserve final64, prompts, retrieved passages and model.',
        'goal_max_tokens':GOAL_CAP,'final_max_tokens':64,'original90_results_preserved':True,
        'main_evaluation_remains_paused':True,'runner_sha256':run.sha(Path(__file__))}
    run.save(out/'PROTOCOL.json',protocol)
    with (run.PLAN/'DECISIONS.jsonl').open('a',encoding='utf8') as handle:
        handle.write(json.dumps({'id':run.HERE.name.upper()+'_GOAL_CAP512_SINGLE',**protocol},ensure_ascii=False)+'\n')
    pipeline=run.module()
    def exact_final(question,nodes,root,readout,examples):
        values={'ORIGINAL_QUESTION':question,'INTERMEDIATE_QA_BLOCK':'','GOAL_STEP_ID':1,
            'GOAL_ANSWER':nodes[-1]['answer'],'ORIGINAL_QUERY_RETRIEVED_PASSAGES':pipeline.passages(root)}
        prompt=prompts.FINAL.format_map(values)
        context=prompt.split('Actual task:\n\nOriginal question:\n'+question+'\n\nContext:\n',1)[1].split('\n\nFinal response requirements:\n',1)[0]
        block=context.split('\n\nRetrieved passages:\n',1)[0]
        return prompt,context,block
    pipeline.render_final=exact_final
    from scripts.cli_support import temporary_api_key
    from experiments.query_decomposition.p02_formal1500_v1_20261021.network import provider_dns
    class Events:
        def __call__(self,value):
            with (out/'trace.jsonl').open('a',encoding='utf8') as h:h.write(json.dumps({'utc':run.utc(),'id':CASE_ID,**value},ensure_ascii=False)+'\n')
        def event(self,event,**fields):self({'event':event,**fields})
    events=Events()
    with ExitStack() as stack:
        seed=stack.enter_context(closing(run.ro(source/'seed.sqlite3')))
        cache=stack.enter_context(run.Cache(out/'checkpoint.sqlite3'))
        budget=stack.enter_context(closing(run.Budget(config,out,config['budget']['cumulative_cap_cny'])))
        stack.enter_context(temporary_api_key('api_code_qwen.txt',allowed_root=run.ROOT,provider='dashscope'))
        stack.enter_context(provider_dns(urlsplit(config['generation']['base_url']).hostname,events))
        live=stack.enter_context(closing(run.RecoveryAsk(cache,seed,config,budget,events)))
        def retrieve(query):
            row=seed.execute('SELECT packet FROM retrieval WHERE dataset=? AND qid=? AND query=?',(case['dataset'],case['qid'],query)).fetchone()
            assert row is not None,'This diagnostic must not retrieve new documents'
            return deepcopy(json.loads(row[0]))
        def ask(stage,prompt,maximum,allow_final_truncation=False):
            actual=GOAL_CAP if stage=='legacy/answer/1' else maximum
            events({'event':'cap','stage':stage,'original_max_tokens':maximum,'actual_max_tokens':actual})
            return live(stage,prompt,actual,allow_final_truncation=allow_final_truncation)
        result=pipeline.Pipeline(config).execute(case,retrieve,ask,record_node=lambda n:cache.put('node',str(n['step_id']),n))
        cache.put('prediction',CASE_ID,result)
        receipts=cache.requests()
        assert all(r['result'].get('finish_reason')=='stop' for r in receipts)
    run.save(out/'RESULT.json',result)
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    ref=next(r for r in run.rows(run.COHORT/'references90.local.frozen.jsonl') if r['id']==CASE_ID)
    score=metric('musique')(result['prediction'],ref)
    usage={'input':sum(r['result']['usage']['prompt_tokens'] for r in receipts),
        'output':sum(r['result']['usage']['completion_tokens'] for r in receipts)}
    summary={'status':'completed','case_id':CASE_ID,'question':case['question'],'goal_max_tokens':GOAL_CAP,
        'goal_answer':result['nodes'][-1]['answer'],'final_answer':result['prediction'],'finish_reason':result['finish_reason'],
        'usage':usage,'estimated_generation_CNY':usage['input']*0.8e-6+usage['output']*2e-6,
        'native_metrics':score,'receipt_details':[{'stage':r['stage'],'finish_reason':r['result']['finish_reason'],'usage':r['result']['usage']} for r in receipts],
        'original90_results_preserved':True,'main_evaluation_remains_paused':True}
    run.save(out/'SUMMARY.json',summary)
    (out/'COMPLETE_ANSWER.txt').write_text('Question:\n'+case['question']+'\n\nGoal factual answer:\n'+summary['goal_answer']+'\n\nFinal short answer:\n'+summary['final_answer']+'\n',encoding='utf8')
    run.emit(summary)


if __name__=='__main__':
    main()
