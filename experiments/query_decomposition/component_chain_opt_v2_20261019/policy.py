"""Two isolated edits; frozen component modules remain unchanged."""
import json

from experiments.query_decomposition.component_chain_c08_c02_v1_20261019 import pipeline as frozen
from .prepare import ROOT,read,HERE


class IntermediateLength(RuntimeError):
    def __init__(self,stage,request_id,answer):
        super().__init__('Known intermediate length termination')
        self.stage,self.request_id,self.answer=stage,request_id,answer


def final_edit(prompt,enabled):
    if not enabled:return prompt
    assert prompt.endswith(frozen.final_prompts.FINAL_READOUT)
    return prompt[:-len(frozen.final_prompts.FINAL_READOUT)]+read(HERE/'PROTOCOL.frozen.json')['phase_B']['P1_readout']


def execute(case,retrieve,ask,condition,record_node=None):
    base = condition if condition in ('legacy','c08_only','c02_only','c08_c02') else 'c08_c02'
    recover = condition in ('r1','r1p1')
    concise = condition in ('p1','r1p1')
    recorded,actual_final=[],None
    def record(node):
        recorded.append(node)
        if record_node:record_node(node)
    def forward(stage,prompt,maximum,**kwargs):
        nonlocal actual_final
        if stage.endswith('/final'):
            prompt=final_edit(prompt,concise);actual_final=prompt
        return ask(stage,prompt,maximum,**kwargs)
    try:
        output=frozen.execute(case,retrieve,forward,variant=base,record_node=record)
    except IntermediateLength as failure:
        if not recover or '/answer/' not in failure.stage:raise
        root=retrieve(case['question'])
        history=[{'step_id':n['step_id'],'request':n['request'],'answer':n['answer']} for n in recorded]
        usable=frozen.c08.auxiliary_pairs(history)
        helper=frozen.legacy_child.render_helper(usable,None,'Answer') if usable else ''
        context=helper+'\n\n'+frozen.passages(root) if helper else frozen.passages(root)
        examples=json.loads((ROOT/'experiments/query_decomposition/multigrain_prompt_v3_20261017/final_examples.json').read_text(encoding='utf-8'))
        prompt=frozen.final_prompts.final(case['question'],context,examples)
        rid,prediction,finish=forward(base+'/final',prompt,64,allow_final_truncation=True)
        output={**case,'variant':base,'status':'success_fallback' if finish=='stop' else 'success_fallback_truncated',
                'nodes':recorded,'history_all':history,'history_usable':usable,'helper':helper,
                'original_retrieval':root,'final_context':context,'final_prompt':prompt,
                'prediction':prediction,'request_id':rid,'finish_reason':finish,
                'fallback':{'reason':'child_length','failed_stage':failure.stage,'failed_request_id':failure.request_id,
                            'unpropagated_truncated_answer':failure.answer,'remaining_nodes_skipped':True,
                            'normal_goal_path_changed':False}}
    output['condition']=condition
    output['base_final_prompt']=output['final_prompt']
    output['final_prompt']=actual_final
    output['isolated_edits']={'child_length_recovery':recover,'final_readout_P1':concise}
    return output
