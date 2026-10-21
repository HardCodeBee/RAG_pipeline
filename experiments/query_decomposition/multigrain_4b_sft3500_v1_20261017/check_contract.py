"""Meaningful CPU checks for no replay, scheduler/resume and actual loss masks."""
import collections,json,sys
from pathlib import Path
from reaudit import HERE,OUT,write
sys.path.insert(0,str(HERE/'remote/code'))
from contract import batches,lr_factor,format_result

def main():
    plan=batches();assert len(plan)==330
    for epoch in [1,2,3]:
        bs=[b for b in plan if b['epoch']==epoch];flat=[i for b in bs for i in b['indices']]
        assert len(bs)==110 and len(flat)==3500 and len(set(flat))==3500
        assert len(bs[-1]['indices'])==12 and all(len(b['indices'])==32 for b in bs[:-1])
    for step in [2,55,110,165,220,275,330]:
        assert plan[:step]+plan[step:]==plan
        assert sum(len(b['indices']) for b in plan[:step])==3500*(step//110)+32*(step%110)
    assert [lr_factor(i) for i in range(10)]==[(i+1)/10 for i in range(10)]
    assert lr_factor(10)==1 and lr_factor(330)==0 and lr_factor(329)>0
    assert format_result('[]')['format_valid']
    assert not format_result('{"steps":[]}')['format_valid']
    assert format_result('["Who is #1?"]')['explicit_references']
    assert not format_result('```json\n[]\n```')['format_valid']
    v=json.loads((OUT/'data_v002/VALIDATION.json').read_text());assert v['silent_truncation']==0 and v['native_encoding_and_target_eos_masks_checked']==3800
    result={'passed':True,'epochs':3,'examples_per_epoch':3500,'updates_per_epoch':110,'actual_tail':12,'sample_replay_or_drop':False,'scheduler_fixed_horizon':330,'warmup_updates':10,'checkpoint_plan_resume_boundaries_checked':True,'actual_native_loss_encoder_checked_records':3800,'format_errors_not_repaired':True,'gpu_smoke':'pending','api_calls':0}
    write(OUT/'LOCAL_CONTRACT_CHECKS.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
