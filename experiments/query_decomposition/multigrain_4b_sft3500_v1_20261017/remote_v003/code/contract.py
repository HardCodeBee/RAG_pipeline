"""Training schedule/data invariants, independent of CUDA and model packages."""
import hashlib,json,random
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def batches(size=3500,epochs=3,seed=1729):
    plan=[]
    for epoch in range(epochs):
        order=list(range(size));random.Random(seed+epoch).shuffle(order)
        plan.extend({'epoch':epoch+1,'indices':order[i:i+32]} for i in range(0,size,32))
    return plan

def lr_factor(step):
    # LambdaLR calls step=0 before the first update; horizon is never shortened.
    return (step+1)/10 if step<10 else max(0.,(330-step)/320)

def validate_config(c):
    expected={'train_records':3500,'dev_records':150,'micro_batch':1,'max_length':1024,'seed':1729,'epochs_max':3,'warmup_steps':10,'scheduler_horizon':330,'expected_adapter_parameters':66060288,'allow_tf32':False,'gradient_checkpointing':True,'template_policy':'qwen3_instruct_native_nonthinking','expected_base_parameters':4022468096}
    for k,v in expected.items():assert c[k]==v,(k,c.get(k),v)
    assert c['optimizer']=={'lr':0.0001,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.01,'clip':1.0}
    for name,digest in c['data_sha256'].items():assert sha(Path(c['data_dir'])/name)==digest,name
    assert sha(c['prompt_file'])==c['prompt_sha256']
    assert c.get('adapter_path') is None,'Formal initialization must use new adapter'

def format_result(raw):
    try:
        parsed=json.loads(raw)
        valid=isinstance(parsed,list) and all(isinstance(q,str) and q.strip() for q in parsed)
    except (ValueError,TypeError):parsed=None;valid=False
    import re
    refs=bool(re.search(r'#\d+',raw or ''))
    return {'parsed':parsed,'format_valid':valid,'explicit_references':refs}
