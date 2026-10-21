"""Check token-normalized tail gradients and AdamW/scheduler exact CPU resume."""
import copy,importlib.util,json,sys,tempfile
from pathlib import Path
from reaudit import HERE,OUT,write
sys.path[:0]=[str(HERE/'remote/code'),str(HERE/'remote/runtime')]
from contract import lr_factor
from model_runtime import collate

def main():
    import torch
    torch.manual_seed(1729)
    records=[]
    for i in range(12):
        prompt=[1]*(1+i%4);target=[2+j%7 for j in range(2+i%3)]
        records.append({'input_ids':prompt+target,'labels':[-100]*len(prompt)+target,'target_tokens':len(target)})
    # A tiny linear computation checks numerical aggregation only, not model quality.
    w=torch.randn(11,11,requires_grad=True);w2=w.detach().clone().requires_grad_(True)
    def loss(param,batch):
        logits=param[batch['input_ids']]
        return torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,11),batch['labels'][:,1:].reshape(-1),ignore_index=-100)
    joint=loss(w,collate(records,0,torch,device='cpu'));joint.backward()
    count=sum(r['target_tokens'] for r in records)
    micro_loss=0.
    for r in records:
        value=loss(w2,collate([r],0,torch,device='cpu'));scale=r['target_tokens']/count
        (value*scale).backward();micro_loss+=float(value.detach())*scale
    error=float((w.grad-w2.grad).abs().max());assert torch.allclose(w.grad,w2.grad,rtol=1e-5,atol=1e-7)
    assert abs(float(joint.detach())-micro_loss)<1e-6
    a=torch.tensor([1.,2.],requires_grad=True)
    def optimizer(p):
        opt=torch.optim.AdamW([p],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        return opt,torch.optim.lr_scheduler.LambdaLR(opt,lr_factor)
    opt,sched=optimizer(a)
    def update(p,opt,sched):opt.zero_grad();(p.square().sum()).backward();opt.step();sched.step()
    for _ in range(55):update(a,opt,sched)
    b=a.detach().clone().requires_grad_(True);opt2,sched2=optimizer(b)
    # Real serialization through CPU preserves optimizer moments and scheduler state.
    with tempfile.TemporaryFile() as f:
        torch.save({'optimizer':opt.state_dict(),'scheduler':sched.state_dict()},f);f.seek(0);saved=torch.load(f,weights_only=False)
    opt2.load_state_dict(saved['optimizer']);sched2.load_state_dict(saved['scheduler'])
    update(a,opt,sched);update(b,opt2,sched2)
    assert torch.equal(a,b) and opt.param_groups[0]['lr']==opt2.param_groups[0]['lr']
    result={'passed':True,'tail_examples':12,'target_tokens':count,'weighted_microbatch_vs_joint_max_gradient_error':error,'actual_padding_labels_masked':True,'adamw_scheduler_serialized_resume_at55_matches_uninterrupted56':True,'fixed_horizon':330,'scope':'Synthetic CPU numerical check, not 4B GPU smoke or semantic evaluation.','torch':torch.__version__}
    write(OUT/'CPU_NUMERICAL_CHECKS.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
