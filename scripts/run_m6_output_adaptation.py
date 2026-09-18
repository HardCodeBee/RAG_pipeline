"""M6-output adaptation primitives and a frozen, fit-only implementation pilot."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0',
                  USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import run_m6_objective_readout as old

ROOT=Path(__file__).resolve().parents[1]
BASE=old.BASE
OUT=ROOT/'outputs/router/hotpotqa_bd_router_v1/runs/m6_output_adaptation_v1'
PLAN=ROOT/'analysis/hotpotqa_router/m6_output_adaptation_plan_20260915.md'
MODEL=Path('C:/Users/12442/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')
CONFIG=dict(epochs=4,batch_size=8,max_length=128,steps_per_epoch=768,
    total_steps=3072,warmup_steps=307,encoder_lr=2e-5,head_lr=.001,
    encoder_weight_decay=.01,head_weight_decay=0.,head_lambda=.001,
    clip_norm=1.,order_seed=2026091517,model_seed=2026091515,
    pilot_steps=8,pilot_fold=0,cal_tie_atol=1e-12,
    bootstrap_draws=20000,bootstrap_seed=2026091519,
    quantiles=[.05/8,1-.05/8],minimum_increment=.002,minimum_fixed_gain=.01)
PRIMARY=['A_minus_C','A_minus_B','A_minus_Dense','A_minus_BM25']
sha,read,write,save=old.sha,old.read,old.write,old.save


def versions():
    return {n:importlib.metadata.version(n) for n in ('numpy','torch','transformers','scipy')}


def inputs():
    return old.input_paths()+[BASE/'layer_pooling_v1/tokens.npz',MODEL/'model.safetensors',MODEL/'config.json']


def sources():
    return [Path(__file__).resolve(),PLAN,ROOT/'scripts/check_m6_output_adaptation_pilot.py',
        ROOT/'scripts/run_m6_output_adaptation_formal.py',Path(old.__file__),
        ROOT/'scripts/m6_objective_math.py']


def data():
    d=old.data()
    with np.load(BASE/'layer_pooling_v1/tokens.npz',allow_pickle=False) as z:
        d['tokens']={k:z[k].copy() for k in z.files}
    assert set(d['tokens'])=={'input_ids','attention_mask','token_type_ids','special_tokens_mask'}
    assert all(v.shape==(9600,128) and v.dtype==np.int64 for v in d['tokens'].values())
    return d


def freeze():
    assert not OUT.exists()
    d=data()
    assert all(p.exists() for p in sources()+inputs())
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',dict(status='frozen_before_M6_output_adaptation_pilot_and_formal_training',
        created_at_utc=datetime.now(timezone.utc).isoformat(),config=CONFIG,primary=PRIMARY,
        versions=versions(),source_sha256={str(p):sha(p) for p in sources()},
        input_sha256={str(p):sha(p) for p in inputs()},queries=len(d['gap']),
        scope='Consumed old9600; direct mean6 supervision; no independent-source confirmation; no external calls'))
    print(json.dumps(dict(status='frozen',protocol_sha256=sha(OUT/'protocol.json'))),flush=True)


def bound(binding):
    assert binding and sha(OUT/'protocol.json')==binding
    p=read(OUT/'protocol.json')
    assert p['config']==CONFIG and p['primary']==PRIMARY and p['versions']==versions()
    for section in ('source_sha256','input_sha256'):
        assert all(sha(k)==v for k,v in p[section].items()),section
    return p,data()


def model(torch,fold,arm):
    from transformers import AutoModel
    assert arm in ('A','C') and fold in range(5)
    torch.manual_seed(CONFIG['model_seed']+fold)
    torch.cuda.manual_seed_all(CONFIG['model_seed']+fold)
    encoder=AutoModel.from_pretrained(MODEL,local_files_only=True,use_safetensors=True)
    assert len(encoder.encoder.layer)==12 and encoder.config.hidden_size==384
    encoder.encoder.layer=torch.nn.ModuleList(encoder.encoder.layer[:6])
    encoder.config.num_hidden_layers=6
    encoder.pooler=None
    encoder.requires_grad_(arm=='A').to('cuda').eval()
    head=torch.nn.Linear(384,1)
    initial=torch.load(BASE/f'layer_pooling_v1/fold{fold}_M6_head.pt',map_location='cpu',weights_only=True)
    head.load_state_dict(initial,strict=True)
    assert all(v.dtype==torch.float32 for v in initial.values())
    head.requires_grad_(True).to('cuda').eval()
    assert all(not module.training for module in encoder.modules())
    return encoder,head


def named(encoder,head):
    return [('encoder.'+n,p) for n,p in encoder.named_parameters()]+[('head.'+n,p) for n,p in head.named_parameters()]


def digest_parameters(parameters):
    h=hashlib.sha256()
    for name,p in parameters:
        a=p.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(np.asarray(a.shape,np.int64).tobytes());h.update(a.tobytes())
    return h.hexdigest()


def feature(encoder,batch,special,torch):
    hidden=encoder(**batch).last_hidden_state
    mask=batch['attention_mask'].bool() & ~special.bool()
    count=mask.sum(1)
    mean=(hidden.float()*mask.unsqueeze(-1)).sum(1)/count.clamp_min(1).float().unsqueeze(1)
    mean=torch.where((count==0).unsqueeze(1),hidden[:,0].float(),mean)
    return torch.nn.functional.normalize(mean,p=2,dim=1)


def infer(encoder,head,tokens,indices,torch):
    assert len(indices)>0 and len(indices)%8==0
    xs=[];scores=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(indices),8):
            ids=indices[start:start+8]
            batch={k:torch.from_numpy(v[ids]).to('cuda') for k,v in tokens.items() if k!='special_tokens_mask'}
            special=torch.from_numpy(tokens['special_tokens_mask'][ids]).to('cuda')
            x=feature(encoder,batch,special,torch)
            xs.append(x.float().cpu().numpy());scores.append(head(x).squeeze(1).float().cpu().numpy())
    x,s=np.concatenate(xs),np.concatenate(scores)
    assert x.dtype==s.dtype==np.float32 and np.isfinite(x).all() and np.isfinite(s).all()
    return x,s


def optimizer_for(encoder,head,arm,torch):
    groups=[]
    if arm=='A':
        for decay in (False,True):
            ps=[p for n,p in encoder.named_parameters() if (not(n.endswith('bias') or 'LayerNorm.weight' in n))==decay]
            groups.append(dict(params=ps,lr=CONFIG['encoder_lr'],base_lr=CONFIG['encoder_lr'],
                weight_decay=CONFIG['encoder_weight_decay'] if decay else 0.))
    groups.append(dict(params=list(head.parameters()),lr=CONFIG['head_lr'],base_lr=CONFIG['head_lr'],weight_decay=0.))
    return torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,foreach=False)


def lr_multiplier(step):
    assert 0<=step<CONFIG['total_steps']
    s=step+1
    return s/CONFIG['warmup_steps'] if s<CONFIG['warmup_steps'] else (CONFIG['total_steps']-s)/(CONFIG['total_steps']-CONFIG['warmup_steps'])


def train_epoch(encoder,head,arm,tokens,fit,fit_gap,optimizer,fold,epoch,torch,max_steps=None):
    """Only this fold's fit labels enter gradient updates. Epoch is zero-based."""
    assert arm in ('A','C') and len(fit)==len(fit_gap)==6144 and epoch in range(CONFIG['epochs'])
    assert max_steps is None or (epoch==0 and max_steps==CONFIG['pilot_steps'])
    assert not encoder.training and not head.training
    w=np.where(abs(fit_gap)>1e-12,abs(fit_gap),0.)
    norm=float(w.mean());assert norm>0
    rng=np.random.default_rng(CONFIG['order_seed']+fold)
    for _ in range(epoch+1): order_local=rng.permutation(len(fit))
    order=fit[order_local]
    trainable=[(n,p) for n,p in named(encoder,head) if p.requires_grad]
    frozen=[(n,p) for n,p in named(encoder,head) if not p.requires_grad]
    encps=[p for n,p in trainable if n.startswith('encoder.')]
    headps=list(head.parameters())
    first=None;rows=[];started=time.perf_counter()
    for batch_no,start in enumerate(range(0,len(order),8)):
        if max_steps is not None and batch_no>=max_steps: break
        ids=order[start:start+8];local=order_local[start:start+8]
        batch={k:torch.from_numpy(v[ids]).to('cuda') for k,v in tokens.items() if k!='special_tokens_mask'}
        special=torch.from_numpy(tokens['special_tokens_mask'][ids]).to('cuda')
        target=torch.tensor(fit_gap[local]>0,dtype=torch.float32,device='cuda')
        weights=torch.tensor(w[local],dtype=torch.float32,device='cuda')
        optimizer.zero_grad(set_to_none=True)
        step=epoch*CONFIG['steps_per_epoch']+batch_no
        mult=lr_multiplier(step)
        for group in optimizer.param_groups: group['lr']=group['base_lr']*mult
        with torch.autocast('cuda',dtype=torch.bfloat16):
            x=feature(encoder,batch,special,torch)
            logits=head(x).squeeze(1).float()
            supervised=(weights*torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')).mean()/norm
        penalty=CONFIG['head_lambda']*.5*head.weight.float().square().sum()
        loss=supervised+penalty
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None for _,p in trainable)
        assert torch.stack([torch.isfinite(p.grad).all() for _,p in trainable]).all()
        assert all(p.grad is None for _,p in frozen)
        if batch_no==0:
            first=dict(indices=ids.copy(),features=x.detach().float().cpu().numpy(),
                logits=logits.detach().cpu().numpy(),targets=target.cpu().numpy(),weights=weights.cpu().numpy(),
                normalizer=np.array(norm),head_weight=head.weight.detach().cpu().numpy().copy(),
                head_bias=head.bias.detach().cpu().numpy().copy(),head_weight_grad=head.weight.grad.detach().cpu().numpy().copy(),
                head_bias_grad=head.bias.grad.detach().cpu().numpy().copy(),
                module_gradient_norm={n:float(p.grad.detach().float().norm()) for n,p in trainable})
        encnorm=float(torch.nn.utils.clip_grad_norm_(encps,CONFIG['clip_norm'])) if encps else 0.
        headnorm=float(torch.nn.utils.clip_grad_norm_(headps,CONFIG['clip_norm']))
        assert np.isfinite([encnorm,headnorm]).all()
        rows.append([float(supervised.detach()),float(penalty.detach()),float(loss.detach()),encnorm,headnorm,mult,float(weights.sum()==0)])
        optimizer.step()
    torch.cuda.synchronize()
    table=np.asarray(rows,np.float64)
    actual_order=order[:len(rows)*8].astype(np.int64)
    summary=dict(fold=fold,arm=arm,epoch=epoch+1,optimizer_steps=len(rows),
        batch_order_sha256=hashlib.sha256(actual_order.tobytes()).hexdigest(),
        fit_mean_weight=norm,online_weighted_BCE=float(table[:,0].mean()),
        online_head_penalty=float(table[:,1].mean()),online_loss=float(table[:,2].mean()),
        zero_weight_batches=int(table[:,6].sum()),all_losses_and_gradients_finite=True,
        mean_encoder_head_gradnorm_before_clip=table[:,3:5].mean(0).tolist(),
        first_lr_multiplier=float(table[0,5]),last_lr_multiplier=float(table[-1,5]),
        seconds=time.perf_counter()-started)
    return summary,first,table,actual_order


def state(encoder,head,arm):
    return {n:p.detach().cpu().clone() for n,p in named(encoder,head) if p.requires_grad}


def save_checkpoint(path,encoder,head,arm,fold,epoch,binding):
    import torch
    assert not Path(path).exists()
    with Path(path).open('xb') as f:
        torch.save(dict(trained_parameters=state(encoder,head,arm),arm=arm,fold=fold,epoch=epoch,
            protocol_sha256=binding,core_sha256=sha(__file__)),f)


def restore(torch,path,binding):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    assert cp['protocol_sha256']==binding and cp['core_sha256']==sha(__file__)
    encoder,head=model(torch,cp['fold'],cp['arm'])
    expected={n:p for n,p in named(encoder,head) if p.requires_grad}
    assert set(cp['trained_parameters'])==set(expected)
    with torch.no_grad():
        for n,p in expected.items():
            v=cp['trained_parameters'][n]
            assert v.dtype==torch.float32 and v.shape==p.shape and torch.isfinite(v).all()
            p.copy_(v.to(p.device))
    return encoder,head,cp


def bce(scores,gap):
    w=np.where(abs(gap)>1e-12,abs(gap),0.)
    return float(w@np.logaddexp(0.,(1-2*(gap>0))*scores.astype(float))/w.sum())


def pilot(binding):
    _,d=bound(binding)
    assert not (OUT/'pilot_started.json').exists()
    write(OUT/'pilot_started.json',dict(protocol_sha256=binding,started_at_utc=datetime.now(timezone.utc).isoformat()))
    torch=old.gpu();fold=CONFIG['pilot_fold'];fit=d['folds'][fold][0];probe=fit[:64]
    torch.cuda.reset_peak_memory_stats();records=[];artifacts=[];started=time.perf_counter()
    for arm in ('A','C'):
        encoder,head=model(torch,fold,arm)
        trainable=[(n,p) for n,p in named(encoder,head) if p.requires_grad]
        frozen=[(n,p) for n,p in named(encoder,head) if not p.requires_grad]
        before={n:p.detach().cpu().clone() for n,p in named(encoder,head)}
        frozen_before=digest_parameters(frozen)
        initial_hash=digest_parameters(named(encoder,head))
        x0,s0=infer(encoder,head,d['tokens'],fit,torch)
        with np.load(BASE/f'layer_pooling_v1/fold{fold}_M6_fit.npz',allow_pickle=False) as z:
            assert np.array_equal(s0,z['final_native_fit_logits'])
        assert np.array_equal(x0,d['features'][fit])
        optimizer=optimizer_for(encoder,head,arm,torch)
        assert not optimizer.state
        summary,first,trace,order=train_epoch(encoder,head,arm,d['tokens'],fit,d['gap'][fit],optimizer,fold,0,torch,max_steps=CONFIG['pilot_steps'])
        x1,s1=infer(encoder,head,d['tokens'],probe,torch)
        changes={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in named(encoder,head)}
        assert frozen_before==digest_parameters(frozen)
        assert any(changes[n]>0 for n,_ in trainable if n.startswith('head.'))
        if arm=='A':
            modules=['encoder.embeddings.']+[f'encoder.encoder.layer.{k}.' for k in range(6)]
            assert all(any(changes[n]>0 for n,_ in trainable if n.startswith(prefix)) for prefix in modules)
            assert all(any(v>0 for n,v in first['module_gradient_norm'].items() if n.startswith(prefix)) for prefix in modules)
            assert not np.array_equal(x1,x0[:64])
        else:
            assert all(changes[n]==0 for n,_ in frozen) and np.array_equal(x1,x0[:64])
        stem='pilot_'+arm
        cp=OUT/(stem+'.pt');save_checkpoint(cp,encoder,head,arm,fold,-1,binding);artifacts.append(cp)
        gradients=first.pop('module_gradient_norm')
        ap=OUT/(stem+'.npz')
        save(ap,probe_indices=probe,initial_features=x0[:64],initial_logits=s0[:64],final_features=x1,final_logits=s1,
            training_order=order,step_trace=trace,**{'first_'+k:v for k,v in first.items()})
        artifacts.append(ap)
        row=dict(summary,initial_parameters_sha256=initial_hash,frozen_before_sha256=frozen_before,
            frozen_after_sha256=digest_parameters(frozen),trained_parameter_count=sum(p.numel() for _,p in trainable),
            trained_names=[n for n,_ in trainable],first_parameter_gradient_norm=gradients,
            parameter_max_changes=changes,full_fit_initial_feature_and_score_exact=True,
            mean6_probe_max_change=float(np.max(abs(x1-x0[:64]))),
            final_parameters_sha256=digest_parameters(named(encoder,head)))
        del optimizer,encoder,head,trainable,frozen,before;gc.collect();torch.cuda.empty_cache()
        encoder,head,_=restore(torch,cp,binding)
        xr,sr=infer(encoder,head,d['tokens'],probe,torch)
        assert np.array_equal(xr,x1) and np.array_equal(sr,s1)
        row['checkpoint_probe_replay_exact']=True
        records.append(row)
        del encoder,head;gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(pilot_arm=arm,steps=summary['optimizer_steps'],mean6_probe_max_change=row['mean6_probe_max_change'])),flush=True)
    assert records[0]['initial_parameters_sha256']==records[1]['initial_parameters_sha256']
    assert records[0]['batch_order_sha256']==records[1]['batch_order_sha256']
    write(OUT/'pilot.json',dict(status='complete_M6_output_pilot_pending_independent_check',protocol_sha256=binding,
        arms=records,elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),pilot_optimizer_steps=16,
        encoder_query_forwards=2*(6144+64+64+64),cal_quality_evaluations=0,test_quality_evaluations=0,
        new_api_calls=0,formal_training_started=False,artifact_sha256={str(p):sha(p) for p in artifacts}))
    print(json.dumps(dict(status='pilot_complete',seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','pilot']);p.add_argument('--protocol-sha256')
    a=p.parse_args()
    if a.mode=='freeze': freeze()
    else: pilot(a.protocol_sha256)
