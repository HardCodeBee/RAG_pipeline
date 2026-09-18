"""Fixed base/H encoder by mean6/CLS12 readout experiment on consumed old9600."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0',
                  USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
import m6_probe_math as hard
import run_m6_objective_readout as old

ROOT=Path(__file__).resolve().parents[1]
BASE=old.BASE
HIST=BASE/'lp_ft_v2'
OUT=ROOT/'outputs/router/hotpotqa_bd_router_v1/runs/m6_trained_readout_v1'
PLAN=ROOT/'analysis/hotpotqa_router/m6_trained_readout_plan_20260915.md'
MODEL=Path('C:/Users/12442/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')
ARMS=['B12','H6','H12']
POLICIES=['B6','B12','H6','H12','H_native','Dense','BM25']
PRIMARY=['H6_minus_B6','H12_minus_B12','H6_minus_H12','readout_interaction','H6_minus_Dense','H6_minus_BM25']
CONFIG=dict(regularization=.001,formal_solves=15,pilot_solves=3,new_encoder_fits=0,
    new_api_calls=0,new_thresholds=0,bootstrap_draws=20000,bootstrap_seed=2026091513,
    quantiles=[.05/12,1-.05/12],minimum_increment=.002,minimum_fixed_gain=.01,
    batch_size=8,max_length=128,gradient_acceptance=1e-8)
sha,read,write,save=old.sha,old.read,old.write,old.save


def versions():
    return {n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','transformers','threadpoolctl')}


def sources():
    return [Path(__file__).resolve(),PLAN,ROOT/'scripts/check_m6_trained_readout.py',
        ROOT/'scripts/check_m6_student_transfer.py',
        Path(hard.__file__),Path(old.__file__),ROOT/'scripts/m6_objective_math.py',
        BASE/'weighted_linear_probe.py',BASE/'weighted_linear_probe_refined.py']


def inputs():
    return old.input_paths()+[BASE/'layer_pooling_v1/tokens.npz',
        BASE/'layer_pooling_v1/protocol.json',HIST/'protocol.json',HIST/'completion_record.json',
        HIST/'separate_checks.json',HIST/'all_endpoints_frozen.json',HIST/'predictions.npz',
        MODEL/'model.safetensors',MODEL/'config.json']+[
        HIST/f'fold{f}{suffix}' for f in range(5) for suffix in
        ('.pt','_fit.npz','_training.json','_lp_head.pt','_lp_predictions.npz')]+[
        BASE/n for n in ('run_lp_ft_v2.py','encoder_scope_training.py','run_e11c.py','run_e10.py','run_e02.py')]


def data():
    d=old.data()
    with np.load(BASE/'layer_pooling_v1/features.npz',allow_pickle=False) as z:
        d['C12']=z['C12'].copy()
    with np.load(HIST/'predictions.npz',allow_pickle=False) as z:
        assert np.array_equal(z['query_ids'],d['query_ids']) and np.array_equal(z['group_ids'],d['group_ids'])
        assert np.array_equal(z['utility'],d['utility'])
        d['H_native']=z['H_native'].copy();d['C12_prior']=z['P_native'].copy()
    with np.load(BASE/'layer_pooling_v1/tokens.npz',allow_pickle=False) as z:
        d['tokens']={k:z[k].copy() for k in z.files}
    assert set(d['tokens'])=={'input_ids','attention_mask','token_type_ids','special_tokens_mask'}
    assert all(v.shape==(9600,128) and v.dtype==np.int64 for v in d['tokens'].values())
    return d


def validate_prior():
    c=read(HIST/'separate_checks.json');p=read(HIST/'protocol.json')
    assert c['status']=='passed_independent_LP_H_F_OOF_training_and_four_comparison_checks'
    assert c['protocol_sha256']==sha(HIST/'protocol.json') and c['results_sha256']==sha(HIST/'results.json')
    completion=read(HIST/'completion_record.json')
    assert completion['checks_sha256']==sha(HIST/'separate_checks.json')
    bindings={**read(BASE/'layer_pooling_v1/completion_record.json')['artifact_sha256'],
              **completion['artifact_sha256'],**c['source_sha256'],**c['artifact_sha256']}
    for path in inputs():
        if str(path) in bindings: assert sha(path)==bindings[str(path)],str(path)
    for f in range(5):
        stat=read(HIST/f'fold{f}_training.json')
        assert stat['fold']==f and stat['calibration_quality_evaluations']==0 and stat['outer_test_quality_evaluations']==0
        assert stat['stopping_state']=='criterion_met' and stat['all_losses_and_gradients_finite']
    assert p['training_rule']['max_length']==128 and p['training_rule']['batch_size']==8


def bound(binding):
    assert binding and sha(OUT/'protocol.json')==binding
    p=read(OUT/'protocol.json');assert p['config']==CONFIG and p['primary']==PRIMARY and p['versions']==versions()
    for section in ('source_sha256','input_sha256'):
        assert all(sha(path)==value for path,value in p[section].items()),section
    return p,data()


def freeze():
    assert not OUT.exists()
    validate_prior();d=data()
    assert d['C12'].shape==(9600,384) and d['C12'].dtype==np.float32
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',dict(status='frozen_before_new_H_views_and_readouts',
        created_at_utc=datetime.now(timezone.utc).isoformat(),config=CONFIG,primary=PRIMARY,policies=POLICIES,
        versions=versions(),source_sha256={str(p):sha(p) for p in sources()},
        input_sha256={str(p):sha(p) for p in inputs()},
        scope='Consumed old9600; fixed H endpoints; no inner CV on supervised representations; no independent source validation'))
    print(json.dumps(dict(status='frozen',protocol_sha256=sha(OUT/'protocol.json'))),flush=True)


def model(torch,fold=None):
    from transformers import AutoModel
    encoder=AutoModel.from_pretrained(MODEL,local_files_only=True,use_safetensors=True)
    head=None
    if fold is not None:
        checkpoint=torch.load(HIST/f'fold{fold}.pt',map_location='cpu',weights_only=True)
        assert checkpoint['fold']==fold and checkpoint['protocol_sha256']==sha(HIST/'protocol.json')
        assert checkpoint['runner_sha256']==sha(BASE/'run_lp_ft_v2.py')
        assert checkpoint['protocol_id']==read(HIST/'protocol.json')['id']
        params=checkpoint['trained_parameters'];assert len(params)==199
        update={k.removeprefix('encoder.'):v for k,v in params.items() if k.startswith('encoder.')}
        missing=encoder.load_state_dict(update,strict=False)
        assert set(missing.missing_keys)=={'pooler.dense.weight','pooler.dense.bias'} and not missing.unexpected_keys
        assert all(torch.equal(encoder.state_dict()[k],v) for k,v in update.items())
        head=torch.nn.Linear(384,1)
        head.load_state_dict({k.removeprefix('answer.'):v for k,v in params.items() if k.startswith('answer.')},strict=True)
        head.requires_grad_(False).to('cuda').eval()
    encoder.requires_grad_(False).to('cuda').eval()
    return encoder,head


def infer(encoder,head,tokens,indices,torch):
    assert len(indices)%8==0
    views=[];native=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(indices),8):
            ix=indices[start:start+8]
            batch={k:torch.from_numpy(v[ix]).to('cuda') for k,v in tokens.items() if k!='special_tokens_mask'}
            hidden=encoder(**batch,output_hidden_states=True).hidden_states
            content=batch['attention_mask'].bool() & ~torch.from_numpy(tokens['special_tokens_mask'][ix]).to('cuda').bool()
            count=content.sum(1)
            mean=(hidden[6].float()*content.unsqueeze(-1)).sum(1)/count.clamp_min(1).float().unsqueeze(1)
            mean=torch.where((count==0).unsqueeze(1),hidden[6][:,0].float(),mean)
            x6=torch.nn.functional.normalize(mean,p=2,dim=1)
            x12=torch.nn.functional.normalize(hidden[12][:,0],p=2,dim=1)
            assert x6.dtype==x12.dtype==torch.float32
            views.append(torch.stack([x6,x12]).cpu().numpy())
            if head is not None: native.append(head(x12).squeeze(1).float().cpu().numpy())
    result=np.concatenate(views,axis=1)
    assert np.isfinite(result).all() and result.dtype==np.float32
    assert np.max(abs(np.linalg.norm(result.astype(float),axis=2)-1))<2e-6
    return result,(np.concatenate(native) if head is not None else None)


def solve(x,gap,stem,metadata,journal=None):
    w=np.where(abs(gap)>1e-12,abs(gap),0.)
    m=hard.refined.fit(x,(gap>0).astype(float),w,regularization=.001)
    record=dict(metadata,**{k:v for k,v in m.items() if k!='coef'})
    save(OUT/(stem+'.npz'),coef=m['coef'],intercept=np.array(m['intercept']))
    write(OUT/(stem+'.json'),record)
    if journal is not None:
        journal.write(json.dumps(dict(stem=stem,**record),allow_nan=False)+'\n');journal.flush()
    assert m['accepted'],'Keep failed solution; do not increase numerical budget'
    return m,[OUT/(stem+'.npz'),OUT/(stem+'.json')]


def bce(scores,gap):
    w=np.where(abs(gap)>1e-12,abs(gap),0.)
    return float(w@np.logaddexp(0.,(1-2*(gap>0))*scores.astype(float))/w.sum())


def pilot(binding):
    _,d=bound(binding);assert not (OUT/'pilot.json').exists()
    torch=old.gpu();controls=old.old_replay(d,torch)
    ix=d['folds'][0][0][:64]
    encoder,head=model(torch)
    baseline,_=infer(encoder,head,d['tokens'],ix,torch)
    assert np.array_equal(baseline[0],d['features'][ix]) and np.array_equal(baseline[1],d['C12'][ix])
    del encoder;gc.collect();torch.cuda.empty_cache()
    encoder,head=model(torch,0);views,native=infer(encoder,head,d['tokens'],ix,torch)
    with np.load(HIST/'fold0_fit.npz',allow_pickle=False) as z: assert np.array_equal(native,z['fit_logits'][:64])
    captured=[]
    handle=encoder.encoder.layer[5].register_forward_hook(lambda mod,args,result:captured.append(result[0].detach().float().cpu().numpy()))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,64,8):
            ids=ix[start:start+8]
            encoder(**{k:torch.from_numpy(v[ids]).to('cuda') for k,v in d['tokens'].items() if k!='special_tokens_mask'})
    handle.remove();h=np.concatenate(captured).astype(float)
    mask=d['tokens']['attention_mask'][ix].astype(bool)&~d['tokens']['special_tokens_mask'][ix].astype(bool)
    counts=mask.sum(1);means=np.sum(h*mask[:,:,None],axis=1)/np.maximum(counts,1)[:,None]
    means[counts==0]=h[counts==0,0]
    independent=means/np.linalg.norm(means,axis=1)[:,None]
    pool_error=float(np.max(abs(independent-views[0])))
    assert pool_error<2e-6
    save(OUT/'pilot_features.npz',indices=ix,base_views=baseline,H_views=views,H_native=native)
    artifacts=[OUT/'pilot_features.npz']
    with threadpool_limits(limits=2):
        for arm,x in zip(ARMS,[baseline[1],views[0],views[1]]):
            m,paths=solve(x,d['gap'][ix],'pilot_'+arm,dict(role='pilot',arm=arm,rows=64))
            artifacts+=paths
            scores=old.native(x,m['coef'],m['intercept'],torch)
            path=OUT/('pilot_'+arm+'_scores.npz');save(path,scores=scores);artifacts.append(path)
    write(OUT/'pilot.json',dict(status='passed_base_views_H_native_pooling_and_three_head_pilot',
        protocol_sha256=binding,original_native_controls=controls,base_feature_max_difference=0.,
        H_native_max_difference=0.,independent_FP64_pool_max_difference=pool_error,
        new_encoder_query_forwards=192,new_encoder_fits=0,new_policy_effects=False,
        artifact_sha256={str(p):sha(p) for p in artifacts}))
    print(json.dumps(dict(status='pilot_passed',pilot_solves=3,pool_error=pool_error)),flush=True)


def run(binding):
    _,d=bound(binding);pilot_record=read(OUT/'pilot.json')
    assert pilot_record['protocol_sha256']==binding and pilot_record['status'].startswith('passed_')
    assert all(sha(p)==v for p,v in pilot_record['artifact_sha256'].items())
    assert not (OUT/'fit_started.json').exists()
    write(OUT/'fit_started.json',dict(protocol_sha256=binding,started_at_utc=datetime.now(timezone.utc).isoformat()))
    started=time.perf_counter();torch=old.gpu();heads=[];artifacts=[];records=[]
    with threadpool_limits(limits=2),(OUT/'solutions.jsonl').open('x',encoding='utf-8') as journal:
        for f,(fit,cal,test) in enumerate(d['folds']):
            encoder,original_head=model(torch,f)
            views,native=infer(encoder,original_head,d['tokens'],np.r_[fit,cal],torch)
            with np.load(HIST/f'fold{f}_fit.npz',allow_pickle=False) as z:
                assert np.array_equal(z['fit_indices'],fit) and np.array_equal(native[:len(fit)],z['fit_logits'])
            path=OUT/f'fold{f}_features_fit_cal.npz'
            save(path,fit_indices=fit,cal_indices=cal,fit_features=views[:,:len(fit)],cal_features=views[:,len(fit):],
                fit_H_native=native[:len(fit)],cal_H_native=native[len(fit):]);artifacts.append(path)
            models=[];fs=[];cs=[]
            for arm,xf,xc in zip(ARMS,[d['C12'][fit],views[0,:len(fit)],views[1,:len(fit)]],
                                     [d['C12'][cal],views[0,len(fit):],views[1,len(fit):]]):
                m,paths=solve(xf,d['gap'][fit],f'fold{f}_{arm}',dict(role='formal',fold=f,arm=arm),journal)
                models.append(m);artifacts+=paths
                fs.append(old.native(xf,m['coef'],m['intercept'],torch));cs.append(old.native(xc,m['coef'],m['intercept'],torch))
            if f>0:
                prior=torch.load(HIST/f'fold{f}_lp_head.pt',map_location='cpu',weights_only=True)
                assert np.array_equal(np.asarray(models[0]['coef'],np.float32),prior['weight'].numpy().reshape(384))
                assert np.float32(models[0]['intercept'])==prior['bias'].numpy()[0]
                with np.load(HIST/f'fold{f}_lp_predictions.npz',allow_pickle=False) as z:
                    assert np.array_equal(fs[0],z['final_native_fit_logits'])
            path=OUT/f'fold{f}_scores_fit_cal.npz'
            save(path,fit_indices=fit,cal_indices=cal,fit_scores=np.stack(fs),cal_scores=np.stack(cs));artifacts.append(path)
            records.append(dict(fold=f,fit_BCE=[bce(s,d['gap'][fit]) for s in fs],
                cal_BCE=[bce(s,d['gap'][cal]) for s in cs],base_C12_original_lambda_control=(f>0)))
            heads.append(models)
            del encoder,original_head,views,native;gc.collect();torch.cuda.empty_cache()
            print(json.dumps(dict(status='fold_heads_complete',fold=f,formal_solves=(f+1)*3)),flush=True)
    artifacts.append(OUT/'solutions.jsonl')
    write(OUT/'fit_completion.json',dict(status='all_15_heads_frozen_before_new_test_views_and_predictions',
        protocol_sha256=binding,formal_solves=15,encoder_fit_cal_query_forwards=38400,
        artifact_sha256={str(p):sha(p) for p in artifacts},completed_at_utc=datetime.now(timezone.utc).isoformat()))
    scores=np.full((3,9600),np.nan,np.float32);fold_id=np.full(9600,-1,int);test_artifacts=[]
    for f,(_,_,test) in enumerate(d['folds']):
        encoder,original_head=model(torch,f);views,native=infer(encoder,original_head,d['tokens'],test,torch)
        assert np.array_equal(native,d['H_native'][test])
        path=OUT/f'fold{f}_features_test.npz';save(path,test_indices=test,test_features=views,H_native=native);test_artifacts.append(path)
        for j,(m,x) in enumerate(zip(heads[f],[d['C12'][test],views[0],views[1]])):
            scores[j,test]=old.native(x,m['coef'],m['intercept'],torch)
        if f>0: assert np.array_equal(scores[0,test],d['C12_prior'][test])
        fold_id[test]=f
        del encoder,original_head,views,native;gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(status='test_predictions_complete',fold=f)),flush=True)
    assert np.isfinite(scores).all() and np.all(fold_id>=0)
    save(OUT/'predictions.npz',query_ids=d['query_ids'],group_ids=d['group_ids'],utility=d['utility'],
        arm_scores=scores,fold_id=fold_id,B6_scores=d['L_scores'],H_native_scores=d['H_native'])
    write(OUT/'predictions_frozen.json',dict(status='all_predictions_before_effects',protocol_sha256=binding,
        predictions_sha256=sha(OUT/'predictions.npz'),fit_completion_sha256=sha(OUT/'fit_completion.json'),
        test_feature_sha256={str(p):sha(p) for p in test_artifacts},encoder_test_query_forwards=9600))
    actions={arm:scores[j]>0 for j,arm in enumerate(ARMS)}
    actions.update(B6=d['L_scores']>0,H_native=d['H_native']>0,Dense=np.zeros(9600,bool),BM25=np.ones(9600,bool))
    u={name:np.where(a,d['utility'][:,0],d['utility'][:,1]) for name,a in actions.items()}
    values=np.column_stack([u['H6']-u['B6'],u['H12']-u['B12'],u['H6']-u['H12'],
        (u['H6']-u['H12'])-(u['B6']-u['B12']),u['H6']-u['Dense'],u['H6']-u['BM25']])
    _,inverse=np.unique(d['group_ids'],return_inverse=True);sizes=np.bincount(inverse)
    sums=np.column_stack([np.bincount(inverse,weights=values[:,j]) for j in range(6)])
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);draws=np.empty((20000,6))
    with threadpool_limits(limits=2):
        for start in range(0,20000,100):
            ix=rng.integers(len(sizes),size=(100,len(sizes)))
            draws[start:start+100]=sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)[:,None]
    save(OUT/'bootstrap.npz',draws=draws)
    ci=np.quantile(draws,CONFIG['quantiles'],axis=0).T
    primary={k:dict(mean=float(values[:,j].mean()),interval=ci[j].tolist()) for j,k in enumerate(PRIMARY)}
    recipe=primary['H6_minus_B6']['mean']>=.002 and primary['H6_minus_B6']['interval'][0]>0
    candidate=recipe and all(primary[k]['mean']>=.01 and primary[k]['interval'][0]>0 for k in PRIMARY[-2:])
    decision='PREPARE_H6_INDEPENDENT_CONFIRMATION' if candidate else ('H6_INTERNAL_INCREMENT_ONLY' if recipe else 'END_FIXED_H_CHECKPOINT_READOUT_NO_CONFIRMED_INCREMENT')
    for f,(_,_,test) in enumerate(d['folds']):
        records[f]['test_BCE']=[bce(s[test],d['gap'][test]) for s in scores]
        records[f]['primary_means']={k:float(values[test,j].mean()) for j,k in enumerate(PRIMARY)}
    result=dict(status='complete_trained_readout_pending_separate_check',protocol_sha256=binding,primary=primary,
        policy={k:old.policy_summary(actions[k],d['utility'],d['gap']) for k in POLICIES},folds=records,
        recipe_gate=recipe,candidate_preparation_gate=candidate,decision=decision,formal_solves=15,pilot_solves=3,
        new_encoder_fits=0,new_encoder_query_forwards=48000,pilot_encoder_query_forwards=192,new_api_calls=0,
        runtime_deployed=False,core_goal_achieved=False,predictions_frozen_sha256=sha(OUT/'predictions_frozen.json'),
        bootstrap_sha256=sha(OUT/'bootstrap.npz'),elapsed_seconds=time.perf_counter()-started,
        scope='Conditional development comparison; H supervised inside own outer-fit only; no inner CV, fresh validation or deployment')
    write(OUT/'results.json',result)
    print(json.dumps(dict(status=result['status'],decision=decision,primary=primary,elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','pilot','run']);parser.add_argument('--protocol-sha')
    args=parser.parse_args()
    try:
        if args.stage=='freeze':freeze()
        elif args.stage=='pilot':pilot(args.protocol_sha)
        else:run(args.protocol_sha)
    except Exception:
        traceback.print_exc();raise
