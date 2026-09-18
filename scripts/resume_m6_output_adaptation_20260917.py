"""Recover an interrupted fixed experiment without changing frozen artifacts.

Completed fold 0 is reused. Incomplete fold 1 A is replayed from initialization;
every existing checkpoint tensor and trace must match before training continues.
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
import psutil
import run_m6_output_adaptation as core

OUT=core.OUT
MANIFEST=OUT/'resume_manifest_20260917.json'
START=OUT/'resume_started_20260917.json'
VERIFIED=OUT/'resume_replay_verified_20260917.json'
COMPLETE=OUT/'resume_completion_20260917.json'
PLAN=core.ROOT/'analysis/hotpotqa_router/m6_output_adaptation_resume_plan_20260917.md'
BINDING='11237536b48159ff9071d2ab140e385f8adba172b5243c449ba0006af5bb35d6'
sha,read,write=core.sha,core.read,core.write


def require_hashes(mapping):
    for path,value in mapping.items():
        assert sha(path)==value,path


def old_process_absent():
    assert not psutil.pid_exists(7868), 'Old PID exists: inspect identity before recovery'
    others=[]
    for process in psutil.process_iter(['pid','name']):
        if process.pid==os.getpid() or 'python' not in (process.info['name'] or '').lower(): continue
        try:
            command=process.cmdline()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            raise AssertionError('Cannot inspect another Python process; do not duplicate work')
        if any('run_m6_output_adaptation' in part or 'resume_m6_output_adaptation' in part for part in command):
            others.append(process.pid)
    assert not others,others
    return dict(old_session=89863,old_session_observed_missing=True,old_pid=7868,
        old_pid_absent=True,other_training_processes=others)


def expected_prefix():
    names=['protocol.json','pilot_started.json','pilot.json','pilot_separate_checks.json','formal_started.json']
    names += [f'pilot_{a}{suffix}' for a in ('A','C') for suffix in ('.pt','.npz')]
    for arm in ('A','C'):
        names += [f'fold0_{arm}_epoch{e}.pt' for e in range(5)]
        names += [f'fold0_{arm}_epoch{e}_training.npz' for e in range(1,5)]
        names += [f'fold0_{arm}{suffix}' for suffix in ('.json','_fit_cal.npz')]
    names += [f'fold1_A_epoch{e}.pt' for e in range(4)]
    names += [f'fold1_A_epoch{e}_training.npz' for e in range(1,4)]
    return {OUT/name for name in names}


def freeze():
    assert not MANIFEST.exists()
    process=old_process_absent();core.bound(BINDING)
    actual={p for p in OUT.iterdir() if p.is_file()}
    assert actual==expected_prefix(),sorted(str(p.name) for p in actual^expected_prefix())
    assert not any((OUT/n).exists() for n in ('all_selected_frozen.json','predictions_frozen.json','results.json'))
    write(MANIFEST,dict(status='frozen_recovery_before_replay_or_new_training',protocol_sha256=BINDING,
        created_at_utc=datetime.now(timezone.utc).isoformat(),process_check=process,
        recovery_source_sha256={str(p):sha(p) for p in (Path(__file__).resolve(),PLAN)},
        existing_artifact_sha256={str(p):sha(p) for p in sorted(actual)},
        completed_trajectories_to_reuse=['fold0_A','fold0_C'],
        incomplete_trajectory='fold1_A',checkpoints_to_replay=[0,1,2,3],trace_epochs_to_replay=[1,2,3],
        replayed_committed_optimizer_steps=2304,
        previous_unlogged_partial_epoch_steps_bounds=[0,768],
        new_scientific_conditions=0,changed_scientific_rules=False,
        original_files_to_overwrite=0,new_api_calls=0))
    print(json.dumps(dict(status='recovery_frozen',manifest_sha256=sha(MANIFEST))),flush=True)


def run(manifest_sha):
    assert sha(MANIFEST)==manifest_sha
    manifest=read(MANIFEST);require_hashes(manifest['recovery_source_sha256'])
    require_hashes(manifest['existing_artifact_sha256']);old_process_absent();core.bound(BINDING)
    assert not START.exists() and not VERIFIED.exists() and not COMPLETE.exists()
    write(START,dict(protocol_sha256=BINDING,resume_manifest_sha256=manifest_sha,pid=os.getpid(),
        started_at_utc=datetime.now(timezone.utc).isoformat()))
    path=core.ROOT/'scripts/run_m6_output_adaptation_formal.py'
    spec=importlib.util.spec_from_file_location('_fixed_M6_formal_recovery',path)
    formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal)
    proxy=SimpleNamespace(**vars(core));formal.core=proxy
    reused=[];replayed_checkpoints=[];replayed_traces=[];started=time.perf_counter()
    journal_path=OUT/'resume_journal_20260917.jsonl'
    journal=journal_path.open('x',encoding='utf-8',newline='\n')

    def event(value):
        journal.write(json.dumps(value,allow_nan=False)+'\n');journal.flush()
        print(json.dumps(value,allow_nan=False),flush=True)

    def same_archive(path,**arrays):
        path=Path(path)
        if not path.exists(): return core.save(path,**arrays)
        assert str(path) in manifest['existing_artifact_sha256']
        assert path.name in {f'fold1_A_epoch{e}_training.npz' for e in range(1,4)}
        assert sha(path)==manifest['existing_artifact_sha256'][str(path)]
        with np.load(path,allow_pickle=False) as old:
            assert set(old.files)==set(arrays)
            for key,value in arrays.items():
                assert old[key].dtype==value.dtype and old[key].shape==value.shape
                assert np.array_equal(old[key],value),(path.name,key)
        replayed_traces.append(path.name)
        event(dict(status='existing_training_trace_replayed_exact',file=path.name,steps=768))

    def same_checkpoint(path,encoder,head,arm,fold,epoch,binding):
        path=Path(path)
        if not path.exists(): return core.save_checkpoint(path,encoder,head,arm,fold,epoch,binding)
        import torch
        assert str(path) in manifest['existing_artifact_sha256'] and (fold,arm)==(1,'A') and epoch in range(4)
        assert sha(path)==manifest['existing_artifact_sha256'][str(path)]
        old=torch.load(path,map_location='cpu',weights_only=True)
        assert (old['fold'],old['arm'],old['epoch'],old['protocol_sha256'],old['core_sha256'])==(fold,arm,epoch,binding,sha(core.__file__))
        current=core.state(encoder,head,arm)
        assert list(current)==list(old['trained_parameters'])
        assert all(value.dtype==old['trained_parameters'][name].dtype and
            value.shape==old['trained_parameters'][name].shape and
            torch.equal(value,old['trained_parameters'][name]) for name,value in current.items())
        replayed_checkpoints.append(path.name)
        event(dict(status='existing_checkpoint_replayed_exact',file=path.name,tensors=len(current)))
        del old,current;gc.collect()

    def metadata():
        return dict(resume_manifest_sha256=manifest_sha,resume_source_sha256=sha(__file__),
            original_formal_started_sha256=manifest['existing_artifact_sha256'][str(OUT/'formal_started.json')],
            reused_trajectories=reused,replayed_checkpoint_files=replayed_checkpoints,
            replayed_trace_files=replayed_traces,replayed_committed_optimizer_steps=2304,
            original_unlogged_partial_epoch_steps_bounds=[0,768],
            logical_protocol_optimizer_steps=30720,resumed_optimizer_steps=24576,
            total_recorded_optimizer_steps_across_attempts=33024,
            logical_protocol_encoder_query_forwards=649040,resumed_encoder_query_forwards=523088,
            replayed_committed_encoder_query_forwards=49152,
            total_recorded_encoder_query_forwards_across_attempts=698192,
            original_unlogged_encoder_query_forwards_bounds=[0,6144],
            scientific_rule_changes=False,original_file_changes=False)

    def record(path,value):
        path=Path(path)
        if path==OUT/'formal_started.json':
            assert sha(path)==manifest['existing_artifact_sha256'][str(path)]
            old=read(path)
            for key in ('protocol_sha256','pilot_sha256','pilot_separate_checks_sha256'):
                assert old[key]==value[key]
            event(dict(status='original_start_record_preserved',actual_resume_pid=os.getpid()))
            return
        if path==OUT/'all_selected_frozen.json':
            assert reused==['fold0_A','fold0_C']
            assert replayed_checkpoints==[f'fold1_A_epoch{e}.pt' for e in range(4)]
            assert replayed_traces==[f'fold1_A_epoch{e}_training.npz' for e in range(1,4)]
            require_hashes(manifest['existing_artifact_sha256'])
            verification=dict(status='completed_prefix_reused_and_incomplete_prefix_replayed_exact_before_test',
                protocol_sha256=BINDING,**metadata(),existing_artifact_sha256=manifest['existing_artifact_sha256'],
                resume_manifest_path=str(MANIFEST),resume_started_sha256=sha(START),
                completed_at_utc=datetime.now(timezone.utc).isoformat())
            write(VERIFIED,verification)
            value=dict(value,execution_resume=dict(**metadata(),resume_replay_verified_sha256=sha(VERIFIED)),
                elapsed_seconds_scope='resumed driver only; excludes original completed-fold time, includes deterministic replay')
        if path==OUT/'results.json':
            assert VERIFIED.exists()
            value=dict(value,execution_resume=dict(**metadata(),resume_replay_verified_sha256=sha(VERIFIED)),
                elapsed_seconds_scope='resumed driver only; excludes original completed-fold time, includes deterministic replay')
        return write(path,value)

    original_train=formal._train_arm
    def train_or_reuse(d,fold,arm,binding,torch,artifacts):
        if fold!=0: return original_train(d,fold,arm,binding,torch,artifacts)
        path=OUT/f'fold{fold}_{arm}.json';row=read(path)
        assert row['fold']==fold and row['arm']==arm and row['epochs_completed']==4 and row['optimizer_steps']==3072
        checks={str(path):manifest['existing_artifact_sha256'][str(path)],row['fit_cal_path']:row['fit_cal_sha256']}
        checks.update({r['checkpoint']:r['checkpoint_sha256'] for r in row['candidates']})
        checks.update({r['trace_path']:r['trace_sha256'] for r in row['epoch_training']})
        assert len(checks)==11
        require_hashes(checks)
        for name,digest in checks.items(): assert manifest['existing_artifact_sha256'][name]==digest
        with np.load(row['fit_cal_path'],allow_pickle=False) as cache:
            fit,cal,_=d['folds'][fold]
            assert np.array_equal(cache['fit_indices'],fit) and np.array_equal(cache['cal_indices'],cal)
            gains=[float(np.where(s>0,d['gap'][cal],0.).mean()) for s in cache['cal_logits']]
        assert row['selection']==formal.choose_epoch(gains,core.CONFIG['cal_tie_atol'])
        selected=row['selection']['selected_epoch'];chosen=row['candidates'][selected]
        assert row['selected_checkpoint']==chosen['checkpoint'] and row['selected_checkpoint_sha256']==chosen['checkpoint_sha256']
        artifacts.update(checks);reused.append(f'fold{fold}_{arm}')
        event(dict(status='completed_trajectory_reused',fold=fold,arm=arm,optimizer_steps_reused=3072))
        return row,dict(fold=fold,arm=arm,epoch=selected,checkpoint=chosen['checkpoint'],
            checkpoint_sha256=chosen['checkpoint_sha256'],record_path=str(path),record_sha256=checks[str(path)])

    proxy.save=same_archive;proxy.save_checkpoint=same_checkpoint;proxy.write=record
    formal._train_arm=train_or_reuse
    try:
        formal.run(BINDING)
        require_hashes(manifest['existing_artifact_sha256']);require_hashes(manifest['recovery_source_sha256'])
        write(COMPLETE,dict(status='resumed_fixed_experiment_complete_pending_independent_checks',
            protocol_sha256=BINDING,**metadata(),results_sha256=sha(OUT/'results.json'),
            resume_replay_verified_sha256=sha(VERIFIED),
            resume_elapsed_seconds=time.perf_counter()-started,completed_at_utc=datetime.now(timezone.utc).isoformat()))
        event(dict(status='resumed_formal_complete',results_sha256=sha(OUT/'results.json')))
    except Exception:
        failure=OUT/'resume_failure_20260917.json'
        if not failure.exists():
            write(failure,dict(status='recovery_failed_preserve_evidence',traceback=traceback.format_exc(),
                protocol_sha256=BINDING,resume_manifest_sha256=manifest_sha))
        raise
    finally:
        journal.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','run']);parser.add_argument('--manifest-sha256')
    args=parser.parse_args()
    if args.mode=='freeze': freeze()
    else: run(args.manifest_sha256)
