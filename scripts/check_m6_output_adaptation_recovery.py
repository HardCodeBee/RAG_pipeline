"""CPU-only provenance audit for the explicitly frozen 2026-09-17 recovery.

This checks preserved artifacts and executed equality-gate evidence. It does
not replay training. The separately bound formal checker validates the model
artifacts, predictions and scientific calculations, including sampled GPU work.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_output_adaptation_v1'
MANIFEST = OUT / 'resume_manifest_20260917.json'
START = OUT / 'resume_started_20260917.json'
VERIFIED = OUT / 'resume_replay_verified_20260917.json'
COMPLETE = OUT / 'resume_completion_20260917.json'
JOURNAL = OUT / 'resume_journal_20260917.jsonl'
CHECKER_READY = OUT / 'formal_checker_ready_20260917.json'
WRAPPER = ROOT / 'scripts/resume_m6_output_adaptation_20260917.py'
PLAN = ROOT / 'analysis/hotpotqa_router/m6_output_adaptation_resume_plan_20260917.md'
PROTOCOL_SHA = '11237536b48159ff9071d2ab140e385f8adba172b5243c449ba0006af5bb35d6'
MANIFEST_SHA = '786cf18fd03ac25ecf6d683136b69b84616e4b11628761da268a30006f1d66f4'
WRAPPER_SHA = 'a4455a93af75be4d0513c08749db16a39f1e87b41f3f2fdc09b5961bbd3d71c6'
FORMAL_SHA = '070a9dc2ad4188f3127f62860501e08e4315691da6c5d20884dd622f6f25bf8d'
FORMAL_CHECKER_SHA = 'ca5d8dfc2485b072df8a9290b25613a5bc42e87b849e86386d0f354dad868404'
CHECKER_READY_SHA = '0f3554130670f5608656e9fc380c65bae99fa5d78a19d23d58b076276f022ee1'
ELAPSED_SCOPE = 'resumed driver only; excludes original completed-fold time, includes deterministic replay'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(mapping):
    result = {str(Path(path).resolve()): value for path, value in mapping.items()}
    assert len(result) == len(mapping)
    return result


def bindings(mapping, expected_paths=None):
    assert isinstance(mapping, dict) and mapping
    if expected_paths is not None:
        assert set(canonical(mapping)) == {str(Path(p).resolve()) for p in expected_paths}
    for path, expected in mapping.items():
        assert isinstance(expected, str) and len(expected) == 64
        assert sha(path) == expected, path


def historical_paths():
    names = ['protocol.json', 'pilot_started.json', 'pilot.json',
             'pilot_separate_checks.json', 'formal_started.json']
    names += [f'pilot_{arm}{suffix}' for arm in ('A', 'C') for suffix in ('.pt', '.npz')]
    for arm in ('A', 'C'):
        names += [f'fold0_{arm}_epoch{epoch}.pt' for epoch in range(5)]
        names += [f'fold0_{arm}_epoch{epoch}_training.npz' for epoch in range(1, 5)]
        names += [f'fold0_{arm}{suffix}' for suffix in ('.json', '_fit_cal.npz')]
    names += [f'fold1_A_epoch{epoch}.pt' for epoch in range(4)]
    names += [f'fold1_A_epoch{epoch}_training.npz' for epoch in range(1, 4)]
    assert len(names) == len(set(names)) == 38
    return [OUT / name for name in names]


def resources():
    """Derive physical accounting from committed prefixes, not wrapper totals."""
    folds, arms, epochs, fit, cal, test, batch = 5, 2, 4, 6144, 1536, 1920, 8
    trajectories, reused, partial_epochs = folds * arms, 2, 3
    steps_per_epoch = fit // batch
    trajectory_steps = epochs * steps_per_epoch
    trajectory_forwards = epochs * fit + (epochs + 1) * (fit + cal)
    test_and_restore = trajectories * (test + 8)
    logical_steps = trajectories * trajectory_steps
    logical_forwards = trajectories * trajectory_forwards + test_and_restore
    replay_steps = partial_epochs * steps_per_epoch
    replay_forwards = partial_epochs * fit + (partial_epochs + 1) * (fit + cal)
    resumed_steps = (trajectories - reused) * trajectory_steps
    resumed_forwards = (trajectories - reused) * trajectory_forwards + test_and_restore
    previous_steps = reused * trajectory_steps + replay_steps
    previous_forwards = reused * trajectory_forwards + replay_forwards
    return dict(replayed_committed_optimizer_steps=replay_steps,
        original_unlogged_partial_epoch_steps_bounds=[0, steps_per_epoch],
        logical_protocol_optimizer_steps=logical_steps,
        resumed_optimizer_steps=resumed_steps,
        total_recorded_optimizer_steps_across_attempts=previous_steps + resumed_steps,
        logical_protocol_encoder_query_forwards=logical_forwards,
        resumed_encoder_query_forwards=resumed_forwards,
        replayed_committed_encoder_query_forwards=replay_forwards,
        total_recorded_encoder_query_forwards_across_attempts=previous_forwards + resumed_forwards,
        original_unlogged_encoder_query_forwards_bounds=[0, fit])


def expected_events(pid, result_sha=None):
    rows = [dict(status='original_start_record_preserved', actual_resume_pid=pid)]
    rows += [dict(status='completed_trajectory_reused', fold=0, arm=arm,
                  optimizer_steps_reused=3072) for arm in ('A', 'C')]
    rows.append(dict(status='existing_checkpoint_replayed_exact', file='fold1_A_epoch0.pt', tensors=103))
    for epoch in range(1, 4):
        rows.append(dict(status='existing_training_trace_replayed_exact',
                         file=f'fold1_A_epoch{epoch}_training.npz', steps=768))
        rows.append(dict(status='existing_checkpoint_replayed_exact',
                         file=f'fold1_A_epoch{epoch}.pt', tensors=103))
    if result_sha is not None:
        rows.append(dict(status='resumed_formal_complete', results_sha256=result_sha))
    return rows


def validate_events(rows, pid, complete=False, result_sha=None):
    wanted = expected_events(pid, result_sha if complete else None)
    if complete:
        assert rows == wanted, 'Complete journal event order/content differs'
    else:
        # A read-only contract check can run during training. Do not infer
        # process liveness or completed recovery from a valid event prefix.
        assert len(rows) <= len(wanted) and rows == wanted[:len(rows)]
    return rows


def journal_check(path, pid, complete=False, result_sha=None):
    text = Path(path).read_text(encoding='utf-8')
    assert text.endswith('\n'), 'Journal read caught a partial line; retry this audit'
    rows = [json.loads(line) for line in text.splitlines()]
    return validate_events(rows, pid, complete, result_sha)


def source_contract():
    """Pin reviewed sources and inspect the narrow interception structure."""
    assert sha(WRAPPER) == WRAPPER_SHA
    formal_path = ROOT / 'scripts/run_m6_output_adaptation_formal.py'
    assert sha(formal_path) == FORMAL_SHA
    tree = ast.parse(WRAPPER.read_text(encoding='utf-8'))
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name, target in (('same_archive', 'core.save'), ('same_checkpoint', 'core.save_checkpoint')):
        fn = funcs[name]
        guard = fn.body[1]
        assert isinstance(guard, ast.If) and ast.unparse(guard.test) == 'not path.exists()'
        assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Return)
        assert ast.unparse(guard.body[0].value.func) == target
        writes = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) in ('core.save', 'core.save_checkpoint')]
        assert writes == [guard.body[0].value]
        assert not any(isinstance(n, ast.Call) and ast.unparse(n.func).endswith('.open')
                       for n in ast.walk(fn))
    cp_text = ast.unparse(funcs['same_checkpoint'])
    array_text = ast.unparse(funcs['same_archive'])
    for fragment in ('value.dtype ==', 'value.shape ==', 'torch.equal(',
                     "manifest['existing_artifact_sha256']"):
        assert fragment in cp_text
    for fragment in ('old[key].dtype == value.dtype', 'old[key].shape == value.shape',
                     'np.array_equal(old[key], value)'):
        assert fragment in array_text
    record = funcs['record']
    original_start_branch = record.body[1]
    assert isinstance(original_start_branch, ast.If)
    assert ast.unparse(original_start_branch.test) == "path == OUT / 'formal_started.json'"
    assert isinstance(original_start_branch.body[-1], ast.Return)
    assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'write'
                   for n in ast.walk(original_start_branch))
    code = ast.unparse(tree)
    for fragment in ('proxy = SimpleNamespace(**vars(core))', 'formal.core = proxy',
                     'proxy.save = same_archive', 'proxy.save_checkpoint = same_checkpoint',
                     'proxy.write = record', 'original_train = formal._train_arm',
                     'formal._train_arm = train_or_reuse'):
        assert fragment in code
    assert ast.unparse(funcs['train_or_reuse'].body[0].test) == 'fold != 0'
    assert ast.unparse(funcs['train_or_reuse'].body[0].body[0].value.func) == 'original_train'
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert ast.unparse(node.func) not in ('exec', 'eval', 'setattr')
        if isinstance(node, ast.Assign):
            assert not any(ast.unparse(t).startswith('core.') for t in node.targets)
    # All verification-record writes precede the intercepted selection freeze.
    rec_code = ast.unparse(record)
    assert rec_code.index('write(VERIFIED, verification)') < rec_code.rindex('return write(path, value)')
    fixed = ast.parse(formal_path.read_text(encoding='utf-8'))
    fixed_funcs = {n.name: n for n in fixed.body if isinstance(n, ast.FunctionDef)}
    training = ast.unparse(fixed_funcs['_train_arm'])
    assert training.index('optimizer = core.optimizer_for(') < training.index('for epoch in range(')
    assert training.count('optimizer = core.optimizer_for(') == 1
    assert training.index('core.train_epoch(') < training.index('_candidate(')
    runner = ast.unparse(fixed_funcs['run'])
    assert runner.index('core.write(frozen_path,') < runner.index('for item in selected:')
    assert runner.index('core.write(prediction_freeze,') < runner.index('group_bootstrap(')
    return dict(reviewed_wrapper_sha256=WRAPPER_SHA, frozen_formal_runner_sha256=FORMAL_SHA,
        preserved_files_have_compare_only_interceptors=True,
        original_start_record_write_intercepted=True,
        continuous_original_optimizer_loop=True,
        selection_freeze_precedes_test_in_unchanged_runner=True,
        audit_kind='Exact reviewed source binding plus critical AST structure; not a general program proof')


def reused_records(manifest):
    old = canonical(manifest['existing_artifact_sha256'])
    reports = []
    for arm in ('A', 'C'):
        path = OUT / f'fold0_{arm}.json'
        row = read(path)
        assert (row['fold'], row['arm'], row['epochs_completed'], row['optimizer_steps']) == (0, arm, 4, 3072)
        assert row['outer_test_quality_evaluations'] == 0
        assert len(row['candidates']) == 5 and len(row['epoch_training']) == 4
        artifact = {str(path): sha(path), row['fit_cal_path']: row['fit_cal_sha256']}
        for epoch, candidate in enumerate(row['candidates']):
            assert candidate['epoch'] == epoch
            assert Path(candidate['checkpoint']).resolve() == (OUT / f'fold0_{arm}_epoch{epoch}.pt').resolve()
            artifact[candidate['checkpoint']] = candidate['checkpoint_sha256']
        for epoch, trace in enumerate(row['epoch_training'], 1):
            assert (trace['fold'], trace['arm'], trace['epoch'], trace['optimizer_steps']) == (0, arm, epoch, 768)
            assert Path(trace['trace_path']).resolve() == (OUT / f'fold0_{arm}_epoch{epoch}_training.npz').resolve()
            artifact[trace['trace_path']] = trace['trace_sha256']
        assert len(artifact) == 11
        assert all(old[name] == digest for name, digest in canonical(artifact).items())
        chosen = row['candidates'][row['selection']['selected_epoch']]
        assert (row['selected_checkpoint'], row['selected_checkpoint_sha256']) == (chosen['checkpoint'], chosen['checkpoint_sha256'])
        reports.append(dict(fold=0, arm=arm, original_record_sha256=sha(path),
            preserved_artifacts=11, selected_epoch=row['selection']['selected_epoch']))
    return reports


def contract():
    assert sha(MANIFEST) == MANIFEST_SHA and sha(OUT / 'protocol.json') == PROTOCOL_SHA
    manifest, protocol, start = read(MANIFEST), read(OUT / 'protocol.json'), read(START)
    assert manifest['status'] == 'frozen_recovery_before_replay_or_new_training'
    assert manifest['protocol_sha256'] == start['protocol_sha256'] == PROTOCOL_SHA
    bindings(manifest['existing_artifact_sha256'], historical_paths())
    bindings(manifest['recovery_source_sha256'], [WRAPPER, PLAN])
    assert len(protocol['source_sha256']) == 6 and len(protocol['input_sha256']) == 19
    bindings(protocol['source_sha256']); bindings(protocol['input_sha256'])
    assert manifest['process_check'] == dict(old_session=89863, old_session_observed_missing=True,
        old_pid=7868, old_pid_absent=True, other_training_processes=[])
    assert manifest['completed_trajectories_to_reuse'] == ['fold0_A', 'fold0_C']
    assert manifest['incomplete_trajectory'] == 'fold1_A'
    assert manifest['checkpoints_to_replay'] == [0, 1, 2, 3] and manifest['trace_epochs_to_replay'] == [1, 2, 3]
    assert manifest['replayed_committed_optimizer_steps'] == 3 * 6144 // 8
    assert manifest['previous_unlogged_partial_epoch_steps_bounds'] == [0, 6144 // 8]
    assert manifest['changed_scientific_rules'] is False
    assert manifest['new_scientific_conditions'] == manifest['original_files_to_overwrite'] == manifest['new_api_calls'] == 0
    assert set(start) == {'protocol_sha256', 'resume_manifest_sha256', 'pid', 'started_at_utc'}
    assert start['resume_manifest_sha256'] == MANIFEST_SHA and start['pid'] == 21488
    original_start = read(OUT / 'formal_started.json')
    assert original_start['pid'] == 7868 and original_start['protocol_sha256'] == PROTOCOL_SHA
    assert datetime.fromisoformat(original_start['started_at_utc']) < datetime.fromisoformat(manifest['created_at_utc']) <= datetime.fromisoformat(start['started_at_utc'])
    assert sha(CHECKER_READY) == CHECKER_READY_SHA
    ready = read(CHECKER_READY)
    assert ready['status'] == 'formal_checker_sources_fixed_after_readonly_review_before_complete_new_effects'
    assert ready['mustfix_remaining'] == ready['expected_optimizer_steps'] == 0
    assert ready['expected_GPU_queries'] == 2320 and ready['expected_backward_passes'] == 40
    assert ready['full_training_replay'] is False
    bindings(ready['source_sha256'], [ROOT / 'scripts' / name for name in (
        'check_m6_output_adaptation_formal.py', 'check_m6_output_adaptation_pilot.py',
        'check_m6_student_transfer.py')])
    assert ready['source_sha256'][str(ROOT / 'scripts/check_m6_output_adaptation_formal.py')] == FORMAL_CHECKER_SHA
    assert datetime.fromisoformat(start['started_at_utc']) <= datetime.fromisoformat(ready['created_at_utc'])
    source = source_contract()
    preserved = reused_records(manifest)
    return manifest, protocol, start, source, preserved


def expected_metadata(manifest):
    return dict(resume_manifest_sha256=MANIFEST_SHA, resume_source_sha256=WRAPPER_SHA,
        original_formal_started_sha256=manifest['existing_artifact_sha256'][str(OUT / 'formal_started.json')],
        reused_trajectories=['fold0_A', 'fold0_C'],
        replayed_checkpoint_files=[f'fold1_A_epoch{e}.pt' for e in range(4)],
        replayed_trace_files=[f'fold1_A_epoch{e}_training.npz' for e in range(1, 4)],
        **resources(), scientific_rule_changes=False, original_file_changes=False)


def check():
    started = time.perf_counter()
    destination = OUT / 'recovery_separate_checks_20260917.json'
    assert not destination.exists(), 'Preserve an existing recovery audit receipt'
    # Deliberately require the full formal numerical check before this receipt.
    numerical = read(OUT / 'separate_checks.json')
    assert numerical['status'] == 'passed_independent_M6_output_adaptation_formal_checks'
    manifest, protocol, start, source, preserved = contract()
    assert not (OUT / 'resume_failure_20260917.json').exists()
    result = read(OUT / 'results.json')
    result_sha = sha(OUT / 'results.json')
    completion, verified = read(COMPLETE), read(VERIFIED)
    selected, predictions = read(OUT / 'all_selected_frozen.json'), read(OUT / 'predictions_frozen.json')
    metadata = expected_metadata(manifest)
    for item in (verified, completion):
        assert item['protocol_sha256'] == PROTOCOL_SHA
        assert {key: item[key] for key in metadata} == metadata
    assert verified['status'] == 'completed_prefix_reused_and_incomplete_prefix_replayed_exact_before_test'
    assert verified['existing_artifact_sha256'] == manifest['existing_artifact_sha256']
    assert Path(verified['resume_manifest_path']).resolve() == MANIFEST.resolve()
    assert verified['resume_started_sha256'] == sha(START)
    expected_resume = dict(**metadata, resume_replay_verified_sha256=sha(VERIFIED))
    assert selected['execution_resume'] == result['execution_resume'] == expected_resume
    assert selected['elapsed_seconds_scope'] == result['elapsed_seconds_scope'] == ELAPSED_SCOPE
    assert selected['protocol_sha256'] == predictions['protocol_sha256'] == result['protocol_sha256'] == PROTOCOL_SHA
    assert selected['status'] == 'all_ten_selected_endpoints_before_any_formal_test'
    assert selected['test_encoder_query_forwards'] == 0 and selected['optimizer_steps'] == resources()['logical_protocol_optimizer_steps']
    assert result['encoder_query_forwards']['total'] == resources()['logical_protocol_encoder_query_forwards']
    assert predictions['all_selected_frozen_sha256'] == result['all_selected_frozen_sha256'] == sha(OUT / 'all_selected_frozen.json')
    assert predictions['predictions_sha256'] == sha(OUT / 'predictions.npz')
    assert result['predictions_frozen_sha256'] == sha(OUT / 'predictions_frozen.json')
    assert completion['status'] == 'resumed_fixed_experiment_complete_pending_independent_checks'
    assert completion['results_sha256'] == result_sha and completion['resume_replay_verified_sha256'] == sha(VERIFIED)
    assert math.isfinite(completion['resume_elapsed_seconds']) and completion['resume_elapsed_seconds'] > 0
    assert datetime.fromisoformat(start['started_at_utc']) <= datetime.fromisoformat(verified['completed_at_utc']) <= datetime.fromisoformat(completion['completed_at_utc']) <= datetime.fromisoformat(numerical['completed_at_utc'])
    ready = read(CHECKER_READY)
    assert datetime.fromisoformat(ready['created_at_utc']) <= datetime.fromisoformat(verified['completed_at_utc'])
    rows = journal_check(JOURNAL, start['pid'], complete=True, result_sha=result_sha)
    assert numerical['protocol_sha256'] == PROTOCOL_SHA and numerical['results_sha256'] == result_sha
    assert numerical['predictions_sha256'] == sha(OUT / 'predictions.npz')
    assert numerical['checker_sha256'] == FORMAL_CHECKER_SHA == sha(ROOT / 'scripts/check_m6_output_adaptation_formal.py')
    assert numerical['checker_source_sha256'] == ready['source_sha256']
    assert numerical['source_sha256'] == protocol['source_sha256'] and numerical['input_sha256'] == protocol['input_sha256']
    bindings(numerical['checker_source_sha256']); bindings(numerical['artifact_sha256'])
    assert numerical['checkpoint_count'] == 50 and numerical['trace_archive_count'] == 40 and numerical['trace_step_count'] == 30720
    assert numerical['independent_encoder_query_forwards'] == 2320 and numerical['independent_backward_passes'] == 40 and numerical['independent_optimizer_steps'] == 0
    assert numerical['full_formal_training_replayed'] is False
    assert numerical['scientific_rule_changes'] is numerical['numerical_tolerance_changes'] is numerical['storage_dtype_changes'] is False
    # Retained fold0 record payloads appear unchanged inside the full result;
    # the three added fields describe the later test inference only.
    for arm in ('A', 'C'):
        original = read(OUT / f'fold0_{arm}.json')
        reported = result['folds'][0]['arms'][arm]
        assert {key: reported[key] for key in original} == original
        assert set(reported) - set(original) == {'selected_fit8_restore_exact', 'test_archive', 'selected_test_BCE'}
    bound_files = dict(manifest['existing_artifact_sha256'])
    bound_files.update(manifest['recovery_source_sha256'])
    bound_files.update({str(path): sha(path) for path in (MANIFEST, START, VERIFIED, COMPLETE, JOURNAL, CHECKER_READY,
        OUT / 'all_selected_frozen.json', OUT / 'predictions_frozen.json', OUT / 'predictions.npz',
        OUT / 'results.json', OUT / 'separate_checks.json')})
    checker_sha = sha(__file__)
    bindings(bound_files); bindings(protocol['source_sha256']); bindings(protocol['input_sha256'])
    assert sha(__file__) == checker_sha
    receipt = dict(status='passed_independent_M6_output_adaptation_recovery_provenance_checks',
        protocol_sha256=PROTOCOL_SHA, resume_manifest_sha256=MANIFEST_SHA,
        results_sha256=result_sha, predictions_sha256=sha(OUT / 'predictions.npz'),
        formal_separate_checks_sha256=sha(OUT / 'separate_checks.json'),
        formal_checker_ready_sha256=CHECKER_READY_SHA,
        checker_sha256=checker_sha, artifact_sha256=bound_files,
        original_protocol_source_sha256=protocol['source_sha256'], original_protocol_input_sha256=protocol['input_sha256'],
        old_artifacts_preserved=38, recovery_sources_bound=2, original_sources_bound=6,
        original_inputs_bound=19, completed_fold0_artifacts_preserved=22, reused_arms=preserved,
        journal_event_count=len(rows), journal_checkpoint_equality_gates=4, journal_trace_equality_gates=3,
        audited_source_contract=source, execution_metadata=metadata, independently_derived_resources=resources(),
        formal_checker_sha256=FORMAL_CHECKER_SHA,
        new_GPU_forwards=0, new_optimizer_steps=0, new_api_calls=0,
        original_files_changed=False, repeated_prefix_independently_retrained_by_this_checker=False,
        complete_policy_results_checked_by_separate_bound_checker=True,
        proof_boundary='Preserved bytes, frozen reviewed code, ordered successful equality-gate events, linked pre-test recovery/selection freezes, completion and separate numerical receipt. This checker does not itself replay the three training epochs or prove execution from logs alone.',
        incomplete_original_epoch_accounting='Committed-step and forward totals exclude up to one unlogged original training epoch; stated bounds do not include pilots or independent checker forwards.',
        core_goal_achieved=False, elapsed_seconds=time.perf_counter() - started,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt


def self_test():
    derived = resources()
    assert derived == dict(replayed_committed_optimizer_steps=2304,
        original_unlogged_partial_epoch_steps_bounds=[0, 768], logical_protocol_optimizer_steps=30720,
        resumed_optimizer_steps=24576, total_recorded_optimizer_steps_across_attempts=33024,
        logical_protocol_encoder_query_forwards=649040, resumed_encoder_query_forwards=523088,
        replayed_committed_encoder_query_forwards=49152,
        total_recorded_encoder_query_forwards_across_attempts=698192,
        original_unlogged_encoder_query_forwards_bounds=[0, 6144])
    assert derived['total_recorded_optimizer_steps_across_attempts'] == derived['logical_protocol_optimizer_steps'] + derived['replayed_committed_optimizer_steps']
    assert derived['total_recorded_encoder_query_forwards_across_attempts'] == derived['logical_protocol_encoder_query_forwards'] + derived['replayed_committed_encoder_query_forwards']
    events = expected_events(42, 'synthetic')
    assert len(events) == 11
    assert sum(e.get('steps', 0) for e in events) == 2304
    assert sum(e.get('tensors', 0) for e in events) == 4 * 103
    assert sum(e.get('optimizer_steps_reused', 0) for e in events) == 6144
    assert validate_events(events, 42, True, 'synthetic') == events
    for length in range(len(events)):
        assert validate_events(events[:length], 42) == events[:length]
    corruptions = [events[1:], events[:-2] + events[-1:],
                   events[:4] + [events[5], events[4]] + events[6:],
                   events[:4] + [dict(events[4], steps=767)] + events[5:],
                   events[:-1] + [dict(events[-1], results_sha256='changed')]]
    rejected = 0
    for damaged in corruptions:
        try:
            validate_events(damaged, 42, True, 'synthetic')
        except AssertionError:
            rejected += 1
    assert rejected == len(corruptions)
    return dict(status='passed_recovery_checker_synthetic_resource_and_event_checks',
        resources=derived, event_count=len(events), malformed_event_streams_rejected=rejected,
        real_data_reads=0, GPU_forwards=0, files_written=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('self-test', 'contract', 'check'))
    args = parser.parse_args()
    if args.mode == 'self-test':
        answer = self_test()
    elif args.mode == 'contract':
        manifest, protocol, start, source, preserved = contract()
        # Completed runs have a terminal event, so allow it only when its
        # declared result bytes exist and match. This is not the final audit.
        if COMPLETE.exists():
            events = journal_check(JOURNAL, start['pid'], True, sha(OUT / 'results.json'))
        else:
            events = journal_check(JOURNAL, start['pid'])
        answer = dict(status='passed_read_only_recovery_contract_checks', old_artifacts=38,
            source_contract=source, preserved_arms=preserved, observed_journal_events=len(events),
            process_liveness_asserted=False, recovery_completion_asserted=False,
            GPU_forwards=0, files_written=0)
    else:
        receipt = check()
        answer = {name: receipt[name] for name in ('status', 'old_artifacts_preserved',
            'journal_checkpoint_equality_gates', 'journal_trace_equality_gates', 'elapsed_seconds')}
    print(json.dumps(answer, ensure_ascii=False, allow_nan=False))
