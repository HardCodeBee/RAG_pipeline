"""Two explicit, lossless storage-dtype adapters for the frozen checker.

The original checker and first failed amendment remain unchanged. Only the
single int32 fold-ID assertion and five float64 historical H test-score inputs
are adapted. Original numerical checks, tolerances and scientific rules run.
"""
import argparse
import io
import json
from pathlib import Path
import re

import numpy as np

import check_m6_trained_readout_dtype_amended as first


ROOT, OUT = first.ROOT, first.OUT
FIRST = Path(first.__file__).resolve()
FIRST_SHA = 'f9964e0e27cdd0bb4c06c11a8324d10a14e41186706dd245dda62ea1b283535e'
SECOND_FAILURE_SHA = '1ce8da06378dd5f76c788e6bacc427d3fd56e5be6f723c3fa66ed58d1c0e3d35'
PASSED = 'passed_independent_trained_readout_checks_with_storage_dtype_amendments'
sha = first.sha


class StorageProxy(first.FoldDtypeProxy):
    def __init__(self, original, fold_ids, historical_test):
        super().__init__(original, fold_ids)
        assert set(historical_test) == set(range(5))
        self.historical_test = {f: np.asarray(values).copy() for f, values in historical_test.items()}
        assert all(v.shape == (1920,) and v.dtype == np.float64 for v in self.historical_test.values())
        self.H_test_calls = set()

    def native_score_bound(self, features, coef, intercept, scores, counters, name, probe=None):
        match = re.fullmatch(r'fold([0-4])_H_native/test', name)
        if match is None:
            return self.original.native_score_bound(features, coef, intercept, scores, counters, name, probe=probe)
        fold = int(match.group(1))
        assert fold not in self.H_test_calls, 'Each historical H test vector must occur once'
        values = np.asarray(scores)
        assert probe is None and features.shape == (1920, 384)
        assert values.dtype == np.float64 and np.array_equal(values, self.historical_test[fold])
        exact_float32 = values.astype(np.float32)
        assert np.array_equal(exact_float32.astype(np.float64), values), 'Reject lossy score conversion'
        self.H_test_calls.add(fold)
        return self.original.native_score_bound(features, coef, intercept, exact_float32, counters, name)


class ReceiptProxy:
    def __init__(self, storage, destination, amendment):
        self.storage, self.amendment = storage, amendment
        self.destination = Path(destination).resolve()
        self.calls = 0

    def __getattr__(self, name):
        return getattr(json, name)

    def dump(self, receipt, handle, *args, **kwargs):
        assert self.calls == 0 and Path(handle.name).resolve() == self.destination
        assert receipt['status'] == 'passed_independent_trained_readout_checks'
        assert receipt['checker_sha256'] == first.ORIGINAL_SHA
        assert self.storage.calls == self.storage.changed_calls == 1
        assert self.storage.H_test_calls == set(range(5))
        receipt['original_checker_passed_status'] = receipt['status']
        receipt['status'] = PASSED
        receipt['storage_dtype_amendments'] = dict(self.amendment,
            fold_id=dict(affected_calls=1, actual_dtype='int32', original_expected_dtype='int64',
                cast=False, exact_bound_archive_identity=True, original_fold_coverage_checks_unchanged=True),
            historical_H_native_test=dict(affected_calls=5, actual_storage_dtype='float64',
                helper_input_dtype='float32', all_9600_values_roundtrip_exact=True,
                bound_historical_current_and_fold_identity=True),
            numerical_tolerance_changes=False, scientific_rule_changes=False, stored_array_changes=False)
        self.calls += 1
        return json.dump(receipt, handle, *args, **kwargs)


def self_test():
    first_result = first.self_test()
    original = first.original_module().independent

    class FakeBound:
        def __getattr__(self, name):
            return getattr(original, name)

        def native_score_bound(self, features, coef, intercept, scores, counters, name, probe=None):
            assert scores.dtype == np.float32
            counters['calls'] = counters.get('calls', 0) + 1
            return scores

    fold_ids = np.arange(9600, dtype=np.int32) % 5
    references = {f: (np.arange(1920, dtype=np.float32) / 16 + f).astype(np.float64) for f in range(5)}
    proxy = StorageProxy(FakeBound(), fold_ids, references)
    proxy.real_array(fold_ids, (9600,), 'fold ID', np.int64)
    features = np.zeros((1920, 384), np.float32); coef = np.zeros(384); counters = {}
    for f in range(5):
        observed = proxy.native_score_bound(features, coef, 0., references[f], counters, f'fold{f}_H_native/test')
        assert np.array_equal(observed.astype(np.float64), references[f])
    assert counters['calls'] == 5 and proxy.H_test_calls == set(range(5))
    rejected = 0
    cases = ['repeat', 'lossy', 'different_values', 'wrong_dtype']
    for case in cases:
        current = StorageProxy(FakeBound(), fold_ids, references)
        values = references[0].copy()
        if case == 'repeat':
            current.H_test_calls.add(0)
        elif case == 'lossy':
            values[0] = .1
            current.historical_test[0] = values.copy()
        elif case == 'different_values':
            values[0] += 1
        else:
            values = values.astype(np.float32)
        try:
            current.native_score_bound(features, coef, 0., values, {}, 'fold0_H_native/test')
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('Invalid historical H score adaptation accepted: ' + case)
    stream = io.StringIO(); stream.name = str(OUT / 'separate_checks.json')
    receipt_proxy = ReceiptProxy(proxy, stream.name, dict(test_only=True))
    receipt = dict(status='passed_independent_trained_readout_checks', checker_sha256=first.ORIGINAL_SHA)
    receipt_proxy.dump(receipt, stream)
    assert receipt['status'] == json.loads(stream.getvalue())['status'] == PASSED
    return dict(status='passed_synthetic_two_storage_adapters_and_receipt_checks',
        fold_dtype_self_test=first_result['status'], score_roundtrip_cases=5,
        score_invalid_cases_rejected=rejected, amended_calls=[1, 5], new_fits=0, GPU_forwards=0, real_data_reads=0)


def check(protocol_sha):
    assert protocol_sha == first.PROTOCOL_SHA and sha(OUT / 'protocol.json') == protocol_sha
    assert sha(FIRST) == FIRST_SHA and sha(first.ORIGINAL) == first.ORIGINAL_SHA
    assert not (OUT / 'separate_checks.json').exists()
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    inputs = {Path(name).resolve(): value for name, value in protocol['input_sha256'].items()}
    failures = [(OUT / 'independent_check_failure_20260915_01.json', first.FAILURE_SHA),
                (OUT / 'independent_check_failure_20260915_02.json', SECOND_FAILURE_SHA)]
    for path, digest in failures:
        assert sha(path) == digest
        assert json.loads(path.read_text(encoding='utf-8'))['exit_code'] == 1
    predictions = OUT / 'predictions.npz'
    prediction_sha = sha(predictions)
    assert all(json.loads(path.read_text(encoding='utf-8'))['predictions_sha256'] == prediction_sha for path, _ in failures)
    historical_path = ROOT.parent / 'work/router_research/lp_ft_v2/predictions.npz'
    split_path = ROOT.parent / 'work/router_research/e02_results/fold_indices.npz'
    for path in (historical_path, split_path):
        assert path.resolve() in inputs and sha(path) == inputs[path.resolve()]
    with np.load(predictions, allow_pickle=False) as current, np.load(historical_path, allow_pickle=False) as historical:
        fold_ids = current['fold_id'].copy()
        H = current['H_native_scores'].copy()
        assert fold_ids.shape == (9600,) and fold_ids.dtype == np.int32
        assert H.shape == (9600,) and H.dtype == np.float64
        assert historical['H_native'].dtype == np.float64 and np.array_equal(H, historical['H_native'])
        assert np.array_equal(H.astype(np.float32).astype(np.float64), H)
        for key in ('query_ids', 'group_ids'):
            assert np.array_equal(current[key], historical[key])
    with np.load(split_path, allow_pickle=False) as archive:
        historical_test = {}
        for f in range(5):
            test = archive[f'fold{f}_test'].copy()
            assert test.shape == (1920,) and test.dtype.kind in 'iu'
            assert np.array_equal(np.flatnonzero(fold_ids == f), np.sort(test))
            historical_test[f] = H[test]
    module = first.original_module()
    storage = StorageProxy(module.independent, fold_ids, historical_test)
    amendment = dict(original_checker_path=str(first.ORIGINAL), original_checker_sha256=first.ORIGINAL_SHA,
        first_amended_entry_path=str(FIRST), first_amended_entry_sha256=FIRST_SHA,
        wrapper_path=str(Path(__file__).resolve()), wrapper_sha256=sha(Path(__file__)),
        failure_record_sha256={str(path): digest for path, digest in failures},
        protocol_sha256=protocol_sha, predictions_sha256=prediction_sha,
        historical_predictions_sha256=sha(historical_path), fold_indices_sha256=sha(split_path))
    receipt_proxy = ReceiptProxy(storage, OUT / 'separate_checks.json', amendment)
    module.independent = storage
    module.json = receipt_proxy
    receipt = module.check(OUT, protocol_sha)
    assert receipt['status'] == PASSED and receipt_proxy.calls == 1
    assert storage.calls == storage.changed_calls == 1 and storage.H_test_calls == set(range(5))
    assert sha(first.ORIGINAL) == first.ORIGINAL_SHA and sha(FIRST) == FIRST_SHA
    assert sha(Path(__file__)) == amendment['wrapper_sha256'] and sha(predictions) == prediction_sha
    assert all(sha(path) == digest for path, digest in failures)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--protocol-sha')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False))
    else:
        receipt = check(args.protocol_sha)
        print(json.dumps({key: receipt[key] for key in ('status', 'formal_heads_checked', 'pilot_heads_checked',
            'maximum_independent_gradient_inf', 'independent_H_encoder_query_forwards', 'elapsed_seconds')}, allow_nan=False))
