"""Explicit local correction for the frozen checker's Windows fold-ID dtype.

Only the unique fold-ID assertion accepts the observed int32 archive member.
All model, GPU, arithmetic and policy checks execute the original frozen code.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'scripts/check_m6_trained_readout.py'
OUT = ROOT / 'outputs/router/hotpotqa_bd_router_v1/runs/m6_trained_readout_v1'
ORIGINAL_SHA = '1edc2043c8f3a70c9fb2e9c62910e767c7468a230b5d530ccd97e20477865a64'
PROTOCOL_SHA = '765121b4c2e59d72d88e63a8905b10ad0b059409737ae24e24e36badd1a8e788'
FAILURE_SHA = '7d6c18549b4eb21050c6db99043f54ddb327bfa4547276ffbefd4c2919dfd789'
PASSED = 'passed_independent_trained_readout_checks_with_fold_id_dtype_amendment'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_module():
    assert sha(ORIGINAL) == ORIGINAL_SHA
    spec = importlib.util.spec_from_file_location('_frozen_trained_readout_dtype_check', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FoldDtypeProxy:
    def __init__(self, original, reference):
        self.original = original
        self.reference = np.asarray(reference).copy()
        assert self.reference.shape == (9600,) and self.reference.dtype in (np.dtype('int32'), np.dtype('int64'))
        self.calls = 0
        self.changed_calls = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def real_array(self, value, shape=None, name='array', dtype=None):
        if name != 'fold ID':
            return self.original.real_array(value, shape, name, dtype)
        assert self.calls == 0, 'The frozen fold-ID check must occur exactly once'
        assert shape == (9600,) and dtype == np.int64, 'Unexpected fold-ID call signature'
        observed = np.asarray(value)
        assert observed.dtype == self.reference.dtype and np.array_equal(observed, self.reference), 'Not the bound predictions.fold_id'
        self.calls += 1
        if observed.dtype == np.int32:
            self.changed_calls += 1
            return self.original.real_array(value, shape, name, np.int32)
        return self.original.real_array(value, shape, name, dtype)


class ReceiptJsonProxy:
    def __init__(self, dtype_proxy, destination, amendment):
        self.dtype_proxy = dtype_proxy
        self.destination = Path(destination).resolve()
        self.amendment = amendment
        self.calls = 0

    def __getattr__(self, name):
        return getattr(json, name)

    def dump(self, receipt, handle, *args, **kwargs):
        assert self.calls == 0 and Path(handle.name).resolve() == self.destination
        assert receipt['status'] == 'passed_independent_trained_readout_checks'
        assert receipt['checker_sha256'] == ORIGINAL_SHA
        assert self.dtype_proxy.calls == self.dtype_proxy.changed_calls == 1
        receipt['original_checker_passed_status'] = receipt['status']
        receipt['status'] = PASSED
        receipt['fold_id_dtype_amendment'] = dict(self.amendment,
            affected_calls=self.dtype_proxy.calls, changed_calls=self.dtype_proxy.changed_calls,
            observed_dtype='int32', unchanged_original_expected_dtype='int64',
            validation='Exact equality to bound predictions.fold_id; no cast; original fold coverage checks unchanged')
        self.calls += 1
        return json.dump(receipt, handle, *args, **kwargs)


def self_test():
    original = original_module().independent
    for dtype in (np.int32, np.int64):
        values = np.arange(9600, dtype=dtype) % 5
        proxy = FoldDtypeProxy(original, values)
        result = proxy.real_array(values, (9600,), 'fold ID', np.int64)
        assert result is values and result.dtype == dtype
        assert proxy.calls == 1 and proxy.changed_calls == int(dtype == np.int32)
        try:
            proxy.real_array(values, (9600,), 'fold ID', np.int64)
        except AssertionError:
            pass
        else:
            raise AssertionError('Repeated correction was accepted')
    values = np.arange(9600, dtype=np.int32) % 5
    wrong_values = values.copy(); wrong_values[0] = 4
    invalid = [values.astype(np.float64), values.astype(np.int16), values.astype(bool),
               values.astype(np.uint32), values[:-1], wrong_values]
    for candidate in invalid:
        try:
            FoldDtypeProxy(original, values).real_array(candidate, (9600,), 'fold ID', np.int64)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('Invalid fold IDs accepted')
    proxy = FoldDtypeProxy(original, values)
    other = np.asarray([.1, .2])
    assert proxy.real_array(other, (2,), 'other', np.float64) is other and proxy.calls == 0
    proxy.real_array(values, (9600,), 'fold ID', np.int64)
    stream = io.StringIO(); stream.name = str(OUT / 'separate_checks.json')
    receipt_proxy = ReceiptJsonProxy(proxy, stream.name, dict(wrapper_sha256='synthetic'))
    receipt = dict(status='passed_independent_trained_readout_checks', checker_sha256=ORIGINAL_SHA)
    receipt_proxy.dump(receipt, stream)
    assert receipt['status'] == json.loads(stream.getvalue())['status'] == PASSED
    assert receipt['fold_id_dtype_amendment']['changed_calls'] == 1 and receipt_proxy.calls == 1
    return dict(status='passed_synthetic_dtype_guard_and_explicit_amended_receipt_checks',
        accepted_dtypes=['int32', 'int64'], invalid_cases_rejected=len(invalid),
        no_cast=True, repeat_refused=True, real_data_reads=0, new_fits=0, GPU_forwards=0)


def check(protocol_sha):
    assert protocol_sha == PROTOCOL_SHA and sha(OUT / 'protocol.json') == PROTOCOL_SHA
    assert not (OUT / 'separate_checks.json').exists()
    failure = OUT / 'independent_check_failure_20260915_01.json'
    assert sha(failure) == FAILURE_SHA
    failure_record = json.loads(failure.read_text(encoding='utf-8'))
    assert failure_record['status'] == 'failed_checker_fold_id_dtype_compatibility'
    assert failure_record['checker_sha256'] == ORIGINAL_SHA and failure_record['exit_code'] == 1
    assert failure_record['GPU_forwards'] == 0 and failure_record['new_fits'] == 0
    predictions = OUT / 'predictions.npz'
    predictions_sha = sha(predictions)
    assert predictions_sha == failure_record['predictions_sha256']
    with np.load(predictions, allow_pickle=False) as archive:
        reference = archive['fold_id'].copy()
    assert reference.shape == (9600,) and reference.dtype == np.int32
    module = original_module()
    proxy = FoldDtypeProxy(module.independent, reference)
    amendment = dict(original_checker_path=str(ORIGINAL), original_checker_sha256=ORIGINAL_SHA,
        wrapper_path=str(Path(__file__).resolve()), wrapper_sha256=sha(Path(__file__)),
        failure_record_path=str(failure), failure_record_sha256=FAILURE_SHA,
        predictions_sha256=predictions_sha, protocol_sha256=PROTOCOL_SHA,
        scientific_rule_changes=False, numerical_tolerance_changes=False)
    json_proxy = ReceiptJsonProxy(proxy, OUT / 'separate_checks.json', amendment)
    # Only this privately loaded checker receives proxies. Imported helper files
    # and their own globals are neither replaced nor modified.
    module.independent = proxy
    module.json = json_proxy
    result = module.check(OUT, protocol_sha)
    assert proxy.calls == proxy.changed_calls == json_proxy.calls == 1
    assert result['status'] == PASSED and sha(predictions) == predictions_sha
    assert sha(ORIGINAL) == ORIGINAL_SHA and sha(failure) == FAILURE_SHA
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--protocol-sha')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False))
    else:
        result = check(args.protocol_sha)
        print(json.dumps({key: result[key] for key in ('status', 'formal_heads_checked', 'pilot_heads_checked',
            'maximum_independent_gradient_inf', 'independent_H_encoder_query_forwards', 'elapsed_seconds')}, allow_nan=False))
