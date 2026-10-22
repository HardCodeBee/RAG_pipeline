"""Read-only native scoring and fixed-denominator reports for the new runtime.

Reference answers are loaded only here, never by inference controllers. Missing
infrastructure/usage stays None; algorithm failures have the frozen zero score.
"""
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from . import runtime_config as cfg
from .qa_raw_runtime import FINAL_METHOD
from .runtime_inputs import ReceiptReader
from .runtime_ledger import valid_usage
from .metrics import quality_for_primary_table


def score_rows(qids, predictions, references, scorer, *, verify_success=None, verify_algorithm_failure=None):
    rows = []
    for qid in qids:
        record = predictions.get(qid, {'status': 'pending'})
        status = record.get('status', 'pending')
        value = {'f1': None, 'em': None}
        if status == 'algorithm_failure':
            if verify_algorithm_failure is None or verify_algorithm_failure(qid, record):
                value = {'f1': 0., 'em': 0.}
            else:
                status = 'technical_pending_receipt'
        elif status == 'success' and isinstance(record.get('prediction'), str):
            if verify_success is None or verify_success(qid, record):
                value = scorer(record['prediction'], references[qid])
                if set(value) != {'f1', 'em'} or any(not 0 <= v <= 1 for v in value.values()):
                    raise ValueError('Native scorer returned invalid Answer F1/EM')
            else:
                status = 'technical_pending_receipt'
        rows.append({'qid': qid, 'status': status, **value})
    return rows


def quality_summary(rows, *, planned):
    if len(rows) != planned or len({r['qid'] for r in rows}) != planned:
        raise ValueError('Fixed denominator or unique ID set changed')
    available = [r for r in rows if r['f1'] is not None]
    missing = planned - len(available)
    sums = {key: sum(r[key] for r in available) for key in ('f1', 'em')}
    return {'planned': planned, 'scorable': len(available), 'technical_missing': missing,
            'statuses': dict(Counter(r['status'] for r in rows)),
            'status': 'complete' if not missing else 'not_ready',
            'quality_for_primary_table': {k: v/planned for k, v in sums.items()} if not missing else None,
            'fixed_denominator_bounds': {k: [v/planned, (v+missing)/planned] for k, v in sums.items()}}


def main_quality_summary(rows, *, expected_qids):
    """Use the frozen publication gate; partial scoring stays internal."""
    return {**quality_for_primary_table(rows, expected_qids=expected_qids),
            'planned': 500, 'statuses': dict(Counter(r['status'] for r in rows))}


def paired_summary(qa, raw, *, planned):
    a, b = {r['qid']: r for r in qa}, {r['qid']: r for r in raw}
    if len(a) != planned or set(a) != set(b):
        raise ValueError('Pair membership changed')
    details, bounds = [], {k: [0., 0.] for k in ('f1', 'em')}
    for qid in a:
        left, right = a[qid], b[qid]
        row = {'qid': qid, 'QA_status': left['status'], 'Raw_status': right['status']}
        for key in ('f1', 'em'):
            x, y = left[key], right[key]
            row['QA_' + key], row['Raw_' + key] = x, y
            row[key + '_difference'] = x-y if x is not None and y is not None else None
            bounds[key][0] += (x if x is not None else 0.) - (y if y is not None else 1.)
            bounds[key][1] += (x if x is not None else 1.) - (y if y is not None else 0.)
        details.append(row)
    complete = [r for r in details if r['f1_difference'] is not None]
    return {'planned_pairs': planned, 'complete_pairs': len(complete),
            'status': 'complete' if len(complete) == planned else 'not_ready',
            'QA': quality_summary(qa, planned=planned), 'Raw': quality_summary(raw, planned=planned),
            'paired_difference_complete_cases': {k: sum(r[k+'_difference'] for r in complete)/len(complete)
                                                 if complete else None for k in ('f1', 'em')},
            'interpretation': 'Complete-case values are conditional on available pairs; missingness is not assumed random.',
            'paired_difference_fixed_denominator_bounds': {k: [v/planned for v in values] for k, values in bounds.items()},
            'improved_tied_worsened': {'improved': sum(r['f1_difference'] > 0 for r in complete),
                                     'tied': sum(r['f1_difference'] == 0 for r in complete),
                                     'worsened': sum(r['f1_difference'] < 0 for r in complete)},
            'details': details}


def paired_bootstrap(differences_by_dataset, *, iterations=10000, seed=20261023):
    """Fixed paired F1 bootstrap, resampling independently inside each dataset."""
    import numpy as np
    if iterations != 10000 or seed != 20261023 or set(differences_by_dataset) != set(cfg.DATASETS):
        raise ValueError('Frozen bootstrap settings or datasets changed')
    rng = np.random.default_rng(seed)
    samples, per_dataset = [], {}
    for dataset in cfg.DATASETS:
        values = np.asarray(differences_by_dataset[dataset], dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (np.abs(values) > 1).any():
            raise ValueError('All datasets require complete finite paired differences')
        boot = np.empty(iterations)
        for start in range(0, iterations, 500):
            stop = min(start+500, iterations)
            ids = rng.integers(0, len(values), size=(stop-start, len(values)))
            boot[start:stop] = values[ids].mean(axis=1)
        samples.append(boot)
        per_dataset[dataset] = {'n': len(values), 'f1_difference': float(values.mean()),
                                'percentile_95_ci': np.quantile(boot, [.025, .975]).tolist()}
    aggregate = np.mean(samples, axis=0)
    return {'iterations': iterations, 'seed': seed, 'method': 'paired within-dataset percentile bootstrap',
            'datasets': per_dataset,
            'equal_dataset_mean': {'f1_difference': sum(v['f1_difference'] for v in per_dataset.values())/3,
                                   'percentile_95_ci': np.quantile(aggregate, [.025, .975]).tolist()},
            'interval_policy': 'Individual prespecified comparisons; no simultaneous-significance claim.'}


def _predictions(directory):
    path = Path(directory) / 'checkpoint.sqlite3'
    if not path.is_file():
        return {}
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        return {qid: json.loads(raw) for qid, raw in db.execute("SELECT key,value FROM data WHERE kind='prediction'")}


def _verified(reader, dataset, method, qid, record, *, terminal='success', expected_stage=None):
    rid = record.get('request_id')
    receipt = reader.receipt(rid) if rid else None
    result = (receipt or {}).get('result') or {}
    return bool(receipt and receipt['dataset'] == dataset and receipt['method'] == method and
                receipt['qid'] == qid and receipt['status'] == terminal and result.get('status') == terminal and
                (expected_stage is None or receipt['stage'] == expected_stage) and
                (result.get('finish_reason') == 'stop' or terminal == 'algorithm_failure' and
                 result.get('finish_reason') == 'length' and (result.get('usage') or {}).get('completion_tokens') == cfg.common.MAX_OUTPUT) and
                result.get('returned_model') == cfg.common.MODEL and
                valid_usage(result.get('usage')))


def _qa_arm_costs(reader, output, dataset, qids, arm):
    """Each arm includes the same prefix once; new expenditure is finals only."""
    totals = {k: 0 for k in ('upstream_logical_input_tokens', 'upstream_logical_output_tokens',
                            'upstream_logical_CNY', 'final_known_input_tokens', 'final_known_output_tokens',
                            'final_known_CNY', 'new_final_known_CNY', 'unknown_final_reserved_CNY')}
    observed, complete = 0, True
    if reader.ledger is None:
        return {'status': 'not_ready', **{k: None for k in totals}}
    for qid in qids:
        path = output / 'qa_raw' / dataset / qid / 'INPUT.json'
        if not path.exists():
            continue
        pair = cfg.read(path)
        observed += 1
        for rid in pair['upstream']['source_request_ids']:
            saved = reader.receipt(rid)
            result = (saved or {}).get('result') or {}
            u = result.get('usage')
            if not (saved and saved['dataset'] == dataset and saved['qid'] == qid and saved['method'] == cfg.P02 and
                    saved['stage'] != 'legacy/final' and result.get('status') == 'success' and valid_usage(u)):
                complete = False
                continue
            totals['upstream_logical_input_tokens'] += u['prompt_tokens']
            totals['upstream_logical_output_tokens'] += u['completion_tokens']
            totals['upstream_logical_CNY'] += cfg.common.usage_cost(u['prompt_tokens'], u['completion_tokens'])
        rows = reader.ledger.db.execute('''SELECT a.actual,a.reserved,a.usage FROM rt_attempts a
            JOIN rt_stages s ON a.key=s.key WHERE s.dataset=? AND s.method=? AND s.qid=? AND s.stage=?''',
            (dataset, FINAL_METHOD, qid, f'qa_raw/{arm}/final')).fetchall()
        if not rows:
            complete = False
        for actual, reserve, usage in rows:
            if actual is None:
                totals['unknown_final_reserved_CNY'] += reserve
                complete = False
            else:
                totals['new_final_known_CNY'] += actual
            u = json.loads(usage) if usage else None
            if valid_usage(u):
                totals['final_known_input_tokens'] += u['prompt_tokens']
                totals['final_known_output_tokens'] += u['completion_tokens']
                totals['final_known_CNY'] += cfg.common.usage_cost(u['prompt_tokens'], u['completion_tokens'])
    return {'status': 'observed_only' if observed else 'not_ready', 'queries_with_prefix': observed,
            'logical_usage_complete_for_observed_queries': complete if observed else None,
            **(totals if observed else {k: None for k in totals}),
            'logical_arm_known_CNY': totals['upstream_logical_CNY'] + totals['final_known_CNY'] if observed else None,
            'incremental_spend_excludes_shared_upstream': True,
            'local_R02_usage': 'same bound upstream decomposition as P02; reported separately from hosted API tokens'}


def _costs(reader, dataset, method, qids):
    if reader.ledger is None:
        return {'status': 'not_ready', 'logical_input_tokens': None, 'logical_output_tokens': None,
                'logical_known_CNY': None, 'new_known_CNY': None, 'unknown_reserved_CNY': None}
    observed = {r[0] for r in reader.ledger.db.execute('SELECT DISTINCT qid FROM rt_stages WHERE dataset=? AND method=?', (dataset, method))}
    selected = [qid for qid in qids if qid in observed]
    sums = {key: 0 for key in ('logical_input_tokens', 'logical_output_tokens', 'logical_known_CNY',
                              'new_known_CNY', 'unknown_reserved_CNY')}
    complete_usage = True
    for qid in selected:
        summary = reader.ledger.query_summary(dataset, method, qid)
        for key in sums:
            sums[key] += summary[key]
        complete_usage = complete_usage and summary['logical_usage_complete']
    return {'status': 'observed_only' if selected else 'not_ready', 'queries_with_ledger_records': len(selected),
            'logical_usage_complete_for_observed_records': complete_usage if selected else None,
            **(sums if selected else {k: None for k in sums}),
            'scope': 'Observed full-flow hosted-model usage, including known failed attempts; reused logical usage is not newly billed.'}


def _r02_tokens(reader, dataset, qids, questions):
    """Keep already bound local decomposer usage separate from hosted API totals."""
    if reader.ledger is None:
        return {'status': 'not_ready', 'input_tokens': None, 'output_tokens': None, 'gpu_cost_CNY': None}
    observed = {r[0] for r in reader.ledger.db.execute('SELECT DISTINCT qid FROM rt_stages WHERE dataset=? AND method=?', (dataset, cfg.P02))}
    selected = [qid for qid in qids if qid in observed]
    if not selected:
        return {'status': 'not_ready', 'input_tokens': None, 'output_tokens': None, 'gpu_cost_CNY': None}
    numbered = {(q['dataset'], q['qid']): i for i, q in enumerate(cfg.rows(cfg.R02_HERE / 'questions1500.frozen.jsonl'), 1)}
    known, missing, ins, outs = 0, 0, 0, 0
    for qid in selected:
        try:
            cfg.load_r02_case(dataset, questions[qid])
            receipt = cfg.read(cfg.R02_OUTPUT / f'remote_raw/{numbered[dataset, qid]:03d}.json')
            if any(type(receipt.get(k)) is not int or receipt[k] < 0 for k in ('input_tokens', 'output_tokens')):
                raise ValueError('Missing local token usage')
            ins += receipt['input_tokens']; outs += receipt['output_tokens']; known += 1
        except (OSError, ValueError, KeyError):
            missing += 1
    return {'status': 'complete_for_observed_queries' if not missing else 'usage_incomplete',
            'queries_with_bound_usage': known, 'queries_with_missing_usage': missing,
            'known_input_tokens': ins, 'known_output_tokens': outs, 'gpu_cost_CNY': None,
            'role': 'Frozen R02 local decomposition, separate from hosted Qwen reader tokens; no training or new decomposition.'}


def report(*, output=cfg.OUTPUT):
    """Return JSON-compatible report; do not write checkpoints, ledgers or gold."""
    # Native metric imports and reference-answer reads are confined to this
    # offline function; inference modules never import comprehensive.scoring.
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    output = Path(output)
    plan = cfg.read(cfg.HERE / 'PLAN.json')
    question_sets = cfg.load_questions(plan)
    cohort = cfg.read(cfg.REVIEW / 'qa_raw/COHORT.qid_only.json')
    reader = ReceiptReader(output / 'runtime_ledger.sqlite3')
    manifest_path = output / 'qa_raw/INPUT_MANIFEST.json'
    initial = {(r['dataset'], r['qid']): r for r in cfg.read(manifest_path)['rows']} if manifest_path.exists() else {}
    main, qa_raw, main_rows = {}, {}, {}
    try:
        for dataset in cfg.DATASETS:
            questions = {q['qid']: q for q in question_sets[dataset]}
            qids = list(questions)
            refs = {r['qid']: r for r in cfg.rows(cfg.ROOT / plan['input_directory'] / dataset / 'formal.references.local.jsonl')}
            if set(refs) != set(qids):
                raise ValueError('Reference IDs differ from frozen formal500')
            scorer = metric(dataset)
            main[dataset], main_rows[dataset] = {}, {}
            for method in cfg.METHODS:
                predictions = _predictions(output / 'main' / dataset / method)
                if not set(predictions).issubset(questions):
                    raise ValueError('Output contains an out-of-cohort main question')
                rows = score_rows(qids, predictions, refs, scorer,
                                  verify_success=lambda q, r, m=method: _verified(reader, dataset, m, q, r))
                main_rows[dataset][method] = rows
                main[dataset][method] = {**main_quality_summary(rows, expected_qids=qids),
                                         'resources': _costs(reader, dataset, method, qids)}
            main[dataset][cfg.P02]['local_R02_resources'] = _r02_tokens(reader, dataset, qids, questions)
            chosen = [q['qid'] for q in cohort['datasets'][dataset]['queries']]
            if len(chosen) != 100 or len(set(chosen)) != 100 or not set(chosen).issubset(questions):
                raise ValueError('QA/raw fixed100 membership changed')
            predictions = {arm: {qid: initial.get((dataset, qid), {}).get(arm, {'status': 'pending_upstream'})
                                  for qid in chosen} for arm in ('QA', 'Raw')}
            for qid in chosen:
                path = output / 'qa_raw' / dataset / qid / (cfg.common.digest([dataset, qid]) + '.json')
                if not path.exists():
                    continue
                saved = cfg.read(path)
                if saved.get('dataset') != dataset or saved.get('qid') != qid:
                    raise ValueError('QA/raw saved question binding mismatch')
                for arm in predictions:
                    predictions[arm][qid] = saved.get('arms', {}).get(arm, {'status': 'pending'})
            arms = {arm: score_rows(chosen, values, refs, scorer,
                                   verify_success=lambda q, r, a=arm: _verified(reader, dataset, FINAL_METHOD, q, r, expected_stage=f'qa_raw/{a}/final'),
                                   verify_algorithm_failure=lambda q, r, a=arm: _verified(reader, dataset, FINAL_METHOD, q, r, terminal='algorithm_failure', expected_stage=f'qa_raw/{a}/final'))
                    for arm, values in predictions.items()}
            qa_raw[dataset] = paired_summary(arms['QA'], arms['Raw'], planned=100)
            qa_raw[dataset]['new_final_resources'] = _costs(reader, dataset, FINAL_METHOD, chosen)
            qa_raw[dataset]['arm_logical_resources'] = {a: _qa_arm_costs(reader, output, dataset, chosen, a) for a in arms}
        comparisons = {}
        for baseline in cfg.METHODS[1:]:
            ready = all(main[ds][cfg.P02]['status'] == main[ds][baseline]['status'] == 'complete500' for ds in cfg.DATASETS)
            comparisons[baseline] = paired_bootstrap({ds: [p['f1']-b['f1'] for p, b in
                zip(main_rows[ds][cfg.P02], main_rows[ds][baseline])] for ds in cfg.DATASETS}) if ready else {'status': 'not_ready'}
        ready = all(qa_raw[ds]['status'] == 'complete' for ds in cfg.DATASETS)
        qa_effect = paired_bootstrap({ds: [r['f1_difference'] for r in qa_raw[ds]['details']] for ds in cfg.DATASETS}) if ready else {'status': 'not_ready'}
        qa_effect['equal_dataset_fixed_denominator_bounds'] = {
            key: [sum(qa_raw[ds]['paired_difference_fixed_denominator_bounds'][key][i] for ds in cfg.DATASETS)/3 for i in (0, 1)]
            for key in ('f1', 'em')}
        return {'schema': 'unified_six_method_offline_report_v1',
                'status': 'complete' if all(main[d][m]['status'] == 'complete500' for d in cfg.DATASETS for m in cfg.METHODS) and ready else 'not_ready',
                'planned_main_outcomes': 9000, 'planned_qa_raw_queries': 300, 'planned_qa_raw_conditions': 600,
                'main': main, 'main_P02_comparisons': comparisons,
                'qa_raw': qa_raw, 'qa_raw_primary_effect': qa_effect,
                'metric': 'Frozen native Answer F1/EM; 2Wiki aliases/demonyms and MuSiQue aliases unchanged.',
                'references': 'Read only by this offline module; no gold text included in report.',
                'new_model_requests': 0, 'execution_authorized': False}
    finally:
        reader.close()
