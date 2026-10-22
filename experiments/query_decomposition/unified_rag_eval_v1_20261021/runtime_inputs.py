"""Bind fixed QA/raw questions to verified main prefixes, without new upstream work."""
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import write_json, Store
from . import runtime_config as cfg
from .qa_raw_runtime import build_pair, run_pair, PrefixBlocked, FINAL_METHOD
from .runtime_ledger import RuntimeLedger


class ReceiptReader:
    """Read the new ledger without startup recovery mutations or schema creation."""
    def __init__(self, path):
        self.path = Path(path)
        self.ledger = None
        if self.path.is_file():
            self.ledger = object.__new__(RuntimeLedger)
            self.ledger.path = self.path
            self.ledger.db = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)

    def receipt(self, rid):
        return self.ledger.receipt(rid) if self.ledger else None

    def close(self):
        if self.ledger:
            self.ledger.db.close()


def prefix_binding(directory, qid):
    path = directory / 'checkpoint.sqlite3'
    if not path.exists():
        return None
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        row = db.execute("SELECT value FROM data WHERE kind='qa_raw_prefix_binding' AND key=?", (qid,)).fetchone()
        return json.loads(row[0]) if row else None


def prepare_qa_raw(*, output=cfg.OUTPUT):
    """No network, model, retrieval, ledger mutation or cache migration."""
    cfg.ExecutionGate()  # Verify frozen binding, without asking for execution permission.
    cohort = cfg.read(cfg.REVIEW / 'qa_raw/COHORT.qid_only.json')
    questions = {ds: {c['qid']: c for c in cases} for ds, cases in cfg.load_questions().items()}
    output = Path(output)
    reader = ReceiptReader(output / 'runtime_ledger.sqlite3')
    entries = []
    try:
        for dataset, group in cohort['datasets'].items():
            if len(group['queries']) != 100 or len({q['qid'] for q in group['queries']}) != 100:
                raise ValueError('Fixed QA/raw cohort must remain unique100 per dataset')
            namespace = cfg.common.digest(cfg.resource(dataset))
            directory = output / 'main' / dataset / cfg.P02
            for selected in group['queries']:
                qid = selected['qid']
                if qid not in questions[dataset]:
                    raise ValueError('QA/raw question is outside frozen formal500')
                entry = {'dataset': dataset, 'qid': qid, 'status': 'pending_upstream',
                         'QA': {'status': 'pending_upstream'}, 'Raw': {'status': 'pending_upstream'}}
                source = directory / 'prefixes' / (qid + '.json')
                if source.exists():
                    try:
                        binding = prefix_binding(directory, qid)
                        if binding is None or binding['resource_namespace'] != namespace:
                            raise PrefixBlocked('Missing or different main prefix binding')
                        case = cfg.load_r02_case(dataset, questions[dataset][qid])
                        pair = build_pair(cfg.read(source), expected_case=case,
                            expected_namespace=namespace, expected_prefix_sha256=binding['prefix_sha256'],
                            receipt_resolver=lambda receipt: reader.receipt(receipt['request_id']))
                        target = output / 'qa_raw' / dataset / qid / 'INPUT.json'
                        if target.exists() and cfg.read(target) != pair:
                            raise PrefixBlocked('Previously built pair changed; explicit review required')
                        write_json(target, pair)
                        entry.update(status=pair['status'], input=str(target.relative_to(output)))
                        for arm in ('QA', 'Raw'):
                            entry[arm]['status'] = pair['status']
                    except (PrefixBlocked, ValueError, KeyError, FileNotFoundError) as exc:
                        entry.update(status='upstream_not_evaluable', reason=str(exc))
                        for arm in ('QA', 'Raw'):
                            entry[arm]['status'] = 'upstream_not_evaluable'
                entries.append(entry)
    finally:
        reader.close()
    if len(entries) != 300:
        raise ValueError('Fixed300 denominator changed')
    manifest = {'schema': 'qa_raw_runtime_inputs_v1', 'planned_queries': 300,
                'planned_final_conditions': 600, 'question_replacements': 0,
                'status_counts': dict(Counter(r['status'] for r in entries)),
                'execution_authorized': False, 'rows': entries}
    write_json(output / 'qa_raw/INPUT_MANIFEST.json', manifest)
    return {k: v for k, v in manifest.items() if k != 'rows'}


def execute_qa_raw(runner):
    runner.guard()
    prepare_qa_raw(output=runner.output)
    manifest = cfg.read(runner.output / 'qa_raw/INPUT_MANIFEST.json')
    for entry in manifest['rows']:
        if entry['status'] != 'ready':
            continue
        runner.guard()
        dataset, qid = entry['dataset'], entry['qid']
        resources = cfg.resource(dataset)
        store = Store(runner.output / 'qa_raw_requests' / dataset,
                      runner.identity(dataset, FINAL_METHOD, resources))
        try:
            transport = runner.transport_factory(store, runner.client, runner.ledger,
                guard=runner.guard, dataset=dataset, method=FINAL_METHOD)
            transport.qid = qid
            run_pair(cfg.read(runner.output / entry['input']), transport,
                     runner.output / 'qa_raw' / dataset / qid,
                     guard=runner.guard, receipt_lookup=runner.ledger.receipt)
        finally:
            store.export()
            store.db.close()
