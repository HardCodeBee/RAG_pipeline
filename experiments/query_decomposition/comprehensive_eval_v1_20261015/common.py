"""Separate campaign state; reuse only immutable prior code and infrastructure."""
import hashlib
import json
from pathlib import Path
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, OUTPUT, canonical, utc
from experiments.query_decomposition.hotpot_formal_eval_v1.parallel_api import locked_json

HERE = Path(__file__).resolve().parent
RUN = OUTPUT / 'comprehensive_v1_20261015'
DATA = ROOT / 'data/multihop_comprehensive_v1_20261015'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def rows(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(s) for s in f if s.strip()]

def save(path, value):
    locked_json(Path(path), value)

def once(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen state changed: ' + str(path))
    else:
        save(path, value)

def log(event, **fields):
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE/'changes.jsonl').open('a', encoding='utf8') as f:
        f.write(canonical({'utc':utc(), 'event':event, **fields})+'\n')

def audit_prior():
    checks = []
    specs = [
        (OUTPUT/'baseline_compact100_200_20261014/CAMPAIGN_FROZEN.json', 'method_source_files_sha256', ROOT/'experiments/query_decomposition/hotpot_formal_eval_v1'),
        (OUTPUT/'s02_child4_compact_formal200_20261014/CONFIG_FROZEN.json', 'sources_sha256', ROOT),
        (OUTPUT/'parallel_execution_preparation/EXECUTION_EPOCH_FROZEN.json', 'source_sha256', ROOT),
    ]
    for receipt, key, base in specs:
        pins = read(receipt)[key]
        for name, expected in pins.items():
            if sha(base/name) != expected:
                raise ValueError('Prior frozen source differs: '+name)
        checks.append({'receipt':str(receipt.relative_to(ROOT)), 'receipt_sha256':sha(receipt), 'pins_verified':len(pins)})
    return checks

if __name__ == '__main__':
    import sqlite3
    checks = audit_prior()
    with sqlite3.connect('file:'+str(OUTPUT/'budget.sqlite3')+'?mode=ro',uri=True) as db:
        actual, reserve = db.execute('SELECT coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
    result = {'utc':utc(), 'prior_configuration_checks':checks, 'cap_cny':200,
              'returned_usage_cny':actual, 'unknown_reservations_cny':reserve, 'remaining_liability_cny':200-actual-reserve,
              'remote_scope':'query_decomposition_only_manual_upload', 'authorized_now':['three_dataset_main','same_final_prompt_top10','four_core_ablations'],
              'not_started':['transfer','no_retrieval_diagnostic','hallucination_audit','robustness_expansion'],
              'new_model_training':False, 'old_sources_modified':False}
    save(RUN/'AUTHORIZATION_AND_PRIOR_AUDIT.json',result)
    log('execution_authorized_and_prior_sources_verified', **result)
    print(json.dumps(result))
