"""Protocol constants; no gold answers enter method execution."""
import json
from pathlib import Path
from experiments.query_decomposition.comprehensive_eval_v1_20261015.common import ROOT, RUN, OUTPUT, read, rows, sha, save, once, audit_prior
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import canonical, digest, utc

HERE = Path(__file__).resolve().parent
OUT = RUN / 'baseline_integrity_repair_v2_20261017'
PARENT = RUN / 'campaign_api_v1'
INPUTS = ROOT / 'experiments/query_decomposition/comprehensive_eval_v1_20261015/inputs_frozen'
MODEL = 'qwen-plus-2025-12-01'
MAX_OUTPUT = 32768
MAX_INPUT = 997952
CONTEXT = 1000000
PROTECTED_CAP = 195
METHODS = {
    'b00_rag_top10_v1': 'rag_top10_v1', 'b01_rag_top10_cot3': 'rag_top10_cot3',
    'b02_selfask_dense': 'selfask_dense', 'b03_react_dense': 'react_dense',
    'b04_ircot_bge': 'ircot_bge', 'b05_iter_retgen_bge': 'iter_retgen_bge',
    'b06_decomp_bge': 'decomp_bge', 'b07_hybrid_rerank': 'hybrid_rerank',
    'b08_qd_rerank': 'qd_rerank',
}
DATASETS = ['2wiki', 'hotpotqa', 'musique']


class IntegrityError(RuntimeError):
    """A blocked input/output must never be treated as a normal answer."""


def journal(event, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'changes.jsonl').open('a', encoding='utf8') as f:
        f.write(canonical({'utc': utc(), 'event': event, **fields}) + '\n')


def prices(prompt_tokens):
    # Frozen Beijing public list prices, non-thinking, 2025-12-01 snapshot.
    if prompt_tokens <= 128000:
        return .8e-6, 2e-6
    if prompt_tokens <= 256000:
        return 2.4e-6, 20e-6
    return 4.8e-6, 48e-6


def usage_cost(prompt_tokens, completion_tokens):
    a, b = prices(prompt_tokens)
    return prompt_tokens * a + completion_tokens * b


def reserve_cost(byte_upper, max_tokens):
    return usage_cost(byte_upper, max_tokens)


def verify():
    receipt = read(OUT / 'CONFIG_FROZEN.json')
    for name, expected in receipt['source_and_input_sha256'].items():
        if sha(ROOT / name) != expected:
            raise IntegrityError('Frozen repair source/input changed: ' + name)
    audit_prior()
    return receipt
