"""Offline query-metadata inventory; no answers or predictions used to select."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path.cwd().resolve()
OUT = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/query_decomposition'
EXPECTED_R02 = '728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92'

def norm(q):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', q).casefold()).strip(' \t\n?!？！.')

def lines(p):
    with p.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

known_ids, known_text, known_mu, training_mu = set(), set(), set(), set()
inventory = []

def add(r, train=False):
    if not isinstance(r, dict):
        return 0
    found = 0
    for k in ['query', 'question', 'question_text', 'original_question']:
        if isinstance(r.get(k), str):
            known_text.add(norm(r[k]))
            found = 1
    src = r.get('source') if isinstance(r.get('source'), dict) else {}
    for v in [r.get('id'), r.get('qid'), r.get('source_qid'), src.get('id'), src.get('native_id')]:
        if isinstance(v, str):
            known_ids.update([v, v.split(':', 1)[-1]])
            mu_match = re.search(r'([234]hop\d*__[0-9_]+)', v)
            if mu_match:
                parts = set(mu_match.group(1).split('__', 1)[1].split('_'))
                known_mu.update(parts)
                if train:
                    training_mu.update(parts)
    return found

rx = re.compile(r'question|quer|runtime|cases|master|data3632|base3496|candidate|native120|new136|historical16|screen|exclude|split', re.I)
for p in sorted(EXP.rglob('*.jsonl')):
    if not rx.search(p.name) or 'reference' in p.name.lower():
        continue
    if 'b0_targeted_optimization_v' in str(p):
        continue
    n = 0
    try:
        for r in lines(p):
            n += add(r)
    except (ValueError, UnicodeError):
        continue
    if n:
        inventory.append({'path': str(p.relative_to(ROOT)), 'query_records': n})

# Earlier authored few-shot questions are an explicit exclusion source.
def visit(x):
    if isinstance(x, dict):
        add(x)
        for v in x.values():
            if isinstance(v, (dict, list)):
                visit(v)
    elif isinstance(x, list):
        for v in x:
            visit(v)
for p in sorted(EXP.rglob('*example*.json')):
    if p.stat().st_size < 10_000_000:
        try:
            visit(json.loads(p.read_text(encoding='utf-8-sig')))
        except ValueError:
            continue

# Actual final R02 training set supplies the primitive exclusion guarantee.
training_paths = [
    EXP / 'multigrain_4b_full_retrain3632_v1_20261018/DATA3632.frozen.jsonl',
    ROOT / 'outputs/query_decomposition/multigrain_4b_prompt_train3500_v1_20261017/training/MASTER_3500.jsonl',
]
for train_path in training_paths:
    for r in lines(train_path):
        add(r, train=True)

native = {
    'hotpotqa': ROOT / 'analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl',
    '2wiki': ROOT / 'data/multihop_comprehensive_v1_20261015/2wiki/native/dev.json',
    'musique': ROOT / 'data/multihop_comprehensive_v1_20261015/musique/native/musique_ans_v1.0_dev.jsonl',
}
availability = {}
eligible_ids = set()
for ds, p in native.items():
    source = json.loads(p.read_text(encoding='utf8')) if ds == '2wiki' else lines(p)
    totals, basic, train_disjoint, full_primitive_disjoint = Counter(), Counter(), Counter(), Counter()
    for r in source:
        # Do not access answer, supports, decomposed answers, or model outcomes.
        qid = r.get('_id', r.get('id'))
        question = r['question']
        kind = str(len(r['question_decomposition'])) + '-hop' if ds == 'musique' else r['type']
        totals[kind] += 1
        if qid in known_ids or ds + ':' + qid in known_ids or norm(question) in known_text:
            continue
        basic[kind] += 1
        parts = set(qid.split('__', 1)[1].split('_')) if ds == 'musique' else set()
        if parts & training_mu:
            continue
        train_disjoint[kind] += 1
        eligible_ids.add(ds + ':' + qid)
        if not parts & known_mu:
            full_primitive_disjoint[kind] += 1
    availability[ds] = {'native_rows': dict(totals), 'after_known_id_and_text': dict(basic),
                        'also_training_primitive_disjoint': dict(train_disjoint),
                        'also_all_known_primitive_disjoint': dict(full_primitive_disjoint)}

# Cached raw generation is reusable only with the same selected R02 identity.
cached_ids = set()
cache_roots = []
for p in (ROOT / 'outputs/query_decomposition').rglob('*IDENTITY*.json'):
    if 'recovery_contract' in p.parts:
        continue
    try:
        ii = json.loads(p.read_text(encoding='utf8'))
    except (ValueError, UnicodeError):
        continue
    if not isinstance(ii, dict) or ii.get('checkpoint_manifest_sha256') != EXPECTED_R02:
        continue
    n = 0
    raw_directories = [p.parent]
    if p.name == 'RAW_IDENTITY.json':
        raw_directories.append(p.parent / 'raw')
    for directory in raw_directories:
        for q in directory.glob('[0-9][0-9][0-9].json'):
            rr = json.loads(q.read_text(encoding='utf8'))
            if isinstance(rr.get('id'), str) and isinstance(rr.get('query'), str):
                cached_ids.add(rr['id'])
                n += 1
    cache_roots.append({'path': str(p.parent.relative_to(ROOT)), 'raw_records': n})

protocol = json.loads((EXP / 'complete_pipeline_comparison_v1_20261020/EVAL300_PROTOCOL.frozen.json').read_text(encoding='utf8'))
resources = {}
for ds, spec in protocol['resources'].items():
    resources[ds] = {'documents': spec['corpus']['documents'], 'corpus_manifest': spec['corpus']['manifest'],
                     'bge_build': spec['dense']['bge']['build'], 'bge_model': spec['dense']['bge']['query_model']}
    if ds in ['2wiki', 'musique']:
        dense_path = Path(spec['dense']['bge']['build']) / 'DENSE_FROZEN.json'
        dense = json.loads(dense_path.read_text(encoding='utf8'))
        assert dense['documents'] == spec['corpus']['documents']
        assert dense['source_sha256'] == spec['corpus']['chunks_sha256']
        resources[ds].update(dense_manifest=str(dense_path.relative_to(ROOT)),
                             full_corpus_embeddings_count_verified=True,
                             dense_source_sha256_matches_corpus=True)

result = {'metadata_only_selection': True, 'new_api_calls': 0, 'remote_calls': 0,
          'known_input_files': inventory, 'known_id_strings': len(known_ids), 'known_query_strings': len(known_text),
          'known_musique_primitives': len(known_mu), 'training_musique_primitives': len(training_mu),
          'training_query_sources_union': [str(p.relative_to(ROOT)) for p in training_paths],
          'availability': availability, 'resources': resources, 'matching_R02_cache_roots': cache_roots,
          'matching_R02_cached_distinct_ids': len(cached_ids),
          'matching_R02_cache_overlap_with_eligible': len(cached_ids & eligible_ids),
          'limitations': ['Input inventory is conservative but not a complete exposure ledger.',
                         'No guarantee of paraphrase independence or pretrained-model decontamination.',
                         'Corpus document exposure is expected for a fixed retrieval corpus, distinct from query/outcome exposure.',
                         'No actual validation cohort has been frozen and no new native answers used for outcome selection.']}
(OUT / 'HOLDOUT_FEASIBILITY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps({k: v for k, v in result.items() if k not in ['known_input_files', 'resources', 'matching_R02_cache_roots']}, ensure_ascii=False))
