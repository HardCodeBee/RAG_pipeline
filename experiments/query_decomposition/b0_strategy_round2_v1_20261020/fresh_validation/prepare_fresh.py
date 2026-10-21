"""Freeze query-only dev90/sealed150 from metadata, without outcome selection.

No API, SSH, retrieval, prediction, reference-answer access or training occurs here.
Source objects may contain gold fields; only question/ID/type metadata is accessed.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/query_decomposition'
OUT = ROOT / 'outputs/query_decomposition'
PREVIOUS = OUT / 'hotpot_formal_eval_v1/b0_targeted_optimization_v2_20261020/HOLDOUT_FEASIBILITY.json'
SEED = 'b0-strategy-round2-fresh-dev90-sealed150-20261020-v001'
TRAINING = [EXP / 'multigrain_4b_full_retrain3632_v1_20261018/DATA3632.frozen.jsonl',
            OUT / 'multigrain_4b_prompt_train3500_v1_20261017/training/MASTER_3500.jsonl']
NATIVE = {'hotpotqa': ROOT / 'analysis/hotpotqa_difficulty_retrieval_20261002/source_labels/validation_labels.jsonl',
          '2wiki': ROOT / 'data/multihop_comprehensive_v1_20261015/2wiki/native/dev.json',
          'musique': ROOT / 'data/multihop_comprehensive_v1_20261015/musique/native/musique_ans_v1.0_dev.jsonl'}
QUOTAS = {'dev90': {'hotpotqa': {'bridge': 21, 'comparison': 9},
                    '2wiki': {'compositional': 8, 'comparison': 7, 'inference': 8, 'bridge_comparison': 7},
                    'musique': {'2-hop': 11, '3-hop': 10, '4-hop': 9}},
          'sealed150': {'hotpotqa': {'bridge': 35, 'comparison': 15},
                        '2wiki': {'compositional': 12, 'comparison': 13, 'inference': 12, 'bridge_comparison': 13},
                        'musique': {'2-hop': 18, '3-hop': 17, '4-hop': 15}}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def norm(q):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', q).casefold()).strip(' \t\n?!？！.')


def lines(p):
    with p.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def freeze(path, value, jsonl=False):
    text = (''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in value) if jsonl
            else json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, 'Preserve existing frozen artifact: ' + str(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8', newline='\n')


def primitives(qid):
    match = re.search(r'([234]hop\d*__[0-9_]+)', qid)
    return set(match.group(1).split('__', 1)[1].split('_')) if match else set()


def main():
    if (HERE / 'MANIFEST.frozen.json').exists():
        manifest = json.loads((HERE / 'MANIFEST.frozen.json').read_text(encoding='utf-8'))
        for name, record in manifest['files'].items(): assert sha(HERE / name) == record['sha256']
        print(json.dumps({'status': 'existing_immutable_split_verified', 'counts': manifest['counts'], 'new_API_calls': 0}))
        return
    known_ids, known_text, known_mu, training_mu = set(), set(), set(), set()
    inventory = []

    def add(row, train=False):
        if not isinstance(row, dict): return 0
        added = 0
        for k in ['query', 'question', 'question_text', 'original_question']:
            if isinstance(row.get(k), str):
                known_text.add(norm(row[k])); added = 1
        src = row.get('source') if isinstance(row.get('source'), dict) else {}
        for v in [row.get('id'), row.get('qid'), row.get('source_qid'), row.get('parent_id'),
                  row.get('parent_qid'), src.get('id'), src.get('native_id')]:
            if isinstance(v, str):
                known_ids.update([v, v.split(':', 1)[-1]])
                known_mu.update(primitives(v))
                if train: training_mu.update(primitives(v))
        return added

    previous = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    candidates = {ROOT / x['path'] for x in previous['known_input_files']}
    # All explicit experimental JSONL inputs are conservatively treated as seen.
    candidates.update(p for p in EXP.rglob('*.jsonl') if HERE not in p.parents and 'vendor' not in p.parts)
    # Supplement earlier inventory with explicit query/training inputs under outputs.
    input_name = re.compile(r'queries|questions|runtime|cases\.inputs|MASTER|DATA3632|BASE3496|NATIVE120|NEW136|HISTORICAL16|screen|EXCLUDE|query_only', re.I)
    candidates.update(p for p in OUT.rglob('*.jsonl') if input_name.search(p.name) and 'vendor' not in p.parts)
    skipped = []
    for path in sorted(candidates):
        if 'reference' in path.name.lower() or path.stat().st_size > 250_000_000:
            skipped.append({'path': str(path.relative_to(ROOT)), 'reason': 'reference_file_or_above250MB'}); continue
        try:
            count = sum(add(r) for r in lines(path))
        except (ValueError, UnicodeError) as exc:
            skipped.append({'path': str(path.relative_to(ROOT)), 'reason': type(exc).__name__}); continue
        if count: inventory.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'query_records': count, 'role': 'known_query_input'})

    def visit(obj):
        if isinstance(obj, dict):
            n = add(obj)
            return n + sum(visit(v) for v in obj.values() if isinstance(v, (list, dict)))
        if isinstance(obj, list): return sum(visit(v) for v in obj)
        return 0

    for path in sorted(EXP.rglob('*example*.json')):
        if path.stat().st_size > 10_000_000: continue
        try: count = visit(json.loads(path.read_text(encoding='utf-8-sig')))
        except (ValueError, UnicodeError): continue
        if count: inventory.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'query_records': count, 'role': 'few_shot_questions'})
    # Explicitly exclude all592 old same-R02 cached query IDs/texts; no outputs read.
    old_cache_records = 0
    old_cache_ids = set()
    for root in previous['matching_R02_cache_roots']:
        folder = ROOT / root['path']
        for path in sorted(set(folder.glob('[0-9][0-9][0-9].json')) | set((folder / 'raw').glob('[0-9][0-9][0-9].json'))):
            row = json.loads(path.read_text(encoding='utf-8'))
            row = {k: row[k] for k in ['id', 'query'] if k in row}
            if add(row):
                old_cache_records += 1
                old_cache_ids.add(row['id'])
                inventory.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'query_records': 1, 'role': 'old_R02_cache_query_metadata_only'})
    for path in TRAINING:
        count = sum(add(r, train=True) for r in lines(path))
        inventory.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'query_records': count, 'role': 'explicit_training_union_for_Mu_primitives'})
    assert training_mu and len(training_mu) >= previous['training_musique_primitives']

    pools = defaultdict(list)
    availability, source_records = {}, {}
    for ds, path in NATIVE.items():
        source_records[ds] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
        source = json.loads(path.read_text(encoding='utf-8')) if ds == '2wiki' else lines(path)
        total, eligible, excluded = Counter(), Counter(), Counter()
        for row in source:
            # Intentionally never access answer, supports, paragraphs or decomposed questions/answers.
            qid = row.get('_id', row.get('id'))
            query = row['question']
            kind = qid.split('hop', 1)[0] + '-hop' if ds == 'musique' else row['type']
            total[kind] += 1
            if qid in known_ids or ds + ':' + qid in known_ids or norm(query) in known_text:
                excluded['known_id_or_normalized_query'] += 1; continue
            if ds == 'musique' and primitives(qid) & training_mu:
                excluded['training_primitive_overlap'] += 1; continue
            record = {'id': ds + ':' + qid, 'dataset': ds, 'qid': qid, 'query': query, 'type': kind}
            if ds == 'musique': record['hops'] = int(kind[0])
            pools[(ds, kind)].append(record); eligible[kind] += 1
        availability[ds] = {'native_rows': dict(total), 'eligible': dict(eligible), 'excluded': dict(excluded)}

    selected = {}
    used_ids, used_text, dev_mu = set(), set(), set()
    for cohort, quotas in QUOTAS.items():
        rows = []
        for ds, kinds in quotas.items():
            for kind, needed in kinds.items():
                choices = sorted(pools[(ds, kind)], key=lambda r: hashlib.sha256((SEED + '|' + cohort + '|' + r['id']).encode()).hexdigest())
                chosen = []
                for row in choices:
                    if row['id'] in used_ids or norm(row['query']) in used_text: continue
                    if cohort == 'sealed150' and ds == 'musique' and primitives(row['qid']) & dev_mu: continue
                    chosen.append(row); used_ids.add(row['id']); used_text.add(norm(row['query']))
                    if len(chosen) == needed: break
                assert len(chosen) == needed, f'Insufficient metadata-only pool for {cohort}/{ds}/{kind}: {len(chosen)} < {needed}'
                rows.extend(chosen)
        rows.sort(key=lambda r: hashlib.sha256((SEED + '|ordering|' + cohort + '|' + r['id']).encode()).hexdigest())
        selected[cohort] = rows
        if cohort == 'dev90': dev_mu = set().union(*(primitives(r['qid']) for r in rows if r['dataset'] == 'musique'))

    assert len(selected['dev90']) == 90 and len(selected['sealed150']) == 150
    assert len(used_ids) == len(used_text) == 240
    confirm_mu = set().union(*(primitives(r['qid']) for r in selected['sealed150'] if r['dataset'] == 'musique'))
    assert not dev_mu & confirm_mu
    assert not (dev_mu | confirm_mu) & training_mu
    for rows in selected.values():
        assert not {r['id'] for r in rows} & known_ids
        assert not {r['qid'] for r in rows} & known_ids
        assert not {norm(r['query']) for r in rows} & known_text

    for cohort, rows in selected.items():
        freeze(HERE / (cohort + '.query_only.frozen.jsonl'), rows, jsonl=True)
        freeze(HERE / 'remote_inputs' / (cohort + '.id_query.frozen.jsonl'), [{'id': r['id'], 'query': r['query']} for r in rows], jsonl=True)
    freeze(HERE / 'EXCLUSION_SOURCES.frozen.json', {'sources': inventory, 'skipped': skipped})
    policy = {'selection': 'deterministic SHA256 ranking on seed/cohort/dataset-prefixed native ID within metadata type strata',
              'seed': SEED, 'quotas': QUOTAS, 'selection_used_gold_answers': False,
              'selection_used_retrieval_or_model_results': False, 'selected_confirmation_queries_manually_read': False,
              'normalization': 'NFKC + casefold + collapse whitespace + strip terminal whitespace/?/!/Chinese punctuation/dot',
              'known_exposure_count': {'id_strings': len(known_ids), 'normalized_query_strings': len(known_text),
                                       'Mu_known_primitives': len(known_mu), 'Mu_training_union_primitives': len(training_mu),
                                       'old_same_R02_cache_records_excluded': old_cache_records,
                                       'old_same_R02_cache_distinct_ids_excluded': len(old_cache_ids)},
              'training_union': [str(p.relative_to(ROOT)) for p in TRAINING],
              'Mu_primitive_guarantees': {'training_union_overlap': 0, 'dev_vs_confirmation_overlap': 0,
                                         'all_historical_primitive_disjoint': False,
                                         'explanation': 'Previously exposed nontraining Mu primitive reuse is possible; all-known primitive exclusion leaves essentially no3/4-hop cohort. No claim of full primitive novelty.'},
              'confirmation_policy': 'Sealed after metadata-only selection; no adaptive candidate edits from confirmation outcomes. Root controller must freeze chosen candidate and use same controls before opening outcomes.',
              'unseen_scope': 'No exact ID or normalized text in inventoried sources; not a complete historical exposure ledger, paraphrase decontamination or pretrained-model decontamination guarantee.',
              'cost': {'new_API_calls': 0, 'remote_calls': 0, 'budget_cap_cny': 300},
              'availability': availability, 'native_sources': source_records}
    freeze(HERE / 'SELECTION_POLICY.frozen.json', policy)
    created = datetime.now(timezone.utc).isoformat()
    paths = [p for p in HERE.rglob('*') if p.is_file() and p.name not in ['MANIFEST.frozen.json', 'README.txt']]
    manifest = {'created_utc': created, 'counts': {cohort: dict(Counter(r['dataset'] for r in rows)) for cohort, rows in selected.items()},
                'metadata_only': True, 'sealed_confirmation': True,
                'files': {str(p.relative_to(HERE)).replace('\\', '/'): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(paths)}}
    freeze(HERE / 'MANIFEST.frozen.json', manifest)
    text = '''新开发90与封存确认150，元数据分层抽样，2026-10-20。\n\n开发集每数据集30题；确认集每数据集50题，具体类型配额见SELECTION_POLICY。\n两组ID和归一化文本无重叠，并排除已盘点训练/开发/测试/示例输入和旧R02缓存。\nMuSiQue同时排除原3500与完整3632训练集来源primitive并集；另保证开发与确认Mu primitive互不重叠。\n不承诺与所有历史非训练Mu primitive均不重叠，不承诺语义改写独立或预训练无污染。\n选题未使用标准答案、支持文档、检索结果、分解结果或F1，也未人工阅读确认题。\n本目录dev90和sealed150均只含query和元数据，无答案。remote_inputs各行严格只有id/query。\nsealed150需在候选配置冻结后作为一次性确认；不可读取其结果后原地调优再声称独立确认。\n只完成离线准备，未执行远端推理或API调用；预算总上限仍为300元。\n哈希与来源记录见MANIFEST、EXCLUSION_SOURCES、SELECTION_POLICY。\n'''
    (HERE / 'README.txt').write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'frozen', 'counts': manifest['counts'], 'training_Mu_primitives': len(training_mu),
                      'Mu_dev_confirmation_primitive_overlap': 0, 'availability': availability, 'new_API_calls': 0}, ensure_ascii=False))


if __name__ == '__main__': main()
