"""Query/metadata-only group audit and frozen independent dev/formal draw.

Protected project-test questions are not opened. Existing template/exact/family
metadata link their groups; lexical near-duplicate comparison is restricted to
accessible official-dev, actual-train and historical question text.
"""
from collections import Counter, defaultdict
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import argparse
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import ROOT, PLAN, write_json, utc


def sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def template(text):
    wh = {'who','what','where','when','which','how','why','is','was','were','are','did','does','do','in','the','of','a','an'}
    out = []
    for token in re.findall(r'[\w]+|[^\w\s]', text, re.UNICODE):
        key = '<NUM>' if re.fullmatch(r'\d+', token) else '<ENTITY>' if token[:1].isupper() and token.casefold() not in wh else token.casefold()
        if key in {'<ENTITY>', '<NUM>'} and out and out[-1] == key:
            continue
        out.append(key)
    return ' '.join(out)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


class Groups:
    def __init__(self):
        self.parent = []
    def add(self):
        index = len(self.parent); self.parent.append(index); return index
    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]; a = self.parent[a]
        return a
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a,b)] = min(a,b)


def audit(extra_exposures=()):
    manifest_path = ROOT/'outputs/query_decomposition/demand_parser_routes_common_eval_20261010/returned/route2/route2_question_decomposer/data/split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    groups, owner, anchor, blocked = Groups(), {}, {}, defaultdict(set)
    sources = []
    def log(path, purpose):
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'purpose': purpose})
    log(manifest_path, 'metadata anchors; no protected query/target bodies')
    for row in manifest['records']:
        index = anchor.setdefault(row['group_id'], len(groups.parent))
        if index == len(groups.parent): groups.add()
        keys = [('exact', row['query_sha256']), ('template', row['template_sha256'])] + [('family', key) for key in row['family_key_hashes']]
        for key in keys:
            if key in owner: groups.union(index, owner[key])
            else: owner[key] = index
        if row['project_assignment'] in {'project_dev', 'project_test'}:
            blocked[index].add(row['project_assignment'] + '_metadata')
    accessible, official = [], []
    def add_query(qid, text, family, reason=None, kind=None):
        index = groups.add()
        for key in [('exact', sha(norm(text))), ('template', sha(template(text)))] + [('family', sha(f)) for f in family]:
            if key in owner: groups.union(index, owner[key])
            else: owner[key] = index
        row = {'qid': qid, 'question': text.strip(), 'index': index, 'type': kind}
        accessible.append(row)
        if reason: blocked[index].add(reason)
        return row
    metadata = ROOT/'analysis/hotpotqa_difficulty_retrieval_20261002/source_labels'
    path = metadata/'validation_labels.jsonl';log(path, 'official dev query/type metadata only')
    for row in rows(path):
        official.append(add_query(row['id'], row['question'], ['HOTPOT:' + row['id']], kind=row['type']))
    path = manifest_path.parent/'runtime_qwen_train.jsonl';log(path, 'actual training query-only projection')
    for row in rows(path):
        item = add_query(row['id'], row['query'], [], 'actual_4b_training')
        groups.union(item['index'], anchor[row['group_id']])
    historical_paths = [
        ROOT/'experiments/query_decomposition/auxiliary_qa_rag_v1/inputs_frozen/queries600.jsonl',
        ROOT/'experiments/query_decomposition/api_qdmr_suitability_v1/data/queries.jsonl',
        ROOT/'analysis/hotpotqa_query_taxonomy_validation_20261007/samples.jsonl',
    ]
    historical_ids = set()
    for path in historical_paths:
        log(path, 'prior decomposition/RAG or question-curation exposure')
        for row in rows(path):
            qid = row['qid'];historical_ids.add(qid)
            add_query(qid, row['question'], ['HOTPOT:' + qid], 'historical_query_exposure')
    path = ROOT/'outputs/query_decomposition/bpb_bart_hotpot_suitability_v1/bpb_bart_hotpot_suitability_v1/data/sample_metadata.jsonl'
    log(path, 'original 1200-query baseline-decomposition exposure IDs only')
    past_ids = {r['qid'] for r in rows(path)}
    missing_ids = past_ids - historical_ids
    log(metadata/'train_labels.jsonl', 'look up query-only text for previously exposed IDs')
    for row in rows(metadata/'train_labels.jsonl'):
        if row['id'] in missing_ids:
            add_query(row['id'], row['question'], ['HOTPOT:' + row['id']], 'historical_1200_decomposition_exposure')
            missing_ids.remove(row['id'])
    if missing_ids:
        raise ValueError('Historical exposure IDs missing query-only source')
    for path in extra_exposures:
        path = Path(path)
        log(path, 'additional method-development exposure; query-only projection')
        for row in rows(path):
            qid = row.get('qid', row.get('question_id'))
            text = row.get('question', row.get('question_text'))
            if not isinstance(qid, str) or not isinstance(text, str):
                raise ValueError('Additional exposure lacks ID/question')
            add_query(qid, text, ['HOTPOT:' + qid], 'additional_method_development_exposure')
    grams, postings = [], defaultdict(list)
    for index, row in enumerate(accessible):
        tokens = re.findall(r'\w+', norm(row['question']))
        gram = {tuple(tokens[i:i+3]) for i in range(len(tokens)-2)}
        grams.append(gram)
        for g in gram: postings[g].append(index)
    near_pairs, near_edges = 0, 0
    for i, gram in enumerate(grams):
        candidates = set()
        for g in sorted(gram, key=lambda g: (len(postings[g]), g))[:5]:
            if len(postings[g]) <= 200: candidates.update(j for j in postings[g] if j < i)
        for j in sorted(candidates):
            old = grams[j]
            if not old or min(len(old),len(gram))/max(len(old),len(gram)) < .82: continue
            near_pairs += 1
            if len(gram & old)/len(gram | old) >= .82:
                groups.union(accessible[i]['index'], accessible[j]['index']);near_edges += 1
    root_reasons = defaultdict(set)
    for index, reasons in blocked.items(): root_reasons[groups.find(index)].update(reasons)
    all_members = defaultdict(list)
    for row in accessible: all_members[groups.find(row['index'])].append(row['qid'])
    names = {root: 'fg:' + sha('\n'.join(sorted(set(ids))))[:24] for root, ids in all_members.items()}
    eligible, excluded = [], []
    for row in official:
        root = groups.find(row['index'])
        item = {k: row[k] for k in ['qid','question','type']}
        item['group_id'] = names[root]
        reasons = sorted(root_reasons[root])
        if reasons: excluded.append({'qid': row['qid'], 'group_id': names[root], 'reasons': reasons})
        else: eligible.append(item)
    report = {'utc': utc(), 'official_total': len(official), 'eligible_questions': len(eligible),
              'eligible_type_counts': dict(Counter(r['type'] for r in eligible)),
              'eligible_groups': len({r['group_id'] for r in eligible}),
              'eligible_group_representative_types': dict(Counter(v[0]['type'] for v in group_by(eligible).values())),
              'excluded_questions': len(excluded), 'overlapping_reason_counts': dict(Counter(reason for e in excluded for reason in e['reasons'])),
              'accessible_near_pairs_checked': near_pairs, 'accessible_near_edges': near_edges,
              'sources': sources, 'sealed_project_test_question_files_opened': False,
              'historical_scope': 'old 600 includes all current-method answer/prompt optimization; original BART/API1200 and question-taxonomy sample additionally excluded',
              'limitations': ['No protected project-test text is inspected; its exact/template/family metadata are checked, lexical-near matches beyond metadata cannot be ruled out.',
                             'Grouping is a conservative lexical heuristic, not semantic paraphrase or pretraining decontamination.',
                             'Official dev has prior project-wide retrieval analysis; only method-selection/answer/decomposition exposure is excluded.'],
              'eligibility_policy': 'exclude any group joined to actual training, protected project dev/test or historical exposure; no answer/F1-dependent exclusion',
              'additional_exposure_sources': [str(Path(p).relative_to(ROOT)) for p in extra_exposures],
              'status': 'query_metadata_group_audit_completed_with_declared_limits'}
    return eligible, excluded, report


def group_by(rows_):
    result = defaultdict(list)
    for row in rows_: result[row['group_id']].append(row)
    return result


def allocate(n, counts):
    total = sum(counts.values());raw = {k:n*v/total for k,v in counts.items()}
    result = {k:int(v) for k,v in raw.items()}
    for key in sorted(raw, key=lambda k: (-(raw[k]-result[k]), k))[:n-sum(result.values())]: result[key] += 1
    return result


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--freeze', action='store_true');args = parser.parse_args()
    if (PLAN/'inputs_frozen').exists():
        raise FileExistsError('Sample and its eligibility report are frozen; do not redraw or overwrite')
    eligible, excluded, report = audit()
    write_json(PLAN/'preparation/GROUP_EXPOSURE_AUDIT.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'sources','limitations','historical_scope'}}, ensure_ascii=False), flush=True)
    if not args.freeze: return
    output = PLAN/'inputs_frozen'
    if output.exists(): raise FileExistsError('Use existing frozen sample; no redraw/overwrite')
    plan = json.loads((PLAN/'PLAN_CONFIG.json').read_text(encoding='utf8'))
    seed = plan['samples']['sample_seed'];drawn = set();selected = {}
    for split, n in [('dev', plan['samples']['development_target']), ('formal', plan['samples']['formal_target'])]:
        available = [r for r in eligible if r['group_id'] not in drawn]
        quotas = allocate(n, Counter(r['type'] for r in available))
        representatives = [min(v, key=lambda r: sha(f'{seed}:{split}:item:'+r['qid'])) for v in group_by(available).values()]
        records = []
        for kind, count in sorted(quotas.items()):
            candidates = sorted([r for r in representatives if r['type']==kind], key=lambda r:sha(f'{seed}:{split}:group:'+r['group_id']))
            if len(candidates) < count: raise ValueError('Insufficient independent eligible groups for fixed quota')
            records.extend(candidates[:count])
        records.sort(key=lambda r:sha(f'{plan["samples"]["execution_seed"]}:{split}:'+r['qid']))
        drawn.update(r['group_id'] for r in records); selected[split] = records
    import pyarrow.parquet as pq
    gold_path = PLAN/'preparation/offline_gold_source/validation.parquet'
    gold = {r['id']:r for r in pq.read_table(gold_path, columns=['id','question','answer','supporting_facts']).to_pylist()}
    qrels = defaultdict(list)
    for line in (ROOT/'data/beir/hotpotqa/qrels/test.tsv').read_text().splitlines()[1:]:
        qid, doc, score = line.split('\t')
        if int(score)>0: qrels[qid].append(doc)
    output.mkdir()
    hashes = {}
    for split, records in selected.items():
        questions = [{'qid':r['qid'], 'question':r['question']} for r in records]
        references = []
        for r in records:
            g = gold[r['qid']]
            assert g['question'].strip() == r['question']
            references.append({'qid':r['qid'], 'question':r['question'], 'answer':g['answer'], 'type':r['type'],
                               'group_id':r['group_id'], 'supporting_facts':g['supporting_facts'], 'positive_doc_ids':qrels[r['qid']], 'generation_input_forbidden':True})
        for name, data in [(f'{split}.questions.jsonl',questions), (f'{split}.references.local.jsonl',references)]:
            path = output/name;path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data), encoding='utf8');hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = {'utc':utc(), 'seed':seed, 'sample_selection_used_answers':False, 'files_sha256':hashes,
              'splits':{s:{'count':len(rs),'type_counts':dict(Counter(r['type'] for r in rs)), 'groups':len({r['group_id'] for r in rs})} for s,rs in selected.items()},
              'groups_disjoint':True, 'eligibility_report_sha256':hashlib.sha256((PLAN/'preparation/GROUP_EXPOSURE_AUDIT.json').read_bytes()).hexdigest(),
              'gold_source_sha256':hashlib.sha256(gold_path.read_bytes()).hexdigest(), 'scope':'independent sample from eligible official dev; public test answers unavailable'}
    write_json(output/'SAMPLES_FROZEN.json',frozen)
    print(json.dumps(frozen, ensure_ascii=False))


if __name__ == '__main__': main()
