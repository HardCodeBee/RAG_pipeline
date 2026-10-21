"""Freeze the existing model/control, diagnose dev errors, and stage prompt-only trials."""
from pathlib import Path
import collections, datetime, hashlib, json, re, statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'outputs/query_decomposition/multigrain_4b_sft3500_v1_20261017'
OUT = ROOT / 'outputs/query_decomposition' / HERE.name
TRAIN = OLD / 'data_v003_exact_v12/MASTER_3500.jsonl'
DEV = HERE.parent / 'multigrain_4b_sft3500_v1_20261017/dev150.questions.frozen.jsonl'
PROMPT = HERE.parent / 'multigrain_4b_sft3500_v1_20261017/remote_v003/v12.txt'

def rows(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def frozen(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): assert p.read_bytes() == data, f'Frozen artifact changed: {p}'
    else: p.write_bytes(data)

def family(q):
    q = q.lower()
    if q.startswith('which film') and 'director' in q: return 'film_director_comparison'
    if any(x in q for x in ['maternal grand', 'paternal grand', '-in-law']): return 'kinship_composition'
    if any(x in q for x in ['director of', 'composer of', 'performer of']): return 'media_creator_attribute'
    if q.startswith('who ') and any(x in q for x in ['born', 'died', 'older', 'younger', 'lived longer']): return 'person_comparison'
    if q.startswith('which film') or q.startswith('which song'): return 'media_comparison'
    if q.startswith('are '): return 'boolean_comparison'
    if 'father' in q or 'mother' in q: return 'relative_attribute'
    return 'other'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = json.loads((OLD / 'SELECTED_CHECKPOINT.json').read_text(encoding='utf-8'))
    assert selected['selected']['step'] == 165 and selected['selected']['passed'] == 128
    assert sha(PROMPT) == '508709f217ee58ca53d60ecce6001d5e5b0e6d51a858dd4b95b2d6d311864ee8'
    review_path = OLD / 'dev_review/checkpoint-000165.jsonl'
    frozen(OUT / 'control/dev150.questions.frozen.jsonl', DEV.read_bytes())
    frozen(OUT / 'control/checkpoint-000165.review.frozen.jsonl', review_path.read_bytes())
    frozen(OUT / 'control/SELECTION.frozen.json', (OLD / 'SELECTED_CHECKPOINT.json').read_bytes())
    frozen(HERE / 'prompts/b00_v12.txt', PROMPT.read_bytes())
    manifest = {'model': 'Original Qwen3-4B-Instruct-2507 + new ordinary LoRA checkpoint165',
        'remote_package': '/home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package_v003',
        'checkpoint_manifest_sha256': selected['selected']['checkpoint_manifest_sha256'],
        'prompt_sha256': sha(PROMPT), 'train3500_sha256': sha(TRAIN),
        'dev150_sha256': sha(DEV), 'dev_review_sha256': sha(review_path),
        'dev_score': {'passed': 128, 'denominator': 150},
        'formal300_role': 'Original4B-versus-SFT evaluation remains separate; never prompt selection.',
        'frozen_files': {str(p.relative_to(ROOT)): sha(p) for p in [PROMPT, TRAIN, DEV, review_path, OLD / 'SELECTED_CHECKPOINT.json']},
        'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 1024, 'seed': 1729},
        'review': 'Frozen F/S/C/U, current Codex query-only, non-independent, not downstream RAG F1.',
        'policy': 'Do not alter control weights, labels, prompts, selection scores, or review rubric. API calls0; no new training.'}
    frozen(OUT / 'BASELINE_FROZEN.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    train, dev, reviews = rows(TRAIN), rows(DEV), rows(review_path)
    by_id = {x['id']: x for x in reviews}
    categories = {
        'coverage': [4,40,44,48,59,105,119],
        'identification_or_binding': [17,50,61,72,102,104,126],
        'relation_or_attachment': [2,108,117,118,139],
        'unnecessary_need': [60,74,81]}
    labels = {i: name for name, values in categories.items() for i in values}
    failures = [x for x in reviews if x['verdict'] != 'pass']
    assert {x['index'] for x in failures} == set(labels) and len(failures) == 22
    for x in failures:
        x['primary_diagnostic_category'] = labels[x['index']]
        x['literal_semantic_defect_flag'] = 'F' in x['failed_dimensions']
        x['source_wording_caution'] = x['index'] in [4,17,44,48,102,108,118]
        x['scope_note'] = 'Baseline verdict preserved; F/S/C/U failure does not establish downstream answer failure.'
    frozen(OUT / 'diagnosis/ALL22_FAILURES.jsonl', ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in failures).encode())
    stats = {'dataset_types': {}, 'primary_categories': {},
        'overlapping_dimensions': dict(collections.Counter(k for x in failures for k in x['failed_dimensions'])),
        'faithfulness_flagged': sum('F' in x['failed_dimensions'] for x in failures),
        'without_faithfulness_flag': sum('F' not in x['failed_dimensions'] for x in failures),
        'training_sources': dict(collections.Counter(x['source']['dataset'] for x in train)),
        '2wiki_train_types': dict(collections.Counter(x['source'].get('type','unknown') for x in train if x['source']['dataset']=='2wiki')),
        '2wiki_dev_families': dict(collections.Counter(family(x['query']) for x in dev if x['dataset']=='2wiki')),
        '2wiki_train_families': dict(collections.Counter(family(x['query']) for x in train if x['source']['dataset']=='2wiki')),
        '50_successes_exact_binomial_95_lower_assuming_iid': .025 ** (1/50),
        'statistical_caution': 'Template/family dependence makes iid binomial inference optimistic; 50/50 is not the population rate.'}
    for ds in ['hotpotqa','2wiki','musique']:
        stats['dataset_types'][ds] = {}
        for typ in sorted({x['type'] for x in dev if x['dataset']==ds}):
            subset=[x for x in dev if x['dataset']==ds and x['type']==typ]
            stats['dataset_types'][ds][typ]={'n':len(subset),'passed':sum(by_id[x['id']]['verdict']=='pass' for x in subset),
                'median_query_words':statistics.median(len(x['query'].split()) for x in subset)}
    for cat, indices in categories.items():
        stats['primary_categories'][cat]={'count':len(indices),'fraction_of22':len(indices)/22,
            'by_dataset':dict(collections.Counter(x['dataset'] for x in failures if x['index'] in indices))}
    write(OUT / 'diagnosis/STATS.json', stats)
    original = PROMPT.read_text(encoding='utf-8')
    old_shot = 'Input: What is the population of the capital of the country where the author of Book R was born?\nOutput: ["Who wrote Book R?", "Where was the author of Book R born?", "Which country contains the birthplace of the author of Book R?", "What is the capital of the country where the author of Book R was born?"]'
    new_shot = 'Input: What is the population of the city bordering the capital of the province containing Museum Q?\nOutput: ["Which province contains Museum Q?", "What is the capital of the province containing Museum Q?", "Which city borders the capital of the province containing Museum Q?"]'
    assert original.count(old_shot)==1
    coverage = original.replace(old_shot, new_shot)
    old_shot2 = 'Input: Who owns the company where the composer of Song S works?\nOutput: ["Who composed Song S?", "Which company does the composer of Song S work for?"]'
    new_shot2 = 'Input: Where was the American folk band founded in 1983 whose lead singer was Lio formed?\nOutput: ["Which American folk band founded in 1983 had Lio as its lead singer?"]'
    assert original.count(old_shot2)==1
    fidelity = original.replace(old_shot2, new_shot2)
    for name, text in [('c01_adjacent_bridge',coverage),('c02_full_description',fidelity)]:
        frozen(HERE/'prompts'/(name+'.txt'), text.encode())
    protocol = {'phase':'prompt-only development trials; never new training',
        'baseline': 'b00_v12 reused literal dev150 outputs, checkpoint165',
        'variants': {
            'c01_adjacent_bridge':{'changed':'Replace only the first example; retain all instructions and four other examples.',
                'purpose':'Explicitly expose the neighboring-city intermediate; preserve standalone nested descriptions.',
                'diagnostic_targets':[40,119], 'prompt_sha256':sha(HERE/'prompts/c01_adjacent_bridge.txt')},
            'c02_full_description':{'changed':'Replace only the composer-company example with a fully qualified past-singer band example.',
                'purpose':'Retain identifying genre/nationality/year and past-tense relation; no extra rules.',
                'diagnostic_targets':[50,61,72,102,104,126], 'prompt_sha256':sha(HERE/'prompts/c02_full_description.txt')}},
        'queries_sha256':sha(DEV),'n':150,'model_weights_unchanged':True,
        'selection':'Review all150 for each candidate including regressions; choose only a higher full-denominator pass count, then worst-dataset rate, then shorter prompt. Otherwise retain b00.',
        'confirmation':'If a development candidate improves, reserve a new disjoint query-only sample before generating; compare frozen b00 and chosen candidate once. Do not claim generalization from dev150.',
        'excluded_from_selection':'The original frozen final300 test and all its outputs.',
        'no_case_specific_runtime_routing':True,'api_calls':0,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    frozen(HERE/'PROTOCOL_FROZEN.json', (json.dumps(protocol,ensure_ascii=False,indent=2)+'\n').encode())
    frozen(HERE/'queries150.frozen.jsonl',DEV.read_bytes())
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
