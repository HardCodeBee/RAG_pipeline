"""Optional mechanical adapter; retain generated output and score it separately."""
import json,re,unicodedata
from prepare import HERE,OUT,rows,write,freeze,sha

def key(s):
    # Preserve internal punctuation, words and relation order. No paraphrase matching.
    return ' '.join(unicodedata.normalize('NFKC',s).casefold().split()).rstrip(' ?!.')

def adapt(query,array):
    kept=[];removed=[];seen=set();root=key(query)
    for index,q in enumerate(array,1):
        normalized=key(q)
        if normalized==root:removed.append({'index':index,'question':q,'reason':'literal_original_query_already_executed_separately'})
        elif normalized in seen:removed.append({'index':index,'question':q,'reason':'literal_duplicate_supporting_question'})
        else:kept.append(q);seen.add(normalized)
    return kept,removed

def main():
    freeze(HERE/'EXACT_DEDUP_POLICY_FROZEN.json',{
        'purpose':'The original query is separately retrieved/answered. Eliminate only literal duplicate work; not a semantic repair model or filter.',
        'normalization':'NFKC,casefold,collapse whitespace,strip final ?!. only; preserve all internal punctuation, words and order.',
        'valid_inputs_only':'JSON string arrays with no explicit references. Invalid generations remain invalid.',
        'review':'Original generated outputs/grades unchanged. For adapter grades, inherit the raw grade if unchanged; inherit locked baseline if resulting array matches; otherwise inspect every changed result under the same rubric.',
        'comparison':'Show raw and processed outcomes separately. Same Mu/macro/control/F-or-S gates; prefer raw configuration on equal scores. Extra model inference0.',
        'selection_uses_unseen_t02_outputs':False,
        'confirmation':'If a processed nominee qualifies, score raw baseline versus that frozen processed configuration on untouched120; retain the raw nominee result too. Do not pool with targeted rare diagnostic.',
    })
    counts={}
    for name in ['b00_v12','t01_original_mix','t02_mu50']:
        folder=OUT/'raw_screen'/name
        if len(list(folder.glob('[0-9][0-9][0-9].json')))!=90:continue
        results=[]
        for p in sorted(folder.glob('[0-9][0-9][0-9].json')):
            r=json.loads(p.read_text(encoding='utf-8'));original=r['parsed']
            if r['format_valid'] and not r['explicit_references']:
                adapted,removed=adapt(r['query'],original)
            else:adapted,removed=original,[]
            results.append({'index':r['index'],'id':r['id'],'dataset':r['dataset'],'query':r['query'],'generated_output':r['raw_output'],'generated_array':original,'adapted_array':adapted,'removed':removed,'source_sha256':sha(p),'fresh_inference':False})
        write(OUT/'processed_screen'/(name+'.json'),results)
        counts[name]={'changed_cases':sum(bool(r['removed']) for r in results),'changed_indices':[r['index'] for r in results if r['removed']],'removed_items':sum(len(r['removed']) for r in results)}
    write(OUT/'processed_screen/COUNTS.json',counts);print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
