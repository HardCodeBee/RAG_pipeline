"""Select only from fully reviewed dev150; test300 cannot be a selection input."""
import argparse,collections,json
from pathlib import Path
from reaudit import HERE,OUT,rows,sha,write

def choose(table):
    assert table
    return max(table,key=lambda r:(r['passed'],r['worst_dataset_rate'],-r['step']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('returned',type=Path);ap.add_argument('--reviews',type=Path,required=True);a=ap.parse_args()
    state=json.loads((a.returned/'train/STATE.json').read_text())
    assert state['status'] in ['completed','early_stopped'],'Incomplete training cannot select a final checkpoint'
    expected_steps=[s for s in [110,165,220,275,330] if s<=state['global_step']]
    assert len(expected_steps)>=3
    dev={r['id']:r for r in rows(HERE/'dev150.questions.frozen.jsonl')};table=[]
    for step in expected_steps:
        name=f'checkpoint-{step:06d}'
        predictions=[json.loads(p.read_text()) for p in sorted((a.returned/'train/dev_predictions'/name).glob('*.json'))]
        reviews=rows(a.reviews/(name+'.jsonl'))
        assert len(predictions)==len(reviews)==len(dev)==150
        by_id={r['id']:r for r in predictions};judged={r['id']:r for r in reviews}
        assert set(by_id)==set(judged)==set(dev)
        counts=collections.Counter()
        for qid,r in judged.items():
            pred=by_id[qid];assert pred['query']==dev[qid]['query']
            assert r['raw_output']==pred['raw_output'],'Review must name literal reviewed output'
            assert r['verdict'] in ['pass','fail','uncertain'] and r['reason']
            if r['verdict']=='pass':assert pred['format_valid'] and not pred['truncated'] and not pred.get('generation_error') and not pred['explicit_references']
            if r['verdict']=='pass':counts[dev[qid]['dataset']]+=1
        passed=sum(counts.values())
        complete=a.returned/'train/checkpoints'/name/'COMPLETE.json'
        table.append({'step':step,'passed':passed,'denominator':150,'by_dataset_pass':{d:counts[d] for d in ['hotpotqa','2wiki','musique']},'worst_dataset_rate':min(counts[d]/50 for d in ['hotpotqa','2wiki','musique']),'checkpoint_manifest_sha256':json.loads(complete.read_text())['manifest_sha256'],'dev_review_sha256':sha(a.reviews/(name+'.jsonl'))})
    selected=choose(table)
    write(OUT/'SELECTED_CHECKPOINT.json',{'selection_uses_final_eval300':False,'criterion':['highest_dev_complete_pass_count','highest_worst_dataset_pass_rate','earliest_step'],'selected':selected,'all_candidates':table,'returned_training':str(a.returned.resolve()),'dev_questions_sha256':sha(HERE/'dev150.questions.frozen.jsonl'),'review_limit':'Current Codex query-only F/S/C/U, not independent human gold.'})
    print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
