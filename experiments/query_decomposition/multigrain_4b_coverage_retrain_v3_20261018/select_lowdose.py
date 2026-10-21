"""Apply the frozen conditional20 gate without changing the rejected40 decision."""
import datetime,json
from prepare import HERE,OUT,sha,freeze
from review import summary,PARENT

def main():
    first=json.loads((HERE/'NOMINEE_FROZEN.json').read_text(encoding='utf-8'));assert first['selected'] is None
    ss=summary('dev60');b=ss[PARENT];t=ss['r03_expanded20'];assert t['n']==60
    gates={'mu_strictly_higher':t['by_dataset']['musique']['passes']>b['by_dataset']['musique']['passes'],
        'overall_not_lower':t['passed']>=b['passed'],
        'controls_net_loss_at_most1':sum(t['by_dataset'][d]['passes']-b['by_dataset'][d]['passes'] for d in ['hotpotqa','2wiki'])>=-1,
        'mu_FS_not_increased':len(t['mu_FS'])<=len(b['mu_FS'])}
    selected='r03_expanded20' if all(gates.values()) else None
    record={'selected':selected,'gates':gates,'dev_summary':ss,'confirmation_used_for_selection':False,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'role':'Disclosed conditional duration refinement after both40 arms failed; not initial two-arm preplan.','amendment_sha256':sha(HERE/'LOWDOSE_AMENDMENT_FROZEN.json'),'parent_and_original_unchanged':True}
    if selected:
        paths=sorted((OUT/'returned').glob('snapshot_*/results/v002/training20/r03_expanded20/checkpoint-final/COMPLETE.json'));assert paths;marker=json.loads(paths[-1].read_text(encoding='utf-8'))
        record.update({'checkpoint':'/home/projects/u7830488/multigrain_4b_coverage_retrain_v3_20261018/results/v002/training20/r03_expanded20/checkpoint-final','checkpoint_manifest_sha256':marker['manifest_sha256'],'confirm_sha256':sha(HERE/'confirm120.questions.frozen.jsonl')})
    freeze(HERE/'NOMINEE_LOWDOSE20_FROZEN.json',record);freeze(OUT/'NOMINEE_LOWDOSE20_FROZEN.json',record);print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
