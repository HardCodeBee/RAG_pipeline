"""Package post-generation qualitative evidence; never changes runner inputs."""
from collections import Counter
import json
from . import run as r


def main():
    r.check()
    changes=r.rows(r.OUT/'CHANGED.local.jsonl')
    inputs={x['id']:x for x in r.rows(r.HERE/'INPUTS60.frozen.jsonl')}
    refs={x['id']:x for x in r.rows(r.HERE/'REFERENCES60.local.frozen.jsonl')}
    results={(x['arm'],x['id']):x for x in r.rows(r.OUT/'DETAILS.local.jsonl')}
    notes=r.rows(r.OUT/'CASE_REVIEW.local.jsonl') if (r.OUT/'CASE_REVIEW.local.jsonl').exists() else []
    needed={(x['pair'],x['id']) for x in changes}
    actual={(x['pair'],x['id']) for x in notes}
    assert len(actual)==len(notes) and actual<=needed
    focal=r.rows(r.OUT/'FOCAL_REVIEW.local.jsonl') if (r.OUT/'FOCAL_REVIEW.local.jsonl').exists() else []
    changed_ids={x['id'] for x in changes}
    ids=changed_ids|{x['id'] for x in focal}
    native_path=r.ROOT/'data/multihop_comprehensive_v1_20261015/musique/native/musique_ans_v1.0_dev.jsonl'
    native={}
    for line in native_path.open(encoding='utf8'):
        x=json.loads(line);i='musique:'+x['id']
        if i in ids:
            native[i]={k:v for k,v in x.items() if k!='paragraphs'}
            native[i]['supporting_paragraphs']=[p for p in x['paragraphs'] if p.get('is_supporting')]
    bundle=[dict(id=i,input=inputs[i],reference=refs[i],
        variants={a:results[(a,i)] for a in r.templates.ARMS if (a,i) in results},
        changes=[{k:v for k,v in x.items() if k not in ('variants','ordinary_nodes')} for x in changes if x['id']==i],
        notes=[x for x in notes+focal if x['id']==i],native_diagnostic_only=native.get(i)) for i in sorted(ids)]
    r.jsonl(r.OUT/'CASE_EVIDENCE.local.jsonl',bundle)
    summary=dict(changed_final_pairs=len(changes),changed_cases=len(changed_ids),reviewed_pairs=len(notes),focal_cases=len(focal),evidence_cases=len(ids),
        all_changed_final_pairs_reviewed=actual==needed,
        categories=dict(Counter(x['category'] for x in notes)),
        review='Local Codex inspection informed by scores; not blinded or independent human labels; no external judge',
        native_gold_usage='Offline explanation after generation only; never supplied to model',
        native_source=str(native_path.relative_to(r.ROOT)),native_sha256=r.sha(native_path))
    r.save(r.OUT/'QUALITATIVE_REVIEW.json',summary)
    lines=['# 三个独立改动：所有改答案例与完整goal输出','',
        '覆盖G0→A、GF→P、GF→R的全部最终文本变化，包括分数不变的改答；另外记录促成本轮改动的重点案例，包括没有改答者。原生分数不因事后解释而改写。完整冻结检索包、普通QA和本地数据集诊断证据见CASE_EVIDENCE.local.jsonl。','']
    for item in bundle:
        lines+=['## '+item['id'],'',item['input']['question'],'','参考答案：'+str(item['reference']['answer']),'']
        for n in item['notes']:lines += [n['pair']+' / '+n['category']+': '+n['note'],'']
        for a,v in item['variants'].items():
            lines+=['### '+a,'','Goal完整输出：','',v.get('goal_text',''),'','Final：'+str(v.get('prediction')),
                '',f"F1={v['metrics']['f1']:.6f}; EM={v['metrics']['em']:.6f}" if v.get('metrics') else 'Technical hold','']
    text='\n'.join(lines)+'\n'
    (r.OUT/'CASE_NOTES.md').write_text(text,encoding='utf8');(r.REVIEW/'CASE_NOTES.md').write_text(text,encoding='utf8')
    print(json.dumps(summary,ensure_ascii=True))


if __name__=='__main__':main()
