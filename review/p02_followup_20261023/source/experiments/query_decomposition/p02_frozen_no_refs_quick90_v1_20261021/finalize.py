"""Offline result and trajectory review; no additional generation."""
from . import run as r
from collections import Counter
from contextlib import closing
import json
import sqlite3


def main():
    r.check()
    report = r.read(r.OUT / 'REPORT.json')
    assert report['status'] == 'completed'
    controls = {v['id']: v['trajectory'] for v in r.rows(r.OUT / 'control90.jsonl')}
    candidates = {}
    for dataset in r.engine.DATASETS:
        path = r.OUT / 'candidate' / dataset / 'checkpoint.sqlite3'
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            candidates.update({k: json.loads(v) for k, v in db.execute("SELECT key,value FROM data WHERE kind='prediction'")})
            assert db.execute("SELECT count(*) FROM requests WHERE status!='success'").fetchone()[0] == 0
    assert len(candidates) == len(controls) == 90
    counts = {role: Counter() for role in ['ordinary', 'goal']}
    node_changes, score_changes = [], []
    for cid, before in controls.items():
        after = candidates[cid]
        for a, b in zip(before['nodes'], after['nodes']):
            assert a['request'] == b['request']
            c = counts[a['role']]
            c['planned'] += 1
            flags = {
                'answer_changed': a['answer'] != b['answer'],
                'rewrite_changed': a['rewritten_query'] != b['rewritten_query'],
                'reader_docs_changed': [d['doc_id'] for d in a['reader_packet']['docs']] != [d['doc_id'] for d in b['reader_packet']['docs']],
                'prior_answers_changed': a['prior_complete_answers'] != b['prior_complete_answers'],
                'original_docs_changed': [d['doc_id'] for d in a['original_retrieval']['docs']] != [d['doc_id'] for d in b['original_retrieval']['docs']],
            }
            for key, flag in flags.items():
                c[key] += int(flag)
            if any(flags.values()):
                node_changes.append({'id': cid, 'role': a['role'], 'step_id': a['step_id'], 'question': a['request'],
                    **flags, 'baseline_answer': a['answer'], 'candidate_answer': b['answer'],
                    'baseline_rewrite': a['rewritten_query'], 'candidate_rewrite': b['rewritten_query']})
    for row in r.rows(r.OUT / 'PAIRED_DETAILS.jsonl'):
        if abs(row['delta_F1']) > 1e-12:
            score_changes.append({**row, 'question': controls[row['id']]['question'],
                'node_changes': [v for v in node_changes if v['id'] == row['id']]})
    summary = {role: dict(c) for role, c in counts.items()}
    assert summary['ordinary']['planned'] == 176 and summary['goal']['planned'] == 90
    assert summary['ordinary']['original_docs_changed'] == summary['goal']['original_docs_changed'] == 0
    r.save(r.OUT / 'CHANGE_REVIEW.json', {'reference': 'original frozen P02', 'summary': summary,
        'score_changed': score_changes, 'node_changes': node_changes,
        'note': 'Text changes are not error counts. Compare retrieved evidence before judging correctness.'})
    lines = ['# 原冻结P02：两类回答prompt统一移除编号及结果引用', '',
        '固定90题，各数据集30题。对照为原冻结P02（F1 76.6889%），不是此前损伤的完整新版。', '',
        '只删除Resolve each #N解析句、历史/当前问题编号，并将两处教学示例结果引用改为完整描述；其余规则、示例问答/原文、goal附加规则、重复query、重写与最终提示词、检索及合并、模型和长度限制均保持冻结。', '',
        '原冻结完整请求精确回放作为对照；候选只复用原冻结及本轮成功的完整请求，变更后的载荷进行新推理。子答案变化正常传到后续流程。', '',
        '| 数据集 | 完成 | 冻结F1 | 候选F1 | F1变化（百分点） | 冻结EM | 候选EM | F1改善/持平/退步 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds, value in report['metrics'].items():
        lines.append(f"| {ds} | {value['complete']}/{value['planned']} | {100*value['baseline_F1']:.4f}% | {100*value['candidate_F1']:.4f}% | {100*(value['candidate_F1']-value['baseline_F1']):+.4f} | {100*value['baseline_EM']:.4f}% | {100*value['candidate_EM']:.4f}% | {value['improved']}/{value['tied']}/{value['harmed']} |")
    lines += ['', '以下是逐项差异，不是错误数量。', '', '| 阶段 | 改写query变化 | 阅读文档变化 | 答案文字变化 |', '|---|---:|---:|---:|']
    for role, value in summary.items():
        lines.append(f"| {role} | {value['rewrite_changed']}/{value['planned']} | {value['reader_docs_changed']}/{value['planned']} | {value['answer_changed']}/{value['planned']} |")
    lines += ['', '| 数据集 | 冻结完整流程平均输入/输出tokens | 候选完整流程平均输入/输出tokens |', '|---|---:|---:|']
    for ds, usage in report['new_usage'].items():
        lines.append(f"| {ds} | {usage['baseline_input_tokens_per_query']:.2f} / {usage['baseline_output_tokens_per_query']:.2f} | {usage['candidate_input_tokens_per_complete_query']:.2f} / {usage['candidate_output_tokens_per_complete_query']:.2f} |")
    lines += ['', f"本轮新增返回用量估算费用¥{report['new_estimated_CNY']:.7f}；未核销费用预留¥{report['unknown_reserved_CNY']:.7f}（恢复时保留的原未知请求及HTTP429预留，不是未完成题目的数量）。旧实验记录与未知费用不变。", '',
        '范围：已暴露开发集快速对照，不是独立确认集；没有额外评审模型调用。旧冻结版本保留，候选未激活，主评保持暂停。', '',
        '逐题F1/EM见PAIRED_DETAILS.jsonl；子问题与goal的答案/query差异见CHANGE_REVIEW.json；ordinary_CHANGE.diff及goal_CHANGE.diff展示实际prompt差异。', '']
    (r.OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf8')
    delivery = {'utc': r.utc(), 'status': 'completed', 'planned': 90, 'completed': 90, 'control': 'original frozen P02',
        'protocol_sha256': r.sha(r.HERE / 'PROTOCOL.frozen.json'), 'pins_verified': True,
        'metrics': report['metrics']['overall'], 'stage_differences': summary,
        'new_estimated_CNY': report['new_estimated_CNY'], 'unknown_reserved_CNY': report['unknown_reserved_CNY'],
        'main_evaluation_remains_paused': True, 'candidate_not_activated': True,
        'report': str((r.OUT / 'REPORT.json').relative_to(r.ROOT))}
    r.save(r.OUT / 'FINAL_DELIVERY.json', delivery)
    entry_id = r.HERE.name.upper() + '_COMPLETED'
    assert not any(v.get('id') == entry_id for v in r.rows(r.PLAN / 'DECISIONS.jsonl'))
    with (r.PLAN / 'DECISIONS.jsonl').open('a', encoding='utf8') as handle:
        handle.write(json.dumps({'id': entry_id, **delivery}, ensure_ascii=False) + '\n')
    r.emit({'summary': delivery, 'score_changed': score_changes})


if __name__ == '__main__':
    main()
