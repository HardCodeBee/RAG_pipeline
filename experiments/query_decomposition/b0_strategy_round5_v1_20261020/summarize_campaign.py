"""Summarize completed final-reader trials; no inference or candidate tuning."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import OUTPUT


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    names = [f'b0_strategy_round{i}_v1_20261020' for i in [3, 4, 5]]
    records = []
    labels = {
        'q03_question_last': '仅将原题移到材料末尾',
        'e03_goal_evidence': '仅补入一篇已有goal检索原文',
        'r04_scope_contract': '仅替换一条原题目标约定',
        's04_scope_plus_evidence': '目标约定与补原文的组合',
        'e05_example_role_headers': '仅对齐四个示例的辅助角色标题',
    }
    parent = None
    for name in names:
        result = read(OUTPUT / name / 'FINAL_RESULTS.frozen.json')
        selection = read(OUTPUT / name / 'SELECTION.frozen.json')
        assert selection['selected'] == 'parent_p02'
        parent = result['parent300_F1'] if parent is None else parent
        assert result['parent300_F1'] == parent
        for arm, row in result['arms'].items():
            assert row['scorable'] == 300
            records.append({'arm': arm, 'change': labels[arm], **row})
    dev = read(OUTPUT / 'b0_strategy_round2_v1_20261020/fresh_dev90/SCORES.progress.json')
    assert dev['paired']['n_matched'] == 90
    confirmation = OUTPUT / 'b0_strategy_round2_v1_20261020/fresh_confirm150'
    status = read(confirmation / 'CONTROLLER_STATUS.json')
    progress = read(confirmation / 'PROGRESS.json') if (confirmation / 'PROGRESS.json').exists() else None
    stamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect((OUTPUT / 'budget.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        known, reserved = db.execute('SELECT coalesce(sum(actual),0), coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges').fetchone()
        trials = {}
        for name in names:
            row = db.execute('SELECT count(*),coalesce(sum(actual),0),coalesce(sum(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM charges WHERE namespace LIKE ?', (name+'%',)).fetchone()
            trials[name] = dict(zip(['physical_submissions', 'known_cny', 'reserved_cny'], row))
    report = {'utc': stamp, 'selected': 'p02_role_headers', 'sample_role': 'Repeatedly exposed development300; no independent confirmatory claim',
              'parent_F1': parent, 'trials': records, 'fresh90': dev,
              'confirmation_status_only': {'controller': status, 'progress': progress,
                                           'candidate_unchanged': 'p02_role_headers', 'answers_and_scores_used': False},
              'budget': {'known_cny': known, 'unknown_reserved_cny': reserved, 'occupied_cny': known+reserved,
                         'cap_cny': 300, 'trial_namespaces': trials},
              'scope': 'All5 trials completed300; all rejected. Existing g01/f01/p02 and every trial retained. Confirmation150 runs the original fixed p02 plan; it is not a pool for selecting these trials.'}
    here = OUTPUT / names[-1]
    (here / 'CAMPAIGN_SUMMARY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    lines = ['B0候选保留及后续五项优化汇总', 'UTC快照：'+stamp, '',
             '冻结保留：p02_role_headers。以下均为反复使用的开发300题，不作为独立正式测试。',
             '配置\t改变\t完成\tAnswer F1\t相对p02百分点\t升/平/降',
             f'p02_role_headers\t冻结父版本\t300/300\t{100*parent:.5f}%\t0\t0/300/0']
    for row in records:
        lines.append(f"{row['arm']}\t{row['change']}\t300/300\t{100*row['F1']:.5f}%\t{100*row['delta_parent']:+.5f}\t{row['up']}/{row['tie']}/{row['down']}")
    pair = dev['paired']
    lines += ['', '已冻结的p02新90题开发验证：',
              f"g01={100*pair['parent_F1']:.5f}%，p02={100*pair['candidate_F1']:.5f}%，差值{100*pair['delta_F1']:+.5f}个百分点；升/平/降={pair['up']}/{pair['tie']}/{pair['down']}。",
              '配对95%区间跨0；不能称为已证明稳定的总体提升。',
              '变化复核：1题正确原文提取恢复，1题同一事实措辞缩短收益，1题相同过宽列表的词面失分。',
              '', '固定150题确认：',
              '远端R02分解150/150完成，0格式回退；本地完整RAG仍在运行。',
              '先完整g01，再固定p02并复用精确前缀；不改候选、不以确认结果调优。',
              '当前进度只读快照：'+json.dumps(progress, ensure_ascii=False),
              '', '费用为返回用量估算，未核对账单。未知请求预留保留；增量最终回答实验费用不等于完整流程冷启动费用。']
    for name, value in trials.items():
        lines.append(f"{name}：提交{value['physical_submissions']}次，已知¥{value['known_cny']:.6f}，未知预留¥{value['reserved_cny']:.6f}。")
    lines.append(f'累计已知¥{known:.6f}＋未知预留¥{reserved:.6f}＝占用¥{known+reserved:.6f}，上限¥300。')
    (here / 'CAMPAIGN_SUMMARY.txt').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps({'selected': report['selected'], 'trials_complete': len(records), 'budget': report['budget']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
