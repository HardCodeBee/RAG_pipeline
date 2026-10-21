"""Offline report; no model or retrieval calls."""
from collections import Counter
import json
from . import experiment as e


def main():
    result = e.read(e.OUT / 'FINAL_RESULTS.frozen.json')
    selected = e.read(e.OUT / 'SELECTION.frozen.json')
    original = e.read(e.OUTPUT / 'baselines_same300_v1_20261020/FINAL_RESULTS.frozen.json')
    old = {r['method']: r for r in original['ranking']}
    previous = e.read(e.PREV_O / 'FINAL_RESULTS.frozen.json')
    p1 = previous['arms']['C_final_readout_P1']
    lines = [
        'B0有针对性的逐项优化：结果与边界', '',
        '已完成前次六项消融的源码/实际回执核查，再执行两项独立新候选。原B0、S02与前次六项结果均未修改。',
        '当前300题已参与开发诊断，以下只作开发候选选择，不能当作独立泛化验证。', '',
        '一、前次六项独立消融',
        'B0：67.36469%。C08整体替换65.20588%；c02改写67.55317%；P1最终readout68.09326%。',
        'D完整缺答精确过滤、E短句整理删除、F截断回退在原B0的300题887节点均未触发，并非三个有触发证据的无效策略。',
        'P1=11升1降；区间[-0.03,+1.67]个百分点跨0。16→Sixteen单题贡献约45.8%净提升，仍保留原始评分与全部题目。', '',
        '二、新候选与预设门槛',
        '父候选B0_P1_parent精确复用原C组300题。g01只给goal补既有GOAL_RULE；f01只在P1最终读出前加一句按事实核对辅助结论。两项相互独立，没有同时叠加。',
        '先按题目ID哈希固定100题（33/33/34），不按成败选题；完整100题平均F1严格超过父版才扩到完整300题。先导与扩展仍属开发数据，不能称为新留出集。', '',
        '三、先导100题结果（仅比较相同100题）',
        '版本\t题数\t最终F1\t同题P1父F1\t相对父版百分点\t升/平/降\t是否按门槛扩300',
    ]
    for arm in e.ARMS:
        p = e.read(e.OUT / (arm + '.PILOT.frozen.json'))
        lines.append(f"{arm}\t{p['scorable']}\t{100*p['F1']:.5f}%\t{100*p['same_parent_F1']:.5f}%\t{100*p['delta_parent']:+.5f}\t{p['up']}/{p['tie']}/{p['down']}\t{p['expand']}")
    lines += ['', '四、完整300题可比结果', '版本\t题数\t总体F1\tHotpotQA\t2Wiki\tMuSiQue']
    for arm in ['B0_legacy_multigrain_R02', 's02_child4', 'b03_react_dense']:
        r = old[arm]
        lines.append(f"{arm}\t300\t{100*r['f1']:.5f}%\t" + '\t'.join(f"{100*r['by_dataset'][ds]['f1']:.5f}%" for ds in ['hotpotqa','2wiki','musique']))
    lines.append('B0_P1_parent\t300\t' + f"{100*p1['F1']:.5f}%\t" + '\t'.join(f"{100*p1['by_dataset'][ds]['F1']:.5f}%" for ds in ['hotpotqa','2wiki','musique']))
    for arm, r in result['arms'].items():
        if r['scorable'] != 300:
            continue
        lines.append(f"{arm}\t300\t{100*r['F1']:.5f}%\t" + '\t'.join(f"{100*r['by_dataset'][ds]['F1']:.5f}%" for ds in ['hotpotqa','2wiki','musique']))
        lines.append(f"  对P1父版：{100*r['delta_parent']:+.5f}个百分点；分层配对95%区间[{100*r['paired95'][0]:+.5f},{100*r['paired95'][1]:+.5f}]；{r['up']}升/{r['tie']}平/{r['down']}降。")
    lines += ['', '开发候选选择：' + selected['selected'] + f"；F1={100*selected['F1']:.5f}%。",
              '此选择仅根据预先固定的开发集门槛，不表示已通过新题验证，不自动替换原冻结B0/S02，也不保证事实错误减少。', '',
              '五、策略与事实风险',
              '提示只包含通用组合一致性约定，没有题目专用实体、答案、数字或数据集分支。年龄题只是发现组合机制的动机；候选仍可能无法修复该题，必须如实记录。',
              'g01先导中出现：两山共有国家集合判断修复、别名压缩得分提高，也出现错误Zero years恰好与about400years共享years而得分。不能把全部F1提升表述成事实改善；新增GOAL_RULE并未保证阻断未知变零。',
              '原文已支持更多关系而答案变长也可能降低F1；涉及定义压缩、数字拼写、别名的变化必须与语义改变分开描述，原评分仍保留。', '',
              '六、组件与流程的后续优先级',
              '1. 先冻结本轮开发候选，在未参与本轮优化的新query上与原B0及P1父版成对验证；不按验证得分回调同一版本。',
              '2. 普通子回答若再优化，只修改部分事实保留要求，保留既有4个例子，不整换C08、不增加状态字段、不宽泛删除None/No/0。该改动会改变后续检索，必须完整串联看最终F1。',
              '3. 利用已有role区分ordinary事实问答与goal整体候选结论，是尚未验证的轻量呈现候选；不与本轮两项混合。',
              '4. c02只提高0.19个百分点，不因组件成绩更好直接堆叠；D/E/F暂作边界保护，不把未触发机制当作提升来源。', '',
              '七、执行与费用',
              '新检索调用：0；新远端任务：0；新训练：0。重用完整B0原始检索和成功普通节点，不从中间回复事后按gold抽答案。',
              '先导前沙箱DNS失败时付费调用为0，原空round保留，以独立执行命名空间恢复；模型、提示和预算不因网络恢复而改变。',
              '恢复故障注入已通过：goal成功/final失败，以及goal+formatter成功/final失败，恢复均只新增final，不重跑成功前缀。模拟回执在recovery_contract隔离，未混入正式结果。',
              f"新增已返回用量估算 ¥{result['budget']['new_known_cny']:.6f}；新增未核销预留 ¥{result['budget']['new_reserved_cny']:.6f}；新提交 {result['budget']['new_submissions']} 次。",
              f"本次汇总时累计占用 ¥{result['budget']['global_occupied_cny']:.6f}/300，其中旧未知预留仍保留。价格按返回token估算，实际账单未核对。", '',
              '方案/准确提示：experiments/query_decomposition/b0_targeted_optimization_v2_20261020/OPTIMIZATION_PLAN.txt、PROTOCOL.frozen.json。',
              '全部回执与实际prompt：本目录rounds/*/methods及rounds/*/cases。',
              '逐题得分：PAIRED_DETAILS.jsonl；冻结汇总：FINAL_RESULTS.frozen.json；开发选择：SELECTION.frozen.json。',
              '先导机制审阅：ANALYSIS.g01_pilot100.v001.json。',
              '泛化约束：experiments/query_decomposition/b0_targeted_optimization_v2_20261020/GENERALIZATION_REQUIREMENT.v001.txt。']
    report = '\n'.join(lines) + '\n'
    path = e.OUT / 'REPORT_FINAL.txt'
    if path.exists():
        assert path.read_text(encoding='utf8') == report
    else:
        path.write_text(report, encoding='utf8')
    e.log('write_offline_report', selected=selected['selected'], new_api_calls=0,
          report=str(path.relative_to(e.ROOT)), report_source_sha256=e.sha(__file__))
    print(json.dumps({'report': str(path), 'selected': selected, 'budget': result['budget']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
