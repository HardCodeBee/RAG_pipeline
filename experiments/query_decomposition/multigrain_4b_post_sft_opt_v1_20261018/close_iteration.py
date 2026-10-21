"""Close the bounded prompt trials; report sealed-test diagnostics without retuning."""
from pathlib import Path
import collections
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'outputs/query_decomposition' / HERE.name
SFT = ROOT / 'outputs/query_decomposition/multigrain_4b_sft3500_v1_20261017'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_status(path):
    if path.exists():
        target = OUT / 'status_history' / (str(path.relative_to(ROOT)).replace('\\', '__').replace('/', '__'))
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    frozen = read(OUT / 'BASELINE_FROZEN.json')
    protected = {p: sha(ROOT / p) == expected for p, expected in frozen['frozen_files'].items()}
    assert all(protected.values())
    confirmation = read(OUT / 'confirmation_reviews/SUMMARY.json')
    assert confirmation['decision'] == 'retain_b00_no_confirmed_gain'
    assert confirmation['b00_v12']['passed'] == confirmation['c02_full_description']['passed'] == 78
    formal = read(SFT / 'FINAL300_COMPARISON.json')
    review = rows(SFT / 'final300/REVIEW.progress.jsonl')
    failures = {r['index']: r for r in review if r['verdict'] != 'pass'}
    categories = {
        'coverage': [3, 18, 28, 66, 71, 84, 86, 103, 136, 161, 189, 207, 225, 231, 274],
        'identification_or_binding': [38, 45, 75, 87, 102, 131, 141, 147, 148, 177, 188, 219, 220, 238, 244, 251, 252, 256, 262, 278, 280, 284],
        'relation_or_attachment': [17, 23, 30, 42, 50, 64, 67, 80, 137, 143, 170, 171, 204, 206, 216, 232, 234, 235, 271],
        'unnecessary_need': [19, 55, 239, 249, 250, 257],
        'source_ambiguity': [77, 240],
    }
    assigned = [i for ids in categories.values() for i in ids]
    assert len(assigned) == len(set(assigned)) == 64 and set(assigned) == set(failures)
    labels = {
        'coverage': '中间对象或分支覆盖不足',
        'identification_or_binding': '对象限定丢失、错绑或拆散',
        'relation_or_attachment': '关系、角色、属性或修饰归属改变',
        'unnecessary_need': '重复目标或加入无明确用途的需求',
        'source_ambiguity': '原题歧义导致无法确定忠实性',
    }
    counts = {key: {'count': len(ids), 'by_dataset': dict(collections.Counter(failures[i]['dataset'] for i in ids)), 'indices': ids} for key, ids in categories.items()}
    flags = dict(collections.Counter(d for r in failures.values() for d in r['failed_dimensions']))
    pairs = rows(SFT / 'FINAL300_PAIRED_CASES.jsonl')
    types = {}
    for ds in ['hotpotqa', '2wiki', 'musique']:
        types[ds] = {}
        for kind in sorted({p['type'] for p in pairs if p['dataset'] == ds}):
            selected = [p for p in pairs if p['dataset'] == ds and p['type'] == kind]
            types[ds][kind] = {'passed': sum(p['after_pass'] for p in selected), 'n': len(selected)}
    write(OUT / 'diagnosis/FORMAL300_POSTHOC_STATS.json', {
        'scope': 'Post-hoc description AFTER closing optimization. Not used for checkpoint/prompt selection.',
        'n': 300, 'passed': 236, 'failed': 62, 'uncertain': 2,
        'primary_categories_manual_exclusive': counts, 'overlapping_frozen_flags': flags,
        'type_results': types,
        'caution': 'Coverage/usefulness criteria are not all false facts or demonstrated downstream RAG failures; original rubric and judgments unchanged.',
    })
    detail = [
        '正式300题：64道未通过题逐题记录（62失败＋2不确定）',
        '用途：本轮优化结束后分析测试集，不再据此调整本轮提示词或选择checkpoint。',
        '下列主类是诊断性人工归类，互斥；原始F/S/C/U标签可重叠，保留不变。',
        '缺少中间问题或存在冗余不等于事实必错；本报告没有运行下游RAG。', '',
    ]
    for key, ids in categories.items():
        detail += [f'{labels[key]}：{len(ids)}/64；分数据集{counts[key]["by_dataset"]}', '']
        for i in ids:
            r = failures[i]
            detail += [f'{i}. [{r["dataset"]}] {r["id"]}', '原query：' + r['query'],
                       '完整实际分解：' + r['raw_output'],
                       f'判定：{r["verdict"]}；原始标记：{r["failed_dimensions"]}',
                       '审阅理由：' + r['reason'], '']
    (OUT / 'FORMAL300_ALL64_CASES.txt').write_text('\n'.join(detail) + '\n', encoding='utf-8')
    remote = read(OUT / 'REMOTE_FINAL_STATUS.json')
    assert len(remote['queue'].splitlines()) == 1
    assert all(f'{jid}|' in remote['accounting'] for jid in ['153978', '154009', '154018', '154020', '154022'])
    status = {
        'checked_utc': now, 'phase': 'completed', 'baseline': 'checkpoint165 + byte-exact v12',
        'dev150': {'b00': 128, 'c01': 125, 'c02': 129, 'n': 150},
        'refine44': {'baseline': 22, 'candidate': 9, 'fixed': 2, 'regressed': 15, 'decision': 'reject_no_expansion'},
        'confirmation90': confirmation, 'decision': 'retain_b00_no_confirmed_gain',
        'optimization_generation_calls': 300 + 44 + 180,
        'formal300_evaluation_complete': True, 'api_calls': 0, 'new_training_runs_this_iteration': 0,
        'running_or_queued_jobs_for_this_iteration': [], 'remote_evidence': 'REMOTE_FINAL_STATUS.json',
        'frozen_file_checks': protected,
        'further_model_or_prompt_changes_from_confirmation90_or_formal300': False,
    }
    archive_status(OUT / 'RUN_STATUS.json')
    write(OUT / 'RUN_STATUS.json', status)
    write(OUT / 'CLOSURE.json', status)
    decisions = read(OUT / 'DECISIONS.json')
    if not any(d['id'] == 'E06_CLOSE' for d in decisions['decisions']):
        decisions['decisions'] += [
            {'id': 'E06_CLOSE', 'action': 'Complete one-time fresh90: both78/90, identical pass/nonpass; retain immutable b00. Close all prompt changes before reading sealed test300.'},
            {'id': 'E07_TEST_REPORT', 'action': 'After optimization close, finish original before-versus-SFT test300:157->236 passes; describe64 nonpasses without retuning on test outputs.'},
        ]
    decisions['last_updated_utc'] = now
    write(OUT / 'DECISIONS.json', decisions)
    confirmation_note = '''
新90题一次确认完成：b00=78/90（86.67%），c02=78/90（86.67%）。
Hotpot两组均26/30，2Wiki均30/30，MuSiQue均22/30。两组各10失败＋2不确定，均保留90分母。
80/90问题数组完全相同；其余10题逐题审阅，整题通过状态也完全相同。没有确认开发集净增1题的收益。
配对经验bootstrap为[0,0]仅因所有观察到的二元差值为0，不证明总体严格等效。
最终决定：不采用c01/c02/c03；保留checkpoint165＋原v12。没有根据这90题继续修改prompt。
本轮新增优化推理524次（300＋44＋180），API调用0，新训练0。所有相关Slurm作业已完成。
'''
    results_path = OUT / 'PROMPT_TRIAL_RESULTS.txt'
    if '新90题一次确认完成' not in results_path.read_text(encoding='utf-8'):
        results_path.write_text(results_path.read_text(encoding='utf-8') + confirmation_note, encoding='utf-8')
    write(OUT / 'NEXT_STEPS_OUTCOME.json', {
        'frozen_plan_retained': 'NEXT_STEPS_FROZEN.txt',
        'confirmation_complete': True, 'adopt_candidate': False,
        'reason': 'No fresh90 gain; all prespecified regressions retained.',
        'next_research_priority': 'Preserve relation/identifier scope before increasing question counts; diagnose MuSiQue long nested descriptions separately from coverage-only failures.',
        'no_further_run_started': True,
    })
    summary = '''本轮错误分析与优化结论

一、冻结配置和结果口径
保留Qwen3-4B-Instruct-2507＋3500条微调后的checkpoint165、逐字原v12、原生模板、greedy、1024输出上限。
输出仍为独立支持问题字符串数组，不恢复#引用、goal字段或强制逐节点答案依赖。原始query另行处理。
冻结前后文件摘要全部一致。本轮没有API调用、新训练或下游RAG回答实验。
当前Codex逐题审阅，非独立人工金标准；不确定和失败请求必须保留分母。合法JSON不等于分解正确。

二、原开发150题的22道未通过
缺中间对象/分支7；对象限定和绑定7；关系及修饰归属5；多余需求3。
Hotpot7道，MuSiQue15道，2Wiki0道。14题有F标记，8题没有，不能把22题都叫作事实歪曲。
逐题原query、完整分解、原因和针对性建议见ERROR_ANALYSIS_AND_PLAN.txt。

三、为什么2Wiki开发集50/50
2Wiki原论文采用预定义模板和逻辑规则构造，许多本地实例是人物属性比较、亲属链、作品创作者属性等重复结构。
训练集有710条2Wiki，四类型为177/180/178/175，开发四类型各12或13；本地粗粒度结构家族均有训练覆盖。
这解释了较容易学会将问句结构展开，但不是模板记忆或因果机制的证明。训练/dev原生ID和规范文本没有重合，不等于排除了所有语义改写或预训练污染。
同一开发50题在三个checkpoint都50/50，不是150道独立新题。
新90题中的2Wiki30/30；另行正式测试100题为98/100，因此不能称2Wiki普遍100%。
正式252题把Ampelakia, Evros中的Evros也移到Mavrolithari；280题把Robert, Archduke Of Austria-Este删成只剩头衔。这仍是限定归属和完整对象识别问题。
论文来源：https://aclanthology.org/2020.coling-main.580.pdf

四、实际执行的三项轻量优化
c01：只替换一个示例，突出相邻地点链中的未知中间对象；dev125/150，完整修复0、退步3；拒绝。
c02：独立只替换一个示例，突出带年份、类型和过去关系的完整身份描述；dev129/150，修复2、退步1。
实际修复dev61：What band has Jerry Roush as its lead vocalist? 改为 What American metalcore band was founded in mid-2009 and had Jerry Roush as its lead vocalist?
但c02也在原失败139新增the answer to the question循环描述；整题已经失败并不意味着候选没有进一步变坏。
c03：同一个冻结4B固定复核一次，只输入原题＋原分解；诊断44题从22通过降至9通过，修复2、退步15；按预定门槛停止，不扩150。
c03常把原本有效的比较、亲属链问题删成空数组，说明增加复核并不自然提高可靠性。
新90题对照：b00和c02均78/90，逐题通过状态一致。最终保留b00，不宣称开发集净增1题是稳健收益。
所有原始输出、逐题判断、提示词逐字差异和决策都在本目录，负面结果保留。

五、完成原300题微调前后测试（不用于选择prompt）
Hotpot：51/100→84/100；2Wiki：71/100→98/100；MuSiQue：35/100→54/100。
合计157/300（52.33%）→236/300（78.67%）；改善85题、退步6题；2题不确定包含在300分母。
配对分层bootstrap通过率差值95%区间[+21.00,+31.67]个百分点；仍受同审阅者及题目相关性限制。
MuSiQue按跳数：2跳26/35；3跳20/35；4跳8/30，长嵌套组合明显更弱。这是分层观察，不单独证明跳数的因果作用。
正式未通过64题的互斥诊断：对象限定/绑定22，关系/角色/属性19，中间覆盖15，多余需求6，原句歧义2。
这些类别不能取代冻结的F/S/C/U判定；详见FORMAL300_ALL64_CASES.txt和diagnosis/FORMAL300_POSTHOC_STATS.json。

六、针对观察的改善安排
1. 优先关系和限定范围：修复“创建者→导演”“制作人→赠予人”“north of→contains”等变化，保留完整姓名/标题及其自身限定，不把所有root关键词复制到每一步。
2. 再处理覆盖：区分未识别的对象身份与已经命名对象的局部属性。保留较浅与较深的独立问题，不把细粒度问题能否回答变成后续执行的硬门槛。
3. 保持原评判规则，分报忠实性、覆盖、冗余和源query歧义；不靠降低标准、补抄最终原题或增加问题数量追求90%。
4. 简单示例替换和一次自我复核已完成，有效性不足；不默认叠加、不扩大循环，不进行无记录的试错。
5. 后续如继续修改，只在开发来源构造新的对照方案并另留未使用验证题；本轮新90和正式300不再用于候选选择。
6. 本轮分解评估未达到90%，也不能直接推导下游RAG F1。当前交付是冻结基线、完整失败分析和有对照的优化记录。
'''
    (OUT / 'FINAL_ANALYSIS_AND_OPTIMIZATION.txt').write_text(summary, encoding='utf-8')
    # Supersede stale progress pointers while keeping their historical contents.
    final_status = {'checked_utc': now, 'job_id': '154009', 'selected_checkpoint_step': 165,
                    'status': 'completed_and_reviewed', 'completed_cases': 300, 'reviewed_cases': 300,
                    'passed': 236, 'fail': 62, 'uncertain': 2, 'api_calls': 0,
                    'remote_accounting': remote['accounting'], 'queue': remote['queue']}
    archive_status(SFT / 'final300/STATUS.json')
    write(SFT / 'final300/STATUS.json', final_status)
    current_path = SFT / 'CURRENT_RUN.json'
    archive_status(current_path)
    current = read(current_path)
    current['final300_status'] = final_status
    current['semantic_checkpoint_selection'] = 'completed_on_dev150_selected165'
    current['last_updated_utc'] = now
    write(current_path, current)
    status_path = SFT / 'STATUS.txt'
    archive_status(status_path)
    status_path.write_text('Training completed at220 updates by preset early stopping. Selected checkpoint165 using dev150 only (128/150).\nOriginal sealed test300 completed and reviewed:157/300 before ->236/300 after, including2 uncertain nonpasses.\nDataset after: Hotpot84/100,2Wiki98/100,MuSiQue54/100. Target90% not met.\nAll relevant jobs completed. API calls0. Post-SFT prompt trials rejected after fresh90 confirmed no gain; baseline remains165/v12.\nDetailed final report: ../multigrain_4b_post_sft_opt_v1_20261018/FINAL_ANALYSIS_AND_OPTIMIZATION.txt\n', encoding='utf-8')
    execution = SFT / 'EXECUTION_REPORT.txt'
    if '九、实际执行完成结果' not in execution.read_text(encoding='utf-8'):
        execution.write_text(execution.read_text(encoding='utf-8') + '\n\n九、实际执行完成结果\n' + summary, encoding='utf-8')
    changelog = read(SFT / 'CHANGELOG.json')
    if not any(c['id'] == 'E004_FINAL300_REVIEWED' for c in changelog['changes']):
        changelog['changes'].append({'id': 'E004_FINAL300_REVIEWED', 'change': 'Complete original sealed300 review after prompt optimization closure;157->236 passes,85 improvements,6 regressions; preserve2 uncertain in denominator.', 'purpose': 'Finish approved model-training evaluation without test-set checkpoint or prompt selection.', 'evidence': 'FINAL300_COMPARISON.json', 'sha256': sha(SFT / 'FINAL300_COMPARISON.json')})
    changelog['last_updated_utc'] = now
    changelog['completion'] = final_status
    write(SFT / 'CHANGELOG.json', changelog)
    print(json.dumps({'baseline_unchanged': protected, 'formal_categories': counts, 'formal_types': types,
                      'final_baseline': 'checkpoint165/v12', 'formal_passed': 236,
                      'confirmed_prompt_gain': False, 'api_calls': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
