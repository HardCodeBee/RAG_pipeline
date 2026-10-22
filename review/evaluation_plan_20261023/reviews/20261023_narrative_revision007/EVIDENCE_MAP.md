# Revision 007 实验设计／Revision 008 文字修订：论证职责与已有证据复用表

日期：2026-10-23。状态：**评估计划与已有材料整理；没有启动评估、模型推理、检索服务或补跑。** 本文件的三组已有分析新增 API 调用为 **0**，不增加新的实验样本，也不恢复任何未完成 probe。主计划中的六行正式比较和 P02 QA/raw 均为待执行方案，不能引用本表作为已完成回执。

拟检验的研究问题是：**在共用 QwenPlus reader 与 BGE 检索资源、P02 另用冻结专用分解器的条件下，将原问题组织为有序信息需求、逐步取得并传递子问事实，能否改善最终回答；哪些信息在检索、选择、中间回答和最终整合之间丢失？** 方法贡献须由正式配对质量、成本及信息流证据共同限定。QDMR、问题分解、子问 QA 中间表示、依赖答案继续检索、QA/raw 整合对比均已有先例，不写“首次”，也不以单案例声称机制普遍必要。

本地结果总入口为 [RESULTS_MAIN.md](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_followup_20261023/RESULTS.md)（仓库原路径：`analysis/p02_publication_20261023/RESULTS_MAIN.md`）。C01–C24 编号对应已提交的[远端结论索引](../../../p02_followup_20261023/query_decomposition_conclusions_20261023.txt)。以下优先链接每项最终报告，避免根目录旧 partial 报告覆盖后续完成结果。

## 1. 六行主表：每一行回答什么问题

这六行按与研究问题的对应关系选择，**不依据预期分数高低删减对手**。它们分别覆盖原题检索、同证据推理、并行信息需求扩展、在线子问求解、模块化分解求解与本方法。减少方法数意味着缩小对外部方法的覆盖，不能将结果写成“超过所有多跳 RAG”或 SOTA。正式结果出现前固定这些职责与参数。

| 主表方法 | 论证职责与必须保留的原因 | 固定代表性设置及出处 | 公平性和结论边界 |
|---|---|---|---|
| Top10 RAG（b00） | 原问题直接检索的共同起点；判断额外组织和求解步骤是否带来最终收益。 | 原题 BGE Top10；使用冻结直接回答接口。k=10 是共同检索口径，也与 QD+Rerank 原论文 §4.5 的实验值相容。 | 共用 QwenPlus reader、语料和 BGE 检索器，P02 的专用分解器另列；报告真实 tokens、逻辑费用及可选计时，不增加调用数指标。 |
| Top10+CoT（b01） | 在相同原题 Top10 上使用带推理提示的阅读方式，作为不新增信息需求/检索的常规推理参照。 | 与 b00 共享相同 Top10，使用冻结的 3 个 CoT 示例；不新增检索。 | 除推理要求外还存在示例差异，不作为“只改变 CoT 一个变量”的因果消融；也不是等 tokens 对照。 |
| QD+Rerank（b08） | 并行分解检索路线的代表；比较该完整路线与 P02 完整路线的表现。两者同时存在求解、历史、检索合并与最终材料等差异，不能把分数差归因于顺序或历史反馈某一因素。 | 保留原题，加至多 5 个生成子查询；每查询 k=10；合并去重后按原题重排，最终 Top10。[原论文](https://aclanthology.org/2025.acl-srw.32.pdf) §3/Algorithm 1、§4.5、§5.3。 | 保留 reranker，不删除其核心能力。原论文模型、解码及数据范围与本项目不同，标注统一资源适配；W/M 上的设置是固定迁移，不是原论文直接复现。 |
| Self-Ask（b02） | 在线提出下一个子问题并把子回答写入历史，是“边求解边组织信息需求”的代表；与 P02 的预先有序分解形成有意义对照。 | 每次检索 k=5，最多 5 次搜索交互，采用 [Iter-RetGen 原论文](https://aclanthology.org/2023.findings-emnlp.620.pdf) §4.4 对 Self-Ask 的 dense-retrieval 适配设置。[Self-Ask 原论文](https://aclanthology.org/2023.findings-emnlp.378.pdf) §3.1、§3.3 给出在线 follow-up/中间答案机制。 | k=5/max=5 来自后续固定 dense 适配，不谎称是原始搜索引擎版本唯一参数；QwenPlus+BGE 继续标注适配。以最多搜索交互次数计数，不能误当最多所有 LLM 调用数。 |
| DecomP（b06） | 模块化分解、子任务执行、答案与文档回传的同类代表；直接检验 P02 相对已有分解求解框架的增量，不能因其可能较强而移除。 | 官方 ODQA CoT base config：检索 k=2，外层与 `retrieve_odqa` 内层 `max_steps=8`。[官方 HotpotQA 配置](https://github.com/HarshTrivedi/DecomP-ODQA/blob/main/base_configs/decomp_context_cot_qa_codex_hotpotqa.jsonnet)及同目录 2Wiki/MuSiQue 配置；[论文](https://arxiv.org/pdf/2210.02406) §4.4、Appendix A.2。 | 这是官方可复查默认值；论文另做 dev100 上 k=2/4/6 等搜索，故不能称 k=2 是各数据集论文最优配置。保留 CoT 和模块化执行；共用 QwenPlus reader 与 BGE 检索资源下属于适配。 |
| 冻结 P02 | 评价既定组合：有序自包含子问 → 基于历史的检索改写 → 原/改写双路选文 → ordinary/goal QA → QA 与原题 Top10 最终整合。 | 方法 `p02_multigrain_timing_cachefix_v1`（冻结原 P02/R02/v12），新主表执行变体 `p02_shared_baseline_v2`；原题优先交错双路 Top10；final 为普通 QA、goal QA、原题原路线 Top10。结构/原生 prompts 保持，共同完整输入和请求层按主计划执行。 | 不把旧原生分数改名成新统一层成绩；不把 M4、目标改写、ROLE、CoT 等探索性候选自动并入 P02。主表测试整体；各组件解释须依赖有相应控制的分析，不能由整体获胜倒推每个组件必要。 |

Self-Ask 替代主表 Iter-RetGen 的理由是更直接覆盖“子问提出—子回答—再提出”的组织方式，并非预期它更弱。Iter-RetGen 仍是相关工作及 dense 适配参数来源；若论文改为主张普遍优于迭代检索、检索反馈或最佳通用 RAG，需重新评估是否加入 Iter-RetGen/IRCoT 等代表。六行表本身不支持这些更宽的主张。

各方法共用主计划冻结的 QwenPlus reader 与 BGE 检索资源；P02 的分解另由冻结的 `Qwen3-4B-Instruct-2507` + `r02_expanded` LoRA / v12 执行。因此这里不是所有模块使用相同模型，也不声称相同计算量；按用户既有口径报告输入输出 tokens 与完整逻辑成本，不新增 LLM 调用数指标。计时仅限另行批准的可选附录。新主表按各数据集冻结 500 题规划，共 3×500×6=9,000 个方法–问题质量记录；以下历史样本不能补入该分母，也不能将这批已曝光的官方 dev/validation 固定子集称为全新确认集。

## 2. A 组：事实正确性与下游信息充分性（C04、C08、C09）

### A1. 旧协议诊断：正确子回答仍可能不足以解决原题（C04）

- **样本与协议：** 接入统一请求层之前的 723 条完整、非空历史 P02 流程，2Wiki 485、HotpotQA 238；不包含 MuSiQue。语义复核沿用既有标签，未做新的模型生成。
- **数字：** ordinary 节点的事实正确标签为 2Wiki 970/980、HotpotQA 352/381。在 230 条最终非满分流程中，有 **177 条（2Wiki 108、HotpotQA 69；77.0%）**同时满足既有分解、普通子回答及有效改写“正确”的标签条件。这里最终“正确”指标是既有 `F1=1 或 EM=1` 二元标志，不能改写成平均 F1 或完全事实真值。
- **可支持：** 合理弃答、真实但不完整的日期/清单可以保持二元正确，却缺少下一步或最终回答所需的信息。局部正确率与原题可回答性应分别衡量。
- **不可支持：** 177 不是“final 模块造成的错误数”，也不是可修复比例；这些标签不是互斥、穷尽因果归因。观察性关联与同一 Codex 复核不能替代独立金标或受控干预。
- **复用：** 一张诊断表与必要信息定义，放在结果解释/分析部分；保留旧协议和两数据集范围，不升级为新 1,500 题的总体统计。

来源：[诊断 REPORT](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_diagnostic_20261021/REPORT.md)（仓库原路径：`outputs/query_decomposition/p02_error_propagation_diagnostic_20261021/offline_review_v2/REPORT.md`）、[条件概率](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_diagnostic_20261021/CONDITIONAL_PROBABILITIES.json)（仓库原路径：`outputs/query_decomposition/p02_error_propagation_diagnostic_20261021/offline_review_v2/CONDITIONAL_PROBABILITIES.json`）、[描述计数](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_diagnostic_20261021/DIAGNOSTIC_DESCRIPTIVE_COUNTS.json)（仓库原路径：`outputs/query_decomposition/p02_error_propagation_diagnostic_20261021/offline_review_v2/DIAGNOSTIC_DESCRIPTIVE_COUNTS.json`）。

### A2. 三个保存轨迹例：定位信息在哪一步丢失（C08）

| 例子与原始 qid | 已核可见范围与信息路径 | 能支持的表述 | 不应推出 |
|---|---|---|---|
| Hotpot46；`5a7553a95542996c70cfaedf` | 保存完整 case 中五份检索包共 50 行/41 个唯一文档、三份 reader 包及其 prompts/QA/final 均未见 `Wendigo/Windigo`；子问列出 Decepticons/Inhumanoids，goal 随之收窄。 | **保存轨迹中未见**所需实体；不能把此例归因为已检到事实被 QA 压缩丢失。 | 语料全集没有 Wendigo；模型必定无法回答；仅替换 final 表示即可修复。 |
| Hotpot7；`5ab3eeb5554299233954fff4` | 子问 2 改写检索的 rank10、doc `33214212`（Doomsday Prophecy）含 2011 及 Jewel Staite/Alan Dale；两路共 19 个不同候选，交错原题优先 Top10 未纳入该段，后续子答/goal/final 未取得该事实。 | 必要候选已检到，但 candidate→reader 选择丢失。 | 保留该段一定得到正确答案；所有截断均属错误；此例已被 M4 修复。 |
| Hotpot236；`5a87ab905542996e4f3088c1` | 普通 QA 给出 Androscoggin Bank Colisée；goal 用历史实体改写后，改写 rank1 doc `1922204` 的 `4,000 capacity (3,677 seated)` 进入 goal reader rank2，goal QA 保留两数；原题原路线 Top10 与普通材料中没有 3677，final 经 goal QA 才看到 3677，最终仍选 4000。 | 此条保存流程中，3677 经“goal 改写检索→goal 回答→final”独有通路进入 final；信息到达与正确使用是两个问题。 | 所有题都依赖 goal；无 goal 的新反事实一定失败；此例证明 final 一般不会用证据。 |

原轨迹目录：[Hotpot 原始 cases](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_diagnostic_20261021/SOURCE_CROSSWALK.json)（仓库原路径：`outputs/query_decomposition/hotpot_formal_eval_v1/p02_formal1500_v1_20261021/rounds/r000/cases/hotpotqa/p02_multigrain_timing_cachefix_v1`）；每个文件名为对应 qid 加 `.json`。选例索引：[ROOT_CAUSE_CASES.json](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_diagnostic_20261021/ROOT_CAUSE_CASES.json)（仓库原路径：`outputs/query_decomposition/p02_error_propagation_diagnostic_20261021/offline_review_v2/ROOT_CAUSE_CASES.json`）。Hotpot236 的旧显式绑定标签为空属于已发现漏标，实体实际来自 history；展示时注明应为 `[1]`，不修改冻结旧标签或把空标签当作没有绑定。

### A3. 选择截断的规模与已证实损失应分开（C09）

- **样本与协议：** 270 条已复核来源 = 230 非满分 + 40 固定 seed 满分对照；258 完成其声明的必要信息检查、12 待核。只读复现 696 个保存 merge/reader 包，全部与原 reader 一致；这验证材料身份，不补足语义金标。
- **数字：** 双路检索共 303 节点/234 题；287 节点候选 union 超过 10；223/270 题有候选截断；共 2,335 个被截节点–文档组合。**只有 2 条（Hotpot7、Wiki721）独立证实必要材料已检到、随后被 merged10 丢弃。**
- **可支持：** 检索命中与 reader 可见性不同；在已核案例里，选择是信息流断点之一。
- **不可支持：** 223 是错误数，或 2/270 是穷尽发生率；未审完的 12 条不能视为无损失；候选被截不等于必要信息被截。
- **复用：** 可报告“容量现象”和“保守确认案例数”两列，并用 A2/C11 解释；不扩大标签、不补跑、不将技术复现宣称为新语义审阅。

来源：[evidence_selection_readonly/REPORT.md](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_followup_20261023/INFORMATION_FLOW.md)（仓库原路径：`experiments/query_decomposition/unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/evidence_selection_readonly/REPORT.md`）。

## 3. B 组：增加材料或上下文不自动提高可用性（C13、C14）

| 对照 | 旧协议、分母与具体结果 | 可支持 / 不可支持 | 本次复用方式 |
|---|---|---|---|
| C13：双路 union20 对原 merged10 | 已曝光 dev300，三集各 100；修复缓存后的完整配对。timing-only 控制逐题预测和分数与原 P02 一致。F1 **70.7242→70.3601**，差 **−0.3641 个百分点**，95% 配对 bootstrap CI **[−2.2851,+1.4429]**；EM 均 **61.3333**；14 升、9 降、277 平。逻辑输入 tokens **2,837,258→3,310,428（+16.68%）**；同价逻辑生成估费 **¥2.3262524→¥2.7053104（+16.29%）**。 | 本 dev300 未观察到增加到最多 20 段后的总体质量收益，且逻辑用量增加。区间包含正负效应，不能称等效/非劣；不是 QA/raw 对照，不证明所有更多证据都无效。 | 保留完整 300 配对、CI 和逻辑成本。此运行含缓存复用与完整流程重构，不能叫 300 题全部新鲜推理，也不能将逻辑费当本轮实际新账单或计时。 |
| C14：reset / flat / chat | 计划 90×3=270 条件，269 成功，1 flat HTTP400 技术 held。共同 **89** 题（H30/W29/M30）按逐 query 等权：F1 **76.43/69.90/71.59**；EM **67.42/58.43/60.67**；平均输入 tokens **9,580.1/17,555.8/17,565.8**；平均总 tokens **9,679.7/17,646.0/17,656.8**，flat/chat 增 **82.3%/82.4%**。相对 reset，flat 为 2 升 10 降 77 平，chat 为 2 升 7 降 80 平。 | 本版本累积完整上下文没有验证总体质量收益。reset 已使用前序 QA，因此不是“无历史 vs 有历史”；不支持所有长上下文、chat 或选择性记忆都无效。 | 用同期 reset 为主控制，历史 P02 只作非同期参考。保留共同 89 与逐题等权，不填成 90、不与三集等权 macro 混用、不补跑 held。 |

来源：C13 [修复缓存最终 REPORT.json](https://github.com/HardCodeBee/RAG_pipeline/blob/2cbe21300ce91e3e20c4f1bce153a7ecb6cfe1b9/review/p02_20261021/evidence/outputs/query_decomposition/hotpot_formal_eval_v1/p02_merge_cachefix_v1_20261021/reports/REPORT.json)（仓库原路径：`outputs/query_decomposition/hotpot_formal_eval_v1/p02_merge_cachefix_v1_20261021/reports/REPORT.json`）；C14 [ROOT_FINAL_RESULTS.md](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_followup_20261023/SHARED_CONTEXT_RESULTS.md)（仓库原路径：`outputs/query_decomposition/p02_shared_context_fixed90_v1_20261022/ROOT_FINAL_RESULTS.md`）。C13 不使用早期混合缓存结果。既有成本用于解释旧对照，不沿用其旧预算上限作为新计划预算。

两项共同支撑的有限解释是：**取得更多文本与最终得到可用事实需要分开评估。** 它们不直接证明 P02 的 QA 中间表示优于 raw，也不替代新的 P02 表示对照。

## 4. C 组：沿实际信息路径解释局部改善与边界（C10–C12）

| 证据 | 旧样本、干预与数字 | 可支持 | 不可支持及复用方式 |
|---|---|---|---|
| C10：goal 检索通道与 final 消息 | 固定上游 ordinary QA 的同 15 题；共同 goal512/final64 接口，**不是原生 goal128 P02**。G0 F1/EM **85.7143/80.0000**；单独关闭 goal 改写补充路线为 **72.3232/66.6667**，1 升 3 降 11 平；单独移除 final 可见 goal 信息为 **67.7778/60.0000**，0 升 4 降 11 平。后者 2 条 final length 算法失败保留 15 分母计 0。 | 在 Josef Mach 出生地、Wang 父子关系等已核案例，goal 改写补充取得的新事实可经 goal QA 到达 final。原题原路线材料并不总包含这些事实。 | 两个分别干预不等于整个 goal 模块删除；移除消息没有省去 goal 计算，且同时去掉事实与候选结论。一个关改写增分是错误否认句的词面重合，非事实修复。复用原 15 配对表和 1–2 条路径，不外推总体必要性。 |
| C11：M4 的选择→传递→final | M4 只去同 source 有序词和标点相同的排版重复，再从共同候选池回填至 10；不是去掉所有同标题文档。固定 reader 8 节点事实正确均 **8/8**，必要信息 **7/8→8/8**。对应动态完整 8 题 F1 **89.4643→95.7143**、EM **62.5→75**，1 升 0 降 7 平；收益集中在 Wiki721 完整日期 `7 March 1517`。另一个已曝光固定 2Wiki20 题，M0/M4 **F1=EM=85**，20 对全平。 | 同一已知例子中，改写候选原本含完整日期；M4 使其进入 reader，再进入 goal QA，最终从年份 `1517` 变完整日期。事实正确和所问信息完整可以分别变化。 | 不把年份判为事实错误；不把同一 8 题阶段/全流程重复计成独立证据；不把后续 20 称盲测或合并成独立 28 题。必须一并展示 20 题持平，不晋级 M4、不声称普遍安全或总体改善。 |
| C12：实际子问题目标与传播 | 已知失败 8 节点，泛化自查或旁加实体候选仍 **2/8** 正确；用同源检索改写替换实际 Current 后，经补证为 **2/8→5/8**。完整动态 8 题 F1 **24.4643→33.3929**、EM **0→12.5**，2 升 0 降 6 平。另 20 题动态对照 38/40 成功，两条技术 held 来自同一题；共同 **19** 题最终文本和分数全部持平。 | 某些失败源于实际被回答的关系/成员目标，附加实体名不足以修复；局部修复需要沿后继改写、检索和 final 检查是否保留收益。 | 8 题是特选失败集；阶段控制有非同期复用且事实补证不表示补证材料原已在 ordinary reader；不能把局部 5/8 宣称普遍提升。必须同时列 19 题无差异，不将检索改写默认提升为可靠任务定义，也不补齐 held。 |

来源：

- C10：[归档内 COMPONENT_PAIRED_REVIEW.md](https://github.com/HardCodeBee/RAG_pipeline/blob/2cbe21300ce91e3e20c4f1bce153a7ecb6cfe1b9/review/p02_followup_20261023/archives/p02_goal_synthesis_probe_v1_20261022.01.zip)（需打开 ZIP；归档内准确路径：`outputs/query_decomposition/p02_goal_synthesis_probe_v1_20261022/manual_review/COMPONENT_PAIRED_REVIEW.md`）。
- C11：[reader 阶段 8](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_followup_20261023/reports/23_p02_m4_reader_stage_probe8_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md)（仓库原路径：`outputs/query_decomposition/p02_m4_reader_stage_probe8_v1_20261022/ROOT_PAIRED_INTERPRETATION.md`）、[动态 8 与唯一信息路径](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_followup_20261023/reports/24_p02_m4_full8_dynamic_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md)（仓库原路径：`outputs/query_decomposition/p02_m4_full8_dynamic_v1_20261022/ROOT_PAIRED_INTERPRETATION.md`）、[后续固定 2Wiki20](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_followup_20261023/reports/25_p02_m4_wiki20_dynamic_dev_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md)（仓库原路径：`outputs/query_decomposition/p02_m4_wiki20_dynamic_dev_v1_20261022/ROOT_PAIRED_INTERPRETATION.md`）。
- C12：[失败节点目标替换及补证](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_followup_20261023/reports/19_p02_failure8_actual_target_probe_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md)（仓库原路径：`outputs/query_decomposition/p02_failure8_actual_target_probe_v1_20261022/ROOT_PAIRED_INTERPRETATION.md`）、[动态 8 REPORT](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_followup_20261023/reports/20_p02_failure8_dynamic_target_fullrag_draft_20261022/01_REPORT.json)（仓库原路径：`outputs/query_decomposition/p02_failure8_dynamic_target_fullrag_draft_20261022/REPORT.json`）、[20 题计划/19 对完成 REPORT](https://github.com/HardCodeBee/RAG_pipeline/blob/1b156fbc1c21e6b4bd24e2cd6bf7642b8b8c96ec/review/p02_followup_20261023/reports/21_p02_live_target_balanced20_dev_v1_20261022/01_REPORT.json)（仓库原路径：`outputs/query_decomposition/p02_live_target_balanced20_dev_v1_20261022/REPORT.json`）。目录名 `draft` 不代表未执行；实际状态以上述最终报告为准。

这组材料适合呈现为一张信息流图及配对案例：**候选命中 → reader 选入 → ordinary/goal 保留 → final 可见 → final 正确使用**。这五个位置是对当前系统的分析口径，不能把五个节点都画成已由独立随机干预验证的因果机制。

## 5. S02 QA/raw 只能作动机；P02 对照仍待执行

旧 S02 表示实验（C23）冻结上游 200 题，其中 199 题共同可计分：163 对新 final 生成、36 对相同 fallback 复用、1 条上游缺失。QA/raw 的 F1 为 **62.8030/60.0852**，差 **+2.7178 个百分点**，配对 95% CI **[−0.7269,+6.2354]**；平均 final 输入 **2,210.31/3,391.80** tokens，QA 少 **34.83%**；全流程逻辑费用少约 **15.08%**。来源：[历史结论索引 C23](../../../p02_followup_20261023/query_decomposition_conclusions_20261023.txt)（该公开入口提供 C23 结论；原始 `outputs/query_decomposition/hotpot_formal_eval_v1/aux_representation_200_20261015/REPORT.txt` 未在此处直接提供）。

不能据此宣布 QA 在准确率上显著优于 raw：区间包含 0。更不能把它替代 P02 结果，原因是 S02 使用显式引用和两遍 QA 流程，最终 prompt 偏 QA 格式，raw 条件仍支付上游两遍 QA 成本；它没有匹配可见信息或 tokens，而 P02 的自包含子问、改写/选文、goal 与 final 可见材料不同。

P02 QA/raw 应作为**另行冻结、尚未运行的表示对照**：共享同一份本轮统一层 P02 上游轨迹（或逐阶段 C1 核验通过的等价迁移）、相同原题 Top10、共同预先冻结的 final prompt；只把实际 reader 来源形成的 raw 材料与相应 QA 表示作为被比较输入，ordinary 与 goal 均包含。旧原生轨迹不能直接顶替。不得让 raw 偷看未进入该臂 reader 的候选，或把不同新生成轨迹混为纯表示对照。详细段落顺序、稳定去重、问题关联与完整输入规则按主计划及 QA/raw 协议明确，不静默截断，不能由旧 S02 补足。由于该共同预先冻结的 final prompt 与原生 P02 主表接口不同，该对照控制需与原生 P02 主表结果分列。

该问题测试的是本系统中**信息筛选、压缩与最终使用的取舍**，并非只更换字面格式；QA 与 raw 的信息量/长度天然可能不同。应并列准确率、可见信息、输入 tokens、final 增量成本及含固定上游的逻辑成本，不能用复用账单宣称 raw 是省掉全部子问求解的独立算法。共同预先冻结的 prompt 仍规定了证据使用政策；该政策可能与两臂的来源可见性交互，因此测得的是该冻结政策下的综合接口效果，不能解释为纯格式效应。本次仅修订表述，不改动已冻结 prompt。

[ChainRAG 原论文](https://aclanthology.org/2025.acl-long.1089.pdf) §3.4（印刷页 22366）已明确比较 AnsInt（子问 QA）与 CxtInt（检索句子去重/重排）的最终整合。因此 P02 QA/raw 的作用是验证当前系统的设计取舍，**不能作为 QA 中间表示或 QA/raw 比较本身的新颖性**。类似地，[DecomP](https://arxiv.org/pdf/2210.02406) 已有分解执行与答案/文档回传；[Self-Ask](https://aclanthology.org/2023.findings-emnlp.378.pdf) 已有子问答案驱动的在线组织。本项目的可写贡献应落在被明确限定的组织与传递组合，以及对最终可用事实损失的可检验证据上。

## 6. 复用规则、风险与写作界限

1. **历史证据与新主表分开。** 723、270、300、89、15、8、20/19 来自不同旧协议或已曝光开发集，不相加成新样本，不拿作正式 1,500 题的新泛化结果。阶段节点数不等于完整问题数；反复使用的 8/90 题不视为独立重复。
2. **完整呈现局部正负结果。** 同时保留 M4 8 题改善/后续 20 持平、目标替换 8 题改善/后续 19 持平，以及 union20/shared-context 负结果。不能为了叙事只挑成功例子。未完成引用因素动态 90 等 probe 不参与择优，也不为本次文档补跑。
3. **技术缺失与算法失败分开。** HTTP400 等技术 held 不作为质量 0；已定义的 final length 算法失败按原协议保留分母计 0。不得根据结果临时更改口径。
4. **语义正确性与 native 分数分开。** 真实但不完整、正确实体但词面不同、合理弃答缺少桥梁、错误句碰巧命中参考词都可能出现。保留原生 EM/F1，并用可见证据说明事实/充分性；不静默改评分或冻结标签。
5. **代表性参数不等于最优复现。** Self-Ask k5/max5 明确归于后续 dense 适配；DecomP k2/内外 8 明确归于官方 base config；QD 保留核心 reranker。不得以固定参数口径声称已复现各论文的最优系统。
6. **拟议贡献需要实证兑现。** 可以预注册“有序需求组织与逐步可用事实传递”的研究问题；在新质量/成本结果出来前不写已经提升、不保证优于全部同类，不把仅有的候选与案例归因为普遍必要机制。

本次复用所需工作限于离线整理现有结果、引用及图表；**三组已有分析 0 新 API、0 新检索/GPU、0 probe 补跑。** 任何新主表、P02 QA/raw 或后续确认均需按主计划的执行授权与冻结设置处理，本证据表不触发执行。
