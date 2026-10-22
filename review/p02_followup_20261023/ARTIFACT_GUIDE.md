# 材料阅读指南

本包的源路径与源字节保留在ZIP内，直接报告只作为网页预览。原报告中的相对链接仍按原仓库解析；若网页上无法直接跳转，请按下面的原路径查找清单或解压。报告不是由压缩包名称推断的结果，最终报告入口已逐项核准。

## 先读实际实现与提示词

- [P02流程](source/src/p02_multigrain_timing_cachefix_v1/core/pipeline.py)、[ordinary/goal原提示词](source/src/p02_multigrain_timing_cachefix_v1/core/child_prompts.py)、[改写提示词](source/src/p02_multigrain_timing_cachefix_v1/core/rewrite_prompts.py)、[final提示词](source/src/p02_multigrain_timing_cachefix_v1/core/final_prompts.py)。
- [普通CoT冻结模板](source/experiments/query_decomposition/p02_local_cot_stage38_v1_20261022/templates.py)、[完整90题协议](source/experiments/query_decomposition/p02_local_cot_fixed90_v1_20261022/PROTOCOL.frozen.json)。F/R的Support供审阅，只有Answer传给后继。
- 每一项的完整设计与Python实现均在下表；它们是审阅代码，付费运行入口依赖原环境，不应因解包自动执行。

| 记录 | 直接源码／协议 | 原样最终报告 |
|---|---|---|
| 01 `p02_noformatter_quick90_v1_20261021` | [设计与源码](source/experiments/query_decomposition/p02_noformatter_quick90_v1_20261021/PROTOCOL.frozen.json) | [最终入口](reports/01_p02_noformatter_quick90_v1_20261021/01_REPORT.md) |
| 02 `p02_childprompt_selfcontained_quick90_v1_20261021` | [设计与源码](source/experiments/query_decomposition/p02_childprompt_selfcontained_quick90_v1_20261021/PROTOCOL.frozen.json) | [最终入口](reports/02_p02_childprompt_selfcontained_quick90_v1_20261021/01_REPORT.json) |
| 03 `p02_frozen_no_refs_quick90_v1_20261021` | [设计与源码](source/experiments/query_decomposition/p02_frozen_no_refs_quick90_v1_20261021/PROTOCOL.frozen.json) | [最终入口](reports/03_p02_frozen_no_refs_quick90_v1_20261021/01_REPORT.json) |
| 04 `p02_childprompt_minimal4_quick90_v1_20261021` | [设计与源码](source/experiments/query_decomposition/p02_childprompt_minimal4_quick90_v1_20261021/PROTOCOL.frozen.json) | [最终入口](reports/04_p02_childprompt_minimal4_quick90_v1_20261021/01_REPORT.json) |
| 05 `p02_three_prompts_quick90_v1_20261021` | [设计与源码](source/experiments/query_decomposition/p02_three_prompts_quick90_v1_20261021/PROTOCOL.frozen.json) | [最终入口](reports/05_p02_three_prompts_quick90_v1_20261021/01_REPORT.md) |
| 06 `p02_three_prompts_compact_quick90_v2_20261021` | [设计与源码](source/experiments/query_decomposition/p02_three_prompts_compact_quick90_v2_20261021/PROTOCOL.frozen.json) | [最终入口](reports/06_p02_three_prompts_compact_quick90_v2_20261021/01_BUNDLE_COMPARISON.md) |
| 07 `p02_prompt_minimal_ablation_quick90_v1_20261021` | [设计与源码](source/experiments/query_decomposition/p02_prompt_minimal_ablation_quick90_v1_20261021/PROTOCOL.frozen.json) | [最终入口](reports/07_p02_prompt_minimal_ablation_quick90_v1_20261021/01_STAGE_AUDIT.md) |
| 08 `p02_prompt_binding_probe_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_prompt_binding_probe_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/08_p02_prompt_binding_probe_v1_20261022/01_REPORT.json) |
| 09 `p02_goal_synthesis_probe_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_goal_synthesis_probe_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/09_p02_goal_synthesis_probe_v1_20261022/01_PILOT_JOINT_REPORT.md) |
| 10 `p02_goal_composition_challenge_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_goal_composition_challenge_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/10_p02_goal_composition_challenge_v1_20261022/01_REPORT.json) |
| 11 `p02_ordinary_fewshot_probe_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_ordinary_fewshot_probe_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/11_p02_ordinary_fewshot_probe_v1_20261022/01_REPORT.json) |
| 12 `p02_actual_child_binding_probe_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_actual_child_binding_probe_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/12_p02_actual_child_binding_probe_v1_20261022/01_REPORT.json) |
| 13 `p02_final_target_check_fixed90_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_final_target_check_fixed90_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/13_p02_final_target_check_fixed90_v1_20261022/01_REPORT.json) |
| 14 `p02_goal_scope_role_probe_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_goal_scope_role_probe_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/14_p02_goal_scope_role_probe_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md) |
| 15 `p02_goal_fact_role_probe_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_goal_fact_role_probe_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/15_p02_goal_fact_role_probe_v1_20261022/01_REPORT.json) |
| 16 `p02_goal_native_role_fixed90_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_goal_native_role_fixed90_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/16_p02_goal_native_role_fixed90_v1_20261022/01_REPORT.json) |
| 17 `p02_ordinary_failure_challenge8_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_ordinary_failure_challenge8_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/17_p02_ordinary_failure_challenge8_v1_20261022/01_REPORT.json) |
| 18 `p02_failure8_actual_entity_probe_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_failure8_actual_entity_probe_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/18_p02_failure8_actual_entity_probe_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md) |
| 19 `p02_failure8_actual_target_probe_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_failure8_actual_target_probe_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/19_p02_failure8_actual_target_probe_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md) |
| 20 `p02_failure8_dynamic_target_fullrag_draft_20261022` | [设计与源码](source/experiments/query_decomposition/p02_failure8_dynamic_target_fullrag_draft_20261022/PROTOCOL.frozen.json) | [最终入口](reports/20_p02_failure8_dynamic_target_fullrag_draft_20261022/01_REPORT.json) |
| 21 `p02_live_target_balanced20_dev_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_live_target_balanced20_dev_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/21_p02_live_target_balanced20_dev_v1_20261022/01_REPORT.json) |
| 22 `p02_ordinary_scope_input_probe8_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_ordinary_scope_input_probe8_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/22_p02_ordinary_scope_input_probe8_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md) |
| 23 `p02_m4_reader_stage_probe8_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_m4_reader_stage_probe8_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/23_p02_m4_reader_stage_probe8_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md) |
| 24 `p02_m4_full8_dynamic_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_m4_full8_dynamic_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/24_p02_m4_full8_dynamic_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md) |
| 25 `p02_m4_wiki20_dynamic_dev_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_m4_wiki20_dynamic_dev_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/25_p02_m4_wiki20_dynamic_dev_v1_20261022/01_ROOT_PAIRED_INTERPRETATION.md) |
| 26 `p02_shared_context_fixed90_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_shared_context_fixed90_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/26_p02_shared_context_fixed90_v1_20261022/01_ROOT_FINAL_RESULTS.md) |
| 27 `p02_reference_binding_priority_probe_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_reference_binding_priority_probe_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/27_p02_reference_binding_priority_probe_v1_20261022/01_REPORT.json) |
| 28 `p02_reference_binding_priority_fixed90_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_reference_binding_priority_fixed90_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/28_p02_reference_binding_priority_fixed90_v1_20261022/01_STATUS.json) |
| 29 `p02_local_cot_stage38_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_local_cot_stage38_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/29_p02_local_cot_stage38_v1_20261022/01_ROOT_RESULTS.zh.md) |
| 30 `p02_local_cot_fixed90_v1_20261022` | [设计与源码](source/experiments/query_decomposition/p02_local_cot_fixed90_v1_20261022/PROTOCOL.frozen.json) | [最终入口](reports/30_p02_local_cot_fixed90_v1_20261022/01_ROOT_RESULTS.zh.md) |

## 从成绩追到实际输入输出

1. 在`CATALOG.json`按实验找最终报告、原输出路径和archive_paths。早期90题完整结果位于恢复目录；根目录旧partial保留作历史，不替代最终结果。
2. 在`ARCHIVE_MANIFEST.json`以完整source路径查找ZIP成员。`DETAILS.jsonl`、`PAIRED.jsonl`、review文件保存逐题结果与配对；原有case／trace文件保存可读输入输出。
3. 更完整的模型和检索回执在`SQLITE_EXPORTS.json`。用source选正确实验及恢复目录，再找data、requests、request_stages、retired_requests等表。行是原数据库列名→原值；payload/value/result中的JSON仍是字符串，可再次`json.loads`。不要把同一请求的阶段映射或不同恢复库副本累计成独立调用或独立样本。
4. 每次实际生成的messages、上限、响应、usage和状态沿requests查阅；候选检索、reader、历史和outcome沿data及原case文件查阅。不同实验表结构保留原样，以原run.py/data kind命名为准。
5. CoT的90题四臂可读导出位于`analysis/p02_prompt_research_followup_20261022/local_cot90_trace_review/`，20题语义复核同目录父层。完整请求和证据都已包含，不只保留解释文字。
6. 共享上下文的配对语义审阅在原输出的root_blind_review；引用因素暂停说明、完整计划和61条结果分别在原协议、暂停资料与data/outcome。它没有完整90题比较。

## 信息流诊断与旧包

最新材料复核入口为`outputs/query_decomposition/p02_information_flow_followup_20261022/PRIMARY270_REVIEW_COMPLETION.md`；旧150条统计仅是中间进度。全部阶段标签、证据包、待核项与补充材料保留在information_flow归档。

`PUBLICATION_DELTA.json`记录本30项报告相对旧包的差异，并提供历史原始case的旧包成员对应。原架构／正式历史成绩和723条错误传播诊断仍在前两个审核包；旁支8组设计／实验仍在已发布的第三个审核包。本次没有覆盖这些源文件。

## 删减、时间与可复现性

credentials、全局budget数据库、模型权重、全量索引、远端启动日志和重复运行快照不在包内；少量合同测试源／历史技术恢复资料作为来源材料保留，不计为模型实验。实际科学执行以最终报告与outcome记录为准。已知API估费和unknown预留保留，不将未知记录当作免费或成功。

原生计分文件、提示词和原输入保持字节不变。标准库验证只重聚合已保存分数，不调用评分模型，不重新评判事实，不复算所有bootstrap。恢复付费完整流程需要原数据、检索环境与授权；本包的可复现承诺是独立离线核对交付完整性和明确列出的主要统计。
