# 本次实际使用的本地证据入口

这些都是此前实验或冻结实现。本设计未重新执行它们；读取日期为2026-10-22。所有链接均可沿本设计目录回到原记录。

| 用途 | 原记录 |
|---|---|
| 描述式ordinary适配完整90题 | [自包含适配报告](../../../outputs/query_decomposition/hotpot_formal_eval_v1/p02_childprompt_selfcontained_quick90_v1_20261021/approved_recovery004/REPORT.json) |
| 三prompt联改90题 | [联改报告](../../../outputs/query_decomposition/hotpot_formal_eval_v1/p02_three_prompts_quick90_v1_20261021/approved_unknown_recovery003/REPORT.json) |
| compact v2已补齐90题 | [精简版完整报告](../../../outputs/query_decomposition/hotpot_formal_eval_v1/p02_three_prompts_compact_quick90_v2_20261021/delegated_recovery003/REPORT.json) |
| 结果绑定与显式组合操作 | [实验解释](../unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/p02_result_binding_operator_probe_v1_20261022/INTERPRETATION.zh.md) |
| 新60题goal综合与展示对照 | [实验解释](../unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/p02_goal_target_transfer_new60_v1_20261022/INTERPRETATION.zh.md) |
| 充分性、保留与final读入的三项单改 | [实验解释](../unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/p02_goal_sufficiency_retention60_v1_20261022/INTERPRETATION.zh.md) |
| 普通QA短依据、完整回答与接受答案的区别 | [局部结果](../../../outputs/query_decomposition/p02_local_cot_stage38_v1_20261022/ROOT_RESULTS.zh.md)、[锁定统计](../../../outputs/query_decomposition/p02_local_cot_stage38_v1_20261022/ROOT_FINAL_STATS.json) |
| 共享上下文对照的完整/缺失范围 | [最终报告](../../../outputs/query_decomposition/p02_shared_context_fixed90_v1_20261022/ROOT_FINAL_RESULTS.md) |
| 实际Current绑定、reader变动与后继检索 | [累计研究记录](../unified_rag_eval_v1_20261021/reviews/20261022_prompt_binding_goal_synthesis/FROZEN_DECOMPOSER_FINDINGS.md) |
| 真实接线和native输出上限 | [pipeline.py](../../../src/p02_multigrain_timing_cachefix_v1/core/pipeline.py) |
| 当前评估约定与暂停范围 | [AGENTS.md](../AGENTS.md)、[PLAN.json](../unified_rag_eval_v1_20261021/PLAN.json) |

以上记录中组件重放、固定reader实验、开发集完整流程和正式主评估用途不同。不能把所有表中的F1放到同一个无条件排名中。
