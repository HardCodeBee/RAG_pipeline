# 提示词与证据导航

`snapshot/`下保留原路径和字节；`receipts/`覆盖全部checkpoint表。`prompt_examples/`是每组一个完整实际请求/响应入口，其余请求仍在完整回执中。

## 1. 初始42题方案：受阻，后由90题方案接替

- [冻结实验协议](snapshot/experiments/query_decomposition/p02_goal_fewshot_synthesis_v1_20261022/PROTOCOL.frozen.json)
- [实验说明](snapshot/experiments/query_decomposition/p02_goal_fewshot_synthesis_v1_20261022/README.md)
- [完整模板与构造方式](snapshot/experiments/query_decomposition/p02_goal_fewshot_synthesis_v1_20261022/templates.py)
- [新指令I](snapshot/experiments/query_decomposition/p02_goal_fewshot_synthesis_v1_20261022/I.frozen.txt)
- [四个合成示例F](snapshot/experiments/query_decomposition/p02_goal_fewshot_synthesis_v1_20261022/F_EXAMPLES.frozen.txt)
- [实际接线与执行代码](snapshot/experiments/query_decomposition/p02_goal_fewshot_synthesis_v1_20261022/run.py)
- [REPORT.json](snapshot/outputs/query_decomposition/p02_goal_fewshot_synthesis_v1_20261022/REPORT.json)
- [BLOCKED.json](snapshot/outputs/query_decomposition/p02_goal_fewshot_synthesis_v1_20261022/BLOCKED.json)
- [全部请求和checkpoint表的索引](receipts/p02_goal_fewshot_synthesis_v1_20261022/SCHEMA.json)

## 2. Few-shot fixed90：旧G2、新指令I、四示例F

- [冻结实验协议](snapshot/experiments/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/PROTOCOL.frozen.json)
- [实验说明](snapshot/experiments/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/README.md)
- [新指令I](snapshot/experiments/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/I.frozen.txt)
- [四个合成示例F](snapshot/experiments/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/F_EXAMPLES.frozen.txt)
- [实际接线与执行代码](snapshot/experiments/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/run.py)
- [REPORT.md](snapshot/outputs/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/REPORT.md)
- [REPORT.json](snapshot/outputs/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/REPORT.json)
- [SUMMARY.json](snapshot/outputs/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/SUMMARY.json)
- [CASE_REVIEW.md](snapshot/outputs/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/CASE_REVIEW.md)
- [OFFLINE_AUDIT.json](snapshot/outputs/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/OFFLINE_AUDIT.json)
- [全部请求和checkpoint表的索引](receipts/p02_goal_fewshot_fixed90_v1_20261022/SCHEMA.json)
- 实际完整请求与回答：[goal_F](prompt_examples/p02_goal_fewshot_fixed90_v1_20261022/goal_F.json) · [goal_G2](prompt_examples/p02_goal_fewshot_fixed90_v1_20261022/goal_G2.json) · [goal_I](prompt_examples/p02_goal_fewshot_fixed90_v1_20261022/goal_I.json)

## 3. 结果绑定与显式组合操作

- [冻结实验协议](snapshot/experiments/query_decomposition/p02_result_binding_operator_probe_v1_20261022/PROTOCOL.frozen.json)
- [完整模板与构造方式](snapshot/experiments/query_decomposition/p02_result_binding_operator_probe_v1_20261022/templates.py)
- [冻结演示](snapshot/experiments/query_decomposition/p02_result_binding_operator_probe_v1_20261022/DEMOS.frozen.json)
- [goal指令](snapshot/experiments/query_decomposition/p02_result_binding_operator_probe_v1_20261022/GOAL_INSTRUCTION.frozen.txt)
- [实际接线与执行代码](snapshot/experiments/query_decomposition/p02_result_binding_operator_probe_v1_20261022/run.py)
- [REPORT.md](snapshot/outputs/query_decomposition/p02_result_binding_operator_probe_v1_20261022/REPORT.md)
- [COMPLETION.json](snapshot/outputs/query_decomposition/p02_result_binding_operator_probe_v1_20261022/COMPLETION.json)
- [SUMMARY.json](snapshot/outputs/query_decomposition/p02_result_binding_operator_probe_v1_20261022/SUMMARY.json)
- [CASE_NOTES.md](snapshot/outputs/query_decomposition/p02_result_binding_operator_probe_v1_20261022/CASE_NOTES.md)
- [INTERPRETATION.zh.md](snapshot/outputs/query_decomposition/p02_result_binding_operator_probe_v1_20261022/INTERPRETATION.zh.md)
- [OFFLINE_AUDIT.json](snapshot/outputs/query_decomposition/p02_result_binding_operator_probe_v1_20261022/OFFLINE_AUDIT.json)
- [全部请求和checkpoint表的索引](receipts/p02_result_binding_operator_probe_v1_20261022/SCHEMA.json)
- 实际完整请求与回答：[goal_G0](prompt_examples/p02_result_binding_operator_probe_v1_20261022/goal_G0.json) · [goal_GF](prompt_examples/p02_result_binding_operator_probe_v1_20261022/goal_GF.json) · [goal_GO](prompt_examples/p02_result_binding_operator_probe_v1_20261022/goal_GO.json) · [ordinary_B](prompt_examples/p02_result_binding_operator_probe_v1_20261022/ordinary_B.json) · [ordinary_D](prompt_examples/p02_result_binding_operator_probe_v1_20261022/ordinary_D.json)

## 4. 新60题：目标约束与综合文本展示

- [冻结实验协议](snapshot/experiments/query_decomposition/p02_goal_target_transfer_new60_v1_20261022/PROTOCOL.frozen.json)
- [完整模板与构造方式](snapshot/experiments/query_decomposition/p02_goal_target_transfer_new60_v1_20261022/templates.py)
- [实际接线与执行代码](snapshot/experiments/query_decomposition/p02_goal_target_transfer_new60_v1_20261022/run.py)
- [REPORT.md](snapshot/outputs/query_decomposition/p02_goal_target_transfer_new60_v1_20261022/REPORT.md)
- [COMPLETION.json](snapshot/outputs/query_decomposition/p02_goal_target_transfer_new60_v1_20261022/COMPLETION.json)
- [SUMMARY.json](snapshot/outputs/query_decomposition/p02_goal_target_transfer_new60_v1_20261022/SUMMARY.json)
- [CASE_NOTES.md](snapshot/outputs/query_decomposition/p02_goal_target_transfer_new60_v1_20261022/CASE_NOTES.md)
- [INTERPRETATION.zh.md](snapshot/outputs/query_decomposition/p02_goal_target_transfer_new60_v1_20261022/INTERPRETATION.zh.md)
- [OFFLINE_AUDIT.json](snapshot/outputs/query_decomposition/p02_goal_target_transfer_new60_v1_20261022/OFFLINE_AUDIT.json)
- [全部请求和checkpoint表的索引](receipts/p02_goal_target_transfer_new60_v1_20261022/SCHEMA.json)
- 实际完整请求与回答：[goal_G0](prompt_examples/p02_goal_target_transfer_new60_v1_20261022/goal_G0.json) · [goal_GF](prompt_examples/p02_goal_target_transfer_new60_v1_20261022/goal_GF.json) · [goal_GF_SPLIT](prompt_examples/p02_goal_target_transfer_new60_v1_20261022/goal_GF_SPLIT.json) · [goal_T](prompt_examples/p02_goal_target_transfer_new60_v1_20261022/goal_T.json)

## 5. 同60题：信息充分性、保留与final读入

- [冻结实验协议](snapshot/experiments/query_decomposition/p02_goal_sufficiency_retention60_v1_20261022/PROTOCOL.frozen.json)
- [完整模板与构造方式](snapshot/experiments/query_decomposition/p02_goal_sufficiency_retention60_v1_20261022/templates.py)
- [实际接线与执行代码](snapshot/experiments/query_decomposition/p02_goal_sufficiency_retention60_v1_20261022/run.py)
- [REPORT.md](snapshot/outputs/query_decomposition/p02_goal_sufficiency_retention60_v1_20261022/REPORT.md)
- [COMPLETION.json](snapshot/outputs/query_decomposition/p02_goal_sufficiency_retention60_v1_20261022/COMPLETION.json)
- [SUMMARY.json](snapshot/outputs/query_decomposition/p02_goal_sufficiency_retention60_v1_20261022/SUMMARY.json)
- [CASE_NOTES.md](snapshot/outputs/query_decomposition/p02_goal_sufficiency_retention60_v1_20261022/CASE_NOTES.md)
- [INTERPRETATION.zh.md](snapshot/outputs/query_decomposition/p02_goal_sufficiency_retention60_v1_20261022/INTERPRETATION.zh.md)
- [OFFLINE_AUDIT.json](snapshot/outputs/query_decomposition/p02_goal_sufficiency_retention60_v1_20261022/OFFLINE_AUDIT.json)
- [全部请求和checkpoint表的索引](receipts/p02_goal_sufficiency_retention60_v1_20261022/SCHEMA.json)
- 实际完整请求与回答：[goal_A](prompt_examples/p02_goal_sufficiency_retention60_v1_20261022/goal_A.json) · [goal_G0](prompt_examples/p02_goal_sufficiency_retention60_v1_20261022/goal_G0.json) · [goal_GF](prompt_examples/p02_goal_sufficiency_retention60_v1_20261022/goal_GF.json) · [goal_P](prompt_examples/p02_goal_sufficiency_retention60_v1_20261022/goal_P.json) · [goal_R](prompt_examples/p02_goal_sufficiency_retention60_v1_20261022/goal_R.json)

## 6. 新提示词套件设计 r01–r03

- [实验说明](snapshot/experiments/query_decomposition/p02_prompt_suite_design_v1_20261022/README.md)
- [套件设计](snapshot/experiments/query_decomposition/p02_prompt_suite_design_v1_20261022/DESIGN.md)
- [验证计划](snapshot/experiments/query_decomposition/p02_prompt_suite_design_v1_20261022/VALIDATION_PLAN.md)
- [r01 / FULL_PROMPTS.en.md](snapshot/experiments/query_decomposition/p02_prompt_suite_design_v1_20261022/revisions/r01/FULL_PROMPTS.en.md)
- [r01 / INTERFACE.md](snapshot/experiments/query_decomposition/p02_prompt_suite_design_v1_20261022/revisions/r01/INTERFACE.md)
- [r02 / FULL_PROMPTS.en.md](snapshot/experiments/query_decomposition/p02_prompt_suite_design_v1_20261022/revisions/r02/FULL_PROMPTS.en.md)
- [r02 / INTERFACE.md](snapshot/experiments/query_decomposition/p02_prompt_suite_design_v1_20261022/revisions/r02/INTERFACE.md)
- [r03 / FULL_PROMPTS.en.md](snapshot/experiments/query_decomposition/p02_prompt_suite_design_v1_20261022/revisions/r03/FULL_PROMPTS.en.md)
- [r03 / FULL_PROMPTS.zh.md](snapshot/experiments/query_decomposition/p02_prompt_suite_design_v1_20261022/revisions/r03/FULL_PROMPTS.zh.md)
- [r03 / INTERFACE.md](snapshot/experiments/query_decomposition/p02_prompt_suite_design_v1_20261022/revisions/r03/INTERFACE.md)

## 7. r03 stage90：新goal、新final与普通QA

- [冻结实验协议](snapshot/experiments/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/PROTOCOL.frozen.json)
- [实验说明](snapshot/experiments/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/README.md)
- [实际接线与执行代码](snapshot/experiments/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/run.py)
- [REPORT.zh.md](snapshot/outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/REPORT.zh.md)
- [REPORT.json](snapshot/outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/REPORT.json)
- [COMPLETION.json](snapshot/outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/COMPLETION.json)
- [ANALYSIS.json](snapshot/outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/ANALYSIS.json)
- [CHANGED_FINAL_CASES.md](snapshot/outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/CHANGED_FINAL_CASES.md)
- [GOAL90_FINDINGS.zh.md](snapshot/outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/GOAL90_FINDINGS.zh.md)
- [FAILURES.local.jsonl](snapshot/outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/FAILURES.local.jsonl)
- [RECEIPT_INTERFACE_AUDIT.json](snapshot/outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/RECEIPT_INTERFACE_AUDIT.json)
- [EXECUTION_NOTE.md](snapshot/outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/EXECUTION_NOTE.md)
- [全部请求和checkpoint表的索引](receipts/p02_prompt_suite_r03_stage90_v1_20261022/SCHEMA.json)
- 实际完整请求与回答：[goal_C](prompt_examples/p02_prompt_suite_r03_stage90_v1_20261022/goal_C.json) · [goal_G](prompt_examples/p02_prompt_suite_r03_stage90_v1_20261022/goal_G.json) · [goal_GF](prompt_examples/p02_prompt_suite_r03_stage90_v1_20261022/goal_GF.json) · [ordinary_O0](prompt_examples/p02_prompt_suite_r03_stage90_v1_20261022/ordinary_O0.json) · [ordinary_O1](prompt_examples/p02_prompt_suite_r03_stage90_v1_20261022/ordinary_O1.json)

## 8. r04 fixed90：简短综合S、目标规则T

- [冻结实验协议](snapshot/experiments/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/PROTOCOL.frozen.json)
- [完整模板与构造方式](snapshot/experiments/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/templates.py)
- [实际接线与执行代码](snapshot/experiments/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/run.py)
- [历史错误复核](snapshot/experiments/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/ERROR_REVIEW.zh.md)
- [普通回答补充复核](snapshot/experiments/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/ORDINARY_REVIEW_ADDENDUM.zh.md)
- [REPORT.zh.md](snapshot/outputs/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/REPORT.zh.md)
- [COMPLETION.json](snapshot/outputs/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/COMPLETION.json)
- [ANALYSIS.json](snapshot/outputs/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/ANALYSIS.json)
- [QUALITATIVE_SUMMARY.zh.md](snapshot/outputs/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/QUALITATIVE_SUMMARY.zh.md)
- [REVIEW_CASES.md](snapshot/outputs/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/REVIEW_CASES.md)
- [全部请求和checkpoint表的索引](receipts/p02_goal_repair_r04_fixed90_v1_20261023/SCHEMA.json)
- 实际完整请求与回答：[goal_C](prompt_examples/p02_goal_repair_r04_fixed90_v1_20261023/goal_C.json) · [goal_S](prompt_examples/p02_goal_repair_r04_fixed90_v1_20261023/goal_S.json) · [goal_T](prompt_examples/p02_goal_repair_r04_fixed90_v1_20261023/goal_T.json)
