# r03固定90题阶段试验

本目录保存独立试验的冻结输入、生成配置和可复核脚本。r03模板来自相邻的 `p02_prompt_suite_design_v1_20261022/revisions/r03`；没有修改生产提示词。

完整报告及执行状态在仓库的 `outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/REPORT.zh.md` 和 `COMPLETION.json`。在后者生成前，不能把整个622条件试验称作完成。

比较包含90题的C/G/GF三臂及176个普通节点的O0/O1两臂。C/O0来源是冻结role_headers轨迹；详情见 `COMPARATOR_NOTE.md`。本轮固定reader与普通历史，新普通答案不进入goal。没有测试新rewrite或动态完整RAG，不代表正式原始P02的重跑。

`PROTOCOL.frozen.json`、`ANALYSIS.frozen.json` 和 `QUALITATIVE_REVIEW_PLAN.frozen.json` 分别记录推理范围、配对统计和事先选择的普通回答复核样本。28个最终文本变化案例和预选18个普通节点的复核记录均在输出目录。已知算法失败保留，不按质量重采样。

只读推理记录的分析入口为模块 `analyze`、`audit`、`diagnostics`；它们仅更新本试验的本地报告，不调用生成API。`run run` 才是付费推理入口，完成后无须再次运行。主评估保持暂停。
