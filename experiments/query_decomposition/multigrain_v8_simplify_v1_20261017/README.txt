本目录只研究query分解提示词；不训练模型、不修改冻结的RAG。

先读输出目录的REPORT.txt与CONFIRMATION_CONCLUSION.json：
outputs/query_decomposition/hotpot_formal_eval_v1/multigrain_v8_simplify_v1_20261017/

TRAINING_PROMPT_v8.txt：未改动对照，与上一实验逐字节相同。
TRAINING_PROMPT_v10/v11/v12.txt：三次完整候选提示词。
CHANGE_*.json＋DIFF_*.txt：每次修改的父版本、意图、时间、完整差异与长度。
DECISIONS.txt：包括失败尝试在内的过程与结果决策。
SELECTION_FROZEN.json：新样本、排除来源、配额、原始选择与停止规则。
SELECTION_AMENDMENT.json：v10后发现API可用性差异，在后续候选/确认前补充配对语义资格约束。
FINAL_CANDIDATE_FROZEN.json：只依据开发集选中的v11及确认前冻结凭据。
BEST_PROMPT.txt：开发阶段选中的精简候选v11的副本；不是已确认优于v8的主版本。
AUTHORED_dev/holdout.json：每题的原始评审代码与理由。
REVIEW_AMENDMENT_001.json：对含糊原句的评审更正与旧汇总位置。

study.py：新题抽样与原传输调用的隔离封装。v8/v10初轮使用旧批次调度；后续用同并发3的滑动窗口，payload/速率/预算/重试策略不变。
review_progress.py：仅对逐字相同列表沿用原判定；新输出仍由当前Codex明确审阅。
freeze.py：按开发集规则冻结候选，不读取确认集文本。
finish.py：核对完整payload、冻结哈希、单次请求、成本、配对结果和报告。

输出目录各dev_*/holdout_*子目录保留每题原query、完整模型输出、请求payload、用量与请求状态；未返回请求不重试。
*.REVIEW_ALL_ARMS.jsonl为完整逐题对照，*.RESULTS.json为完整统计。
REMAINING_CASES.jsonl为候选版本仍未通过/不确定/不可用的诊断记录；不等于已修正SFT标签，不自动加入训练。

确认后结论：保留v8为主版本。v11明显缩短，完整60题通过数持平，但共同有效58题净少1题，且指代失败增加；不能称为语义优化成功，也未接近90%。不在已看过的确认集上继续改prompt。
