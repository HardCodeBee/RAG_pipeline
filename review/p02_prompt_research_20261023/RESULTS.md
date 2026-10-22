# 各实验、各数据集的结果

数值为百分数（F1 / EM）。表格直接取原始SUMMARY/ANALYSIS文件；独立校验脚本逐题重算后核对。不同实验不合并排名。

## Few-shot fixed90：旧G2、新指令I、四示例F

三组共同可计分86题（HotpotQA26、2Wiki30、MuSiQue30）。all是按题平均；另列数据集等权平均。1条算法失败保留为0分；7个未知条件不强行计分。历史原生P02接口不同，仅作参考。

[原始统计文件](snapshot/outputs/query_decomposition/p02_goal_fewshot_fixed90_v1_20261022/SUMMARY.json)

| 数据集 | 共同题数 | 组别 | F1 | EM |
|---|---:|---|---:|---:|
| all | 86 | native_historical_P02 | 75.99 | 66.28 |
| all | 86 | G2 | 68.38 | 58.14 |
| all | 86 | I | 69.87 | 59.30 |
| all | 86 | F | 68.78 | 58.14 |
| hotpotqa | 26 | native_historical_P02 | 80.50 | 69.23 |
| hotpotqa | 26 | G2 | 76.61 | 65.38 |
| hotpotqa | 26 | I | 76.01 | 61.54 |
| hotpotqa | 26 | F | 71.45 | 57.69 |
| 2wiki | 30 | native_historical_P02 | 94.33 | 90.00 |
| 2wiki | 30 | G2 | 83.01 | 73.33 |
| 2wiki | 30 | I | 86.13 | 76.67 |
| 2wiki | 30 | F | 90.02 | 80.00 |
| musique | 30 | native_historical_P02 | 53.74 | 40.00 |
| musique | 30 | G2 | 46.63 | 36.67 |
| musique | 30 | I | 48.28 | 40.00 |
| musique | 30 | F | 45.23 | 36.67 |
| 三数据集等权 | 26/30/30 | native_historical_P02 | 76.19 | 66.41 |
| 三数据集等权 | 26/30/30 | G2 | 68.75 | 58.46 |
| 三数据集等权 | 26/30/30 | I | 70.14 | 59.40 |
| 三数据集等权 | 26/30/30 | F | 68.90 | 58.12 |

本轮新增返回usage估算费用：1.1401780元；本轮保留未知费用预留：0.0276520元。缓存对应逻辑开销与新增费用另列于原报告，不能混加。

另保留旧42题尝试的未知预留 0.0548024 元，未当作已确认支出。

## 结果绑定与显式组合操作

goal共同90题，每集30。G0/GF/GO共享final接口；另有38对普通节点，普通分支不产生下游F1。

[原始统计文件](snapshot/outputs/query_decomposition/p02_result_binding_operator_probe_v1_20261022/SUMMARY.json)

| 数据集 | 共同题数 | 组别 | F1 | EM |
|---|---:|---|---:|---:|
| all | 90 | historical_native_P02 | 76.69 | 66.67 |
| all | 90 | G0 | 76.62 | 65.56 |
| all | 90 | GF | 73.76 | 63.33 |
| all | 90 | GO | 70.21 | 58.89 |
| hotpotqa | 30 | historical_native_P02 | 81.99 | 70.00 |
| hotpotqa | 30 | G0 | 81.99 | 70.00 |
| hotpotqa | 30 | GF | 75.75 | 60.00 |
| hotpotqa | 30 | GO | 73.20 | 60.00 |
| 2wiki | 30 | historical_native_P02 | 94.33 | 90.00 |
| 2wiki | 30 | G0 | 92.80 | 86.67 |
| 2wiki | 30 | GF | 94.13 | 90.00 |
| 2wiki | 30 | GO | 91.69 | 83.33 |
| musique | 30 | historical_native_P02 | 53.74 | 40.00 |
| musique | 30 | G0 | 55.07 | 40.00 |
| musique | 30 | GF | 51.42 | 40.00 |
| musique | 30 | GO | 45.74 | 33.33 |

本轮新增返回usage估算费用：1.2994076元；本轮保留未知费用预留：0.0000000元。缓存对应逻辑开销与新增费用另列于原报告，不能混加。

## 新60题：目标约束与综合文本展示

共同60题，每集20，排除了当时清单中已用题，但来自既有开发/评估材料，并非完全未见测试。GF_SPLIT复用同一GF文本，只改展示包装。

[原始统计文件](snapshot/outputs/query_decomposition/p02_goal_target_transfer_new60_v1_20261022/SUMMARY.json)

| 数据集 | 共同题数 | 组别 | F1 | EM |
|---|---:|---|---:|---:|
| all | 60 | historical_native_P02 | 67.99 | 55.00 |
| all | 60 | G0 | 67.47 | 56.67 |
| all | 60 | T | 66.97 | 56.67 |
| all | 60 | GF | 68.26 | 60.00 |
| all | 60 | GF_SPLIT | 68.50 | 61.67 |
| hotpotqa | 20 | historical_native_P02 | 74.55 | 55.00 |
| hotpotqa | 20 | G0 | 71.00 | 60.00 |
| hotpotqa | 20 | T | 69.33 | 55.00 |
| hotpotqa | 20 | GF | 73.62 | 60.00 |
| hotpotqa | 20 | GF_SPLIT | 74.33 | 65.00 |
| 2wiki | 20 | historical_native_P02 | 96.39 | 85.00 |
| 2wiki | 20 | G0 | 96.39 | 85.00 |
| 2wiki | 20 | T | 98.89 | 90.00 |
| 2wiki | 20 | GF | 99.44 | 95.00 |
| 2wiki | 20 | GF_SPLIT | 99.44 | 95.00 |
| musique | 20 | historical_native_P02 | 33.02 | 25.00 |
| musique | 20 | G0 | 35.02 | 25.00 |
| musique | 20 | T | 32.69 | 25.00 |
| musique | 20 | GF | 31.72 | 25.00 |
| musique | 20 | GF_SPLIT | 31.72 | 25.00 |

本轮新增返回usage估算费用：0.8297532元；本轮保留未知费用预留：0.0000000元。缓存对应逻辑开销与新增费用另列于原报告，不能混加。

## 同60题：信息充分性、保留与final读入

共同60题，与上一实验相同。G0/GF控制回执严格复用，不能当作独立重复观测。

[原始统计文件](snapshot/outputs/query_decomposition/p02_goal_sufficiency_retention60_v1_20261022/SUMMARY.json)

| 数据集 | 共同题数 | 组别 | F1 | EM |
|---|---:|---|---:|---:|
| all | 60 | G0 | 67.47 | 56.67 |
| all | 60 | A | 66.80 | 56.67 |
| all | 60 | GF | 68.26 | 60.00 |
| all | 60 | P | 67.83 | 60.00 |
| all | 60 | R | 68.26 | 60.00 |
| hotpotqa | 20 | G0 | 71.00 | 60.00 |
| hotpotqa | 20 | A | 71.00 | 60.00 |
| hotpotqa | 20 | GF | 73.62 | 60.00 |
| hotpotqa | 20 | P | 74.33 | 65.00 |
| hotpotqa | 20 | R | 73.62 | 60.00 |
| 2wiki | 20 | G0 | 96.39 | 85.00 |
| 2wiki | 20 | A | 96.39 | 85.00 |
| 2wiki | 20 | GF | 99.44 | 95.00 |
| 2wiki | 20 | P | 96.94 | 90.00 |
| 2wiki | 20 | R | 99.44 | 95.00 |
| musique | 20 | G0 | 35.02 | 25.00 |
| musique | 20 | A | 33.02 | 25.00 |
| musique | 20 | GF | 31.72 | 25.00 |
| musique | 20 | P | 32.22 | 25.00 |
| musique | 20 | R | 31.72 | 25.00 |

本轮新增返回usage估算费用：0.5179048元；本轮保留未知费用预留：0.0000000元。缓存对应逻辑开销与新增费用另列于原报告，不能混加。

## r03 stage90：新goal、新final与普通QA

goal共同90题。C为role_headers历史控制，G只换goal，GF同时换goal与final。另有ordinary176对，其中O1一条length失败；新ordinary未传播到goal/final。

[原始统计文件](snapshot/outputs/query_decomposition/p02_prompt_suite_r03_stage90_v1_20261022/ANALYSIS.json)

| 数据集 | 共同题数 | 组别 | F1 | EM |
|---|---:|---|---:|---:|
| all | 90 | C | 76.12 | 66.67 |
| all | 90 | G | 68.98 | 58.89 |
| all | 90 | GF | 66.80 | 56.67 |
| hotpotqa | 30 | C | 82.06 | 70.00 |
| hotpotqa | 30 | G | 69.92 | 56.67 |
| hotpotqa | 30 | GF | 69.92 | 56.67 |
| 2wiki | 30 | C | 93.57 | 90.00 |
| 2wiki | 30 | G | 86.13 | 76.67 |
| 2wiki | 30 | GF | 86.13 | 76.67 |
| musique | 30 | C | 52.74 | 40.00 |
| musique | 30 | G | 50.90 | 43.33 |
| musique | 30 | GF | 44.36 | 36.67 |

本轮新增返回usage估算费用：1.8302568元；本轮保留未知费用预留：0.0000000元。缓存对应逻辑开销与新增费用另列于原报告，不能混加。

## r04 fixed90：简短综合S、目标规则T

共同90题。C严格复用r03控制；S只改变goal输出约定，T在S上加目标解释块。G_r03为上一实验参考，不算本轮新条件。

[原始统计文件](snapshot/outputs/query_decomposition/p02_goal_repair_r04_fixed90_v1_20261023/ANALYSIS.json)

| 数据集 | 共同题数 | 组别 | F1 | EM |
|---|---:|---|---:|---:|
| all | 90 | C | 76.12 | 66.67 |
| all | 90 | G_r03 | 68.98 | 58.89 |
| all | 90 | S | 68.06 | 57.78 |
| all | 90 | T | 67.98 | 57.78 |
| hotpotqa | 30 | C | 82.06 | 70.00 |
| hotpotqa | 30 | G_r03 | 69.92 | 56.67 |
| hotpotqa | 30 | S | 69.55 | 60.00 |
| hotpotqa | 30 | T | 71.74 | 56.67 |
| 2wiki | 30 | C | 93.57 | 90.00 |
| 2wiki | 30 | G_r03 | 86.13 | 76.67 |
| 2wiki | 30 | S | 86.11 | 76.67 |
| 2wiki | 30 | T | 88.70 | 80.00 |
| musique | 30 | C | 52.74 | 40.00 |
| musique | 30 | G_r03 | 50.90 | 43.33 |
| musique | 30 | S | 48.51 | 36.67 |
| musique | 30 | T | 43.49 | 36.67 |

本轮新增返回usage估算费用：0.7278448元；本轮保留未知费用预留：0.0000000元。缓存对应逻辑开销与新增费用另列于原报告，不能混加。
