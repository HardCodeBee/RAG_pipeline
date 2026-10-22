三段 prompt 第二套整体 quick90 结果

同一批开发90题，各数据集30题；三段均按用户提供原文替换。模型、分解、改写、检索、formatter及256/128/64输出上限沿用原版。
冻结原版为精确回放；两套候选按自身前置回答继续改写与检索，仅成功且完整payload相同才复用。上一套单题goal512诊断不混入该表。

| 数据集 | 题数 | 冻结原版F1 | 上一套F1 | 本套F1 | 本套减原版 | 本套减上一套 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | 30 | 81.99 | 77.61 | 81.04 | -0.95 | +3.43 |
| 2wiki | 30 | 94.33 | 82.80 | 91.90 | -2.43 | +9.10 |
| musique | 30 | 53.74 | 54.71 | 55.19 | +1.45 | +0.48 |
| overall | 90 | 76.69 | 71.71 | 76.04 | -0.64 | +4.34 |

总体EM：原版 66.67%，上一套 58.89%，本套 64.44%。
相对原版F1升/平/降：7/77/6；相对上一套：11/71/8。
相对原版F1差值95%分层配对bootstrap区间：[-4.3628, 2.8677] 个百分点；相对上一套：[-1.1036, 10.2597] 个百分点。
终态：{'success': 90}；length响应：0。算法失败按既定规则保留在90题分母中计0。
实际prompt逐项核对：{'ordinary': 176, 'goal': 90, 'final': 90}；离线额外核对全部6个无普通子问题案例。
本轮新增估算生成费用 0.661530 元，未结算预留 0.096457 元。逻辑tokens和旧费用不重复计作新增费用。
该90题已反复用于开发，本次为三段联合效果，不能判断单段贡献，也不能作为独立确认。未激活候选；其他暂停实验保持暂停。

Technical recovery audit: {"technical_reissues": 13, "maximum_actual_submissions_per_exact_payload": 3, "all_old_receipts_preserved": true, "all_old_unknown_reservations_preserved": true, "old_unknown_reserved_CNY": 0.09645680000000001, "terminal_statuses": {"success": 90}, "actual_prompt_audit": {"ordinary": 176, "goal": 90, "final": 90}, "length_responses": 0}
