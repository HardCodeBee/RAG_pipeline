# M6：BEIR test7405 历史选择使用审计

2026-09-15。有界、只读的本地历史审计；接续[库存快照](m6_confirmation_inventory_20260915.md)与[完整 group 身份审计](m6_test_group_identity_results_20260915.md)。[证据 JSON](m6_test_selection_history_audit_20260915.json)保存 14 个来源的路径、当前 SHA256、行号与结论，以及搜索范围和未解决问题。

## 1. 资格结论

**7405 条 official test 已参与检索政策和参数探索，并用于形成后续研究建议，不能再描述为未经研究使用的 fresh 确认来源。** 这有正面证据，不依赖“没有找到其他缓存”的推断。

完整身份审计得到的 **7169 查询 / 7158 组**，仅与已消费 Router 来源的 canonical group 不交。去掉共享组的 236 条查询，不会消除整个 test 来源已有的检索分析和研究设计使用史。本次保留全部 7405 条，不选择新验证子集，也不认证新的独立确认集。

本轮范围内没有找到 test7405 的生成答案 F1/EM 或已训练 Router 效果曝光证据；这项有限的未发现，**不能证明历史上从未发生**。若以后在该来源评价答案效用，应披露既有检索结果的使用，并单独论证所声称的独立性；本审计不自动赋予该资格。

## 2. 新补齐的证据链

| 已确认事项 | 直接证据 | 能支持的判断 |
|---|---|---|
| 全 test 检索比较已经完成并报告 | [原 BEIR 报告](../../docs/beir_experiment_results.md)，5–28、35–46、66–76 行：截至 8 月 8 日，HotpotQA 共 7405 条；比较 BM25/Dense 与 BGE 重排条件，使用 BGE-small encoder | 已暴露检索指标与路径优劣；报告明确不含生成答案指标 |
| Top-k 是已有分析维度 | 同一报告的 Top-5/Top-10；[回放配置](../../configs/beir/topk_selected7_5_10_20_50_v1.yaml)，4、11、17、23、29 行列出 5/10/20/50 及 HotpotQA 原运行/派生运行 | 不能将该来源当作对评价深度毫无历史检查的来源；配置本身不证明最终 `final_k=5` 的选择原因 |
| HotpotQA 原结果进入 test-qrels 政策探索 | [源清单](../../docs/assets/history_aware_retriever_policy/analysis/source_files.csv)，8–9 行，直接绑定原四条件的 Top-50 BM25/Dense 派生结果；[metadata](../../docs/assets/history_aware_retriever_policy/analysis/metadata.json)，11–40 行 | 分析使用 21 个 α、equal50/full100 两预算，以及 test-best α 和 query oracle；不只是汇报单个固定检索器 |
| 这些比较确实形成研究方向建议 | [history-aware 报告](../../docs/history_aware_retriever_policy_report.md)，187–205、219–247 行 | 明确建议 Dense 默认先验、lexical/hybrid 例外，并将 HotpotQA 多证据查询列为 hybrid gate 重点候选；oracle 仍是描述性上界，并非训练好的策略 |
| BGE/DPR/Contriever 比较进入 Router 研究论证 | [8 月 19 日来源记录](../../../work/experiment_records/2026-08-19/data/source_runs.md)，13–40 行，连接原二路结果与 `beir_selected7_dense_baselines_v2`；[结论](../../../work/experiment_records/2026-08-19/conclusion.md)，23–27、172–191 行包含 HotpotQA 7405 | 文档明确写“BM25 vs BGE 是合理的第一阶段 routing 任务”，并讨论 Dense 模型偏好；同时将 answer utility 列为后续标签方向 |
| Router 原配置已承认 official test 的历史角色 | [归档配置](phases/phase00_26/results/source_configs/hotpotqa_bd_router_config.yaml)，1–7 行 | `official_test: historical_retrieval_evidence_only`；该标记在 Git `c9aa553` 中也存在，历史角色并非本轮新推定 |

源清单当前 SHA256 为 `bbd8d8c307d129c5b1e5549f16ffa7d391bb844b17971381d3a8d716e4e54860`，与历史 metadata 及[8 月 9 日通过的回放验证](../../docs/assets/history_aware_retriever_policy/analysis/deterministic_replay_validation.json)一致。清单共 36 行，HotpotQA 两行直接指向 `beir_hotpotqa_four_condition_full_v2_top50_offline_v1`。本次只核对清单与记录的绑定；没有打开其 per-query payload，也没有重新计算那些 payload 的当前 SHA。

## 3. 哪些选择因果仍不能认定

- **初选 BGE 的原因：** 原四条件报告在 8 月 8 日已使用 BGE；Dense-family 矩阵的[完成 metadata](../../outputs/beir_selected7_dense_baselines_v2/metadata.json)，376–377 行，是 8 月 16 日。因此，不能把后发生的 BGE/DPR/Contriever 矩阵当成更早初选 BGE 的原因；它确实被后续研究论证使用。
- **重排与最终深度：** [四路径历史结论](../../../work/experiment_records/2026-08-08/2/conclusion.md)，501–519 行推荐 HotpotQA 使用 Dense+BGE 重排；实际 Router 配置却为 `reranker: none`、`candidate_k: 50`、`final_k: 5`。建议存在与建议被采用是两件事。尚缺具体决策记录来确定最终无重排、深度 5 的选择依据。BGE-small encoder 与 BGE reranker 也应分开表述。
- **当前 M6/Student：** 未找到 test7405 直接选择 M6 的层、读出头或本轮 Student 配方的证据。不能把一般 Router 研究建议扩大为对这些具体选择的已证实因果链。
- **答案效用：** 历史来源记录明确限定检索指标、排除生成答案指标；早期库存也未找到 test7405 答案/Router 缓存。已知后续 Contriever 答案实验使用的是冻结的 **train1200**，见[完整旧报告](complete_research_report_20260901.md)，650–661 行，不能据此声称 test7405 答案已暴露。删除的、远程的、未归档的评价或人工查看仍未被排除。

## 4. 范围、未知历史与下一步边界

本轮定向检索 repo 的 `docs/configs/scripts/analysis`、README，以及 `../work/experiment_records` 和 `../work/router_research` 的历史文本，围绕两运行名、BGE/DPR/Contriever、test-qrels、α、routing 与 official-test 角色追踪来源；再按指定路径和字符串查 Git 历史。早期记忆索引仅用于定位，结论均回到当前本地文件核实。未搜索原始会话日志。

Git 保留了 8 月 12 日的回放/政策分析工具（`f649046`）、8 月 20 日的 Dense matrix 工具（`5cdfbd9`）和 9 月 1 日的 Router 配置（`c9aa553`）。但 `docs/` 被 `.gitignore` 第 32 行忽略，sibling work 也不在该 Git 仓库内；文档自报日期不等于获得完整版本历史认证。未命中的 Git 字符串搜索不证明没有历史使用。

现有正面证据已足以否定“未经检索研究使用”的描述；继续搜索不能让这项已发生的暴露消失。若仍考虑较窄的、预先冻结的答案效用验证，尚需明确：

1. 哪些历史结果实际用于锁定 encoder、retriever、reranker、top-k、M6 和评价条件，以及相应的带时间决策记录。
2. 既有答案生成、答案评分及 Router 评价运行的来源清单，覆盖本地库存以外的可能记录；不能用当前文件缺失代替这一证明。
3. 该验证要支持的具体命题及独立性边界。若目标是未经来源选择影响的独立确认，应另行寻找并论证合格来源；仅删除共享组不足以达成。

本次新增读取问题正文、参考元字段、答案 payload 均为 **0**；新增策略效果计算、拟合、网络与收费调用均为 **0**。只写本报告和证据 JSON，未修改冻结旧文件或 sibling progress。
