# 数据字典与信息边界

## 1. 紧凑逐次结果

以下三个 `*.jsonl.gz` 使用同一字段结构：

- `data/phase2/phase2_bd_600_outcomes.jsonl.gz`：600 queries × 2 actions × 3 repeats = 3,600 rows；
- `data/phase26/phase26_bd_4800_outcomes.jsonl.gz`：4,800 queries × 2 actions × 3 repeats = 28,800 rows；
- `data/contriever/phase3_contriever_1200_outcomes.jsonl.gz`：1,200 queries × 3 actions × 3 repeats = 10,800 rows。

主键为 `(query_id, action, repeat_id)`；Phase 2 还需结合 `split`，Phase 2.6/Contriever 可用 `query_rank` 保持冻结顺序。

| 字段组 | 字段 | 含义 |
|---|---|---|
| 标识 | `query_id`, `group_id`, `split`, `query_rank` | query、信息需求分组、分区和冻结顺序 |
| 输入/参考 | `question`, `reference_answers`, `supporting_titles`, `gold_doc_ids` | 仅用于离线误差分析，不是可部署 Router 输入 |
| 动作 | `action`, `repeat_id`, `status` | 检索动作和重复生成编号 |
| 输出 | `prediction` | 生成的短答案 |
| 答案指标 | `normalized_token_f1`, `normalized_exact_match`, `answer_correctness` | F1/EM；AC 仅在已执行 AC judging 的数据中非空 |
| 检索诊断 | `retrieved_doc_ids`, `retrieval_scores`, `evidence_page_recall`, `retrieval_hit` | 检索后诊断，禁止作为严格预检索 Router 输入 |
| 成本/延迟 | retrieval/generation latency、context token count、input/output tokens | 运行成本诊断 |

未导出 `context.text`、provider request/response、judge 原始回复、API 凭据。

## 2. Query-level 汇总

三个 `*_query_summary.csv.gz` 将同一 query/action 的三次生成先取均值，再计算 winner 和 oracle：

- `*_mean_f1`, `*_mean_em`, `*_mean_ac`：action 内三次均值；
- `f1_winner`, `ac_winner`：严格大于时的唯一胜者，否则为 `tie`；
- `f1_oracle`, `ac_oracle`：逐 query 事后最大值，仅为非部署上界；
- `f1_gap_*`：两动作均值差；
- evidence recall/hit：三次重复的均值，实际上同一检索动作通常一致。

## 3. 特征矩阵

### Phase 2.6

`features/phase26/phase26_features_4800.npz`：

- `query_ids`: `(4800,)`
- `group_ids`: `(4800,)`
- `lexical`: `(4800, 17)`
- `dense`: `(4800, 13)`，这里表示 corpus-static prototype summaries，不是检索结果
- `embedding`: `(4800, 384)`，BGE query embedding

三块合计 414 维。完整字段名见 `features/phase26/feature_schema.json`。

### Contriever 修订

`features/contriever/contriever_features_1200.npz`：

- `query_ids`: `(1200,)`
- `group_ids`: `(1200,)`
- `lexical`: `(1200, 17)`
- `dense`: `(1200, 13)`
- `embedding`: `(1200, 768)`，Contriever query embedding

## 4. 泄漏边界

- 可部署预检索输入：query 文本变换、词法/IDF/DF、冻结 query embedding、corpus-static prototypes。
- 特权离线标签：qrels、gold evidence、F1、EM、AC、oracle winner。
- 检索后诊断：doc IDs、scores、evidence recall/hit。
- 生成后诊断：prediction、F1/EM/AC、token usage。

对方进行建模时必须只从特征矩阵取输入；其他字段只能作为 label、分层变量或诊断证据。

## 5. 评分注意事项

- BM25 分数、BGE/Contriever inner product 不在同一标尺上，不可直接比较数值大小。
- Query oracle、qrels winner、F1/AC winner 都是事后标签，不是在线策略。
- `bge` 在 Contriever 数据中指原 BGE Dense action；`dense` 在 B/D 数据中同样指 BGE-small-en-v1.5。
- CQADupStack/Selected-7 小表只是上游检索背景，不能与 HotpotQA Answer F1/AC 混作同一指标。
