# M6：BEIR test7405 身份审计计划（2026-09-15）

## 目的与当前边界

补齐完整 BEIR test7405 的 canonical group hash，核实它与已消费 Router 来源的组级交叉。此次只做身份审计，保存全部 7405 行；不选择新的验证子集，不拟合、不计算新效用、不调用 API。

**执行顺序：先审阅本计划、独立源码与纯合成检查；根代理明确确认后才 freeze / run。** 编写与 self-test 阶段不读取真实 queries 的元字段。freeze 只读取输入字节并保存哈希；run 才进行下述 92405 行元字段投影。冻结后不改源码或本计划；输出拒绝覆盖，失败后不得无记录重跑。

本次 `metadata.answer` 与 supporting-facts 标题读取专用于已事先确定的身份规则，是参考元数据身份审计。它们不作为质量标签、路由输入、训练目标或选择依据；不输出这些原文本。完整 test7405 已参与历史检索基准，**是否曾影响方法或候选选择仍须审计，组隔离不能认证新独立确认集**。

## 输入与允许读取

入口：[audit_m6_test_group_identities.py](../../scripts/audit_m6_test_group_identities.py)。固定输出目录：`outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1`。

| 输入 | 使用字段或用途 |
|---|---|
| `data/beir/hotpotqa/queries/queries.jsonl` | 每行 `_id`；仅 train85000 + test7405 的 `metadata.answer` 与 `metadata.supporting_facts[*][0]` 标题 |
| 同数据目录 `qrels/train.tsv`、`dev.tsv`、`test.tsv` | 只解码首列 query ID；其余列不用 |
| `outputs/router/hotpotqa_bd_router_v1/split.json` | assignments 中 query_id / group_id；验证 old85000 原组对应 |
| `../work/router_research/beir_dev_group_identities.npz` | 已保存 dev5447 ID、hash、三个交叉布尔数组 |
| `../work/router_research/m6_beir_validation_v1/identities.npz`、`protocol.json` | 已验证 5209 ID / 5206 hash 与冻结 sample 哈希 |
| `beir_dev_eligibility.json`、`beir_dev_eligibility_checks.json`、`beir_dev_validation_supplement.json`（均位于 work/router_research） | 原分组结果、独立复核、官方归档成员字节绑定；不重新联网 |
| `m6_validation_sources.json`、`m6_validation_sources_checks.json`（同目录） | 原消费来源清单与“内部无未消费组剩余”复核绑定 |
| `analysis/hotpotqa_router/m6_confirmation_inventory_20260915.json` | 既有全 test7405 检索暴露的身份清单；不重扫结果文件 |

脚本按字节扫描 JSON；要求 `_id` 为首字段。`text` 仅检查字符串边界和转义并跳过，绝不 JSON 解码。dev5447 的整段 metadata 仅按 JSON 结构跳过；其已保存 hash 直接复用。选中行仅解码 answer 和标题，sentence index 只验证整数语法、不求其值。哈希后不保留字段正文。异常只输出固定错误码，禁止原始 decoder 异常或带正文的 traceback。

## 唯一 canonical 规则

1. answer 使用 `lower()`，删除 Python `string.punctuation` 中 ASCII 标点。
2. 用 `re.sub(r'\b(a|an|the)\b', ' ', value)` 删除冠词；按 Python Unicode 空白 `.split()` 后单空格连接。
3. 标题保持原始大小写、标点与空白；按 Python 字符串排序后去重。标题顺序、重复事实及句子索引不影响 key。
4. 计算 `SHA256(json.dumps([answer_normalized, sorted(set(titles))], ensure_ascii=False, separators=(',', ':')).encode('utf-8'))`。

query ID、问题正文、qrel 文档与相关性、检索分数、模型分数、生成答案、F1 均不参加分组。`lower` 不换成 `casefold`；标题不做 answer 归一化。

## 固定验收

- qrels 三个 ID 集分别为 85000 / 5447 / 7405，互斥并覆盖 queries 中全部 97852 行。实际 reference 元字段投影恰为 **92405 行**。
- old85000 ID 与原 split 完全一致；每个原 group_id 唯一映射 canonical hash，且每个 canonical hash 唯一映射原 group_id：**双向 bijection，共 83167 组**。
- dev5447 已存 hash 的三组交叉标记必须逐行重现；旧组交叉 225 查询 / 220 组；test 交叉 21 查询 / 21 组；同时交叉 8 查询 / 8 组；两者都不交叉 5209 查询 / 5206 组。
- 冻结验证 5209 IDs/hash 必须逐行等于 dev 的既有 `no_either_group_overlap` 投影。test 与这 5206 组交叉必须为 0。
- 既有 train∩dev∩test 桥接 **8 个 distinct hash** 必须全部出现在新重建的 train∩test 中；完整 test↔old 交叉组数至少为 8，不能被误报为零。
- 已消费 Router 来源采用保守集合：old83167 全组 ∪ 已验证 dev5206 组，共 88373 组。目标自身的历史检索暴露另存为全 true，不能把自身加入前述交叉集合造成循环判定。
- 当前输入绑定须匹配旧分组审计、独立 checker、官方归档成员校验及验证 sample；run 前后重新核对固定输入字节 SHA256。

## 输出 schema

所有文件只包含 ID/hash、计数、布尔标记或协议元数据，无参考文本、问题正文和效应。

- `protocol.json`：源码、计划及上述输入路径 / SHA256；固定 config；合成检查记录。
- `run_started.json`：绑定协议、开始时间与身份投影边界，防止无记录重复运行。
- `old85000_group_identities.npz`：按 ID 排序的 `query_ids`、`original_group_ids`、`group_key_sha256`，保留完整双向对应证据。
- `test7405_group_identities.npz`：按 ID 排序的全部 `query_ids`、`group_key_sha256`，及以下逐行 bool：
  - `overlaps_internal85000_group`
  - `overlaps_validated5209_group`
  - `overlaps_dev5447_group`
  - `overlaps_other_consumed_router_group`（old ∪ validated）
  - `shares_train_dev_test_bridge_group`
  - `canonical_group_isolated_from_router_history`（前述 union 交叉的补集，**不是资格认证**）
  - `prior_retrieval_benchmark_exposure`（全部 true，根据已保存六份全量 ID 清单）
- `results.json`：全体/组计数、双向映射验收、dev 桥接复核、test 对各集合交叉的 query / distinct-group 数与 hash、字段投影计数、零效应读取边界及产物 SHA256。
- `completion.json`：终态和全部产物 SHA256；终态始终保留 `selection_history_unresolved`。

交叉的 query 数按目标 test 行计数，group 数按目标命中 hash 去重；跨集合数字不应直接相加。输出完整 7405 行，不生成“推荐样本”文件或基于交叉的选样动作。

## 纯合成检查与运行命令

合成检查包括：独立字符级 answer 归一化对照、Unicode/ASCII 区别、lower/casefold 区别、标题去重与精确大小写、转义假 metadata 与跳过问题正文、非目标 metadata 不解码、ID/正文/句子索引不影响 key、非法字段/索引/转义拒绝、双向 bijection 两种冲突、query 与 group 交叉分母。它不读取任何真实数据。

```powershell
& 'C:/Users/12442/anaconda3/python.exe' -B -X utf8 scripts/audit_m6_test_group_identities.py self-test
# 以下两步必须等根代理审阅确认后执行；freeze 输出 protocol SHA256。
& 'C:/Users/12442/anaconda3/python.exe' -B -X utf8 scripts/audit_m6_test_group_identities.py freeze
& 'C:/Users/12442/anaconda3/python.exe' -B -X utf8 scripts/audit_m6_test_group_identities.py run --protocol-sha256 '<freeze 输出的完整 SHA256>'
```

## 能回答与仍不能回答

这次能补齐全部 7405 的历史 canonical group 身份缺口，并给出可复算的组交叉，避免只用 query ID 零交叉声称独立。不能证明语义近重复缺失、未记录/远端/人工选择史缺失，也不独立重现原始 HotpotQA ID 映射或 BEIR 随机 split 算法。是否能作为新的确认来源，须先完成历史检索结果及方法选择使用史审计，再另行冻结候选、数据规则和验证协议；本次不执行这些后续步骤。
