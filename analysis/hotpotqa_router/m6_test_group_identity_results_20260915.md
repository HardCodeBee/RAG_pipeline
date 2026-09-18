# M6：BEIR test7405 完整 group 身份审计结果

2026-09-15。按[预先冻结计划](m6_test_group_identity_plan_20260915.md)完成身份审计；根代理执行原入口，随后另一数值路径复核保存的身份数组。**全部 7405 条已保存，完整 group 缺口已补齐；独立确认资格仍待历史选择使用审计。**

原产物：[协议](../../outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1/protocol.json)、[结果](../../outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1/results.json)、[终态绑定](../../outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1/completion.json)、[独立复核](../../outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1/separate_checks.json)。

## 1. 完整交叉计数

BEIR test 共 **7405 查询 / 7392 个 canonical group**。下表 query 数均以 test 行为分母，group 数对命中的 hash 去重；对照集合之间有重叠，不能直接相加。

| 对照集合 | 对照查询 / 组 | 命中的 test 查询 | 共享组 |
|---|---:|---:|---:|
| 内部 BEIR train 全体 | 85000 / 83167 | 236 | 234 |
| 已验证、已消费 BEIR dev | 5209 / 5206 | 0 | 0 |
| dev 全体（桥接对照） | 5447 / 5439 | 22 | 21 |
| 已消费 Router 组集合：内部 ∪ 已验证 dev | 90209 / 88373 | 236 | 234 |
| train∩dev∩test 桥接组 | 8 个组 | 9 | 8 |

test 与内部、dev 全体、已验证 dev 的 **query ID 交集均为 0**，但其与内部的 canonical group 交集为 234。原来“至少 8 组”的桥接下界得到完整补齐。

**236 条是共享组的查询，绝不是质量失败。** 其补集为 **7169 查询 / 7158 组**，仅表示与上述已消费 Router 组集合无 canonical-key 交叉。审计未据此选择新的验证样本；[test 身份归档](../../outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1/test7405_group_identities.npz)仍包含全部 7405 行及所有交叉标记。

### dev 侧 21 条与 test 侧 22 条为什么不同

两侧共享同样的 21 个组，但按各自查询行计数：dev 为 **21 查询 / 21 组**，test 为 **22 查询 / 21 组**。某个共享组在 test 有两条查询，因此两侧 query 数不同。三方桥接也相同：dev 侧 **8 查询 / 8 组**，test 侧 **9 查询 / 8 组**。这不是计数矛盾，也不是重复验证样本可当独立组。

## 2. 规则与复核

canonical key 沿用原规则：answer 做 lowercase、ASCII 标点删除、冠词删除与空白归一化；supporting-facts 标题按原文精确去重排序；二者的紧凑 UTF-8 JSON 取 SHA256。问题正文、ID、句子索引、检索或答案效用不参与 key。

- old85000 的原始 query/group 配对逐行对应；原 group_id ↔ canonical hash **双向 bijection，共 83167 组**。[完整映射](../../outputs/router/hotpotqa_bd_router_v1/runs/m6_test_group_identity_audit_v1/old85000_group_identities.npz)已保存。
- 既有 dev 三个交叉标记逐行重现：对内部 **225 查询 / 220 组**；对 test **21 查询 / 21 组**；同时命中 **8 查询 / 8 组**；两者皆不命中 **5209 查询 / 5206 组**。
- 已验证 5209 的 ID/hash 与原 dev 排除投影逐行一致；内部与已验证 dev 的组集合互斥，联合为 **88373 组**。
- 独立复核未导入原入口，用 `Counter` 计数和集合交叉重算全部表格、标记与分母，用两个方向的集合映射检查 bijection；**104 项检查全部通过**。源文件、协议、历史身份输入、原 results/completion 及数组的 SHA256 均核对一致。

正式投影记录：97852 个 query ID，跳过全部 97852 个 `text` 字符串；仅 train85000 + test7405 的 **92405 行**解码所需参考元字段，共 220790 个标题字符串；dev5447 元字段跳过并复用既有 hash。原文本无输出，新答案 payload、效应、拟合、网络及收费调用均为 0。

独立复核只读已保存 NPZ/JSON 并校验冻结源字节，**没有再次打开 queries 或 qrels**。对投影边界验证的是源绑定及记录计数的一致性；没有第二次从参考正文重建 hash，也不把 220790 个标题的记录数说成独立重放所得。

## 3. 尚未解决的资格问题

[前一库存快照](m6_confirmation_inventory_20260915.md)已发现全 test7405 参与四条件及 BGE/DPR/Contriever 检索比较；本次再次核对六份全量身份清单的记录。目标自身的检索暴露另记全 true，没有循环并入“与其他已消费 Router 来源交叉”的分母。

因此，7169 查询 / 7158 组也不能直接叫“新确认集”。还须核实历史检索结果是否影响当前检索器、encoder、重排、top-k 或评价条件选择；库存未找到答案效用缓存，不等于证明所有历史或人工使用不存在。canonical key 无交叉也不排除语义近重复。本次未计算任何新质量结果，未评价、选择或认证新确认样本。

绑定摘要：协议 `c92d5c51d92751dc480772224ee9dd89ae58e8c283edec98d4e1d63b15255037`；原结果 `a2056afc3174790958371bcd34f3e3ded3771bbc477afcab37cfb6083c54a91c`；独立复核 `964f4a0cc7b00e99c7611a0a9dc5d799edcd8eb5f35ef53dfa9d6a13ef1d9c15`。
