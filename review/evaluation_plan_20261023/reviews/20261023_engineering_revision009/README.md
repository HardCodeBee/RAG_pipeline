# 修订009：六方法评估工程接线

2026-10-23（Australia/Sydney）。研究设计沿用修订007，修订008的解释与证据勘误保留。本轮只完成工程实现、离线验证和推送审核；正式评估继续暂停。

## 已接通的路径

- `runtime.py` 直接调用冻结的六个控制器，绑定相同formal500×3、方法参数、完整Qwen/BGE输入与独立输出目录。QD参数名映射到实际实现，DecomP两层均8步；不走旧run.main，不启动dev20、matched25或其他基线。检索模型与索引句柄按数据集复用。
- `runtime_transport.py` / `runtime_ledger.py` 在真实发送位置登记逻辑阶段与实际尝试，SDK/HTTP隐藏重试均关闭。新旧可核实等价阶段尝试去重后累计最多5次，跨重启不清零；unknown/in_flight先查回执，不自动重发。未核清的历史请求只阻断相关阶段。
- 费用保留已知实际支出、未知预留、逻辑用量及历史来源，旧账本只读；新运行器无195/200元隐含帽。主表费用提醒保留3倍/最近20个完整fresh均费1.5倍阈值，提醒写入持久账本并输出给运行监控，附历史参考及当前参数。QA/raw尚无同口径历史均费，明确记录缺参考，不偷用P02整流费用充当final参考。
- `runtime_reuse.py` 逐阶段只读验证C1完整payload、同题/方法/阶段、stop+usage、模型与检索输入；不按分数选缓存。真实旧记录仅做2个保存样例检查，不能据此将3866候选算作完成。历史回执核实结果可写入新账本覆盖记录，原数据库及负债原样保留；逐条证明“未提交且无费”才解除该unknown，尝试计数保留。本轮没有登记任何真实恢复覆盖或迁移。
- `qa_raw_runtime.py` 以持久回执重放冻结P02至原生final之前，核验完整ordinary/goal前缀、reader材料、去重和问题关联。两臂仅替换辅助材料，原提示字节不改。原生final失败不阻断已合格前缀；每臂独立保留成功、技术待定和算法失败。
- `runtime_inputs.py` 为固定300题形成材料/状态清单；`runtime_report.py` 独立读取原生参考答案，保留aliases和固定分母。主表未满500时不发布部分正式分数或scorable列；QA/raw保留缺失界。完整配对使用既定10000次、seed20261023 bootstrap。推理控制器不读取答案。

## 离线验证及边界

[96项测试、7项子测试全部通过](OFFLINE_TESTS.json)，验证进程禁用真实socket连接，实际尝试0次。覆盖六个真实控制器→新请求层→账本、unknown当场/重启/崩溃恢复、旧4次后仅剩1次、C1逻辑费用与新支出分离、费用提醒重新触发、QA/raw材料及逐臂恢复、原生评分与分母。

[136份受保护文件字节不变](VALIDATION.json)，含方法实现、示例、P02提示、样本及STOP标志。预算数值与实验统计口径不变。

[本机资源检查](RESOURCE_PREFLIGHT.json)验证冻结清单、模型身份及文件存在性，没有加载模型或索引；完整二进制校验仍由原冻结loader在实际运行时执行。实际部署GPU/远端环境尚未验证，不能称为真实硬件通过。

[1500条已有R02绑定通过只读核验](R02_READONLY_CHECK.json)，只检查题目、提示、模型身份与已接受输出一致；不作分解语义质量或完整RAG复用结论。

[正式QA/raw状态](QA_RAW_INPUT_STATUS.json)：300题全部保留为pending_upstream，0个已验证生产配对；这是未启动新主表的实际状态。[C1检查](C1_READONLY_CHECK.json)仅2个保存样例。[报告就绪检查](REPORT_READINESS.json)仍为not_ready，没有新分数。

## 入口和恢复

从仓库根目录执行默认入口只做离线检查：

```text
python -m experiments.query_decomposition.unified_rag_eval_v1_20261021.runtime
python -m experiments.query_decomposition.unified_rag_eval_v1_20261021.runtime --prepare-qa-raw
python -m experiments.query_decomposition.unified_rag_eval_v1_20261021.runtime --report
python analysis/engineering_wiring_revision009_20261023/verify_offline.py
```

复核数据需在原工作环境提供已冻结的语料、模型、样本和历史记录；它们不是新复制到审核包的数据集。新入口的`--execute --stage main`或`--execute --stage qa_raw`仍需与计划版本相符的明确执行批准，且所有用户STOP必须由后续明确恢复决定处理。当前命令会在创建客户端、加载模型之前拒绝执行。

新回执核对接口为`RuntimeLedger.reconcile`；历史未提交覆盖为`record_historical_reconciliation`，要求来源、请求、payload和核对证据对应。接口实现和合成测试不代表真实未知费用已经核清。完成主表不需要先核完全部历史候选，QA/raw只要求相应配对上游合格；可选计时继续单独决定。

## 发布完整性

远端运行目录的PLAN现在同步为009，并发布RUNTIME_BINDING.json及必需协议材料；审核目录保留同一计划副本。远端原来缺少基线methods目录，本次补入五个参照及QD依赖的hybrid辅助模块、示例和reranker身份文件；它们均为本地已冻结的原文件，不是新增方法或提示修改。旧运行器/原始报告/账本不覆盖。

研究范围仍为六行主表9000条件、三组已有分析、固定QA/raw300题/600final；没有真实API、GPU、检索、训练、judge或正式评估执行。
