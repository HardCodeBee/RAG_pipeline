# M6 输出端直接适配：小试与独立复核

2026-09-15。**实现小试及独立复核均已完成并通过；正式五折实验已启动，session89863 / PID7868 正在运行。本文不报告正式 cal/test 效果，不表示适配已经产生路由收益。** 小试 session76359 与独立检查 session23967 均真实 exit0。

## 1. 问题与固定对照

既有 H 在最后层 CLS 训练后改读 mean6，未恢复超过原 M6 的所需增量。本轮直接从原 M6 的同折决策函数出发，检验让监督作用于第6层均值、更新其上游表示是否有效；这与事后换读取视图是不同干预。[冻结方案](m6_output_adaptation_plan_20260915.md)、[设计审查](m6_output_adaptation_design_review_20260915.md)

| 对象 | 起点 | 可训练参数 |
|---|---|---|
| A | 原始 BGE 前六层＋同折原 M6 头 | embeddings、block0–5、head，共22,565,761参数 |
| C | 与 A 相同 | 仅 head，共385参数 |
| B | 原同折 M6 | 不更新，复用既有 OOF 基线 |

仅使用已消费旧9600 / 9559 group，以及原五折 fit6144/cal1536/test1920。输入为固定 query tokens；第6层普通有效 token 做 FP32 mean/L2，原生 BF16 分数>0选 BM25。训练沿用已有两动作三重复平均答案 F1 差的标签与权重。两臂显式加入 `.001·||head.weight||²/2`，bias不罚、head AdamW decay=0，保持相同 minibatch、学习率日程和4 epoch预算；optimizer均从空状态开始。

正式选择仅限每臂 epoch0–4 的五个端点，在 own-cal 上按固定零阈值的答案效用选全局最大值，容差内选最早；所有十个选择先冻结，再统一 test。配方推进要求 A 同时超过 C 和 B，不能只超过漂移后的头控制。完整主比较和门槛见冻结方案。

## 2. 小试实际做了什么

仅使用 fold0 的 fit：两臂训练各8步，使用正式固定排列的前64条；完整 fit 的平均权重仍作为损失分母。恢复与表示检查固定使用 `fit[:64]`，与训练的64条不是同一集合。没有使用 cal/test 效果调整范围、学习率或轮数。

| 检查 | A | C |
|---|---:|---:|
| 更新步数 | 8 | 8 |
| 可训练 tensor 数 | 103 | 2 |
| 初始6144条 fit 特征与原 M6 分数 | 逐值一致 | 逐值一致 |
| 起始全部参数、实际训练排列 | 两臂相同 | 两臂相同 |
| 64条复查集 mean6 最大绝对变化 | 0.002912037 | 0 |
| 保存后恢复的特征与分数 | 逐值一致 | 逐值一致 |

A 的 embeddings 及六个 block 均有模块梯度和实际更新；C 的 encoder 逐参数保持不变，表示精确不变。两臂头均已更新，全部 loss/gradient 有限。这些事实验证了指定的干预已发生，不说明干预有益。[原小试记录](../../outputs/router/hotpotqa_bd_router_v1/runs/m6_output_adaptation_v1/pilot.json)

## 3. 独立检查的证据与边界

独立检查通过，终态为 `passed_independent_M6_output_adaptation_pilot_checks`。它绑定6个源码、19个输入及小试产物，使用原十二层模型的默认 forward，在第6层 hook 取得表示并独立实现 masked mean/L2 和训练计算图；没有调用正式训练入口。[检查实现](../../scripts/check_m6_output_adaptation_pilot.py)、[完整收据](../../outputs/router/hotpotqa_bd_router_v1/runs/m6_output_adaptation_v1/pilot_separate_checks.json)

- **真实优化重放：** 两臂共16步，首批梯度范数、完整 step trace、末端全部可训练 tensor，以及初始/末端特征和分数均逐值一致；重放误差为0。
- **独立数值计算：** NumPy FP64 池化最大差为 **7.084745e−8**。首步损失与头梯度按实际 BF16 前向/反向舍入规则核对；损失最大差为 **1.285063e−7**，头梯度落在对应舍入区间内。这不等于声称 BF16 训练等同理想 FP64 梯度。
- **原始头核查：** 五份原 M6 头在缓存 FP32 表示上的理想 FP64 目标梯度无穷范数最大为 **3.312497e−9**；其30,720条保存原生分数通过误差界检查。它核对起点，不作为小试改善门，也不是新适配模型的收敛证明。

独立检查共执行 **384个十二层 encoder query forward**，并重放上述16次优化更新。这是小试重复核验，新增正式拟合和实验条件均为0；不能将384条检查称为全正式训练重放。

## 4. 实际资源与当前状态

| 阶段 | 墙钟时间 | encoder query forward | optimizer步骤 |
|---|---:|---:|---:|
| 原始小试 | 26.26秒 | 12,672 | 16 |
| 独立小试检查 | 58.37秒 | 384 | 16次重放 |

原小试 GPU 最大 allocated 为 **476,107,776 bytes**，最大 reserved 为 **606,076,928 bytes**。12,672次包含两臂完整 fit 起点检查，以及小试训练、末端和恢复检查；这些本地墙钟时间不是线上单请求延迟。两阶段新增 cal/test 质量评价与 API 调用均为0。

小试记录中的 `formal_training_started=false` 是写入该记录时的快照；随后独立实现门通过，根代理才启动正式 session89863。本文写作只读取小试产物和冻结方案，没有读取正式训练中的 cal/test 质量。正式结果必须等全部预定训练、端点选择、test与独立复核完成后另行报告。

## 5. 冻结身份

| 文件 | SHA256 |
|---|---|
| [protocol.json](../../outputs/router/hotpotqa_bd_router_v1/runs/m6_output_adaptation_v1/protocol.json) | `11237536b48159ff9071d2ab140e385f8adba172b5243c449ba0006af5bb35d6` |
| pilot.json | `cb44d0de6650ae1fce47d9ef7df8af577847299989ac30fce1c83bb2cbc9d4b0` |
| pilot_separate_checks.json | `e6c842f2af3acd5ed4fb7e842f11eff3ceff1a88432bd7c8d25bde15a371d973` |
| 小试独立 checker 源码 | `907c0423402c3e4abaa46345e8f5aa9a7235c59a33ebcda34a99a6d1d5fab4cf` |

小试通过只允许按已冻结方案执行这一个有界比较；没有据此增加范围、种子、训练轮数或新的付费验证。
