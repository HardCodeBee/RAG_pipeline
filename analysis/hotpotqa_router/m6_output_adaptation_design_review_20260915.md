# M6 输出端直接适配：最小公平对照审查

2026-09-15。状态：**只读设计审查完成，以下适配尚未训练。** 核读本地 M6、六层推理、E11C/F/H、E13 及本日 H 重读的源码与记录，并仅在 CPU 读取五份原 M6 头的参数 schema；没有 encoder forward、新效果计算或 API 调用。本文不是冻结实验协议，也不预言结果。

## 1. 是否已经做过完全相同的实验

当前有界检索未找到“从同折原 M6 头初始化，直接以 mean6 为训练出口，比较更新 encoder 与仅更新头”的已执行实验。这个范围结论不代表所有外部历史绝对不存在。

- 全层 F、LP→FT 的 H 都已经训练过，但输出为最后层 CLS；H 只继承原 C12 线性头。其实际调用见 [run_lp_ft_v2.py](../../../work/router_research/run_lp_ft_v2.py) 206–218 行；范围与初始化替换见 [encoder_scope_training.py](../../../work/router_research/encoder_scope_training.py) 54–68 行、[lp_ft_training.py](../../../work/router_research/lp_ft_training.py) 12–17、95–107 行。
- C6/M6/C12/M12 是原始冻结 BGE 的读取视图比较，没有 encoder 训练；见 [run_layer_pooling.py](../../../work/router_research/run_layer_pooling.py) 156–173、235–275 行。
- 本日 H6/H12 只重新提取既有 H 的视图并拟合新头，H6−原 M6 未过门；它没有让训练目标直接作用于 mean6。[完整结果](m6_trained_readout_results_20260915.md) §2、§7。
- cal 按答案效用选 epoch 也已经做过。E13 比较 cal-BCE 与 cal-效用选择，U−Dense 为 −.001075，97.5% 条件区间 [−.003556,+.001391]，未确认收益；见 [E13 报告](../../../work/router_research/e13_report.md) 1–5、15–27 行。因此本轮不能把 cal 选择本身作为新方法或已经有效的补救。

## 2. 可直接复用的起点与六层路径

五份 `layer_pooling_v1/fold{0..4}_M6_head.pt` 均只有 `weight` 与 `bias`：形状分别为 (1,384)/(1,)，CPU FP32，无 optimizer state；对应五份 `fold{f}_M6.json` 的 selected_lambda 均为 .001。头由同折 own-fit 的线性求解映射保存，出处为 [run_layer_pooling.py](../../../work/router_research/run_layer_pooling.py) 241–265 行及 [lp_ft_training.py](../../../work/router_research/lp_ft_training.py) 20–33 行。

已有六层实现只保留 block0–5，并移除无用 pooler；mean6 是普通有效 WordPiece 的 FP32 均值再 L2 归一化，排除 padding/special token，空行回退 CLS。见 [m6_early_exit.py](../../../work/router_research/m6_early_exit.py) 18–34、45–57 行。旧工程检查已验证全部 9600 特征、全池候选头分数与动作逐值一致，见 [check_m6_early_exit.py](../../../work/router_research/check_m6_early_exit.py) 108–133 行。

**不能直接沿用该类的默认头。** 其第13行指向 `m6_candidate_v1/M6_head.pt`，这是全旧池候选头；新 outer-fold 实验必须显式载入对应 `fold{f}_M6_head.pt`。六层特征的既有等价证据可以复用，同折初始头的 fit 缓存仍需重放。训练 forward 还必须去掉旧推理函数的 `inference_mode`，保留 eval 模式关闭 dropout，并启用所声明参数的梯度。

选择 embeddings＋block0–5 有结构依据：它们是 mean6 的全部上游参数。原全模型“末两层”block10–11位于出口之后，不会改变 mean6；前缀末两层 block4–5 则是另一个范围假设，不应本轮追加选择。保留完整十二层但仍只在 mean6 计算损失，不会给 block6–11 带来该损失的数据梯度，增加这些参数不能解释为更充分的出口适配。

## 3. 建议冻结的三臂与训练目标

| 臂 | 起点 | 更新范围 | 作用 |
|---|---|---|---|
| A | 原始 BGE 前六层＋同折原 M6 头 | embeddings、block0–5、head | 检验出口直接适配的完整配方 |
| C | 与 A 完全相同 | 仅 head | 控制 minibatch 优化、头漂移与端点选择 |
| B | 同折原 M6 | 不更新，复用原 OOF | 要求新配方也超过原始有效起点 |

旧 9600 / 9559 group、原五折 fit6144/cal1536/test1920 不变。两训练臂同 Y=1[d>0]、W=|d|·1[|d|>1e−12]，d 为已有三重复平均答案 F1 的 BM25−Dense 差；只对 own-fit 梯度训练。固定 batch8、max128、FP32 master 参数、BF16 原生头、eval 无 dropout；两臂头参数相同、optimizer state 均重置，采用独立于模型执行顺序的相同 minibatch 排列。

原 M6 目标是 `sum(W·BCE)/sum(W) + .001·||beta||²/2`，bias 不罚；W 在 FP64 中按 max/sum 归一，见 [weighted_linear_probe.py](../../../work/router_research/weighted_linear_probe.py) 41–62 行。推荐两臂每步使用：

`mean_batch(W·BCE) / fixed_mean_ownfit(W) + .001·||beta||²/2`。

固定 fit 均值包含零权重行；正则在数据项归一后加入，使用 FP32 master beta，不再除以 fit/batch 权重。head AdamW decay=0，避免重复惩罚，bias 不罚。encoder 单独沿用 lr2e−5、AdamW decay=.01，排除 bias/LayerNorm weight；head lr=.001，encoder/head 分别 clip norm1。AdamW 的解耦衰减不等于凸目标中的 λ。

**与旧 H 的区别要明确记录。** H 在 LP 后仅复制参数，实际微调损失没有显式 λ 项，head.weight 受 AdamW decay=.01；见 [run_e11c.py](../../../work/router_research/run_e11c.py) 135–174 行。旧 H 全 tie batch 的 loss/梯度为零断言，不能复制到本轮：显式头正则仍会产生梯度，且两臂都应继续 optimizer/scheduler。

原头是 FP64 凸目标的数值解，但训练使用 BF16 原生路径及随机 minibatch，不能要求头控制在每一步都不变，也不能对新的非凸 AdamW 训练沿用“完整梯度≤1e−8”的凸求解停止门。应验证初始函数、梯度/更新范围和有限性，保留实际优化预算。

## 4. 固定预算与 cal 选择

推荐本轮仅执行两臂各 4 个完整 epoch，即每臂每折 3072 次更新，采用相同 warmup-linear 日程；不因小试拟合速度改变 LR、范围或 epoch 数。这是事先限定的短适配实验，4 epoch 不是经数据证明的最优预算。

旧 H 是首次达到 own-fit BCE≤.1×prior 且加权错误率≤.01 即停止，否则最多32 epoch，完全不看 cal；见 [run_e11c.py](../../../work/router_research/run_e11c.py) 45–54、190–201 行。该规则检查拟合能力，不是泛化最优停止；用于两臂也不保证相同步数。本轮不应追求两个模型都达到旧近插值门。

本轮端点只限 epoch0、1、2、3、4。两臂各自在全部候选生成后，以 cal 上 `mean(d·1[native_logit>0])` 最大选择；将与**全局最大值**相差≤1e−12的候选列为并列，再选最早 epoch。不要按顺序作链式近似比较。epoch0 保留原 M6，因此选中的 cal 点值在容差意义下不会更差，但这不是 test 保底。cal 的监督不进入梯度，也不选阈值、范围或超参数；观察不能改变 RNG、模型或训练模式。E13 对此已有观察器身份检查，见 [run_e13.py](../../../work/router_research/run_e13.py) 46–67、111–119 行。

同样五个候选和同样 cal 标签预算使 A/C 比较可解释，但不消除 cal 的选择乐观性。A−C 检验的是两臂各自按同一规则选点后的完整配方差异；不能把它说成固定 epoch 下纯 encoder 参数的因果效应。所有十个所选端点与其选择记录先冻结，再统一运行 test；不生成中间 epoch 的 test 曲线。

诊断 BCE 应明确分母：推荐每个分区按自身 `sum(W)` 归一，与本日重读表一致。旧 H 的 `loss_diagnostics` 则把 fit_mean_weight 也用于保留分区，见 [run_e10.py](../../../work/router_research/run_e10.py) 94–99 行；不能不换口径就把历史 BCE 表直接相减。

## 5. 最小判定与解释边界

四个主比较固定为 A−C、A−B、A−Dense、A−BM25；共用 20000 次 group bootstrap，四项各 98.75% 条件区间。配方门要求 A−C 与 A−B **均点值≥.002且下界>0**；候选准备另要求 A 对两固定动作 **均点值≥.01且下界>0**。A 只超过漂移后的 C 不足以推进。

小试训练只用 fold0 正式固定排列的前64条、8步；恢复与表示变化复查使用 fit 索引中的前64条。检查初始表示与同折头重放、参数/梯度范围、更新与恢复，不看新 cal/test 效果，也不凭小试拟合表现选择参数。训练与检查都是新增工程工作，应分别记录实际更新、query forward 与资源。

若通过，只支持已消费旧池上这一个匹配配方，仍需确认来源资格；若未通过，结束的是固定范围、4 epoch、五端点选择的方案。它不能证明更长或其他 M6 训练必然无效，也不能仅凭 fit 改善、cal 被选中或较低 BCE 宣称路由收益。当前不追加 F、层位、范围、正则、种子或重复目标的搜索。
