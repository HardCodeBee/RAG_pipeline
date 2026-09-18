# 固定 M6：严格 OOF 的软 BCE Student 迁移对照

2026-09-15。正式拟合及效果计算前冻结。研究问题：匹配 M6/BCE 下已确认的检索后 probe 增量，能否通过只使用 query 的 Student 改善答案质量？本轮使用已消费的旧9600开发池，不是独立确认。

## 1. 依据与竞争性假设

[匹配 probe 实验](m6_probe_readout_results_20260915.md)中，P−M6=+0.009581，99%条件区间[+0.004892,+0.014284]；P同时超过两份组置乱控制，通过原定迁移研究准备门。P需要先执行检索。本轮不把它当成可部署模型，也不直接复用其全局OOF预测作为Student训练目标。

[目标一致性推导](m6_distillation_rationale_20260915.md)说明：理想效用加权Teacher的软概率，在保留原效用权重的BCE下，对任意pre-only Student保持相同总体目标。有限Teacher可能改善监督估计，也可能引入偏差；这不是有限样本收益的证明。

| 假设 | 可区分的预测 |
|---|---|
| probe带来的监督估计可迁移 | Probe Student同时优于Pre Student及Direct |
| 只有一般软化/Teacher估计作用 | Pre优于Direct，而Probe没有明确额外增量 |
| 信息不能由当前M6线性类利用，或Teacher误差抵消收益 | probe Teacher较强，但Probe Student没有达到迁移门 |
| 泄漏或软损失实现错误造成差异 | 组隔离、训练几何、软目标重建或独立梯度复核失败；此时不解释效果 |

本轮无法单独区分最后一类科学解释中的表示限制与Teacher误差；失败只结束当前固定配方。

## 2. 数据、参数与输入边界

完全沿用原9600 query / 9559 group、五外层折，每折fit6144/cal1536/test1920。各折三内层组划分使用原`lp_ft_inner_v1`规则。Student的cal部分不参与训练、目标生成、选参或决策阈值。所有外层test各被评价一次。

X是冻结384D M6 FP32特征：第6层有效普通WordPiece均值、L2归一化。Teacher额外信息仅为旧缓存的50D检索后probe；只加载probe、query_ids、group_ids，不加载同文件gold块。probe schema不使用qrels或生成质量，缓存原准备流程还处理过其他gold块，不据此宣称整个准备流程gold-free。

所有Teacher、Student预固定lambda=.001；软标签混合alpha=.5；原生logit>0选择BM25，等于0选择Dense。没有温度、概率标准化、概率截断、额外校准、阈值或参数搜索。.001来自已消费的原M6五头及匹配probe十五头均选中该值的开发证据，不恢复评价独立性。

令d=F1(BM25)−F1(Dense)，Y=1[d>0]，W=abs(d)×1[abs(d)>1e−12]。F1是两动作各三次回答的normalized-token F1均值；本轮不生成新回答或计算语义Answer Correctness。

## 3. 严格嵌套的Teacher目标

每个Student outer-fit分别完成三份组OOF。每份Teacher仅用该outer-fit的另两份组训练；目标组、Student cal/test均不进入该Teacher的训练。两类Teacher使用同一训练/目标行，原W及硬Y：

- T_pre：原384D M6加性线性BCE。
- T_probe：不改变M6，追加50D probe。每个Teacher仅用自身训练全部行计算FP64均值、总体标准差ddof=0；std≤1e−12列归零；其余z-score除sqrt(50)，转换FP32后拼接。无行归一化或拼接后归一化。

每个Teacher保存训练/目标indices、组分配、头、求解诊断；probe Teacher还保存自身几何及实际变换缓存。目标行取Teacher原生CUDA BF16分数转FP64后sigmoid。T_pre使用原batch8单线性头；T_probe使用原batch8的384D带bias分支加50D无bias分支，各转FP32后相加，不再BF16舍入。

不能用最新P的全局OOF分数，因为其训练集合可能包含当前Student外层保留组。固定Student参数避免在Student内层选参中间接使用其验证标签；本轮无新的Student内层选参。

## 4. 三个pre-only Student与数值规则

| Student | 训练目标Q | 输入 |
|---|---|---|
| Direct | Y | 原384D M6 |
| Pre | .5Y+.5sigmoid(T_pre OOF logit) | 原384D M6 |
| Probe | .5Y+.5sigmoid(T_probe OOF logit) | 原384D M6 |

三者保持同一W归一、lambda=.001、未惩罚截距与零决策边界。软BCE稳定值使用`Q*logaddexp(0,-s)+(1-Q)*logaddexp(0,s)`；不能把Q代入只适用于二值Y的signed-softplus。先删除零权重行，W除最大值再除总和。初始化beta=0，bias=log(sum(WQ))−log(sum(W(1−Q)))。

Direct及硬Teacher调用原refined solver；新软模块只扩展目标。沿用L-BFGS-B的maxiter5000/maxfun50000/maxls50/maxcor20/ftol1e−15/gtol1e−10，以及原至多8步Newton、每步25次backtrack预算。优化成功且原坐标完整梯度≤1e−8才验收。原已接受硬解保持不变。失败保存并停止，不增加预算或把失败算作阴性。

Direct每折重新拟合，检查其FP32参数与原M6头完全一致，全部fit/cal原生分数逐值相等，最终全部test分数及动作也须完全相等。这是恢复相同基线的必要条件，不允许静默替换基线。

T_pre重新拟合所得的全部30720条own-fit OOF原生分数，还须与原M6各折已保存的lambda=.001内层分数逐值一致。仅把旧分数用于重放验收；实际Teacher目标仍由本轮严格内部训练产生。

## 5. 小试、执行顺序与规模

冻结前先完成合成软目标值/梯度/Hessian核对、硬标签退化、零权重处理、存在可迁移规律的保留样本正对照及无输入信息负对照。合成只检查实现，不用于选alpha/lambda。

冻结后真实pilot取fold0 fit中按原顺序最先192条全局单行group；前128训练两类Teacher，后64产生目标并各拟合三种Student，共5次pilot拟合。只检查组隔离、几何、有限分数、有效软目标与梯度，不计算策略效果。另外重放原M6的480条fit/cal/test原生分数。

正式每外层折6个Teacher+3个Student，共30个Teacher、15个Student、**45次拟合**。所有OOF目标与全部头先固定并绑定摘要；完成记录之后统一生成全部Student外层预测；预测冻结之后才计算效果。每个fit的目标完全覆盖一次，不能漏行、重复行或混合折来源。先拟合先保存的顺序记录不等于外部可信时间戳。

只使用已有本地特征/标签：0 encoder forward、0检索、0新生成回答、0外部API。按实际记录CPU/GPU耗时，不作端到端延迟声明。所有新产物保存在workspace的`m6_student_transfer_v1`独立目录，冻结源及输入绑定SHA256。

## 6. 主比较、区间和停止规则

五项主比较固定为Pre−Direct、Probe−Pre、Probe−Direct、Probe−Dense、Probe−BM25。共享20000次group bootstrap，seed=2026091509；每次抽9559个group有放回，按抽中query总数归一；每项双侧99%区间[.005,.995]，对应五项名义Bonferroni家族。

区间条件于当前固定OOF预测，不包含重训练、Teacher估计、全部历史选参的不确定性。样本量为可完整匹配两动作标签的全部旧9600；没有基于小试效应扩容，不声称对.002已有特定功效。

**当前固定配方的迁移门：Probe−Pre与Probe−Direct均为点值≥.002且区间下界>0。** Pre−Direct用于辨别一般软化作用。**独立候选准备门额外要求Probe相对Dense和BM25均为点值≥.01且区间下界>0。** 通过只允许随后准备独立确认；不能宣布核心目标完成或部署。

| 结果 | 决定 |
|---|---|
| 迁移门和候选准备门都过 | 固定该Student，另审查来源并冻结独立验证 |
| 仅迁移门过 | 确认当前开发条件的迁移增量，但没有足够绝对收益；结束该配方候选推进，依据实际误差提出新问题 |
| 迁移门未过 | 结束固定alpha=.5/lambda=.001/M6线性Student配方；不自动追加alpha、温度或参数网格 |

未过门不证明一切蒸馏无效；只有相应区间上界低于.002才在条件推断范围排除该幅度。报告全部三Student和两固定策略F1、BM25切换数、benefit/harm/tie数与质量贡献、全部五折主比较、Student fit/cal/test硬标签BCE，以及两Teacher在其各自OOF目标上的硬标签BCE。Teacher训练规模与最近P外层不同，不将OOF BCE当作重做P效果验证。

独立checker不导入新runner/math、不重新训练或重复GPU推理。它核对源和数据绑定、组隔离、Teacher几何与目标覆盖、soft sigmoid/混合及W、全部45解目标/梯度、原基线和Student输入维度、保存的原生分数路径数值范围、动作/贡献、全部bootstrap及决定。任何复核失败先解释测量，暂不宣布科学结论。
