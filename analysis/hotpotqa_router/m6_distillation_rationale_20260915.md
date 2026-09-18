# M6 probe 增量之后：与答案效用目标一致的蒸馏问题

2026-09-15。**本文保留实验前的理论推导与设计依据。后续Student已冻结、完成45次正式拟合并通过独立复核，未确认迁移，见[实际实验报告](m6_student_transfer_results_20260915.md)。** 作为本推导依据的匹配probe实验中，P−M6为+0.009581，99%条件区间[+0.004892,+0.014284]，同时超过两份置乱控制，符合预定的迁移研究准备门。结果仍是旧池上的检索后诊断。

## 1. 新问题与E04的区别

问题：将同一M6/BCE条件下已经证实有增量的probe Teacher，用严格折外软目标教给仅使用M6的Student，能否改善保留数据上的动作收益？

E04已经实际做过严格OOF蒸馏，旧62D Ridge Student及固定混合回归配方未确认迁移。本轮依据新增的匹配M6证据，拟保持M6线性Student和原答案效用加权目标，改变监督估计方式。Teacher更强不保证Student会改善。

拟设三臂：Direct使用真实动作标签；Pre使用pre-only Teacher软目标；Probe使用pre＋probe Teacher软目标。Pre−Direct衡量一般软化/Teacher估计影响，Probe−Pre衡量增加probe后的增量，Probe−Direct衡量净收益。后两项都需要报告，避免仅因Pre变差就称迁移成功。

## 2. Teacher概率不能当作F1差

令答案效用差为d，Y=1[d>0]，W=abs(d)×1[abs(d)>1e−12]。所有臂使用同一实际权重及近零容差。X为M6，Z=(X,probe)。在正条件权重质量处，理想加权BCE Teacher的概率为

$$
p_T(Z)=\frac{\mathbb{E}[WY\mid Z]}{\mathbb{E}[W\mid Z]}.
$$

忽略容差记号，令a为条件正效用质量，b为条件负效用质量，则p_T=a/(a+b)，而条件期望gap为a−b。概率缺少总效用质量a+b；其logit同样不是答案F1差，不能将两者混合成同单位回归标签。

## 3. 保留效用权重的软BCE为何在总体上目标一致

取软目标Q=(1−α)Y＋αp_T(Z)，仍以原W加权。由于X包含在Z中，条件期望给出

$$
\mathbb{E}[W p_T(Z)\mid X]
=\mathbb{E}\!\left[\mathbb{E}[W\mid Z]p_T(Z)\mid X\right]
=\mathbb{E}[WY\mid X].
$$

因此E[WQ|X]=E[WY|X]。软BCE对其目标是线性的：

$$
\ell(Q,s)=\log(1+e^s)-Qs.
$$

对任意仅依赖X的Student分数函数s(X)，均有

$$
\mathbb{E}[W\ell(Q,s(X))]
=\mathbb{E}[W\ell(Y,s(X))].
$$

保留相同总体权重归一常数及正则后，两者总体目标一致；该等式不要求Student具有无限表达能力。零条件权重部分对风险无贡献。

这是对理想Teacher的目标一致性推导，**不是有限样本改善或方差必然下降的证明**。实际Teacher具有估计误差、正则偏差及原生数值误差。若其概率误差为ε(Z)，软目标相对直接标签在总体未归一风险中增加−αE[Wε(Z)s(X)]，可以有利也可以有害。保留相同Student函数类也不会凭蒸馏增加检索前信息。

## 4. 省略权重可能直接改变最优动作

一个纯合成例子：某类query的两种Z状态概率为0.1/0.9，对应W为1/0.01，正动作标签概率为0.1/0.9。无权重平均Teacher概率得到0.82，会偏向BM25；正确效用加权概率约0.166055，选择Dense。真实期望BM25−Dense gap为−0.0728。

因此本设计应保持原效用权重，而不是把Teacher概率当普通胜率标签平均。这个例子只说明数学目标，不是项目实测质量。

另用60状态的人工联合分布，在α=0/0.5/1、各100个Student分数函数下核对上述风险等式，最大差4.44e−16。[合成数值核验](m6_distillation_objective_checks_20260915.json)没有拟合真实模型；代数推导与数值检查相互补充。

## 5. 有界设计草案及防泄漏要求

拟预固定α=.5、所有Teacher/Student lambda=.001，不进行新的Student内层选参。本轮P/S0/S1的15个头和原M6的5个头均曾独立选中.001；这可作为已消费开发依据，但不能恢复旧池评价独立性。Direct应核验与原M6硬标签目标和原生路径一致，不能默默更换基线。

- 每个Student outer-fit内部按原三组分别生成Teacher目标；Teacher训练不含该目标所属组，也不含Student cal/test。
- Probe Teacher的均值/标准差仅由它自己的训练组计算，目标行只应用这些参数。
- 两类Teacher共享相同内层划分、权重、lambda及数值预算；唯一信息差异为是否追加probe。
- Teacher原生logit转FP64后取sigmoid，得到[0,1]软目标；不直接使用logit、不额外添加温度、概率标准化或cal校准。
- Student输入始终为原384D M6；使用相同权重、正则和零动作边界。
- 固定lambda避免了Student选参过程中，预先生成的Teacher目标间接包含Student内层验证标签的问题。若未来重新引入Student内层选参，必须把Teacher目标生成完整嵌入每份Student训练部分。
- 不能直接复用刚完成P的全局OOF分数作为当前Student训练目标；这些Teacher的训练集合可能包含当前Student外层保留组。

按此草案，5个外层折×(3份Teacher OOF×2类Teacher＋3个Student)=45次拟合。下一轮仍须先明确主比较、区间、最小增量及停止标准，实现软BCE与独立复核，验证硬标签退化和真实小试，才能冻结后运行。

撰写上述推导时只完成理论和设计准备。随后草案已转为[冻结计划](m6_student_transfer_plan_20260915.md)并执行；实际结果限制当前固定配方，未产生独立来源确认或部署结论。
