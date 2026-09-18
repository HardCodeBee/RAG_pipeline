# M6软BCE蒸馏：一次正则补偿机制对照

2026-09-15。方案准备；运行前以独立protocol.json绑定本页与源码。不得据此视为已拟合或已有收益。只使用已消费旧9600，独立确认资格不因此恢复。

## 为什么在固定配方结束后还有一个新问题

前一轮45次拟合未确认迁移，固定alpha=.5/lambda=.001配方已结束。随后预固定的[诊断计划](m6_student_localization_plan_20260915.md)显示：Pre头的beta范数仅为Direct的.690—.725，实际目标moment与解析收缩项cosine为.877—.989。模型确实改变了参数和动作，不能称蒸馏没有作用。

新的数学依据是：当同一训练集上的正则化Teacher就是Direct，使用其概率混合硬标签，并在Student上继续施加相同lambda时，Direct处会多出`alpha*lambda*[beta_D,0]`梯度。唯一解析补偿值为`lambda_Student=(1−alpha)*lambda_Teacher=.0005`；此时理想自预测条件的Direct恢复为驻点。真实OOF、probe Teacher不满足该理想条件，没有收益保证。

已有诊断只在Direct处计算该补偿的梯度，尚未求解新头：Pre的梯度范数变为原来的.158—.482；Probe为.436—.777。公共收缩在Probe−Pre中抵消，因此“两个头都缩小”不能单独解释特权增量失败。需要匹配的低正则Direct控制和交互项，检验降低Student正则是否释放额外probe价值。这是由恒等式指定的单点机制检验，不是重开lambda/alpha搜索。

## 固定设计

- 原旧9600/9559group、原五折fit6144/cal1536/test1920不变；仅使用原384D M6。
- 直接复用上一轮各Student outer-fit内部严格生成并独立核验的Teacher OOF概率、Q和W。Teacher训练lambda=.001，alpha=.5均不变；无新Teacher训练、无新目标构造、无Student选参。
- 旧三臂Direct/Pre/Probe的lambda=.001和全部预测固定，作为O格；新增三臂DirectC/PreC/ProbeC都用lambda=.0005，作为C格。DirectC使用硬Y，PreC/ProbeC使用与其旧臂逐值相同的Q。
- 三新臂保留原效用权重、未惩罚截距、数值预算、梯度≤1e−8、原native CUDA BF16 batch8、logit>0选BM25。没有cal阈值或温度。
- 5折×3臂=15次正式拟合；另用上轮已固定的64行pilot Student训练集及目标，完成三次小试，只验数值，不看策略效果；再重放480条原M6原生分数。
- 全部15新头及fit/cal缓存先保存并绑定；之后统一保存9600行外层预测；预测冻结后才计算效果。拒绝解保存并停止，不扩大预算。
- 0 encoder forward、0检索、0答案生成、0外部API。新的独立来源评价另需资格与预算，不在本协议内。

## 六项主比较与决定

1. 净蒸馏增量的补偿交互：`(ProbeC−DirectC)−(Probe−Direct)`，检验降低lambda是否比对直接监督更有利于Probe Student。
2. ProbeC−PreC。
3. ProbeC−DirectC：控制一般降低正则的影响。
4. ProbeC−Direct：相对原M6的净增量。
5. ProbeC−Dense。
6. ProbeC−BM25。

共享20000次group bootstrap，seed=2026091512，逐次按抽中query数归一；六项均使用分位点[.05/12,1−.05/12]，即双侧99又1/6%条件区间，对应六项名义Bonferroni家族。旧效应及诊断已知，所有结果仍是已消费开发证据；不覆盖重训练或全部历史研究选择不确定性。

**配方推进门：第2—4项均满足点增量≥.002且下界>0。候选准备门额外要求第5—6项均点值≥.01且下界>0。** 交互项用于解释补偿是否比对直接监督更有利于Probe Student；第2项直接核查新的probe特有增量。交互不能独自代替净收益门。两门未过就结束本次解析补偿配方，不再围绕它追加正则、alpha或阈值。

报告8个固定政策的F1、BM25数量与有益/有害/平局贡献，五折全部主比较、原生fit/cal/test硬BCE、三个新头的系数范数与旧对应头的差异。PreC相对Direct的参数距离与旧Pre相对Direct距离作描述，检查补偿是否朝理论预期移动；更接近Direct并不等于产生迁移收益。六格各臂C−O的点值只作描述，不增加主检验或事后选出胜者。

独立checker复用已验证的另一套目标/梯度与BF16数值界公式，不导入新runner或训练模块；核对输入目标逐值沿用、18份解、15个新头、全部分数/动作/贡献、六项完整bootstrap及预定决定。没有独立GPU重跑。冻结前核验旧诊断已通过独立复核，之后才执行本对照。
