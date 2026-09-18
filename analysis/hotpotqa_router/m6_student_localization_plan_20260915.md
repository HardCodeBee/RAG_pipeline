# M6 Student负结果：固定产物的监督、参数和动作分解

2026-09-15。旧9600已消费开发诊断；前一轮总体效果已知，本计划在计算以下新诊断量前固定范围。只读取已保存的Teacher目标、三Student头和原生预测，不拟合、选参、造新策略或计算新显著性检验。

## 问题与竞争解释

固定软BCE Student未确认迁移，不能直接解释为query缺少信息。需要区分：(1) Teacher额外目标在当前线性模型可用的方向上抵消；(2) 它确实改变模型与分数，但固定动作边界上的有益/有害选择相互抵消；(3) 一般Teacher软化主要带来收缩，probe的变化叠加在这个效应之上。三种现象可能同时出现，描述性分解不能证明因果或总体无信息。

## 精确数学边界

每个fit内令U=[X,1]，theta=[beta,bias]，w为原W在正权重行的归一，D对beta为1、bias为0。三Student的X、w、lambda=.001相同，区别仅为Q。

对任意两臂a,b，令dq=Q_b−Q_a，m=U^T(w*dq)，则对任意theta：

`J_b(theta)−J_a(theta)=−theta^T m`。

这是当前有限样本和固定线性类的恒等式：目标在Student损失中的所有影响由m表达。它不证明整个query没有信息。分开记录bias moment=sum(w*dq)、中心化特征moment=m_beta−mean_w(X)*m_bias，避免把总体偏移误称为特征关联。

令delta=theta_b−theta_a、s_a=U theta_a、s_b=U theta_b、p=sigmoid(s)，有

`Hbar*delta = U^T[w*(p_b−p_a)] + lambda*D*delta = m + grad_b−grad_a`。

Hbar为连线上积分Hessian。无需近似求积分或新拟合，就能计算其对delta的作用。进一步

`delta^T m ≈ sum(w*(s_b−s_a)*(p_b−p_a)) + lambda*||delta_beta||²`，

等号误差是已保存解的驻点残差。记录data_response与regularization_response及其占比；这只是响应几何，正则响应占比大不能单独证明应调整lambda。

再以Direct的FP64概率p_D、原生fit概率p_D_native及残余梯度g_D为参照，对每个soft Student定义p_T为其OOF Teacher概率，固定alpha=.5，有精确四向量分解：

`m_soft−Direct = −alpha*lambda*[beta_D,0] + alpha*g_D + alpha*U^T[w*(p_D_native−p_D)] + alpha*U^T[w*(p_T−p_D_native)]`。

分别命名为Direct自预测的正则收缩项、解的数值残差、原生数值差异、OOF与完整拟合预测差异；最后一项混合训练子集和估计差异，不是纯OOF因果作用。保存向量、范数和与实际m的cosine，不将范数相加或当解释比例。公共收缩项在Probe−Pre中抵消，因此它不能单独解释probe特有增量失败。

只描述Direct处把Student正则设为`(1−alpha)*lambda=.0005`时的软目标梯度，相对当前lambda的梯度范数如何改变；不拟合该模型或评价该策略。同训练集且Teacher概率恰为Direct自身FP64概率时，该补偿会恢复Direct驻点；真实OOF和probe情形不保证如此。

## 固定输出

五折全部报告三项有向比较Pre−Direct、Probe−Pre、Probe−Direct，不选择子折。

1. fit正权重行数、目标差的加权mean/RMS/中心化RMS、原moment与中心化moment范数；中心化moment除以`RMS_center(dq)*sqrt(Ew||X−mean_w X||²)`，只称归一化一阶关联，不能当解释方差或可迁移信息比例。
2. 三个Student的beta范数、bias、相对Direct的beta范数比与cosine；每项delta_beta范数、FP64 fit分数变化的加权RMS、dq与Direct FP64分数的加权协方差、以上损失等式在theta_a/theta_b处的误差、驻点响应误差和能量分解。保存m、delta及上述四项收缩分解全向量供独立复核。
3. 同一比较中心化moment在五个fit之间的全部两两cosine。fit高度重叠，不作独立重复或显著性解释；均值小或norm小不能单独判定无信号。
4. 由冻结原生test分数，记录每折及全9600的signed变化均值、RMS、Pearson相关；固定四种动作迁移D→D、D→B、B→D、B→B。逐格保存query数、相对a的改善/损害/容差平局数、真实质量变化总和及占全test/9600的贡献，完整加总重放前一轮三项效应。报告同动作行上的分数RMS，不调整阈值、选择top分歧或读取问题正文。

统计量为描述性，0新增bootstrap或候选门；历史迁移门不改变。若分母为0，标为null，不生成NaN。数值恒等误差≤1e−11，响应残差按保存解的两梯度逐值解释，并确认每端梯度≤1e−8。动作数量和质量加总必须与已绑定前轮结果一致。

## 实施、校验与决定

先用合成矩阵验证目标线性恒等式、正/零响应、中心化bias分解及四格动作贡献，再绑定计划、脚本和输入摘要后运行。另一路计算复核moment、梯度/能量和动作加总。没有重新GPU推理，所有新Student政策仍为前轮冻结政策。

输出不足以唯一分开有限Teacher误差、表示限制、代理损失与效用边界。只有观察到模型没有获得任何可见变化时才讨论消失发生在目标投影环节；若参数和动作都改变，后续问题必须转向变化的方向与收益，不能称“蒸馏没有实施”。一般收缩若出现，先推导其是否由Teacher正则传递造成，不能见norm变小就自动追加lambda网格。任何纠正实验须有独立数学或数据依据，另外冻结；若没有新的可区分问题，关闭当前线性迁移路线。
