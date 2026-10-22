# M4完整RAG固定8题：锁后审核与信息路径

在这8道已曝光的2Wiki开发题上，M4完整RAG的F1/EM高于同期M0和旧同题参考；收益全部集中在Wiki721的一处完整日期恢复，其他7题最终文本逐字相同。不能将该差值外推到正式500题或总体表现。

| 同题8题口径 | F1 | EM | F1=1或EM=1 |
|---|---:|---:|---:|
| 旧公开source参考（r000/r100，非同期） | 89.4643% | 62.50% | 5/8 |
| 新鲜M0，原merge | 89.4643% | 62.50% | 5/8 |
| 新鲜M4，去同source排版重复并回填 | 95.7143% | 75.00% | 6/8 |

M4−M0为F1 **+6.25个百分点**、EM **+12.5个百分点**，1改善、7持平、0下降。16条流程全部成功，无技术缺失或算法终止。唯一不同配对的2条最终回答已先做方法遮蔽评审并锁定；其余7对相同文本没有重复贴标签，也不由此推定事实正确。

Wiki721原题直接问Cardinal-Infante Afonso of Portugal母亲的死亡日期。本轮可核的数据路径为：

1. 两臂ordinary第1节点完全相同，均正确绑定母亲Maria of Aragon。goal原题、原/改写检索包、前答history、rewrite完整prompt及request_id均相同。
2. 首次分叉只在goal第2节点reader选文。正确的“Maria of Aragon, Queen of Portugal”段已在两臂共同的改写路线Top10第6名；M0交错截取reader未纳入，M4去除同source有序词标点相同的排版重复后，将该段纳入reader第9名。原段给出Queen身份、Manuel I配偶身份和死亡日期7 March 1517。
3. M0 goal回答“Maria of Aragon…died in 1517”；M4 goal回答“Maria of Aragon, Queen of Portugal, died on 7 March 1517”。两臂均未触发formatter。
4. 两个实际final输入的root检索包保持完全相同，正确Queen日期文章没有直接进入final root。完整日期通过M4自己的goal结论进入实际可见helper。两臂完整final context和最终请求prompt只在这一条goal回答字符串上不同，所有其他字节相同。
5. M0 final为`1517`，原生F1=0.5、EM=0；M4 final为`7 March 1517`，F1=1、EM=1。锁定语义评审确认二者均为真实且回应原题的回答，区别是所问日期的完整程度。这里没有把正确年份判为事实错，没有年龄比较，也没有把隐藏goal reader当作final直接证据。

这条链验证了本次隔离merge变更下的“已检到→reader保留→goal传递→helper可见→final输出”信息路径；单个已知开发案例和单轮生成不能估计总体因果效应或证明普遍安全。

全部8题共有每臂17个ordinary和8个goal节点。实际M4改变7个ordinary及6个goal的reader成员；普通回答3处、goal回答2处变化，final prompt3处变化，最终文本只有Wiki721变化。所有两臂原/改写检索包和接受的改写query都相同：runner确实重算各臂后继，但本轮未观察到不同检索query，不能把fake中验证的动态query变化当成本轮真实结果。

旧source→新M0对照另行记录：8题最终文本、逐题F1/EM均相同；但有4个final prompt、1个改写、14个QA prompt、4个节点回答变化，且12个original-route、7个rewrite-route、11个reader的内容/顺序不完全相同。旧快照非同期且含r100；这些漂移不算M4收益，也不是全部字节等价或纯随机方差估计。本轮主要比较仍为共享资源且检索包相同的同期M0/M4。

已核128项冻结pin、3个暂停hold、完整16条flow离线回放、真实query/resource与39个检索包接线、全部模型/三个非负整数usage及求和、每阶段cap和final原生length例外。本轮无formatter或final length触发，100次逻辑stage使用对应70份不同的新成功回执，不作为额外独立样本。最多1次实际技术尝试，无退休unknown、无孤立或未知预留，首次标签与所有冻结源/原报告保持。

按usage及冻结单价估算的新费用为¥0.0949644，预留0；全流程逻辑费用M0 ¥0.0648192、M4 ¥0.0649792。逻辑费含本批复用，不与实际新费用相加；实际付款按第一次提交归属，不能当两方法独立冷启动成本。16个完整条件不能构成20-query费用窗口。

远端闭环文件已核：本地`REMOTE_GPU_FINISHED.json`为stopped、39次检索、A6000且无LLM请求；root另核job159027为COMPLETED、用时11m47、观察器exit0/队列空。这是运行闭环时间，不是单query处理时间。本审计没有发起SSH、API或GPU操作。

该8题全部已曝光，并按先前设计包含已知Wiki721和成功/非满分分层。已显示这个固定开发小队列的完整RAG正向结果；正式评估仍暂停，不自动修改主方法、晋级或授权更大任务。

证据：[严格审计](ROOT_NATIVE_RECEIPT_AUDIT.json)、[完整配对/路径及闭环补充](ROOT_PAIRED_INTERPRETATION.json)、[全8对最终差异范围锁](root_blind_review/ROOT_FINAL_DIFFERENCE_SCOPE_LOCK.json)、[首次两条语义标签](root_blind_review/batch001/SEMANTIC_REVIEWS.blinded.jsonl)。
