# L0/L1：固定90题、原生goal128＋原formatter＋原final包装

本轮仅离线准备数据、纯渲染合同与独立执行器；没有RUN_AUTHORIZATION，没有启动模型。`prepare`、`dry-run`和未启动状态的`report`均不提交请求，`run`必须先通过独立授权检查。后续是否执行须由root在T语义与端点评审锁定后另行决定。正式评估及全部STOP保持不变。

固定W90相同原题（每集30）、同实际历史R02/v12分解、全部ordinaryQA、goal改写、原双路检索及合并reader和final原query检索root。W90属于反复暴露的开发集，其真实分解出自冻结R02/v12旧生成批，不等同formal1500更新后的完整解码合同；没有人工corrected60替换。仅重生成goal及其必要formatter、final；不重跑检索或上游，不声称新鲜完整RAG耗时、独立确认或正式测试。

L0为source原goal完整payload的新鲜本轮控制，goal上限128。L1只在原主指令末、唯一`\n\nOriginal question:\n`之前插入`\n`加exact ROLE单句。原始材料、历史QA、goal末尾既有GOAL_RULE、请求参数和全部其他字节保留。ROLE与T1逐字同句，但本实验不使用T的512上限、综合包装或历史输出。

> Before answering, check that each named entity keeps its stated role in every relation and that the final conclusion preserves the original subject, object, relationship direction, generation and requested attribute, including the distinction between a parent and a grandparent.

goal响应先`strip()`，少于4个空白分隔词时触发原128-token formatter。原formatter的完整指令、原始当前goal问题、候选回答和原reader passages原样保留；两臂同条件、同模板、同上限。不是固定触发次数：即使历史90未触发，新响应也可能触发。格式化完成后才进入原`core.render_final`，使用原ordinaryQA、原goal问题及新goal回答和原root；保留原helper、示例、readout与64-token final。不得改为新综合候选展示。

合法模型身份、stop/完整usage的goal或formatter成功才向下游传递。goal/formatter/final的length、content_filter、refusal或empty，在完整匹配模型和三项非负整数usage（排除bool且total等于输入加输出）下是算法终止，整条件F1/EM记0且不重抽；goal或formatter终止时不发送下游阶段。缺usage、模型身份不符、技术unknown/最多5次耗尽归held，不能强记0。这里按本次明确的完整输出判定处理final length，不能把截断答案当stop成功。

执行器为本目录`run.py`，使用新独立checkpoint，L0/L1共同新鲜运行，StrictAsk seed=None，不导入历史goal/formatter/final响应。仅允许本实验相同完整payload的已验证成功回执复用；重复final等只按完整消息命中，不能按答案或分数挑选。技术attempt最多5次（含initial和restart），SDK/HTTP retries0，统一provider gate、NoCapBudget、旧unknown预留和已退休回执永久保留。对ledger已returned_usage或任何响应迹象的unknown/failed均hold，避免ledger先提交而cache未提交的崩溃窗口；不得改老shared runtime。

按固定seed20261022散列轮换每题L0/L1先后，题序沿W90，计划180条件。固定90原样保留6条没有ordinary子问的goal-only路线，另按执行前已知的分解结构分列84条有ordinary子问的敏感性结果，不依据得分选题或替换主90分母。每条件最多goal、formatter、final三个逻辑阶段，共最多540个阶段payload（技术尝试前）；历史formatter0次不能当新分支调用上界。所有条件保留，不依据T或L结果选题、挑最佳或重试算法失败。

输入不含gold、本题历史goal回答或最终预测；历史goal/formatter/final回执独立保存在local审查文件，严禁作为fresh seed或消息使用。普通QA属于合法冻结上游输入，不是reference。参考答案另文件保存，只在本地report join，各数据集原生scorer初始化一次。主要比较同期L1/L0最终官方F1/EM，历史P02单列参考；goal局部语义正确率不能替代最终端点。完整180条件与held/算法终止分别记录；有技术held时主表不报告虚假完整F1，仅给固定分母界限与已解析信息。

tokens与费用按每query冻结prefix usage加新goal/实际formatter/final归属完整流程逻辑开销；新实付、同payload共享费用归属和unknown预留另列，不重复把历史支出算成新增费用。不报告LLM调用数指标，不设固定预算上限；沿用相对历史阶段单条3倍及最近20条均值1.5倍提醒。新触发formatter无本题历史formatter开销时单列，不伪造配对历史0基线。

源public proof沿用W90已核定公开source文件hash，不宣称本轮重新联网验证。离线审查逐条核原goal及实际formatter/final完整body、stop、returned model与三项usage；历史receipt缺requested_model时由精确payload绑定。用原历史goal回答仅在本地复现final renderer，不能混入新控制。短回答formatter用合成分支对照原core真实分支，合成结果不进入任何质量统计。
