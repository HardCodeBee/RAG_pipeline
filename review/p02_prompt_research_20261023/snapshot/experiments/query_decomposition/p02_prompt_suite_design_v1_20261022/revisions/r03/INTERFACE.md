# r03 输入与输出约定

这是独立设计快照，不是生产实现、模型运行结果或已批准的新主评估协议。

1. 分解器、描述式子问题、原始步骤顺序、阶段独立上下文、双路检索与原query的root Top10不变。无普通子步骤时允许直接goal；这只是有效路径覆盖，不更改旧诊断样本。
2. rewrite消费原始当前问题与此前Answer事实句。无历史不新增改写调用。只输出一条检索问题；原始当前问题仍是ordinary的任务。
3. ordinary输入原题、原始当前子问题、全部此前Answer、当前合并reader及实际检索问题。最后一个字段是新增且可能错误的上下文，必须作为独立变化披露。
4. ordinary原始Support与Answer全文保存；仅Answer进入后续history、改写及final普通QA区。Answer必须带上本地结果的实体、关系和必要条件。Support不会累积为新上下文或检索证据。
5. goal输入原题、普通Answer历史、goal合并reader和实际goal检索问题。输出Facts、Link、Candidate全文进入final；Facts和Link不是已验证证据，也不能证明模型内部推理忠实。
6. final输入原题、普通Answer历史、goal三字段和原query的root Top10。不会额外偷加goal原始reader。goal来源的新事实可以通过综合段落进入final；未出现在root不等于被root反驳。
7. 候选ordinary不再调用旧短句LLM formatter，避免把Support/Answer误当一句话处理。旧P02对照保留原行为。这是方法包差异，不能归因于提示文字或CoT单一因素。
8. 只有完成原因为stop的响应进入接受解析。缺失、重复、空字段、标签顺序错误、前导解释或代码围栏记为格式失败；保留原文，不调用额外模型修复，不从Support猜Answer，不复用旧答案补洞。解析不会核验事实、完整性或逻辑。
9. 输出按完整内容保存；50/140词都是软目标，不由客户端按词数裁剪。具体API上限未冻结，后续验证必须同时记录上限、finish_reason及对照条件。length响应不作为成功字段结果，也不得通过截取正确前半段改善分数。
10. few-shot只用人工虚构题材。示例元数据、预期行为、禁止推断、标准答案等审核字段不进入运行时真实题目。final的错误Candidate示例只是在final输入端展示纠正任务，不作为goal正确输出。

离线design_tools.py仅是接口草图。它不连接模型、不访问账务缓存，也不意味着生产runner已经接入。最终短答原样保留；解析不自动删除前缀、解释或补充限定，不利用参考答案挑选输出。
