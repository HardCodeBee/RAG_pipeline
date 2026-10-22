# 固定实际分解的8题完整动态下游验证（待审核）

本目录准备独立的同期开发对照，不恢复正式主评。准备和冻结均不代表执行许可；必须由root审核整套本地与远端代码，再记录绑定协议hash的独立`RUN_AUTHORIZATION.json`与决策记录。所有旧source、标签、成绩、代码及主评STOP保留。本目录名称含`draft`，冻结执行后也不通过更名掩盖其暴露开发用途。

## 问题与范围

前面的C阶段screen只改变一个保存reader上的实际子问接口，出现局部事实修复，未证明完整RAG最终F1提升。此处检验真实部署策略：所有普通子问都使用本轮运行产生、并通过原acceptance规则的实际`rewritten_query`作为QA的Current subquestion。它会改变后续历史、改写、检索、goal和final。参考目标始终是原始当前子问，错误改写造成的对象偏移仍应报告。

保留先前选定的全部8个原题（HotpotQA5、2Wiki3），包括6个原普通子回答错误和2个合理弃答边界，不只选择C获胜题或原锚节点。固定正式R02/v12实际接受的`supporting_questions`及原题全文，不用人工修订60、不改训练。来源与原始remote_raw数组逐字核对，原公开review归档hash证明单列。该集合已多次查看、按既有诊断选取，没有独立确认或正式500题泛化含义。

`INPUTS8.jsonl`只含id/qid/dataset/question/supporting_questions。参考答案仅保存于单独本地`references8.local.jsonl`，只被report读取；它不进入Pipeline、模板、worker、模型或检索请求。旧质量结果、旧模型回答和旧retrieval packet均不作为运行seed。`HISTORICAL_COSTS8.local.jsonl`只供账务比较和非同期历史参考，不用于选择输出。

## 两臂与唯一干预

- D0：同期fresh原P02控制，普通QA Current保持原分解子问。
- D1：普通QA仅将Actual task段的Current subquestion值换成该臂当前历史生成、经原规则接受的`rewritten_query`。原prompt其他字符、fewshot、Original question、可见history、reader及128上限不改。

每个普通节点统一应用D1。第1个普通节点没有历史，原流程不改写，因此自然保持原问；拒绝的改写亦回退原问。原acceptance为strip后非空、以`?`结尾且无换行。原完整分解文本和`history_all[].request`保持原问，记录`actual_question_interface`区分原评判目标和真正送到QA的current；`child_prompt`记录实际发送文本，`raw_P02_child_prompt`保存原模板渲染。node.rewritten_query与D1实际current一致。

使用原`core.Pipeline`执行，不复制或修改core算法。rewrite256，ordinary/goal128，原双路Top10交错merge，原goal事实句和组合规则，少于4词时原formatter128、formatter的Question仍为raw当前问题。goal不采用C替换，也不加入ROLE或综合段落。final使用原helper、原query root检索和原包装，64上限。D1前答变化后所有后继rewrite均重新从新历史生成；实际改变的查询必须搜索冻结全语料，不能复用旧reader并称完整动态流程。

按固定case顺序和seed20261022轮换臂的开始顺序，各4题D0先/4题D1先，不根据旧分数、标签或新输出调整。首个分歧前完整payload相同，可精确复用本次新checkpoint中的成功回执；不将历史回执称作fresh同期控制。底层StrictAsk `seed=None`，新增实际费用按唯一request计一次，各臂逻辑完整流程usage则包含共享请求的完整归属。

## 本地API与远端检索

API请求、共享provider gate、账本与checkpoint全部留在本地。端点、模型、温度、thinking、请求配置继承原P02 config。每次真实SDK请求持跨进程gate并至少间隔3秒；SDK与HTTP内置重试为0。远端只接收dataset/qid/query/k和资源identity，不接收LLM消息、gold或API凭据。

优先使用远端GPU编码查询，原冻结corpus向量仍通过CPU精确FAISS搜索。BGE model `BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`，384维，归一化内积，max512，单query batch1，4线程，50000行corpus块；原Top11过滤原qid后取10，原doc_id、正文、source、score、rank和block保留。跨GPU仍可能有边界浮差，因此保存硬件及资源记录，不声称位等价。

`RESOURCE_CONTRACTS.json`的每ds值严格为`{identity,contract}`。路径无关contract包含完整portable manifest SHA（绑定所有源文件与embedding part SHA）、chunks/offsets SHA、doccount、模型revision和space、上述算术设置；identity为canonical JSON的SHA256。root已有只读login preflight仅证明资源manifest及登录可读，未证明此刻分配GPU。新的worker必须在GPU_READY前核对上传代码hash、真实manifest及资源identity、分配和CUDA；无法核对则held，不以空packet代替。

`remote_worker.py`和`remote_bridge.py`由root负责独立新job/spool。`open_bridge(resources,output,authorization)`为contextmanager，`.for_dataset(ds).retrieve(qid,query,10)`返回原route packet。只在run动态import桥。桥在任何连接/部署/submit前需要独占部署锁，避免第二CLI先重复申请GPU再被checkpoint锁拒绝。密码仅本地内存或getpass，不写进文件/日志；worker从未使用API凭据。主评与其他旧job不接管。

worker停止或连接/任务状态不明时不静默申请另一个GPU任务。即使Qwen技术请求可以在既定5次界限内恢复，缺少新查询检索仍须held，由root核对job/回执后在本小实验范围内另行授权技术恢复。若本地16条件均已质量终止，再次run只读导出report，不打开桥、不重新部署或分配GPU。

本地retrieval缓存仅对本轮新packet按完整resource identity+dataset+qid+原样query+k精确复用。不同history引发不同query就是真实新搜索。原路和改写路分别保存，不以合并reader冒充retrieval缓存。默认不迁移任何历史packet，减少跨路径资源迁移歧义。

## 成功、失败与续跑

每条返回在进入Pipeline前要求status success、requested/returned model与冻结模型一致，prompt/completion/total三项tokens为非负int（不含bool）且总数相加一致。rewrite、普通QA、goal和formatter须stop、非空且无refusal。合法length/empty/refusal/filter为算法终止，记录并按该题0计，不重抽。缺usage、模型不匹配或格式不明是技术held，不能伪装算法0。

原生final允许完整归属、非空、无refusal的length返回，记`success_truncated`并照常官方计分。共享StrictAsk本身禁止allow_final_truncation；新NativeAsk始终以false调用底层，在捕获已保存合法final length回执后从同条回执返回length，绝不第二次请求。goal/formatter/rewrite length仍终止0。

技术失败最多5次实际尝试（含首次和重启），持久retired请求与关联stage、未知费用预留。只有没有任何返回迹象且没有已记returned_usage的unknown可在该界限内恢复。response_id/raw_response/usage/returned_model/answer/content/refusal/finish_reason任一迹象，或ledger已先记actual/returned_usage，都held，不盲目重发。5次仍失败则保存技术工作项，继续其他题，不偷偷换题。stage和node不可覆盖；重启从原Pipeline开头精确回放本轮已保存前缀，后续保持同一臂历史。

所有report和监测只以SQLite `mode=ro`读取，不构造Cache。只有run写入新目录自己的Cache/账本。正式主评STOP及PLAN.execution_approval=false必须在freeze/run/report检查时保持。

## 报告与费用

以每臂固定8题给native官方answerF1/EM，算法失败计0；有技术held时不给伪完整点估计，报告固定分母上下界和缺失状态。每数据集官方scorer只构造一次。配对给改善/不变/损害题数与F1/EM差异，分数据集，并保留每题结果。与8题旧P02结果的比较明确为非同期历史参考；主因果开发比较为fresh D1对fresh D0。2Wiki官方alias/demonym规则保留。

每题记录逻辑完整下游流程输入/输出tokens和估算费；实际新增returned usage费用按ledger去重，unknown预留单列，不记录LLM调用数指标。固定D的历史生成费用不是本轮新增，后续完整流程耗时也不因缓存复用而声称fresh测得。此处不执行正式TTFT/本地组件时间补测。

沿用户“不设固定预算、合理支出且无异常”约定。新增实际付费按每个active及retired request最初提交的condition归属，再join账本actual，计入技术失败尝试已经返回的费用；同payload共享回执不重复计费，unknown预留单列。原source同题完整流程费用作为参考：该condition已知新增实际费用超过历史完整query的3倍即写expense_alert供root提醒。使用共享前缀的完整variant只能作为同期费用代理，另保留各臂完整逻辑tokens/cold费，不能把部分新增费用当作无缓存部署费用。

近期趋势的单位是完整query：只收success/success_truncated、actual大于0且无unknown预留的已完成condition，排除纯缓存及技术partial，按完成时间比较最近20个完整query的实际均费和对应历史完整query均费，超过1.5倍提醒。此批最多16个完整condition，本来不足20，因此不会凑stage数触发20query趋势，也不混入其他实验以补齐分母。重启由持久回执和账本恢复归属，不把费用警报用于挑答案或忽略失败。

## 离线合同与执行门槛

`validate_offline`只读原source/旧DB，并在临时目录用fake客户端与fake检索验证：原8完整wire/reader/history/formatter/final回放；两臂全部16普通节点覆盖；当前live rewrite与actual current一致；变化历史引发新下游rewrite和query；formatter仍问raw问题；goal不直接改；goal-only无干预；算法终止不重试；三usage/model严格；合法final length一次fake调用且重启0新增；返回迹象和ledger崩溃窗保留；5次技术尝试跨重启；桥identity/packet精确检查。不会载入GPU、连远端或创建真实checkpoint。

先完成本地及root桥/worker合同，再freeze全部代码、输入、资源、原core及STOP pin，root独立审核并决定执行。这个文档本身不授权API、SSH、上传、GPU任务或恢复主评。
