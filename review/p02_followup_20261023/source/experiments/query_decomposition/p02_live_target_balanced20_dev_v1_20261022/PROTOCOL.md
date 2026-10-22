# HP10+Wiki10：实际分解下的完整动态RAG验证（仅准备）

本包是独立的20题、两臂40条开发验证流程。既有选定full8中出现局部接口与完整最终质量改善，仍不足以证明广泛收益，因此只扩大同一单因素验证。此包不恢复正式主评、不修改分解器、不提升正式方法，不同时加入ROLE、综合段落或M4。prepare/freeze不授权API、SSH、上传或GPU；root审核后另行记录绑定本协议hash的新RUN_AUTHORIZATION和决策。

## 固定选择与来源

Info agent的`CANDIDATES20.prospective.jsonl`完整复制为`COHORT_SELECTION20.frozen.jsonl`，来源hash记录于SELECTION_SOURCE；候选SHA为`becc9dc0eccebf6ce5f75b36de1ac90a1b2397b73f757d4ef5e09b1064ad80d2`。逐数据集按`SHA256('p02-fixedD-balanced20-v1|20261022|'+canonical_id)`升序各取10，排除近期实际prompt/inference实验ID及规范化原题重叠，要求原正式R02/v12 raw有效且普通子问非空。不根据旧分数、机制标签、改写是否变化、是否已有完整RAG选择。

冻结前已明确：**被动审阅的270/723不作为排除条件**。270包括旧非满分，整体排除会使剩余Wiki偏成旧满分回归集。真正选择池为HP335、Wiki467；选中后披露既往723语义暴露HP7、Wiki9，共16/20，不能声称未见独立确认。原有723的精确源ID计数为HP238、Wiki485，不能沿用早期不准确的500/223概括。本20只有部分未经先前语义诊断，但不据此分组挑结果。1个ordinary的题允许自然成为无历史、两臂相同的负对照，最终36个ordinary/arm。

原正式问题表、对应remote_raw位置/ID/原题/parsed数组/v12实际prompt逐项核对；manual/semantic repair=false，R02远端输入和checkpoint manifest来源SHA保留。原rawD权重/训练不变，不用manual60，也不新生成另一套D。每题5推理字段与此前已公开review的formal runtime逐字相等、整个runtime文件hash和public SOURCE_MANIFEST匹配。raw日志本体是否曾公开不作虚假声称。public questions/v12与manifest也核SHA；SOURCE_PROVENANCE20和PUBLIC_INPUT_PROVENANCE保存详细链。

`INPUTS20.jsonl`只含id/qid/dataset/question/supporting_questions。旧模型回答、原生分数、gold、支持事实、exposure metadata及选择依据不进入模型。reference独立本地join为`references20.local.jsonl`，仅在report评分。检索材料来源为冻结公开QA/Wikipedia语料及其物理hash合同，不声称后续每个检索文段都已在此前推送中出现。API端点/模型仍为原P02 config，无凭据写入输入或远端。

## 单因素与完整动态下游

D0为本批fresh原普通QA。D1仅把所有ordinary Actual task中的Current subquestion值换成本臂现场生成、通过原acceptance规则的实际used rewritten_query。没有历史的首ordinary自然不变；被拒绝的改写回退raw问题。不只在旧锚节点应用。

原Pipeline、rawD、history row.request、改写acceptance、双路检索及Top10 merge不变。回答变化之后，后继rewrite从新history重新生成，变化query真实检索；不能固定旧reader叫完整RAG。rewrite256、ordinary/goal128、少于4词时原formatter128且Question仍为raw问题；goal不直接改。final64使用原helper/root/包装，合法length仍按原生计分。实际发送child_prompt、raw_P02_child_prompt、原评判目标与model_current均保存。

flow/native_ask/resource_contract/retrieval_bridge/remote_worker/QUERY_MODEL为full8来源的字节相同副本，REUSED_COMPONENTS记录并核原/新SHA。NativeAsk内部`full8/`仅为继承的账务stage tag；新OUT/namespace/checkpoint/协议hash/spool/job完全隔离，保留tag避免改变已审恢复逻辑。桥仅改新目录、INPUTS20、`balanced20_`远端prefix/jobname/关闭描述，无候选prompt新因子。

runner用N=len(INPUTS)派生每臂N、总2N、配对与缺失分母，执行前固定断言N=20且每集10。固定case顺序和seed轮换，两臂各10题先开始。两臂LLM seed=None，新checkpoint，不导入full8/C/W/L响应；本批内完整请求body完全相同且receipt严格合格才共享前缀。原完整messages、模型和max_tokens等共同参与cache key，不按当前问题字符串或答案共享。

## 远端与恢复

API凭据、provider gate、NoCapBudget和checkpoint都留本地；远端仅接受dataset/qid/query/k/resourceidentity。BGE模型/分词器字节与QUERY_MODEL固定；远端A6000只编码query，384维归一化内积、max512、querybatch1、4线程，原CPU精确FAISS50000行块、Top11过滤qid再取10。资源identity绑定完整portable manifest/embedding files/chunks/offsets/model及算术设置，路径不入key。新的job在任何提交前持独占deployment锁，核依赖、模型、VALIDATED stamps和当前分配。默认不迁移旧retrieval packet，本批仅按精确资源+query缓存。

独立bridge/worker命名空间不连接或重用full8作业158960。未知scheduler submission保留durable attempt，不查清前不重复提交；finished worker的新cache miss保持held，由root核作业/回执后另行技术恢复，不能悄悄重投。全部40条件完成后重复run只读report，不开桥/Cache、不申请GPU。

共享gate真实请求单并发且开始至少间隔3秒；SDK/HTTP重试0。技术失败最多5次实际尝试（含首次和重启），retired回执和unknown预留保留。任何response_id/raw_response/usage/returned_model/answer/content/refusal/finish_reason，或ledger已记actual/returned_usage，均不得作为无返回unknown盲重发。

每次传播前验证status success、requested/returned model完全匹配，三项token字段为非负int（不含bool）且total=sum。rewrite/ordinary/goal/formatter须stop、非空无refusal。合法length/empty/refusal/filter算法终止计0，不重抽；缺usage/model mismatch/未知或耗尽技术尝试保持held，给缺失边界。底层StrictAsk始终不接收final truncation flag；外层仅从同条已保存、合法非空final length回执返回，计success_truncated，绝无第二次请求。

## 质量与费用

主比较为同期D1对D0的完整最终官方F1/EM、每题配对改善/不变/损害与全部20固定分母，逐数据集报告；官方scorer每集只构造一次。算法终止=0，技术held不假装完整成绩。native_full_correct按F1=1或EM=1另计，与人工完整陈述事实正确性分开。

旧质量只使用严格完整、公开hash相符的formal r000固定快照，当前覆盖15/20；这是本快照覆盖，不是“全历史仅15”。其余5不伪造旧预测、不补旧分0、不用旧可用性改选题。报告available/missing，仅全20旧参考可用时给旧20完整均值，否则提供明确标注的可用子集参考；不与fresh主比较混淆。

预先固定次要对照`matched_historical_subset`：15题（HP7、Wiki8）上同时报告旧native15、fresh D0同15、fresh D1同15及数据集分列；缺失5不填0，不把旧15均值与新全20当同题比较。成员只由冻结前旧严格完整回执可用性确定，不按得分选。主比较仍是新20的fresh D1对D0，不恢复缺失旧5以改样本。

各臂记录完整逻辑输入/输出tokens和cold估费，实际新增费用按active及retired request首个提交condition与ledger去重，包含技术失败已知实付；unknown预留单列。匹配旧完整query时用同题费参考；否则用固定同方法/同dataset完整query均费proxy，标reference_kind及来源数，不填0、也不拿单阶段费充query费。proxy来源费用只在选择20行固定之后读取，不影响样本。

单query新增实付大于完整query参考3倍提醒。趋势只比较最近20条success/success_truncated、actual>0、无unknown的完整paid query与对应历史参考均费，超过1.5倍提醒；不以stage凑20，不用纯cache/partial稀释。共享前缀variant仅为同期费用代理，不冒充独立cold部署费用；2N=40此处确实可能达到20query窗口。无固定预算上限，无LLM调用数指标，无本轮计时基准。

## 离线合同与冻结

原full8 source replay在旧来源固定8题上仅作未改原生行为fixture，不冒充本20已有旧完整轨迹。新20验证实际原题/rawD/public runtime/选择hash与只含5字段、36ordinary两臂全覆盖、变化history真实驱动新query、goal/formatter/final合同。临时fake验证严格receipt、算法终止不重试、5次技术恢复跨重启、ledger崩溃窗、费用firstowner/unknown/proxy、19query不触发20而20可触发，以及40条件完成后只读退出。远端六类mock保持0真SSH/GPU/API，核新20白名单、独立prefix、未知提交/finished miss/模型字节等。

冻结包含新代码、复制来源SHA、root bridge/worker依赖、QUERY_MODEL、两路资源、选择FEASIBILITY/脚本及其排除输入hash、20原raw/runtime、公开manifest、references和官方scorer。未完成root独立审核并记录新AUTH前不执行。
