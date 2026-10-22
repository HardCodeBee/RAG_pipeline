# 本设计采用的经验及适用边界

## 本地实验

1. 描述式子问题适配、三prompt联改及精简版本均有完整90题记录。原P02 F1=76.69；自包含适配74.10；联改71.71；后来补齐的精简版76.04。依据：outputs/query_decomposition/hotpot_formal_eval_v1内对应approved_recovery004、approved_unknown_recovery003、delegated_recovery003/REPORT.json。格式一致不保证提升，精简v2现已补齐，不能继续沿用旧记忆的30/90状态。
2. p02_result_binding_operator_probe_v1_20261022：共同90题协议G0/GF/GO=76.62/73.76/70.21。显式操作说明不是默认升级；普通38节点结果绑定说明也无净改善。固定reader而非新完整流程，案例与语义复核有暴露。
3. p02_goal_target_transfer_new60_v1_20261022：60题G0/GF=67.47/68.26；拆分展示68.50，唯一额外涨分来自去掉near。不同样本与具体模板可能呈不同方向，不能宣布综合段落普遍无效或普遍有效。
4. p02_goal_sufficiency_retention60_v1_20261022：A/P/R相对各自对照F1为−0.67/−0.43/0.00百分点。Chadwick保留事实有局部收益；Shirley Temple指令未执行；篮球家庭矛盾被再次传播。以此设计任务结构和示例，不继续堆叠同类泛化提醒。
5. p02_shared_context_fixed90_v1_20261022：完整reset/chat两臂F1约76.69/71.91；flat有一技术暂挂，不能把可用89题写成90题完整成绩。当前新套件保留阶段重置和紧凑事实历史，不默认为共享全部对话更好。
6. p02_local_cot_stage38_v1_20261022：普通QA的短Support+Answer出现局部修复，也有错误依据或有用信息减少。最新revision002下，R256对N128：37个可判节点的接受答案3改善1退步；完整原始回复3改善2退步，1个节点待核。不是完整RAG F1，也不把正确答案掩盖错误Support。新普通输出借鉴的是待验证方向。
7. FROZEN_DECOMPOSER_FINDINGS.md记录实际Current改用改写在失效8题获得局部修复，而新抽样19个可用配对未见最终变化；reader去重在机制8题有一处改善，新20题F1持平。不要把检索措辞直接升格为任务，也不要声称所有差表现都能靠prompt修复。

各条完整源路径见[SOURCE_INDEX.md](SOURCE_INDEX.md)；冻结核心的哈希只用于确认本设计没有覆盖生产实现。

## 本轮核对的原始研究

- [Madaan等，2023，What Makes Chain-of-Thought Prompting Effective?](https://aclanthology.org/2023.findings-emnlp.101/)：作者通过改变few-shot中的符号和模式研究任务理解。对本项目的启发是分别检验表面符号与依赖/任务示范；不能推断P02的#没有作用，更不把描述展开视为只有字符变化。
- [Wang等，2023，Towards Understanding Chain-of-Thought Prompting](https://aclanthology.org/2023.acl-long.153/)：其任务中与问题相关和步骤组织对示范效果很重要。这里用少量相互衔接、内容正确的合成示例；论文不保证Qwen的局部或端到端收益。
- [Trivedi等，2023，IRCoT](https://aclanthology.org/2023.acl-long.557.pdf)：检索与中间推导相互影响，说明改变上游回答后须验证实际下游检索。本候选保持P02既有双路架构，不能以IRCoT结果宣称获得同样收益。
- [Lyu等，2023，Faithful Chain-of-Thought Reasoning](https://arxiv.org/abs/2301.13379)：其方法用符号中间表示与确定性求解器连接推导和答案。本设计没有这样的求解器；自然语言Support/Link只作可检查说明，不保证反映模型内部过程，也不保证正确。

上述研究只作为设计启发；本目录的人工例句与离线结构测试都不能代替真实模型验证。
