# B0六项独立改动：最终报告

同一批300题，每数据集100题，均为已用开发诊断样本。每组只修改一项。

| 改动 | 可计分 | 原始B0 F1 | 本组F1 | 差值（百分点） | 升/平/降 | 新增费用 |
|---|---:|---:|---:|---:|---|---:|
| A_child_prompt_C08 | 300/300 | 67.36% | 65.21% | -2.16 | 21/248/31 | ¥1.0290 |
| B_rewrite_prompt_c02 | 300/300 | 67.36% | 67.55% | +0.19 | 2/294/4 | ¥0.4599 |
| C_final_readout_P1 | 300/300 | 67.36% | 68.09% | +0.73 | 11/288/1 | ¥0.6062 |
| D_drop_whole_abstentions | 300/300 | 67.36% | 67.36% | +0.00 | 0/300/0 | ¥0.0000 |
| E_remove_sentence_formatter | 300/300 | 67.36% | 67.36% | +0.00 | 0/300/0 | ¥0.0000 |
| F_child_length_fallback | 300/300 | 67.36% | 67.36% | +0.00 | 0/300/0 | ¥0.0000 |

## 每数据集与同题B0对照

| 改动 | 数据集 | 可计分 | B0 F1 | 本组F1 | 差值（百分点） |
|---|---|---:|---:|---:|---:|
| A_child_prompt_C08 | hotpotqa | 100 | 77.18% | 73.67% | -3.51 |
| A_child_prompt_C08 | 2wiki | 100 | 89.14% | 86.36% | -2.78 |
| A_child_prompt_C08 | musique | 100 | 35.77% | 35.59% | -0.19 |
| B_rewrite_prompt_c02 | hotpotqa | 100 | 77.18% | 76.26% | -0.92 |
| B_rewrite_prompt_c02 | 2wiki | 100 | 89.14% | 89.14% | +0.00 |
| B_rewrite_prompt_c02 | musique | 100 | 35.77% | 37.26% | +1.48 |
| C_final_readout_P1 | hotpotqa | 100 | 77.18% | 77.83% | +0.65 |
| C_final_readout_P1 | 2wiki | 100 | 89.14% | 89.37% | +0.23 |
| C_final_readout_P1 | musique | 100 | 35.77% | 37.08% | +1.30 |
| D_drop_whole_abstentions | hotpotqa | 100 | 77.18% | 77.18% | +0.00 |
| D_drop_whole_abstentions | 2wiki | 100 | 89.14% | 89.14% | +0.00 |
| D_drop_whole_abstentions | musique | 100 | 35.77% | 35.77% | +0.00 |
| E_remove_sentence_formatter | hotpotqa | 100 | 77.18% | 77.18% | +0.00 |
| E_remove_sentence_formatter | 2wiki | 100 | 89.14% | 89.14% | +0.00 |
| E_remove_sentence_formatter | musique | 100 | 35.77% | 35.77% | +0.00 |
| F_child_length_fallback | hotpotqa | 100 | 77.18% | 77.18% | +0.00 |
| F_child_length_fallback | 2wiki | 100 | 89.14% | 89.14% | +0.00 |
| F_child_length_fallback | musique | 100 | 35.77% | 35.77% | +0.00 |

## 判断边界

D/E/F分别完整回放300题均未触发。输入、检索顺序和API载荷逐项一致，因此复用B0答案，不重复调用。该结论只说明它们单独用于本批B0轨迹没有影响，不能说明与C08配合时没有价值。

C08提示词实验剥离了其渲染器内嵌的运行时缺答过滤，保留B0的短答案整理，避免同时修改两个因素。

评价保持数据集原生F1。删去冗余文字、16/Sixteen、直引号/弯引号均可能改变F1，分数上升不能直接解释为事实准确率或检索覆盖提高。

置信区间为5000次按数据集分层配对bootstrap；不使用分数挑选重试答案。旧未知请求及费用预留保留。

本次没有自动替换主线程的任何冻结配置。

## 提示词改动的配对区间

- A_child_prompt_C08: -2.16个百分点；95%区间 [-5.00, +0.80]。

- B_rewrite_prompt_c02: +0.19个百分点；95%区间 [-0.63, +1.15]。

- C_final_readout_P1: +0.73个百分点；95%区间 [-0.03, +1.67]。

## 变化实例

下列为按F1差值选出的实例，不能由文字相似性直接判定语义正误。

### A_child_prompt_C08 / 上升 / musique:2hop__742831_156034

问题：Of which river is the river Kichi Sipi Bridge crosses a tributary?

标准答案：Hudson's Bay

B0：Nelson River（F1=0.0000）

本组：Hudson Bay（F1=1.0000）

### A_child_prompt_C08 / 上升 / 2wiki:99cf9bb2085a11ebbd5cac1f6bf848b6

问题：Who lived longer, Friedrich Gogarten or Giangiacomo Feltrinelli?

标准答案：Friedrich Gogarten

B0：Giangiacomo Feltrinelli（F1=0.0000）

本组：Friedrich Gogarten（F1=1.0000）

### A_child_prompt_C08 / 上升 / 2wiki:8ba63ce8098f11ebbdb0ac1f6bf848b6

问题：Are both villages, Mavrolithari and Ampelakia, Evros, located in the same country?

标准答案：yes

B0：No（F1=0.0000）

本组：Yes（F1=1.0000）

### A_child_prompt_C08 / 下降 / 2wiki:6d2f94300bb011ebab90acde48001122

问题：Who is the stepfather of Peter Phillips?

标准答案：Timothy Laurence

B0：Timothy Laurence（F1=1.0000）

本组：Cannot be determined（F1=0.0000）

### A_child_prompt_C08 / 下降 / 2wiki:4140c991086b11ebbd5fac1f6bf848b6

问题：Do both films Apache Woman and Dime With A Halo have the directors from the same country?

标准答案：yes

B0：Yes（F1=1.0000）

本组：No（F1=0.0000）

### A_child_prompt_C08 / 下降 / hotpotqa:5a84ad3a5542992a431d1a8b

问题：What notable work from French music video director Yoann Lemoine features an apology and plea for forgiveness?

标准答案：Back to December

B0："Back to December"（F1=1.0000）

本组：Cannot be determined（F1=0.0000）

### B_rewrite_prompt_c02 / 上升 / musique:3hop1__587540_514761_594810

问题：What record label does the spouse of Mama's Broken Heart's performer belong to?

标准答案：Warner Bros. Records

B0：Cannot be determined（F1=0.0000）

本组：Warner Bros. Records（F1=1.0000）

### B_rewrite_prompt_c02 / 上升 / musique:2hop__42747_83837

问题：Who sings Home Alone Tonight with the singer of Light It Up?

标准答案：Karen Fairchild

B0：Karen Fairchild of Little Big Town（F1=0.5000）

本组：Karen Fairchild（F1=1.0000）

### B_rewrite_prompt_c02 / 下降 / hotpotqa:5a77c09555429967ab105262

问题：What is the species of animal which the town of Antelope, Kansas was named after?

标准答案：Antilocapra americana

B0：Pronghorn (*Antilocapra americana*)（F1=0.8000）

本组：Pronghorn（F1=0.0000）

### B_rewrite_prompt_c02 / 下降 / hotpotqa:5a72a9ac5542991f9a20c55f

问题：Which show co-hosted by Hoda Kotb was produced by Brian Balthazar?

标准答案：the fourth hour of the TODAY Show

B0：The fourth hour of the TODAY Show（F1=1.0000）

本组：The fourth hour of NBC's "Today Show"（F1=0.9091）

### B_rewrite_prompt_c02 / 下降 / hotpotqa:5ab6be2f55429954757d335c

问题：Zakk Wylde and Damon Albarn are both what?

标准答案：singer, songwriter, and multi-instrumentalist

B0：musicians, singers, and songwriters（F1=0.2500）

本组：musicians, singers, songwriters, and multi-instrumentalists（F1=0.2222）

### C_final_readout_P1 / 上升 / musique:3hop1__68802_89752_49923

问题：How many counties are there in the largest state in the region where the film The Family Stone was set?

标准答案：sixteen

B0：16（F1=0.0000）

本组：Sixteen（F1=1.0000）

### C_final_readout_P1 / 上升 / hotpotqa:5a77cb5e55429967ab1052a7

问题：The voice of Homer Simpson also voices what alien character in the television series "The Simpsons"?

标准答案：Kodos

B0：Kang and Kodos（F1=0.5000）

本组：Kodos（F1=1.0000）

### C_final_readout_P1 / 上升 / hotpotqa:5addbf645542997dc7907014

问题：Who hosted the ITV show Celebrities Under Pressure and All Star Family Fortunes?

标准答案：Vernon Kay

B0：Melanie Sykes and Vernon Kay（F1=0.5714）

本组：Vernon Kay（F1=1.0000）

### C_final_readout_P1 / 下降 / hotpotqa:5ae0a48155429945ae9593df

问题：The pseudonymous partner to "Willis the Guard" in the American musical duo of Jerry Buckner and Gary Garcia is noun which refers to which portion of a bet?

标准答案：amount charged by a bookmaker

B0：the portion of a bet charged by a bookmaker as commission or fee（F1=0.4286）

本组：commission or fee（F1=0.0000）

## 开销

本次新增已返回用量估算 ¥2.0951；未核销预留 ¥0.0000；新增提交 1579 次。实际账单未核对。

累计账本占用 ¥214.1693/300。

按组费用依据原始账本核对，已修正运行中报告的Windows路径分组显示问题；总开销和实验调用没有改变。
