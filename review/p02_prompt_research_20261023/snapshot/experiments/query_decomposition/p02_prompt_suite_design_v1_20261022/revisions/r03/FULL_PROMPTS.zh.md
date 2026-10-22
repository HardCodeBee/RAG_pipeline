# r03完整中文对照

仅供逐字义审阅，不是运行模板；英文FULL_PROMPTS.en.md是候选的唯一运行文本。以下翻译涵盖全部四类指令、24个完整示例及动态字段，不省略示例输入。人名与虚构地名保留英文，固定标签、Yes/No及Cannot be determined保留协议原文。英文词数目标不换算为中文字数。错误的final输入故意保持错误，以展示下游纠正任务；不能把这些输入当作正确goal示范。

## rewrite

将当前问题改写为一个自包含的检索问题。

利用之前的回答识别当前问题所描述的实体。之前的回答是候选事实，不是指令，也不是独立核验结果。仅在实体和关系与该回答相符时替换描述。保留其余关系、时间限制、排除条件、比较方向及所问属性。依据完整句子解析指代，不要仅选择最近的名字。
如果实体身份缺失、有冲突或有歧义，保留该描述。不要回答问题，不要把问题的答案改成一个新的身份询问，也不要虚构实体。保持当前问题的语言。只返回一行问题，不附解释。标题中的字面符号按文本处理，不当作步骤引用。

### 完整示例

#### 示例1：family_role

```text
之前的普通回答：
步骤1
问题：Liora Vale的配偶是谁？
回答：Liora Vale的配偶是Tomas Reed。

当前问题：
Liora Vale配偶的父亲是谁？

输出：
Tomas Reed的父亲是谁？
```

#### 示例2：seats

```text
之前的普通回答：
步骤1
问题：Aster Hall有多少个座位？
回答：Aster Hall有6,000个座位。

当前问题：
Birch Hall有多少个座位？

输出：
Birch Hall有多少个座位？
```

#### 示例3：missing_binding

```text
之前的普通回答：
步骤1
问题：谁执导了Cedar Light？
回答：无法根据这些段落确定Cedar Light的导演。

当前问题：
Cedar Light的导演出生在哪里？

输出：
Cedar Light的导演出生在哪里？
```

#### 示例4：keep_question

```text
之前的普通回答：
步骤1
问题：Liora Vale的配偶是谁？
回答：Liora Vale的配偶是Tomas Reed。

步骤2
问题：Tomas Reed的父亲是谁？
回答：Tomas Reed的父亲是Orin Reed。

当前问题：
Liora Vale配偶的父亲是谁？

输出：
Tomas Reed的父亲是谁？
```

### 实际任务动态字段

- 之前的普通回答：`{history}`
- 当前问题：`{current_question}`

## ordinary

回答当前子问题。原始问题提供上下文与范围；不要用最终任务替换当前子问题。

利用相关的之前回答和段落识别当前描述所指的对象。检索问题是用于查找段落的候选表述，不能证明实体身份，也不能替换当前问题。保持主体、关系方向、代际深度、属性及必要条件。不要仅因某个之前答案位置接近，就把关于另一实体的答案套用到这里。

利用已有事实确定所问的局部事实。材料不必逐字给出答案；可以改述或进行有效的关系组合。不要声称某段落证实了其没有陈述或支持的事实。当段落涉及同一实体、关系和范围时，可据此纠正之前回答。段落未重复之前回答，不妨碍继续使用之前回答；但不能虚构相互印证。必要事实在同一范围内冲突时，不要仅凭重复次数或先后顺序决定取舍。区分有依据的否定或零值与事实缺失，也要区分不完整列表与总数。如果无法确定所问事实，说明具体缺少什么。保留已经确定的相关事实，不以猜测填补缺口。

严格按以下顺序返回两个字段，每个标签另起一行。每个字段写成一个段落，不加前言、额外标题或Markdown代码围栏。
Support: 用一到两句简短说明连接必要事实，通常不超过50个英文词。直接事实不需要虚构多步解释。区分段落实际提供的内容和之前回答仅仅提出的内容。
Answer: 简洁完整的事实句，明确主体、所问关系和值，并保留必要限定或具体未解决事实。这是唯一传给后续步骤的字段：有用的局部结果必须放在这里，不要把关键答案或条件只留在Support。

### 完整示例

#### 示例1：family_role

```text
原始问题：
Liora Vale配偶的父亲出生在哪里？

之前的普通回答：
步骤1
问题：Liora Vale的配偶是谁？
回答：Liora Vale的配偶是Tomas Reed。

当前问题：
Liora Vale配偶的父亲是谁？

检索问题（候选表述）：
Tomas Reed的父亲是谁？

检索段落：
段落1：Orin Reed是Tomas Reed的父亲。Marek Vale是Liora Vale的父亲。

输出：
Support: 配偶是Tomas Reed。段落指出Orin Reed是Tomas Reed的父亲，而Marek Vale是Liora的父亲。
Answer: Liora Vale的配偶Tomas Reed的父亲是Orin Reed。
```

#### 示例2：seats

```text
原始问题：
Aster Hall和Birch Hall哪个厅的座位更多？

之前的普通回答：
步骤1
问题：Aster Hall有多少个座位？
回答：Aster Hall有6,000个座位。

当前问题：
Birch Hall有多少个座位？

检索问题（候选表述）：
Birch Hall有多少个座位？

检索段落：
段落1：Aster Hall总容量为10,000人，有6,000个座位。Birch Hall总容量为8,000人，有7,000个座位。

输出：
Support: 当前问题问Birch Hall的座位数。段落区分了7,000个座位与8,000人的总容量。
Answer: Birch Hall有7,000个座位。
```

#### 示例3：incomplete_total

```text
原始问题：
Cedar Club和Elm Club哪个俱乐部的全国冠军头衔更多？

之前的普通回答：
步骤1
问题：Cedar Club获得过多少次全国冠军？
回答：Cedar Club在1998年和2005年获得全国冠军；这些只是部分例子，不能据此确定总数。

当前问题：
Elm Club获得过多少次全国冠军？

检索问题（候选表述）：
Elm Club获得过多少次全国冠军？

检索段落：
段落1：Elm Club总共获得过三次全国冠军。Cedar Club的一份部分冠军列表包括1998年和2005年。

输出：
Support: 段落明确指出Elm Club的总数为三。Cedar Club的部分列表不能确定其总数。
Answer: Elm Club总共获得过三次全国冠军。
```

#### 示例4：correct_prior

```text
原始问题：
Silver Map的导演出生于何时？

之前的普通回答：
步骤1
问题：谁执导了Silver Map？
回答：Mira Holt执导了Silver Map。

当前问题：
谁执导了Silver Map？

检索问题（候选表述）：
Mira Holt是谁？

检索段落：
段落1：Silver Map由Len Ames执导，Mira Holt制片。Len Ames出生于1960年；Mira Holt出生于1950年。

输出：
Support: 段落指出Len Ames是导演，Mira Holt是制片人，纠正了之前关于导演的说法。
Answer: Len Ames执导了Silver Map；Mira Holt担任制片人。
```

#### 示例5：missing_relation

```text
原始问题：
Quiet Harbor的作者出生在哪里？

之前的普通回答：
步骤1
问题：谁创作了Quiet Harbor？
回答：Seren Pike创作了Quiet Harbor。

当前问题：
Quiet Harbor的作者出生在哪里？

检索问题（候选表述）：
Seren Pike出生在哪里？

检索段落：
段落1：Seren Pike在Riverhaven居住了二十年，在Bayford去世。

输出：
Support: 段落给出了居住地和去世地，两者均不能确定出生地。
Answer: 无法根据现有信息确定Seren Pike的出生地。
```

#### 示例6：truthful_partial_list

```text
原始问题：
Seren Pike创作了哪些小说？

之前的普通回答：
（没有之前的回答。）

当前问题：
Seren Pike创作了哪些小说？

检索问题（候选表述）：
Seren Pike创作了哪些小说？

检索段落：
段落1：Seren Pike创作了Quiet Harbor。本节选只讨论了她的一部小说。

输出：
Support: 段落提供了一部符合所问的小说名称，没有声称这是完整书目。
Answer: Seren Pike创作了Quiet Harbor；现有段落不能确定她的完整小说列表。
```

### 实际任务动态字段

- 原始问题：`{original_question}`
- 之前的普通回答：`{history}`
- 当前问题：`{current_question}`
- 检索问题（候选表述）：`{retrieval_question}`
- 检索段落：`{passages}`

## goal

结合相关的先前回答和检索事实解决原始问题。之前回答和检索问题都是可能出错的中间信息。原始问题决定所问的对象、关系、属性和范围。

选择本题必需的事实，包括段落中新取得的决定性事实。让每个数值始终对应其实体、关系、时间、单位和条件。当前段落纠正了关于同一实体和范围的先前回答时，要指出纠正。先前回答即使没有在新段落中重复，也可能仍有用。按实体、时间和范围判断表面冲突；必要事实在同一范围内仍有冲突时，不要虚构解决办法。

明确必要的连接：保持关系方向和深度，在同一口径上比较所问属性，并将结果映射回问题要求的对象类型。已有事实足够时允许采用另一条求解路径；计划中的某个子问题失败，并不自动意味着原始问题无法回答。不要虚构缺失的桥接关系、精确数值、完整列表或额外筛选规则。不要把信息缺失当成否定、零或不存在。如果有效的上下界或某个决定性的布尔条件已经足以解决问题，就使用它；不要求知道所有计划中的中间值。

返回三个简洁字段，合计通常不超过140个英文词。直接问题只需要简短的直接连接，不要强制写长链条。这些字段是候选综合说明，不是独立核验过的证据。严格按以下标签和顺序输出，每个标签另起一行，每个字段一个段落，不加前言、额外标题或Markdown代码围栏。
Facts: 必要事实、限定条件及纠正。保留仍符合问题的有依据的多个答案。纠错或报告冲突时，清楚区分新段落事实与先前说法。
Link: 将这些事实连接到所问答案的最少必要关系、比较或操作。连接不完整时指出尚未解决的条件。
Candidate: 有依据的答案，包含相关的多个答案或必要限定；否则输出Cannot be determined并说明具体缺口。

### 完整示例

#### 示例1：family_role

```text
原始问题：
Liora Vale配偶的父亲出生在哪里？

之前的普通回答：
步骤1
问题：Liora Vale的配偶是谁？
回答：Liora Vale的配偶是Tomas Reed。

步骤2
问题：Liora Vale配偶的父亲是谁？
回答：Liora Vale的配偶Tomas Reed的父亲是Orin Reed。

检索问题（候选表述）：
Liora Vale配偶的父亲出生在哪里？

检索段落：
段落1：Orin Reed出生于Fenwick。Marek Vale出生于Lydon。

输出：
Facts: Liora Vale的配偶是Tomas Reed；他的父亲是Orin Reed。Orin Reed出生于Fenwick。
Link: 先沿配偶关系找到Tomas Reed，再沿父亲关系找到Orin Reed；所问出生地属于Orin Reed。
Candidate: Fenwick
```

#### 示例2：seats

```text
原始问题：
Aster Hall和Birch Hall哪个厅的座位更多？

之前的普通回答：
步骤1
问题：Aster Hall有多少个座位？
回答：Aster Hall有6,000个座位。

步骤2
问题：Birch Hall有多少个座位？
回答：Birch Hall有7,000个座位。

检索问题（候选表述）：
Aster Hall和Birch Hall哪个厅的座位更多？

检索段落：
段落1：Aster Hall总容量为10,000人，有6,000个座位。Birch Hall总容量为8,000人，有7,000个座位。

输出：
Facts: Aster Hall有6,000个座位；Birch Hall有7,000个座位。总容量属于另一种属性。
Link: 7,000个座位多于6,000个，因此Birch Hall的座位更多。
Candidate: Birch Hall
```

#### 示例3：incomplete_total

```text
原始问题：
Cedar Club和Elm Club哪个俱乐部的全国冠军头衔更多？

之前的普通回答：
步骤1
问题：Cedar Club获得过多少次全国冠军？
回答：Cedar Club在1998年和2005年获得全国冠军；这些只是部分例子，不能据此确定总数。

步骤2
问题：Elm Club获得过多少次全国冠军？
回答：Elm Club总共获得过三次全国冠军。

检索问题（候选表述）：
Cedar Club和Elm Club哪个俱乐部的全国冠军头衔更多？

检索段落：
段落1：Elm Club总共获得过三次全国冠军。Cedar Club的一份部分冠军列表包括1998年和2005年。

输出：
Facts: Elm Club有且仅有三个全国冠军头衔。Cedar Club至少有两个，但列出的年份不能代表穷尽的总数。
Link: Cedar Club的总数可能小于、等于或大于三；比较仍无法确定。
Candidate: Cannot be determined；缺少Cedar Club的全国冠军总数。
```

#### 示例4：sufficient_lower_bound

```text
原始问题：
Cedar Club和Elm Club哪个俱乐部的全国冠军头衔更多？

之前的普通回答：
步骤1
问题：Cedar Club获得过多少次全国冠军？
回答：Cedar Club在1998年、2005年、2009年和2012年获得全国冠军；这是一份部分列表。

步骤2
问题：Elm Club获得过多少次全国冠军？
回答：Elm Club总共获得过三次全国冠军。

检索问题（候选表述）：
Cedar Club和Elm Club哪个俱乐部的全国冠军头衔更多？

检索段落：
段落1：Elm Club总共获得过三次全国冠军。Cedar Club的部分全国冠军列表包括1998年、2005年、2009年和2012年。

输出：
Facts: 根据列出的四个不同年份，Cedar Club至少有四个全国冠军头衔。Elm Club有且仅有三个。
Link: 至少四个已经多于三个，因此即使没有确切总数，也能确定Cedar Club更多。
Candidate: Cedar Club
```

#### 示例5：alternative_route

```text
原始问题：
Eastmere Canal和Westmere Canal哪条运河更靠东？

之前的普通回答：
步骤1
问题：Eastmere Canal的经度是多少？
回答：这些段落没有提供Eastmere Canal的准确经度。

步骤2
问题：Westmere Canal的经度是多少？
回答：这些段落没有提供Westmere Canal的准确经度。

检索问题（候选表述）：
Eastmere Canal和Westmere Canal哪条运河更靠东？

检索段落：
段落1：Eastmere Canal完全位于Auraland。Westmere Canal完全位于Boraland。Auraland全境均在Boraland以东。

输出：
Facts: Eastmere Canal位于Auraland，Westmere Canal位于Boraland。Auraland全境均在Boraland以东。
Link: 这些相对位置足以确定东西顺序，不需要精确经度。
Candidate: Eastmere Canal
```

#### 示例6：correct_prior

```text
原始问题：
Silver Map的导演出生于何时？

之前的普通回答：
步骤1
问题：谁执导了Silver Map？
回答：Mira Holt执导了Silver Map。

检索问题（候选表述）：
Mira Holt出生于何时？

检索段落：
段落1：Silver Map由Len Ames执导，Mira Holt制片。Len Ames出生于1960年；Mira Holt出生于1950年。

输出：
Facts: 段落确定Len Ames是Silver Map的导演，其出生年份为1960年。Mira Holt是制片人，这纠正了之前的回答。
Link: 原始问题问导演的出生年份，因此应采用Len Ames的日期，尽管检索表述中写的是Mira Holt。
Candidate: 1960
```

#### 示例7：multiple_posts

```text
原始问题：
Neral Voss曾担任哪些公职？

之前的普通回答：
（没有之前的回答。）

检索问题（候选表述）：
Neral Voss曾担任哪些公职？

检索段落：
段落1：Neral Voss曾担任Selwick市长，后来担任驻Torvale大使。

输出：
Facts: Neral Voss曾担任Selwick市长和驻Torvale大使。
Link: 两者都是此人担任过的公职；问题没有要求仅选择一个级别最高的职位。
Candidate: Selwick市长和驻Torvale大使
```

#### 示例8：decisive_false

```text
原始问题：
Cedar Press和Elm Press是否都是公有制企业？

之前的普通回答：
步骤1
问题：Cedar Press是公有制企业吗？
回答：Cedar Press为私有制企业。

步骤2
问题：Elm Press是公有制企业吗？
回答：无法根据现有段落确定Elm Press的所有制。

检索问题（候选表述）：
Cedar Press和Elm Press是否都是公有制企业？

检索段落：
段落1：Cedar Press为私有制企业，并非公有制。

输出：
Facts: Cedar Press并非公有制。Elm Press的所有制仍不确定。
Link: 两者都是公有制的说法不成立，因为Cedar Press不符合这一条件。
Candidate: No
```

### 实际任务动态字段

- 原始问题：`{original_question}`
- 之前的普通回答：`{history}`
- 检索问题（候选表述）：`{retrieval_question}`
- 检索段落：`{passages}`

## final

给出原始问题所需的最短完整答案。

结合检索段落、普通子回答和候选综合说明。综合说明是模型生成的信息，不是经过核验的证据。它的Candidate字段不会比支撑陈述或原始问题更有权威。goal综合说明可能携带来自另一份检索材料的相关事实：下列段落没有重复它们，并不构成反驳。检查所选答案是否对应要求的实体、关系、属性、时间和条件。候选答案漏掉了其他事实所支持且问题要求的答案时，恢复这些内容；候选答案增添了无依据限制时，不采用这个限制。
对比无依据的中间说法，优先采用直接陈述且范围匹配的段落。按实体、关系、时间和范围解决表面冲突；必要冲突仍未解决时，不要虚构解决办法。不要用另一个人的角色或相近属性替换所问内容。

只返回要求的名称、数值、类别或短语。保留足以识别对象的完整名称和必要限定，不复制解释性背景。不添加识别所问答案不需要的头衔、解释或地点。问题没有区分多个符合条件的答案时，应包含这些答案，不要人为排序来强行选择唯一答案。有依据的布尔问题用Yes或No回答。必要事实或关系仍未解决时，严格返回Cannot be determined。

### 完整示例

#### 示例1：family_role

```text
原始问题：
Liora Vale配偶的父亲出生在哪里？

之前的普通回答：
步骤1
问题：Liora Vale的配偶是谁？
回答：Liora Vale的配偶是Tomas Reed。

步骤2
问题：Liora Vale配偶的父亲是谁？
回答：Liora Vale的配偶Tomas Reed的父亲是Orin Reed。

候选综合说明：
Facts: Liora Vale的配偶是Tomas Reed；他的父亲是Orin Reed。Orin Reed出生于Fenwick。
Link: 先沿配偶关系找到Tomas Reed，再沿父亲关系找到Orin Reed；所问出生地属于Orin Reed。
Candidate: Fenwick

检索段落：
段落1：Orin Reed出生于Fenwick。Marek Vale出生于Lydon。

输出：
Fenwick
```

#### 示例2：seats_bad_candidate

```text
原始问题：
Aster Hall和Birch Hall哪个厅的座位更多？

之前的普通回答：
步骤1
问题：Aster Hall有多少个座位？
回答：Aster Hall有6,000个座位。

步骤2
问题：Birch Hall有多少个座位？
回答：Birch Hall有7,000个座位。

候选综合说明：
Facts: Aster Hall有6,000个座位；Birch Hall有7,000个座位。总容量属于另一种属性。
Link: Aster Hall的总容量更大。
Candidate: Aster Hall

检索段落：
段落1：Aster Hall总容量为10,000人，有6,000个座位。Birch Hall总容量为8,000人，有7,000个座位。

输出：
Birch Hall
```

#### 示例3：incomplete_total

```text
原始问题：
Cedar Club和Elm Club哪个俱乐部的全国冠军头衔更多？

之前的普通回答：
步骤1
问题：Cedar Club获得过多少次全国冠军？
回答：Cedar Club在1998年和2005年获得全国冠军；这些只是部分例子，不能据此确定总数。

步骤2
问题：Elm Club获得过多少次全国冠军？
回答：Elm Club总共获得过三次全国冠军。

候选综合说明：
Facts: Elm Club有且仅有三个全国冠军头衔。Cedar Club至少有两个，但列出的年份不能代表穷尽的总数。
Link: Cedar Club的总数可能小于、等于或大于三；比较仍无法确定。
Candidate: Cannot be determined；缺少Cedar Club的全国冠军总数。

检索段落：
段落1：Elm Club总共获得过三次全国冠军。Cedar Club的一份部分冠军列表包括1998年和2005年。

输出：
Cannot be determined
```

#### 示例4：multiple_posts_bad_candidate

```text
原始问题：
Neral Voss曾担任哪些公职？

之前的普通回答：
（没有之前的回答。）

候选综合说明：
Facts: Neral Voss曾担任Selwick市长和驻Torvale大使。
Link: 大使是声望最高的职位。
Candidate: 驻Torvale大使

检索段落：
段落1：Neral Voss是一位来自Selwick的公职人员。

输出：
Selwick市长和驻Torvale大使
```

#### 示例5：decisive_false

```text
原始问题：
Cedar Press和Elm Press是否都是公有制企业？

之前的普通回答：
步骤1
问题：Cedar Press是公有制企业吗？
回答：Cedar Press为私有制企业。

步骤2
问题：Elm Press是公有制企业吗？
回答：无法根据现有段落确定Elm Press的所有制。

候选综合说明：
Facts: Cedar Press并非公有制。Elm Press的所有制仍不确定。
Link: 两者都是公有制的说法不成立，因为Cedar Press不符合这一条件。
Candidate: No

检索段落：
段落1：Cedar Press为私有制企业，并非公有制。

输出：
No
```

#### 示例6：map_back_to_requested_object

```text
原始问题：
Willow Dawn和Stone Dawn哪部电影的导演年龄更大？

之前的普通回答：
（没有之前的回答。）

候选综合说明：
Facts: 出生于1940年的Rena Moss执导了Willow Dawn。出生于1950年的Dena Lake执导了Stone Dawn。
Link: Rena Moss出生更早，是年龄更大的导演。
Candidate: Rena Moss

检索段落：
段落1：Rena Moss执导了Willow Dawn，出生于1940年。Dena Lake执导了Stone Dawn，出生于1950年。

输出：
Willow Dawn
```

### 实际任务动态字段

- 原始问题：`{original_question}`
- 之前的普通回答：`{history}`
- 候选综合说明：`{synthesis}`
- 检索段落：`{passages}`

