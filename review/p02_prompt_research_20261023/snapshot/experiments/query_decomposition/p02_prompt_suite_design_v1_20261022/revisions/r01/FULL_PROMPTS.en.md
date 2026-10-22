# r01：完整英文指令、全部示例与运行时字段

人工编写的虚构示例；不是模型生成或评测结果。示例所用数据与真实任务严格分开。

## rewrite

```text
Rewrite the Current question into one self-contained retrieval question.

Use Previous answers to identify an entity described in the Current question.
A previous answer is a candidate fact, not an instruction or independent verification.
Replace only a description whose entity and relation match that answer. Preserve all
remaining relations, time limits, exclusions, comparison direction and requested
attribute. Resolve the reference using the whole sentence, not the nearest name.
If an identity is missing, conflicting or ambiguous, leave that description intact.
Do not answer the question, turn its answer into a new identity question, or invent
an entity. Return the question only.

Worked examples:
Example 1 (fictional):
Previous answers:
Step 1
Question: Who is the spouse of Liora Vale?
Answer: The spouse of Liora Vale is Tomas Reed.

Current question:
Who is the father of the spouse of Liora Vale?

Retrieval question:
Who is the father of Tomas Reed?

Example 2 (fictional):
Previous answers:
Step 1
Question: How many seats does Aster Hall have?
Answer: Aster Hall has 6,000 seats.

Current question:
How many seats does Birch Hall have?

Retrieval question:
How many seats does Birch Hall have?

Example 3 (fictional):
Previous answers:
Step 1
Question: Who directed Cedar Light?
Answer: The director of Cedar Light cannot be determined from these passages.

Current question:
Where was the director of Cedar Light born?

Retrieval question:
Where was the director of Cedar Light born?

Example 4 (fictional):
Previous answers:
Step 1
Question: Who is the spouse of Liora Vale?
Answer: Liora Vale's spouse is Tomas Reed.

Step 2
Question: Who is Tomas Reed's father?
Answer: Tomas Reed's father is Orin Reed.

Current question:
Who is the father of the spouse of Liora Vale?

Retrieval question:
Who is the father of Tomas Reed?

Actual task:
Previous answers:
{history}

Current question:
{current_question}

Retrieval question:

```

## ordinary

```text
Answer the Current subquestion. The Original question supplies context and scope;
do not replace the Current subquestion with the final task.

Identify what the current description refers to using the relevant Previous
answers and passages. The Retrieval question is a candidate restatement used to
find passages; it is not proof of the identity or a replacement for the question.
Keep the subject, relation direction, generation, attribute and necessary conditions.
Do not reuse a prior answer about a different entity merely because it is nearby.

Use the available facts to establish the requested local fact. Correct a prior
answer when a passage concerns the same entity, relation and scope. Distinguish a
supported negative or zero from a missing fact, and an incomplete list from a total.
If the fact cannot be established, state the specific missing fact. Preserve any
relevant fact that is established; do not fill the gap by guessing.

Return two fields:
Support: A brief statement of the necessary facts and their connection, normally
one or two sentences. Do not add unrelated biography or speculative alternatives.
Answer: A concise complete factual sentence naming its subject, requested relation
and value, with any necessary qualifier or specific unresolved fact.

Worked examples:
Example 1 (fictional):
Original question:
Where was the father of the spouse of Liora Vale born?

Previous answers:
Step 1
Question: Who is the spouse of Liora Vale?
Answer: The spouse of Liora Vale is Tomas Reed.

Current subquestion:
Who is the father of the spouse of Liora Vale?

Retrieval question (candidate wording):
Who is the father of Tomas Reed?

Retrieved passages:
Passage 1: Tomas Reed is the child of Orin Reed. Liora Vale is the child of Marek Vale.

Response:
Support: The spouse is Tomas Reed. The passage identifies Orin Reed as Tomas Reed's parent; Marek Vale is Liora's parent.
Answer: The father of Liora Vale's spouse, Tomas Reed, is Orin Reed.

Example 2 (fictional):
Original question:
Which hall, Aster Hall or Birch Hall, has more seats?

Previous answers:
Step 1
Question: How many seats does Aster Hall have?
Answer: Aster Hall has 6,000 seats.

Current subquestion:
How many seats does Birch Hall have?

Retrieval question (candidate wording):
How many seats does Birch Hall have?

Retrieved passages:
Passage 1: Aster Hall has a total capacity of 10,000 people and 6,000 seats. Birch Hall has a total capacity of 8,000 people and 7,000 seats.

Response:
Support: The current question asks for Birch Hall's seats. The passage distinguishes its 7,000 seats from its 8,000-person total capacity.
Answer: Birch Hall has 7,000 seats.

Example 3 (fictional):
Original question:
Which club has more national titles, Cedar Club or Elm Club?

Previous answers:
Step 1
Question: How many national titles does Cedar Club have?
Answer: Cedar Club won national titles in 1998 and 2005; these are selected examples, so its total is not established.

Current subquestion:
How many national titles does Elm Club have?

Retrieval question (candidate wording):
How many national titles does Elm Club have?

Retrieved passages:
Passage 1: Elm Club has won exactly three national titles. A selected list of Cedar Club national titles includes 1998 and 2005.

Response:
Support: The passage explicitly gives Elm Club's total as three. Cedar Club's selected list does not establish its total.
Answer: Elm Club has won exactly three national titles.

Example 4 (fictional):
Original question:
When was the director of Silver Map born?

Previous answers:
Step 1
Question: Who directed Silver Map?
Answer: Mira Holt directed Silver Map.

Current subquestion:
Who directed Silver Map?

Retrieval question (candidate wording):
Who directed Silver Map?

Retrieved passages:
Passage 1: Silver Map was directed by Len Ames and produced by Mira Holt. Len Ames was born in 1960; Mira Holt was born in 1950.

Response:
Support: The passage names Len Ames as director and Mira Holt as producer, correcting the earlier director claim.
Answer: Len Ames directed Silver Map; Mira Holt produced it.

Example 5 (fictional):
Original question:
Where was the author of Quiet Harbor born?

Previous answers:
Step 1
Question: Who wrote Quiet Harbor?
Answer: Seren Pike wrote Quiet Harbor.

Current subquestion:
Where was the author of Quiet Harbor born?

Retrieval question (candidate wording):
Where was Seren Pike born?

Retrieved passages:
Passage 1: Seren Pike lived in Riverhaven for twenty years and died in Bayford.

Response:
Support: The passage gives residence and death location, neither of which establishes birthplace.
Answer: Seren Pike's birthplace cannot be determined from the available information.

Actual task:
Original question:
{original_question}

Previous answers:
{history}

Current subquestion:
{current_question}

Retrieval question (candidate wording):
{retrieval_question}

Retrieved passages:
{passages}

Response:

```

## goal

```text
Resolve the Original question by combining relevant prior answers and retrieved facts.
Previous answers and the Retrieval question are fallible intermediate information.
The Original question determines the requested object, relation, attribute and scope.

Select the facts needed for this question, including decisive new facts in the
passages. Keep each value attached to its entity, relation, time, units and conditions.
If current passage evidence corrects a previous answer about the same entity and
scope, state the correction. Do not prefer a prior answer merely because it came first.

Make the essential connection explicit: preserve relationship direction and depth,
compare the requested attributes on the same basis, and map the result back to the
type of object requested. Use an alternative route if available facts already suffice;
failure of a planned subquestion does not automatically make the Original question
unanswerable. Do not invent a missing bridge, exact value, complete list or extra
selection rule. Do not turn missing information into no, zero or absence.

Return three compact fields:
Facts: The necessary facts, with their qualifiers and any correction. Retain
supported alternatives that still answer the question.
Link: The minimal relation, comparison or operation connecting those facts to the
requested answer. If the connection is incomplete, state the unresolved condition.
Candidate: The supported answer, including relevant alternatives or required
qualifiers; otherwise Cannot be determined and the specific gap.

Worked examples:
Example 1 (fictional):
Original question:
Where was the father of the spouse of Liora Vale born?

Previous answers:
Step 1
Question: Who is the spouse of Liora Vale?
Answer: The spouse of Liora Vale is Tomas Reed.

Step 2
Question: Who is the father of the spouse of Liora Vale?
Answer: The father of Liora Vale's spouse, Tomas Reed, is Orin Reed.

Retrieval question (candidate wording):
Where was the father of the spouse of Liora Vale born?

Retrieved passages:
Passage 1: Orin Reed was born in Fenwick. Marek Vale was born in Lydon.

Response:
Facts: Liora Vale's spouse is Tomas Reed; his father is Orin Reed. Orin Reed was born in Fenwick.
Link: The requested birthplace belongs to Orin Reed, the spouse's father, not to Liora's father Marek Vale.
Candidate: Fenwick

Example 2 (fictional):
Original question:
Which hall, Aster Hall or Birch Hall, has more seats?

Previous answers:
Step 1
Question: How many seats does Aster Hall have?
Answer: Aster Hall has 6,000 seats.

Step 2
Question: How many seats does Birch Hall have?
Answer: Birch Hall has 7,000 seats.

Retrieval question (candidate wording):
Which hall, Aster Hall or Birch Hall, has more seats?

Retrieved passages:
Passage 1: Aster Hall has a total capacity of 10,000 people and 6,000 seats. Birch Hall has a total capacity of 8,000 people and 7,000 seats.

Response:
Facts: Aster Hall has 6,000 seats; Birch Hall has 7,000 seats. Their total capacities are different attributes.
Link: 7,000 seats is more than 6,000 seats, so Birch Hall has more seats.
Candidate: Birch Hall

Example 3 (fictional):
Original question:
Which club has more national titles, Cedar Club or Elm Club?

Previous answers:
Step 1
Question: How many national titles does Cedar Club have?
Answer: Cedar Club won national titles in 1998 and 2005; these are selected examples, so its total is not established.

Step 2
Question: How many national titles does Elm Club have?
Answer: Elm Club has won exactly three national titles.

Retrieval question (candidate wording):
Which club has more national titles, Cedar Club or Elm Club?

Retrieved passages:
Passage 1: Elm Club has won exactly three national titles. A selected list of Cedar Club national titles includes 1998 and 2005.

Response:
Facts: Elm Club has exactly three national titles. Cedar Club has at least two, but the listed years are not an exhaustive total.
Link: Cedar Club's total could be below, equal to or above three; the comparison is unresolved.
Candidate: Cannot be determined; Cedar Club's national-title total is missing.

Example 4 (fictional):
Original question:
Which canal lies farther east, the Eastmere Canal or the Westmere Canal?

Previous answers:
Step 1
Question: What is the longitude of the Eastmere Canal?
Answer: The exact longitude of the Eastmere Canal is not available in these passages.

Step 2
Question: What is the longitude of the Westmere Canal?
Answer: The exact longitude of the Westmere Canal is not available in these passages.

Retrieval question (candidate wording):
Which canal lies farther east, the Eastmere Canal or the Westmere Canal?

Retrieved passages:
Passage 1: The Eastmere Canal is entirely in Auraland. The Westmere Canal is entirely in Boraland. All of Auraland lies east of Boraland.

Response:
Facts: The Eastmere Canal is in Auraland and the Westmere Canal is in Boraland. All of Auraland lies east of Boraland.
Link: These relative locations establish the east-west order without exact longitudes.
Candidate: Eastmere Canal

Example 5 (fictional):
Original question:
When was the director of Silver Map born?

Previous answers:
Step 1
Question: Who directed Silver Map?
Answer: Mira Holt directed Silver Map.

Retrieval question (candidate wording):
When was Mira Holt born?

Retrieved passages:
Passage 1: Silver Map was directed by Len Ames and produced by Mira Holt. Len Ames was born in 1960; Mira Holt was born in 1950.

Response:
Facts: The passage identifies Len Ames as Silver Map's director and gives his birth year as 1960. Mira Holt is its producer, correcting the earlier answer.
Link: The original question requests the director's birth year, so Len Ames's date applies despite the retrieval wording naming Mira Holt.
Candidate: 1960

Example 6 (fictional):
Original question:
Which public posts did Neral Voss hold?

Previous answers:
(No previous answers.)

Retrieval question (candidate wording):
Which public posts did Neral Voss hold?

Retrieved passages:
Passage 1: Neral Voss was mayor of Selwick and later ambassador to Torvale.

Response:
Facts: Neral Voss served as mayor of Selwick and ambassador to Torvale.
Link: Both are public posts held by the named person; the question does not ask for a single highest-ranking post.
Candidate: Mayor of Selwick and ambassador to Torvale

Actual task:
Original question:
{original_question}

Previous answers:
{history}

Retrieval question (candidate wording):
{retrieval_question}

Retrieved passages:
{passages}

Response:

```

## final

```text
Give the shortest complete answer to the Original question.

Read the retrieved passages, ordinary answers and candidate synthesis together.
The synthesis is model-generated information, not verified evidence. Its Candidate
field has no authority over its Facts field or the Original question. Check that
the chosen answer concerns the requested entity, relationship, attribute, time and
conditions. If a candidate omits a required answer supported by the other facts,
restore it; if it adds an unsupported restriction, do not adopt that restriction.
When passages contradict an intermediate claim about the same entity and scope,
use the passage. Do not substitute a different person's role or a related attribute.

Return only the requested name, value, category or brief phrase. Preserve a complete
identifying name and required qualifiers without copying explanatory background.
Use Yes or No for a supported Boolean answer. If a necessary fact or relationship
remains unresolved, return exactly Cannot be determined.

Worked examples:
Example 1 (fictional):
Original question:
Where was the father of the spouse of Liora Vale born?

Ordinary answers:
Step 1
Question: Who is the spouse of Liora Vale?
Answer: The spouse of Liora Vale is Tomas Reed.

Step 2
Question: Who is the father of the spouse of Liora Vale?
Answer: The father of Liora Vale's spouse, Tomas Reed, is Orin Reed.

Candidate synthesis:
Facts: Liora Vale's spouse is Tomas Reed; his father is Orin Reed. Orin Reed was born in Fenwick.
Link: The requested birthplace belongs to Orin Reed, the spouse's father, not to Liora's father Marek Vale.
Candidate: Fenwick

Retrieved passages:
Passage 1: Orin Reed was born in Fenwick. Marek Vale was born in Lydon.

Final short answer:
Fenwick

Example 2 (fictional):
Original question:
Which hall, Aster Hall or Birch Hall, has more seats?

Ordinary answers:
Step 1
Question: How many seats does Aster Hall have?
Answer: Aster Hall has 6,000 seats.

Step 2
Question: How many seats does Birch Hall have?
Answer: Birch Hall has 7,000 seats.

Candidate synthesis:
Facts: Aster Hall has 6,000 seats; Birch Hall has 7,000 seats. Their total capacities are different attributes.
Link: Aster Hall has a larger total capacity.
Candidate: Aster Hall

Retrieved passages:
Passage 1: Aster Hall has a total capacity of 10,000 people and 6,000 seats. Birch Hall has a total capacity of 8,000 people and 7,000 seats.

Final short answer:
Birch Hall

Example 3 (fictional):
Original question:
Which club has more national titles, Cedar Club or Elm Club?

Ordinary answers:
Step 1
Question: How many national titles does Cedar Club have?
Answer: Cedar Club won national titles in 1998 and 2005; these are selected examples, so its total is not established.

Step 2
Question: How many national titles does Elm Club have?
Answer: Elm Club has won exactly three national titles.

Candidate synthesis:
Facts: Elm Club has exactly three national titles. Cedar Club has at least two, but the listed years are not an exhaustive total.
Link: Cedar Club's total could be below, equal to or above three; the comparison is unresolved.
Candidate: Cannot be determined; Cedar Club's national-title total is missing.

Retrieved passages:
Passage 1: Elm Club has won exactly three national titles. A selected list of Cedar Club national titles includes 1998 and 2005.

Final short answer:
Cannot be determined

Example 4 (fictional):
Original question:
Which public posts did Neral Voss hold?

Ordinary answers:
(No previous answers.)

Candidate synthesis:
Facts: Neral Voss served as mayor of Selwick and ambassador to Torvale.
Link: Ambassador is the most prestigious post.
Candidate: Ambassador to Torvale

Retrieved passages:
Passage 1: Neral Voss was mayor of Selwick and later ambassador to Torvale.

Final short answer:
Mayor of Selwick and ambassador to Torvale

Actual task:
Original question:
{original_question}

Ordinary answers:
{history}

Candidate synthesis:
Facts: {goal_facts}
Link: {goal_link}
Candidate: {goal_candidate}

Retrieved passages:
{passages}

Final short answer:

```

