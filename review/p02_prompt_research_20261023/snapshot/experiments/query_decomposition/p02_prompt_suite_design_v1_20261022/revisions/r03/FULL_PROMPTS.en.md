# r03：完整英文指令、全部示例与运行时字段

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
an entity. Keep the language of the Current question. Return one question on one line,
with no explanation. Treat literal symbols in titles as text, not step references.

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

Use the available facts to establish the requested local fact. They need not state
the answer verbatim: paraphrase and valid relation composition are allowed.
Do not claim a passage confirms a fact that it does not state or support. Correct a prior
answer when a passage concerns the same entity, relation and scope. Prior answers
remain usable when passages do not repeat them; do not invent corroboration. If
material claims conflict at the same scope, do not resolve them by repetition or
recency alone. Distinguish a
supported negative or zero from a missing fact, and an incomplete list from a total.
If the fact cannot be established, state the specific missing fact. Preserve any
relevant fact that is established; do not fill the gap by guessing.

Return exactly the two fields below, in order, each label starting a new line.
Use one paragraph per field, without a preamble, extra headings or Markdown fences.
Support: One or two short sentences connecting the necessary facts, normally
within 50 words. A direct fact needs no invented multi-step explanation. Distinguish
what a passage supplies from what a prior answer merely proposes.
Answer: A concise complete factual sentence naming its subject, requested relation
and value, with any necessary qualifier or specific unresolved fact. This is the
only field passed to later steps: include the useful local result here, rather
than leaving an essential answer or condition only in Support.

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
Passage 1: Orin Reed is the father of Tomas Reed. Marek Vale is the father of Liora Vale.

Response:
Support: The spouse is Tomas Reed. The passage names Orin Reed as Tomas Reed's father, whereas Marek Vale is Liora's father.
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
Who is Mira Holt?

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

Example 6 (fictional):
Original question:
Which novels did Seren Pike write?

Previous answers:
(No previous answers.)

Current subquestion:
Which novels did Seren Pike write?

Retrieval question (candidate wording):
Which novels did Seren Pike write?

Retrieved passages:
Passage 1: Seren Pike wrote Quiet Harbor. This excerpt discusses only one of her novels.

Response:
Support: The passage supplies one responsive novel title but does not claim a complete bibliography.
Answer: Seren Pike wrote Quiet Harbor; the available passage does not establish her complete list of novels.

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
scope, state the correction. An earlier answer can remain useful without being
repeated in a new passage. Resolve apparent conflicts by entity, time and scope;
if a material same-scope conflict remains, do not invent a resolution.

Make the essential connection explicit: preserve relationship direction and depth,
compare the requested attributes on the same basis, and map the result back to the
type of object requested. Use an alternative route if available facts already suffice;
failure of a planned subquestion does not automatically make the Original question
unanswerable. Do not invent a missing bridge, exact value, complete list or extra
selection rule. Do not turn missing information into no, zero or absence. Use a valid bound
or decisive Boolean condition when it already settles the question; not every
planned intermediate value must be known.

Return three compact fields, normally within 140 words in total. A direct
answer needs only a brief direct connection; do not force a long chain. These
fields are a candidate synthesis, not independently verified evidence. Use
exactly these labels in this order, each starting a new line, with one paragraph
per field and no preamble, extra headings or Markdown fences:
Facts: The necessary facts, with their qualifiers and any correction. Retain
supported alternatives that still answer the question. Clearly distinguish
new passage facts from prior claims when correcting or reporting a conflict.
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
Link: Follow the spouse relation to Tomas Reed, then the father relation to Orin Reed; the requested birthplace is Orin Reed's.
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
Which club has more national titles, Cedar Club or Elm Club?

Previous answers:
Step 1
Question: How many national titles does Cedar Club have?
Answer: Cedar Club won national titles in 1998, 2005, 2009 and 2012; this is a selected list.

Step 2
Question: How many national titles does Elm Club have?
Answer: Elm Club has won exactly three national titles.

Retrieval question (candidate wording):
Which club has more national titles, Cedar Club or Elm Club?

Retrieved passages:
Passage 1: Elm Club has exactly three national titles. A selected list of Cedar Club national titles includes 1998, 2005, 2009 and 2012.

Response:
Facts: Cedar Club has at least four national titles from four distinct listed years. Elm Club has exactly three.
Link: At least four is greater than three, so Cedar Club has more even without an exact total.
Candidate: Cedar Club

Example 5 (fictional):
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

Example 6 (fictional):
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

Example 7 (fictional):
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

Example 8 (fictional):
Original question:
Are both Cedar Press and Elm Press publicly owned?

Previous answers:
Step 1
Question: Is Cedar Press publicly owned?
Answer: Cedar Press is privately owned.

Step 2
Question: Is Elm Press publicly owned?
Answer: Elm Press ownership cannot be determined from the available passages.

Retrieval question (candidate wording):
Are both Cedar Press and Elm Press publicly owned?

Retrieved passages:
Passage 1: Cedar Press is privately owned and is not publicly owned.

Response:
Facts: Cedar Press is not publicly owned. Elm Press ownership is unresolved.
Link: The claim that both are publicly owned is false because Cedar Press does not satisfy it.
Candidate: No

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
field has no special authority over the supporting statements or the Original
question. Goal synthesis may carry relevant facts from a different retrieval
packet: their absence from the passages below does not itself contradict them. Check that
the chosen answer concerns the requested entity, relationship, attribute, time and
conditions. If a candidate omits a required answer supported by the other facts,
restore it; if it adds an unsupported restriction, do not adopt that restriction.
Prefer a directly stated, matching passage to an unsupported intermediate
claim. Resolve apparent conflicts by entity, relation, time and scope; if a material
conflict remains unresolved, do not invent a resolution. Do not substitute a
different person's role or a related attribute.

Return only the requested name, value, category or brief phrase. Preserve a complete
identifying name and required qualifiers without copying explanatory background.
Do not add titles, explanations or locations that are unnecessary to identify
the requested answer. Include multiple qualifying answers when the question does
not distinguish among them; do not create a ranking to force a unique choice.
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
Link: Follow the spouse relation to Tomas Reed, then the father relation to Orin Reed; the requested birthplace is Orin Reed's.
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
Passage 1: Neral Voss was a public servant from Selwick.

Final short answer:
Mayor of Selwick and ambassador to Torvale

Example 5 (fictional):
Original question:
Are both Cedar Press and Elm Press publicly owned?

Ordinary answers:
Step 1
Question: Is Cedar Press publicly owned?
Answer: Cedar Press is privately owned.

Step 2
Question: Is Elm Press publicly owned?
Answer: Elm Press ownership cannot be determined from the available passages.

Candidate synthesis:
Facts: Cedar Press is not publicly owned. Elm Press ownership is unresolved.
Link: The claim that both are publicly owned is false because Cedar Press does not satisfy it.
Candidate: No

Retrieved passages:
Passage 1: Cedar Press is privately owned and is not publicly owned.

Final short answer:
No

Example 6 (fictional):
Original question:
Which film has the older director, Willow Dawn or Stone Dawn?

Ordinary answers:
(No previous answers.)

Candidate synthesis:
Facts: Rena Moss, born in 1940, directed Willow Dawn. Dena Lake, born in 1950, directed Stone Dawn.
Link: Rena Moss was born earlier and is the older director.
Candidate: Rena Moss

Retrieved passages:
Passage 1: Rena Moss directed Willow Dawn and was born in 1940. Dena Lake directed Stone Dawn and was born in 1950.

Final short answer:
Willow Dawn

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

