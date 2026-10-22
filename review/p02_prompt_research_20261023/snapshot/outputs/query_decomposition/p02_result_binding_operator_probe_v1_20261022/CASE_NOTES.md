# Selected mechanism checks

These are informed post-hoc illustrations selected after inspecting outputs. They are not an exhaustive prevalence estimate. Native scores are unchanged.

## musique:3hop1__635099_131926_89261

Where does the body of water by the city where Write This Down was formed empty into the Gulf of Mexico?

All arms identify Minneapolis and the Mississippi River. GO also states that the river empties into the Gulf of Mexico at its delta, then calls the essential relation the Gulf as mouth and returns Gulf of Mexico. The reference asks for Mississippi River Delta. G0/GF return Mississippi River and score0.8 through partial overlap; GO scores0.

Neither high partial lexical overlap nor an explicit relation sentence establishes correct target selection. The requested endpoint must be distinguished from the destination already given in the question.

## musique:2hop__557411_652060

Who is the spouse of the creator of The Nanny Reunion: A Nosh to Remember?

G0 and GF return the reference name Peter Marc Jacobson. GO first states that passages confirm the spouse, immediately says that fact is not in the chunks, and then abstains. GF explicitly appeals to external knowledge rather than a supplied spouse passage.

This contrasts evidence-policy/abstention behavior and an internally inconsistent explanation. Reference-name matching alone does not validate the temporal spouse statement or its claimed citation; do not label it a clean multi-step operation repair.

## musique:2hop__289517_139339

What was the cast member in After Hours nominated for?

GF separates the actor-identifying film from the film associated with his nomination and returns Academy Award for Best Live Action Short Film. G0 and GO instead add for his role in After Hours, which the question does not require. GO names the correct nomination earlier in its paragraph but concludes no nomination for that role; final becomes Cannot be determined.

The decisive fact is present. Explicit operation demonstrations do not by themselves preserve the original question's conditions. This is the same unresolved scope mechanism observed in the preceding few-shot experiment, now appearing with different fictional examples.

## musique:2hop__582045_161450

Where is the province of Gonbad-e Qabus County located?

GF and GO both state that Golestan is in the north-east of Iran, south of the Caspian Sea, but their concluding sentence and final output drop south of the Caspian Sea. G0 retains that phrase. Native scores are0.588235,0,0.153846 for G0/GF/GO respectively.

This mixes answer-boundary and wording penalties with loss of a requested location descriptor. The useful fact survives in the body but is lost from the candidate conclusion/final; retrieving or explaining it again is not the missing step.

## 2wiki:f37818400bd911eba7f7acde48001122

Where did Elisabeth Magdalena Of Pomerania's husband die?

All arms identify Mitau. GF/GO explicitly acknowledge that their passage gives a birthplace but no death location, then infer the death location from the ruler's residence/seat of power and lack of an alternative. GO final adds now Jelgava and native F1 drops from1 to0.5.

A residence and absence of another stated place do not logically establish a death location. A reference-matching entity does not validate this explanation; the final-score difference here additionally reflects the alias phrase.

## 2wiki:923b15ba0bb011ebab90acde48001122

Who is the paternal grandfather of Araripe Júnior?

All arms return Tristao Goncalves de Alencar for the grandfather. GF/GO show the two father links and omit later Araripe from final. Native F1 rises from0.5 to0.666667, but none reaches the reference's short name exactly.

The explicit father-of-father explanation is valid in the supplied chain, while the observed F1 change is due to the returned name boundary, not a changed ancestor.

## hotpotqa:5addcfd65542997dc790703e

Which company had a ten year contract with the anti-hero from Escape From New York played by Kurt Russel? 

All three goal inputs contain Kurt Russell signing a ten-year Disney contract at reader rank9 and the actor-character link at ranks1/3. G0 and GF return Disney. GO instead insists that the fictional character cannot sign contracts, asserts no company, and final follows that interpretation.

The facts and bridge are available. An operation-style explanation can harden an unhelpfully literal interpretation of an ambiguous question. This is not a missing-number or retrieval-budget failure.

## hotpotqa:5ae4c01e55429913cc2044f3

Which Captain launched the attack which led to more casualties than any other incident in the war fought between the settlers of the nascent colony of New Netherland and the native Lenape population?

Frozen prior QA identifies a1643 Kieft-ordered massacre. The goal reader does not name Captain John Underhill; G0 nevertheless returns that reference name. GF substitutes Director-General Kieft for the requested Captain. GO adds an ungrounded explanation that such officials were called Captain. Both synthesis finals answer Kieft.

Do not call G0 fully evidence-validated merely because its final matches the reference. The synthesis failure includes missing role evidence and unsupported justification, not just a faulty explicit operator.

## hotpotqa:5ab8c3c65542991b5579effd

How many albums did the company who released Tift Merritt's debut album release for Tif Merritt?

G0 final is Two, GF and GO finals are2. All goal outputs state two albums. Native Hotpot scoring gives1 to Two and0 to2.

This native-score loss is not a changed album count. Preserve the official score while documenting numerical equivalence.

## hotpotqa:5a71474b5542994082a3e74f

What was triggered by the resignation of Conservative MP Stephen Phillips on 4 November 2016 and was when Caroline Johnson became a Member of Parliament?

All arms identify the Sleaford and North Hykeham by-election. GF/GO final adds on8December2016, lowering native F1 from1 to0.714286.

A more verbose candidate conclusion can alter final answer length without changing the identified event.

## hotpotqa:5ab55455554299488d4d993b

 Mount Stimson is the second highest peak in national park that encompasses how many acres?

G0 final is1million acres, while GF/GO retain over1million acres. The goal texts all include over. Native F1 rises from0.666667 to0.857143.

The final stage can lose or preserve a meaningful qualifier despite equivalent upstream facts. This is not evidence that an additional multi-step operation was required.

## hotpotqa:5a84d2ea5542992a431d1aa8

What career titles do Ed Wood and Fernand Rivers share?

GF includes shared writer/screenwriter alongside actor, producer and director. GO explains an intersection but rejects writer versus screenwriter unless equivalence is separately confirmed, omitting writer from final. G0 also omits it.

Explicitly naming intersection does not resolve how semantic categories align. The demonstration teaches an operation on labels; the real task also requires interpreting those labels.

## hotpotqa:5abce85755429959677d6b3e

Le Rêve was created by the director behind what Cirque du Soleil production from 1998 onward?

GF and GO say the available list does not identify a unique Cirque du Soleil production. GO even calls Franco Dragone the production in one sentence. Nevertheless all final outputs are O and score1.

Independent final reading can recover from a poor goal synthesis. Goal fluency/correctness and final-score effect must be evaluated separately.

## 2wiki:cdb8b7540bd911eba7f7acde48001122

Where was the place of burial of the director of film Broadway Bad?

All arms identify Westwood Village Memorial Park Cemetery. G0/GO include in Los Angeles in final and scoreF1=0.769231; GF omits the city and scores1.

The GF improvement is answer-boundary selection, with no changed burial entity or repaired relation chain.

## 2wiki:7b4cc520091d11ebbdaeac1f6bf848b6

Which film came out earlier, Irina Palm or Big Man On Campus?

GF and GO state Irina Palm2007 versus Big Man on Campus1989 and explicitly compare years. All three final answers identify Big Man on Campus and score1.

This is a successful synthesis example, but its explicit comparison does not add a score gain over the short-answer control.
