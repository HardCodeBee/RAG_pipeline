# B8 post-lock paired interpretation

The eight selected ordinary-QA nodes have **2 correct / 6 incorrect in every arm**, with no semantic repair or regression. The two correct nodes are the same reasonable-abstention boundaries. These are exposed development challenges, not an estimate of complete-RAG accuracy or F1.

The original 24 blinded reviews remain unchanged (SHA256 `e5468a8d8967d8cd6ba85ec2f2fbc39b446948d85ebba78315129d70cbdf0828`). All 24 joined labels match that first lock. The reviewer previously designed/reviewed I8 and these cases; the labels are not independent human gold. Full per-candidate answers, original actual question, evidence-based reasons and receipt checks are in [the audit JSON](ROOT_PAIRED_INTERPRETATION.json).

B0 exactly reuses the previously paid fresh I0 response. B1 adds a candidate-description block containing the original current question; B2 changes only that block's candidate to the source's existing rewritten query. The actual Current subquestion remains unchanged and authoritative in all arms. B0 is a historical exact reuse, not a newly synchronized B control.

| Case | Text change | Why it does not repair the task |
|---|---|---|
| 2Wiki371, Richard | B2 changes punctuation/capitalization and `was` to `is`. | The complete statement still attaches Richard II to the wrong generation of Richard's family. The actual current question asks for William's father's father; an entity-name hit does not repair the stated relation. The original decomposition already walks one generation too far relative to the root question. |
| 2Wiki506, Liu | B2 explicitly names Liu Cong instead of describing him as Empress Liu E's spouse. | It still assigns Liu Yin as Liu Cong's father. The passages instead connect Liu Yin to Liu E and Liu Yuan to Liu Cong. |
| Hotpot31, Wills | B2 omits the host-description clause. | It still includes Maury Wills as a pitcher despite the reader describing him as a shortstop/switch hitter. A true but incomplete list would be allowed; this is an incorrect extra membership. Bob Gibson's positive pitcher relation is not independently established by this reader, so necessary-information sufficiency remains a pending boundary. |
| 2Wiki248, Tsutsunava | B2 shortens the abstention. | The reader identifies the director and preserves the death endpoint `25 October 1955`; all arms still fail to extract it. The damaged birth field is not reconstructed. |
| Hotpot50, Cretan pack | B1/B2 add the supported background phrase `from Crete`. | The requested pack bridge remains absent. This is a reasonable abstention with a supported extra fact, not recovery of the missing relation. |

B0→B1 changes one answer text; B1→B2 changes four. Three B1/B2 pairs have byte-identical full payloads and reuse the same actual receipt, so they are not independent repeated samples.

The independent receipt audit regenerated and matched all 24 full payload contracts, including the unchanged 128-token cap. All 13 new unique completed payloads have nonempty `stop` responses, matching requested/returned model and three nonnegative integer token fields (excluding booleans), with total equal to prompt plus completion. Eight B0 proofs match their source database receipts and previously recorded ledger charges.

| Cost category | CNY | Interpretation |
|---|---:|---|
| New known completed cost | 0.0283508 | 13 new unique receipts |
| Retired unknown reservation | 0.0067248 | One timed-out submission; retained, not declared free |
| B0 previously paid attribution | 0.0164192 | Already charged in I0; no new B0 charge |

The 14 B namespace ledger entries are precisely 13 returned-usage charges plus the retained unknown reservation; no orphan reservation was found. The eight source B0 request IDs are absent from this new-charge namespace.

I16 likewise had 2 correct / 6 incorrect in I0 and I1, with none of the six targeted failures repaired. Together these results stop expansion of the tested generic-verification sentence and candidate-description block. They do not establish that every entity-binding strategy fails. No retrieval, decomposition, goal or final stage was rerun here; no complete-RAG benefit has been shown. Original labels, scripts and reports are preserved. This audit made no API/network calls and did not read L outputs.
