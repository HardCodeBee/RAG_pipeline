# 合成边界检查清单

这些输入和预期由本次设计人工编写，用于逐条审查提示词与接口。不是模型通过率、独立测试集或最终F1。预期与禁止推断字段不得进入真实任务prompt。初稿保留于reviews/CHALLENGES.first_draft.json。除数字空格整理外，C37已补足第一份公职的明确证据，详见第五轮复核。

| ID | 阶段 | 应有行为 | 必须避免 |
|---|---|---|---|
| C01_parent_not_father | ordinary | State that the father is not established. | Infer that Sorel is the father from parent alone. |
| C02_spouse_parent | ordinary | Cora is the mother of Ada's spouse Ben. | Use Dana, the mother of Ada herself. |
| C03_generation | ordinary | Garen. | Return Fenn or Halen, or state a wrong relation around Garen. |
| C04_direction | ordinary | Kalen. | Return Iris's parent Jora. |
| C05_birth_vs_death | ordinary | Birthplace is unresolved. | Treat residence or death location as birthplace. |
| C06_producer_vs_director | goal | Correct the role and answer 1971. | Choose 1962 by trusting the rewrite or earlier answer. |
| C07_residence_not_citizenship | ordinary | Citizenship is unresolved. | Infer citizenship from residence or birthplace alone. |
| C08_wrong_search_county | ordinary | Brook, preserving the original descriptive target. | Answer Dune solely because it answers the search question. |
| C09_ambiguous_binding | rewrite | Retain the ambiguous description. | Choose Lera or Mera without a disambiguating condition. |
| C10_irrelevant_history | rewrite | Keep the question about Boren. | Replace Boren with Aven. |
| C11_known_answer_not_identity | rewrite | Who is the father of Mio? | Who is Nio? |
| C12_missing_binding | rewrite | Retain the current question. | Substitute Cannot be determined as a person name. |
| C13_missing_not_negative | ordinary | Native range is unresolved. | No, or zero native populations. |
| C14_and_false | goal | No; Arven falsifies the conjunction. | Require Birven's ownership before answering. |
| C15_or_true | goal | Yes. | Abstain because Birven is unknown. |
| C16_and_unknown | goal | Cannot be determined; Birven remains unknown. | Yes from only Arven, or No from missing information. |
| C17_incomplete_count | goal | The comparison is unresolved. | Conclude Boren wins because only two Aven titles are listed. |
| C18_lower_bound | goal | Aven; its lower bound of four exceeds three. | Abstain solely because Aven's exact total is unknown. |
| C19_no_items_not_zero | ordinary | The count is unresolved. | Zero species. |
| C20_scope_mismatch | goal | National-title comparison is unresolved. | Compare 13 all-titles with 5 national-titles as equivalent counts. |
| C21_equal_values | goal | Neither; their stated heights are equal. | Choose a winner or claim height evidence is missing. |
| C22_coarse_route | goal | Hara; no exact longitude is needed. | Treat failed precision subquestions as proof the root is unanswerable. |
| C23_map_back_to_film | final | Red Window. | Return the intermediate person Lena. |
| C24_body_beats_wrong_candidate | final | Bronze. | Use the total-capacity comparison or repeat Candidate. |
| C25_goal_only_new_fact | final | Orford, treating the synthesis as usable candidate information. | Reject the fact solely because root passages do not repeat it; call it independently verified. |
| C26_unrequested_ranking | final | Mayor of Quen and ambassador to Ruen. | Invent a prestige filter. |
| C27_truthful_partial_list | ordinary | State that Sera wrote Night Garden without claiming an exhaustive list. | Abstain despite a responsive true item, or claim it is the only novel. |
| C28_no_ordinary_steps | goal | Tera by direct evidence. | Invent missing substeps or require nonempty history. |
| C29_source_attribution | ordinary | Distinguish the prior Ura claim from the passage, which only gives display location. | State that the retrieved passage names Ura as painter. |
| C30_local_target_vs_root | ordinary | Yern answers the actual current subquestion; keep its relation explicit. | Silently shorten the current relation to answer the root instead. |
| C31_goal_original_target | goal | Xern. | Use the last subanswer Yern as the root answer. |
| C32_exclusion_zero | ordinary | Zero other species. | Treat supported zero as uncertainty. |
| C33_units | goal | Berco; Arco is 30.48 metres. | Compare 100 and 50 without conversion. |
| C34_frozen_time | ordinary | Dera in 2018. | Use the later team Era. |
| C35_literal_hash | rewrite | Preserve #9 as a title component. | Resolve #9 as a step reference without evidence. |
| C36_name_qualifier | final | Glenford-on-Mere. | Drop the required disambiguating suffix. |
| C37_new_post_not_required | final | Mayor of Quen. | Retain every post despite the explicit first condition. |
| C38_inconsistent_quantities | goal | The comparison is unresolved unless the conflict can be scoped. | Average 20 and 40 or choose a convenient source without justification. |
| C39_scope_resolved_conflict | goal | Iena. | Treat different years as an irreconcilable same-scope conflict. |
| C40_answer_already_available | goal | Use the available candidate answer Kera, without pretending a passage confirmed it. | Require a redundant new fact or invent additional biography. |
