"""Synthetic review questions, kept separate from demonstrations and live inputs."""
from datetime import datetime,timezone
import json
from .design_tools import HERE


def qa(q,a,s=1):return dict(step_id=s,question=q,answer=a)
def ordinary(q,docs,*,original=None,history=None,search=None):
    return dict(original_question=original or q,history=history or [],current_question=q,retrieval_question=search or q,passages=docs)
def goal(q,docs,*,history=None,search=None):
    return dict(original_question=q,history=history or [],retrieval_question=search or q,passages=docs)
def final(q,docs,facts,link,candidate,history=None):
    return dict(original_question=q,history=history or [],synthesis=dict(Facts=facts,Link=link,Candidate=candidate),passages=docs)


def main():
    cases=[]
    def add(key,stage,data,expected,forbidden,reason):
        cases.append(dict(id=key,stage=stage,input=data,expected_behavior=expected,
                          forbidden_inference=forbidden,reason=reason,source='Authored fictional scenario; not a model result'))
    add('C01_parent_not_father','ordinary',ordinary('Who is Iven\'s father?',['Sorel is a parent of Iven.']),
        'State that the father is not established.','Infer that Sorel is the father from parent alone.','Relation specificity')
    add('C02_spouse_parent','ordinary',ordinary('Who is the mother of Ada\'s spouse?',
        ['Ada is married to Ben. Ben\'s mother is Cora. Ada\'s mother is Dana.'],history=[qa('Who is Ada\'s spouse?','Ada\'s spouse is Ben.')]),
        'Cora is the mother of Ada\'s spouse Ben.','Use Dana, the mother of Ada herself.','Entity binding')
    add('C03_generation','ordinary',ordinary('Who is the father of the father of Evin?',
        ['Evin\'s father is Fenn. Fenn\'s father is Garen. Garen\'s father is Halen.']),
        'Garen.','Return Fenn or Halen, or state a wrong relation around Garen.','Depth and whole-statement correctness')
    add('C04_direction','ordinary',ordinary('Who is the child of Iris?',
        ['Iris is the child of Jora. Kalen is the child of Iris.']),
        'Kalen.','Return Iris\'s parent Jora.','Directed relation')
    add('C05_birth_vs_death','ordinary',ordinary('Where was Niven born?',
        ['Niven lived in Oakford and died in Pineford.']),
        'Birthplace is unresolved.','Treat residence or death location as birthplace.','Wrong attribute inference')
    add('C06_producer_vs_director','goal',goal('When was the director of Moon Ledger born?',
        ['Moon Ledger was directed by Rena and produced by Sera. Rena was born in 1971. Sera was born in 1962.'],
        history=[qa('Who directed Moon Ledger?','Sera directed Moon Ledger.')],search='When was Sera born?'),
        'Correct the role and answer1971.','Choose1962 by trusting the rewrite or earlier answer.','Fallible intermediate identity')
    add('C07_residence_not_citizenship','ordinary',ordinary('What is Tavin\'s citizenship?',
        ['Tavin resides in Uland and was born in Valen.']),
        'Citizenship is unresolved.','Infer citizenship from residence or birthplace alone.','Attribute and scope')
    add('C08_wrong_search_county','ordinary',ordinary('What is the seat of the county containing Wren?',
        ['Wren is in Alder County. Alder County\'s seat is Brook. Cedar County\'s seat is Dune.'],
        history=[qa('Which county contains Wren?','Wren is in Alder County.')],search='What is the seat of Cedar County?'),
        'Brook, preserving the original descriptive target.','Answer Dune solely because it answers the search question.','Retrieval wording is not the task')
    add('C09_ambiguous_binding','rewrite',dict(history=[qa('Who wrote Dusk?','Two books titled Dusk were written by Lera and Mera, respectively.')],current_question='Where was the author of Dusk born?'),
        'Retain the ambiguous description.','Choose Lera or Mera without a disambiguating condition.','No invented identity')
    add('C10_irrelevant_history','rewrite',dict(history=[qa('How many species are in genus Aven?','Aven has seven species.')],current_question='How many species are in genus Boren?'),
        'Keep the question about Boren.','Replace Boren with Aven.','Independent current task')
    add('C11_known_answer_not_identity','rewrite',dict(history=[qa('Who is Lia\'s spouse?','Lia\'s spouse is Mio.'),qa('Who is Mio\'s father?','Mio\'s father is Nio.',2)],current_question='Who is the father of Lia\'s spouse?'),
        'Who is the father of Mio?','Who is Nio?','Keep the requested relation even if an answer is already known')
    add('C12_missing_binding','rewrite',dict(history=[qa('Who directed Pale Road?','The director cannot be determined from these passages.')],current_question='When was the director of Pale Road born?'),
        'Retain the current question.','Substitute Cannot be determined as a person name.','Unknown is not an entity')
    add('C13_missing_not_negative','ordinary',ordinary('Is Cedar Fox native to Vela?',
        ['Cedar Fox has reddish fur.']),
        'Native range is unresolved.','No, or zero native populations.','Absence of evidence')
    add('C14_and_false','goal',goal('Are both Arven and Birven publicly owned?',
        ['Arven is not publicly owned. Birven\'s ownership is not documented.']),
        'No; Arven falsifies the conjunction.','Require Birven\'s ownership before answering.','Decisive Boolean condition')
    add('C15_or_true','goal',goal('Is at least one of Arven and Birven publicly owned?',
        ['Arven is publicly owned. Birven\'s ownership is not documented.']),
        'Yes.','Abstain because Birven is unknown.','Decisive Boolean condition')
    add('C16_and_unknown','goal',goal('Are both Arven and Birven publicly owned?',
        ['Arven is publicly owned. Birven\'s ownership is not documented.']),
        'Cannot be determined; Birven remains unknown.','Yes from only Arven, or No from missing information.','Three-valued conjunction')
    add('C17_incomplete_count','goal',goal('Which has more national titles, Aven or Boren?',
        ['Selected national-title years for Aven are1991and1998. Boren has exactly three national titles.']),
        'The comparison is unresolved.','Conclude Boren wins because only two Aven titles are listed.','Partial list is not a total')
    add('C18_lower_bound','goal',goal('Which has more national titles, Aven or Boren?',
        ['Selected distinct national-title years for Aven are1991,1998,2001and2007. Boren has exactly three national titles.']),
        'Aven; its lower bound of four exceeds three.','Abstain solely because Aven\'s exact total is unknown.','Sufficient bound')
    add('C19_no_items_not_zero','ordinary',ordinary('How many species does genus Civen contain?',
        ['Civen has large flowers; no species are enumerated in this excerpt.']),
        'The count is unresolved.','Zero species.','Missing list versus empty set')
    add('C20_scope_mismatch','goal',goal('Which club has more national titles, Daren or Eren?',
        ['Daren has thirteen titles across all competitions. Eren has five national titles.']),
        'National-title comparison is unresolved.','Compare13all-titles with5national-titles as equivalent counts.','Comparable units and scope')
    add('C21_equal_values','goal',goal('Which tower is taller, Fera or Gera?',
        ['Fera is exactly80metres tall. Gera is exactly80metres tall.']),
        'Neither; their stated heights are equal.','Choose a winner or claim height evidence is missing.','Equality is a supported outcome')
    add('C22_coarse_route','goal',goal('Which canal is farther east, Hara or Iara?',
        ['Hara lies entirely in Jara. Iara lies entirely in Kara. Jara lies entirely east of Kara.'],
        history=[qa('What is Hara\'s longitude?','Its exact longitude is unavailable.'),qa('What is Iara\'s longitude?','Its exact longitude is unavailable.',2)]),
        'Hara; no exact longitude is needed.','Treat failed precision subquestions as proof the root is unanswerable.','Alternative sufficient route')
    add('C23_map_back_to_film','final',final('Which film has the older director, Red Window or Blue Window?',
        ['Lena directed Red Window and was born in1940. Mina directed Blue Window and was born in1950.'],
        'Lena is older than Mina. Lena directed Red Window.','The older director is Lena.','Lena'),
        'Red Window.','Return the intermediate person Lena.','Answer type and mapping')
    add('C24_body_beats_wrong_candidate','final',final('Which hall has more seats, Amber or Bronze?',
        ['Amber has900seats and1200totalcapacity. Bronze has1000seats and1100totalcapacity.'],
        'Amber has900seats; Bronze has1000seats.','Amber has higher total capacity.','Amber'),
        'Bronze.','Use the total-capacity comparison or repeat Candidate.','Final independent attribute check')
    add('C25_goal_only_new_fact','final',final('Where was the director of Paper Sun born?',
        ['Paper Sun is a film directed by Nera.'],
        'The goal retrieval identified Nera\'s birthplace as Orford.','Nera is the requested director.','Orford'),
        'Orford, treating the synthesis as usable candidate information.','Reject the fact solely because root passages do not repeat it; call it independently verified.','Do not erase goal supplemental retrieval')
    add('C26_unrequested_ranking','final',final('Which public posts did Pera hold?',
        ['Pera worked in public service.'],
        'Pera served as mayor of Quen and ambassador to Ruen.','Ambassador is more prestigious.','Ambassador to Ruen'),
        'Mayor of Quen and ambassador to Ruen.','Invent a prestige filter.','Retain relevant alternatives')
    add('C27_truthful_partial_list','ordinary',ordinary('Which novels did Sera write?',
        ['Sera wrote Night Garden. This excerpt discusses only one of her novels.']),
        'State that Sera wrote Night Garden without claiming an exhaustive list.','Abstain despite a responsive true item, or claim it is the only novel.','Correct but partial answer')
    add('C28_no_ordinary_steps','goal',goal('Who designed the Lumen clock?',
        ['Tera designed the Lumen clock.']),
        'Tera by direct evidence.','Invent missing substeps or require nonempty history.','Valid empty decomposition route')
    add('C29_source_attribution','ordinary',ordinary('Who painted Distant Shore?',
        ['Distant Shore is displayed in the Vale museum.'],history=[qa('Who painted Distant Shore?','Ura painted Distant Shore.')]),
        'Distinguish the prior Ura claim from the passage, which only gives display location.','State that the retrieved passage names Ura as painter.','Do not fabricate corroboration')
    add('C30_local_target_vs_root','ordinary',ordinary('Who is the father of the father of the father of Venn?',
        ['Venn\'s father is Wern. Wern\'s father is Xern. Xern\'s father is Yern.'],original='Who is Venn\'s grandfather?'),
        'Yern answers the actual current subquestion; keep its relation explicit.','Silently shorten the current relation to answer the root instead.','Local execution is not decomposition repair')
    add('C31_goal_original_target','goal',goal('Who is Venn\'s grandfather?',
        ['Venn\'s father is Wern. Wern\'s father is Xern. Xern\'s father is Yern.'],
        history=[qa('Who is Venn\'s great-grandfather?','Venn\'s great-grandfather is Yern.')]),
        'Xern.','Use the last subanswer Yern as the root answer.','Goal answers original task')
    add('C32_exclusion_zero','ordinary',ordinary('How many members of genus Zera are there other than Zera alba?',
        ['Zera has exactly one member species, Zera alba.']),
        'Zero other species.','Treat supported zero as uncertainty.','Supported exclusion and zero')
    add('C33_units','goal',goal('Which pole is taller, Arco or Berco?',
        ['Arco is100feet tall. Berco is50metres tall. One foot equals0.3048metres.']),
        'Berco; Arco is30.48metres.','Compare100and50without conversion.','Units before comparison')
    add('C34_frozen_time','ordinary',ordinary('Which team did Cera play for in2018?',
        ['Cera played for Dera in2018 and joined Era in2020.'],search='Which team does Cera play for?'),
        'Dera in2018.','Use the later team Era.','Requested temporal scope')
    add('C35_literal_hash','rewrite',dict(history=[qa('Who wrote First Book?','Fera wrote First Book.')],current_question='Who wrote the novel titled #9?'),
        'Preserve #9 as a title component.','Resolve #9 as a step reference without evidence.','Symbols are contextual')
    add('C36_name_qualifier','final',final('Which of the two towns named Glenford is in North Province?',
        ['Glenford-on-Mere is in North Province. Glenford-on-Hill is in South Province.'],
        'Glenford-on-Mere is the town in North Province.','The suffix distinguishes the two towns.','Glenford'),
        'Glenford-on-Mere.','Drop the required disambiguating suffix.','Short yet identifying answer')
    add('C37_new_post_not_required','final',final('What was Pera\'s first public post?',
        ['Pera became mayor of Quen in1980 and ambassador to Ruen in1990.'],
        'Pera held both posts.','Both are relevant public positions.','Mayor of Quen and ambassador to Ruen'),
        'Mayor of Quen.','Retain every post despite the explicit first condition.','Preservation must obey scope')
    add('C38_inconsistent_quantities','goal',goal('Which basin is larger, Hena or Iena?',
        ['Two undated sources put Hena\'s area at20or40square kilometres without explaining the difference. Iena\'s area is30square kilometres.']),
        'The comparison is unresolved unless the conflict can be scoped.','Average20and40orchoosea convenient source without justification.','Unresolved conflict')
    add('C39_scope_resolved_conflict','goal',goal('Which basin had a larger area in2010, Hena or Iena?',
        ['Hena had20square kilometres in2010 and40in2020. Iena had30square kilometres in2010.']),
        'Iena.','Treat different years as an irreconcilable same-scope conflict.','Resolve a seeming contradiction by scope')
    add('C40_answer_already_available','goal',goal('Who founded Jera College?',
        [],history=[qa('Who founded Jera College?','Kera founded Jera College.')]),
        'Use the available candidate answer Kera, without pretending a passage confirmed it.','Require a redundant new fact or invent additional biography.','Prior facts remain usable')
    path=HERE/'CHALLENGES.local.json';assert not path.exists()
    path.write_text(json.dumps({'status':'Authored design challenge bank; zero model executions','cases':cases},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# 合成边界检查清单','',
           '这些输入和预期由本次设计人工编写，用于逐条审查提示词与接口。不是模型通过率、独立测试集或最终F1。预期与禁止推断字段不得进入真实任务prompt。','',
           '| ID | 阶段 | 应有行为 | 必须避免 |','|---|---|---|---|']
    for x in cases:lines.append('| '+x['id']+' | '+x['stage']+' | '+x['expected_behavior']+' | '+x['forbidden_inference']+' |')
    (HERE/'CHALLENGE_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (HERE/'WORKLOG.jsonl').open('a',encoding='utf8') as h:h.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage='authored_challenges',count=len(cases),model_executions=0),ensure_ascii=False)+'\n')
    print('Authored',len(cases),'design checks; model executions0')


if __name__=='__main__':main()
