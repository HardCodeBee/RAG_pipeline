"""Author synthetic demonstration fixtures. No model, retriever or API imports."""
from pathlib import Path
from copy import deepcopy
import json

HERE=Path(__file__).resolve().parent


def qa(step,question,answer):
    return dict(step_id=step,question=question,answer=answer)


def item(key,original,current,history,passages,retrieval=None):
    return dict(id=key,original_question=original,current_question=current,
                retrieval_question=retrieval or current,history=history,passages=passages)


def main():
    examples={k:[] for k in ('rewrite','ordinary','goal','final')}
    original='Where was the father of the spouse of Liora Vale born?'
    hist=[qa(1,'Who is the spouse of Liora Vale?','The spouse of Liora Vale is Tomas Reed.')]
    relation=item('family_role',original,'Who is the father of the spouse of Liora Vale?',hist,
        ['Tomas Reed is the child of Orin Reed. Liora Vale is the child of Marek Vale.'],
        'Who is the father of Tomas Reed?')
    x=deepcopy(relation);x['output']='Who is the father of Tomas Reed?';examples['rewrite'].append(x)
    x=deepcopy(relation);x['output']=dict(Support='The spouse is Tomas Reed. The passage identifies Orin Reed as Tomas Reed\'s parent; Marek Vale is Liora\'s parent.',Answer='The father of Liora Vale\'s spouse, Tomas Reed, is Orin Reed.');examples['ordinary'].append(x)
    goal=item('family_role',original,original,hist+[qa(2,relation['current_question'],x['output']['Answer'])],
        ['Orin Reed was born in Fenwick. Marek Vale was born in Lydon.'])
    goal['output']=dict(Facts='Liora Vale\'s spouse is Tomas Reed; his father is Orin Reed. Orin Reed was born in Fenwick.',Link='The requested birthplace belongs to Orin Reed, the spouse\'s father, not to Liora\'s father Marek Vale.',Candidate='Fenwick')
    examples['goal'].append(goal)
    final=deepcopy(goal);final['synthesis']=deepcopy(goal['output']);final['output']='Fenwick';examples['final'].append(final)

    original='Which hall, Aster Hall or Birch Hall, has more seats?'
    hist=[qa(1,'How many seats does Aster Hall have?','Aster Hall has 6,000 seats.')]
    hall=item('seats',original,'How many seats does Birch Hall have?',hist,
        ['Aster Hall has a total capacity of 10,000 people and 6,000 seats. Birch Hall has a total capacity of 8,000 people and 7,000 seats.'])
    x=deepcopy(hall);x['output']=hall['current_question'];examples['rewrite'].append(x)
    x=deepcopy(hall);x['output']=dict(Support='The current question asks for Birch Hall\'s seats. The passage distinguishes its 7,000 seats from its 8,000-person total capacity.',Answer='Birch Hall has 7,000 seats.');examples['ordinary'].append(x)
    goal=item('seats',original,original,hist+[qa(2,hall['current_question'],x['output']['Answer'])],hall['passages'])
    goal['output']=dict(Facts='Aster Hall has 6,000 seats; Birch Hall has 7,000 seats. Their total capacities are different attributes.',Link='7,000 seats is more than 6,000 seats, so Birch Hall has more seats.',Candidate='Birch Hall');examples['goal'].append(goal)
    final=deepcopy(goal);final['id']='seats_bad_candidate';final['synthesis']=dict(Facts=goal['output']['Facts'],Link='Aster Hall has a larger total capacity.',Candidate='Aster Hall');final['output']='Birch Hall';final['perturbation']='Deliberately wrong candidate synthesis for final-only correction demonstration; not presented as a valid goal output.';examples['final'].append(final)

    original='Which club has more national titles, Cedar Club or Elm Club?'
    hist=[qa(1,'How many national titles does Cedar Club have?','Cedar Club won national titles in 1998 and 2005; these are selected examples, so its total is not established.')]
    club=item('incomplete_total',original,'How many national titles does Elm Club have?',hist,
        ['Elm Club has won exactly three national titles. A selected list of Cedar Club national titles includes 1998 and 2005.'])
    x=deepcopy(club);x['output']=dict(Support='The passage explicitly gives Elm Club\'s total as three. Cedar Club\'s selected list does not establish its total.',Answer='Elm Club has won exactly three national titles.');examples['ordinary'].append(x)
    goal=item('incomplete_total',original,original,hist+[qa(2,club['current_question'],x['output']['Answer'])],club['passages'])
    goal['output']=dict(Facts='Elm Club has exactly three national titles. Cedar Club has at least two, but the listed years are not an exhaustive total.',Link='Cedar Club\'s total could be below, equal to or above three; the comparison is unresolved.',Candidate='Cannot be determined; Cedar Club\'s national-title total is missing.');examples['goal'].append(goal)
    final=deepcopy(goal);final['synthesis']=deepcopy(goal['output']);final['output']='Cannot be determined';examples['final'].append(final)

    original='Which canal lies farther east, the Eastmere Canal or the Westmere Canal?'
    hist=[qa(1,'What is the longitude of the Eastmere Canal?','The exact longitude of the Eastmere Canal is not available in these passages.'),qa(2,'What is the longitude of the Westmere Canal?','The exact longitude of the Westmere Canal is not available in these passages.')]
    goal=item('alternative_route',original,original,hist,['The Eastmere Canal is entirely in Auraland. The Westmere Canal is entirely in Boraland. All of Auraland lies east of Boraland.'])
    goal['output']=dict(Facts='The Eastmere Canal is in Auraland and the Westmere Canal is in Boraland. All of Auraland lies east of Boraland.',Link='These relative locations establish the east-west order without exact longitudes.',Candidate='Eastmere Canal');examples['goal'].append(goal)

    original='When was the director of Silver Map born?'
    hist=[qa(1,'Who directed Silver Map?','Mira Holt directed Silver Map.')]
    correction=item('correct_prior',original,'Who directed Silver Map?',hist,
        ['Silver Map was directed by Len Ames and produced by Mira Holt. Len Ames was born in 1960; Mira Holt was born in 1950.'])
    x=deepcopy(correction);x['output']=dict(Support='The passage names Len Ames as director and Mira Holt as producer, correcting the earlier director claim.',Answer='Len Ames directed Silver Map; Mira Holt produced it.');examples['ordinary'].append(x)
    goal=item('correct_prior',original,original,hist,correction['passages'],'When was Mira Holt born?')
    goal['output']=dict(Facts='The passage identifies Len Ames as Silver Map\'s director and gives his birth year as 1960. Mira Holt is its producer, correcting the earlier answer.',Link='The original question requests the director\'s birth year, so Len Ames\'s date applies despite the retrieval wording naming Mira Holt.',Candidate='1960');examples['goal'].append(goal)

    original='Which public posts did Neral Voss hold?'
    goal=item('multiple_posts',original,original,[],['Neral Voss was mayor of Selwick and later ambassador to Torvale.'])
    goal['output']=dict(Facts='Neral Voss served as mayor of Selwick and ambassador to Torvale.',Link='Both are public posts held by the named person; the question does not ask for a single highest-ranking post.',Candidate='Mayor of Selwick and ambassador to Torvale');examples['goal'].append(goal)
    final=deepcopy(goal);final['id']='multiple_posts_bad_candidate';final['synthesis']=dict(Facts=goal['output']['Facts'],Link='Ambassador is the most prestigious post.',Candidate='Ambassador to Torvale');final['output']='Mayor of Selwick and ambassador to Torvale';final['perturbation']='Final-only demonstration of an unrequested ranking restriction.';examples['final'].append(final)

    original='Where was the author of Quiet Harbor born?'
    hist=[qa(1,'Who wrote Quiet Harbor?','Seren Pike wrote Quiet Harbor.')]
    birthplace=item('missing_relation',original,'Where was the author of Quiet Harbor born?',hist,
        ['Seren Pike lived in Riverhaven for twenty years and died in Bayford.'],'Where was Seren Pike born?')
    x=deepcopy(birthplace);x['output']=dict(Support='The passage gives residence and death location, neither of which establishes birthplace.',Answer='Seren Pike\'s birthplace cannot be determined from the available information.');examples['ordinary'].append(x)

    x=item('missing_binding','Where was the director of Cedar Light born?','Where was the director of Cedar Light born?',
        [qa(1,'Who directed Cedar Light?','The director of Cedar Light cannot be determined from these passages.')],[])
    x['output']=x['current_question'];examples['rewrite'].append(x)
    x=item('keep_question','Who is the father of the spouse of Liora Vale?','Who is the father of the spouse of Liora Vale?',
        [qa(1,'Who is the spouse of Liora Vale?','Liora Vale\'s spouse is Tomas Reed.'),qa(2,'Who is Tomas Reed\'s father?','Tomas Reed\'s father is Orin Reed.')],[])
    x['output']='Who is the father of Tomas Reed?';examples['rewrite'].append(x)

    doc={'authorship':'Synthetic instructional fixtures authored for this design; no API/model outputs or benchmark cases. Names and facts are fictional.',
         'stages':examples}
    path=HERE/'revisions/r01/examples.json'
    assert not path.exists();path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print({k:len(v) for k,v in examples.items()})


if __name__=='__main__':main()
