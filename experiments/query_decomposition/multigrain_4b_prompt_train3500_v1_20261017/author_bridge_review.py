"""Explicitly authored rare-form corrections after all175 query/span reviews."""
from training_sources import HERE,write
spec={
3092:('country','Naalu Pennungal',"Can'T Complain"),
3106:('death',"Who'S Your Brother?","The Night Won'T Talk"),
3118:('birth','Our Crazy Aunts','A Little Sex'),
3142:('country',"Er Kann'S Nicht Lassen",'Barbed Wire (1952 Film)'),
3143:('death','The Gal Who Took the West','Twenty Plus Two'),
3148:('birth','Smiley Gets A Gun','Frankenstein And The Monster From Hell'),
3159:('birth','The Legacy Of Pretoria',"My Wife'S Enemy"),
3170:('death','Down Missouri Way','The Hands Of Orlac (1924 Film)'),
3175:('death','Innalenkil Nale','Amores De Ayer'),
3177:('birth','Days of Being Wild','A Cool, Dry Place'),
3186:('country','Sweet Sixteen (1983 Film)','Time Freak (2018 Film)'),
3187:('country','Leviathan (1989 film)','Jinx (1919 film)'),
3199:('country','Make Haste to Live','Pinocchio (1957 TV program)'),
3201:('nationality','A Caixa','Kab? Kyoon? Aur Kahan?'),
3204:('producer','Some Kind of Wonderful (film)','Guns of El Chupacabra'),
3211:('country','The Man from Montana','With Love... from the Age of Reason'),
3220:('birth','A Song For Tomorrow','The Belgian'),
3223:('country','Velvet Fingers','Bitnoon')
}
revisions={}
for i,(op,a,b) in spec.items():
 role='producer' if op=='producer' else 'director'
 target=[f'Who is the {role} of the film {a}?',f'Who is the {role} of the film {b}?']
 for title in [a,b]:
  s=f'the {role} of the film {title}'
  target.append(f'When was {s} born?' if op=='birth' else f'When did {s} die?' if op=='death' else f'What is the nationality of {s}?' if op=='nationality' else f'What country is {s} from?')
 revisions[str(i)]=target
write(HERE/'training_reviews/new2wiki_3065_3239.json',{'reviewed_range':[3065,3239],'reviewer':'Current Codex inspected all175 original queries, exact subject/title spans and property assignments, plus deterministic question-rendering code; template/span review, same-context nonblind.','revisions':revisions,'quarantine':{},'note':'Explicit rare-form corrections preserve producer versus director, younger/older versus death dates, title punctuation and full identifiers. Authored correction specs are in author_bridge_review.py.'})
print('Wrote175 reviewed bridge cases;18 authored corrections.')
