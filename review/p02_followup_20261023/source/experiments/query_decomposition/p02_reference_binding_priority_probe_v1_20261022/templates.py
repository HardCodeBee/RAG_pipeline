"""Independent, pure single-factor ordinary-QA prompt interventions.

Candidate span mapping is mechanical string evidence, not entity recognition
or a semantic guarantee. No facts, gold, reader text or current model answers
are consulted when constructing the mapping.
"""
from difflib import SequenceMatcher
import json
import re

ARMS=("C0","S1","B1","P1")
SCOPE="The original question supplies scope; answer only the current subquestion."
BINDING_HEADER="Candidate span bindings from prior answers (not verified):"
TOKEN_PATTERN=re.compile(r"\w+|[^\w\s]",re.UNICODE)
REFERENCE=re.compile(r"#(?:\d+|N)\b")
FUNCTION_WORDS=frozenset("""a an the this that these those who whom whose what which when where why how
am is are was were be been being do does did have has had can could may might must shall should will would
of to in on at for from by with without as into over under between among through after before during
and or nor but if then than whether both either neither each every any all some such other same
i me my mine we us our ours you your yours he him his she her hers it its they them their theirs
no not none nothing cannot insufficient unknown undetermined yes also only respectively
""".split())


def tokens(text):
    return [dict(text=match.group(),start=match.start(),end=match.end())
            for match in TOKEN_PATTERN.finditer(text)]


def substantive(text):
    return any(any(char.isalnum()for char in token["text"])
               and token["text"].casefold()not in FUNCTION_WORDS for token in tokens(text))


def named_or_numeric_looking(text):
    """Frozen lexical screen, not NER: substantive capitals or any digit.

    Capitalization alone does not verify an entity or requested relation. Keep
    these limitations instead of hand-repairing a mapper to fit known cases.
    """
    for token in tokens(text):
        word=token["text"]
        if word.casefold()in FUNCTION_WORDS:continue
        if any(char.isdigit()for char in word)or(word and word[0].isupper()):return True
    return False


def candidate_bindings(item):
    raw=item["current_question"];rewrite=item["accepted_rewrite"]
    left,right=tokens(raw),tokens(rewrite)
    opcodes=SequenceMatcher(None,[row["text"].casefold()for row in left],
        [row["text"].casefold()for row in right],autojunk=False).get_opcodes()
    different=[opcode for opcode in opcodes if opcode[0]!="equal"]
    if not different:return []
    # One contiguous substitution window envelopes EVERY changed opcode and
    # intervening equal tokens, preserving the common outer question frame.
    _,a0,_,b0,_=different[0]
    _,_,a1,_,b1=different[-1]
    if a0==a1 or b0==b1:return []
    history=sorted(item["history"],key=lambda row:row["step_id"])
    assert len({row["step_id"]for row in history})==len(history)
    assert all(type(row["step_id"])is int and 0<row["step_id"]<item["step_id"]for row in history)

    def supports_for(start,stop):
        replacement=rewrite[right[start]["start"]:right[stop-1]["end"]]
        if not substantive(replacement):return []
        first=right[start]["text"].casefold()
        if first in FUNCTION_WORDS and first not in {"a","an","the"}:return []
        wanted=[row["text"].casefold()for row in right[start:stop]]
        supports=[]
        for prior in history:
            answer=prior["answer"];answer_tokens=tokens(answer);width=len(wanted)
            for index in range(len(answer_tokens)-width+1):
                part=answer_tokens[index:index+width]
                if [row["text"].casefold()for row in part]!=wanted:continue
                begin,end=part[0]["start"],part[-1]["end"]
                copied=answer[begin:end]
                if copied.casefold()!=replacement.casefold()or not named_or_numeric_looking(copied):continue
                supports.append(dict(step_id=prior["step_id"],answer_span=[begin,end],answer_text=copied))
        return supports

    supports=supports_for(b0,b1)
    if not supports:return []
    original_window=dict(raw_span=[left[a0]["start"],left[a1-1]["end"]],
        rewrite_span=[right[b0]["start"],right[b1-1]["end"]])
    extended=0
    while a0>0 and b0>0:
        previous_left,previous_right=left[a0-1]["text"],right[b0-1]["text"]
        if(previous_left.casefold()!=previous_right.casefold()
            or previous_left.casefold()in FUNCTION_WORDS
            or not previous_left[0].isupper()or not previous_right[0].isupper()):break
        candidate=supports_for(b0-1,b1)
        if not candidate:break
        a0-=1;b0-=1;extended+=1;supports=candidate
    lo,hi=left[a0]["start"],left[a1-1]["end"]
    rlo,rhi=right[b0]["start"],right[b1-1]["end"]
    if not substantive(raw[lo:hi]):return []
    return [dict(raw_span=[lo,hi],raw_text=raw[lo:hi],rewrite_span=[rlo,rhi],rewrite_text=rewrite[rlo:rhi],
        candidate_text=supports[0]["answer_text"],supporting_step_ids=sorted({row["step_id"]for row in supports}),
        supports=supports,original_difference_window=original_window,shared_named_prefix_tokens_extended=extended,
        match_policy="single_all_non_equal_opcode_window_casefold_tokens_autojunkFalse_contiguous_answer_copy",
        limitations="Candidate lexical substitution only; capitalization is not NER and no semantic role correctness is guaranteed")]


def bindings(item):
    """Stable pure entry point shared with a later separately authorized flow."""
    return candidate_bindings(item)


def binding_block(bindings):
    return BINDING_HEADER+"\n"+"\n".join(
        "- "+json.dumps(row["raw_text"],ensure_ascii=False)+" -> "+json.dumps(row["candidate_text"],ensure_ascii=False)
        +" (prior answer "+", ".join("#"+str(sid)for sid in row["supporting_step_ids"])+")."
        for row in bindings)


def parts(item):
    original=item["control_prompt"]
    assert item["role"]=="ordinary"and original.count("\n\nActual task:\n")==1
    prefix,actual=original.split("\n\nActual task:\n",1)
    marker=f'Current subquestion #{item["step_id"]}:\n{item["current_question"]}\n\nRetrieved passages:\n'
    assert actual.count(marker)==1
    return prefix,actual,marker


def prompt(item,arm):
    if arm not in ARMS:raise ValueError(arm)
    prefix,actual,marker=parts(item)
    if arm=="C0":return item["control_prompt"]
    if arm=="S1":
        # Whitelist the resolver and example QUESTION fields. Example reader
        # and answers remain verbatim even if their text contains a # token.
        main,demos=prefix.split("Worked examples:\n\n",1)
        assert main.count("Resolve each #N")==1
        main=main.replace("Resolve each #N","Resolve each @N",1)
        fragments=re.split(r"(?m)(^Example \d+\n)",demos)
        for index,fragment in enumerate(fragments):
            if not fragment or re.fullmatch(r"Example \d+\n",fragment):continue
            assert fragment.count("\n\nRetrieved passages:\n")==1
            questions,rest=fragment.split("\n\nRetrieved passages:\n",1)
            # Example prior-answer strings are also immutable. Structural
            # headers and question values can carry the symbolic references.
            lines=questions.splitlines(keepends=True);rename_next=False;renamed=[]
            for line in lines:
                question_line=rename_next or bool(re.match(r"^#\d+ Question:",line))
                header=bool(re.match(r"^Current subquestion #\d+:\s*$",line))
                if question_line or header:
                    line=REFERENCE.sub(lambda match:"@"+match.group()[1:],line)
                renamed.append(line)
                rename_next=line.rstrip("\r\n")=="Original question:"or bool(re.match(r"^Current subquestion @\d+:\s*$",line))
            fragments[index]="".join(renamed)+"\n\nRetrieved passages:\n"+rest
        prefix=main+"Worked examples:\n\n"+"".join(fragments)
        before,current_and_reader=actual.split(marker,1)
        history=sorted(item["history"],key=lambda row:row["step_id"])
        prior="\n\n".join(f'#{row["step_id"]} Question: {row["request"]}\nAnswer: {row["answer"]}'for row in history)if history else"No prior question-answer pairs are required."
        newprior="\n\n".join(f'@{row["step_id"]} Question: {row["request"]}\nAnswer: {row["answer"]}'for row in history)if history else prior
        assert before.endswith(prior+"\n\n")
        before=before[:-len(prior+"\n\n")]+newprior+"\n\n"
        renamed=marker.replace(f"Current subquestion #{item['step_id']}:",f"Current subquestion @{item['step_id']}:",1)
        return prefix+"\n\nActual task:\n"+before+renamed+current_and_reader
    if arm=="P1":
        assert prefix.count(SCOPE)==1 and actual.count(SCOPE)==0
        # Keep the original sentence and every other original byte. Only its
        # location changes; no simultaneous header, task or example rewriting.
        prefix=prefix.replace(SCOPE+"\n","",1)
        return prefix+"\n\nActual task:\n"+actual.replace(marker,SCOPE+"\n"+marker,1)
    bindings=candidate_bindings(item)
    if not bindings:return item["control_prompt"]
    return prefix+"\n\nActual task:\n"+actual.replace(marker,binding_block(bindings)+"\n\n"+marker,1)


def needs_formatter(answer):return len(answer.strip().split())<4


def formatter_prompt(item,answer):
    context="\n\n".join(doc["block"]for doc in item["reader_packet"]["docs"])
    return ("Express the candidate answer below as a concise complete factual sentence that names its subject and relation. "
        "Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. "
        "If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n"
        +"Question:\n"+item["current_question"]+"\n\nCandidate answer:\n"+answer+"\n\nContext:\n"+context)
