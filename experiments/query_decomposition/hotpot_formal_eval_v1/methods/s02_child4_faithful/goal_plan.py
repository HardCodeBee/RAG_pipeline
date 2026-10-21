"""Copied pure frozen goal decision rules; no historical database reader."""
import re

MAX_QUERY_UTF8_BYTES = 512

MAX_QUERY_WORDS = 64

MAX_ENUM_ANSWER_UTF8_BYTES = 256

MAX_ENUM_ANSWER_WORDS = 32

_REF = re.compile(r"#([1-9][0-9]*)\b")

_MISSING = re.compile(
    r"^(?:none\b|null\b|unknown\b|n/?a\b|not\s+(?:specified|provided|known|available|"
    r"mentioned|stated|applicable|determined)\b|(?:i\s+)?(?:cannot|can't|could not|unable to)\s+"
    r"(?:determine|answer|find|identify|be determined)\b|i\s+(?:do not|don't)\s+know\b|"
    r"insufficient\s+(?:information|context|evidence)\b)", re.I,
)

_MISSING_IN_CONTEXT = re.compile(
    r"\b(?:no|not)\b.*\b(?:provided|specified|mentioned|available|found|given|"
    r"included)\b.*\b(?:context|chunks?|documents?|information|evidence)\b|"
    r"\bno\b.*\b(?:information|evidence)\b|"
    r"\b(?:context|chunks?|documents?)\b.*\b(?:does not|doesn't|do not|don't)\s+"
    r"(?:specify|provide|mention|include|state|identify)\b", re.I,
)

_ENUM = re.compile(
    r"^(?:what\s+(?:types?|kinds?|categories)\s+of\b.*\bare\s+there\b|"
    r"(?:types?|kinds?|categories)\s+of\b|who\s+are\s+(?:the\s+)?(?:actors|people)\b)", re.I,
)

_COUNT = re.compile(r"\b(?:how many|number of|count of)\b|^count\b", re.I)

def _refs(request):
    matches = list(_REF.finditer(request))
    if request.count("#") != len(matches):
        return [], "malformed_reference"
    return list(dict.fromkeys(int(match[1]) for match in matches)), None

def _missing(node):
    answer = node.get("answer")
    if node.get("available") is not True or node.get("status", "success") != "success":
        return "unavailable"
    if node.get("pure_refusal") or node.get("refusal"):
        return "refusal"
    if not isinstance(answer, str) or not answer.strip():
        return "null_or_empty_answer"
    answer = answer.strip().strip("\"'")
    if _MISSING.search(answer) or _MISSING_IN_CONTEXT.search(answer):
        return "missing_answer"
    return None

def _pure_combination(request, refs, answers):
    """An intentionally small whitelist; factual predicates default to retrieval."""
    text = request.lower().strip().rstrip("?.")
    if not refs:
        return False
    # Count the referenced collection itself, not facts about a referenced entity.
    if re.fullmatch(r"(?:what is (?:the )?)?(?:the )?(?:number|count) of #[1-9]\d*", text):
        return True
    if re.fullmatch(r"how many (?:items|members|elements|entries|sets) are in #[1-9]\d*", text):
        return True
    booleans = all(value.strip().lower().rstrip(".!?") in {"yes", "no", "true", "false"} for value in answers)
    if booleans and re.fullmatch(r"(?:are|is|were|was) both #[1-9]\d* and #[1-9]\d* true", text):
        return True
    if booleans and re.fullmatch(r"if #[1-9]\d* then #[1-9]\d*", text):
        return True
    if len(refs) < 2:
        return False
    pair = r"#[1-9]\d* and #[1-9]\d*"
    # Equality, intersection, and extraction from already supplied descriptions.
    if re.fullmatch(r"is #[1-9]\d* the same as #[1-9]\d*", text):
        return True
    if re.fullmatch(r"are " + pair + r" (?:different|the same(?: areas)?)", text):
        return True
    if re.fullmatch(r"what do " + pair + r" have in common", text):
        return True
    attribute = (
        r"(?:occupations?|professions?|jobs?|roles?|nationalit(?:y|ies)|countries?|"
        r"places?|locations?|continents?|genres?|types?(?: of [a-z -]+)?|"
        r"kinds?(?: of [a-z -]+)?|(?:specific )?types|categories|animals?|media|"
        r"music genres?|film types?|groups?|companies|production compan(?:y|ies)|"
        r"us courts?|musicians?|years?|dates?)"
    )
    if re.fullmatch(r"(?:what|which)(?: " + attribute + r")? (?:is|are) (?:common to|in both) " + pair, text):
        return True
    if re.fullmatch(r"what " + attribute + r" (?:is|are) represented (?:by|in) " + pair, text):
        return True
    if re.fullmatch(r"where are both " + pair, text):
        return True
    if re.fullmatch(r"is #[1-9]\d* (?:after|before|greater than|less than|bigger than|smaller than) #[1-9]\d*", text):
        return True
    if re.fullmatch(r"compare #[1-9]\d* (?:and|with|to) #[1-9]\d*", text):
        return True
    selection = re.match(r"(?:which|who)\b", text)
    according = re.search(r"\baccording to " + pair + r"(?:, respectively)?$", text)
    if selection and booleans:
        if according or re.search(r"\b(?:true statement|true proposition) among " + pair + r"(?:, respectively)?$", text):
            return True
    # Dates/counts must actually be supplied, not merely the referenced entities.
    scalar_words = set((
        "january february march april may june july august september october november december "
        "jan feb mar apr jun jul aug sep sept oct nov dec "
        "about over under approximately circa before after prior to late early mid and "
        "years year months month days day miles mile kilometers kilometer kilometres kilometre "
        "km m meters meter metres metre feet foot ft people persons species percent million billion"
    ).split())

    def scalar(value):
        words = re.findall(r"[a-z]+", value.lower())
        return bool(re.search(r"\d", value)) and len(words) <= 12 and all(word in scalar_words for word in words)

    scalars = all(scalar(value) for value in answers)
    literal_numbers = re.search(r"\d", _REF.sub("", text))
    comparative = r"(?:earlier|earliest|later|latest|older|oldest|younger|youngest|more|most|less|least|greater|greatest|smaller|larger|highest|lowest|higher|lower|longer|shorter|first|recently|farther|further|closer|broader)"
    if selection and scalars:
        if according and (re.search(r"\b" + comparative + r"\b", text) or literal_numbers):
            return True
        if re.search(r"\bmatching\b|\bcorresponds to (?:the )?description date\b", text) and literal_numbers:
            return True
    # Source decompositions explicitly provide the values for these comparisons.
    if selection and re.search(r"\b" + comparative + r"\b", text):
        if re.search(r"\baccording to #[1-9]\d*", text):
            return True
        if re.search(r"\b(?:greater|earlier|later|larger|smaller|higher|lower|highest|lowest) of #[1-9]\d* (?:and|,) #[1-9]\d*", text):
            return True
        if re.fullmatch(r"which of #[1-9]\d* and #[1-9]\d* is (?:the )?" + comparative, text):
            return True
    return False

def plan_goal(qid, decomposition, nodes):
    """Return {action, query, reason, refs}; never call retrieval, APIs, or write.

    Nodes may be the source list or a mapping of step IDs to saved node records.
    Direct goal refs are returned; all their transitive dependencies are checked.
    Unknown classification defaults to retrieve, but unsafe binding always skips.
    """
    direct_refs = []

    def result(action, reason, query=None):
        return {"action": action, "query": query, "reason": reason, "refs": direct_refs}

    if not isinstance(decomposition, dict) or decomposition.get("qid", qid) != qid:
        return result("skip", "invalid_decomposition")
    target = decomposition.get("qdmr", decomposition)
    if not isinstance(target, dict):
        return result("skip", "invalid_goal")
    steps, goal = target.get("steps"), target.get("goal")
    if not isinstance(steps, list) or not steps or type(goal) is not int or not 1 <= goal <= len(steps):
        return result("skip", "invalid_goal")
    requests, dependencies = {}, {}
    for index, step in enumerate(steps, 1):
        if (not isinstance(step, dict) or type(step.get("id")) is not int or step["id"] != index
                or not isinstance(step.get("request"), str) or not step["request"].strip()):
            return result("skip", "invalid_step_schema")
        requests[index] = step["request"].strip()
        dependencies[index], error = _refs(requests[index])
        if error or any(ref >= index for ref in dependencies[index]):
            return result("skip", f"invalid_reference_in_step:{index}")
    direct_refs = dependencies[goal]
    if len(steps) == 1:
        return result("skip", "goal_only_no_intermediate_information")
    records = nodes.values() if isinstance(nodes, dict) else nodes
    if not isinstance(records, (list, tuple)) and not isinstance(nodes, dict):
        return result("skip", "invalid_nodes")
    by_id = {}
    for node in records:
        if not isinstance(node, dict) or type(node.get("step_id")) is not int or node.get("qid", qid) != qid:
            return result("skip", "invalid_node_record")
        if node["step_id"] in by_id:
            return result("skip", "duplicate_node_record")
        by_id[node["step_id"]] = node

    def check(ref):
        if ref not in by_id:
            return f"unavailable_dependency:{ref}"
        problem = _missing(by_id[ref])
        if problem:
            return f"dependency:{ref}:{problem}"
        for ancestor in dependencies[ref]:
            problem = check(ancestor)
            if problem:
                prefix = "untrusted_count_dependency" if _COUNT.search(requests[ref]) else "untrusted_dependency"
                return f"{prefix}:{ref}->{problem}"
        answer = by_id[ref]["answer"].strip()
        size, words = len(answer.encode("utf-8")), len(answer.split())
        if _ENUM.search(requests[ref]) and (size > MAX_ENUM_ANSWER_UTF8_BYTES or words > MAX_ENUM_ANSWER_WORDS):
            return f"generic_enumeration_answer_too_long:{ref}:bytes={size},words={words}"
        return None

    for ref in direct_refs:
        problem = check(ref)
        if problem:
            return result("skip", problem)
    answers = [by_id[ref]["answer"].strip() for ref in direct_refs]
    if _pure_combination(requests[goal], direct_refs, answers):
        return result("combine", "explicit_pure_combination")
    query = _REF.sub(lambda match: by_id[int(match[1])]["answer"].strip(), requests[goal])
    size, words = len(query.encode("utf-8")), len(query.split())
    if size > MAX_QUERY_UTF8_BYTES or words > MAX_QUERY_WORDS:
        return result("skip", f"bound_query_too_long:bytes={size},words={words}")
    return result("retrieve", "factual_or_unclassified_goal;semantic_binding_not_verified", query)
