"""Frozen syntax/reference and availability functions; no semantic repair."""
import re
REF=re.compile(r"#([1-9][0-9]*)\b")

def references(request):
    matches = list(REF.finditer(request))
    if request.count("#") != len(matches):
        raise ValueError("malformed_reference")
    return list(dict.fromkeys(int(m[1]) for m in matches))

def structural_errors(target):
    """Only format/reference legality; no natural-language semantic rules."""
    if not isinstance(target, dict) or set(target) != {"steps", "goal"}:
        return ["invalid_qdmr_fields"]
    steps, goal = target["steps"], target["goal"]
    if not isinstance(steps, list) or not steps or type(goal) is not int or not 1 <= goal <= len(steps):
        return ["invalid_steps_or_goal"]
    errors = []
    for sid, step in enumerate(steps, 1):
        if (not isinstance(step, dict) or set(step) != {"id", "request"}
                or type(step["id"]) is not int or step["id"] != sid
                or not isinstance(step["request"], str) or not step["request"].strip()):
            errors.append(f"invalid_step:{sid}")
            continue
        try:
            refs = references(step["request"])
            if any(ref >= sid for ref in refs):
                errors.append(f"nonprior_reference:{sid}")
        except ValueError:
            errors.append(f"malformed_reference:{sid}")
    return errors

def returned(node):
    return bool(node and node.get("status") == "success" and node.get("answer", "").strip()
                and node.get("finish_reason") != "length" and not node.get("refusal"))

def bind(request, step_id, answers):
    """Replace exactly once. Never resolve references introduced by answers."""
    refs = [int(m.group(1)) for m in re.finditer(r"#(\d+)\b", request)]
    if any(n <= 0 or n >= step_id for n in refs):
        return None, "illegal_reference"
    if any(n not in answers for n in refs):
        return None, "unavailable_dependency"
    return re.sub(r"#(\d+)\b", lambda m: answers[int(m.group(1))], request), None

def pure_refusal(answer, phrases):
    def norm(s):
        return " ".join(s.casefold().split()).rstrip(".!?。！？ ")
    return norm(answer) in {norm(s) for s in phrases}
