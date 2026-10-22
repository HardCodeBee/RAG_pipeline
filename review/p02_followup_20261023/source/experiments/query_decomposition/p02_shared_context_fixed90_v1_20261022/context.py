"""Pure, per-original-query context; never retain a previous expanded wire."""
from copy import deepcopy
import hashlib
import json

ARMS = ("reset", "flat", "chat")


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def messages(value):
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    assert isinstance(value, list) and value
    assert all(set(row) == {"role", "content"} and row["role"] in ("user", "assistant")
               and isinstance(row["content"], str) and row["content"] for row in value)
    return deepcopy(value)


class Conversation:
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.turns = arm, []

    def render(self, step_id, current_prompt):
        assert type(step_id) is int and step_id == len(self.turns) + 1
        assert isinstance(current_prompt, str) and current_prompt
        if self.arm == "reset" or not self.turns:
            return current_prompt
        history = [dict(role=role, content=content) for turn in self.turns
                   for role, content in (("user", turn["raw_child_prompt"]),
                                         ("assistant", turn["accepted_answer"]))]
        history.append(dict(role="user", content=current_prompt))
        if self.arm == "chat":
            return history
        # Delimiters are presentation labels, not extra instructions. Every
        # previous content string and the current native prompt are unchanged.
        return "\n\n".join(("User:\n" if row["role"] == "user" else "Assistant:\n")
                             + row["content"] for row in history)

    def commit(self, node):
        assert node["step_id"] == len(self.turns) + 1
        assert node["role"] in ("ordinary", "goal")
        assert isinstance(node["child_prompt"], str) and node["child_prompt"]
        assert isinstance(node["answer"], str) and node["answer"].strip()
        # Pipeline calls record_node only after optional formatting succeeds.
        # The committed assistant content is the same accepted answer that its
        # ordinary history, future rewriter and final helper receive.
        self.turns.append(dict(step_id=node["step_id"], role=node["role"],
            raw_child_prompt=node["child_prompt"], accepted_answer=node["answer"],
            initial_answer=node["initial_answer"], answer_request_id=node["answer_request_id"],
            sentence_format_request_id=node["sentence_format_request_id"]))

    def manifest(self):
        return [dict(step_id=turn["step_id"], role=turn["role"],
            raw_child_prompt_sha256=digest(turn["raw_child_prompt"]),
            accepted_answer_sha256=digest(turn["accepted_answer"]),
            answer_request_id=turn["answer_request_id"],
            sentence_format_request_id=turn["sentence_format_request_id"])
            for turn in self.turns]
