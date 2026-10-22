"""Wrap the unchanged core Pipeline; only ordinary/goal request context changes."""
from copy import deepcopy
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case
from .context import ARMS, Conversation, messages, digest


def execute_arm(case, retrieve, ask, arm, *, pipeline=None, record_node=None, record_event=None):
    if arm not in ARMS:
        raise ValueError(arm)
    clean = safe_case(case)
    conversation, rendered, recorded = Conversation(arm), {}, {}
    events = record_event or (lambda value: None)

    def wrapped_ask(stage, native_prompt, maximum, allow_final_truncation=False):
        assert isinstance(native_prompt, str)
        wire = native_prompt
        if stage.startswith("legacy/answer/"):
            sid = int(stage.rsplit("/", 1)[1])
            assert maximum == 128 and not allow_final_truncation
            wire = conversation.render(sid, native_prompt)
            rendered[sid] = dict(raw_prompt=native_prompt, wire=deepcopy(wire),
                prior_turns=conversation.manifest())
            events(dict(event="shared_context_request", stage=stage, arm=arm,
                raw_current_prompt=native_prompt, wire_input=deepcopy(wire),
                current_raw_prompt_sha256=digest(native_prompt),
                wire_messages_sha256=digest(messages(wire)),
                historical_turns=len(conversation.turns), message_count=len(messages(wire)),
                message_content_characters=sum(len(r["content"]) for r in messages(wire))))
        # Rewriting, <4-word formatting and final reading are independent calls.
        return ask(stage, wire, maximum, allow_final_truncation=allow_final_truncation)

    def record(node):
        node = deepcopy(node)
        sid = node["step_id"]
        pending = rendered[sid]
        assert node["child_prompt"] == pending["raw_prompt"]
        node["raw_P02_child_prompt"] = node["child_prompt"]
        node["shared_context"] = dict(arm=arm, historical_turns=len(pending["prior_turns"]),
            previous_turn_manifest=pending["prior_turns"],
            wire_input=deepcopy(pending["wire"]),
            wire_messages_sha256=digest(messages(pending["wire"])),
            committed_assistant_answer="accepted_post_optional_formatter",
            old_wire_never_nested=True)
        if record_node:
            record_node(deepcopy(node))
        # A failed recorder/formatter never commits a partially completed turn.
        conversation.commit(node)
        recorded[sid] = node

    result = (pipeline or Pipeline()).execute(clean, retrieve, wrapped_ask, record_node=record)
    assert len(conversation.turns) == len(clean["supporting_questions"]) + 1
    result["nodes"] = [recorded[node["step_id"]] for node in result["nodes"]]
    result["context_arm"] = arm
    result["context_turn_manifest"] = conversation.manifest()
    result["context_scope"] = "Within one original query; ordinary+goal only; native raw prompts/accepted answers; final independent"
    return result
