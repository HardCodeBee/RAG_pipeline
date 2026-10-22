"""Inject one ordinary-QA factor into the unchanged native Pipeline.

All services are supplied by the caller. This module creates no model,
retriever, provider, persistent cache, authorization or run controller.
"""
from copy import deepcopy

from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case
from experiments.query_decomposition.p02_reference_binding_priority_probe_v1_20261022 import templates

ARMS = templates.ARMS


def accepted_rewrite(raw, question):
    candidate = raw.strip()
    return candidate if candidate and candidate.endswith("?") and "\n" not in candidate else question


def execute_arm(case, retrieve, ask, arm, *, pipeline=None, record_node=None, record_event=None):
    if arm not in ARMS:
        raise ValueError(arm)
    clean = safe_case(case)
    ordinary = clean["supporting_questions"]
    accepted, rendered, completed = {}, {}, {}
    event = record_event or (lambda value: None)

    def prior_rows(sid):
        assert set(completed) == set(range(1, sid))
        return [dict(step_id=n["step_id"], request=n["request"], answer=n["answer"])
                for _, n in sorted(completed.items())]

    def wrapped_ask(stage, native_prompt, maximum, allow_final_truncation=False):
        actual = native_prompt
        if stage.startswith("legacy/answer/"):
            sid = int(stage.rsplit("/", 1)[1])
            assert maximum == 128 and not allow_final_truncation
            history = prior_rows(sid)
            maps = []
            if sid <= len(ordinary):
                raw = ordinary[sid - 1]
                item = dict(role="ordinary", step_id=sid, original_question=clean["question"],
                            current_question=raw, history=history, control_prompt=native_prompt,
                            accepted_rewrite=accepted.get(sid, raw))
                actual = templates.prompt(item, arm)
                if arm == "B1":
                    maps = templates.bindings(item)
            rendered[sid] = dict(raw_prompt=native_prompt, actual_prompt=actual,
                                 prior_rows=deepcopy(history), bindings=deepcopy(maps),
                                 accepted_rewrite=accepted.get(sid, ordinary[sid - 1] if sid <= len(ordinary) else clean["question"]))
            event(dict(event="reference_binding_priority_rendered", arm=arm, stage=stage,
                       prompt_changed=actual != native_prompt, binding_count=len(maps),
                       ordinary=sid <= len(ordinary)))
        result = ask(stage, actual, maximum, allow_final_truncation=allow_final_truncation)
        if stage.startswith("legacy/rewrite/"):
            sid = int(stage.rsplit("/", 1)[1])
            assert maximum == 256 and result[2] == "stop"
            raw = ordinary[sid - 1] if sid <= len(ordinary) else clean["question"]
            accepted[sid] = accepted_rewrite(result[1], raw)
        return result

    def record(node):
        node = deepcopy(node)
        sid = node["step_id"]
        item = rendered[sid]
        assert node["child_prompt"] == item["raw_prompt"]
        assert node["prior_complete_answers"] == item["prior_rows"]
        assert node["rewritten_query"] == item["accepted_rewrite"]
        node["raw_P02_child_prompt"] = item["raw_prompt"]
        node["child_prompt"] = item["actual_prompt"]
        node["reference_binding_priority"] = dict(
            arm=arm, evaluation_target=node["request"],
            prompt_changed=item["actual_prompt"] != item["raw_prompt"],
            candidate_bindings=item["bindings"],
            policy="ordinary_only; own accepted history and live accepted rewrite; goal/formatter/final native")
        if record_node:
            record_node(deepcopy(node))
        completed[sid] = node

    result = (pipeline or Pipeline()).execute(clean, retrieve, wrapped_ask, record_node=record)
    result["nodes"] = [completed[n["step_id"]] for n in result["nodes"]]
    result["prompt_factor_arm"] = arm
    result["prompt_factor_scope"] = "Actual R02 ordinary QA only; complete native dynamic downstream"
    return result
