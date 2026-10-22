"""Use the unchanged Pipeline; change only actual ordinary-QA current text.

No model, network, retriever, persistent Cache, auth or run launcher is created.
All services are injected. This draft is not an execution authorization.
"""

from copy import deepcopy

from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case
from experiments.query_decomposition.p02_failure8_actual_target_probe_v1_20261022.templates import prompt as stage_prompt

ARMS = ("D0", "D1")


def accepted_rewrite(raw, question):
    """Exactly the existing Pipeline's successful-rewrite acceptance rule."""
    candidate = raw.strip()
    return candidate if candidate and candidate.endswith("?") and "\n" not in candidate else question


def execute_arm(case, retrieve, ask, arm, *, pipeline=None, record_node=None, record_event=None):
    """Apply to EVERY ordinary node, preserving raw question as audit target.

    The rewrite is obtained live from this same arm's current history. A changed
    answer therefore flows through the original subsequent rewrite/retrieve/QA,
    goal and final code. No saved reader or selected anchor is injected here.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    case = safe_case(case)
    ordinary = case["supporting_questions"]
    accepted, actual_prompts, question_metadata = {}, {}, {}
    events = record_event or (lambda event: None)

    def wrapped_ask(stage, original_prompt, maximum, allow_final_truncation=False):
        actual = original_prompt
        if stage.startswith("legacy/answer/"):
            sid = int(stage.rsplit("/", 1)[1])
            if 1 <= sid <= len(ordinary):
                raw = ordinary[sid - 1]
                model_current = raw if arm == "D0" else accepted.get(sid, raw)
                # The stage probe's already-reviewed helper changes exactly the
                # Actual task Current subquestion field, never fewshot examples.
                actual = stage_prompt(dict(control_prompt=original_prompt,
                    step_id=sid, current_question=raw,
                    frozen_rewritten_query=model_current), "C1" if arm == "D1" else "C0")
                question_metadata[sid] = dict(evaluation_target=raw, model_current=model_current,
                    policy="all_ordinary_live_accepted_rewrite" if arm == "D1" else "original_P02")
            actual_prompts[sid] = (original_prompt, actual)
        events(dict(event="rendered_stage", stage=stage, maximum=maximum,
            prompt_changed=actual != original_prompt,
            allow_final_truncation=allow_final_truncation))
        result = ask(stage, actual, maximum, allow_final_truncation=allow_final_truncation)
        if stage.startswith("legacy/rewrite/") and result[2] == "stop":
            sid = int(stage.rsplit("/", 1)[1])
            raw_question = ordinary[sid - 1] if sid <= len(ordinary) else case["question"]
            accepted[sid] = accepted_rewrite(result[1], raw_question)
        return result

    def enrich(node):
        node = deepcopy(node)
        raw_prompt, actual_prompt = actual_prompts[node["step_id"]]
        node["raw_P02_child_prompt"] = raw_prompt
        node["child_prompt"] = actual_prompt
        node["actual_question_interface"] = question_metadata.get(node["step_id"],
            dict(evaluation_target=node["request"], model_current=node["request"], policy="goal_unchanged"))
        return node

    result = (pipeline or Pipeline()).execute(case, retrieve, wrapped_ask,
        record_node=(lambda node: record_node(enrich(node))) if record_node else None)
    result["nodes"] = [enrich(node) for node in result["nodes"]]
    result["draft_arm"] = arm
    result["draft_execution_scope"] = "all ordinary; dynamic original Pipeline downstream"
    return result
