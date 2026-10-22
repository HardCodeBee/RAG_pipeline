"""Ordinary-only output transformation over the unchanged native Pipeline.

The caller supplies every service. NativeAsk persists complete raw model
receipts; this adapter passes only parsed Answer to the native Pipeline, which
retains its original formatter, history, rewrite, retrieval, goal and final.
"""
from copy import deepcopy

from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case, merge
from experiments.query_decomposition.p02_local_cot_stage38_v1_20261022 import templates
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.native_ask import AlgorithmTerminal

ARMS = templates.ARMS


class OutputFormatTerminal(AlgorithmTerminal):
    """Complete stop-ended raw output failed parsing. It is never retried."""


def execute_arm(case, retrieve, ask, arm, *, pipeline=None, record_node=None,
                record_output=None, record_event=None):
    if arm not in ARMS: raise ValueError(arm)
    clean = safe_case(case)
    ordinary = clean["supporting_questions"]
    completed, pending, retrieved, rewrites = {}, {}, {}, {}
    event = record_event or (lambda value: None)
    output = record_output or (lambda value: None)

    def history_before(sid):
        assert set(completed) == set(range(1, sid))
        return [dict(step_id=n["step_id"], request=n["request"], answer=n["answer"])
                for _, n in sorted(completed.items())]

    def tracked_retrieve(query):
        packet = retrieve(query)
        retrieved[query] = deepcopy(packet)
        return packet

    def wrapped_ask(stage, native_prompt, maximum, allow_final_truncation=False):
        if not stage.startswith("legacy/answer/"):
            # Rewriting, formatting and final reading keep exact native wires.
            result = ask(stage, native_prompt, maximum, allow_final_truncation=allow_final_truncation)
            if stage.startswith("legacy/rewrite/"):
                sid = int(stage.rsplit("/", 1)[1]); candidate = result[1].strip()
                question = ordinary[sid - 1] if sid <= len(ordinary) else clean["question"]
                rewrites[sid] = candidate if candidate and candidate.endswith("?") and "\n" not in candidate else question
            return result
        sid = int(stage.rsplit("/", 1)[1])
        assert maximum == 128 and not allow_final_truncation
        history = history_before(sid)
        is_ordinary = sid <= len(ordinary)
        item = dict(role="ordinary", step_id=sid, original_question=clean["question"],
            current_question=ordinary[sid - 1] if is_ordinary else clean["question"],
            history=history, control_prompt=native_prompt)
        actual = templates.prompt(item, arm) if is_ordinary else native_prompt
        cap = templates.cap(arm) if is_ordinary else maximum
        question = item["current_question"]; rewritten = rewrites.get(sid, question)
        reader = merge(retrieved[question], retrieved[rewritten] if rewritten != question else None)
        record = dict(step_id=sid, role="ordinary" if is_ordinary else "goal", stage=stage,
            raw_P02_child_prompt=native_prompt, actual_child_prompt=actual, output_cap=cap,
            prior_complete_answers=deepcopy(history), raw_output=None, support=None,
            parsed_answer=None, parser_status="not_reached", arm=arm, current_question=question,
            reader_packet=reader, original_retrieval=deepcopy(retrieved[question]), accepted_rewrite=rewritten)
        pending[sid] = record
        event(dict(event="local_cot_rendered", stage=stage, ordinary=is_ordinary, output_cap=cap,
            prompt_changed=actual != native_prompt, support_forwarded=False))
        try:
            rid, raw, finish = ask(stage, actual, cap, allow_final_truncation=False)
        except AlgorithmTerminal as exc:
            record.update(raw_output=exc.value.get("answer"), request_id=exc.value.get("request_id"),
                finish_reason=exc.value.get("finish_reason"), generation_terminal_reason=exc.value["terminal_reason"])
            output(deepcopy(record))
            raise
        record.update(raw_output=raw, request_id=rid, finish_reason=finish)
        # NativeAsk enforces stop and complete receipts. Fail closed for a caller
        # that violates the contract; do not silently parse a truncated answer.
        if finish != "stop" or not isinstance(raw, str) or not raw.strip():
            record["generation_terminal_reason"] = "non_native_incomplete_answer"
            output(deepcopy(record))
            raise AlgorithmTerminal(dict(stage=stage, answer=raw, request_id=rid, finish_reason=finish,
                terminal_reason="non_native_incomplete_answer"))
        if is_ordinary:
            try: parsed = templates.parse(arm, raw)
            except templates.FormatFailure as exc:
                record.update(parser_status="format_failure", parser_reason=exc.reason)
                output(deepcopy(record))
                raise OutputFormatTerminal(dict(stage=stage, answer=raw, request_id=rid, finish_reason=finish,
                    terminal_reason="invalid_support_answer_format", parser_reason=exc.reason,
                    raw_receipt_is_complete=True, accepted_answer=None)) from None
        else:
            parsed = dict(support=None, answer=raw.strip())
        record.update(support=parsed["support"], parsed_answer=parsed["answer"], parser_status="parsed")
        output(deepcopy(record))
        # Only this answer enters native <4-word formatting and later history.
        # NativeAsk.stages continues to retain raw (Support + Answer) text.
        return rid, parsed["answer"], finish

    def record(node):
        node = deepcopy(node); sid = node["step_id"]; record = pending[sid]
        assert node["child_prompt"] == record["raw_P02_child_prompt"]
        assert node["prior_complete_answers"] == record["prior_complete_answers"]
        assert node["initial_answer"] == record["parsed_answer"]
        node["raw_P02_child_prompt"] = record["raw_P02_child_prompt"]
        node["child_prompt"] = record["actual_child_prompt"]
        node["local_cot"] = dict(arm=arm, role=record["role"], raw_output=record["raw_output"],
            support=record["support"], parsed_answer=record["parsed_answer"], accepted_answer=node["answer"],
            parser_status=record["parser_status"], output_cap=record["output_cap"],
            support_forwarded=False, accepted_answer_is_post_native_formatter=True)
        if record_node: record_node(deepcopy(node))
        # A formatter/recorder failure cannot commit a partial node to adapter history.
        completed[sid] = node

    result = (pipeline or Pipeline()).execute(clean, tracked_retrieve, wrapped_ask, record_node=record)
    result["nodes"] = [completed[n["step_id"]] for n in result["nodes"]]
    result["local_cot_arm"] = arm
    result["local_cot_scope"] = "ordinary only; raw Support audit only; own accepted answers drive native dynamic downstream"
    return result
