"""Pure fake complete flows: no Cache, provider, SSH, GPU, or scientific labels."""
from copy import deepcopy
import hashlib
import json

from . import preparation as p
from . import flow
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case
from experiments.query_decomposition.p02_local_cot_stage38_v1_20261022 import templates
from experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.native_ask import AlgorithmTerminal

SENTINEL = "SUPPORT_AUDIT_ONLY_SENTINEL"
CASE = dict(id="hotpotqa:fixture", qid="fixture", dataset="hotpotqa", question="When did the author's city open?",
    supporting_questions=["Which city is associated with the author?", "When did the author's associated city open?"])


def packet(query):
    docs = []
    for i in range(10):
        text = "Fixture public retrieval material for query " + query + " with index " + str(i)
        docs.append(dict(rank=i + 1, doc_id=hashlib.sha256(text.encode()).hexdigest(), source="Fixture", text=text,
            block=f"[Chunk {i + 1} | source=Fixture, pages=1-1]\n" + text))
    return dict(qid="fixture", query=query, k=10, docs=docs)


class FakeServices:
    def __init__(self, arm, ordinary=2, *, malformed=False, fail_stage=None, final_length=False):
        self.arm, self.ordinary = arm, ordinary
        self.malformed, self.fail_stage, self.final_length = malformed, fail_stage, final_length
        self.calls, self.queries, self.outputs = [], [], []
        self.city = {"N128": "Delta", "N256": "Delta", "F256": "Echo", "R256": "Foxtrot"}[arm]
    def retrieve(self, query):
        assert SENTINEL not in query
        self.queries.append(query); return packet(query)
    def ask(self, stage, prompt, maximum, allow_final_truncation=False):
        self.calls.append(dict(stage=stage, prompt=prompt, maximum=maximum, allow_final_truncation=allow_final_truncation))
        rid = "fake-" + str(len(self.calls))
        if self.fail_stage == stage:
            raise AlgorithmTerminal(dict(stage=stage, answer="Truncated fixture output", request_id=rid,
                finish_reason="length", terminal_reason="terminal_length"))
        if stage.startswith("legacy/answer/"):
            sid = int(stage.rsplit("/", 1)[1]); ordinary = sid <= self.ordinary
            assert maximum == (templates.cap(self.arm) if ordinary else 128)
            assert allow_final_truncation is False
            if sid > 1:
                assert "The target city is " + self.city + "." in prompt
                assert SENTINEL not in prompt, "Previous Support leaked into later QA"
            answer = self.city if sid == 1 and ordinary else "The opening year is 2001."
            if ordinary and self.arm in ("F256", "R256"):
                answer = "An unstructured response that must terminate." if self.malformed else f"Support: {SENTINEL} mentions the requested city.\nAnswer: {answer}"
        elif stage.startswith("legacy/rewrite/"):
            assert maximum == 256 and not allow_final_truncation
            assert SENTINEL not in prompt and "The target city is " + self.city + "." in prompt
            answer = f"When did {self.city} open?"
        elif stage.startswith("legacy/sentence_format/"):
            assert maximum == 128 and not allow_final_truncation
            assert SENTINEL not in prompt and "Candidate answer:\n" + self.city + "\n\nContext:" in prompt
            answer = "The target city is " + self.city + "."
        else:
            assert stage == "legacy/final" and maximum == 64 and allow_final_truncation is True
            assert SENTINEL not in prompt
            answer = "2001"
        self.outputs.append(dict(stage=stage, raw=answer))
        return rid, answer, "length" if stage == "legacy/final" and self.final_length else "stop"


def main():
    p.prepare(); checks = []; pipeline = Pipeline()
    # Control must be exactly native in both requests and retrieved query order.
    native = FakeServices("N128"); wrapped = FakeServices("N128")
    original = pipeline.execute(safe_case(CASE), native.retrieve, native.ask)
    result = flow.execute_arm(CASE, wrapped.retrieve, wrapped.ask, "N128", pipeline=pipeline)
    assert native.calls == wrapped.calls and native.queries == wrapped.queries
    assert all(result[k] == original[k] for k in ("history_all", "helper", "final_prompt", "final_context", "prediction", "status"))
    for old, new in zip(original["nodes"], result["nodes"]): assert all(new[k] == v for k, v in old.items())
    checks.append("N128_native_wire_retrieval_history_formatter_goal_final_exact")
    control_calls = wrapped.calls
    for arm in flow.ARMS:
        fake = FakeServices(arm); records = []; candidate_outputs = []
        result = flow.execute_arm(CASE, fake.retrieve, fake.ask, arm, pipeline=pipeline,
            record_node=records.append, record_output=candidate_outputs.append)
        assert len(records) == len(candidate_outputs) == 3
        assert result["history_all"][0]["answer"] == "The target city is " + fake.city + "."
        assert records[1]["prior_complete_answers"][0]["answer"] == result["history_all"][0]["answer"]
        assert f"When did {fake.city} open?" in fake.queries
        assert records[0]["initial_answer"] == fake.city and records[0]["local_cot"]["parsed_answer"] == fake.city
        assert records[0]["local_cot"]["accepted_answer"] == records[0]["answer"]
        assert records[0]["sentence_format_request_id"] is not None
        assert records[-1]["local_cot"]["support"] is None and records[-1]["local_cot"]["output_cap"] == 128
        assert result["helper"].count(SENTINEL) == result["final_prompt"].count(SENTINEL) == 0
        if arm == "N256":
            assert len(fake.calls) == len(control_calls)
            for before, after in zip(control_calls, fake.calls):
                expected = deepcopy(before)
                if before["stage"] in ("legacy/answer/1", "legacy/answer/2"): expected["maximum"] = 256
                assert after == expected
        if arm in ("F256", "R256"):
            assert SENTINEL in candidate_outputs[0]["raw_output"] and SENTINEL in candidate_outputs[0]["support"]
            assert candidate_outputs[0]["parsed_answer"] == fake.city
            assert SENTINEL in records[0]["local_cot"]["raw_output"]
            # The raw model output remains separate from the transformed return.
            assert fake.outputs[0]["raw"] == records[0]["local_cot"]["raw_output"] != records[0]["initial_answer"]
    checks.extend(["N256_changes_only_ordinary_cap", "own_postformatter_answer_drives_next_rewrite_and_actual_retrieval",
        "Support_audit_only_never_in_formatter_history_goal_final_or_retrieval", "raw_support_parsed_and_accepted_answer_remain_distinct"])
    for arm in ("F256", "R256"):
        fake = FakeServices(arm, malformed=True); nodes = []; outputs = []
        try: flow.execute_arm(CASE, fake.retrieve, fake.ask, arm, pipeline=pipeline, record_node=nodes.append, record_output=outputs.append)
        except flow.OutputFormatTerminal as exc:
            assert exc.value["terminal_reason"] == "invalid_support_answer_format" and exc.value["raw_receipt_is_complete"] is True
        else: raise AssertionError("Parser failure did not stop algorithm")
        assert not nodes and len(fake.calls) == len(fake.queries) == len(outputs) == 1
        assert outputs[0]["parser_status"] == "format_failure" and outputs[0]["raw_output"]
    checks.append("parser_failure_is_algorithm_terminal_no_formatter_no_reroll_no_later_retrieval")
    for failing in ("legacy/answer/1", "legacy/sentence_format/1", "legacy/rewrite/2", "legacy/answer/3"):
        fake = FakeServices("R256", fail_stage=failing); outputs = []; nodes = []
        try: flow.execute_arm(CASE, fake.retrieve, fake.ask, "R256", pipeline=pipeline, record_node=nodes.append, record_output=outputs.append)
        except AlgorithmTerminal as exc: assert exc.value["stage"] == failing and exc.value["terminal_reason"] == "terminal_length"
        else: raise AssertionError("Native length terminal was swallowed")
        assert fake.calls[-1]["stage"] == failing and not any(c["stage"] == "legacy/final" for c in fake.calls)
        if failing == "legacy/sentence_format/1":
            assert not nodes and outputs[0]["parsed_answer"] == fake.city and outputs[0]["support"]
    checks.append("ordinary_formatter_rewrite_goal_length_terminal_keeps_raw_partial_without_committing_failed_node")
    fake = FakeServices("R256", final_length=True)
    result = flow.execute_arm(CASE, fake.retrieve, fake.ask, "R256", pipeline=pipeline)
    assert result["status"] == "success_truncated" and result["prediction"] == "2001"
    checks.append("native_final64_length_remains_accepted")
    goal_case = {**CASE, "supporting_questions": []}
    wires = []
    for arm in flow.ARMS:
        fake = FakeServices(arm, ordinary=0)
        result = flow.execute_arm(goal_case, fake.retrieve, fake.ask, arm, pipeline=pipeline)
        assert len(result["nodes"]) == 1 and result["nodes"][0]["role"] == "goal"
        wires.append((fake.calls, fake.queries))
    assert all(v == wires[0] for v in wires)
    checks.append("goal_only_noop_cases_retained_all_four_wires_identical")
    # Render every frozen ordinary question with actual R02 bodies, empty prior
    # history, and fake retrieval. No historical QA/rewrite or truth is loaded.
    from p02_multigrain_timing_cachefix_v1.core import child_prompts
    rendered = 0
    for case in p.rows(p.INPUTS):
        for sid, question in enumerate(case["supporting_questions"], 1):
            native = child_prompts.child(case["question"], sid, question, [], "Fixture reader", pipeline.child_examples, role="ordinary")
            for arm in flow.ARMS:
                item = dict(role="ordinary", control_prompt=native)
                assert templates.prompt(item, arm)
                rendered += 1
    checks.append("all_fixed90_R02_ordinary_bodies_render_without_historical_answers_or_labels")
    value = dict(passed=True, checks=checks, flow_sha256=p.sha(p.HERE / "flow.py"),
        template_sha256=p.sha(p.STAGE / "templates.py"), inputs_sha256=p.sha(p.INPUTS), order_sha256=p.sha(p.ORDER),
        source_questions=90, planned_conditions=360, goal_only_cases=6, rendered_ordinary_requests=rendered,
        new_API_SSH_GPU_requests=0, production_cache_or_ledger_writes=0, no_quality_claim=True)
    p.atomic_json(p.HERE / "OFFLINE_FLOW_CHECKS.json", value)
    return value


if __name__ == "__main__": print(json.dumps(main(), ensure_ascii=True))
