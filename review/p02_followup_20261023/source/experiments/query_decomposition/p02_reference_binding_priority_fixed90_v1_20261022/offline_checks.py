"""Reproducible injected-service black-box checks; no provider or retrieval job.

Run: python -m experiments.query_decomposition.p02_reference_binding_priority_fixed90_v1_20261022.offline_checks
Only OFFLINE_CHECKS.json beside this file is written. No Cache, authorization,
credential, scorer, remote bridge or future experiment output is opened.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline, safe_case, passages
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from . import flow
from experiments.query_decomposition.p02_reference_binding_priority_probe_v1_20261022 import templates

SOURCE = HERE.parent / "p02_shared_context_fixed90_v1_20261022/INPUTS90.jsonl"
SOURCE_SHA = "f5ea82408a13c0bfb4d37c35dd0ce0c8dc7d22325f237f183593556dc447e85d"
PACKAGE = ROOT / "src/p02_multigrain_timing_cachefix_v1"
GENERATION = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))["generation"]


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class FakeServices:
    """Deterministic service tape, never a model/client or persistent cache."""
    def __init__(self, case, *, dynamic=False, invalid_rewrite=False, final_length=False):
        self.case = deepcopy(case)
        self.dynamic, self.invalid_rewrite, self.final_length = dynamic, invalid_rewrite, final_length
        self.requests, self.queries = [], []

    def retrieve(self, query):
        self.queries.append(query)
        stem = sha256(query.encode()).hexdigest()[:20]
        docs = []
        for rank in range(1, 11):
            text = f"Offline fixture for {query} at rank {rank}; document literal #12 is immutable."
            source = "offline-source-" + str(rank)
            docs.append(dict(doc_id=stem + ":" + str(rank), rank=rank, source=source, text=text,
                score=1 / rank, block=f"[Chunk {rank} | source={source}, pages=1-1]\n{text}"))
        return dict(qid=self.case["qid"], query=query, k=10, docs=docs)

    def ask(self, stage, prompt, maximum, allow_final_truncation=False):
        assert isinstance(prompt, str)
        payload = build_payload(GENERATION, prompt, maximum)
        self.requests.append(dict(stage=stage, prompt=prompt, maximum=maximum,
            allow_final_truncation=allow_final_truncation, payload=payload))
        assert bool(allow_final_truncation) == (stage == "legacy/final")
        expected_cap = 64 if stage == "legacy/final" else 256 if stage.startswith("legacy/rewrite/") else 128
        assert maximum == expected_cap
        sid = int(stage.rsplit("/", 1)[1]) if stage != "legacy/final" else None
        if stage.startswith("legacy/rewrite/"):
            if self.invalid_rewrite:
                answer = "not a complete question"
            elif not self.dynamic:
                answer = f"Where is Synthetic Entity {sid}?"
            elif sid == 2:
                answer = "Where was Arlen Vale born?"
            else:
                actual = prompt.rsplit("Actual task:", 1)[1]
                place = "Cedar Bay" if "Cedar Bay" in actual else "Birch Port"
                answer = f"Which country contains {place}?"
        elif stage.startswith("legacy/sentence_format/"):
            candidate = prompt.split("\n\nCandidate answer:\n", 1)[1].split("\n\nContext:\n", 1)[0]
            if candidate == "Arlen Vale":
                answer = "Mira's father was Arlen Vale."
            elif candidate in ("Cedar Bay", "Birch Port"):
                answer = "Arlen Vale was born in " + candidate + "."
            else:
                answer = "The requested country is " + candidate + "."
        elif stage == "legacy/final":
            answer = ("Lumeria" if "Cedar Bay" in prompt else "Norvia") if self.dynamic else "Final fixture response."
        elif self.dynamic:
            if sid == 1:
                answer = "Arlen Vale"
            elif sid == 2:
                actual = prompt.split("\n\nActual task:\n", 1)[1]
                answer = "Cedar Bay" if templates.BINDING_HEADER in actual else "Birch Port"
            elif sid == 3:
                actual = prompt.split("\n\nActual task:\n", 1)[1]
                answer = "Cedar Bay lies in Lumeria." if "Cedar Bay" in actual else "Birch Port lies in Norvia."
            else:
                answer = "Lumeria" if "Cedar Bay" in prompt else "Norvia"
        else:
            answer = f"Recorded response for stage {sid}."
        finish = "length" if stage == "legacy/final" and self.final_length else "stop"
        return digest(dict(stage=stage, payload=payload)), answer, finish


def native(case, **kwargs):
    fake = FakeServices(case, **kwargs)
    result = Pipeline().execute(safe_case(case), fake.retrieve, fake.ask)
    return result, fake


def adapted(case, arm, **kwargs):
    fake = FakeServices(case, **kwargs)
    recorded, events = [], []
    result = flow.execute_arm(case, fake.retrieve, fake.ask, arm,
        record_node=recorded.append, record_event=events.append)
    assert recorded == result["nodes"]
    assert len(events) == len(result["nodes"])
    return result, fake


def remove_adapter_metadata(result):
    result = deepcopy(result)
    result.pop("prompt_factor_arm")
    result.pop("prompt_factor_scope")
    for node in result["nodes"]:
        node.pop("raw_P02_child_prompt")
        node.pop("reference_binding_priority")
    return result


def nonordinary(fake, n):
    return [r for r in fake.requests if not r["stage"].startswith("legacy/answer/")
        or int(r["stage"].rsplit("/", 1)[1]) > n]


def check_native_interfaces(case, result, fake, arm):
    index = {r["stage"]: r for r in fake.requests}
    n = len(case["supporting_questions"])
    for node in result["nodes"]:
        sid = node["step_id"]
        assert node["request"] == (case["supporting_questions"][sid - 1] if sid <= n else case["question"])
        qa = index[f"legacy/answer/{sid}"]
        assert node["child_prompt"] == qa["prompt"] and qa["maximum"] == 128
        item = dict(role="ordinary", original_question=case["question"], step_id=sid,
            current_question=node["request"], history=node["prior_complete_answers"],
            accepted_rewrite=node["rewritten_query"], control_prompt=node["raw_P02_child_prompt"])
        if sid <= n:
            assert qa["prompt"] == templates.prompt(item, arm)
            actual = qa["prompt"].split("\n\nActual task:\n", 1)[1]
            label = "@" if arm == "S1" else "#"
            assert f"Current subquestion {label}{sid}:\n{node['request']}\n\nRetrieved passages:\n" in actual
            assert actual.endswith("\n\nRetrieved passages:\n" + passages(node["reader_packet"]) + "\n\nAnswer:")
            for prior in node["prior_complete_answers"]:
                assert f"{label}{prior['step_id']} Question: {prior['request']}\nAnswer: {prior['answer']}" in actual
            expected_maps = templates.bindings(item) if arm == "B1" else []
            assert node["reference_binding_priority"]["candidate_bindings"] == expected_maps
        else:
            assert qa["prompt"] == node["raw_P02_child_prompt"]
            assert node["reference_binding_priority"]["candidate_bindings"] == []
            assert "Actual task:" not in qa["prompt"]
        if node["sentence_format_request_id"]:
            formatter = index[f"legacy/sentence_format/{sid}"]
            item["reader_packet"] = node["reader_packet"]
            assert formatter["prompt"] == templates.formatter_prompt(item, node["initial_answer"])
            assert formatter["maximum"] == 128 and formatter["allow_final_truncation"] is False
        if node["rewritten_retrieval"] is not None:
            assert node["rewritten_retrieval"]["query"] == node["rewritten_query"]
    final = index["legacy/final"]
    assert final["prompt"] == result["final_prompt"] and final["maximum"] == 64 and final["allow_final_truncation"]
    assert result["original_retrieval"] == result["nodes"][-1]["original_retrieval"]
    assert result["original_retrieval"]["query"] == case["question"]
    return len([r for r in fake.requests if r["stage"].startswith("legacy/sentence_format/")])


def all90_checks():
    assert sha(SOURCE) == SOURCE_SHA
    data = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(data) == 90 and Counter(c["dataset"] for c in data) == {"hotpotqa": 30, "2wiki": 30, "musique": 30}
    comparisons, ordinary_prompts, empty = 0, 0, 0
    for position, case in enumerate(data):
        frozen = deepcopy(case)
        original, baseline = native(case, final_length=position % 2 == 0)
        for arm in flow.ARMS:
            result, fake = adapted(case, arm, final_length=position % 2 == 0)
            check_native_interfaces(case, result, fake, arm)
            assert case == frozen
            assert fake.queries == baseline.queries
            assert nonordinary(fake, len(case["supporting_questions"])) == nonordinary(baseline, len(case["supporting_questions"]))
            assert result["final_prompt"] == original["final_prompt"] and result["final_context"] == original["final_context"]
            assert result["helper"] == original["helper"] and result["prediction"] == original["prediction"]
            if arm == "C0":
                assert fake.requests == baseline.requests
                assert remove_adapter_metadata(result) == original
                comparisons += 1
            if not case["supporting_questions"]:
                assert fake.requests == baseline.requests
                assert remove_adapter_metadata(result) == original
                empty += 1
            ordinary_prompts += len(case["supporting_questions"])
    return dict(cases=90, arms=list(flow.ARMS), adapted_flows=360, native_control_flows=90,
        exact_C0_native_request_and_trajectory_comparisons=comparisons,
        ordinary_prompt_contracts=ordinary_prompts, empty_decomposition_arm_noops=empty,
        fixed_fake_answers_keep_all_nonordinary_requests_and_actual_query_sequences_identical=True,
        final_stop_and_length_cases_both_covered=True,
        scope="Equality under identical injected deterministic services, not fresh provider inference or a new historical receipt audit")


def dynamic_checks():
    case = dict(id="2wiki:offline_dynamic_binding", qid="offline_dynamic_binding", dataset="2wiki",
        question="In which country was Mira's father born?", supporting_questions=[
            "Who is Mira's father?", "Where was Mira's father born?",
            "Which country contains the birthplace of Mira's father?"])
    control, cfake = adapted(case, "C0", dynamic=True)
    candidate, bfake = adapted(case, "B1", dynamic=True)
    assert control["prediction"] == "Norvia" and candidate["prediction"] == "Lumeria"
    assert control["nodes"][1]["answer"] == "Arlen Vale was born in Birch Port."
    assert candidate["nodes"][1]["answer"] == "Arlen Vale was born in Cedar Bay."
    third = candidate["nodes"][2]
    assert third["prior_complete_answers"][1]["answer"] == candidate["nodes"][1]["answer"]
    assert third["rewritten_query"] == "Which country contains Cedar Bay?"
    assert third["rewritten_retrieval"]["query"] in bfake.queries
    assert "Which country contains Birch Port?" in cfake.queries
    assert any("Cedar Bay" in doc["text"] for doc in third["reader_packet"]["docs"])
    maps = third["reference_binding_priority"]["candidate_bindings"]
    assert maps and maps[0]["candidate_text"] == "Cedar Bay" and maps[0]["supporting_step_ids"] == [2]
    for node in candidate["nodes"]:
        for mapping in node["reference_binding_priority"]["candidate_bindings"]:
            for support in mapping["supports"]:
                answer = next(row["answer"] for row in node["prior_complete_answers"] if row["step_id"] == support["step_id"])
                assert answer[support["answer_span"][0]:support["answer_span"][1]] == support["answer_text"]
            assert mapping["candidate_text"] in [support["answer_text"] for support in mapping["supports"]]
    assert candidate["nodes"][0]["initial_answer"] == "Arlen Vale"
    assert candidate["nodes"][0]["answer"] == "Mira's father was Arlen Vale."
    assert candidate["nodes"][0]["reference_binding_priority"]["candidate_bindings"] == []
    formatters = check_native_interfaces(case, candidate, bfake, "B1")
    assert formatters == 3  # First and second ordinary nodes and goal.
    check_native_interfaces(case, control, cfake, "C0")
    for arm in flow.ARMS:
        result, fake = adapted(case, arm, dynamic=True, invalid_rewrite=True)
        check_native_interfaces(case, result, fake, arm)
        for node in result["nodes"][1:]:
            assert node["rewritten_query"] == node["request"] and node["rewritten_retrieval"] is None
            assert node["rewrite_rejection"] == "not_one_complete_question; original retained"
            assert node["reference_binding_priority"]["candidate_bindings"] == []
    return dict(passed=True, own_new_postformatter_history_used_for_binding=True,
        own_live_accepted_rewrite_used_for_binding=True,
        downstream_rewrite_actual_retrieval_packet_reader_and_final_depend_on_new_answer=True,
        raw_current_preserved_for_evaluation_and_formatter=True,
        ordinary_and_goal_formatter128_both_verified=True, formatter_stages_in_dynamic_candidate=formatters,
        first_node_no_history_B1_noop=True, invalid_rewrite_native_fallback_all4arms=True,
        candidate_rhs_copied_from_actual_prior_answer_spans=True,
        fixture_answers_are_synthetic_not_dataset_gold_or_claimed_quality=True)


def blocked(*args, **kwargs):
    raise AssertionError("Network or subprocess is forbidden in offline_checks")


def main():
    # Services remain ordinary Python callables; accidental network/process
    # operations also fail instead of silently converting this into a run.
    with patch.object(socket.socket, "connect", blocked), patch.object(socket.socket, "connect_ex", blocked), \
         patch.object(socket, "create_connection", blocked), patch.object(subprocess, "Popen", blocked):
        checks = dict(fixed90=all90_checks(), dynamic_binding=dynamic_checks())
    deps = [Path(__file__), HERE / "flow.py", HERE / "README.md", Path(templates.__file__), SOURCE,
        PACKAGE / "core/pipeline.py", PACKAGE / "core/child_prompts.py", PACKAGE / "core/rewrite_prompts.py",
        PACKAGE / "core/final_prompts.py", PACKAGE / "core/llm.py", PACKAGE / "config.json",
        PACKAGE / "prompts/child_examples.json", PACKAGE / "prompts/final_examples.json"]
    result = dict(passed=True, checks=checks,
        checked_hashes={str(path.resolve().relative_to(ROOT)).replace("\\", "/"): sha(path) for path in deps},
        created_provider_clients=0, new_API_SSH_GPU_requests=0, production_Cache_or_ledger_created=False,
        native_scores_or_new_context_outputs_opened=False,
        limitations="Pure flow/template fake tests only. No runner/auth/resource/receipt/budget correctness or semantic/fullRAG benefit is inferred; those require a separately prepared execution protocol.")
    output = HERE / "OFFLINE_CHECKS.json"
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    output.write_text(encoded, encoding="utf-8")
    print(json.dumps(dict(passed=True, output=str(output.resolve()), sha256=sha(output), checks=checks), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
