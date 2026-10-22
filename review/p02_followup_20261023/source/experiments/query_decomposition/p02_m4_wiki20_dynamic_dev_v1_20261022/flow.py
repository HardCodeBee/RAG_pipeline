"""Isolated native execute copy: only merge-selection line differs; no global patch."""
from copy import deepcopy
from p02_multigrain_timing_cachefix_v1.core import METHOD
from p02_multigrain_timing_cachefix_v1.core import child_prompts,rewrite_prompts
from p02_multigrain_timing_cachefix_v1.core.pipeline import Pipeline,safe_case,passages,render_final
from experiments.query_decomposition.p02_m4_reader_stage_probe8_v1_20261022 import selection
ARMS=("M0","M4")

class ReaderPipeline(Pipeline):
    def __init__(self,pipeline,arm):
        self.__dict__.update(pipeline.__dict__)
        self.reader_arm=arm

    def execute(self, case, retrieve, ask, *, record_node=None):
        examples = self.child_examples
        history, nodes = [], []
        root = None
        for sid, question in enumerate(
            case["supporting_questions"] + [case["question"]], 1
        ):
            is_goal = sid == len(case["supporting_questions"]) + 1
            available = deepcopy(history)
            original = retrieve(question)
            assert (
                len(original["docs"])
                == len({d["doc_id"] for d in original["docs"]})
                == 10
            )
            rewrite, rewrite_id, raw_rewrite, rejection, rewrite_prompt = (
                question,
                None,
                None,
                None,
                None,
            )
            if available:
                rewrite_prompt = rewrite_prompts.rewrite(question, available)
                rewrite_id, raw_rewrite, finish = ask(
                    f"legacy/rewrite/{sid}", rewrite_prompt, 256
                )
                if finish != "stop":
                    raise RuntimeError("Incomplete rewrite is not propagated")
                candidate = raw_rewrite.strip()
                if candidate and candidate.endswith("?") and "\n" not in candidate:
                    rewrite = candidate
                else:
                    rejection = "not_one_complete_question; original retained"
            rewritten = retrieve(rewrite) if rewrite != question else None
            packet = selection.packet(original, rewritten, self.reader_arm)
            if is_goal:
                root = deepcopy(original)
            packet_text = passages(packet)
            raw_child = child_prompts.child(
                case["question"],
                sid,
                question,
                available,
                packet_text,
                examples,
                role="goal" if is_goal else "ordinary",
            )
            prompt = raw_child
            if is_goal:
                prompt = (
                    prompt[: -len("Answer:")] + child_prompts.GOAL_RULE + "\n\nAnswer:"
                )
            answer_id, answer, finish = ask(f"legacy/answer/{sid}", prompt, 128)
            if finish != "stop" or not answer.strip():
                raise RuntimeError("Incomplete/empty child answer is not propagated")
            answer = answer.strip()
            initial = answer
            format_id = None
            if len(answer.split()) < 4:
                format_prompt = (
                    "Express the candidate answer below as a concise complete factual sentence that names its subject and relation. "
                    "Use only the current question, candidate answer, and supplied context. Preserve uncertainty; do not add facts. "
                    "If evidence is missing, state the specific unresolved fact in a complete sentence. Return only the sentence.\n\n"
                    + "Question:\n"
                    + question
                    + "\n\nCandidate answer:\n"
                    + answer
                    + "\n\nContext:\n"
                    + packet_text
                )
                format_id, answer, finish = ask(
                    f"legacy/sentence_format/{sid}", format_prompt, 128
                )
                if finish != "stop" or not answer.strip():
                    raise RuntimeError(
                        "Incomplete/empty sentence formatting is not propagated"
                    )
                answer = answer.strip()
            row = {"step_id": sid, "request": question, "answer": answer}
            node = {
                **row,
                "role": "goal" if is_goal else "ordinary",
                "prior_complete_answers": available,
                "excluded_prior_step_ids": [],
                "original_retrieval": original,
                "rewritten_query": rewrite,
                "rewritten_retrieval": rewritten,
                "reader_packet": packet,
                "rewrite_prompt": rewrite_prompt,
                "raw_rewrite": raw_rewrite,
                "rewrite_request_id": rewrite_id,
                "rewrite_rejection": rejection,
                "child_prompt": prompt,
                "answer_request_id": answer_id,
                "initial_answer": initial,
                "sentence_format_request_id": format_id,
                "is_whole_no_answer": None,
            }
            if is_goal:
                node["raw_B0_child_prompt"] = raw_child
            history.append(row)
            nodes.append(node)
            if record_node:
                record_node(deepcopy(node))
        prompt, context, block = render_final(
            case["question"], nodes, root, self.readout, self.final_examples
        )
        rid, prediction, finish = ask(
            "legacy/final", prompt, 64, allow_final_truncation=True
        )
        if finish not in {"stop", "length"}:
            raise RuntimeError("Final generation not successful")
        return {
            **case,
            "method": METHOD,
            "status": "success" if finish == "stop" else "success_truncated",
            "nodes": nodes,
            "history_all": history,
            "helper": block,
            "original_retrieval": root,
            "final_prompt": prompt,
            "final_context": context,
            "prediction": prediction.strip(),
            "request_id": rid,
            "finish_reason": finish,
        }


def execute(case, retrieve, ask, *, record_node=None):
    """Validate a standalone case and run it with the fixed prompt resources."""
    return Pipeline().execute(safe_case(case), retrieve, ask, record_node=record_node)


def execute_arm(case,retrieve,ask,arm,*,pipeline=None,record_node=None,record_event=None):
    if arm not in ARMS:raise ValueError(arm)
    clean=safe_case(case)
    if record_event:record_event(dict(event="reader_selection_arm",arm=arm,scope="allordinaryandgoal"))
    return ReaderPipeline(pipeline or Pipeline(),arm).execute(clean,retrieve,ask,record_node=record_node)
