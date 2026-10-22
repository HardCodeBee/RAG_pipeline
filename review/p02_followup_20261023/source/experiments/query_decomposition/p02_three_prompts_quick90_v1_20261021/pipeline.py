"""One frozen multigrain candidate; no ancestor experiment imports."""

from copy import deepcopy
import json

from p02_multigrain_timing_cachefix_v1.core import ROOT
METHOD = 'p02_three_prompts_user_v1'
from p02_multigrain_timing_cachefix_v1.core import rewrite_prompts
from . import prompts as child_prompts, prompts as final_prompts


def safe_case(case):
    forbidden = {
        "answer",
        "answers",
        "gold",
        "gold_answer",
        "supporting_facts",
        "gold_documents",
        "question_decomposition",
        "prediction",
        "f1",
    }
    if forbidden.intersection(case):
        raise ValueError("Reference/result fields cannot enter inference")
    result = {
        key: deepcopy(case[key])
        for key in ["id", "qid", "dataset", "question", "supporting_questions"]
    }
    if (
        result["dataset"] not in {"hotpotqa", "2wiki", "musique"}
        or result["id"] != result["dataset"] + ":" + result["qid"]
    ):
        raise ValueError("Dataset/query identity mismatch")
    if not isinstance(result["question"], str) or not result["question"].strip():
        raise ValueError("Empty original question")
    if not isinstance(result["supporting_questions"], list) or not all(
        isinstance(q, str) and q.strip() for q in result["supporting_questions"]
    ):
        raise ValueError("supporting_questions must be an ordered string array")
    return result


def passages(packet):
    return "\n\n".join(d["block"] for d in packet["docs"])


def merge(original, rewritten):
    if rewritten is None:
        return deepcopy(original)
    docs, seen = [], set()
    for rank in range(10):
        for route in (original, rewritten):
            if rank < len(route["docs"]):
                d = route["docs"][rank]
                if d["doc_id"] not in seen:
                    seen.add(d["doc_id"])
                    docs.append(d)
    docs = [
        {
            **d,
            "rank": i,
            "block": f'[Chunk {i} | source={d["source"]}, pages=1-1]\n{d["text"]}',
        }
        for i, d in enumerate(docs[:10], 1)
    ]
    if len(docs) != 10:
        raise ValueError("Merged retrieval must contain ten documents")
    return {
        "qid": original["qid"],
        "query": original["query"],
        "k": 10,
        "docs": docs,
        "merge_policy": "interleave_original_first_top10",
    }


def helper(nodes):
    ordinary, goals = nodes[:-1], nodes[-1:]
    sections = ["Auxiliary question-answer pairs"]
    if ordinary:
        sections += [
            "Intermediate candidate results",
            "\n\n".join(
                f"#{n['step_id']} Question: {n['request']}\nAnswer: {n['answer']}"
                for n in ordinary
            ),
        ]
    sections += [
        "Candidate conclusion for the Original question",
        "\n\n".join(
            f"#{n['step_id']} Question: {n['request']}\nAnswer: {n['answer']}"
            for n in goals
        ),
    ]
    return "\n\n".join(sections)


def render_final(question, nodes, root, readout, examples):
    block = helper(nodes)
    context = block + "\n\nRetrieved passages:\n" + passages(root)
    prompt = final_prompts.final(question, context, examples, readout)
    return prompt, context, block


class Pipeline:
    """Load fixed prompt resources once; execute cases already checked by safe_case."""

    def __init__(self, config=None):
        if config is None:
            config = json.loads((ROOT / "config.json").read_text(encoding="utf8"))
        self.readout = config["readout"]
        self.child_examples = json.loads(
            (ROOT / "prompts/child_examples.json").read_text(encoding="utf8")
        )
        self.final_examples = json.loads(
            (ROOT / "prompts/final_examples.json").read_text(encoding="utf8")
        )

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
            packet = merge(original, rewritten)
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
