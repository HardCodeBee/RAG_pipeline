"""Dense Self-Ask; cumulative raw passages precede original question."""
import json
from pathlib import Path
import re

INSTRUCTIONS = "Solve the original Question using Follow-up questions and Intermediate answers. Use the actual retrieved passages. Once the passages and completed answers resolve the original Question, stop immediately with Final answer: and its short answer. Do not ask further questions to reconfirm known information or request details beyond the original Question. Output only the next continuation, never repeat the Question, completed record, or passages. Stop after the next Follow-up or Final answer. Never invent passages or answers for a Follow-up not yet retrieved. A Final answer must answer the original Question."


def parse(text, expect_answer, current_followup=None):
    # Chat models may echo the exact already-executed question before their
    # labelled answer. That echo is not a new search action. Only a matching
    # leading echo can be skipped, and only when a later explicit answer exists.
    leading = re.match(r"\s*Follow-up\s*:\s*([^\n]+)", text, re.I)
    if expect_answer and current_followup and leading:
        normalize = lambda value: ' '.join(value.split())
        if normalize(leading.group(1)) == normalize(current_followup):
            marker = re.search(r"(?m)^\s*(?:Intermediate answer|Final answer|So the final answer is)\s*:", text[leading.end():], re.I)
            if marker:
                text = text[leading.end()+marker.start():]
    follow = re.search(r"Follow-up\s*:\s*([^\n]+)", text, re.I)
    final = re.search(r"(?:Final answer|So the final answer is)\s*:\s*([^\n]+)", text, re.I)
    first = min([(match.start(), kind, match) for kind, match in [("follow", follow), ("final", final)] if match], default=None)
    intermediate = ""
    if expect_answer:
        prefix = text[:first[0]] if first else text
        intermediate = re.sub(r"^\s*Intermediate answer\s*:\s*", "", prefix, flags=re.I).strip()
    return intermediate, (first[1], first[2].group(1).strip()) if first else None


def run(cases, retrieval, transport, store, config):
    demo = "\n\n".join(e["selfask"] for e in json.loads(Path(__file__).with_name("examples.json").read_text(encoding="utf-8"))["examples"])
    for case in cases:
        transport.qid = case['qid']
        if store.get("prediction", case["qid"]) is not None:
            continue
        history, documents, searches, expect_answer, current_followup = [], {}, 0, False, None
        prediction = {"qid": case["qid"], "question": case["question"], "prediction": "", "status": "algorithm_failure", "reason": "no_final_within_budget"}
        # One planner call and at most an answer and planner call per search.
        # A combined answer+next-action still uses one call. Search cap is unchanged.
        for turn in range(2 * config["max_search"] + 1):
            context = "\n\n".join(d["block"] for d in documents.values())
            prompt = INSTRUCTIONS + "\n\n" + demo + "\n\nRetrieved passages:\n" + context + "\n\nQuestion: " + case["question"] + "\n" + "\n".join(history)
            if expect_answer:
                prompt += "\nNext continuation: answer the most recent Follow-up with Intermediate answer:. Then output Final answer: if the original Question is resolved. Only if a required fact is still missing, output Follow-up: with a new question."
            else:
                prompt += "\nNext continuation: output Final answer: if the original Question is resolved. Only if a required fact is still missing, output Follow-up: to request it. Do not repeat an Intermediate answer."
            if searches == config["max_search"]:
                prompt += "\nThe search budget is exhausted. No further Follow-up is allowed. Finish this continuation with Final answer: and your best short answer to the original Question from the available passages and completed answers."
            key, result = transport.ask(f"continuation_{turn+1}", prompt, 512)
            if result["status"] != "success":
                prediction.update(prediction=None, status=result["status"], reason="transport_failure", request_id=key)
                break
            intermediate, next_action = parse(result["answer"], expect_answer, current_followup)
            if expect_answer:
                if not intermediate and not (next_action and next_action[0] == "final"):
                    prediction["reason"] = "missing_intermediate_answer"
                    break
                if intermediate:
                    history.append("Intermediate answer: " + intermediate)
            store.put("continuation", f"{case['qid']}/{turn+1}", {"request_id": key, "intermediate": intermediate, "next_action": next_action})
            if next_action is None:
                # A returned intermediate may require another continuation, without a new search.
                if expect_answer:
                    expect_answer = False
                    continue
                prediction["reason"] = "invalid_continuation"
                break
            operation, argument = next_action
            if operation == "final":
                prediction.update(prediction=argument, status="success", reason=None, request_id=key, finish_reason=result.get("finish_reason"))
                break
            if searches == config["max_search"]:
                break
            history.append("Follow-up: " + argument)
            current_followup = argument
            packet = retrieval.retrieve(case["qid"], argument, config["k"])
            searches += 1
            for doc in packet["docs"]:
                documents.setdefault(doc["doc_id"], doc)
            expect_answer = True
        prediction.update(searches=searches, all_seen_doc_ids=list(documents))
        store.put("prediction", case["qid"], prediction)
        store.export()
