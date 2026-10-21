"""Real observations and private ReAct action/history parsing."""
import json
from pathlib import Path
import re

INSTRUCTIONS = "Solve the Question by alternating a Thought and one Action. Actions are Search[search query] and Finish[short final answer]. Search observations are returned by the program. Produce only the next Thought and Action; never invent an Observation."


def action(text):
    before_observation = re.split(r"\n\s*Observation\s*:", text, maxsplit=1, flags=re.I)[0]
    match = re.search(r"(?:Action\s*:\s*)?(Search|Finish)\[([^\]\n]+)\]", before_observation, re.I)
    if not match:
        return None
    return match.group(1).casefold(), match.group(2).strip(), before_observation[:match.end()]


def run(cases, retrieval, transport, store, config):
    demo = "\n\n".join(e["react"] for e in json.loads(Path(__file__).with_name("examples.json").read_text(encoding="utf-8"))["examples"])
    for case in cases:
        transport.qid = case['qid']
        if store.get("prediction", case["qid"]) is not None:
            continue
        history, searches, seen = [], 0, set()
        prediction = {"qid": case["qid"], "question": case["question"], "prediction": "", "status": "algorithm_failure", "reason": "no_finish_within_budget"}
        for turn in range(config["max_search"] + 1):
            prompt = INSTRUCTIONS + "\n\n" + demo + "\n\nQuestion: " + case["question"] + "\n" + "\n".join(history) + "\nNext Thought and Action:"
            key, result = transport.ask(f"action_{turn+1}", prompt, 512)
            if result["status"] != "success":
                prediction.update(prediction=None, status=result["status"], reason="transport_failure", request_id=key)
                break
            parsed = action(result["answer"])
            store.put("action", f"{case['qid']}/{turn+1}", {"request_id": key, "parsed": parsed})
            if parsed is None:
                prediction["reason"] = "invalid_action"
                break
            operation, argument, generated = parsed
            history.append(generated)
            if operation == "finish":
                prediction.update(prediction=argument, status="success", reason=None, request_id=key, finish_reason=result.get("finish_reason"))
                break
            if searches == config["max_search"]:
                break
            packet = retrieval.retrieve(case["qid"], argument, config["k"])
            searches += 1
            seen.update(d["doc_id"] for d in packet["docs"])
            history.append("Observation: " + "\n\n".join(d["block"] for d in packet["docs"]))
        prediction.update(searches=searches, all_seen_doc_ids=sorted(seen))
        store.put("prediction", case["qid"], prediction)
        store.export()
