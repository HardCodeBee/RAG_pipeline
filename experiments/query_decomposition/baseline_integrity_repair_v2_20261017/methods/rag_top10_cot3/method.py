"""Single retrieval with a private three-example CoT reader."""
import json
from pathlib import Path
import re

INSTRUCTIONS = "Answer the Question using the retrieved passages. Reason through the required relations or comparison, then end with Final answer: followed by the short answer."


def examples():
    path = Path(__file__).with_name("examples.json")
    return "\n\n".join(e["cot"] for e in json.loads(path.read_text(encoding="utf-8"))["examples"])


from ...answers import extract


def run(cases, retrieval, transport, store, config):
    demo = examples()
    packets = retrieval.retrieve_many([{"qid": c["qid"], "query": c["question"]} for c in cases], 10)
    for case, packet in zip(cases, packets):
        transport.qid = case['qid']
        if store.get("prediction", case["qid"]) is not None:
            continue
        prompt = INSTRUCTIONS + "\n\n" + demo + "\n\nRetrieved passages:\n" + "\n\n".join(d["block"] for d in packet["docs"]) + "\n\nQuestion: " + case["question"] + "\nAnswer:"
        key, result = transport.ask("final_cot", prompt, 512)
        prediction = {"qid": case["qid"], "question": case["question"], "prediction": extract(result["answer"]) if result["status"] == "success" else None, "status": result["status"], "request_id": key, "finish_reason": result.get("finish_reason"), "final_doc_ids": [d["doc_id"] for d in packet["docs"]]}
        if prediction["status"]=="success" and prediction["prediction"]=="":
            prediction.update(status="algorithm_failure",reason="missing_final_answer")
        store.put("prediction", case["qid"], prediction)
        store.export()
