"""Ordinary RAG; owns its prompt and answering policy."""

PROMPT = "Answer the question using only the provided context.\nReturn only the short answer. Do not add citations or explanation.\n\nQuestion:\n{question}\n\nContext:\n{context}\n\nAnswer:"


def run(cases, retrieval, transport, store, config):
    packets = retrieval.retrieve_many([{"qid": c["qid"], "query": c["question"]} for c in cases], 10)
    for case, packet in zip(cases, packets):
        transport.qid = case['qid']
        if store.get("prediction", case["qid"]) is not None:
            continue
        prompt = PROMPT.format(question=case["question"], context="\n\n".join(doc["block"] for doc in packet["docs"]))
        request_id, result = transport.ask("final", prompt, 64)
        prediction = {"qid": case["qid"], "question": case["question"], "prediction": result.get("answer") if result["status"] == "success" else None,
                      "status": result["status"], "request_id": request_id, "finish_reason": result.get("finish_reason"), "final_doc_ids": [d["doc_id"] for d in packet["docs"]]}
        store.put("prediction", case["qid"], prediction)
        store.export()
