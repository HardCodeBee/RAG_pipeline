"""Recover an actually returned length-limited final output; zero new API calls.

Score returned final text as with the frozen S02 reader. Keep the token-limit
flag; do not confuse it with provider missingness or request a replacement.
"""

import json
import sqlite3
from types import SimpleNamespace

from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import digest, canonical
from . import run as frozen
from .prepare import HERE, OUT, ROOT, OUTPUT, INPUTS, rows, load, write


def main():
    dataset = "musique"
    qid = "4hop1__151650_5274_458768_33637"
    path = OUT / "cases" / dataset / "dual_original_rewritten" / (qid + ".json")
    if path.exists():
        print("Already recovered; no replay")
        return
    directory = OUT / load(OUT / "API_STORES.json")[dataset]
    with sqlite3.connect((directory / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        identity = json.loads(db.execute("SELECT value FROM data WHERE kind='meta' AND key='identity'").fetchone()[0])
    owner = SimpleNamespace(directory=directory, identity=identity, namespace=str(directory.relative_to(OUTPUT)))
    proof = []
    def exact_returned(transport, stage, prompt, maximum):
        payload = {"model": transport.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": maximum, "extra_body": {"enable_thinking": False}}
        rid = digest([transport.store.namespace, payload])
        row = transport.store.db.execute("SELECT payload,status,result FROM requests WHERE id=?", (rid,)).fetchone()
        assert row is not None and row[1] == "success" and canonical(json.loads(row[0])) == canonical(payload)
        response = json.loads(row[2])
        assert response["finish_reason"] == "stop" or stage.endswith("/final") and response["finish_reason"] == "length"
        proof.append({"request_id": rid, "stage": stage, "finish_reason": response["finish_reason"], "exact_payload_verified": True})
        return rid, response["answer"].strip()
    class CachedRetrieval:
        resources = identity["resources"]
        store = owner
        def retrieve(self, question_id, query, k):
            packet = self.store.get("retrieval", digest([question_id, query, k, self.resources]))
            assert packet is not None, "Offline recovery cannot perform fresh retrieval"
            return packet
    case = next(r for r in rows(HERE / "cases.frozen.jsonl") if r["qid"] == qid)
    protocol = {**load(HERE / "protocol.json"), "matched_single_control": False}
    examples = load(ROOT / "experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json")
    original_out, original_ask = frozen.OUT, frozen.ask
    try:
        frozen.OUT = OUT / "offline_exact_replay"
        frozen.ask = exact_returned
        frozen.run_case(case, protocol, owner, CachedRetrieval(), None, protocol["model"], examples, load(INPUTS / dataset / "formal.positions.json"))
        recovered = load(frozen.OUT / "cases" / dataset / "dual_original_rewritten" / (qid + ".json"))
    finally:
        frozen.OUT, frozen.ask = original_out, original_ask
    assert proof[-1]["finish_reason"] == "length" and proof[-1]["stage"] == "dual_original_rewritten/final"
    recovered.update({"status": "success_truncated", "finish_reason": "length", "offline_recovered_exact_returned_final": True, "generation_changed": False})
    write(path, recovered)
    write(OUT / "OFFLINE_FINAL_RECOVERY.json", {"qid": qid, "dataset": dataset, "new_api_requests": 0, "new_retrieval": False,
        "reason": "strict stop guard discarded an actually returned64-token final response; retain its raw text for native F1 instead of calling it API missingness", "scoring_policy": "score the actual returned text without completion or replacement; retain length flag, consistent with S02 raw-final scoring", "exact_requests_replayed": proof, "final_request_id": recovered["request_id"], "original_failure_record_preserved": True})
    print("Recovered actual truncated final; no API calls or new retrieval")


if __name__ == "__main__":
    main()
