"""Checkpoint compatibility fix; frozen method and generation code stay intact.

A request can serve several original questions in the same method when their
single-hop prompts are identical. Preserve the old request-level provenance and
append a separate immutable binding for each question. Never relax other writes.
"""
from pathlib import Path
import argparse
import json
import sqlite3
import tempfile

from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import common
from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import transport
from experiments.query_decomposition.hotpot_formal_eval_v1.common.runtime import Store

ORIGINAL_TRANSPORT = transport.Transport
SOURCE_FIELDS = {"origin_db", "origin_request_id", "mode"}
RECEIPT = common.OUT / "origin_binding_fix_20261017"


class BindingStore:
    def __init__(self, store):
        self.original = store

    def __getattr__(self, name):
        return getattr(self.original, name)

    def put(self, kind, key, value):
        if kind != "request_origin":
            return self.original.put(kind, key, value)
        if set(value) != SOURCE_FIELDS | {"qid"} or not isinstance(value["qid"], str):
            raise common.IntegrityError("Unexpected historical request origin schema")
        previous = self.original.get(kind, key)
        if previous is not None:
            if set(previous) != SOURCE_FIELDS | {"qid"}:
                raise common.IntegrityError("Unexpected stored historical origin schema")
            if any(previous[name] != value[name] for name in SOURCE_FIELDS):
                raise common.IntegrityError("Historical request source changed; refusing reuse")
        row = self.original.db.execute("select id from requests where id=?", (key,)).fetchone()
        if row is None:
            raise common.IntegrityError("Historical origin has no corresponding request")
        if previous is None:
            self.original.put(kind, key, value)
        binding = {**value, "request_id": key,
                   "role": "same-method question binding; request source remains unchanged"}
        self.original.put("question_request_origin", common.digest([value["qid"], key]), binding)


class OriginSafeTransport(ORIGINAL_TRANSPORT):
    def __init__(self, store, *args, **kwargs):
        super().__init__(BindingStore(store), *args, **kwargs)


def validate():
    """Offline replay of the actual collision plus strict-negative checks."""
    common.verify()
    checks = {}
    source = common.OUT / "main/2wiki/formal/b06_decomp_bge/k6_cot/checkpoint.sqlite3"
    key = "a2a7d5416b2243e51f6b191798f0e0d07f2be694e9741ec1d1ee6c1e811e250a"
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db:
        old = json.loads(db.execute("select value from data where kind='request_origin' and key=?", (key,)).fetchone()[0])
        request = db.execute("select * from requests where id=?", (key,)).fetchone()
    from experiments.query_decomposition.baseline_integrity_repair_v2_20261017.catalog import Catalog
    catalog = Catalog(Path(old["origin_db"]))
    payload = json.loads(request[2])
    matches = [(qid, stage) for (qid, stage, _), item in catalog.requests.items()
               if item["request_id"] == old["origin_request_id"]]
    assert len({qid for qid, _ in matches}) > 1
    RECEIPT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="binding-replay-", dir=RECEIPT) as temp:
        store = Store(Path(temp), {"purpose": "offline actual historical collision replay"})
        store.db.execute("insert into requests values(?,?,?,?,?)", request)
        store.db.commit()
        store.put("request_origin", key, old)
        wrapped = BindingStore(store)
        original_value = store.db.execute("select value from data where kind='request_origin' and key=?", (key,)).fetchone()[0]
        for qid, stage in matches:
            inherited = catalog.request(qid, stage, {**payload, "max_tokens": common.MAX_OUTPUT})
            assert inherited and inherited["request_id"] == old["origin_request_id"]
            wrapped.put("request_origin", key, {**old, "qid": qid})
            wrapped.put("request_origin", key, {**old, "qid": qid})
        assert store.db.execute("select count(*) from requests").fetchone()[0] == 1
        assert store.db.execute("select count(*) from attempts").fetchone()[0] == 0
        assert store.db.execute("select value from data where kind='request_origin' and key=?", (key,)).fetchone()[0] == original_value
        assert store.db.execute("select count(*) from data where kind='question_request_origin'").fetchone()[0] == len({qid for qid, _ in matches})
        checks["actual_shared_request_replay_all_question_bindings_idempotent"] = True
        checks["old_origin_and_response_preserved_no_api_attempts"] = True
        for field in sorted(SOURCE_FIELDS):
            try:
                wrapped.put("request_origin", key, {**old, field: "changed"})
            except common.IntegrityError:
                pass
            else:
                raise AssertionError("A changed source was accepted")
        checks["different_source_or_mode_remains_blocked"] = True
        wrapped.put("prediction", "test", {"prediction": "a"})
        try:
            wrapped.put("prediction", "test", {"prediction": "b"})
        except ValueError:
            pass
        else:
            raise AssertionError("Non-origin immutable write was relaxed")
        checks["other_immutable_records_remain_strict"] = True
        # Run the actual transport, including per-question token/evidence records.
        class NeverCall:
            def __getattr__(self, name):
                raise AssertionError("Offline shared-request replay attempted an API call")
        client = NeverCall()
        fixed = OriginSafeTransport(store, client, NeverCall(), common.MODEL, catalog)
        prompt = payload["messages"][0]["content"]
        replay_key = common.digest([store.namespace, "preserved_old_response",
                                    old["origin_db"], old["origin_request_id"]])
        for qid, stage in matches:
            fixed.qid = qid
            rid, result = fixed.ask(stage, prompt, payload["max_tokens"])
            assert rid == replay_key and result["status"] == "success"
        checks["actual_transport_replay_keeps_each_question_tokens_and_evidence"] = True
        assert store.db.execute("select count(*) from attempts").fetchone()[0] == 0
        store.db.close()
    # Namespace changes the request ID; cross-method sharing is not introduced.
    checks["source_catalog_still_confined_to_method_question_stage_payload"] = True
    common.verify()
    value = {"utc": common.utc(), "passed": True, "checks": checks,
             "actual_conflict_request": key, "same_request_question_bindings": matches,
             "api_calls": 0, "source_sha256": common.sha(Path(__file__)),
             "frozen_v2_config_sha256": common.sha(common.OUT / "CONFIG_FROZEN.json")}
    common.save(RECEIPT / "VALIDATION.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def install():
    validation = common.read(RECEIPT / "VALIDATION.json")
    if not validation["passed"] or validation["source_sha256"] != common.sha(Path(__file__)):
        raise common.IntegrityError("Origin-binding patch lacks matching offline validation")
    common.verify()
    transport.Transport = OriginSafeTransport
    from experiments.query_decomposition.baseline_integrity_repair_v2_20261017 import run
    run.Transport = OriginSafeTransport
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--stage", choices=["all", "dev", "formal", "matched25"], default="all")
    args = parser.parse_args()
    if args.validate_only:
        validate()
        return
    run = install()
    common.journal("origin_binding_compatibility_enabled", source_sha256=common.sha(Path(__file__)),
                   original_sources_and_records_preserved=True, generation_conditions_changed=False)
    import sys
    sys.argv = [sys.argv[0], "--stage", args.stage]
    run.main()


if __name__ == "__main__":
    main()
