"""Run the frozen method with exact caches and cumulative budget accounting."""

from contextlib import ExitStack, closing
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3

from . import ROOT, METHOD
from .cache import Cache, canonical, run_lock
from .llm import LLM, build_payload
from .pipeline import Pipeline, execute, safe_case


def read_rows(path):
    rows = [
        safe_case(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(rows) != len({row["id"] for row in rows}):
        raise ValueError("Duplicate query identity")
    return rows


def check_resources(config, rows):
    from .retrieval import embedding_source

    missing = []
    for dataset in sorted({row["dataset"] for row in rows}):
        resources = config["resources"][dataset]
        corpus, spec = resources["corpus"], resources["dense"]["bge"]
        paths = [
            corpus["manifest"],
            corpus["chunks"],
            corpus["offsets"],
            embedding_source(resources),
            spec["query_snapshot"],
        ]
        missing.extend(str(path) for path in paths if not Path(path).exists())
    if missing:
        raise FileNotFoundError(
            "Configure external frozen resources: " + ", ".join(missing)
        )


def positive_budget(maximum):
    if maximum is None or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Paid execution requires a finite positive --max-cost-cny")
    return maximum


def resource_identity(resources):
    """Compare active settings when opening checkpoints made by the older package."""
    return {
        "corpus": {
            key: resources["corpus"][key]
            for key in ("dataset", "manifest", "chunks", "offsets", "documents")
        },
        "dense": {
            "bge": {
                key: resources["dense"]["bge"][key]
                for key in (
                    "build",
                    "query_model",
                    "query_revision",
                    "query_snapshot",
                    "space",
                )
            }
        },
    }


def bind_identity(cache, identity):
    previous = cache.get("meta", "identity")
    if previous is None:
        cache.put("meta", "identity", identity)
        return
    config = previous["config"]
    previous = {
        **previous,
        "config": {
            **config,
            "resources": {
                dataset: resource_identity(resources)
                for dataset, resources in config["resources"].items()
            },
        },
    }
    if previous != identity:
        raise ValueError("Immutable checkpoint changed: meta/identity")


class Budget:
    def __init__(self, config, output, maximum):
        self.maximum = positive_budget(maximum)
        ledger = Path(config["budget"]["ledger"])
        if not ledger.is_file():
            raise FileNotFoundError(
                "Cumulative budget ledger is required: " + str(ledger)
            )
        self.db = sqlite3.connect(ledger, timeout=30)
        self.namespace = (
            "candidate_packages/" + METHOD + "/" + str(Path(output).resolve())
        )
        self.cap = config["budget"]["cumulative_cap_cny"]

    def reserve(self, request_id, request):
        reserve = (
            len(request["messages"][0]["content"].encode("utf8")) * 0.8e-6
            + request["max_tokens"] * 2e-6
        )
        self.db.execute("BEGIN IMMEDIATE")
        try:
            total = self.db.execute(
                "SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges"
            ).fetchone()[0]
            local = self.db.execute(
                "SELECT coalesce(sum(coalesce(actual,reserved)),0) FROM charges WHERE namespace=?",
                (self.namespace,),
            ).fetchone()[0]
            if total + reserve > self.cap or local + reserve > self.maximum:
                raise RuntimeError("Global or per-run budget ceiling; no submission")
            self.db.execute(
                "INSERT INTO charges(id,namespace,reserved,actual,status) VALUES(?,?,?,NULL,'reserved')",
                (request_id, self.namespace, reserve),
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def cancel(self, request_id):
        # Only used for a local failure before any provider submission.
        with self.db:
            self.db.execute(
                "DELETE FROM charges WHERE id=? AND status='reserved' AND actual IS NULL",
                (request_id,),
            )

    def finish(self, request_id, result):
        usage = result.get("usage")
        if usage and "prompt_tokens" in usage and "completion_tokens" in usage:
            actual = usage["prompt_tokens"] * 0.8e-6 + usage["completion_tokens"] * 2e-6
            with self.db:
                self.db.execute(
                    "UPDATE charges SET actual=?,status='returned_usage' WHERE id=?",
                    (actual, request_id),
                )
        # Unknown submissions keep their full reservation.

    def close(self):
        self.db.close()


class Ask:
    def __init__(self, cache, seed, config, budget, events):
        self.cache, self.seed = cache, seed
        self.settings = config["generation"]
        self.budget, self.events = budget, events
        self.client = None

    def __call__(self, stage, prompt, maximum, allow_final_truncation=False):
        request = build_payload(self.settings, prompt, maximum)
        seed_result = None
        reserved_id = None
        source, reused = "current_run", True

        def prepare(request_id):
            nonlocal seed_result, reserved_id, source, reused
            row = self.seed.execute(
                "SELECT request_id,result FROM requests WHERE payload=?",
                (canonical(request),),
            ).fetchone()
            if row:
                seed_result = json.loads(row[1])
                source = "frozen_replay/" + row[0]
            else:
                if self.client is None:
                    self.client = LLM(self.settings)
                self.budget.reserve(request_id, request)
                reserved_id = request_id
                source, reused = "provider", False

        try:
            request_id, result = self.cache.begin_request(
                stage, request, before_start=prepare
            )
        except BaseException:
            if reserved_id is not None:
                self.budget.cancel(reserved_id)
            raise
        if result is None:
            if seed_result is not None:
                result = seed_result
            else:
                result = self.client.request_once(request)
                self.budget.finish(request_id, result)
            self.cache.finish_request(request_id, result)
        self.events(
            {
                "event": "model_call",
                "stage": stage,
                "request_id": request_id,
                "reused": reused,
                "source": source,
                "status": result.get("status"),
                "usage": result.get("usage"),
            }
        )
        finish = result.get("finish_reason")
        if result.get("status") != "success" or not (
            finish == "stop" or allow_final_truncation and finish == "length"
        ):
            raise RuntimeError(
                "Recorded incomplete/failed/unknown request; no automatic retry: "
                + request_id
            )
        return request_id, result["answer"].strip(), finish

    def close(self):
        if self.client is not None:
            self.client.close()


def run(inputs, output, *, limit=None, max_cost=None, device="cpu"):
    positive_budget(max_cost)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf8"))
    rows = read_rows(inputs)[:limit]
    check_resources(config, rows)
    pipeline = Pipeline(config)
    from .retrieval import Retriever

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_uri = (ROOT / "data/replay.sqlite3").resolve().as_uri() + "?mode=ro"
    with run_lock(output), closing(sqlite3.connect(seed_uri, uri=True)) as seed, Cache(
        output / "checkpoint.sqlite3"
    ) as cache:
        bind_identity(cache, {"method": METHOD, "config": config, "questions": rows})

        def event(value):
            with (output / "trace.jsonl").open("a", encoding="utf8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")

        with ExitStack() as stack:
            budget = stack.enter_context(closing(Budget(config, output, max_cost)))
            ask = stack.enter_context(closing(Ask(cache, seed, config, budget, event)))
            retrievers = {}
            try:
                for case in rows:
                    if cache.get("prediction", case["id"]) is not None:
                        continue
                    resources = config["resources"][case["dataset"]]

                    def retrieve(query):
                        # The checkpoint identity already fixes all resource settings.
                        key = canonical([case["dataset"], case["qid"], query, 10])
                        packet = cache.get("retrieval", key)
                        if packet is None:
                            saved = seed.execute(
                                "SELECT packet FROM retrieval WHERE dataset=? AND qid=? AND query=?",
                                (case["dataset"], case["qid"], query),
                            ).fetchone()
                            if saved:
                                packet = json.loads(saved[0])
                            else:
                                if case["dataset"] not in retrievers:
                                    retriever = Retriever(resources, device=device)
                                    stack.callback(retriever.close)
                                    retrievers[case["dataset"]] = retriever
                                packet = retrievers[case["dataset"]](case["qid"], query)
                            cache.put("retrieval", key, packet)
                        return deepcopy(packet)

                    result = pipeline.execute(
                        case,
                        retrieve,
                        ask,
                        record_node=lambda node: cache.put(
                            "node", case["id"] + "/" + str(node["step_id"]), node
                        ),
                    )
                    cache.put("prediction", case["id"], result)
                    event(
                        {
                            "event": "case_complete",
                            "id": case["id"],
                            "status": result["status"],
                        }
                    )
            finally:
                predictions = list(cache.all("prediction").values())
                (output / "results.jsonl").write_text(
                    "".join(
                        json.dumps(value, ensure_ascii=False) + "\n"
                        for value in predictions
                    ),
                    encoding="utf8",
                )
    return {
        "method": METHOD,
        "planned": len(rows),
        "completed": len(predictions),
        "output": str(output),
    }
