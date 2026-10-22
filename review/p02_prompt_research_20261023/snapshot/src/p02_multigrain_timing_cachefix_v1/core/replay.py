"""Full recorded-payload replay; never opens API, model or corpus resources."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from . import ROOT, METHOD
from .pipeline import Pipeline, safe_case
from .cache import canonical
from .llm import build_payload
from contextlib import closing


def verify():
    record = json.loads((ROOT / "FROZEN.json").read_text(encoding="utf8"))
    for name in record["files"]:
        if not (ROOT / name).is_file():
            raise FileNotFoundError("Missing package file: " + name)
    if record["method"] != METHOD:
        raise ValueError("Package method identity mismatch")
    return {
        "method": METHOD,
        "passed": True,
        "files_checked": len(record["files"]),
        "file_check": "existence_only; behavior checked by replay",
        "new_API_calls": 0,
    }


def replay(output=None, limit=None):
    checked = verify()
    counts = Counter()
    scores = defaultdict(list)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf8"))
    pipeline = Pipeline(config)
    with closing(
        sqlite3.connect(
            (ROOT / "data/replay.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
        )
    ) as db:
        expected = [
            json.loads(s)
            for s in (ROOT / "data/expected300.jsonl")
            .read_text(encoding="utf8")
            .splitlines()
        ]
        runtime = [
            json.loads(s)
            for s in (ROOT / "data/runtime300.jsonl")
            .read_text(encoding="utf8")
            .splitlines()
        ]
        by_id = {v["id"]: v for v in expected}
        for case in runtime[:limit]:
            source = by_id[case["id"]]
            calls = []
            queries = []

            def retrieve(query):
                row = db.execute(
                    "SELECT packet FROM retrieval WHERE dataset=? AND qid=? AND query=?",
                    (case["dataset"], case["qid"], query),
                ).fetchone()
                if row is None:
                    raise ValueError("Replay requires a new retrieval: " + query)
                queries.append(query)
                return json.loads(row[0])

            def ask(stage, prompt, maximum, allow_final_truncation=False):
                request = build_payload(config["generation"], prompt, maximum)
                row = db.execute(
                    "SELECT request_id,result FROM requests WHERE payload=?",
                    (canonical(request),),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        "Replay requires a new model input: " + case["id"] + " " + stage
                    )
                result = json.loads(row[1])
                assert result["status"] == "success"
                finish = result["finish_reason"]
                assert finish == "stop" or (
                    allow_final_truncation and finish == "length"
                )
                calls.append({"stage": stage, "request_id": row[0]})
                return row[0], result["answer"].strip(), finish

            actual = pipeline.execute(safe_case(case), retrieve, ask)
            for key in ["prediction", "final_prompt", "final_context", "finish_reason"]:
                assert actual[key] == source[key], (case["id"], key)
            assert len(actual["nodes"]) == len(source["nodes"])
            for node, previous in zip(actual["nodes"], source["nodes"]):
                for key, value in previous.items():
                    assert node.get(key) == value, (case["id"], node["step_id"], key)
            assert actual["original_retrieval"] == source["original_retrieval"]
            assert len(calls) == source["logical_model_calls"]
            counts.update(
                cases=1,
                nodes=len(actual["nodes"]),
                logical_model_calls=len(calls),
                logical_retrievals=len(queries),
            )
            scores[case["dataset"]].append(source["native_answer_f1"])
    result = {
        **checked,
        "counts": dict(counts),
        "all_prompts_outputs_and_packets_exact": True,
        "native_Answer_F1": sum(sum(v) for v in scores.values()) / counts["cases"],
        "by_dataset": {
            ds: {"n": len(v), "F1": sum(v) / len(v)} for ds, v in scores.items()
        },
        "scope": "Recorded exact-input replay, not fresh inference or independent generalization",
        "score_scope": "Recorded native scores, verified predictions are identical; no new judge",
        "new_API_calls": 0,
        "new_index_searches": 0,
        "new_remote_calls": 0,
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("Keep prior verification: " + str(path))
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
        )
    return result
