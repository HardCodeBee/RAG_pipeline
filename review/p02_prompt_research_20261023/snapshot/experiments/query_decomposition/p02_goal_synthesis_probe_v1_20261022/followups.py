"""Two independently reported goal controls on the existing frozen pilot15.

These controls use G0's common512 interface, not native historical P02.
No frozen generation or rendering code is modified.
"""
from argparse import ArgumentParser
from contextlib import closing
from pathlib import Path
import hashlib
import json
import sqlite3

from . import run as base

PROTOCOL = base.HERE / "FOLLOWUP_PROTOCOL.frozen.json"


def prepare():
    base.check()
    value = {
        "id": "p02_goal_controls_pilot15_20261022",
        "authority": "P02_PROMPT_RESEARCH_AUTONOMOUS_SMALL_EVALUATIONS_APPROVED_20261022",
        "base_protocol_sha256": base.sha(base.HERE / "PROTOCOL.frozen.json"),
        "script_sha256": base.sha(Path(__file__)),
        "planned_queries_per_control": 15,
        "goal_rewrite_off": "Only substitute original-route packet text in the historical G0 goal prompt; regenerate goal and final at common512/64 settings.",
        "final_goal_none": "Retain original ordinary QA and root passages; omit only candidate goal information from final. Goal computing expense is not claimed saved by this visibility intervention.",
        "control": "G0 from pilot15, with common512 cap, no sentence formatter and the same candidate-information wrapper; historical native P02 is separate.",
        "main_evaluation_remains_paused": True,
    }
    if PROTOCOL.exists():
        assert base.read(PROTOCOL) == value
    else:
        base.save(PROTOCOL, value)
    return value


def check():
    base.check()
    value = base.read(PROTOCOL)
    assert value["base_protocol_sha256"] == base.sha(base.HERE / "PROTOCOL.frozen.json")
    assert value["script_sha256"] == base.sha(Path(__file__))
    return value


def coverage(run_id):
    path = base.OUT / run_id / "checkpoint.sqlite3"
    if not path.exists():
        return 0
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return db.execute("SELECT count(*) FROM data WHERE kind='prediction'").fetchone()[0]


def execute(which):
    check()
    original_report = base.report

    def control_report(run_id):
        value = original_report(run_id)
        value["expected_predictions"] = 15
        value["completed_predictions"] = coverage(run_id)
        value["status"] = "complete" if value["completed_predictions"] == 15 else "partial"
        value["followup_protocol_sha256"] = base.sha(PROTOCOL)
        value["interpretation"] = (
            "Goal information visibility, not deletion of goal computation."
            if run_id == "final_goal_none" else
            "Goal original-route versus dual-route evidence at the common512 interface."
        )
        base.save(base.OUT / run_id / "REPORT.json", value)
        return value

    base.report = control_report
    try:
        if which == "rewrite-off":
            return base.execute(["G0"], rewrite_off=True)
        return base.execute([], final_none=True)
    finally:
        base.report = original_report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "rewrite-off", "final-none"))
    args = parser.parse_args()
    result = prepare() if args.action == "prepare" else execute(args.action)
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
