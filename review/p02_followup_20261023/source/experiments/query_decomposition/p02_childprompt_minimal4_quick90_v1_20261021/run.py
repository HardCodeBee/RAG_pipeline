"""Isolated fixed90 execution using the existing native-P02 quick-evaluation engine."""
from contextlib import closing
from pathlib import Path
from threading import Lock
import argparse
import ast
import difflib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021 import run as engine
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021.run_r002 import RecordingLLM
from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021.run_r004 import RecoveryAsk as ConnectionAsk
from experiments.query_decomposition.p02_frozen_no_refs_quick90_v1_20261021.recover import retire
from p02_multigrain_timing_cachefix_v1.core.cache import canonical, utc
from p02_multigrain_timing_cachefix_v1.core.llm import build_payload
from . import child_prompts

PACKAGE, COHORT, PLAN = engine.PACKAGE, engine.COHORT, engine.PLAN_DIR
OUT = engine.OUT.parent / HERE.name
read, rows, sha, save, emit = engine.read, engine.rows, engine.sha, engine.save, engine.emit
AUTHORITY = "User: 给出经过这四组最小改动后的prompt 然后再次在90题集上快速评估"
SUBMIT_LOCK = Lock()
LAST_SUBMIT = 0.0


def check():
    protocol = read(HERE / "PROTOCOL.frozen.json")
    for path, expected in protocol["pins"].items():
        assert sha(ROOT / path) == expected, path
    assert read(PLAN / "PLAN.json")["execution_approval"]["approved"] is False
    return protocol


def prepare():
    engine.check_pins()
    assert not (HERE / "PROTOCOL.frozen.json").exists()
    source = (PACKAGE / "core/pipeline.py").read_text(encoding="utf8")
    changed = source.replace("from . import METHOD, ROOT",
        "from p02_multigrain_timing_cachefix_v1.core import ROOT\nMETHOD = 'p02_childprompt_minimal4_quick90_v1'", 1)
    changed = changed.replace("from . import child_prompts, rewrite_prompts, final_prompts",
        "from p02_multigrain_timing_cachefix_v1.core import rewrite_prompts, final_prompts\nfrom . import child_prompts", 1)
    ast.parse(changed)
    (HERE / "pipeline.py").write_text(changed, encoding="utf8")
    (HERE / "PIPELINE_CHANGE.diff").write_text("".join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True), fromfile="frozen_P02", tofile="minimal4_candidate")), encoding="utf8")
    candidate = child_prompts.template() + "\n"
    (HERE / "ordinary_child_prompt.frozen.txt").write_text(candidate, encoding="utf8")
    old = child_prompts.frozen.CHILD_INSTRUCTIONS + "\n\nWorked examples:\n\n" + "\n\n".join(
        f"Example {i+1}\n{r['text']}" for i, r in enumerate(child_prompts.ORIGINAL_EXAMPLES))
    old += "\n\nActual task:\nOriginal question:\n{original_question}\n\nRelevant prior question-answer pairs:\n{prior_question_answer_pairs}\n\nCurrent subquestion #{current_step_id}:\n{current_request}\n\nRetrieved passages:\n{context}\n\nAnswer:\n"
    (HERE / "PROMPT_CHANGE.diff").write_text("".join(difflib.unified_diff(old.splitlines(True), candidate.splitlines(True),
        fromfile="frozen_ordinary_prompt", tofile="minimal4_ordinary_prompt")), encoding="utf8")
    previous = read(engine.HERE / "PROTOCOL.frozen.json")
    pins = {k: v for group in ["inputs", "original_package_pins", "hold_pins"] for k, v in previous[group].items()}
    dependencies = [HERE / n for n in ["run.py", "child_prompts.py", "pipeline.py", "ordinary_child_prompt.frozen.txt"]]
    dependencies += [engine.HERE / n for n in ["run.py", "run_r002.py", "run_r004.py"]]
    dependencies += [PACKAGE / "config.json", PACKAGE / "prompts/child_examples.json", PACKAGE / "prompts/final_examples.json"]
    pins.update({str(p.relative_to(ROOT)): sha(p) for p in dependencies})
    protocol = {"id": HERE.name, "utc": utc(), "authority": AUTHORITY, "planned": 90,
        "sample": "Existing quick90; 30 per dataset; frozen query-only R02 decompositions; no resampling.",
        "control": "Original frozen P02 exact-payload replay, not the previously revised prompt.",
        "changes": ["Replace numbered-reference resolution with descriptive-reference interpretation; retain prior terminology.",
                    "Add relation to the entity/scope match required for evidence correction.",
                    "Example1: self-contained current question; original goal becomes the1000mi comparison.",
                    "Example3: self-contained current question; original goal asks whether other species exist."],
        "preserved": "No only/exclusively additions, no label/placeholder/empty-history/whitespace edits; ordinary record IDs remain; goal, rewriter, final, formatter, model, native token caps, decomposition, retrieval and merge unchanged.",
        "execution": "Paired complete downstream flows. Changed intermediate answers may change later rewrites/retrievals. Reuse only exact successful full payloads from original P02 or this run.",
        "retry": "Serial provider, minimum3seconds; transport connect-before-send retries2; SDK POST retries0. Known transient HTTP408/429/5xx and proved-unsent connect errors at most5 logical attempts; unresolved sent requests held without automatic replay.",
        "metrics": "Native final Answer-F1/EM, paired gain/tie/loss, full90 denominator/missingness, tokens/estimated cost; control replay is not fresh inference or independent confirmation.",
        "destination": read(PACKAGE / "config.json")["generation"]["base_url"],
        "main_evaluation_remains_paused": True, "pins": pins}
    save(HERE / "PROTOCOL.frozen.json", protocol)
    # Record this explicit scoped approval; preserve all main execution flags.
    main = read(PLAN / "PLAN.json")
    assert main["execution_approval"]["approved"] is False
    entry = {"decision_id": HERE.name.upper(), "scope": protocol["sample"] + " Four reviewed ordinary-prompt changes only.",
        "protocol": str((HERE / "PROTOCOL.frozen.json").relative_to(ROOT)), "campaign_resume_authorized": False, "new_inference": True}
    main.setdefault("standalone_diagnostics", []).append(entry)
    save(PLAN / "PLAN.json", main)
    with (PLAN / "DECISIONS.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps({"id": entry["decision_id"], "utc": utc(), "authority": AUTHORITY,
            "status": "approved_scoped_quick90", **entry, "main_evaluation_remains_paused": True}, ensure_ascii=False) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    return {"prepared": True, "planned": 90, "prompt": str(HERE / "ordinary_child_prompt.frozen.txt")}


class SerialLLM(RecordingLLM):
    def request_once(self, payload):
        global LAST_SUBMIT
        with SUBMIT_LOCK:
            pause = max(0, LAST_SUBMIT + 3 - time.monotonic())
            if pause:
                time.sleep(pause)
            LAST_SUBMIT = time.monotonic()
            return super().request_once(payload)


class ScopedAsk(ConnectionAsk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        previous = self.client
        self.client = SerialLLM(self.settings, client=previous.client)
        previous.owns_client = False
        self.client.owns_client = True
        self.cache.db.execute("CREATE TABLE IF NOT EXISTS retired_requests AS SELECT * FROM requests WHERE 0")
        self.cache.db.commit()

    def __call__(self, stage, prompt, maximum, allow_final_truncation=False):
        payload = canonical(build_payload(self.settings, prompt, maximum))
        while True:
            try:
                return super().__call__(stage, prompt, maximum, allow_final_truncation=allow_final_truncation)
            except RuntimeError:
                row = self.cache.db.execute("SELECT id,status,result FROM requests WHERE payload=?", (payload,)).fetchone()
                result = json.loads(row[2]) if row and row[2] else {}
                retryable = row and (result.get("proved_not_submitted") or
                    row[1] == "failed" and result.get("http_status") in {408, 429, 500, 502, 503, 504})
                attempts = 1 + self.cache.db.execute("SELECT count(*) FROM retired_requests WHERE payload=?", (payload,)).fetchone()[0]
                if not retryable or attempts >= 5:
                    raise
                retire(self.cache.db, row[0], "scoped_quick90_bounded_known_transient_or_unsent_recovery")
                self.events({"event": "known_transient_recovery", "stage": stage,
                    "retired_request_id": row[0], "attempts_used": attempts, "maximum": 5})
                time.sleep(min(5 * attempts, 30))


def configure():
    engine.HERE, engine.OUT = HERE, OUT
    engine.MODULE = "experiments.query_decomposition." + HERE.name
    engine.check_pins = check
    engine.ScopedAsk = ScopedAsk
    from p02_multigrain_timing_cachefix_v1.core.retrieval import Retriever
    previous_load = Retriever.load
    loading = Lock()
    def serialized_load(self):
        with loading:
            return previous_load(self)
    Retriever.load = serialized_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "offline", "run", "report"])
    action = parser.parse_args().action
    if action == "prepare":
        emit(prepare())
        return
    configure()
    emit(getattr(engine, action)())


if __name__ == "__main__":
    main()
