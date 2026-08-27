"""Complete only the missing train BM25/Dense outcomes for the Phase 2.7 9,600 audit."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_router_phase1 import (  # noqa: E402
    GENERATOR_PRICES,
    _core_config,
    _load_router_config,
    _prompt_for_row,
    _usage_cost,
)
from src.evaluators.hotpot_answer import answer_metrics  # noqa: E402
from src.generators.answer_generator import LLMGenerator  # noqa: E402


PROTOCOL_ID = "hotpotqa_bd_router_phase27_9600_completion_v1"
ACTIONS = ("bm25", "dense")
QUERY_LIMIT = 9600
DEFAULT_RUN_DIR = (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase3_f1_pairwise_v1"
)
DEFAULT_DECISION = (
    "outputs/router/hotpotqa_bd_router_v1/runs/phase27_model_audit_v1/decision.json"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="outputs/router/hotpotqa_bd_router_v1/config.yaml"
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--decision", default=DEFAULT_DECISION)
    parser.add_argument("--stage", choices=("status", "generate"), required=True)
    parser.add_argument("--max-new-calls", type=int, default=None)
    parser.add_argument("--hard-budget-usd", type=float, default=4.50)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _connect(run_dir: Path, *, read_only: bool = False) -> sqlite3.Connection:
    path = (run_dir / "state.sqlite3").resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if read_only:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=60.0)
    else:
        connection = sqlite3.connect(path, timeout=60.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_authorization(decision_path: Path) -> dict[str, Any]:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != "REQUEST_9600_AUTHORIZATION":
        raise ValueError("Stage 9 did not request 9,600 authorization")
    if decision.get("does_not_authorize_paid_work") is not True:
        raise ValueError("Decision artifact has an unexpected authorization boundary")
    return decision


def _counts(connection: sqlite3.Connection) -> dict[str, dict[str, int]]:
    rows = connection.execute(
        """
        SELECT o.action, o.status, COUNT(*)
        FROM outcomes AS o
        JOIN sample_rows AS s ON s.query_id = o.query_id
        WHERE o.partition = 'train'
          AND s.query_rank < ?
          AND o.action IN ('bm25', 'dense')
        GROUP BY o.action, o.status
        ORDER BY o.action, o.status
        """,
        (QUERY_LIMIT,),
    )
    result = {action: {} for action in ACTIONS}
    for action, status_name, count in rows:
        result[str(action)][str(status_name)] = int(count)
    return result


def _protocol_usage(connection: sqlite3.Connection) -> dict[str, Any]:
    calls = 0
    successes = 0
    failures = 0
    input_tokens = 0
    output_tokens = 0
    spent = 0.0
    rows = connection.execute(
        """
        SELECT status, payload
        FROM outcomes
        WHERE partition = 'train' AND action IN ('bm25', 'dense')
        """
    )
    for status_name, payload_text in rows:
        payload = json.loads(payload_text)
        if payload.get("completion_protocol") != PROTOCOL_ID:
            continue
        calls += 1
        successes += status_name == "success"
        failures += status_name == "generation_failure"
        generation = payload.get("generation")
        usage = generation.get("token_usage", {}) if isinstance(generation, Mapping) else {}
        provider = usage.get("provider_reported") if isinstance(usage, Mapping) else None
        if isinstance(provider, Mapping):
            if isinstance(provider.get("input_tokens"), int):
                input_tokens += int(provider["input_tokens"])
            if isinstance(provider.get("output_tokens"), int):
                output_tokens += int(provider["output_tokens"])
        spent += _usage_cost(usage, GENERATOR_PRICES)
    return {
        "attempted_cells": calls,
        "successful_cells": successes,
        "failed_cells": failures,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": spent,
    }


def status(run_dir: Path) -> dict[str, Any]:
    connection = _connect(run_dir, read_only=True)
    try:
        counts = _counts(connection)
        fresh_dev = int(
            connection.execute(
                "SELECT COUNT(*) FROM outcomes WHERE partition = 'fresh_dev'"
            ).fetchone()[0]
        )
        train_queries = int(
            connection.execute(
                "SELECT COUNT(*) FROM sample_rows WHERE partition = 'train'"
            ).fetchone()[0]
        )
        protocol_usage = _protocol_usage(connection)
    finally:
        connection.close()
    pending = sum(value.get("pending_generation", 0) for value in counts.values())
    failed = sum(value.get("generation_failure", 0) for value in counts.values())
    return {
        "protocol_id": PROTOCOL_ID,
        "query_limit": QUERY_LIMIT,
        "actions": list(ACTIONS),
        "train_sample_queries": train_queries,
        "counts": counts,
        "pending_or_retryable_cells": pending + failed,
        "fresh_dev_outcomes": fresh_dev,
        "protocol_usage": protocol_usage,
        "complete": pending == 0 and failed == 0,
    }


def _pending_rows(
    connection: sqlite3.Connection, *, max_new_calls: int | None
) -> list[tuple[str, str, int, str]]:
    rows = connection.execute(
        """
        SELECT o.query_id, o.action, o.repeat_id, o.payload
        FROM outcomes AS o
        JOIN sample_rows AS s ON s.query_id = o.query_id
        WHERE o.partition = 'train'
          AND s.query_rank < ?
          AND o.action IN ('bm25', 'dense')
          AND o.status IN ('pending_generation', 'generation_failure')
        ORDER BY s.query_rank,
                 CASE o.action WHEN 'bm25' THEN 0 ELSE 1 END,
                 o.repeat_id
        """,
        (QUERY_LIMIT,),
    ).fetchall()
    if max_new_calls is not None:
        if max_new_calls <= 0:
            raise ValueError("max-new-calls must be positive")
        rows = rows[:max_new_calls]
    return [(str(q), str(a), int(r), str(p)) for q, a, r, p in rows]


def _generator_view(result: Any) -> dict[str, Any]:
    return {
        "attempted": True,
        "status": "success",
        "latency_ms": result.latency_ms,
        "token_usage": result.token_usage,
    }


def generate(
    router: Mapping[str, Any],
    run_dir: Path,
    *,
    max_new_calls: int | None,
    hard_budget_usd: float,
    workers: int,
) -> dict[str, Any]:
    if hard_budget_usd <= 0.0:
        raise ValueError("hard-budget-usd must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    generation = _core_config(router, method="bm25")["generation"]
    connection = _connect(run_dir)
    local = threading.local()
    created: list[LLMGenerator] = []
    created_lock = threading.Lock()
    try:
        before = status(run_dir)
        if before["fresh_dev_outcomes"] != 0:
            raise RuntimeError("Fresh-dev outcomes must remain sealed")
        prior_usage = _protocol_usage(connection)
        spent = float(prior_usage["estimated_cost_usd"])
        if spent >= hard_budget_usd:
            raise RuntimeError("Phase 2.7 completion budget is already exhausted")
        pending = _pending_rows(connection, max_new_calls=max_new_calls)
        if not pending:
            return before

        def worker(item: Sequence[Any]) -> tuple[str, str, int, dict[str, Any]]:
            query_id, action, repeat_id, payload_text = item
            generator = getattr(local, "generator", None)
            if generator is None:
                generator = LLMGenerator(
                    provider=generation["provider"],
                    model=generation["model"],
                    temperature=generation["temperature"],
                    max_output_tokens=generation["max_output_tokens"],
                    timeout_seconds=generation["timeout_seconds"],
                    max_retries=generation["max_retries"],
                )
                local.generator = generator
                with created_lock:
                    created.append(generator)
            row = json.loads(str(payload_text))
            row["completion_protocol"] = PROTOCOL_ID
            try:
                result = generator.generate_from_prompt(
                    _prompt_for_row(row, str(router["prompt"]["version"])),
                    row["question"],
                    [],
                )
                prediction = result.answer.strip()
                metrics = answer_metrics(prediction, row["reference_answers"])
                metrics.pop("answer_correctness", None)
                row.update(
                    {
                        "status": "success",
                        "prediction": prediction,
                        "metrics": metrics,
                        "generation": _generator_view(result),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "generation_failure",
                        "generation": {
                            "attempted": True,
                            "status": "failure",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    }
                )
            return str(query_id), str(action), int(repeat_id), row

        iterator = iter(pending)
        futures: dict[Future[Any], tuple[str, str, int, str]] = {}
        completed = 0

        def submit_one(pool: ThreadPoolExecutor) -> bool:
            if spent >= hard_budget_usd:
                return False
            try:
                item = next(iterator)
            except StopIteration:
                return False
            futures[pool.submit(worker, item)] = item
            return True

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(min(workers, len(pending))):
                submit_one(pool)
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    query_id, action, repeat_id, row = future.result()
                    connection.execute(
                        """
                        UPDATE outcomes SET status = ?, payload = ?
                        WHERE partition = 'train'
                          AND query_id = ? AND action = ? AND repeat_id = ?
                        """,
                        (
                            str(row["status"]),
                            json.dumps(row, ensure_ascii=False, allow_nan=False),
                            query_id,
                            action,
                            repeat_id,
                        ),
                    )
                    completed += 1
                    generation_view = row.get("generation")
                    if row["status"] == "success" and isinstance(generation_view, Mapping):
                        spent += _usage_cost(
                            generation_view.get("token_usage", {}), GENERATOR_PRICES
                        )
                    if completed % 50 == 0:
                        connection.commit()
                    if completed % 100 == 0:
                        print(
                            json.dumps(
                                {
                                    "protocol_id": PROTOCOL_ID,
                                    "generation_new_cells": completed,
                                    "selected_cells": len(pending),
                                    "estimated_protocol_cost_usd": spent,
                                }
                            ),
                            flush=True,
                        )
                    submit_one(pool)
        connection.commit()
    finally:
        connection.commit()
        connection.close()
        for generator in created:
            close = getattr(getattr(generator, "_client", None), "close", None)
            if callable(close):
                close()

    after = status(run_dir)
    state = {
        **after,
        "authorized_action_scope": list(ACTIONS),
        "provider": generation["provider"],
        "model": generation["model"],
        "prompt_version": router["prompt"]["version"],
        "temperature": generation["temperature"],
        "max_output_tokens": generation["max_output_tokens"],
        "workers": workers,
        "hard_budget_usd": hard_budget_usd,
        "updated_at_unix": time.time(),
    }
    _write_json(run_dir / "phase27_9600_completion_state.json", state)
    if float(after["protocol_usage"]["estimated_cost_usd"]) > hard_budget_usd:
        raise RuntimeError("Phase 2.7 completion exceeded the hard budget")
    return state


def main() -> int:
    args = _arguments()
    config_path = _resolve(args.config)
    run_dir = _resolve(args.run_dir)
    decision_path = _resolve(args.decision)
    router = _load_router_config(config_path)
    _validate_authorization(decision_path)
    if tuple(GENERATOR_PRICES) != (0.40, 1.60):
        raise ValueError("GPT-4.1 mini price constants changed; re-audit before paid work")
    workers = int(args.workers or router["phase3"]["provider_workers"])
    if args.stage == "status":
        result = status(run_dir)
    else:
        result = generate(
            router,
            run_dir,
            max_new_calls=args.max_new_calls,
            hard_budget_usd=float(args.hard_budget_usd),
            workers=workers,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
