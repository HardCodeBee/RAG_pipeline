#!/usr/bin/env python3
"""Inspect and verify the local HotpotQA Router experiment workspace."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.router_experiments.data import load_yaml, read_json, resolve_path, sha256


REGISTRY = "analysis/hotpotqa_router/registry.yaml"
MARKDOWN_LINK = re.compile(r"\]\(([^)]+)\)")


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="show frozen phase status")
    verify = subparsers.add_parser("verify", help="verify local records and artifacts")
    verify.add_argument("phase", nargs="?", help="optional registry phase id")
    rebuild = subparsers.add_parser("rebuild", help="run a deterministic local rebuild")
    rebuild.add_argument("target", choices=("phase28_views",))
    return parser.parse_args(argv)


def load_registry() -> dict[str, Any]:
    registry = load_yaml(REGISTRY)
    if registry.get("storage") != "local_only":
        raise ValueError("Router registry must remain local-only")
    phases = registry.get("phases")
    if not isinstance(phases, Mapping) or not phases:
        raise ValueError("Router registry has no phase entries")
    return registry


def _phase_entries(
    registry: Mapping[str, Any], phase: str | None = None
) -> list[tuple[str, Mapping[str, Any]]]:
    phases = registry["phases"]
    if phase is not None:
        if phase not in phases:
            raise KeyError(f"Unknown phase: {phase}")
        return [(phase, phases[phase])]
    return list(phases.items())


def status_rows(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "phase": phase,
            "spec": entry["spec_status"],
            "result": entry["result_status"],
            "decision": entry["decision"],
            "deployable": bool(entry["deployable"]),
            "consumed": bool(entry.get("consumed", False)),
        }
        for phase, entry in _phase_entries(registry)
    ]


def verify_registry(
    registry: Mapping[str, Any], phase: str | None = None
) -> list[str]:
    failures: list[str] = []
    for phase_id, entry in _phase_entries(registry, phase):
        paths = [entry["config"], entry["conclusion"], entry["results"]]
        paths.extend(entry.get("required_paths", []))
        for value in paths:
            if not resolve_path(value).exists():
                failures.append(f"{phase_id}: missing {value}")

        conclusion = resolve_path(entry["conclusion"])
        if conclusion.exists():
            for target in MARKDOWN_LINK.findall(conclusion.read_text(encoding="utf-8")):
                target = target.split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                if not (conclusion.parent / target).resolve().exists():
                    failures.append(f"{phase_id}: broken conclusion link {target}")

        expected_hash = entry.get("config_sha256")
        if expected_hash and resolve_path(entry["config"]).exists():
            actual_hash = sha256(entry["config"])
            if actual_hash != expected_hash:
                failures.append(
                    f"{phase_id}: config sha256 {actual_hash} != {expected_hash}"
                )

        check = entry.get("decision_check")
        if check:
            path = check["path"]
            if not resolve_path(path).exists():
                failures.append(f"{phase_id}: missing decision file {path}")
            else:
                actual = read_json(path).get(check["field"])
                if actual != check["equals"]:
                    failures.append(
                        f"{phase_id}: {check['field']}={actual!r}, "
                        f"expected {check['equals']!r}"
                    )
    return failures


def _print_status(registry: Mapping[str, Any]) -> None:
    print("phase\tspec\tresult\tdeployable\tconsumed\tdecision")
    for row in status_rows(registry):
        print(
            f"{row['phase']}\t{row['spec']}\t{row['result']}\t"
            f"{str(row['deployable']).lower()}\t{str(row['consumed']).lower()}\t"
            f"{row['decision']}"
        )


def _rebuild_phase28_views() -> int:
    output_dir = PROJECT_ROOT / "tmp/router_rebuild/phase28_views"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Rebuild target is not empty: {output_dir}")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/build_router_phase28_training_views.py"),
        "--phase28-config",
        str(PROJECT_ROOT / "analysis/hotpotqa_router/phases/phase28/config.yaml"),
        "--phase27-template",
        str(PROJECT_ROOT / "analysis/hotpotqa_router/phases/phase27/config.yaml"),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        return completed.returncode
    failures = _verify_phase28_rebuild(output_dir)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"rebuild verified: {output_dir.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


def _same_gzip_payload(left: Path, right: Path) -> bool:
    import gzip

    with gzip.open(left, "rb") as first, gzip.open(right, "rb") as second:
        while True:
            first_block = first.read(1024 * 1024)
            second_block = second.read(1024 * 1024)
            if first_block != second_block:
                return False
            if not first_block:
                return True


def _verify_phase28_rebuild(output_dir: Path) -> list[str]:
    import numpy as np

    canonical = PROJECT_ROOT / (
        "outputs/router/hotpotqa_bd_router_v1/runs/phase28_query_expansion_v1"
    )
    failures: list[str] = []
    for view in ("T0_tiecap", "T1_winner2000", "T2_winner3000"):
        expected = canonical / "training_views" / view / "snapshot"
        rebuilt = output_dir / "training_views" / view / "snapshot"
        for name in ("outcomes.jsonl.gz", "query_summary.csv.gz"):
            if not _same_gzip_payload(expected / name, rebuilt / name):
                failures.append(f"{view}: rebuilt {name} differs")
        if (expected / "feature_schema.json").read_bytes() != (
            rebuilt / "feature_schema.json"
        ).read_bytes():
            failures.append(f"{view}: rebuilt feature schema differs")
        with np.load(expected / "features.npz", allow_pickle=False) as old, np.load(
            rebuilt / "features.npz", allow_pickle=False
        ) as new:
            if old.files != new.files:
                failures.append(f"{view}: rebuilt feature keys differ")
            else:
                for name in old.files:
                    if not np.array_equal(old[name], new[name]):
                        failures.append(f"{view}: rebuilt feature array {name} differs")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    registry = load_registry()
    if args.command == "status":
        _print_status(registry)
        return 0
    if args.command == "verify":
        failures = verify_registry(registry, args.phase)
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        target = args.phase or "all phases"
        print(f"verified: {target}")
        return 0
    if args.command == "rebuild" and args.target == "phase28_views":
        return _rebuild_phase28_views()
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
