"""Run, inspect or replay this frozen standalone candidate."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(ROOT.parent))
    __package__ = ROOT.name

from .core import METHOD
from .core.replay import verify, replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--replay", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/runtime300.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-cost-cny", type=float)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.verify:
        result = verify()
    elif args.replay:
        result = replay(args.output, args.limit)
    else:
        from .core.runtime import read_rows, check_resources, run

        verify()
        if args.dry_run:
            config = json.loads((ROOT / "config.json").read_text(encoding="utf8"))
            rows = read_rows(args.inputs)[: args.limit]
            check_resources(config, rows)
            result = {
                "method": METHOD,
                "questions": len(rows),
                "resources_exist": True,
                "new_API_calls": 0,
                "scope": "Resources and inputs checked; no model or index loaded",
            }
        else:
            if args.output is None:
                parser.error("--run requires an explicit --output directory")
            result = run(
                args.inputs,
                args.output,
                limit=args.limit,
                max_cost=args.max_cost_cny,
                device=args.device,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        KeyError,
    ) as exc:
        print(METHOD + ": " + str(exc), file=sys.stderr)
        raise SystemExit(2)
