"""Query-only/base-only contracts. Standard library only, never calls an API."""
import hashlib
import json
from pathlib import Path
import re

ARMS = ("v6", "v8", "v11", "v12")
REPO = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BASE_MANIFEST = "271c937fb07875d2dd9b1b9d21ec7a199f6d78f22c9c77027d35e642c42f39ea"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def queries(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    assert len(rows) == 100 and len({r["id"] for r in rows}) == 100
    assert all(set(r) == {"id", "query"} and all(isinstance(v, str) and v.strip() for v in r.values()) for r in rows)
    return rows


def validate_config(config):
    assert config["repo_id"] == REPO and config["revision"] == REVISION
    assert config.get("adapter_path") is None and config["expected_adapter_parameters"] == 0
    assert "lora" not in config and config["allow_upstream_checkpoint"] is True
    assert config["expected_base_parameters"] == 4022468096
    assert config["template_policy"] == "qwen3_instruct_native_nonthinking"
    assert config["max_new_tokens"] == 1024 and config["num_beams"] == 1 and config["seed"] == 1729
    assert config["allow_tf32"] is False and config["attn_implementation"] == "sdpa"
    assert config["runtime_device"] == "cuda" and config["dtype"] == "bfloat16"
    assert tuple(config["arms"]) == ARMS


def verify_package(package):
    package = Path(package)
    manifest = read_json(package / "PACKAGE_MANIFEST.json")
    for name, expected in manifest["files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert sha(package / name) == expected, name
    assert manifest["api_calls"] == 0 and manifest["adapter_loaded"] is False
    assert manifest["expected_outputs"] == 400
    return manifest


def parse_array(raw):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return {"format_valid": False, "parsed_questions": None, "parse_error": "not_literal_json", "explicit_references": None, "duplicate_count": None}
    valid = isinstance(value, list) and all(isinstance(q, str) and q.strip() for q in value)
    return {"format_valid": valid, "parsed_questions": value if valid else None,
            "parse_error": None if valid else "expected_array_of_nonempty_question_strings",
            "explicit_references": any(re.search(r"#\d+|\bprevious (?:question|answer)\b", q, re.I) for q in value) if valid else None,
            "duplicate_count": len(value) - len(set(q.strip() for q in value)) if valid else None}


def summary(records):
    stats = {}
    for arm in ARMS:
        selected = [r for r in records if r["arm"] == arm]
        stats[arm] = {"planned": 100, "completed_attempts": len(selected),
                      "generation_errors": sum(bool(r.get("generation_error")) for r in selected),
                      "truncated": sum(bool(r.get("truncated")) for r in selected),
                      "format_valid": sum(bool(r.get("format_valid")) for r in selected),
                      "explicit_reference_outputs": sum(bool(r.get("explicit_references")) for r in selected)}
    return {"planned": 400, "completed_attempts": len(records), "arms": stats,
            "semantic_review": "pending_current_Codex_F_S_C_U_review", "api_calls": 0}


def read_cases(out, rows):
    records = []
    for index, row in enumerate(rows, 1):
        for arm in ARMS:
            p = Path(out) / "cases" / arm / f"{index:03d}.json"
            if p.exists():
                r = read_json(p)
                assert r["id"] == row["id"] and r["query"] == row["query"] and r["arm"] == arm
                records.append(r)
    return records
