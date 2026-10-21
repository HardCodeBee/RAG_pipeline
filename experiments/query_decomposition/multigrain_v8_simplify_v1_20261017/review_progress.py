"""Reuse only exact literal-output judgments; expose all new strings for review."""
import argparse
import json

from .study import HERE, OUT, api


def show(phase, arm, start, end, reuse):
    path = HERE / ("AUTHORED_" + phase + ".json")
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"reviewer": "Current Codex same-context nonblind F/S/C/U literal review, no answers, reference plans or documents.", "cases": []}
    authored = {r["index"]: r for r in doc["cases"]}
    for c in api.rows(HERE / (phase + ".questions.frozen.jsonl"))[start-1:end]:
        raw_path = OUT / (phase + "_" + arm) / "cases" / arm / c["dataset"] / (c["qid"] + ".json")
        if not raw_path.exists():
            continue
        r = json.loads(raw_path.read_text(encoding="utf-8"))
        j = authored.setdefault(c["index"], {"index": c["index"]})
        code, reason = None, None
        if r["status"] != "success" or r.get("finish_reason") != "stop" or not r.get("mechanical", {}).get("array_of_nonempty_strings"):
            code, reason = "X", "无完整可解析返回，保留不可用，不作语义评判。"
        else:
            for before in ("v8", "v10", "v11", "v12"):
                if before == arm or before not in j or j[before] == "X":
                    continue
                p = OUT / (phase + "_" + before) / "cases" / before / c["dataset"] / (c["qid"] + ".json")
                if p.exists() and json.loads(p.read_text(encoding="utf-8")).get("mechanical", {}).get("questions") == r["mechanical"]["questions"]:
                    code = j[before]
                    reason = "与已审阅" + before + "逐字问题列表相同，沿用该输出判定。" + j.get("reasons", {}).get(before, j.get("reason", ""))
                    break
        if reuse and code is not None and arm not in j:
            j[arm] = code
            j.setdefault("reasons", {})[arm] = reason
        marker = j.get(arm, "NEEDS_REVIEW")
        print(str(c["index"]) + " " + c["dataset"] + " [" + marker + "] " + c["question"])
        if marker == "NEEDS_REVIEW":
            print(json.dumps(r.get("mechanical", {}).get("questions"), ensure_ascii=False))
    if reuse:
        doc["cases"] = sorted(authored.values(), key=lambda r: r["index"])
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="dev", choices=["dev", "holdout"])
    p.add_argument("--arm", required=True)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=60)
    p.add_argument("--reuse-exact", action="store_true")
    a = p.parse_args()
    show(a.phase, a.arm, a.start, a.end, a.reuse_exact)
