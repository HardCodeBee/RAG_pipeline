"""400 raw base-4B generations; no old-QDMR parser, adapter, APIs or training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from contracts import (ARMS, BASE_MANIFEST, digest, parse_array, queries, read_cases,
                       read_json, save_json, sha, summary, validate_config, verify_package)

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(PACKAGE / "runtime"))
from model_runtime import (StopRequest, allocation_guard, encode_prompt, load_libraries,
                           load_model, runtime_info, seed_all)


def generate(model, tokenizer, row, prompt, torch, config):
    ids = encode_prompt(tokenizer, row["query"], prompt, model.config.model_type)
    limit = int(model.config.max_position_embeddings)
    if len(ids) + config["max_new_tokens"] > limit:
        raise ValueError("Input+output exceeds context; truncation forbidden")
    tensor = torch.tensor([ids], dtype=torch.long, device="cuda")
    decode = {"do_sample": False, "num_beams": 1, "max_new_tokens": 1024,
              "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
              "use_cache": True}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), **decode)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = output[0, len(ids):].tolist()
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return {"raw_output": raw, **parse_array(raw), "input_token_ids": ids,
            "output_token_ids": generated, "input_tokens": len(ids), "output_tokens": len(generated),
            "decoding_config": decode, "elapsed_seconds": elapsed,
            "truncated": bool(len(generated) >= 1024 and generated[-1] != tokenizer.eos_token_id),
            "generation_error": None}


def export(out, rows):
    records = read_cases(out, rows)
    save_json(out / "SUMMARY.json", summary(records))
    p = out / "predictions.jsonl"
    tmp = p.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    tmp.replace(p)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    verify_package(PACKAGE)
    config = read_json(args.config)
    validate_config(config)
    rows = queries(PACKAGE / "queries100.query_only.jsonl")
    prompts = {a:(PACKAGE / "prompts" / (a + ".txt")).read_text(encoding="utf-8") for a in ARMS}
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {"model_manifest": BASE_MANIFEST, "adapter_loaded": False,
                    "input_sha256": sha(PACKAGE / "queries100.query_only.jsonl"),
                    "prompt_sha256": {a:sha(PACKAGE / "prompts" / (a + ".txt")) for a in ARMS},
                    "runtime_config_sha256": digest(config),
                    "package_manifest_sha256": sha(PACKAGE / "PACKAGE_MANIFEST.json")}
        if (out / "IDENTITY.json").exists():
            assert read_json(out / "IDENTITY.json") == identity, "Resume identity differs"
        else:
            assert not list((out / "cases").glob("*/*.json")), "Existing results lack identity"
            save_json(out / "IDENTITY.json", identity)
        existing = read_cases(out, rows)
        for r in existing:
            assert r["identity_sha256"] == digest(identity)
        if len(existing) == 400:
            export(out, rows)
            save_json(out / "STATE.json", {"completed_attempts": 400, "planned": 400, "status": "completed", "api_calls": 0, "adapter_loaded": False})
            print("All400 completed attempts retained; no generation replay.", flush=True)
            return
        stop = StopRequest()
        stop.install()
        allocation = allocation_guard()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch, np, transformers = load_libraries(allocation)
        seed_all(1729, torch, np, config)
        model, tokenizer, report = load_model(config, torch, transformers, train=False)
        assert report["adapter"] is None and report["adapter_parameters"] == 0
        assert report["base_identity"]["manifest_sha256"] == BASE_MANIFEST
        model.eval()
        save_json(out / "model_report.json", report)
        save_json(out / "RUNTIME.json", runtime_info(torch, transformers, allocation, config))
        save_json(out / "prompt_template.json", {"chat_template": tokenizer.chat_template,
                  "sha256": report["chat_template_sha256"], "artificial_thinking_prefix": False})
        token_lengths = {a:[len(encode_prompt(tokenizer, r["query"], prompts[a], model.config.model_type)) for r in rows] for a in ARMS}
        save_json(out / "TOKEN_PREFLIGHT.json", {"input_tokens": token_lengths, "max_new_tokens": 1024,
                  "context_limit": model.config.max_position_embeddings, "truncation_allowed": False})
        assert max(max(v) for v in token_lengths.values()) + 1024 <= model.config.max_position_embeddings
        failed = False
        for index, row in enumerate(rows, 1):
            order = list(ARMS[index % 4:]) + list(ARMS[:index % 4])
            for arm in order:
                target = out / "cases" / arm / f"{index:03d}.json"
                if target.exists():
                    continue
                if stop.requested:
                    break
                record = {"id": row["id"], "query": row["query"], "index": index, "arm": arm,
                          "identity_sha256": digest(identity), "prompt_sha256": identity["prompt_sha256"][arm],
                          "model_manifest": BASE_MANIFEST, "adapter_loaded": False,
                          "created_utc": datetime.now(timezone.utc).isoformat()}
                try:
                    record.update(generate(model, tokenizer, row, prompts[arm], torch, config))
                except Exception as error:
                    record.update({"raw_output": None, **parse_array(None), "generation_error": type(error).__name__+": "+str(error), "truncated": False})
                    failed = True
                save_json(target, record)
                export(out, rows)
                print(json.dumps({"index": index, "arm": arm, "format_valid": record["format_valid"], "generation_error": bool(record["generation_error"])}), flush=True)
                if failed:
                    break
            if failed or stop.requested:
                break
        records = export(out, rows)
        save_json(out / "STATE.json", {"completed_attempts": len(records), "planned": 400,
                  "status": "completed" if len(records) == 400 else "generation_error_stopped" if failed else "interrupted",
                  "signal": stop.signal, "api_calls": 0, "adapter_loaded": False})
        if failed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
