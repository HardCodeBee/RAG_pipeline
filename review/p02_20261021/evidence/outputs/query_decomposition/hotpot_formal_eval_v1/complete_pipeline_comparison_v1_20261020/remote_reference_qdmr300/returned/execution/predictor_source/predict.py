#!/usr/bin/env python3
"""Query-only generation: predict.py --config CONFIG.json; resumes completed IDs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from adapter_runtime import verify_base_identity, read_identity
from model_runtime import (
    INSTRUCTION, StopRequest, allocation_guard, checkpoint_manifest, generate_one,
    json_dump, jsonl_append, load_libraries, load_model, object_hash, read_config,
    runtime_info, seed_all, sha256_file, verify_checkpoint,
)


def query_records(path):
    """Read and retain exclusively ID/query; no target fields enter model runtime."""
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get('id'), str) or not isinstance(row.get('query'), str):
                raise ValueError(f'Invalid query-only input at {path}:{number}')
            rows.append({'id': row['id'], 'query': row['query']})
    rows.sort(key=lambda row: row['id'])
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Repeated input IDs')
    return rows


def read_existing(path):
    if not path.exists():
        return {}
    complete = {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    consumed = 0
    for idx, line in enumerate(lines):
        try:
            record = json.loads(line)
        except Exception:
            if idx != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError(f'Corrupt completed output row {idx + 1}')
            backup = path.with_name(path.name + f'.partial-tail-{int(time.time())}')
            backup.write_bytes(line)
            with path.open('r+b') as stream:
                stream.truncate(consumed)
            break
        if record['id'] in complete:
            raise ValueError(f'Duplicate saved output {record["id"]}')
        complete[record['id']] = record
        consumed += len(line)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    out = Path(config['output_dir']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    output = out / 'predictions.jsonl'
    stop = StopRequest()
    stop.install()
    rows = query_records(config['input_file'])
    finished = read_existing(output)
    allocation = allocation_guard()
    torch, np, transformers = load_libraries(allocation)
    seed_all(int(config.get('seed', 1729)), torch, np, config)
    torch.cuda.reset_peak_memory_stats()
    base_identity = verify_base_identity(config['base_identity_file'], config['model_path'])
    adapter_path = config.get('adapter_path')
    adapter_manifest = None
    if adapter_path:
        adapter_manifest = verify_checkpoint(adapter_path)
        if read_identity(Path(adapter_path) / 'BASE_MODEL_IDENTITY.json') != base_identity:
            raise ValueError('Prediction adapter belongs to another frozen base')
    elif config.get('allow_upstream_checkpoint') is True:
        pass
    else:
        raise ValueError('A completed adapter checkpoint is required; official baseline needs allow_upstream_checkpoint=true')
    manifest = {'manifest_sha256': object_hash({'base_model_manifest': base_identity['manifest_sha256'],
                                               'adapter_manifest': adapter_manifest['manifest_sha256'] if adapter_manifest else None})}
    identity = {'input_sha256': sha256_file(config['input_file']),
                'model_manifest': manifest['manifest_sha256'],
                'base_model_identity': base_identity,
                'adapter_manifest_sha256': adapter_manifest['manifest_sha256'] if adapter_manifest else None,
                'config_fingerprint': object_hash({k: v for k, v in config.items() if k not in ('_config_path', 'output_dir')}),
                'query_order': [row['id'] for row in rows],
                'instruction': config.get('instruction', INSTRUCTION)}
    identity_file = out / 'IDENTITY.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError('Resume rejected: input, model, instruction or decoding config changed')
    json_dump(identity, identity_file)
    expected_queries = {row['id']: row['query'] for row in rows}
    if set(finished) - set(expected_queries):
        raise ValueError('Existing predictions contain unknown IDs')
    for rid, record in finished.items():
        if record['query'] != expected_queries[rid]:
            raise ValueError('Existing prediction query changed')
        if record['checkpoint_sha256_or_manifest'] != manifest['manifest_sha256']:
            raise ValueError('Existing prediction checkpoint changed')
    model, tokenizer, report = load_model(config, torch, transformers, train=False)
    model.eval()
    json_dump(report, out / 'model_report.json')
    json_dump(runtime_info(torch, transformers, allocation, config), out / f'runtime-{allocation["job_id"]}.json')
    json_dump(config, out / 'config.json')
    json_dump({'instruction': identity['instruction'], 'chat_template': tokenizer.chat_template,
               'enable_thinking': None, 'template_policy': config['template_policy']}, out / 'prompt_template.json')
    start = time.perf_counter()
    failures = 0
    for row in rows:
        if row['id'] in finished:
            continue
        if stop.requested:
            break
        try:
            result = generate_one(model, tokenizer, row, torch, config)
        except Exception as error:
            failures += 1
            result = {'id': row['id'], 'query': row['query'], 'raw_output': None,
                      'parsed_target': None, 'references': [], 'elapsed_seconds': None,
                      'truncated': False, 'parse_error': None,
                      'generation_error': {'type': type(error).__name__, 'message': str(error),
                                           'traceback': traceback.format_exc()},
                      'input_token_ids': None, 'output_token_ids': None,
                      'input_tokens': None, 'output_tokens': None,
                      'decoding_config': {key: config.get(key) for key in ('max_new_tokens', 'num_beams')},
                      'failure_is_complete_attempt': True}
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        result['checkpoint_sha256_or_manifest'] = manifest['manifest_sha256']
        result['slurm_job_id'] = allocation['job_id']
        result['scope'] = config.get('scope', 'frozen_checkpoint_query_only_generation')
        jsonl_append(result, output)
        finished[row['id']] = result
        json_dump({'status': 'running', 'completed_ids': sorted(finished),
                   'total': len(rows), 'last_id': row['id']}, out / 'STATE.json')
        print(json.dumps({'id': row['id'], 'elapsed_seconds': result['elapsed_seconds'],
                          'parse_error': result['parse_error'],
                          'generation_error': result['generation_error']}, ensure_ascii=False), flush=True)
    # Retain stable ID ordering even if a resumed file was produced with a past partial tail.
    summary = {'status': 'interrupted' if stop.requested else 'completed',
               'scope': config.get('scope', 'frozen_checkpoint_query_only_generation'),
               'total_inputs': len(rows), 'completed_attempts': len(finished),
               'generation_failures': sum(row.get('generation_error') is not None for row in finished.values()),
               'parse_failures': sum(row.get('parse_error') is not None for row in finished.values()),
               'truncated_outputs': sum(bool(row.get('truncated')) for row in finished.values()),
               'current_job_elapsed_seconds': time.perf_counter() - start,
               'sum_generation_seconds': sum(row.get('elapsed_seconds') or 0 for row in finished.values()),
               'output_tokens': sum(row.get('output_tokens') or 0 for row in finished.values()),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'checkpoint_sha256_or_manifest': manifest['manifest_sha256'],
               'base_model_manifest_sha256': base_identity['manifest_sha256'],
               'adapter_manifest_sha256': adapter_manifest['manifest_sha256'] if adapter_manifest else None,
               'signal': stop.signal, 'input_hash': identity['input_sha256']}
    json_dump(summary, out / 'SUMMARY.json')
    json_dump(dict(summary, completed_ids=sorted(finished)), out / 'STATE.json')
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 85 if stop.requested else (2 if summary['generation_failures'] else 0)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
