"""Prepare the query-only R02 dev90 submission; never run remote work here."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = '/home/projects/u7830488/b0_strategy_round2_v1_20261020/dev90'
BASE = '/home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package_v003'
CHECKPOINT = '/home/projects/u7830488/multigrain_4b_coverage_retrain_v3_20261018/results/v001/training40/r02_expanded/checkpoint-final'
PYTHON = '/home/projects/u7830488/demand_parser_route2_training_v1/20261009T191901Z/route2_question_decomposer/environment/venv/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    query = HERE / 'remote_inputs/dev90.id_query.frozen.jsonl'
    values = [json.loads(line) for line in query.read_text(encoding='utf8').splitlines()]
    assert len(values) == 90 and all(set(v) == {'id', 'query'} for v in values)
    script = HERE / 'remote_infer_id_query_v001.py'
    runner = (
        '#!/usr/bin/env bash\n#SBATCH --job-name=b0r2-dev90\n'
        '#SBATCH --account=db4ai\n#SBATCH --partition=db4ai\n#SBATCH --qos=db4ai\n'
        '#SBATCH --gres=gpu:a6000:1\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=32G\n'
        '#SBATCH --time=00:45:00\n#SBATCH --output=infer-%j.out\n#SBATCH --error=infer-%j.err\n'
        'set -euo pipefail\nexport HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1\n'
        f'exec bash {BASE}/runtime/run_stage.slurm {PYTHON} {REMOTE}/{script.name} '
        f'--base-package {BASE} --checkpoint {CHECKPOINT} '
        '--checkpoint-sha 728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92 '
        f'--queries {REMOTE}/{query.name} --queries-sha {sha(query)} '
        f'--prompt {BASE}/v12.txt --output {REMOTE}/results\n'
    )
    path = HERE / 'dev90.slurm'
    if path.exists():
        assert path.read_text(encoding='utf8') == runner
    else:
        path.write_text(runner, encoding='utf8', newline='\n')
    record = {'scope': 'Query decomposition only; no answers, retrieval, API, or training.',
              'remote_directory': REMOTE, 'input_count': 90,
              'source_infer_sha256': '05457658ce1543553669d3bde14804bec59e88df885fef260c52b18339072b77',
              'wrapper_change': 'Only derive output dataset logging from query id prefix. All model inputs and generation remain unchanged.',
              'files': {str(p.relative_to(HERE)): sha(p) for p in [query, script, path]},
              'checkpoint_manifest_sha256': '728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92',
              'prompt_sha256': '508709f217ee58ca53d60ecce6001d5e5b0e6d51a858dd4b95b2d6d311864ee8',
              'sealed_confirmation_uploaded': False}
    manifest = HERE / 'REMOTE_DEV90.frozen.json'
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding='utf8')) == record
    else:
        manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'prepared': True, 'query_count': 90, 'api_calls': 0, 'remote_work_started': False}))


if __name__ == '__main__':
    main()
