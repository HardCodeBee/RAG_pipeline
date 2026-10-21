"""Open the fixed query-only confirmation input only after the dev90 gate."""
import json
from pathlib import Path
from .prepare_remote import BASE, CHECKPOINT, PYTHON, sha
from ..fresh_confirm import prerequisites

HERE = Path(__file__).resolve().parent
REMOTE = '/home/projects/u7830488/b0_strategy_round2_v1_20261020/confirm150'


def main():
    checked = prerequisites()
    assert checked['selected'] == 'p02_role_headers'
    query = HERE / 'remote_inputs/sealed150.id_query.frozen.jsonl'
    values = [json.loads(s) for s in query.read_text(encoding='utf8').splitlines()]
    assert len(values) == 150 and all(set(v) == {'id', 'query'} for v in values)
    assert sha(query) == '585b91d8f5094f9c472a83f44a9dee3660770b99f8b70d5af952450618ea8b94'
    script = HERE / 'remote_infer_id_query_v001.py'
    runner = (
        '#!/usr/bin/env bash\n#SBATCH --job-name=b0r2-confirm150\n'
        '#SBATCH --account=db4ai\n#SBATCH --partition=db4ai\n#SBATCH --qos=db4ai\n'
        '#SBATCH --gres=gpu:a6000:1\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=32G\n'
        '#SBATCH --time=00:45:00\n#SBATCH --output=infer-%j.out\n#SBATCH --error=infer-%j.err\n'
        'set -euo pipefail\nexport HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1\n'
        f'exec bash {BASE}/runtime/run_stage.slurm {PYTHON} {REMOTE}/{script.name} '
        f'--base-package {BASE} --checkpoint {CHECKPOINT} '
        '--checkpoint-sha 728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92 '
        f'--queries {REMOTE}/{query.name} --queries-sha {sha(query)} '
        f'--prompt {BASE}/v12.txt --output {REMOTE}/results\n')
    path = HERE / 'confirm150.slurm'
    if path.exists():
        assert path.read_text(encoding='utf8') == runner
    else:
        path.write_text(runner, encoding='utf8', newline='\n')
    gate = HERE.parent.parents[3] / 'outputs/query_decomposition/hotpot_formal_eval_v1/b0_strategy_round2_v1_20261020/fresh_dev90/GATE.frozen.json'
    # Keep outcomes local; the remote submission contains only id/query and inference code.
    record = {'selected': checked['selected'], 'gate_passed_before_opening_query_input': True,
              'scope': 'Query decomposition only. No training, retrieval, answering, answers or model scores uploaded.',
              'remote_directory': REMOTE, 'input_count': 150,
              'files': {str(p.relative_to(HERE)): sha(p) for p in [query, script, path]},
              'model_and_decoding': 'Exactly frozen R02/v12 explicit greedy generation configuration, seed1729, max1024.'}
    manifest = HERE / 'REMOTE_CONFIRM150.frozen.json'
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding='utf8')) == record
    else:
        manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'prepared': True, 'query_count': 150, 'gate_verified': True, 'new_API_calls': 0}))


if __name__ == '__main__':
    main()
