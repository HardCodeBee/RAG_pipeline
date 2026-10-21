"""One fixed held-out120 run for a strictly improved dev-selected checkpoint."""
import argparse,json,subprocess,sys
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--base-package',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True);a=ap.parse_args()
s=json.loads((a.experiment/'NOMINEE_FROZEN.json').read_text());assert s['selected'] and not s['confirmation_used_for_selection']
assert all(s['gates'][s['selected']].values())
c=json.loads((a.experiment/'config.json').read_text())
scorer=Path('/home/projects/u7830488/multigrain_4b_coverage_retrain_v3_20261018/package_v001/score_model.py')
subprocess.run([sys.executable,str(scorer),'--base-package',str(a.base_package),'--checkpoint',s['checkpoint'],
  '--checkpoint-sha',s['checkpoint_manifest_sha256'],'--queries',str(a.experiment/'confirm120.questions.frozen.jsonl'),
  '--queries-sha',c['data_sha256']['confirm120.questions.frozen.jsonl'],'--prompt',str(a.experiment/'v12.txt'),
  '--output',str(a.experiment/'results/confirmation')],check=True)
