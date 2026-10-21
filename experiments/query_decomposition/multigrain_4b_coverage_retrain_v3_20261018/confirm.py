"""One-time new120 paired confirmation, only after a frozen eligible nominee."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-package',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--historical',type=Path);ap.add_argument('--historical-sha');a=ap.parse_args()
    plan=json.loads((a.experiment/'PLAN_FROZEN.json').read_text());s=json.loads(a.selection.read_text());assert s['selected'] and s['confirmation_used_for_selection'] is False
    cases=[('p00_parent_t02',plan['parent_checkpoint'],plan['parent_sha256']),
           (s['selected'],s['checkpoint'],s['checkpoint_manifest_sha256']),
           ('original165',str(a.base_package/'results/train/checkpoints/checkpoint-000165'),'eb78aaaec6a8cfb853daeca9e739ca6143487d8d8f9b3bd1c4bb3494873a9192')]
    for name,cp,h in cases:
        subprocess.run([sys.executable,str(a.experiment/'score_model.py'),'--base-package',str(a.base_package),'--checkpoint',cp,'--checkpoint-sha',h,
            '--queries',str(a.experiment/'confirm120.questions.frozen.jsonl'),'--queries-sha',plan['confirm_roster_sha256'],
            '--prompt',str(a.experiment/'v12.txt'),'--output',str(a.output/name)],check=True)
    if a.historical:
        assert a.historical_sha
        subprocess.run([sys.executable,str(a.experiment/'score_model.py'),'--base-package',str(a.base_package),'--checkpoint',s['checkpoint'],'--checkpoint-sha',s['checkpoint_manifest_sha256'],
            '--queries',str(a.historical),'--queries-sha',a.historical_sha,'--prompt',str(a.experiment/'v12.txt'),'--output',str(a.output.parent/'historical16'/s['selected'])],check=True)

if __name__=='__main__':main()
