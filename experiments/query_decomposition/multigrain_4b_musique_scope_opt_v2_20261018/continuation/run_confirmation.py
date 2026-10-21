"""One-time baseline/nominee confirmation; each adapter loads in a fresh process."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-package',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    selected=json.loads((a.experiment/'TRAIN_NOMINEE_FROZEN.json').read_text());baseline=json.loads((a.experiment/'BASELINE_FROZEN.json').read_text())
    assert selected['screen_gate_passed'] is True and selected['selection_uses_confirmation_outputs'] is False
    rosters=a.experiment/'confirm120.questions.frozen.jsonl'
    for name,cp,h in [('b00_v12',a.base_package/'results/train/checkpoints/checkpoint-000165',baseline['checkpoint_manifest_sha256']),(selected['name'],Path(selected['checkpoint']),selected['checkpoint_manifest_sha256'])]:
        cmd=[sys.executable,str(a.experiment/'continuation/score_model.py'),'--base-package',str(a.base_package),'--checkpoint',str(cp),'--checkpoint-sha',h,'--queries',str(rosters),'--queries-sha',selected['confirmation_roster_sha256'],'--prompt',str(a.experiment/'prompts/b00_v12.txt'),'--output',str(a.output/name)]
        subprocess.run(cmd,check=True)
    # Keep the smaller purposive rare-prefix diagnostic separate from the main120.
    rare=a.experiment/'rare_queries.frozen.jsonl'
    if rare.exists():
        for name,cp,h in [('b00_v12',a.base_package/'results/train/checkpoints/checkpoint-000165',baseline['checkpoint_manifest_sha256']),(selected['name'],Path(selected['checkpoint']),selected['checkpoint_manifest_sha256'])]:
            cmd=[sys.executable,str(a.experiment/'continuation/score_model.py'),'--base-package',str(a.base_package),'--checkpoint',str(cp),'--checkpoint-sha',h,'--queries',str(rare),'--queries-sha',selected['rare_roster_sha256'],'--prompt',str(a.experiment/'prompts/b00_v12.txt'),'--output',str(a.output.parent/'rare16'/name)]
            subprocess.run(cmd,check=True)

if __name__=='__main__':main()
