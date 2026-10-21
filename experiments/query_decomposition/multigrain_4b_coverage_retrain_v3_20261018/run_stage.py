"""Score frozen parent first, then execute two controlled training trials."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-package',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    p=json.loads((a.experiment/'PLAN_FROZEN.json').read_text())
    subprocess.run([sys.executable,str(a.experiment/'score_model.py'),'--base-package',str(a.base_package),'--checkpoint',p['parent_checkpoint'],'--checkpoint-sha',p['parent_sha256'],'--queries',str(a.experiment/'dev60.questions.frozen.jsonl'),'--queries-sha',p['dev_roster_sha256'],'--prompt',str(a.experiment/'v12.txt'),'--output',str(a.output/'parent_dev60')],check=True)
    subprocess.run([sys.executable,str(a.experiment/'train.py'),'--base-package',str(a.base_package),'--experiment',str(a.experiment),'--output',str(a.output/'training40')],check=True)

if __name__=='__main__':main()
