"""Separate fresh processes for smoke and full training, no smoke-weight reuse."""
import argparse,subprocess,sys
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--base-package',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True);a=ap.parse_args()
for mode in ['smoke','train']:
    subprocess.run([sys.executable,str(a.experiment/'train_full.py'),'--base-package',str(a.base_package),'--experiment',str(a.experiment),'--mode',mode],check=True)
