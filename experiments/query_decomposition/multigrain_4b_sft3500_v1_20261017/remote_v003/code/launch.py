"""One allocation, isolated smoke process followed by freshly initialized SFT."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[1]
from contract import sha
child=None;stopping=False

def forward(signum,_frame):
    global stopping
    stopping=True
    if child and child.poll() is None:child.send_signal(signum)

def main():
    global child
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--resume',type=Path);a=ap.parse_args()
    for name,digest in json.loads((PACKAGE/'PACKAGE_MANIFEST.json').read_text())['files'].items():
        assert sha(PACKAGE/name)==digest,('package_changed',name)
    gate=json.loads((PACKAGE/'EXECUTION_READY.json').read_text())
    assert gate['local_data_validation_passed'] and gate['baseline300_semantic_review_completed'] and gate['local_contract_checks_passed']
    signal.signal(signal.SIGUSR1,forward);signal.signal(signal.SIGTERM,forward)
    c=json.loads(a.config.read_text());root=Path(c['output_root'])
    stages=[] if (root/'smoke/SMOKE_VERIFIED.json').exists() else ['smoke']
    stages.append('train')
    for mode in stages:
        cmd=[sys.executable,str(PACKAGE/'code/train.py'),'--config',str(a.config),'--mode',mode]
        if mode=='train' and a.resume:cmd+=['--resume',str(a.resume)]
        child=subprocess.Popen(cmd);code=child.wait()
        if code or stopping:raise SystemExit(code or 75)
        if mode=='train':
            state=json.loads((root/'train/STATE.json').read_text())
            if state['status']=='interrupted':raise SystemExit(75)

if __name__=='__main__':main()
