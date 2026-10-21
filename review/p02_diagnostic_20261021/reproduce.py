"""Recompute locked-label statistics in a disposable directory, without RAG."""
from pathlib import Path
import json,os,subprocess,sys,tempfile,zipfile
from verify_bundle import verify
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 verify()
 import numpy
 expected=read(HERE/'CONDITIONAL_PROBABILITIES.json')
 with tempfile.TemporaryDirectory(prefix='p02_offline_diagnostic_') as directory:
  root=Path(directory)
  # Paths and member hashes were verified before extraction.
  with zipfile.ZipFile(HERE/'analysis_bundle.zip') as z:z.extractall(root)
  script=root/'analysis/p02_error_propagation_diagnostic_20261021/aggregate_diagnostic.py'
  env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'}
  subprocess.run([sys.executable,'-B',str(script)],cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',check=True)
  actual=read(root/'outputs/query_decomposition/p02_error_propagation_diagnostic_20261021/offline_review_v2/CONDITIONAL_PROBABILITIES.json')
  # Generation times and hashes of newly serialized labels are intentionally excluded.
  for field in ['final_correctness','bootstrap','summary','tables','association_not_causal']:
   assert actual[field]==expected[field],field
 print(json.dumps({'passed':True,'tables_reproduced':len(actual['tables']),'query_count':723,'bootstrap_replicates':2000,'seed':20261021,'pending_goal_facts':7,'numpy_version':numpy.__version__,'new_inference':False,'network_or_API_or_GPU_execution':False},ensure_ascii=False))
if __name__=='__main__':main()
