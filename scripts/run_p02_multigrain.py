"""Run the separately frozen p02 multigrain candidate."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/p02_multigrain_timing_cachefix_v1/run.py'),run_name='__main__')
