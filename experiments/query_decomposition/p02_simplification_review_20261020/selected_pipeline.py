"""User-selected simplification: delay original-query retrieval only.

All prompt, merge, ordinary QA, goal pre-answer and final-answer behavior stays
with frozen p02. No API client or retrieval resource is created by this wrapper.
"""
import json
from pathlib import Path

from experiments.query_decomposition.p02_flow_simplify_v1_20261020 import pipeline


def execute(case, retrieve, ask, *, record_node=None):
    protocol = Path(pipeline.__file__).parent / 'PROTOCOL.frozen.json'
    readout = json.loads(protocol.read_text(encoding='utf8'))['readout']
    return pipeline.execute(case, retrieve, ask, arm='timing_only',
                            readout=readout, record_node=record_node)
