"""Frozen second candidate: p02 + delayed root retrieval + exact-input cache.

This module changes no model-facing input. Retrieval, successful-payload reuse,
transport, and budget accounting remain injected by the selected executor.
"""
from experiments.query_decomposition.p02_simplification_review_20261020.selected_pipeline import execute

CANDIDATE_ID = 'p02_multigrain_timing_cachefix_v1'
SELECTED_ARM = 'timing_only'
CACHE_PROVIDER = 'experiments.query_decomposition.p02_merge_cachefix_v1_20261021.cache.catalog'
EXECUTION_RUNNER = 'experiments.query_decomposition.p02_merge_cachefix_v1_20261021.run'

