"""Public modeling interface for Router experiment stages.

The Phase 2.7 runner remains the implementation owner. Later stages import
this module instead of reaching into private names in an older stage script.
"""

from scripts.run_router_phase27_model_audit import (
    CandidateSpec,
    RouterData,
    TIE_ATOL,
    _candidate_by_id,
    _fit_affine_calibration,
    _fit_predict_candidate,
    _fit_predict_late_fusion,
    _load_config,
    load_frozen_data,
    make_group_stratified_folds,
    policy_metrics,
    run_candidate_split,
)

candidate_by_id = _candidate_by_id
fit_affine_calibration = _fit_affine_calibration
fit_predict_candidate = _fit_predict_candidate
fit_predict_late_fusion = _fit_predict_late_fusion
load_config = _load_config

__all__ = [
    "CandidateSpec",
    "RouterData",
    "TIE_ATOL",
    "candidate_by_id",
    "fit_affine_calibration",
    "fit_predict_candidate",
    "fit_predict_late_fusion",
    "load_config",
    "load_frozen_data",
    "make_group_stratified_folds",
    "policy_metrics",
    "run_candidate_split",
]

