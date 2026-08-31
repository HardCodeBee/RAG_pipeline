"""Public runtime interface shared by stateful Router experiment stages."""

from scripts.run_router_phase1 import (
    GENERATOR_PRICES,
    _core_config,
    _deduplicated_hits,
    _load_router_config,
    _prompt_for_row,
    _usage_cost,
)
from scripts.run_router_phase2 import _load_examples
from scripts.run_router_phase3 import (
    _base_without_dataset,
    _connect,
    _existing_query_ids,
    _insert_repeats,
    _spent_generation_cost,
    _sync_sample,
)
from scripts.run_router_phase3_train import _dense_summary, _lexical_features

core_config = _core_config
deduplicated_hits = _deduplicated_hits
load_router_config = _load_router_config
prompt_for_row = _prompt_for_row
usage_cost = _usage_cost
load_examples = _load_examples
base_without_dataset = _base_without_dataset
connect = _connect
existing_query_ids = _existing_query_ids
insert_repeats = _insert_repeats
spent_generation_cost = _spent_generation_cost
sync_sample = _sync_sample
dense_summary = _dense_summary
lexical_features = _lexical_features

__all__ = [
    "GENERATOR_PRICES",
    "base_without_dataset",
    "connect",
    "core_config",
    "deduplicated_hits",
    "dense_summary",
    "existing_query_ids",
    "insert_repeats",
    "load_examples",
    "load_router_config",
    "lexical_features",
    "prompt_for_row",
    "spent_generation_cost",
    "sync_sample",
    "usage_cost",
]
