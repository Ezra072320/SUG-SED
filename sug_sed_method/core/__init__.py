"""Core shared infrastructure for SUG-SED v2 (Pure-Star-Meso + SUG-CrossStar)."""

from sug_sed_method.core.data_loader import (
    SocialDataset,
    load_socialed_npy,
    split_daily_blocks,
    split_fixed_size_blocks,
)
from sug_sed_method.core.graph_builder import MessageGraph, build_message_graph
from sug_sed_method.core.key_loader import load_provider_key
from sug_sed_method.core.metrics_audit import full_partition_audit
from sug_sed_method.core.partition_cache import load_or_solve_2d_se
from sug_sed_method.core.se_solver import compute_2d_se, solve_2d_se
from sug_sed_method.core.token_ledger import (
    CausalSlidingBucket,
    CausalUniformBucket,
    CumulativeTokenBucket,
    TokenLedger,
)

__all__ = [
    "SocialDataset",
    "load_socialed_npy",
    "split_daily_blocks",
    "split_fixed_size_blocks",
    "MessageGraph",
    "build_message_graph",
    "compute_2d_se",
    "solve_2d_se",
    "load_or_solve_2d_se",
    "TokenLedger",
    "CausalUniformBucket",
    "CausalSlidingBucket",
    "CumulativeTokenBucket",
    "full_partition_audit",
    "load_provider_key",
]
