"""Standalone Track A method package: SUG-SED v2 (Pure-Star-Meso + SUG-CrossStar)."""

from sug_sed_method.core import (
    CausalUniformBucket,
    MessageGraph,
    SocialDataset,
    TokenLedger,
    build_message_graph,
    full_partition_audit,
    load_or_solve_2d_se,
    load_provider_key,
    load_socialed_npy,
    solve_2d_se,
    split_daily_blocks,
)
from sug_sed_method.method import (
    AnonymizationAuditStore,
    BudgetedLLMClient,
    GlobalEventMemory,
    GuardedUnionFind,
    QueryAnonymizer,
    detect_clean_seams,
    evaluate_cross_day_stream,
    load_or_extract_block_embeddings,
    run_pure_star_meso,
)

__all__ = [
    "SocialDataset",
    "load_socialed_npy",
    "split_daily_blocks",
    "MessageGraph",
    "build_message_graph",
    "solve_2d_se",
    "load_or_solve_2d_se",
    "TokenLedger",
    "CausalUniformBucket",
    "full_partition_audit",
    "load_provider_key",
    "load_or_extract_block_embeddings",
    "BudgetedLLMClient",
    "QueryAnonymizer",
    "AnonymizationAuditStore",
    "GuardedUnionFind",
    "detect_clean_seams",
    "run_pure_star_meso",
    "GlobalEventMemory",
    "evaluate_cross_day_stream",
]
