"""Method modules for SUG-SED v2 (Pure-Star-Meso + SUG-CrossStar)."""

from sug_sed_method.method.anonymizer import (
    AnonymizationAuditStore,
    QueryAnonymizer,
    QueryItem,
    check_summary_reconstruction_leakage,
)
from sug_sed_method.method.cross_day import (
    GlobalEventMemory,
    HistoricalEvent,
    evaluate_cross_day_stream,
)
from sug_sed_method.method.embed_cache import encode_texts_hf, load_or_extract_block_embeddings
from sug_sed_method.method.llm_client import (
    BudgetedLLMClient,
    build_cross_star_prompt,
    build_seam_split_prompt,
    build_star_merge_prompt,
)
from sug_sed_method.method.star_meso import (
    ClusterSummary,
    GuardedUnionFind,
    SeamCandidate,
    build_cluster_summaries,
    detect_clean_seams,
    run_pure_star_meso,
)

__all__ = [
    "encode_texts_hf",
    "load_or_extract_block_embeddings",
    "BudgetedLLMClient",
    "build_star_merge_prompt",
    "build_seam_split_prompt",
    "build_cross_star_prompt",
    "QueryItem",
    "QueryAnonymizer",
    "AnonymizationAuditStore",
    "check_summary_reconstruction_leakage",
    "ClusterSummary",
    "build_cluster_summaries",
    "GuardedUnionFind",
    "SeamCandidate",
    "detect_clean_seams",
    "run_pure_star_meso",
    "HistoricalEvent",
    "GlobalEventMemory",
    "evaluate_cross_day_stream",
]
