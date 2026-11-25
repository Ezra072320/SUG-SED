"""Gate A Pre-Flight Diagnostic (Zero API Cost): Real Event2012 Dev Blocks.

Strictly uses ONLY designated dev blocks (M_02..M_05 of Event2012) while leaving
M_06..M_21 completely untouched as reserved val/test blocks.

Uses locally cached `sentence-transformers/all-MiniLM-L6-v2` on GTX 1650 (cached
to disk as float16 .npy so subsequent runs take <2s) and evaluates:
  1. Initial 2D Structural Entropy clustering quality & error structure
  2. Router diagnostics: Seam True-Split Rate & Multi-Route Candidate Recall@4
  3. Oracle Upper-Bound Gate A Comparison across 4 budgeted strategies:
     - Random
     - Rep-Div (RagSEDE-style)
     - Cequel-Edge (Cequel-style pairwise)
     - SUG-Router (Ours: Seam + Bridge + Seed)
  4. Strict same-decision comparison of:
     - Branch 1 (Direct Assignment)
     - Branch 2 (Topology Repair + Exact 2D SE Re-Clustering)
     measuring Full-Partition AMI / Macro-B3, Unqueried-Post Net Gain,
     Resolved vs. Introduced False Merges & Splits, and Tail Discovery.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure workspace root is importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import build_message_graph, compute_hf_embeddings
from core_shared.metrics_audit import full_partition_audit
from core_shared.se_solver import solve_2d_se
from trackA_sug_sed.high_leverage_router import (
    build_cluster_summaries,
    select_cequel_edge_queries,
    select_random_queries,
    select_rep_div_queries,
    select_sug_queries,
)
from trackA_sug_sed.topology_surgery import (
    apply_direct_assignment,
    apply_topology_repair_and_recluster,
    build_oracle_decisions,
)

CACHE_DIR = ROOT / "cache" / "embeddings"
RESULTS_DIR = ROOT / "results" / "gateA_oracle"
DEV_BLOCK_IDS = [2, 3, 4, 5]  # Strictly dev only; M_06..M_21 reserved untouched!
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_or_extract_block_embeddings(
    dataset_name: str,
    block_id: int,
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "__")
    cache_path = CACHE_DIR / f"{dataset_name}_M{block_id:02d}_{safe_model}.npy"
    if cache_path.exists():
        emb = np.load(cache_path).astype(np.float32)
        if emb.shape[0] == len(texts):
            return emb
    t0 = time.time()
    emb = compute_hf_embeddings(texts, model_name=model_name, batch_size=64, max_length=128)
    np.save(cache_path, emb.astype(np.float16))
    print(f"  [Emb] Extracted & cached {cache_path.name} shape={emb.shape} in {time.time()-t0:.1f}s")
    return emb


def run_diagnostic(
    dataset_name: str = "Event2012",
    budgets: List[int] = [15, 30, 50],
    knn_k: int = 15,
    knn_sim_threshold: float = 0.35,
    attr_alpha: float = 0.5,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {dataset_name} (embed_model={model_name}) ...")
    ds_full = load_socialed_npy(dataset_name, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
    print(f"Total blocks: {len(blocks)}. Using STRICT DEV blocks: {DEV_BLOCK_IDS}")

    all_reports: Dict[str, Any] = {
        "dataset": dataset_name,
        "dev_block_ids": DEV_BLOCK_IDS,
        "model": model_name,
        "graph_params": {
            "knn_k": knn_k,
            "knn_sim_threshold": knn_sim_threshold,
            "attr_alpha": attr_alpha,
        },
        "blocks": {},
    }

    # Aggregate table rows across dev blocks
    agg: Dict[str, List[Dict[str, float]]] = {}

    for bid in DEV_BLOCK_IDS:
        bname, bidx = blocks[bid]
        sub_ds = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        print(f"\n================ M_{bid:02d} [{bname}] (n={len(sub_ds)}, true_events={len(set(sub_ds.labels.tolist()))}) ================")

        emb = load_or_extract_block_embeddings(dataset_name, bid, sub_ds.texts, model_name=model_name)
        t0 = time.time()
        graph = build_message_graph(
            sub_ds,
            emb,
            knn_k=knn_k,
            knn_sim_threshold=knn_sim_threshold,
            attr_alpha=attr_alpha,
        )
        labels_init, h2_init = solve_2d_se(graph.adj, refine_moves=True)
        dt_init = time.time() - t0
        summaries = build_cluster_summaries(sub_ds, emb, labels_init)
        init_audit = full_partition_audit(sub_ds.labels, labels_init)

        print(
            f"  [Init 2D-SE] K={init_audit['n_pred_clusters']} (true {init_audit['n_true_events']}) "
            f"| AMI={init_audit['ami']:.4f} | NMI={init_audit['nmi']:.4f} "
            f"| Macro-B3={init_audit['bcubed_full']['macro_b3_f1']:.4f} "
            f"| Micro-B3={init_audit['bcubed_full']['micro_b3_f1']:.4f} "
            f"| FM={init_audit['pairwise_errors']['false_merge_pairs']} "
            f"| FS={init_audit['pairwise_errors']['false_split_pairs']} ({dt_init:.2f}s)"
        )

        block_report: Dict[str, Any] = {
            "block_name": bname,
            "n_posts": len(sub_ds),
            "h2_init": h2_init,
            "init_audit": init_audit,
            "budgets": {},
        }

        for b_q in budgets:
            budget_res: Dict[str, Any] = {}
            strategies = {
                "random": select_random_queries(sub_ds, graph, labels_init, b_q, seed=42 + bid),
                "rep_div": select_rep_div_queries(sub_ds, graph, labels_init, b_q),
                "cequel_edge": select_cequel_edge_queries(sub_ds, graph, labels_init, b_q),
                "sug_router": select_sug_queries(sub_ds, graph, labels_init, b_q),
            }

            print(f"\n  --- Budget Q={b_q} ({100.0*b_q/len(sub_ds):.2f}% of block) ---")
            for sname, queries in strategies.items():
                decisions, dstats = build_oracle_decisions(queries, sub_ds.labels, labels_init, summaries)

                # Branch 1: Direct Assignment
                lab_dir, qmask_dir = apply_direct_assignment(labels_init, queries, decisions)
                aud_dir = full_partition_audit(
                    sub_ds.labels, lab_dir, y_before=labels_init, queried_mask=qmask_dir
                )

                # Branch 2: Topology Repair + Exact 2D SE Re-Clustering
                lab_rep, qmask_rep, surg_stats = apply_topology_repair_and_recluster(
                    graph, labels_init, summaries, queries, decisions
                )
                aud_rep = full_partition_audit(
                    sub_ds.labels, lab_rep, y_before=labels_init, queried_mask=qmask_rep
                )

                budget_res[sname] = {
                    "decision_stats": dstats,
                    "surgery_stats": surg_stats,
                    "branch1_direct": aud_dir,
                    "branch2_repair": aud_rep,
                }

                for branch_tag, aud in [("direct", aud_dir), ("repair", aud_rep)]:
                    key = f"Q{b_q:02d}_{sname}_{branch_tag}"
                    prop = aud["propagation_audit"]
                    et = aud["error_transition"]
                    agg.setdefault(key, []).append(
                        {
                            "ami": aud["ami"],
                            "delta_ami": aud["delta_ami"],
                            "macro_b3": aud["bcubed_full"]["macro_b3_f1"],
                            "delta_macro_b3": aud["delta_macro_b3_f1"],
                            "micro_b3": aud["bcubed_full"]["micro_b3_f1"],
                            "delta_micro_b3": aud["delta_micro_b3_f1"],
                            "unq_delta_micro_b3": prop["unqueried_delta_micro_b3_f1"],
                            "unq_delta_macro_b3": prop["unqueried_delta_macro_b3_f1"],
                            "unq_net_posts": prop["unqueried_net_improved_posts"],
                            "net_fm_red": et["net_false_merge_reduction"],
                            "net_fs_red": et["net_false_split_reduction"],
                            "tail_b3": aud["stratified"]["tail"]["macro_b3_f1"],
                            "tokens": dstats["total_tokens"],
                            "cand_recall": dstats["candidate_recall_at_m"],
                            "seam_split_rate": dstats["seam_true_split_rate"],
                        }
                    )

                p1 = aud_dir["propagation_audit"]
                p2 = aud_rep["propagation_audit"]
                print(
                    f"    {sname:11s} [Rec@4={dstats['candidate_recall_at_m']:.2f}, "
                    f"MrgHit={dstats.get('merge_true_hit_rate', 0.0):.2f}, "
                    f"SmHit={dstats['seam_true_split_rate']:.2f}, Tok={int(dstats['total_tokens']):5d}] | "
                    f"Direct: dAMI={aud_dir['delta_ami']:+.4f}, dMacB3={aud_dir['delta_macro_b3_f1']:+.4f}, "
                    f"UnqNet={p1['unqueried_net_improved_posts']:+4d} || "
                    f"Repair: dAMI={aud_rep['delta_ami']:+.4f}, dMacB3={aud_rep['delta_macro_b3_f1']:+.4f}, "
                    f"UnqNet={p2['unqueried_net_improved_posts']:+4d}"
                )

            block_report["budgets"][str(b_q)] = budget_res
        all_reports["blocks"][f"M_{bid:02d}"] = block_report

    # Print clean cross-block mean summary
    print("\n================ MEAN ACROSS DEV BLOCKS (M_02..M_05) ================")
    summary_table: Dict[str, Dict[str, float]] = {}
    header = (
        f"{'Setting':26s} | {'Tok':>6s} | {'AMI':>6s} | {'dAMI':>7s} | {'MacB3':>6s} | "
        f"{'dMacB3':>7s} | {'Unq_dMic':>8s} | {'Unq_dMac':>8s} | {'UnqNet':>7s} | {'NetFM':>7s} | {'NetFS':>7s}"
    )
    print(header)
    print("-" * len(header))
    for key, rows in agg.items():
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()}
        summary_table[key] = m
        print(
            f"{key:26s} | {m['tokens']:6.0f} | {m['ami']:6.4f} | {m['delta_ami']:+7.4f} | "
            f"{m['macro_b3']:6.4f} | {m['delta_macro_b3']:+7.4f} | "
            f"{m['unq_delta_micro_b3']:+8.4f} | {m['unq_delta_macro_b3']:+8.4f} | "
            f"{m['unq_net_posts']:+7.1f} | {m['net_fm_red']:+7.0f} | {m['net_fs_red']:+7.0f}"
        )

    all_reports["dev_mean_summary"] = summary_table
    out_path = RESULTS_DIR / f"{dataset_name}_gateA_oracle_dev.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full diagnostic JSON to: {out_path}")
    return all_reports


if __name__ == "__main__":
    run_diagnostic()
