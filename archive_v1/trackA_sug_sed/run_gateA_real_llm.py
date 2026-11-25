"""Gate A Real-LLM Decisive Experiment Runner (Strict Real-Token Accounting).

Runs controlled comparisons on identical streaming blocks, identical initial
clusters, identical candidate pools, identical open-source LLM, and identical
REAL (prompt_tokens + completion_tokens) budgets:
  - Query selectors: random, rep_div (RagSEDE-style), cequel_edge (Cequel-style), sug_router (Ours)
  - Same-decision post-processing split:
      Branch 1: Direct Assignment / Direct Partition Edit (no graph re-clustering)
      Branch 2: Graph Topology Repair + Constrained Warm-Start 2D SE Re-Clustering

Never writes API keys to disk or logs. Every API call is logged in an append-only
JSONL ledger (`journal.jsonl` + `responses.jsonl`) with exact usage tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import build_message_graph
from core_shared.key_loader import load_provider_key
from core_shared.metrics_audit import full_partition_audit
from core_shared.se_solver import solve_2d_se
from core_shared.token_ledger import CumulativeTokenBucket, TokenLedger
from trackA_sug_sed.high_leverage_router import (
    build_cluster_summaries,
    select_cequel_edge_queries,
    select_random_queries,
    select_rep_div_queries,
    select_sug_queries,
)
from trackA_sug_sed.llm_client import BudgetedLLMClient, build_query_prompt
from trackA_sug_sed.run_gateA_oracle_diagnostic import load_or_extract_block_embeddings
from trackA_sug_sed.topology_surgery import (
    apply_direct_assignment,
    apply_topology_repair_and_recluster,
)


def _select_pool(
    strategy: str,
    ds,
    graph,
    labels_init: np.ndarray,
    max_queries: int,
    seed: int,
):
    if strategy == "random":
        return select_random_queries(ds, graph, labels_init, max_queries, seed=seed)
    if strategy == "rep_div":
        return select_rep_div_queries(ds, graph, labels_init, max_queries)
    if strategy == "cequel_edge":
        return select_cequel_edge_queries(ds, graph, labels_init, max_queries)
    if strategy == "sug_router":
        return select_sug_queries(ds, graph, labels_init, max_queries)
    if strategy == "sug_meso_only":
        return select_sug_queries(
            ds,
            graph,
            labels_init,
            max_queries,
            merge_frac=0.65,
            seam_frac=0.35,
            bridge_frac=0.0,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate A Real Open-Source LLM Decisive Experiment")
    parser.add_argument("--dataset", default="Event2012", choices=["Event2012", "Event2012_100", "Event2018"])
    parser.add_argument("--blocks", default="2,3,4,5", help="Comma-separated dev block IDs")
    parser.add_argument("--strategies", default="rep_div,cequel_edge,sug_router")
    parser.add_argument("--total-token-budget", type=int, default=40000, help="Strict total real token budget across selected blocks")
    parser.add_argument("--max-queries-per-block", type=int, default=120)
    parser.add_argument("--provider", default="deepseek", help="Open-source key provider in APIkey.txt")
    parser.add_argument("--model", default=None, help="Override open-source model name")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--knn-sim-threshold", type=float, default=0.35)
    parser.add_argument("--attr-alpha", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.6)
    parser.add_argument("--price-prompt-cny", type=float, default=2.0)
    parser.add_argument("--price-completion-cny", type=float, default=3.0)
    parser.add_argument("--run-name", default="dev_deepseek_40k")
    parser.add_argument("--dry-run", action="store_true", help="Estimate prompt tokens without calling API")
    args = parser.parse_args()

    block_ids = [int(x.strip()) for x in args.blocks.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]

    api_key = ""
    if not args.dry_run:
        prov_url, prov_model, api_key = load_provider_key(args.provider)
        base_url = args.base_url or prov_url
        model_name = args.model or prov_model
    else:
        base_url = args.base_url or "dry-run"
        model_name = args.model or "deepseek-chat"

    out_dir = ROOT / "results" / "gateA_llm" / f"{args.dataset}_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_full = load_socialed_npy(args.dataset, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)

    # Precompute initial graphs & partitions identically for all strategies
    block_cache: Dict[int, Dict[str, Any]] = {}
    total_posts = 0
    for bid in block_ids:
        bname, bidx = blocks[bid]
        sub_ds = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        emb = load_or_extract_block_embeddings(args.dataset, bid, sub_ds.texts, model_name=args.embed_model)
        graph = build_message_graph(
            sub_ds,
            emb,
            knn_k=args.knn_k,
            knn_sim_threshold=args.knn_sim_threshold,
            attr_alpha=args.attr_alpha,
        )
        labels_init, h2_init = solve_2d_se(graph.adj, refine_moves=True)
        summaries = build_cluster_summaries(sub_ds, emb, labels_init)
        init_audit = full_partition_audit(sub_ds.labels, labels_init)
        block_cache[bid] = {
            "name": bname,
            "ds": sub_ds,
            "graph": graph,
            "labels_init": labels_init,
            "summaries": summaries,
            "init_audit": init_audit,
        }
        total_posts += len(sub_ds)
        print(
            f"[Init] M_{bid:02d} [{bname}] n={len(sub_ds)} K={init_audit['n_pred_clusters']} "
            f"AMI={init_audit['ami']:.4f} MacB3={init_audit['bcubed_full']['macro_b3_f1']:.4f} "
            f"MicB3={init_audit['bcubed_full']['micro_b3_f1']:.4f}"
        )

    safe_config = {k: v for k, v in vars(args).items() if "key" not in k.lower()}
    safe_config["resolved_model"] = model_name
    safe_config["resolved_base_url"] = base_url

    summary_report: Dict[str, Any] = {
        "config": safe_config,
        "init_mean": {
            "ami": float(np.mean([block_cache[b]["init_audit"]["ami"] for b in block_ids])),
            "macro_b3": float(np.mean([block_cache[b]["init_audit"]["bcubed_full"]["macro_b3_f1"] for b in block_ids])),
            "micro_b3": float(np.mean([block_cache[b]["init_audit"]["bcubed_full"]["micro_b3_f1"] for b in block_ids])),
        },
        "strategies": {},
        "mean_comparison": {},
    }

    agg: Dict[str, List[Dict[str, float]]] = {}

    for strat in strategies:
        print(f"\n================ Strategy: {strat} | Model: {model_name} | Budget={args.total_token_budget} real tokens ================")
        ledger = TokenLedger(
            out_dir / f"ledger_{strat}",
            price_per_1m_prompt_cny=args.price_prompt_cny,
            price_per_1m_completion_cny=args.price_completion_cny,
        )
        bucket = CumulativeTokenBucket(
            total_token_budget=args.total_token_budget,
            num_blocks=len(block_ids),
        )
        client = BudgetedLLMClient(
            model=model_name,
            ledger=ledger,
            base_url=base_url,
            api_key=api_key,
        )

        strat_blocks: Dict[str, Any] = {}
        for bid in block_ids:
            bc = block_cache[bid]
            sub_ds = bc["ds"]
            graph = bc["graph"]
            labels_init = bc["labels_init"]
            summaries = bc["summaries"]

            pool = _select_pool(strat, sub_ds, graph, labels_init, args.max_queries_per_block, seed=42 + bid)
            mean_u = float(np.mean([q.score for q in pool])) if pool else 1.0
            cap_tok = bucket.start_block(block_weight=len(sub_ds) / total_posts, mean_uncertainty=mean_u)

            if args.dry_run:
                est_list = []
                cum = 0
                count = 0
                for q in pool:
                    sm, um = build_query_prompt(q, sub_ds, summaries)
                    est = int((len(sm) + len(um)) / 3.5) + 60
                    if cum + est > cap_tok:
                        break
                    cum += est
                    count += 1
                    est_list.append(q.query_type)
                types_dict = {str(k): int(v) for k, v in zip(*np.unique(est_list, return_counts=True))} if est_list else {}
                print(f"  [DRY-RUN] M_{bid:02d} cap={cap_tok} tok -> ~{count} queries (~{cum} tok), types={types_dict}")
                bucket.record_spend(cum, 0)
                continue

            decisions = client.execute_queries(pool, sub_ds, summaries, block_id=bid, bucket=bucket)
            dec_ids = {d.query_id for d in decisions}
            used_queries = [q for q in pool if q.query_id in dec_ids]

            # Audit real LLM decision distribution & accuracy vs ground truth (diagnostic only!)
            action_counts: Dict[str, int] = {}
            for d in decisions:
                action_counts[d.action] = action_counts.get(d.action, 0) + 1

            # Branch 1 vs Branch 2 on the EXACT SAME LLM decisions
            lab_dir, qmask_dir = apply_direct_assignment(
                labels_init, used_queries, decisions, conf_threshold=args.conf_threshold
            )
            aud_dir = full_partition_audit(
                sub_ds.labels, lab_dir, y_before=labels_init, queried_mask=qmask_dir
            )

            lab_rep, qmask_rep, surg_stats = apply_topology_repair_and_recluster(
                graph, labels_init, summaries, used_queries, decisions, conf_threshold=args.conf_threshold
            )
            aud_rep = full_partition_audit(
                sub_ds.labels, lab_rep, y_before=labels_init, queried_mask=qmask_rep
            )

            p1 = aud_dir["propagation_audit"]
            p2 = aud_rep["propagation_audit"]
            block_tok = sum(d.prompt_tokens + d.completion_tokens for d in decisions)
            print(
                f"  M_{bid:02d} Q={len(decisions):2d} Tok={block_tok:5d}/{cap_tok:5d} {action_counts} | "
                f"Direct: dAMI={aud_dir['delta_ami']:+.4f}, dMacB3={aud_dir['delta_macro_b3_f1']:+.4f}, UnqNet={p1['unqueried_net_improved_posts']:+4d} || "
                f"Repair: dAMI={aud_rep['delta_ami']:+.4f}, dMacB3={aud_rep['delta_macro_b3_f1']:+.4f}, UnqNet={p2['unqueried_net_improved_posts']:+4d}"
            )

            for btag, aud in [("direct", aud_dir), ("repair", aud_rep)]:
                key = f"{strat}_{btag}"
                prop = aud["propagation_audit"]
                et = aud["error_transition"]
                agg.setdefault(key, []).append(
                    {
                        "queries": float(len(decisions)),
                        "tokens": float(block_tok),
                        "ami": aud["ami"],
                        "delta_ami": aud["delta_ami"],
                        "macro_b3": aud["bcubed_full"]["macro_b3_f1"],
                        "delta_macro_b3": aud["delta_macro_b3_f1"],
                        "micro_b3": aud["bcubed_full"]["micro_b3_f1"],
                        "delta_micro_b3": aud["delta_micro_b3_f1"],
                        "unq_delta_micro_b3": prop["unqueried_delta_micro_b3_f1"],
                        "unq_delta_macro_b3": prop["unqueried_delta_macro_b3_f1"],
                        "unq_net_posts": float(prop["unqueried_net_improved_posts"]),
                        "net_fm_red": float(et["net_false_merge_reduction"]),
                        "net_fs_red": float(et["net_false_split_reduction"]),
                        "tail_b3": aud["stratified"]["tail"]["macro_b3_f1"],
                    }
                )

            strat_blocks[f"M_{bid:02d}"] = {
                "n_queries": len(decisions),
                "block_tokens": block_tok,
                "block_cap_tokens": cap_tok,
                "action_counts": action_counts,
                "surgery_stats": surg_stats,
                "branch1_direct": aud_dir,
                "branch2_repair": aud_rep,
            }

        summary_report["strategies"][strat] = {
            "ledger_summary": ledger.summary(),
            "blocks": strat_blocks,
        }

    if not args.dry_run:
        print("\n================ REAL OPEN-SOURCE LLM GATE A SUMMARY (MEAN ACROSS DEV BLOCKS) ================")
        im = summary_report["init_mean"]
        print(f"Init 2D-SE Mean: AMI={im['ami']:.4f} | Macro-B3={im['macro_b3']:.4f} | Micro-B3={im['micro_b3']:.4f}")
        header = (
            f"{'Setting':22s} | {'TotTok':>7s} | {'Q/blk':>5s} | {'AMI':>6s} | {'dAMI':>7s} | "
            f"{'MacB3':>6s} | {'dMacB3':>7s} | {'Unq_dMic':>8s} | {'Unq_dMac':>8s} | {'UnqNet':>7s} | {'NetFM':>7s} | {'NetFS':>7s}"
        )
        print(header)
        print("-" * len(header))
        for key, rows in agg.items():
            m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()}
            tot_tok = float(sum(r["tokens"] for r in rows))
            m["total_tokens_all_blocks"] = tot_tok
            summary_report["mean_comparison"][key] = m
            print(
                f"{key:22s} | {tot_tok:7.0f} | {m['queries']:5.1f} | {m['ami']:6.4f} | {m['delta_ami']:+7.4f} | "
                f"{m['macro_b3']:6.4f} | {m['delta_macro_b3']:+7.4f} | "
                f"{m['unq_delta_micro_b3']:+8.4f} | {m['unq_delta_macro_b3']:+8.4f} | "
                f"{m['unq_net_posts']:+7.1f} | {m['net_fm_red']:+7.0f} | {m['net_fs_red']:+7.0f}"
            )

        res_path = out_dir / "gateA_real_llm_report.json"
        with res_path.open("w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved full real-LLM Gate A report to: {res_path}")


if __name__ == "__main__":
    main()
