"""Multi-Budget Real-Token Pareto Curve Runner (Qwen/Qwen3-8B, Strict Real-Token Billing).

Sweeps strict cumulative real (prompt_tokens + completion_tokens) budgets — e.g.,
0 (Init 2D-SE), 20k, 40k, 80k — across all 5 query strategies on identical blocks,
identical initial partitions, and identical Qwen/Qwen3-8B endpoints:
  - random
  - rep_div      (RagSEDE-style MMR representativeness + diversity)
  - cequel_edge  (Cequel-style pairwise edge uncertainty)
  - sug_router   (Full symmetric mesoscopic + bridge/seed router)
  - sug_meso_only (Pure binary mesoscopic MERGE_PAIR + SPLIT_SEAM router)

Reuses a shared persistent valid-response cache (`results/llm_cache/...`) across
budget tiers so overlapping prefixes cost zero duplicate API calls while STILL
billing every single executed/replayed query's exact prompt+completion tokens to
each budget tier's isolated CumulativeTokenBucket and journal.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
from trackA_sug_sed.high_leverage_router import build_cluster_summaries
from trackA_sug_sed.llm_client import BudgetedLLMClient, build_query_prompt
from trackA_sug_sed.run_gateA_oracle_diagnostic import load_or_extract_block_embeddings
from trackA_sug_sed.run_gateA_real_llm import _select_pool
from trackA_sug_sed.topology_surgery import (
    apply_direct_assignment,
    apply_topology_repair_and_recluster,
)


def _discover_seed_caches(dataset: str, model_name: str) -> List[Path]:
    """Collect valid response ledgers from earlier runs of the same dataset & model."""
    caches: List[Path] = []
    safe_tag = "qwen3_8b" if "qwen3-8b" in model_name.lower() else model_name.lower().replace("/", "_")
    gate_dir = ROOT / "results" / "gateA_llm"
    if gate_dir.exists():
        for run_dir in sorted(gate_dir.glob(f"{dataset}_*{safe_tag}*")):
            for led_dir in sorted(run_dir.glob("ledger_*")):
                rp = led_dir / "responses.jsonl"
                if rp.exists():
                    caches.append(rp)
    return caches


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Budget Real-Token Pareto Curve Runner")
    parser.add_argument("--dataset", required=True, choices=["Event2012", "Event2012_100", "Event2018"])
    parser.add_argument("--blocks", default="2,3,4,5", help="Comma-separated block IDs (default dev M_02..M_05)")
    parser.add_argument("--budgets", default="20000,40000,80000", help="Comma-separated strict real-token budgets")
    parser.add_argument(
        "--strategies",
        default="random,rep_div,cequel_edge,sug_router,sug_meso_only",
    )
    parser.add_argument("--max-queries-per-block", type=int, default=120)
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--knn-sim-threshold", type=float, default=0.35)
    parser.add_argument("--attr-alpha", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.6)
    parser.add_argument("--price-prompt-cny", type=float, default=0.0)
    parser.add_argument("--price-completion-cny", type=float, default=0.0)
    parser.add_argument("--run-tag", default="dev_qwen3_8b")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    embed_model = args.embed_model or (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if args.dataset == "Event2018"
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    block_ids = [int(x.strip()) for x in args.blocks.split(",") if x.strip()]
    budgets = sorted([int(x.strip()) for x in args.budgets.split(",") if x.strip()])
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]

    if not args.dry_run:
        prov_url, prov_model, api_key = load_provider_key(args.provider)
        base_url = args.base_url or prov_url
        model_name = args.model or prov_model
    else:
        api_key = ""
        base_url = args.base_url or "dry-run"
        model_name = args.model

    safe_model = model_name.replace("/", "__")
    out_dir = ROOT / "results" / "budget_curves" / f"{args.dataset}_{args.run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_cache_file = ROOT / "results" / "llm_cache" / f"{args.dataset}_{safe_model}" / "responses.jsonl"
    seed_caches = _discover_seed_caches(args.dataset, model_name)
    print(f"[Cache] Shared cache: {shared_cache_file} | Seed files found: {len(seed_caches)}")

    ds_full = load_socialed_npy(args.dataset, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)

    # Precompute initial graphs, partitions, and deterministic query pools once
    block_cache: Dict[int, Dict[str, Any]] = {}
    total_posts = 0
    for bid in block_ids:
        bname, bidx = blocks[bid]
        sub_ds = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        emb = load_or_extract_block_embeddings(args.dataset, bid, sub_ds.texts, model_name=embed_model)
        graph = build_message_graph(
            sub_ds,
            emb,
            knn_k=args.knn_k,
            knn_sim_threshold=args.knn_sim_threshold,
            attr_alpha=args.attr_alpha,
        )
        labels_init, _ = solve_2d_se(graph.adj, refine_moves=True)
        summaries = build_cluster_summaries(sub_ds, emb, labels_init)
        init_audit = full_partition_audit(sub_ds.labels, labels_init)
        pools = {
            strat: _select_pool(strat, sub_ds, graph, labels_init, args.max_queries_per_block, seed=42 + bid)
            for strat in strategies
        }
        block_cache[bid] = {
            "name": bname,
            "ds": sub_ds,
            "graph": graph,
            "labels_init": labels_init,
            "summaries": summaries,
            "init_audit": init_audit,
            "pools": pools,
        }
        total_posts += len(sub_ds)
        pool_lens = {s: len(p) for s, p in pools.items()}
        print(
            f"[Init] M_{bid:02d} [{bname}] n={len(sub_ds)} K={init_audit['n_pred_clusters']} "
            f"AMI={init_audit['ami']:.4f} MacB3={init_audit['bcubed_full']['macro_b3_f1']:.4f} "
            f"MicB3={init_audit['bcubed_full']['micro_b3_f1']:.4f} | pools={pool_lens}"
        )

    init_mean = {
        "ami": float(np.mean([block_cache[b]["init_audit"]["ami"] for b in block_ids])),
        "nmi": float(np.mean([block_cache[b]["init_audit"]["nmi"] for b in block_ids])),
        "ari": float(np.mean([block_cache[b]["init_audit"]["ari"] for b in block_ids])),
        "macro_b3": float(np.mean([block_cache[b]["init_audit"]["bcubed_full"]["macro_b3_f1"] for b in block_ids])),
        "micro_b3": float(np.mean([block_cache[b]["init_audit"]["bcubed_full"]["micro_b3_f1"] for b in block_ids])),
        "tail_b3": float(np.mean([block_cache[b]["init_audit"]["stratified"]["tail"]["macro_b3_f1"] for b in block_ids])),
    }

    if args.dry_run:
        # Check how many queries fit and how many are already cached vs live needed
        probe_ledger = TokenLedger(
            out_dir / "_dry_probe",
            extra_cache_dirs=seed_caches,
            shared_cache_file=shared_cache_file,
        )
        from core_shared.token_ledger import canonical_request_hash

        for B in budgets:
            print(f"\n--- DRY-RUN Budget={B} real tokens ---")
            for strat in strategies:
                bucket = CumulativeTokenBucket(total_token_budget=B, num_blocks=len(block_ids))
                tot_q = tot_cached = tot_live = tot_est = 0
                for bid in block_ids:
                    bc = block_cache[bid]
                    pool = bc["pools"][strat]
                    mean_u = float(np.mean([q.score for q in pool])) if pool else 1.0
                    cap = bucket.start_block(len(bc["ds"]) / total_posts, mean_u)
                    for q in pool:
                        sm, um = build_query_prompt(q, bc["ds"], bc["summaries"])
                        est = int((len(sm) + len(um)) / 3.5) + 120
                        if not bucket.can_spend(est):
                            break
                        rh = canonical_request_hash(
                            {
                                "model": model_name,
                                "temperature": 0.0,
                                "messages": [
                                    {"role": "system", "content": sm},
                                    {"role": "user", "content": um},
                                ],
                            }
                        )
                        hit = probe_ledger.lookup_cache(rh)
                        if hit is not None:
                            pt, ct = int(hit["prompt_tokens"]), int(hit["completion_tokens"])
                            tot_cached += 1
                            bucket.record_spend(pt, ct)
                            tot_est += pt + ct
                        else:
                            tot_live += 1
                            bucket.record_spend(est - 60, 60)
                            tot_est += est
                        tot_q += 1
                print(
                    f"  {strat:14s}: total_q={tot_q:3d} (cached={tot_cached:3d}, live_needed={tot_live:3d}), est_tok={tot_est:6d}/{B}"
                )
        return

    # Execute largest budget FIRST so smaller budgets replay deterministically from shared cache
    exec_budgets = sorted(budgets, reverse=True)
    results_by_budget: Dict[int, Dict[str, Any]] = {}
    csv_rows: List[Dict[str, Any]] = []

    # Add budget=0 baseline rows for every strategy/branch
    for strat in strategies:
        for branch in ["direct", "repair"]:
            csv_rows.append(
                {
                    "dataset": args.dataset,
                    "model": model_name,
                    "budget_cap": 0,
                    "strategy": strat,
                    "branch": branch,
                    "total_tokens": 0,
                    "live_calls": 0,
                    "cached_calls": 0,
                    "queries_per_block": 0.0,
                    "ami": init_mean["ami"],
                    "delta_ami": 0.0,
                    "nmi": init_mean["nmi"],
                    "ari": init_mean["ari"],
                    "macro_b3": init_mean["macro_b3"],
                    "delta_macro_b3": 0.0,
                    "micro_b3": init_mean["micro_b3"],
                    "delta_micro_b3": 0.0,
                    "unq_delta_micro_b3": 0.0,
                    "unq_delta_macro_b3": 0.0,
                    "unq_net_posts": 0.0,
                    "net_fm_red": 0.0,
                    "net_fs_red": 0.0,
                    "tail_b3": init_mean["tail_b3"],
                }
            )

    t_all = time.time()
    for B in exec_budgets:
        print(f"\n==================== BUDGET TIER: {B:,} REAL TOKENS ({args.dataset} · {model_name}) ====================")
        tier_dir = out_dir / f"budget_{B}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        tier_summary: Dict[str, Any] = {"budget_cap": B, "strategies": {}, "mean_comparison": {}}

        for strat in strategies:
            t_s = time.time()
            # Fresh journal for this (budget, strategy), backed by shared persistent valid-response cache
            led_dir = tier_dir / f"ledger_{strat}"
            if (led_dir / "journal.jsonl").exists():
                (led_dir / "journal.jsonl").unlink()
            ledger = TokenLedger(
                led_dir,
                price_per_1m_prompt_cny=args.price_prompt_cny,
                price_per_1m_completion_cny=args.price_completion_cny,
                extra_cache_dirs=seed_caches,
                shared_cache_file=shared_cache_file,
            )
            bucket = CumulativeTokenBucket(total_token_budget=B, num_blocks=len(block_ids))
            client = BudgetedLLMClient(
                model=model_name,
                ledger=ledger,
                base_url=base_url,
                api_key=api_key,
            )

            strat_blocks: Dict[str, Any] = {}
            branch_rows: Dict[str, List[Dict[str, float]]] = {"direct": [], "repair": []}

            for bid in block_ids:
                bc = block_cache[bid]
                sub_ds = bc["ds"]
                graph = bc["graph"]
                labels_init = bc["labels_init"]
                summaries = bc["summaries"]
                pool = bc["pools"][strat]

                mean_u = float(np.mean([q.score for q in pool])) if pool else 1.0
                cap_tok = bucket.start_block(block_weight=len(sub_ds) / total_posts, mean_uncertainty=mean_u)

                decisions = client.execute_queries(pool, sub_ds, summaries, block_id=bid, bucket=bucket)
                dec_ids = {d.query_id for d in decisions}
                used_queries = [q for q in pool if q.query_id in dec_ids]

                action_counts: Dict[str, int] = {}
                for d in decisions:
                    action_counts[d.action] = action_counts.get(d.action, 0) + 1

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

                block_tok = sum(d.prompt_tokens + d.completion_tokens for d in decisions)
                for btag, aud in [("direct", aud_dir), ("repair", aud_rep)]:
                    prop = aud["propagation_audit"]
                    et = aud["error_transition"]
                    branch_rows[btag].append(
                        {
                            "queries": float(len(decisions)),
                            "tokens": float(block_tok),
                            "ami": aud["ami"],
                            "delta_ami": aud["delta_ami"],
                            "nmi": aud["nmi"],
                            "ari": aud["ari"],
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

            lsum = ledger.summary()
            live_calls = lsum["total_calls"] - lsum["cached_calls"]
            tier_summary["strategies"][strat] = {
                "ledger_summary": lsum,
                "blocks": strat_blocks,
            }

            for btag in ["direct", "repair"]:
                rows = branch_rows[btag]
                m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()}
                tot_tok = int(sum(r["tokens"] for r in rows))
                m["total_tokens_all_blocks"] = float(tot_tok)
                tier_summary["mean_comparison"][f"{strat}_{btag}"] = m
                csv_rows.append(
                    {
                        "dataset": args.dataset,
                        "model": model_name,
                        "budget_cap": B,
                        "strategy": strat,
                        "branch": btag,
                        "total_tokens": tot_tok,
                        "live_calls": live_calls,
                        "cached_calls": lsum["cached_calls"],
                        "queries_per_block": round(m["queries"], 2),
                        "ami": round(m["ami"], 6),
                        "delta_ami": round(m["delta_ami"], 6),
                        "nmi": round(m["nmi"], 6),
                        "ari": round(m["ari"], 6),
                        "macro_b3": round(m["macro_b3"], 6),
                        "delta_macro_b3": round(m["delta_macro_b3"], 6),
                        "micro_b3": round(m["micro_b3"], 6),
                        "delta_micro_b3": round(m["delta_micro_b3"], 6),
                        "unq_delta_micro_b3": round(m["unq_delta_micro_b3"], 6),
                        "unq_delta_macro_b3": round(m["unq_delta_macro_b3"], 6),
                        "unq_net_posts": round(m["unq_net_posts"], 2),
                        "net_fm_red": round(m["net_fm_red"], 2),
                        "net_fs_red": round(m["net_fs_red"], 2),
                        "tail_b3": round(m["tail_b3"], 6),
                    }
                )

            md = tier_summary["mean_comparison"][f"{strat}_direct"]
            mr = tier_summary["mean_comparison"][f"{strat}_repair"]
            print(
                f"  [{strat:14s}] tok={lsum['total_tokens']:6d}/{B} (live={live_calls:3d}, cache={lsum['cached_calls']:3d}, {time.time()-t_s:4.1f}s) | "
                f"Dir: dAMI={md['delta_ami']:+.4f} dMacB3={md['delta_macro_b3']:+.4f} UnqMic={md['unq_delta_micro_b3']:+.4f} || "
                f"Rep: dAMI={mr['delta_ami']:+.4f} dMacB3={mr['delta_macro_b3']:+.4f} UnqMic={mr['unq_delta_micro_b3']:+.4f}"
            )

        results_by_budget[B] = tier_summary

    # Sort CSV rows cleanly by (budget_cap, strategy, branch)
    strat_order = {s: i for i, s in enumerate(strategies)}
    csv_rows.sort(key=lambda r: (r["budget_cap"], strat_order.get(r["strategy"], 99), r["branch"]))

    csv_path = out_dir / f"{args.dataset}_{args.run_tag}_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    full_report = {
        "dataset": args.dataset,
        "blocks": block_ids,
        "model": model_name,
        "embed_model": embed_model,
        "budgets": [0] + budgets,
        "init_mean": init_mean,
        "elapsed_sec": round(time.time() - t_all, 2),
        "tiers": {str(b): results_by_budget[b] for b in budgets},
        "table_rows": csv_rows,
    }
    json_path = out_dir / f"{args.dataset}_{args.run_tag}_curve.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n==================== SUMMARY TABLE ({args.dataset} · {model_name}) ====================")
    print(f"Init 2D-SE: AMI={init_mean['ami']:.4f}, Macro-B3={init_mean['macro_b3']:.4f}, Micro-B3={init_mean['micro_b3']:.4f}")
    hdr = f"{'Budget':>7s} | {'Setting':21s} | {'RealTok':>7s} | {'Q/blk':>5s} | {'AMI':>6s} | {'dAMI':>7s} | {'MacB3':>6s} | {'dMacB3':>7s} | {'UnqMic':>7s} | {'UnqNet':>7s} | {'NetFS':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in csv_rows:
        if r["budget_cap"] == 0:
            continue
        setting = f"{r['strategy']}_{r['branch']}"
        print(
            f"{r['budget_cap']:7d} | {setting:21s} | {r['total_tokens']:7d} | {r['queries_per_block']:5.1f} | "
            f"{r['ami']:6.4f} | {r['delta_ami']:+7.4f} | {r['macro_b3']:6.4f} | {r['delta_macro_b3']:+7.4f} | "
            f"{r['unq_delta_micro_b3']:+7.4f} | {r['unq_net_posts']:+7.1f} | {r['net_fs_red']:+7.0f}"
        )
    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
