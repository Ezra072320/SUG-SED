"""End-to-End Cross-Block Streaming SED Pipeline (Within-Block + Longitudinal Memory).

Evaluates two complementary streaming benchmarks under strict chronological arrival
and strict cumulative real-token accounting (Qwen/Qwen3-8B):

  1. Standard Per-Block Streaming Evaluation (KPGNN / HISEvent / RPLM-SED protocol):
     Processes blocks M_start..M_end sequentially under a single global
     CumulativeTokenBucket, comparing Init 2D-SE, Random, Rep-Div (RagSEDE),
     Cequel-Edge, SUG-Router, and SUG-Meso (both Branch 1 Direct and Branch 2 Repair).

  2. Longitudinal Cross-Block Event Tracking (Multi-Week Global Partition):
     Maintains a lightweight O(|E_active| * d) GlobalEventMemory (running weighted
     prototype, top-3 medoid texts, top-5 keywords, last-seen block) and links
     each day's finalized clusters to ongoing historical events via three modes:
       - `no_link`       : Independent daily cluster IDs (lower bound)
       - `proto_only`    : Zero-LLM cosine prototype threshold linking
       - `sug_cross_llm` : High-sim auto-link (>= tau_hi) + Qwen3-8B Extract-Then-Compare
                           adjudication in the ambiguous band [tau_lo, tau_hi)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset, load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import build_message_graph
from core_shared.key_loader import load_provider_key
from core_shared.metrics_audit import full_partition_audit
from core_shared.se_solver import solve_2d_se
from core_shared.token_ledger import CumulativeTokenBucket, TokenLedger, canonical_request_hash
from trackA_sug_sed.high_leverage_router import ClusterSummary, build_cluster_summaries
from trackA_sug_sed.llm_client import BudgetedLLMClient, _extract_json, build_query_prompt
from trackA_sug_sed.run_gateA_oracle_diagnostic import load_or_extract_block_embeddings
from trackA_sug_sed.run_gateA_real_llm import _select_pool
from trackA_sug_sed.topology_surgery import (
    apply_direct_assignment,
    apply_topology_repair_and_recluster,
)


CROSS_BLOCK_SYS_MSG = (
    "You decide whether social-media posts or post groups refer to the SAME specific real-world event.\n"
    "Rules:\n"
    "- SAME EVENT (true / ASSIGN) if both cover the same specific incident, match, announcement, or developing story, "
    "even when they highlight different details, updates, players/angles, or reactions "
    "(e.g., 'Team X beats Team Y 4-2' and 'Player Z scores for Team X against Team Y' -> true; "
    "'Official resigns' and 'Official quits over scandal' -> true).\n"
    "- DIFFERENT EVENT (false / NEW_EVENT) if they cover different incidents, different matches, different countries, "
    "or different main subjects even if sharing a broad topic "
    "(e.g., 'Team X vs Team Y' and 'Team A vs Team B' -> false; 'Scandal at Org 1' and 'Scandal at Org 2' -> false).\n"
    "Always write concise 5-10 word factual event summaries FIRST, then output your final boolean or action in strict JSON."
)


@dataclass
class HistoricalEvent:
    global_id: int
    prototype: np.ndarray          # L2-normalized running mean embedding
    total_count: int
    first_block: int
    last_block: int
    rep_texts: List[str]           # Up to 3 central/recent medoid texts
    top_keywords: List[str] = field(default_factory=list)


class GlobalEventMemory:
    """Lightweight O(|E| * d) streaming memory for cross-block event linking."""

    def __init__(self, window_blocks: int = 14) -> None:
        self.window_blocks = int(window_blocks)
        self.events: Dict[int, HistoricalEvent] = {}
        self.next_gid: int = 0

    def active_events(self, current_block: int) -> List[HistoricalEvent]:
        return [
            ev for ev in self.events.values()
            if (current_block - ev.last_block) <= self.window_blocks
        ]

    def spawn_event(
        self,
        block_id: int,
        summ: ClusterSummary,
        ds: SocialDataset,
    ) -> int:
        gid = self.next_gid
        self.next_gid += 1
        rep_texts = [
            ds.texts[m].replace("\n", " ").strip()[:140]
            for m in summ.medoids[:3]
        ]
        self.events[gid] = HistoricalEvent(
            global_id=gid,
            prototype=summ.prototype.astype(np.float32, copy=True),
            total_count=int(summ.size),
            first_block=int(block_id),
            last_block=int(block_id),
            rep_texts=rep_texts,
            top_keywords=list(summ.top_keywords[:5]),
        )
        return gid

    def update_event(
        self,
        gid: int,
        block_id: int,
        summ: ClusterSummary,
        ds: SocialDataset,
        ema_cap_weight: int = 200,
    ) -> None:
        ev = self.events[gid]
        w_old = float(min(ev.total_count, ema_cap_weight))
        w_new = float(summ.size)
        proto = (w_old * ev.prototype + w_new * summ.prototype)
        norm = float(np.linalg.norm(proto))
        if norm > 1e-12:
            proto = proto / norm
        ev.prototype = proto.astype(np.float32)
        ev.total_count += int(summ.size)
        ev.last_block = int(block_id)
        # Keep representative texts fresh: 2 prior + 1 newest medoid
        new_txt = ds.texts[summ.medoids[0]].replace("\n", " ").strip()[:140] if summ.medoids else ""
        if new_txt and new_txt not in ev.rep_texts:
            ev.rep_texts = (ev.rep_texts[:2] + [new_txt])[:3]
        # Merge keywords keeping order
        merged_kw = list(dict.fromkeys(ev.top_keywords[:3] + summ.top_keywords[:3]))[:5]
        ev.top_keywords = merged_kw


def _adjudicate_cross_block_llm(
    client: BudgetedLLMClient,
    hist_ev: HistoricalEvent,
    cur_summ: ClusterSummary,
    cur_ds: SocialDataset,
    block_id: int,
    bucket: Optional[CumulativeTokenBucket],
) -> Tuple[Optional[bool], int]:
    """Ask frozen Qwen/Qwen3-8B Extract-Then-Compare whether cur_summ continues hist_ev."""
    h_lines = "\n".join(f"  - {t}" for t in hist_ev.rep_texts[:3])
    c_lines = "\n".join(
        f"  - {cur_ds.texts[m].replace(chr(10), ' ').strip()[:140]}"
        for m in cur_summ.medoids[:3]
    )
    h_kw = ", ".join(hist_ev.top_keywords[:5])
    c_kw = ", ".join(cur_summ.top_keywords[:5])
    user_msg = (
        f"Group A (ongoing historical event, size={hist_ev.total_count}, keywords: {h_kw}):\n{h_lines}\n\n"
        f"Group B (new daily cluster, size={cur_summ.size}, keywords: {c_kw}):\n{c_lines}\n\n"
        "First summarize Group A's specific event (event_a) and Group B's specific event (event_b) in 5-10 words each, "
        "then set same_event to true ONLY if both describe the exact same specific incident.\n"
        'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", "same_event": true or false}'
    )
    est_tokens = int((len(CROSS_BLOCK_SYS_MSG) + len(user_msg)) / 3.5) + 120
    if bucket is not None and not bucket.can_spend(est_tokens):
        return None, 0

    req_payload = {
        "model": client.model,
        "temperature": client.temperature,
        "messages": [
            {"role": "system", "content": CROSS_BLOCK_SYS_MSG},
            {"role": "user", "content": user_msg},
        ],
    }
    req_hash = canonical_request_hash(req_payload)
    qid = f"cross_M{block_id:02d}_g{hist_ev.global_id}_c{cur_summ.cluster_id}"
    cached = client.ledger.lookup_cache(req_hash)

    if cached is not None:
        parsed = cached["response"]
        pt = int(cached["prompt_tokens"])
        ct = int(cached["completion_tokens"])
        client.ledger.log_call(
            request_id=qid,
            block_id=block_id,
            strategy="sug_cross_llm",
            query_type="cross_merge",
            model=client.model,
            request_hash=req_hash,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_sec=0.0,
            status="CACHED_REPLAY",
            response_payload=parsed,
            from_cache=True,
        )
    else:
        pt, ct, latency = 0, 0, 0.0
        try:
            raw_text, pt, ct, latency, finish_reason = client._call_http(req_payload["messages"])
            try:
                parsed = _extract_json(raw_text)
                status = "OK"
            except Exception as parse_exc:
                parsed = {"error": str(parse_exc), "finish_reason": finish_reason}
                status = f"PARSE_ERROR:{type(parse_exc).__name__}"
        except Exception as http_exc:
            parsed = {"error": str(http_exc)}
            status = f"HTTP_ERROR:{type(http_exc).__name__}"

        client.ledger.log_call(
            request_id=qid,
            block_id=block_id,
            strategy="sug_cross_llm",
            query_type="cross_merge",
            model=client.model,
            request_hash=req_hash,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_sec=latency,
            status=status,
            response_payload=parsed,
            from_cache=False,
        )

    if bucket is not None:
        bucket.record_spend(pt, ct)

    if "error" in parsed or "same_event" not in parsed:
        return None, pt + ct
    return bool(parsed["same_event"]), pt + ct


def link_block_to_memory(
    memory: GlobalEventMemory,
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    block_labels: np.ndarray,
    mode: str = "proto_only",
    tau_proto: float = 0.68,
    tau_lo: float = 0.50,
    tau_hi: float = 0.78,
    min_cross_size: int = 3,
    client: Optional[BudgetedLLMClient] = None,
    bucket: Optional[CumulativeTokenBucket] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Map block_labels to persistent global event IDs and update GlobalEventMemory."""
    summaries = build_cluster_summaries(ds, embeddings, block_labels)
    # Process larger clusters first so major events claim/update their historical ID cleanly
    ordered_cids = sorted(summaries.keys(), key=lambda c: -summaries[c].size)

    active = memory.active_events(block_id)
    if active:
        act_gids = [ev.global_id for ev in active]
        act_protos = np.vstack([ev.prototype for ev in active])
    else:
        act_gids = []
        act_protos = np.zeros((0, embeddings.shape[1]), dtype=np.float32)

    local_to_global: Dict[int, int] = {}
    stats = {
        "n_clusters": len(ordered_cids),
        "auto_linked": 0,
        "llm_queried": 0,
        "llm_linked": 0,
        "llm_rejected": 0,
        "spawned_new": 0,
        "cross_tokens": 0,
    }

    for cid in ordered_cids:
        summ = summaries[cid]
        if mode == "no_link" or not act_gids:
            gid = memory.spawn_event(block_id, summ, ds)
            local_to_global[cid] = gid
            stats["spawned_new"] += 1
            continue

        sims = act_protos @ summ.prototype
        best_pos = int(np.argmax(sims))
        best_sim = float(sims[best_pos])
        best_gid = act_gids[best_pos]

        if mode == "proto_only":
            if best_sim >= tau_proto:
                memory.update_event(best_gid, block_id, summ, ds)
                local_to_global[cid] = best_gid
                stats["auto_linked"] += 1
            else:
                gid = memory.spawn_event(block_id, summ, ds)
                local_to_global[cid] = gid
                stats["spawned_new"] += 1
        elif mode == "sug_cross_llm":
            if best_sim >= tau_hi:
                memory.update_event(best_gid, block_id, summ, ds)
                local_to_global[cid] = best_gid
                stats["auto_linked"] += 1
            elif best_sim >= tau_lo and summ.size >= min_cross_size and client is not None:
                verdict, tok = _adjudicate_cross_block_llm(
                    client, memory.events[best_gid], summ, ds, block_id, bucket
                )
                stats["cross_tokens"] += tok
                if verdict is not None:
                    stats["llm_queried"] += 1
                if verdict is True:
                    memory.update_event(best_gid, block_id, summ, ds)
                    local_to_global[cid] = best_gid
                    stats["llm_linked"] += 1
                elif verdict is False:
                    gid = memory.spawn_event(block_id, summ, ds)
                    local_to_global[cid] = gid
                    stats["llm_rejected"] += 1
                    stats["spawned_new"] += 1
                else:
                    # Budget exhausted or error -> fall back cleanly to prototype threshold
                    if best_sim >= tau_proto:
                        memory.update_event(best_gid, block_id, summ, ds)
                        local_to_global[cid] = best_gid
                        stats["auto_linked"] += 1
                    else:
                        gid = memory.spawn_event(block_id, summ, ds)
                        local_to_global[cid] = gid
                        stats["spawned_new"] += 1
            else:
                # Below tau_lo OR tiny micro-cluster (size < min_cross_size): use tau_proto fallback
                if best_sim >= tau_proto:
                    memory.update_event(best_gid, block_id, summ, ds)
                    local_to_global[cid] = best_gid
                    stats["auto_linked"] += 1
                else:
                    gid = memory.spawn_event(block_id, summ, ds)
                    local_to_global[cid] = gid
                    stats["spawned_new"] += 1
        else:
            raise ValueError(f"Unknown cross-block link mode: {mode}")

    global_labels = np.array([local_to_global[int(c)] for c in block_labels.tolist()], dtype=np.int64)
    return global_labels, stats


def calibrate_cross_thresholds_on_dev(
    dataset_name: str,
    embed_model: str,
    dev_end_block: int = 5,
    knn_k: int = 15,
    knn_sim_threshold: float = 0.35,
    attr_alpha: float = 0.5,
) -> Dict[str, Any]:
    """Zero-LLM dev-only calibration of cross-block prototype similarity thresholds (M_00..M_05)."""
    ds_full = load_socialed_npy(dataset_name, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
    use_blocks = list(range(0, min(dev_end_block + 1, len(blocks))))

    print(f"\n[Dev Cross-Block Calibration] {dataset_name} blocks M_00..M_{use_blocks[-1]:02d} (strictly dev-only!)")
    block_data = []
    for bid in use_blocks:
        bname, bidx = blocks[bid]
        sub_ds = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        emb = load_or_extract_block_embeddings(dataset_name, bid, sub_ds.texts, model_name=embed_model)
        t0 = time.time()
        graph = build_message_graph(
            sub_ds, emb, knn_k=knn_k, knn_sim_threshold=knn_sim_threshold, attr_alpha=attr_alpha
        )
        labels_init, _ = solve_2d_se(graph.adj, refine_moves=True)
        print(f"  M_{bid:02d} [{bname}] n={len(sub_ds):5d} K={len(set(labels_init.tolist())):3d} solved in {time.time()-t0:.1f}s")
        block_data.append((bid, sub_ds, emb, labels_init))

    # Sweep tau_proto on dev stream M_00..M_05
    best_tau = 0.68
    best_ami = -1.0
    sweep_res = []
    for tau in [0.55, 0.60, 0.65, 0.68, 0.72, 0.76, 0.80]:
        mem = GlobalEventMemory(window_blocks=14)
        all_true, all_pred = [], []
        for bid, sub_ds, emb, lab in block_data:
            glab, _ = link_block_to_memory(mem, bid, sub_ds, emb, lab, mode="proto_only", tau_proto=tau)
            if bid >= 2:
                all_true.append(sub_ds.labels)
                all_pred.append(glab)
        y_t = np.concatenate(all_true)
        y_p = np.concatenate(all_pred)
        aud = full_partition_audit(y_t, y_p)
        row = {
            "tau_proto": tau,
            "global_dev_ami": round(aud["ami"], 4),
            "global_dev_macro_b3": round(aud["bcubed_full"]["macro_b3_f1"], 4),
            "global_dev_micro_b3": round(aud["bcubed_full"]["micro_b3_f1"], 4),
        }
        sweep_res.append(row)
        print(f"  tau_proto={tau:.2f} -> M_02..M_05 Stream AMI={row['global_dev_ami']:.4f}, MacB3={row['global_dev_macro_b3']:.4f}, MicB3={row['global_dev_micro_b3']:.4f}")
        if aud["ami"] > best_ami:
            best_ami = aud["ami"]
            best_tau = tau

    # Also evaluate no_link lower bound on M_02..M_05
    mem0 = GlobalEventMemory(window_blocks=14)
    yt0, yp0 = [], []
    for bid, sub_ds, emb, lab in block_data:
        glab, _ = link_block_to_memory(mem0, bid, sub_ds, emb, lab, mode="no_link")
        if bid >= 2:
            yt0.append(sub_ds.labels)
            yp0.append(glab)
    aud0 = full_partition_audit(np.concatenate(yt0), np.concatenate(yp0))
    print(f"  no_link        -> M_02..M_05 Stream AMI={aud0['ami']:.4f}, MacB3={aud0['bcubed_full']['macro_b3_f1']:.4f}, MicB3={aud0['bcubed_full']['micro_b3_f1']:.4f}")

    return {"dataset": dataset_name, "best_tau_proto": best_tau, "no_link": aud0, "sweep": sweep_res}


def run_streaming_evaluation(args: argparse.Namespace) -> None:
    embed_model = args.embed_model or (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if args.dataset == "Event2018"
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    if args.calibrate_dev_only:
        res = calibrate_cross_thresholds_on_dev(
            args.dataset,
            embed_model=embed_model,
            dev_end_block=5,
            knn_k=args.knn_k,
            knn_sim_threshold=args.knn_sim_threshold,
            attr_alpha=args.attr_alpha,
        )
        out_p = ROOT / "results" / "streaming" / f"{args.dataset}_dev_cross_calibration.json"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved dev calibration to {out_p}")
        return

    ds_full = load_socialed_npy(args.dataset, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
    n_blocks_total = len(blocks)

    if args.eval_blocks == "heldout":
        eval_block_ids = list(range(6, n_blocks_total))
        warmup_block_ids = list(range(0, 6))
    elif args.eval_blocks == "dev":
        eval_block_ids = [2, 3, 4, 5]
        warmup_block_ids = [0, 1]
    else:
        eval_block_ids = [int(x.strip()) for x in args.eval_blocks.split(",") if x.strip()]
        warmup_block_ids = list(range(0, min(eval_block_ids)))

    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    within_budget = args.within_token_budget
    if within_budget is None:
        within_budget = args.tokens_per_block * len(eval_block_ids)
    cross_budget = args.cross_token_budget
    if cross_budget is None:
        cross_budget = args.cross_tokens_per_block * len(eval_block_ids)

    if not args.dry_run:
        prov_url, prov_model, api_key = load_provider_key(args.provider)
        base_url = args.base_url or prov_url
        model_name = args.model or prov_model
    else:
        api_key = ""
        base_url = args.base_url or "dry-run"
        model_name = args.model

    safe_model = model_name.replace("/", "__")
    out_dir = ROOT / "results" / "streaming" / f"{args.dataset}_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shared_cache_file = ROOT / "results" / "llm_cache" / f"{args.dataset}_{safe_model}" / "responses.jsonl"

    print(
        f"================ STREAMING SED: {args.dataset} ({model_name}) ================\n"
        f"Warmup blocks : {warmup_block_ids} (memory seed only, 0 LLM tokens)\n"
        f"Eval blocks   : M_{eval_block_ids[0]:02d}..M_{eval_block_ids[-1]:02d} ({len(eval_block_ids)} blocks)\n"
        f"Within budget : {within_budget:,} real tokens ({within_budget / len(eval_block_ids):.0f}/blk)\n"
        f"Cross budget  : {cross_budget:,} real tokens ({cross_budget / len(eval_block_ids):.0f}/blk)\n"
        f"Strategies    : {strategies}"
    )

    # 1. Precompute base 2D-SE partitions for warmup + eval blocks once
    all_needed = sorted(set(warmup_block_ids + eval_block_ids))
    block_cache: Dict[int, Dict[str, Any]] = {}
    eval_total_posts = 0
    t_pre = time.time()
    for bid in all_needed:
        bname, bidx = blocks[bid]
        sub_ds = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        emb = load_or_extract_block_embeddings(args.dataset, bid, sub_ds.texts, model_name=embed_model)
        t0 = time.time()
        graph = build_message_graph(
            sub_ds, emb, knn_k=args.knn_k, knn_sim_threshold=args.knn_sim_threshold, attr_alpha=args.attr_alpha
        )
        labels_init, _ = solve_2d_se(graph.adj, refine_moves=True)
        summaries = build_cluster_summaries(sub_ds, emb, labels_init)
        init_audit = full_partition_audit(sub_ds.labels, labels_init)
        block_cache[bid] = {
            "name": bname,
            "ds": sub_ds,
            "emb": emb,
            "graph": graph,
            "labels_init": labels_init,
            "summaries": summaries,
            "init_audit": init_audit,
            "solve_sec": round(time.time() - t0, 2),
        }
        if bid in eval_block_ids:
            eval_total_posts += len(sub_ds)
        tag = "EVAL" if bid in eval_block_ids else "SEED"
        print(
            f"  [{tag}] M_{bid:02d} [{bname:21s}] n={len(sub_ds):5d} K={init_audit['n_pred_clusters']:3d} "
            f"AMI={init_audit['ami']:.4f} MacB3={init_audit['bcubed_full']['macro_b3_f1']:.4f} "
            f"MicB3={init_audit['bcubed_full']['micro_b3_f1']:.4f} ({block_cache[bid]['solve_sec']:.1f}s)"
        )
    print(f"[Precompute] Built & solved {len(all_needed)} blocks in {time.time()-t_pre:.1f}s | eval_posts={eval_total_posts:,}")

    if args.dry_run:
        probe_ledger = TokenLedger(out_dir / "_dry_probe", shared_cache_file=shared_cache_file)
        for strat in strategies:
            bucket = CumulativeTokenBucket(total_token_budget=within_budget, num_blocks=len(eval_block_ids))
            tot_q = tot_cached = tot_live = tot_est = 0
            for bid in eval_block_ids:
                bc = block_cache[bid]
                pool = _select_pool(strat, bc["ds"], bc["graph"], bc["labels_init"], args.max_queries_per_block, seed=42 + bid)
                mean_u = float(np.mean([q.score for q in pool])) if pool else 1.0
                bucket.start_block(len(bc["ds"]) / eval_total_posts, mean_u)
                for q in pool:
                    sm, um = build_query_prompt(q, bc["ds"], bc["summaries"])
                    est = int((len(sm) + len(um)) / 3.5) + 120
                    if not bucket.can_spend(est):
                        break
                    rh = canonical_request_hash(
                        {
                            "model": model_name,
                            "temperature": 0.0,
                            "messages": [{"role": "system", "content": sm}, {"role": "user", "content": um}],
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
            print(f"  [DRY-RUN] {strat:14s}: total_q={tot_q:4d} (cached={tot_cached:3d}, live={tot_live:4d}), est_tok={tot_est:7d}/{within_budget}")
        # Estimate cross-block ambiguous pairs on init partitions
        mem_est = GlobalEventMemory(window_blocks=args.memory_window)
        ambig_cnt = 0
        for bid in warmup_block_ids:
            bc = block_cache[bid]
            link_block_to_memory(mem_est, bid, bc["ds"], bc["emb"], bc["labels_init"], mode="proto_only", tau_proto=args.tau_proto)
        for bid in eval_block_ids:
            bc = block_cache[bid]
            summs = build_cluster_summaries(bc["ds"], bc["emb"], bc["labels_init"])
            active = mem_est.active_events(bid)
            if active:
                protos = np.vstack([e.prototype for e in active])
                for s in summs.values():
                    mx = float(np.max(protos @ s.prototype))
                    if args.tau_lo <= mx < args.tau_hi and s.size >= args.min_cross_size:
                        ambig_cnt += 1
            link_block_to_memory(mem_est, bid, bc["ds"], bc["emb"], bc["labels_init"], mode="proto_only", tau_proto=args.tau_proto)
        print(f"  [DRY-RUN] cross-block ambiguous seams (size>={args.min_cross_size}, [{args.tau_lo},{args.tau_hi})): ~{ambig_cnt} calls (~{ambig_cnt*560:,} est tok, budget={cross_budget:,})")
        return

    # 2. Execute full streaming per-block comparison across strategies
    init_eval_mean = {
        "ami": float(np.mean([block_cache[b]["init_audit"]["ami"] for b in eval_block_ids])),
        "nmi": float(np.mean([block_cache[b]["init_audit"]["nmi"] for b in eval_block_ids])),
        "ari": float(np.mean([block_cache[b]["init_audit"]["ari"] for b in eval_block_ids])),
        "macro_b3": float(np.mean([block_cache[b]["init_audit"]["bcubed_full"]["macro_b3_f1"] for b in eval_block_ids])),
        "micro_b3": float(np.mean([block_cache[b]["init_audit"]["bcubed_full"]["micro_b3_f1"] for b in eval_block_ids])),
        "tail_b3": float(np.mean([block_cache[b]["init_audit"]["stratified"]["tail"]["macro_b3_f1"] for b in eval_block_ids])),
    }

    report: Dict[str, Any] = {
        "dataset": args.dataset,
        "model": model_name,
        "embed_model": embed_model,
        "warmup_blocks": warmup_block_ids,
        "eval_blocks": eval_block_ids,
        "within_token_budget": within_budget,
        "cross_token_budget": cross_budget,
        "cross_thresholds": {
            "tau_proto": args.tau_proto,
            "tau_lo": args.tau_lo,
            "tau_hi": args.tau_hi,
            "min_cross_size": args.min_cross_size,
            "memory_window": args.memory_window,
        },
        "init_eval_mean": init_eval_mean,
        "strategies": {},
        "mean_comparison": {},
        "longitudinal_tracking": {},
    }

    saved_block_labels: Dict[str, Dict[int, np.ndarray]] = {
        "init_2dse": {b: block_cache[b]["labels_init"] for b in eval_block_ids}
    }
    per_block_csv_rows: List[Dict[str, Any]] = []

    for strat in strategies:
        t_s = time.time()
        print(f"\n================ Streaming Strategy: {strat} | Within Budget={within_budget:,} real tokens ================")
        led_dir = out_dir / f"ledger_{strat}"
        if (led_dir / "journal.jsonl").exists():
            (led_dir / "journal.jsonl").unlink()
        ledger = TokenLedger(led_dir, shared_cache_file=shared_cache_file)
        bucket = CumulativeTokenBucket(total_token_budget=within_budget, num_blocks=len(eval_block_ids))
        client = BudgetedLLMClient(model=model_name, ledger=ledger, base_url=base_url, api_key=api_key)

        strat_blocks: Dict[str, Any] = {}
        agg_branch: Dict[str, List[Dict[str, float]]] = {"direct": [], "repair": []}
        saved_block_labels[f"{strat}_direct"] = {}
        saved_block_labels[f"{strat}_repair"] = {}

        for bid in eval_block_ids:
            bc = block_cache[bid]
            sub_ds = bc["ds"]
            graph = bc["graph"]
            labels_init = bc["labels_init"]
            summaries = bc["summaries"]

            pool = _select_pool(strat, sub_ds, graph, labels_init, args.max_queries_per_block, seed=42 + bid)
            mean_u = float(np.mean([q.score for q in pool])) if pool else 1.0
            cap_tok = bucket.start_block(len(sub_ds) / eval_total_posts, mean_u)

            decisions = client.execute_queries(pool, sub_ds, summaries, block_id=bid, bucket=bucket)
            dec_ids = {d.query_id for d in decisions}
            used_queries = [q for q in pool if q.query_id in dec_ids]

            action_counts: Dict[str, int] = {}
            for d in decisions:
                action_counts[d.action] = action_counts.get(d.action, 0) + 1

            lab_dir, qmask_dir = apply_direct_assignment(
                labels_init, used_queries, decisions, conf_threshold=args.conf_threshold
            )
            aud_dir = full_partition_audit(sub_ds.labels, lab_dir, y_before=labels_init, queried_mask=qmask_dir)

            lab_rep, qmask_rep, surg_stats = apply_topology_repair_and_recluster(
                graph, labels_init, summaries, used_queries, decisions, conf_threshold=args.conf_threshold
            )
            aud_rep = full_partition_audit(sub_ds.labels, lab_rep, y_before=labels_init, queried_mask=qmask_rep)

            saved_block_labels[f"{strat}_direct"][bid] = lab_dir
            saved_block_labels[f"{strat}_repair"][bid] = lab_rep

            block_tok = sum(d.prompt_tokens + d.completion_tokens for d in decisions)
            p1, p2 = aud_dir["propagation_audit"], aud_rep["propagation_audit"]
            print(
                f"  M_{bid:02d} Q={len(decisions):2d} Tok={block_tok:5d}/{cap_tok:5d} | "
                f"Dir: dAMI={aud_dir['delta_ami']:+.4f} dMac={aud_dir['delta_macro_b3_f1']:+.4f} Unq={p1['unqueried_net_improved_posts']:+4d} || "
                f"Rep: dAMI={aud_rep['delta_ami']:+.4f} dMac={aud_rep['delta_macro_b3_f1']:+.4f} Unq={p2['unqueried_net_improved_posts']:+4d}"
            )

            for btag, aud in [("direct", aud_dir), ("repair", aud_rep)]:
                prop = aud["propagation_audit"]
                et = aud["error_transition"]
                row = {
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
                    "win_ami": 1.0 if aud["delta_ami"] > 1e-6 else 0.0,
                    "win_macb3": 1.0 if aud["delta_macro_b3_f1"] > 1e-6 else 0.0,
                }
                agg_branch[btag].append(row)
                per_block_csv_rows.append(
                    {
                        "dataset": args.dataset,
                        "block_id": bid,
                        "block_name": bc["name"],
                        "n_posts": len(sub_ds),
                        "strategy": strat,
                        "branch": btag,
                        "init_ami": round(bc["init_audit"]["ami"], 6),
                        "init_macro_b3": round(bc["init_audit"]["bcubed_full"]["macro_b3_f1"], 6),
                        **{k: round(v, 6) for k, v in row.items()},
                    }
                )

            strat_blocks[f"M_{bid:02d}"] = {
                "n_posts": len(sub_ds),
                "n_queries": len(decisions),
                "block_tokens": block_tok,
                "block_cap_tokens": cap_tok,
                "action_counts": action_counts,
                "surgery_stats": surg_stats,
                "branch1_direct": aud_dir,
                "branch2_repair": aud_rep,
            }

        lsum = ledger.summary()
        report["strategies"][strat] = {
            "elapsed_sec": round(time.time() - t_s, 2),
            "ledger_summary": lsum,
            "blocks": strat_blocks,
        }
        for btag in ["direct", "repair"]:
            rows = agg_branch[btag]
            m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()}
            m["total_tokens_all_blocks"] = float(sum(r["tokens"] for r in rows))
            m["blocks_improved_ami"] = int(sum(r["win_ami"] for r in rows))
            m["blocks_improved_macb3"] = int(sum(r["win_macb3"] for r in rows))
            m["n_eval_blocks"] = len(eval_block_ids)
            report["mean_comparison"][f"{strat}_{btag}"] = m

    # 3. Longitudinal Cross-Block Event Tracking Evaluation
    # Compare across Init 2D-SE and our primary mesoscopic partitions (sug_meso_only_direct, sug_router_repair)
    if not args.skip_cross_block:
        print(f"\n================ LONGITUDINAL CROSS-BLOCK TRACKING (M_{eval_block_ids[0]:02d}..M_{eval_block_ids[-1]:02d}) ================")
        track_sources = [("init_2dse", "init_2dse")]
        for base_k in ["random_direct", "rep_div_direct", "cequel_edge_repair"]:
            if base_k in saved_block_labels:
                track_sources.append((base_k, base_k))
        if "sug_meso_only_direct" in saved_block_labels:
            track_sources.append(("sug_meso_only_direct", "sug_meso_only_direct"))
        if "sug_router_repair" in saved_block_labels:
            track_sources.append(("sug_router_repair", "sug_router_repair"))

        for src_name, src_key in track_sources:
            modes_to_run = ["no_link", "proto_only"]
            if src_name in {"sug_meso_only_direct", "sug_router_repair"} and cross_budget > 0:
                modes_to_run.append("sug_cross_llm")

            for mode in modes_to_run:
                mem = GlobalEventMemory(window_blocks=args.memory_window)
                # Warmup memory with M_00..M_05 (0 LLM tokens, proto_only)
                for bid in warmup_block_ids:
                    bc = block_cache[bid]
                    link_block_to_memory(
                        mem, bid, bc["ds"], bc["emb"], bc["labels_init"],
                        mode=("no_link" if mode == "no_link" else "proto_only"),
                        tau_proto=args.tau_proto,
                    )

                cross_client = None
                cross_bucket = None
                cross_ledger = None
                if mode == "sug_cross_llm":
                    cled_dir = out_dir / f"ledger_cross_{src_name}"
                    if (cled_dir / "journal.jsonl").exists():
                        (cled_dir / "journal.jsonl").unlink()
                    cross_ledger = TokenLedger(cled_dir, shared_cache_file=shared_cache_file)
                    cross_bucket = CumulativeTokenBucket(total_token_budget=cross_budget, num_blocks=len(eval_block_ids))
                    cross_client = BudgetedLLMClient(
                        model=model_name, ledger=cross_ledger, base_url=base_url, api_key=api_key
                    )

                eval_true_all, eval_pred_all = [], []
                tot_stats = defaultdict(int)
                for bid in eval_block_ids:
                    bc = block_cache[bid]
                    if cross_bucket is not None:
                        cross_bucket.start_block(len(bc["ds"]) / eval_total_posts, 1.0)
                    glab, st = link_block_to_memory(
                        mem, bid, bc["ds"], bc["emb"], saved_block_labels[src_key][bid],
                        mode=mode,
                        tau_proto=args.tau_proto,
                        tau_lo=args.tau_lo,
                        tau_hi=args.tau_hi,
                        min_cross_size=args.min_cross_size,
                        client=cross_client,
                        bucket=cross_bucket,
                    )
                    for k, v in st.items():
                        tot_stats[k] += int(v)
                    eval_true_all.append(bc["ds"].labels)
                    eval_pred_all.append(glab)

                y_true_cat = np.concatenate(eval_true_all)
                y_pred_cat = np.concatenate(eval_pred_all)
                gaud = full_partition_audit(y_true_cat, y_pred_cat)
                tag = f"{src_name}__{mode}"
                report["longitudinal_tracking"][tag] = {
                    "source_partition": src_name,
                    "link_mode": mode,
                    "n_total_posts": int(len(y_true_cat)),
                    "n_true_events": int(len(set(y_true_cat.tolist()))),
                    "n_pred_events": int(gaud["n_pred_clusters"]),
                    "global_ami": gaud["ami"],
                    "global_nmi": gaud["nmi"],
                    "global_ari": gaud["ari"],
                    "global_macro_b3": gaud["bcubed_full"]["macro_b3_f1"],
                    "global_micro_b3": gaud["bcubed_full"]["micro_b3_f1"],
                    "global_tail_b3": gaud["stratified"]["tail"]["macro_b3_f1"],
                    "link_stats": dict(tot_stats),
                    "cross_ledger": cross_ledger.summary() if cross_ledger is not None else None,
                }
                print(
                    f"  {tag:36s} | K={gaud['n_pred_clusters']:4d} | Global AMI={gaud['ami']:.4f} "
                    f"MacB3={gaud['bcubed_full']['macro_b3_f1']:.4f} MicB3={gaud['bcubed_full']['micro_b3_f1']:.4f} "
                    f"| auto={tot_stats['auto_linked']:3d} llm_q={tot_stats['llm_queried']:3d} "
                    f"(yes={tot_stats['llm_linked']:2d}, no={tot_stats['llm_rejected']:2d}) tok={tot_stats['cross_tokens']:6d}"
                )

    # 4. Print final summary table & save outputs
    print(f"\n================ FINAL STREAMING PER-BLOCK SUMMARY ({args.dataset} · M_{eval_block_ids[0]:02d}..M_{eval_block_ids[-1]:02d}) ================")
    im = init_eval_mean
    print(f"Init 2D-SE Mean ({len(eval_block_ids)} blocks): AMI={im['ami']:.4f} | MacB3={im['macro_b3']:.4f} | MicB3={im['micro_b3']:.4f} | TailB3={im['tail_b3']:.4f}")
    hdr = (
        f"{'Setting':22s} | {'TotTok':>7s} | {'Q/blk':>5s} | {'AMI':>6s} | {'dAMI':>7s} | "
        f"{'MacB3':>6s} | {'dMacB3':>7s} | {'UnqMic':>7s} | {'UnqNet':>7s} | {'WinAMI':>6s} | {'WinMac':>6s} | {'NetFS':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for key, m in report["mean_comparison"].items():
        print(
            f"{key:22s} | {m['total_tokens_all_blocks']:7.0f} | {m['queries']:5.1f} | "
            f"{m['ami']:6.4f} | {m['delta_ami']:+7.4f} | {m['macro_b3']:6.4f} | {m['delta_macro_b3']:+7.4f} | "
            f"{m['unq_delta_micro_b3']:+7.4f} | {m['unq_net_posts']:+7.1f} | "
            f"{m['blocks_improved_ami']:2d}/{m['n_eval_blocks']:<2d}  | {m['blocks_improved_macb3']:2d}/{m['n_eval_blocks']:<2d}  | "
            f"{m['net_fs_red']:+7.0f}"
        )

    json_path = out_dir / "streaming_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out_dir / "per_block_metrics.csv"
    if per_block_csv_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_block_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_block_csv_rows)
    print(f"\nSaved streaming JSON: {json_path}")
    print(f"Saved per-block CSV : {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-End Cross-Block Streaming SED Pipeline")
    parser.add_argument("--dataset", required=True, choices=["Event2012", "Event2012_100", "Event2018"])
    parser.add_argument("--eval-blocks", default="heldout", help="'heldout' (M_06..end), 'dev' (M_02..M_05), or comma list")
    parser.add_argument("--strategies", default="random,rep_div,cequel_edge,sug_router,sug_meso_only")
    parser.add_argument("--tokens-per-block", type=int, default=10000, help="Default within-block real tokens/block")
    parser.add_argument("--within-token-budget", type=int, default=None, help="Override total within-block real token budget")
    parser.add_argument("--cross-tokens-per-block", type=int, default=3000, help="Default cross-block real tokens/block")
    parser.add_argument("--cross-token-budget", type=int, default=None, help="Override total cross-block real token budget")
    parser.add_argument("--max-queries-per-block", type=int, default=120)
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--knn-sim-threshold", type=float, default=0.35)
    parser.add_argument("--attr-alpha", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.6)
    parser.add_argument("--tau-proto", type=float, default=0.68)
    parser.add_argument("--tau-lo", type=float, default=0.50)
    parser.add_argument("--tau-hi", type=float, default=0.78)
    parser.add_argument("--min-cross-size", type=int, default=3)
    parser.add_argument("--memory-window", type=int, default=14)
    parser.add_argument("--skip-cross-block", action="store_true")
    parser.add_argument("--calibrate-dev-only", action="store_true")
    parser.add_argument("--run-name", default="heldout_qwen3_8b")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_streaming_evaluation(args)


if __name__ == "__main__":
    main()
