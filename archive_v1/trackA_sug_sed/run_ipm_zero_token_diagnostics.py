"""IP&M Stage 1 Zero-Token & Cached Decisive Diagnostic Suite (D1-D6).

Executes ZERO live LLM API calls. Uses cached dense embeddings, disk-cached Stage-1
2D-SE initial partitions, and frozen held-out / dev Qwen3-8B ledgers read-only to
produce all missing audits, causal/uncontaminated recomputations, zero-LLM baselines,
statistical tests, failure root-cause audits, and anonymized case studies in:
  - results/ipm_audit/stage1_diagnostics.json
  - results/ipm_audit/stage1_summary.md
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import sparse, stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset, load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import MessageGraph, build_message_graph
from core_shared.metrics_audit import full_partition_audit, pairwise_error_counts
from core_shared.partition_cache import load_or_solve_2d_se
from core_shared.token_ledger import CumulativeTokenBucket, canonical_request_hash
from trackA_sug_sed.high_leverage_router import (
    ClusterSummary,
    QueryItem,
    build_cluster_summaries,
    detect_merge_pair_queries,
    detect_seam_queries,
)
from trackA_sug_sed.llm_client import parse_llm_response_to_decision
from trackA_sug_sed.run_gateA_oracle_diagnostic import load_or_extract_block_embeddings
from trackA_sug_sed.run_gateA_real_llm import _select_pool
from trackA_sug_sed.streaming_sed import CROSS_BLOCK_SYS_MSG, GlobalEventMemory, link_block_to_memory
from trackA_sug_sed.topology_surgery import (
    LLMDecision,
    apply_cequel_transitive_assignment,
    apply_direct_assignment,
    apply_topology_repair_and_recluster,
    collect_prompt_shown_nodes,
)


def _anonymize_text(text: str, max_len: int = 160) -> str:
    s = text.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"https?://\S+", "[URL]", s)
    s = re.sub(r"@[A-Za-z0-9_]+", "@USER", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _load_all_cached_responses(dataset: str) -> Dict[str, Dict[str, Any]]:
    """Load all non-error cached responses across shared cache & historical ledgers."""
    cache: Dict[str, Dict[str, Any]] = {}
    cand_files: List[Path] = []
    shared = ROOT / "results" / "llm_cache" / f"{dataset}_Qwen__Qwen3-8B" / "responses.jsonl"
    if shared.exists():
        cand_files.append(shared)
    for base in [
        ROOT / "results" / "streaming" / f"{dataset}_heldout_qwen3_8b",
        ROOT / "results" / "ablations" / f"{dataset}_heldout_qwen3_8b",
        ROOT / "results" / "budget_curves" / f"{dataset}_dev_qwen3_8b",
        ROOT / "results" / "gateA_llm",
    ]:
        if base.exists():
            cand_files.extend(sorted(base.rglob("responses.jsonl")))

    for path in cand_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                resp = obj.get("response", {})
                pt = int(obj.get("prompt_tokens", 0))
                ct = int(obj.get("completion_tokens", 0))
                if isinstance(resp, dict) and "error" in resp and pt == 0 and ct == 0:
                    continue
                rh = obj.get("request_hash")
                if rh and rh not in cache:
                    cache[rh] = obj
    return cache


def _read_journal_by_block(journal_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    by_blk: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with journal_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_blk[int(rec["block_id"])].append(rec)
    return by_blk


def _majority_label(arr: np.ndarray) -> int:
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _wilcoxon_and_effects(x_ours: Sequence[float], x_base: Sequence[float]) -> Dict[str, float]:
    """Compute paired two-sided Wilcoxon signed-rank test, Cohen's dz, and rank-biserial r."""
    a = np.asarray(x_ours, dtype=np.float64)
    b = np.asarray(x_base, dtype=np.float64)
    diff = a - b
    n = len(diff)
    mean_d = float(np.mean(diff))
    sd_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    dz = (mean_d / sd_d) if sd_d > 1e-12 else 0.0
    med_d = float(np.median(diff))
    q25, q75 = (float(np.percentile(diff, 25)), float(np.percentile(diff, 75))) if n > 0 else (0.0, 0.0)

    nz = diff[np.abs(diff) > 1e-12]
    if len(nz) == 0:
        return {
            "n": n,
            "mean_diff": mean_d,
            "sd_diff": sd_d,
            "median_diff": med_d,
            "iqr_low": q25,
            "iqr_high": q75,
            "cohens_dz": 0.0,
            "rank_biserial_r": 0.0,
            "p_value": 1.0,
            "wins": 0,
            "ties": n,
            "losses": 0,
        }

    # Manual signed ranks for exact rank-biserial correlation
    abs_nz = np.abs(nz)
    ranks = stats.rankdata(abs_nz, method="average")
    w_pos = float(np.sum(ranks[nz > 0]))
    w_neg = float(np.sum(ranks[nz < 0]))
    denom = float(len(nz) * (len(nz) + 1) / 2.0)
    r_rb = (w_pos - w_neg) / denom if denom > 0 else 0.0

    # Zero-diff guard before calling wilcoxon
    res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", correction=False)
    p_val = float(res.pvalue)
    return {
        "n": n,
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "median_diff": med_d,
        "iqr_low": q25,
        "iqr_high": q75,
        "cohens_dz": float(dz),
        "rank_biserial_r": float(r_rb),
        "p_value": p_val,
        "wins": int(np.sum(diff > 1e-12)),
        "ties": int(np.sum(np.abs(diff) <= 1e-12)),
        "losses": int(np.sum(diff < -1e-12)),
    }


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * float(p_values[idx])
        running_max = max(running_max, min(1.0, val))
        adjusted[int(idx)] = running_max
    return adjusted.tolist()


def _summarize_metric_arrays(per_block: Dict[str, List[float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, vals in per_block.items():
        arr = np.asarray(vals, dtype=np.float64)
        if len(arr) == 0:
            continue
        out[k] = {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "per_block": [float(v) for v in arr.tolist()],
        }
    return out


def preload_dataset_blocks(dataset: str) -> Dict[str, Any]:
    embed_model = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if dataset == "Event2018"
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    ds_full = load_socialed_npy(dataset, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
    block_data: Dict[int, Dict[str, Any]] = {}
    t0 = time.time()
    for bid in range(len(blocks)):
        bname, bidx = blocks[bid]
        sub_ds = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        emb = load_or_extract_block_embeddings(dataset, bid, sub_ds.texts, model_name=embed_model)
        graph = build_message_graph(sub_ds, emb, knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.5)
        labels_init, solve_sec = load_or_solve_2d_se(
            dataset, bid, graph.adj, knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.5, refine_moves=True
        )
        summaries = build_cluster_summaries(sub_ds, emb, labels_init)
        init_audit = full_partition_audit(sub_ds.labels, labels_init)
        block_data[bid] = {
            "name": bname,
            "ds": sub_ds,
            "emb": emb,
            "graph": graph,
            "labels_init": labels_init,
            "summaries": summaries,
            "init_audit": init_audit,
            "solve_sec": solve_sec,
        }
    print(f"[Preload] {dataset}: {len(blocks)} blocks ready in {time.time() - t0:.1f}s")
    return {
        "dataset": dataset,
        "embed_model": embed_model,
        "n_blocks": len(blocks),
        "warmup_ids": list(range(0, 6)),
        "dev_ids": [2, 3, 4, 5],
        "heldout_ids": list(range(6, len(blocks))),
        "blocks": block_data,
    }


def run_d1_replay_and_queried_audit(
    dataset: str,
    pdata: Dict[str, Any],
    resp_cache: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """D1: Exact AUD-01 queried-post accounting, uncontaminated U_t, and Cequel-Transitive."""
    report_path = ROOT / "results" / "streaming" / f"{dataset}_heldout_qwen3_8b" / "streaming_report.json"
    ref_report = json.loads(report_path.read_text(encoding="utf-8"))
    eval_ids: List[int] = pdata["heldout_ids"]
    blocks = pdata["blocks"]
    eval_total_posts = sum(len(blocks[b]["ds"]) for b in eval_ids)

    strategies = ["random", "rep_div", "cequel_edge", "sug_router", "sug_meso_only"]
    d1_out: Dict[str, Any] = {"eval_total_posts": eval_total_posts, "strategies": {}, "sanity_checks": {}}
    saved_partitions: Dict[str, Dict[int, np.ndarray]] = {
        "init_2dse": {b: blocks[b]["labels_init"] for b in eval_ids}
    }
    decisions_store: Dict[str, Dict[int, Any]] = defaultdict(dict)
    per_block_metrics_store: Dict[str, Dict[str, List[float]]] = {}

    for strat in strategies:
        journal_path = (
            ROOT / "results" / "streaming" / f"{dataset}_heldout_qwen3_8b" / f"ledger_{strat}" / "journal.jsonl"
        )
        by_blk = _read_journal_by_block(journal_path)
        branch_records: Dict[str, Dict[str, List[float]]] = {
            "direct": defaultdict(list),
            "repair": defaultdict(list),
        }
        cequel_trans_metrics: Dict[str, List[float]] = defaultdict(list)

        mask_totals = {
            "direct": {"calls": 0, "prim": 0, "targ": 0, "all": 0, "unq_net_prim": 0, "unq_net_all": 0},
            "repair": {"calls": 0, "prim": 0, "targ": 0, "all": 0, "unq_net_prim": 0, "unq_net_all": 0},
        }
        saved_partitions[f"{strat}_direct"] = {}
        saved_partitions[f"{strat}_repair"] = {}
        if strat == "cequel_edge":
            saved_partitions["cequel_transitive"] = {}

        for bid in eval_ids:
            bc = blocks[bid]
            sub_ds = bc["ds"]
            graph = bc["graph"]
            labels_init = bc["labels_init"]
            summaries = bc["summaries"]

            pool = _select_pool(strat, sub_ds, graph, labels_init, 120, seed=42 + bid)
            qmap_pool = {q.query_id: q for q in pool}
            recs = by_blk.get(bid, [])

            used_queries: List[QueryItem] = []
            decisions: List[LLMDecision] = []
            parsed_raw: Dict[str, Dict[str, Any]] = {}
            for rec in recs:
                qid = rec["request_id"]
                q = qmap_pool[qid]
                rh = rec["request_hash"]
                cached_obj = resp_cache.get(rh)
                if cached_obj is None:
                    raise KeyError(f"Missing cached response for {dataset} {strat} M_{bid:02d} {qid} hash={rh}")
                parsed = cached_obj["response"]
                dec = parse_llm_response_to_decision(
                    q, parsed, prompt_tokens=int(rec["prompt_tokens"]), completion_tokens=int(rec["completion_tokens"])
                )
                used_queries.append(q)
                decisions.append(dec)
                parsed_raw[qid] = parsed

            decisions_store[strat][bid] = {
                "used_queries": used_queries,
                "decisions": decisions,
                "parsed_raw": parsed_raw,
            }

            n_posts = len(sub_ds)
            qmask_prim = np.zeros(n_posts, dtype=bool)
            for q in used_queries:
                for node in q.primary_nodes:
                    qmask_prim[int(node)] = True
            qmask_targ, qmask_all = collect_prompt_shown_nodes(
                used_queries, decisions, n_posts, summaries=summaries, include_candidate_medoids=True
            )

            lab_dir, _ = apply_direct_assignment(
                labels_init, used_queries, decisions, conf_threshold=0.6, summaries=summaries, mask_mode="all_shown"
            )
            lab_rep, _, _ = apply_topology_repair_and_recluster(
                graph, labels_init, summaries, used_queries, decisions, conf_threshold=0.6, mask_mode="all_shown"
            )
            saved_partitions[f"{strat}_direct"][bid] = lab_dir
            saved_partitions[f"{strat}_repair"][bid] = lab_rep

            aud_dir_prim = full_partition_audit(sub_ds.labels, lab_dir, y_before=labels_init, queried_mask=qmask_prim)
            aud_dir_all = full_partition_audit(sub_ds.labels, lab_dir, y_before=labels_init, queried_mask=qmask_all)
            aud_rep_prim = full_partition_audit(sub_ds.labels, lab_rep, y_before=labels_init, queried_mask=qmask_prim)
            aud_rep_all = full_partition_audit(sub_ds.labels, lab_rep, y_before=labels_init, queried_mask=qmask_all)

            for btag, ap, aa in [
                ("direct", aud_dir_prim, aud_dir_all),
                ("repair", aud_rep_prim, aud_rep_all),
            ]:
                mask_totals[btag]["calls"] += len(decisions)
                mask_totals[btag]["prim"] += int(qmask_prim.sum())
                mask_totals[btag]["targ"] += int(qmask_targ.sum())
                mask_totals[btag]["all"] += int(qmask_all.sum())
                mask_totals[btag]["unq_net_prim"] += int(ap["propagation_audit"]["unqueried_net_improved_posts"])
                mask_totals[btag]["unq_net_all"] += int(aa["propagation_audit"]["unqueried_net_improved_posts"])

                branch_records[btag]["ami"].append(aa["ami"])
                branch_records[btag]["delta_ami"].append(aa["delta_ami"])
                branch_records[btag]["nmi"].append(aa["nmi"])
                branch_records[btag]["ari"].append(aa["ari"])
                branch_records[btag]["macro_b3"].append(aa["bcubed_full"]["macro_b3_f1"])
                branch_records[btag]["delta_macro_b3"].append(aa["delta_macro_b3_f1"])
                branch_records[btag]["micro_b3"].append(aa["bcubed_full"]["micro_b3_f1"])
                branch_records[btag]["delta_micro_b3"].append(aa["delta_micro_b3_f1"])
                branch_records[btag]["tail_b3"].append(aa["stratified"]["tail"]["macro_b3_f1"])
                branch_records[btag]["legacy_unq_delta_micro_b3"].append(ap["propagation_audit"]["unqueried_delta_micro_b3_f1"])
                branch_records[btag]["legacy_unq_delta_macro_b3"].append(ap["propagation_audit"]["unqueried_delta_macro_b3_f1"])
                branch_records[btag]["legacy_unq_net_posts"].append(float(ap["propagation_audit"]["unqueried_net_improved_posts"]))
                branch_records[btag]["allshown_unq_delta_micro_b3"].append(aa["propagation_audit"]["unqueried_delta_micro_b3_f1"])
                branch_records[btag]["allshown_unq_delta_macro_b3"].append(aa["propagation_audit"]["unqueried_delta_macro_b3_f1"])
                branch_records[btag]["allshown_unq_net_posts"].append(float(aa["propagation_audit"]["unqueried_net_improved_posts"]))

            if strat == "cequel_edge":
                lab_ct, _ = apply_cequel_transitive_assignment(
                    labels_init, used_queries, decisions, conf_threshold=0.6, summaries=summaries, mask_mode="all_shown"
                )
                saved_partitions["cequel_transitive"][bid] = lab_ct
                aud_ct = full_partition_audit(sub_ds.labels, lab_ct, y_before=labels_init, queried_mask=qmask_all)
                cequel_trans_metrics["ami"].append(aud_ct["ami"])
                cequel_trans_metrics["delta_ami"].append(aud_ct["delta_ami"])
                cequel_trans_metrics["nmi"].append(aud_ct["nmi"])
                cequel_trans_metrics["ari"].append(aud_ct["ari"])
                cequel_trans_metrics["macro_b3"].append(aud_ct["bcubed_full"]["macro_b3_f1"])
                cequel_trans_metrics["delta_macro_b3"].append(aud_ct["delta_macro_b3_f1"])
                cequel_trans_metrics["micro_b3"].append(aud_ct["bcubed_full"]["micro_b3_f1"])
                cequel_trans_metrics["delta_micro_b3"].append(aud_ct["delta_micro_b3_f1"])
                cequel_trans_metrics["tail_b3"].append(aud_ct["stratified"]["tail"]["macro_b3_f1"])
                cequel_trans_metrics["allshown_unq_delta_micro_b3"].append(
                    aud_ct["propagation_audit"]["unqueried_delta_micro_b3_f1"]
                )
                cequel_trans_metrics["allshown_unq_delta_macro_b3"].append(
                    aud_ct["propagation_audit"]["unqueried_delta_macro_b3_f1"]
                )
                cequel_trans_metrics["allshown_unq_net_posts"].append(
                    float(aud_ct["propagation_audit"]["unqueried_net_improved_posts"])
                )

        # Sanity check against frozen streaming_report.json
        for btag in ["direct", "repair"]:
            ref_m = ref_report["mean_comparison"][f"{strat}_{btag}"]
            rec_ami = float(np.mean(branch_records[btag]["ami"]))
            rec_mac = float(np.mean(branch_records[btag]["macro_b3"]))
            rec_unq = float(np.mean(branch_records[btag]["legacy_unq_delta_micro_b3"]))
            diff_ami = abs(rec_ami - ref_m["ami"])
            diff_mac = abs(rec_mac - ref_m["macro_b3"])
            diff_unq = abs(rec_unq - ref_m["unq_delta_micro_b3"])
            max_diff = max(diff_ami, diff_mac, diff_unq)
            d1_out["sanity_checks"][f"{strat}_{btag}"] = {
                "diff_ami": diff_ami,
                "diff_macro_b3": diff_mac,
                "diff_legacy_unq_micro": diff_unq,
                "max_diff": max_diff,
            }
            if max_diff > 5e-4:
                raise RuntimeError(
                    f"D1 sanity check failed on {dataset} {strat}_{btag}: max_diff={max_diff:.6f}"
                )

            mt = mask_totals[btag]
            calls = max(1, mt["calls"])
            all_shown = max(1, mt["all"])
            d1_out["strategies"][f"{strat}_{btag}"] = {
                "counts": {
                    "total_calls": mt["calls"],
                    "calls_pct": 100.0 * mt["calls"] / eval_total_posts,
                    "primary_posts": mt["prim"],
                    "primary_pct": 100.0 * mt["prim"] / eval_total_posts,
                    "target_shown_posts": mt["targ"],
                    "target_shown_pct": 100.0 * mt["targ"] / eval_total_posts,
                    "all_shown_posts": mt["all"],
                    "all_shown_pct": 100.0 * mt["all"] / eval_total_posts,
                    "unq_net_total_legacy_prim": mt["unq_net_prim"],
                    "unq_net_total_all_shown": mt["unq_net_all"],
                    "unq_net_per_call_all_shown": mt["unq_net_all"] / calls,
                    "unq_net_per_shown_post_all_shown": mt["unq_net_all"] / all_shown,
                },
                "metrics": _summarize_metric_arrays(branch_records[btag]),
            }
            per_block_metrics_store[f"{strat}_{btag}"] = {k: list(v) for k, v in branch_records[btag].items()}

        if strat == "cequel_edge":
            d1_out["strategies"]["cequel_transitive"] = {
                "counts": dict(d1_out["strategies"]["cequel_edge_direct"]["counts"]),
                "metrics": _summarize_metric_arrays(cequel_trans_metrics),
            }
            per_block_metrics_store["cequel_transitive"] = {k: list(v) for k, v in cequel_trans_metrics.items()}

    # Add Init 2D-SE per-block arrays
    init_arrays: Dict[str, List[float]] = defaultdict(list)
    for bid in eval_ids:
        ia = blocks[bid]["init_audit"]
        init_arrays["ami"].append(ia["ami"])
        init_arrays["nmi"].append(ia["nmi"])
        init_arrays["ari"].append(ia["ari"])
        init_arrays["macro_b3"].append(ia["bcubed_full"]["macro_b3_f1"])
        init_arrays["micro_b3"].append(ia["bcubed_full"]["micro_b3_f1"])
        init_arrays["tail_b3"].append(ia["stratified"]["tail"]["macro_b3_f1"])
        init_arrays["delta_ami"].append(0.0)
        init_arrays["delta_macro_b3"].append(0.0)
    per_block_metrics_store["init_2dse"] = dict(init_arrays)
    d1_out["init_2dse_metrics"] = _summarize_metric_arrays(init_arrays)

    print(
        f"[D1 {dataset}] Sanity max_diff={max(v['max_diff'] for v in d1_out['sanity_checks'].values()):.2e} | "
        f"SUG-Meso shown={d1_out['strategies']['sug_meso_only_direct']['counts']['all_shown_posts']} "
        f"({d1_out['strategies']['sug_meso_only_direct']['counts']['all_shown_pct']:.2f}%), "
        f"UnqNet(all_shown)={d1_out['strategies']['sug_meso_only_direct']['counts']['unq_net_total_all_shown']}/blk="
        f"{d1_out['strategies']['sug_meso_only_direct']['metrics']['allshown_unq_net_posts']['mean']:.1f}"
    )
    return d1_out, saved_partitions, {"decisions": decisions_store, "per_block": per_block_metrics_store}


def _instrumented_cross_eval(
    pdata: Dict[str, Any],
    part_dict: Dict[int, np.ndarray],
    mode: str,
    tau_proto: float = 0.72,
    tau_lo: float = 0.55,
    tau_hi: float = 0.80,
    min_cross_size: int = 3,
    cross_token_budget: Optional[int] = None,
    cross_journal_by_blk: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    resp_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    oracle_lo: float = 0.55,
    oracle_hi: float = 0.80,
    oracle_min_size: int = 3,
) -> Dict[str, Any]:
    """Run global cross-block memory with optional LLM replay / Oracle and full confusion audit."""
    blocks = pdata["blocks"]
    warmup_ids: List[int] = pdata["warmup_ids"]
    eval_ids: List[int] = pdata["heldout_ids"]
    eval_total_posts = sum(len(blocks[b]["ds"]) for b in eval_ids)

    cross_bucket: Optional[CumulativeTokenBucket] = None
    if cross_token_budget is not None and cross_token_budget > 0:
        cross_bucket = CumulativeTokenBucket(total_token_budget=cross_token_budget, num_blocks=len(eval_ids))

    mem = GlobalEventMemory(window_blocks=14)
    gid_true_counts: Dict[int, Counter] = defaultdict(Counter)

    def _seed_warmup(bid_list: List[int]) -> None:
        for bid in bid_list:
            bc = blocks[bid]
            sub_ds, emb, lab = bc["ds"], bc["emb"], bc["labels_init"]
            summaries = build_cluster_summaries(sub_ds, emb, lab)
            ordered = sorted(summaries.keys(), key=lambda c: -summaries[c].size)
            active = mem.active_events(bid)
            act_gids = [ev.global_id for ev in active]
            act_protos = np.vstack([ev.prototype for ev in active]) if active else np.zeros((0, emb.shape[1]), dtype=np.float32)
            for cid in ordered:
                summ = summaries[cid]
                if mode == "no_link" or not act_gids:
                    gid = mem.spawn_event(bid, summ, sub_ds)
                else:
                    best_pos = int(np.argmax(act_protos @ summ.prototype))
                    best_sim = float((act_protos @ summ.prototype)[best_pos])
                    if best_sim >= tau_proto:
                        gid = act_gids[best_pos]
                        mem.update_event(gid, bid, summ, sub_ds)
                    else:
                        gid = mem.spawn_event(bid, summ, sub_ds)
                gid_true_counts[gid].update(sub_ds.labels[summ.members].tolist())

    _seed_warmup(warmup_ids)

    eval_true_all, eval_pred_all = [], []
    cm = Counter()
    sim_bins = defaultdict(lambda: {"total": 0, "same_true": 0})
    auto_linked = spawned_new = llm_queried = llm_linked = llm_rejected = 0

    def _bin_name(s: float) -> str:
        if s < 0.55:
            return "<0.55"
        if s < 0.60:
            return "[0.55,0.60)"
        if s < 0.65:
            return "[0.60,0.65)"
        if s < 0.70:
            return "[0.65,0.70)"
        if s < 0.75:
            return "[0.70,0.75)"
        if s < 0.80:
            return "[0.75,0.80)"
        if s < 0.85:
            return "[0.80,0.85)"
        return "[0.85,1.00]"

    for bid in eval_ids:
        bc = blocks[bid]
        sub_ds, emb = bc["ds"], bc["emb"]
        block_labels = part_dict[bid]
        summaries = build_cluster_summaries(sub_ds, emb, block_labels)
        ordered = sorted(summaries.keys(), key=lambda c: -summaries[c].size)
        active = mem.active_events(bid)
        act_gids = [ev.global_id for ev in active]
        act_protos = np.vstack([ev.prototype for ev in active]) if active else np.zeros((0, emb.shape[1]), dtype=np.float32)

        if cross_bucket is not None:
            cross_bucket.start_block(len(sub_ds) / eval_total_posts, 1.0)

        jid_map = {}
        if cross_journal_by_blk is not None:
            jid_map = {rec["request_id"]: rec for rec in cross_journal_by_blk.get(bid, [])}

        local_to_global: Dict[int, int] = {}
        for cid in ordered:
            summ = summaries[cid]
            maj_cur = _majority_label(sub_ds.labels[summ.members])
            if mode == "no_link" or not act_gids:
                gid = mem.spawn_event(bid, summ, sub_ds)
                local_to_global[cid] = gid
                gid_true_counts[gid].update(sub_ds.labels[summ.members].tolist())
                spawned_new += 1
                continue

            sims = act_protos @ summ.prototype
            best_pos = int(np.argmax(sims))
            best_sim = float(sims[best_pos])
            best_gid = act_gids[best_pos]
            maj_hist = gid_true_counts[best_gid].most_common(1)[0][0]
            same_true = (maj_hist == maj_cur)

            if summ.size >= min_cross_size:
                bname = _bin_name(best_sim)
                sim_bins[bname]["total"] += 1
                if same_true:
                    sim_bins[bname]["same_true"] += 1

            if mode == "proto_only":
                if best_sim >= tau_proto:
                    mem.update_event(best_gid, bid, summ, sub_ds)
                    local_to_global[cid] = best_gid
                    gid_true_counts[best_gid].update(sub_ds.labels[summ.members].tolist())
                    auto_linked += 1
                else:
                    gid = mem.spawn_event(bid, summ, sub_ds)
                    local_to_global[cid] = gid
                    gid_true_counts[gid].update(sub_ds.labels[summ.members].tolist())
                    spawned_new += 1
            elif mode == "sug_cross_llm_replay":
                if best_sim >= tau_hi:
                    mem.update_event(best_gid, bid, summ, sub_ds)
                    local_to_global[cid] = best_gid
                    gid_true_counts[best_gid].update(sub_ds.labels[summ.members].tolist())
                    auto_linked += 1
                elif best_sim >= tau_lo and summ.size >= min_cross_size:
                    qid = f"cross_M{bid:02d}_g{best_gid}_c{cid}"
                    hist_ev = mem.events[best_gid]
                    h_lines = "\n".join(f"  - {t}" for t in hist_ev.rep_texts[:3])
                    c_lines = "\n".join(
                        f"  - {sub_ds.texts[m].replace(chr(10), ' ').strip()[:140]}"
                        for m in summ.medoids[:3]
                    )
                    h_kw = ", ".join(hist_ev.top_keywords[:5])
                    c_kw = ", ".join(summ.top_keywords[:5])
                    user_msg = (
                        f"Group A (ongoing historical event, size={hist_ev.total_count}, keywords: {h_kw}):\n{h_lines}\n\n"
                        f"Group B (new daily cluster, size={summ.size}, keywords: {c_kw}):\n{c_lines}\n\n"
                        "First summarize Group A's specific event (event_a) and Group B's specific event (event_b) in 5-10 words each, "
                        "then set same_event to true ONLY if both describe the exact same specific incident.\n"
                        'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", "same_event": true or false}'
                    )
                    est_tokens = int((len(CROSS_BLOCK_SYS_MSG) + len(user_msg)) / 3.5) + 120
                    if cross_bucket is not None and not cross_bucket.can_spend(est_tokens):
                        if best_sim >= tau_proto:
                            mem.update_event(best_gid, bid, summ, sub_ds)
                            local_to_global[cid] = best_gid
                            gid_true_counts[best_gid].update(sub_ds.labels[summ.members].tolist())
                            auto_linked += 1
                        else:
                            gid = mem.spawn_event(bid, summ, sub_ds)
                            local_to_global[cid] = gid
                            gid_true_counts[gid].update(sub_ds.labels[summ.members].tolist())
                            spawned_new += 1
                        continue
                    rec = jid_map.get(qid)
                    if rec is None:
                        raise KeyError(f"Missing cross record {qid} in replay journal")
                    if cross_bucket is not None:
                        cross_bucket.record_spend(int(rec["prompt_tokens"]), int(rec["completion_tokens"]))
                    parsed = resp_cache[rec["request_hash"]]["response"]
                    verdict = bool(parsed["same_event"])
                    llm_queried += 1
                    if verdict and same_true:
                        cm["TP"] += 1
                    elif verdict and not same_true:
                        cm["FP"] += 1
                    elif (not verdict) and same_true:
                        cm["FN"] += 1
                    else:
                        cm["TN"] += 1
                    if verdict:
                        mem.update_event(best_gid, bid, summ, sub_ds)
                        local_to_global[cid] = best_gid
                        gid_true_counts[best_gid].update(sub_ds.labels[summ.members].tolist())
                        llm_linked += 1
                    else:
                        gid = mem.spawn_event(bid, summ, sub_ds)
                        local_to_global[cid] = gid
                        gid_true_counts[gid].update(sub_ds.labels[summ.members].tolist())
                        llm_rejected += 1
                        spawned_new += 1
                else:
                    if best_sim >= tau_proto:
                        mem.update_event(best_gid, bid, summ, sub_ds)
                        local_to_global[cid] = best_gid
                        gid_true_counts[best_gid].update(sub_ds.labels[summ.members].tolist())
                        auto_linked += 1
                    else:
                        gid = mem.spawn_event(bid, summ, sub_ds)
                        local_to_global[cid] = gid
                        gid_true_counts[gid].update(sub_ds.labels[summ.members].tolist())
                        spawned_new += 1
            elif mode == "oracle_band":
                if best_sim >= oracle_hi:
                    mem.update_event(best_gid, bid, summ, sub_ds)
                    local_to_global[cid] = best_gid
                    gid_true_counts[best_gid].update(sub_ds.labels[summ.members].tolist())
                    auto_linked += 1
                elif best_sim >= oracle_lo and summ.size >= oracle_min_size:
                    if cross_bucket is not None:
                        if not cross_bucket.can_spend(560):
                            if best_sim >= tau_proto:
                                mem.update_event(best_gid, bid, summ, sub_ds)
                                local_to_global[cid] = best_gid
                                gid_true_counts[best_gid].update(sub_ds.labels[summ.members].tolist())
                                auto_linked += 1
                            else:
                                gid = mem.spawn_event(bid, summ, sub_ds)
                                local_to_global[cid] = gid
                                gid_true_counts[gid].update(sub_ds.labels[summ.members].tolist())
                                spawned_new += 1
                            continue
                        cross_bucket.record_spend(520, 40)
                    llm_queried += 1
                    if same_true:
                        mem.update_event(best_gid, bid, summ, sub_ds)
                        local_to_global[cid] = best_gid
                        gid_true_counts[best_gid].update(sub_ds.labels[summ.members].tolist())
                        llm_linked += 1
                    else:
                        gid = mem.spawn_event(bid, summ, sub_ds)
                        local_to_global[cid] = gid
                        gid_true_counts[gid].update(sub_ds.labels[summ.members].tolist())
                        llm_rejected += 1
                        spawned_new += 1
                else:
                    if best_sim >= tau_proto:
                        mem.update_event(best_gid, bid, summ, sub_ds)
                        local_to_global[cid] = best_gid
                        gid_true_counts[best_gid].update(sub_ds.labels[summ.members].tolist())
                        auto_linked += 1
                    else:
                        gid = mem.spawn_event(bid, summ, sub_ds)
                        local_to_global[cid] = gid
                        gid_true_counts[gid].update(sub_ds.labels[summ.members].tolist())
                        spawned_new += 1
            else:
                raise ValueError(mode)

        glab = np.array([local_to_global[int(c)] for c in block_labels.tolist()], dtype=np.int64)
        eval_true_all.append(sub_ds.labels)
        eval_pred_all.append(glab)

    aud = full_partition_audit(np.concatenate(eval_true_all), np.concatenate(eval_pred_all))
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    prec = tp / max(1, tp + fp)
    rec_v = tp / max(1, tp + fn)
    f1 = (2 * prec * rec_v / (prec + rec_v)) if (prec + rec_v) > 0 else 0.0
    return {
        "ami": aud["ami"],
        "nmi": aud["nmi"],
        "ari": aud["ari"],
        "macro_b3": aud["bcubed_full"]["macro_b3_f1"],
        "micro_b3": aud["bcubed_full"]["micro_b3_f1"],
        "tail_b3": aud["stratified"]["tail"]["macro_b3_f1"],
        "n_pred_events": aud["n_pred_clusters"],
        "n_true_events": aud["n_true_events"],
        "stats": {
            "auto_linked": auto_linked,
            "llm_queried": llm_queried,
            "llm_linked": llm_linked,
            "llm_rejected": llm_rejected,
            "spawned_new": spawned_new,
        },
        "confusion": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": prec,
            "recall": rec_v,
            "f1": f1,
        },
        "sim_bins": dict(sim_bins),
    }


def _cross_sim_bins_on_dev(pdata: Dict[str, Any], min_cross_size: int = 3) -> Dict[str, Any]:
    """Check empirical similarity-bin ground-truth match rate on Dev blocks M_02..M_05."""
    blocks = pdata["blocks"]
    mem = GlobalEventMemory(window_blocks=14)
    gid_true_counts: Dict[int, Counter] = defaultdict(Counter)
    bins = defaultdict(lambda: {"total": 0, "same_true": 0})

    def _bname(s: float) -> str:
        if s < 0.55:
            return "<0.55"
        if s < 0.60:
            return "[0.55,0.60)"
        if s < 0.65:
            return "[0.60,0.65)"
        if s < 0.70:
            return "[0.65,0.70)"
        if s < 0.75:
            return "[0.70,0.75)"
        if s < 0.80:
            return "[0.75,0.80)"
        if s < 0.85:
            return "[0.80,0.85)"
        return "[0.85,1.00]"

    for bid in range(0, 6):
        bc = blocks[bid]
        ds, emb, lab = bc["ds"], bc["emb"], bc["labels_init"]
        summaries = build_cluster_summaries(ds, emb, lab)
        ordered = sorted(summaries.keys(), key=lambda c: -summaries[c].size)
        active = mem.active_events(bid)
        act_gids = [ev.global_id for ev in active]
        act_protos = np.vstack([ev.prototype for ev in active]) if active else np.zeros((0, emb.shape[1]), dtype=np.float32)
        for cid in ordered:
            summ = summaries[cid]
            maj_cur = _majority_label(ds.labels[summ.members])
            if not act_gids:
                gid = mem.spawn_event(bid, summ, ds)
                gid_true_counts[gid].update(ds.labels[summ.members].tolist())
                continue
            sims = act_protos @ summ.prototype
            best_pos = int(np.argmax(sims))
            best_sim = float(sims[best_pos])
            best_gid = act_gids[best_pos]
            maj_hist = gid_true_counts[best_gid].most_common(1)[0][0]
            if bid >= 2 and summ.size >= min_cross_size:
                bn = _bname(best_sim)
                bins[bn]["total"] += 1
                if maj_hist == maj_cur:
                    bins[bn]["same_true"] += 1
            if best_sim >= 0.72:
                mem.update_event(best_gid, bid, summ, ds)
                gid_true_counts[best_gid].update(ds.labels[summ.members].tolist())
            else:
                gid = mem.spawn_event(bid, summ, ds)
                gid_true_counts[gid].update(ds.labels[summ.members].tolist())
    return dict(bins)


def run_d2_cross_day_audit(
    dataset: str,
    pdata: Dict[str, Any],
    saved_partitions: Dict[str, Dict[int, np.ndarray]],
    resp_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """D2: Zero-token ProtoOnly sweep vs SUG-CrossLLM confusion matrix & Oracle bound."""
    ref_report = json.loads(
        (ROOT / "results" / "streaming" / f"{dataset}_heldout_qwen3_8b" / "streaming_report.json").read_text(
            encoding="utf-8"
        )
    )
    taus = [0.55, 0.60, 0.65, 0.68, 0.72, 0.76, 0.78, 0.80, 0.82, 0.85, 0.90]
    sweep_meso = {}
    sweep_init = {}

    for src_name, target_dict in [
        ("sug_meso_only_direct", sweep_meso),
        ("init_2dse", sweep_init),
    ]:
        part = saved_partitions[src_name]
        target_dict["no_link"] = _instrumented_cross_eval(pdata, part, mode="no_link")
        for tau in taus:
            target_dict[f"tau_{tau:.2f}"] = _instrumented_cross_eval(pdata, part, mode="proto_only", tau_proto=tau)

    # Sanity check proto_only(0.72)
    rec72 = sweep_meso["tau_0.72"]
    ref72 = ref_report["longitudinal_tracking"]["sug_meso_only_direct__proto_only"]
    max_diff_72 = max(abs(rec72["ami"] - ref72["global_ami"]), abs(rec72["macro_b3"] - ref72["global_macro_b3"]))
    if max_diff_72 > 5e-4:
        raise RuntimeError(f"D2 proto_only(0.72) sanity diff={max_diff_72:.6f} on {dataset}")

    # Replay SUG-CrossLLM(0.55, 0.80) + Oracle upper bounds
    cross_journal_path = (
        ROOT / "results" / "streaming" / f"{dataset}_heldout_qwen3_8b"
        / "ledger_cross_sug_meso_only_direct" / "journal.jsonl"
    )
    cross_by_blk = _read_journal_by_block(cross_journal_path)
    cross_budget = int(ref_report["cross_token_budget"])
    cross_replay = _instrumented_cross_eval(
        pdata,
        saved_partitions["sug_meso_only_direct"],
        mode="sug_cross_llm_replay",
        tau_proto=0.72,
        tau_lo=0.55,
        tau_hi=0.80,
        min_cross_size=3,
        cross_token_budget=cross_budget,
        cross_journal_by_blk=cross_by_blk,
        resp_cache=resp_cache,
    )
    ref_llm = ref_report["longitudinal_tracking"]["sug_meso_only_direct__sug_cross_llm"]
    max_diff_llm = max(
        abs(cross_replay["ami"] - ref_llm["global_ami"]), abs(cross_replay["macro_b3"] - ref_llm["global_macro_b3"])
    )
    if max_diff_llm > 5e-4:
        raise RuntimeError(f"D2 sug_cross_llm replay sanity diff={max_diff_llm:.6f} on {dataset}")

    oracle_band_055_080_budgeted = _instrumented_cross_eval(
        pdata,
        saved_partitions["sug_meso_only_direct"],
        mode="oracle_band",
        cross_token_budget=cross_budget,
        oracle_lo=0.55,
        oracle_hi=0.80,
        oracle_min_size=3,
    )
    oracle_band_055_080 = _instrumented_cross_eval(
        pdata,
        saved_partitions["sug_meso_only_direct"],
        mode="oracle_band",
        cross_token_budget=None,
        oracle_lo=0.55,
        oracle_hi=0.80,
        oracle_min_size=3,
    )
    oracle_band_040_085 = _instrumented_cross_eval(
        pdata,
        saved_partitions["sug_meso_only_direct"],
        mode="oracle_band",
        oracle_lo=0.40,
        oracle_hi=0.85,
        oracle_min_size=1,
    )
    oracle_router_055_080 = _instrumented_cross_eval(
        pdata,
        saved_partitions["sug_router_repair"],
        mode="oracle_band",
        oracle_lo=0.55,
        oracle_hi=0.80,
        oracle_min_size=3,
    )
    router_tau80 = _instrumented_cross_eval(
        pdata, saved_partitions["sug_router_repair"], mode="proto_only", tau_proto=0.80
    )

    dev_bins = _cross_sim_bins_on_dev(pdata, min_cross_size=3)

    best_tau_key = max([f"tau_{t:.2f}" for t in taus], key=lambda k: sweep_meso[k]["ami"] + sweep_meso[k]["macro_b3"])
    print(
        f"[D2 {dataset}] Sanity(0.72/LLM)=({max_diff_72:.1e}/{max_diff_llm:.1e}) | "
        f"CrossLLM AMI={cross_replay['ami']:.4f} MacB3={cross_replay['macro_b3']:.4f} "
        f"CM(TP={cross_replay['confusion']['TP']},FP={cross_replay['confusion']['FP']},"
        f"TN={cross_replay['confusion']['TN']},FN={cross_replay['confusion']['FN']}) | "
        f"Proto(0.80) AMI={sweep_meso['tau_0.80']['ami']:.4f} MacB3={sweep_meso['tau_0.80']['macro_b3']:.4f} | "
        f"Oracle[0.55,0.80) AMI={oracle_band_055_080['ami']:.4f} MacB3={oracle_band_055_080['macro_b3']:.4f}"
    )
    return {
        "sanity_max_diff_proto72": max_diff_72,
        "sanity_max_diff_cross_llm": max_diff_llm,
        "proto_sweep_meso": sweep_meso,
        "proto_sweep_init": sweep_init,
        "best_proto_sum_key": best_tau_key,
        "sug_router_repair_proto80": router_tau80,
        "sug_cross_llm_replay": cross_replay,
        "oracle_band_055_080_budgeted": oracle_band_055_080_budgeted,
        "oracle_band_055_080": oracle_band_055_080,
        "oracle_band_040_085": oracle_band_040_085,
        "oracle_router_055_080": oracle_router_055_080,
        "dev_sim_bins": dev_bins,
    }


def _audit_block_list(
    blocks: Dict[int, Dict[str, Any]],
    block_ids: List[int],
    pred_labels_map: Dict[int, np.ndarray],
) -> Dict[str, List[float]]:
    res: Dict[str, List[float]] = defaultdict(list)
    for bid in block_ids:
        bc = blocks[bid]
        ds = bc["ds"]
        li = bc["labels_init"]
        lp = pred_labels_map[bid]
        aud = full_partition_audit(ds.labels, lp, y_before=li)
        res["ami"].append(aud["ami"])
        res["delta_ami"].append(aud["delta_ami"])
        res["nmi"].append(aud["nmi"])
        res["ari"].append(aud["ari"])
        res["macro_b3"].append(aud["bcubed_full"]["macro_b3_f1"])
        res["delta_macro_b3"].append(aud["delta_macro_b3_f1"])
        res["micro_b3"].append(aud["bcubed_full"]["micro_b3_f1"])
        res["delta_micro_b3"].append(aud["delta_micro_b3_f1"])
        res["tail_b3"].append(aud["stratified"]["tail"]["macro_b3_f1"])
    return dict(res)


def run_d3_zero_llm_baselines_and_sensitivity(
    dataset: str,
    pdata: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[float]]]]:
    """D3: Vanilla 2D-SE, Semantic-only 2D-SE, Dev-tuned Stage 1, Louvain, Agglom, Heuristic-Meso."""
    blocks = pdata["blocks"]
    dev_ids: List[int] = pdata["dev_ids"]
    held_ids: List[int] = pdata["heldout_ids"]

    # B1a: Vanilla 2D-SE (refine_moves=False) on held-out
    van_labels = {}
    for bid in held_ids:
        lab_v, _ = load_or_solve_2d_se(dataset, bid, blocks[bid]["graph"].adj, 15, 0.35, 0.5, False)
        van_labels[bid] = lab_v
    van_m = _audit_block_list(blocks, held_ids, van_labels)

    # B1b: Pure Semantic k-NN 2D-SE (attr_alpha=0.0) on held-out
    sem_labels = {}
    for bid in held_ids:
        bc = blocks[bid]
        g_sem = build_message_graph(bc["ds"], bc["emb"], knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.0)
        lab_s, _ = load_or_solve_2d_se(dataset, bid, g_sem.adj, 15, 0.35, 0.0, True)
        sem_labels[bid] = lab_s
    sem_m = _audit_block_list(blocks, held_ids, sem_labels)

    # B2: Clean one-factor-at-a-time + joint Stage-1 sensitivity on Dev M_02..M_05
    dev_grid_cfgs = [
        (15, 0.35, 0.50),
        (10, 0.35, 0.50),
        (25, 0.35, 0.50),
        (15, 0.25, 0.50),
        (15, 0.45, 0.50),
        (15, 0.35, 0.00),
        (15, 0.35, 0.25),
        (15, 0.35, 0.80),
    ]
    dev_sens_rows = []
    for k, tau, alpha in dev_grid_cfgs:
        lab_map = {}
        for bid in dev_ids:
            bc = blocks[bid]
            if (k, tau, alpha) == (15, 0.35, 0.50):
                lab_map[bid] = bc["labels_init"]
            else:
                g = build_message_graph(bc["ds"], bc["emb"], knn_k=k, knn_sim_threshold=tau, attr_alpha=alpha)
                lab_map[bid], _ = load_or_solve_2d_se(dataset, bid, g.adj, k, tau, alpha, True)
        m = _audit_block_list(blocks, dev_ids, lab_map)
        mean_ami = float(np.mean(m["ami"]))
        mean_mac = float(np.mean(m["macro_b3"]))
        dev_sens_rows.append(
            {
                "knn_k": k,
                "tau_sim": tau,
                "attr_alpha": alpha,
                "dev_ami": mean_ami,
                "dev_macro_b3": mean_mac,
                "dev_micro_b3": float(np.mean(m["micro_b3"])),
                "dev_tail_b3": float(np.mean(m["tail_b3"])),
                "dev_score": 0.5 * (mean_ami + mean_mac),
            }
        )

    # Check marginal joint winner if different
    best_k = max([r for r in dev_sens_rows if r["tau_sim"] == 0.35 and r["attr_alpha"] == 0.50], key=lambda r: r["dev_score"])["knn_k"]
    best_t = max([r for r in dev_sens_rows if r["knn_k"] == 15 and r["attr_alpha"] == 0.50], key=lambda r: r["dev_score"])["tau_sim"]
    best_a = max([r for r in dev_sens_rows if r["knn_k"] == 15 and r["tau_sim"] == 0.35], key=lambda r: r["dev_score"])["attr_alpha"]
    if (best_k, best_t, best_a) not in dev_grid_cfgs:
        lab_map = {}
        for bid in dev_ids:
            bc = blocks[bid]
            g = build_message_graph(bc["ds"], bc["emb"], knn_k=best_k, knn_sim_threshold=best_t, attr_alpha=best_a)
            lab_map[bid], _ = load_or_solve_2d_se(dataset, bid, g.adj, best_k, best_t, best_a, True)
        m = _audit_block_list(blocks, dev_ids, lab_map)
        mean_ami = float(np.mean(m["ami"]))
        mean_mac = float(np.mean(m["macro_b3"]))
        dev_sens_rows.append(
            {
                "knn_k": best_k,
                "tau_sim": best_t,
                "attr_alpha": best_a,
                "dev_ami": mean_ami,
                "dev_macro_b3": mean_mac,
                "dev_micro_b3": float(np.mean(m["micro_b3"])),
                "dev_tail_b3": float(np.mean(m["tail_b3"])),
                "dev_score": 0.5 * (mean_ami + mean_mac),
            }
        )

    best_s1 = max(dev_sens_rows, key=lambda r: r["dev_score"])
    bk, bt, ba = int(best_s1["knn_k"]), float(best_s1["tau_sim"]), float(best_s1["attr_alpha"])
    if (bk, bt, ba) == (15, 0.35, 0.50):
        tuned_labels = {bid: blocks[bid]["labels_init"] for bid in held_ids}
    else:
        tuned_labels = {}
        for bid in held_ids:
            bc = blocks[bid]
            g = build_message_graph(bc["ds"], bc["emb"], knn_k=bk, knn_sim_threshold=bt, attr_alpha=ba)
            tuned_labels[bid], _ = load_or_solve_2d_se(dataset, bid, g.adj, bk, bt, ba, True)
    tuned_m = _audit_block_list(blocks, held_ids, tuned_labels)

    # B3: Zero-LLM Heuristic Mesoscopic Post-Processing (Dev-calibrated -> Held-out)
    dev_cand_pools = {}
    for bid in dev_ids:
        bc = blocks[bid]
        merges = detect_merge_pair_queries(bc["ds"], bc["graph"], bc["labels_init"], bc["summaries"], min_proto_sim=0.45)
        seams = detect_seam_queries(bc["ds"], bc["graph"], bc["labels_init"], bc["summaries"], max_conductance=0.50)
        dev_cand_pools[bid] = (merges, seams)

    seam_cfgs = [
        ("none", 999.0, 0.0),
        ("tight", 0.30, 0.25),
        ("med", 0.25, 0.30),
        ("loose", 0.20, 0.35),
        ("default_all", 0.08, 0.50),
    ]
    tau_m_grid = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 999.0]

    def _apply_heur(bc: Dict[str, Any], merges: List[QueryItem], seams: List[QueryItem], tau_m: float, min_sep: float, max_cond: float) -> Tuple[np.ndarray, int]:
        qs: List[QueryItem] = []
        decs: List[LLMDecision] = []
        for q in merges:
            if q.meta.get("proto_sim", 0.0) >= tau_m:
                qs.append(q)
                decs.append(LLMDecision(query_id=q.query_id, query_type="merge_pair", action="MERGE", confidence=1.0))
        for q in seams:
            sem_sep = q.meta.get("intra_pock", 0.0) - q.meta.get("cross_sim", 1.0)
            cond = q.meta.get("conductance", 1.0)
            if sem_sep >= min_sep and cond <= max_cond:
                qs.append(q)
                decs.append(LLMDecision(query_id=q.query_id, query_type="seam", action="SPLIT", target_cluster=q.seam_cluster_id, confidence=1.0))
        lab, _ = apply_direct_assignment(bc["labels_init"], qs, decs, conf_threshold=0.6)
        return lab, len(decs)

    heur_dev_rows = []
    for tm in tau_m_grid:
        for sname, min_sep, max_cond in seam_cfgs:
            lab_map = {}
            qcnts = []
            for bid in dev_ids:
                merges, seams = dev_cand_pools[bid]
                lab_h, qc = _apply_heur(blocks[bid], merges, seams, tm, min_sep, max_cond)
                lab_map[bid] = lab_h
                qcnts.append(qc)
            m = _audit_block_list(blocks, dev_ids, lab_map)
            mean_ami = float(np.mean(m["ami"]))
            mean_mac = float(np.mean(m["macro_b3"]))
            heur_dev_rows.append(
                {
                    "tau_m": tm,
                    "seam_cfg": sname,
                    "min_sem_sep": min_sep,
                    "max_cond": max_cond,
                    "mean_ops": float(np.mean(qcnts)),
                    "dev_ami": mean_ami,
                    "dev_macro_b3": mean_mac,
                    "dev_score": 0.5 * (mean_ami + mean_mac),
                }
            )

    best_heur_merge = max([r for r in heur_dev_rows if r["seam_cfg"] == "none"], key=lambda r: r["dev_score"])
    best_heur_full = max(heur_dev_rows, key=lambda r: r["dev_score"])

    def _eval_heur_heldout(cfg: Dict[str, Any]) -> Tuple[Dict[str, List[float]], float]:
        lab_map = {}
        ops = []
        for bid in held_ids:
            bc = blocks[bid]
            merges = detect_merge_pair_queries(bc["ds"], bc["graph"], bc["labels_init"], bc["summaries"], min_proto_sim=0.45)
            seams = detect_seam_queries(bc["ds"], bc["graph"], bc["labels_init"], bc["summaries"], max_conductance=0.50)
            lab_h, qc = _apply_heur(bc, merges, seams, cfg["tau_m"], cfg["min_sem_sep"], cfg["max_cond"])
            lab_map[bid] = lab_h
            ops.append(qc)
        return _audit_block_list(blocks, held_ids, lab_map), float(np.mean(ops))

    heur_merge_m, heur_merge_ops = _eval_heur_heldout(best_heur_merge)
    heur_full_m, heur_full_ops = _eval_heur_heldout(best_heur_full)

    # B4a: Louvain on identical Stage-1 MessageGraph
    louvain_dev_graphs = {bid: nx.from_scipy_sparse_array(blocks[bid]["graph"].adj) for bid in dev_ids}
    res_grid = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    louv_dev_rows = []
    for res in res_grid:
        lab_map = {}
        for bid in dev_ids:
            comms = nx.community.louvain_communities(louvain_dev_graphs[bid], weight="weight", resolution=res, seed=42)
            lab = np.empty(len(blocks[bid]["ds"]), dtype=np.int64)
            for cid, members in enumerate(comms):
                for node in members:
                    lab[int(node)] = cid
            lab_map[bid] = lab
        m = _audit_block_list(blocks, dev_ids, lab_map)
        louv_dev_rows.append(
            {
                "resolution": res,
                "dev_ami": float(np.mean(m["ami"])),
                "dev_macro_b3": float(np.mean(m["macro_b3"])),
                "dev_score": 0.5 * (float(np.mean(m["ami"])) + float(np.mean(m["macro_b3"]))),
            }
        )
    best_res = max(louv_dev_rows, key=lambda r: r["dev_score"])["resolution"]
    louv_held_labels = {}
    for bid in held_ids:
        g_nx = nx.from_scipy_sparse_array(blocks[bid]["graph"].adj)
        comms = nx.community.louvain_communities(g_nx, weight="weight", resolution=best_res, seed=42)
        lab = np.empty(len(blocks[bid]["ds"]), dtype=np.int64)
        for cid, members in enumerate(comms):
            for node in members:
                lab[int(node)] = cid
        louv_held_labels[bid] = lab
    louv_m = _audit_block_list(blocks, held_ids, louv_held_labels)

    # B4b: Agglomerative Cosine (average linkage) on identical dense embeddings
    dist_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    agg_dev_rows = []
    for dth in dist_grid:
        lab_map = {}
        for bid in dev_ids:
            ac = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=dth)
            lab_map[bid] = ac.fit_predict(blocks[bid]["emb"]).astype(np.int64)
        m = _audit_block_list(blocks, dev_ids, lab_map)
        agg_dev_rows.append(
            {
                "distance_threshold": dth,
                "dev_ami": float(np.mean(m["ami"])),
                "dev_macro_b3": float(np.mean(m["macro_b3"])),
                "dev_score": 0.5 * (float(np.mean(m["ami"])) + float(np.mean(m["macro_b3"]))),
            }
        )
    best_dist = max(agg_dev_rows, key=lambda r: r["dev_score"])["distance_threshold"]
    agg_held_labels = {}
    for bid in held_ids:
        ac = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=best_dist)
        agg_held_labels[bid] = ac.fit_predict(blocks[bid]["emb"]).astype(np.int64)
    agg_m = _audit_block_list(blocks, held_ids, agg_held_labels)

    print(
        f"[D3 {dataset}] VanillaAMI={np.mean(van_m['ami']):.4f} SemOnlyAMI={np.mean(sem_m['ami']):.4f} "
        f"DevTuned({bk},{bt},{ba})AMI={np.mean(tuned_m['ami']):.4f} "
        f"Louvain(r={best_res})AMI={np.mean(louv_m['ami']):.4f} Agglom(d={best_dist})AMI={np.mean(agg_m['ami']):.4f} "
        f"Heur0Tok(tm={best_heur_full['tau_m']},{best_heur_full['seam_cfg']})AMI={np.mean(heur_full_m['ami']):.4f}/Mac={np.mean(heur_full_m['macro_b3']):.4f}"
    )

    d3_out = {
        "vanilla_2dse_no_refine": _summarize_metric_arrays(van_m),
        "semantic_only_2dse_alpha0": _summarize_metric_arrays(sem_m),
        "dev_stage1_sensitivity": dev_sens_rows,
        "dev_tuned_2dse_cfg": {"knn_k": bk, "tau_sim": bt, "attr_alpha": ba},
        "dev_tuned_2dse": _summarize_metric_arrays(tuned_m),
        "heur_meso_merge_only_cfg": best_heur_merge,
        "heur_meso_merge_only_ops_per_blk": heur_merge_ops,
        "heur_meso_merge_only": _summarize_metric_arrays(heur_merge_m),
        "heur_meso_full_cfg": best_heur_full,
        "heur_meso_full_ops_per_blk": heur_full_ops,
        "heur_meso_full": _summarize_metric_arrays(heur_full_m),
        "louvain_dev_sweep": louv_dev_rows,
        "louvain_best_res": best_res,
        "louvain": _summarize_metric_arrays(louv_m),
        "agglom_dev_sweep": agg_dev_rows,
        "agglom_best_dist": best_dist,
        "agglom_cosine": _summarize_metric_arrays(agg_m),
    }
    d3_arrays = {
        "vanilla_2dse": van_m,
        "semantic_only_2dse": sem_m,
        "dev_tuned_2dse": tuned_m,
        "heur_meso_merge_only": heur_merge_m,
        "heur_meso_0tok": heur_full_m,
        "louvain": louv_m,
        "agglom_cosine": agg_m,
    }
    return d3_out, d3_arrays


def run_d4_pathology_and_recurrence(dataset: str, pdata: Dict[str, Any]) -> Dict[str, Any]:
    """D4: Exact Dev vs. Held-Out error pathology, interior zero-entropy fraction, impure census & recurrence."""
    blocks = pdata["blocks"]
    out: Dict[str, Any] = {}

    for split_name, eval_ids in [("dev", pdata["dev_ids"]), ("heldout", pdata["heldout_ids"])]:
        seen_events = set()
        for bid in range(0, min(eval_ids)):
            seen_events.update(np.unique(blocks[bid]["ds"].labels).tolist())

        fm_tot = fs_tot = 0
        ratios = []
        tot_nodes = 0
        zero_ent_incl_iso = 0
        zero_ent_non_iso = 0
        non_iso_nodes = 0

        tot_c10 = 0
        impure_c10 = 0
        minority_shares: List[float] = []
        recurr_posts = 0
        block_recurr = []

        for bid in eval_ids:
            bc = blocks[bid]
            ds = bc["ds"]
            adj = bc["graph"].adj
            lab = bc["labels_init"]
            y_true = ds.labels
            n = len(ds)
            tot_nodes += n

            pw = pairwise_error_counts(y_true, lab)
            fm = int(pw["false_merge_pairs"])
            fs = int(pw["false_split_pairs"])
            fm_tot += fm
            fs_tot += fs
            ratios.append(fs / max(1, fm))

            indptr, indices = adj.indptr, adj.indices
            for i in range(n):
                s, e = indptr[i], indptr[i + 1]
                if s == e:
                    zero_ent_incl_iso += 1
                    continue
                non_iso_nodes += 1
                nbr_labs = lab[indices[s:e]]
                if np.all(nbr_labs == lab[i]):
                    zero_ent_incl_iso += 1
                    zero_ent_non_iso += 1

            c_members: Dict[int, List[int]] = defaultdict(list)
            for idx, c in enumerate(lab.tolist()):
                c_members[int(c)].append(idx)
            for members in c_members.values():
                if len(members) >= 10:
                    tot_c10 += 1
                    vals, cnts = np.unique(y_true[members], return_counts=True)
                    pur = float(cnts.max() / len(members))
                    if pur < 0.90:
                        impure_c10 += 1
                        minority_shares.append(1.0 - pur)

            rec_in_blk = int(sum(1 for ev in y_true.tolist() if ev in seen_events))
            recurr_posts += rec_in_blk
            block_recurr.append(rec_in_blk / max(1, n))
            seen_events.update(np.unique(y_true).tolist())

        ms = np.asarray(minority_shares, dtype=np.float64)
        out[split_name] = {
            "n_blocks": len(eval_ids),
            "n_posts": tot_nodes,
            "total_fm_pairs": fm_tot,
            "total_fs_pairs": fs_tot,
            "overall_fs_fm_ratio": fs_tot / max(1, fm_tot),
            "block_fs_fm_ratio_min": float(np.min(ratios)),
            "block_fs_fm_ratio_median": float(np.median(ratios)),
            "block_fs_fm_ratio_max": float(np.max(ratios)),
            "zero_boundary_entropy_pct_all_nodes": 100.0 * zero_ent_incl_iso / max(1, tot_nodes),
            "zero_boundary_entropy_pct_non_isolated": 100.0 * zero_ent_non_iso / max(1, non_iso_nodes),
            "clusters_ge_10": tot_c10,
            "impure_clusters_pur_lt_090": impure_c10,
            "impure_pct_of_ge_10": 100.0 * impure_c10 / max(1, tot_c10),
            "minority_share_mean": float(np.mean(ms)) if len(ms) > 0 else 0.0,
            "minority_share_median": float(np.median(ms)) if len(ms) > 0 else 0.0,
            "minority_share_q25": float(np.percentile(ms, 25)) if len(ms) > 0 else 0.0,
            "minority_share_q75": float(np.percentile(ms, 75)) if len(ms) > 0 else 0.0,
            "minority_share_pct_in_10_25": float(100.0 * np.mean((ms >= 0.10) & (ms <= 0.25))) if len(ms) > 0 else 0.0,
            "minority_share_pct_in_05_30": float(100.0 * np.mean((ms >= 0.05) & (ms <= 0.30))) if len(ms) > 0 else 0.0,
            "cross_day_recurrence_pct": 100.0 * recurr_posts / max(1, tot_nodes),
            "block_recurrence_pct_min": 100.0 * float(np.min(block_recurr)),
            "block_recurrence_pct_max": 100.0 * float(np.max(block_recurr)),
        }
    print(
        f"[D4 {dataset}] Dev FS/FM={out['dev']['overall_fs_fm_ratio']:.1f}x Rec={out['dev']['cross_day_recurrence_pct']:.2f}% | "
        f"Held FS/FM={out['heldout']['overall_fs_fm_ratio']:.1f}x ({out['heldout']['block_fs_fm_ratio_min']:.2f}x..{out['heldout']['block_fs_fm_ratio_max']:.2f}x) "
        f"H_bnd=0: {out['heldout']['zero_boundary_entropy_pct_all_nodes']:.1f}% Rec={out['heldout']['cross_day_recurrence_pct']:.2f}%"
    )
    return out


def run_d5_stats_and_ledger_audit(
    dataset: str,
    pdata: Dict[str, Any],
    d1_pb: Dict[str, Dict[str, List[float]]],
    d3_arrays: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Any]:
    """D5: Paired Wilcoxon + Holm-Bonferroni + Effect Sizes comparing SUG-Meso (Dir) vs all comparators."""
    eval_ids: List[int] = pdata["heldout_ids"]
    blocks = pdata["blocks"]

    all_methods: Dict[str, Dict[str, List[float]]] = {}
    all_methods.update(d1_pb)
    all_methods.update(d3_arrays)

    # Load all 5 held-out ablations from ablation_report.json
    abl_path = ROOT / "results" / "ablations" / f"{dataset}_heldout_qwen3_8b" / "ablation_report.json"
    abl_rep = json.loads(abl_path.read_text(encoding="utf-8"))
    init_amis = [blocks[b]["init_audit"]["ami"] for b in eval_ids]
    init_macs = [blocks[b]["init_audit"]["bcubed_full"]["macro_b3_f1"] for b in eval_ids]

    for abl_key, abl_obj in abl_rep["ablations"].items():
        bdet = abl_obj["blocks"]
        d_ami = [float(bdet[f"M_{b:02d}"]["delta_ami"]) for b in eval_ids]
        d_mac = [float(bdet[f"M_{b:02d}"]["delta_macro_b3"]) for b in eval_ids]
        all_methods[f"abl_{abl_key}"] = {
            "delta_ami": d_ami,
            "delta_macro_b3": d_mac,
            "ami": [i + d for i, d in zip(init_amis, d_ami)],
            "macro_b3": [i + d for i, d in zip(init_macs, d_mac)],
        }

    anchor_key = "sug_meso_only_direct"
    comparators = [k for k in all_methods.keys() if k != anchor_key]

    tests_by_metric: Dict[str, Dict[str, Any]] = {}
    for metric in ["ami", "macro_b3", "micro_b3", "tail_b3"]:
        eligible = [c for c in comparators if metric in all_methods[c]]
        raw_rows = {}
        pvals = []
        for c in eligible:
            res = _wilcoxon_and_effects(all_methods[anchor_key][metric], all_methods[c][metric])
            raw_rows[c] = res
            pvals.append(res["p_value"])
        holm_pvals = _holm_bonferroni(pvals)
        for c, ph in zip(eligible, holm_pvals):
            raw_rows[c]["p_holm"] = float(ph)
        tests_by_metric[metric] = raw_rows

    summary_stats = {m: _summarize_metric_arrays({k: v for k, v in arrs.items() if k in {"ami", "delta_ami", "macro_b3", "delta_macro_b3", "micro_b3", "tail_b3"}}) for m, arrs in all_methods.items()}
    return {
        "n_eval_blocks": len(eval_ids),
        "summary_stats": summary_stats,
        "paired_vs_sug_meso_direct": tests_by_metric,
    }


def audit_all_ledgers() -> Dict[str, Any]:
    """Audit EVERY journal.jsonl under results/streaming, results/ablations, results/budget_curves."""
    rows = []
    grand = Counter()
    for folder in ["streaming", "ablations", "budget_curves"]:
        base = ROOT / "results" / folder
        if not base.exists():
            continue
        for jp in sorted(base.rglob("journal.jsonl")):
            statuses = Counter()
            pt = ct = live = cached = 0
            lat_sum = 0.0
            with jp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    st = str(rec.get("status", "UNKNOWN"))
                    statuses[st] += 1
                    pt += int(rec.get("prompt_tokens", 0))
                    ct += int(rec.get("completion_tokens", 0))
                    if bool(rec.get("from_cache", False)):
                        cached += 1
                    else:
                        live += 1
                        lat_sum += float(rec.get("latency_sec", 0.0))
            n_tot = sum(statuses.values())
            n_err = sum(v for k, v in statuses.items() if k not in {"OK", "CACHED_REPLAY"})
            rows.append(
                {
                    "path": str(jp.relative_to(ROOT).as_posix()),
                    "calls": n_tot,
                    "live": live,
                    "cached": cached,
                    "errors": n_err,
                    "statuses": dict(statuses),
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                    "mean_live_latency_sec": (lat_sum / live) if live > 0 else 0.0,
                }
            )
            grand["calls"] += n_tot
            grand["live"] += live
            grand["cached"] += cached
            grand["errors"] += n_err
            grand["tokens"] += pt + ct
    return {"grand_totals": dict(grand), "ledgers": rows}


def run_d6_failure_audits_and_cases(
    p2012: Dict[str, Any],
    p2018: Dict[str, Any],
    extra2012: Dict[str, Any],
    extra2018: Dict[str, Any],
    resp2012: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """D6: Failure Audit 1 (Event2018 M_16), Failure Audit 2 (Event2012 Dev 20k->40k dip), Anonymized Case Bank."""
    # --- Failure Audit 1: Event2018 M_16 (-1.73 pp Macro-B3 in SUG-Meso Direct) ---
    bc16 = p2018["blocks"][16]
    ds16, li16, summs16 = bc16["ds"], bc16["labels_init"], bc16["summaries"]
    d16_info = extra2018["decisions"]["sug_meso_only"][16]
    uq16: List[QueryItem] = d16_info["used_queries"]
    decs16: List[LLMDecision] = d16_info["decisions"]
    raw16: Dict[str, Dict[str, Any]] = d16_info["parsed_raw"]

    lab_full16, _ = apply_direct_assignment(li16, uq16, decs16, conf_threshold=0.6)
    aud_full16 = full_partition_audit(ds16.labels, lab_full16, y_before=li16)

    fa1_queries = []
    for idx, (q, d) in enumerate(zip(uq16, decs16)):
        # Leave-one-in effect
        lab_soi, _ = apply_direct_assignment(li16, [q], [d], conf_threshold=0.6)
        aud_soi = full_partition_audit(ds16.labels, lab_soi, y_before=li16)
        # Leave-one-out effect compared to full16
        rem_q = [x for j, x in enumerate(uq16) if j != idx]
        rem_d = [x for j, x in enumerate(decs16) if j != idx]
        lab_loo, _ = apply_direct_assignment(li16, rem_q, rem_d, conf_threshold=0.6)
        aud_loo = full_partition_audit(ds16.labels, lab_loo, y_before=li16)

        if q.query_type == "merge_pair":
            m1, m2 = summs16[int(q.merge_c1)].members, summs16[int(q.merge_c2)].members
            c1_ev = Counter(ds16.labels[m1].tolist()).most_common(2)
            c2_ev = Counter(ds16.labels[m2].tolist()).most_common(2)
            gt_same = c1_ev[0][0] == c2_ev[0][0]
            verdict = "TP" if (gt_same and d.action == "MERGE") else ("FP" if (not gt_same and d.action == "MERGE") else ("FN" if (gt_same and d.action == "SEPARATE") else "TN"))
            sizes = (len(m1), len(m2))
        else:
            c1_ev = Counter(ds16.labels[q.seam_sub1].tolist()).most_common(2)
            c2_ev = Counter(ds16.labels[q.seam_sub2].tolist()).most_common(2)
            gt_same = c1_ev[0][0] == c2_ev[0][0]
            verdict = "TP" if ((not gt_same) and d.action == "SPLIT") else ("FP" if (gt_same and d.action == "SPLIT") else ("FN" if ((not gt_same) and d.action == "KEEP_MERGED") else "TN"))
            sizes = (len(q.seam_sub1), len(q.seam_sub2))

        fa1_queries.append(
            {
                "order": idx,
                "query_id": q.query_id,
                "query_type": q.query_type,
                "action": d.action,
                "verdict": verdict,
                "sizes": sizes,
                "top_events_A": c1_ev,
                "top_events_B": c2_ev,
                "event_a_llm": raw16[q.query_id].get("event_a", ""),
                "event_b_llm": raw16[q.query_id].get("event_b", ""),
                "solo_delta_ami": aud_soi["delta_ami"],
                "solo_delta_macro_b3": aud_soi["delta_macro_b3_f1"],
                "marginal_loo_delta_macro_b3": aud_full16["delta_macro_b3_f1"] - aud_loo["delta_macro_b3_f1"],
                "marginal_loo_delta_ami": aud_full16["delta_ami"] - aud_loo["delta_ami"],
            }
        )

    # --- Failure Audit 2: Event2012 Dev Budget Curve 20k -> 40k Macro-B3 Dip ---
    curve_path = ROOT / "results" / "budget_curves" / "Event2012_dev_qwen3_8b" / "Event2012_dev_qwen3_8b_curve.json"
    curve_json = json.loads(curve_path.read_text(encoding="utf-8"))
    per_blk_20_40 = {}
    j20 = _read_journal_by_block(
        ROOT / "results" / "budget_curves" / "Event2012_dev_qwen3_8b" / "budget_20000" / "ledger_sug_meso_only" / "journal.jsonl"
    )
    j40 = _read_journal_by_block(
        ROOT / "results" / "budget_curves" / "Event2012_dev_qwen3_8b" / "budget_40000" / "ledger_sug_meso_only" / "journal.jsonl"
    )
    dip_details = []
    for bid in p2012["dev_ids"]:
        btag = f"M_{bid:02d}"
        b20 = curve_json["tiers"]["20000"]["strategies"]["sug_meso_only"]["blocks"][btag]["branch1_direct"]
        b40 = curve_json["tiers"]["40000"]["strategies"]["sug_meso_only"]["blocks"][btag]["branch1_direct"]
        d20 = b20["delta_macro_b3_f1"]
        d40 = b40["delta_macro_b3_f1"]
        per_blk_20_40[btag] = {
            "dMac_20k": d20,
            "dMac_40k": d40,
            "diff_40k_minus_20k": d40 - d20,
            "q_20k": len(j20.get(bid, [])),
            "q_40k": len(j40.get(bid, [])),
        }
        if d40 < d20 - 1e-6:
            bc = p2012["blocks"][bid]
            ds, g, li, summs = bc["ds"], bc["graph"], bc["labels_init"], bc["summaries"]
            pool = _select_pool("sug_meso_only", ds, g, li, 120, seed=42 + bid)
            qmap_p = {q.query_id: q for q in pool}
            q20_ids = {r["request_id"] for r in j20.get(bid, [])}
            added_recs = [r for r in j40.get(bid, []) if r["request_id"] not in q20_ids]
            all_40_q, all_40_d = [], []
            for r in j40.get(bid, []):
                q = qmap_p[r["request_id"]]
                parsed = resp2012[r["request_hash"]]["response"]
                all_40_q.append(q)
                all_40_d.append(parse_llm_response_to_decision(q, parsed))
            for r in added_recs:
                q = qmap_p[r["request_id"]]
                parsed = resp2012[r["request_hash"]]["response"]
                dec = parse_llm_response_to_decision(q, parsed)
                # Incremental effect over 20k prefix
                pref_idx = [i for i, x in enumerate(all_40_q) if x.query_id in q20_ids]
                lab_20, _ = apply_direct_assignment(li, [all_40_q[i] for i in pref_idx], [all_40_d[i] for i in pref_idx])
                lab_plus, _ = apply_direct_assignment(
                    li,
                    [all_40_q[i] for i in pref_idx] + [q],
                    [all_40_d[i] for i in pref_idx] + [dec],
                )
                a20 = full_partition_audit(ds.labels, lab_20, y_before=li)
                aplus = full_partition_audit(ds.labels, lab_plus, y_before=li)
                if q.query_type == "merge_pair":
                    evA = Counter(ds.labels[summs[int(q.merge_c1)].members].tolist()).most_common(2)
                    evB = Counter(ds.labels[summs[int(q.merge_c2)].members].tolist()).most_common(2)
                else:
                    evA = Counter(ds.labels[q.seam_sub1].tolist()).most_common(2)
                    evB = Counter(ds.labels[q.seam_sub2].tolist()).most_common(2)
                dip_details.append(
                    {
                        "block": btag,
                        "query_id": q.query_id,
                        "query_type": q.query_type,
                        "action": dec.action,
                        "delta_mac_over_20k": aplus["bcubed_full"]["macro_b3_f1"] - a20["bcubed_full"]["macro_b3_f1"],
                        "delta_ami_over_20k": aplus["ami"] - a20["ami"],
                        "top_events_A": evA,
                        "top_events_B": evB,
                        "event_a_llm": parsed.get("event_a", ""),
                        "event_b_llm": parsed.get("event_b", ""),
                    }
                )

    # --- Anonymized Qualitative Case Bank (3 EN, 3 FR) ---
    def _extract_lang_cases(pdata: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        blocks = pdata["blocks"]
        best_merge = None
        best_split = None
        best_fp = None
        for bid in pdata["heldout_ids"]:
            bc = blocks[bid]
            ds, li, summs = bc["ds"], bc["labels_init"], bc["summaries"]
            info = extra["decisions"]["sug_meso_only"][bid]
            for q, d in zip(info["used_queries"], info["decisions"]):
                parsed = info["parsed_raw"][q.query_id]
                lab_s, _ = apply_direct_assignment(li, [q], [d], conf_threshold=0.6)
                aud_s = full_partition_audit(ds.labels, lab_s, y_before=li)
                et = aud_s["error_transition"]
                if q.query_type == "merge_pair":
                    m1, m2 = summs[int(q.merge_c1)].members, summs[int(q.merge_c2)].members
                    ev1 = Counter(ds.labels[m1].tolist()).most_common(2)
                    ev2 = Counter(ds.labels[m2].tolist()).most_common(2)
                    gt_same = ev1[0][0] == ev2[0][0]
                    snippet1 = [_anonymize_text(ds.texts[m]) for m in q.merge_c1_medoids[:2]]
                    snippet2 = [_anonymize_text(ds.texts[m]) for m in q.merge_c2_medoids[:2]]
                    rec = {
                        "block": f"M_{bid:02d}",
                        "query_id": q.query_id,
                        "type": "MERGE_PAIR",
                        "action": d.action,
                        "sizes": [len(m1), len(m2)],
                        "proto_sim": q.meta.get("proto_sim", 0.0),
                        "event_a_llm": parsed.get("event_a", ""),
                        "event_b_llm": parsed.get("event_b", ""),
                        "true_events_A": ev1,
                        "true_events_B": ev2,
                        "delta_ami": aud_s["delta_ami"],
                        "delta_macro_b3": aud_s["delta_macro_b3_f1"],
                        "net_fs_red": et["net_false_split_reduction"],
                        "net_fm_red": et["net_false_merge_reduction"],
                        "medoids_A": snippet1,
                        "medoids_B": snippet2,
                    }
                    if gt_same and d.action == "MERGE":
                        score = et["net_false_split_reduction"]
                        if best_merge is None or score > best_merge[0]:
                            best_merge = (score, rec)
                    elif (not gt_same) and d.action == "MERGE":
                        score = -aud_s["delta_macro_b3_f1"]
                        if best_fp is None or score > best_fp[0]:
                            best_fp = (score, rec)
                elif q.query_type == "seam":
                    ev1 = Counter(ds.labels[q.seam_sub1].tolist()).most_common(2)
                    ev2 = Counter(ds.labels[q.seam_sub2].tolist()).most_common(2)
                    gt_same = ev1[0][0] == ev2[0][0]
                    snippet1 = [_anonymize_text(ds.texts[m]) for m in q.seam_sub1_medoids[:2]]
                    snippet2 = [_anonymize_text(ds.texts[m]) for m in q.seam_sub2_medoids[:2]]
                    rec = {
                        "block": f"M_{bid:02d}",
                        "query_id": q.query_id,
                        "type": "SPLIT_SEAM",
                        "action": d.action,
                        "sizes": [len(q.seam_sub1), len(q.seam_sub2)],
                        "conductance": q.meta.get("conductance", 0.0),
                        "phi_non_ent": q.meta.get("phi_non_ent", 0.0),
                        "sem_sep": q.meta.get("intra_pock", 0.0) - q.meta.get("cross_sim", 0.0),
                        "event_a_llm": parsed.get("event_a", ""),
                        "event_b_llm": parsed.get("event_b", ""),
                        "true_events_A": ev1,
                        "true_events_B": ev2,
                        "delta_ami": aud_s["delta_ami"],
                        "delta_macro_b3": aud_s["delta_macro_b3_f1"],
                        "net_fm_red": et["net_false_merge_reduction"],
                        "net_fs_red": et["net_false_split_reduction"],
                        "medoids_A": snippet1,
                        "medoids_B": snippet2,
                    }
                    if (not gt_same) and d.action == "SPLIT" and aud_s["delta_macro_b3_f1"] > 0:
                        score = aud_s["delta_macro_b3_f1"]
                        if best_split is None or score > best_split[0]:
                            best_split = (score, rec)
                    elif gt_same and d.action == "SPLIT":
                        score = -aud_s["delta_macro_b3_f1"]
                        if best_fp is None or score > best_fp[0]:
                            best_fp = (score, rec)
        return {
            "tp_merge": best_merge[1] if best_merge else None,
            "tp_split": best_split[1] if best_split else None,
            "fp_error": best_fp[1] if best_fp else None,
        }

    return {
        "failure_audit_1_event2018_M16": {
            "full_delta_ami": aud_full16["delta_ami"],
            "full_delta_macro_b3": aud_full16["delta_macro_b3_f1"],
            "queries": fa1_queries,
        },
        "failure_audit_2_event2012_dev_20k_to_40k": {
            "per_block_20k_vs_40k": per_blk_20_40,
            "added_queries_in_dip_blocks": dip_details,
        },
        "case_bank": {
            "Event2012": _extract_lang_cases(p2012, extra2012),
            "Event2018": _extract_lang_cases(p2018, extra2018),
        },
    }


def write_markdown_summary(report: Dict[str, Any], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# IP&M Stage 1 Decisive Zero-Token Diagnostic Report (D1–D6)\n")
    lines.append(f"- **Ledger Grand Audit**: `{report['ledger_audit']['grand_totals']}`\n")

    for ds in ["Event2012", "Event2018"]:
        d1 = report[ds]["D1_queried_audit"]
        d2 = report[ds]["D2_cross_day"]
        d3 = report[ds]["D3_zero_llm_baselines"]
        d4 = report[ds]["D4_pathology"]
        d5 = report[ds]["D5_stats"]

        lines.append(f"\n---\n\n## {ds} (`n_blocks_eval={d5['n_eval_blocks']}`, `N_eval={d1['eval_total_posts']:,}`)\n")

        # D1 Table
        lines.append("### D1: Corrected Prompt-Shown Post Accounting & Uncontaminated $U_t$\n")
        lines.append("| Strategy | Calls (%) | Primary (%) | All-Shown (%) | Unq ΔMic-B³ (All-Shown) | UnqNet/blk (Legacy→All-Shown) | UnqNet/Call | AMI | Mac-B³ |\n")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for key in ["random_direct", "rep_div_direct", "cequel_edge_direct", "cequel_edge_repair", "cequel_transitive", "sug_router_repair", "sug_meso_only_direct", "sug_meso_only_repair"]:
            if key not in d1["strategies"]:
                continue
            s = d1["strategies"][key]
            c = s["counts"]
            m = s["metrics"]
            leg_unq = m.get("legacy_unq_net_posts", m["allshown_unq_net_posts"])["mean"]
            lines.append(
                f"| `{key}` | {c['total_calls']} ({c['calls_pct']:.2f}%) | {c['primary_posts']} ({c['primary_pct']:.2f}%) | "
                f"{c['all_shown_posts']} ({c['all_shown_pct']:.2f}%) | "
                f"{m['allshown_unq_delta_micro_b3']['mean']:+.4f} | "
                f"{leg_unq:+.1f} -> {m['allshown_unq_net_posts']['mean']:+.1f} | "
                f"{c['unq_net_per_call_all_shown']:+.1f} | "
                f"{m['ami']['mean']:.4f} | {m['macro_b3']['mean']:.4f} |\n"
            )

        # D2 Table
        lines.append("\n### D2: Cross-Block `ProtoOnly` Threshold Sweep vs. `SUG-CrossLLM` & Oracle\n")
        lines.append("| Variant (`SUG-Meso Direct` stream) | Global AMI | Global Mac-B³ | Global Mic-B³ | Tail-B³ | AutoLinked / LLM / Spawned |\n")
        lines.append("| :--- | ---: | ---: | ---: | ---: | :--- |\n")
        lines.append(
            f"| `no_link` | {d2['proto_sweep_meso']['no_link']['ami']:.4f} | {d2['proto_sweep_meso']['no_link']['macro_b3']:.4f} | "
            f"{d2['proto_sweep_meso']['no_link']['micro_b3']:.4f} | {d2['proto_sweep_meso']['no_link']['tail_b3']:.4f} | 0 / 0 / {d2['proto_sweep_meso']['no_link']['stats']['spawned_new']} |\n"
        )
        for tau in [0.55, 0.65, 0.68, 0.72, 0.76, 0.78, 0.80, 0.82, 0.85, 0.90]:
            k = f"tau_{tau:.2f}"
            v = d2["proto_sweep_meso"][k]
            st = v["stats"]
            lines.append(
                f"| `ProtoOnly(τ={tau:.2f})` | {v['ami']:.4f} | {v['macro_b3']:.4f} | {v['micro_b3']:.4f} | {v['tail_b3']:.4f} | "
                f"{st['auto_linked']} / 0 / {st['spawned_new']} |\n"
            )
        cl = d2["sug_cross_llm_replay"]
        cm = cl["confusion"]
        st = cl["stats"]
        lines.append(
            f"| **`SUG-CrossLLM(0.55,0.80)`** | **{cl['ami']:.4f}** | **{cl['macro_b3']:.4f}** | **{cl['micro_b3']:.4f}** | {cl['tail_b3']:.4f} | "
            f"{st['auto_linked']} / {st['llm_queried']} (L={st['llm_linked']},R={st['llm_rejected']}) / {st['spawned_new']} |\n"
        )
        obb = d2["oracle_band_055_080_budgeted"]
        lines.append(
            f"| `Oracle-Band[0.55,0.80) (Same Budget)` | {obb['ami']:.4f} | {obb['macro_b3']:.4f} | {obb['micro_b3']:.4f} | {obb['tail_b3']:.4f} | "
            f"{obb['stats']['auto_linked']} / {obb['stats']['llm_queried']} (L={obb['stats']['llm_linked']},R={obb['stats']['llm_rejected']}) / {obb['stats']['spawned_new']} |\n"
        )
        ob = d2["oracle_band_055_080"]
        lines.append(
            f"| `Oracle-Band[0.55,0.80) (Full)` | {ob['ami']:.4f} | {ob['macro_b3']:.4f} | {ob['micro_b3']:.4f} | {ob['tail_b3']:.4f} | "
            f"{ob['stats']['auto_linked']} / {ob['stats']['llm_queried']} (L={ob['stats']['llm_linked']},R={ob['stats']['llm_rejected']}) / {ob['stats']['spawned_new']} |\n"
        )
        ob2 = d2["oracle_band_040_085"]
        lines.append(
            f"| `Oracle-Band[0.40,0.85)` | {ob2['ami']:.4f} | {ob2['macro_b3']:.4f} | {ob2['micro_b3']:.4f} | {ob2['tail_b3']:.4f} | "
            f"{ob2['stats']['auto_linked']} / {ob2['stats']['llm_queried']} (L={ob2['stats']['llm_linked']},R={ob2['stats']['llm_rejected']}) / {ob2['stats']['spawned_new']} |\n"
        )
        lines.append(
            f"\n- **CrossLLM Confusion Matrix**: `TP={cm['TP']}, FP={cm['FP']}, TN={cm['TN']}, FN={cm['FN']}` "
            f"(Prec={cm['precision']:.3f}, Rec={cm['recall']:.3f}, F1={cm['f1']:.3f})\n"
        )
        lines.append("- **Held-Out Similarity Bins (`size>=3`) True Match Rate**: ")
        lines.append(
            ", ".join(
                f"`{bn}`: {vals['same_true']}/{vals['total']} ({100*vals['same_true']/max(1,vals['total']):.1f}%)"
                for bn, vals in sorted(cl["sim_bins"].items())
            )
            + "\n"
        )

        # D3 Table
        lines.append("\n### D3: Zero-LLM & Dev-Tuned Baselines (Held-Out Mean ± SD)\n")
        lines.append("| Method | AMI (Mean ± SD) | Mac-B³ (Mean ± SD) | Mic-B³ | Tail-B³ | p_Wilcoxon vs SUG-Meso (Dir) [AMI / Mac-B³] |\n")
        lines.append("| :--- | ---: | ---: | ---: | ---: | :--- |\n")
        for mkey, label in [
            ("init_2dse", "Init 2D-SE (Default)"),
            ("vanilla_2dse", "Vanilla 2D-SE (w/o Node-Moves)"),
            ("semantic_only_2dse", "Semantic-Only 2D-SE (α=0)"),
            ("dev_tuned_2dse", f"Dev-Tuned 2D-SE {d3['dev_tuned_2dse_cfg']}"),
            ("louvain", f"Louvain (res={d3['louvain_best_res']})"),
            ("agglom_cosine", f"Agglom-Cosine (d={d3['agglom_best_dist']})"),
            ("heur_meso_merge_only", f"Meso-Heur MergeOnly (τ={d3['heur_meso_merge_only_cfg']['tau_m']})"),
            ("heur_meso_0tok", f"Meso-Heur 0-Tok (τ={d3['heur_meso_full_cfg']['tau_m']},{d3['heur_meso_full_cfg']['seam_cfg']})"),
            ("cequel_transitive", "Cequel-Transitive (LLM)"),
            ("sug_meso_only_direct", "**SUG-Meso (Direct, LLM)**"),
        ]:
            st_m = d5["summary_stats"][mkey]
            if mkey == "sug_meso_only_direct":
                p_str = "Anchor"
            else:
                pa = d5["paired_vs_sug_meso_direct"]["ami"].get(mkey, {})
                pm = d5["paired_vs_sug_meso_direct"]["macro_b3"].get(mkey, {})
                p_str = f"p={pa.get('p_value', 1.0):.2e} (H={pa.get('p_holm', 1.0):.2e}, dz={pa.get('cohens_dz', 0.0):.2f}) / p={pm.get('p_value', 1.0):.2e} (H={pm.get('p_holm', 1.0):.2e}, dz={pm.get('cohens_dz', 0.0):.2f})"
            lines.append(
                f"| {label} | {st_m['ami']['mean']:.4f} ± {st_m['ami']['sd']:.4f} | "
                f"{st_m['macro_b3']['mean']:.4f} ± {st_m['macro_b3']['sd']:.4f} | "
                f"{st_m['micro_b3']['mean']:.4f} | {st_m['tail_b3']['mean']:.4f} | {p_str} |\n"
            )

        # D5 LLM & Ablation Statistical Tests vs SUG-Meso (Direct)
        lines.append("\n### D5: Paired Wilcoxon + Holm-Bonferroni Tests (`SUG-Meso Direct` vs. LLM Baselines, Repair, & Ablations)\n")
        lines.append("| Comparator | AMI (Mean ± SD) | Mac-B³ (Mean ± SD) | ΔAMI vs Anchor (p / p_Holm / dz) | ΔMac-B³ vs Anchor (p / p_Holm / dz) |\n")
        lines.append("| :--- | ---: | ---: | :--- | :--- |\n")
        for mkey in [
            "random_direct",
            "rep_div_direct",
            "cequel_edge_repair",
            "cequel_transitive",
            "sug_router_repair",
            "sug_meso_only_repair",
            "abl_merge_only",
            "abl_seam_only",
            "abl_fiedler_only_seam",
            "abl_no_extract_compare",
            "abl_uniform_budget",
        ]:
            if mkey not in d5["summary_stats"]:
                continue
            st_m = d5["summary_stats"][mkey]
            pa = d5["paired_vs_sug_meso_direct"]["ami"].get(mkey, {})
            pm = d5["paired_vs_sug_meso_direct"]["macro_b3"].get(mkey, {})
            lines.append(
                f"| `{mkey}` | {st_m['ami']['mean']:.4f} ± {st_m['ami']['sd']:.4f} | "
                f"{st_m['macro_b3']['mean']:.4f} ± {st_m['macro_b3']['sd']:.4f} | "
                f"{pa.get('mean_diff', 0.0):+.4f} (p={pa.get('p_value', 1.0):.2e}, H={pa.get('p_holm', 1.0):.2e}, dz={pa.get('cohens_dz', 0.0):+.2f}) | "
                f"{pm.get('mean_diff', 0.0):+.4f} (p={pm.get('p_value', 1.0):.2e}, H={pm.get('p_holm', 1.0):.2e}, dz={pm.get('cohens_dz', 0.0):+.2f}) |\n"
            )

        # D4 Summary
        lines.append("\n### D4: Dev vs. Held-Out Initial Error Pathology & Cross-Day Recurrence\n")
        for sp in ["dev", "heldout"]:
            p = d4[sp]
            lines.append(
                f"- **{sp}**: N={p['n_posts']:,}, FS/FM={p['overall_fs_fm_ratio']:.2f}× (blk {p['block_fs_fm_ratio_min']:.2f}×–{p['block_fs_fm_ratio_max']:.2f}×), "
                f"H_bnd=0={p['zero_boundary_entropy_pct_all_nodes']:.1f}%, "
                f"Impure(|C|>=10, pur<0.90)={p['impure_clusters_pur_lt_090']}/{p['clusters_ge_10']} ({p['impure_pct_of_ge_10']:.1f}%), "
                f"minority share median={p['minority_share_median']:.3f} [IQR {p['minority_share_q25']:.3f}–{p['minority_share_q75']:.3f}] "
                f"({p['minority_share_pct_in_10_25']:.1f}% in [0.10,0.25]), Recurrence={p['cross_day_recurrence_pct']:.2f}%.\n"
            )

    # D6 Failure Audits
    d6 = report["D6_failures_and_cases"]
    lines.append("\n---\n\n## D6: Root-Cause Failure Audits\n")
    fa1 = d6["failure_audit_1_event2018_M16"]
    lines.append(
        f"### Failure Audit 1: `Event2018 M_16` (`ΔAMI={fa1['full_delta_ami']:+.4f}`, `ΔMac-B³={fa1['full_delta_macro_b3']:+.4f}`)\n"
    )
    lines.append("| Query | Type | Action | Verdict | Sizes | Top Events A vs B | Solo ΔMac-B³ | LOO Marginal ΔMac-B³ | LLM Event A / B |\n")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: | :--- |\n")
    for q in fa1["queries"]:
        lines.append(
            f"| `{q['query_id']}` | {q['query_type']} | {q['action']} | **{q['verdict']}** | {q['sizes']} | "
            f"{q['top_events_A']} vs {q['top_events_B']} | {q['solo_delta_macro_b3']:+.4f} | {q['marginal_loo_delta_macro_b3']:+.4f} | "
            f"`{q['event_a_llm']}` // `{q['event_b_llm']}` |\n"
        )

    fa2 = d6["failure_audit_2_event2012_dev_20k_to_40k"]
    lines.append("\n### Failure Audit 2: `Event2012` Dev Budget Curve `20k -> 40k` Macro-B³ Dip\n")
    for btag, vals in fa2["per_block_20k_vs_40k"].items():
        lines.append(
            f"- `{btag}`: `20k dMac={vals['dMac_20k']:+.4f}` ({vals['q_20k']}q) -> "
            f"`40k dMac={vals['dMac_40k']:+.4f}` ({vals['q_40k']}q) [diff=`{vals['diff_40k_minus_20k']:+.4f}`]\n"
        )
    for dq in fa2["added_queries_in_dip_blocks"]:
        lines.append(
            f"  - Added `{dq['block']} {dq['query_id']}` ({dq['action']}): `ΔMac over 20k={dq['delta_mac_over_20k']:+.4f}`, "
            f"`ΔAMI={dq['delta_ami_over_20k']:+.4f}`, events={dq['top_events_A']} vs {dq['top_events_B']} | "
            f"`{dq['event_a_llm']}` // `{dq['event_b_llm']}`\n"
        )

    lines.append("\n### Anonymized Qualitative Case Bank\n")
    for ds_name, cdict in d6["case_bank"].items():
        lines.append(f"\n#### {ds_name}\n")
        for ctype in ["tp_merge", "tp_split", "fp_error"]:
            c = cdict.get(ctype)
            if c is None:
                continue
            lines.append(
                f"- **{ctype}** (`{c['block']} {c['query_id']}`, action=`{c['action']}`, sizes=`{c['sizes']}`, "
                f"`ΔAMI={c['delta_ami']:+.4f}`, `ΔMac-B³={c['delta_macro_b3']:+.4f}`, "
                f"`NetFS={c['net_fs_red']:+d}`, `NetFM={c['net_fm_red']:+d}`): "
                f"LLM: `{c['event_a_llm']}` // `{c['event_b_llm']}`\n"
                f"  - A medoids: {c['medoids_A']}\n"
                f"  - B medoids: {c['medoids_B']}\n"
            )

    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    t_start = time.time()
    out_dir = ROOT / "results" / "ipm_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    p2012 = preload_dataset_blocks("Event2012")
    p2018 = preload_dataset_blocks("Event2018")
    resp2012 = _load_all_cached_responses("Event2012")
    resp2018 = _load_all_cached_responses("Event2018")

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ledger_audit": audit_all_ledgers(),
    }

    extras = {}
    for ds_name, pdata, rcache in [
        ("Event2012", p2012, resp2012),
        ("Event2018", p2018, resp2018),
    ]:
        d1_out, saved_parts, extra = run_d1_replay_and_queried_audit(ds_name, pdata, rcache)
        d2_out = run_d2_cross_day_audit(ds_name, pdata, saved_parts, rcache)
        d3_out, d3_arrays = run_d3_zero_llm_baselines_and_sensitivity(ds_name, pdata)
        d4_out = run_d4_pathology_and_recurrence(ds_name, pdata)
        d5_out = run_d5_stats_and_ledger_audit(ds_name, pdata, extra["per_block"], d3_arrays)
        report[ds_name] = {
            "D1_queried_audit": d1_out,
            "D2_cross_day": d2_out,
            "D3_zero_llm_baselines": d3_out,
            "D4_pathology": d4_out,
            "D5_stats": d5_out,
        }
        extras[ds_name] = extra

    report["D6_failures_and_cases"] = run_d6_failure_audits_and_cases(
        p2012, p2018, extras["Event2012"], extras["Event2018"], resp2012
    )
    report["elapsed_sec"] = round(time.time() - t_start, 2)

    json_path = out_dir / "stage1_diagnostics.json"
    md_path = out_dir / "stage1_summary.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_summary(report, md_path)

    print(f"\n[DONE] Stage 1 diagnostics finished in {report['elapsed_sec']:.1f}s")
    print(f"  JSON: {json_path}")
    print(f"  MD  : {md_path}")


if __name__ == "__main__":
    main()
