"""IP&M Stage 2A Experiment Suite (Qwen/Qwen3-8B, Strictly Causal Real-Token Protocol).

Covers on both Event2012 (EN) and Event2018 (FR):
  - S2-1A: Dev-only (M_02..M_05) prompt-schema calibration (extract_compare vs action_first)
  - S2-1B: Strictly causal held-out (M_06+) protocol comparing CausalUniformFixed,
           CausalUniformCarryover, CausalSliding, multi-seed Random (3 seeds),
           and clean 0-error causal ablations (merge_only, seam_only, fiedler_seam)
  - S2-2 : Structural routing & Eq. (5)/(6) ablations + 3-seed Random-Meso
           (eq5_no_leverage, eq5_cosine_only, eq6_no_conductance, eq6_no_non_entity, eq6_raw_tfidf)
  - S2-3 : Upgraded causal cross-day tracking SUG-CrossLLM-v2 ([0.65, 0.85), tau_fb=0.85)
           calibrated on Dev and evaluated on Held-out vs ProtoOnly(0.72/0.80/0.85/0.90) & Oracle
  - S2-6 : Hyperparameter sensitivity (MERGE/SEAM ratio 80/20, 65/35, 50/50; max_conductance 0.35/0.50/0.65)

All outputs are written strictly under results/ipm_experiments/stage2a_qwen3/ while reusing
results/llm_cache/{dataset}_Qwen__Qwen3-8B/responses.jsonl with honest real-token billing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import numpy as np
from scipy import sparse, stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset, load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import MessageGraph, build_message_graph
from core_shared.key_loader import load_provider_key
from core_shared.metrics_audit import full_partition_audit
from core_shared.partition_cache import load_or_solve_2d_se
from core_shared.token_ledger import (
    CausalSlidingBucket,
    CausalUniformBucket,
    TokenLedger,
    canonical_request_hash,
)
from trackA_sug_sed.high_leverage_router import (
    ClusterSummary,
    QueryItem,
    _interleave_query_queues,
    build_cluster_summaries,
    detect_merge_pair_queries,
    detect_seam_queries,
    select_random_meso_queries,
    select_random_queries,
    select_sug_queries,
)
from trackA_sug_sed.llm_client import (
    BudgetedLLMClient,
    _extract_json,
    build_action_first_prompt,
    build_cross_block_prompt,
    build_query_prompt,
)
from trackA_sug_sed.run_ablations_qwen3 import _detect_fiedler_only_seams
from trackA_sug_sed.run_gateA_oracle_diagnostic import load_or_extract_block_embeddings
from trackA_sug_sed.streaming_sed import GlobalEventMemory
from trackA_sug_sed.topology_surgery import (
    LLMDecision,
    apply_direct_assignment,
    apply_topology_repair_and_recluster,
)


def _discover_all_caches(dataset: str) -> Tuple[Path, List[Path]]:
    shared = ROOT / "results" / "llm_cache" / f"{dataset}_Qwen__Qwen3-8B" / "responses.jsonl"
    extras: List[Path] = []
    for folder in [
        ROOT / "results" / "streaming" / f"{dataset}_heldout_qwen3_8b",
        ROOT / "results" / "ablations" / f"{dataset}_heldout_qwen3_8b",
        ROOT / "results" / "budget_curves" / f"{dataset}_dev_qwen3_8b",
        ROOT / "results" / "gateA_llm",
    ]:
        if folder.exists():
            extras.extend(sorted(folder.rglob("responses.jsonl")))
    return shared, extras


def _build_raw_tfidf(ds: SocialDataset) -> sparse.csr_matrix:
    """TF-IDF over all lemmas >= 2 chars WITHOUT stripping entity tokens (for Eq. 6 ablation)."""
    docs = [" ".join([t for t in lemmas if len(t) >= 2]) for lemmas in ds.tokens_lemma]
    vec = TfidfVectorizer(min_df=2, max_df=0.8, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    try:
        mat = vec.fit_transform(docs)
    except ValueError:
        mat = sparse.csr_matrix((len(docs), 1), dtype=np.float64)
    return normalize(mat, norm="l2", axis=1).tocsr()


def _majority_label(arr: np.ndarray) -> int:
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _wilcoxon_and_effects(x_ours: Sequence[float], x_base: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(x_ours, dtype=np.float64)
    b = np.asarray(x_base, dtype=np.float64)
    diff = a - b
    n = len(diff)
    mean_d = float(np.mean(diff))
    sd_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    dz = (mean_d / sd_d) if sd_d > 1e-12 else 0.0
    nz = diff[np.abs(diff) > 1e-12]
    if len(nz) == 0:
        return {
            "n": n,
            "mean_diff": mean_d,
            "sd_diff": sd_d,
            "cohens_dz": 0.0,
            "rank_biserial_r": 0.0,
            "p_value": 1.0,
        }
    ranks = stats.rankdata(np.abs(nz), method="average")
    w_pos = float(np.sum(ranks[nz > 0]))
    w_neg = float(np.sum(ranks[nz < 0]))
    denom = float(len(nz) * (len(nz) + 1) / 2.0)
    r_rb = (w_pos - w_neg) / denom if denom > 0 else 0.0
    res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", correction=False)
    return {
        "n": n,
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "cohens_dz": float(dz),
        "rank_biserial_r": float(r_rb),
        "p_value": float(res.pvalue),
    }


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adj = np.empty(m, dtype=np.float64)
    rmax = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * float(p_values[idx])
        rmax = max(rmax, min(1.0, val))
        adj[int(idx)] = rmax
    return adj.tolist()


def _summarize_arrays(per_blk: Dict[str, List[float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, vals in per_blk.items():
        arr = np.asarray(vals, dtype=np.float64)
        if len(arr) == 0:
            continue
        out[k] = {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "per_block": [float(x) for x in arr.tolist()],
        }
    return out


def _cm_stats(cm: Counter) -> Dict[str, Any]:
    tp, fp, tn, fn = int(cm["TP"]), int(cm["FP"]), int(cm["TN"]), int(cm["FN"])
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": round(tp / max(1, tp + fp), 4),
        "recall": round(tp / max(1, tp + fn), 4),
        "specificity": round(tn / max(1, tn + fp), 4),
    }


def _update_merge_seam_cm(
    ds: SocialDataset,
    summaries: Dict[int, ClusterSummary],
    used_queries: List[QueryItem],
    decisions: List[LLMDecision],
    conf_threshold: float,
    cm_m: Counter,
    cm_s: Counter,
) -> None:
    y = ds.labels
    qmap = {q.query_id: q for q in used_queries}
    maj_c = {cid: _majority_label(y[summ.members]) for cid, summ in summaries.items()}
    for d in decisions:
        if d.confidence < conf_threshold:
            continue
        q = qmap[d.query_id]
        if d.query_type == "merge_pair":
            same_true = maj_c[int(q.merge_c1)] == maj_c[int(q.merge_c2)]
            pred = d.action == "MERGE"
            if pred and same_true:
                cm_m["TP"] += 1
            elif pred and not same_true:
                cm_m["FP"] += 1
            elif (not pred) and same_true:
                cm_m["FN"] += 1
            else:
                cm_m["TN"] += 1
        elif d.query_type == "seam":
            diff_true = _majority_label(y[q.seam_sub1]) != _majority_label(y[q.seam_sub2])
            pred = d.action == "SPLIT"
            if pred and diff_true:
                cm_s["TP"] += 1
            elif pred and not diff_true:
                cm_s["FP"] += 1
            elif (not pred) and diff_true:
                cm_s["FN"] += 1
            else:
                cm_s["TN"] += 1


def _get_std_merges(bc: Dict[str, Any]) -> List[QueryItem]:
    if "std_merges" not in bc:
        bc["std_merges"] = detect_merge_pair_queries(
            bc["ds"], bc["graph"], bc["labels_init"], bc["summaries"]
        )
    return bc["std_merges"]


def _get_std_seams(bc: Dict[str, Any]) -> List[QueryItem]:
    if "std_seams" not in bc:
        bc["std_seams"] = detect_seam_queries(
            bc["ds"], bc["graph"], bc["labels_init"], bc["summaries"]
        )
    return bc["std_seams"]


def _build_variant_pool(
    pool_kind: str,
    bc: Dict[str, Any],
    bid: int,
    max_queries: int = 120,
) -> List[QueryItem]:
    lock = bc.get("pool_lock")
    if lock is not None:
        with lock:
            pcache = bc.setdefault("pool_cache", {})
            if pool_kind in pcache:
                return pcache[pool_kind]
            pool = _build_variant_pool_uncached(pool_kind, bc, bid, max_queries=max_queries)
            pcache[pool_kind] = pool
            return pool
    pcache = bc.setdefault("pool_cache", {})
    if pool_kind in pcache:
        return pcache[pool_kind]
    pool = _build_variant_pool_uncached(pool_kind, bc, bid, max_queries=max_queries)
    pcache[pool_kind] = pool
    return pool


def _build_variant_pool_uncached(
    pool_kind: str,
    bc: Dict[str, Any],
    bid: int,
    max_queries: int = 120,
) -> List[QueryItem]:
    ds = bc["ds"]
    graph = bc["graph"]
    labels = bc["labels_init"]
    summaries = bc["summaries"]
    raw_tfidf = bc["raw_tfidf"]

    if pool_kind.startswith("random_s"):
        seed = int(pool_kind.split("random_s", 1)[1])
        return select_random_queries(ds, graph, labels, max_queries, seed=seed + bid)
    if pool_kind.startswith("random_meso_s"):
        seed = int(pool_kind.split("random_meso_s", 1)[1])
        return select_random_meso_queries(ds, graph, labels, max_queries, merge_frac=0.65, seam_frac=0.35, seed=seed + bid)
    if pool_kind == "sug_meso":
        return _interleave_query_queues([("merge", _get_std_merges(bc), 0.65), ("seam", _get_std_seams(bc), 0.35)], max_queries)
    if pool_kind == "merge_only":
        return _interleave_query_queues([("merge", _get_std_merges(bc), 1.0)], max_queries)
    if pool_kind == "seam_only":
        return _interleave_query_queues([("seam", _get_std_seams(bc), 1.0)], max_queries)
    if pool_kind == "fiedler_seam":
        seams = _detect_fiedler_only_seams(ds, graph, summaries)
        return _interleave_query_queues([("merge", _get_std_merges(bc), 0.65), ("seam", seams, 0.35)], max_queries)
    if pool_kind == "eq5_no_leverage":
        merges = detect_merge_pair_queries(ds, graph, labels, summaries, use_leverage=False)
        return _interleave_query_queues([("merge", merges, 0.65), ("seam", _get_std_seams(bc), 0.35)], max_queries)
    if pool_kind == "eq5_cosine_only":
        merges = detect_merge_pair_queries(ds, graph, labels, summaries, use_leverage=False, use_kw_edge_bonus=False)
        return _interleave_query_queues([("merge", merges, 0.65), ("seam", _get_std_seams(bc), 0.35)], max_queries)
    if pool_kind == "eq6_no_conductance":
        seams = detect_seam_queries(ds, graph, labels, summaries, use_conductance=False)
        return _interleave_query_queues([("merge", _get_std_merges(bc), 0.65), ("seam", seams, 0.35)], max_queries)
    if pool_kind == "eq6_no_non_entity":
        seams = detect_seam_queries(ds, graph, labels, summaries, use_non_entity=False)
        return _interleave_query_queues([("merge", _get_std_merges(bc), 0.65), ("seam", seams, 0.35)], max_queries)
    if pool_kind == "eq6_raw_tfidf":
        seams = detect_seam_queries(ds, graph, labels, summaries, override_tfidf=raw_tfidf)
        return _interleave_query_queues([("merge", _get_std_merges(bc), 0.65), ("seam", seams, 0.35)], max_queries)
    if pool_kind == "ratio_80_20":
        return _interleave_query_queues([("merge", _get_std_merges(bc), 0.80), ("seam", _get_std_seams(bc), 0.20)], max_queries)
    if pool_kind == "ratio_50_50":
        return _interleave_query_queues([("merge", _get_std_merges(bc), 0.50), ("seam", _get_std_seams(bc), 0.50)], max_queries)
    if pool_kind == "cond_035":
        seams = detect_seam_queries(ds, graph, labels, summaries, max_conductance=0.35)
        return _interleave_query_queues([("merge", _get_std_merges(bc), 0.65), ("seam", seams, 0.35)], max_queries)
    if pool_kind == "cond_065":
        seams = detect_seam_queries(ds, graph, labels, summaries, max_conductance=0.65)
        return _interleave_query_queues([("merge", _get_std_merges(bc), 0.65), ("seam", seams, 0.35)], max_queries)
    raise ValueError(f"Unknown pool_kind: {pool_kind!r}")


def _prompt_fn_for(mode: str) -> Callable[..., Tuple[str, str]]:
    if mode == "extract_compare":
        return build_query_prompt
    if mode == "action_first":
        return build_action_first_prompt
    raise ValueError(f"Unknown prompt_mode: {mode!r}")


def _simulate_dry_run(
    variants: List[Dict[str, Any]],
    block_cache: Dict[int, Dict[str, Any]],
    probe_ledger: TokenLedger,
    model_name: str,
    seen_new_hashes: Dict[str, Tuple[int, int]],
) -> Dict[str, int]:
    """Simulate exact query prefixes without HTTP calls and count unique live requests."""
    tot_calls = tot_cached = tot_live = tot_tok = 0
    for var in variants:
        bids: List[int] = var["block_ids"]
        pfn = _prompt_fn_for(var["prompt_mode"])
        ctrl_type = var["controller"]
        if ctrl_type == "fixed":
            bucket = None
        elif ctrl_type == "carryover":
            bucket = CausalUniformBucket(tokens_per_block=10000, allow_carryover=True, max_carryover_mult=1.5)
        elif ctrl_type == "sliding":
            bucket = CausalSlidingBucket(tokens_per_block=10000)
            for wb in var.get("warmup_ids", []):
                bucket.observe_warmup_block(len(block_cache[wb]["ds"]))
        else:
            raise ValueError(ctrl_type)

        for bid in bids:
            bc = block_cache[bid]
            pool = _build_variant_pool(var["pool_kind"], bc, bid)
            mean_u = float(np.mean([q.score for q in pool])) if pool else 1.0
            if bucket is None:
                blk_bucket = CausalUniformBucket(tokens_per_block=10000, allow_carryover=False)
                blk_bucket.start_block(mean_uncertainty=mean_u)
            else:
                if ctrl_type == "sliding":
                    bucket.start_block(len(bc["ds"]), mean_uncertainty=mean_u)
                else:
                    bucket.start_block(mean_uncertainty=mean_u)
                blk_bucket = bucket

            for q in pool:
                sm, um = pfn(q, bc["ds"], bc["summaries"])
                est = int((len(sm) + len(um)) / 3.5) + 120
                if not blk_bucket.can_spend(est):
                    break
                rh = canonical_request_hash(
                    {
                        "model": model_name,
                        "temperature": 0.0,
                        "messages": [{"role": "system", "content": sm}, {"role": "user", "content": um}],
                    }
                )
                hit = probe_ledger.lookup_cache(rh)
                tot_calls += 1
                if hit is not None:
                    pt, ct = int(hit["prompt_tokens"]), int(hit["completion_tokens"])
                    tot_cached += 1
                    blk_bucket.record_spend(pt, ct)
                    tot_tok += pt + ct
                elif rh in seen_new_hashes:
                    pt, ct = seen_new_hashes[rh]
                    tot_cached += 1
                    blk_bucket.record_spend(pt, ct)
                    tot_tok += pt + ct
                else:
                    est_pt = int((len(sm) + len(um)) / 3.5)
                    est_ct = 40
                    seen_new_hashes[rh] = (est_pt, est_ct)
                    tot_live += 1
                    blk_bucket.record_spend(est_pt, est_ct)
                    tot_tok += est_pt + est_ct
    return {
        "total_calls": tot_calls,
        "cached_or_dedup": tot_cached,
        "unique_live_needed": tot_live,
        "total_tokens": tot_tok,
    }


def _run_within_block_variant(
    var: Dict[str, Any],
    dataset: str,
    block_cache: Dict[int, Dict[str, Any]],
    out_dir: Path,
    shared_cache_file: Path,
    extra_cache_files: List[Path],
    model_name: str,
    base_url: str,
    api_key: str,
    workers: int,
    conf_threshold: float = 0.6,
) -> Tuple[Dict[str, Any], Dict[int, np.ndarray]]:
    name = var["name"]
    bids: List[int] = var["block_ids"]
    pfn = _prompt_fn_for(var["prompt_mode"])
    ctrl_type = var["controller"]
    eval_repair = bool(var.get("eval_repair", False))

    led_dir = out_dir / dataset / f"ledger_{name}"
    if (led_dir / "journal.jsonl").exists():
        (led_dir / "journal.jsonl").unlink()
    ledger = TokenLedger(
        led_dir,
        price_per_1m_prompt_cny=0.5,
        price_per_1m_completion_cny=2.0,
        extra_cache_dirs=extra_cache_files,
        shared_cache_file=shared_cache_file,
    )
    client = BudgetedLLMClient(model=model_name, ledger=ledger, base_url=base_url, api_key=api_key)

    # Phase A (main thread, sequential): build all pools & mean uncertainties before threads start
    pools: Dict[int, List[QueryItem]] = {}
    mean_us: Dict[int, float] = {}
    for bid in bids:
        bc = block_cache[bid]
        pool = _build_variant_pool(var["pool_kind"], bc, bid)
        pools[bid] = pool
        mean_us[bid] = float(np.mean([q.score for q in pool])) if pool else 1.0

    # Phase B (HTTP-only concurrency for independent fixed-budget blocks; sequential for stateful budgets)
    def _query_one_block(bid: int, blk_bucket: Any) -> Tuple[int, int, List[LLMDecision]]:
        bc = block_cache[bid]
        ds, summs = bc["ds"], bc["summaries"]
        pool = pools[bid]
        mean_u = mean_us[bid]
        if ctrl_type == "sliding":
            cap = blk_bucket.start_block(len(ds), mean_uncertainty=mean_u)
        else:
            cap = blk_bucket.start_block(mean_uncertainty=mean_u)
        decisions = client.execute_queries(
            pool, ds, summs, block_id=bid, bucket=blk_bucket, prompt_fn=pfn, strategy_tag=name
        )
        return bid, cap, decisions

    raw_decisions: Dict[int, Tuple[int, List[LLMDecision]]] = {}
    if ctrl_type == "fixed" and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for bid in bids:
                bkt = CausalUniformBucket(tokens_per_block=10000, allow_carryover=False)
                futs[ex.submit(_query_one_block, bid, bkt)] = bid
            for fut in as_completed(futs):
                bid_r, cap_r, decs_r = fut.result()
                raw_decisions[bid_r] = (cap_r, decs_r)
    else:
        if ctrl_type == "fixed":
            shared_bkt = None
        elif ctrl_type == "carryover":
            shared_bkt = CausalUniformBucket(tokens_per_block=10000, allow_carryover=True, max_carryover_mult=1.5)
        elif ctrl_type == "sliding":
            shared_bkt = CausalSlidingBucket(tokens_per_block=10000)
            for wb in var.get("warmup_ids", []):
                shared_bkt.observe_warmup_block(len(block_cache[wb]["ds"]))
        else:
            raise ValueError(ctrl_type)

        for bid in bids:
            bkt = CausalUniformBucket(tokens_per_block=10000, allow_carryover=False) if shared_bkt is None else shared_bkt
            bid_r, cap_r, decs_r = _query_one_block(bid, bkt)
            raw_decisions[bid_r] = (cap_r, decs_r)

    # Phase C (main thread, sequential): topology surgery, repair, and partition auditing
    per_blk_results: Dict[int, Dict[str, Any]] = {}
    for bid in bids:
        bc = block_cache[bid]
        ds, graph, li, summs = bc["ds"], bc["graph"], bc["labels_init"], bc["summaries"]
        pool = pools[bid]
        cap, decisions = raw_decisions[bid]
        dec_ids = {d.query_id for d in decisions}
        used_queries = [q for q in pool if q.query_id in dec_ids]
        btok = sum(d.prompt_tokens + d.completion_tokens for d in decisions)

        lab_dir, qmask_dir = apply_direct_assignment(
            li, used_queries, decisions, conf_threshold=conf_threshold, summaries=summs, mask_mode="all_shown"
        )
        aud_dir = full_partition_audit(ds.labels, lab_dir, y_before=li, queried_mask=qmask_dir)

        lab_rep = None
        aud_rep = None
        if eval_repair:
            lab_rep, qmask_rep, _ = apply_topology_repair_and_recluster(
                graph, li, summs, used_queries, decisions, conf_threshold=conf_threshold, mask_mode="all_shown"
            )
            aud_rep = full_partition_audit(ds.labels, lab_rep, y_before=li, queried_mask=qmask_rep)

        per_blk_results[bid] = {
            "bid": bid,
            "n_posts": len(ds),
            "cap": cap,
            "tokens": btok,
            "decisions": decisions,
            "used_queries": used_queries,
            "lab_dir": lab_dir,
            "aud_dir": aud_dir,
            "qmask_dir": qmask_dir,
            "lab_rep": lab_rep,
            "aud_rep": aud_rep,
        }

    # Aggregate in deterministic sorted block order
    eval_total_posts = sum(len(block_cache[b]["ds"]) for b in bids)
    cm_m, cm_s = Counter(), Counter()
    branch_arrays: Dict[str, Dict[str, List[float]]] = {"direct": defaultdict(list)}
    if eval_repair:
        branch_arrays["repair"] = defaultdict(list)

    saved_parts_dir: Dict[int, np.ndarray] = {}
    saved_parts_rep: Dict[int, np.ndarray] = {}
    total_calls = total_shown = total_unq_dir = total_unq_rep = 0

    for bid in bids:
        r = per_blk_results[bid]
        bc = block_cache[bid]
        _update_merge_seam_cm(
            bc["ds"], bc["summaries"], r["used_queries"], r["decisions"], conf_threshold, cm_m, cm_s
        )
        total_calls += len(r["decisions"])
        total_shown += int(r["qmask_dir"].sum())
        saved_parts_dir[bid] = r["lab_dir"]

        for btag, aud in [("direct", r["aud_dir"])] + ([("repair", r["aud_rep"])] if eval_repair else []):
            prop = aud["propagation_audit"]
            arrs = branch_arrays[btag]
            arrs["queries"].append(float(len(r["decisions"])))
            arrs["tokens"].append(float(r["tokens"]))
            arrs["ami"].append(float(aud["ami"]))
            arrs["delta_ami"].append(float(aud["delta_ami"]))
            arrs["nmi"].append(float(aud["nmi"]))
            arrs["ari"].append(float(aud["ari"]))
            arrs["macro_b3"].append(float(aud["bcubed_full"]["macro_b3_f1"]))
            arrs["delta_macro_b3"].append(float(aud["delta_macro_b3_f1"]))
            arrs["micro_b3"].append(float(aud["bcubed_full"]["micro_b3_f1"]))
            arrs["delta_micro_b3"].append(float(aud["delta_micro_b3_f1"]))
            arrs["tail_b3"].append(float(aud["stratified"]["tail"]["macro_b3_f1"]))
            arrs["unq_delta_micro_b3"].append(float(prop["unqueried_delta_micro_b3_f1"]))
            arrs["unq_delta_macro_b3"].append(float(prop["unqueried_delta_macro_b3_f1"]))
            arrs["unq_net_posts"].append(float(prop["unqueried_net_improved_posts"]))
            if btag == "direct":
                total_unq_dir += int(prop["unqueried_net_improved_posts"])
            else:
                total_unq_rep += int(prop["unqueried_net_improved_posts"])
                saved_parts_rep[bid] = r["lab_rep"]

    # Ledger health check
    statuses = Counter()
    jpath = led_dir / "journal.jsonl"
    if jpath.exists():
        with jpath.open("r", encoding="utf-8") as jf:
            for line in jf:
                if line.strip():
                    statuses[json.loads(line).get("status", "UNKNOWN")] += 1
    n_err = sum(v for k, v in statuses.items() if k not in {"OK", "CACHED_REPLAY"})
    lsum = ledger.summary()
    lsum["errors"] = int(n_err)
    lsum["statuses"] = dict(statuses)

    m_dir = _summarize_arrays(branch_arrays["direct"])
    print(
        f"  [{dataset:9s} | {name:34s}] calls={total_calls:3d} ({100*total_shown/max(1,eval_total_posts):.2f}% shown) "
        f"tok={lsum['total_tokens']:6d} (live={lsum['total_calls']-lsum['cached_calls']:3d}, err={n_err}) | "
        f"AMI={m_dir['ami']['mean']:.4f} ({m_dir['delta_ami']['mean']:+.4f}) "
        f"MacB3={m_dir['macro_b3']['mean']:.4f} ({m_dir['delta_macro_b3']['mean']:+.4f}) "
        f"Tail={m_dir['tail_b3']['mean']:.4f} UnqNet={m_dir['unq_net_posts']['mean']:+.1f}"
    )

    result_obj: Dict[str, Any] = {
        "name": name,
        "pool_kind": var["pool_kind"],
        "prompt_mode": var["prompt_mode"],
        "controller": ctrl_type,
        "block_ids": bids,
        "eval_total_posts": eval_total_posts,
        "ledger": lsum,
        "counts": {
            "total_calls": total_calls,
            "calls_pct": 100.0 * total_calls / max(1, eval_total_posts),
            "all_shown_posts": total_shown,
            "all_shown_pct": 100.0 * total_shown / max(1, eval_total_posts),
            "unq_net_total_direct": total_unq_dir,
            "unq_net_per_call_direct": total_unq_dir / max(1, total_calls),
            "unq_net_per_shown_post_direct": total_unq_dir / max(1, total_shown),
        },
        "confusion_merge": _cm_stats(cm_m),
        "confusion_seam": _cm_stats(cm_s),
        "direct": m_dir,
    }
    if eval_repair:
        result_obj["repair"] = _summarize_arrays(branch_arrays["repair"])
        result_obj["counts"]["unq_net_total_repair"] = total_unq_rep
    return result_obj, {"direct": saved_parts_dir, "repair": saved_parts_rep}


def _parse_cross_verdict(parsed: Dict[str, Any]) -> Optional[bool]:
    if not isinstance(parsed, dict) or "error" in parsed:
        return None
    if "same_event" in parsed:
        return bool(parsed["same_event"])
    if "action" in parsed:
        act = str(parsed["action"]).upper().strip()
        if act in {"MERGE", "ASSIGN", "SAME"}:
            return True
        if act in {"SEPARATE", "NEW_EVENT", "SPLIT"}:
            return False
    return None


def _run_cross_day_suite(
    dataset: str,
    tag: str,
    part_dict: Dict[int, np.ndarray],
    block_cache: Dict[int, Dict[str, Any]],
    warmup_ids: List[int],
    eval_ids: List[int],
    out_dir: Path,
    shared_cache_file: Path,
    extra_cache_files: List[Path],
    model_name: str,
    base_url: str,
    api_key: str,
    dry_run: bool,
    seen_new_hashes: Dict[str, Tuple[int, int]],
) -> Dict[str, Any]:
    """Run S2-3 longitudinal cross-day tracking on a fixed daily partition stream."""
    results: Dict[str, Any] = {}

    def _run_single_cross(
        mode: str,
        tau_proto: float = 0.85,
        tau_lo: float = 0.65,
        tau_hi: float = 0.85,
        tau_fallback: float = 0.85,
        min_cross_size: int = 3,
        prompt_mode: str = "action_first",
        tokens_per_block: int = 6000,
    ) -> Dict[str, Any]:
        mem = GlobalEventMemory(window_blocks=14)
        gid_true_counts: Dict[int, Counter] = defaultdict(Counter)

        # Warmup blocks strictly with zero LLM tokens using consistent prototype threshold
        for bid in warmup_ids:
            bc = block_cache[bid]
            ds, emb, lab = bc["ds"], bc["emb"], bc["labels_init"]
            summs = build_cluster_summaries(ds, emb, lab)
            ordered = sorted(summs.keys(), key=lambda c: -summs[c].size)
            active = mem.active_events(bid)
            act_gids = [ev.global_id for ev in active]
            act_protos = np.vstack([ev.prototype for ev in active]) if active else np.zeros((0, emb.shape[1]), dtype=np.float32)
            for cid in ordered:
                summ = summs[cid]
                if mode == "no_link" or not act_gids:
                    gid = mem.spawn_event(bid, summ, ds)
                else:
                    sims = act_protos @ summ.prototype
                    bp = int(np.argmax(sims))
                    if float(sims[bp]) >= tau_proto:
                        gid = act_gids[bp]
                        mem.update_event(gid, bid, summ, ds)
                    else:
                        gid = mem.spawn_event(bid, summ, ds)
                gid_true_counts[gid].update(ds.labels[summ.members].tolist())

        ledger = None
        client = None
        bucket = None
        probe = None
        run_label = f"cross_{tag}_{mode}_{prompt_mode}" if "llm" in mode else f"cross_{tag}_{mode}_{tau_proto:.2f}"

        if mode == "sug_cross_llm_v2":
            bucket = CausalUniformBucket(tokens_per_block=tokens_per_block, allow_carryover=True, max_carryover_mult=1.5)
            led_dir = out_dir / dataset / f"ledger_{run_label}"
            if dry_run:
                probe = TokenLedger(
                    out_dir / "_dry_probe_cross",
                    extra_cache_dirs=extra_cache_files,
                    shared_cache_file=shared_cache_file,
                )
            else:
                if (led_dir / "journal.jsonl").exists():
                    (led_dir / "journal.jsonl").unlink()
                ledger = TokenLedger(
                    led_dir,
                    price_per_1m_prompt_cny=0.5,
                    price_per_1m_completion_cny=2.0,
                    extra_cache_dirs=extra_cache_files,
                    shared_cache_file=shared_cache_file,
                )
                client = BudgetedLLMClient(model=model_name, ledger=ledger, base_url=base_url, api_key=api_key)

        eval_true_all, eval_pred_all = [], []
        cm = Counter()
        auto_linked = spawned_new = llm_queried = llm_linked = llm_rejected = 0
        dry_stats = {"calls": 0, "cached": 0, "live": 0, "tokens": 0}

        for bid in eval_ids:
            bc = block_cache[bid]
            ds, emb = bc["ds"], bc["emb"]
            lab = part_dict[bid]
            summs = build_cluster_summaries(ds, emb, lab)
            ordered = sorted(summs.keys(), key=lambda c: -summs[c].size)
            active = mem.active_events(bid)
            act_gids = [ev.global_id for ev in active]
            act_protos = np.vstack([ev.prototype for ev in active]) if active else np.zeros((0, emb.shape[1]), dtype=np.float32)
            if bucket is not None:
                bucket.start_block()

            l2g: Dict[int, int] = {}
            for cid in ordered:
                summ = summs[cid]
                maj_cur = _majority_label(ds.labels[summ.members])
                if mode == "no_link" or not act_gids:
                    gid = mem.spawn_event(bid, summ, ds)
                    l2g[cid] = gid
                    gid_true_counts[gid].update(ds.labels[summ.members].tolist())
                    spawned_new += 1
                    continue

                sims = act_protos @ summ.prototype
                bp = int(np.argmax(sims))
                bs = float(sims[bp])
                bg = act_gids[bp]
                maj_hist = gid_true_counts[bg].most_common(1)[0][0]
                same_true = maj_hist == maj_cur

                if mode == "proto_only":
                    if bs >= tau_proto:
                        mem.update_event(bg, bid, summ, ds)
                        l2g[cid] = bg
                        gid_true_counts[bg].update(ds.labels[summ.members].tolist())
                        auto_linked += 1
                    else:
                        gid = mem.spawn_event(bid, summ, ds)
                        l2g[cid] = gid
                        gid_true_counts[gid].update(ds.labels[summ.members].tolist())
                        spawned_new += 1
                elif mode == "oracle_band":
                    if bs >= tau_hi:
                        mem.update_event(bg, bid, summ, ds)
                        l2g[cid] = bg
                        gid_true_counts[bg].update(ds.labels[summ.members].tolist())
                        auto_linked += 1
                    elif bs >= tau_lo and summ.size >= min_cross_size:
                        llm_queried += 1
                        if same_true:
                            mem.update_event(bg, bid, summ, ds)
                            l2g[cid] = bg
                            gid_true_counts[bg].update(ds.labels[summ.members].tolist())
                            llm_linked += 1
                        else:
                            gid = mem.spawn_event(bid, summ, ds)
                            l2g[cid] = gid
                            gid_true_counts[gid].update(ds.labels[summ.members].tolist())
                            llm_rejected += 1
                            spawned_new += 1
                    else:
                        if bs >= tau_fallback:
                            mem.update_event(bg, bid, summ, ds)
                            l2g[cid] = bg
                            gid_true_counts[bg].update(ds.labels[summ.members].tolist())
                            auto_linked += 1
                        else:
                            gid = mem.spawn_event(bid, summ, ds)
                            l2g[cid] = gid
                            gid_true_counts[gid].update(ds.labels[summ.members].tolist())
                            spawned_new += 1
                elif mode == "sug_cross_llm_v2":
                    if bs >= tau_hi:
                        mem.update_event(bg, bid, summ, ds)
                        l2g[cid] = bg
                        gid_true_counts[bg].update(ds.labels[summ.members].tolist())
                        auto_linked += 1
                    elif bs >= tau_lo and summ.size >= min_cross_size:
                        hist_ev = mem.events[bg]
                        sm, um = build_cross_block_prompt(hist_ev, summ, ds, prompt_mode=prompt_mode)
                        est = int((len(sm) + len(um)) / 3.5) + 120
                        verdict: Optional[bool] = None
                        if bucket is None or bucket.can_spend(est):
                            req_payload = {
                                "model": model_name,
                                "temperature": 0.0,
                                "messages": [{"role": "system", "content": sm}, {"role": "user", "content": um}],
                            }
                            rh = canonical_request_hash(req_payload)
                            if dry_run:
                                dry_stats["calls"] += 1
                                hit = probe.lookup_cache(rh) if probe else None
                                if hit is not None:
                                    pt, ct = int(hit["prompt_tokens"]), int(hit["completion_tokens"])
                                    dry_stats["cached"] += 1
                                    verdict = _parse_cross_verdict(hit["response"])
                                elif rh in seen_new_hashes:
                                    pt, ct = seen_new_hashes[rh]
                                    dry_stats["cached"] += 1
                                    verdict = False  # conservative fallback assumption for dry-run chain
                                else:
                                    pt, ct = int((len(sm) + len(um)) / 3.5), 40
                                    seen_new_hashes[rh] = (pt, ct)
                                    dry_stats["live"] += 1
                                    verdict = False
                                dry_stats["tokens"] += pt + ct
                                if bucket is not None:
                                    bucket.record_spend(pt, ct)
                            else:
                                cached = client.ledger.lookup_cache(rh)
                                qid = f"cross_M{bid:02d}_g{bg}_c{cid}"
                                if cached is not None:
                                    parsed = cached["response"]
                                    pt, ct = int(cached["prompt_tokens"]), int(cached["completion_tokens"])
                                    client.ledger.log_call(
                                        request_id=qid,
                                        block_id=bid,
                                        strategy=f"sug_cross_v2_{prompt_mode}",
                                        query_type="cross_merge",
                                        model=model_name,
                                        request_hash=rh,
                                        prompt_tokens=pt,
                                        completion_tokens=ct,
                                        latency_sec=0.0,
                                        status="CACHED_REPLAY",
                                        response_payload=parsed,
                                        from_cache=True,
                                    )
                                else:
                                    pt = ct = 0
                                    latency = 0.0
                                    try:
                                        raw_t, pt, ct, latency, fr = client._call_http(req_payload["messages"])
                                        try:
                                            parsed = _extract_json(raw_t)
                                            st = "OK"
                                        except Exception as pe:
                                            parsed = {"error": str(pe), "finish_reason": fr}
                                            st = f"PARSE_ERROR:{type(pe).__name__}"
                                    except Exception as he:
                                        parsed = {"error": str(he)}
                                        st = f"HTTP_ERROR:{type(he).__name__}"
                                    client.ledger.log_call(
                                        request_id=qid,
                                        block_id=bid,
                                        strategy=f"sug_cross_v2_{prompt_mode}",
                                        query_type="cross_merge",
                                        model=model_name,
                                        request_hash=rh,
                                        prompt_tokens=pt,
                                        completion_tokens=ct,
                                        latency_sec=latency,
                                        status=st,
                                        response_payload=parsed,
                                        from_cache=False,
                                    )
                                if bucket is not None:
                                    bucket.record_spend(pt, ct)
                                verdict = _parse_cross_verdict(parsed)

                        if verdict is not None:
                            llm_queried += 1
                            if verdict and same_true:
                                cm["TP"] += 1
                            elif verdict and not same_true:
                                cm["FP"] += 1
                            elif (not verdict) and same_true:
                                cm["FN"] += 1
                            else:
                                cm["TN"] += 1

                        if verdict is True:
                            mem.update_event(bg, bid, summ, ds)
                            l2g[cid] = bg
                            gid_true_counts[bg].update(ds.labels[summ.members].tolist())
                            llm_linked += 1
                        elif verdict is False:
                            gid = mem.spawn_event(bid, summ, ds)
                            l2g[cid] = gid
                            gid_true_counts[gid].update(ds.labels[summ.members].tolist())
                            llm_rejected += 1
                            spawned_new += 1
                        else:
                            if bs >= tau_fallback:
                                mem.update_event(bg, bid, summ, ds)
                                l2g[cid] = bg
                                gid_true_counts[bg].update(ds.labels[summ.members].tolist())
                                auto_linked += 1
                            else:
                                gid = mem.spawn_event(bid, summ, ds)
                                l2g[cid] = gid
                                gid_true_counts[gid].update(ds.labels[summ.members].tolist())
                                spawned_new += 1
                    else:
                        if bs >= tau_fallback:
                            mem.update_event(bg, bid, summ, ds)
                            l2g[cid] = bg
                            gid_true_counts[bg].update(ds.labels[summ.members].tolist())
                            auto_linked += 1
                        else:
                            gid = mem.spawn_event(bid, summ, ds)
                            l2g[cid] = gid
                            gid_true_counts[gid].update(ds.labels[summ.members].tolist())
                            spawned_new += 1
                else:
                    raise ValueError(mode)

            glab = np.array([l2g[int(c)] for c in lab.tolist()], dtype=np.int64)
            eval_true_all.append(ds.labels)
            eval_pred_all.append(glab)

        if dry_run:
            return {"dry_stats": dry_stats}

        aud = full_partition_audit(np.concatenate(eval_true_all), np.concatenate(eval_pred_all))
        lsum = ledger.summary() if ledger is not None else None
        out = {
            "ami": float(aud["ami"]),
            "nmi": float(aud["nmi"]),
            "ari": float(aud["ari"]),
            "macro_b3": float(aud["bcubed_full"]["macro_b3_f1"]),
            "micro_b3": float(aud["bcubed_full"]["micro_b3_f1"]),
            "tail_b3": float(aud["stratified"]["tail"]["macro_b3_f1"]),
            "n_pred_events": int(aud["n_pred_clusters"]),
            "stats": {
                "auto_linked": auto_linked,
                "llm_queried": llm_queried,
                "llm_linked": llm_linked,
                "llm_rejected": llm_rejected,
                "spawned_new": spawned_new,
            },
            "confusion": _cm_stats(cm) if "llm" in mode else None,
            "ledger": lsum,
        }
        print(
            f"  [Cross {dataset} | {tag} | {run_label}] AMI={out['ami']:.4f} MacB3={out['macro_b3']:.4f} "
            f"MicB3={out['micro_b3']:.4f} Tail={out['tail_b3']:.4f} Q={llm_queried}(L={llm_linked},R={llm_rejected})"
        )
        return out

    if not dry_run:
        results["no_link"] = _run_single_cross("no_link", tau_proto=0.85)
        for tau in [0.72, 0.80, 0.85, 0.90]:
            results[f"proto_{tau:.2f}"] = _run_single_cross("proto_only", tau_proto=tau)
        results["oracle_065_085"] = _run_single_cross(
            "oracle_band", tau_proto=0.85, tau_lo=0.65, tau_hi=0.85, tau_fallback=0.85
        )
    results["sug_cross_v2_extract"] = _run_single_cross(
        "sug_cross_llm_v2", tau_proto=0.85, tau_lo=0.65, tau_hi=0.85, tau_fallback=0.85, prompt_mode="extract_compare"
    )
    results["sug_cross_v2_action"] = _run_single_cross(
        "sug_cross_llm_v2", tau_proto=0.85, tau_lo=0.65, tau_hi=0.85, tau_fallback=0.85, prompt_mode="action_first"
    )
    return results


def _build_variant_specs(dev_ids: List[int], held_ids: List[int], warmup_ids: List[int]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        # S2-1A: Dev-only prompt schema calibration (M_02..M_05)
        {"name": "dev_sug_meso_extract", "pool_kind": "sug_meso", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": dev_ids},
        {"name": "dev_sug_meso_action", "pool_kind": "sug_meso", "prompt_mode": "action_first", "controller": "fixed", "block_ids": dev_ids},
        # S2-1B: Strictly causal main protocol & controllers on all held-out blocks
        {"name": "sug_meso_causal_fixed_extract", "pool_kind": "sug_meso", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids, "eval_repair": True},
        {"name": "sug_meso_causal_fixed_action", "pool_kind": "sug_meso", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids, "eval_repair": True},
        {"name": "sug_meso_causal_carryover_extract", "pool_kind": "sug_meso", "prompt_mode": "extract_compare", "controller": "carryover", "block_ids": held_ids},
        {"name": "sug_meso_causal_carryover_action", "pool_kind": "sug_meso", "prompt_mode": "action_first", "controller": "carryover", "block_ids": held_ids},
        {"name": "sug_meso_causal_sliding_action", "pool_kind": "sug_meso", "prompt_mode": "action_first", "controller": "sliding", "block_ids": held_ids, "warmup_ids": warmup_ids},
        # 3-seed causal Random
        {"name": "random_seed42", "pool_kind": "random_s42", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "random_seed100", "pool_kind": "random_s100", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "random_seed200", "pool_kind": "random_s200", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        # Clean 0-error causal ablations
        {"name": "causal_merge_only_action", "pool_kind": "merge_only", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "causal_seam_only_action", "pool_kind": "seam_only", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "causal_fiedler_seam_action", "pool_kind": "fiedler_seam", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "causal_fiedler_seam_extract", "pool_kind": "fiedler_seam", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        # S2-2: Random-Meso (3 seeds) + Eq. (5)/(6) structural ablations under BOTH schemas
        {"name": "random_meso_extract_s42", "pool_kind": "random_meso_s42", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "random_meso_extract_s100", "pool_kind": "random_meso_s100", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "random_meso_extract_s200", "pool_kind": "random_meso_s200", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "random_meso_action_s42", "pool_kind": "random_meso_s42", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "random_meso_action_s100", "pool_kind": "random_meso_s100", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "random_meso_action_s200", "pool_kind": "random_meso_s200", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "eq5_no_leverage_extract", "pool_kind": "eq5_no_leverage", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "eq5_cosine_only_extract", "pool_kind": "eq5_cosine_only", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "eq6_no_conductance_extract", "pool_kind": "eq6_no_conductance", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "eq6_no_non_entity_extract", "pool_kind": "eq6_no_non_entity", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "eq6_raw_tfidf_extract", "pool_kind": "eq6_raw_tfidf", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "eq5_no_leverage_action", "pool_kind": "eq5_no_leverage", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "eq5_cosine_only_action", "pool_kind": "eq5_cosine_only", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "eq6_no_conductance_action", "pool_kind": "eq6_no_conductance", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "eq6_no_non_entity_action", "pool_kind": "eq6_no_non_entity", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "eq6_raw_tfidf_action", "pool_kind": "eq6_raw_tfidf", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        # S2-6: Hyperparameter sensitivity (MERGE/SEAM ratio & max_conductance)
        {"name": "sens_ratio_80_20_action", "pool_kind": "ratio_80_20", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "sens_ratio_50_50_action", "pool_kind": "ratio_50_50", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "sens_cond_035_action", "pool_kind": "cond_035", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "sens_cond_065_action", "pool_kind": "cond_065", "prompt_mode": "action_first", "controller": "fixed", "block_ids": held_ids},
        {"name": "sens_ratio_80_20_extract", "pool_kind": "ratio_80_20", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "sens_ratio_50_50_extract", "pool_kind": "ratio_50_50", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "sens_cond_035_extract", "pool_kind": "cond_035", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
        {"name": "sens_cond_065_extract", "pool_kind": "cond_065", "prompt_mode": "extract_compare", "controller": "fixed", "block_ids": held_ids},
    ]
    return specs


def _compute_stage2a_stats(
    dataset: str,
    ds_res: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    held_variants = {
        k: v["direct"] for k, v in ds_res["within_block"].items() if not k.startswith("dev_")
    }
    # Add 3-seed means cleanly
    for base, seeds in [
        ("random_mean3", ["random_seed42", "random_seed100", "random_seed200"]),
        ("random_meso_extract_mean3", ["random_meso_extract_s42", "random_meso_extract_s100", "random_meso_extract_s200"]),
        ("random_meso_action_mean3", ["random_meso_action_s42", "random_meso_action_s100", "random_meso_action_s200"]),
    ]:
        if all(s in held_variants for s in seeds):
            avg_blk: Dict[str, List[float]] = {}
            for m in ["ami", "macro_b3", "micro_b3", "tail_b3"]:
                mat = np.vstack([held_variants[s][m]["per_block"] for s in seeds])
                avg_blk[m] = mat.mean(axis=0).tolist()
            held_variants[base] = _summarize_arrays(avg_blk)
            ds_res["within_block"][base] = {"direct": held_variants[base]}

    if stage1_json and dataset in stage1_json:
        s1_stats = stage1_json[dataset]["D5_stats"]["summary_stats"]
        for sk in ["init_2dse", "heur_meso_0tok", "cequel_transitive"]:
            if sk in s1_stats:
                held_variants[sk] = s1_stats[sk]

    tests: Dict[str, Any] = {}
    for anchor in ["sug_meso_causal_fixed_action", "sug_meso_causal_fixed_extract"]:
        if anchor not in held_variants:
            continue
        tests[anchor] = {}
        comps = [k for k in held_variants.keys() if k != anchor]
        for metric in ["ami", "macro_b3", "micro_b3", "tail_b3"]:
            elig = [c for c in comps if metric in held_variants[c]]
            pvals = []
            rows = {}
            for c in elig:
                r = _wilcoxon_and_effects(held_variants[anchor][metric]["per_block"], held_variants[c][metric]["per_block"])
                rows[c] = r
                pvals.append(r["p_value"])
            for c, ph in zip(elig, _holm_bonferroni(pvals)):
                rows[c]["p_holm"] = float(ph)
            tests[anchor][metric] = rows
    return tests


def _write_summary_md(report: Dict[str, Any], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# IP&M Stage 2A Causal Qwen/Qwen3-8B Experiment Report\n")
    lines.append(f"- Generated: `{report['generated_at']}` | Model: `{report['model']}`\n")
    lines.append(f"- Ledger Totals across Stage 2A: `{report['ledger_totals']}`\n")

    for ds in ["Event2012", "Event2018"]:
        dr = report[ds]
        lines.append(f"\n---\n\n## {ds}\n")
        lines.append("### S2-1A: Dev-Only Prompt Calibration (`M_02..M_05`, `CausalUniformBucket(10000)`)\n")
        for k in ["dev_sug_meso_extract", "dev_sug_meso_action"]:
            v = dr["within_block"][k]
            d = v["direct"]
            lines.append(
                f"- `{k}`: AMI={d['ami']['mean']:.4f}, Mac-B³={d['macro_b3']['mean']:.4f}, "
                f"Mic-B³={d['micro_b3']['mean']:.4f}, Tail-B³={d['tail_b3']['mean']:.4f} | "
                f"Merge={v['confusion_merge']} | Seam={v['confusion_seam']}\n"
            )
        lines.append(f"- **Selected Dev-Best Schema**: `{dr['dev_selected_prompt']}`\n")

        lines.append("\n### S2-1B / S2-2 / S2-6: Causal Held-Out Results (`10k/blk`, `all_shown` uncontaminated $U_t$)\n")
        lines.append("| Variant | Prompt | Ctrl | Calls (Shown%) | AMI | Mac-B³ | Tail-B³ | Unq ΔMic | UnqNet/blk | Merge P/R/S | Seam P/R/S | Err |\n")
        lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: |\n")
        for k, v in dr["within_block"].items():
            if k.startswith("dev_") or k.endswith("_mean3"):
                continue
            d = v["direct"]
            c = v["counts"]
            cm_m, cm_s = v["confusion_merge"], v["confusion_seam"]
            lines.append(
                f"| `{k}` | {v['prompt_mode']} | {v['controller']} | {c['total_calls']} ({c['all_shown_pct']:.2f}%) | "
                f"{d['ami']['mean']:.4f} | {d['macro_b3']['mean']:.4f} | {d['tail_b3']['mean']:.4f} | "
                f"{d['unq_delta_micro_b3']['mean']:+.4f} | {d['unq_net_posts']['mean']:+.1f} | "
                f"{cm_m['precision']:.2f}/{cm_m['recall']:.2f}/{cm_m['specificity']:.2f} | "
                f"{cm_s['precision']:.2f}/{cm_s['recall']:.2f}/{cm_s['specificity']:.2f} | "
                f"{v['ledger']['errors']} |\n"
            )

        for anchor in ["sug_meso_causal_fixed_action", "sug_meso_causal_fixed_extract"]:
            if anchor in dr["within_block"]:
                lines.append(f"\n### Paired Wilcoxon vs `{anchor}` (Key Comparators)\n")
                st_a = dr["statistical_tests"][anchor]
                for comp in [
                    "init_2dse",
                    "heur_meso_0tok",
                    "cequel_transitive",
                    "random_mean3",
                    "random_meso_action_mean3",
                    "random_meso_extract_mean3",
                    "eq5_no_leverage_action",
                    "eq5_cosine_only_action",
                    "eq6_no_conductance_action",
                    "eq6_no_non_entity_action",
                    "eq6_raw_tfidf_action",
                    "eq5_cosine_only_extract",
                    "eq6_no_conductance_extract",
                    "eq6_raw_tfidf_extract",
                ]:
                    if comp in st_a["ami"]:
                        pa = st_a["ami"][comp]
                        pm = st_a["macro_b3"][comp]
                        lines.append(
                            f"- vs `{comp}`: ΔAMI={pa['mean_diff']:+.4f} (p={pa['p_value']:.2e}, H={pa['p_holm']:.2e}, dz={pa['cohens_dz']:+.2f}) | "
                            f"ΔMac={pm['mean_diff']:+.4f} (p={pm['p_value']:.2e}, H={pm['p_holm']:.2e}, dz={pm['cohens_dz']:+.2f})\n"
                        )

        lines.append("\n### S2-3: Upgraded Cross-Day Tracking (`[0.65, 0.85)`, `tau_fb=0.85`, causal 6k/blk carryover)\n")
        for tag, cres in dr["cross_day"].items():
            lines.append(f"\n#### Base Stream: `{tag}`\n")
            lines.append("| Mode | Global AMI | Global Mac-B³ | Global Mic-B³ | Tail-B³ | Link Stats | Confusion |\n")
            lines.append("| :--- | ---: | ---: | ---: | ---: | :--- | :--- |\n")
            for mk, mv in cres.items():
                lines.append(
                    f"| `{mk}` | {mv['ami']:.4f} | {mv['macro_b3']:.4f} | {mv['micro_b3']:.4f} | {mv['tail_b3']:.4f} | "
                    f"{mv['stats']} | {mv['confusion']} |\n"
                )

    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="IP&M Stage 2A Causal Qwen3-8B Experiment Suite")
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "results" / "ipm_experiments" / "stage2a_qwen3"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = ROOT / "results" / "ipm_audit" / "stage1_diagnostics.json"
    stage1_json = json.loads(stage1_path.read_text(encoding="utf-8")) if stage1_path.exists() else None

    if args.dry_run:
        api_key = ""
        base_url = "dry-run"
        model_name = args.model
    else:
        prov_url, prov_model, api_key = load_provider_key(args.provider)
        base_url = prov_url
        model_name = args.model or prov_model

    t_start = time.time()
    all_report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "provider": args.provider,
    }
    grand = Counter()

    for ds_name in ["Event2012", "Event2018"]:
        embed_model = (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            if ds_name == "Event2018"
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        ds_full = load_socialed_npy(ds_name, sort_chronologically=True)
        blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
        warmup_ids = list(range(0, 6))
        dev_ids = [2, 3, 4, 5]
        held_ids = list(range(6, len(blocks)))

        t_ds_pre = time.time()
        block_cache: Dict[int, Dict[str, Any]] = {}
        for bid in range(len(blocks)):
            bname, bidx = blocks[bid]
            sub_ds = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
            emb = load_or_extract_block_embeddings(ds_name, bid, sub_ds.texts, model_name=embed_model)
            graph = build_message_graph(sub_ds, emb, knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.5)
            labels_init, _ = load_or_solve_2d_se(ds_name, bid, graph.adj, 15, 0.35, 0.5, True)
            summaries = build_cluster_summaries(sub_ds, emb, labels_init)
            block_cache[bid] = {
                "name": bname,
                "ds": sub_ds,
                "emb": emb,
                "graph": graph,
                "labels_init": labels_init,
                "summaries": summaries,
                "raw_tfidf": _build_raw_tfidf(sub_ds) if bid in held_ids else sparse.csr_matrix((len(sub_ds), 1)),
                "pool_lock": threading.Lock(),
                "pool_cache": {},
            }
        print(f"[Preload {ds_name}] {len(blocks)} blocks ready in {time.time()-t_ds_pre:.1f}s", flush=True)

        shared_cache_file, extra_caches = _discover_all_caches(ds_name)
        specs = _build_variant_specs(dev_ids, held_ids, warmup_ids)

        if args.dry_run:
            probe = TokenLedger(out_dir / "_dry_probe", extra_cache_dirs=extra_caches, shared_cache_file=shared_cache_file)
            seen_hashes: Dict[str, Tuple[int, int]] = {}
            w_dry = _simulate_dry_run(specs, block_cache, probe, model_name, seen_hashes)
            c_dry_dev = _run_cross_day_suite(
                ds_name, "dev", {b: block_cache[b]["labels_init"] for b in dev_ids},
                block_cache, [0, 1], dev_ids, out_dir, shared_cache_file, extra_caches,
                model_name, base_url, api_key, True, seen_hashes,
            )
            c_dry_held = _run_cross_day_suite(
                ds_name, "held_init", {b: block_cache[b]["labels_init"] for b in held_ids},
                block_cache, warmup_ids, held_ids, out_dir, shared_cache_file, extra_caches,
                model_name, base_url, api_key, True, seen_hashes,
            )
            c_tok = sum(v["dry_stats"]["tokens"] for d in [c_dry_dev, c_dry_held] for v in d.values())
            c_live = sum(v["dry_stats"]["live"] for d in [c_dry_dev, c_dry_held] for v in d.values())
            est_live_tok = w_dry["unique_live_needed"] * 560 + c_live * 560
            est_cost = est_live_tok * 0.6 / 1_000_000.0
            print(
                f"[DRY-RUN {ds_name}] Within calls={w_dry['total_calls']} (cached/dedup={w_dry['cached_or_dedup']}, "
                f"unique_live={w_dry['unique_live_needed']}), Cross unique_live~{c_live} (scaled x2 streams ~{c_live*2}) | "
                f"Est live CNY ~ {est_cost:.3f}"
            )
            grand["live_needed"] += w_dry["unique_live_needed"] + c_live * 2
            grand["tot_calls"] += w_dry["total_calls"] + c_live * 2
            grand["est_cost_milli_cny"] += int(round((w_dry["unique_live_needed"] + c_live * 2) * 560 * 0.6 / 1000.0))
            continue

        print(f"\n==================== STAGE 2A LIVE: {ds_name} ({model_name}) ====================")
        ds_within: Dict[str, Any] = {}
        saved_streams: Dict[str, Dict[int, np.ndarray]] = {}
        for var in specs:
            res, parts = _run_within_block_variant(
                var, ds_name, block_cache, out_dir, shared_cache_file, extra_caches,
                model_name, base_url, api_key, workers=args.workers,
            )
            ds_within[var["name"]] = res
            if var["name"] in {"dev_sug_meso_extract", "dev_sug_meso_action", "sug_meso_causal_fixed_extract", "sug_meso_causal_fixed_action"}:
                saved_streams[var["name"]] = parts["direct"]
            grand["calls"] += res["ledger"]["total_calls"]
            grand["live"] += res["ledger"]["total_calls"] - res["ledger"]["cached_calls"]
            grand["tokens"] += res["ledger"]["total_tokens"]
            grand["errors"] += res["ledger"]["errors"]

        dev_ext = ds_within["dev_sug_meso_extract"]["direct"]
        dev_act = ds_within["dev_sug_meso_action"]["direct"]
        s_ext = 0.5 * (dev_ext["ami"]["mean"] + dev_ext["macro_b3"]["mean"])
        s_act = 0.5 * (dev_act["ami"]["mean"] + dev_act["macro_b3"]["mean"])
        dev_best = "action_first" if s_act >= s_ext else "extract_compare"

        # S2-3 Cross-Day Tracking: Dev calibration + Held-out on Extract and Action causal streams
        cross_res: Dict[str, Any] = {}
        dev_part = saved_streams["dev_sug_meso_action"] if dev_best == "action_first" else saved_streams["dev_sug_meso_extract"]
        seen_dummy: Dict[str, Tuple[int, int]] = {}
        cross_res["dev_calibrated_stream"] = _run_cross_day_suite(
            ds_name, "dev", dev_part, block_cache, [0, 1], dev_ids,
            out_dir, shared_cache_file, extra_caches, model_name, base_url, api_key, False, seen_dummy,
        )
        cross_res["heldout_extract_stream"] = _run_cross_day_suite(
            ds_name, "held_extract", saved_streams["sug_meso_causal_fixed_extract"], block_cache, warmup_ids, held_ids,
            out_dir, shared_cache_file, extra_caches, model_name, base_url, api_key, False, seen_dummy,
        )
        cross_res["heldout_action_stream"] = _run_cross_day_suite(
            ds_name, "held_action", saved_streams["sug_meso_causal_fixed_action"], block_cache, warmup_ids, held_ids,
            out_dir, shared_cache_file, extra_caches, model_name, base_url, api_key, False, seen_dummy,
        )
        for cdict in cross_res.values():
            for mv in cdict.values():
                if mv.get("ledger"):
                    grand["calls"] += mv["ledger"]["total_calls"]
                    grand["live"] += mv["ledger"]["total_calls"] - mv["ledger"]["cached_calls"]
                    grand["tokens"] += mv["ledger"]["total_tokens"]

        ds_res = {
            "dev_selected_prompt": dev_best,
            "within_block": ds_within,
            "cross_day": cross_res,
        }
        ds_res["statistical_tests"] = _compute_stage2a_stats(ds_name, ds_res, stage1_json)
        all_report[ds_name] = ds_res

    if args.dry_run:
        print(f"[DRY-RUN TOTAL] Unique live calls est={grand['live_needed']}, Total cost est~{grand['est_cost_milli_cny']/1000.0:.3f} CNY")
        return

    all_report["ledger_totals"] = dict(grand)
    # Audit all Stage 2A journals for zero HTTP/parse errors across both within-block & cross-day
    j_errors = 0
    for jp in sorted(out_dir.rglob("journal.jsonl")):
        with jp.open("r", encoding="utf-8") as jf:
            for line in jf:
                if line.strip():
                    st = json.loads(line).get("status", "UNKNOWN")
                    if st not in {"OK", "CACHED_REPLAY"}:
                        j_errors += 1
    all_report["ledger_totals"]["all_journal_errors"] = j_errors
    all_report["elapsed_sec"] = round(time.time() - t_start, 2)
    json_path = out_dir / "stage2a_report.json"
    md_path = out_dir / "stage2a_summary.md"
    json_path.write_text(json.dumps(all_report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_md(all_report, md_path)
    print(f"\n[DONE] Stage 2A finished in {all_report['elapsed_sec']:.1f}s | errors={grand['errors']}")
    print(f"  JSON: {json_path}")
    print(f"  MD  : {md_path}")


if __name__ == "__main__":
    main()
