"""Gate 2 Full 27-Block Held-Out Benchmark Suite (Event2012 M_06..M_21 + Event2018 M_06..M_16).

Strictly Dev-locked hyperparameters (zero tuning on held-out M_06+):
  - Stage `0tok`          : 4 zero-LLM published papers + 4 internal 0-LLM controls + FS/FM structural audit
  - Stage `qwen3_main`    : Pure-Star-Meso (+ HiSEvent Default & Dev-Tuned) vs 6 published LLM papers + 3-seed controls
                            with paired two-sided Wilcoxon signed-rank tests + Holm-Bonferroni correction (EN, FR, Pooled)
  - Stage `qwen3_abl`     : 5 core pillar ablations across all 27 held-out blocks
  - Stage `qwen3_crossday`: Chronological M_00..M_end stream (warmup M_00..M_05, eval M_06+), full ProtoOnly 0.60..0.95
                            + Proto+Attr + SUG-CrossStar [0.65, 0.90) + Oracle Band
  - Stage `qwen25_full`   : Full 2nd-model evaluation (Qwen/Qwen2.5-7B-Instruct): all 6 LLM papers + RandomMeso + Ours + Cross-Day
  - Stage `budget_anon`   : Causal budget curves B in {0, 2k, 5k, 10k, 20k} across all 27 held-out blocks +
                            4-mode Anti-Memorization Audit (raw, pseudonym, extract/blind, end-to-end e2e_pseudonym)
"""

from __future__ import annotations

import os

for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset, load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import MessageGraph, build_message_graph
from core_shared.key_loader import load_provider_key
from core_shared.metrics_audit import full_partition_audit
from core_shared.se_solver import solve_2d_se
from core_shared.token_ledger import CausalUniformBucket, TokenLedger
from trackA_sug_sed.anonymizer import (
    AnonymizationAuditStore,
    QueryAnonymizer,
    _ALLCAPS_RE,
    _AUDIT_STOPWORDS,
    _CAP_FUNCTION_WORDS,
    _HASHTAG_RE,
    _MENTION_RE,
    _MID_CAP_RE,
    _MULTI_CAP_RE,
    _PLACEHOLDER_RE,
    _TOKEN_RE,
    _UPPER_BOILERPLATE,
    _URL_RE,
    _WORD_CH,
    _YEAR_RE,
    _build_accent_pattern,
    normalize_surface_key,
)
from trackA_sug_sed.baselines_llm import (
    run_automerge_only,
    run_cequel,
    run_clusterllm,
    run_fewshot_llm_post,
    run_lhg_budgeted,
    run_lsed_budgeted,
    run_meso_heur_best,
    run_ragsede_budgeted,
    run_random_llm,
)
from trackA_sug_sed.baselines_sed import run_eventx, run_hisevent, run_hypersed, run_sbert_agglom
from trackA_sug_sed.cross_day import evaluate_cross_day_stream
from trackA_sug_sed.embed_cache import encode_texts_hf, load_or_extract_block_embeddings
from trackA_sug_sed.llm_client import BudgetedLLMClient
from trackA_sug_sed.star_meso import run_pure_star_meso

HELDOUT_BLOCKS = {
    "Event2012": list(range(6, 22)),
    "Event2018": list(range(6, 17)),
}
WARMUP_BLOCKS = list(range(0, 6))
STRATIFIED_ANON_BLOCKS = {
    "Event2012": [6, 9, 14, 19],
    "Event2018": [6, 9, 12, 15],
}

# Hyperparameters 100% LOCKED from Gate 1 Dev (M_02..M_05); ZERO held-out tuning.
DEV_LOCKED_0TOK = {
    "Event2012": {
        "eventx_res": 2.0,
        "sbert_dist": 0.60,
        "hisevent_tuned": (0.50, 0.5),
        "hypersed_ar": 0.26,
        "meso_heur": (0.80, 0.99),
    },
    "Event2018": {
        "eventx_res": 1.5,
        "sbert_dist": 0.60,
        "hisevent_tuned": (0.55, 1.0),
        "hypersed_ar": 0.14,
        "meso_heur": (0.80, 0.99),
    },
}

BUDGET_PER_BLK = 10000
EMBED_MODELS = {
    "Event2012": "sentence-transformers/all-MiniLM-L6-v2",
    "Event2018": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

OUT_DIR = ROOT / "results" / "heldout_gate2"
PRED_DIR = OUT_DIR / "preds"


# ---------------------------------------------------------------------------
# Metrics Aggregation, Prediction Persistence, and Wilcoxon + Holm Tests
# ---------------------------------------------------------------------------

def _aggregate_block_metrics(
    block_ids: Sequence[int],
    ds_list: Sequence[SocialDataset],
    pred_list: Sequence[np.ndarray],
    init_list: Optional[Sequence[np.ndarray]] = None,
    shown_list: Optional[Sequence[np.ndarray]] = None,
    stats_list: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    per_block: Dict[str, Any] = {}
    amis, macs, mics, tails, aris = [], [], [], [], []
    unq_nets, unq_dmics = [], []
    leak_q = leak_t = calls = toks = shown = posts_tot = errs = 0
    cpu_s = api_s = wall_s = 0.0

    for i, bid in enumerate(block_ids):
        ds = ds_list[i]
        pred = pred_list[i]
        yb = init_list[i] if init_list is not None else None
        qmask = shown_list[i] if shown_list is not None else None
        aud = full_partition_audit(ds.labels, pred, y_before=yb, queried_mask=qmask)

        ami = aud["ami"] * 100.0
        mac = aud["bcubed_full"]["macro_b3_f1"] * 100.0
        mic = aud["bcubed_full"]["micro_b3_f1"] * 100.0
        tail = aud["stratified"]["tail"]["macro_b3_f1"] * 100.0
        ari = aud["ari"] * 100.0
        amis.append(ami)
        macs.append(mac)
        mics.append(mic)
        tails.append(tail)
        aris.append(ari)
        posts_tot += len(ds)

        b_entry: Dict[str, Any] = {
            "ami": round(ami, 4),
            "macro_b3": round(mac, 4),
            "micro_b3": round(mic, 4),
            "tail_b3": round(tail, 4),
            "ari": round(ari, 4),
            "n_posts": int(len(ds)),
            "n_true_events": int(len(set(ds.labels.tolist()))),
            "n_pred_clusters": int(aud["n_pred_clusters"]),
            "false_merge_pairs": int(aud["pairwise_errors"]["false_merge_pairs"]),
            "false_split_pairs": int(aud["pairwise_errors"]["false_split_pairs"]),
        }
        if "propagation_audit" in aud:
            pa = aud["propagation_audit"]
            unq_nets.append(pa["unqueried_net_improved_posts"])
            unq_dmics.append(pa["unqueried_delta_micro_b3_f1"] * 100.0)
            b_entry["unq_net_posts"] = int(pa["unqueried_net_improved_posts"])
            b_entry["unq_delta_micro_b3"] = round(pa["unqueried_delta_micro_b3_f1"] * 100.0, 2)
        if qmask is not None:
            shown += int(np.sum(qmask))
        if stats_list is not None:
            st = stats_list[i]
            calls += int(st.get("total_calls", 0))
            toks += int(st.get("total_tokens", 0))
            errs += int(st.get("parse_errors", 0))
            leak_q += int(st.get("anon_leak_queries", 0))
            leak_t += int(st.get("anon_leak_terms", 0))
            api_s += float(st.get("llm_api_sec", 0.0))
            w_val = float(st.get("total_wall_sec", st.get("runtime_sec", 0.0)))
            wall_s += w_val
            cpu_s += float(st.get("router_cpu_sec", max(0.0, w_val - float(st.get("llm_api_sec", 0.0)))))
            b_entry["stats"] = st
        per_block[f"M_{bid:02d}"] = b_entry

    summary: Dict[str, Any] = {
        "n_blocks": len(block_ids),
        "n_posts": int(posts_tot),
        "ami": round(float(np.mean(amis)), 2),
        "macro_b3": round(float(np.mean(macs)), 2),
        "micro_b3": round(float(np.mean(mics)), 2),
        "tail_b3": round(float(np.mean(tails)), 2),
        "ari": round(float(np.mean(aris)), 2),
        "total_calls": calls,
        "total_tokens": toks,
        "tok_per_post": round(toks / max(1, posts_tot), 2),
        "shown_posts": shown,
        "shown_pct": round(shown * 100.0 / max(1, posts_tot), 2),
        "parse_errors": errs,
        "anon_leak_queries": leak_q,
        "anon_leak_terms": leak_t,
        "router_cpu_sec": round(cpu_s, 3),
        "llm_api_sec": round(api_s, 3),
        "total_wall_sec": round(wall_s, 3),
        "per_block": per_block,
    }
    if unq_nets:
        summary["unq_net_mean"] = round(float(np.mean(unq_nets)), 1)
        summary["unq_delta_micro_mean"] = round(float(np.mean(unq_dmics)), 2)
    return summary


def _average_seed_summaries(runs: Sequence[Dict[str, Any]], block_ids: Sequence[int]) -> Dict[str, Any]:
    """Average per-block metrics across multiple random seeds for paired Wilcoxon testing."""
    per_block: Dict[str, Any] = {}
    for bid in block_ids:
        bk = f"M_{bid:02d}"
        b0 = runs[0]["per_block"][bk]
        per_block[bk] = {
            "ami": round(float(np.mean([r["per_block"][bk]["ami"] for r in runs])), 4),
            "macro_b3": round(float(np.mean([r["per_block"][bk]["macro_b3"] for r in runs])), 4),
            "micro_b3": round(float(np.mean([r["per_block"][bk]["micro_b3"] for r in runs])), 4),
            "tail_b3": round(float(np.mean([r["per_block"][bk]["tail_b3"] for r in runs])), 4),
            "ari": round(float(np.mean([r["per_block"][bk]["ari"] for r in runs])), 4),
            "n_posts": b0["n_posts"],
            "n_true_events": b0["n_true_events"],
        }
    out = {
        k: round(float(np.mean([r.get(k, 0.0) for r in runs])), 2)
        for k in [
            "ami",
            "macro_b3",
            "micro_b3",
            "tail_b3",
            "ari",
            "total_calls",
            "total_tokens",
            "tok_per_post",
            "shown_pct",
            "parse_errors",
            "router_cpu_sec",
            "llm_api_sec",
            "total_wall_sec",
            "unq_net_mean",
        ]
    }
    out["n_blocks"] = len(block_ids)
    out["n_posts"] = runs[0]["n_posts"]
    out["per_block"] = per_block
    return out


def _build_live_latency_index(ledger_roots: Sequence[Path]) -> Dict[str, float]:
    """Scan journal.jsonl files once to map request_hash -> recorded live API latency_sec."""
    lat_by_hash: Dict[str, float] = {}
    for root_dir in ledger_roots:
        if not root_dir.exists():
            continue
        for jp in root_dir.rglob("journal.jsonl"):
            try:
                text = jp.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("from_cache"):
                    continue
                lat = float(obj.get("latency_sec", 0.0) or 0.0)
                rh = obj.get("request_hash")
                if rh and lat > 0.0 and rh not in lat_by_hash:
                    lat_by_hash[rh] = lat
    return lat_by_hash


def _attach_recorded_live_api_sec(
    sub_journal: Path,
    agg: Dict[str, Any],
    live_lat_index: Dict[str, float],
) -> None:
    """Recover true cumulative API latency for cached/mixed runs without touching router_cpu_sec."""
    if not sub_journal.exists() or int(agg.get("total_calls", 0)) <= 0:
        return
    by_bid: Dict[int, List[Dict[str, Any]]] = {}
    for line in sub_journal.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        bid = int(obj.get("block_id", -1))
        by_bid.setdefault(bid, []).append(obj)

    total_api = 0.0
    per_block = agg.get("per_block", {})
    for bk, bv in per_block.items():
        if not bk.startswith("M_"):
            continue
        bid = int(bk[2:])
        st = bv.get("stats", {})
        n_calls = int(st.get("total_calls", 0))
        if n_calls <= 0:
            continue
        recs = by_bid.get(bid, [])
        tail = recs[-n_calls:] if len(recs) >= n_calls else recs
        blk_api = 0.0
        for rec in tail:
            lat = float(rec.get("latency_sec", 0.0) or 0.0)
            if (not rec.get("from_cache")) and lat > 0.0:
                blk_api += lat
            else:
                blk_api += float(live_lat_index.get(rec.get("request_hash", ""), 0.0))
        if blk_api > 0.0:
            st["llm_api_sec"] = round(blk_api, 4)
            total_api += blk_api
        else:
            total_api += float(st.get("llm_api_sec", 0.0))

    if total_api > 0.0:
        agg["llm_api_sec"] = round(total_api, 3)


def _holm_bonferroni(p_vals: Sequence[float]) -> List[float]:
    m = len(p_vals)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_vals), key=lambda x: x[1])
    adj = [1.0] * m
    prev_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, float(p) * (m - rank))
        val = max(prev_max, val)
        prev_max = val
        adj[orig_idx] = val
    return adj


def compute_wilcoxon_holm(
    ours_entries: Sequence[Tuple[str, Dict[str, Any], Sequence[int]]],
    baseline_map: Dict[str, Sequence[Tuple[str, Dict[str, Any], Sequence[int]]]],
) -> Dict[str, Dict[str, Any]]:
    """Compute paired two-sided Wilcoxon + Holm-Bonferroni across baselines."""
    metrics = ("ami", "macro_b3", "micro_b3", "tail_b3")
    raw_table: Dict[str, Dict[str, Any]] = {}
    base_names = list(baseline_map.keys())
    for b_name in base_names:
        raw_table[b_name] = {}

    for m in metrics:
        p_raws: List[float] = []
        for b_name in base_names:
            b_entries = baseline_map[b_name]
            ours_vec: List[float] = []
            base_vec: List[float] = []
            for (_ds_o, r_o, bids_o), (_ds_b, r_b, bids_b) in zip(ours_entries, b_entries):
                for bid in bids_o:
                    bk = f"M_{bid:02d}"
                    ours_vec.append(float(r_o["per_block"][bk][m]))
                    base_vec.append(float(r_b["per_block"][bk][m]))
            diffs = np.asarray(ours_vec) - np.asarray(base_vec)
            nz = diffs[np.abs(diffs) > 1e-9]
            if len(nz) == 0:
                p_val = 1.0
            else:
                no_ties = len(np.unique(np.round(np.abs(nz), 9))) == len(nz)
                method = "exact" if (len(nz) <= 50 and no_ties) else "auto"
                res = wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method=method)
                p_val = float(res.pvalue)
            raw_table[b_name][m] = {
                "mean_diff": round(float(np.mean(diffs)), 2),
                "n_pairs": len(diffs),
                "n_wins": int(np.sum(diffs > 1e-9)),
                "n_ties": int(np.sum(np.abs(diffs) <= 1e-9)),
                "n_losses": int(np.sum(diffs < -1e-9)),
                "p_raw": float(f"{p_val:.6g}"),
            }
            p_raws.append(p_val)
        p_holms = _holm_bonferroni(p_raws)
        for b_name, ph in zip(base_names, p_holms):
            raw_table[b_name][m]["p_holm"] = float(f"{float(ph):.6g}")
    return raw_table


def _save_preds(path: Path, block_ids: Sequence[int], preds: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"M_{b:02d}": preds[i] for i, b in enumerate(block_ids)})


def _load_preds(path: Path, block_ids: Sequence[int]) -> Optional[List[np.ndarray]]:
    if not path.exists():
        return None
    z = np.load(path)
    out = [z[f"M_{b:02d}"] for b in block_ids if f"M_{b:02d}" in z]
    return out if len(out) == len(block_ids) else None


# ---------------------------------------------------------------------------
# Held-Out Context Loader & Concurrency Wrapper
# ---------------------------------------------------------------------------

def _prepare_context(dataset_name: str, block_ids: Sequence[int], need_tuned_diff: bool = True) -> Dict[str, Any]:
    ds_full = load_socialed_npy(dataset_name, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
    embed_model = EMBED_MODELS[dataset_name]
    tuned_tau, tuned_alpha = DEV_LOCKED_0TOK[dataset_name]["hisevent_tuned"]

    ds_map: Dict[int, SocialDataset] = {}
    emb_map: Dict[int, np.ndarray] = {}
    graph_pure_map: Dict[int, MessageGraph] = {}
    init_pure_map: Dict[int, np.ndarray] = {}
    stats_pure_map: Dict[int, Dict[str, Any]] = {}
    graph_def_map: Dict[int, MessageGraph] = {}
    init_def_map: Dict[int, np.ndarray] = {}
    stats_def_map: Dict[int, Dict[str, Any]] = {}
    graph_tun_map: Dict[int, MessageGraph] = {}
    init_tun_map: Dict[int, np.ndarray] = {}
    stats_tun_map: Dict[int, Dict[str, Any]] = {}

    def _hisev_stat(st: Dict[str, Any]) -> Dict[str, Any]:
        sec = float(st.get("e2e_local_sec", st.get("se_solve_sec", st.get("runtime_sec", 0.0))))
        return {"runtime_sec": round(sec, 3), "router_cpu_sec": round(sec, 3), "total_wall_sec": round(sec, 3)}

    print(f"\n[Prepare Held-Out Context] {dataset_name} blocks={list(block_ids)} ({embed_model})", flush=True)
    for bid in block_ids:
        _bname, bidx = blocks[bid]
        sd = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        em = load_or_extract_block_embeddings(dataset_name, bid, sd.texts, model_name=embed_model)
        g_pure = build_message_graph(
            sd, em, knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5, compute_non_entity_tfidf=False
        )
        l_pure, st_pure = run_hisevent(
            dataset_name, bid, sd, em, knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5, embed_model=embed_model
        )
        g_def = build_message_graph(
            sd, em, knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.5, compute_non_entity_tfidf=False
        )
        l_def, st_def = run_hisevent(
            dataset_name, bid, sd, em, knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.5, embed_model=embed_model
        )

        if need_tuned_diff and (abs(tuned_tau - 0.50) > 1e-6 or abs(tuned_alpha - 0.5) > 1e-6):
            g_tun = build_message_graph(
                sd, em, knn_k=15, knn_sim_threshold=tuned_tau, attr_alpha=tuned_alpha, compute_non_entity_tfidf=False
            )
            l_tun, st_tun = run_hisevent(
                dataset_name, bid, sd, em, knn_k=15, knn_sim_threshold=tuned_tau, attr_alpha=tuned_alpha, embed_model=embed_model
            )
        else:
            g_tun, l_tun, st_tun = g_pure, l_pure, st_pure

        ds_map[bid] = sd
        emb_map[bid] = em
        graph_pure_map[bid] = g_pure
        init_pure_map[bid] = l_pure
        stats_pure_map[bid] = _hisev_stat(st_pure)
        graph_def_map[bid] = g_def
        init_def_map[bid] = l_def
        stats_def_map[bid] = _hisev_stat(st_def)
        graph_tun_map[bid] = g_tun
        init_tun_map[bid] = l_tun
        stats_tun_map[bid] = _hisev_stat(st_tun)

    return {
        "dataset": dataset_name,
        "embed_model": embed_model,
        "ds": ds_map,
        "emb": emb_map,
        "graph_pure": graph_pure_map,
        "init_pure": init_pure_map,
        "stats_pure": stats_pure_map,
        "graph_def": graph_def_map,
        "init_def": init_def_map,
        "stats_def": stats_def_map,
        "graph_tuned": graph_tun_map,
        "init_tuned": init_tun_map,
        "stats_tuned": stats_tun_map,
    }


def _make_client(provider: str, model: str, log_dir: Path, shared_cache_file: Path) -> BudgetedLLMClient:
    base_url, _def_m, api_key = load_provider_key(provider)
    ledger = TokenLedger(log_dir=log_dir, shared_cache_file=shared_cache_file)
    return BudgetedLLMClient(
        model=model,
        ledger=ledger,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
        max_tokens=384,
        timeout_sec=45.0,
    )


def _eval_llm_parallel(
    tag: str,
    bids: Sequence[int],
    dss: Sequence[SocialDataset],
    init_list: Sequence[np.ndarray],
    fn_per_block,
    budget_per_blk: int = BUDGET_PER_BLK,
    max_workers: int = 8,
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    def _one(i: int, bid: int):
        bk = CausalUniformBucket(tokens_per_block=budget_per_blk, allow_carryover=False) if budget_per_blk > 0 else None
        if bk is not None:
            bk.start_block()
        return fn_per_block(i, bid, bk)

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(bids)))) as pool:
        futs = [pool.submit(_one, i, bid) for i, bid in enumerate(bids)]
        outs = [f.result() for f in futs]

    preds = [o[0] for o in outs]
    showns = [o[1] for o in outs]
    sts = [o[2] for o in outs]
    agg = _aggregate_block_metrics(bids, dss, preds, init_list=init_list, shown_list=showns, stats_list=sts)
    print(
        f"  [{tag:30s}] calls={agg['total_calls']:4d} tok={agg['total_tokens']:7d} ({agg['tok_per_post']:.2f}/p) "
        f"show={agg['shown_pct']:4.2f}% err={agg['parse_errors']} | "
        f"AMI={agg['ami']:5.2f} Mac={agg['macro_b3']:5.2f} Mic={agg['micro_b3']:5.2f} Tail={agg['tail_b3']:5.2f} Unq={agg.get('unq_net_mean', 0):+6.1f}",
        flush=True,
    )
    return agg, preds


# ---------------------------------------------------------------------------
# Stage 1: Zero-LLM Published Baselines + Internal 0-Tok Controls (27 Blocks)
# ---------------------------------------------------------------------------

def run_stage_0tok(contexts: Dict[str, Dict[str, Any]], resume: bool = True) -> Dict[str, Any]:
    out_path = OUT_DIR / "stage_0tok.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if (resume and out_path.exists()) else {}

    for ds_name in ["Event2012", "Event2018"]:
        ctx = contexts[ds_name]
        bids = HELDOUT_BLOCKS[ds_name]
        dss = [ctx["ds"][b] for b in bids]
        embs = [ctx["emb"][b] for b in bids]
        graphs_p = [ctx["graph_pure"][b] for b in bids]
        inits_p = [ctx["init_pure"][b] for b in bids]
        graphs_d = [ctx["graph_def"][b] for b in bids]
        inits_d = [ctx["init_def"][b] for b in bids]
        inits_t = [ctx["init_tuned"][b] for b in bids]
        cfg = DEV_LOCKED_0TOK[ds_name]

        ds_res: Dict[str, Any] = dict(results.get(ds_name, {})) if resume else {}
        results[ds_name] = ds_res
        print(f"\n=== [Gate 2 Stage 0-Tok] {ds_name} Held-Out (M_{bids[0]:02d}..M_{bids[-1]:02d}, n={len(bids)}) ===", flush=True)

        if "eventx_tkdd20" not in ds_res:
            preds, sts = zip(*[run_eventx(ds, resolution=cfg["eventx_res"]) for ds in dss])
            ds_res["eventx_tkdd20"] = _aggregate_block_metrics(bids, dss, preds, stats_list=sts)
            ds_res["eventx_tkdd20"]["cfg"] = {"resolution": cfg["eventx_res"]}

        if "sbert_emnlp19" not in ds_res:
            preds, sts = zip(*[run_sbert_agglom(em, distance_threshold=cfg["sbert_dist"]) for em in embs])
            ds_res["sbert_emnlp19"] = _aggregate_block_metrics(bids, dss, preds, stats_list=sts)
            ds_res["sbert_emnlp19"]["cfg"] = {"distance_threshold": cfg["sbert_dist"]}

        if "hisevent_default" not in ds_res:
            ds_res["hisevent_default"] = _aggregate_block_metrics(
                bids, dss, inits_d, stats_list=[ctx["stats_def"][b] for b in bids]
            )

        if "hisevent_dev_tuned" not in ds_res:
            ds_res["hisevent_dev_tuned"] = _aggregate_block_metrics(
                bids, dss, inits_t, stats_list=[ctx["stats_tuned"][b] for b in bids]
            )
            ds_res["hisevent_dev_tuned"]["cfg"] = {"tau_sim": cfg["hisevent_tuned"][0], "attr_alpha": cfg["hisevent_tuned"][1]}

        if "hypersed_aaai25" not in ds_res:
            preds, sts = zip(*[run_hypersed(ds, em, graph=g, anchor_ratio=cfg["hypersed_ar"], epochs=60) for ds, em, g in zip(dss, embs, graphs_p)])
            ds_res["hypersed_aaai25"] = _aggregate_block_metrics(bids, dss, preds, stats_list=sts)
            ds_res["hypersed_aaai25"]["cfg"] = {"anchor_ratio": cfg["hypersed_ar"]}

        if "ctrl_pure_2dse_0tok" not in ds_res:
            ds_res["ctrl_pure_2dse_0tok"] = _aggregate_block_metrics(
                bids, dss, inits_p, stats_list=[ctx["stats_pure"][b] for b in bids]
            )

        if "ctrl_automerge_only" not in ds_res:
            preds, sts = zip(*[run_automerge_only(ds, em, lp, tau_auto=0.85, guard_sim=0.40) for ds, em, lp in zip(dss, embs, inits_p)])
            ds_res["ctrl_automerge_only"] = _aggregate_block_metrics(bids, dss, preds, init_list=inits_p, stats_list=sts)

        if "ctrl_meso_heur_pure" not in ds_res:
            tm, ss_th = cfg["meso_heur"]
            preds, sts = zip(*[run_meso_heur_best(ds, em, g, lp, tau_merge=tm, min_seam_sep=ss_th) for ds, em, g, lp in zip(dss, embs, graphs_p, inits_p)])
            ds_res["ctrl_meso_heur_pure"] = _aggregate_block_metrics(bids, dss, preds, init_list=inits_p, stats_list=sts)
            ds_res["ctrl_meso_heur_pure"]["cfg"] = {"tau_merge": tm, "min_seam_sep": ss_th}

        if "ctrl_meso_heur_def" not in ds_res:
            tm, ss_th = cfg["meso_heur"]
            preds, sts = zip(*[run_meso_heur_best(ds, em, g, ld, tau_merge=tm, min_seam_sep=ss_th) for ds, em, g, ld in zip(dss, embs, graphs_d, inits_d)])
            ds_res["ctrl_meso_heur_def"] = _aggregate_block_metrics(bids, dss, preds, init_list=inits_d, stats_list=sts)

        # Structural FS/FM ratio audit across held-out blocks
        fm_p = sum(b["false_merge_pairs"] for b in ds_res["ctrl_pure_2dse_0tok"]["per_block"].values())
        fs_p = sum(b["false_split_pairs"] for b in ds_res["ctrl_pure_2dse_0tok"]["per_block"].values())
        fm_d = sum(b["false_merge_pairs"] for b in ds_res["hisevent_default"]["per_block"].values())
        fs_d = sum(b["false_split_pairs"] for b in ds_res["hisevent_default"]["per_block"].values())
        ds_res["structural_fs_fm_audit"] = {
            "pure_2dse_fm": fm_p,
            "pure_2dse_fs": fs_p,
            "pure_fs_over_fm": round(fs_p / max(1, fm_p), 2),
            "hisevent_def_fm": fm_d,
            "hisevent_def_fs": fs_d,
            "def_fs_over_fm": round(fs_d / max(1, fm_d), 2),
        }
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        for k in ["eventx_tkdd20", "sbert_emnlp19", "hisevent_default", "hisevent_dev_tuned", "hypersed_aaai25", "ctrl_pure_2dse_0tok", "ctrl_automerge_only", "ctrl_meso_heur_pure"]:
            r = ds_res[k]
            print(f"  [{k:22s}] AMI={r['ami']:5.2f} Mac={r['macro_b3']:5.2f} Mic={r['micro_b3']:5.2f} Tail={r['tail_b3']:5.2f}", flush=True)
        print(f"  [FS/FM Audit] Pure FS/FM={ds_res['structural_fs_fm_audit']['pure_fs_over_fm']}x | Def FS/FM={ds_res['structural_fs_fm_audit']['def_fs_over_fm']}x", flush=True)

    return results


# ---------------------------------------------------------------------------
# Stage 2: Qwen3-8B Held-Out Main + Plug-and-Play + 6 LLM Papers + Controls + Stats
# ---------------------------------------------------------------------------

def run_stage_qwen3_main(contexts: Dict[str, Dict[str, Any]], max_workers: int = 8, resume: bool = True) -> Dict[str, Any]:
    model_name = "Qwen/Qwen3-8B"
    shared_cache = OUT_DIR / "ledgers" / "qwen3_8b_shared_responses.jsonl"
    out_path = OUT_DIR / "stage_qwen3_main.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if (resume and out_path.exists()) else {"model": model_name, "provider": "siliconflow"}
    live_lat_index = _build_live_latency_index([OUT_DIR / "ledgers", ROOT / "results" / "dev_gate1" / "ledgers"])

    def _ckpt() -> None:
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for ds_name in ["Event2012", "Event2018"]:
        ctx = contexts[ds_name]
        bids = HELDOUT_BLOCKS[ds_name]
        dss = [ctx["ds"][b] for b in bids]
        embs = [ctx["emb"][b] for b in bids]
        graphs_p = [ctx["graph_pure"][b] for b in bids]
        inits_p = [ctx["init_pure"][b] for b in bids]
        graphs_d = [ctx["graph_def"][b] for b in bids]
        inits_d = [ctx["init_def"][b] for b in bids]
        graphs_t = [ctx["graph_tuned"][b] for b in bids]
        inits_t = [ctx["init_tuned"][b] for b in bids]
        embed_model = ctx["embed_model"]

        def _client(sub: str) -> BudgetedLLMClient:
            return _make_client("siliconflow", model_name, OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / sub, shared_cache)

        def _sub_journal(sub: str) -> Path:
            return OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / sub / "journal.jsonl"

        ds_out: Dict[str, Any] = dict(results.get(ds_name, {})) if resume else {}
        results[ds_name] = ds_out
        print(f"\n=== [Gate 2 Qwen3-8B Main] {ds_name} Held-Out (M_{bids[0]:02d}..M_{bids[-1]:02d}, n={len(bids)}) ===", flush=True)

        pred_path = PRED_DIR / f"qwen3_{ds_name}.npz"
        if not (resume and "our_pure_star_meso" in ds_out and pred_path.exists()):
            cl_main = _client("pure_star_meso")
            agg_main, main_preds = _eval_llm_parallel(
                "OUR.Pure-Star-Meso(Pure)", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_main: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag="pure_star_meso", graph=graphs_p[i], labels_init=inits_p[i])
                ),
                max_workers=max_workers,
            )
            _attach_recorded_live_api_sec(_sub_journal("pure_star_meso"), agg_main, live_lat_index)
            ds_out["our_pure_star_meso"] = agg_main
            _save_preds(pred_path, bids, main_preds)
            cl_path = pred_path.with_suffix(".cl.json")
            cl_payload = {
                str(b): agg_main["per_block"].get(f"M_{b:02d}", {}).get("stats", {}).get("intra_cannot_link_pairs", [])
                for b in bids
            }
            cl_path.write_text(json.dumps(cl_payload, indent=2), encoding="utf-8")
            _ckpt()

        if not (resume and "our_star_meso_on_hisevent_def" in ds_out):
            cl_def = _client("star_on_def")
            agg_def, _ = _eval_llm_parallel(
                "OUR.Star-Meso(HiSEventDef)", bids, dss, inits_d,
                lambda i, b, bk, _c=cl_def: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag="star_meso_def", use_purity_base=False, graph=graphs_d[i], labels_init=inits_d[i])
                ),
                max_workers=max_workers,
            )
            _attach_recorded_live_api_sec(_sub_journal("star_on_def"), agg_def, live_lat_index)
            ds_out["our_star_meso_on_hisevent_def"] = agg_def
            _ckpt()

        if not (resume and "our_star_meso_on_hisevent_tuned" in ds_out):
            tuned_tau, tuned_alpha = DEV_LOCKED_0TOK[ds_name]["hisevent_tuned"]
            if abs(tuned_tau - 0.50) <= 1e-6 and abs(tuned_alpha - 0.5) <= 1e-6:
                ds_out["our_star_meso_on_hisevent_tuned"] = dict(ds_out["our_pure_star_meso"])
                ds_out["our_star_meso_on_hisevent_tuned"]["note"] = "Dev-Tuned (0.50, 0.5) identical to Pure Base"
            else:
                cl_tun = _client("star_on_tuned")
                agg_tun, _ = _eval_llm_parallel(
                    "OUR.Star-Meso(DevTuned)", bids, dss, inits_t,
                    lambda i, b, bk, _c=cl_tun: (lambda r: (r[0], r[2], r[3]))(
                        run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag="star_meso_tuned", graph=graphs_t[i], labels_init=inits_t[i])
                    ),
                    max_workers=max_workers,
                )
                _attach_recorded_live_api_sec(_sub_journal("star_on_tuned"), agg_tun, live_lat_index)
                ds_out["our_star_meso_on_hisevent_tuned"] = agg_tun
            _ckpt()

        if not (resume and "clusterllm_emnlp23" in ds_out):
            cl_c = _client("clusterllm")
            sbert_inits = [
                run_sbert_agglom(em, distance_threshold=DEV_LOCKED_0TOK[ds_name]["sbert_dist"])[0]
                for em in embs
            ]
            agg_cllm, _ = _eval_llm_parallel(
                "5.ClusterLLM-NoFT(23)", bids, dss, sbert_inits,
                lambda i, b, bk, _c=cl_c, _si=sbert_inits: run_clusterllm(b, dss[i], embs[i], _si[i], _c, bk),
                max_workers=max_workers,
            )
            _attach_recorded_live_api_sec(_sub_journal("clusterllm"), agg_cllm, live_lat_index)
            ds_out["clusterllm_emnlp23"] = agg_cllm
            _ckpt()

        if not (resume and "fewshot_post_tacl24" in ds_out):
            cl_f = _client("fewshot")
            agg_fs, _ = _eval_llm_parallel(
                "6.FewShotPost(TACL24)", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_f: run_fewshot_llm_post(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk),
                max_workers=max_workers,
            )
            _attach_recorded_live_api_sec(_sub_journal("fewshot"), agg_fs, live_lat_index)
            ds_out["fewshot_post_tacl24"] = agg_fs
            _ckpt()

        if not (resume and "cequel_edge_25" in ds_out and "cequel_trans_25" in ds_out):
            cl_ceq = _client("cequel")
            with ThreadPoolExecutor(max_workers=min(max_workers, len(bids))) as pool:
                futs = [
                    pool.submit(
                        lambda i=i, bid=bid: (
                            lambda bk: (bk.start_block(), run_cequel(bid, dss[i], embs[i], graphs_p[i], inits_p[i], cl_ceq, bk))[1]
                        )(CausalUniformBucket(tokens_per_block=BUDGET_PER_BLK, allow_carryover=False))
                    )
                    for i, bid in enumerate(bids)
                ]
                ceq_outs = [f.result() for f in futs]
            c_e = [o[0] for o in ceq_outs]
            c_t = [o[1] for o in ceq_outs]
            c_s = [o[2] for o in ceq_outs]
            c_st = [o[3] for o in ceq_outs]
            agg_ce = _aggregate_block_metrics(bids, dss, c_e, init_list=inits_p, shown_list=c_s, stats_list=c_st)
            agg_ct = _aggregate_block_metrics(bids, dss, c_t, init_list=inits_p, shown_list=c_s, stats_list=c_st)
            _attach_recorded_live_api_sec(_sub_journal("cequel"), agg_ce, live_lat_index)
            _attach_recorded_live_api_sec(_sub_journal("cequel"), agg_ct, live_lat_index)
            ds_out["cequel_edge_25"] = agg_ce
            ds_out["cequel_trans_25"] = agg_ct
            for tc, ac in [("7a.Cequel-Edge(25)", ds_out["cequel_edge_25"]), ("7b.Cequel-Trans(25)", ds_out["cequel_trans_25"])]:
                print(f"  [{tc:30s}] calls={ac['total_calls']:4d} tok={ac['total_tokens']:7d} AMI={ac['ami']:5.2f} Mac={ac['macro_b3']:5.2f} Tail={ac['tail_b3']:5.2f}", flush=True)
            _ckpt()

        if not (resume and "lsed_acl25" in ds_out):
            cl_l = _client("lsed")
            agg_lsed, _ = _eval_llm_parallel(
                "8.LSED-Bud(ACL25)", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_l: run_lsed_budgeted(b, dss[i], embs[i], inits_p[i], embed_model, _c, bk, knn_sim_threshold=0.50),
                max_workers=max_workers,
            )
            _attach_recorded_live_api_sec(_sub_journal("lsed"), agg_lsed, live_lat_index)
            ds_out["lsed_acl25"] = agg_lsed
            _ckpt()

        if not (resume and "ragsede_www26" in ds_out):
            cl_r = _client("ragsede")
            agg_rag, _ = _eval_llm_parallel(
                "9.RagSEDE-Bud(WWW26)", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_r: run_ragsede_budgeted(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk),
                max_workers=max_workers,
            )
            _attach_recorded_live_api_sec(_sub_journal("ragsede"), agg_rag, live_lat_index)
            ds_out["ragsede_www26"] = agg_rag
            _ckpt()

        if not (resume and "lhg_www26" in ds_out):
            cl_h = _client("lhg")
            agg_lhg, _ = _eval_llm_parallel(
                "10.LHG-Bud(WWW26)", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_h: run_lhg_budgeted(b, dss[i], embs[i], graphs_p[i], inits_p[i], embed_model, _c, bk),
                max_workers=max_workers,
            )
            _attach_recorded_live_api_sec(_sub_journal("lhg"), agg_lhg, live_lat_index)
            ds_out["lhg_www26"] = agg_lhg
            _ckpt()

        if not (resume and "ctrl_random_mean3" in ds_out):
            rand_runs = []
            for sd_i in [42, 100, 200]:
                cl_rd = _client(f"rand_s{sd_i}")
                agg_rd, _ = _eval_llm_parallel(
                    f"Ctrl.Random(s={sd_i})", bids, dss, inits_p,
                    lambda i, b, bk, _s=sd_i, _c=cl_rd: run_random_llm(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk, seed=_s),
                    max_workers=max_workers,
                )
                _attach_recorded_live_api_sec(_sub_journal(f"rand_s{sd_i}"), agg_rd, live_lat_index)
                rand_runs.append(agg_rd)
            ds_out["ctrl_random_mean3"] = _average_seed_summaries(rand_runs, bids)
            _ckpt()

        if not (resume and "ctrl_random_meso_mean3" in ds_out and "ctrl_random_meso_s42" in ds_out):
            rm_runs = []
            for sd_i in [42, 100, 200]:
                cl_rm = _client(f"rmeso_s{sd_i}")
                agg_rm, _ = _eval_llm_parallel(
                    f"Ctrl.RandomMeso(s={sd_i})", bids, dss, inits_p,
                    lambda i, b, bk, _s=sd_i, _c=cl_rm: (lambda r: (r[0], r[2], r[3]))(
                        run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag="random_meso", random_order=True, seed=_s, graph=graphs_p[i], labels_init=inits_p[i])
                    ),
                    max_workers=max_workers,
                )
                _attach_recorded_live_api_sec(_sub_journal(f"rmeso_s{sd_i}"), agg_rm, live_lat_index)
                rm_runs.append(agg_rm)
            ds_out["ctrl_random_meso_s42"] = rm_runs[0]
            ds_out["ctrl_random_meso_mean3"] = _average_seed_summaries(rm_runs, bids)
            _ckpt()

    # Compute paired two-sided Wilcoxon + Holm-Bonferroni tests (EN n=16, FR n=11, Pooled n=27)
    s0 = json.loads((OUT_DIR / "stage_0tok.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_0tok.json").exists() else {}
    stat_family = [
        ("EventX (TKDD'20)", "0tok", "eventx_tkdd20"),
        ("SBERT-Agglom (EMNLP'19)", "0tok", "sbert_emnlp19"),
        ("HiSEvent Default (AAAI'24)", "0tok", "hisevent_default"),
        ("HiSEvent Dev-Tuned (AAAI'24)", "0tok", "hisevent_dev_tuned"),
        ("HyperSED (AAAI'25)", "0tok", "hypersed_aaai25"),
        ("Meso-Heur-Best (0-Tok)", "0tok", "ctrl_meso_heur_pure"),
        ("ClusterLLM-NoFT (EMNLP'23)", "q3", "clusterllm_emnlp23"),
        ("Few-Shot LLM-Post (TACL'24)", "q3", "fewshot_post_tacl24"),
        ("Cequel-Edge (arXiv'25)", "q3", "cequel_edge_25"),
        ("Cequel-Trans (arXiv'25)", "q3", "cequel_trans_25"),
        ("LSED-Bud (ACL'25)", "q3", "lsed_acl25"),
        ("RagSEDE-Bud (WWW'26)", "q3", "ragsede_www26"),
        ("LHG-Bud (WWW'26)", "q3", "lhg_www26"),
        ("Random-Meso (3-seed)", "q3", "ctrl_random_meso_mean3"),
    ]
    wilcoxon_out: Dict[str, Any] = {}
    for scope_name, ds_list_s in [("Event2012", ["Event2012"]), ("Event2018", ["Event2018"]), ("Pooled27", ["Event2012", "Event2018"])]:
        ours_ent = [(d, results[d]["our_pure_star_meso"], HELDOUT_BLOCKS[d]) for d in ds_list_s if d in results and "our_pure_star_meso" in results[d]]
        base_map: Dict[str, List[Tuple[str, Dict[str, Any], Sequence[int]]]] = {}
        for label, src, key in stat_family:
            ents = []
            ok = True
            for d in ds_list_s:
                src_obj = s0 if src == "0tok" else results
                if d not in src_obj or key not in src_obj[d]:
                    ok = False
                    break
                ents.append((d, src_obj[d][key], HELDOUT_BLOCKS[d]))
            if ok and len(ents) == len(ours_ent):
                base_map[label] = ents
        if ours_ent and base_map:
            wilcoxon_out[scope_name] = compute_wilcoxon_holm(ours_ent, base_map)
    results["wilcoxon_holm"] = wilcoxon_out
    _ckpt()
    return results


# ---------------------------------------------------------------------------
# Stage 3: Qwen3-8B Five-Pillar Held-Out Ablations (27 Blocks)
# ---------------------------------------------------------------------------

def run_stage_qwen3_abl(contexts: Dict[str, Dict[str, Any]], max_workers: int = 8, resume: bool = True) -> Dict[str, Any]:
    model_name = "Qwen/Qwen3-8B"
    shared_cache = OUT_DIR / "ledgers" / "qwen3_8b_shared_responses.jsonl"
    out_path = OUT_DIR / "stage_qwen3_abl.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if (resume and out_path.exists()) else {"model": model_name}
    live_lat_index = _build_live_latency_index([OUT_DIR / "ledgers", ROOT / "results" / "dev_gate1" / "ledgers"])

    abl_specs = [
        ("abl_no_automerge", dict(use_auto_merge=False)),
        ("abl_no_bayes_band", dict(use_bayes_band=False)),
        ("abl_no_star_packing", dict(use_star_packing=False)),
        ("abl_no_guarded_uf", dict(use_guarded_uf=False)),
        ("abl_no_seam_split", dict(max_seam_calls=0)),
    ]

    for ds_name in ["Event2012", "Event2018"]:
        ctx = contexts[ds_name]
        bids = HELDOUT_BLOCKS[ds_name]
        dss = [ctx["ds"][b] for b in bids]
        embs = [ctx["emb"][b] for b in bids]
        graphs_p = [ctx["graph_pure"][b] for b in bids]
        inits_p = [ctx["init_pure"][b] for b in bids]
        ds_out: Dict[str, Any] = dict(results.get(ds_name, {})) if resume else {}
        results[ds_name] = ds_out
        print(f"\n=== [Gate 2 Qwen3-8B Ablations] {ds_name} Held-Out (n={len(bids)}) ===", flush=True)

        for abl_key, kw in abl_specs:
            if resume and abl_key in ds_out:
                continue
            log_dir = OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / abl_key
            cl_ab = _make_client("siliconflow", model_name, log_dir, shared_cache)
            agg_ab, _ = _eval_llm_parallel(
                abl_key, bids, dss, inits_p,
                lambda i, b, bk, _k=abl_key, _kw=kw, _c=cl_ab: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag=_k, graph=graphs_p[i], labels_init=inits_p[i], **_kw)
                ),
                max_workers=max_workers,
            )
            _attach_recorded_live_api_sec(log_dir / "journal.jsonl", agg_ab, live_lat_index)
            ds_out[abl_key] = agg_ab
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Stage 4: Full Chronological Cross-Day Evaluation (M_00..M_end, eval M_06+)
# ---------------------------------------------------------------------------

def _load_block_cl(path: Path) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): [(int(a), int(b)) for a, b in v] for k, v in raw.items()}


def _eval_crossday_for_model(
    ds_name: str,
    ctx: Dict[str, Any],
    held_preds: Sequence[np.ndarray],
    client: BudgetedLLMClient,
    full_sweep: bool = True,
    block_cannot_links: Optional[Dict[int, Sequence[Tuple[int, int]]]] = None,
) -> Dict[str, Any]:
    held_bids = HELDOUT_BLOCKS[ds_name]
    pred_map: Dict[int, np.ndarray] = {b: ctx["init_pure"][b] for b in WARMUP_BLOCKS}
    for b, pr in zip(held_bids, held_preds):
        pred_map[b] = pr
    all_bids = WARMUP_BLOCKS + list(held_bids)
    stream = [(b, ctx["ds"][b], ctx["emb"][b], pred_map[b]) for b in all_bids]

    out: Dict[str, Any] = {}
    out["no_link"] = evaluate_cross_day_stream(stream, "no_link", eval_start_block=6)
    p_list = [0.60, 0.65, 0.72, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95] if full_sweep else [0.80, 0.88, 0.90, 0.92]
    for tau_p in p_list:
        out[f"proto_{tau_p:.2f}"] = evaluate_cross_day_stream(stream, "proto_only", eval_start_block=6, tau_proto=tau_p)
    if full_sweep:
        for tau_pa in [0.80, 0.85, 0.90, 0.95]:
            out[f"proto_attr_{tau_pa:.2f}"] = evaluate_cross_day_stream(stream, "proto_attr_only", eval_start_block=6, tau_proto=tau_pa, attr_weight=0.15)
    out["sug_cross_star"] = evaluate_cross_day_stream(
        stream, "sug_cross_star", eval_start_block=6, tau_lo=0.65, tau_hi=0.90, tau_fb=0.90, min_cross_size=3,
        client=client, budget_per_block=6000, block_cannot_links=block_cannot_links,
    )
    out["oracle_band"] = evaluate_cross_day_stream(
        stream, "oracle_band", eval_start_block=6, tau_lo=0.65, tau_hi=0.90, tau_fb=0.90, min_cross_size=3,
        block_cannot_links=block_cannot_links,
    )
    return out


def run_stage_qwen3_crossday(contexts: Dict[str, Dict[str, Any]], resume: bool = True) -> Dict[str, Any]:
    model_name = "Qwen/Qwen3-8B"
    shared_cache = OUT_DIR / "ledgers" / "qwen3_8b_shared_responses.jsonl"
    out_path = OUT_DIR / "stage_qwen3_crossday.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if (resume and out_path.exists()) else {"model": model_name}

    for ds_name in ["Event2012", "Event2018"]:
        if resume and ds_name in results and "sug_cross_star" in results[ds_name]:
            continue
        ctx = contexts[ds_name]
        bids = HELDOUT_BLOCKS[ds_name]
        held_preds = _load_preds(PRED_DIR / f"qwen3_{ds_name}.npz", bids)
        if held_preds is None:
            raise RuntimeError(f"Missing saved Qwen3 held-out predictions at {PRED_DIR / f'qwen3_{ds_name}.npz'}; run qwen3_main first.")
        block_cl = _load_block_cl(PRED_DIR / f"qwen3_{ds_name}.cl.json")
        print(f"\n=== [Gate 2 Qwen3-8B Cross-Day] {ds_name} (warmup M_00..M_05, eval M_{bids[0]:02d}..M_{bids[-1]:02d}) ===", flush=True)
        cl = _make_client("siliconflow", model_name, OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / "cross_star", shared_cache)
        results[ds_name] = _eval_crossday_for_model(ds_name, ctx, held_preds, cl, full_sweep=True, block_cannot_links=block_cl)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        for ck in ["no_link", "proto_0.80", "proto_0.88", "proto_0.90", "proto_attr_0.90", "sug_cross_star", "oracle_band"]:
            r = results[ds_name][ck]
            print(f"  {ck:16s} | G-AMI={r['g_ami']:5.2f} G-Mac={r['g_mac_b3']:5.2f} G-Mic={r['g_mic_b3']:5.2f} G-Tail={r['g_tail_b3']:5.2f} Q={r['stats']['llm_queried']}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Stage 5: Full Second-Model Evaluation (Qwen2.5-7B-Instruct, 27 Blocks + Cross-Day)
# ---------------------------------------------------------------------------

def run_stage_qwen25_full(contexts: Dict[str, Dict[str, Any]], max_workers: int = 8, resume: bool = True) -> Dict[str, Any]:
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    shared_cache = OUT_DIR / "ledgers" / "qwen25_7b_shared_responses.jsonl"
    out_path = OUT_DIR / "stage_qwen25_full.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if (resume and out_path.exists()) else {"model": model_name, "provider": "siliconflow"}

    def _ckpt() -> None:
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for ds_name in ["Event2012", "Event2018"]:
        ctx = contexts[ds_name]
        bids = HELDOUT_BLOCKS[ds_name]
        dss = [ctx["ds"][b] for b in bids]
        embs = [ctx["emb"][b] for b in bids]
        graphs_p = [ctx["graph_pure"][b] for b in bids]
        inits_p = [ctx["init_pure"][b] for b in bids]
        embed_model = ctx["embed_model"]

        def _cl(sub: str) -> BudgetedLLMClient:
            return _make_client("siliconflow", model_name, OUT_DIR / "ledgers" / "qwen25_7b" / ds_name / sub, shared_cache)

        ds_out: Dict[str, Any] = dict(results.get(ds_name, {})) if resume else {}
        results[ds_name] = ds_out
        print(f"\n=== [Gate 2 Qwen2.5-7B Full] {ds_name} Held-Out (M_{bids[0]:02d}..M_{bids[-1]:02d}, n={len(bids)}) ===", flush=True)

        pred_path = PRED_DIR / f"qwen25_{ds_name}.npz"
        if not (resume and "our_pure_star_meso" in ds_out and pred_path.exists()):
            cl_m = _cl("pure_star_meso")
            ds_out["our_pure_star_meso"], q25_preds = _eval_llm_parallel(
                "q25.OUR.Pure-Star-Meso", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_m: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag="pure_star_meso", graph=graphs_p[i], labels_init=inits_p[i])
                ),
                max_workers=max_workers,
            )
            _save_preds(pred_path, bids, q25_preds)
            cl_q25_path = pred_path.with_suffix(".cl.json")
            cl_q25 = {
                str(b): ds_out["our_pure_star_meso"]["per_block"].get(f"M_{b:02d}", {}).get("stats", {}).get("intra_cannot_link_pairs", [])
                for b in bids
            }
            cl_q25_path.write_text(json.dumps(cl_q25, indent=2), encoding="utf-8")
            _ckpt()

        if not (resume and "clusterllm_emnlp23" in ds_out):
            cl_c = _cl("clusterllm")
            sbert_inits = [
                run_sbert_agglom(em, distance_threshold=DEV_LOCKED_0TOK[ds_name]["sbert_dist"])[0]
                for em in embs
            ]
            ds_out["clusterllm_emnlp23"], _ = _eval_llm_parallel(
                "q25.5.ClusterLLM-NoFT", bids, dss, sbert_inits,
                lambda i, b, bk, _c=cl_c, _si=sbert_inits: run_clusterllm(b, dss[i], embs[i], _si[i], _c, bk),
                max_workers=max_workers,
            )
            _ckpt()

        if not (resume and "fewshot_post_tacl24" in ds_out):
            cl_f = _cl("fewshot")
            ds_out["fewshot_post_tacl24"], _ = _eval_llm_parallel(
                "q25.6.FewShotPost", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_f: run_fewshot_llm_post(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk),
                max_workers=max_workers,
            )
            _ckpt()

        if not (resume and "cequel_edge_25" in ds_out and "cequel_trans_25" in ds_out):
            cl_ceq = _cl("cequel")
            with ThreadPoolExecutor(max_workers=min(max_workers, len(bids))) as pool:
                futs = [
                    pool.submit(
                        lambda i=i, bid=bid: (
                            lambda bk: (bk.start_block(), run_cequel(bid, dss[i], embs[i], graphs_p[i], inits_p[i], cl_ceq, bk))[1]
                        )(CausalUniformBucket(tokens_per_block=BUDGET_PER_BLK, allow_carryover=False))
                    )
                    for i, bid in enumerate(bids)
                ]
                ceq_outs = [f.result() for f in futs]
            c_e, c_t, c_s, c_st = [o[0] for o in ceq_outs], [o[1] for o in ceq_outs], [o[2] for o in ceq_outs], [o[3] for o in ceq_outs]
            ds_out["cequel_edge_25"] = _aggregate_block_metrics(bids, dss, c_e, init_list=inits_p, shown_list=c_s, stats_list=c_st)
            ds_out["cequel_trans_25"] = _aggregate_block_metrics(bids, dss, c_t, init_list=inits_p, shown_list=c_s, stats_list=c_st)
            print(f"  [q25.7b.Cequel-Trans          ] AMI={ds_out['cequel_trans_25']['ami']:5.2f} Mac={ds_out['cequel_trans_25']['macro_b3']:5.2f}", flush=True)
            _ckpt()

        if not (resume and "lsed_acl25" in ds_out):
            cl_l = _cl("lsed")
            ds_out["lsed_acl25"], _ = _eval_llm_parallel(
                "q25.8.LSED-Bud", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_l: run_lsed_budgeted(b, dss[i], embs[i], inits_p[i], embed_model, _c, bk, knn_sim_threshold=0.50),
                max_workers=max_workers,
            )
            _ckpt()

        if not (resume and "ragsede_www26" in ds_out):
            cl_r = _cl("ragsede")
            ds_out["ragsede_www26"], _ = _eval_llm_parallel(
                "q25.9.RagSEDE-Bud", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_r: run_ragsede_budgeted(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk),
                max_workers=max_workers,
            )
            _ckpt()

        if not (resume and "lhg_www26" in ds_out):
            cl_h = _cl("lhg")
            ds_out["lhg_www26"], _ = _eval_llm_parallel(
                "q25.10.LHG-Bud", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_h: run_lhg_budgeted(b, dss[i], embs[i], graphs_p[i], inits_p[i], embed_model, _c, bk),
                max_workers=max_workers,
            )
            _ckpt()

        if not (resume and "ctrl_random_meso_s42" in ds_out):
            cl_rm = _cl("rmeso42")
            ds_out["ctrl_random_meso_s42"], _ = _eval_llm_parallel(
                "q25.RandomMeso(s42)", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_rm: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag="random_meso_s42", random_order=True, seed=42, graph=graphs_p[i], labels_init=inits_p[i])
                ),
                max_workers=max_workers,
            )
            _ckpt()

        if not (resume and "cross_day" in ds_out):
            q25_held = _load_preds(pred_path, bids)
            assert q25_held is not None
            q25_cl = _load_block_cl(pred_path.with_suffix(".cl.json"))
            cl_cd = _cl("cross_star")
            ds_out["cross_day"] = _eval_crossday_for_model(ds_name, ctx, q25_held, cl_cd, full_sweep=False, block_cannot_links=q25_cl)
            for ck in ["no_link", "proto_0.88", "proto_0.90", "sug_cross_star", "oracle_band"]:
                r = ds_out["cross_day"][ck]
                print(f"  [q25 CrossDay {ck:14s}] G-AMI={r['g_ami']:5.2f} G-Mac={r['g_mac_b3']:5.2f} Q={r['stats']['llm_queried']}", flush=True)
            _ckpt()

    return results


# ---------------------------------------------------------------------------
# Stage 6: Causal Budget Curves (0..20k) + Anti-Memorization (incl. E2E Pseudonym)
# ---------------------------------------------------------------------------

_PH_SPLIT_RE = re.compile(f"({_PLACEHOLDER_RE.pattern})")
_SPAN_SEP_RE = re.compile(r"[\s\-’'./:,;&]+")
_LOWER_2CH_FUNC: set[str] = {
    "us", "as", "my", "me", "by", "up", "go", "so", "no", "ok",
    "am", "pm", "vs", "rt", "en", "fr", "id", "un", "et",
}


def _build_phrase_table(
    items: Sequence[Tuple[str, str]],
    max_ngram: int = 8,
) -> Tuple[Dict[int, Dict[Tuple[str, ...], str]], int]:
    """Build length-indexed token-tuple lookup for longest-match O(N) replacement."""
    by_len: Dict[int, Dict[Tuple[str, ...], str]] = {}
    actual_max = 0
    for key, val in items:
        toks = tuple(key.split())
        if not toks or len(toks) > max_ngram:
            continue
        by_len.setdefault(len(toks), {})
        if toks not in by_len[len(toks)]:
            by_len[len(toks)][toks] = val
            if len(toks) > actual_max:
                actual_max = len(toks)
    return by_len, actual_max


def _replace_token_spans(
    seg: str,
    phrase_by_len: Dict[int, Dict[Tuple[str, ...], str]],
    max_ngram: int,
    require_upper: bool = False,
) -> str:
    """Replace longest matching entity/hashtag token spans in a non-placeholder text segment."""
    if not seg or max_ngram <= 0:
        return seg
    matches = list(_TOKEN_RE.finditer(seg))
    if not matches:
        return seg
    norm_toks = [normalize_surface_key(m.group(0)) for m in matches]
    n_tok = len(matches)
    out_parts: List[str] = []
    cursor = 0
    idx = 0
    while idx < n_tok:
        matched_len = 0
        matched_val = ""
        upper_l = min(max_ngram, n_tok - idx)
        for span_l in range(upper_l, 0, -1):
            table = phrase_by_len.get(span_l)
            if not table:
                continue
            valid_sep = True
            for k in range(span_l - 1):
                between = seg[matches[idx + k].end() : matches[idx + k + 1].start()]
                if len(between) > 4 or not _SPAN_SEP_RE.fullmatch(between):
                    valid_sep = False
                    break
            if not valid_sep:
                continue
            cand_tup = tuple(norm_toks[idx : idx + span_l])
            if "" in cand_tup:
                continue
            if span_l == 1:
                nt = cand_tup[0]
                if len(nt) == 2 and nt in _LOWER_2CH_FUNC and matches[idx].group(0).islower():
                    continue
            matched_surface = seg[matches[idx].start() : matches[idx + span_l - 1].end()]
            if require_upper and not any(ch.isupper() for ch in matched_surface):
                continue
            val = table.get(cand_tup)
            if val is not None:
                matched_len = span_l
                matched_val = val
                break
        if matched_len > 0:
            start_pos = matches[idx].start()
            end_pos = matches[idx + matched_len - 1].end()
            next_idx = idx + matched_len
            # Consume trailing possessive 's / ’s token cleanly
            if next_idx < n_tok:
                gap = seg[end_pos : matches[next_idx].start()]
                if gap in {"'", "’"} and matches[next_idx].group(0).lower() == "s":
                    end_pos = matches[next_idx].end()
                    next_idx += 1
            out_parts.append(seg[cursor:start_pos])
            out_parts.append(matched_val)
            cursor = end_pos
            idx = next_idx
        else:
            idx += 1
    out_parts.append(seg[cursor:])
    return "".join(out_parts)


def _deidentify_block_full(ds: SocialDataset, bid: int, embed_model: str) -> Tuple[SocialDataset, np.ndarray, Dict[str, Any]]:
    """Build an end-to-end de-identified block dataset with per-post zero-bleed scrubbing,
    predicate-only MiniLM embeddings, and opaque graph attribute IDs."""
    ner_surfs: set[str] = set()
    tag_surfs: set[str] = set()
    mention_surfs: set[str] = set()

    for i, raw in enumerate(ds.texts):
        for e in ds.entities[i]:
            es = str(e).strip()
            if len(es) >= 2 and normalize_surface_key(es) not in _AUDIT_STOPWORDS:
                ner_surfs.add(es)
        for h in ds.hashtags[i]:
            hs = str(h).strip().lstrip("#")
            if len(hs) >= 2:
                tag_surfs.add(hs)
        for m in ds.mentioned_users[i]:
            ms = str(m).strip().lstrip("@")
            if len(ms) >= 2:
                mention_surfs.add(ms)
        cleaned = _URL_RE.sub(" ", raw)
        for hm in _HASHTAG_RE.finditer(cleaned):
            hb = hm.group(0).lstrip("#")
            if len(hb) >= 2:
                tag_surfs.add(hb)
        for mm in _MENTION_RE.finditer(cleaned):
            mention_surfs.add(mm.group(0).lstrip("@"))

    # 1. Block-consistent NER / hashtag / mention ID maps (no global heuristic word bleed)
    ent_map: Dict[str, str] = {}
    ent_idx = 0
    for s in sorted(ner_surfs, key=lambda s: (-len(s), normalize_surface_key(s))):
        k = normalize_surface_key(s)
        if not k or len(k) < 2 or k in _AUDIT_STOPWORDS:
            continue
        if k not in ent_map:
            ent_idx += 1
            ent_map[k] = f"[ENT_{ent_idx}]"

    tag_map: Dict[str, str] = {}
    tag_idx = 0
    for h in sorted(tag_surfs, key=lambda s: (-len(s), normalize_surface_key(s))):
        k = normalize_surface_key(h)
        if k and len(k) >= 2:
            if k in ent_map:
                tag_map[k] = "#" + ent_map[k]
            elif k not in tag_map:
                tag_idx += 1
                tag_map[k] = f"#[TAG_{tag_idx}]"

    men_map: Dict[str, str] = {}
    for idx_m, ms in enumerate(sorted(mention_surfs, key=lambda s: normalize_surface_key(s)), start=1):
        mk = normalize_surface_key(ms)
        if mk and mk not in men_map:
            men_map[mk] = f"[USER_{idx_m}]"

    # Cross-post NER lookup (applied with require_upper=True to catch missed capitalized entities)
    block_ner_items = sorted(ent_map.items(), key=lambda kv: (-len(kv[0].split()), -len(kv[0]), kv[0]))
    block_ner_table, block_ner_max = _build_phrase_table(block_ner_items, max_ngram=8)

    # 2. Per-post scrubbing (post-local NER + capitalized cross-post NER + post-local heuristics)
    extra_ent_map: Dict[str, str] = {}
    anon_texts: List[str] = []
    post_extra_keys: List[set[str]] = []

    def _apply_on_non_ph(text_val: str, table: Dict[int, Dict[Tuple[str, ...], str]], max_ng: int, req_upper: bool) -> str:
        if max_ng <= 0:
            return text_val
        parts = _PH_SPLIT_RE.split(text_val)
        for pi in range(0, len(parts), 2):
            if parts[pi]:
                parts[pi] = _replace_token_spans(parts[pi], table, max_ng, require_upper=req_upper)
        return "".join(parts)

    for i, raw in enumerate(ds.texts):
        t = _URL_RE.sub("[URL]", raw)
        t = _YEAR_RE.sub("[YEAR]", t)
        t = _MENTION_RE.sub(lambda m: men_map.get(normalize_surface_key(m.group(0)), "[USER]"), t)
        t = _HASHTAG_RE.sub(lambda m: tag_map.get(normalize_surface_key(m.group(0)), "[HASHTAG]"), t)

        # (a) Post's own NER entities (require_upper=False for 100% coverage)
        p_ner_items: List[Tuple[str, str]] = []
        for e in ds.entities[i]:
            ek = normalize_surface_key(str(e))
            if ek in ent_map:
                p_ner_items.append((ek, ent_map[ek]))
        if p_ner_items:
            p_ner_items.sort(key=lambda kv: (-len(kv[0].split()), -len(kv[0]), kv[0]))
            p_tbl, p_max = _build_phrase_table(p_ner_items, max_ngram=16)
            t = _apply_on_non_ph(t, p_tbl, p_max, req_upper=False)

        # (b) Cross-post capitalized NER entities (require_upper=True)
        t = _apply_on_non_ph(t, block_ner_table, block_ner_max, req_upper=True)

        # (c) Post-local capitalized proper-noun heuristics on remaining non-placeholder segments
        local_props: set[str] = set()
        parts_rem = _PH_SPLIT_RE.split(t)
        for pi in range(0, len(parts_rem), 2):
            seg = parts_rem[pi]
            if not seg:
                continue
            for pm in _MULTI_CAP_RE.finditer(seg):
                parts = pm.group(1).strip().split()
                if all(p.lower() not in _AUDIT_STOPWORDS and p.lower() not in _CAP_FUNCTION_WORDS for p in parts):
                    cand = " ".join(parts)
                    if len(cand) >= 4:
                        local_props.add(cand)
            for am in _ALLCAPS_RE.finditer(seg):
                tok = am.group(1)
                if tok not in _UPPER_BOILERPLATE and tok.lower() not in _AUDIT_STOPWORDS and len(tok) >= 2:
                    local_props.add(tok)
            for cm in _MID_CAP_RE.finditer(seg):
                tok = cm.group(1)
                if tok.lower() not in _AUDIT_STOPWORDS and len(tok) >= 3:
                    local_props.add(tok)

        local_keys_i: set[str] = set()
        if local_props:
            local_items: List[Tuple[str, str]] = []
            for lp in sorted(local_props, key=lambda s: (-len(s), normalize_surface_key(s))):
                lk = normalize_surface_key(lp)
                if not lk or len(lk) < 2 or lk in _AUDIT_STOPWORDS:
                    continue
                if lk in ent_map:
                    ph = ent_map[lk]
                elif lk in tag_map:
                    ph = tag_map[lk].lstrip("#")
                elif lk in extra_ent_map:
                    ph = extra_ent_map[lk]
                else:
                    ent_idx += 1
                    ph = f"[ENT_{ent_idx}]"
                    extra_ent_map[lk] = ph
                local_items.append((lk, ph))
                local_keys_i.add(lk)
            if local_items:
                local_items.sort(key=lambda kv: (-len(kv[0].split()), -len(kv[0]), kv[0]))
                l_tbl, l_max = _build_phrase_table(local_items, max_ngram=8)
                t = _apply_on_non_ph(t, l_tbl, l_max, req_upper=True)

        anon_texts.append(t)
        post_extra_keys.append(local_keys_i)

    # 3. Clean predicate-only MiniLM embeddings (no raw entities AND no [ENT_k] bracket collapse)
    emb_texts = [re.sub(r"\s+", " ", _PLACEHOLDER_RE.sub(" ", t)).strip() or "post" for t in anon_texts]
    anon_emb = encode_texts_hf(emb_texts, model_name=embed_model, batch_size=64, max_length=128)

    # Extract opaque anonymized attribute IDs & entity-free lemmas
    opaque_ents: List[List[str]] = []
    opaque_tags: List[List[str]] = []
    for t in anon_texts:
        es = sorted({x.strip("[]").lower() for x in re.findall(r"\[ENT_\d+\]", t)})
        ts = sorted({x.lstrip("#").strip("[]").lower() for x in re.findall(r"#\[(?:TAG|ENT)_\d+\]", t)})
        opaque_ents.append(es)
        opaque_tags.append(ts)

    global_single_ent_keys = {
        k for k in ent_map if (" " not in k) and len(k) >= 3
    } | {k for k in tag_map if (" " not in k) and len(k) >= 3}
    anon_lemmas: List[List[str]] = []
    for i, toks in enumerate(ds.tokens_lemma):
        post_blocked = set(global_single_ent_keys)
        for e in ds.entities[i]:
            ek = normalize_surface_key(str(e))
            if ek:
                post_blocked.add(ek)
                for pt in ek.split():
                    if len(pt) >= 3 and pt not in _AUDIT_STOPWORDS:
                        post_blocked.add(pt)
        for lk in post_extra_keys[i]:
            post_blocked.add(lk)
            for pt in lk.split():
                if len(pt) >= 3 and pt not in _AUDIT_STOPWORDS:
                    post_blocked.add(pt)
        anon_lemmas.append([w for w in toks if normalize_surface_key(w) not in post_blocked])

    uid_map: Dict[str, str] = {}
    anon_uids: List[str] = []
    for u in ds.user_ids:
        if u not in uid_map:
            uid_map[u] = f"u_anon_{len(uid_map)+1}"
        anon_uids.append(uid_map[u])

    # Upstream block-level masking coverage audit against raw dataset entities & hashtags
    blk_ent_total = blk_ent_masked = 0
    blk_tag_total = blk_tag_masked = 0
    for raw_t, anon_t, ents_i in zip(ds.texts, anon_texts, ds.entities):
        norm_raw = " " + normalize_surface_key(raw_t) + " "
        norm_anon = " " + normalize_surface_key(_PLACEHOLDER_RE.sub(" ", anon_t)) + " "
        seen_k: set[str] = set()
        for e in ents_i:
            kn = normalize_surface_key(str(e))
            if len(kn) >= 3 and kn not in _AUDIT_STOPWORDS and kn not in seen_k:
                seen_k.add(kn)
                if f" {kn} " in norm_raw:
                    blk_ent_total += 1
                    if f" {kn} " not in norm_anon:
                        blk_ent_masked += 1
        r_tags = {m.group(0).lstrip("#").lower() for m in _HASHTAG_RE.finditer(raw_t) if len(m.group(0)) >= 3}
        a_tags = {m.group(0).lstrip("#").lower() for m in _HASHTAG_RE.finditer(_PLACEHOLDER_RE.sub(" ", anon_t)) if len(m.group(0)) >= 3}
        blk_tag_total += len(r_tags)
        blk_tag_masked += len(r_tags - a_tags)

    anon_ds = SocialDataset(
        name=f"{ds.name}_e2e",
        tweet_ids=ds.tweet_ids,
        texts=anon_texts,
        labels=ds.labels,
        tokens_lemma=anon_lemmas,
        entities=opaque_ents,
        hashtags=opaque_tags,
        user_ids=anon_uids,
        mentioned_users=[[] for _ in anon_texts],
        timestamps=ds.timestamps,
        dates=["[DATE]"] * len(anon_texts),
    )
    meta = {
        "n_posts": len(anon_texts),
        "n_ent_mapped": len(ent_map) + len(extra_ent_map),
        "n_tag_mapped": len(tag_map),
        "n_user_mapped": len(men_map),
        "block_ent_total": blk_ent_total,
        "block_ent_masked": blk_ent_masked,
        "block_ent_mask_rate": round((blk_ent_masked / blk_ent_total) if blk_ent_total > 0 else 1.0, 6),
        "block_tag_total": blk_tag_total,
        "block_tag_masked": blk_tag_masked,
        "block_tag_mask_rate": round((blk_tag_masked / blk_tag_total) if blk_tag_total > 0 else 1.0, 6),
    }
    return anon_ds, anon_emb, meta


def run_stage_budget_anon(contexts: Dict[str, Dict[str, Any]], max_workers: int = 8, resume: bool = True) -> Dict[str, Any]:
    model_name = "Qwen/Qwen3-8B"
    shared_cache = OUT_DIR / "ledgers" / "qwen3_8b_shared_responses.jsonl"
    out_path = OUT_DIR / "stage_budget_anon.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if (resume and out_path.exists()) else {"model": model_name, "budget_curves": {}, "anonymization": {}}
    live_lat_index = _build_live_latency_index([OUT_DIR / "ledgers", ROOT / "results" / "dev_gate1" / "ledgers"])

    q3_main = json.loads((OUT_DIR / "stage_qwen3_main.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_qwen3_main.json").exists() else {}

    def _ckpt() -> None:
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- Part A: Causal Budget Curves B in {0, 2000, 5000, 10000, 20000} on all 27 held-out blocks ---
    b_curves: Dict[str, Any] = results.setdefault("budget_curves", {})
    for ds_name in ["Event2012", "Event2018"]:
        ctx = contexts[ds_name]
        bids = HELDOUT_BLOCKS[ds_name]
        dss = [ctx["ds"][b] for b in bids]
        embs = [ctx["emb"][b] for b in bids]
        graphs_p = [ctx["graph_pure"][b] for b in bids]
        inits_p = [ctx["init_pure"][b] for b in bids]

        ds_bc: Dict[str, Any] = b_curves.setdefault(ds_name, {})
        print(f"\n=== [Gate 2 Budget Curves] {ds_name} Held-Out (n={len(bids)}, B={{0,2k,5k,10k,20k}}) ===", flush=True)

        if not (resume and "B0" in ds_bc):
            ds_bc["B0"] = {"pure_2dse": _aggregate_block_metrics(bids, dss, inits_p)}
            _ckpt()

        for b_cap in [2000, 5000, 10000, 20000]:
            b_key = f"B{b_cap}"
            b_obj: Dict[str, Any] = ds_bc.setdefault(b_key, {})

            copied_from_main: set[str] = set()
            if b_cap == 10000 and ds_name in q3_main:
                for mkey in ["our_pure_star_meso", "cequel_trans_25", "fewshot_post_tacl24", "ctrl_random_meso_s42"]:
                    if mkey in q3_main[ds_name]:
                        b_obj[mkey] = q3_main[ds_name][mkey]
                        copied_from_main.add(mkey)
                _ckpt()

            max_sc = 40 if b_cap > 10000 else 24
            if "our_pure_star_meso" not in copied_from_main and not (resume and "our_pure_star_meso" in b_obj):
                ld_our = OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / f"b{b_cap}_our"
                cl = _make_client("siliconflow", model_name, ld_our, shared_cache)
                agg_our, _ = _eval_llm_parallel(
                    f"B{b_cap}.OUR.PureStarMeso", bids, dss, inits_p,
                    lambda i, b, bk, _c=cl, _ms=max_sc: (lambda r: (r[0], r[2], r[3]))(
                        run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag="pure_star_meso", max_star_calls=_ms, graph=graphs_p[i], labels_init=inits_p[i])
                    ),
                    budget_per_blk=b_cap, max_workers=max_workers,
                )
                _attach_recorded_live_api_sec(ld_our / "journal.jsonl", agg_our, live_lat_index)
                b_obj["our_pure_star_meso"] = agg_our
                _ckpt()

            if "fewshot_post_tacl24" not in copied_from_main and not (resume and "fewshot_post_tacl24" in b_obj):
                ld_fs = OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / f"b{b_cap}_fewshot"
                cl = _make_client("siliconflow", model_name, ld_fs, shared_cache)
                agg_fs, _ = _eval_llm_parallel(
                    f"B{b_cap}.FewShotPost", bids, dss, inits_p,
                    lambda i, b, bk, _c=cl: run_fewshot_llm_post(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk),
                    budget_per_blk=b_cap, max_workers=max_workers,
                )
                _attach_recorded_live_api_sec(ld_fs / "journal.jsonl", agg_fs, live_lat_index)
                b_obj["fewshot_post_tacl24"] = agg_fs
                _ckpt()

            if "cequel_trans_25" not in copied_from_main and not (resume and "cequel_trans_25" in b_obj):
                ld_ceq = OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / f"b{b_cap}_cequel"
                cl_ceq = _make_client("siliconflow", model_name, ld_ceq, shared_cache)
                with ThreadPoolExecutor(max_workers=min(max_workers, len(bids))) as pool:
                    futs = [
                        pool.submit(
                            lambda i=i, bid=bid: (
                                lambda bk: (bk.start_block(), run_cequel(bid, dss[i], embs[i], graphs_p[i], inits_p[i], cl_ceq, bk))[1]
                            )(CausalUniformBucket(tokens_per_block=b_cap, allow_carryover=False))
                        )
                        for i, bid in enumerate(bids)
                    ]
                    co = [f.result() for f in futs]
                agg_ct = _aggregate_block_metrics(bids, dss, [o[1] for o in co], init_list=inits_p, shown_list=[o[2] for o in co], stats_list=[o[3] for o in co])
                _attach_recorded_live_api_sec(ld_ceq / "journal.jsonl", agg_ct, live_lat_index)
                b_obj["cequel_trans_25"] = agg_ct
                print(f"  [B{b_cap}.Cequel-Trans          ] AMI={b_obj['cequel_trans_25']['ami']:5.2f} Mac={b_obj['cequel_trans_25']['macro_b3']:5.2f}", flush=True)
                _ckpt()

            if "ctrl_random_meso_s42" not in copied_from_main and not (resume and "ctrl_random_meso_s42" in b_obj):
                ld_rm = OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / f"b{b_cap}_rmeso42"
                cl = _make_client("siliconflow", model_name, ld_rm, shared_cache)
                agg_rm, _ = _eval_llm_parallel(
                    f"B{b_cap}.RandomMeso(s42)", bids, dss, inits_p,
                    lambda i, b, bk, _c=cl, _ms=max_sc: (lambda r: (r[0], r[2], r[3]))(
                        run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag="random_meso_s42", random_order=True, seed=42, max_star_calls=_ms, graph=graphs_p[i], labels_init=inits_p[i])
                    ),
                    budget_per_blk=b_cap, max_workers=max_workers,
                )
                _attach_recorded_live_api_sec(ld_rm / "journal.jsonl", agg_rm, live_lat_index)
                b_obj["ctrl_random_meso_s42"] = agg_rm
                _ckpt()

    # --- Part B: Anti-Memorization Anonymization & End-to-End De-ID Audit (Stratified 8 Held-Out Blocks) ---
    anon_out: Dict[str, Any] = results.setdefault("anonymization", {})
    for ds_name in ["Event2012", "Event2018"]:
        ctx = contexts[ds_name]
        abids = STRATIFIED_ANON_BLOCKS[ds_name]
        dss = [ctx["ds"][b] for b in abids]
        embs = [ctx["emb"][b] for b in abids]
        graphs_p = [ctx["graph_pure"][b] for b in abids]
        inits_p = [ctx["init_pure"][b] for b in abids]
        embed_model = ctx["embed_model"]

        ds_an: Dict[str, Any] = anon_out.setdefault(ds_name, {})
        print(f"\n=== [Gate 2 Anti-Memorization Audit] {ds_name} Stratified Blocks={abids} ===", flush=True)

        if not (resume and "base_2dse_0tok" in ds_an):
            ds_an["base_2dse_0tok"] = _aggregate_block_metrics(abids, dss, inits_p)
            _ckpt()

        if not (resume and "raw" in ds_an):
            ld_raw = OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / "anon_raw"
            cl = _make_client("siliconflow", model_name, ld_raw, shared_cache)
            agg_raw, _ = _eval_llm_parallel(
                "anon.raw", abids, dss, inits_p,
                lambda i, b, bk, _c=cl: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag="pure_star_meso", graph=graphs_p[i], labels_init=inits_p[i])
                ),
                max_workers=4,
            )
            _attach_recorded_live_api_sec(ld_raw / "journal.jsonl", agg_raw, live_lat_index)
            ds_an["raw"] = agg_raw
            _ckpt()

        for mode_label, q_mode in [("pseudonym", "pseudonym"), ("extract_blind", "blind")]:
            if resume and mode_label in ds_an:
                continue
            store = AnonymizationAuditStore()
            ld_mode = OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / f"anon_{mode_label}"
            cl = _make_client("siliconflow", model_name, ld_mode, shared_cache)
            agg_a, _ = _eval_llm_parallel(
                f"anon.{mode_label}", abids, dss, inits_p,
                lambda i, b, bk, _c=cl, _qm=q_mode, _st=store: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(ds_name, b, dss[i], embs[i], _c, bk, strategy_tag=f"anon_{_qm}", graph=graphs_p[i], labels_init=inits_p[i], anonymizer=QueryAnonymizer(_qm), audit_store=_st)
                ),
                max_workers=4,
            )
            _attach_recorded_live_api_sec(ld_mode / "journal.jsonl", agg_a, live_lat_index)
            agg_a["mask_audit"] = store.summarize_for_dataset(ds_name)
            ds_an[mode_label] = agg_a
            _ckpt()

        if not (
            resume
            and "e2e_pseudonym" in ds_an
            and "e2e_pseudonym_base_2dse" in ds_an
            and "e2e_pseudonym_automerge_0tok" in ds_an
            and "e2e_pseudonym_meso_heur_0tok" in ds_an
        ):
            tm_mh, ss_mh = DEV_LOCKED_0TOK[ds_name]["meso_heur"]
            e2e_dss, e2e_embs, e2e_graphs, e2e_inits, e2e_metas = [], [], [], [], []
            e2e_am_preds, e2e_am_sts = [], []
            e2e_mh_preds, e2e_mh_sts = [], []
            for i, b in enumerate(abids):
                ads, aemb, ameta = _deidentify_block_full(dss[i], b, embed_model)
                ag = build_message_graph(
                    ads, aemb, knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5, compute_non_entity_tfidf=False
                )
                al, _ = solve_2d_se(ag.adj, refine_moves=True)
                am_p, am_s = run_automerge_only(ads, aemb, al, tau_auto=0.85, guard_sim=0.40)
                mh_p, mh_s = run_meso_heur_best(ads, aemb, ag, al, tau_merge=tm_mh, min_seam_sep=ss_mh)
                e2e_dss.append(ads)
                e2e_embs.append(aemb)
                e2e_graphs.append(ag)
                e2e_inits.append(al)
                e2e_metas.append(ameta)
                e2e_am_preds.append(am_p)
                e2e_am_sts.append(am_s)
                e2e_mh_preds.append(mh_p)
                e2e_mh_sts.append(mh_s)
            ds_an["e2e_pseudonym_base_2dse"] = _aggregate_block_metrics(abids, e2e_dss, e2e_inits)
            ds_an["e2e_pseudonym_automerge_0tok"] = _aggregate_block_metrics(
                abids, e2e_dss, e2e_am_preds, init_list=e2e_inits, stats_list=e2e_am_sts
            )
            ds_an["e2e_pseudonym_meso_heur_0tok"] = _aggregate_block_metrics(
                abids, e2e_dss, e2e_mh_preds, init_list=e2e_inits, stats_list=e2e_mh_sts
            )
            ds_an["e2e_pseudonym_meso_heur_0tok"]["cfg"] = {"tau_merge": tm_mh, "min_seam_sep": ss_mh}
            store_e2e = AnonymizationAuditStore()
            ld_e2e = OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / "anon_e2e"
            cl_e2e = _make_client("siliconflow", model_name, ld_e2e, shared_cache)
            agg_e2e, _ = _eval_llm_parallel(
                "anon.e2e_pseudonym", abids, e2e_dss, e2e_inits,
                lambda i, b, bk, _c=cl_e2e, _st=store_e2e: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(
                        ds_name, b, e2e_dss[i], e2e_embs[i], _c, bk,
                        strategy_tag="anon_e2e", graph=e2e_graphs[i], labels_init=e2e_inits[i],
                        anonymizer=None, audit_store=_st,
                        pre_anonymized=True, orig_ds_for_audit=dss[i],
                    )
                ),
                max_workers=4,
            )
            _attach_recorded_live_api_sec(ld_e2e / "journal.jsonl", agg_e2e, live_lat_index)
            agg_e2e["mask_audit"] = store_e2e.summarize_for_dataset(ds_name)
            agg_e2e["block_deid_stats"] = e2e_metas
            ds_an["e2e_pseudonym"] = agg_e2e
            print(
                f"  [anon.e2e Base -> AutoMerge -> MesoHeur -> +LLM] "
                f"AMI={ds_an['e2e_pseudonym_base_2dse']['ami']:5.2f} -> "
                f"{ds_an['e2e_pseudonym_automerge_0tok']['ami']:5.2f} -> "
                f"{ds_an['e2e_pseudonym_meso_heur_0tok']['ami']:5.2f} -> "
                f"{agg_e2e['ami']:5.2f} Mac={agg_e2e['macro_b3']:5.2f}",
                flush=True,
            )
            _ckpt()

    return results


# ---------------------------------------------------------------------------
# Report Generator: Publication-Ready Markdown + Master JSON
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    if p <= 0.0:
        return "0"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4g}"


def build_gate2_report() -> None:
    s0 = json.loads((OUT_DIR / "stage_0tok.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_0tok.json").exists() else {}
    q3m = json.loads((OUT_DIR / "stage_qwen3_main.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_qwen3_main.json").exists() else {}
    q3a = json.loads((OUT_DIR / "stage_qwen3_abl.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_qwen3_abl.json").exists() else {}
    q3c = json.loads((OUT_DIR / "stage_qwen3_crossday.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_qwen3_crossday.json").exists() else {}
    q25 = json.loads((OUT_DIR / "stage_qwen25_full.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_qwen25_full.json").exists() else {}
    ba = json.loads((OUT_DIR / "stage_budget_anon.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_budget_anon.json").exists() else {}

    lines: List[str] = ["# Gate 2 Held-Out Full Benchmark Report (27 Blocks: Event2012 M_06..M_21 + Event2018 M_06..M_16)\n"]

    for ds_name in ["Event2012", "Event2018"]:
        if ds_name not in s0 or ds_name not in q3m:
            continue
        bids = HELDOUT_BLOCKS[ds_name]
        lines.append(f"## {ds_name} Held-Out (`M_{bids[0]:02d}..M_{bids[-1]:02d}`, {len(bids)} blocks, {s0[ds_name]['ctrl_pure_2dse_0tok']['n_posts']} posts)\n")
        if "structural_fs_fm_audit" in s0[ds_name]:
            fa = s0[ds_name]["structural_fs_fm_audit"]
            lines.append(
                f"> Structural FS/FM Ratio: Pure 2D-SE = `{fa['pure_fs_over_fm']}x` "
                f"(FS={fa['pure_2dse_fs']}, FM={fa['pure_2dse_fm']}); "
                f"HiSEvent Def = `{fa['def_fs_over_fm']}x` (FS={fa['hisevent_def_fs']}, FM={fa['hisevent_def_fm']}).\n"
            )
        base_pure_cpu = float(s0[ds_name]["ctrl_pure_2dse_0tok"].get("router_cpu_sec", s0[ds_name]["ctrl_pure_2dse_0tok"].get("total_wall_sec", 0.0)))
        def_cpu = float(s0[ds_name]["hisevent_default"].get("router_cpu_sec", s0[ds_name]["hisevent_default"].get("total_wall_sec", 0.0)))
        tuned_cpu = float(s0[ds_name]["hisevent_dev_tuned"].get("router_cpu_sec", s0[ds_name]["hisevent_dev_tuned"].get("total_wall_sec", 0.0)))
        lines.append("| Group | Method | Tok | Show% | Err | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet | CPU(s) | E2E CPU(s) | API(s) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for grp, nm, k, base_c in [
            ("0-LLM Paper 1", "EventX (TKDD'20)", "eventx_tkdd20", 0.0),
            ("0-LLM Paper 2", "SBERT-Agglom (EMNLP'19)", "sbert_emnlp19", 0.0),
            ("0-LLM Paper 3a", "HiSEvent Default (AAAI'24)", "hisevent_default", 0.0),
            ("0-LLM Paper 3b", "HiSEvent Dev-Tuned (AAAI'24)", "hisevent_dev_tuned", 0.0),
            ("0-LLM Paper 4", "HyperSED (AAAI'25)", "hypersed_aaai25", 0.0),
            ("Internal 0-Tok", "Pure 2D-SE Base (0.50)", "ctrl_pure_2dse_0tok", 0.0),
            ("Internal 0-Tok", "AutoMerge-Only (0.85)", "ctrl_automerge_only", base_pure_cpu),
            ("Internal 0-Tok", "Meso-Heur-Best (Pure)", "ctrl_meso_heur_pure", base_pure_cpu),
        ]:
            if k in s0[ds_name]:
                r = s0[ds_name][k]
                m_cpu = float(r.get("router_cpu_sec", r.get("total_wall_sec", 0.0)))
                e2e_cpu = base_c + m_cpu
                lines.append(f"| {grp} | {nm} | 0 | 0.00% | 0 | {r['ami']:.2f} | {r['macro_b3']:.2f} | {r['micro_b3']:.2f} | {r['tail_b3']:.2f} | +0.0 | {m_cpu:.1f} | {e2e_cpu:.1f} | 0.0 |")
        for grp, nm, k, base_c in [
            ("LLM Paper 5", "ClusterLLM-NoFT (EMNLP'23)", "clusterllm_emnlp23", 0.0),
            ("LLM Paper 6", "Few-Shot LLM-Post (TACL'24)", "fewshot_post_tacl24", base_pure_cpu),
            ("LLM Paper 7a", "Cequel-Edge (arXiv'25)", "cequel_edge_25", base_pure_cpu),
            ("LLM Paper 7b", "Cequel-Transitive (arXiv'25)", "cequel_trans_25", base_pure_cpu),
            ("LLM Paper 8", "LSED-Budgeted (ACL'25)", "lsed_acl25", base_pure_cpu),
            ("LLM Paper 9", "RagSEDE-Budgeted (WWW'26)", "ragsede_www26", base_pure_cpu),
            ("LLM Paper 10", "LHG-Budgeted (WWW'26)", "lhg_www26", base_pure_cpu),
            ("Internal Ctrl", "Random (3-seed mean)", "ctrl_random_mean3", base_pure_cpu),
            ("Internal Ctrl", "Random-Meso (Star-Meso Rand-Order, 3-seed)", "ctrl_random_meso_mean3", base_pure_cpu),
            ("**OUR METHOD**", "**Pure-Star-Meso (Qwen3-8B)**", "our_pure_star_meso", base_pure_cpu),
            ("Plug-and-Play", "Star-Meso on HiSEvent Default", "our_star_meso_on_hisevent_def", def_cpu),
            ("Plug-and-Play", "Star-Meso on HiSEvent Dev-Tuned", "our_star_meso_on_hisevent_tuned", tuned_cpu),
        ]:
            if k in q3m[ds_name]:
                r = q3m[ds_name][k]
                m_cpu = float(r.get("router_cpu_sec", 0.0))
                e2e_cpu = base_c + m_cpu
                lines.append(
                    f"| {grp} | {nm} | {int(r['total_tokens'])} | {r['shown_pct']:.2f}% | {int(r.get('parse_errors', 0))} | "
                    f"{r['ami']:.2f} | {r['macro_b3']:.2f} | {r['micro_b3']:.2f} | {r['tail_b3']:.2f} | {r.get('unq_net_mean', 0):+.1f} | "
                    f"{m_cpu:.1f} | {e2e_cpu:.1f} | {r.get('llm_api_sec', 0):.1f} |"
                )
        lines.append("\n*Note: `CPU(s)` reports method/router local CPU time excluding API latency and RPM sleep; `E2E CPU(s)` includes Stage-1 2D-SE graph+solve time where applicable. `ClusterLLM-NoFT (EMNLP'23)` omits contrastive encoder fine-tuning under the zero-training budgeted protocol and computes `UnqNet` relative to its 0-LLM hierarchical base `SBERT-Agglom (0.60)`. `HyperSED` caps `n_anchors <= 360`.*\n")

    if "wilcoxon_holm" in q3m:
        lines.append("## Paired Two-Sided Wilcoxon + Holm-Bonferroni Tests (`Pure-Star-Meso Qwen3-8B`)\n")
        for scope, tbl in q3m["wilcoxon_holm"].items():
            lines.append(f"### Scope: {scope}\n")
            lines.append("| Baseline | ΔAMI (p_raw / p_Holm) | ΔMac-$B^3$ (p_raw / p_Holm) | ΔMic-$B^3$ (p_raw / p_Holm) | ΔTail-$B^3$ (p_raw / p_Holm) |")
            lines.append("|---|---|---|---|---|")
            for bnm, d in tbl.items():
                cells = []
                for m in ("ami", "macro_b3", "micro_b3", "tail_b3"):
                    v = d[m]
                    cells.append(f"{v['mean_diff']:+.2f} ({_fmt_p(float(v['p_raw']))} / **{_fmt_p(float(v['p_holm']))}**)")
                lines.append(f"| {bnm} | " + " | ".join(cells) + " |")
            lines.append("")

    if q3a:
        lines.append("## Five-Pillar Held-Out Ablations (`Qwen/Qwen3-8B`, 27 Blocks)\n")
        for ds_name in ["Event2012", "Event2018"]:
            if ds_name not in q3a or ds_name not in q3m:
                continue
            lines.append(f"### {ds_name}\n")
            lines.append("| Variant | Calls | Tok | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            rf = q3m[ds_name]["our_pure_star_meso"]
            lines.append(f"| **Full Pure-Star-Meso** | {rf['total_calls']} | {rf['total_tokens']} | {rf['ami']:.2f} | {rf['macro_b3']:.2f} | {rf['micro_b3']:.2f} | {rf['tail_b3']:.2f} | {rf.get('unq_net_mean',0):+.1f} |")
            for ak, albl in [
                ("abl_no_automerge", "w/o AutoMerge (>=0.85)"),
                ("abl_no_bayes_band", "w/o Two-Tier Band"),
                ("abl_no_star_packing", "w/o Star Packing (k=1 Pair)"),
                ("abl_no_guarded_uf", "w/o Guarded UF"),
                ("abl_no_seam_split", "w/o Seam Split"),
            ]:
                if ak in q3a[ds_name]:
                    ra = q3a[ds_name][ak]
                    lines.append(f"| {albl} | {ra['total_calls']} | {ra['total_tokens']} | {ra['ami']:.2f} | {ra['macro_b3']:.2f} | {ra['micro_b3']:.2f} | {ra['tail_b3']:.2f} | {ra.get('unq_net_mean',0):+.1f} |")
            lines.append("")

    if q3c:
        lines.append("## Chronological Cross-Day Stream (`Qwen/Qwen3-8B`, Warmup `M_00..M_05`, Eval `M_06+`)\n")
        for ds_name in ["Event2012", "Event2018"]:
            if ds_name not in q3c:
                continue
            lines.append(f"### {ds_name}\n")
            lines.append("| Mode | G-AMI | G-Mac$B^3$ | G-Mic$B^3$ | G-Tail$B^3$ | Q | Tok |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for ck, cr in q3c[ds_name].items():
                lines.append(f"| {ck} | {cr['g_ami']:.2f} | {cr['g_mac_b3']:.2f} | {cr['g_mic_b3']:.2f} | {cr['g_tail_b3']:.2f} | {cr['stats']['llm_queried']} | {cr['stats']['cross_tokens']} |")
            lines.append("")

    if q25:
        lines.append("## Second-Model Full Held-Out Suite (`Qwen/Qwen2.5-7B-Instruct`, 27 Blocks)\n")
        for ds_name in ["Event2012", "Event2018"]:
            if ds_name not in q25:
                continue
            lines.append(f"### {ds_name} (`Qwen/Qwen2.5-7B-Instruct`)\n")
            lines.append("| Method | Calls | Tok | Show% | Err | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for k in ["clusterllm_emnlp23", "fewshot_post_tacl24", "cequel_edge_25", "cequel_trans_25", "lsed_acl25", "ragsede_www26", "lhg_www26", "ctrl_random_meso_s42", "our_pure_star_meso"]:
                if k in q25[ds_name]:
                    r = q25[ds_name][k]
                    lines.append(f"| {k} | {r['total_calls']} | {r['total_tokens']} | {r['shown_pct']:.2f}% | {r['parse_errors']} | {r['ami']:.2f} | {r['macro_b3']:.2f} | {r['micro_b3']:.2f} | {r['tail_b3']:.2f} | {r.get('unq_net_mean',0):+.1f} |")
            if "cross_day" in q25[ds_name]:
                lines.append("\nCross-Day (`eval M_06+`):")
                for ck in ["no_link", "proto_0.80", "proto_0.88", "proto_0.90", "sug_cross_star", "oracle_band"]:
                    if ck in q25[ds_name]["cross_day"]:
                        cr = q25[ds_name]["cross_day"][ck]
                        lines.append(f"- `{ck}`: G-AMI={cr['g_ami']:.2f}, G-Mac$B^3$={cr['g_mac_b3']:.2f}, G-Mic$B^3$={cr['g_mic_b3']:.2f}, G-Tail$B^3$={cr['g_tail_b3']:.2f}, Q={cr['stats']['llm_queried']}")
            lines.append("")

    if ba:
        bc = ba.get("budget_curves", {})
        if bc:
            lines.append("## Budget Curves Across All 27 Held-Out Blocks (`Qwen/Qwen3-8B`)\n")
            for ds_name in ["Event2012", "Event2018"]:
                if ds_name not in bc:
                    continue
                lines.append(f"### {ds_name}\n")
                lines.append("| Budget/Blk | Method | Tok | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ |")
                lines.append("|---|---|---:|---:|---:|---:|---:|")
                if "B0" in bc[ds_name]:
                    r0 = bc[ds_name]["B0"]["pure_2dse"]
                    lines.append(f"| 0 | Pure 2D-SE Base | 0 | {r0['ami']:.2f} | {r0['macro_b3']:.2f} | {r0['micro_b3']:.2f} | {r0['tail_b3']:.2f} |")
                for b_cap in [2000, 5000, 10000, 20000]:
                    bk = f"B{b_cap}"
                    if bk not in bc[ds_name]:
                        continue
                    for mk in ["fewshot_post_tacl24", "cequel_trans_25", "ctrl_random_meso_s42", "our_pure_star_meso"]:
                        if mk in bc[ds_name][bk]:
                            rm = bc[ds_name][bk][mk]
                            lines.append(f"| {b_cap} | {mk} | {rm['total_tokens']} | {rm['ami']:.2f} | {rm['macro_b3']:.2f} | {rm['micro_b3']:.2f} | {rm['tail_b3']:.2f} |")
                lines.append("")

        an = ba.get("anonymization", {})
        if an:
            lines.append("## Anti-Memorization & End-to-End De-Identification Audit (Stratified 8 Held-Out Blocks)\n")
            for ds_name in ["Event2012", "Event2018"]:
                if ds_name not in an:
                    continue
                lines.append(f"### {ds_name} (`{STRATIFIED_ANON_BLOCKS[ds_name]}`)\n")
                lines.append("| Condition | Tok | AMI | Mac-$B^3$ | Tail-$B^3$ | UnqNet | EntMask% | TagMask% | LeakQ |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
                for ck in ["base_2dse_0tok", "raw", "pseudonym", "extract_blind", "e2e_pseudonym_base_2dse", "e2e_pseudonym_automerge_0tok", "e2e_pseudonym_meso_heur_0tok", "e2e_pseudonym"]:
                    if ck not in an[ds_name]:
                        continue
                    rc = an[ds_name][ck]
                    ma = rc.get("mask_audit", {})
                    em = f"{ma['entity_mask_rate']*100:.1f}%" if "entity_mask_rate" in ma else "-"
                    tm = f"{ma['hashtag_mask_rate']*100:.1f}%" if "hashtag_mask_rate" in ma else "-"
                    lines.append(
                        f"| {ck} | {rc.get('total_tokens', 0)} | {rc['ami']:.2f} | {rc['macro_b3']:.2f} | {rc['tail_b3']:.2f} | "
                        f"{rc.get('unq_net_mean', 0):+.1f} | {em} | {tm} | {rc.get('anon_leak_queries', 0)} |"
                    )
                lines.append("")

    p1_path = OUT_DIR / "stage_p1_mechanisms.json"
    p1: Dict[str, Any] = {}
    if p1_path.exists():
        p1 = json.loads(p1_path.read_text(encoding="utf-8"))
        lines.append("## Mechanistic Attribution & Boundary Calibration (P1 Audit)\n")
        lines.append("*(Zero-API mechanistic audit; full tables in [`p1_mechanisms_report.md`](./p1_mechanisms_report.md) and [`stage_p1_mechanisms.json`](./stage_p1_mechanisms.json))*\n")
        fa = p1["factor_attribution"]["transitions"]
        lines.append("### Controlled 3-Factor Attribution (`Pooled27`)\n")
        lines.append("| Factor Transition | ΔAMI | ΔMac-$B^3$ | ΔMic-$B^3$ | ΔTail-$B^3$ |")
        lines.append("|---|---:|---:|---:|---:|")
        for tid in ("factor1_structural_prior", "factor2_priority_scheduling", "factor3_llm_over_heuristic", "factor_total_over_random"):
            t = fa[tid]
            sc = t["scopes"]["Pooled27"]
            lines.append(
                f"| {t['title']} | {sc['ami']['mean_diff']:+.2f} | {sc['macro_b3']['mean_diff']:+.2f} | "
                f"{sc['micro_b3']['mean_diff']:+.2f} | {sc['tail_b3']['mean_diff']:+.2f} |"
            )
        lines.append("")
        ec = p1["encoder_cosine_calibration"]
        gu = p1["guarded_uf_audit"]
        lines.append("### EN vs FR Encoder Cosine Calibration (`AutoMerge >=0.85`) & `GuardedUF` Risk Insurance\n")
        lines.append("| Dataset | Diff-Event Cos `p95 / p99` | `>=0.85` Prec% (Size-Wtd%) | Ablation `Full-NoAM` ΔFM / ΔFS | GuardedUF Prevented FM (`+% vs Full`) |")
        lines.append("|---|---:|---:|---|---|")
        for ds_name in ("Event2012", "Event2018"):
            c = ec[ds_name]
            g = gu[ds_name]
            am_bin = next(b for b in c["bins"] if b["bin"] == ">=0.85 (AutoMerge)")
            dc = c["diff_event_pair_cosine"]
            lines.append(
                f"| {ds_name} | {dc['p95']:.3f} / {dc['p99']:.3f} | "
                f"{am_bin['precision_pct']:.1f}% ({c['automerge_size_weighted_precision_pct']:.1f}%) | "
                f"{c['automerge_ablation_delta_fm']:+,d} / {c['automerge_ablation_delta_fs']:+,d} | "
                f"{g['prevented_fm_pairs']:+,d} (+{g['prevented_fm_pct_of_full']:.1f}%) |"
            )
        lines.append("")

    (OUT_DIR / "gate2_full_report.md").write_text("\n".join(lines), encoding="utf-8")
    master = {
        "stage_0tok": s0,
        "stage_qwen3_main": q3m,
        "stage_qwen3_abl": q3a,
        "stage_qwen3_crossday": q3c,
        "stage_qwen25_full": q25,
        "stage_budget_anon": ba,
        "stage_p1_mechanisms": p1,
    }
    (OUT_DIR / "gate2_full_report.json").write_text(json.dumps(master, indent=2), encoding="utf-8")
    print(f"\nSaved Gate 2 full report -> {OUT_DIR / 'gate2_full_report.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2 Held-Out Full Benchmark Suite (27 Held-Out Blocks)")
    parser.add_argument(
        "--stage",
        choices=["0tok", "qwen3_main", "qwen3_abl", "qwen3_crossday", "qwen25_full", "budget_anon", "report", "all"],
        default="all",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true", help="Re-run metrics cleanly rather than skipping finished JSON keys")
    args = parser.parse_args()
    resume = not args.no_resume

    t0 = time.time()
    if args.stage != "report":
        need_warmup = args.stage in {"qwen3_crossday", "qwen25_full", "all"}
        contexts = {
            ds_name: _prepare_context(
                ds_name,
                (WARMUP_BLOCKS if need_warmup else []) + HELDOUT_BLOCKS[ds_name],
                need_tuned_diff=(args.stage in {"0tok", "qwen3_main", "all"}),
            )
            for ds_name in ["Event2012", "Event2018"]
        }
        if args.stage in {"0tok", "all"}:
            run_stage_0tok(contexts, resume=resume)
        if args.stage in {"qwen3_main", "all"}:
            run_stage_qwen3_main(contexts, max_workers=args.max_workers, resume=resume)
        if args.stage in {"qwen3_abl", "all"}:
            run_stage_qwen3_abl(contexts, max_workers=args.max_workers, resume=resume)
        if args.stage in {"qwen3_crossday", "all"}:
            run_stage_qwen3_crossday(contexts, resume=resume)
        if args.stage in {"qwen25_full", "all"}:
            run_stage_qwen25_full(contexts, max_workers=args.max_workers, resume=resume)
        if args.stage in {"budget_anon", "all"}:
            run_stage_budget_anon(contexts, max_workers=args.max_workers, resume=resume)

    build_gate2_report()
    print(f"\n[Gate 2 finished in {time.time() - t0:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
