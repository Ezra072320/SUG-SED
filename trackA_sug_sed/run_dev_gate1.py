"""Gate 1 Benchmark: Strictly DEV-Only (M_02..M_05) Validation Before Held-Out Runs.

Evaluates the redesigned Pure-Star-Meso framework against all 10 published paper
baselines, internal 0-LLM / random controls, 6 pillar ablations, cross-day
ProtoOnly frontier (0.65..0.95), and 3 open-source LLMs (Qwen3-8B, Qwen2.5-7B,
Llama-3.1-8B-Instruct). NEVER touches held-out blocks M_06+.
"""

from __future__ import annotations

import os

for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset, load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import MessageGraph, build_message_graph
from core_shared.key_loader import load_provider_key
from core_shared.metrics_audit import full_partition_audit
from core_shared.token_ledger import CausalUniformBucket, TokenLedger
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
from trackA_sug_sed.embed_cache import load_or_extract_block_embeddings
from trackA_sug_sed.llm_client import BudgetedLLMClient
from trackA_sug_sed.star_meso import run_pure_star_meso

DEV_EVAL_BLOCKS = [2, 3, 4, 5]
ALL_DEV_STREAM_BLOCKS = [0, 1, 2, 3, 4, 5]  # M_00..M_01 are 0-token cross-day memory seeds
BUDGET_PER_DEV_BLOCK = 10000

EMBED_MODELS = {
    "Event2012": "sentence-transformers/all-MiniLM-L6-v2",
    "Event2018": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

OUT_DIR = ROOT / "results" / "dev_gate1"


def _assert_dev_only(block_ids: Sequence[int]) -> None:
    if any(int(b) >= 6 for b in block_ids):
        raise RuntimeError(f"GATE 1 GUARDRAIL: held-out block >= 6 requested: {block_ids}")


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
    calls = toks = shown = posts_tot = errs = 0
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
            "ami": round(ami, 2),
            "macro_b3": round(mac, 2),
            "micro_b3": round(mic, 2),
            "tail_b3": round(tail, 2),
            "ari": round(ari, 2),
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
            api_s += float(st.get("llm_api_sec", 0.0))
            wall_s += float(st.get("total_wall_sec", st.get("runtime_sec", 0.0)))
            cpu_s += float(st.get("router_cpu_sec", max(0.0, float(st.get("total_wall_sec", st.get("runtime_sec", 0.0))) - float(st.get("llm_api_sec", 0.0)))))
            b_entry["stats"] = st
        per_block[f"M_{bid:02d}"] = b_entry

    summary: Dict[str, Any] = {
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
        "router_cpu_sec": round(cpu_s, 3),
        "llm_api_sec": round(api_s, 3),
        "total_wall_sec": round(wall_s, 3),
        "per_block": per_block,
    }
    if unq_nets:
        summary["unq_net_mean"] = round(float(np.mean(unq_nets)), 1)
        summary["unq_delta_micro_mean"] = round(float(np.mean(unq_dmics)), 2)
    return summary


def _prepare_dev_context(dataset_name: str, block_ids: Sequence[int]) -> Dict[str, Any]:
    _assert_dev_only(block_ids)
    ds_full = load_socialed_npy(dataset_name, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
    embed_model = EMBED_MODELS[dataset_name]

    ds_map: Dict[int, SocialDataset] = {}
    emb_map: Dict[int, np.ndarray] = {}
    graph_pure_map: Dict[int, MessageGraph] = {}
    init_pure_map: Dict[int, np.ndarray] = {}
    graph_def_map: Dict[int, MessageGraph] = {}
    init_def_map: Dict[int, np.ndarray] = {}

    print(f"\n[Prepare Dev Context] {dataset_name} blocks={list(block_ids)} ({embed_model})", flush=True)
    for bid in block_ids:
        _bname, bidx = blocks[bid]
        sd = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        em = load_or_extract_block_embeddings(dataset_name, bid, sd.texts, model_name=embed_model)
        g_pure = build_message_graph(
            sd, em, knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5, compute_non_entity_tfidf=False
        )
        l_pure, _ = run_hisevent(
            dataset_name, bid, sd, em, knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5, embed_model=embed_model
        )
        g_def = build_message_graph(
            sd, em, knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.5, compute_non_entity_tfidf=False
        )
        l_def, _ = run_hisevent(
            dataset_name, bid, sd, em, knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.5, embed_model=embed_model
        )

        ds_map[bid] = sd
        emb_map[bid] = em
        graph_pure_map[bid] = g_pure
        init_pure_map[bid] = l_pure
        graph_def_map[bid] = g_def
        init_def_map[bid] = l_def
        print(f"  M_{bid:02d} n={len(sd)} events={len(set(sd.labels.tolist()))} | K_pure={len(set(l_pure.tolist()))} K_def={len(set(l_def.tolist()))}", flush=True)

    return {
        "dataset": dataset_name,
        "embed_model": embed_model,
        "ds": ds_map,
        "emb": emb_map,
        "graph_pure": graph_pure_map,
        "init_pure": init_pure_map,
        "graph_def": graph_def_map,
        "init_def": init_def_map,
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


# ---------------------------------------------------------------------------
# Stage 1: Zero-LLM Published SED Papers + Internal 0-Token Controls
# ---------------------------------------------------------------------------

def run_stage_0tok(contexts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for ds_name, ctx in contexts.items():
        print(f"\n=== [Stage 0-Tok] {ds_name} Dev (M_02..M_05) ===", flush=True)
        bids = DEV_EVAL_BLOCKS
        dss = [ctx["ds"][b] for b in bids]
        embs = [ctx["emb"][b] for b in bids]
        inits_pure = [ctx["init_pure"][b] for b in bids]
        graphs_pure = [ctx["graph_pure"][b] for b in bids]
        inits_def = [ctx["init_def"][b] for b in bids]
        graphs_def = [ctx["graph_def"][b] for b in bids]

        ds_res: Dict[str, Any] = {}

        # Paper 1: EventX sweep on Dev
        best_evx = None
        evx_sweep = []
        for res_v in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            preds, sts = zip(*[run_eventx(ds, resolution=res_v) for ds in dss])
            agg = _aggregate_block_metrics(bids, dss, preds, stats_list=sts)
            evx_sweep.append({"resolution": res_v, "ami": agg["ami"], "macro_b3": agg["macro_b3"], "tail_b3": agg["tail_b3"]})
            if best_evx is None or (agg["ami"] + agg["macro_b3"]) > (best_evx["ami"] + best_evx["macro_b3"]):
                best_evx = agg
        best_evx["dev_sweep"] = evx_sweep
        ds_res["eventx_tkdd20"] = best_evx
        print(f"  [1] EventX (TKDD'20)      | AMI={best_evx['ami']:5.2f} Mac={best_evx['macro_b3']:5.2f} Mic={best_evx['micro_b3']:5.2f} Tail={best_evx['tail_b3']:5.2f}")

        # Paper 2: SBERT Agglom sweep on Dev
        best_sbert = None
        sb_sweep = []
        for dth in [0.45, 0.50, 0.55, 0.60, 0.65]:
            preds, sts = zip(*[run_sbert_agglom(em, distance_threshold=dth) for em in embs])
            agg = _aggregate_block_metrics(bids, dss, preds, stats_list=sts)
            sb_sweep.append({"distance_threshold": dth, "ami": agg["ami"], "macro_b3": agg["macro_b3"], "tail_b3": agg["tail_b3"]})
            if best_sbert is None or (agg["ami"] + agg["macro_b3"]) > (best_sbert["ami"] + best_sbert["macro_b3"]):
                best_sbert = agg
        best_sbert["dev_sweep"] = sb_sweep
        ds_res["sbert_emnlp19"] = best_sbert
        print(f"  [2] SBERT-Agg (EMNLP'19)  | AMI={best_sbert['ami']:5.2f} Mac={best_sbert['macro_b3']:5.2f} Mic={best_sbert['micro_b3']:5.2f} Tail={best_sbert['tail_b3']:5.2f}")

        # Paper 3: HiSEvent Default & Dev-Tuned
        agg_his_def = _aggregate_block_metrics(bids, dss, inits_def)
        ds_res["hisevent_default"] = agg_his_def
        print(f"  [3a] HiSEvent Default     | AMI={agg_his_def['ami']:5.2f} Mac={agg_his_def['macro_b3']:5.2f} Mic={agg_his_def['micro_b3']:5.2f} Tail={agg_his_def['tail_b3']:5.2f}")

        best_his_tuned = None
        his_sweep = []
        for (tsim, aa) in [
            (0.35, 0.5),
            (0.45, 0.5),
            (0.45, 0.8),
            (0.50, 0.5),
            (0.50, 0.8),
            (0.50, 1.0),
            (0.55, 0.5),
            (0.55, 0.8),
            (0.55, 1.0),
            (0.55, 1.2),
            (0.60, 0.5),
            (0.60, 0.8),
            (0.60, 1.0),
        ]:
            preds, sts = zip(*[run_hisevent(ds_name, b, ctx["ds"][b], ctx["emb"][b], knn_sim_threshold=tsim, attr_alpha=aa) for b in bids])
            agg = _aggregate_block_metrics(bids, dss, preds, stats_list=sts)
            his_sweep.append({"tau_sim": tsim, "attr_alpha": aa, "ami": agg["ami"], "macro_b3": agg["macro_b3"], "tail_b3": agg["tail_b3"]})
            if best_his_tuned is None or (agg["ami"] + agg["macro_b3"]) > (best_his_tuned["ami"] + best_his_tuned["macro_b3"]):
                best_his_tuned = agg
        best_his_tuned["dev_sweep"] = his_sweep
        ds_res["hisevent_dev_tuned"] = best_his_tuned
        print(f"  [3b] HiSEvent Dev-Tuned   | AMI={best_his_tuned['ami']:5.2f} Mac={best_his_tuned['macro_b3']:5.2f} Mic={best_his_tuned['micro_b3']:5.2f} Tail={best_his_tuned['tail_b3']:5.2f}")

        # Paper 4: HyperSED (AAAI 2025)
        best_hyp = None
        hyp_sweep = []
        for ar in [0.10, 0.14, 0.18, 0.22, 0.26]:
            preds, sts = zip(*[run_hypersed(ds, em, graph=g, anchor_ratio=ar, epochs=60) for ds, em, g in zip(dss, embs, graphs_pure)])
            agg = _aggregate_block_metrics(bids, dss, preds, stats_list=sts)
            hyp_sweep.append({"anchor_ratio": ar, "ami": agg["ami"], "macro_b3": agg["macro_b3"], "tail_b3": agg["tail_b3"], "wall": agg["total_wall_sec"]})
            if best_hyp is None or (agg["ami"] + agg["macro_b3"]) > (best_hyp["ami"] + best_hyp["macro_b3"]):
                best_hyp = agg
        best_hyp["dev_sweep"] = hyp_sweep
        ds_res["hypersed_aaai25"] = best_hyp
        print(f"  [4] HyperSED (AAAI'25)    | AMI={best_hyp['ami']:5.2f} Mac={best_hyp['macro_b3']:5.2f} Mic={best_hyp['micro_b3']:5.2f} Tail={best_hyp['tail_b3']:5.2f} ({best_hyp['total_wall_sec']}s)")

        # Internal 0-LLM baselines on Pure 2D-SE (0.50, 0.5)
        agg_pure = _aggregate_block_metrics(bids, dss, inits_pure)
        ds_res["ctrl_pure_2dse_0tok"] = agg_pure
        print(f"  [C0] Pure 2D-SE Base(0.5) | AMI={agg_pure['ami']:5.2f} Mac={agg_pure['macro_b3']:5.2f} Mic={agg_pure['micro_b3']:5.2f} Tail={agg_pure['tail_b3']:5.2f}")

        am_preds, am_sts = zip(*[run_automerge_only(ds, em, lp, tau_auto=0.85, guard_sim=0.45) for ds, em, lp in zip(dss, embs, inits_pure)])
        agg_am = _aggregate_block_metrics(bids, dss, am_preds, init_list=inits_pure, stats_list=am_sts)
        ds_res["ctrl_automerge_only"] = agg_am
        print(f"  [C1] AutoMerge-Only(0.85) | AMI={agg_am['ami']:5.2f} Mac={agg_am['macro_b3']:5.2f} Mic={agg_am['micro_b3']:5.2f} Tail={agg_am['tail_b3']:5.2f}")

        best_mh = None
        for tm in [0.75, 0.80, 0.85]:
            for ss_th in [0.22, 0.28, 0.99]:  # 0.99 = merge-only heur
                preds, sts = zip(*[
                    run_meso_heur_best(ds, em, g, lp, tau_merge=tm, min_seam_sep=ss_th)
                    for ds, em, g, lp in zip(dss, embs, graphs_pure, inits_pure)
                ])
                agg = _aggregate_block_metrics(bids, dss, preds, init_list=inits_pure, stats_list=sts)
                if best_mh is None or (agg["ami"] + agg["macro_b3"]) > (best_mh["ami"] + best_mh["macro_b3"]):
                    best_mh = agg
                    best_mh["cfg"] = {"tau_merge": tm, "min_seam_sep": ss_th}
        ds_res["ctrl_meso_heur_pure"] = best_mh
        print(f"  [C2] Meso-Heur-Best(Pure) | AMI={best_mh['ami']:5.2f} Mac={best_mh['macro_b3']:5.2f} Mic={best_mh['micro_b3']:5.2f} Tail={best_mh['tail_b3']:5.2f} cfg={best_mh['cfg']}")

        # Meso-Heur on HiSEvent Default (0.35) for plug-and-play comparison
        best_mh_def = None
        for tm in [0.75, 0.80, 0.85]:
            preds, sts = zip(*[
                run_meso_heur_best(ds, em, g, ld, tau_merge=tm, min_seam_sep=0.99)
                for ds, em, g, ld in zip(dss, embs, graphs_def, inits_def)
            ])
            agg = _aggregate_block_metrics(bids, dss, preds, init_list=inits_def, stats_list=sts)
            if best_mh_def is None or (agg["ami"] + agg["macro_b3"]) > (best_mh_def["ami"] + best_mh_def["macro_b3"]):
                best_mh_def = agg
                best_mh_def["cfg"] = {"tau_merge": tm}
        ds_res["ctrl_meso_heur_def"] = best_mh_def
        print(f"  [C2b] Meso-Heur-Best(Def) | AMI={best_mh_def['ami']:5.2f} Mac={best_mh_def['macro_b3']:5.2f} Mic={best_mh_def['micro_b3']:5.2f} Tail={best_mh_def['tail_b3']:5.2f}")

        results[ds_name] = ds_res

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stage_0tok.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Stage 2: Qwen/Qwen3-8B Full Published LLM Baselines + Pure-Star-Meso + Ablations + Cross-Day
# ---------------------------------------------------------------------------

def _eval_llm_runner_across_dev(
    tag: str,
    bids: Sequence[int],
    dss: Sequence[SocialDataset],
    init_list: Sequence[np.ndarray],
    fn_per_block,
    budget_per_blk: int = BUDGET_PER_DEV_BLOCK,
    max_workers: int = 4,
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """Run all 4 Dev blocks concurrently (independent per-block causal buckets -> 4x faster)."""
    def _one(i: int, bid: int):
        bk = CausalUniformBucket(tokens_per_block=budget_per_blk, allow_carryover=False)
        bk.start_block()
        return fn_per_block(i, bid, bk)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(bids))) as pool:
        futs = [pool.submit(_one, i, bid) for i, bid in enumerate(bids)]
        outs = [f.result() for f in futs]

    preds = [o[0] for o in outs]
    showns = [o[1] for o in outs]
    sts = [o[2] for o in outs]
    agg = _aggregate_block_metrics(bids, dss, preds, init_list=init_list, shown_list=showns, stats_list=sts)
    print(
        f"  [{tag:28s}] calls={agg['total_calls']:3d} tok={agg['total_tokens']:6d} ({agg['tok_per_post']:.2f}/p) "
        f"show={agg['shown_pct']:4.2f}% err={agg['parse_errors']} | "
        f"AMI={agg['ami']:5.2f} Mac={agg['macro_b3']:5.2f} Mic={agg['micro_b3']:5.2f} Tail={agg['tail_b3']:5.2f} UnqNet={agg.get('unq_net_mean', 0):+6.1f}",
        flush=True,
    )
    return agg, preds


def run_stage_qwen3(contexts: Dict[str, Dict[str, Any]], reuse_baselines: bool = False) -> Dict[str, Any]:
    model_name = "Qwen/Qwen3-8B"
    shared_cache = OUT_DIR / "ledgers" / "qwen3_8b_shared_responses.jsonl"
    out_path = OUT_DIR / "stage_qwen3.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prev_data = json.loads(out_path.read_text(encoding="utf-8")) if (reuse_baselines and out_path.exists()) else {}
    results: Dict[str, Any] = {"model": model_name, "provider": "siliconflow"}

    def _ckpt() -> None:
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for ds_name, ctx in contexts.items():
        print(f"\n=== [Stage Qwen3-8B] {ds_name} Dev (10k tok/blk, 4-block parallel) ===", flush=True)
        bids = DEV_EVAL_BLOCKS
        dss = [ctx["ds"][b] for b in bids]
        embs = [ctx["emb"][b] for b in bids]
        graphs_p = [ctx["graph_pure"][b] for b in bids]
        inits_p = [ctx["init_pure"][b] for b in bids]
        graphs_d = [ctx["graph_def"][b] for b in bids]
        inits_d = [ctx["init_def"][b] for b in bids]
        embed_model = ctx["embed_model"]

        def _client(sub: str) -> BudgetedLLMClient:
            return _make_client("siliconflow", model_name, OUT_DIR / "ledgers" / "qwen3_8b" / ds_name / sub, shared_cache)

        prev_ds = prev_data.get(ds_name, {}) if reuse_baselines else {}
        ds_out: Dict[str, Any] = dict(prev_ds)
        results[ds_name] = ds_out

        # --- 1. OUR PRIMARY METHOD FIRST (fast early signal) + Plug-and-Play + 5 Clean Ablations ---
        main_dev_preds: Optional[List[np.ndarray]] = None
        if reuse_baselines and "our_pure_star_meso" in prev_ds:
            ds_out["our_pure_star_meso"] = prev_ds["our_pure_star_meso"]
        else:
            cl_main = _client("pure_star_meso")
            main_agg, main_dev_preds = _eval_llm_runner_across_dev(
                "OUR.Pure-Star-Meso(Pure)", bids, dss, inits_p,
                lambda i, b, bk, _c=cl_main: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(
                        ds_name, b, dss[i], embs[i], _c, bk,
                        strategy_tag="pure_star_meso",
                        graph=graphs_p[i], labels_init=inits_p[i],
                    )
                ),
            )
            ds_out["our_pure_star_meso"] = main_agg
            _ckpt()

        if reuse_baselines and "our_star_meso_on_hisevent_def" in prev_ds:
            ds_out["our_star_meso_on_hisevent_def"] = prev_ds["our_star_meso_on_hisevent_def"]
        else:
            cl_def = _client("star_on_def")
            ds_out["our_star_meso_on_hisevent_def"], _ = _eval_llm_runner_across_dev(
                "OUR.Star-Meso(HiSEventDef)", bids, dss, inits_d,
                lambda i, b, bk, _c=cl_def: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(
                        ds_name, b, dss[i], embs[i], _c, bk,
                        strategy_tag="star_meso_def", use_purity_base=False,
                        graph=graphs_d[i], labels_init=inits_d[i],
                    )
                ),
            )
            _ckpt()

        abl_specs = [
            ("abl_no_automerge", dict(use_auto_merge=False)),
            ("abl_no_bayes_band", dict(use_bayes_band=False)),
            ("abl_no_star_packing", dict(use_star_packing=False)),
            ("abl_no_guarded_uf", dict(use_guarded_uf=False)),
            ("abl_no_seam_split", dict(max_seam_calls=0)),
        ]
        for abl_key, kw in abl_specs:
            if reuse_baselines and abl_key in prev_ds:
                ds_out[abl_key] = prev_ds[abl_key]
                continue
            cl_ab = _client(abl_key)
            ds_out[abl_key], _ = _eval_llm_runner_across_dev(
                abl_key, bids, dss, inits_p,
                lambda i, b, bk, _k=abl_key, _kw=kw, _c=cl_ab: (lambda r: (r[0], r[2], r[3]))(
                    run_pure_star_meso(
                        ds_name, b, dss[i], embs[i], _c, bk,
                        strategy_tag=_k, graph=graphs_p[i], labels_init=inits_p[i], **_kw,
                    )
                ),
            )
            _ckpt()

        # --- 2. Dev Cross-Day Evaluation (M_00..M_05, eval M_02..M_05) ---
        if reuse_baselines and "cross_day_dev" in prev_ds:
            ds_out["cross_day_dev"] = prev_ds["cross_day_dev"]
        else:
            if main_dev_preds is None:
                cl_main = _client("pure_star_meso")
                _, main_dev_preds = _eval_llm_runner_across_dev(
                    "OUR.Pure-Star-Meso(Pure)", bids, dss, inits_p,
                    lambda i, b, bk, _c=cl_main: (lambda r: (r[0], r[2], r[3]))(
                        run_pure_star_meso(
                            ds_name, b, dss[i], embs[i], _c, bk,
                            strategy_tag="pure_star_meso",
                            graph=graphs_p[i], labels_init=inits_p[i],
                        )
                    ),
                )
            print("  --- Dev Cross-Day Stream (M_00..M_05, eval M_02..M_05) ---", flush=True)
            pred_by_bid = {0: ctx["init_pure"][0], 1: ctx["init_pure"][1]}
            for b, pr in zip(bids, main_dev_preds):
                pred_by_bid[b] = pr
            stream = [(b, ctx["ds"][b], ctx["emb"][b], pred_by_bid[b]) for b in ALL_DEV_STREAM_BLOCKS]

            cd_res: Dict[str, Any] = {}
            cd_res["no_link"] = evaluate_cross_day_stream(stream, "no_link", eval_start_block=2)
            for tau_p in [0.60, 0.65, 0.72, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]:
                cd_res[f"proto_{tau_p:.2f}"] = evaluate_cross_day_stream(stream, "proto_only", eval_start_block=2, tau_proto=tau_p)
            for tau_pa in [0.80, 0.85, 0.90, 0.95]:
                cd_res[f"proto_attr_{tau_pa:.2f}"] = evaluate_cross_day_stream(stream, "proto_attr_only", eval_start_block=2, tau_proto=tau_pa, attr_weight=0.15)
            cd_res["sug_cross_star"] = evaluate_cross_day_stream(
                stream, "sug_cross_star", eval_start_block=2, tau_lo=0.65, tau_hi=0.90, tau_fb=0.90, min_cross_size=3,
                client=_client("cross_star"), budget_per_block=6000,
            )
            cd_res["oracle_band"] = evaluate_cross_day_stream(
                stream, "oracle_band", eval_start_block=2, tau_lo=0.65, tau_hi=0.90, tau_fb=0.90, min_cross_size=3,
            )
            ds_out["cross_day_dev"] = cd_res
            for ck in ["no_link", "proto_0.60", "proto_0.80", "proto_0.85", "proto_0.88", "proto_0.90", "proto_0.92", "proto_attr_0.90", "sug_cross_star", "oracle_band"]:
                if ck in cd_res:
                    r = cd_res[ck]
                    print(f"    {ck:16s} | G-AMI={r['g_ami']:5.2f} G-MacB3={r['g_mac_b3']:5.2f} G-MicB3={r['g_mic_b3']:5.2f} Q={r['stats']['llm_queried']}", flush=True)
            _ckpt()

        # --- 3. 6 Published LLM Paper Baselines + Random(3-seed) Control ---
        if reuse_baselines and "clusterllm_emnlp23" in prev_ds:
            ds_out["clusterllm_emnlp23"] = prev_ds["clusterllm_emnlp23"]
        else:
            cl_cllm = _client("clusterllm")
            sbert_inits = [run_sbert_agglom(em, distance_threshold=0.60)[0] for em in embs]
            ds_out["clusterllm_emnlp23"], _ = _eval_llm_runner_across_dev(
                "5.ClusterLLM-NoFT(23)",
                bids, dss, sbert_inits,
                lambda i, b, bk, _c=cl_cllm, _si=sbert_inits: run_clusterllm(b, dss[i], embs[i], _si[i], _c, bk),
            )
            _ckpt()

        if reuse_baselines and "fewshot_post_tacl24" in prev_ds:
            ds_out["fewshot_post_tacl24"] = prev_ds["fewshot_post_tacl24"]
        else:
            cl_fs = _client("fewshot")
            ds_out["fewshot_post_tacl24"], _ = _eval_llm_runner_across_dev(
                "6.FewShotPost(TACL24)",
                bids, dss, inits_p,
                lambda i, b, bk, _c=cl_fs: run_fewshot_llm_post(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk),
            )
            _ckpt()

        if reuse_baselines and "cequel_edge_25" in prev_ds and "cequel_trans_25" in prev_ds:
            ds_out["cequel_edge_25"] = prev_ds["cequel_edge_25"]
            ds_out["cequel_trans_25"] = prev_ds["cequel_trans_25"]
        else:
            cl_ceq = _client("cequel")
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [
                    pool.submit(
                        lambda i=i, bid=bid: (
                            lambda bk: (bk.start_block(), run_cequel(bid, dss[i], embs[i], graphs_p[i], inits_p[i], cl_ceq, bk))[1]
                        )(CausalUniformBucket(tokens_per_block=BUDGET_PER_DEV_BLOCK, allow_carryover=False))
                    )
                    for i, bid in enumerate(bids)
                ]
                ceq_outs = [f.result() for f in futs]
            c_edge_preds = [o[0] for o in ceq_outs]
            c_trans_preds = [o[1] for o in ceq_outs]
            c_showns = [o[2] for o in ceq_outs]
            c_sts = [o[3] for o in ceq_outs]
            ds_out["cequel_edge_25"] = _aggregate_block_metrics(bids, dss, c_edge_preds, init_list=inits_p, shown_list=c_showns, stats_list=c_sts)
            ds_out["cequel_trans_25"] = _aggregate_block_metrics(bids, dss, c_trans_preds, init_list=inits_p, shown_list=c_showns, stats_list=c_sts)
            for tag_c, ag_c in [("7a.Cequel-Edge(25)", ds_out["cequel_edge_25"]), ("7b.Cequel-Trans(25)", ds_out["cequel_trans_25"])]:
                print(f"  [{tag_c:28s}] calls={ag_c['total_calls']:3d} tok={ag_c['total_tokens']:6d} show={ag_c['shown_pct']:4.2f}% | AMI={ag_c['ami']:5.2f} Mac={ag_c['macro_b3']:5.2f} Tail={ag_c['tail_b3']:5.2f} UnqNet={ag_c.get('unq_net_mean',0):+6.1f}", flush=True)
            _ckpt()

        if reuse_baselines and "lsed_acl25" in prev_ds:
            ds_out["lsed_acl25"] = prev_ds["lsed_acl25"]
        else:
            cl_lsed = _client("lsed")
            ds_out["lsed_acl25"], _ = _eval_llm_runner_across_dev(
                "8.LSED-Bud(ACL25)",
                bids, dss, inits_p,
                lambda i, b, bk, _c=cl_lsed: run_lsed_budgeted(b, dss[i], embs[i], inits_p[i], embed_model, _c, bk, knn_sim_threshold=0.50),
            )
            _ckpt()

        if reuse_baselines and "ragsede_www26" in prev_ds:
            ds_out["ragsede_www26"] = prev_ds["ragsede_www26"]
        else:
            cl_rag = _client("ragsede")
            ds_out["ragsede_www26"], _ = _eval_llm_runner_across_dev(
                "9.RagSEDE-Bud(WWW26)",
                bids, dss, inits_p,
                lambda i, b, bk, _c=cl_rag: run_ragsede_budgeted(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk),
            )
            _ckpt()

        if reuse_baselines and "lhg_www26" in prev_ds:
            ds_out["lhg_www26"] = prev_ds["lhg_www26"]
        else:
            cl_lhg = _client("lhg")
            ds_out["lhg_www26"], _ = _eval_llm_runner_across_dev(
                "10.LHG-Bud(WWW26)",
                bids, dss, inits_p,
                lambda i, b, bk, _c=cl_lhg: run_lhg_budgeted(b, dss[i], embs[i], graphs_p[i], inits_p[i], embed_model, _c, bk),
            )
            _ckpt()

        if reuse_baselines and "ctrl_random_mean3" in prev_ds:
            ds_out["ctrl_random_mean3"] = prev_ds["ctrl_random_mean3"]
        else:
            rand_runs = []
            for sd_i in [42, 100, 200]:
                cl_rd = _client(f"rand_s{sd_i}")
                agg_rd, _ = _eval_llm_runner_across_dev(
                    f"Ctrl.Random(s={sd_i})", bids, dss, inits_p,
                    lambda i, b, bk, _s=sd_i, _c=cl_rd: run_random_llm(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk, seed=_s),
                )
                rand_runs.append(agg_rd)
            ds_out["ctrl_random_mean3"] = {
                k: round(float(np.mean([r[k] for r in rand_runs])), 2)
                for k in ["ami", "macro_b3", "micro_b3", "tail_b3", "total_calls", "total_tokens", "shown_pct", "unq_net_mean"]
            }
            _ckpt()

        # --- 4. Random-Meso (3-seed) Control (uses latest Pure-Star-Meso domain) ---
        if reuse_baselines and "ctrl_random_meso_mean3" in prev_ds:
            ds_out["ctrl_random_meso_mean3"] = prev_ds["ctrl_random_meso_mean3"]
        else:
            rmeso_runs = []
            for sd_i in [42, 100, 200]:
                cl_rm = _client(f"rmeso_s{sd_i}")
                agg_rm, _ = _eval_llm_runner_across_dev(
                    f"Ctrl.RandomMeso(s={sd_i})", bids, dss, inits_p,
                    lambda i, b, bk, _s=sd_i, _c=cl_rm: (lambda r: (r[0], r[2], r[3]))(
                        run_pure_star_meso(
                            ds_name, b, dss[i], embs[i], _c, bk,
                            strategy_tag="random_meso", random_order=True, seed=_s,
                            graph=graphs_p[i], labels_init=inits_p[i],
                        )
                    ),
                )
                rmeso_runs.append(agg_rm)
            ds_out["ctrl_random_meso_mean3"] = {
                k: round(float(np.mean([r[k] for r in rmeso_runs])), 2)
                for k in ["ami", "macro_b3", "micro_b3", "tail_b3", "total_calls", "total_tokens", "shown_pct", "unq_net_mean"]
            }
        _ckpt()

    return results


# ---------------------------------------------------------------------------
# Stage 3: Cross-Model Generalization (Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct)
# ---------------------------------------------------------------------------

def run_stage_cross_model(
    contexts: Dict[str, Dict[str, Any]],
    model_tags: Optional[Sequence[str]] = None,
    reuse_existing: bool = False,
) -> Dict[str, Any]:
    specs = [
        ("qwen25_7b", "siliconflow", "Qwen/Qwen2.5-7B-Instruct"),
        ("llama31_8b", "cloudflare", "@cf/meta/llama-3.1-8b-instruct-fast"),
    ]
    if model_tags is not None:
        tag_set = {t.strip() for t in model_tags if t.strip()}
        specs = [s for s in specs if s[0] in tag_set]
    out_path = OUT_DIR / "stage_cross_model.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = (
        json.loads(out_path.read_text(encoding="utf-8"))
        if (reuse_existing and out_path.exists())
        else {}
    )

    def _ckpt() -> None:
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for m_tag, prov, m_name in specs:
        print(f"\n=== [Stage Cross-Model] {m_name} ({prov}) ===", flush=True)
        shared_cache = OUT_DIR / "ledgers" / f"{m_tag}_shared_responses.jsonl"
        m_res: Dict[str, Any] = dict(results.get(m_tag, {})) if reuse_existing else {}
        m_res["model"] = m_name
        m_res["provider"] = prov
        results[m_tag] = m_res
        for ds_name, ctx in contexts.items():
            bids = DEV_EVAL_BLOCKS
            dss = [ctx["ds"][b] for b in bids]
            embs = [ctx["emb"][b] for b in bids]
            graphs_p = [ctx["graph_pure"][b] for b in bids]
            inits_p = [ctx["init_pure"][b] for b in bids]
            embed_model = ctx["embed_model"]
            print(f"  -- {ds_name} --", flush=True)

            def _cl(sub: str) -> BudgetedLLMClient:
                return _make_client(prov, m_name, OUT_DIR / "ledgers" / m_tag / ds_name / sub, shared_cache)

            ds_m: Dict[str, Any] = dict(m_res.get(ds_name, {})) if reuse_existing else {}
            m_res[ds_name] = ds_m

            if not (reuse_existing and "our_pure_star_meso" in ds_m):
                cl_main = _cl("pure_star_meso")
                ds_m["our_pure_star_meso"], _ = _eval_llm_runner_across_dev(
                    f"{m_tag}.OUR.PureStarMeso", bids, dss, inits_p,
                    lambda i, b, bk, _c=cl_main: (lambda r: (r[0], r[2], r[3]))(
                        run_pure_star_meso(
                            ds_name, b, dss[i], embs[i], _c, bk,
                            strategy_tag="pure_star_meso",
                            graph=graphs_p[i], labels_init=inits_p[i],
                        )
                    ),
                )
                _ckpt()

            if not (reuse_existing and "clusterllm_emnlp23" in ds_m):
                cl_cllm = _cl("clusterllm")
                sbert_inits = [run_sbert_agglom(em, distance_threshold=0.60)[0] for em in embs]
                ds_m["clusterllm_emnlp23"], _ = _eval_llm_runner_across_dev(
                    f"{m_tag}.5.ClusterLLM-NoFT",
                    bids, dss, sbert_inits,
                    lambda i, b, bk, _c=cl_cllm, _si=sbert_inits: run_clusterllm(b, dss[i], embs[i], _si[i], _c, bk),
                )
                _ckpt()

            if not (reuse_existing and "fewshot_post_tacl24" in ds_m):
                cl_fs = _cl("fewshot")
                ds_m["fewshot_post_tacl24"], _ = _eval_llm_runner_across_dev(
                    f"{m_tag}.6.FewShotPost", bids, dss, inits_p,
                    lambda i, b, bk, _c=cl_fs: run_fewshot_llm_post(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk),
                )
                _ckpt()

            if not (reuse_existing and "cequel_edge_25" in ds_m and "cequel_trans_25" in ds_m):
                cl_ceq = _cl("cequel")
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futs = [
                        pool.submit(
                            lambda i=i, bid=bid: (
                                lambda bk: (bk.start_block(), run_cequel(bid, dss[i], embs[i], graphs_p[i], inits_p[i], cl_ceq, bk))[1]
                            )(CausalUniformBucket(tokens_per_block=BUDGET_PER_DEV_BLOCK, allow_carryover=False))
                        )
                        for i, bid in enumerate(bids)
                    ]
                    ceq_outs = [f.result() for f in futs]
                c_e = [o[0] for o in ceq_outs]
                c_t = [o[1] for o in ceq_outs]
                c_sh = [o[2] for o in ceq_outs]
                c_st = [o[3] for o in ceq_outs]
                ds_m["cequel_edge_25"] = _aggregate_block_metrics(bids, dss, c_e, init_list=inits_p, shown_list=c_sh, stats_list=c_st)
                ds_m["cequel_trans_25"] = _aggregate_block_metrics(bids, dss, c_t, init_list=inits_p, shown_list=c_sh, stats_list=c_st)
                print(f"  [{m_tag}.7a.Cequel-Edge  ] AMI={ds_m['cequel_edge_25']['ami']:5.2f} Mac={ds_m['cequel_edge_25']['macro_b3']:5.2f} Tail={ds_m['cequel_edge_25']['tail_b3']:5.2f}", flush=True)
                print(f"  [{m_tag}.7b.Cequel-Trans ] AMI={ds_m['cequel_trans_25']['ami']:5.2f} Mac={ds_m['cequel_trans_25']['macro_b3']:5.2f} Tail={ds_m['cequel_trans_25']['tail_b3']:5.2f}", flush=True)
                _ckpt()

            if not (reuse_existing and "lsed_acl25" in ds_m):
                cl_lsed = _cl("lsed")
                ds_m["lsed_acl25"], _ = _eval_llm_runner_across_dev(
                    f"{m_tag}.8.LSED-Bud",
                    bids, dss, inits_p,
                    lambda i, b, bk, _c=cl_lsed: run_lsed_budgeted(b, dss[i], embs[i], inits_p[i], embed_model, _c, bk, knn_sim_threshold=0.50),
                )
                _ckpt()

            if not (reuse_existing and "ragsede_www26" in ds_m):
                cl_rag = _cl("ragsede")
                ds_m["ragsede_www26"], _ = _eval_llm_runner_across_dev(
                    f"{m_tag}.9.RagSEDE-Bud",
                    bids, dss, inits_p,
                    lambda i, b, bk, _c=cl_rag: run_ragsede_budgeted(b, dss[i], embs[i], graphs_p[i], inits_p[i], _c, bk),
                )
                _ckpt()

            if not (reuse_existing and "lhg_www26" in ds_m):
                cl_lhg = _cl("lhg")
                ds_m["lhg_www26"], _ = _eval_llm_runner_across_dev(
                    f"{m_tag}.10.LHG-Bud",
                    bids, dss, inits_p,
                    lambda i, b, bk, _c=cl_lhg: run_lhg_budgeted(b, dss[i], embs[i], graphs_p[i], inits_p[i], embed_model, _c, bk),
                )
                _ckpt()

            if not (reuse_existing and "ctrl_random_meso_s42" in ds_m):
                cl_rm = _cl("rmeso42")
                ds_m["ctrl_random_meso_s42"], _ = _eval_llm_runner_across_dev(
                    f"{m_tag}.RandomMeso(s42)", bids, dss, inits_p,
                    lambda i, b, bk, _c=cl_rm: (lambda r: (r[0], r[2], r[3]))(
                        run_pure_star_meso(
                            ds_name, b, dss[i], embs[i], _c, bk,
                            strategy_tag="random_meso_s42", random_order=True, seed=42,
                            graph=graphs_p[i], labels_init=inits_p[i],
                        )
                    ),
                )
                _ckpt()

    return results


def build_gate1_report() -> None:
    s0 = json.loads((OUT_DIR / "stage_0tok.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_0tok.json").exists() else {}
    sq = json.loads((OUT_DIR / "stage_qwen3.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_qwen3.json").exists() else {}
    scm = json.loads((OUT_DIR / "stage_cross_model.json").read_text(encoding="utf-8")) if (OUT_DIR / "stage_cross_model.json").exists() else {}

    lines: List[str] = ["# Gate 1 Report (Strictly DEV M_02..M_05, Zero Held-Out Touched)\n"]
    for ds_name in ["Event2012", "Event2018"]:
        if ds_name not in s0 or ds_name not in sq:
            continue
        lines.append(f"## {ds_name} Dev (`M_02..M_05`)\n")
        lines.append("| Group | Method | Tok | Show% | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet | Wall(s) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        rows_0 = [
            ("0-LLM Paper 1", "EventX (TKDD'20)", s0[ds_name]["eventx_tkdd20"]),
            ("0-LLM Paper 2", "SBERT-Agglom (EMNLP'19)", s0[ds_name]["sbert_emnlp19"]),
            ("0-LLM Paper 3a", "HiSEvent Default (AAAI'24)", s0[ds_name]["hisevent_default"]),
            ("0-LLM Paper 3b", "HiSEvent Dev-Tuned (AAAI'24)", s0[ds_name]["hisevent_dev_tuned"]),
            ("0-LLM Paper 4", "HyperSED (AAAI'25)", s0[ds_name]["hypersed_aaai25"]),
            ("Internal 0-Tok", "Pure 2D-SE Base (0.50)", s0[ds_name]["ctrl_pure_2dse_0tok"]),
            ("Internal 0-Tok", "AutoMerge-Only (0.85)", s0[ds_name]["ctrl_automerge_only"]),
            ("Internal 0-Tok", "Meso-Heur-Best (Pure)", s0[ds_name]["ctrl_meso_heur_pure"]),
        ]
        for grp, nm, r in rows_0:
            lines.append(
                f"| {grp} | {nm} | 0 | 0.00% | {r['ami']:.2f} | {r['macro_b3']:.2f} | {r['micro_b3']:.2f} | {r['tail_b3']:.2f} | {r.get('unq_net_mean', 0):+.1f} | {r['total_wall_sec']:.1f} |"
            )
        rows_q = [
            ("LLM Paper 5", "ClusterLLM-NoFT (EMNLP'23)", sq[ds_name]["clusterllm_emnlp23"]),
            ("LLM Paper 6", "Few-Shot LLM-Post (TACL'24)", sq[ds_name]["fewshot_post_tacl24"]),
            ("LLM Paper 7a", "Cequel-Edge (arXiv'25)", sq[ds_name]["cequel_edge_25"]),
            ("LLM Paper 7b", "Cequel-Transitive (arXiv'25)", sq[ds_name]["cequel_trans_25"]),
            ("LLM Paper 8", "LSED-Budgeted (ACL'25)", sq[ds_name]["lsed_acl25"]),
            ("LLM Paper 9", "RagSEDE-Budgeted (WWW'26)", sq[ds_name]["ragsede_www26"]),
            ("LLM Paper 10", "LHG-Budgeted (WWW'26)", sq[ds_name]["lhg_www26"]),
            ("Internal Ctrl", "Random (3-seed mean)", sq[ds_name]["ctrl_random_mean3"]),
            ("Internal Ctrl", "Random-Meso (3-seed mean)", sq[ds_name]["ctrl_random_meso_mean3"]),
            ("**OUR METHOD**", "**Pure-Star-Meso (Qwen3-8B)**", sq[ds_name]["our_pure_star_meso"]),
            ("Plug-and-Play", "Star-Meso on HiSEvent Default", sq[ds_name]["our_star_meso_on_hisevent_def"]),
        ]
        for grp, nm, r in rows_q:
            lines.append(
                f"| {grp} | {nm} | {int(r['total_tokens'])} | {r['shown_pct']:.2f}% | {r['ami']:.2f} | {r['macro_b3']:.2f} | {r['micro_b3']:.2f} | {r['tail_b3']:.2f} | {r.get('unq_net_mean', 0):+.1f} | {r.get('total_wall_sec', 0):.1f} |"
            )
        lines.append("\n*Note: `ClusterLLM-NoFT (EMNLP'23)` omits contrastive encoder fine-tuning under the zero-training budgeted protocol and computes `UnqNet` relative to its 0-LLM hierarchical base `SBERT-Agglom (0.60)`. `HyperSED` caps `n_anchors <= 360`.*\n")

        lines.append(f"### {ds_name} Pillar Ablations (`Qwen/Qwen3-8B`)\n")
        lines.append("| Variant | Calls | Tok | AMI | Mac-$B^3$ | Tail-$B^3$ | UnqNet |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for key, label in [
            ("our_pure_star_meso", "**Full Pure-Star-Meso**"),
            ("abl_no_automerge", "w/o AutoMerge (>=0.85)"),
            ("abl_no_bayes_band", "w/o Bayes Band & AutoMerge"),
            ("abl_no_star_packing", "w/o Star Packing (k=1 Pair)"),
            ("abl_no_guarded_uf", "w/o Guarded UF"),
            ("abl_no_seam_split", "w/o Seam Split"),
        ]:
            if key not in sq[ds_name]:
                continue
            r = sq[ds_name][key]
            lines.append(
                f"| {label} | {r['total_calls']} | {r['total_tokens']} | {r['ami']:.2f} | {r['macro_b3']:.2f} | {r['tail_b3']:.2f} | {r.get('unq_net_mean', 0):+.1f} |"
            )
        lines.append("")

        if "cross_day_dev" in sq[ds_name]:
            lines.append(f"### {ds_name} Dev Cross-Day Frontier (`M_00..M_05`, eval `M_02..M_05`)\n")
            lines.append("| Mode | G-AMI | G-Mac$B^3$ | G-Mic$B^3$ | G-Tail$B^3$ | Q |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for ck, cr in sq[ds_name]["cross_day_dev"].items():
                lines.append(f"| {ck} | {cr['g_ami']:.2f} | {cr['g_mac_b3']:.2f} | {cr['g_mic_b3']:.2f} | {cr['g_tail_b3']:.2f} | {cr['stats']['llm_queried']} |")
            lines.append("")

    if scm:
        lines.append("## Cross-Model Dev Comparison (`M_02..M_05`)\n")
        lines.append("| Model | Dataset | Method | Calls | Tok | Show% | Err | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m_tag, m_obj in scm.items():
            for ds_name in ["Event2012", "Event2018"]:
                if ds_name not in m_obj:
                    continue
                for k in [
                    "clusterllm_emnlp23",
                    "fewshot_post_tacl24",
                    "cequel_edge_25",
                    "cequel_trans_25",
                    "lsed_acl25",
                    "ragsede_www26",
                    "lhg_www26",
                    "ctrl_random_meso_s42",
                    "our_pure_star_meso",
                ]:
                    if k not in m_obj[ds_name]:
                        continue
                    r = m_obj[ds_name][k]
                    lines.append(
                        f"| {m_obj['model']} | {ds_name} | {k} | {r['total_calls']} | {r['total_tokens']} | {r['shown_pct']:.2f}% | {r['parse_errors']} | {r['ami']:.2f} | {r['macro_b3']:.2f} | {r['micro_b3']:.2f} | {r['tail_b3']:.2f} | {r.get('unq_net_mean', 0):+.1f} |"
                    )

    (OUT_DIR / "dev_gate1_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "dev_gate1_report.json").write_text(json.dumps({"stage_0tok": s0, "stage_qwen3": sq, "stage_cross_model": scm}, indent=2), encoding="utf-8")
    print(f"\nSaved Gate 1 report -> {OUT_DIR / 'dev_gate1_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gate 1 DEV-only SUG-SED Benchmark")
    parser.add_argument("--stage", choices=["0tok", "qwen3", "cross_model", "report", "all"], default="all")
    parser.add_argument("--reuse-baselines", action="store_true", help="Reuse cached published baselines in stage_qwen3.json")
    parser.add_argument(
        "--cross-models",
        type=str,
        default="qwen25_7b,llama31_8b",
        help="Comma-separated cross-model tags to evaluate (qwen25_7b, llama31_8b)",
    )
    args = parser.parse_args()
    cross_tags = [t.strip() for t in args.cross_models.split(",") if t.strip()]

    t_start = time.time()
    if args.stage != "report":
        need_blocks = ALL_DEV_STREAM_BLOCKS if args.stage in {"qwen3", "all"} else DEV_EVAL_BLOCKS
        contexts = {
            "Event2012": _prepare_dev_context("Event2012", need_blocks),
            "Event2018": _prepare_dev_context("Event2018", need_blocks),
        }
        if args.stage in {"0tok", "all"}:
            run_stage_0tok(contexts)
        if args.stage in {"qwen3", "all"}:
            run_stage_qwen3(contexts, reuse_baselines=args.reuse_baselines)
        if args.stage in {"cross_model", "all"}:
            run_stage_cross_model(contexts, model_tags=cross_tags, reuse_existing=args.reuse_baselines)
    build_gate1_report()
    print(f"\n[Gate 1 finished in {time.time() - t_start:.1f}s]")


if __name__ == "__main__":
    main()
