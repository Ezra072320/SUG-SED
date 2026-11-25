"""Track B (`Evid-MSED`) Phase 2 End-to-End Runner on NED (`validation` -> `holdout`).

Executes:
  - Stage 1: All 6 unsupervised 2D-SE variants (`text_only_2dse`, `naive_mv_2dse`,
    `cntd_raw_mv_2dse`, `cntd_cond_mv_2dse`, `visual_idf_no_impute_mv_2dse`,
    `visual_idf_mv_2dse`) on Clean and optional Stress-30% competing-image corruption.
  - Stage 2: High-leverage multimodal seam routing + paired `text_only` & `multimodal`
    VLM adjudication (`Qwen/Qwen3-VL-8B-Instruct` via SiliconFlow) comparing 5 gating
    policies (`text_only_vlm`, `naive_vlm`, `evid_self_gate_inline`,
    `evid_self_gate_clean_fallback`, `evid_comp_gate_clean_fallback`) across budget prefixes.

Enforces strict SHA-256 frozen protocol validation before unlocking `holdout`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.key_loader import load_provider_key
from core_shared.token_ledger import TokenLedger
from trackB_evid_msed.mm_seam_adjudicator import (
    MMSeamQuery,
    decisions_to_llm_decisions,
    route_mm_seams,
)
from trackB_evid_msed.mv_2dse_pipeline import (
    build_text_graph_and_2dse,
    corrupt_images_competing,
    load_split_with_cache,
    run_all_stage1_variants,
)
from trackB_evid_msed.vlm_client import BudgetedVLMClient, SeamVLDecision

OUT_DIR = ROOT / "results" / "trackB_phase2"

POLICIES = [
    "text_only_vlm",
    "naive_vlm",
    "evid_self_gate_inline",
    "evid_self_gate_clean_fallback",
    "evid_comp_gate_clean_fallback",
]


def _verify_protocol_hashes(proto_path: Path) -> Dict[str, Any]:
    if not proto_path.exists():
        raise FileNotFoundError(f"Frozen protocol required for holdout not found: {proto_path}")
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    checksums: Dict[str, str] = proto.get("code_sha256", {})
    if not checksums:
        raise ValueError("Protocol missing code_sha256 map.")
    mismatches = []
    for rel_path, expected in checksums.items():
        fp = ROOT / rel_path
        if not fp.exists():
            mismatches.append(f"{rel_path}: MISSING")
            continue
        actual = hashlib.sha256(fp.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(f"{rel_path}: expected {expected[:12]} got {actual[:12]}")
    if mismatches:
        raise RuntimeError("Protocol SHA-256 verification FAILED:\n  " + "\n  ".join(mismatches))
    print(f"[Protocol OK] Verified {len(checksums)} frozen file SHA-256 digests from {proto_path.name}")
    return proto


def _run_vlm_calls(
    routed: List[MMSeamQuery],
    client: BudgetedVLMClient,
    max_workers: int,
) -> Tuple[Dict[str, SeamVLDecision], Dict[str, SeamVLDecision]]:
    """Run paired `text_only` and `multimodal` seam VLM calls safely with bounded concurrency."""
    tasks = []
    for idx, mq in enumerate(routed):
        qid = mq.query.query_id
        qt = mq.query.query_type
        tasks.append((idx, "text_only", qid, qt, mq.brief_a, mq.brief_b, None, None))
        tasks.append((idx, "multimodal", qid, qt, mq.brief_a, mq.brief_b, mq.rep_img_a, mq.rep_img_b))

    d_txt: Dict[str, SeamVLDecision] = {}
    d_mm: Dict[str, SeamVLDecision] = {}
    t0 = time.time()
    n_tot = len(tasks)

    def _exec(item: Tuple[int, str, str, str, Any, Any, Optional[Path], Optional[Path]]) -> Tuple[str, str, SeamVLDecision]:
        _, mode, qid, qt, ba, bb, pa, pb = item
        dec = client.adjudicate_cluster_seam(
            query_id=f"{qid}_{mode}",
            query_type=qt,
            mode=mode,
            brief_a=ba,
            brief_b=bb,
            img_path_a=pa,
            img_path_b=pb,
            max_side=384,
            quality=82,
        )
        return qid, mode, dec

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futs = [pool.submit(_exec, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            qid, mode, dec = fut.result()
            if mode == "text_only":
                d_txt[qid] = dec
            else:
                d_mm[qid] = dec
            done += 1
            if done % 40 == 0 or done == n_tot:
                print(f"    VLM progress: {done}/{n_tot} ({time.time()-t0:.1f}s)", flush=True)

    return d_txt, d_mm


def _extract_case_logs(
    routed: List[MMSeamQuery],
    ds_labels: np.ndarray,
    summaries: Dict[int, Any],
    d_txt: Dict[str, SeamVLDecision],
    d_mm: Dict[str, SeamVLDecision],
    corrupted_mask: np.ndarray,
    limit: int = 8,
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect representative visual-rescue and conflict-blocked seam examples (no raw event names)."""
    rescues: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []

    for mq in routed:
        q = mq.query
        qid = q.query_id
        dt = d_txt[qid]
        dm = d_mm[qid]

        if q.query_type == "merge_pair":
            m1 = summaries[int(q.merge_c1)].members
            m2 = summaries[int(q.merge_c2)].members
        else:
            m1, m2 = q.seam_sub1, q.seam_sub2
        v1, c1 = np.unique(ds_labels[m1], return_counts=True)
        v2, c2 = np.unique(ds_labels[m2], return_counts=True)
        oracle_same = int(v1[int(np.argmax(c1))]) == int(v2[int(np.argmax(c2))])

        a_corr = bool(mq.rep_post_a is not None and corrupted_mask[mq.rep_post_a])
        b_corr = bool(mq.rep_post_b is not None and corrupted_mask[mq.rep_post_b])

        rec = {
            "query_id": qid,
            "query_type": q.query_type,
            "route_source": mq.route_source,
            "oracle_same": oracle_same,
            "rep_corrupted": a_corr or b_corr,
            "keywords_a": mq.brief_a[0][:4],
            "keywords_b": mq.brief_b[0][:4],
            "txt_pred_same": dt.same_event_text_only,
            "mm_pred_same": dm.same_event_multimodal,
            "mm_inline_txt_same": dm.same_event_text_only,
            "visual_discriminative": dm.visual_discriminative,
            "visual_conflict_with_text": dm.visual_conflict_with_text,
            "visual_clue_a": dm.visual_clue_a[:90],
            "visual_clue_b": dm.visual_clue_b[:90],
            "confidence": dm.confidence,
        }

        if (
            len(rescues) < limit
            and dm.visual_discriminative is True
            and dm.visual_conflict_with_text is False
            and dt.same_event_text_only is not None
            and dm.same_event_multimodal is not None
            and dt.same_event_text_only != oracle_same
            and dm.same_event_multimodal == oracle_same
        ):
            rescues.append(rec)

        if (
            len(conflicts) < limit
            and dm.visual_conflict_with_text is True
        ):
            conflicts.append(rec)

    return {"visual_rescues": rescues, "conflict_blocked": conflicts}


def run_condition(
    split: str,
    cond: str,
    ds: Any,
    txt_minilm: np.ndarray,
    clip_img: np.ndarray,
    decoded_ok: np.ndarray,
    dhash64: np.ndarray,
    precomputed_text: Tuple[Any, Any, np.ndarray, Dict[str, Any]],
    run_stage2: bool,
    budget_queries: int,
    merge_ratio: float,
    conf_threshold: float,
    stage1_cfg: Dict[str, Any],
    router_cfg: Dict[str, Any],
    provider: str,
    model_name: str,
    max_workers: int,
    prefix_steps: List[int],
) -> Dict[str, Any]:
    print(f"\n=== [{split.upper()} | cond={cond}] Running Stage-1 (6 variants) ===", flush=True)
    if cond == "stress30":
        clip_cur, img_paths_cur, corrupted_mask = corrupt_images_competing(
            ds, clip_img, decoded_ok, dhash64, rate=0.30, seed=20261122
        )
        print(f"  Corrupted {int(corrupted_mask.sum())}/{int(decoded_ok.sum())} valid images (Stress-30%)", flush=True)
    else:
        clip_cur = clip_img
        img_paths_cur = ds.image_paths
        corrupted_mask = np.zeros(len(ds), dtype=bool)

    s1 = run_all_stage1_variants(
        ds,
        txt_minilm,
        clip_cur,
        decoded_ok,
        lam_v=float(stage1_cfg["lam_v"]),
        tau_imp=float(stage1_cfg["tau_imp"]),
        gamma_imp=float(stage1_cfg["gamma_imp"]),
        k_v=int(stage1_cfg["k_v"]),
        vis_sim_thresh=float(stage1_cfg["vis_sim_thresh"]),
        block_size=int(stage1_cfg.get("block_size", 1024)),
        precomputed_text=precomputed_text,
    )
    for k in [
        "text_only_2dse",
        "naive_mv_2dse",
        "cntd_raw_mv_2dse",
        "cntd_cond_mv_2dse",
        "visual_idf_no_impute_mv_2dse",
        "visual_idf_mv_2dse",
    ]:
        m = s1["metrics"][k]
        print(f"  Stage1 {k:30s}: AMI={m['ami']:.4f} MacroB3={m['macro_b3']:.4f} MicroB3={m['micro_b3']:.4f} K={m['n_clusters']}", flush=True)

    cond_out: Dict[str, Any] = {
        "n_corrupted_images": int(corrupted_mask.sum()),
        "stage1": s1["metrics"],
    }

    if not run_stage2:
        return cond_out

    print(f"\n=== [{split.upper()} | cond={cond}] Routing & Stage-2 VLM ({model_name}) ===", flush=True)
    routed, summaries = route_mm_seams(
        ds,
        s1["social_ds"],
        s1["vis_idf_comb_graph"],
        s1["vis_idf_labels"],
        txt_minilm,
        clip_cur,
        decoded_ok,
        s1["evid_scores"]["visual_idf"],
        k_total=budget_queries,
        merge_ratio=merge_ratio,
        min_text_sim=float(router_cfg["min_text_sim"]),
        vis_rescue_sim=float(router_cfg["vis_rescue_sim"]),
        vis_rescue_min_txt=float(router_cfg["vis_rescue_min_txt"]),
        vis_rescue_min_vidf=float(router_cfg["vis_rescue_min_vidf"]),
        max_pairs_per_cluster=int(router_cfg.get("max_pairs_per_cluster", 3)),
        img_paths_override=img_paths_cur,
    )
    n_merge = sum(1 for mq in routed if mq.query.query_type == "merge_pair")
    n_seam = sum(1 for mq in routed if mq.query.query_type == "seam")
    n_vis_rescue = sum(1 for mq in routed if mq.route_source == "visual_rescue")
    n_both_imgs = sum(1 for mq in routed if mq.rep_img_a is not None and mq.rep_img_b is not None)
    n_corrupt_rep = sum(
        1
        for mq in routed
        if (mq.rep_post_a is not None and corrupted_mask[mq.rep_post_a])
        or (mq.rep_post_b is not None and corrupted_mask[mq.rep_post_b])
    )
    print(
        f"  Routed={len(routed)} (merge={n_merge}, seam={n_seam}, vis_rescue_pairs={n_vis_rescue}, "
        f"both_imgs={n_both_imgs}, corrupt_reps={n_corrupt_rep})",
        flush=True,
    )

    safe_model = model_name.replace("/", "__")
    shared_cache = ROOT / "results" / "llm_cache" / f"NED_phase2_{safe_model}" / "responses.jsonl"
    ledger_dir = OUT_DIR / f"ledger_{split}_{cond}_{safe_model}"
    prov_url, _, api_key = load_provider_key(provider)
    ledger = TokenLedger(ledger_dir, shared_cache_file=shared_cache)
    client = BudgetedVLMClient(model=model_name, ledger=ledger, base_url=prov_url, api_key=api_key, max_tokens=320)

    d_txt_map, d_mm_map = _run_vlm_calls(routed, client, max_workers=max_workers)
    status_txt = Counter(d.status for d in d_txt_map.values())
    status_mm = Counter(d.status for d in d_mm_map.values())
    ledger_summary = ledger.summary()
    print(f"  Statuses text_only={dict(status_txt)} | multimodal={dict(status_mm)}", flush=True)
    print(f"  Ledger summary={ledger_summary}", flush=True)

    # Full-budget evaluation for all 5 policies
    policies_full: Dict[str, Any] = {}
    for pol in POLICIES:
        pm, _ = decisions_to_llm_decisions(
            routed,
            ds.labels,
            summaries,
            d_txt_map,
            d_mm_map,
            s1["vis_idf_labels"],
            policy=pol,
            conf_threshold=conf_threshold,
        )
        policies_full[pol] = pm
        print(
            f"  Stage2 [{pol:30s}]: AMI={pm['ami']:.4f} ({pm['delta_ami_vs_stage1']:+.4f}) "
            f"MacroB3={pm['macro_b3']:.4f} ({pm['delta_macro_b3_vs_stage1']:+.4f}) "
            f"SeamAcc={pm['overall_seam_acc']:.4f} Rej={pm['gate_reject_rate']:.2%} "
            f"Tok={pm['real_tokens_spent']} Fix/Q={pm['unqueried_posts_fixed_per_query']}",
            flush=True,
        )

    # Prefix budget curve
    clean_steps = sorted({s for s in prefix_steps if 0 < s <= len(routed)} | ({len(routed)} if routed else set()))
    prefix_curves: Dict[str, List[Dict[str, Any]]] = {pol: [] for pol in POLICIES}
    for k_step in clean_steps:
        sub_routed = routed[:k_step]
        for pol in POLICIES:
            pm_k, _ = decisions_to_llm_decisions(
                sub_routed,
                ds.labels,
                summaries,
                d_txt_map,
                d_mm_map,
                s1["vis_idf_labels"],
                policy=pol,
                conf_threshold=conf_threshold,
            )
            prefix_curves[pol].append({"k": k_step, **pm_k})

    case_logs = _extract_case_logs(routed, ds.labels, summaries, d_txt_map, d_mm_map, corrupted_mask)

    cond_out["routing"] = {
        "n_routed": len(routed),
        "n_merge": n_merge,
        "n_seam": n_seam,
        "n_vis_rescue_pairs": n_vis_rescue,
        "n_both_imgs": n_both_imgs,
        "n_corrupt_reps": n_corrupt_rep,
    }
    cond_out["vlm_health"] = {
        "status_text_only": dict(status_txt),
        "status_multimodal": dict(status_mm),
        "ledger": ledger_summary,
    }
    cond_out["stage2_policies"] = policies_full
    cond_out["stage2_prefix_curves"] = prefix_curves
    cond_out["case_studies"] = case_logs
    return cond_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Track B Phase 2 Evid-MSED Runner on NED.")
    parser.add_argument("--split", required=True, choices=["validation", "holdout"])
    parser.add_argument("--budget-queries", type=int, default=60)
    parser.add_argument("--merge-ratio", type=float, default=0.70)
    parser.add_argument("--conf-threshold", type=float, default=0.60)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--run-stress", action="store_true", help="Run Stage 1 + Stage 2 on Stress-30 pct.")
    parser.add_argument("--run-stage1-stress", action="store_true", help="Run Stage 1 Stress-30 pct (0 API cost).")
    parser.add_argument("--protocol", type=str, default=None, help="Frozen protocol JSON (required for holdout).")
    parser.add_argument("--provider", type=str, default="siliconflow")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    stage1_cfg = {
        "lam_v": 0.65,
        "tau_imp": 0.55,
        "gamma_imp": 0.70,
        "k_v": 10,
        "vis_sim_thresh": 0.72,
        "block_size": 1024,
    }
    router_cfg = {
        "min_text_sim": 0.42,
        "vis_rescue_sim": 0.68,
        "vis_rescue_min_txt": 0.18,
        "vis_rescue_min_vidf": 0.50,
        "max_pairs_per_cluster": 3,
    }
    budget_queries = args.budget_queries
    merge_ratio = args.merge_ratio
    conf_threshold = args.conf_threshold
    model_name = args.model

    if args.split == "holdout":
        if not args.protocol:
            print("[ERROR] --protocol trackB_evid_msed/protocol_trackB_v1.json is REQUIRED for holdout.", file=sys.stderr)
            sys.exit(2)
        proto = _verify_protocol_hashes(Path(args.protocol))
        stage1_cfg = proto["stage1"]
        router_cfg = proto["stage2_router"]
        budget_queries = int(proto["holdout"]["budget_queries"])
        merge_ratio = float(proto["stage2_router"]["merge_ratio"])
        conf_threshold = float(proto["stage2_router"]["conf_threshold"])
        model_name = str(proto["llm"]["model"])

    ds, txt_minilm, clip_img, _, decoded_ok, dhash64 = load_split_with_cache(args.split)
    print(f"[Loaded] split={args.split} n_posts={len(ds)} n_events={len(ds.label_to_event)} valid_img={int(decoded_ok.sum())}", flush=True)

    precomputed_text = build_text_graph_and_2dse(ds, txt_minilm)
    prefix_steps = [20, 40, 60] if args.split == "validation" else [60, 120, 180]

    results: Dict[str, Any] = {
        "split": args.split,
        "n_posts": len(ds),
        "n_events": len(ds.label_to_event),
        "n_valid_images": int(decoded_ok.sum()),
        "model": model_name,
        "budget_queries": budget_queries,
        "merge_ratio": merge_ratio,
        "conf_threshold": conf_threshold,
        "stage1_cfg": stage1_cfg,
        "router_cfg": router_cfg,
        "protocol_used": args.protocol,
        "conditions": {},
    }

    results["conditions"]["clean"] = run_condition(
        split=args.split,
        cond="clean",
        ds=ds,
        txt_minilm=txt_minilm,
        clip_img=clip_img,
        decoded_ok=decoded_ok,
        dhash64=dhash64,
        precomputed_text=precomputed_text,
        run_stage2=True,
        budget_queries=budget_queries,
        merge_ratio=merge_ratio,
        conf_threshold=conf_threshold,
        stage1_cfg=stage1_cfg,
        router_cfg=router_cfg,
        provider=args.provider,
        model_name=model_name,
        max_workers=args.max_workers,
        prefix_steps=prefix_steps,
    )

    if args.run_stress or args.run_stage1_stress:
        results["conditions"]["stress30"] = run_condition(
            split=args.split,
            cond="stress30",
            ds=ds,
            txt_minilm=txt_minilm,
            clip_img=clip_img,
            decoded_ok=decoded_ok,
            dhash64=dhash64,
            precomputed_text=precomputed_text,
            run_stage2=bool(args.run_stress),
            budget_queries=budget_queries,
            merge_ratio=merge_ratio,
            conf_threshold=conf_threshold,
            stage1_cfg=stage1_cfg,
            router_cfg=router_cfg,
            provider=args.provider,
            model_name=model_name,
            max_workers=args.max_workers,
            prefix_steps=prefix_steps,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"ned_{args.split}_results.json"
    tmp_file = OUT_DIR / f"ned_{args.split}_results.json.tmp"
    tmp_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_file.replace(out_file)
    print(f"\n[Saved] Phase 2 results saved cleanly to {out_file}", flush=True)


if __name__ == "__main__":
    main()
