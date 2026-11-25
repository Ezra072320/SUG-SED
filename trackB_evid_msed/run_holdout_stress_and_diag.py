"""Extension runner (Option 1): Holdout Stress-30% Stage 2 (K=60) + 24-Event Stratified & Action-Confusion Diagnosis.

Strictly preserves all 9 frozen SHA-256 files in `protocol_trackB_v1.json`.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.graph_builder import MessageGraph
from core_shared.key_loader import load_provider_key
from core_shared.metrics_audit import full_partition_audit, item_bcubed_scores
from core_shared.token_ledger import TokenLedger
from trackB_evid_msed.mm_seam_adjudicator import (
    MMSeamQuery,
    _oracle_same_event,
    decisions_to_llm_decisions,
    route_mm_seams,
)
from trackB_evid_msed.mv_2dse_pipeline import (
    build_text_graph_and_2dse,
    compute_visual_evidence_chunked,
    corrupt_images_competing,
    load_split_with_cache,
    run_mv_2dse_variant,
)
from trackB_evid_msed.run_phase2_ned import (
    POLICIES,
    _extract_case_logs,
    _run_vlm_calls,
    _verify_protocol_hashes,
)
from trackB_evid_msed.vlm_client import BudgetedVLMClient, SeamVLDecision

OUT_DIR = ROOT / "results" / "trackB_phase2"
PROTO_PATH = ROOT / "trackB_evid_msed" / "protocol_trackB_v1.json"

CORE_POLICIES = [
    "text_only_vlm",
    "naive_vlm",
    "evid_self_gate_inline",
    "evid_self_gate_clean_fallback",
]


def _stratum_of(size: int) -> str:
    if size < 200:
        return "small"
    if size < 600:
        return "medium"
    return "head"


def _compute_stratified_b3(
    labels_true: np.ndarray,
    event_ids: List[str],
    label_to_event: List[str],
    labels_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute exact per-event B3-P/R/F1 and Small(<200)/Medium(200-599)/Head(>=600) strata averages."""
    p_item, r_item, _ = item_bcubed_scores(labels_true, labels_pred)
    ev_idxs: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(labels_true.tolist()):
        ev_idxs[int(y)].append(i)

    per_event: Dict[str, Dict[str, Any]] = {}
    by_stratum: Dict[str, List[Tuple[float, float, float]]] = {"small": [], "medium": [], "head": []}

    for y_idx, ev_name in enumerate(label_to_event):
        idxs = ev_idxs[y_idx]
        ep = float(np.mean(p_item[idxs]))
        er = float(np.mean(r_item[idxs]))
        ef1 = (2.0 * ep * er / (ep + er)) if (ep + er) > 0 else 0.0
        sz = len(idxs)
        st = _stratum_of(sz)
        per_event[ev_name] = {
            "event_id": ev_name,
            "size": sz,
            "stratum": st,
            "b3_p": round(ep, 4),
            "b3_r": round(er, 4),
            "b3_f1": round(ef1, 4),
        }
        by_stratum[st].append((ep, er, ef1))

    strata_summary: Dict[str, Any] = {}
    all_f1s: List[float] = []
    for st in ["small", "medium", "head"]:
        items = by_stratum[st]
        ps = [x[0] for x in items]
        rs = [x[1] for x in items]
        f1s = [x[2] for x in items]
        all_f1s.extend(f1s)
        strata_summary[st] = {
            "n_events": len(items),
            "macro_p": round(float(np.mean(ps)), 4),
            "macro_r": round(float(np.mean(rs)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
        }
    strata_summary["overall_macro_b3_f1"] = round(float(np.mean(all_f1s)), 4)
    return {"strata": strata_summary, "per_event": per_event}


def _compute_action_confusion(
    routed: List[MMSeamQuery],
    ds_labels: np.ndarray,
    event_ids: List[str],
    summaries: Dict[int, Any],
    d_txt_map: Dict[str, SeamVLDecision],
    d_mm_map: Dict[str, SeamVLDecision],
    policy: str,
    conf_threshold: float,
) -> Dict[str, Any]:
    """Compute Stage-2 executed decision confusion matrix and head/medium/small seam exposure."""
    cnt = {
        "merge_tp": 0,
        "merge_fp": 0,
        "separate_tn": 0,
        "separate_fn": 0,
        "split_tp": 0,
        "split_fp": 0,
        "keep_tn": 0,
        "keep_fn": 0,
        "skipped_low_conf": 0,
    }
    ev_sizes = defaultdict(int)
    for e in event_ids:
        ev_sizes[e] += 1

    # Track strata involvement of routed queries (only counted once per policy run)
    stratum_touched = {"small": 0, "medium": 0, "head": 0}
    stratum_merge_fp = {"small": 0, "medium": 0, "head": 0}

    def _maj_ev(idxs: List[int]) -> str:
        vals, cs = np.unique(ds_labels[idxs], return_counts=True)
        # Map int label to event_id from first matching member
        maj_int = int(vals[int(np.argmax(cs))])
        for m in idxs:
            if int(ds_labels[m]) == maj_int:
                return event_ids[m]
        return event_ids[idxs[0]]

    for mq in routed:
        q = mq.query
        d_txt = d_txt_map[q.query_id]
        d_mm = d_mm_map[q.query_id]
        oracle_same = _oracle_same_event(q, ds_labels, summaries)

        if q.query_type == "merge_pair":
            m1 = summaries[int(q.merge_c1)].members
            m2 = summaries[int(q.merge_c2)].members
        else:
            m1, m2 = q.seam_sub1, q.seam_sub2
        e1, e2 = _maj_ev(m1), _maj_ev(m2)
        st_pair = {_stratum_of(ev_sizes[e1]), _stratum_of(ev_sizes[e2])}
        for st in st_pair:
            stratum_touched[st] += 1

        if policy == "text_only_vlm":
            same = d_txt.same_event_text_only
            conf = d_txt.confidence
        elif policy == "naive_vlm":
            same = d_mm.same_event_multimodal
            conf = d_mm.confidence
        elif policy == "evid_self_gate_inline":
            accept = (d_mm.visual_discriminative is True) and (d_mm.visual_conflict_with_text is False)
            same = d_mm.same_event_multimodal if accept else d_mm.same_event_text_only
            conf = d_mm.confidence
        elif policy == "evid_self_gate_clean_fallback":
            accept = (d_mm.visual_discriminative is True) and (d_mm.visual_conflict_with_text is False)
            if accept:
                same = d_mm.same_event_multimodal
                conf = d_mm.confidence
            else:
                same = d_txt.same_event_text_only
                conf = d_txt.confidence
        else:
            raise ValueError(policy)

        if same is None or float(conf) < conf_threshold:
            cnt["skipped_low_conf"] += 1
            continue

        if q.query_type == "merge_pair":
            if same:
                if oracle_same:
                    cnt["merge_tp"] += 1
                else:
                    cnt["merge_fp"] += 1
                    for st in st_pair:
                        stratum_merge_fp[st] += 1
            else:
                if oracle_same:
                    cnt["separate_fn"] += 1
                else:
                    cnt["separate_tn"] += 1
        else:
            if not same:  # SPLIT
                if not oracle_same:
                    cnt["split_tp"] += 1
                else:
                    cnt["split_fp"] += 1
            else:  # KEEP_MERGED
                if oracle_same:
                    cnt["keep_tn"] += 1
                else:
                    cnt["keep_fn"] += 1

    m_exec = cnt["merge_tp"] + cnt["merge_fp"]
    s_exec = cnt["split_tp"] + cnt["split_fp"]
    cnt["merge_prec"] = round(cnt["merge_tp"] / m_exec, 4) if m_exec > 0 else 0.0
    cnt["split_prec"] = round(cnt["split_tp"] / s_exec, 4) if s_exec > 0 else 0.0
    cnt["stratum_touched"] = stratum_touched
    cnt["stratum_merge_fp"] = stratum_merge_fp
    return cnt


def _top_event_deltas(
    base_pe: Dict[str, Dict[str, Any]],
    after_pe: Dict[str, Dict[str, Any]],
    n_top: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    rows = []
    for ev, b in base_pe.items():
        a = after_pe[ev]
        rows.append(
            {
                "event_id": ev,
                "size": b["size"],
                "stratum": b["stratum"],
                "s1_p": b["b3_p"],
                "s1_r": b["b3_r"],
                "s1_f1": b["b3_f1"],
                "s2_p": a["b3_p"],
                "s2_r": a["b3_r"],
                "s2_f1": a["b3_f1"],
                "delta_p": round(a["b3_p"] - b["b3_p"], 4),
                "delta_r": round(a["b3_r"] - b["b3_r"], 4),
                "delta_f1": round(a["b3_f1"] - b["b3_f1"], 4),
            }
        )
    by_delta = sorted(rows, key=lambda r: -r["delta_f1"])
    return {
        "top_improved": by_delta[:n_top],
        "top_degraded": list(reversed(by_delta[-n_top:])),
    }


def main() -> None:
    proto = _verify_protocol_hashes(PROTO_PATH)
    s1_cfg = proto["stage1"]
    r_cfg = proto["stage2_router"]
    model_name = str(proto["llm"]["model"])
    provider = str(proto["llm"]["provider"])
    merge_ratio = float(r_cfg["merge_ratio"])
    conf_threshold = float(r_cfg["conf_threshold"])
    safe_model = model_name.replace("/", "__")
    shared_cache = ROOT / "results" / "llm_cache" / f"NED_phase2_{safe_model}" / "responses.jsonl"

    print("\n=== Loading NED Holdout & Building Text 2D-SE Base ===", flush=True)
    ds, txt_minilm, clip_img, _, decoded_ok, dhash64 = load_split_with_cache("holdout")
    social_ds, text_graph, labels_txt_se, txt_aud = build_text_graph_and_2dse(ds, txt_minilm)
    precomputed_text = (social_ds, text_graph, labels_txt_se, txt_aud)
    print(
        f"  Holdout loaded: n={len(ds)}, events={len(ds.label_to_event)}, "
        f"valid_img={int(decoded_ok.sum())}, text_only AMI={txt_aud['ami']:.4f} MacroB3={txt_aud['macro_b3']:.4f}",
        flush=True,
    )

    # ==================================================================
    # PART A: Holdout Stress-30% End-to-End Stage 2 (K=60, 120 live calls)
    # ==================================================================
    print("\n=== PART A: Holdout Stress-30% Stage-1 & Stage-2 (K=60) ===", flush=True)
    clip_stress, img_paths_stress, corrupted_mask = corrupt_images_competing(
        ds, clip_img, decoded_ok, dhash64, rate=0.30, seed=20261122
    )
    print(f"  Corrupted {int(corrupted_mask.sum())}/{int(decoded_ok.sum())} valid images.", flush=True)
    evid_stress = compute_visual_evidence_chunked(
        txt_minilm, clip_stress, decoded_ok, block_size=int(s1_cfg["block_size"])
    )
    lab_naive_stress, _, _, _, m_naive_stress = run_mv_2dse_variant(
        ds, text_graph, labels_txt_se, clip_stress, decoded_ok, evid_stress["naive"],
        lam_v=float(s1_cfg["lam_v"]), tau_imp=float(s1_cfg["tau_imp"]), gamma_imp=float(s1_cfg["gamma_imp"]),
        use_soft_imputation=True, k_v=int(s1_cfg["k_v"]), vis_sim_thresh=float(s1_cfg["vis_sim_thresh"]),
        block_size=int(s1_cfg["block_size"]),
    )
    lab_vis_stress, comb_adj_stress, _, _, m_vis_stress = run_mv_2dse_variant(
        ds, text_graph, labels_txt_se, clip_stress, decoded_ok, evid_stress["visual_idf"],
        lam_v=float(s1_cfg["lam_v"]), tau_imp=float(s1_cfg["tau_imp"]), gamma_imp=float(s1_cfg["gamma_imp"]),
        use_soft_imputation=True, k_v=int(s1_cfg["k_v"]), vis_sim_thresh=float(s1_cfg["vis_sim_thresh"]),
        block_size=int(s1_cfg["block_size"]),
    )
    print(f"  Stress-30% naive_mv_2dse     : {m_naive_stress}", flush=True)
    print(f"  Stress-30% visual_idf_mv_2dse: {m_vis_stress}", flush=True)
    if abs(m_vis_stress["ami"] - 0.5599) > 1e-4 or abs(m_vis_stress["macro_b3"] - 0.3773) > 1e-4:
        raise RuntimeError(f"Holdout Stress-30% Stage-1 mismatch: got {m_vis_stress}, expected AMI=0.5599 MacroB3=0.3773")

    comb_graph_stress = MessageGraph(
        adj=comb_adj_stress,
        sem_adj=text_graph.sem_adj,
        attr_adj=text_graph.attr_adj,
        embeddings=txt_minilm,
        non_entity_tfidf=text_graph.non_entity_tfidf,
    )
    routed_stress_60, summaries_stress = route_mm_seams(
        ds,
        social_ds,
        comb_graph_stress,
        lab_vis_stress,
        txt_minilm,
        clip_stress,
        decoded_ok,
        evid_stress["visual_idf"],
        k_total=60,
        merge_ratio=merge_ratio,
        min_text_sim=float(r_cfg["min_text_sim"]),
        vis_rescue_sim=float(r_cfg["vis_rescue_sim"]),
        vis_rescue_min_txt=float(r_cfg["vis_rescue_min_txt"]),
        vis_rescue_min_vidf=float(r_cfg["vis_rescue_min_vidf"]),
        max_pairs_per_cluster=int(r_cfg["max_pairs_per_cluster"]),
        img_paths_override=img_paths_stress,
    )
    n_m_st = sum(1 for mq in routed_stress_60 if mq.query.query_type == "merge_pair")
    n_s_st = sum(1 for mq in routed_stress_60 if mq.query.query_type == "seam")
    n_vr_st = sum(1 for mq in routed_stress_60 if mq.route_source == "visual_rescue")
    n_cr_st = sum(
        1
        for mq in routed_stress_60
        if (mq.rep_post_a is not None and corrupted_mask[mq.rep_post_a])
        or (mq.rep_post_b is not None and corrupted_mask[mq.rep_post_b])
    )
    print(
        f"  Stress-30% Routed=60 (merge={n_m_st}, seam={n_s_st}, vis_rescue={n_vr_st}, corrupt_reps={n_cr_st})",
        flush=True,
    )

    prov_url, _, api_key = load_provider_key(provider)
    ledger_stress_dir = OUT_DIR / f"ledger_holdout_stress30_k60_{safe_model}"
    ledger_stress = TokenLedger(ledger_stress_dir, shared_cache_file=shared_cache)
    client_stress = BudgetedVLMClient(
        model=model_name, ledger=ledger_stress, base_url=prov_url, api_key=api_key, max_tokens=320
    )
    d_txt_stress, d_mm_stress = _run_vlm_calls(routed_stress_60, client_stress, max_workers=2)
    from collections import Counter

    st_txt = Counter(d.status for d in d_txt_stress.values())
    st_mm = Counter(d.status for d in d_mm_stress.values())
    stress_ledger_sum = ledger_stress.summary()
    print(f"  Stress-30% VLM statuses text={dict(st_txt)} | mm={dict(st_mm)}", flush=True)
    print(f"  Stress-30% Ledger summary={stress_ledger_sum}", flush=True)

    stress_pol_metrics: Dict[str, Any] = {}
    stress_pol_labels: Dict[str, np.ndarray] = {}
    for pol in POLICIES:
        pm, lab_out = decisions_to_llm_decisions(
            routed_stress_60,
            ds.labels,
            summaries_stress,
            d_txt_stress,
            d_mm_stress,
            lab_vis_stress,
            policy=pol,
            conf_threshold=conf_threshold,
        )
        stress_pol_metrics[pol] = pm
        stress_pol_labels[pol] = lab_out
        print(
            f"  Holdout Stress30 [{pol:30s}]: AMI={pm['ami']:.4f} MacroB3={pm['macro_b3']:.4f} "
            f"MicroB3={pm['micro_b3']:.4f} SeamAcc={pm['overall_seam_acc']:.4f} Rej={pm['gate_reject_rate']:.2%} "
            f"Fix/Q={pm['unqueried_posts_fixed_per_query']}",
            flush=True,
        )

    stress_prefix: Dict[str, List[Dict[str, Any]]] = {pol: [] for pol in POLICIES}
    for k_step in [20, 40, 60]:
        for pol in POLICIES:
            pm_k, _ = decisions_to_llm_decisions(
                routed_stress_60[:k_step],
                ds.labels,
                summaries_stress,
                d_txt_stress,
                d_mm_stress,
                lab_vis_stress,
                policy=pol,
                conf_threshold=conf_threshold,
            )
            stress_prefix[pol].append({"k": k_step, **pm_k})

    stress_cases = _extract_case_logs(
        routed_stress_60, ds.labels, summaries_stress, d_txt_stress, d_mm_stress, corrupted_mask
    )

    # ==================================================================
    # PART B: Zero-Cost Holdout Clean Replay + Stratified & Confusion Audit
    # ==================================================================
    print("\n=== PART B: Zero-Cost Clean Replay & 24-Event Stratified / Action Diagnosis ===", flush=True)
    evid_clean = compute_visual_evidence_chunked(
        txt_minilm, clip_img, decoded_ok, block_size=int(s1_cfg["block_size"])
    )
    lab_vis_clean, comb_adj_clean, _, _, m_vis_clean = run_mv_2dse_variant(
        ds, text_graph, labels_txt_se, clip_img, decoded_ok, evid_clean["visual_idf"],
        lam_v=float(s1_cfg["lam_v"]), tau_imp=float(s1_cfg["tau_imp"]), gamma_imp=float(s1_cfg["gamma_imp"]),
        use_soft_imputation=True, k_v=int(s1_cfg["k_v"]), vis_sim_thresh=float(s1_cfg["vis_sim_thresh"]),
        block_size=int(s1_cfg["block_size"]),
    )
    if abs(m_vis_clean["ami"] - 0.5850) > 1e-4 or abs(m_vis_clean["macro_b3"] - 0.4124) > 1e-4:
        raise RuntimeError(f"Holdout Clean Stage-1 mismatch: {m_vis_clean}")

    comb_graph_clean = MessageGraph(
        adj=comb_adj_clean,
        sem_adj=text_graph.sem_adj,
        attr_adj=text_graph.attr_adj,
        embeddings=txt_minilm,
        non_entity_tfidf=text_graph.non_entity_tfidf,
    )
    routed_clean_180, summaries_clean = route_mm_seams(
        ds,
        social_ds,
        comb_graph_clean,
        lab_vis_clean,
        txt_minilm,
        clip_img,
        decoded_ok,
        evid_clean["visual_idf"],
        k_total=180,
        merge_ratio=merge_ratio,
        min_text_sim=float(r_cfg["min_text_sim"]),
        vis_rescue_sim=float(r_cfg["vis_rescue_sim"]),
        vis_rescue_min_txt=float(r_cfg["vis_rescue_min_txt"]),
        vis_rescue_min_vidf=float(r_cfg["vis_rescue_min_vidf"]),
        max_pairs_per_cluster=int(r_cfg["max_pairs_per_cluster"]),
    )

    # Fresh separate replay dir + disabled HTTP endpoint so ZERO live calls can ever fire!
    replay_dir = OUT_DIR / f"ledger_holdout_clean_diag_replay_{safe_model}"
    if replay_dir.exists():
        shutil.rmtree(replay_dir)
    ledger_replay = TokenLedger(replay_dir, shared_cache_file=shared_cache)
    client_replay = BudgetedVLMClient(
        model=model_name,
        ledger=ledger_replay,
        base_url="http://127.0.0.1:0/cache_only",
        api_key="CACHE_ONLY_NO_LIVE_HTTP",
        max_tokens=320,
    )
    d_txt_clean, d_mm_clean = _run_vlm_calls(routed_clean_180, client_replay, max_workers=1)
    replay_sum = ledger_replay.summary()
    print(f"  Holdout Clean Replay Ledger: {replay_sum}", flush=True)
    if replay_sum["cached_calls"] != 360 or replay_sum["total_calls"] != 360:
        raise RuntimeError(f"Expected 360/360 CACHED_REPLAY on Clean, got {replay_sum}")

    # Collect Clean K=60 and K=180 partitions and action confusion matrices
    strata_clean: Dict[str, Any] = {
        "text_only_2dse": _compute_stratified_b3(ds.labels, ds.event_ids, ds.label_to_event, labels_txt_se),
        "visual_idf_mv_2dse": _compute_stratified_b3(ds.labels, ds.event_ids, ds.label_to_event, lab_vis_clean),
    }
    action_conf: Dict[str, Any] = {"clean_k60": {}, "clean_k180": {}, "stress30_k60": {}}

    clean_k60_labels: Dict[str, np.ndarray] = {}
    clean_k180_labels: Dict[str, np.ndarray] = {}
    for pol in CORE_POLICIES:
        m60, l60 = decisions_to_llm_decisions(
            routed_clean_180[:60], ds.labels, summaries_clean, d_txt_clean, d_mm_clean, lab_vis_clean,
            policy=pol, conf_threshold=conf_threshold,
        )
        m180, l180 = decisions_to_llm_decisions(
            routed_clean_180, ds.labels, summaries_clean, d_txt_clean, d_mm_clean, lab_vis_clean,
            policy=pol, conf_threshold=conf_threshold,
        )
        clean_k60_labels[pol] = l60
        clean_k180_labels[pol] = l180
        strata_clean[f"k60_{pol}"] = _compute_stratified_b3(ds.labels, ds.event_ids, ds.label_to_event, l60)
        strata_clean[f"k180_{pol}"] = _compute_stratified_b3(ds.labels, ds.event_ids, ds.label_to_event, l180)
        action_conf["clean_k60"][pol] = _compute_action_confusion(
            routed_clean_180[:60], ds.labels, ds.event_ids, summaries_clean, d_txt_clean, d_mm_clean, pol, conf_threshold
        )
        action_conf["clean_k180"][pol] = _compute_action_confusion(
            routed_clean_180, ds.labels, ds.event_ids, summaries_clean, d_txt_clean, d_mm_clean, pol, conf_threshold
        )

    # Verify exact replay match vs Phase 2 Clean Holdout K=180
    re_aud_inline = full_partition_audit(ds.labels, clean_k180_labels["evid_self_gate_inline"])
    if abs(re_aud_inline["ami"] - 0.5858) > 1e-4:
        raise RuntimeError(f"Replay verification mismatch: {re_aud_inline['ami']}")

    # Stress-30% stratified B3 + action confusion
    strata_stress: Dict[str, Any] = {
        "text_only_2dse": _compute_stratified_b3(ds.labels, ds.event_ids, ds.label_to_event, labels_txt_se),
        "naive_mv_2dse": _compute_stratified_b3(ds.labels, ds.event_ids, ds.label_to_event, lab_naive_stress),
        "visual_idf_mv_2dse": _compute_stratified_b3(ds.labels, ds.event_ids, ds.label_to_event, lab_vis_stress),
    }
    for pol in CORE_POLICIES:
        strata_stress[f"k60_{pol}"] = _compute_stratified_b3(
            ds.labels, ds.event_ids, ds.label_to_event, stress_pol_labels[pol]
        )
        action_conf["stress30_k60"][pol] = _compute_action_confusion(
            routed_stress_60, ds.labels, ds.event_ids, summaries_stress, d_txt_stress, d_mm_stress, pol, conf_threshold
        )

    # Top-3 improved & top-3 degraded events (Stage-2 vs Stage-1 Clean)
    s1_pe = strata_clean["visual_idf_mv_2dse"]["per_event"]
    top_deltas = {
        "clean_k60_inline": _top_event_deltas(s1_pe, strata_clean["k60_evid_self_gate_inline"]["per_event"]),
        "clean_k180_inline": _top_event_deltas(s1_pe, strata_clean["k180_evid_self_gate_inline"]["per_event"]),
        "clean_k180_cleanfb": _top_event_deltas(s1_pe, strata_clean["k180_evid_self_gate_clean_fallback"]["per_event"]),
        "stress30_k60_cleanfb": _top_event_deltas(
            strata_stress["visual_idf_mv_2dse"]["per_event"],
            strata_stress["k60_evid_self_gate_clean_fallback"]["per_event"],
        ),
    }

    print("\n=== Summary: Holdout Stratified B3-F1 (Clean vs Stress-30%) ===", flush=True)
    for k, v in strata_clean.items():
        st = v["strata"]
        print(
            f"  [CLEAN ] {k:34s}: Small={st['small']['macro_f1']:.4f} Med={st['medium']['macro_f1']:.4f} "
            f"Head={st['head']['macro_f1']:.4f} | Overall={st['overall_macro_b3_f1']:.4f}",
            flush=True,
        )
    for k, v in strata_stress.items():
        st = v["strata"]
        print(
            f"  [STRESS] {k:34s}: Small={st['small']['macro_f1']:.4f} Med={st['medium']['macro_f1']:.4f} "
            f"Head={st['head']['macro_f1']:.4f} | Overall={st['overall_macro_b3_f1']:.4f}",
            flush=True,
        )

    out_payload = {
        "split": "holdout",
        "n_posts": len(ds),
        "n_events": len(ds.label_to_event),
        "protocol_used": "trackB_evid_msed/protocol_trackB_v1.json",
        "holdout_stress30_end_to_end": {
            "n_corrupted_images": int(corrupted_mask.sum()),
            "stage1": {
                "text_only_2dse": txt_aud,
                "naive_mv_2dse": m_naive_stress,
                "visual_idf_mv_2dse": m_vis_stress,
            },
            "routing_k60": {
                "n_routed": len(routed_stress_60),
                "n_merge": n_m_st,
                "n_seam": n_s_st,
                "n_vis_rescue_pairs": n_vr_st,
                "n_corrupt_reps": n_cr_st,
            },
            "vlm_health": {
                "status_text_only": dict(st_txt),
                "status_multimodal": dict(st_mm),
                "ledger": stress_ledger_sum,
            },
            "stage2_policies_k60": stress_pol_metrics,
            "stage2_prefix_curves": stress_prefix,
            "case_studies": stress_cases,
        },
        "clean_replay_ledger": replay_sum,
        "stratified_b3": {
            "clean": strata_clean,
            "stress30": strata_stress,
        },
        "action_confusion": action_conf,
        "top_event_deltas": top_deltas,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ned_holdout_stress_and_diag.json"
    tmp_path = OUT_DIR / "ned_holdout_stress_and_diag.json.tmp"
    tmp_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)
    print(f"\n[Saved] Holdout Stress-30% & Diagnostics written cleanly to {out_path}", flush=True)


if __name__ == "__main__":
    main()
