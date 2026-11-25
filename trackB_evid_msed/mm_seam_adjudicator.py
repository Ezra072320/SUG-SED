"""Stage-2 Multimodal Cluster Seam Router & Evidence-Gated Adjudicator.

Bridges Stage-1 Visual-IDF MV-2DSE clusters with Track A's mesoscopic routing
(`MERGE_PAIR` + spherical-KMeans++ `SPLIT_SEAM`) and Gate B's winning
cross-modal conflict & discriminability gate (`Qwen/Qwen3-VL-8B-Instruct`).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import MessageGraph
from core_shared.metrics_audit import full_partition_audit
from trackA_sug_sed.high_leverage_router import (
    ClusterSummary,
    QueryItem,
    build_cluster_summaries,
    detect_seam_queries,
)
from trackA_sug_sed.topology_surgery import LLMDecision, apply_direct_assignment
from trackB_evid_msed.ned_loader import NEDDataset
from trackB_evid_msed.vlm_client import SeamVLDecision


@dataclass
class MMSeamQuery:
    query: QueryItem
    brief_a: Tuple[List[str], List[str]]
    brief_b: Tuple[List[str], List[str]]
    rep_img_a: Optional[Path]
    rep_img_b: Optional[Path]
    rep_post_a: Optional[int]
    rep_post_b: Optional[int]
    route_source: str  # 'text_bridge' | 'visual_rescue' | 'intra_seam'


def _extract_brief(
    ds: NEDDataset,
    members: List[int],
    medoids: List[int],
    max_keywords: int = 5,
    max_medoids: int = 3,
) -> Tuple[List[str], List[str]]:
    """Build strictly anonymized cluster brief (top hashtags/tokens + medoid texts)."""
    kw_cnt: Dict[str, int] = defaultdict(int)
    for m in members:
        for h in ds.hashtags[m]:
            kw_cnt[f"#{h}"] += 2
        for tk in ds.tokens[m]:
            if len(tk) >= 3:
                kw_cnt[tk] += 1
    top_kw = [k for k, _ in sorted(kw_cnt.items(), key=lambda kv: -kv[1])[:max_keywords]]
    med_texts = [ds.texts[int(m)] for m in medoids[:max_medoids]]
    return top_kw, med_texts


def select_cluster_rep_image(
    members: List[int],
    clip_img: np.ndarray,
    decoded_ok: np.ndarray,
    b_idf: np.ndarray,
) -> Tuple[Optional[int], Optional[np.ndarray], float]:
    """Pick the most Visually-IDF-weighted representative image post in `members`."""
    vis_members = [int(m) for m in members if decoded_ok[m]]
    if not vis_members:
        return None, None, 0.0

    idx = np.asarray(vis_members, dtype=np.int64)
    wts = np.maximum(0.05, b_idf[idx]).astype(np.float32)
    proto = (clip_img[idx] * wts[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(proto))
    if norm <= 1e-12:
        return vis_members[0], clip_img[vis_members[0]].copy(), float(np.mean(b_idf[idx]))
    proto = (proto / norm).astype(np.float32)

    scores = wts * (clip_img[idx] @ proto)
    best_local = int(np.argmax(scores))
    rep_idx = int(idx[best_local])
    mean_vidf = float(np.mean(b_idf[idx]))
    return rep_idx, proto, mean_vidf


def route_mm_seams(
    ds: NEDDataset,
    social_ds: SocialDataset,
    comb_graph: MessageGraph,
    labels_init: np.ndarray,
    txt_minilm: np.ndarray,
    clip_img: np.ndarray,
    decoded_ok: np.ndarray,
    b_idf: np.ndarray,
    k_total: int,
    merge_ratio: float = 0.70,
    min_text_sim: float = 0.42,
    vis_rescue_sim: float = 0.68,
    vis_rescue_min_txt: float = 0.18,
    vis_rescue_min_vidf: float = 0.50,
    min_cluster_size: int = 2,
    max_pairs_per_cluster: int = 3,
    img_paths_override: Optional[List[Optional[Path]]] = None,
) -> Tuple[List[MMSeamQuery], Dict[int, ClusterSummary]]:
    """Route and deterministically interleave multimodal merge-pair and split-seam queries."""
    img_paths = img_paths_override if img_paths_override is not None else ds.image_paths
    summaries = build_cluster_summaries(social_ds, txt_minilm, labels_init)

    # Precompute cluster visual representatives
    cids = sorted([c for c, s in summaries.items() if s.size >= min_cluster_size])
    vis_info: Dict[int, Tuple[Optional[int], Optional[np.ndarray], float]] = {}
    for c in cids:
        vis_info[c] = select_cluster_rep_image(summaries[c].members, clip_img, decoded_ok, b_idf)

    # Inter-cluster edge weights in combined multi-view graph
    coo = comb_graph.adj.tocoo()
    inter_w: Dict[Tuple[int, int], float] = defaultdict(float)
    for u, v, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if u >= v:
            continue
        cu, cv = int(labels_init[u]), int(labels_init[v])
        if cu != cv:
            key = (cu, cv) if cu < cv else (cv, cu)
            inter_w[key] += float(w)

    # 1. Detect candidate multimodal merge_pair queries (text_bridge + visual_rescue)
    merge_candidates: List[Tuple[float, int, int, float, float, float, float, float, str]] = []
    if len(cids) >= 2:
        protos = np.vstack([summaries[c].prototype for c in cids])
        sim_mat = protos @ protos.T
        for i in range(len(cids)):
            c1 = cids[i]
            s1 = summaries[c1]
            kw1 = set(s1.top_keywords)
            _, vp1, vidf1 = vis_info[c1]
            for j in range(i + 1, len(cids)):
                c2 = cids[j]
                s2 = summaries[c2]
                t_sim = float(sim_mat[i, j])
                cut_w = inter_w.get((c1, c2), 0.0)
                _, vp2, vidf2 = vis_info[c2]
                if vp1 is not None and vp2 is not None:
                    v_sim = float(np.dot(vp1, vp2))
                    min_vidf = min(vidf1, vidf2)
                else:
                    v_sim = 0.0
                    min_vidf = 0.0

                is_txt = (t_sim >= min_text_sim) or (cut_w >= 2.0)
                is_vis = (
                    (v_sim >= vis_rescue_sim)
                    and (t_sim >= vis_rescue_min_txt)
                    and (min_vidf >= vis_rescue_min_vidf)
                )
                if not (is_txt or is_vis):
                    continue

                route_src = "visual_rescue" if (is_vis and not is_txt) else "text_bridge"
                kw_ov = len(kw1.intersection(s2.top_keywords)) / max(
                    1, min(len(kw1), len(s2.top_keywords))
                )
                leverage = float(np.log1p(min(s1.size, s2.size)) * np.log1p(s1.size + s2.size))
                edge_bonus = 0.15 if cut_w > 0.0 else 0.0
                score = (
                    max(0.0, t_sim)
                    + 0.35 * max(0.0, v_sim - 0.55) * min_vidf
                    + 0.15 * kw_ov
                    + edge_bonus
                ) * (0.5 + 0.15 * leverage)
                merge_candidates.append(
                    (score, c1, c2, t_sim, v_sim, min_vidf, cut_w, kw_ov, route_src)
                )

    merge_candidates.sort(key=lambda x: -x[0])

    merge_mm: List[MMSeamQuery] = []
    cluster_use: Dict[int, int] = defaultdict(int)
    for score, c1, c2, t_sim, v_sim, min_vidf, cut_w, kw_ov, route_src in merge_candidates:
        if cluster_use[c1] >= max_pairs_per_cluster or cluster_use[c2] >= max_pairs_per_cluster:
            continue
        cluster_use[c1] += 1
        cluster_use[c2] += 1
        s1, s2 = summaries[c1], summaries[c2]
        rep1, _, _ = vis_info[c1]
        rep2, _, _ = vis_info[c2]
        p_nodes = list(dict.fromkeys([n for n in [s1.medoids[0], s2.medoids[0], rep1, rep2] if n is not None]))
        q_item = QueryItem(
            query_id=f"merge_c{c1}_c{c2}",
            strategy="evid_msed",
            query_type="merge_pair",
            score=score,
            primary_nodes=p_nodes,
            affected_group=s1.members + s2.members,
            candidate_clusters=[c1, c2],
            merge_c1=c1,
            merge_c2=c2,
            merge_c1_medoids=s1.medoids[:3],
            merge_c2_medoids=s2.medoids[:3],
            meta={
                "proto_sim": t_sim,
                "vis_sim": v_sim,
                "min_vidf": min_vidf,
                "inter_cut_w": cut_w,
                "kw_overlap": kw_ov,
                "size1": float(s1.size),
                "size2": float(s2.size),
                "is_visual_rescue": 1.0 if route_src == "visual_rescue" else 0.0,
            },
        )
        merge_mm.append(
            MMSeamQuery(
                query=q_item,
                brief_a=_extract_brief(ds, s1.members, s1.medoids[:3]),
                brief_b=_extract_brief(ds, s2.members, s2.medoids[:3]),
                rep_img_a=img_paths[rep1] if rep1 is not None else None,
                rep_img_b=img_paths[rep2] if rep2 is not None else None,
                rep_post_a=rep1,
                rep_post_b=rep2,
                route_source=route_src,
            )
        )

    # 2. Detect intra-cluster false-merge seams over combined graph
    raw_seams = detect_seam_queries(
        social_ds,
        comb_graph,
        labels_init,
        summaries,
        min_cluster_size=10,
        min_pocket_size=4,
        max_conductance=0.50,
    )
    seam_mm: List[MMSeamQuery] = []
    for sq in raw_seams:
        rep_a, _, _ = select_cluster_rep_image(sq.seam_sub1, clip_img, decoded_ok, b_idf)
        rep_b, _, _ = select_cluster_rep_image(sq.seam_sub2, clip_img, decoded_ok, b_idf)
        p_nodes = list(
            dict.fromkeys(
                [n for n in [sq.seam_sub1_medoids[0], sq.seam_sub2_medoids[0], rep_a, rep_b] if n is not None]
            )
        )
        sq.strategy = "evid_msed"
        sq.primary_nodes = p_nodes
        seam_mm.append(
            MMSeamQuery(
                query=sq,
                brief_a=_extract_brief(ds, sq.seam_sub1, sq.seam_sub1_medoids),
                brief_b=_extract_brief(ds, sq.seam_sub2, sq.seam_sub2_medoids),
                rep_img_a=img_paths[rep_a] if rep_a is not None else None,
                rep_img_b=img_paths[rep_b] if rep_b is not None else None,
                rep_post_a=rep_a,
                rep_post_b=rep_b,
                route_source="intra_seam",
            )
        )

    # 3. Interleave up to k_total maintaining merge_ratio across every prefix slice
    k_merge_target = int(round(k_total * merge_ratio))
    k_seam_target = max(0, k_total - k_merge_target)
    m_pool = merge_mm[:k_merge_target]
    s_pool = seam_mm[:k_seam_target]
    # If one pool runs short, fill remaining slots cleanly from the other unused tail
    if len(m_pool) + len(s_pool) < k_total:
        need = k_total - (len(m_pool) + len(s_pool))
        extra_m = merge_mm[k_merge_target : k_merge_target + need]
        m_pool = m_pool + extra_m
        need = k_total - (len(m_pool) + len(s_pool))
        if need > 0:
            s_pool = s_pool + seam_mm[k_seam_target : k_seam_target + need]

    routed: List[MMSeamQuery] = []
    mi = 0
    si = 0
    eff_merge_ratio = len(m_pool) / max(1, len(m_pool) + len(s_pool))
    while mi < len(m_pool) or si < len(s_pool):
        if len(routed) >= k_total:
            break
        take_seam = False
        if mi >= len(m_pool):
            take_seam = True
        elif si < len(s_pool):
            cur_s_ratio = si / max(1, mi + si)
            if mi >= 1 and cur_s_ratio < (1.0 - eff_merge_ratio):
                take_seam = True
        if take_seam:
            routed.append(s_pool[si])
            si += 1
        else:
            routed.append(m_pool[mi])
            mi += 1

    return routed, summaries


def _oracle_same_event(q: QueryItem, y_true: np.ndarray, summaries: Dict[int, ClusterSummary]) -> bool:
    """Compute true cluster/pocket majority compatibility visible to an Oracle judge."""
    if q.query_type == "merge_pair":
        m1 = summaries[int(q.merge_c1)].members
        m2 = summaries[int(q.merge_c2)].members
    else:
        m1 = q.seam_sub1
        m2 = q.seam_sub2
    v1, c1 = np.unique(y_true[m1], return_counts=True)
    v2, c2 = np.unique(y_true[m2], return_counts=True)
    return int(v1[int(np.argmax(c1))]) == int(v2[int(np.argmax(c2))])


def decisions_to_llm_decisions(
    routed_queries: List[MMSeamQuery],
    ds_labels: np.ndarray,
    summaries: Dict[int, ClusterSummary],
    d_txt_map: Dict[str, SeamVLDecision],
    d_mm_map: Dict[str, SeamVLDecision],
    labels_stage1: np.ndarray,
    policy: str,
    conf_threshold: float = 0.60,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Convert raw text+multimodal VLM responses into gated LLMDecisions and audit partitions."""
    llm_decs: List[LLMDecision] = []
    raw_queries = [mq.query for mq in routed_queries]

    merge_ok = merge_tot = 0
    seam_ok = seam_tot = 0
    n_rejected = 0
    n_conflict_flagged = 0
    real_tokens_spent = 0

    for mq in routed_queries:
        q = mq.query
        d_txt = d_txt_map[q.query_id]
        d_mm = d_mm_map[q.query_id]

        if d_mm.visual_conflict_with_text is True:
            n_conflict_flagged += 1

        rejected = False
        if policy == "text_only_vlm":
            same = d_txt.same_event_text_only
            conf = d_txt.confidence
            toks = d_txt.prompt_tokens + d_txt.completion_tokens
        elif policy == "naive_vlm":
            same = d_mm.same_event_multimodal
            conf = d_mm.confidence
            toks = d_mm.prompt_tokens + d_mm.completion_tokens
        elif policy == "evid_self_gate_inline":
            accept_vis = (d_mm.visual_discriminative is True) and (d_mm.visual_conflict_with_text is False)
            if accept_vis:
                same = d_mm.same_event_multimodal
            else:
                same = d_mm.same_event_text_only
                rejected = True
            conf = d_mm.confidence
            toks = d_mm.prompt_tokens + d_mm.completion_tokens
        elif policy == "evid_self_gate_clean_fallback":
            accept_vis = (d_mm.visual_discriminative is True) and (d_mm.visual_conflict_with_text is False)
            if accept_vis:
                same = d_mm.same_event_multimodal
                conf = d_mm.confidence
                toks = d_mm.prompt_tokens + d_mm.completion_tokens
            else:
                same = d_txt.same_event_text_only
                conf = d_txt.confidence
                toks = (d_mm.prompt_tokens + d_mm.completion_tokens) + (
                    d_txt.prompt_tokens + d_txt.completion_tokens
                )
                rejected = True
        elif policy == "evid_comp_gate_clean_fallback":
            accept_vis = (
                (d_mm.visual_discriminative is True)
                and (d_mm.visual_conflict_with_text is False)
                and (d_mm.confidence >= 0.70)
            )
            if accept_vis:
                same = d_mm.same_event_multimodal
                conf = d_mm.confidence
                toks = d_mm.prompt_tokens + d_mm.completion_tokens
            else:
                same = d_txt.same_event_text_only
                conf = d_txt.confidence
                toks = (d_mm.prompt_tokens + d_mm.completion_tokens) + (
                    d_txt.prompt_tokens + d_txt.completion_tokens
                )
                rejected = True
        else:
            raise ValueError(f"Unknown Stage-2 gating policy: {policy}")

        if rejected:
            n_rejected += 1
        real_tokens_spent += int(toks)

        oracle_same = _oracle_same_event(q, ds_labels, summaries)
        pred_correct = (same is not None) and (bool(same) == bool(oracle_same))
        if q.query_type == "merge_pair":
            merge_tot += 1
            if pred_correct:
                merge_ok += 1
            action = "MERGE" if same is True else "SEPARATE"
        else:
            seam_tot += 1
            if pred_correct:
                seam_ok += 1
            action = "KEEP_MERGED" if same is True else "SPLIT"

        if same is None:
            conf = 0.0

        llm_decs.append(
            LLMDecision(
                query_id=q.query_id,
                query_type=q.query_type,
                action=action,
                target_cluster=q.seam_cluster_id if q.query_type == "seam" else None,
                confidence=float(conf),
                event_cue=f"{d_mm.event_a} | {d_mm.event_b}",
                prompt_tokens=int(toks),
                completion_tokens=0,
            )
        )

    labels_after, queried_mask = apply_direct_assignment(
        labels_stage1,
        raw_queries,
        llm_decs,
        conf_threshold=conf_threshold,
    )
    aud = full_partition_audit(
        ds_labels,
        labels_after,
        y_before=labels_stage1,
        queried_mask=queried_mask,
    )
    et = aud["error_transition"]
    pa = aud["propagation_audit"]
    n_q = max(1, len(routed_queries))

    metrics = {
        "n_queries": int(len(routed_queries)),
        "ami": round(aud["ami"], 4),
        "delta_ami_vs_stage1": round(aud["delta_ami"], 4),
        "nmi": round(aud["nmi"], 4),
        "ari": round(aud["ari"], 4),
        "macro_b3": round(aud["bcubed_full"]["macro_b3_f1"], 4),
        "delta_macro_b3_vs_stage1": round(aud["delta_macro_b3_f1"], 4),
        "micro_b3": round(aud["bcubed_full"]["micro_b3_f1"], 4),
        "delta_micro_b3_vs_stage1": round(aud["delta_micro_b3_f1"], 4),
        "n_clusters": int(aud["n_pred_clusters"]),
        "net_fm_red": int(et["net_false_merge_reduction"]),
        "net_fs_red": int(et["net_false_split_reduction"]),
        "unqueried_improved_posts": int(pa["unqueried_improved_posts"]),
        "unqueried_degraded_posts": int(pa["unqueried_degraded_posts"]),
        "unqueried_net_improved_posts": int(pa["unqueried_net_improved_posts"]),
        "unqueried_posts_fixed_per_query": round(pa["unqueried_net_improved_posts"] / n_q, 2),
        "merge_acc": round(merge_ok / max(1, merge_tot), 4),
        "merge_total": int(merge_tot),
        "seam_acc": round(seam_ok / max(1, seam_tot), 4),
        "seam_total": int(seam_tot),
        "overall_seam_acc": round((merge_ok + seam_ok) / n_q, 4),
        "gate_reject_rate": round(n_rejected / n_q, 4),
        "conflict_flag_rate": round(n_conflict_flagged / n_q, 4),
        "real_tokens_spent": int(real_tokens_spent),
    }
    return metrics, labels_after
