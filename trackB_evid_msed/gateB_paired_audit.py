"""Gate B Two-Tier Paired Causal Visual Evidence Audit on Concurrent NED Dev Stream.

Implements the pre-registered Gate B diagnostic from Proposal_TrackB_Evid-MSED.md:
  - Strictly uses ONLY the 8 NED development events (`E15, E23, E22, E33, E40, E11, E24, E38`),
    preserving complete isolation from validation and holdout splits.
  - Eliminates the 72-hour NED time shortcut via event-relative concurrent timelines and
    forbids exposing event names or calendar dates/years to any model or prompt.

Two complementary evaluation tiers:
  - Tier 1 (Full Dev Stream, n=2,665 posts / 2,131 images, 0 API cost):
      1A. Paired 3-condition candidate classification (Text-Only / Orig-Img / Mismatched-Img)
          measuring exact N_rescue, N_mislead, N_mismatch_flip, Net Gain, Acc & Macro-F1 across
          Text-Only, Naive-Visual, Visual-IDF, Raw CNTD, Time/Competitor-CNTD, and Comp-Margin.
      1B. Unsupervised Multi-View 2D-SE Clustering over the full concurrent stream (testing our
          corrected Proposal B §3.3 unified soft-imputation formula without the broken delta_i*delta_j).
  - Tier 2 (`Qwen/Qwen3-VL-8B-Instruct` 3-Condition Paired Causal VLM Audit, N=120 x 3 = 360 calls):
      Strict real-token ledger via SiliconFlow evaluating Text-Only vs. Orig-Img vs. Mismatched-Img
      and Competitor-Conditioned VLM Gating, plus extracting concrete rescue/mislead case studies.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import build_message_graph
from core_shared.key_loader import load_provider_key
from core_shared.metrics_audit import full_partition_audit
from core_shared.se_solver import solve_2d_se
from core_shared.token_ledger import CumulativeTokenBucket, TokenLedger
from trackB_evid_msed.ned_loader import (
    DEV_COMPETITOR_PAIRS,
    NEDDataset,
    dhash_hamming,
    load_ned_split,
)
from trackB_evid_msed.vlm_client import BudgetedVLMClient, VLDecision

CACHE_DIR = ROOT / "cache" / "trackB" / "ned_development"
RESULTS_DIR = ROOT / "results" / "trackB_gateB"


def _load_cached_features() -> Tuple[NEDDataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = load_ned_split(split="development")
    txt_minilm = np.load(CACHE_DIR / "text_minilm.npy").astype(np.float32)
    clip_img = np.load(CACHE_DIR / "clip_img.npy").astype(np.float32)
    clip_txt = np.load(CACHE_DIR / "clip_txt.npy").astype(np.float32)
    meta = np.load(CACHE_DIR / "image_meta.npz")
    decoded_ok = meta["decoded_ok"].astype(bool)
    dhash64 = meta["dhash64"].astype(np.uint64)
    return ds, txt_minilm, clip_img, clip_txt, decoded_ok, dhash64


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> float:
    f1s = []
    for c in labels:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        if tp + fp + fn == 0:
            continue
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1s.append((2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _build_anchors_and_queries(
    ds: NEDDataset,
    txt_minilm: np.ndarray,
    clip_img: np.ndarray,
    decoded_ok: np.ndarray,
    dhash64: np.ndarray,
    anchor_frac: float = 0.25,
) -> Dict[str, Any]:
    """Chronological first 25% of each dev event forms unlabeled-to-VLM cluster anchors; rest is query pool."""
    n = len(ds)
    n_events = len(ds.label_to_event)
    ev_to_idx = {ev: i for i, ev in enumerate(ds.label_to_event)}

    anchor_indices_by_ev: Dict[int, List[int]] = defaultdict(list)
    query_img_indices: List[int] = []

    for ev_i in range(n_events):
        members = [i for i in range(n) if int(ds.labels[i]) == ev_i]
        members.sort(key=lambda idx: (ds.raw_timestamps[idx], ds.post_ids[idx]))
        n_anc = max(5, int(round(anchor_frac * len(members))))
        anc = members[:n_anc]
        qry = [idx for idx in members[n_anc:] if decoded_ok[idx]]
        anchor_indices_by_ev[ev_i] = anc
        query_img_indices.extend(qry)

    query_img_indices.sort()
    anchor_all = [idx for lst in anchor_indices_by_ev.values() for idx in lst]
    anchor_img_hashes = [int(dhash64[idx]) for idx in anchor_all if decoded_ok[idx]]

    proto_txt = np.zeros((n_events, txt_minilm.shape[1]), dtype=np.float32)
    proto_vis = np.zeros((n_events, clip_img.shape[1]), dtype=np.float32)
    anchor_briefs: Dict[int, Tuple[List[str], List[str]]] = {}

    for ev_i, anc in anchor_indices_by_ev.items():
        mt = txt_minilm[anc].mean(axis=0)
        mt /= max(float(np.linalg.norm(mt)), 1e-12)
        proto_txt[ev_i] = mt

        vis_anc = [idx for idx in anc if decoded_ok[idx]]
        if vis_anc:
            mv = clip_img[vis_anc].mean(axis=0)
            mv /= max(float(np.linalg.norm(mv)), 1e-12)
            proto_vis[ev_i] = mv

        # Pick 3 most representative medoid anchor texts + 5 keywords (NO event_name!)
        anc_arr = np.asarray(anc, dtype=np.int64)
        sims = txt_minilm[anc_arr] @ mt
        med_pos = np.argsort(-sims)[:3]
        med_texts = [ds.texts[int(anc_arr[p])] for p in med_pos]
        kw_cnt: Dict[str, int] = defaultdict(int)
        for idx in anc:
            for h in ds.hashtags[idx]:
                kw_cnt[f"#{h}"] += 2
            for tk in ds.tokens[idx]:
                kw_cnt[tk] += 1
        top_kw = [k for k, _ in sorted(kw_cnt.items(), key=lambda kv: -kv[1])[:5]]
        anchor_briefs[ev_i] = (top_kw, med_texts)

    # Check which query images are visually novel vs anchors (dHash Hamming > 6)
    novel_mask = np.zeros(len(query_img_indices), dtype=bool)
    for pos, qidx in enumerate(query_img_indices):
        qh = int(dhash64[qidx])
        min_hd = min((dhash_hamming(qh, ah) for ah in anchor_img_hashes), default=64)
        novel_mask[pos] = min_hd > 6

    # Candidate sets C_cand(i) (4 competing events: true + same-theme competitor + top-2 nearest others)
    cand_sets: Dict[int, List[int]] = {}
    mismatch_map: Dict[int, int] = {}
    rng = np.random.default_rng(20261122)

    valid_img_by_ev = {
        ev_i: [idx for idx in range(n) if int(ds.labels[idx]) == ev_i and decoded_ok[idx]]
        for ev_i in range(n_events)
    }

    for qidx in query_img_indices:
        y = int(ds.labels[qidx])
        ev_str = ds.label_to_event[y]
        comp_str = DEV_COMPETITOR_PAIRS[ev_str]
        comp_y = ev_to_idx[comp_str]

        t_sims = proto_txt @ txt_minilm[qidx]
        v_sims = proto_vis @ clip_img[qidx]
        comb = np.maximum(t_sims, v_sims)
        others = [c for c in range(n_events) if c not in {y, comp_y}]
        others.sort(key=lambda c: -float(comb[c]))
        cands = [y, comp_y] + others[:2]
        cand_sets[qidx] = cands

        # Deterministic mismatched image from competing event (must differ by dHash > 6 from query)
        pool_comp = valid_img_by_ev[comp_y]
        qh = int(dhash64[qidx])
        non_dup = [j for j in pool_comp if dhash_hamming(qh, int(dhash64[j])) > 6]
        pick_pool = non_dup if non_dup else pool_comp
        mismatch_map[qidx] = int(pick_pool[int(rng.integers(0, len(pick_pool)))])

    return {
        "anchor_by_ev": anchor_indices_by_ev,
        "anchor_set": set(anchor_all),
        "query_indices": query_img_indices,
        "novel_mask": novel_mask,
        "proto_txt": proto_txt,
        "proto_vis": proto_vis,
        "anchor_briefs": anchor_briefs,
        "cand_sets": cand_sets,
        "mismatch_map": mismatch_map,
    }


def _compute_post_evidence_scores(
    txt_minilm: np.ndarray,
    clip_img: np.ndarray,
    decoded_ok: np.ndarray,
    proto_vis: np.ndarray,
    cand_sets: Dict[int, List[int]],
) -> Dict[str, np.ndarray]:
    """Compute unsupervised visual evidence scores for all dev posts (zero label usage)."""
    n = len(decoded_ok)
    vis_idx = np.where(decoded_ok)[0]
    emb_v = clip_img[vis_idx]
    emb_t = txt_minilm[vis_idx]

    # Visual similarity matrix among all valid image posts
    vsim = emb_v @ emb_v.T
    np.fill_diagonal(vsim, -1.0)

    # 1. Visual IDF (downweights high-repeat / viral template images with >= 0.88 CLIP neighbors)
    b_idf = np.zeros(n, dtype=np.float32)
    # 2. Raw CNTD (penalizes ANY image whose top visual neighbors have dispersed texts)
    b_cntd_raw = np.zeros(n, dtype=np.float32)
    # 3. Conflict-Conditioned CNTD (only penalizes if visual neighbors disagree across distant text clusters!)
    b_cntd_cond = np.zeros(n, dtype=np.float32)

    for li, gi in enumerate(vis_idx.tolist()):
        sims_row = vsim[li]
        near_dup_cnt = int(np.sum(sims_row >= 0.88))
        b_idf[gi] = float(np.log(2.0) / np.log(2.0 + near_dup_cnt))

        # Top visual neighbors (sim >= 0.65, up to 8)
        nb_pos = np.where(sims_row >= 0.65)[0]
        if len(nb_pos) >= 2:
            if len(nb_pos) > 8:
                top_order = np.argsort(-sims_row[nb_pos])[:8]
                nb_pos = nb_pos[top_order]
            nb_t = emb_t[nb_pos]
            pw = nb_t @ nb_t.T
            m = len(nb_pos)
            tri = pw[np.triu_indices(m, k=1)]
            mean_t_sim = float(np.mean(tri)) if len(tri) > 0 else 1.0
            dispersion = max(0.0, 1.0 - mean_t_sim)
            b_cntd_raw[gi] = float(max(0.05, 1.0 - dispersion))

            # Time/compatibility-aware variant: only penalize when high dispersion coincides
            # with near-zero semantic consistency (< 0.15) AND repeated image hubbing
            hard_conflict = float(np.mean(tri < 0.12)) if len(tri) > 0 else 0.0
            b_cntd_cond[gi] = float(max(0.10, 1.0 - 0.7 * hard_conflict)) * b_idf[gi]
        else:
            b_cntd_raw[gi] = 1.0
            b_cntd_cond[gi] = b_idf[gi]

    # 4. Competitor-Conditioned Discriminability (Top-1 vs Top-2 visual margin over local candidates)
    b_comp_margin = np.zeros(n, dtype=np.float32)
    # Full visual prototypes on all dev vs candidate sets
    all_v_protos = proto_vis  # (8, 512)
    for gi in vis_idx.tolist():
        if gi in cand_sets:
            cands = cand_sets[gi]
            vs = np.sort(all_v_protos[cands] @ clip_img[gi])[::-1]
        else:
            vs = np.sort(all_v_protos @ clip_img[gi])[::-1]
        margin = max(0.0, float(vs[0] - vs[1])) if len(vs) >= 2 else 0.0
        # Normalized saturating margin * visual specificity
        margin_score = float(min(1.0, margin / 0.10))
        b_comp_margin[gi] = margin_score * b_idf[gi]

    return {
        "naive": decoded_ok.astype(np.float32),
        "visual_idf": b_idf,
        "cntd_raw": b_cntd_raw,
        "cntd_cond": b_cntd_cond,
        "comp_margin": b_comp_margin,
    }


def run_tier1_audit(
    ds: NEDDataset,
    txt_minilm: np.ndarray,
    clip_img: np.ndarray,
    decoded_ok: np.ndarray,
    dhash64: np.ndarray,
    setup: Dict[str, Any],
    evid_scores: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Run Tier 1A (3-condition candidate classification) & Tier 1B (multi-view 2D-SE)."""
    query_idx = setup["query_indices"]
    novel_mask = setup["novel_mask"]
    proto_txt = setup["proto_txt"]
    proto_vis = setup["proto_vis"]
    cand_sets = setup["cand_sets"]
    mismatch_map = setup["mismatch_map"]

    y_true = np.array([int(ds.labels[i]) for i in query_idx], dtype=np.int64)
    ev_labels = list(range(len(ds.label_to_event)))

    # 1A: Paired classification under lambda_V sweep
    def _eval_scorer(
        b_arr: np.ndarray,
        lam_v: float,
        img_vec_override: Optional[Dict[int, np.ndarray]] = None,
        use_text_conf_guard: bool = False,
    ) -> np.ndarray:
        preds = np.empty(len(query_idx), dtype=np.int64)
        for pos, gi in enumerate(query_idx):
            cands = np.asarray(cand_sets[gi], dtype=np.int64)
            ts = proto_txt[cands] @ txt_minilm[gi]
            t_sorted = np.sort(ts)[::-1]
            t_margin = float(t_sorted[0] - t_sorted[1]) if len(t_sorted) >= 2 else 1.0

            v_vec = img_vec_override[gi] if img_vec_override is not None else clip_img[gi]
            if img_vec_override is not None:
                # Recompute competitor margin dynamically against actual supplied image (for mismatched_img!)
                vs_raw = proto_vis[cands] @ v_vec
                vs_s = np.sort(vs_raw)[::-1]
                dyn_margin = float(min(1.0, max(0.0, vs_s[0] - vs_s[1]) / 0.10))
                w_vis = dyn_margin * b_arr[gi] if use_text_conf_guard else b_arr[gi]
            else:
                vs_raw = proto_vis[cands] @ v_vec
                w_vis = float(b_arr[gi])

            # Optional confidence guard: if text already has huge margin (>0.15) & weak visual evidence, damp visual override
            eff_lam = lam_v
            if use_text_conf_guard and t_margin > 0.14 and w_vis < 0.45:
                eff_lam = 0.15 * lam_v

            tot = ts + eff_lam * w_vis * vs_raw
            preds[pos] = int(cands[int(np.argmax(tot))])
        return preds

    # Compute Text-Only baseline predictions
    pred_txt = _eval_scorer(np.zeros(len(ds), dtype=np.float32), lam_v=0.0)
    txt_ok = pred_txt == y_true
    mismatch_vecs = {gi: clip_img[mismatch_map[gi]] for gi in query_idx}

    def _pair_metrics(pred_orig: np.ndarray, pred_mis: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if mask is None:
            mask = np.ones(len(query_idx), dtype=bool)
        yt = y_true[mask]
        pt = txt_ok[mask]
        po = pred_orig[mask] == yt
        pm = pred_mis[mask] == yt

        n_rescue = int(np.sum((~pt) & po))
        n_mislead = int(np.sum(pt & (~po)))
        n_mismatch_flip = int(np.sum(pt & (~pm)))
        return {
            "n_eval": int(mask.sum()),
            "text_acc": round(float(np.mean(pt)), 4),
            "orig_acc": round(float(np.mean(po)), 4),
            "orig_macro_f1": round(_macro_f1(yt, pred_orig[mask], ev_labels), 4),
            "mismatch_acc": round(float(np.mean(pm)), 4),
            "n_rescue": n_rescue,
            "n_mislead": n_mislead,
            "net_rescue_minus_mislead": n_rescue - n_mislead,
            "n_mismatch_flip": n_mismatch_flip,
        }

    # For mismatched_img under comp_margin, pass visual_idf vector so _eval_scorer recomputes dynamic margin!
    b_idf = evid_scores["visual_idf"]
    scorers_spec = [
        ("text_only", evid_scores["naive"] * 0.0, False, False),
        ("naive_visual", evid_scores["naive"], False, False),
        ("visual_idf", evid_scores["visual_idf"], False, False),
        ("cntd_raw", evid_scores["cntd_raw"], False, False),
        ("cntd_cond", evid_scores["cntd_cond"], False, False),
        ("comp_margin", evid_scores["comp_margin"], False, False),
        ("comp_margin_guarded", evid_scores["comp_margin"], True, True),
    ]

    tier1a_table: Dict[str, Any] = {}
    for lam in [0.50, 0.75, 1.00]:
        lam_rows: Dict[str, Any] = {}
        for sname, barr, is_guarded, dyn_mis in scorers_spec:
            if sname == "text_only":
                po = pred_txt
                pm = pred_txt
            else:
                po = _eval_scorer(barr, lam_v=lam, use_text_conf_guard=is_guarded)
                pm = _eval_scorer(
                    b_idf if dyn_mis or sname == "comp_margin" else barr,
                    lam_v=lam,
                    img_vec_override=mismatch_vecs,
                    use_text_conf_guard=is_guarded,
                )
            lam_rows[sname] = {
                "all_query": _pair_metrics(po, pm, None),
                "novel_non_dup": _pair_metrics(po, pm, novel_mask),
            }
        tier1a_table[f"lambda_{lam:.2f}"] = lam_rows

    # 1B: Unsupervised Multi-View 2D-SE Clustering on FULL concurrent stream (including missing-image posts!)
    # Uses Proposal B §3.3 unified corrected soft-imputation formula (NO delta_i * delta_j!)
    social_ds = SocialDataset(
        name="NED_dev_concurrent",
        tweet_ids=np.asarray(ds.post_ids, dtype=object),
        texts=ds.texts,
        labels=ds.labels,
        tokens_lemma=ds.tokens,
        entities=[[] for _ in range(len(ds))],
        hashtags=ds.hashtags,
        user_ids=["" for _ in range(len(ds))],
        mentioned_users=ds.mentions,
        timestamps=ds.rel_timestamps,  # Concurrent timeline! No 72h shortcut!
        dates=["concurrent" for _ in range(len(ds))],
    )
    text_graph = build_message_graph(
        social_ds,
        txt_minilm,
        knn_k=12,
        knn_sim_threshold=0.38,
        attr_alpha=0.45,
        time_decay_per_day=0.0,
    )
    labels_txt_se, _ = solve_2d_se(text_graph.adj, refine_moves=True)
    aud_txt_se = full_partition_audit(ds.labels, labels_txt_se)

    n = len(ds)
    adj_t_csr = text_graph.adj.tocsr()

    def _run_mv_2dse(name: str, b_raw: np.ndarray, lam_v: float = 0.65, tau_imp: float = 0.55) -> Dict[str, Any]:
        # Proposal B §3.3 Unified Imputation without delta_i * delta_j
        z_eff = clip_img.copy()
        b_eff = np.where(decoded_ok, b_raw, 0.0).astype(np.float32)

        for i in range(n):
            if decoded_ok[i]:
                continue
            st, en = adj_t_csr.indptr[i], adj_t_csr.indptr[i + 1]
            if st == en:
                continue
            nbs = adj_t_csr.indices[st:en]
            w_nbs = adj_t_csr.data[st:en]
            img_nb = decoded_ok[nbs]
            if not np.any(img_nb):
                continue
            b_mean = float(np.mean(b_raw[nbs[img_nb]]))
            if b_mean >= tau_imp:
                wts = w_nbs[img_nb] * np.maximum(0.05, b_raw[nbs[img_nb]])
                proto = (clip_img[nbs[img_nb]] * wts[:, None]).sum(axis=0)
                norm = float(np.linalg.norm(proto))
                if norm > 1e-12:
                    z_eff[i] = proto / norm
                    b_eff[i] = 0.7 * b_mean

        has_eff = b_eff > 0.0
        eff_idx = np.where(has_eff)[0]
        if len(eff_idx) < 2:
            return {"ami": aud_txt_se["ami"], "macro_b3": aud_txt_se["bcubed_full"]["macro_b3_f1"]}

        z_sub = z_eff[eff_idx]
        sims = z_sub @ z_sub.T
        np.fill_diagonal(sims, -1.0)
        rows_l, cols_l, vals_l = [], [], []
        k_v = min(10, len(eff_idx) - 1)
        top_pos = np.argpartition(-sims, kth=k_v - 1, axis=1)[:, :k_v]
        for li in range(len(eff_idx)):
            gi = int(eff_idx[li])
            for pj in top_pos[li]:
                s = float(sims[li, pj])
                if s >= 0.72:
                    gj = int(eff_idx[pj])
                    w_v = float(b_eff[gi] * b_eff[gj] * s)
                    if w_v > 0.0:
                        rows_l.extend([gi, gj])
                        cols_l.extend([gj, gi])
                        vals_l.extend([w_v, w_v])

        w_vis_mat = sparse.csr_matrix((vals_l, (rows_l, cols_l)), shape=(n, n), dtype=np.float64)
        w_vis_mat = w_vis_mat.maximum(w_vis_mat.T)
        comb_adj = (text_graph.adj + lam_v * w_vis_mat).tocsr()
        comb_adj.setdiag(0.0)
        comb_adj.eliminate_zeros()

        lab_mv, _ = solve_2d_se(comb_adj, refine_moves=True)
        aud_mv = full_partition_audit(ds.labels, lab_mv, y_before=labels_txt_se)
        et = aud_mv["error_transition"]
        return {
            "ami": round(aud_mv["ami"], 4),
            "delta_ami": round(aud_mv["delta_ami"], 4),
            "macro_b3": round(aud_mv["bcubed_full"]["macro_b3_f1"], 4),
            "delta_macro_b3": round(aud_mv["delta_macro_b3_f1"], 4),
            "micro_b3": round(aud_mv["bcubed_full"]["micro_b3_f1"], 4),
            "delta_micro_b3": round(aud_mv["delta_micro_b3_f1"], 4),
            "n_clusters": int(aud_mv["n_pred_clusters"]),
            "net_fm_red": int(et["net_false_merge_reduction"]),
            "net_fs_red": int(et["net_false_split_reduction"]),
        }

    # For unsupervised MV-2DSE, construct label-free local-neighborhood margin evidence
    b_unsub_margin = np.zeros(n, dtype=np.float32)
    vis_all = np.where(decoded_ok)[0]
    # Top-2 distinct text cluster visual prototype margin using labels_txt_se (strictly unsupervised!)
    txt_cids = [c for c in np.unique(labels_txt_se).tolist() if np.sum((labels_txt_se == c) & decoded_ok) >= 3]
    if len(txt_cids) >= 2:
        t_vis_protos = []
        for c in txt_cids:
            idx_c = np.where((labels_txt_se == c) & decoded_ok)[0]
            mv = clip_img[idx_c].mean(axis=0)
            mv /= max(float(np.linalg.norm(mv)), 1e-12)
            t_vis_protos.append(mv)
        tvp = np.vstack(t_vis_protos)
        for gi in vis_all.tolist():
            sv = np.sort(tvp @ clip_img[gi])[::-1]
            marg = min(1.0, max(0.0, float(sv[0] - sv[1])) / 0.10)
            b_unsub_margin[gi] = marg * evid_scores["visual_idf"][gi]

    tier1b_clustering = {
        "text_only_2dse": {
            "ami": round(aud_txt_se["ami"], 4),
            "macro_b3": round(aud_txt_se["bcubed_full"]["macro_b3_f1"], 4),
            "micro_b3": round(aud_txt_se["bcubed_full"]["micro_b3_f1"], 4),
            "n_clusters": int(aud_txt_se["n_pred_clusters"]),
        },
        "naive_mv_2dse": _run_mv_2dse("naive", evid_scores["naive"]),
        "visual_idf_mv_2dse": _run_mv_2dse("visual_idf", evid_scores["visual_idf"]),
        "cntd_raw_mv_2dse": _run_mv_2dse("cntd_raw", evid_scores["cntd_raw"]),
        "cntd_cond_mv_2dse": _run_mv_2dse("cntd_cond", evid_scores["cntd_cond"]),
        "unsup_comp_margin_mv_2dse": _run_mv_2dse("unsup_margin", b_unsub_margin),
    }

    return {
        "n_dev_posts": n,
        "n_valid_images": int(decoded_ok.sum()),
        "n_query_images": len(query_idx),
        "n_novel_query_images": int(novel_mask.sum()),
        "tier1a_paired_classification": tier1a_table,
        "tier1b_unsupervised_mv_2dse": tier1b_clustering,
    }


def _select_stratified_vlm_sample(
    ds: NEDDataset,
    txt_minilm: np.ndarray,
    setup: Dict[str, Any],
    evid_scores: Dict[str, np.ndarray],
    n_total: int = 120,
) -> List[int]:
    """Select n_total novel-image query posts balanced across all 8 dev events & text difficulty."""
    query_idx = setup["query_indices"]
    novel_mask = setup["novel_mask"]
    proto_txt = setup["proto_txt"]
    cand_sets = setup["cand_sets"]
    n_events = len(ds.label_to_event)

    # Group novel query indices by true event, computing text margin (hard vs easy)
    by_ev: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for pos, gi in enumerate(query_idx):
        if not novel_mask[pos]:
            continue
        cands = np.asarray(cand_sets[gi], dtype=np.int64)
        y = int(ds.labels[gi])
        ts = proto_txt[cands] @ txt_minilm[gi]
        pred_t = int(cands[int(np.argmax(ts))])
        t_sorted = np.sort(ts)[::-1]
        margin = float(t_sorted[0] - t_sorted[1]) if len(t_sorted) >= 2 else 1.0
        # Prioritize balanced coverage + borderline text cases where vision truly decides
        # Combine exact wrong/low-margin text + clear high-margin text
        by_ev[y].append((margin, gi))

    # Proportional quota with floor >= 8 per event (ensures rare events like E24/E40/E38 well covered)
    selected: List[int] = []
    # Base minimum 10 per event (8*10=80), remaining 40 weighted by event query size
    rem_pool: List[Tuple[float, int]] = []
    for ev_i in range(n_events):
        pool = by_ev[ev_i]
        pool.sort(key=lambda x: x[0])  # lowest text margin -> highest text margin
        k_take = min(len(pool), 10)
        # Uniform quantile sampling across text-margin spectrum so both text-wrong & text-right are represented
        if k_take > 0:
            picks = np.linspace(0, len(pool) - 1, num=k_take, dtype=int)
            picked_set = set()
            for p in picks.tolist():
                gid = pool[p][1]
                if gid not in picked_set:
                    selected.append(gid)
                    picked_set.add(gid)
            for m_val, gid in pool:
                if gid not in picked_set:
                    rem_pool.append((m_val, gid))

    need = max(0, n_total - len(selected))
    if need > 0 and rem_pool:
        rem_pool.sort(key=lambda x: x[0])
        # Sample evenly from bottom 70% (competing/hard cases) + top 30% (clear text cases)
        picks = np.linspace(0, len(rem_pool) - 1, num=min(need, len(rem_pool)), dtype=int)
        for p in picks.tolist():
            gid = rem_pool[p][1]
            if gid not in selected:
                selected.append(gid)

    return selected[:n_total]


def run_tier2_vlm_audit(
    ds: NEDDataset,
    txt_minilm: np.ndarray,
    clip_img: np.ndarray,
    setup: Dict[str, Any],
    evid_scores: Dict[str, np.ndarray],
    vlm_sample_idx: List[int],
    provider: str,
    model_name: str,
    dry_run: bool,
) -> Dict[str, Any]:
    """Run Tier 2 3-condition paired causal VLM audit (`Qwen/Qwen3-VL-8B-Instruct`)."""
    out_dir = RESULTS_DIR / "ned_dev_qwen3_vl_8b"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "__")
    shared_cache = ROOT / "results" / "llm_cache" / f"NED_dev_{safe_model}" / "responses.jsonl"

    anchor_briefs = setup["anchor_briefs"]
    cand_sets = setup["cand_sets"]
    mismatch_map = setup["mismatch_map"]
    slots = ["A", "B", "C", "D"]
    rng = np.random.default_rng(20261122)

    # Pre-build anonymized candidate briefs (shuffled slot order per post so slot letter never leaks truth!)
    vlm_tasks = []
    for gi in vlm_sample_idx:
        cands = list(cand_sets[gi])
        perm = rng.permutation(len(cands)).tolist()
        shuf_cands = [cands[p] for p in perm]
        cand_briefs = [
            (slots[k], ev_i, anchor_briefs[ev_i][0], anchor_briefs[ev_i][1])
            for k, ev_i in enumerate(shuf_cands)
        ]
        # Verify zero event_name strings slipped into anchor keywords/medoids
        vlm_tasks.append((gi, cand_briefs))

    if dry_run:
        print(f"\n[Tier 2 DRY-RUN] N={len(vlm_tasks)} posts x 3 conditions = {len(vlm_tasks)*3} VLM calls ({model_name})")
        from trackB_evid_msed.vlm_client import build_vlm_candidate_prompt

        gi0, cb0 = vlm_tasks[0]
        sm, um = build_vlm_candidate_prompt(ds.texts[gi0], [(s, kw, md) for s, _, kw, md in cb0], "orig_img")
        print("--- Sample Anonymized VLM Prompt (Cluster A..D, zero event_name/year metadata) ---")
        print(um[:550] + "\n...")
        est_tok = len(vlm_tasks) * (420 + 620 + 620)
        print(f"Estimated total real tokens across 360 calls: ~{est_tok:,} tokens")
        return {"dry_run": True, "n_posts": len(vlm_tasks), "est_tokens": est_tok}

    prov_url, _, api_key = load_provider_key(provider)
    ledger = TokenLedger(out_dir / "ledger_vlm", shared_cache_file=shared_cache)
    client = BudgetedVLMClient(
        model=model_name,
        ledger=ledger,
        base_url=prov_url,
        api_key=api_key,
    )

    records = []
    t0 = time.time()
    for step, (gi, cand_briefs) in enumerate(vlm_tasks, 1):
        pid = ds.post_ids[gi]
        true_ev = int(ds.labels[gi])
        orig_path = ds.image_paths[gi]
        mis_gi = mismatch_map[gi]
        mis_path = ds.image_paths[mis_gi]

        d_txt = client.adjudicate_paired(f"{pid}_text", "text_only", ds.texts[gi], cand_briefs, None)
        d_orig = client.adjudicate_paired(f"{pid}_orig", "orig_img", ds.texts[gi], cand_briefs, orig_path)
        d_mis = client.adjudicate_paired(f"{pid}_mis", "mismatched_img", ds.texts[gi], cand_briefs, mis_path)

        records.append({
            "idx": int(gi),
            "post_id": pid,
            "true_event_id": ds.label_to_event[true_ev],
            "true_ev_int": true_ev,
            "text": ds.texts[gi],
            "orig_img": str(orig_path.name) if orig_path else "",
            "mis_img": str(mis_path.name) if mis_path else "",
            "mis_event_id": ds.event_ids[mis_gi],
            "comp_margin_score": float(evid_scores["comp_margin"][gi]),
            "visual_idf_score": float(evid_scores["visual_idf"][gi]),
            "cntd_raw_score": float(evid_scores["cntd_raw"][gi]),
            "text_only": d_txt.__dict__,
            "orig_img_res": d_orig.__dict__,
            "mismatched_img_res": d_mis.__dict__,
        })
        if step % 15 == 0 or step == len(vlm_tasks):
            ls = ledger.summary()
            print(
                f"  [VLM {step:3d}/{len(vlm_tasks)}] tok={ls['total_tokens']:,} "
                f"(cached={ls['cached_calls']}/{ls['total_calls']}) elapsed={time.time()-t0:.1f}s"
            )

    # Analyze Tier 2 results + Competitor-Conditioned VLM Gating
    y_t = np.array([r["true_ev_int"] for r in records], dtype=np.int64)
    p_txt = np.array([r["text_only"]["chosen_event_idx"] if r["text_only"]["chosen_event_idx"] is not None else -1 for r in records])
    p_ori = np.array([r["orig_img_res"]["chosen_event_idx"] if r["orig_img_res"]["chosen_event_idx"] is not None else -1 for r in records])
    p_mis = np.array([r["mismatched_img_res"]["chosen_event_idx"] if r["mismatched_img_res"]["chosen_event_idx"] is not None else -1 for r in records])

    # Evaluate post-hoc evidence-gated VLM variants (choosing between d_orig and d_txt!)
    def _gate_preds(cond_tag: str) -> Tuple[np.ndarray, np.ndarray]:
        out_o = np.empty(len(records), dtype=np.int64)
        out_m = np.empty(len(records), dtype=np.int64)
        for k, r in enumerate(records):
            t_choice = p_txt[k]
            t_conf = float(r["text_only"]["confidence"])
            o_choice = p_ori[k]
            o_disc = bool(r["orig_img_res"]["visual_discriminative"])
            o_conf_txt = bool(r["orig_img_res"]["visual_conflict_with_text"])
            m_choice = p_mis[k]
            m_disc = bool(r["mismatched_img_res"]["visual_discriminative"])
            m_conf_txt = bool(r["mismatched_img_res"]["visual_conflict_with_text"])

            if cond_tag == "visual_idf_gate":
                accept_o = r["visual_idf_score"] >= 0.70
                accept_m = True
            elif cond_tag == "cntd_raw_gate":
                accept_o = r["cntd_raw_score"] >= 0.65
                accept_m = True
            elif cond_tag == "vlm_self_gate":
                # Accept multimodal shift only if image is discriminative & does NOT conflict with text
                accept_o = o_disc and (not o_conf_txt)
                accept_m = m_disc and (not m_conf_txt)
            elif cond_tag == "comp_evid_gate":
                # Competitor-conditioned: require no visual_conflict_with_text AND either discriminative visual clue
                # with non-tiny embedding evidence OR low text confidence with discriminative visual clue
                accept_o = (not o_conf_txt) and o_disc and ((r["comp_margin_score"] >= 0.25) or (t_conf < 0.75))
                accept_m = (not m_conf_txt) and m_disc and (t_conf < 0.75)
            else:
                accept_o = accept_m = True

            out_o[k] = o_choice if accept_o else t_choice
            out_m[k] = m_choice if accept_m else t_choice
        return out_o, out_m

    ev_labels = list(range(len(ds.label_to_event)))
    ok_txt = p_txt == y_t

    def _summ_vlm(po_arr: np.ndarray, pm_arr: np.ndarray) -> Dict[str, Any]:
        ok_o = po_arr == y_t
        ok_m = pm_arr == y_t
        n_res = int(np.sum((~ok_txt) & ok_o))
        n_mis = int(np.sum(ok_txt & (~ok_o)))
        n_mflip = int(np.sum(ok_txt & (~ok_m)))
        return {
            "text_acc": round(float(np.mean(ok_txt)), 4),
            "orig_acc": round(float(np.mean(ok_o)), 4),
            "orig_macro_f1": round(_macro_f1(y_t, po_arr, ev_labels), 4),
            "mismatch_acc": round(float(np.mean(ok_m)), 4),
            "n_rescue": n_res,
            "n_mislead": n_mis,
            "net_rescue_minus_mislead": n_res - n_mis,
            "n_mismatch_flip": n_mflip,
        }

    vlm_comparison = {
        "vlm_text_only": _summ_vlm(p_txt, p_txt),
        "vlm_naive_orig": _summ_vlm(p_ori, p_mis),
        "vlm_visual_idf_gate": _summ_vlm(*_gate_preds("visual_idf_gate")),
        "vlm_cntd_raw_gate": _summ_vlm(*_gate_preds("cntd_raw_gate")),
        "vlm_self_gate": _summ_vlm(*_gate_preds("vlm_self_gate")),
        "vlm_comp_evid_gate": _summ_vlm(*_gate_preds("comp_evid_gate")),
    }

    # Case studies: up to 3 visual rescues, up to 3 visual misleads, and 3 mismatch detections
    rescues = [r for k, r in enumerate(records) if (p_txt[k] != y_t[k]) and (p_ori[k] == y_t[k])][:4]
    misleads = [r for k, r in enumerate(records) if (p_txt[k] == y_t[k]) and (p_ori[k] != y_t[k])][:4]
    mismatch_flips = [r for k, r in enumerate(records) if (p_txt[k] == y_t[k]) and (p_mis[k] != y_t[k])][:4]

    st_cnt = Counter()
    for line in (out_dir / "ledger_vlm" / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            st_cnt[json.loads(line)["status"]] += 1

    tier2_out = {
        "model": model_name,
        "n_posts": len(records),
        "ledger_summary": ledger.summary(),
        "statuses": dict(st_cnt),
        "vlm_comparison": vlm_comparison,
        "case_studies": {
            "rescues": rescues,
            "misleads": misleads,
            "mismatch_flips": mismatch_flips,
        },
        "records": records,
    }
    (out_dir / "tier2_vlm_audit.json").write_text(
        json.dumps(tier2_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return tier2_out


def _plot_gateB_summary(tier1: Dict[str, Any], tier2: Optional[Dict[str, Any]], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=220)

    # Panel 1: Tier 1A Full-Dev (lambda=0.75) Rescues vs Misleads & Net Gain
    ax = axes[0]
    t1 = tier1["tier1a_paired_classification"]["lambda_0.75"]
    keys1 = ["naive_visual", "visual_idf", "cntd_raw", "cntd_cond", "comp_margin_guarded"]
    lbls1 = ["Naive\nVisual", "Visual\nIDF", "CNTD\n(Raw)", "CNTD\n(Cond)", "Comp-\nMargin"]
    res1 = [t1[k]["novel_non_dup"]["n_rescue"] for k in keys1]
    mis1 = [-t1[k]["novel_non_dup"]["n_mislead"] for k in keys1]
    net1 = [t1[k]["novel_non_dup"]["net_rescue_minus_mislead"] for k in keys1]
    xs = np.arange(len(keys1))
    ax.bar(xs - 0.18, res1, width=0.35, color="#16A34A", alpha=0.88, label="Rescues (+)")
    ax.bar(xs + 0.18, mis1, width=0.35, color="#DC2626", alpha=0.88, label="Misleads (-)")
    ax.plot(xs, net1, color="#1D4ED8", marker="o", lw=2.0, label="Net Gain (Rescue - Mislead)")
    ax.axhline(0, color="#334155", lw=0.9)
    ax.set_ylim(-44, 120)
    for xi, r_v, m_v in zip(xs, res1, mis1):
        ax.text(xi - 0.18, r_v + 2.5, f"+{r_v}", ha="center", va="bottom", fontsize=7.5)
        ax.text(xi + 0.18, m_v - 2.5, f"{m_v}", ha="center", va="top", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(lbls1, fontsize=8.5)
    ax.set_title("Tier 1A: Novel-Image Causal Audit (n=1,527)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Paired Image Posts Count")
    ax.legend(fontsize=7.8, frameon=False, loc="upper right")
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: Tier 1B Concurrent 8-Event 2D-SE Clustering AMI & Macro-B3
    ax = axes[1]
    t1b = tier1["tier1b_unsupervised_mv_2dse"]
    keys2 = ["text_only_2dse", "naive_mv_2dse", "visual_idf_mv_2dse", "cntd_raw_mv_2dse", "cntd_cond_mv_2dse"]
    lbls2 = ["Text\nOnly", "Naive\nMV", "Vis-IDF\nMV", "CNTD\n(Raw) MV", "CNTD\n(Cond) MV"]
    amis = [t1b[k]["ami"] * 100 for k in keys2]
    macs = [t1b[k]["macro_b3"] * 100 for k in keys2]
    xs2 = np.arange(len(keys2))
    b1 = ax.bar(xs2 - 0.18, amis, width=0.35, color="#2563EB", alpha=0.88, label="AMI (x100)")
    b2 = ax.bar(xs2 + 0.18, macs, width=0.35, color="#F59E0B", alpha=0.88, label="Event Macro-B³ (x100)")
    lo = max(0, min(amis + macs) - 6)
    hi = min(100, max(amis + macs) + 9)
    ax.set_ylim(lo, hi)
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs2)
    ax.set_xticklabels(lbls2, fontsize=8.4)
    ax.set_title("Tier 1B: Concurrent 8-Event 2D-SE (2,665 posts)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Score (x100)")
    ax.legend(fontsize=8.0, frameon=False, loc="upper left")
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: Tier 2 Qwen3-VL-8B-Instruct Paired Comparison
    ax = axes[2]
    if tier2 and "vlm_comparison" in tier2:
        vc = tier2["vlm_comparison"]
        keys3 = ["vlm_text_only", "vlm_naive_orig", "vlm_visual_idf_gate", "vlm_cntd_raw_gate", "vlm_self_gate", "vlm_comp_evid_gate"]
        lbls3 = ["Text\nOnly", "Orig\nNaive", "VisIDF\nGate", "CNTD\nGate", "Self-\nGate", "CompEvid\nGate"]
        accs = [vc[k]["orig_acc"] * 100 for k in keys3]
        macf = [vc[k]["orig_macro_f1"] * 100 for k in keys3]
        misa = [vc[k]["mismatch_acc"] * 100 for k in keys3]
        xs3 = np.arange(len(keys3))
        w3 = 0.26
        r1 = ax.bar(xs3 - w3, accs, width=w3, color="#1D4ED8", label="Orig Acc")
        r2 = ax.bar(xs3, macf, width=w3, color="#16A34A", label="Orig Macro-F1")
        r3 = ax.bar(xs3 + w3, misa, width=w3, color="#DC2626", alpha=0.82, label="Mismatched Acc")
        ax.set_ylim(0, 114)
        for rect in list(r1) + list(r2) + list(r3):
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=6.4)
        ax.set_xticks(xs3)
        ax.set_xticklabels(lbls3, fontsize=8.0)
        ax.set_title("Tier 2: Qwen3-VL-8B-Instruct Paired Audit (n=120)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Accuracy / Macro-F1 (%)")
        ax.legend(fontsize=7.6, frameon=False, loc="upper center", ncol=3)
        ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Gate B: Visual Rescue vs. Visual Mislead Causal Diagnostic (NED Concurrent Dev Split)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved Gate B diagnostic figure: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B Paired Causal Visual Evidence Audit")
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--n-vlm", type=int, default=120)
    parser.add_argument("--tier1-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replot-only", action="store_true")
    args = parser.parse_args()

    if args.replot_only:
        t1 = json.loads((RESULTS_DIR / "tier1_audit.json").read_text(encoding="utf-8"))
        t2 = json.loads((RESULTS_DIR / "ned_dev_qwen3_vl_8b" / "tier2_vlm_audit.json").read_text(encoding="utf-8"))
        _plot_gateB_summary(t1, t2, ROOT / "results" / "figures" / "gateB_ned_dev_qwen3_vl_8b.png")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds, txt_minilm, clip_img, _clip_txt, decoded_ok, dhash64 = _load_cached_features()
    setup = _build_anchors_and_queries(ds, txt_minilm, clip_img, decoded_ok, dhash64)
    evid_scores = _compute_post_evidence_scores(txt_minilm, clip_img, decoded_ok, setup["proto_vis"], setup["cand_sets"])

    print("================ GATE B TIER 1: FULL CONCURRENT DEV AUDIT ================")
    tier1 = run_tier1_audit(ds, txt_minilm, clip_img, decoded_ok, dhash64, setup, evid_scores)
    (RESULTS_DIR / "tier1_audit.json").write_text(json.dumps(tier1, indent=2), encoding="utf-8")
    print(
        f"Dev posts={tier1['n_dev_posts']}, valid_img={tier1['n_valid_images']}, "
        f"query_img={tier1['n_query_images']}, novel_non_dup_img={tier1['n_novel_query_images']}"
    )
    print("\n--- Tier 1A (lambda_V = 0.75, Novel Non-Duplicate Query Images) ---")
    for sname, vals in tier1["tier1a_paired_classification"]["lambda_0.75"].items():
        nv = vals["novel_non_dup"]
        al = vals["all_query"]
        print(
            f"  {sname:20s} | Novel(n={nv['n_eval']}): txt={nv['text_acc']:.4f} orig={nv['orig_acc']:.4f} "
            f"F1={nv['orig_macro_f1']:.4f} mis={nv['mismatch_acc']:.4f} | "
            f"Resc={nv['n_rescue']:+3d} Misld={nv['n_mislead']:3d} Net={nv['net_rescue_minus_mislead']:+4d} MisFlip={nv['n_mismatch_flip']:3d} || "
            f"All(n={al['n_eval']}): orig={al['orig_acc']:.4f} Net={al['net_rescue_minus_mislead']:+4d}"
        )

    print("\n--- Tier 1B (Concurrent 8-Event Unsupervised MV-2DSE Clustering, Corrected §3.3 Formula) ---")
    for k, v in tier1["tier1b_unsupervised_mv_2dse"].items():
        print(f"  {k:26s}: {v}")

    if args.tier1_only:
        return

    vlm_sample_idx = _select_stratified_vlm_sample(ds, txt_minilm, setup, evid_scores, n_total=args.n_vlm)
    print(f"\n================ GATE B TIER 2: Qwen3-VL-8B-Instruct ({len(vlm_sample_idx)} posts x 3 conditions) ================")
    tier2 = run_tier2_vlm_audit(
        ds, txt_minilm, clip_img, setup, evid_scores, vlm_sample_idx,
        provider=args.provider, model_name=args.model, dry_run=args.dry_run,
    )
    if args.dry_run:
        return

    print("\n--- Tier 2 Paired Causal VLM Verdict Summary ---")
    for k, v in tier2["vlm_comparison"].items():
        print(
            f"  {k:22s}: txt={v['text_acc']:.4f} orig={v['orig_acc']:.4f} F1={v['orig_macro_f1']:.4f} "
            f"mis={v['mismatch_acc']:.4f} | Resc={v['n_rescue']:2d} Misld={v['n_mislead']:2d} "
            f"Net={v['net_rescue_minus_mislead']:+3d} MisFlip={v['n_mismatch_flip']:2d}"
        )

    _plot_gateB_summary(tier1, tier2, ROOT / "results" / "figures" / "gateB_ned_dev_qwen3_vl_8b.png")


if __name__ == "__main__":
    main()
