"""Scalable Stage-1 Evid-MV-2DSE engine for NED validation & holdout streams.

Implements:
  - Chunked visual evidence computation (naive, visual_idf, cntd_raw, cntd_cond)
    without O(N^2) dense matrix blowup.
  - Proposal B §3.3 unified soft-imputation without the broken delta_i * delta_j factor,
    plus an explicit No-Impute (delta_i * delta_j) ablation.
  - Deterministic competing-event image corruption (`Stress-30%`) using topic groups
    from `ned_splits.json`.
  - Regression self-check (`--verify-dev`) against Gate B Tier 1B exact numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import MessageGraph, build_message_graph
from core_shared.metrics_audit import full_partition_audit
from core_shared.se_solver import solve_2d_se
from trackB_evid_msed.ned_loader import (
    SPLITS_PATH,
    NEDDataset,
    dhash_hamming,
    load_ned_split,
)

CACHE_ROOT = ROOT / "cache" / "trackB"


def load_split_with_cache(
    split: str,
) -> Tuple[NEDDataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NED split and pre-extracted MiniLM / CLIP / dHash64 arrays."""
    ds = load_ned_split(split=split)
    cdir = CACHE_ROOT / f"ned_{split}"
    txt_minilm = np.load(cdir / "text_minilm.npy").astype(np.float32)
    clip_img = np.load(cdir / "clip_img.npy").astype(np.float32)
    clip_txt = np.load(cdir / "clip_txt.npy").astype(np.float32)
    meta = np.load(cdir / "image_meta.npz")
    decoded_ok = meta["decoded_ok"].astype(bool)
    dhash64 = meta["dhash64"].astype(np.uint64)
    if len(ds) != txt_minilm.shape[0] or len(ds) != clip_img.shape[0]:
        raise ValueError(f"Cache length mismatch for split={split}: ds={len(ds)}, cache={txt_minilm.shape[0]}")
    return ds, txt_minilm, clip_img, clip_txt, decoded_ok, dhash64


def to_social_dataset(ds: NEDDataset) -> SocialDataset:
    """Wrap NEDDataset as SocialDataset along its concurrent relative timeline."""
    return SocialDataset(
        name=f"NED_{ds.split}_concurrent",
        tweet_ids=np.asarray(ds.post_ids, dtype=object),
        texts=ds.texts,
        labels=ds.labels,
        tokens_lemma=ds.tokens,
        entities=[[] for _ in range(len(ds))],
        hashtags=ds.hashtags,
        user_ids=["" for _ in range(len(ds))],
        mentioned_users=ds.mentions,
        timestamps=ds.rel_timestamps,
        dates=["concurrent" for _ in range(len(ds))],
    )


def corrupt_images_competing(
    ds: NEDDataset,
    clip_img: np.ndarray,
    decoded_ok: np.ndarray,
    dhash64: np.ndarray,
    rate: float = 0.30,
    seed: int = 20261122,
) -> Tuple[np.ndarray, List[Optional[Path]], np.ndarray]:
    """Replace `rate` of valid images with non-duplicate images from competing events.

    Prefers a different event within the same topic group (`ned_splits.json` groups)
    and falls back to any different event in the split if the topic is singleton here.
    Requires `dHash Hamming > 6` whenever available.
    """
    groups_obj = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))["groups"]
    ev_to_topic: Dict[str, str] = {}
    for topic, evs in groups_obj.items():
        for ev in evs:
            ev_to_topic[ev] = topic

    valid_idx = np.where(decoded_ok)[0]
    n_corrupt = int(round(float(rate) * len(valid_idx)))
    rng = np.random.default_rng(seed)

    by_ev: Dict[str, List[int]] = defaultdict(list)
    by_topic: Dict[str, List[int]] = defaultdict(list)
    for gi in valid_idx.tolist():
        ev = ds.event_ids[gi]
        by_ev[ev].append(gi)
        tp = ev_to_topic.get(ev, "other")
        by_topic[tp].append(gi)

    chosen = rng.choice(valid_idx, size=min(n_corrupt, len(valid_idx)), replace=False)
    clip_out = clip_img.copy()
    paths_out: List[Optional[Path]] = list(ds.image_paths)
    corrupted_mask = np.zeros(len(ds), dtype=bool)

    for gi in chosen.tolist():
        ev = ds.event_ids[gi]
        tp = ev_to_topic.get(ev, "other")
        same_topic_other = [j for j in by_topic.get(tp, []) if ds.event_ids[j] != ev]
        pool = same_topic_other if same_topic_other else [j for j in valid_idx.tolist() if ds.event_ids[j] != ev]
        if not pool:
            continue
        qh = int(dhash64[gi])
        # Sample up to 32 candidates cleanly to find dHash > 6 without full scan
        donor = int(pool[int(rng.integers(0, len(pool)))])
        if dhash_hamming(qh, int(dhash64[donor])) <= 6:
            non_dup = [j for j in pool if dhash_hamming(qh, int(dhash64[j])) > 6]
            if non_dup:
                donor = int(non_dup[int(rng.integers(0, len(non_dup)))])
        clip_out[gi] = clip_img[donor]
        paths_out[gi] = ds.image_paths[donor]
        corrupted_mask[gi] = True

    return clip_out, paths_out, corrupted_mask


def compute_visual_evidence_chunked(
    txt_minilm: np.ndarray,
    clip_img: np.ndarray,
    decoded_ok: np.ndarray,
    block_size: int = 1024,
) -> Dict[str, np.ndarray]:
    """Compute unsupervised visual evidence scores (naive, visual_idf, cntd_raw, cntd_cond)."""
    n = len(decoded_ok)
    vis_idx = np.where(decoded_ok)[0]
    n_vis = len(vis_idx)

    b_idf = np.zeros(n, dtype=np.float32)
    b_cntd_raw = np.zeros(n, dtype=np.float32)
    b_cntd_cond = np.zeros(n, dtype=np.float32)

    if n_vis == 0:
        return {
            "naive": np.zeros(n, dtype=np.float32),
            "visual_idf": b_idf,
            "cntd_raw": b_cntd_raw,
            "cntd_cond": b_cntd_cond,
        }

    emb_v = np.ascontiguousarray(clip_img[vis_idx], dtype=np.float32)
    emb_t = np.ascontiguousarray(txt_minilm[vis_idx], dtype=np.float32)
    emb_v_t = emb_v.T

    for st in range(0, n_vis, block_size):
        en = min(n_vis, st + block_size)
        sims_blk = emb_v[st:en] @ emb_v_t
        for r in range(en - st):
            sims_blk[r, st + r] = -1.0
            sims_row = sims_blk[r]
            gi = int(vis_idx[st + r])

            near_dup_cnt = int(np.sum(sims_row >= 0.88))
            b_idf[gi] = float(np.log(2.0) / np.log(2.0 + near_dup_cnt))

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

                hard_conflict = float(np.mean(tri < 0.12)) if len(tri) > 0 else 0.0
                b_cntd_cond[gi] = float(max(0.10, 1.0 - 0.7 * hard_conflict)) * b_idf[gi]
            else:
                b_cntd_raw[gi] = 1.0
                b_cntd_cond[gi] = b_idf[gi]

    return {
        "naive": decoded_ok.astype(np.float32),
        "visual_idf": b_idf,
        "cntd_raw": b_cntd_raw,
        "cntd_cond": b_cntd_cond,
    }


def build_text_graph_and_2dse(
    ds: NEDDataset,
    txt_minilm: np.ndarray,
    knn_k: int = 12,
    knn_sim_threshold: float = 0.38,
    attr_alpha: float = 0.45,
) -> Tuple[SocialDataset, MessageGraph, np.ndarray, Dict[str, Any]]:
    """Construct concurrent text+attribute graph and solve Text-Only 2D-SE."""
    social_ds = to_social_dataset(ds)
    text_graph = build_message_graph(
        social_ds,
        txt_minilm,
        knn_k=knn_k,
        knn_sim_threshold=knn_sim_threshold,
        attr_alpha=attr_alpha,
        time_decay_per_day=0.0,
    )
    labels_txt_se, _ = solve_2d_se(text_graph.adj, refine_moves=True)
    aud_txt_se = full_partition_audit(ds.labels, labels_txt_se)
    summary = {
        "ami": round(aud_txt_se["ami"], 4),
        "nmi": round(aud_txt_se["nmi"], 4),
        "ari": round(aud_txt_se["ari"], 4),
        "macro_b3": round(aud_txt_se["bcubed_full"]["macro_b3_f1"], 4),
        "micro_b3": round(aud_txt_se["bcubed_full"]["micro_b3_f1"], 4),
        "n_clusters": int(aud_txt_se["n_pred_clusters"]),
    }
    return social_ds, text_graph, labels_txt_se, summary


def run_mv_2dse_variant(
    ds: NEDDataset,
    text_graph: MessageGraph,
    labels_txt_se: np.ndarray,
    clip_img: np.ndarray,
    decoded_ok: np.ndarray,
    b_raw: np.ndarray,
    lam_v: float = 0.65,
    tau_imp: float = 0.55,
    gamma_imp: float = 0.70,
    use_soft_imputation: bool = True,
    k_v: int = 10,
    vis_sim_thresh: float = 0.72,
    block_size: int = 1024,
) -> Tuple[np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run one Multi-View 2D-SE variant using Proposal B §3.3 unified soft-imputation."""
    n = len(ds)
    adj_t_csr = text_graph.adj.tocsr()
    z_eff = clip_img.copy()
    b_eff = np.where(decoded_ok, b_raw, 0.0).astype(np.float32)

    imputed_count = 0
    if use_soft_imputation:
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
                    b_eff[i] = float(gamma_imp * b_mean)
                    imputed_count += 1

    eff_idx = np.where(b_eff > 0.0)[0]
    n_eff = len(eff_idx)
    if n_eff < 2:
        aud = full_partition_audit(ds.labels, labels_txt_se, y_before=labels_txt_se)
        return labels_txt_se.copy(), adj_t_csr, z_eff, b_eff, {
            "ami": round(aud["ami"], 4),
            "macro_b3": round(aud["bcubed_full"]["macro_b3_f1"], 4),
            "micro_b3": round(aud["bcubed_full"]["micro_b3_f1"], 4),
            "n_clusters": int(aud["n_pred_clusters"]),
            "imputed_count": imputed_count,
        }

    z_sub = np.ascontiguousarray(z_eff[eff_idx], dtype=np.float32)
    z_sub_t = z_sub.T
    k_v_eff = min(int(k_v), n_eff - 1)

    rows_l: List[int] = []
    cols_l: List[int] = []
    vals_l: List[float] = []

    for st in range(0, n_eff, block_size):
        en = min(n_eff, st + block_size)
        sims_blk = z_sub[st:en] @ z_sub_t
        for r in range(en - st):
            sims_blk[r, st + r] = -1.0
        top_pos = np.argpartition(-sims_blk, kth=k_v_eff - 1, axis=1)[:, :k_v_eff]
        for r in range(en - st):
            gi = int(eff_idx[st + r])
            bi = float(b_eff[gi])
            row_s = sims_blk[r]
            for pj in top_pos[r].tolist():
                s = float(row_s[pj])
                if s >= vis_sim_thresh:
                    gj = int(eff_idx[pj])
                    w_v = float(bi * float(b_eff[gj]) * s)
                    if w_v > 0.0:
                        rows_l.append(gi)
                        cols_l.append(gj)
                        vals_l.append(w_v)
                        rows_l.append(gj)
                        cols_l.append(gi)
                        vals_l.append(w_v)

    w_vis_mat = sparse.csr_matrix((vals_l, (rows_l, cols_l)), shape=(n, n), dtype=np.float64)
    w_vis_mat = w_vis_mat.maximum(w_vis_mat.T)
    comb_adj = (text_graph.adj + float(lam_v) * w_vis_mat).tocsr()
    comb_adj.setdiag(0.0)
    comb_adj.eliminate_zeros()

    lab_mv, _ = solve_2d_se(comb_adj, refine_moves=True)
    aud_mv = full_partition_audit(ds.labels, lab_mv, y_before=labels_txt_se)
    et = aud_mv["error_transition"]
    audit_dict = {
        "ami": round(aud_mv["ami"], 4),
        "delta_ami": round(aud_mv["delta_ami"], 4),
        "nmi": round(aud_mv["nmi"], 4),
        "ari": round(aud_mv["ari"], 4),
        "macro_b3": round(aud_mv["bcubed_full"]["macro_b3_f1"], 4),
        "delta_macro_b3": round(aud_mv["delta_macro_b3_f1"], 4),
        "micro_b3": round(aud_mv["bcubed_full"]["micro_b3_f1"], 4),
        "delta_micro_b3": round(aud_mv["delta_micro_b3_f1"], 4),
        "n_clusters": int(aud_mv["n_pred_clusters"]),
        "net_fm_red": int(et["net_false_merge_reduction"]),
        "net_fs_red": int(et["net_false_split_reduction"]),
        "imputed_count": int(imputed_count),
    }
    return lab_mv, comb_adj, z_eff, b_eff, audit_dict


def run_all_stage1_variants(
    ds: NEDDataset,
    txt_minilm: np.ndarray,
    clip_img: np.ndarray,
    decoded_ok: np.ndarray,
    lam_v: float = 0.65,
    tau_imp: float = 0.55,
    gamma_imp: float = 0.70,
    k_v: int = 10,
    vis_sim_thresh: float = 0.72,
    block_size: int = 1024,
    precomputed_text: Optional[Tuple[SocialDataset, MessageGraph, np.ndarray, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run all 6 Stage-1 variants and return metrics + visual_idf_mv_2dse artifacts."""
    t0 = time.time()
    if precomputed_text is None:
        social_ds, text_graph, labels_txt_se, txt_aud = build_text_graph_and_2dse(ds, txt_minilm)
    else:
        social_ds, text_graph, labels_txt_se, txt_aud = precomputed_text

    evid_scores = compute_visual_evidence_chunked(txt_minilm, clip_img, decoded_ok, block_size=block_size)

    _, _, _, _, m_naive = run_mv_2dse_variant(
        ds, text_graph, labels_txt_se, clip_img, decoded_ok, evid_scores["naive"],
        lam_v=lam_v, tau_imp=tau_imp, gamma_imp=gamma_imp, use_soft_imputation=True,
        k_v=k_v, vis_sim_thresh=vis_sim_thresh, block_size=block_size,
    )
    _, _, _, _, m_cntd_raw = run_mv_2dse_variant(
        ds, text_graph, labels_txt_se, clip_img, decoded_ok, evid_scores["cntd_raw"],
        lam_v=lam_v, tau_imp=tau_imp, gamma_imp=gamma_imp, use_soft_imputation=True,
        k_v=k_v, vis_sim_thresh=vis_sim_thresh, block_size=block_size,
    )
    _, _, _, _, m_cntd_cond = run_mv_2dse_variant(
        ds, text_graph, labels_txt_se, clip_img, decoded_ok, evid_scores["cntd_cond"],
        lam_v=lam_v, tau_imp=tau_imp, gamma_imp=gamma_imp, use_soft_imputation=True,
        k_v=k_v, vis_sim_thresh=vis_sim_thresh, block_size=block_size,
    )
    _, _, _, _, m_vis_noimp = run_mv_2dse_variant(
        ds, text_graph, labels_txt_se, clip_img, decoded_ok, evid_scores["visual_idf"],
        lam_v=lam_v, tau_imp=tau_imp, gamma_imp=gamma_imp, use_soft_imputation=False,
        k_v=k_v, vis_sim_thresh=vis_sim_thresh, block_size=block_size,
    )
    lab_vidf, comb_adj, z_eff, b_eff, m_vis_idf = run_mv_2dse_variant(
        ds, text_graph, labels_txt_se, clip_img, decoded_ok, evid_scores["visual_idf"],
        lam_v=lam_v, tau_imp=tau_imp, gamma_imp=gamma_imp, use_soft_imputation=True,
        k_v=k_v, vis_sim_thresh=vis_sim_thresh, block_size=block_size,
    )

    comb_graph = MessageGraph(
        adj=comb_adj,
        sem_adj=text_graph.sem_adj,
        attr_adj=text_graph.attr_adj,
        embeddings=txt_minilm,
        non_entity_tfidf=text_graph.non_entity_tfidf,
    )

    metrics = {
        "text_only_2dse": txt_aud,
        "naive_mv_2dse": m_naive,
        "cntd_raw_mv_2dse": m_cntd_raw,
        "cntd_cond_mv_2dse": m_cntd_cond,
        "visual_idf_no_impute_mv_2dse": m_vis_noimp,
        "visual_idf_mv_2dse": m_vis_idf,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    return {
        "metrics": metrics,
        "social_ds": social_ds,
        "text_graph": text_graph,
        "labels_txt_se": labels_txt_se,
        "vis_idf_labels": lab_vidf,
        "vis_idf_comb_graph": comb_graph,
        "z_eff": z_eff,
        "b_eff": b_eff,
        "evid_scores": evid_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 Evid-MV-2DSE verification.")
    parser.add_argument("--verify-dev", action="store_true", help="Verify against Gate B Tier 1B exact dev numbers.")
    args = parser.parse_args()

    if args.verify_dev:
        ds, txt_minilm, clip_img, _, decoded_ok, _ = load_split_with_cache("development")
        res = run_all_stage1_variants(ds, txt_minilm, clip_img, decoded_ok)
        m = res["metrics"]
        print("=== Stage-1 Regression Check on NED development ===")
        for k in [
            "text_only_2dse",
            "naive_mv_2dse",
            "cntd_raw_mv_2dse",
            "cntd_cond_mv_2dse",
            "visual_idf_no_impute_mv_2dse",
            "visual_idf_mv_2dse",
        ]:
            print(f"  {k:30s}: {m[k]}")

        # Assert exact match to Gate B Tier 1B targets (AMI=0.4747, Macro-B3=0.5973, n_clusters=98)
        t_ok = abs(m["text_only_2dse"]["ami"] - 0.4544) <= 1e-4 and abs(m["text_only_2dse"]["macro_b3"] - 0.5234) <= 1e-4
        v_ok = (
            abs(m["visual_idf_mv_2dse"]["ami"] - 0.4747) <= 1e-4
            and abs(m["visual_idf_mv_2dse"]["macro_b3"] - 0.5973) <= 1e-4
            and m["visual_idf_mv_2dse"]["n_clusters"] == 98
        )
        print(f"Regression assertions: text_ok={t_ok}, vis_idf_ok={v_ok}")
        if not (t_ok and v_ok):
            sys.exit(1)


if __name__ == "__main__":
    main()
