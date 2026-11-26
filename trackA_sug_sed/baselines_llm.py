"""Published Same-Budget LLM Clustering / SED Baselines + Internal Controls:

All LLM baselines share the exact same zero-lookahead `CausalUniformBucket`, cached
`BudgetedLLMClient`, frozen MiniLM embeddings, and strict `all_shown` unqueried masking:

  5. ClusterLLM          (Zhang et al., EMNLP 2023): hierarchical tree + entropy pairwise granularity calibration
  6. Few-Shot LLM-Post   (Viswanathan et al., TACL 2024): low-margin boundary post-correction (ASSIGN/NEW_EVENT)
  7. Cequel              (Wang et al., arXiv 2025): EdgeLLM uncertain cross-boundary pairs (Direct & Transitive)
  8. LSED-Budgeted       (Qiu et al., ACL 2025): budgeted LLM short-text disambiguation/rewriting + re-embedding + 2D-SE
  9. RagSEDE-Budgeted    (Liu et al., WWW 2026): MMR diverse selection + Top-K RAG neighbor context + SE update
 10. LHG-Budgeted        (Qiu et al., WWW 2026): HIG hub pseudo-labeling + MPLSS meta-path triplet selector + Poincare space

Internal Zero-Token & Random Routing Controls:
  - AutoMerge-Only (0-Tok, strict proto_sim >= 0.85 + Guarded UF)
  - Meso-Heur-Best (0-Tok, cosine + seam threshold on identical mesoscopic pool)
  - Random (Single-post uniform selection under equal token budget)
  - Random-Meso (Star-Meso candidate pool queried in random order under equal token budget)
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_COMPUTE_LOCK = threading.Lock()

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import MessageGraph, _ensure_l2_normalized, build_message_graph
from core_shared.se_solver import (
    minimize_2d_se_agglomerative,
    refine_2d_se_node_moves,
    solve_2d_se,
)
from trackA_sug_sed.baselines_sed import _poincare_dist, _poincare_exp0, _proj_poincare
from trackA_sug_sed.embed_cache import encode_texts_hf
from trackA_sug_sed.llm_client import (
    BudgetedLLMClient,
    build_lhg_pseudolabel_prompt,
    build_lsed_rewrite_prompt,
    build_pairwise_same_event_prompt,
    build_post_assign_prompt,
    select_display_medoids,
)
from trackA_sug_sed.star_meso import (
    ClusterSummary,
    GuardedUnionFind,
    _parse_bool,
    build_cluster_summaries,
    detect_clean_seams,
)


# ---------------------------------------------------------------------------
# Shared Helpers (Candidate Retrieval & Margins)
# ---------------------------------------------------------------------------

def _multi_route_candidates(
    node_idx: int,
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    summaries: Dict[int, ClusterSummary],
    proto_matrix: np.ndarray,
    proto_cids: np.ndarray,
    max_candidates: int = 4,
    min_cand_size: int = 2,
) -> List[int]:
    cands: List[int] = []
    seen: Set[int] = set()

    def _add(cid: int) -> None:
        if cid not in seen and cid in summaries and summaries[cid].size >= min_cand_size:
            seen.add(cid)
            cands.append(cid)

    row = graph.adj.getrow(node_idx)
    if row.nnz > 0:
        comm_w: Dict[int, float] = defaultdict(float)
        for nb, w in zip(row.indices.tolist(), row.data.tolist()):
            comm_w[int(labels[nb])] += w
        for cid, _ in sorted(comm_w.items(), key=lambda kv: -kv[1])[:2]:
            _add(cid)

    if len(proto_cids) > 0:
        sims = proto_matrix @ graph.embeddings[node_idx]
        for pos in np.argsort(-sims)[: max_candidates + 2]:
            _add(int(proto_cids[pos]))
            if len(cands) >= max_candidates - 1:
                break

    node_terms = set(ds.entities[node_idx]) | {f"#{h}" for h in ds.hashtags[node_idx]} | set(ds.tokens_lemma[node_idx])
    if node_terms and len(cands) < max_candidates:
        best_cid = None
        best_ov = 0
        for cid, s in summaries.items():
            if cid in seen or s.size < min_cand_size:
                continue
            ov = len(node_terms.intersection(s.top_keywords))
            if ov > best_ov:
                best_ov = ov
                best_cid = cid
        if best_cid is not None:
            _add(best_cid)

    if len(proto_cids) > 0 and len(cands) < max_candidates:
        sims = proto_matrix @ graph.embeddings[node_idx]
        for pos in np.argsort(-sims):
            _add(int(proto_cids[pos]))
            if len(cands) >= max_candidates:
                break

    return cands[:max_candidates]


def _proto_index(summaries: Dict[int, ClusterSummary], emb_dim: int, min_size: int = 2):
    valid = sorted([cid for cid, s in summaries.items() if s.size >= min_size])
    if not valid:
        return np.array([], dtype=np.int64), np.zeros((0, emb_dim), dtype=np.float32)
    return np.array(valid, dtype=np.int64), np.vstack([summaries[c].prototype for c in valid])


def _loo_cluster_sim(
    node_idx: int,
    cid: int,
    emb: np.ndarray,
    summs: Dict[int, ClusterSummary],
    comp_sums: Dict[int, np.ndarray],
) -> Optional[float]:
    """Leave-one-out cosine similarity between post `node_idx` and cluster `cid`."""
    s = summs.get(cid)
    if s is None or s.size < 2 or cid not in comp_sums:
        return None
    diff = comp_sums[cid] - emb[node_idx].astype(np.float64)
    nm = float(np.linalg.norm(diff))
    if nm <= 1e-12:
        return None
    return float(np.dot(emb[node_idx].astype(np.float64), diff / nm))


# ---------------------------------------------------------------------------
# Internal Control Baselines (Zero-LLM & Random)
# ---------------------------------------------------------------------------

def run_automerge_only(
    ds: SocialDataset,
    embeddings: np.ndarray,
    labels_init: np.ndarray,
    tau_auto: float = 0.85,
    guard_sim: float = 0.40,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Zero-token strict AutoMerge (proto_sim >= tau_auto) + CentroidGuardedUF."""
    t0 = time.time()
    summs = build_cluster_summaries(ds, embeddings, labels_init)
    cids = sorted([c for c, s in summs.items() if s.size >= 2])
    if len(cids) < 2:
        return np.asarray(labels_init, dtype=np.int64).copy(), {"runtime_sec": 0.0, "auto_merge_ops": 0}
    protos = np.vstack([summs[c].prototype for c in cids])
    sizes = [summs[c].size for c in cids]
    comp_vecs = [embeddings[summs[c].members].sum(axis=0, dtype=np.float64) for c in cids]
    sim_mat = protos @ protos.T
    uf = GuardedUnionFind(
        cids, sim_mat, guard_sim=guard_sim, enforce_guards=True, protos=protos, sizes=sizes, comp_vecs=comp_vecs
    )
    pairs = []
    for i in range(len(cids)):
        for j in np.where(sim_mat[i] >= tau_auto)[0]:
            if j > i:
                pairs.append((float(sim_mat[i, j]), cids[i], cids[int(j)]))
    pairs.sort(key=lambda x: -x[0])
    ops = 0
    for _s, c1, c2 in pairs:
        r1, r2 = uf.root(c1), uf.root(c2)
        if r1 == r2:
            continue
        if float(np.dot(uf._comp_proto(r1), uf._comp_proto(r2))) < tau_auto:
            uf.blocked_count += 1
            continue
        if uf.merge(r1, r2):
            ops += 1
    labels = uf.apply_to_labels(labels_init)
    return labels, {"runtime_sec": round(time.time() - t0, 3), "auto_merge_ops": ops}


def run_meso_heur_best(
    ds: SocialDataset,
    embeddings: np.ndarray,
    graph: MessageGraph,
    labels_init: np.ndarray,
    tau_merge: float = 0.80,
    max_seam_cross_sim: float = 0.55,
    min_seam_sep: float = 0.28,
    guard_sim: float = 0.40,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Zero-token heuristic operating on the identical Star-Meso candidate pool."""
    t0 = time.time()
    labels = np.asarray(labels_init, dtype=np.int64).copy()
    summs0 = build_cluster_summaries(ds, embeddings, labels)
    seams = detect_clean_seams(
        graph,
        summs0,
        min_cluster_size=8,
        min_pocket_size=3,
        max_cross_sim=max_seam_cross_sim,
        min_sem_sep=min_seam_sep,
        use_seam_gate=True,
    )
    next_cid = int(labels.max()) + 1 if len(labels) > 0 else 0
    seam_ops = 0
    for sc in seams:
        labels[sc.sub2] = next_cid
        next_cid += 1
        seam_ops += 1

    summs = build_cluster_summaries(ds, embeddings, labels)
    cids = sorted([c for c, s in summs.items() if s.size >= 2])
    merge_ops = 0
    if len(cids) >= 2:
        protos = np.vstack([summs[c].prototype for c in cids])
        sizes = [summs[c].size for c in cids]
        comp_vecs = [embeddings[summs[c].members].sum(axis=0, dtype=np.float64) for c in cids]
        sim_mat = protos @ protos.T
        uf = GuardedUnionFind(
            cids, sim_mat, guard_sim=guard_sim, enforce_guards=True, protos=protos, sizes=sizes, comp_vecs=comp_vecs
        )
        pairs = []
        for i in range(len(cids)):
            for j in np.where(sim_mat[i] >= tau_merge)[0]:
                if j > i:
                    pairs.append((float(sim_mat[i, j]), cids[i], cids[int(j)]))
        pairs.sort(key=lambda x: -x[0])
        for _s, c1, c2 in pairs:
            r1, r2 = uf.root(c1), uf.root(c2)
            if r1 == r2:
                continue
            if float(np.dot(uf._comp_proto(r1), uf._comp_proto(r2))) < tau_merge:
                uf.blocked_count += 1
                continue
            if uf.merge(r1, r2):
                merge_ops += 1
        labels = uf.apply_to_labels(labels)
    return labels, {"runtime_sec": round(time.time() - t0, 3), "seam_ops": seam_ops, "merge_ops": merge_ops}


def run_random_llm(
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    graph: MessageGraph,
    labels_init: np.ndarray,
    client: BudgetedLLMClient,
    bucket: Any,
    seed: int = 42,
    strategy_tag: str = "random",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Baseline control: uniform random post selection + post-assignment."""
    t0 = time.perf_counter()
    t_thread0 = time.thread_time()
    emb = _ensure_l2_normalized(embeddings)
    n = len(ds)
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels_init, dtype=np.int64).copy()
    summs = build_cluster_summaries(ds, emb, labels)
    proto_cids, proto_mat = _proto_index(summs, emb.shape[1], min_size=2)

    all_shown = np.zeros(n, dtype=bool)
    next_cid = int(labels.max()) + 1 if n > 0 else 0
    order = rng.permutation(n).tolist()

    calls = pt_sum = ct_sum = errs = 0
    llm_sec = api_wall_sec = 0.0
    for idx in order:
        cands = _multi_route_candidates(idx, ds, graph, labels, summs, proto_mat, proto_cids, max_candidates=4)
        cur_c = int(labels[idx])
        if cur_c not in cands and cur_c in summs and summs[cur_c].size >= 2:
            cands = ([cur_c] + cands)[:4]
        if not cands:
            continue
        sys_msg, user_msg = build_post_assign_prompt(idx, cands, summs, ds, max_medoids=2)
        t_api0 = time.perf_counter()
        res = client.call_json(
            request_id=f"rand_M{block_id:02d}_n{idx}",
            block_id=block_id,
            strategy=strategy_tag,
            query_type="node",
            sys_msg=sys_msg,
            user_msg=user_msg,
            bucket=bucket,
            est_completion_tokens=75,
        )
        api_wall_sec += time.perf_counter() - t_api0
        if res is None:
            break
        parsed, pt, ct, lat, _ = res
        pt_sum += pt
        ct_sum += ct
        llm_sec += lat
        calls += 1
        all_shown[idx] = True
        for cid in cands:
            if cid in summs:
                for m in select_display_medoids(summs[cid], exclude_idx=idx, max_medoids=2):
                    all_shown[int(m)] = True
        if "error" in parsed:
            errs += 1
            continue
        act = str(parsed.get("action", "")).upper()
        tgt = parsed.get("target_cluster", None)
        if act == "ASSIGN" and tgt is not None:
            try:
                ti = int(tgt)
                if ti in cands:
                    labels[idx] = ti
            except (ValueError, TypeError):
                pass
        elif act == "NEW_EVENT":
            if cur_c in summs and summs[cur_c].size >= 2:
                labels[idx] = next_cid
                next_cid += 1

    _, clean = np.unique(labels, return_inverse=True)
    total_wall = time.perf_counter() - t0
    router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
    return clean.astype(np.int64), all_shown, {
        "total_calls": calls,
        "total_tokens": pt_sum + ct_sum,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": errs,
        "router_cpu_sec": round(router_cpu, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }


# ---------------------------------------------------------------------------
# 5. ClusterLLM (Zhang et al., EMNLP 2023)
# ---------------------------------------------------------------------------

def run_clusterllm(
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    labels_init: np.ndarray,
    client: BudgetedLLMClient,
    bucket: Any,
    strategy_tag: str = "clusterllm",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """ClusterLLM (EMNLP 2023): hierarchical clustering + entropy pairwise granularity selection."""
    t0 = time.perf_counter()
    t_thread0 = time.thread_time()
    emb = _ensure_l2_normalized(embeddings)
    n = len(ds)
    if n <= 3:
        return np.asarray(labels_init, dtype=np.int64).copy(), np.zeros(n, dtype=bool), {
            "total_calls": 0,
            "total_tokens": 0,
            "router_cpu_sec": 0.0,
            "api_wall_sec": 0.0,
            "llm_api_sec": 0.0,
            "total_wall_sec": 0.0,
        }

    with _COMPUTE_LOCK:
        dists = pdist(emb.astype(np.float64), metric="cosine")
        dists = np.clip(np.nan_to_num(dists, nan=1.0), 0.0, 2.0)
        Z = linkage(dists, method="average")

        cut_grid = [0.45, 0.50, 0.55, 0.60, 0.65]
        cuts = {d: fcluster(Z, t=d, criterion="distance") for d in cut_grid}
        fine_cut = fcluster(Z, t=0.40, criterion="distance")

    # Build representative medoids on fine_cut (< min(cut_grid)) so candidate pairs
    # span all transition thresholds across cut_grid rather than collapsing below 0.55.
    fine_summs = build_cluster_summaries(ds, emb, fine_cut)
    cids, p_mat = _proto_index(fine_summs, emb.shape[1], min_size=1)
    medoid_by_cid = {cid: fine_summs[cid].medoids[0] for cid in cids.tolist()}
    size_by_cid = {cid: fine_summs[cid].size for cid in cids.tolist()}

    sim_cc = p_mat @ p_mat.T if len(cids) >= 2 else np.zeros((0, 0), dtype=np.float32)
    band_buckets: Dict[float, List[Tuple[float, int, int]]] = {d: [] for d in cut_grid}
    fallback_pool: List[Tuple[float, int, int]] = []

    if len(cids) >= 2:
        rows, cols = np.where(np.triu(sim_cc, 1) >= 0.35)
        for r_idx, c_idx in zip(rows.tolist(), cols.tolist()):
            s = float(sim_cc[r_idx, c_idx])
            if s > 0.95:
                continue
            ci, cj = int(cids[r_idx]), int(cids[c_idx])
            u, v = medoid_by_cid[ci], medoid_by_cid[cj]
            sz_min = min(size_by_cid[ci], size_by_cid[cj])

            first_d: Optional[float] = None
            same_count = 0
            for d in cut_grid:
                if cuts[d][u] == cuts[d][v]:
                    same_count += 1
                    if first_d is None:
                        first_d = d

            same_frac = same_count / float(len(cut_grid))
            ent = -(
                same_frac * math.log2(max(same_frac, 1e-9))
                + (1.0 - same_frac) * math.log2(max(1.0 - same_frac, 1e-9))
            )
            if first_d is not None:
                score = (0.5 + ent) * math.log1p(sz_min) * max(0.05, 1.0 - abs(s - (1.0 - first_d)))
                band_buckets[first_d].append((score, u, v))
            elif s >= 0.45 and s <= 0.90:
                score = 0.5 * math.log1p(sz_min) * max(0.05, 1.0 - abs(s - 0.72))
                fallback_pool.append((score, u, v))

    for d in cut_grid:
        band_buckets[d].sort(key=lambda x: -x[0])
    fallback_pool.sort(key=lambda x: -x[0])

    # Round-robin interleave across transition bands in entropy-first order, then fallback
    band_order = [0.55, 0.60, 0.50, 0.65, 0.45]
    cand_pairs: List[Tuple[float, int, int]] = []
    pos_by_band = {d: 0 for d in band_order}
    while True:
        progressed = False
        for d in band_order:
            pos = pos_by_band[d]
            if pos < len(band_buckets[d]):
                cand_pairs.append(band_buckets[d][pos])
                pos_by_band[d] = pos + 1
                progressed = True
        if not progressed:
            break
    cand_pairs.extend(fallback_pool)

    all_shown = np.zeros(n, dtype=bool)
    verdicts: List[Tuple[int, int, bool]] = []
    calls = pt_sum = ct_sum = errs = 0
    llm_sec = api_wall_sec = 0.0

    for _sc, u, v in cand_pairs:
        sys_msg, user_msg = build_pairwise_same_event_prompt(ds.texts[u], ds.texts[v], ds.dates[u], ds.dates[v])
        t_api0 = time.perf_counter()
        res = client.call_json(
            request_id=f"cllm_M{block_id:02d}_u{u}_v{v}",
            block_id=block_id,
            strategy=strategy_tag,
            query_type="clusterllm_pair",
            sys_msg=sys_msg,
            user_msg=user_msg,
            bucket=bucket,
            est_completion_tokens=75,
        )
        api_wall_sec += time.perf_counter() - t_api0
        if res is None:
            break
        parsed, pt, ct, lat, _ = res
        pt_sum += pt
        ct_sum += ct
        llm_sec += lat
        calls += 1
        all_shown[u] = True
        all_shown[v] = True
        if "error" in parsed:
            errs += 1
            continue
        verdicts.append((u, v, _parse_bool(parsed.get("same_event", False))))

    # Granularity determination: maximize balanced accuracy (invariant to pair class skew),
    # then F1, accuracy, and proximity to 0.55
    best_d = 0.55
    best_key = (-1.0, -1.0, -1.0, -1.0)
    n_pos = n_neg = 0
    if verdicts:
        n_pos = sum(1 for _u, _v, same in verdicts if same)
        n_neg = len(verdicts) - n_pos
        for d in cut_grid:
            tp = tn = fp = fn = 0
            for u, v, same in verdicts:
                pred_same = bool(cuts[d][u] == cuts[d][v])
                if pred_same and same:
                    tp += 1
                elif not pred_same and not same:
                    tn += 1
                elif pred_same and not same:
                    fp += 1
                else:
                    fn += 1
            acc = (tp + tn) / float(len(verdicts))
            if n_pos > 0 and n_neg > 0:
                bal_acc = 0.5 * ((tp / float(n_pos)) + (tn / float(n_neg)))
            else:
                bal_acc = acc
            f1 = (2.0 * tp / float(2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            key = (round(bal_acc, 6), round(f1, 6), round(acc, 6), -round(abs(d - 0.55), 6))
            if key > best_key:
                best_key = key
                best_d = d

    labels = cuts[best_d].astype(np.int64).copy()
    # Apply verified MUST_LINK merges guarded by CANNOT_LINK constraints and centroid cohesion
    must_link_merges = 0
    cannot_link_splits = 0
    guarded_blocks = 0
    base_summs = build_cluster_summaries(ds, emb, labels)
    base_cids = sorted(base_summs.keys())
    if len(base_cids) >= 2 and verdicts:
        protos = np.vstack([base_summs[c].prototype for c in base_cids])
        sizes = [base_summs[c].size for c in base_cids]
        comp_vecs = [emb[base_summs[c].members].sum(axis=0, dtype=np.float64) for c in base_cids]
        sim_mat = protos @ protos.T
        uf = GuardedUnionFind(
            base_cids,
            sim_mat,
            guard_sim=0.40,
            enforce_guards=True,
            protos=protos,
            sizes=sizes,
            comp_vecs=comp_vecs,
            min_cl_votes=1,
        )
        for u, v, same in verdicts:
            if not same and int(labels[u]) != int(labels[v]):
                uf.add_cannot_link(int(labels[u]), int(labels[v]))
        for u, v, same in verdicts:
            if same and uf.merge(int(labels[u]), int(labels[v])):
                must_link_merges += 1
        labels = uf.apply_to_labels(labels)
        guarded_blocks = uf.blocked_count

    next_cid = int(labels.max()) + 1
    for u, v, same in verdicts:
        if not same and labels[u] == labels[v]:
            cid_uv = int(labels[u])
            if fine_cut[u] != fine_cut[v]:
                fu_nodes = np.where((labels == cid_uv) & (fine_cut == fine_cut[u]))[0]
                fv_nodes = np.where((labels == cid_uv) & (fine_cut == fine_cut[v]))[0]
                peel = fu_nodes if len(fu_nodes) < len(fv_nodes) else fv_nodes
                labels[peel] = next_cid
            else:
                labels[v] = next_cid
            next_cid += 1
            cannot_link_splits += 1

    _, clean = np.unique(labels, return_inverse=True)
    total_wall = time.perf_counter() - t0
    router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
    return clean.astype(np.int64), all_shown, {
        "total_calls": calls,
        "selected_cut_dist": best_d,
        "verdicts_pos": n_pos,
        "verdicts_neg": n_neg,
        "must_link_merges": must_link_merges,
        "cannot_link_splits": cannot_link_splits,
        "guarded_blocks": guarded_blocks,
        "total_tokens": pt_sum + ct_sum,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": errs,
        "router_cpu_sec": round(router_cpu, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }


# ---------------------------------------------------------------------------
# 6. Few-Shot LLM-Post (Viswanathan et al., TACL 2024)
# ---------------------------------------------------------------------------

def run_fewshot_llm_post(
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    graph: MessageGraph,
    labels_init: np.ndarray,
    client: BudgetedLLMClient,
    bucket: Any,
    strategy_tag: str = "fewshot_llm_post",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Viswanathan et al. (TACL 2024): Post-clustering low-margin LLM assignment correction."""
    t0 = time.perf_counter()
    t_thread0 = time.thread_time()
    emb = _ensure_l2_normalized(embeddings)
    n = len(ds)
    labels = np.asarray(labels_init, dtype=np.int64).copy()
    summs = build_cluster_summaries(ds, emb, labels)
    comp_sums = {c: emb[s.members].sum(axis=0, dtype=np.float64) for c, s in summs.items()}
    proto_cids, proto_mat = _proto_index(summs, emb.shape[1], min_size=2)
    cid_to_row = {int(c): i for i, c in enumerate(proto_cids.tolist())}

    # Rank boundary & high-affinity unattached singletons by leave-one-out confidence margin
    scored_nodes: List[Tuple[float, int]] = []
    if len(proto_cids) >= 2:
        sims_all = emb @ proto_mat.T
        for i in range(n):
            c_cur = int(labels[i])
            if summs[c_cur].size < 2:
                top2 = np.sort(sims_all[i])[-2:]
                top1_sim, top2_sim = float(top2[-1]), float(top2[0])
                if top1_sim >= 0.45:
                    margin = max(-0.05, (top1_sim - top2_sim) - 0.15)
                else:
                    margin = 1.0 + (1.0 - top1_sim)
                scored_nodes.append((margin, i))
                continue
            r_cur = cid_to_row.get(c_cur)
            if r_cur is None:
                continue
            row = sims_all[i].copy()
            s_cur = _loo_cluster_sim(i, c_cur, emb, summs, comp_sums)
            if s_cur is None:
                s_cur = float(row[r_cur])
            row[r_cur] = -2.0
            s_alt = float(np.max(row))
            margin = s_cur - s_alt
            scored_nodes.append((margin, i))
    scored_nodes.sort(key=lambda x: x[0])

    all_shown = np.zeros(n, dtype=bool)
    next_cid = int(labels.max()) + 1 if n > 0 else 0
    calls = pt_sum = ct_sum = errs = reassigned = new_events = 0
    llm_sec = api_wall_sec = 0.0

    for _margin, idx in scored_nodes:
        cands = _multi_route_candidates(
            idx, ds, graph, labels, summs, proto_mat, proto_cids, max_candidates=4, min_cand_size=2
        )
        cur_c = int(labels[idx])
        if cur_c not in cands and cur_c in summs and summs[cur_c].size >= 2:
            cands = ([cur_c] + cands)[:4]
        if not cands:
            continue
        sys_msg, user_msg = build_post_assign_prompt(idx, cands, summs, ds, max_medoids=2)
        t_api0 = time.perf_counter()
        res = client.call_json(
            request_id=f"fslp_M{block_id:02d}_n{idx}",
            block_id=block_id,
            strategy=strategy_tag,
            query_type="post_correct",
            sys_msg=sys_msg,
            user_msg=user_msg,
            bucket=bucket,
            est_completion_tokens=75,
        )
        api_wall_sec += time.perf_counter() - t_api0
        if res is None:
            break
        parsed, pt, ct, lat, _ = res
        pt_sum += pt
        ct_sum += ct
        llm_sec += lat
        calls += 1
        all_shown[idx] = True
        for cid in cands:
            if cid in summs:
                for m in select_display_medoids(summs[cid], exclude_idx=idx, max_medoids=2):
                    all_shown[int(m)] = True
        if "error" in parsed:
            errs += 1
            continue
        act = str(parsed.get("action", "")).upper()
        tgt = parsed.get("target_cluster", None)
        if act == "ASSIGN" and tgt is not None:
            try:
                ti = int(tgt)
                if ti in cands and ti != cur_c:
                    labels[idx] = ti
                    reassigned += 1
            except (ValueError, TypeError):
                pass
        elif act == "NEW_EVENT":
            if cur_c in summs and summs[cur_c].size >= 2:
                labels[idx] = next_cid
                next_cid += 1
                new_events += 1

    _, clean = np.unique(labels, return_inverse=True)
    total_wall = time.perf_counter() - t0
    router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
    return clean.astype(np.int64), all_shown, {
        "total_calls": calls,
        "reassigned": reassigned,
        "new_events": new_events,
        "total_tokens": pt_sum + ct_sum,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": errs,
        "router_cpu_sec": round(router_cpu, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }
# ---------------------------------------------------------------------------

def run_cequel(
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    graph: MessageGraph,
    labels_init: np.ndarray,
    client: BudgetedLLMClient,
    bucket: Any,
    strategy_tag: str = "cequel",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Cequel EdgeLLM: greedy uncertain boundary edge queries -> Direct & Transitive partitions."""
    t0 = time.perf_counter()
    t_thread0 = time.thread_time()
    n = len(ds)
    labels = np.asarray(labels_init, dtype=np.int64).copy()
    summs = build_cluster_summaries(ds, embeddings, labels)

    # Collect uncertain cross-cluster pairs (high cosine & graph-adjacent or medoid-close)
    coo = graph.adj.tocoo()
    cand_edges: List[Tuple[float, int, int]] = []
    seen_pair: Set[Tuple[int, int]] = set()
    for u, v, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if u >= v or int(labels[u]) == int(labels[v]):
            continue
        cos_uv = float(np.dot(embeddings[u], embeddings[v]))
        if cos_uv < 0.48:
            continue
        seen_pair.add((u, v))
        ambig = 1.0 - min(1.0, abs(cos_uv - 0.72) / 0.28)
        sz_u = summs[int(labels[u])].size
        sz_v = summs[int(labels[v])].size
        score = ambig * math.log1p(min(sz_u, sz_v)) * (1.0 + 0.15 * w)
        cand_edges.append((score, u, v))

    # Also include cross-cluster medoid pairs in ambiguous band [0.55, 0.90)
    cids = sorted([c for c, s in summs.items() if s.size >= 2])
    if len(cids) >= 2:
        protos = np.vstack([summs[c].prototype for c in cids])
        sim_cc = protos @ protos.T
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                s = float(sim_cc[i, j])
                if 0.55 <= s < 0.90:
                    u = summs[cids[i]].medoids[0]
                    v = summs[cids[j]].medoids[0]
                    ek = (u, v) if u < v else (v, u)
                    if ek not in seen_pair:
                        seen_pair.add(ek)
                        ambig = 1.0 - min(1.0, abs(s - 0.72) / 0.25)
                        score = ambig * math.log1p(min(summs[cids[i]].size, summs[cids[j]].size))
                        cand_edges.append((score, ek[0], ek[1]))

    cand_edges.sort(key=lambda x: -x[0])

    # Diversify across cluster pairs so budget covers broad edges
    all_shown = np.zeros(n, dtype=bool)
    verdicts: List[Tuple[int, int, bool]] = []
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    calls = pt_sum = ct_sum = errs = 0
    llm_sec = api_wall_sec = 0.0

    for _sc, u, v in cand_edges:
        cu, cv = int(labels[u]), int(labels[v])
        ck = (cu, cv) if cu < cv else (cv, cu)
        if pair_counts[ck] >= 2:
            continue
        sys_msg, user_msg = build_pairwise_same_event_prompt(ds.texts[u], ds.texts[v], ds.dates[u], ds.dates[v])
        t_api0 = time.perf_counter()
        res = client.call_json(
            request_id=f"ceq_M{block_id:02d}_u{u}_v{v}",
            block_id=block_id,
            strategy=strategy_tag,
            query_type="pair",
            sys_msg=sys_msg,
            user_msg=user_msg,
            bucket=bucket,
            est_completion_tokens=75,
        )
        api_wall_sec += time.perf_counter() - t_api0
        if res is None:
            break
        pair_counts[ck] += 1
        parsed, pt, ct, lat, _ = res
        pt_sum += pt
        ct_sum += ct
        llm_sec += lat
        calls += 1
        all_shown[u] = True
        all_shown[v] = True
        if "error" in parsed:
            errs += 1
            continue
        verdicts.append((u, v, _parse_bool(parsed.get("same_event", False))))

    # 1) Direct Edge edits
    lbl_edge = labels.copy()
    next_e = int(lbl_edge.max()) + 1 if n > 0 else 0
    for u, v, same in verdicts:
        if same and lbl_edge[u] != lbl_edge[v]:
            lbl_edge[v] = lbl_edge[u]
        elif not same and lbl_edge[u] == lbl_edge[v]:
            lbl_edge[v] = next_e
            next_e += 1
    _, clean_edge = np.unique(lbl_edge, return_inverse=True)

    # 2) Cluster-level Transitive Closure guarded by CANNOT_LINK
    lbl_trans = labels.copy()
    parent: Dict[int, int] = {int(c): int(c) for c in np.unique(lbl_trans).tolist()}

    def _find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cannot_pairs = [(u, v) for u, v, s in verdicts if not s]
    must_pairs = [(u, v) for u, v, s in verdicts if s]
    fallback = []
    for u, v in must_pairs:
        r1, r2 = _find(int(labels[u])), _find(int(labels[v]))
        if r1 == r2:
            continue
        blocked = any(
            (_find(int(labels[cu])) == r1 and _find(int(labels[cv])) == r2)
            or (_find(int(labels[cu])) == r2 and _find(int(labels[cv])) == r1)
            for cu, cv in cannot_pairs
        )
        if blocked:
            fallback.append((u, v))
        else:
            parent[r2] = r1
    for i in range(n):
        lbl_trans[i] = _find(int(lbl_trans[i]))
    for u, v in fallback:
        lbl_trans[v] = lbl_trans[u]
    next_t = int(lbl_trans.max()) + 1 if n > 0 else 0
    for u, v in cannot_pairs:
        if lbl_trans[u] == lbl_trans[v]:
            lbl_trans[v] = next_t
            next_t += 1
    _, clean_trans = np.unique(lbl_trans, return_inverse=True)

    total_wall = time.perf_counter() - t0
    router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
    stats = {
        "total_calls": calls,
        "total_tokens": pt_sum + ct_sum,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": errs,
        "router_cpu_sec": round(router_cpu, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }
    return clean_edge.astype(np.int64), clean_trans.astype(np.int64), all_shown, stats


# ---------------------------------------------------------------------------
# 8. LSED-Budgeted (Qiu et al., ACL 2025)
# ---------------------------------------------------------------------------

def run_lsed_budgeted(
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    labels_init: np.ndarray,
    embed_model: str,
    client: BudgetedLLMClient,
    bucket: Any,
    knn_sim_threshold: float = 0.50,
    strategy_tag: str = "lsed_budgeted",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """LSED (ACL 2025) under equal causal budget: rewrite noisy uncertain posts + re-embed + 2D-SE."""
    t0 = time.perf_counter()
    t_thread0 = time.thread_time()
    emb = _ensure_l2_normalized(embeddings)
    n = len(ds)
    summs = build_cluster_summaries(ds, emb, labels_init)
    comp_sums = {c: emb[s.members].sum(axis=0, dtype=np.float64) for c, s in summs.items()}
    proto_cids, proto_mat = _proto_index(summs, emb.shape[1], min_size=2)
    cid_to_row = {int(c): i for i, c in enumerate(proto_cids.tolist())}
    sims_all = emb @ proto_mat.T if len(proto_cids) >= 2 else np.zeros((n, 0))

    # Rank uncertain/short/isolated posts needing disambiguation using leave-one-out margin
    scored: List[Tuple[float, int]] = []
    for i in range(n):
        c_cur = int(labels_init[i])
        margin = 0.0
        if c_cur in summs and summs[c_cur].size >= 2 and sims_all.shape[1] >= 2 and c_cur in cid_to_row:
            r = cid_to_row[c_cur]
            row = sims_all[i].copy()
            s_cur = _loo_cluster_sim(i, c_cur, emb, summs, comp_sums)
            if s_cur is None:
                s_cur = float(row[r])
            row[r] = -2.0
            margin = s_cur - float(np.max(row))
        len_pen = min(1.0, len(ds.texts[i]) / 80.0)
        ent_pen = min(1.0, len(ds.entities[i]) * 0.5)
        uncertainty = (1.0 - margin) + 0.3 * (1.0 - len_pen) + 0.2 * (1.0 - ent_pen)
        scored.append((uncertainty, i))
    scored.sort(key=lambda x: -x[0])

    all_shown = np.zeros(n, dtype=bool)
    rewritten_idx: List[int] = []
    rewritten_texts: List[str] = []
    calls = pt_sum = ct_sum = errs = 0
    llm_sec = api_wall_sec = 0.0

    for _sc, idx in scored:
        sys_msg, user_msg = build_lsed_rewrite_prompt(ds.texts[idx], ds.dates[idx])
        t_api0 = time.perf_counter()
        res = client.call_json(
            request_id=f"lsed_M{block_id:02d}_n{idx}",
            block_id=block_id,
            strategy=strategy_tag,
            query_type="lsed_rewrite",
            sys_msg=sys_msg,
            user_msg=user_msg,
            bucket=bucket,
            est_completion_tokens=65,
        )
        api_wall_sec += time.perf_counter() - t_api0
        if res is None:
            break
        parsed, pt, ct, lat, _ = res
        pt_sum += pt
        ct_sum += ct
        llm_sec += lat
        calls += 1
        all_shown[idx] = True
        if "error" in parsed or not isinstance(parsed.get("rewritten_event"), str):
            errs += 1
            continue
        rewritten_idx.append(idx)
        rewritten_texts.append(parsed["rewritten_event"].strip())

    emb_new = emb.copy()
    if rewritten_idx:
        with _COMPUTE_LOCK:
            new_vecs = encode_texts_hf(rewritten_texts, model_name=embed_model, batch_size=32)
            new_vecs = _ensure_l2_normalized(new_vecs)
            for k, idx in enumerate(rewritten_idx):
                # LSED blends normalized disambiguated event statement vector cleanly
                vec = 0.7 * new_vecs[k] + 0.3 * emb_new[idx]
                nm = float(np.linalg.norm(vec))
                emb_new[idx] = vec / max(nm, 1e-12)
            g_new = build_message_graph(
                ds,
                emb_new,
                knn_k=15,
                knn_sim_threshold=knn_sim_threshold,
                attr_alpha=0.5,
                compute_non_entity_tfidf=False,
            )
            labels_out, _ = solve_2d_se(g_new.adj, refine_moves=True)
    else:
        labels_out = np.asarray(labels_init, dtype=np.int64).copy()

    _, clean = np.unique(labels_out, return_inverse=True)
    total_wall = time.perf_counter() - t0
    router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
    return clean.astype(np.int64), all_shown, {
        "total_calls": calls,
        "rewritten_ok": len(rewritten_idx),
        "total_tokens": pt_sum + ct_sum,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": errs,
        "router_cpu_sec": round(router_cpu, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }


# ---------------------------------------------------------------------------
# 9. RagSEDE-Budgeted (Liu et al., WWW 2026)
# ---------------------------------------------------------------------------

def run_ragsede_budgeted(
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    graph: MessageGraph,
    labels_init: np.ndarray,
    client: BudgetedLLMClient,
    bucket: Any,
    strategy_tag: str = "ragsede_budgeted",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """RagSEDE (WWW 2026) under equal causal budget: MMR selection + RAG retrieval + SE repair."""
    t0 = time.perf_counter()
    t_thread0 = time.thread_time()
    emb = _ensure_l2_normalized(embeddings)
    n = len(ds)
    labels = np.asarray(labels_init, dtype=np.int64).copy()
    summs = build_cluster_summaries(ds, emb, labels)
    comp_sums = {c: emb[s.members].sum(axis=0, dtype=np.float64) for c, s in summs.items()}
    proto_cids, proto_mat = _proto_index(summs, emb.shape[1], min_size=2)
    cid_to_row = {int(c): i for i, c in enumerate(proto_cids.tolist())}

    # Base MMR representativeness/uncertainty scores (leave-one-out for multi-post; top1 for singletons)
    deg = np.asarray(graph.adj.sum(axis=1)).ravel()
    norm_deg = deg / max(float(deg.max()), 1.0)
    sims_all = emb @ proto_mat.T if len(proto_cids) >= 2 else np.zeros((n, 0))

    rep_score = np.zeros(n, dtype=np.float64)
    for i in range(n):
        amb = 0.5
        c_cur = int(labels[i])
        if sims_all.shape[1] >= 2:
            if c_cur in summs and summs[c_cur].size >= 2 and c_cur in cid_to_row:
                r = cid_to_row[c_cur]
                row = sims_all[i].copy()
                s_cur = _loo_cluster_sim(i, c_cur, emb, summs, comp_sums)
                if s_cur is None:
                    s_cur = float(row[r])
                row[r] = -2.0
                amb = 1.0 - max(0.0, s_cur - float(np.max(row)))
            else:
                top1_sim = float(np.max(sims_all[i]))
                amb = top1_sim if top1_sim >= 0.45 else 0.2 * max(0.0, top1_sim)
        rep_score[i] = 0.6 * amb + 0.4 * float(norm_deg[i])

    # MMR iterative selection + RAG Top-2 retrieval
    pool_top = np.argsort(-rep_score)[: min(n, 300)].tolist()
    selected_vecs: List[np.ndarray] = []
    remaining = set(pool_top)
    all_shown = np.zeros(n, dtype=bool)
    adj_lil = graph.adj.tocsr().tolil(copy=True)
    warm_labels = labels.copy()
    next_cid = int(warm_labels.max()) + 1 if n > 0 else 0
    movable = np.zeros(n, dtype=bool)
    calls = pt_sum = ct_sum = errs = reassigned = new_events = 0
    llm_sec = api_wall_sec = 0.0

    while remaining:
        if not selected_vecs:
            idx = max(remaining, key=lambda x: rep_score[x])
        else:
            sel_mat = np.vstack(selected_vecs)
            best_val, idx = -1e9, -1
            for cand in remaining:
                div = float(np.max(sel_mat @ emb[cand]))
                val = 0.5 * rep_score[cand] - 0.5 * div
                if val > best_val:
                    best_val = val
                    idx = cand
        remaining.remove(idx)
        selected_vecs.append(emb[idx])

        cands = _multi_route_candidates(
            idx, ds, graph, labels, summs, proto_mat, proto_cids, max_candidates=4, min_cand_size=2
        )
        cur_c = int(labels[idx])
        if cur_c not in cands and cur_c in summs and summs[cur_c].size >= 2:
            cands = ([cur_c] + cands)[:4]
        if not cands:
            continue

        # Retrieve top-2 dense semantic RAG neighbors (excluding idx)
        cos_n = emb @ emb[idx]
        cos_n[idx] = -2.0
        rag_idx = np.argsort(-cos_n)[:2].tolist()
        rag_txts = [ds.texts[r] for r in rag_idx]

        sys_msg, user_msg = build_post_assign_prompt(
            idx, cands, summs, ds, rag_context=rag_txts, max_medoids=2, max_rag=2
        )
        t_api0 = time.perf_counter()
        res = client.call_json(
            request_id=f"ragsede_M{block_id:02d}_n{idx}",
            block_id=block_id,
            strategy=strategy_tag,
            query_type="ragsede_node",
            sys_msg=sys_msg,
            user_msg=user_msg,
            bucket=bucket,
            est_completion_tokens=80,
        )
        api_wall_sec += time.perf_counter() - t_api0
        if res is None:
            break
        parsed, pt, ct, lat, _ = res
        pt_sum += pt
        ct_sum += ct
        llm_sec += lat
        calls += 1
        all_shown[idx] = True
        for r in rag_idx[:2]:
            all_shown[int(r)] = True
        for cid in cands:
            if cid in summs:
                for m in select_display_medoids(summs[cid], exclude_idx=idx, max_medoids=2):
                    all_shown[int(m)] = True
        if "error" in parsed:
            errs += 1
            continue

        act = str(parsed.get("action", "")).upper()
        tgt = parsed.get("target_cluster", None)
        if act == "ASSIGN" and tgt is not None:
            try:
                ti = int(tgt)
                if ti in cands and ti != cur_c and ti in summs:
                    orig_nbs = list(adj_lil.rows[idx])
                    for nb in orig_nbs:
                        if int(labels[nb]) == cur_c:
                            adj_lil[idx, nb] = 0.0
                            adj_lil[nb, idx] = 0.0
                    ti_meds = select_display_medoids(summs[ti], exclude_idx=idx, max_medoids=2)
                    for med in ti_meds:
                        adj_lil[idx, med] = max(float(adj_lil[idx, med]), 1.5)
                        adj_lil[med, idx] = max(float(adj_lil[med, idx]), 1.5)
                    warm_labels[idx] = ti
                    movable[idx] = True
                    reassigned += 1

                    # Propagate repair to up to 5 tightly-coupled neighbors aligned with ti
                    if ti_meds and cur_c in summs and summs[cur_c].size >= 2:
                        primary_med = ti_meds[0]
                        prop_cands: List[Tuple[float, int, float]] = []
                        for nb in orig_nbs:
                            nb_i = int(nb)
                            if nb_i == idx or int(labels[nb_i]) != cur_c:
                                continue
                            s_in = float(np.dot(emb[idx], emb[nb_i]))
                            s_ti = float(np.dot(emb[nb_i], summs[ti].prototype))
                            if s_in >= 0.72 and s_ti >= 0.55:
                                prop_cands.append((s_in * s_ti, nb_i, s_ti))
                        prop_cands.sort(key=lambda x: -x[0])
                        for _sc, nb_i, s_ti in prop_cands[:5]:
                            movable[nb_i] = True
                            w_br = 0.5 * s_ti
                            adj_lil[nb_i, primary_med] = max(float(adj_lil[nb_i, primary_med]), w_br)
                            adj_lil[primary_med, nb_i] = max(float(adj_lil[primary_med, nb_i]), w_br)
            except (ValueError, TypeError):
                pass
        elif act == "NEW_EVENT":
            if cur_c in summs and summs[cur_c].size >= 2:
                for nb in list(adj_lil.rows[idx]):
                    if int(labels[nb]) == cur_c:
                        adj_lil[idx, nb] = 0.0
                        adj_lil[nb, idx] = 0.0
                warm_labels[idx] = next_cid
                next_cid += 1
                movable[idx] = True
                new_events += 1

    rep_adj = adj_lil.tocsr()
    rep_adj.setdiag(0.0)
    rep_adj.eliminate_zeros()
    if np.any(movable):
        with _COMPUTE_LOCK:
            out_labels, _ = minimize_2d_se_agglomerative(rep_adj, initial_labels=warm_labels)
            out_labels, _ = refine_2d_se_node_moves(rep_adj, out_labels, max_iters=6, movable_mask=movable)
    else:
        out_labels = labels

    _, clean = np.unique(out_labels, return_inverse=True)
    total_wall = time.perf_counter() - t0
    router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
    return clean.astype(np.int64), all_shown, {
        "total_calls": calls,
        "reassigned": reassigned,
        "new_events": new_events,
        "movable_nodes": int(movable.sum()),
        "total_tokens": pt_sum + ct_sum,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": errs,
        "router_cpu_sec": round(router_cpu, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }


# ---------------------------------------------------------------------------
# 10. LHG-Budgeted (Qiu et al., WWW 2026) — MPLSS + Hyperbolic HIG
# ---------------------------------------------------------------------------

def run_lhg_budgeted(
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    graph: MessageGraph,
    labels_init: np.ndarray,
    embed_model: str,
    client: BudgetedLLMClient,
    bucket: Any,
    knn_sim_threshold: float = 0.50,
    seed: int = 42,
    strategy_tag: str = "lhg_budgeted",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """LHG (WWW 2026) under equal causal budget: HIG hub pseudo-labels + MPLSS + Poincare triplet tuning."""
    import torch

    t0 = time.perf_counter()
    t_thread0 = time.thread_time()
    n = len(ds)
    attr_csr = graph.attr_adj.tocsr()
    attr_deg = np.asarray(attr_csr.sum(axis=1)).ravel()
    comb_deg = np.asarray(graph.adj.sum(axis=1)).ravel()

    # Select diverse HIG hubs prioritizing cross-community heterogeneous hubs
    score_hub = 0.7 * (attr_deg / max(float(attr_deg.max()), 1.0)) + 0.3 * (comb_deg / max(float(comb_deg.max()), 1.0))
    hub_order = np.argsort(-score_hub).tolist()

    all_shown = np.zeros(n, dtype=bool)
    pl_nodes: List[int] = []
    pl_texts: List[str] = []
    calls = pt_sum = ct_sum = errs = 0
    llm_sec = api_wall_sec = 0.0

    adj_csr = graph.adj.tocsr()
    for u in hub_order:
        r_start, r_end = adj_csr.indptr[u], adj_csr.indptr[u + 1]
        nbs = adj_csr.indices[r_start:r_end]
        w_nbs = adj_csr.data[r_start:r_end]
        top_nbs = nbs[np.argsort(-w_nbs)][:3].tolist() if len(nbs) > 0 else []
        nb_snips = [ds.texts[nb] for nb in top_nbs]

        sys_msg, user_msg = build_lhg_pseudolabel_prompt(ds.texts[u], nb_snips, ds.dates[u])
        t_api0 = time.perf_counter()
        res = client.call_json(
            request_id=f"lhg_M{block_id:02d}_n{u}",
            block_id=block_id,
            strategy=strategy_tag,
            query_type="lhg_pseudo",
            sys_msg=sys_msg,
            user_msg=user_msg,
            bucket=bucket,
            est_completion_tokens=60,
        )
        api_wall_sec += time.perf_counter() - t_api0
        if res is None:
            break
        parsed, pt, ct, lat, _ = res
        pt_sum += pt
        ct_sum += ct
        llm_sec += lat
        calls += 1
        all_shown[u] = True
        for nb in top_nbs:
            all_shown[int(nb)] = True
        if "error" in parsed or not isinstance(parsed.get("event_label"), str):
            errs += 1
            continue
        pl_nodes.append(int(u))
        pl_texts.append(parsed["event_label"].strip())

    # MPLSS: Meta-Path Guided Label Similarity Selector -> Hyperbolic Triplet Head
    if len(pl_nodes) >= 6:
        with _COMPUTE_LOCK:
            pl_emb = encode_texts_hf(pl_texts, model_name=embed_model, batch_size=32)
            pl_sim = pl_emb @ pl_emb.T
            m_nodes = len(pl_nodes)
            sub_attr = attr_csr[pl_nodes][:, pl_nodes].toarray()

            anchors, positives, negatives = [], [], []
            for i in range(m_nodes):
                pos_cands = [
                    j
                    for j in range(m_nodes)
                    if j != i and ((sub_attr[i, j] > 0.0 and pl_sim[i, j] >= 0.60) or pl_sim[i, j] >= 0.78)
                ]
                neg_cands = [
                    j for j in range(m_nodes) if j != i and pl_sim[i, j] < 0.45
                ]
                if pos_cands and neg_cands:
                    p_best = max(pos_cands, key=lambda j: float(pl_sim[i, j]))
                    n_hard = max(neg_cands, key=lambda j: float(embeddings[pl_nodes[i]] @ embeddings[pl_nodes[j]]))
                    anchors.append(pl_nodes[i])
                    positives.append(pl_nodes[p_best])
                    negatives.append(pl_nodes[n_hard])

            if len(anchors) >= 3:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                torch.manual_seed(seed)
                x_all = torch.tensor(embeddings, dtype=torch.float32, device=device)
                proj = torch.nn.Linear(embeddings.shape[1], 32, bias=False).to(device)
                torch.nn.init.xavier_uniform_(proj.weight, gain=0.4)
                opt = torch.optim.Adam(proj.parameters(), lr=0.03, weight_decay=1e-4)

                a_t = torch.tensor(anchors, dtype=torch.long, device=device)
                p_t = torch.tensor(positives, dtype=torch.long, device=device)
                n_t = torch.tensor(negatives, dtype=torch.long, device=device)

                coo_e = graph.adj.tocoo()
                upper = coo_e.row < coo_e.col
                eu, ev = coo_e.row[upper], coo_e.col[upper]
                rng = np.random.default_rng(seed)

                for _ep in range(45):
                    opt.zero_grad()
                    z = _proj_poincare(_poincare_exp0(proj(x_all), c=1.0), c=1.0)
                    d_ap = _poincare_dist(z[a_t], z[p_t], c=1.0)
                    d_an = _poincare_dist(z[a_t], z[n_t], c=1.0)
                    loss_trip = torch.mean(torch.relu(d_ap - d_an + 0.6))
                    if len(eu) > 0:
                        samp = rng.integers(0, len(eu), size=min(len(eu), 1024))
                        su = torch.tensor(eu[samp], dtype=torch.long, device=device)
                        sv = torch.tensor(ev[samp], dtype=torch.long, device=device)
                        loss_g = torch.mean(_poincare_dist(z[su], z[sv], c=1.0) ** 2)
                        loss = 2.0 * loss_trip + 0.2 * loss_g
                    else:
                        loss = loss_trip
                    loss.backward()
                    opt.step()

                with torch.no_grad():
                    z_out = _proj_poincare(_poincare_exp0(proj(x_all), c=1.0), c=1.0)
                    if len(eu) > 0:
                        eu_t = torch.tensor(eu, dtype=torch.long, device=device)
                        ev_t = torch.tensor(ev, dtype=torch.long, device=device)
                        dh = _poincare_dist(z_out[eu_t], z_out[ev_t], c=1.0).cpu().numpy()
                        hw = coo_e.data[upper] * np.exp(-0.25 * dh)
                        r_all = np.concatenate([eu, ev])
                        c_all = np.concatenate([ev, eu])
                        w_all = np.concatenate([hw, hw])
                        adj_hyp = sparse.csr_matrix((w_all, (r_all, c_all)), shape=(n, n), dtype=np.float64)
                        adj_lil = adj_hyp.tolil(copy=True)
                        for ai, pi, ni in zip(anchors, positives, negatives):
                            adj_lil[ai, pi] = max(float(adj_lil[ai, pi]), 1.5)
                            adj_lil[pi, ai] = max(float(adj_lil[pi, ai]), 1.5)
                            if adj_lil[ai, ni] > 0:
                                adj_lil[ai, ni] = 0.0
                                adj_lil[ni, ai] = 0.0
                        adj_hyp = adj_lil.tocsr()
                        adj_hyp.setdiag(0.0)
                        adj_hyp.eliminate_zeros()
                        labels_out, _ = solve_2d_se(adj_hyp, refine_moves=True)
                        _, clean = np.unique(labels_out, return_inverse=True)
                        total_wall = time.perf_counter() - t0
                        router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
                        return clean.astype(np.int64), all_shown, {
                            "total_calls": calls,
                            "mplss_triplets": len(anchors),
                            "total_tokens": pt_sum + ct_sum,
                            "shown_posts": int(all_shown.sum()),
                            "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
                            "parse_errors": errs,
                            "router_cpu_sec": round(router_cpu, 4),
                            "api_wall_sec": round(api_wall_sec, 4),
                            "llm_api_sec": round(llm_sec, 4),
                            "total_wall_sec": round(total_wall, 4),
                        }

    # Fallback if < 3 verified triplets constructed under budget
    _, clean = np.unique(labels_init, return_inverse=True)
    total_wall = time.perf_counter() - t0
    router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
    return clean.astype(np.int64), all_shown, {
        "total_calls": calls,
        "mplss_triplets": 0,
        "total_tokens": pt_sum + ct_sum,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": errs,
        "router_cpu_sec": round(router_cpu, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }
