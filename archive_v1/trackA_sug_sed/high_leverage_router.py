"""High-leverage mesoscopic structural routing & budgeted baseline selectors for Gate A.

Implements:
  1. SUG-Router v2.1 (Symmetric Mesoscopic Topology Router):
     - Type-I (`merge_pair`): Inter-Cluster False-Split Seam detector! Ranks candidate
       cluster pairs (Ca, Cb) by prototype similarity, shared attribute cues, and
       log-leverage to catch fragmented sibling clusters (1 query fixes dozens/hundreds
       of unqueried posts!).
     - Type-II (`seam`): Intra-Cluster False-Merge Seam detector! Uses Fiedler /
       non-entity bipartition inside clusters with log-leverage + semantic bimodality
       (preventing giant pure clusters from drowning out truly impure clusters).
     - Type-III (`bridge` / `seed`): Boundary bridge nodes & small micro-cluster nuclei
       with multi-route candidate retrieval (graph + dense + lexical/entity).
  2. Fair same-budget comparison selectors:
     - Random node selector
     - Rep-Div selector (RagSEDE-style representativeness + diversity)
     - Cequel-Edge selector (Cequel-style uncertain pair must-link / cannot-link)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import MessageGraph


@dataclass
class ClusterSummary:
    cluster_id: int
    members: List[int]
    size: int
    prototype: np.ndarray            # L2-normalized mean embedding
    medoids: List[int]               # Up to 3 most central post indices
    top_keywords: List[str]          # Top entities / hashtags / lemmas
    mean_timestamp: float


@dataclass
class QueryItem:
    query_id: str
    strategy: str                    # 'sug_router' | 'random' | 'rep_div' | 'cequel_edge'
    query_type: str                  # 'merge_pair' | 'seam' | 'bridge' | 'seed' | 'node' | 'pair'
    score: float
    primary_nodes: List[int]         # Directly queried representative/medoid/pair nodes
    affected_group: List[int]        # Members structurally targeted by this query
    candidate_clusters: List[int] = field(default_factory=list)
    # Populated for query_type == 'seam' (intra-cluster split)
    seam_cluster_id: Optional[int] = None
    seam_sub1: List[int] = field(default_factory=list)
    seam_sub2: List[int] = field(default_factory=list)
    seam_sub1_medoids: List[int] = field(default_factory=list)
    seam_sub2_medoids: List[int] = field(default_factory=list)
    seam_cut_edges: List[Tuple[int, int]] = field(default_factory=list)
    # Populated for query_type == 'merge_pair' (inter-cluster merge/separate)
    merge_c1: Optional[int] = None
    merge_c2: Optional[int] = None
    merge_c1_medoids: List[int] = field(default_factory=list)
    merge_c2_medoids: List[int] = field(default_factory=list)
    meta: Dict[str, float] = field(default_factory=dict)


def build_cluster_summaries(
    ds: SocialDataset,
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_medoids: int = 3,
    max_keywords: int = 5,
) -> Dict[int, ClusterSummary]:
    """Summarize each cluster by prototype, central medoids, and top keywords."""
    comm_to_members: Dict[int, List[int]] = defaultdict(list)
    for i, c in enumerate(np.asarray(labels).tolist()):
        comm_to_members[int(c)].append(i)

    summaries: Dict[int, ClusterSummary] = {}
    for cid, members in comm_to_members.items():
        idx = np.asarray(members, dtype=np.int64)
        sub_emb = embeddings[idx]
        proto = sub_emb.mean(axis=0)
        norm = float(np.linalg.norm(proto))
        if norm > 1e-12:
            proto = proto / norm

        sims = sub_emb @ proto
        top_local = np.argsort(-sims)[:max_medoids]
        medoids = [int(idx[k]) for k in top_local]

        kw_counts: Dict[str, int] = defaultdict(int)
        for m in members:
            for e in ds.entities[m]:
                kw_counts[e] += 2
            for h in ds.hashtags[m]:
                kw_counts[f"#{h}"] += 2
            for tok in ds.tokens_lemma[m]:
                if len(tok) >= 3:
                    kw_counts[tok] += 1
        top_kw = [k for k, _ in sorted(kw_counts.items(), key=lambda kv: -kv[1])[:max_keywords]]

        summaries[cid] = ClusterSummary(
            cluster_id=cid,
            members=members,
            size=len(members),
            prototype=proto.astype(np.float32),
            medoids=medoids,
            top_keywords=top_kw,
            mean_timestamp=float(np.mean(ds.timestamps[idx])),
        )
    return summaries


def detect_merge_pair_queries(
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    summaries: Dict[int, ClusterSummary],
    min_cluster_size: int = 2,
    min_proto_sim: float = 0.45,
    max_pairs_per_cluster: int = 3,
    use_leverage: bool = True,
    use_kw_edge_bonus: bool = True,
    random_order: bool = False,
    seed: int = 42,
) -> List[QueryItem]:
    """Type-I: Inter-Cluster False-Split Seam detector (Cluster-to-Cluster Merge/Separate).

    Identifies pairs of distinct clusters (Ca, Cb) with high prototype similarity
    and/or existing weak inter-cluster edges. One LLM decision can unify an entire
    fragmented event (or permanently sever a deceptive cross-event similarity).
    """
    cids = sorted([c for c, s in summaries.items() if s.size >= min_cluster_size])
    if len(cids) < 2:
        return []

    # Precompute inter-cluster cut edge weights
    coo = graph.adj.tocoo()
    inter_w: Dict[Tuple[int, int], float] = defaultdict(float)
    for u, v, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if u >= v:
            continue
        cu, cv = int(labels[u]), int(labels[v])
        if cu != cv:
            key = (cu, cv) if cu < cv else (cv, cu)
            inter_w[key] += w

    # Pairwise prototype cosine matrix
    protos = np.vstack([summaries[c].prototype for c in cids])
    sim_mat = protos @ protos.T

    raw_pairs: List[Tuple[float, int, int, float, float, float]] = []
    for i in range(len(cids)):
        c1 = cids[i]
        s1 = summaries[c1]
        kw1 = set(s1.top_keywords)
        for j in range(i + 1, len(cids)):
            c2 = cids[j]
            sim = float(sim_mat[i, j])
            cut_w = inter_w.get((c1, c2), 0.0)
            if sim < min_proto_sim and cut_w < 3.0:
                continue
            s2 = summaries[c2]
            kw_ov = len(kw1.intersection(s2.top_keywords)) / max(1, min(len(kw1), len(s2.top_keywords)))
            # High similarity + gentle log-leverage so both large fragmented events and medium/tail fragments get covered
            leverage = float(np.log1p(min(s1.size, s2.size)) * np.log1p(s1.size + s2.size))
            edge_bonus = 0.15 if cut_w > 0.0 else 0.0
            base_score = max(0.0, sim) + (0.15 * kw_ov + edge_bonus if use_kw_edge_bonus else 0.0)
            score = base_score * (0.5 + 0.15 * leverage) if use_leverage else base_score
            raw_pairs.append((score, c1, c2, sim, cut_w, kw_ov))

    if random_order:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(raw_pairs))
        raw_pairs = [raw_pairs[int(idx)] for idx in perm]
    else:
        raw_pairs.sort(key=lambda x: -x[0])

    # Diversify so a single fragmented hub does not consume the entire merge budget
    queries: List[QueryItem] = []
    cluster_use: Dict[int, int] = defaultdict(int)
    for score, c1, c2, sim, cut_w, kw_ov in raw_pairs:
        if cluster_use[c1] >= max_pairs_per_cluster or cluster_use[c2] >= max_pairs_per_cluster:
            continue
        cluster_use[c1] += 1
        cluster_use[c2] += 1
        s1, s2 = summaries[c1], summaries[c2]
        queries.append(
            QueryItem(
                query_id=f"merge_c{c1}_c{c2}",
                strategy="sug_router",
                query_type="merge_pair",
                score=score,
                primary_nodes=s1.medoids[:1] + s2.medoids[:1],
                affected_group=s1.members + s2.members,
                candidate_clusters=[c1, c2],
                merge_c1=c1,
                merge_c2=c2,
                merge_c1_medoids=s1.medoids[:3],
                merge_c2_medoids=s2.medoids[:3],
                meta={
                    "proto_sim": sim,
                    "inter_cut_w": cut_w,
                    "kw_overlap": kw_ov,
                    "size1": float(s1.size),
                    "size2": float(s2.size),
                },
            )
        )
    return queries


def _bipartition_cluster(
    sub_adj: sparse.csr_matrix,
    sub_emb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Deterministic 2-way bipartition returning (local_idx_1, local_idx_2, cut_conductance)."""
    m = sub_adj.shape[0]
    if m < 2:
        return np.arange(m, dtype=np.int64), np.array([], dtype=np.int64), 1.0

    n_comp, comp_labels = sparse.csgraph.connected_components(sub_adj, directed=False)
    if n_comp >= 2:
        counts = np.bincount(comp_labels)
        main_c = int(np.argmax(counts))
        g1 = np.where(comp_labels == main_c)[0]
        g2 = np.where(comp_labels != main_c)[0]
        return g1, g2, 0.0

    deg = np.asarray(sub_adj.sum(axis=1)).ravel()
    total_vol = float(deg.sum())
    if float(deg.min()) > 1e-12 and m >= 4 and total_vol > 1e-12:
        inv_sqrt_d = 1.0 / np.sqrt(deg)
        d_mat = sparse.diags(inv_sqrt_d)
        norm_adj = d_mat @ sub_adj @ d_mat
        try:
            if m <= 80:
                eigvals, eigvecs = np.linalg.eigh(norm_adj.toarray())
                fiedler = eigvecs[:, -2] * inv_sqrt_d
            else:
                eigvals, eigvecs = eigsh(norm_adj, k=2, which="LA", tol=1e-4, maxiter=500)
                order = np.argsort(eigvals)
                fiedler = eigvecs[:, order[0]] * inv_sqrt_d

            order = np.argsort(fiedler)
            min_side = max(2, min(4, m // 4))
            best_cond = float("inf")
            best_split = m // 2

            candidates = np.unique(
                np.linspace(min_side, m - min_side, num=min(30, max(1, m - 2 * min_side + 1)), dtype=int)
            )
            sub_coo = sub_adj.tocoo()
            rank = np.empty(m, dtype=np.int64)
            rank[order] = np.arange(m)
            r_u = rank[sub_coo.row]
            r_v = rank[sub_coo.col]
            w_uv = sub_coo.data
            prefix_vol = np.cumsum(deg[order])

            for cut_pos in candidates:
                cut_mask = (r_u < cut_pos) & (r_v >= cut_pos)
                cut_w = float(w_uv[cut_mask].sum())
                vol1 = float(prefix_vol[cut_pos - 1])
                vol2 = total_vol - vol1
                denom = min(vol1, vol2)
                cond = cut_w / denom if denom > 1e-12 else float("inf")
                if cond < best_cond:
                    best_cond = cond
                    best_split = int(cut_pos)

            g1 = order[:best_split]
            g2 = order[best_split:]
            if len(g1) > 0 and len(g2) > 0:
                return g1, g2, float(min(1.0, best_cond))
        except Exception:
            pass

    centered = sub_emb - sub_emb.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[0]
    med = float(np.median(proj))
    g1 = np.where(proj <= med)[0]
    g2 = np.where(proj > med)[0]
    if len(g1) == 0 or len(g2) == 0:
        half = m // 2
        g1, g2 = np.arange(half), np.arange(half, m)
    return g1, g2, 0.5


def _spherical_kmeans_numpy(
    x: np.ndarray,
    k: int,
    n_iters: int = 15,
    seed: int = 42,
) -> np.ndarray:
    """Fast, deterministic farthest-point spherical k-means in pure NumPy."""
    n = x.shape[0]
    if k <= 1 or n <= k:
        return np.arange(n, dtype=np.int64) % max(1, k)
    rng = np.random.default_rng(seed)
    # Deterministic farthest-point seeding on cosine distance (1 - x_i . c)
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    first = int(np.argmax(np.sum(x @ x.T, axis=1)))
    centers[0] = x[first]
    min_dist = 1.0 - (x @ centers[0])
    for c_idx in range(1, k):
        probs = np.maximum(min_dist, 0.0) ** 2
        s = float(probs.sum())
        if s <= 1e-12:
            pick = int(rng.integers(0, n))
        else:
            pick = int(np.argmax(probs))  # deterministic farthest-point for reproducibility
        centers[c_idx] = x[pick]
        min_dist = np.minimum(min_dist, 1.0 - (x @ centers[c_idx]))

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iters):
        sims = x @ centers.T
        new_labels = np.argmax(sims, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c_idx in range(k):
            mask = labels == c_idx
            if np.any(mask):
                mu = x[mask].mean(axis=0)
                norm = float(np.linalg.norm(mu))
                if norm > 1e-12:
                    centers[c_idx] = mu / norm
    return labels


def detect_seam_queries(
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    summaries: Dict[int, ClusterSummary],
    min_cluster_size: int = 10,
    min_pocket_size: int = 4,
    max_conductance: float = 0.50,
    max_medoids: int = 3,
    enforce_fiedler_max_frac: Optional[float] = None,
    use_conductance: bool = True,
    use_non_entity: bool = True,
    override_tfidf: Optional[sparse.csr_matrix] = None,
    use_leverage: bool = True,
    random_order: bool = False,
    seed: int = 42,
) -> List[QueryItem]:
    """Type-II: Unbalanced Minority-Pocket + Bottleneck Intra-Cluster False-Merge Detector.

    Upgrades v2.1 balanced Fiedler cutting (which split large pure majority events
    in half and missed 10%-25% secondary false-merged pockets):
      - Runs deterministic farthest-point spherical KMeans (k=3..6) inside each cluster C_k
      - Evaluates every candidate minority pocket S_p (4 <= |S_p| <= 0.48 |C_k|) by:
          (a) Semantic pocket isolation: intra_sim(S_p) - cos(mu_p, mu_rest)
          (b) Graph structural bottleneck: (1 - conductance(S_p))^2  (penalizes tightly
              connected retweet sub-cliques that have cond > 0.55!)
          (c) Non-entity cut divergence: (1 - phi_non_ent(cut_edges))
          (d) Gentle log-leverage: log(1 + |S_p|)
      - Also checks the Fiedler 2-way cut and keeps the higher-scoring seam per cluster.
    """
    adj = graph.adj
    emb = graph.embeddings
    non_ent = override_tfidf if override_tfidf is not None else graph.non_entity_tfidf

    queries: List[QueryItem] = []
    for cid, summ in summaries.items():
        if summ.size < min_cluster_size:
            continue
        members = np.asarray(summ.members, dtype=np.int64)
        sub_adj = adj[members][:, members].tocsr()
        sub_emb = emb[members]

        k_sub = min(6, max(3, summ.size // 12))
        klab = _spherical_kmeans_numpy(sub_emb, k=k_sub, n_iters=15, seed=42 + cid)

        candidate_splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for p in range(k_sub):
            p_loc = np.where(klab == p)[0]
            r_loc = np.where(klab != p)[0]
            if len(p_loc) >= min_pocket_size and len(p_loc) <= int(0.48 * summ.size) and len(r_loc) >= min_pocket_size:
                candidate_splits.append((r_loc, p_loc))

        # Also include Fiedler bipartition as a candidate
        fg1, fg2, _ = _bipartition_cluster(sub_adj, sub_emb)
        if len(fg1) >= min_pocket_size and len(fg2) >= min_pocket_size:
            if len(fg1) < len(fg2):
                fg1, fg2 = fg2, fg1
            if enforce_fiedler_max_frac is None or len(fg2) <= int(enforce_fiedler_max_frac * summ.size):
                candidate_splits.append((fg1, fg2))

        best_q: Optional[QueryItem] = None
        sub_coo = sub_adj.tocoo()
        sub_row = sub_coo.row
        sub_col = sub_coo.col

        for main_loc, pocket_loc in candidate_splits:
            g_main = members[main_loc]
            g_pock = members[pocket_loc]

            p_main = emb[g_main].mean(axis=0)
            p_pock = emb[g_pock].mean(axis=0)
            p_main = p_main / max(float(np.linalg.norm(p_main)), 1e-12)
            p_pock = p_pock / max(float(np.linalg.norm(p_pock)), 1e-12)

            intra_pock = float(np.mean(emb[g_pock] @ p_pock))
            cross_sim = float(np.dot(p_main, p_pock))
            # Skip tiny near-duplicate retweet clumps (size < 6 & intra > 0.96)
            if len(g_pock) < 6 and intra_pock > 0.96:
                continue

            sem_sep = max(0.0, intra_pock - cross_sim)
            if sem_sep < 0.08:
                continue

            vol_pock = float(sub_adj[pocket_loc].sum())
            cut_w = float(sub_adj[pocket_loc][:, main_loc].sum())
            cond = cut_w / max(vol_pock, 1e-12) if vol_pock > 1e-12 else 0.0
            if use_conductance and cond > max_conductance:
                continue

            in_pock = np.zeros(len(members), dtype=bool)
            in_pock[pocket_loc] = True
            cut_mask = in_pock[sub_row] & (~in_pock[sub_col])
            if np.any(cut_mask):
                cu = members[sub_row[cut_mask]]
                cv = members[sub_col[cut_mask]]
                cut_edges = list(zip(cu.tolist(), cv.tolist()))
                if use_non_entity and non_ent.shape[1] > 0:
                    phi_non_ent = float(np.asarray(non_ent[cu].multiply(non_ent[cv]).sum(axis=1)).mean())
                else:
                    phi_non_ent = 0.0
            else:
                cut_edges = []
                phi_non_ent = 0.0

            cond_factor = ((1.0 - min(0.95, cond)) ** 2) if use_conductance else 1.0
            non_ent_factor = max(0.15, 1.0 - phi_non_ent) if use_non_entity else 1.0
            lev_factor = float(np.log1p(len(g_pock))) if use_leverage else 1.0
            score = sem_sep * cond_factor * non_ent_factor * lev_factor

            if best_q is None or score > best_q.score:
                m_main = [int(g_main[k]) for k in np.argsort(-(emb[g_main] @ p_main))[:max_medoids]]
                m_pock = [int(g_pock[k]) for k in np.argsort(-(emb[g_pock] @ p_pock))[:max_medoids]]
                best_q = QueryItem(
                    query_id=f"seam_c{cid}",
                    strategy="sug_router",
                    query_type="seam",
                    score=score,
                    primary_nodes=m_main[:1] + m_pock[:1],
                    affected_group=summ.members,
                    candidate_clusters=[cid],
                    seam_cluster_id=cid,
                    seam_sub1=g_main.tolist(),
                    seam_sub2=g_pock.tolist(),
                    seam_sub1_medoids=m_main,
                    seam_sub2_medoids=m_pock,
                    seam_cut_edges=cut_edges,
                    meta={
                        "pocket_size": float(len(g_pock)),
                        "intra_pock": intra_pock,
                        "cross_sim": cross_sim,
                        "conductance": cond,
                        "phi_non_ent": phi_non_ent,
                        "n_cut_edges": float(len(cut_edges)),
                    },
                )

        if best_q is not None:
            queries.append(best_q)

    if random_order:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(queries))
        queries = [queries[int(idx)] for idx in perm]
    else:
        queries.sort(key=lambda q: -q.score)
    return queries


def _multi_route_candidates(
    node_idx: int,
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    summaries: Dict[int, ClusterSummary],
    proto_matrix: np.ndarray,
    proto_cids: np.ndarray,
    max_candidates: int = 4,
    exclude_cid: Optional[int] = None,
    min_cand_size: int = 2,
) -> List[int]:
    """Multi-route candidate retrieval: graph neighbors + dense prototypes + lexical/entity."""
    cands: List[int] = []
    seen = set()
    if exclude_cid is not None:
        seen.add(exclude_cid)

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
        best_lex_cid = None
        best_lex = 0
        for cid, summ in summaries.items():
            if cid in seen or summ.size < min_cand_size:
                continue
            ov = len(node_terms.intersection(summ.top_keywords))
            if ov > best_lex:
                best_lex = ov
                best_lex_cid = cid
        if best_lex_cid is not None:
            _add(best_lex_cid)

    if len(proto_cids) > 0 and len(cands) < max_candidates:
        sims = proto_matrix @ graph.embeddings[node_idx]
        for pos in np.argsort(-sims):
            _add(int(proto_cids[pos]))
            if len(cands) >= max_candidates:
                break

    return cands[:max_candidates]


def detect_bridge_and_seed_queries(
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    summaries: Dict[int, ClusterSummary],
    max_candidates: int = 4,
    seed_max_size: int = 5,
    dedup_sim_threshold: float = 0.92,
) -> Tuple[List[QueryItem], List[QueryItem]]:
    """Type-III: Inter-Cluster Boundary Bridges & Micro-Cluster Seeds."""
    adj = graph.adj
    emb = graph.embeddings
    n = len(ds)

    valid_cids = [cid for cid, s in summaries.items() if s.size >= 2]
    if valid_cids:
        proto_cids = np.array(valid_cids, dtype=np.int64)
        proto_matrix = np.vstack([summaries[c].prototype for c in valid_cids])
    else:
        proto_cids = np.array([], dtype=np.int64)
        proto_matrix = np.zeros((0, emb.shape[1]), dtype=np.float32)

    bridge_raw: List[Tuple[float, int, Dict[int, float]]] = []
    indptr, indices, data = adj.indptr, adj.indices, adj.data
    for i in range(n):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue
        comm_w: Dict[int, float] = defaultdict(float)
        tot_w = 0.0
        for p in range(start, end):
            w = float(data[p])
            c_nb = int(labels[indices[p]])
            comm_w[c_nb] += w
            tot_w += w
        if len(comm_w) < 2 or tot_w <= 0.0:
            continue
        probs = np.array([w / tot_w for w in comm_w.values()], dtype=np.float64)
        ent = float(-np.sum(probs * np.log2(probs + 1e-12)))
        score = ent * (1.0 - float(probs.max())) * float(np.log1p(end - start))
        bridge_raw.append((score, i, comm_w))

    bridge_raw.sort(key=lambda x: -x[0])

    bridge_queries: List[QueryItem] = []
    used_bridge_nodes = set()
    for score, i, comm_w in bridge_raw:
        if i in used_bridge_nodes:
            continue
        group = [i]
        row = adj.getrow(i)
        for nb in row.indices.tolist():
            if nb not in used_bridge_nodes and float(np.dot(emb[i], emb[nb])) >= dedup_sim_threshold:
                group.append(nb)
        for g in group:
            used_bridge_nodes.add(g)

        cands = _multi_route_candidates(
            i, ds, graph, labels, summaries, proto_matrix, proto_cids,
            max_candidates=max_candidates, exclude_cid=None, min_cand_size=2,
        )
        cur_c = int(labels[i])
        if cur_c not in cands:
            cands = [cur_c] + cands[: max_candidates - 1]

        bridge_queries.append(
            QueryItem(
                query_id=f"bridge_n{i}",
                strategy="sug_router",
                query_type="bridge",
                score=score,
                primary_nodes=[i],
                affected_group=group,
                candidate_clusters=cands[:max_candidates],
                meta={"n_incident_comms": float(len(comm_w)), "group_size": float(len(group))},
            )
        )

    seed_queries: List[QueryItem] = []
    for cid, summ in summaries.items():
        if summ.size > seed_max_size:
            continue
        medoid = summ.medoids[0]
        cands = _multi_route_candidates(
            medoid, ds, graph, labels, summaries, proto_matrix, proto_cids,
            max_candidates=max_candidates, exclude_cid=cid, min_cand_size=2,
        )
        if not cands:
            continue
        cand_sims = sorted(
            [float(np.dot(emb[medoid], summaries[c].prototype)) for c in cands], reverse=True
        )
        s1 = cand_sims[0]
        s2 = cand_sims[1] if len(cand_sims) > 1 else 0.0
        size_bonus = float(np.log1p(summ.size)) if summ.size >= 2 else 0.35
        ambiguity = (1.0 - (s1 - s2)) + 0.5 * s1
        score = size_bonus * ambiguity

        seed_queries.append(
            QueryItem(
                query_id=f"seed_c{cid}_n{medoid}",
                strategy="sug_router",
                query_type="seed",
                score=score,
                primary_nodes=[medoid],
                affected_group=summ.members,
                candidate_clusters=cands,
                meta={"seed_size": float(summ.size), "top1_sim": s1, "margin": s1 - s2},
            )
        )

    seed_queries.sort(key=lambda q: -q.score)
    return bridge_queries, seed_queries


def _interleave_query_queues(
    queues: List[Tuple[str, List[QueryItem], float]],
    budget_queries: int,
) -> List[QueryItem]:
    """Deficit-based interleaved scheduler across query types."""
    active_queues = [(n, ql, f) for n, ql, f in queues if f > 0.0]
    ptrs = {name: 0 for name, _, _ in active_queues}
    counts = {name: 0 for name, _, _ in active_queues}
    qmap = {name: ql for name, ql, _ in active_queues}

    selected: List[QueryItem] = []
    while len(selected) < budget_queries:
        best_name = None
        best_deficit = -1e9
        total_so_far = max(1, len(selected))
        for name, qlist, target_frac in active_queues:
            if ptrs[name] >= len(qlist):
                continue
            deficit = target_frac - (counts[name] / total_so_far)
            if deficit > best_deficit:
                best_deficit = deficit
                best_name = name
        if best_name is None:
            break
        selected.append(qmap[best_name][ptrs[best_name]])
        ptrs[best_name] += 1
        counts[best_name] += 1

    return selected


def select_sug_queries(
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    budget_queries: int,
    merge_frac: float = 0.45,
    seam_frac: float = 0.25,
    bridge_frac: float = 0.15,
    max_candidates: int = 4,
) -> List[QueryItem]:
    """Assemble symmetric mesoscopic SUG-Router query stream in proportional interleaved order.

    Interleaving ensures that when a real-token budget bucket truncates the stream
    early at ANY token cap, the executed subset preserves the balanced ratio of
    Merge-Pair, Split-Seam, Bridge, and Seed queries.
    """
    summaries = build_cluster_summaries(ds, graph.embeddings, labels)
    merges = detect_merge_pair_queries(ds, graph, labels, summaries)
    seams = detect_seam_queries(ds, graph, labels, summaries)
    bridges, seeds = detect_bridge_and_seed_queries(
        ds, graph, labels, summaries, max_candidates=max_candidates
    )

    seed_frac = max(0.0, 1.0 - merge_frac - seam_frac - bridge_frac)
    queues = [
        ("merge", merges, merge_frac),
        ("seam", seams, seam_frac),
        ("bridge", bridges, bridge_frac),
        ("seed", seeds, seed_frac),
    ]
    return _interleave_query_queues(queues, budget_queries)


def select_random_meso_queries(
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    budget_queries: int,
    merge_frac: float = 0.65,
    seam_frac: float = 0.35,
    seed: int = 42,
) -> List[QueryItem]:
    """Ablation selector: uniformly random ordering over the exact candidate meso pool."""
    summaries = build_cluster_summaries(ds, graph.embeddings, labels)
    merges = detect_merge_pair_queries(ds, graph, labels, summaries, random_order=True, seed=seed)
    seams = detect_seam_queries(ds, graph, labels, summaries, random_order=True, seed=seed + 1000)
    for q in merges:
        q.strategy = "random_meso"
    for q in seams:
        q.strategy = "random_meso"
    queues = [
        ("merge", merges, merge_frac),
        ("seam", seams, seam_frac),
    ]
    return _interleave_query_queues(queues, budget_queries)


def select_random_queries(
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    budget_queries: int,
    max_candidates: int = 4,
    seed: int = 42,
) -> List[QueryItem]:
    """Baseline 1: Uniform random node selection with identical multi-route candidate pool."""
    rng = np.random.default_rng(seed)
    summaries = build_cluster_summaries(ds, graph.embeddings, labels)
    valid_cids = [cid for cid, s in summaries.items() if s.size >= 2]
    proto_cids = np.array(valid_cids, dtype=np.int64)
    proto_matrix = (
        np.vstack([summaries[c].prototype for c in valid_cids])
        if valid_cids else np.zeros((0, graph.embeddings.shape[1]), dtype=np.float32)
    )

    chosen = rng.choice(len(ds), size=min(budget_queries, len(ds)), replace=False)
    queries: List[QueryItem] = []
    for idx in chosen.tolist():
        cands = _multi_route_candidates(
            idx, ds, graph, labels, summaries, proto_matrix, proto_cids,
            max_candidates=max_candidates, exclude_cid=None, min_cand_size=2,
        )
        cur_c = int(labels[idx])
        if cur_c not in cands:
            cands = [cur_c] + cands[: max_candidates - 1]
        queries.append(
            QueryItem(
                query_id=f"rand_n{idx}",
                strategy="random",
                query_type="node",
                score=1.0,
                primary_nodes=[idx],
                affected_group=[idx],
                candidate_clusters=cands[:max_candidates],
            )
        )
    return queries


def select_rep_div_queries(
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    budget_queries: int,
    max_candidates: int = 4,
    div_lambda: float = 0.5,
) -> List[QueryItem]:
    """Baseline 2: RagSEDE-style Representativeness + Diversity (MMR) node sampling."""
    summaries = build_cluster_summaries(ds, graph.embeddings, labels)
    valid_cids = [cid for cid, s in summaries.items() if s.size >= 2]
    proto_cids = np.array(valid_cids, dtype=np.int64)
    proto_matrix = (
        np.vstack([summaries[c].prototype for c in valid_cids])
        if valid_cids else np.zeros((0, graph.embeddings.shape[1]), dtype=np.float32)
    )

    emb = graph.embeddings
    n = len(ds)
    deg = np.asarray(graph.adj.sum(axis=1)).ravel()
    rep = deg / max(float(deg.max()), 1e-12)

    selected_idx: List[int] = []
    max_sim_to_sel = np.full(n, -1.0, dtype=np.float32)
    chosen_mask = np.zeros(n, dtype=bool)

    for _ in range(min(budget_queries, n)):
        if not selected_idx:
            best = int(np.argmax(rep))
        else:
            mmr = (1.0 - div_lambda) * rep - div_lambda * max_sim_to_sel
            mmr[chosen_mask] = -1e9
            best = int(np.argmax(mmr))
        selected_idx.append(best)
        chosen_mask[best] = True
        sims_new = emb @ emb[best]
        max_sim_to_sel = np.maximum(max_sim_to_sel, sims_new)

    queries: List[QueryItem] = []
    for rank, idx in enumerate(selected_idx):
        cands = _multi_route_candidates(
            idx, ds, graph, labels, summaries, proto_matrix, proto_cids,
            max_candidates=max_candidates, exclude_cid=None, min_cand_size=2,
        )
        cur_c = int(labels[idx])
        if cur_c not in cands:
            cands = [cur_c] + cands[: max_candidates - 1]
        queries.append(
            QueryItem(
                query_id=f"repdiv_n{idx}",
                strategy="rep_div",
                query_type="node",
                score=float(budget_queries - rank),
                primary_nodes=[idx],
                affected_group=[idx],
                candidate_clusters=cands[:max_candidates],
            )
        )
    return queries


def select_cequel_edge_queries(
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    budget_queries: int,
    sim_target: float = 0.45,
) -> List[QueryItem]:
    """Baseline 3: Cequel-style uncertain edge / pair querying (must-link / cannot-link)."""
    adj = graph.adj
    emb = graph.embeddings
    coo = adj.tocoo()

    cand_edges: List[Tuple[float, int, int, float, bool]] = []
    for u, v, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if u >= v:
            continue
        sim = float(np.dot(emb[u], emb[v]))
        same_c = int(labels[u]) == int(labels[v])
        uncertainty = 1.0 - abs(sim - sim_target)
        if not same_c and sim >= sim_target:
            uncertainty += 0.25
        cand_edges.append((uncertainty, u, v, sim, same_c))

    cand_edges.sort(key=lambda x: -x[0])

    queries: List[QueryItem] = []
    node_use: Dict[int, int] = defaultdict(int)
    for unc, u, v, sim, same_c in cand_edges:
        if len(queries) >= budget_queries:
            break
        if node_use[u] >= 2 or node_use[v] >= 2:
            continue
        node_use[u] += 1
        node_use[v] += 1
        queries.append(
            QueryItem(
                query_id=f"cequel_e_{u}_{v}",
                strategy="cequel_edge",
                query_type="pair",
                score=unc,
                primary_nodes=[u, v],
                affected_group=[u, v],
                candidate_clusters=list({int(labels[u]), int(labels[v])}),
                meta={"edge_sim": sim, "same_init_cluster": float(same_c)},
            )
        )
    return queries
