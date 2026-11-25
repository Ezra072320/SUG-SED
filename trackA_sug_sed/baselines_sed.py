"""Published Zero-LLM Social Event Detection (SED) Paper Baselines:

  1. EventX    (Liu et al., Story Forest, ACM TKDD 2020): keyword/entity co-occurrence Louvain community detection
  2. SBERT     (Reimers & Gurevych, EMNLP 2019): frozen Sentence-BERT + cosine average-linkage agglomerative clustering
  3. HiSEvent  (Cao et al., AAAI 2024): heterogeneous message graph + 2D Structural Entropy minimization (Default & Dev-Tuned)
  4. HyperSED  (Yu et al., AAAI 2025, arXiv:2412.10712): Semantic-based Anchor Message Graph (SAMG) + Poincare hyperbolic
               distortion optimization + differentiable hyperbolic structural-entropy partitioning
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import MessageGraph, build_message_graph
from core_shared.partition_cache import load_or_solve_2d_se
from core_shared.se_solver import solve_2d_se
from trackA_sug_sed.star_meso import _spherical_kmeans_deterministic


# ---------------------------------------------------------------------------
# 1. EventX (Liu et al., ACM TKDD 2020)
# ---------------------------------------------------------------------------

def run_eventx(
    ds: SocialDataset,
    resolution: float = 1.0,
    max_bucket_size: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keyword/entity co-occurrence message graph + Louvain community detection (EventX)."""
    t0 = time.time()
    n = len(ds)
    if n <= 1:
        return np.zeros(n, dtype=np.int64), {"runtime_sec": 0.0}

    inv_index: Dict[str, List[int]] = defaultdict(list)
    for i in range(n):
        seen_terms = set()
        for e in ds.entities[i]:
            seen_terms.add(f"e:{e.lower()}")
        for h in ds.hashtags[i]:
            seen_terms.add(f"h:{h.lower()}")
        for tok in ds.tokens_lemma[i]:
            if len(tok) >= 3:
                seen_terms.add(f"w:{tok.lower()}")
        for term in seen_terms:
            inv_index[term].append(i)

    edge_w: Dict[Tuple[int, int], float] = {}
    ts = ds.timestamps
    for term, nodes in inv_index.items():
        df = len(nodes)
        if df < 2 or df > max_bucket_size:
            continue
        idf = math.log((n + 1.0) / (df + 1.0)) + 1.0
        base = idf * (1.4 if term.startswith(("e:", "h:")) else 0.7) / math.log2(2.0 + df)
        nodes_sorted = sorted(nodes, key=lambda idx: ts[idx])
        half_win = 6
        for pos, u in enumerate(nodes_sorted):
            for pos2 in range(pos + 1, min(df, pos + half_win + 1)):
                v = nodes_sorted[pos2]
                key = (u, v) if u < v else (v, u)
                edge_w[key] = edge_w.get(key, 0.0) + base

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for (u, v), w in edge_w.items():
        g.add_edge(u, v, weight=float(w))

    comms = nx.algorithms.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
    labels = np.empty(n, dtype=np.int64)
    for cid, comm in enumerate(comms):
        for u in comm:
            labels[int(u)] = cid
    _, clean = np.unique(labels, return_inverse=True)
    return clean.astype(np.int64), {"runtime_sec": round(time.time() - t0, 3), "resolution": resolution}


# ---------------------------------------------------------------------------
# 2. SBERT Agglomerative (Reimers & Gurevych, EMNLP 2019)
# ---------------------------------------------------------------------------

def run_sbert_agglom(
    embeddings: np.ndarray,
    distance_threshold: float = 0.55,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Average-linkage cosine agglomerative clustering on frozen SBERT embeddings."""
    t0 = time.time()
    n = len(embeddings)
    if n <= 1:
        return np.zeros(n, dtype=np.int64), {"runtime_sec": 0.0, "distance_threshold": distance_threshold}
    cl = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=float(distance_threshold),
    )
    labels = cl.fit_predict(embeddings).astype(np.int64)
    return labels, {"runtime_sec": round(time.time() - t0, 3), "distance_threshold": distance_threshold}


# ---------------------------------------------------------------------------
# 3. HiSEvent (Cao et al., AAAI 2024)
# ---------------------------------------------------------------------------

def run_hisevent(
    dataset_name: str,
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    knn_k: int = 15,
    knn_sim_threshold: float = 0.35,
    attr_alpha: float = 0.5,
    embed_model: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Heterogeneous semantic + attribute message graph 2D Structural Entropy (HiSEvent)."""
    t0 = time.time()
    graph = build_message_graph(
        ds,
        embeddings,
        knn_k=knn_k,
        knn_sim_threshold=knn_sim_threshold,
        attr_alpha=attr_alpha,
        compute_non_entity_tfidf=False,
    )
    labels, se_sec = load_or_solve_2d_se(
        dataset_name=dataset_name,
        block_id=block_id,
        adj=graph.adj,
        knn_k=knn_k,
        knn_sim_threshold=knn_sim_threshold,
        attr_alpha=attr_alpha,
        refine_moves=True,
        embed_model=embed_model,
    )
    return labels, {
        "runtime_sec": round(time.time() - t0, 3),
        "se_solve_sec": se_sec,
        "knn_sim_threshold": knn_sim_threshold,
        "attr_alpha": attr_alpha,
    }


# ---------------------------------------------------------------------------
# 4. HyperSED (Yu et al., AAAI 2025, arXiv:2412.10712)
# ---------------------------------------------------------------------------

def _poincare_exp0(v, c: float = 1.0):
    import torch

    sqrt_c = math.sqrt(c)
    norm_v = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=1e-7)
    coeff = torch.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
    return coeff * v


def _poincare_dist(x, y, c: float = 1.0):
    import torch

    diff_sq = torch.sum((x - y) ** 2, dim=-1)
    nx2 = torch.sum(x**2, dim=-1).clamp(max=(1.0 / c) - 1e-4)
    ny2 = torch.sum(y**2, dim=-1).clamp(max=(1.0 / c) - 1e-4)
    denom = ((1.0 - c * nx2) * (1.0 - c * ny2)).clamp(min=1e-6)
    arg = (1.0 + 2.0 * c * diff_sq / denom).clamp(min=1.0 + 1e-6)
    return (1.0 / math.sqrt(c)) * torch.acosh(arg)


def _proj_poincare(x, c: float = 1.0, eps: float = 1e-4):
    import torch

    max_norm = (1.0 / math.sqrt(c)) - eps
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    cond = norm > max_norm
    return torch.where(cond, x * (max_norm / norm.clamp(min=1e-9)), x)


def run_hypersed(
    ds: SocialDataset,
    embeddings: np.ndarray,
    graph: Optional[MessageGraph] = None,
    anchor_ratio: float = 0.12,
    hyp_dim: int = 32,
    epochs: int = 60,
    lr: float = 0.03,
    knn_anchor: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Faithful PyTorch HyperSED (Yu et al., AAAI 2025): SAMG + Hyperbolic Anchor Tree."""
    import torch

    t0 = time.time()
    n = len(ds)
    if n <= 32:
        return np.zeros(n, dtype=np.int64), {"runtime_sec": 0.0}

    if graph is None:
        graph = build_message_graph(ds, embeddings, knn_k=15, knn_sim_threshold=0.40, attr_alpha=0.5)

    # 1. Semantic-based Anchor Message Graph (SAMG)
    n_anchors = int(min(360, max(24, round(n * float(anchor_ratio)))))
    n_anchors = min(n_anchors, n - 1)
    anchor_assign = _spherical_kmeans_deterministic(embeddings, k=n_anchors, n_iters=10)
    # Reindex dense active anchors
    uniq_a, anchor_assign = np.unique(anchor_assign, return_inverse=True)
    A = len(uniq_a)

    anchor_emb = np.zeros((A, embeddings.shape[1]), dtype=np.float32)
    for a in range(A):
        mu = embeddings[anchor_assign == a].mean(axis=0)
        nm = float(np.linalg.norm(mu))
        anchor_emb[a] = mu / max(nm, 1e-12)

    # Aggregate message-graph edge weights to anchor graph P^T W P
    P = sparse.csr_matrix(
        (np.ones(n, dtype=np.float64), (np.arange(n, dtype=np.int64), anchor_assign)),
        shape=(n, A),
    )
    W_anchor = (P.T @ graph.adj @ P).tocsr()
    W_anchor.setdiag(0.0)
    W_anchor.eliminate_zeros()

    # Add semantic anchor k-NN edges so disconnected tail anchors stay structured
    sim_aa = anchor_emb @ anchor_emb.T
    np.fill_diagonal(sim_aa, -2.0)
    k_eff = min(knn_anchor, A - 1)
    rows_k, cols_k, vals_k = [], [], []
    top_idx = np.argpartition(-sim_aa, kth=k_eff - 1, axis=1)[:, :k_eff]
    for u in range(A):
        for v in top_idx[u].tolist():
            s = float(sim_aa[u, v])
            if s >= 0.45:
                rows_k.extend([u, v])
                cols_k.extend([v, u])
                vals_k.extend([s, s])
    W_sem = sparse.csr_matrix((vals_k, (rows_k, cols_k)), shape=(A, A), dtype=np.float64)
    W_sem = W_sem.maximum(W_sem.T)
    W_samg = (W_anchor + W_sem).tocsr()
    W_samg.setdiag(0.0)
    W_samg.eliminate_zeros()

    # Normalize edge weights for hyperbolic attraction
    coo = W_samg.tocoo()
    upper = coo.row < coo.col
    e_u = coo.row[upper]
    e_v = coo.col[upper]
    e_w = coo.data[upper]
    if len(e_u) > 0:
        w_max = float(np.percentile(e_w, 95)) + 1e-6
        e_w_norm = np.clip(e_w / w_max, 0.05, 1.5)
    else:
        e_w_norm = np.array([], dtype=np.float64)

    # 2. Hyperbolic Poincare Space Optimization on GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x_t = torch.tensor(anchor_emb, dtype=torch.float32, device=device)
    proj_layer = torch.nn.Linear(embeddings.shape[1], hyp_dim, bias=False).to(device)
    torch.nn.init.xavier_uniform_(proj_layer.weight, gain=0.4)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=lr, weight_decay=1e-4)

    if len(e_u) > 0:
        u_t = torch.tensor(e_u, dtype=torch.long, device=device)
        v_t = torch.tensor(e_v, dtype=torch.long, device=device)
        w_t = torch.tensor(e_w_norm, dtype=torch.float32, device=device)
        # Pre-filter negative candidate mask via low Euclidean similarity
        low_mask = sim_aa < 0.25
        for _ep in range(epochs):
            opt.zero_grad()
            z = _proj_poincare(_poincare_exp0(proj_layer(x_t), c=1.0), c=1.0)
            d_pos = _poincare_dist(z[u_t], z[v_t], c=1.0)
            loss_pos = torch.mean(w_t * (d_pos**2))

            neg_u = rng.integers(0, A, size=min(len(e_u), 1024))
            neg_v = rng.integers(0, A, size=min(len(e_u), 1024))
            keep = low_mask[neg_u, neg_v]
            if np.any(keep):
                nu_t = torch.tensor(neg_u[keep], dtype=torch.long, device=device)
                nv_t = torch.tensor(neg_v[keep], dtype=torch.long, device=device)
                d_neg = _poincare_dist(z[nu_t], z[nv_t], c=1.0)
                loss_neg = torch.mean(torch.relu(1.6 - d_neg) ** 2)
                loss = loss_pos + 0.8 * loss_neg
            else:
                loss = loss_pos
            loss.backward()
            opt.step()

    with torch.no_grad():
        z_final = _proj_poincare(_poincare_exp0(proj_layer(x_t), c=1.0), c=1.0)
        # Pairwise hyperbolic distance matrix on anchors (A <= 360 -> tiny!)
        diff = z_final.unsqueeze(1) - z_final.unsqueeze(0)
        d2 = torch.sum(diff**2, dim=-1)
        nz = (1.0 - torch.sum(z_final**2, dim=-1).clamp(max=0.9999)).clamp(min=1e-5)
        arg = (1.0 + 2.0 * d2 / (nz.unsqueeze(1) * nz.unsqueeze(0))).clamp(min=1.0 + 1e-6)
        hyp_dist = torch.acosh(arg).cpu().numpy()

    # 3. Hyperbolic Geometry-Aware Structural Entropy Partition Tree over Anchors
    # Edge weights decay exponentially with hyperbolic distance + preserve top hyperbolic neighbors
    if len(e_u) > 0:
        hyp_w = e_w * np.exp(-0.5 * hyp_dist[e_u, e_v])
        r_all = np.concatenate([e_u, e_v])
        c_all = np.concatenate([e_v, e_u])
        w_all = np.concatenate([hyp_w, hyp_w])
        adj_hyp = sparse.csr_matrix((w_all, (r_all, c_all)), shape=(A, A), dtype=np.float64)
    else:
        adj_hyp = sparse.csr_matrix((A, A), dtype=np.float64)
    adj_hyp.setdiag(0.0)
    adj_hyp.eliminate_zeros()

    anchor_labels, _ = solve_2d_se(adj_hyp, refine_moves=True)
    msg_labels = anchor_labels[anchor_assign]
    _, clean = np.unique(msg_labels, return_inverse=True)
    return clean.astype(np.int64), {
        "runtime_sec": round(time.time() - t0, 3),
        "n_anchors": A,
        "device": device,
        "anchor_ratio": anchor_ratio,
    }
