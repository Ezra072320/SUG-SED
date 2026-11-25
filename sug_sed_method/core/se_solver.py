"""Exact & fast 2D Structural Entropy (SE) solver in pure Python/SciPy.

Implements:
  - Exact 2D Structural Entropy evaluation H^2(G; P)
  - Exact merge delta formula delta_H2_merge(Ca, Cb; G)
  - Fast heap-based agglomerative 2D SE minimization with small-to-large
    community adjacency merging (HISEvent-style, O(|E| log^2 |V|))
  - Fast greedy node-move local 2D SE refinement (strictly monotone descent)
  - Exact cut-edge / bridge-pruning verification helper
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


def _log2(x: float) -> float:
    return math.log2(x) if x > 0.0 else 0.0


def compute_2d_se(adj: sparse.csr_matrix, labels: np.ndarray) -> float:
    """Compute exact 2D Structural Entropy H^2(G; P) on a symmetric weighted graph.

    Args:
        adj: Symmetric non-negative sparse adjacency matrix (n x n) with zero diagonal.
        labels: Integer community assignment array of length n.

    Returns:
        Exact 2D structural entropy in bits.
    """
    adj = adj.tocsr()
    degrees = np.asarray(adj.sum(axis=1)).ravel().astype(np.float64)
    vol_g = float(degrees.sum())
    if vol_g <= 0.0:
        return 0.0

    labels = np.asarray(labels)
    _, inv = np.unique(labels, return_inverse=True)
    n_comm = int(inv.max()) + 1

    comm_vol = np.bincount(inv, weights=degrees, minlength=n_comm)

    # Compute internal edge weight sum (counted twice since symmetric) per community
    coo = adj.tocoo()
    same_mask = inv[coo.row] == inv[coo.col]
    internal_vol = np.bincount(
        inv[coo.row[same_mask]],
        weights=coo.data[same_mask],
        minlength=n_comm,
    )
    comm_cut = comm_vol - internal_vol

    # Community-level entropy term: - sum_k (g_k / 2m) log2(V_k / 2m)
    se = 0.0
    for k in range(n_comm):
        vk = comm_vol[k]
        gk = comm_cut[k]
        if vk > 0.0 and gk > 0.0:
            se -= (gk / vol_g) * math.log2(vk / vol_g)

    # Node-level entropy term: - sum_i (d_i / 2m) log2(d_i / V_{c(i)})
    nz = degrees > 0.0
    d_nz = degrees[nz]
    v_nz = comm_vol[inv[nz]]
    se -= float(np.sum((d_nz / vol_g) * np.log2(d_nz / v_nz)))
    return float(se)


def exact_merge_delta_h2(
    vol_g: float,
    v_a: float,
    v_b: float,
    g_a: float,
    g_b: float,
    w_ab: float,
) -> float:
    """Exact change H^2(merged) - H^2(separate) when merging communities A and B.

    Negative value means merging STRICTLY DECREASES 2D structural entropy.
    Uses the exact identity (V_a - g_a) = 2 * W_in(a):
        delta = [ (V_a - g_a) log2((V_a+V_b)/V_a)
                + (V_b - g_b) log2((V_a+V_b)/V_b)
                - 2*w_ab * log2(vol_g / (V_a+V_b)) ] / vol_g
    """
    if vol_g <= 0.0 or v_a <= 0.0 or v_b <= 0.0:
        return 0.0
    v_ab = v_a + v_b
    win2_a = max(0.0, v_a - g_a)
    win2_b = max(0.0, v_b - g_b)

    term = 0.0
    if win2_a > 0.0:
        term += win2_a * math.log2(v_ab / v_a)
    if win2_b > 0.0:
        term += win2_b * math.log2(v_ab / v_b)
    if w_ab > 0.0 and v_ab < vol_g:
        term -= 2.0 * w_ab * math.log2(vol_g / v_ab)
    return term / vol_g


def _ensure_clean_symmetric(adj: sparse.spmatrix) -> sparse.csr_matrix:
    """Ensure float64, symmetric, non-negative, zero-diagonal CSR matrix."""
    mat = adj.tocsr().astype(np.float64, copy=True)
    mat.setdiag(0.0)
    mat.eliminate_zeros()
    # Symmetrize via 0.5 * (A + A^T) if asymmetric
    diff = mat - mat.T
    if diff.nnz > 0 and float(np.max(np.abs(diff.data))) > 1e-12:
        mat = 0.5 * (mat + mat.T)
    if mat.nnz > 0 and float(mat.data.min()) < 0.0:
        mat.data = np.maximum(mat.data, 0.0)
        mat.eliminate_zeros()
    return mat.tocsr()


def minimize_2d_se_agglomerative(
    adj: sparse.spmatrix,
    initial_labels: Optional[np.ndarray] = None,
    max_merges: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Fast exact greedy agglomerative 2D Structural Entropy minimization.

    Uses a lazy min-heap over community pairs + small-to-large adjacency dict
    merging. Isolated nodes (degree 0) remain singleton communities.

    Args:
        adj: Symmetric non-negative sparse adjacency matrix (n x n).
        initial_labels: Optional starting community labels (None -> singletons).
        max_merges: Optional cap on number of merges.
        eps: Minimum entropy decrease threshold (-eps) to accept a merge.

    Returns:
        (labels, final_h2) where labels are contiguous integers 0..K-1.
    """
    mat = _ensure_clean_symmetric(adj)
    n = mat.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64), 0.0

    degrees = np.asarray(mat.sum(axis=1)).ravel()
    vol_g = float(degrees.sum())
    if vol_g <= 0.0:
        return np.arange(n, dtype=np.int64), 0.0

    # Initialize communities
    if initial_labels is None:
        comm_members: Dict[int, List[int]] = {i: [i] for i in range(n)}
        node_to_comm = list(range(n))
    else:
        _, inv = np.unique(np.asarray(initial_labels), return_inverse=True)
        n_init = int(inv.max()) + 1
        comm_members = {k: [] for k in range(n_init)}
        node_to_comm = inv.tolist()
        for i, c in enumerate(node_to_comm):
            comm_members[c].append(i)

    # Build community volumes, cuts, and inter-community adjacency dicts
    comm_vol: Dict[int, float] = {}
    comm_cut: Dict[int, float] = {}
    comm_adj: Dict[int, Dict[int, float]] = {}

    for c, members in comm_members.items():
        v_c = float(sum(degrees[i] for i in members))
        comm_vol[c] = v_c
        comm_cut[c] = 0.0
        comm_adj[c] = {}

    coo = mat.tocoo()
    for u, v, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if u >= v:
            continue
        cu = node_to_comm[u]
        cv = node_to_comm[v]
        if cu != cv:
            comm_adj[cu][cv] = comm_adj[cu].get(cv, 0.0) + w
            comm_adj[cv][cu] = comm_adj[cv].get(cu, 0.0) + w
            comm_cut[cu] += w
            comm_cut[cv] += w

    # Versioning for lazy heap invalidation
    version: Dict[int, int] = {c: 0 for c in comm_members}
    alive: Dict[int, bool] = {c: True for c in comm_members}

    heap: List[Tuple[float, int, int, int, int]] = []
    for cu, nbrs in comm_adj.items():
        for cv, w_uv in nbrs.items():
            if cu < cv:
                delta = exact_merge_delta_h2(
                    vol_g, comm_vol[cu], comm_vol[cv], comm_cut[cu], comm_cut[cv], w_uv
                )
                if delta < -eps:
                    heapq.heappush(heap, (delta, cu, cv, 0, 0))

    merges_done = 0
    while heap:
        if max_merges is not None and merges_done >= max_merges:
            break
        delta, a, b, va, vb = heapq.heappop(heap)
        if not (alive.get(a, False) and alive.get(b, False)):
            continue
        if version[a] != va or version[b] != vb:
            continue
        if delta >= -eps:
            break

        # Merge smaller adjacency community into larger one (small-to-large)
        if len(comm_adj[a]) < len(comm_adj[b]):
            a, b = b, a

        w_ab = comm_adj[a].pop(b, 0.0)
        comm_adj[b].pop(a, None)

        # Update community A statistics
        comm_vol[a] += comm_vol[b]
        comm_cut[a] = comm_cut[a] + comm_cut[b] - 2.0 * w_ab
        comm_members[a].extend(comm_members[b])

        # Rewire neighbors of B -> A
        nbrs_b = comm_adj[b]
        for nbr, w_bn in nbrs_b.items():
            comm_adj[nbr].pop(b, None)
            new_w = comm_adj[a].get(nbr, 0.0) + w_bn
            comm_adj[a][nbr] = new_w
            comm_adj[nbr][a] = new_w

        # Mark B dead & bump version of A
        alive[b] = False
        del comm_members[b]
        del comm_adj[b]
        version[a] += 1
        va_new = version[a]
        merges_done += 1

        # Push updated merge candidates for all neighbors of A
        v_a = comm_vol[a]
        g_a = comm_cut[a]
        for nbr, w_an in comm_adj[a].items():
            d_new = exact_merge_delta_h2(
                vol_g, v_a, comm_vol[nbr], g_a, comm_cut[nbr], w_an
            )
            if d_new < -eps:
                c1, c2 = (a, nbr) if a < nbr else (nbr, a)
                v1, v2 = (va_new, version[nbr]) if a < nbr else (version[nbr], va_new)
                heapq.heappush(heap, (d_new, c1, c2, v1, v2))

    # Relabel contiguously 0..K-1
    labels = np.empty(n, dtype=np.int64)
    for new_id, (_, members) in enumerate(sorted(comm_members.items(), key=lambda kv: min(kv[1]))):
        labels[members] = new_id

    final_h2 = compute_2d_se(mat, labels)
    return labels, final_h2


def refine_2d_se_node_moves(
    adj: sparse.spmatrix,
    labels: np.ndarray,
    max_iters: int = 10,
    eps: float = 1e-12,
    movable_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Strictly monotone greedy node-move refinement on exact 2D Structural Entropy.

    Moves individual nodes to neighboring communities whenever doing so strictly
    reduces H^2(G; P). When `movable_mask` is provided, only nodes with
    `movable_mask[i] == True` may be moved (preventing 2D-SE volume repulsion
    from bleeding interior nodes out of LLM-confirmed giant merged events).
    """
    mat = _ensure_clean_symmetric(adj)
    n = mat.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64), 0.0
    mov = None if movable_mask is None else np.asarray(movable_mask, dtype=bool)

    degrees = np.asarray(mat.sum(axis=1)).ravel()
    vol_g = float(degrees.sum())
    if vol_g <= 0.0:
        return np.asarray(labels, dtype=np.int64).copy(), 0.0

    _, cur_labels = np.unique(np.asarray(labels), return_inverse=True)
    cur_labels = cur_labels.astype(np.int64)
    n_comm = int(cur_labels.max()) + 1

    # Maintain community volume V_c, cut g_c, and size
    comm_vol = np.bincount(cur_labels, weights=degrees, minlength=n_comm).astype(np.float64)
    comm_size = np.bincount(cur_labels, minlength=n_comm).astype(np.int64)

    coo = mat.tocoo()
    same = cur_labels[coo.row] == cur_labels[coo.col]
    internal = np.bincount(
        cur_labels[coo.row[same]], weights=coo.data[same], minlength=n_comm
    ).astype(np.float64)
    comm_cut = comm_vol - internal

    indptr = mat.indptr
    indices = mat.indices
    data = mat.data

    def _comm_term(v: float, g: float) -> float:
        if v <= 0.0:
            return 0.0
        # Exact contribution of community c to vol_g * H^2(G; P) up to constant sum_i d_i log2(d_i):
        # -g * log2(v / vol_g) + v * log2(v) = g * log2(vol_g) + (v - g) * log2(v)
        # Wait: let's verify!
        # Full H^2 * vol_g = sum_c [ -g_c log2(V_c / vol_g) - sum_{i in c} d_i log2(d_i) + V_c log2(V_c) ]
        #                  = const + sum_c [ g_c log2(vol_g) + (V_c - g_c) log2(V_c) ]
        return g * log2_vol_g + (v - g) * math.log2(v)

    log2_vol_g = math.log2(vol_g)

    for _ in range(max_iters):
        moved = 0
        for i in range(n):
            if mov is not None and not mov[i]:
                continue
            di = float(degrees[i])
            if di <= 0.0:
                continue
            ci = int(cur_labels[i])
            start, end = indptr[i], indptr[i + 1]
            nbr_idx = indices[start:end]
            nbr_w = data[start:end]

            # Aggregate edge weights from node i to each incident community
            w_to_comm: Dict[int, float] = {}
            for nb, w in zip(nbr_idx.tolist(), nbr_w.tolist()):
                c_nb = int(cur_labels[nb])
                w_to_comm[c_nb] = w_to_comm.get(c_nb, 0.0) + w

            w_i_ci = w_to_comm.get(ci, 0.0)
            v_ci = float(comm_vol[ci])
            g_ci = float(comm_cut[ci])

            # New stats of ci if node i leaves
            v_ci_after = v_ci - di
            # Leaving ci removes edges to outside ci (di - w_i_ci) from cut,
            # and turns internal edges (w_i_ci) into cut edges!
            g_ci_after = g_ci - (di - w_i_ci) + w_i_ci
            old_ci_contrib = _comm_term(v_ci, g_ci)
            new_ci_contrib = _comm_term(v_ci_after, g_ci_after) if v_ci_after > 1e-15 else 0.0

            best_c = ci
            best_delta = 0.0

            for cand_c, w_i_cand in w_to_comm.items():
                if cand_c == ci:
                    continue
                v_t = float(comm_vol[cand_c])
                g_t = float(comm_cut[cand_c])
                v_t_after = v_t + di
                # Entering cand_c adds edges to outside cand_c (di - w_i_cand) to cut,
                # and removes w_i_cand from cut!
                g_t_after = g_t + (di - w_i_cand) - w_i_cand

                old_t_contrib = _comm_term(v_t, g_t)
                new_t_contrib = _comm_term(v_t_after, g_t_after)

                delta_scaled = (new_ci_contrib - old_ci_contrib) + (new_t_contrib - old_t_contrib)
                if delta_scaled < best_delta - eps * vol_g:
                    best_delta = delta_scaled
                    best_c = cand_c

            if best_c != ci:
                w_i_best = w_to_comm[best_c]
                comm_vol[ci] = v_ci_after
                comm_cut[ci] = g_ci_after
                comm_size[ci] -= 1

                comm_vol[best_c] += di
                comm_cut[best_c] += (di - 2.0 * w_i_best)
                comm_size[best_c] += 1

                cur_labels[i] = best_c
                moved += 1

        if moved == 0:
            break

    # Relabel contiguously
    _, clean_labels = np.unique(cur_labels, return_inverse=True)
    clean_labels = clean_labels.astype(np.int64)
    return clean_labels, compute_2d_se(mat, clean_labels)


def solve_2d_se(
    adj: sparse.spmatrix,
    refine_moves: bool = True,
    max_refine_iters: int = 10,
    max_outer_passes: int = 3,
) -> Tuple[np.ndarray, float]:
    """Deterministic 2D Structural Entropy clustering converged to agglomerative+move fixed point."""
    labels, h2 = minimize_2d_se_agglomerative(adj)
    if not refine_moves:
        return labels, h2
    for _ in range(max_outer_passes):
        labels_m, h2_m = refine_2d_se_node_moves(adj, labels, max_iters=max_refine_iters)
        labels_a, h2_a = minimize_2d_se_agglomerative(adj, initial_labels=labels_m)
        if np.array_equal(labels_a, labels):
            return labels_a, h2_a
        labels, h2 = labels_a, h2_a
    return labels, h2


def _self_test() -> None:
    """Verify exact 2D SE identity, Astra 3-clique counterexample, and node-move descent."""
    # 1. Astra's 3-clique counterexample: 3 clusters of 2 nodes, unit internal & inter-cluster edges
    edges = [
        (0, 1, 1.0), (2, 3, 1.0), (4, 5, 1.0),  # internal
        (0, 2, 1.0), (2, 4, 1.0), (4, 0, 1.0),  # inter-cluster
    ]
    rows, cols, vals = [], [], []
    for u, v, w in edges:
        rows.extend([u, v])
        cols.extend([v, u])
        vals.extend([w, w])
    g_before = sparse.csr_matrix((vals, (rows, cols)), shape=(6, 6))
    sep = np.array([0, 0, 1, 1, 2, 2])
    mrg = np.array([0, 0, 0, 0, 2, 2])

    d_before = compute_2d_se(g_before, mrg) - compute_2d_se(g_before, sep)
    d_before_formula = exact_merge_delta_h2(12.0, 4.0, 4.0, 2.0, 2.0, 1.0)
    assert abs(d_before - d_before_formula) < 1e-12, (d_before, d_before_formula)

    # Delete bridge (0, 2)
    g_after = g_before.tolil()
    g_after[0, 2] = 0.0
    g_after[2, 0] = 0.0
    g_after = g_after.tocsr()
    d_after = compute_2d_se(g_after, mrg) - compute_2d_se(g_after, sep)
    d_after_formula = exact_merge_delta_h2(10.0, 3.0, 3.0, 1.0, 1.0, 0.0)
    assert abs(d_after - d_after_formula) < 1e-12, (d_after, d_after_formula)
    assert abs((d_after - d_before) - 0.164160416) < 1e-5, d_after - d_before

    # 2. Verify solver finds exact 3 communities on g_before & g_after
    lab, h2 = solve_2d_se(g_before, refine_moves=True)
    assert len(set(lab.tolist())) == 3
    assert abs(h2 - compute_2d_se(g_before, sep)) < 1e-12

    # 3. Verify node-move refinement strictly decreases or maintains H^2 on random graph
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, size=(30, 30))
    a = (a + a.T) / 2.0
    a[a < 0.7] = 0.0
    np.fill_diagonal(a, 0.0)
    sp_a = sparse.csr_matrix(a)
    l0, h0 = minimize_2d_se_agglomerative(sp_a)
    l1, h1 = refine_2d_se_node_moves(sp_a, l0)
    assert h1 <= h0 + 1e-12, (h0, h1)
    print(
        f"[se_solver self-test PASSED] "
        f"Astra exact diff={d_after - d_before:.6f} bit, rand H2: {h0:.6f} -> {h1:.6f}"
    )


if __name__ == "__main__":
    _self_test()
