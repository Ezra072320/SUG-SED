"""Exact Branch 1 (Direct Edit) vs. Branch 2 (Constrained Topology Repair + 2D SE Re-Clustering).

Both branches consume the EXACT SAME frozen list of `LLMDecision` objects so that
Gate A cleanly separates:
  (a) Query-level leverage (mesoscopic MERGE_PAIR / SPLIT_SEAM vs single-post routing)
  (b) Graph-propagation leverage (Branch 2 re-clustering cascades over Branch 1 direct edit)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core_shared.graph_builder import MessageGraph
from core_shared.se_solver import minimize_2d_se_agglomerative, refine_2d_se_node_moves
from trackA_sug_sed.high_leverage_router import ClusterSummary, QueryItem


@dataclass
class LLMDecision:
    query_id: str
    query_type: str               # 'merge_pair' | 'seam' | 'bridge' | 'seed' | 'node' | 'pair'
    action: str                   # 'MERGE' | 'SEPARATE' | 'SPLIT' | 'KEEP_MERGED' | 'ASSIGN' | 'NEW_EVENT' | 'NOISE' | 'MUST_LINK' | 'CANNOT_LINK'
    target_cluster: Optional[int] = None
    confidence: float = 1.0
    event_cue: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


def build_oracle_decisions(
    queries: List[QueryItem],
    y_true: np.ndarray,
    labels_init: np.ndarray,
    summaries: Dict[int, ClusterSummary],
    estimated_prompt_tokens: int = 380,
    estimated_completion_tokens: int = 45,
) -> Tuple[List[LLMDecision], Dict[str, float]]:
    """Zero-API-cost Oracle simulator respecting the exact info visible to an LLM."""
    y_true = np.asarray(y_true)
    cluster_majority_true: Dict[int, int] = {}
    for cid, summ in summaries.items():
        vals, cnts = np.unique(y_true[summ.members], return_counts=True)
        cluster_majority_true[cid] = int(vals[int(np.argmax(cnts))])

    decisions: List[LLMDecision] = []
    cand_hits = cand_total = 0
    seam_splits = seam_total = 0
    merge_hits = merge_total = 0

    for q in queries:
        if q.query_type == "merge_pair":
            merge_total += 1
            c1, c2 = int(q.merge_c1), int(q.merge_c2)
            if cluster_majority_true[c1] == cluster_majority_true[c2]:
                merge_hits += 1
                action = "MERGE"
            else:
                action = "SEPARATE"
            decisions.append(
                LLMDecision(
                    query_id=q.query_id,
                    query_type="merge_pair",
                    action=action,
                    confidence=1.0,
                    prompt_tokens=estimated_prompt_tokens,
                    completion_tokens=estimated_completion_tokens,
                )
            )
        elif q.query_type == "seam":
            seam_total += 1
            v1, c1 = np.unique(y_true[q.seam_sub1], return_counts=True)
            v2, c2 = np.unique(y_true[q.seam_sub2], return_counts=True)
            maj1 = int(v1[int(np.argmax(c1))])
            maj2 = int(v2[int(np.argmax(c2))])
            if maj1 != maj2:
                seam_splits += 1
                action = "SPLIT"
            else:
                action = "KEEP_MERGED"
            decisions.append(
                LLMDecision(
                    query_id=q.query_id,
                    query_type="seam",
                    action=action,
                    target_cluster=q.seam_cluster_id,
                    confidence=1.0,
                    prompt_tokens=estimated_prompt_tokens,
                    completion_tokens=estimated_completion_tokens,
                )
            )
        elif q.query_type == "pair":
            u, v = q.primary_nodes[0], q.primary_nodes[1]
            action = "MUST_LINK" if int(y_true[u]) == int(y_true[v]) else "CANNOT_LINK"
            decisions.append(
                LLMDecision(
                    query_id=q.query_id,
                    query_type="pair",
                    action=action,
                    confidence=1.0,
                    prompt_tokens=int(estimated_prompt_tokens * 0.55),
                    completion_tokens=20,
                )
            )
        else:
            cand_total += 1
            node = q.primary_nodes[0]
            true_ev = int(y_true[node])
            matched_cid = None
            for cid in q.candidate_clusters:
                if cluster_majority_true.get(cid) == true_ev:
                    matched_cid = cid
                    break
            if matched_cid is not None:
                cand_hits += 1
                decisions.append(
                    LLMDecision(
                        query_id=q.query_id,
                        query_type=q.query_type,
                        action="ASSIGN",
                        target_cluster=matched_cid,
                        confidence=1.0,
                        prompt_tokens=estimated_prompt_tokens,
                        completion_tokens=estimated_completion_tokens,
                    )
                )
            else:
                decisions.append(
                    LLMDecision(
                        query_id=q.query_id,
                        query_type=q.query_type,
                        action="NEW_EVENT",
                        target_cluster=None,
                        confidence=1.0,
                        event_cue=f"event_{true_ev}",
                        prompt_tokens=estimated_prompt_tokens,
                        completion_tokens=estimated_completion_tokens,
                    )
                )

    stats = {
        "n_queries": float(len(queries)),
        "merge_total": float(merge_total),
        "merge_true_hit_rate": (merge_hits / merge_total) if merge_total > 0 else 0.0,
        "seam_total": float(seam_total),
        "seam_true_split_rate": (seam_splits / seam_total) if seam_total > 0 else 0.0,
        "node_cand_total": float(cand_total),
        "candidate_recall_at_m": (cand_hits / cand_total) if cand_total > 0 else 0.0,
        "total_tokens": float(sum(d.prompt_tokens + d.completion_tokens for d in decisions)),
    }
    return decisions, stats


def collect_prompt_shown_nodes(
    queries: List[QueryItem],
    decisions: List[LLMDecision],
    n_posts: int,
    summaries: Optional[Dict[int, ClusterSummary]] = None,
    include_candidate_medoids: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build boolean masks (target_shown_mask, all_prompt_shown_mask) of length n_posts.

    Ensures Unqueried-Post Net Gain (U_t) excludes EVERY post whose text was
    shown to the LLM in prompts (including all 6 displayed medoids for
    merge_pair / seam and displayed candidate cluster medoids for node/bridge/seed).
    """
    target_mask = np.zeros(n_posts, dtype=bool)
    all_mask = np.zeros(n_posts, dtype=bool)
    qmap = {q.query_id: q for q in queries}

    for dec in decisions:
        q = qmap.get(dec.query_id)
        if q is None:
            continue
        for node in q.primary_nodes:
            idx = int(node)
            if 0 <= idx < n_posts:
                target_mask[idx] = True
                all_mask[idx] = True

        if q.query_type == "merge_pair":
            for m in (q.merge_c1_medoids[:3] + q.merge_c2_medoids[:3]):
                idx = int(m)
                if 0 <= idx < n_posts:
                    target_mask[idx] = True
                    all_mask[idx] = True
        elif q.query_type == "seam":
            for m in (q.seam_sub1_medoids[:3] + q.seam_sub2_medoids[:3]):
                idx = int(m)
                if 0 <= idx < n_posts:
                    target_mask[idx] = True
                    all_mask[idx] = True
        elif q.query_type != "pair":
            if include_candidate_medoids and summaries is not None:
                for cid in q.candidate_clusters:
                    summ = summaries.get(int(cid))
                    if summ is not None:
                        for m in summ.medoids[:3]:
                            idx = int(m)
                            if 0 <= idx < n_posts:
                                all_mask[idx] = True

    return target_mask, all_mask


def _resolve_queried_mask(
    queries: List[QueryItem],
    decisions: List[LLMDecision],
    n_posts: int,
    summaries: Optional[Dict[int, ClusterSummary]],
    mask_mode: str,
) -> np.ndarray:
    if mask_mode == "primary_only":
        mask = np.zeros(n_posts, dtype=bool)
        qmap = {q.query_id: q for q in queries}
        for dec in decisions:
            q = qmap.get(dec.query_id)
            if q is None:
                continue
            for node in q.primary_nodes:
                idx = int(node)
                if 0 <= idx < n_posts:
                    mask[idx] = True
        return mask
    target_mask, all_mask = collect_prompt_shown_nodes(
        queries, decisions, n_posts, summaries=summaries, include_candidate_medoids=True
    )
    if mask_mode == "target_shown":
        return target_mask
    if mask_mode == "all_shown":
        return all_mask
    raise ValueError(f"Unknown mask_mode: {mask_mode!r}")


def apply_direct_assignment(
    labels_init: np.ndarray,
    queries: List[QueryItem],
    decisions: List[LLMDecision],
    conf_threshold: float = 0.6,
    summaries: Optional[Dict[int, ClusterSummary]] = None,
    mask_mode: str = "all_shown",
) -> Tuple[np.ndarray, np.ndarray]:
    """Branch 1: Direct partition edit WITHOUT graph edge surgery or 2D SE re-clustering.

    Order: SPLIT impure seams first (so impure clusters don't contaminate merges),
    then MERGE confirmed cluster pairs, then apply node/pair edits.
    """
    n = len(labels_init)
    labels = np.asarray(labels_init, dtype=np.int64).copy()
    next_cid = int(labels.max()) + 1 if n > 0 else 0
    queried_mask = _resolve_queried_mask(queries, decisions, n, summaries, mask_mode)
    qmap = {q.query_id: q for q in queries}

    # Pass 1: Apply confirmed SPLITs first so impure pieces detach before merges
    for dec in decisions:
        q = qmap[dec.query_id]
        if dec.confidence < conf_threshold:
            continue
        if dec.query_type == "seam" and dec.action == "SPLIT" and q.seam_sub2:
            labels[q.seam_sub2] = next_cid
            next_cid += 1

    # Pass 2: Apply confirmed MERGEs via union-find on remaining cluster IDs
    parent: Dict[int, int] = {int(c): int(c) for c in np.unique(labels).tolist()}

    def _find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for dec in decisions:
        q = qmap[dec.query_id]
        if dec.confidence < conf_threshold:
            continue
        if dec.query_type == "merge_pair" and dec.action == "MERGE":
            # Use majority current label of c1/c2 medoids in case a split ran on part of c1/c2
            c1_cur = int(labels[q.merge_c1_medoids[0]])
            c2_cur = int(labels[q.merge_c2_medoids[0]])
            r1, r2 = _find(c1_cur), _find(c2_cur)
            if r1 != r2:
                parent[r2] = r1

    for i in range(n):
        labels[i] = _find(int(labels[i]))

    # Pass 3: Apply single-node / pair direct edits
    for dec in decisions:
        q = qmap[dec.query_id]
        if dec.confidence < conf_threshold:
            continue
        if dec.query_type == "pair":
            u, v = q.primary_nodes[0], q.primary_nodes[1]
            if dec.action == "MUST_LINK" and labels[u] != labels[v]:
                labels[v] = labels[u]
            elif dec.action == "CANNOT_LINK" and labels[u] == labels[v]:
                labels[v] = next_cid
                next_cid += 1
        elif dec.query_type not in {"merge_pair", "seam"}:
            targets = q.primary_nodes
            if dec.action == "ASSIGN" and dec.target_cluster is not None:
                tgt = _find(int(dec.target_cluster))
                labels[targets] = tgt
            elif dec.action == "NEW_EVENT":
                labels[targets] = next_cid
                next_cid += 1

    _, clean = np.unique(labels, return_inverse=True)
    return clean.astype(np.int64), queried_mask


def apply_cequel_transitive_assignment(
    labels_init: np.ndarray,
    queries: List[QueryItem],
    decisions: List[LLMDecision],
    conf_threshold: float = 0.6,
    summaries: Optional[Dict[int, ClusterSummary]] = None,
    mask_mode: str = "all_shown",
) -> Tuple[np.ndarray, np.ndarray]:
    """Transitive Cequel-Edge baseline: cluster-level MUST_LINK closure guarded by CANNOT_LINK."""
    n = len(labels_init)
    labels = np.asarray(labels_init, dtype=np.int64).copy()
    queried_mask = _resolve_queried_mask(queries, decisions, n, summaries, mask_mode)
    if n == 0:
        return labels, queried_mask

    qmap = {q.query_id: q for q in queries}
    parent: Dict[int, int] = {int(c): int(c) for c in np.unique(labels).tolist()}

    def _find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Gather hard CANNOT_LINK node pairs first to guard against contradictory transitive merges
    cannot_pairs: List[Tuple[int, int]] = []
    must_pairs: List[Tuple[int, int]] = []
    for dec in decisions:
        if dec.confidence < conf_threshold or dec.query_type != "pair":
            continue
        q = qmap[dec.query_id]
        u, v = int(q.primary_nodes[0]), int(q.primary_nodes[1])
        if dec.action == "CANNOT_LINK":
            cannot_pairs.append((u, v))
        elif dec.action == "MUST_LINK":
            must_pairs.append((u, v))

    fallback_moves: List[Tuple[int, int]] = []
    for u, v in must_pairs:
        r1 = _find(int(labels_init[u]))
        r2 = _find(int(labels_init[v]))
        if r1 == r2:
            continue
        contradicted = False
        for cu, cv in cannot_pairs:
            rc1 = _find(int(labels_init[cu]))
            rc2 = _find(int(labels_init[cv]))
            if (rc1 == r1 and rc2 == r2) or (rc1 == r2 and rc2 == r1):
                contradicted = True
                break
        if contradicted:
            fallback_moves.append((u, v))
        else:
            parent[r2] = r1

    for i in range(n):
        labels[i] = _find(int(labels[i]))

    for u, v in fallback_moves:
        labels[v] = labels[u]

    next_cid = int(labels.max()) + 1
    for u, v in cannot_pairs:
        if labels[u] == labels[v]:
            labels[v] = next_cid
            next_cid += 1

    _, clean = np.unique(labels, return_inverse=True)
    return clean.astype(np.int64), queried_mask


def apply_topology_repair_and_recluster(
    graph: MessageGraph,
    labels_init: np.ndarray,
    summaries: Dict[int, ClusterSummary],
    queries: List[QueryItem],
    decisions: List[LLMDecision],
    conf_threshold: float = 0.6,
    bridge_weight: float = 1.0,
    anchor_weight: float = 1.2,
    refine_moves: bool = True,
    mask_mode: str = "all_shown",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Branch 2: Distributed Graph Topology Surgery + Constrained Warm-Start 2D SE.

    Fixes v2.1 medoid-ripping artifact:
      1. Uses distributed moderate cross-boundary semantic bridges (not 50x weights
         on 6 medoids) for confirmed MERGEs, and complete edge pruning (Proposition 1)
         for confirmed SEPARATE / SPLIT decisions.
      2. Warm-starts exact 2D SE agglomerative merging + node-move refinement from
         the constraint-respecting partition (dissolving reassigned boundary/seed
         nodes so they & their unqueried neighbors can re-attach freely, while allowing
         split pieces to re-merge with external true siblings and merged clusters to
         pull in 2nd-order unqueried fragments!).
    """
    n = len(labels_init)
    adj_csr = graph.adj.tocsr()
    emb = graph.embeddings
    adj_lil = adj_csr.tolil(copy=True)
    queried_mask = _resolve_queried_mask(queries, decisions, n, summaries, mask_mode)
    qmap = {q.query_id: q for q in queries}

    # Start warm partition from labels_init
    warm_labels = np.asarray(labels_init, dtype=np.int64).copy()
    next_cid = int(warm_labels.max()) + 1 if n > 0 else 0

    edges_pruned = 0
    edges_added = 0
    new_event_cues: Dict[str, List[int]] = {}
    dissolve_nodes = set()

    # Pass 1: Execute SPLITs (Proposition 1 complete cut)
    for dec in decisions:
        q = qmap[dec.query_id]
        if dec.confidence < conf_threshold:
            continue
        if dec.query_type == "seam" and dec.action == "SPLIT":
            for u, v in q.seam_cut_edges:
                if adj_lil[u, v] > 0.0:
                    adj_lil[u, v] = 0.0
                    adj_lil[v, u] = 0.0
                    edges_pruned += 1
            if q.seam_sub2:
                warm_labels[q.seam_sub2] = next_cid
                next_cid += 1

    # Pass 2: Execute MERGE / SEPARATE on cluster pairs
    parent: Dict[int, int] = {int(c): int(c) for c in np.unique(warm_labels).tolist()}

    def _find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for dec in decisions:
        q = qmap[dec.query_id]
        if dec.confidence < conf_threshold:
            continue
        c = float(dec.confidence)
        if dec.query_type == "merge_pair":
            c1, c2 = int(q.merge_c1), int(q.merge_c2)
            m1 = np.asarray(summaries[c1].members, dtype=np.int64)
            m2 = np.asarray(summaries[c2].members, dtype=np.int64)
            if dec.action == "MERGE":
                # Scale-invariant distributed semantic cross-boundary bridges:
                # Scales edge count & degree-matched weight with cluster size so 400+ node
                # clusters (e.g. Event2018) stay coherently merged during 2D SE refinement!
                min_sz = min(len(m1), len(m2))
                r = min(max(15, int(0.35 * min_sz)), 160)
                k1 = min(max(30, int(0.5 * len(m1))), 120, len(m1))
                k2 = min(max(30, int(0.5 * len(m2))), 120, len(m2))
                cand1 = m1[np.argsort(-(emb[m1] @ summaries[c1].prototype))[:k1]]
                cand2 = m2[np.argsort(-(emb[m2] @ summaries[c2].prototype))[:k2]]
                sims = emb[cand1] @ emb[cand2].T
                flat_top = np.argsort(-sims.ravel())[:r]
                deg_vec = np.asarray(adj_csr.sum(axis=1)).ravel()
                for flat_idx in flat_top.tolist():
                    i1, i2 = divmod(flat_idx, len(cand2))
                    u, v = int(cand1[i1]), int(cand2[i2])
                    if u != v:
                        deg_scale = max(bridge_weight, 0.20 * float(np.sqrt(max(1.0, deg_vec[u] * deg_vec[v]))))
                        w_new = max(float(adj_lil[u, v]), deg_scale * c)
                        adj_lil[u, v] = w_new
                        adj_lil[v, u] = w_new
                        edges_added += 1
                # Union in warm partition so 2D SE starts merged & can pull in 2nd-order siblings
                r1, r2 = _find(int(warm_labels[q.merge_c1_medoids[0]])), _find(int(warm_labels[q.merge_c2_medoids[0]]))
                if r1 != r2:
                    parent[r2] = r1
            elif dec.action == "SEPARATE":
                set2 = set(summaries[c2].members)
                for u in summaries[c1].members:
                    for nb in list(adj_lil.rows[u]):
                        if nb in set2:
                            adj_lil[u, nb] = 0.0
                            adj_lil[nb, u] = 0.0
                            edges_pruned += 1

    for i in range(n):
        warm_labels[i] = _find(int(warm_labels[i]))

    # Pass 3: Execute Node / Bridge / Seed / Pair edits with STRICT NO-OP GUARD
    for dec in decisions:
        q = qmap[dec.query_id]
        if dec.confidence < conf_threshold:
            continue
        c = float(dec.confidence)

        if dec.query_type == "pair":
            u, v = q.primary_nodes[0], q.primary_nodes[1]
            same_init = int(warm_labels[u]) == int(warm_labels[v])
            if dec.action == "CANNOT_LINK" and same_init:
                if adj_lil[u, v] > 0.0:
                    adj_lil[u, v] = 0.0
                    adj_lil[v, u] = 0.0
                    edges_pruned += 1
                dissolve_nodes.update([u, v])
            elif dec.action == "MUST_LINK" and not same_init:
                w_new = max(float(adj_lil[u, v]), anchor_weight * c)
                adj_lil[u, v] = w_new
                adj_lil[v, u] = w_new
                edges_added += 1
                dissolve_nodes.update([u, v])
        elif dec.query_type not in {"merge_pair", "seam"}:
            group = q.affected_group if q.affected_group else q.primary_nodes
            primary = q.primary_nodes[0]
            cur_cid = int(labels_init[primary])

            if dec.action == "ASSIGN" and dec.target_cluster is not None:
                tgt_cid = int(dec.target_cluster)
                if tgt_cid == cur_cid:
                    continue
                tgt_medoids = summaries[tgt_cid].medoids if tgt_cid in summaries else []
                tgt_warm_cid = _find(tgt_cid)
                for u in group:
                    for nb in list(adj_lil.rows[u]):
                        if int(labels_init[nb]) == cur_cid and nb not in group:
                            adj_lil[u, nb] = 0.0
                            adj_lil[nb, u] = 0.0
                            edges_pruned += 1
                    for med in tgt_medoids:
                        if med != u:
                            w_new = max(float(adj_lil[u, med]), anchor_weight * c)
                            adj_lil[u, med] = w_new
                            adj_lil[med, u] = w_new
                            edges_added += 1
                    warm_labels[u] = tgt_warm_cid
            elif dec.action == "NEW_EVENT":
                for u in group:
                    for nb in list(adj_lil.rows[u]):
                        if nb not in group and summaries[int(labels_init[nb])].size > 5:
                            adj_lil[u, nb] = 0.0
                            adj_lil[nb, u] = 0.0
                            edges_pruned += 1
                    warm_labels[u] = next_cid
                next_cid += 1
                if dec.event_cue:
                    new_event_cues.setdefault(dec.event_cue, []).extend(group)

    # Link NEW_EVENT seeds sharing matching cues
    for _cue, nodes in new_event_cues.items():
        uniq = list(dict.fromkeys(nodes))
        if len(uniq) >= 2:
            anchor = uniq[0]
            anchor_c = int(warm_labels[anchor])
            for other in uniq[1:]:
                w_new = max(float(adj_lil[anchor, other]), anchor_weight)
                adj_lil[anchor, other] = w_new
                adj_lil[other, anchor] = w_new
                warm_labels[other] = anchor_c
                edges_added += 1

    for u in dissolve_nodes:
        warm_labels[u] = next_cid
        next_cid += 1

    repaired_adj = adj_lil.tocsr()
    repaired_adj.setdiag(0.0)
    repaired_adj.eliminate_zeros()

    # Constrained Warm-Start 2D SE:
    # 1) Agglomerative merging from warm_labels -> allows split pieces to re-merge into external true
    #    clusters AND merged clusters to pull in 2nd-order unqueried fragments!
    # 2) Boundary-locked node-move 2D SE refinement -> only nodes touched by SPLIT / ASSIGN / NEW_EVENT /
    #    pair constraints and their 1-hop graph neighbors may move, preventing 2D-SE volume repulsion from
    #    bleeding interior nodes out of giant LLM-confirmed MERGE clusters!
    movable_mask = np.zeros(n, dtype=bool)
    for dec in decisions:
        q = qmap[dec.query_id]
        if dec.confidence < conf_threshold:
            continue
        if dec.query_type == "seam" and dec.action == "SPLIT":
            for u in q.seam_sub2:
                movable_mask[u] = True
            for u, v in q.seam_cut_edges:
                movable_mask[u] = True
                movable_mask[v] = True
        elif dec.query_type == "pair" and dec.action in {"MUST_LINK", "CANNOT_LINK"}:
            for u in q.primary_nodes:
                movable_mask[u] = True
        elif dec.query_type not in {"merge_pair", "seam"}:
            primary = q.primary_nodes[0]
            if dec.action == "NEW_EVENT" or (
                dec.action == "ASSIGN" and dec.target_cluster is not None and int(dec.target_cluster) != int(labels_init[primary])
            ):
                for u in (q.affected_group or q.primary_nodes):
                    movable_mask[u] = True

    # Expand movable_mask by 1-hop neighborhood on repaired graph (excluding interior of confirmed MERGEs)
    merged_core_mask = np.zeros(n, dtype=bool)
    for dec in decisions:
        q = qmap[dec.query_id]
        if dec.confidence >= conf_threshold and dec.query_type == "merge_pair" and dec.action == "MERGE":
            for u in summaries[int(q.merge_c1)].members:
                merged_core_mask[u] = True
            for u in summaries[int(q.merge_c2)].members:
                merged_core_mask[u] = True

    if np.any(movable_mask):
        seed_idx = np.where(movable_mask)[0]
        for u in seed_idx.tolist():
            for nb in repaired_adj.indices[repaired_adj.indptr[u] : repaired_adj.indptr[u + 1]].tolist():
                if not merged_core_mask[nb]:
                    movable_mask[nb] = True
        # Ensure SPLIT pockets remain movable even if their parent cluster was also merged
        for dec in decisions:
            q = qmap[dec.query_id]
            if dec.confidence >= conf_threshold and dec.query_type == "seam" and dec.action == "SPLIT":
                for u in q.seam_sub2:
                    movable_mask[u] = True

    new_labels, _ = minimize_2d_se_agglomerative(repaired_adj, initial_labels=warm_labels)
    if refine_moves and np.any(movable_mask):
        new_labels, _ = refine_2d_se_node_moves(
            repaired_adj, new_labels, max_iters=10, movable_mask=movable_mask
        )

    surgery_stats = {
        "edges_pruned": edges_pruned,
        "edges_added": edges_added,
        "nnz_before": int(adj_csr.nnz // 2),
        "nnz_after": int(repaired_adj.nnz // 2),
    }
    return new_labels, queried_mask, surgery_stats
