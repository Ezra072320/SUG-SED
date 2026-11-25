"""Core SUG-SED Framework (Pure-Star-Meso).

Empirically calibrated two-layer architecture verified across 27 held-out blocks:
  Core Universal Engines (cross-lingual statistically significant gains):
    1. Dynamic Contrastive Star-Merge (k_nb=4, two-tier [0.55, 0.85) -> [0.45, 0.85)):
       evaluates an anchor component against up to k_nb=4 distinct current-component
       neighbors per LLM call, halving call count and preventing single-post dilution.
    2. Clean Seam Split (detect_clean_seams, min_sem_sep=0.12): spherical k-means +
       Fiedler pocket proposal with balanced subgroup comparison before merging.
  Efficiency & Tail-Risk Guardrails (calibrated trade-offs):
    3. Purity-First 2D-SE Base Partition (tau_sim=0.50, alpha=0.5): isolates events
       into high-purity fragments, shifting hard False Merges into recoverable False Splits.
    4. Zero-Token AutoMerge (tau_auto=0.85): cosine-calibrated load reducer dependent on
       encoder high-end isotropy.
    5. Centroid-Cohesive Guarded Union-Find (guard_sim=0.40, min_cl_votes=1): enforces
       zero CANNOT-LINK violations and a centroid cohesion floor to block transitive
       snowball collapses.
"""

from __future__ import annotations

import math
import threading
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_MATH_LOCK = threading.Lock()

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import MessageGraph, _ensure_l2_normalized, build_message_graph
from core_shared.partition_cache import load_or_solve_2d_se
from trackA_sug_sed.llm_client import (
    BudgetedLLMClient,
    build_seam_split_prompt,
    build_star_merge_prompt,
)


def _parse_bool(val: Any, default: bool = False) -> bool:
    """Parse boolean JSON values safely without treating 'false' or '0' as True."""
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n", "null", "none", ""}:
            return False
    return bool(val)


@dataclass
class ClusterSummary:
    cluster_id: int
    members: List[int]
    size: int
    prototype: np.ndarray            # L2-normalized centroid
    medoids: List[int]               # Up to 3 closest-to-centroid posts
    top_keywords: List[str]          # Up to 5 entities/hashtags/lemmas
    mean_timestamp: float


def _compute_cluster_idf(
    ds: SocialDataset,
    comm_to_members: Dict[int, List[int]],
) -> Dict[str, float]:
    """Compute cluster-level IDF across all clusters in a block so generic words do not drown discriminative cues."""
    n_clusters = max(1, len(comm_to_members))
    cluster_df: Dict[str, int] = defaultdict(int)
    for members in comm_to_members.values():
        seen_terms: Set[str] = set()
        for m in members:
            for e in ds.entities[m]:
                seen_terms.add(e)
            for h in ds.hashtags[m]:
                seen_terms.add(f"#{h}")
            for tok in ds.tokens_lemma[m]:
                if len(tok) >= 3:
                    seen_terms.add(tok)
        for t in seen_terms:
            cluster_df[t] += 1
    return {
        t: math.log((1.0 + n_clusters) / (1.0 + df)) + 1.0
        for t, df in cluster_df.items()
    }


def _summarize_members(
    cid: int,
    members: List[int],
    ds: SocialDataset,
    embeddings: np.ndarray,
    term_idf: Optional[Dict[str, float]] = None,
    max_medoids: int = 3,
    max_keywords: int = 5,
) -> ClusterSummary:
    """Build or refresh a ClusterSummary from a member index list using TF-IDF keywords."""
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

    if term_idf is not None:
        scored_kw = sorted(
            kw_counts.items(),
            key=lambda kv: (-(1.0 + math.log(max(1, kv[1]))) * term_idf.get(kv[0], 1.0), -kv[1], kv[0]),
        )
    else:
        scored_kw = sorted(kw_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_kw = [k for k, _ in scored_kw[:max_keywords]]

    ts_vals = ds.timestamps[idx]
    valid_ts = ts_vals[~np.isnan(ts_vals)]
    mean_ts = float(np.mean(valid_ts)) if len(valid_ts) > 0 else float("nan")

    return ClusterSummary(
        cluster_id=cid,
        members=list(members),
        size=len(members),
        prototype=proto.astype(np.float32),
        medoids=medoids,
        top_keywords=top_kw,
        mean_timestamp=mean_ts,
    )


def build_cluster_summaries(
    ds: SocialDataset,
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_medoids: int = 3,
    max_keywords: int = 5,
) -> Dict[int, ClusterSummary]:
    """Summarize each cluster by prototype, medoids, and IDF-weighted top keywords."""
    comm_to_members: Dict[int, List[int]] = defaultdict(list)
    for i, c in enumerate(np.asarray(labels).tolist()):
        comm_to_members[int(c)].append(i)

    term_idf = _compute_cluster_idf(ds, comm_to_members)
    summaries: Dict[int, ClusterSummary] = {}
    for cid, members in comm_to_members.items():
        summaries[cid] = _summarize_members(
            cid,
            members,
            ds,
            embeddings,
            term_idf=term_idf,
            max_medoids=max_medoids,
            max_keywords=max_keywords,
        )
    return summaries


class GuardedUnionFind:
    """Constraint-consistent, centroid-cohesion-bounded Union-Find over cluster IDs."""

    def __init__(
        self,
        cids: Sequence[int],
        sim_mat: np.ndarray,
        guard_sim: float = 0.40,
        enforce_guards: bool = True,
        protos: Optional[np.ndarray] = None,
        sizes: Optional[Sequence[int]] = None,
        min_cl_votes: int = 1,
    ) -> None:
        self.cids = list(cids)
        self.cid_to_pos = {c: i for i, c in enumerate(self.cids)}
        self.sim_mat = sim_mat
        self.guard_sim = float(guard_sim)
        self.enforce_guards = bool(enforce_guards)
        self.min_cl_votes = max(1, int(min_cl_votes))

        self.comp_of: Dict[int, int] = {c: c for c in self.cids}
        self.comp_members: Dict[int, Set[int]] = {c: {c} for c in self.cids}
        self.use_centroid = protos is not None
        if protos is not None:
            if len(protos) != len(self.cids):
                raise ValueError(f"protos length {len(protos)} != cids length {len(self.cids)}")
            if sizes is not None and len(sizes) != len(self.cids):
                raise ValueError(f"sizes length {len(sizes)} != cids length {len(self.cids)}")
            sz_arr = list(sizes) if sizes is not None else [1] * len(self.cids)
            self.comp_vec: Dict[int, np.ndarray] = {
                c: np.asarray(protos[i], dtype=np.float64) * max(1, int(sz_arr[i]))
                for i, c in enumerate(self.cids)
            }
            self.comp_size: Dict[int, int] = {c: max(1, int(sz_arr[i])) for i, c in enumerate(self.cids)}
        else:
            self.comp_vec = {}
            self.comp_size = {}
        self.cannot_pairs: Set[Tuple[int, int]] = set()
        self.comp_cl_counts: Dict[int, Dict[int, int]] = {c: {} for c in self.cids}
        self.blocked_count: int = 0

    @staticmethod
    def _ekey(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def root(self, c: int) -> int:
        return self.comp_of[c]

    def _comp_proto(self, r: int) -> np.ndarray:
        v = self.comp_vec[r]
        norm = float(np.linalg.norm(v))
        return (v / norm) if norm > 1e-12 else v

    def can_merge(self, c1: int, c2: int) -> bool:
        r1, r2 = self.comp_of[c1], self.comp_of[c2]
        if r1 == r2:
            return False
        if not self.enforce_guards:
            return True
        if self.comp_cl_counts[r1].get(r2, 0) >= self.min_cl_votes:
            return False
        if self.guard_sim > 0.0:
            if self.use_centroid:
                if float(np.dot(self._comp_proto(r1), self._comp_proto(r2))) < self.guard_sim:
                    return False
            else:
                sims = [
                    float(self.sim_mat[self.cid_to_pos[a], self.cid_to_pos[b]])
                    for a in self.comp_members[r1]
                    for b in self.comp_members[r2]
                ]
                if sims and (sum(sims) / len(sims)) < self.guard_sim:
                    return False
        return True

    def add_cannot_link(self, c1: int, c2: int) -> None:
        ek = self._ekey(c1, c2)
        if ek in self.cannot_pairs:
            return
        self.cannot_pairs.add(ek)
        r1, r2 = self.comp_of[c1], self.comp_of[c2]
        if r1 != r2:
            self.comp_cl_counts[r1][r2] = self.comp_cl_counts[r1].get(r2, 0) + 1
            self.comp_cl_counts[r2][r1] = self.comp_cl_counts[r2].get(r1, 0) + 1

    def merge(self, c1: int, c2: int) -> bool:
        r1, r2 = self.comp_of[c1], self.comp_of[c2]
        if r1 == r2:
            return False
        if self.enforce_guards and not self.can_merge(c1, c2):
            self.blocked_count += 1
            return False
        # Small-to-large root update
        if len(self.comp_members[r1]) < len(self.comp_members[r2]):
            r1, r2 = r2, r1
        for m in self.comp_members[r2]:
            self.comp_of[m] = r1
        self.comp_members[r1].update(self.comp_members[r2])
        del self.comp_members[r2]

        cl_r2 = self.comp_cl_counts.pop(r2, {})
        cl_r2.pop(r1, None)
        self.comp_cl_counts[r1].pop(r2, None)
        for nbr_r, cnt in cl_r2.items():
            self.comp_cl_counts[r1][nbr_r] = self.comp_cl_counts[r1].get(nbr_r, 0) + cnt
            if nbr_r in self.comp_cl_counts:
                self.comp_cl_counts[nbr_r].pop(r2, None)
                self.comp_cl_counts[nbr_r][r1] = self.comp_cl_counts[nbr_r].get(r1, 0) + cnt

        if self.use_centroid:
            self.comp_vec[r1] = self.comp_vec[r1] + self.comp_vec[r2]
            self.comp_size[r1] += self.comp_size[r2]
            del self.comp_vec[r2]
            del self.comp_size[r2]
        return True

    def apply_to_labels(self, labels: np.ndarray) -> np.ndarray:
        out = np.asarray(labels, dtype=np.int64).copy()
        for i, c in enumerate(out.tolist()):
            if c in self.comp_of:
                out[i] = self.comp_of[c]
        _, clean = np.unique(out, return_inverse=True)
        return clean.astype(np.int64)


# ---------------------------------------------------------------------------
# Deterministic Spherical k-Means + Spectral Fallback Seam Detector
# ---------------------------------------------------------------------------

def _spherical_kmeans_deterministic(x: np.ndarray, k: int, n_iters: int = 15) -> np.ndarray:
    n = x.shape[0]
    if k <= 1 or n <= k:
        return np.arange(n, dtype=np.int64) % max(1, k)
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    first = int(np.argmax(np.sum(x @ x.T, axis=1)))
    centers[0] = x[first]
    min_dist = 1.0 - (x @ centers[0])
    for c_idx in range(1, k):
        probs = np.maximum(min_dist, 0.0) ** 2
        pick = int(np.argmax(probs)) if float(probs.sum()) > 1e-12 else (c_idx % n)
        centers[c_idx] = x[pick]
        min_dist = np.minimum(min_dist, 1.0 - (x @ centers[c_idx]))

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iters):
        new_labels = np.argmax(x @ centers.T, axis=1).astype(np.int64)
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


def _fiedler_bipartition(sub_adj: sparse.csr_matrix, sub_emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = sub_adj.shape[0]
    if m < 4:
        return np.arange(m // 2, dtype=np.int64), np.arange(m // 2, m, dtype=np.int64)
    n_comp, comp_labels = sparse.csgraph.connected_components(sub_adj, directed=False)
    if n_comp >= 2:
        counts = np.bincount(comp_labels)
        main_c = int(np.argmax(counts))
        return np.where(comp_labels == main_c)[0], np.where(comp_labels != main_c)[0]

    deg = np.asarray(sub_adj.sum(axis=1)).ravel()
    total_vol = float(deg.sum())
    if float(deg.min()) > 1e-12 and total_vol > 1e-12:
        inv_sqrt = 1.0 / np.sqrt(deg)
        d_mat = sparse.diags(inv_sqrt)
        norm_adj = d_mat @ sub_adj @ d_mat
        try:
            with _MATH_LOCK:
                if m <= 80:
                    _, eigvecs = np.linalg.eigh(norm_adj.toarray())
                    fiedler = eigvecs[:, -2] * inv_sqrt
                else:
                    eigvals, eigvecs = eigsh(norm_adj, k=2, which="LA", tol=1e-4, maxiter=500)
                    fiedler = eigvecs[:, int(np.argsort(eigvals)[0])] * inv_sqrt
            order = np.argsort(fiedler)
            min_side = max(2, min(4, m // 4))
            candidates = np.unique(
                np.linspace(min_side, m - min_side, num=min(30, max(1, m - 2 * min_side + 1)), dtype=int)
            )
            sub_coo = sub_adj.tocoo()
            rank = np.empty(m, dtype=np.int64)
            rank[order] = np.arange(m)
            r_u, r_v, w_uv = rank[sub_coo.row], rank[sub_coo.col], sub_coo.data
            pref_vol = np.cumsum(deg[order])
            best_cond, best_cut = float("inf"), m // 2
            for cp in candidates:
                cut_w = float(w_uv[(r_u < cp) & (r_v >= cp)].sum())
                v1 = float(pref_vol[cp - 1])
                denom = min(v1, total_vol - v1)
                cond = cut_w / denom if denom > 1e-12 else float("inf")
                if cond < best_cond:
                    best_cond = cond
                    best_cut = int(cp)
            g1, g2 = order[:best_cut], order[best_cut:]
            if len(g1) > 0 and len(g2) > 0:
                return g1, g2
        except Exception:
            pass

    centered = sub_emb - sub_emb.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[0]
    med = float(np.median(proj))
    g1, g2 = np.where(proj <= med)[0], np.where(proj > med)[0]
    if len(g1) == 0 or len(g2) == 0:
        return np.arange(m // 2), np.arange(m // 2, m)
    return g1, g2


@dataclass
class SeamCandidate:
    query_id: str
    cluster_id: int
    sub1: List[int]
    sub2: List[int]
    sub1_medoids: List[int]
    sub2_medoids: List[int]
    score: float
    cross_sim: float
    sem_sep: float


def detect_clean_seams(
    graph: MessageGraph,
    summaries: Dict[int, ClusterSummary],
    min_cluster_size: int = 8,
    min_pocket_size: int = 3,
    max_cross_sim: float = 1.0,
    min_sem_sep: float = 0.12,
    use_seam_gate: bool = False,
    random_order: bool = False,
    seed: int = 42,
) -> List[SeamCandidate]:
    """Detect minority false-merge pockets without conductance cap or non-entity TF-IDF."""
    adj = graph.adj
    emb = _ensure_l2_normalized(graph.embeddings)
    out: List[SeamCandidate] = []

    for cid, summ in sorted(summaries.items()):
        if summ.size < min_cluster_size:
            continue
        members = np.asarray(summ.members, dtype=np.int64)
        sub_adj = adj[members][:, members].tocsr()
        sub_emb = emb[members]

        k_sub = min(6, max(3, summ.size // 12))
        k_values = [2, k_sub] if summ.size < 18 else [k_sub]

        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for kv in k_values:
            klab = _spherical_kmeans_deterministic(sub_emb, k=kv, n_iters=15)
            for p in range(kv):
                p_loc = np.where(klab == p)[0]
                r_loc = np.where(klab != p)[0]
                if len(p_loc) >= min_pocket_size and len(p_loc) <= len(r_loc) and len(r_loc) >= min_pocket_size:
                    splits.append((r_loc, p_loc))

        fg1, fg2 = _fiedler_bipartition(sub_adj, sub_emb)
        if len(fg1) < len(fg2):
            fg1, fg2 = fg2, fg1
        if len(fg2) >= min_pocket_size:
            splits.append((fg1, fg2))

        best: Optional[SeamCandidate] = None
        for m_loc, p_loc in splits:
            g_main, g_pock = members[m_loc], members[p_loc]
            p_main = emb[g_main].mean(axis=0)
            p_pock = emb[g_pock].mean(axis=0)
            p_main = p_main / max(float(np.linalg.norm(p_main)), 1e-12)
            p_pock = p_pock / max(float(np.linalg.norm(p_pock)), 1e-12)

            intra_pock = float(np.mean(emb[g_pock] @ p_pock))
            cross_sim = float(np.dot(p_main, p_pock))
            # Guard against same-event exact-retweet bursts with moderate cross_sim hijacking Seam budget
            if len(g_pock) < 6 and intra_pock > 0.96 and cross_sim > 0.65:
                continue
            sem_sep = max(0.0, intra_pock - cross_sim)
            if sem_sep < min_sem_sep:
                continue
            if use_seam_gate and cross_sim > max_cross_sim:
                continue

            score = sem_sep * math.log1p(len(g_pock))
            if best is None or score > best.score:
                m1 = [int(g_main[k]) for k in np.argsort(-(emb[g_main] @ p_main))[:3]]
                m2 = [int(g_pock[k]) for k in np.argsort(-(emb[g_pock] @ p_pock))[:3]]
                best = SeamCandidate(
                    query_id=f"seam_c{cid}",
                    cluster_id=cid,
                    sub1=g_main.tolist(),
                    sub2=g_pock.tolist(),
                    sub1_medoids=m1,
                    sub2_medoids=m2,
                    score=score,
                    cross_sim=cross_sim,
                    sem_sep=sem_sep,
                )
        if best is not None:
            out.append(best)

    if random_order:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(out))
        out = [out[int(idx)] for idx in perm]
    else:
        out.sort(key=lambda s: -s.score)
    return out


# ---------------------------------------------------------------------------
# Full Pure-Star-Meso Block Runner (with Clean Pillar Ablation Toggles)
# ---------------------------------------------------------------------------

def run_pure_star_meso(
    dataset_name: str,
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    client: Optional[BudgetedLLMClient],
    bucket: Optional[Any] = None,
    strategy_tag: str = "pure_star_meso",
    *,
    # Pillar toggles for Dev & Held-out ablations
    use_purity_base: bool = True,
    use_auto_merge: bool = True,
    use_bayes_band: bool = True,
    use_star_packing: bool = True,
    use_guarded_uf: bool = True,
    use_seam_gate: bool = False,
    random_order: bool = False,
    seed: int = 42,
    # Locked Dev hyperparameters
    tau_auto: float = 0.85,
    tau_lo: float = 0.55,
    guard_sim: float = 0.40,
    k_nb: int = 4,
    max_star_calls: int = 24,
    max_seam_calls: int = 3,
    max_cross_sim: float = 1.0,
    min_sem_sep: float = 0.12,
    graph: Optional[MessageGraph] = None,
    labels_init: Optional[np.ndarray] = None,
    dry_run_oracle: bool = False,
    anonymizer: Optional[Any] = None,
    audit_store: Optional[Any] = None,
    pre_anonymized: bool = False,
    orig_ds_for_audit: Optional[SocialDataset] = None,
    embed_model: Optional[str] = None,
    max_seam_budget_frac: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run Pure-Star-Meso on one streaming block under causal budget & all_shown accounting.

    Returns:
        (final_labels, labels_init, all_shown_mask, stats)
    """
    t_cpu0 = time.perf_counter()
    n = len(ds)
    embeddings = _ensure_l2_normalized(embeddings)
    if anonymizer is not None or (pre_anonymized and orig_ds_for_audit is not None):
        from trackA_sug_sed.anonymizer import (
            QueryItem,
            audit_pre_anonymized_query,
            check_summary_reconstruction_leakage,
        )

    knn_tau = 0.50 if use_purity_base else 0.35
    if graph is not None and getattr(graph, "knn_sim_threshold", None) is not None:
        if abs(float(graph.knn_sim_threshold) - knn_tau) > 1e-6:
            warnings.warn(
                f"run_pure_star_meso received pre-built graph with knn_sim_threshold={graph.knn_sim_threshold}, "
                f"which overrides use_purity_base={use_purity_base} (expected {knn_tau}).",
                RuntimeWarning,
                stacklevel=2,
            )
    if graph is None:
        graph = build_message_graph(
            ds,
            embeddings,
            knn_k=15,
            knn_sim_threshold=knn_tau,
            attr_alpha=0.5,
            compute_non_entity_tfidf=False,
        )
    if labels_init is None:
        labels_init, _ = load_or_solve_2d_se(
            dataset_name=dataset_name,
            block_id=block_id,
            adj=graph.adj,
            knn_k=15,
            knn_sim_threshold=knn_tau,
            attr_alpha=0.5,
            refine_moves=True,
            embed_model=embed_model,
        )

    labels = np.asarray(labels_init, dtype=np.int64).copy()
    all_shown = np.zeros(n, dtype=bool)
    llm_api_sec = 0.0
    api_wall_sec = 0.0
    total_calls = seam_calls = seam_splits = star_calls = star_edges_checked = star_merges = 0
    prompt_tok = comp_tok = parse_errors = auto_merge_ops = blocked_guards = 0
    anon_leak_queries = anon_leak_terms = 0
    _anon_note = (
        "\nNote: Entities, hashtags, mentions, URLs, and years have been anonymized; "
        "decide strictly from event structure and placeholder identity correspondence."
    )

    # --- Step 1: Zero-token Auto-Merge (>= tau_auto) with Centroid-Cohesive GuardedUF ---
    eff_tau_auto = float(tau_auto) if use_auto_merge else 1.01
    if eff_tau_auto <= 1.0:
        summs0 = build_cluster_summaries(ds, embeddings, labels)
        cids0 = sorted([c for c, s in summs0.items() if s.size >= 2])
        if len(cids0) >= 2:
            protos0 = np.vstack([summs0[c].prototype for c in cids0])
            sizes0 = [summs0[c].size for c in cids0]
            sim0 = protos0 @ protos0.T
            uf0 = GuardedUnionFind(
                cids0, sim0, guard_sim=guard_sim, enforce_guards=use_guarded_uf, protos=protos0, sizes=sizes0
            )
            for c in cids0:
                uf0.comp_vec[c] = embeddings[summs0[c].members].sum(axis=0, dtype=np.float64)
            pairs0: List[Tuple[float, int, int]] = []
            for i in range(len(cids0)):
                row = sim0[i]
                for j in np.where(row >= eff_tau_auto)[0]:
                    if j > i:
                        pairs0.append((float(row[j]), cids0[i], cids0[int(j)]))
            pairs0.sort(key=lambda x: -x[0])
            for _s, c1, c2 in pairs0:
                r1, r2 = uf0.root(c1), uf0.root(c2)
                if r1 == r2:
                    continue
                if use_guarded_uf:
                    live_cos = float(np.dot(uf0._comp_proto(r1), uf0._comp_proto(r2)))
                    if live_cos < eff_tau_auto:
                        uf0.blocked_count += 1
                        continue
                if uf0.merge(c1, c2):
                    auto_merge_ops += 1
            blocked_guards += uf0.blocked_count
            labels = uf0.apply_to_labels(labels)

    # --- Step 2: Clean SPLIT_SEAM adjudication BEFORE Star-Merge ---
    seam_split_node_pairs: List[Tuple[int, int]] = []
    if max_seam_calls > 0:
        summs_pre_seam = build_cluster_summaries(ds, embeddings, labels)
        seams = detect_clean_seams(
            graph,
            summs_pre_seam,
            min_cluster_size=8,
            min_pocket_size=3,
            max_cross_sim=max_cross_sim,
            min_sem_sep=min_sem_sep,
            use_seam_gate=use_seam_gate,
            random_order=random_order,
            seed=seed,
        )
        next_cid = int(labels.max()) + 1 if n > 0 else 0
        seam_tokens_spent = 0
        seam_cap = (
            int(float(bucket.current_block_cap) * float(max_seam_budget_frac))
            if (bucket is not None and float(max_seam_budget_frac) > 0.0 and getattr(bucket, "current_block_cap", 0) > 0)
            else None
        )
        for sc in seams[:max_seam_calls]:
            seam_audit = None
            raw_seam_user = ""
            if anonymizer is not None and not dry_run_oracle:
                _, raw_seam_user = build_seam_split_prompt(
                    sc.sub1_medoids, sc.sub2_medoids, len(sc.sub1), len(sc.sub2), ds
                )
                q_item = QueryItem(
                    query_id=sc.query_id,
                    query_type="seam",
                    primary_nodes=list(sc.sub1_medoids[:3]) + list(sc.sub2_medoids[:3]),
                    candidate_clusters=[sc.cluster_id],
                    seam_cluster_id=sc.cluster_id,
                    seam_sub1_medoids=list(sc.sub1_medoids[:3]),
                    seam_sub2_medoids=list(sc.sub2_medoids[:3]),
                )
                anon_texts, _anon_s, seam_audit = anonymizer.build_query_anonymized_inputs(q_item, ds, summs_pre_seam)
                sys_msg, user_msg = build_seam_split_prompt(
                    sc.sub1_medoids, sc.sub2_medoids, len(sc.sub1), len(sc.sub2), ds, text_override=anon_texts
                )
                sys_msg = sys_msg + _anon_note
            else:
                sys_msg, user_msg = build_seam_split_prompt(
                    sc.sub1_medoids, sc.sub2_medoids, len(sc.sub1), len(sc.sub2), ds
                )
                if pre_anonymized and not dry_run_oracle:
                    sys_msg = sys_msg + _anon_note
                    if orig_ds_for_audit is not None:
                        seam_shown = list(sc.sub1_medoids[:3]) + list(sc.sub2_medoids[:3])
                        seam_audit = audit_pre_anonymized_query(
                            query_id=sc.query_id,
                            shown_medoids=seam_shown,
                            shown_keywords=[],
                            orig_ds=orig_ds_for_audit,
                            anon_ds=ds,
                        )
                        if audit_store is not None:
                            _, raw_seam_user = build_seam_split_prompt(
                                sc.sub1_medoids, sc.sub2_medoids, len(sc.sub1), len(sc.sub2), orig_ds_for_audit
                            )

            if dry_run_oracle:
                v1, c1 = np.unique(ds.labels[sc.sub1], return_counts=True)
                v2, c2 = np.unique(ds.labels[sc.sub2], return_counts=True)
                do_split = int(v1[int(np.argmax(c1))]) != int(v2[int(np.argmax(c2))])
                res_tuple = ({"different_events": do_split}, 0, 0, 0.0, "ORACLE")
            else:
                assert client is not None
                est_seam_tok = int((len(sys_msg) + len(user_msg)) / 3.4) + 85
                if seam_cap is not None and (seam_tokens_spent + est_seam_tok) > seam_cap:
                    break
                t_api0 = time.perf_counter()
                res_tuple = client.call_json(
                    request_id=sc.query_id,
                    block_id=block_id,
                    strategy=strategy_tag,
                    query_type="seam",
                    sys_msg=sys_msg,
                    user_msg=user_msg,
                    bucket=bucket,
                    est_completion_tokens=85,
                )
                api_wall_sec += time.perf_counter() - t_api0
                if res_tuple is None:
                    break

            if seam_audit is not None and audit_store is not None:
                audit_store.record(ds.name, sc.query_id, seam_audit, raw_seam_user, user_msg)

            parsed, pt, ct, lat, _st = res_tuple
            prompt_tok += pt
            comp_tok += ct
            seam_tokens_spent += pt + ct
            llm_api_sec += lat
            total_calls += 1
            seam_calls += 1
            for m in (sc.sub1_medoids[:3] + sc.sub2_medoids[:3]):
                all_shown[int(m)] = True
            if "error" in parsed:
                parse_errors += 1
                continue
            if seam_audit is not None:
                leaked = check_summary_reconstruction_leakage(
                    str(parsed.get("event_a", "")),
                    str(parsed.get("event_b", "")),
                    seam_audit.get("masked_originals", []),
                )
                if leaked:
                    anon_leak_queries += 1
                    anon_leak_terms += len(leaked)
            if _parse_bool(parsed.get("different_events", False)):
                labels[sc.sub2] = next_cid
                next_cid += 1
                seam_splits += 1
                if sc.sub1 and sc.sub2:
                    seam_split_node_pairs.append((int(sc.sub1[0]), int(sc.sub2[0])))

    # --- Step 3: Two-Tier Adaptive Contrastive Star-Merge with Online Component Refresh ---
    comm_to_members: Dict[int, List[int]] = defaultdict(list)
    for i_node, c_lab in enumerate(labels.tolist()):
        comm_to_members[int(c_lab)].append(i_node)
    term_idf = _compute_cluster_idf(ds, comm_to_members)
    summs = {
        cid: _summarize_members(cid, mems, ds, embeddings, term_idf=term_idf)
        for cid, mems in comm_to_members.items()
    }
    all_cids = sorted(summs.keys())
    uf: Optional[GuardedUnionFind] = None
    if len(all_cids) >= 2 and max_star_calls > 0:
        cid_to_pos = {c: i for i, c in enumerate(all_cids)}
        protos = np.vstack([summs[c].prototype for c in all_cids]).astype(np.float32)
        sizes = [summs[c].size for c in all_cids]
        sim_mat = protos @ protos.T
        uf = GuardedUnionFind(
            all_cids, sim_mat, guard_sim=guard_sim, enforce_guards=use_guarded_uf, protos=protos, sizes=sizes
        )
        for c in all_cids:
            uf.comp_vec[c] = embeddings[summs[c].members].sum(axis=0, dtype=np.float64)
        for n1, n2 in seam_split_node_pairs:
            ca, cb = int(labels[n1]), int(labels[n2])
            if ca != cb and ca in uf.comp_of and cb in uf.comp_of:
                uf.add_cannot_link(ca, cb)

        comp_summs: Dict[int, ClusterSummary] = dict(summs)
        active_roots: Set[int] = set(all_cids)
        eff_k_nb = int(k_nb) if use_star_packing else 1
        eff_max_star_calls = max_star_calls if use_star_packing else max(80, max_star_calls * 3)
        queried_root_nbrs: Dict[int, Set[int]] = {c: set() for c in all_cids}
        rng = np.random.default_rng(seed)

        def _components_queried(r1: int, r2: int) -> bool:
            return r2 in queried_root_nbrs[r1]

        if not use_bayes_band:
            tiers: List[Tuple[float, float, int]] = [(0.35, eff_tau_auto, 1)]
        elif random_order:
            tiers = [(0.45, eff_tau_auto, 1)]
        else:
            tiers = [(float(tau_lo), eff_tau_auto, 2), (0.45, eff_tau_auto, 1)]

        # Majority true labels per live root component (only used when dry_run_oracle=True)
        maj_true: Dict[int, int] = {}
        if dry_run_oracle:
            for c in all_cids:
                vs, cs = np.unique(ds.labels[comp_summs[c].members], return_counts=True)
                maj_true[c] = int(vs[int(np.argmax(cs))])

        step_idx = 0
        budget_exhausted = False
        for tier_lo, tier_hi, min_sz in tiers:
            if budget_exhausted or step_idx >= eff_max_star_calls:
                break
            while step_idx < eff_max_star_calls:
                live_roots = sorted(active_roots)
                if len(live_roots) < 2:
                    break
                live_pos = np.asarray([cid_to_pos[r] for r in live_roots], dtype=np.int64)
                live_sizes = np.asarray([comp_summs[r].size for r in live_roots], dtype=np.int64)

                candidates_for_step: List[Tuple[float, int, List[Tuple[float, int]]]] = []
                for idx1, r1 in enumerate(live_roots):
                    s1 = int(live_sizes[idx1])
                    if s1 < min_sz:
                        continue
                    row_sims = sim_mat[live_pos[idx1], live_pos]
                    mask = (row_sims >= tier_lo) & (row_sims < tier_hi) & (live_sizes >= min_sz)
                    mask[idx1] = False
                    js = np.where(mask)[0]
                    if len(js) == 0:
                        continue
                    js = js[np.argsort(-row_sims[js])]
                    pack: List[Tuple[float, int]] = []
                    for j in js:
                        r2 = live_roots[int(j)]
                        s2 = int(live_sizes[int(j)])
                        sim_val = float(row_sims[int(j)])
                        if s1 == 1 and s2 == 1 and sim_val < 0.55:
                            continue
                        if _components_queried(r1, r2):
                            continue
                        if not uf.can_merge(r1, r2):
                            continue
                        pack.append((sim_val, r2))
                        if len(pack) >= eff_k_nb:
                            break
                    if not pack:
                        continue
                    # Calibrated Bayes utility over LIVE component sizes & similarities
                    utility = sum(
                        (s**2) * (1.0 + 0.25 * math.log1p(min(s1, comp_summs[r2].size)))
                        for s, r2 in pack
                    )
                    candidates_for_step.append((utility, r1, pack))

                if not candidates_for_step:
                    break

                if random_order:
                    pick_pos = int(rng.integers(0, len(candidates_for_step)))
                    _u, anchor_r, pack = candidates_for_step[pick_pos]
                else:
                    candidates_for_step.sort(key=lambda x: -x[0])
                    _u, anchor_r, pack = candidates_for_step[0]

                cand_roots = [r2 for _s, r2 in pack]
                cand_summs = [comp_summs[r2] for r2 in cand_roots]
                anchor_summ = comp_summs[anchor_r]
                star_audit = None
                raw_star_user = ""
                qid_star = (
                    f"star_s{step_idx}_c{anchor_r}_n{len(cand_roots)}"
                    if use_star_packing
                    else f"pair_s{step_idx}_c{anchor_r}_c{cand_roots[0]}"
                )

                if dry_run_oracle:
                    matched_indices = [
                        idx for idx, r2 in enumerate(cand_roots, start=1) if maj_true[anchor_r] == maj_true[r2]
                    ]
                    res_tuple = (
                        {"matches": [{"id": idx, "shared_event": "oracle"} for idx in matched_indices]},
                        0,
                        0,
                        0.0,
                        "ORACLE",
                    )
                else:
                    # Both k=4 (Star Packing) and k=1 (w/o Star Packing) use identical prompt richness & rules
                    if anonymizer is not None:
                        _, raw_star_user = build_star_merge_prompt(anchor_summ, cand_summs, ds)
                        q_star = QueryItem(
                            query_id=qid_star,
                            query_type="star_merge",
                            primary_nodes=list(anchor_summ.medoids[:2])
                            + [m for cs in cand_summs for m in cs.medoids[:2]],
                            candidate_clusters=[anchor_r] + cand_roots,
                        )
                        anon_texts, anon_summs, star_audit = anonymizer.build_query_anonymized_inputs(
                            q_star, ds, comp_summs
                        )
                        sys_msg, user_msg = build_star_merge_prompt(
                            anon_summs[anchor_r],
                            [anon_summs[r2] for r2 in cand_roots],
                            ds,
                            text_override=anon_texts,
                        )
                        sys_msg = sys_msg + _anon_note
                    else:
                        sys_msg, user_msg = build_star_merge_prompt(anchor_summ, cand_summs, ds)
                        if pre_anonymized:
                            sys_msg = sys_msg + _anon_note
                            if orig_ds_for_audit is not None:
                                star_shown = list(anchor_summ.medoids[:2]) + [
                                    m for cs in cand_summs for m in cs.medoids[:2]
                                ]
                                shown_kws = list(anchor_summ.top_keywords[:5]) + [
                                    kw for cs in cand_summs for kw in cs.top_keywords[:5]
                                ]
                                star_audit = audit_pre_anonymized_query(
                                    query_id=qid_star,
                                    shown_medoids=star_shown,
                                    shown_keywords=shown_kws,
                                    orig_ds=orig_ds_for_audit,
                                    anon_ds=ds,
                                )
                                if audit_store is not None:
                                    _, raw_star_user = build_star_merge_prompt(
                                        anchor_summ, cand_summs, orig_ds_for_audit
                                    )
                    assert client is not None
                    t_api0 = time.perf_counter()
                    res_tuple = client.call_json(
                        request_id=qid_star,
                        block_id=block_id,
                        strategy=strategy_tag,
                        query_type="star_merge" if use_star_packing else "merge_pair",
                        sys_msg=sys_msg,
                        user_msg=user_msg,
                        bucket=bucket,
                        est_completion_tokens=105 if use_star_packing else 85,
                    )
                    api_wall_sec += time.perf_counter() - t_api0

                if res_tuple is None:
                    budget_exhausted = True
                    break

                if star_audit is not None and audit_store is not None:
                    audit_store.record(ds.name, qid_star, star_audit, raw_star_user, user_msg)

                step_idx += 1
                parsed, pt, ct, lat, _st = res_tuple
                prompt_tok += pt
                comp_tok += ct
                llm_api_sec += lat
                total_calls += 1
                star_calls += 1
                star_edges_checked += len(cand_roots)

                # Strict all_shown marking: anchor medoids + every shown candidate medoid!
                for m in anchor_summ.medoids[:2]:
                    all_shown[int(m)] = True
                for cs in cand_summs:
                    for m in cs.medoids[:2]:
                        all_shown[int(m)] = True

                for r2 in cand_roots:
                    queried_root_nbrs[anchor_r].add(r2)
                    queried_root_nbrs[r2].add(anchor_r)

                if "error" in parsed:
                    parse_errors += 1
                    continue
                if star_audit is not None:
                    shared_evs = " ".join(
                        str(it.get("shared_event", "")) for it in parsed.get("matches", []) if isinstance(it, dict)
                    ) if isinstance(parsed.get("matches", []), list) else ""
                    leaked = check_summary_reconstruction_leakage(
                        str(parsed.get("anchor_event", "")),
                        shared_evs,
                        star_audit.get("masked_originals", []),
                    )
                    if leaked:
                        anon_leak_queries += 1
                        anon_leak_terms += len(leaked)

                matched_pos: Set[int] = set()
                parse_ok = True
                if not use_star_packing and "matches" not in parsed and "same_event" in parsed:
                    if _parse_bool(parsed.get("same_event", False)):
                        matched_pos.add(1)
                else:
                    raw_matches = parsed.get("matches", None)
                    if not isinstance(raw_matches, list):
                        parse_ok = False
                    else:
                        for item in raw_matches:
                            mid: Optional[int] = None
                            if isinstance(item, int) and not isinstance(item, bool):
                                mid = item
                            elif isinstance(item, str) and item.strip().isdigit():
                                mid = int(item.strip())
                            elif isinstance(item, dict) and "id" in item:
                                raw_id = item["id"]
                                if isinstance(raw_id, int) and not isinstance(raw_id, bool):
                                    mid = raw_id
                                elif isinstance(raw_id, str) and raw_id.strip().isdigit():
                                    mid = int(raw_id.strip())
                            if mid is None or mid < 1 or mid > len(cand_roots):
                                parse_ok = False
                                break
                            matched_pos.add(mid)

                if not parse_ok:
                    parse_errors += 1
                    continue

                # Apply CANNOT-LINK first on unmatched candidates, then GUARDED MERGE with live component refresh
                for pos, r2 in enumerate(cand_roots, start=1):
                    if pos not in matched_pos:
                        uf.add_cannot_link(anchor_r, r2)

                cur_anchor = anchor_r
                for pos, r2 in enumerate(cand_roots, start=1):
                    if pos not in matched_pos:
                        continue
                    cur_anchor = uf.root(cur_anchor)
                    cur_r2 = uf.root(r2)
                    if cur_anchor == cur_r2:
                        continue
                    if uf.merge(cur_anchor, cur_r2):
                        star_merges += 1
                        new_r = uf.root(cur_anchor)
                        old_r = cur_r2 if new_r == cur_anchor else cur_anchor
                        merged_members = comp_summs[cur_anchor].members + comp_summs[cur_r2].members
                        new_summ = _summarize_members(new_r, merged_members, ds, embeddings, term_idf=term_idf)
                        comp_summs[new_r] = new_summ
                        comp_summs.pop(old_r, None)
                        active_roots.discard(old_r)

                        old_q = queried_root_nbrs.pop(old_r, set())
                        old_q.discard(new_r)
                        for qnbr in old_q:
                            if qnbr in queried_root_nbrs:
                                queried_root_nbrs[qnbr].discard(old_r)
                                queried_root_nbrs[qnbr].add(new_r)
                                queried_root_nbrs[new_r].add(qnbr)
                        queried_root_nbrs[new_r].discard(new_r)

                        uf.comp_vec[new_r] = embeddings[merged_members].sum(axis=0, dtype=np.float64)
                        uf.comp_size[new_r] = len(merged_members)

                        pos_new = cid_to_pos[new_r]
                        protos[pos_new] = new_summ.prototype
                        new_row = protos @ new_summ.prototype
                        sim_mat[pos_new, :] = new_row
                        sim_mat[:, pos_new] = new_row

                        if dry_run_oracle:
                            vs, cs = np.unique(ds.labels[merged_members], return_counts=True)
                            maj_true[new_r] = int(vs[int(np.argmax(cs))])
                        cur_anchor = new_r

        blocked_guards += uf.blocked_count
        labels = uf.apply_to_labels(labels)

    cl_final: Set[Tuple[int, int]] = set()
    for n1, n2 in seam_split_node_pairs:
        l1, l2 = int(labels[n1]), int(labels[n2])
        if l1 != l2:
            cl_final.add((l1, l2) if l1 < l2 else (l2, l1))
    if uf is not None:
        for a, b in uf.cannot_pairs:
            na = summs[a].members[0]
            nb = summs[b].members[0]
            la, lb = int(labels[na]), int(labels[nb])
            if la != lb:
                cl_final.add((la, lb) if la < lb else (lb, la))

    total_wall = time.perf_counter() - t_cpu0
    router_cpu_sec = max(0.0, total_wall - api_wall_sec)
    stats = {
        "total_calls": total_calls,
        "auto_merge_ops": auto_merge_ops,
        "seam_calls": seam_calls,
        "seam_splits": seam_splits,
        "star_calls": star_calls,
        "star_edges_checked": star_edges_checked,
        "star_merges": star_merges,
        "blocked_by_guard": blocked_guards,
        "intra_cannot_link_pairs": sorted(cl_final),
        "anon_leak_queries": anon_leak_queries,
        "anon_leak_terms": anon_leak_terms,
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "total_tokens": prompt_tok + comp_tok,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": parse_errors,
        "router_cpu_sec": round(router_cpu_sec, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_api_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }
    return labels, np.asarray(labels_init, dtype=np.int64), all_shown, stats
