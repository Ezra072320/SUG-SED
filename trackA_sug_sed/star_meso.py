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
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_MATH_LOCK = threading.Lock()

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import MessageGraph, build_message_graph
from core_shared.partition_cache import load_or_solve_2d_se
from trackA_sug_sed.llm_client import (
    BudgetedLLMClient,
    build_pairwise_same_event_prompt,
    build_seam_split_prompt,
    build_star_merge_prompt,
)


@dataclass
class ClusterSummary:
    cluster_id: int
    members: List[int]
    size: int
    prototype: np.ndarray            # L2-normalized centroid
    medoids: List[int]               # Up to 3 closest-to-centroid posts
    top_keywords: List[str]          # Up to 5 entities/hashtags/lemmas
    mean_timestamp: float


def build_cluster_summaries(
    ds: SocialDataset,
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_medoids: int = 3,
    max_keywords: int = 5,
) -> Dict[int, ClusterSummary]:
    """Summarize each cluster by prototype, medoids, and top keywords."""
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
        if self._ekey(c1, c2) in self.cannot_pairs:
            return False
        cl_hits = 0
        for a in self.comp_members[r1]:
            for b in self.comp_members[r2]:
                if self._ekey(a, b) in self.cannot_pairs:
                    cl_hits += 1
                    if cl_hits >= self.min_cl_votes:
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
        self.cannot_pairs.add(self._ekey(c1, c2))

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
    emb = graph.embeddings
    out: List[SeamCandidate] = []

    for cid, summ in sorted(summaries.items()):
        if summ.size < min_cluster_size:
            continue
        members = np.asarray(summ.members, dtype=np.int64)
        sub_adj = adj[members][:, members].tocsr()
        sub_emb = emb[members]

        k_sub = min(6, max(3, summ.size // 12))
        klab = _spherical_kmeans_deterministic(sub_emb, k=k_sub, n_iters=15)

        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for p in range(k_sub):
            p_loc = np.where(klab == p)[0]
            r_loc = np.where(klab != p)[0]
            if len(p_loc) >= min_pocket_size and len(p_loc) <= int(0.48 * summ.size) and len(r_loc) >= min_pocket_size:
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
            if len(g_pock) < 6 and intra_pock > 0.96:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run Pure-Star-Meso on one streaming block under causal budget & all_shown accounting.

    Returns:
        (final_labels, labels_init, all_shown_mask, stats)
    """
    t_cpu0 = time.perf_counter()
    n = len(ds)
    if anonymizer is not None or (pre_anonymized and orig_ds_for_audit is not None):
        from trackA_sug_sed.anonymizer import (
            QueryItem,
            audit_pre_anonymized_query,
            check_summary_reconstruction_leakage,
        )

    knn_tau = 0.50 if use_purity_base else 0.35
    if graph is None:
        graph = build_message_graph(ds, embeddings, knn_k=15, knn_sim_threshold=knn_tau, attr_alpha=0.5)
    if labels_init is None:
        labels_init, _ = load_or_solve_2d_se(
            dataset_name=dataset_name,
            block_id=block_id,
            adj=graph.adj,
            knn_k=15,
            knn_sim_threshold=knn_tau,
            attr_alpha=0.5,
            refine_moves=True,
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
    eff_tau_auto = float(tau_auto) if (use_auto_merge and use_bayes_band) else 1.01
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
            pairs0: List[Tuple[float, int, int]] = []
            for i in range(len(cids0)):
                row = sim0[i]
                for j in np.where(row >= eff_tau_auto)[0]:
                    if j > i:
                        pairs0.append((float(row[j]), cids0[i], cids0[int(j)]))
            pairs0.sort(key=lambda x: -x[0])
            for _s, c1, c2 in pairs0:
                if uf0.merge(c1, c2):
                    auto_merge_ops += 1
            blocked_guards += uf0.blocked_count
            labels = uf0.apply_to_labels(labels)

    # --- Step 2: Clean SPLIT_SEAM adjudication BEFORE Star-Merge ---
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
            if bool(parsed.get("different_events", False)):
                labels[sc.sub2] = next_cid
                next_cid += 1
                seam_splits += 1

    # --- Step 3: Two-Tier Adaptive Contrastive Star-Merge with Online Guarded UF ---
    summs = build_cluster_summaries(ds, embeddings, labels)
    all_cids = sorted(summs.keys())
    if len(all_cids) >= 2 and max_star_calls > 0:
        protos = np.vstack([summs[c].prototype for c in all_cids])
        sizes = [summs[c].size for c in all_cids]
        sim_mat = protos @ protos.T
        uf = GuardedUnionFind(
            all_cids, sim_mat, guard_sim=guard_sim, enforce_guards=use_guarded_uf, protos=protos, sizes=sizes
        )
        eff_k_nb = int(k_nb) if use_star_packing else 1
        eff_max_star_calls = max_star_calls if use_star_packing else max(80, max_star_calls * 3)
        queried_pairs: Set[Tuple[int, int]] = set()
        rng = np.random.default_rng(seed)

        if not use_bayes_band:
            tiers: List[Tuple[float, float, int]] = [(0.35, 1.01, 1)]
        elif random_order:
            tiers = [(0.45, eff_tau_auto, 1)]
        else:
            tiers = [(float(tau_lo), eff_tau_auto, 2), (0.45, eff_tau_auto, 1)]

        # Majority true labels per cluster only used when dry_run_oracle=True
        maj_true: Dict[int, int] = {}
        if dry_run_oracle:
            for c in all_cids:
                vs, cs = np.unique(ds.labels[summs[c].members], return_counts=True)
                maj_true[c] = int(vs[int(np.argmax(cs))])

        step_idx = 0
        budget_exhausted = False
        for tier_lo, tier_hi, min_sz in tiers:
            if budget_exhausted or step_idx >= eff_max_star_calls:
                break
            while step_idx < eff_max_star_calls:
                candidates_for_step: List[Tuple[float, int, List[Tuple[float, int]]]] = []
                for i, c1 in enumerate(all_cids):
                    if summs[c1].size < min_sz:
                        continue
                    r1 = uf.root(c1)
                    row_sims = sim_mat[i]
                    js = np.where((row_sims >= tier_lo) & (row_sims < tier_hi))[0]
                    if len(js) == 0:
                        continue
                    js = js[np.argsort(-row_sims[js])]
                    pack: List[Tuple[float, int]] = []
                    seen_roots = {r1}
                    for j in js:
                        c2 = all_cids[int(j)]
                        if summs[c2].size < min_sz:
                            continue
                        if summs[c1].size == 1 and summs[c2].size == 1 and float(row_sims[int(j)]) < 0.55:
                            continue
                        r2 = uf.root(c2)
                        if r2 in seen_roots:
                            continue
                        ek = (c1, c2) if c1 < c2 else (c2, c1)
                        if ek in queried_pairs:
                            continue
                        if not uf.can_merge(c1, c2):
                            continue
                        pack.append((float(row_sims[int(j)]), c2))
                        seen_roots.add(r2)
                        if len(pack) >= eff_k_nb:
                            break
                    if not pack:
                        continue
                    # Calibrated Bayes utility: sim^2 * bounded size leverage (log1p min size)
                    utility = sum(
                        (s**2) * (1.0 + 0.25 * math.log1p(min(summs[c1].size, summs[c2].size)))
                        for s, c2 in pack
                    )
                    candidates_for_step.append((utility, c1, pack))

                if not candidates_for_step:
                    break

                if random_order:
                    pick_pos = int(rng.integers(0, len(candidates_for_step)))
                    _u, anchor_c, pack = candidates_for_step[pick_pos]
                else:
                    candidates_for_step.sort(key=lambda x: -x[0])
                    _u, anchor_c, pack = candidates_for_step[0]

                cand_cids = [c2 for _s, c2 in pack]
                cand_summs = [summs[c2] for c2 in cand_cids]
                star_audit = None
                raw_star_user = ""

                if dry_run_oracle:
                    matched_indices = [
                        idx for idx, c2 in enumerate(cand_cids, start=1) if maj_true[anchor_c] == maj_true[c2]
                    ]
                    res_tuple = (
                        {"matches": [{"id": idx, "shared_event": "oracle"} for idx in matched_indices]},
                        0,
                        0,
                        0.0,
                        "ORACLE",
                    )
                elif not use_star_packing:
                    # 1-on-1 static pairwise ablation uses standard pairwise prompt
                    c2 = cand_cids[0]
                    t1 = " | ".join(ds.texts[m].replace("\n", " ")[:120] for m in summs[anchor_c].medoids[:2])
                    t2 = " | ".join(ds.texts[m].replace("\n", " ")[:120] for m in summs[c2].medoids[:2])
                    sys_msg, user_msg = build_pairwise_same_event_prompt(t1, t2)
                    assert client is not None
                    t_api0 = time.perf_counter()
                    res_tuple = client.call_json(
                        request_id=f"pair_s{step_idx}_c{anchor_c}_c{c2}",
                        block_id=block_id,
                        strategy=strategy_tag,
                        query_type="merge_pair",
                        sys_msg=sys_msg,
                        user_msg=user_msg,
                        bucket=bucket,
                        est_completion_tokens=75,
                    )
                    api_wall_sec += time.perf_counter() - t_api0
                else:
                    qid_star = f"star_s{step_idx}_c{anchor_c}_n{len(cand_cids)}"
                    if anonymizer is not None:
                        _, raw_star_user = build_star_merge_prompt(summs[anchor_c], cand_summs, ds)
                        q_star = QueryItem(
                            query_id=qid_star,
                            query_type="star_merge",
                            primary_nodes=list(summs[anchor_c].medoids[:2])
                            + [m for cs in cand_summs for m in cs.medoids[:2]],
                            candidate_clusters=[anchor_c] + cand_cids,
                        )
                        anon_texts, anon_summs, star_audit = anonymizer.build_query_anonymized_inputs(q_star, ds, summs)
                        sys_msg, user_msg = build_star_merge_prompt(
                            anon_summs[anchor_c],
                            [anon_summs[c] for c in cand_cids],
                            ds,
                            text_override=anon_texts,
                        )
                        sys_msg = sys_msg + _anon_note
                    else:
                        sys_msg, user_msg = build_star_merge_prompt(summs[anchor_c], cand_summs, ds)
                        if pre_anonymized:
                            sys_msg = sys_msg + _anon_note
                            if orig_ds_for_audit is not None:
                                star_shown = list(summs[anchor_c].medoids[:2]) + [
                                    m for cs in cand_summs for m in cs.medoids[:2]
                                ]
                                shown_kws = list(summs[anchor_c].top_keywords[:5]) + [
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
                                        summs[anchor_c], cand_summs, orig_ds_for_audit
                                    )
                    assert client is not None
                    t_api0 = time.perf_counter()
                    res_tuple = client.call_json(
                        request_id=qid_star,
                        block_id=block_id,
                        strategy=strategy_tag,
                        query_type="star_merge",
                        sys_msg=sys_msg,
                        user_msg=user_msg,
                        bucket=bucket,
                        est_completion_tokens=105,
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
                star_edges_checked += len(cand_cids)

                # Strict all_shown marking: anchor medoids + every shown candidate medoid!
                for m in summs[anchor_c].medoids[:2]:
                    all_shown[int(m)] = True
                for cs in cand_summs:
                    for m in cs.medoids[:2]:
                        all_shown[int(m)] = True

                for c2 in cand_cids:
                    ek = (anchor_c, c2) if anchor_c < c2 else (c2, anchor_c)
                    queried_pairs.add(ek)

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
                if not use_star_packing and "same_event" in parsed:
                    if bool(parsed.get("same_event", False)):
                        matched_pos.add(1)
                else:
                    raw_matches = parsed.get("matches", [])
                    if isinstance(raw_matches, list):
                        for item in raw_matches:
                            if isinstance(item, int):
                                matched_pos.add(item)
                            elif isinstance(item, dict) and "id" in item:
                                try:
                                    mid = int(item["id"])
                                    matched_pos.add(mid)
                                except (ValueError, TypeError):
                                    pass

                # Apply CANNOT-LINK first on unmatched candidates, then GUARDED MERGE on matched
                for pos, c2 in enumerate(cand_cids, start=1):
                    if pos not in matched_pos:
                        uf.add_cannot_link(anchor_c, c2)
                for pos, c2 in enumerate(cand_cids, start=1):
                    if pos in matched_pos:
                        if uf.merge(anchor_c, c2):
                            star_merges += 1

        blocked_guards += uf.blocked_count
        labels = uf.apply_to_labels(labels)

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
