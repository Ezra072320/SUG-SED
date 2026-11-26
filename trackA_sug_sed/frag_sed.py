"""Simplified FRAG-SED: Fragment-Level Selective LLM Verification for Streaming SED.

Implements the clean 3-step intra-block framework specified in FRAG_SED_Paper_Design.md (§6):
  - Step 0: Plug-and-play unsupervised base partition (default: HISEvent-style 2D-SE, FS >> FM).
  - Step 1: Budget-capped Split Verification (<= rho * B) via deterministic spherical 2-means
            + unified operator Q(R, {P}) (m = 1).
  - Step 2: Utility-guided k-way Merge Verification via unified operator Q(a, B_k(a)) (m <= k)
            and a CleanConstraintDSU tracking exact member vector sums and cannot-link propagation.
  - Step 3: Output contiguous partition labels y_hat_t and strict prompt exposure mask all_shown.

All pruned v2.1 heuristics (AutoMerge+p95, Two-Tier Band, GuardedUF guard_sim, Fiedler dual
proposal, retweet exemption, and cross-day LLM arbitration) are deliberately excluded.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from core_shared.data_loader import SocialDataset
from core_shared.graph_builder import _ensure_l2_normalized
from trackA_sug_sed.llm_client import BudgetedLLMClient, build_frag_select_prompt, estimate_tokens
from trackA_sug_sed.star_meso import (
    ClusterSummary,
    _compute_cluster_idf,
    _summarize_members,
    build_cluster_summaries,
)


def deterministic_spherical_2means(
    sub_emb: np.ndarray,
    max_iters: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic farthest-point spherical 2-means on unit-normalized embeddings.

    Returns (R_local, P_local) where |P_local| <= |R_local|.
    """
    m = len(sub_emb)
    if m < 2:
        return np.arange(m, dtype=np.int64), np.array([], dtype=np.int64)

    mean_vec = sub_emb.sum(axis=0, dtype=np.float64)
    i1 = int(np.argmax(sub_emb @ mean_vec))
    c1 = sub_emb[i1].astype(np.float64)
    c1 = c1 / max(float(np.linalg.norm(c1)), 1e-12)

    i2 = int(np.argmin(sub_emb @ c1))
    if i2 == i1:
        i2 = 1 if i1 == 0 else 0
    c2 = sub_emb[i2].astype(np.float64)
    c2 = c2 / max(float(np.linalg.norm(c2)), 1e-12)

    assign = np.zeros(m, dtype=np.int64)
    for _ in range(max_iters):
        s1 = sub_emb @ c1
        s2 = sub_emb @ c2
        new_assign = (s2 > s1).astype(np.int64)
        if np.array_equal(new_assign, assign) and _ > 0:
            break
        assign = new_assign
        m0 = assign == 0
        m1 = ~m0
        if not np.any(m0) or not np.any(m1):
            break
        v1 = sub_emb[m0].sum(axis=0, dtype=np.float64)
        v2 = sub_emb[m1].sum(axis=0, dtype=np.float64)
        c1 = v1 / max(float(np.linalg.norm(v1)), 1e-12)
        c2 = v2 / max(float(np.linalg.norm(v2)), 1e-12)

    idx0 = np.where(assign == 0)[0]
    idx1 = np.where(assign == 1)[0]
    if len(idx1) <= len(idx0):
        return idx0.astype(np.int64), idx1.astype(np.int64)
    return idx1.astype(np.int64), idx0.astype(np.int64)


class CleanConstraintDSU:
    """Clean Disjoint-Set Union with exact member embedding sums and symmetric cannot-link / queried tracking."""

    def __init__(
        self,
        frag_to_members: Dict[int, List[int]],
        embeddings: np.ndarray,
        initial_cannot_links: Sequence[Tuple[int, int]] = (),
    ) -> None:
        self.active_roots: Set[int] = set(frag_to_members.keys())
        self.parent: Dict[int, int] = {fid: fid for fid in frag_to_members}
        self.comp_members_post: Dict[int, List[int]] = {
            fid: list(members) for fid, members in frag_to_members.items()
        }
        self.comp_vec: Dict[int, np.ndarray] = {
            fid: embeddings[np.asarray(members, dtype=np.int64)].sum(axis=0, dtype=np.float64)
            for fid, members in frag_to_members.items()
        }
        self.N_CL: Dict[int, Set[int]] = {fid: set() for fid in frag_to_members}
        self.N_Q: Dict[int, Set[int]] = {fid: set() for fid in frag_to_members}
        self.cannot_pairs: Set[Tuple[int, int]] = set()
        self.blocked_by_cl: int = 0

        for u, v in initial_cannot_links:
            self.add_cannot_link(int(u), int(v))

    def find(self, x: int) -> int:
        r = x
        while self.parent[r] != r:
            r = self.parent[r]
        while self.parent[x] != r:
            nxt = self.parent[x]
            self.parent[x] = r
            x = nxt
        return r

    def comp_proto(self, r: int) -> np.ndarray:
        v = self.comp_vec[r]
        nrm = float(np.linalg.norm(v))
        return (v / nrm) if nrm > 1e-12 else v

    def comp_size(self, r: int) -> int:
        return len(self.comp_members_post[r])

    def add_cannot_link(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        ek = (ra, rb) if ra < rb else (rb, ra)
        self.cannot_pairs.add(ek)
        if ra == rb:
            return
        self.N_CL[ra].add(rb)
        self.N_CL[rb].add(ra)

    def add_queried(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.N_Q[ra].add(rb)
        self.N_Q[rb].add(ra)

    def can_merge(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        return rb not in self.N_CL[ra]

    def union(self, a: int, b: int) -> Optional[int]:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if rb in self.N_CL[ra]:
            self.blocked_by_cl += 1
            return None

        if len(self.comp_members_post[ra]) < len(self.comp_members_post[rb]):
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.active_roots.discard(rb)
        self.comp_members_post[ra].extend(self.comp_members_post.pop(rb))
        self.comp_vec[ra] = self.comp_vec[ra] + self.comp_vec.pop(rb)

        # Remap symmetric N_CL pointers excluding {ra, rb}
        cl_rb = self.N_CL.pop(rb)
        self.N_CL[ra].discard(rb)
        self.N_CL[ra].discard(ra)
        for u in cl_rb:
            if u == ra or u == rb:
                continue
            self.N_CL[u].discard(rb)
            self.N_CL[u].add(ra)
            self.N_CL[ra].add(u)

        # Remap symmetric N_Q pointers excluding {ra, rb}
        q_rb = self.N_Q.pop(rb)
        self.N_Q[ra].discard(rb)
        self.N_Q[ra].discard(ra)
        for v in q_rb:
            if v == ra or v == rb:
                continue
            self.N_Q[v].discard(rb)
            self.N_Q[v].add(ra)
            self.N_Q[ra].add(v)

        return ra


def _validate_matches(parsed: Dict[str, Any], n_cands: int) -> Optional[List[int]]:
    """Strictly validate JSON {"matches": [1..n_cands]} without coercing invalid payloads."""
    if not isinstance(parsed, dict) or "error" in parsed:
        return None
    raw_m = parsed.get("matches")
    if not isinstance(raw_m, list):
        return None
    out: List[int] = []
    seen: Set[int] = set()
    for item in raw_m:
        if not isinstance(item, int) or isinstance(item, bool):
            return None
        if item < 1 or item > n_cands:
            return None
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _mark_displayed_medoids(
    all_shown: np.ndarray,
    anchor_summ: ClusterSummary,
    cand_summs: Sequence[ClusterSummary],
    max_medoids: int = 2,
) -> None:
    for m in anchor_summ.medoids[:max_medoids]:
        all_shown[int(m)] = True
    for cs in cand_summs:
        for m in cs.medoids[:max_medoids]:
            all_shown[int(m)] = True


def run_frag_sed(
    block_id: int,
    ds: SocialDataset,
    embeddings: np.ndarray,
    labels_init: np.ndarray,
    client: BudgetedLLMClient,
    bucket: Optional[Any] = None,
    tau_lo: float = 0.50,
    k: int = 4,
    m_split: int = 8,
    delta: float = 0.12,
    rho: float = 0.30,
    max_iters: int = 2000,
    strategy_tag: str = "frag_sed",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run simplified 3-step FRAG-SED over an initial base partition `labels_init`."""
    t0 = time.perf_counter()
    t_thread0 = time.thread_time()
    emb = _ensure_l2_normalized(embeddings)
    n = len(ds)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=bool), {
            "total_calls": 0,
            "total_tokens": 0,
            "parse_errors": 0,
        }

    # Canonicalize initial fragment IDs to 0..K0-1
    _, inv0 = np.unique(np.asarray(labels_init, dtype=np.int64), return_inverse=True)
    frag_to_members: Dict[int, List[int]] = defaultdict(list)
    for idx, fid in enumerate(inv0.tolist()):
        frag_to_members[int(fid)].append(idx)

    summs0 = build_cluster_summaries(ds, emb, inv0, max_medoids=2, max_keywords=5)
    all_shown = np.zeros(n, dtype=bool)
    calls = pt_sum = ct_sum = errs = 0
    llm_sec = api_wall_sec = 0.0

    # --- Step 1: Budget-Capped Split Verification (<= rho * B) ---
    split_candidates: List[Tuple[float, int, List[int], List[int]]] = []
    for cid, summ in summs0.items():
        if summ.size < m_split:
            continue
        members = summ.members
        r_loc, p_loc = deterministic_spherical_2means(emb[np.asarray(members, dtype=np.int64)])
        if len(p_loc) < 3 or len(p_loc) > len(r_loc):
            continue
        r_idx = [members[int(i)] for i in r_loc]
        p_idx = [members[int(i)] for i in p_loc]
        v_r = emb[np.asarray(r_idx, dtype=np.int64)].sum(axis=0, dtype=np.float64)
        v_p = emb[np.asarray(p_idx, dtype=np.int64)].sum(axis=0, dtype=np.float64)
        mu_r = v_r / max(float(np.linalg.norm(v_r)), 1e-12)
        mu_p = v_p / max(float(np.linalg.norm(v_p)), 1e-12)
        intra_p = float(np.mean(emb[np.asarray(p_idx, dtype=np.int64)] @ mu_p))
        sep = intra_p - float(np.dot(mu_r, mu_p))
        if sep >= delta:
            score = sep * math.log1p(len(p_idx))
            split_candidates.append((score, cid, r_idx, p_idx))

    split_candidates.sort(key=lambda x: -x[0])

    next_fid = max(frag_to_members.keys()) + 1
    initial_cl: List[Tuple[int, int]] = []
    split_calls = split_accepted = 0
    split_cap = int(rho * bucket.current_block_cap) if bucket is not None else None

    for _sc, cid, r_idx, p_idx in split_candidates:
        r_summ = _summarize_members(cid, r_idx, ds, emb, max_medoids=2, max_keywords=5)
        p_summ = _summarize_members(next_fid, p_idx, ds, emb, max_medoids=2, max_keywords=5)
        sys_msg, user_msg = build_frag_select_prompt(r_summ, [p_summ], ds)
        est_tok = estimate_tokens(sys_msg, user_msg, est_completion_tokens=64)
        if split_cap is not None and (bucket.current_block_spent + est_tok > split_cap):
            break

        t_api0 = time.perf_counter()
        res = client.call_json(
            request_id=f"fragsed_split_M{block_id:02d}_c{cid}",
            block_id=block_id,
            strategy=strategy_tag,
            query_type="frag_split",
            sys_msg=sys_msg,
            user_msg=user_msg,
            bucket=bucket,
            est_completion_tokens=64,
        )
        api_wall_sec += time.perf_counter() - t_api0
        if res is None:
            break
        parsed, pt, ct, lat, _ = res
        pt_sum += pt
        ct_sum += ct
        llm_sec += lat
        calls += 1
        split_calls += 1
        _mark_displayed_medoids(all_shown, r_summ, [p_summ], max_medoids=2)

        matches = _validate_matches(parsed, n_cands=1)
        if matches is None:
            errs += 1
            continue
        if 1 not in matches:
            frag_to_members[cid] = r_idx
            frag_to_members[next_fid] = p_idx
            initial_cl.append((cid, next_fid))
            next_fid += 1
            split_accepted += 1

    # --- Step 2: Utility-Guided k-Way Merge Verification ---
    term_idf = _compute_cluster_idf(ds, frag_to_members)
    dsu = CleanConstraintDSU(frag_to_members, emb, initial_cannot_links=initial_cl)
    all_cids = sorted(dsu.active_roots)
    cid_to_pos = {c: i for i, c in enumerate(all_cids)}
    summs: Dict[int, ClusterSummary] = {
        c: _summarize_members(
            c, dsu.comp_members_post[c], ds, emb, term_idf=term_idf, max_medoids=2, max_keywords=5
        )
        for c in all_cids
    }
    protos = np.vstack([dsu.comp_proto(c) for c in all_cids]).astype(np.float64)
    sim_mat = protos @ protos.T
    np.fill_diagonal(sim_mat, -2.0)

    failed_pairs: Set[Tuple[int, int]] = set()
    merge_calls = merge_ops = 0

    for _it in range(max_iters):
        best_u = -1.0
        best_a: Optional[int] = None
        best_pack: List[int] = []

        active_list = sorted(dsu.active_roots)
        if len(active_list) < 2:
            break

        for a in active_list:
            if dsu.comp_size(a) < 2:
                continue
            pa = cid_to_pos[a]
            row = sim_mat[pa]
            cand_pairs: List[Tuple[float, int]] = []
            for c in active_list:
                if c == a:
                    continue
                s_ac = float(row[cid_to_pos[c]])
                if s_ac < tau_lo:
                    continue
                if c in dsu.N_CL[a] or c in dsu.N_Q[a]:
                    continue
                ek = (a, c) if a < c else (c, a)
                if ek in failed_pairs:
                    continue
                cand_pairs.append((s_ac, c))
            if not cand_pairs:
                continue
            cand_pairs.sort(key=lambda x: -x[0])
            pack = cand_pairs[:k]
            sz_a = dsu.comp_size(a)
            u_val = sum(s_ac * math.log1p(min(sz_a, dsu.comp_size(c))) for s_ac, c in pack)
            if u_val > best_u:
                best_u = u_val
                best_a = a
                best_pack = [c for _s, c in pack]

        if best_a is None or not best_pack:
            break

        a_summ = summs[best_a]
        c_summs = [summs[c] for c in best_pack]
        sys_msg, user_msg = build_frag_select_prompt(a_summ, c_summs, ds)
        t_api0 = time.perf_counter()
        res = client.call_json(
            request_id=f"fragsed_merge_M{block_id:02d}_it{merge_calls}_a{best_a}",
            block_id=block_id,
            strategy=strategy_tag,
            query_type="frag_merge",
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
        merge_calls += 1
        _mark_displayed_medoids(all_shown, a_summ, c_summs, max_medoids=2)

        matches = _validate_matches(parsed, n_cands=len(best_pack))
        if matches is None:
            errs += 1
            for c in best_pack:
                ek = (best_a, c) if best_a < c else (c, best_a)
                failed_pairs.add(ek)
            continue

        for c in best_pack:
            dsu.add_queried(best_a, c)

        match_set = set(matches)
        for j_idx, c in enumerate(best_pack, start=1):
            if j_idx not in match_set:
                dsu.add_cannot_link(best_a, c)

        # best_pack is already ordered by descending similarity
        for j_idx, c in enumerate(best_pack, start=1):
            if j_idx not in match_set:
                continue
            ra = dsu.find(best_a)
            rc = dsu.find(c)
            if ra == rc or not dsu.can_merge(ra, rc):
                if ra != rc:
                    dsu.blocked_by_cl += 1
                continue
            r_new = dsu.union(ra, rc)
            if r_new is None:
                continue
            merge_ops += 1
            r_old = rc if r_new == ra else ra

            summs.pop(r_old, None)
            summs[r_new] = _summarize_members(
                r_new,
                dsu.comp_members_post[r_new],
                ds,
                emb,
                term_idf=term_idf,
                max_medoids=2,
                max_keywords=5,
            )
            p_new = cid_to_pos[r_new]
            p_old = cid_to_pos[r_old]
            mu_new = dsu.comp_proto(r_new)
            protos[p_new] = mu_new
            new_row = protos @ mu_new
            sim_mat[p_new, :] = new_row
            sim_mat[:, p_new] = new_row
            sim_mat[p_new, p_new] = -2.0
            sim_mat[p_old, :] = -2.0
            sim_mat[:, p_old] = -2.0

            failed_pairs = {
                ek for ek in failed_pairs if ek[0] not in (ra, rc) and ek[1] not in (ra, rc)
            }

    # --- Step 3: Output contiguous partition labels ---
    raw_out = np.empty(n, dtype=np.int64)
    for r in dsu.active_roots:
        for post_i in dsu.comp_members_post[r]:
            raw_out[post_i] = r
    _, clean_labels = np.unique(raw_out, return_inverse=True)

    total_wall = time.perf_counter() - t0
    router_cpu = min(max(0.0, total_wall - api_wall_sec), max(0.0, time.thread_time() - t_thread0))
    return clean_labels.astype(np.int64), all_shown, {
        "total_calls": calls,
        "split_calls": split_calls,
        "split_accepted": split_accepted,
        "merge_calls": merge_calls,
        "merge_ops": merge_ops,
        "blocked_by_cl": dsu.blocked_by_cl,
        "total_tokens": pt_sum + ct_sum,
        "prompt_tokens": pt_sum,
        "completion_tokens": ct_sum,
        "shown_posts": int(all_shown.sum()),
        "shown_pct": float(all_shown.sum() * 100.0 / max(1, n)),
        "parse_errors": errs,
        "n_init_frags": len(summs0),
        "n_final_clusters": int(clean_labels.max() + 1) if n > 0 else 0,
        "router_cpu_sec": round(router_cpu, 4),
        "api_wall_sec": round(api_wall_sec, 4),
        "llm_api_sec": round(llm_sec, 4),
        "total_wall_sec": round(total_wall, 4),
    }
