"""Longitudinal Cross-Day Streaming Event Tracking (GlobalEventMemory & Cross-Star).

Evaluates chronological multi-block cluster continuity without future leakage:
  - `no_link`          : Independent daily cluster IDs (lower bound)
  - `proto_only`       : Unsupervised cosine prototype linking across full sweep 0.65..0.95
  - `proto_attr_only`  : Unsupervised prototype cosine + keyword/entity Jaccard (0 tokens)
  - `sug_cross_star`   : Bayes ambiguity-band [0.65, 0.90) Top-3 contrastive LLM linker
                         with causal 6k/blk budget and tau_fb = 0.90 fallback
  - `oracle_band`      : Oracle upper bound within the identical [0.65, 0.90) band
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from core_shared.data_loader import SocialDataset
from core_shared.metrics_audit import full_partition_audit
from core_shared.token_ledger import CausalUniformBucket
from trackA_sug_sed.llm_client import BudgetedLLMClient, build_cross_star_prompt
from trackA_sug_sed.star_meso import ClusterSummary, build_cluster_summaries


@dataclass
class HistoricalEvent:
    global_id: int
    prototype: np.ndarray
    total_count: int
    first_block: int
    last_block: int
    rep_texts: List[str]
    top_keywords: List[str] = field(default_factory=list)


def _update_rolling_rep_texts(rep_texts: List[str], new_txt: str, max_items: int = 3) -> List[str]:
    if not new_txt or new_txt in rep_texts:
        return rep_texts
    if not rep_texts:
        return [new_txt]
    return ([rep_texts[0], new_txt] + [t for t in rep_texts[1:] if t != new_txt])[:max_items]


class GlobalEventMemory:
    """Lightweight O(|E_active| * d) streaming memory over sliding W blocks."""

    def __init__(self, window_blocks: int = 14) -> None:
        self.window_blocks = int(window_blocks)
        self.events: Dict[int, HistoricalEvent] = {}
        self.next_gid: int = 0

    def active_events(self, current_block: int) -> List[HistoricalEvent]:
        return [
            ev
            for ev in self.events.values()
            if (current_block - ev.last_block) <= self.window_blocks
        ]

    def evict_expired(self, current_block: int) -> int:
        """Evict historical events outside the sliding window to keep O(|E_active|*d) memory/scan."""
        expired = [
            gid
            for gid, ev in self.events.items()
            if (current_block - ev.last_block) > self.window_blocks
        ]
        for gid in expired:
            del self.events[gid]
        return len(expired)

    def spawn_event(self, block_id: int, summ: ClusterSummary, ds: SocialDataset) -> int:
        gid = self.next_gid
        self.next_gid += 1
        rep_texts = [ds.texts[m].replace("\n", " ").strip()[:140] for m in summ.medoids[:3]]
        self.events[gid] = HistoricalEvent(
            global_id=gid,
            prototype=summ.prototype.astype(np.float32, copy=True),
            total_count=int(summ.size),
            first_block=int(block_id),
            last_block=int(block_id),
            rep_texts=rep_texts,
            top_keywords=list(summ.top_keywords[:5]),
        )
        return gid

    def update_event(
        self,
        gid: int,
        block_id: int,
        summ: ClusterSummary,
        ds: SocialDataset,
        ema_cap_weight: int = 200,
    ) -> None:
        ev = self.events[gid]
        w_old = float(min(ev.total_count, ema_cap_weight))
        w_new = float(summ.size)
        proto = w_old * ev.prototype + w_new * summ.prototype
        norm = float(np.linalg.norm(proto))
        if norm > 1e-12:
            proto = proto / norm
        ev.prototype = proto.astype(np.float32)
        ev.total_count += int(summ.size)
        ev.last_block = int(block_id)
        new_txt = ds.texts[summ.medoids[0]].replace("\n", " ").strip()[:140] if summ.medoids else ""
        ev.rep_texts = _update_rolling_rep_texts(ev.rep_texts, new_txt, max_items=3)
        ev.top_keywords = list(dict.fromkeys(ev.top_keywords[:2] + summ.top_keywords[:3] + ev.top_keywords[2:]))[:5]


def _attr_jaccard(kw_a: Sequence[str], kw_b: Sequence[str]) -> float:
    sa = {k.lower() for k in kw_a if k}
    sb = {k.lower() for k in kw_b if k}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def evaluate_cross_day_stream(
    block_stream: Sequence[Tuple[int, SocialDataset, np.ndarray, np.ndarray]],
    mode: str,
    eval_start_block: int = 2,
    *,
    tau_proto: float = 0.85,
    attr_weight: float = 0.15,
    tau_lo: float = 0.65,
    tau_hi: float = 0.90,
    tau_fb: float = 0.90,
    min_cross_size: int = 3,
    top_k_cand: int = 3,
    window_blocks: int = 14,
    client: Optional[BudgetedLLMClient] = None,
    budget_per_block: int = 6000,
    strategy_tag: str = "sug_cross_star",
    enforce_block_uniqueness: Optional[bool] = None,
    block_cannot_links: Optional[Dict[int, Sequence[Tuple[int, int]]]] = None,
    same_block_guard_sim: float = 0.40,
) -> Dict[str, Any]:
    """Evaluate a cross-day linking policy chronologically over `(block_id, ds, emb, labels)`."""
    mem = GlobalEventMemory(window_blocks=window_blocks)
    oracle_hist_maj: Dict[int, int] = {}
    eff_enforce_unique = (
        (block_cannot_links is None)
        if enforce_block_uniqueness is None
        else bool(enforce_block_uniqueness)
    )

    eval_true: List[np.ndarray] = []
    eval_pred: List[np.ndarray] = []
    stats = {
        "auto_linked": 0,
        "llm_queried": 0,
        "llm_linked": 0,
        "llm_rejected": 0,
        "spawned_new": 0,
        "parse_errors": 0,
        "budget_skipped": 0,
        "blocked_same_block_collisions": 0,
        "blocked_cannot_link_collisions": 0,
        "same_block_gid_collisions": 0,
        "cross_tokens": 0,
        "budget_per_block": int(budget_per_block),
        "allow_carryover": True,
    }

    bucket = (
        CausalUniformBucket(tokens_per_block=budget_per_block, allow_carryover=True)
        if (mode == "sug_cross_star" and client is not None)
        else None
    )

    for bid, ds, emb, blk_labels in block_stream:
        is_eval = int(bid) >= int(eval_start_block)
        mem.evict_expired(int(bid))
        summaries = build_cluster_summaries(ds, emb, blk_labels)
        ordered_cids = sorted(summaries.keys(), key=lambda c: -summaries[c].size)
        active = mem.active_events(bid)
        if active:
            act_gids = [ev.global_id for ev in active]
            act_protos = np.vstack([ev.prototype for ev in active])
        else:
            act_gids = []
            act_protos = np.zeros((0, emb.shape[1]), dtype=np.float32)

        gid_to_act_pos: Dict[int, int] = {gid: i for i, gid in enumerate(act_gids)}
        gid_to_cids: Dict[int, List[int]] = defaultdict(list)
        bid_cl_set: Set[Tuple[int, int]] = set()
        if block_cannot_links and int(bid) in block_cannot_links:
            for a, b in block_cannot_links[int(bid)]:
                ca, cb = int(a), int(b)
                bid_cl_set.add((ca, cb) if ca < cb else (cb, ca))

        def _check_gid_allowed(cur_cid: int, target_gid: int) -> Tuple[bool, str]:
            prev_cids = gid_to_cids.get(target_gid)
            if not prev_cids:
                return True, ""
            if eff_enforce_unique:
                return False, "same_block"
            cur_proto = summaries[cur_cid].prototype
            for pc in prev_cids:
                pair = (cur_cid, pc) if cur_cid < pc else (pc, cur_cid)
                if pair in bid_cl_set:
                    return False, "cl"
                if float(np.dot(cur_proto, summaries[pc].prototype)) < same_block_guard_sim:
                    return False, "same_block"
            return True, ""

        def _gid_allowed(cur_cid: int, target_gid: int) -> bool:
            return _check_gid_allowed(cur_cid, target_gid)[0]

        if bucket is not None and is_eval:
            bucket.start_block()
        local_to_global: Dict[int, int] = {}

        def _link_to_gid(cur_cid: int, target_gid: int) -> None:
            mem.update_event(target_gid, bid, summaries[cur_cid], ds)
            pos_act = gid_to_act_pos.get(target_gid)
            if pos_act is not None:
                act_protos[pos_act] = mem.events[target_gid].prototype
            local_to_global[cur_cid] = target_gid
            gid_to_cids[target_gid].append(cur_cid)
            if is_eval and len(gid_to_cids[target_gid]) > 1:
                stats["same_block_gid_collisions"] += 1

        def _spawn_gid(cur_cid: int, cur_maj: int) -> int:
            new_gid = mem.spawn_event(bid, summaries[cur_cid], ds)
            oracle_hist_maj[new_gid] = cur_maj
            local_to_global[cur_cid] = new_gid
            gid_to_cids[new_gid].append(cur_cid)
            return new_gid

        for cid in ordered_cids:
            summ = summaries[cid]
            cur_maj = 0
            if mode == "oracle_band":
                vs, cs = np.unique(ds.labels[summ.members], return_counts=True)
                cur_maj = int(vs[int(np.argmax(cs))])

            if mode == "no_link" or not act_gids:
                _spawn_gid(cid, cur_maj)
                if is_eval:
                    stats["spawned_new"] += 1
                continue

            cos_sims = act_protos @ summ.prototype
            if mode == "proto_attr_only":
                scores = np.array(
                    [
                        float(cos_sims[pos]) + attr_weight * _attr_jaccard(summ.top_keywords, active[pos].top_keywords)
                        for pos in range(len(active))
                    ],
                    dtype=np.float64,
                )
            else:
                scores = cos_sims

            top_pos = np.argsort(-scores)
            allowed_positions = [int(p) for p in top_pos if _gid_allowed(cid, act_gids[int(p)])]
            if len(top_pos) > 0 and is_eval:
                ok_top, blk_reason = _check_gid_allowed(cid, act_gids[int(top_pos[0])])
                if not ok_top:
                    if blk_reason == "cl":
                        stats["blocked_cannot_link_collisions"] += 1
                    else:
                        stats["blocked_same_block_collisions"] += 1

            if not allowed_positions:
                _spawn_gid(cid, cur_maj)
                if is_eval:
                    stats["spawned_new"] += 1
                continue

            best_pos = allowed_positions[0]
            best_score = float(scores[best_pos])
            best_cos = float(cos_sims[best_pos])
            best_gid = act_gids[best_pos]

            if mode in {"proto_only", "proto_attr_only"}:
                if best_score >= tau_proto:
                    _link_to_gid(cid, best_gid)
                    if is_eval:
                        stats["auto_linked"] += 1
                else:
                    _spawn_gid(cid, cur_maj)
                    if is_eval:
                        stats["spawned_new"] += 1
            elif mode == "oracle_band":
                if best_cos >= tau_hi:
                    _link_to_gid(cid, best_gid)
                    if is_eval:
                        stats["auto_linked"] += 1
                elif best_cos >= tau_lo and summ.size >= min_cross_size and is_eval:
                    stats["llm_queried"] += 1
                    matched_gid = None
                    for pos in allowed_positions[:top_k_cand]:
                        cand_gid = act_gids[pos]
                        if float(cos_sims[pos]) >= tau_lo and oracle_hist_maj.get(cand_gid) == cur_maj:
                            matched_gid = cand_gid
                            break
                    if matched_gid is not None:
                        _link_to_gid(cid, matched_gid)
                        stats["llm_linked"] += 1
                    else:
                        _spawn_gid(cid, cur_maj)
                        stats["llm_rejected"] += 1
                        stats["spawned_new"] += 1
                else:
                    if best_cos >= tau_fb:
                        _link_to_gid(cid, best_gid)
                        if is_eval:
                            stats["auto_linked"] += 1
                    else:
                        _spawn_gid(cid, cur_maj)
                        if is_eval:
                            stats["spawned_new"] += 1
            elif mode == "sug_cross_star":
                if best_cos >= tau_hi or (not is_eval and best_cos >= tau_fb):
                    _link_to_gid(cid, best_gid)
                    if is_eval:
                        stats["auto_linked"] += 1
                elif is_eval and best_cos >= tau_lo and summ.size >= min_cross_size and client is not None:
                    cand_events: List[HistoricalEvent] = []
                    for pos in allowed_positions:
                        if float(cos_sims[pos]) >= tau_lo:
                            cand_events.append(active[pos])
                        if len(cand_events) >= top_k_cand:
                            break
                    if not cand_events:
                        cand_events = [active[best_pos]]

                    sys_msg, user_msg = build_cross_star_prompt(summ, cand_events, ds)
                    res = client.call_json(
                        request_id=f"cross_M{bid:02d}_c{cid}_k{len(cand_events)}",
                        block_id=bid,
                        strategy=strategy_tag,
                        query_type="cross_star",
                        sys_msg=sys_msg,
                        user_msg=user_msg,
                        bucket=bucket,
                        est_completion_tokens=95,
                    )
                    if res is None:
                        # Causal cross budget exhausted -> tau_fb fallback (when tau_fb == tau_hi == 0.90, always spawns new)
                        stats["budget_skipped"] += 1
                        if best_cos >= tau_fb:
                            _link_to_gid(cid, best_gid)
                            stats["auto_linked"] += 1
                        else:
                            _spawn_gid(cid, cur_maj)
                            stats["spawned_new"] += 1
                    else:
                        parsed, pt, ct, _lat, _st = res
                        stats["llm_queried"] += 1
                        stats["cross_tokens"] += pt + ct
                        has_api_err = isinstance(parsed, dict) and "error" in parsed
                        call_parse_err = bool(has_api_err)
                        matched_id = parsed.get("matched_id", None) if isinstance(parsed, dict) else None
                        chosen_gid: Optional[int] = None
                        if not has_api_err:
                            if not isinstance(parsed, dict):
                                call_parse_err = True
                            elif matched_id is not None:
                                m_idx: Optional[int] = None
                                if isinstance(matched_id, int) and not isinstance(matched_id, bool):
                                    m_idx = matched_id
                                elif isinstance(matched_id, str) and matched_id.strip().isdigit():
                                    m_idx = int(matched_id.strip())
                                if m_idx is not None and 1 <= m_idx <= len(cand_events):
                                    cand_gid = cand_events[m_idx - 1].global_id
                                    if _gid_allowed(cid, cand_gid):
                                        chosen_gid = cand_gid
                                else:
                                    call_parse_err = True
                        if call_parse_err:
                            stats["parse_errors"] += 1

                        if chosen_gid is not None:
                            _link_to_gid(cid, chosen_gid)
                            stats["llm_linked"] += 1
                        else:
                            _spawn_gid(cid, cur_maj)
                            if not call_parse_err:
                                stats["llm_rejected"] += 1
                            stats["spawned_new"] += 1
                else:
                    if best_cos >= tau_fb:
                        _link_to_gid(cid, best_gid)
                        if is_eval:
                            stats["auto_linked"] += 1
                    else:
                        _spawn_gid(cid, cur_maj)
                        if is_eval:
                            stats["spawned_new"] += 1
            else:
                raise ValueError(f"Unknown cross-day mode: {mode!r}")

        if is_eval:
            glab = np.array([local_to_global[int(c)] for c in blk_labels.tolist()], dtype=np.int64)
            eval_true.append(ds.labels)
            eval_pred.append(glab)

    if not eval_true:
        return {
            "mode": mode,
            "tau_proto": tau_proto,
            "g_ami": 0.0,
            "g_mac_b3": 0.0,
            "g_mic_b3": 0.0,
            "g_ari": 0.0,
            "g_tail_b3": 0.0,
            "n_pred_events": 0,
            "stats": stats,
            "warning": "no_eval_blocks",
        }

    y_t = np.concatenate(eval_true)
    y_p = np.concatenate(eval_pred)
    aud = full_partition_audit(y_t, y_p)
    return {
        "mode": mode,
        "tau_proto": tau_proto,
        "g_ami": round(aud["ami"] * 100.0, 2),
        "g_mac_b3": round(aud["bcubed_full"]["macro_b3_f1"] * 100.0, 2),
        "g_mic_b3": round(aud["bcubed_full"]["micro_b3_f1"] * 100.0, 2),
        "g_ari": round(aud["ari"] * 100.0, 2),
        "g_tail_b3": round(aud["stratified"]["tail"]["macro_b3_f1"] * 100.0, 2),
        "n_pred_events": int(aud["n_pred_clusters"]),
        "stats": stats,
    }
