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

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sug_sed_method.core.data_loader import SocialDataset
from sug_sed_method.core.metrics_audit import full_partition_audit
from sug_sed_method.core.token_ledger import CausalUniformBucket
from sug_sed_method.method.llm_client import BudgetedLLMClient, build_cross_star_prompt
from sug_sed_method.method.star_meso import ClusterSummary, build_cluster_summaries


@dataclass
class HistoricalEvent:
    global_id: int
    prototype: np.ndarray
    total_count: int
    first_block: int
    last_block: int
    rep_texts: List[str]
    top_keywords: List[str] = field(default_factory=list)


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
        if new_txt and new_txt not in ev.rep_texts:
            ev.rep_texts = (ev.rep_texts[:2] + [new_txt])[:3]
        ev.top_keywords = list(dict.fromkeys(ev.top_keywords[:3] + summ.top_keywords[:3]))[:5]


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
) -> Dict[str, Any]:
    """Evaluate a cross-day linking policy chronologically over `(block_id, ds, emb, labels)`."""
    mem = GlobalEventMemory(window_blocks=window_blocks)
    oracle_hist_maj: Dict[int, int] = {}

    eval_true: List[np.ndarray] = []
    eval_pred: List[np.ndarray] = []
    stats = {
        "auto_linked": 0,
        "llm_queried": 0,
        "llm_linked": 0,
        "llm_rejected": 0,
        "spawned_new": 0,
        "parse_errors": 0,
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
        summaries = build_cluster_summaries(ds, emb, blk_labels)
        ordered_cids = sorted(summaries.keys(), key=lambda c: -summaries[c].size)
        active = mem.active_events(bid)
        if active:
            act_gids = [ev.global_id for ev in active]
            act_protos = np.vstack([ev.prototype for ev in active])
        else:
            act_gids = []
            act_protos = np.zeros((0, emb.shape[1]), dtype=np.float32)

        if bucket is not None and is_eval:
            bucket.start_block()
        local_to_global: Dict[int, int] = {}

        for cid in ordered_cids:
            summ = summaries[cid]
            vs, cs = np.unique(ds.labels[summ.members], return_counts=True)
            cur_maj = int(vs[int(np.argmax(cs))])

            if mode == "no_link" or not act_gids:
                gid = mem.spawn_event(bid, summ, ds)
                oracle_hist_maj[gid] = cur_maj
                local_to_global[cid] = gid
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
            best_pos = int(top_pos[0])
            best_score = float(scores[best_pos])
            best_cos = float(cos_sims[best_pos])
            best_gid = act_gids[best_pos]

            if mode in {"proto_only", "proto_attr_only"}:
                if best_score >= tau_proto:
                    mem.update_event(best_gid, bid, summ, ds)
                    local_to_global[cid] = best_gid
                    if is_eval:
                        stats["auto_linked"] += 1
                else:
                    gid = mem.spawn_event(bid, summ, ds)
                    oracle_hist_maj[gid] = cur_maj
                    local_to_global[cid] = gid
                    if is_eval:
                        stats["spawned_new"] += 1
            elif mode == "oracle_band":
                if best_cos >= tau_hi:
                    mem.update_event(best_gid, bid, summ, ds)
                    local_to_global[cid] = best_gid
                    if is_eval:
                        stats["auto_linked"] += 1
                elif best_cos >= tau_lo and summ.size >= min_cross_size and is_eval:
                    stats["llm_queried"] += 1
                    matched_gid = None
                    for pos in top_pos[:top_k_cand]:
                        cand_gid = act_gids[int(pos)]
                        if float(cos_sims[int(pos)]) >= tau_lo and oracle_hist_maj.get(cand_gid) == cur_maj:
                            matched_gid = cand_gid
                            break
                    if matched_gid is not None:
                        mem.update_event(matched_gid, bid, summ, ds)
                        local_to_global[cid] = matched_gid
                        stats["llm_linked"] += 1
                    else:
                        gid = mem.spawn_event(bid, summ, ds)
                        oracle_hist_maj[gid] = cur_maj
                        local_to_global[cid] = gid
                        stats["llm_rejected"] += 1
                        stats["spawned_new"] += 1
                else:
                    if best_cos >= tau_fb:
                        mem.update_event(best_gid, bid, summ, ds)
                        local_to_global[cid] = best_gid
                        if is_eval:
                            stats["auto_linked"] += 1
                    else:
                        gid = mem.spawn_event(bid, summ, ds)
                        oracle_hist_maj[gid] = cur_maj
                        local_to_global[cid] = gid
                        if is_eval:
                            stats["spawned_new"] += 1
            elif mode == "sug_cross_star":
                if best_cos >= tau_hi or (not is_eval and best_cos >= tau_fb):
                    mem.update_event(best_gid, bid, summ, ds)
                    local_to_global[cid] = best_gid
                    if is_eval:
                        stats["auto_linked"] += 1
                elif is_eval and best_cos >= tau_lo and summ.size >= min_cross_size and client is not None:
                    cand_events: List[HistoricalEvent] = []
                    for pos in top_pos:
                        if float(cos_sims[int(pos)]) >= tau_lo:
                            cand_events.append(active[int(pos)])
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
                        if best_cos >= tau_fb:
                            mem.update_event(best_gid, bid, summ, ds)
                            local_to_global[cid] = best_gid
                            stats["auto_linked"] += 1
                        else:
                            gid = mem.spawn_event(bid, summ, ds)
                            local_to_global[cid] = gid
                            stats["spawned_new"] += 1
                    else:
                        parsed, pt, ct, _lat, _st = res
                        stats["llm_queried"] += 1
                        stats["cross_tokens"] += pt + ct
                        if isinstance(parsed, dict) and "error" in parsed:
                            stats["parse_errors"] += 1
                        matched_id = parsed.get("matched_id", None) if isinstance(parsed, dict) else None
                        chosen_gid: Optional[int] = None
                        if matched_id is not None:
                            try:
                                m_idx = int(matched_id)
                                if 1 <= m_idx <= len(cand_events):
                                    chosen_gid = cand_events[m_idx - 1].global_id
                            except (ValueError, TypeError):
                                chosen_gid = None

                        if chosen_gid is not None:
                            mem.update_event(chosen_gid, bid, summ, ds)
                            local_to_global[cid] = chosen_gid
                            stats["llm_linked"] += 1
                        else:
                            gid = mem.spawn_event(bid, summ, ds)
                            local_to_global[cid] = gid
                            stats["llm_rejected"] += 1
                            stats["spawned_new"] += 1
                else:
                    if best_cos >= tau_fb:
                        mem.update_event(best_gid, bid, summ, ds)
                        local_to_global[cid] = best_gid
                        if is_eval:
                            stats["auto_linked"] += 1
                    else:
                        gid = mem.spawn_event(bid, summ, ds)
                        oracle_hist_maj[gid] = cur_maj
                        local_to_global[cid] = gid
                        if is_eval:
                            stats["spawned_new"] += 1
            else:
                raise ValueError(f"Unknown cross-day mode: {mode!r}")

        if is_eval:
            glab = np.array([local_to_global[int(c)] for c in blk_labels.tolist()], dtype=np.int64)
            eval_true.append(ds.labels)
            eval_pred.append(glab)

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
