"""O(N) exact clustering & SED partition auditing metrics for Gate A / Gate B.

Computes:
  - NMI, AMI, ARI
  - Exact O(N) Micro (item-average) B-cubed and Event Macro B-cubed (P, R, F1)
  - Exact O(N) Pairwise Error Decomposition (False Merges, False Splits,
    and Before-vs-After Resolved vs. Newly Introduced False Merges & False Splits)
  - Unqueried-Post Net Gain (delta B-cubed strictly on posts NOT queried by LLM,
    evaluated within the full partition contingency table)
  - Stratified Head / Medium / Tail Event Macro-B3 & Tail Discovery Recall
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def _comb2(n: int) -> int:
    return n * (n - 1) // 2


def item_bcubed_scores(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact per-item B-cubed Precision, Recall, and F1 in O(N)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())
    joint_counts = Counter(zip(y_true.tolist(), y_pred.tolist()))

    p = np.empty(n, dtype=np.float64)
    r = np.empty(n, dtype=np.float64)
    for idx, (yt, yp) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
        jc = joint_counts[(yt, yp)]
        p[idx] = jc / pred_counts[yp]
        r[idx] = jc / true_counts[yt]
    denom = p + r
    f1 = np.where(denom > 0, 2.0 * p * r / denom, 0.0)
    return p, r, f1


def event_macro_bcubed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subset_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute Micro B-cubed and Event Macro B-cubed using full-partition counts.

    When `subset_mask` is given (e.g. unqueried posts only), item-level B-cubed
    scores are STILL computed against the full partition contingency table, and
    then averaged across the masked subset (and per-event within that subset).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_item, r_item, f1_item = item_bcubed_scores(y_true, y_pred)

    if subset_mask is not None:
        mask = np.asarray(subset_mask, dtype=bool)
        if mask.sum() == 0:
            return {
                "micro_b3_p": 0.0,
                "micro_b3_r": 0.0,
                "micro_b3_f1": 0.0,
                "macro_b3_p": 0.0,
                "macro_b3_r": 0.0,
                "macro_b3_f1": 0.0,
                "macro_b3_item_f1": 0.0,
                "num_items": 0,
                "num_events": 0,
            }
        y_eval = y_true[mask]
        p_eval = p_item[mask]
        r_eval = r_item[mask]
        f1_eval = f1_item[mask]
    else:
        y_eval, p_eval, r_eval, f1_eval = y_true, p_item, r_item, f1_item

    micro_p = float(np.mean(p_eval))
    micro_r = float(np.mean(r_eval))
    micro_f1 = (2.0 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    # Group by true event
    event_indices: Dict[Any, list[int]] = defaultdict(list)
    for idx, yt in enumerate(y_eval.tolist()):
        event_indices[yt].append(idx)

    ev_ps, ev_rs, ev_f1s, ev_item_f1s = [], [], [], []
    for idxs in event_indices.values():
        ep = float(np.mean(p_eval[idxs]))
        er = float(np.mean(r_eval[idxs]))
        ef1 = (2.0 * ep * er / (ep + er)) if (ep + er) > 0 else 0.0
        ev_ps.append(ep)
        ev_rs.append(er)
        ev_f1s.append(ef1)
        ev_item_f1s.append(float(np.mean(f1_eval[idxs])))

    return {
        "micro_b3_p": micro_p,
        "micro_b3_r": micro_r,
        "micro_b3_f1": micro_f1,
        "macro_b3_p": float(np.mean(ev_ps)),
        "macro_b3_r": float(np.mean(ev_rs)),
        "macro_b3_f1": float(np.mean(ev_f1s)),
        "macro_b3_item_f1": float(np.mean(ev_item_f1s)),
        "num_items": int(len(y_eval)),
        "num_events": int(len(event_indices)),
    }


def pairwise_error_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Compute exact TP, False Merge pairs, and False Split pairs in O(N)."""
    y_true = np.asarray(y_true).tolist()
    y_pred = np.asarray(y_pred).tolist()

    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    joint_counts = Counter(zip(y_true, y_pred))

    total_true_pairs = sum(_comb2(c) for c in true_counts.values())
    total_pred_pairs = sum(_comb2(c) for c in pred_counts.values())
    tp = sum(_comb2(c) for c in joint_counts.values())

    false_merges = total_pred_pairs - tp
    false_splits = total_true_pairs - tp
    return {
        "total_true_pairs": int(total_true_pairs),
        "total_pred_pairs": int(total_pred_pairs),
        "tp_pairs": int(tp),
        "false_merge_pairs": int(false_merges),
        "false_split_pairs": int(false_splits),
    }


def pairwise_error_transition(
    y_true: np.ndarray, y_before: np.ndarray, y_after: np.ndarray
) -> Dict[str, int]:
    """Exact O(N) decomposition of resolved vs. newly introduced false merges & splits."""
    yt = np.asarray(y_true).tolist()
    yb = np.asarray(y_before).tolist()
    ya = np.asarray(y_after).tolist()

    before = pairwise_error_counts(y_true, y_before)
    after = pairwise_error_counts(y_true, y_after)

    both_pred_counts = Counter(zip(yb, ya))
    both_joint_counts = Counter(zip(yt, yb, ya))

    both_pred_pairs = sum(_comb2(c) for c in both_pred_counts.values())
    tp_both = sum(_comb2(c) for c in both_joint_counts.values())
    fm_both = both_pred_pairs - tp_both

    resolved_fm = before["false_merge_pairs"] - fm_both
    introduced_fm = after["false_merge_pairs"] - fm_both
    resolved_fs = after["tp_pairs"] - tp_both
    introduced_fs = before["tp_pairs"] - tp_both

    return {
        "before_false_merges": before["false_merge_pairs"],
        "after_false_merges": after["false_merge_pairs"],
        "resolved_false_merges": int(resolved_fm),
        "introduced_false_merges": int(introduced_fm),
        "net_false_merge_reduction": int(resolved_fm - introduced_fm),
        "before_false_splits": before["false_split_pairs"],
        "after_false_splits": after["false_split_pairs"],
        "resolved_false_splits": int(resolved_fs),
        "introduced_false_splits": int(introduced_fs),
        "net_false_split_reduction": int(resolved_fs - introduced_fs),
    }


def stratified_tail_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tail_max_size: int = 10,
    head_min_size: int = 50,
) -> Dict[str, Any]:
    """Evaluate Head / Medium / Tail events separately plus Tail Discovery Recall."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())
    joint_counts = Counter(zip(y_true.tolist(), y_pred.tolist()))

    p_item, r_item, _ = item_bcubed_scores(y_true, y_pred)

    strata_events: Dict[str, list[Any]] = {"tail": [], "medium": [], "head": []}
    for ev, cnt in true_counts.items():
        if cnt <= tail_max_size:
            strata_events["tail"].append(ev)
        elif cnt >= head_min_size:
            strata_events["head"].append(ev)
        else:
            strata_events["medium"].append(ev)

    # Map event -> item indices
    ev_to_idx: Dict[Any, list[int]] = defaultdict(list)
    for idx, yt in enumerate(y_true.tolist()):
        ev_to_idx[yt].append(idx)

    # Best matching cluster per event for discovery recall (purity >= 0.5 & recall >= 0.5)
    ev_best_match: Dict[Any, Tuple[float, float]] = {}
    for (yt, yp), jc in joint_counts.items():
        pur = jc / pred_counts[yp]
        rec = jc / true_counts[yt]
        prev = ev_best_match.get(yt, (0.0, 0.0))
        if min(pur, rec) > min(prev[0], prev[1]):
            ev_best_match[yt] = (pur, rec)

    out: Dict[str, Any] = {}
    for name, evs in strata_events.items():
        if not evs:
            out[name] = {"num_events": 0, "macro_b3_f1": 0.0, "discovery_recall": 0.0}
            continue
        f1s = []
        discovered = 0
        for ev in evs:
            idxs = ev_to_idx[ev]
            ep = float(np.mean(p_item[idxs]))
            er = float(np.mean(r_item[idxs]))
            f1s.append((2.0 * ep * er / (ep + er)) if (ep + er) > 0 else 0.0)
            pur, rec = ev_best_match.get(ev, (0.0, 0.0))
            if pur >= 0.5 and rec >= 0.5:
                discovered += 1
        out[name] = {
            "num_events": len(evs),
            "macro_b3_f1": float(np.mean(f1s)),
            "discovery_recall": discovered / len(evs),
        }
    return out


def full_partition_audit(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_before: Optional[Sequence[Any]] = None,
    queried_mask: Optional[Sequence[bool]] = None,
    tail_max_size: int = 10,
    head_min_size: int = 50,
) -> Dict[str, Any]:
    """Comprehensive Gate A partition audit report."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    report: Dict[str, Any] = {
        "n_posts": int(len(yt)),
        "n_true_events": int(len(set(yt.tolist()))),
        "n_pred_clusters": int(len(set(yp.tolist()))),
        "nmi": float(normalized_mutual_info_score(yt, yp)),
        "ami": float(adjusted_mutual_info_score(yt, yp)),
        "ari": float(adjusted_rand_score(yt, yp)),
        "bcubed_full": event_macro_bcubed(yt, yp),
        "pairwise_errors": pairwise_error_counts(yt, yp),
        "stratified": stratified_tail_audit(yt, yp, tail_max_size, head_min_size),
    }

    if y_before is not None:
        yb = np.asarray(y_before)
        report["delta_ami"] = report["ami"] - float(adjusted_mutual_info_score(yt, yb))
        before_b3 = event_macro_bcubed(yt, yb)
        report["delta_macro_b3_f1"] = report["bcubed_full"]["macro_b3_f1"] - before_b3["macro_b3_f1"]
        report["delta_micro_b3_f1"] = report["bcubed_full"]["micro_b3_f1"] - before_b3["micro_b3_f1"]
        report["error_transition"] = pairwise_error_transition(yt, yb, yp)

        if queried_mask is not None:
            qmask = np.asarray(queried_mask, dtype=bool)
            unq_mask = ~qmask
            _, _, f1_b = item_bcubed_scores(yt, yb)
            _, _, f1_a = item_bcubed_scores(yt, yp)

            unq_before = event_macro_bcubed(yt, yb, subset_mask=unq_mask)
            unq_after = event_macro_bcubed(yt, yp, subset_mask=unq_mask)
            q_before = event_macro_bcubed(yt, yb, subset_mask=qmask)
            q_after = event_macro_bcubed(yt, yp, subset_mask=qmask)

            diff_unq = f1_a[unq_mask] - f1_b[unq_mask]
            diff_q = f1_a[qmask] - f1_b[qmask]
            tol = 1e-9

            report["propagation_audit"] = {
                "n_queried": int(qmask.sum()),
                "n_unqueried": int(unq_mask.sum()),
                "queried_delta_micro_b3_f1": q_after["micro_b3_f1"] - q_before["micro_b3_f1"],
                "queried_delta_macro_b3_f1": q_after["macro_b3_f1"] - q_before["macro_b3_f1"],
                "queried_improved_posts": int(np.sum(diff_q > tol)),
                "queried_degraded_posts": int(np.sum(diff_q < -tol)),
                "unqueried_delta_micro_b3_f1": unq_after["micro_b3_f1"] - unq_before["micro_b3_f1"],
                "unqueried_delta_macro_b3_f1": unq_after["macro_b3_f1"] - unq_before["macro_b3_f1"],
                "unqueried_improved_posts": int(np.sum(diff_unq > tol)),
                "unqueried_degraded_posts": int(np.sum(diff_unq < -tol)),
                "unqueried_net_improved_posts": int(
                    np.sum(diff_unq > tol) - np.sum(diff_unq < -tol)
                ),
            }

    return report
