"""Reusable per-action LLM decision confusion-matrix & error auditor.

Supports both Gate A dev runs and full streaming held-out runs, including
CACHED_REPLAY entries backed by shared persistent response caches.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import build_message_graph
from core_shared.se_solver import solve_2d_se
from trackA_sug_sed.high_leverage_router import build_cluster_summaries
from trackA_sug_sed.run_gateA_real_llm import _select_pool


def _load_hash_map(paths: Sequence[Path]) -> Dict[str, dict]:
    hmap: Dict[str, dict] = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = o.get("request_hash")
            resp = o.get("response", {})
            if h and not (isinstance(resp, dict) and "error" in resp and o.get("prompt_tokens", 0) == 0):
                hmap[h] = resp
    return hmap


def audit_run(
    run_dir: Path,
    dataset_name: str = "Event2012",
    strategy: str = "sug_router",
    block_ids: Sequence[int] = (2, 3, 4, 5),
    embed_model: Optional[str] = None,
    shared_cache: Optional[Path] = None,
    verbose_errors: bool = False,
) -> Dict[str, dict]:
    embed_model = embed_model or (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if dataset_name == "Event2018"
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    safe_embed = embed_model.replace("/", "__")

    base = run_dir / f"ledger_{strategy}"
    cache_files = [base / "responses.jsonl"]
    if shared_cache is not None:
        cache_files.insert(0, shared_cache)
    else:
        auto_cache = ROOT / "results" / "llm_cache" / f"{dataset_name}_Qwen__Qwen3-8B" / "responses.jsonl"
        if auto_cache.exists():
            cache_files.insert(0, auto_cache)

    hmap = _load_hash_map(cache_files)
    jmap: Dict[tuple, dict] = {}
    status_cnt = Counter()
    for line in (base / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        status_cnt[o.get("status", "UNKNOWN")] += 1
        resp = hmap.get(o.get("request_hash"))
        if resp is not None:
            jmap[(int(o["block_id"]), o["request_id"])] = resp

    ds_full = load_socialed_npy(dataset_name, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
    cache = ROOT / "cache" / "embeddings"

    cm_m, cm_s, cm_n = Counter(), Counter(), Counter()
    for bid in block_ids:
        _, bidx = blocks[bid]
        sub = ds_full.subset(bidx)
        emb = np.load(cache / f"{dataset_name}_M{bid:02d}_{safe_embed}.npy").astype(np.float32)
        g = build_message_graph(sub, emb, knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.5)
        labels, _ = solve_2d_se(g.adj, refine_moves=True)
        summ = build_cluster_summaries(sub, emb, labels)
        maj = {c: Counter(sub.labels[s.members].tolist()).most_common(1)[0][0] for c, s in summ.items()}
        pool = _select_pool(strategy, sub, g, labels, 120, seed=42 + bid)
        for q in pool:
            if (bid, q.query_id) not in jmap:
                continue
            r = jmap[(bid, q.query_id)]
            if "error" in r:
                continue
            if q.query_type == "merge_pair":
                ts = maj[q.merge_c1] == maj[q.merge_c2]
                pred = bool(r["same_event"]) if "same_event" in r else (str(r.get("action")).upper() == "MERGE")
                cm_m[(ts, pred)] += 1
                if verbose_errors and ts != pred:
                    print(f"  [M{bid:02d} MERGE ERR] {q.query_id} truth={ts} pred={pred} | A={r.get('event_a')!r} | B={r.get('event_b')!r}")
            elif q.query_type == "seam":
                m1 = Counter(sub.labels[q.seam_sub1].tolist()).most_common(1)[0][0]
                m2 = Counter(sub.labels[q.seam_sub2].tolist()).most_common(1)[0][0]
                td = m1 != m2
                pred_split = (not bool(r["same_event"])) if "same_event" in r else (str(r.get("action")).upper() == "SPLIT")
                cm_s[(td, pred_split)] += 1
                if verbose_errors and td != pred_split:
                    print(f"  [M{bid:02d} SEAM  ERR] {q.query_id} true_diff={td} pred_split={pred_split} | A={r.get('event_a')!r} | B={r.get('event_b')!r}")
            else:
                node = q.primary_nodes[0]
                act = str(r.get("action", "NOOP")).upper()
                tgt = r.get("target_cluster")
                true_ev = int(sub.labels[node])
                if act == "ASSIGN" and tgt is not None:
                    try:
                        tgt_i = int(tgt)
                    except Exception:
                        continue
                    ok = maj.get(tgt_i) == true_ev
                    changed = tgt_i != int(labels[node])
                    cm_n[("ASSIGN_CHG" if changed else "ASSIGN_KEEP", ok)] += 1
                else:
                    cm_n[(act, True)] += 1

    def _prf(cm: Counter) -> Dict[str, float]:
        tp = cm[(True, True)]
        fp = cm[(False, True)]
        tn = cm[(False, False)]
        fn = cm[(True, False)]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "precision": round(prec, 4), "recall": round(rec, 4), "specificity": round(spec, 4)}

    m_stats = _prf(cm_m)
    s_stats = _prf(cm_s)
    print(f"\n================ AUDIT: {run_dir.name} ({dataset_name} / {strategy}) ================")
    print("  Statuses :", dict(status_cnt))
    print("  MERGE    :", m_stats)
    print("  SEAM     :", s_stats)
    if cm_n:
        print("  NODE     :", {f"{k[0]}_{k[1]}": v for k, v in cm_n.items()})
    return {"statuses": dict(status_cnt), "merge": m_stats, "seam": s_stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="Event2012")
    parser.add_argument("--strategy", default="sug_router")
    parser.add_argument("--blocks", default="2,3,4,5")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bids = [int(x.strip()) for x in args.blocks.split(",") if x.strip()]
    audit_run(args.run_dir, dataset_name=args.dataset, strategy=args.strategy, block_ids=bids, verbose_errors=args.verbose)


if __name__ == "__main__":
    main()
