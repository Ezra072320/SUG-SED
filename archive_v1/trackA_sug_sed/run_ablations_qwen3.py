"""Comprehensive Controlled Ablation Suite on Held-Out Streaming Blocks (Qwen/Qwen3-8B).

Evaluates 5 surgical single-factor ablations against the frozen SUG-Meso (Direct)
and SUG-Meso (Repair) reference runs across ALL 27 unseen held-out chronological
blocks (Event2012 M_06..M_21 [16 blocks] & Event2018 M_06..M_16 [11 blocks]) under
identical 10k real tokens/block budgets and strict real-token accounting:

  1. `merge_only`        : 100% MERGE_PAIR, 0% SPLIT_SEAM (removes false-merge splitting)
  2. `seam_only`         : 0% MERGE_PAIR, 100% SPLIT_SEAM (removes false-split merging)
  3. `fiedler_only_seam` : Balanced spectral Fiedler bipartition instead of spherical
                           KMeans++ unbalanced minority-pocket seam detection
  4. `no_extract_compare`: Action-First direct JSON (`action`, `confidence`) instead of
                           Balanced Extract-Then-Compare (`event_a`, `event_b`, `same_event`)
  5. `uniform_budget`    : Naive fixed 10k tokens/block quota (no adaptive block-weight /
                           uncertainty scaling and no cumulative balance carryover)

Reuses the persistent shared valid-response cache (`results/llm_cache/...`) while
honestly charging every executed/replayed query's exact prompt+completion tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset, load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import MessageGraph, build_message_graph
from core_shared.key_loader import load_provider_key
from core_shared.metrics_audit import full_partition_audit
from core_shared.se_solver import solve_2d_se
from core_shared.token_ledger import CumulativeTokenBucket, TokenLedger, canonical_request_hash
from trackA_sug_sed.high_leverage_router import (
    ClusterSummary,
    QueryItem,
    _bipartition_cluster,
    build_cluster_summaries,
    detect_merge_pair_queries,
    select_sug_queries,
)
from trackA_sug_sed.llm_client import BudgetedLLMClient, _extract_json, build_query_prompt
from trackA_sug_sed.run_gateA_oracle_diagnostic import load_or_extract_block_embeddings
from trackA_sug_sed.topology_surgery import (
    LLMDecision,
    apply_direct_assignment,
)


def _detect_fiedler_only_seams(
    ds: SocialDataset,
    graph: MessageGraph,
    summaries: Dict[int, ClusterSummary],
    min_cluster_size: int = 10,
    min_pocket_size: int = 4,
    max_conductance: float = 0.50,
    max_medoids: int = 3,
) -> List[QueryItem]:
    """Ablation seam detector: balanced spectral Fiedler bipartition ONLY (no spherical KMeans++)."""
    adj = graph.adj
    emb = graph.embeddings
    non_ent = graph.non_entity_tfidf
    queries: List[QueryItem] = []

    for cid, summ in summaries.items():
        if summ.size < min_cluster_size:
            continue
        members = np.asarray(summ.members, dtype=np.int64)
        sub_adj = adj[members][:, members].tocsr()
        sub_emb = emb[members]

        fg1, fg2, _ = _bipartition_cluster(sub_adj, sub_emb)
        if len(fg1) < min_pocket_size or len(fg2) < min_pocket_size:
            continue
        if len(fg1) < len(fg2):
            fg1, fg2 = fg2, fg1

        g_main = members[fg1]
        g_pock = members[fg2]

        p_main = emb[g_main].mean(axis=0)
        p_pock = emb[g_pock].mean(axis=0)
        p_main = p_main / max(float(np.linalg.norm(p_main)), 1e-12)
        p_pock = p_pock / max(float(np.linalg.norm(p_pock)), 1e-12)

        intra_pock = float(np.mean(emb[g_pock] @ p_pock))
        cross_sim = float(np.dot(p_main, p_pock))
        if len(g_pock) < 6 and intra_pock > 0.96:
            continue

        sem_sep = max(0.0, intra_pock - cross_sim)
        if sem_sep < 0.08:
            continue

        vol_pock = float(sub_adj[fg2].sum())
        cut_w = float(sub_adj[fg2][:, fg1].sum())
        cond = cut_w / max(vol_pock, 1e-12) if vol_pock > 1e-12 else 0.0
        if cond > max_conductance:
            continue

        sub_coo = sub_adj.tocoo()
        in_pock = np.zeros(len(members), dtype=bool)
        in_pock[fg2] = True
        cut_mask = in_pock[sub_coo.row] & (~in_pock[sub_coo.col])
        if np.any(cut_mask):
            cu = members[sub_coo.row[cut_mask]]
            cv = members[sub_coo.col[cut_mask]]
            cut_edges = list(zip(cu.tolist(), cv.tolist()))
            if non_ent.shape[1] > 0:
                phi_non_ent = float(np.asarray(non_ent[cu].multiply(non_ent[cv]).sum(axis=1)).mean())
            else:
                phi_non_ent = 0.0
        else:
            cut_edges = []
            phi_non_ent = 0.0

        score = (
            sem_sep
            * ((1.0 - min(0.95, cond)) ** 2)
            * max(0.15, 1.0 - phi_non_ent)
            * float(np.log1p(len(g_pock)))
        )
        m_main = [int(g_main[k]) for k in np.argsort(-(emb[g_main] @ p_main))[:max_medoids]]
        m_pock = [int(g_pock[k]) for k in np.argsort(-(emb[g_pock] @ p_pock))[:max_medoids]]
        queries.append(
            QueryItem(
                query_id=f"fiedler_seam_c{cid}",
                strategy="fiedler_only_seam",
                query_type="seam",
                score=score,
                primary_nodes=m_main[:1] + m_pock[:1],
                affected_group=summ.members,
                candidate_clusters=[cid],
                seam_cluster_id=cid,
                seam_sub1=g_main.tolist(),
                seam_sub2=g_pock.tolist(),
                seam_sub1_medoids=m_main,
                seam_sub2_medoids=m_pock,
                seam_cut_edges=cut_edges,
            )
        )
    queries.sort(key=lambda q: -q.score)
    return queries


def _select_fiedler_meso_pool(
    ds: SocialDataset,
    graph: MessageGraph,
    labels: np.ndarray,
    budget_queries: int = 120,
    merge_frac: float = 0.65,
    seam_frac: float = 0.35,
) -> List[QueryItem]:
    summaries = build_cluster_summaries(ds, graph.embeddings, labels)
    merges = detect_merge_pair_queries(ds, graph, labels, summaries)
    seams = _detect_fiedler_only_seams(ds, graph, summaries)
    queues = [("merge", merges, merge_frac), ("seam", seams, seam_frac)]
    ptrs = {"merge": 0, "seam": 0}
    counts = {"merge": 0, "seam": 0}
    selected: List[QueryItem] = []
    while len(selected) < budget_queries:
        best_name = None
        best_def = -1e9
        tot = max(1, len(selected))
        for name, _ql, tf in queues:
            if ptrs[name] >= len(_ql):
                continue
            d = tf - (counts[name] / tot)
            if d > best_def:
                best_def = d
                best_name = name
        if best_name is None:
            break
        ql = merges if best_name == "merge" else seams
        selected.append(ql[ptrs[best_name]])
        ptrs[best_name] += 1
        counts[best_name] += 1
    return selected


def _build_action_first_prompt(
    q: QueryItem,
    ds: SocialDataset,
    summaries: Dict[int, ClusterSummary],
) -> Tuple[str, str]:
    """Ablation prompt: identical event rules & tweet evidence, but Action-First JSON without event extraction."""
    sys_msg = (
        "You decide whether social-media posts or post groups refer to the SAME specific real-world event.\n"
        "Rules:\n"
        "- SAME EVENT if both cover the same specific incident, match, announcement, or developing story, "
        "even when they highlight different details, updates, players/angles, or reactions "
        "(e.g., 'Team X beats Team Y 4-2' and 'Player Z scores for Team X against Team Y'; "
        "'Official resigns' and 'Official quits over scandal').\n"
        "- DIFFERENT EVENT if they cover different incidents, different matches, different countries, "
        "or different main subjects even if sharing a broad topic "
        "(e.g., 'Team X vs Team Y' and 'Team A vs Team B'; 'Scandal at Org 1' and 'Scandal at Org 2').\n"
        "Respond ONLY with strict JSON containing action and confidence."
    )
    if q.query_type == "merge_pair":
        s1 = "\n".join(f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}" for m in q.merge_c1_medoids[:3])
        s2 = "\n".join(f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}" for m in q.merge_c2_medoids[:3])
        c1_kw = ", ".join(summaries[q.merge_c1].top_keywords[:5]) if q.merge_c1 in summaries else ""
        c2_kw = ", ".join(summaries[q.merge_c2].top_keywords[:5]) if q.merge_c2 in summaries else ""
        user_msg = (
            f"Group A (size={summaries[q.merge_c1].size}, keywords: {c1_kw}):\n{s1}\n\n"
            f"Group B (size={summaries[q.merge_c2].size}, keywords: {c2_kw}):\n{s2}\n\n"
            "Should Group A and Group B be MERGED into the same event or kept SEPARATE?\n"
            'Return strict JSON: {"action": "MERGE" | "SEPARATE", "confidence": <float 0.0 to 1.0>}'
        )
        return sys_msg, user_msg
    if q.query_type == "seam":
        s1 = "\n".join(f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}" for m in q.seam_sub1_medoids[:3])
        s2 = "\n".join(f"  - {ds.texts[m].replace(chr(10), ' ').strip()[:140]}" for m in q.seam_sub2_medoids[:3])
        user_msg = (
            f"Subgroup A (size={len(q.seam_sub1)}):\n{s1}\n\n"
            f"Subgroup B (size={len(q.seam_sub2)}):\n{s2}\n\n"
            "Should this cluster be SPLIT into two separate events or KEEP_MERGED as one event?\n"
            'Return strict JSON: {"action": "SPLIT" | "KEEP_MERGED", "confidence": <float 0.0 to 1.0>}'
        )
        return sys_msg, user_msg
    return build_query_prompt(q, ds, summaries)


class UniformFixedBucket:
    """Ablation budget controller: fixed B/T token cap per block, zero adaptive scaling, zero carryover."""

    def __init__(self, total_token_budget: int, num_blocks: int) -> None:
        self.total_token_budget = int(total_token_budget)
        self.num_blocks = int(num_blocks)
        self.fixed_cap = int(total_token_budget // num_blocks)
        self.current_block_cap = 0
        self.current_block_spent = 0
        self.cumulative_spent = 0

    def start_block(self, block_weight: float = 0.0, mean_uncertainty: float = 1.0) -> int:
        rem = max(0, self.total_token_budget - self.cumulative_spent)
        self.current_block_cap = min(self.fixed_cap, rem)
        self.current_block_spent = 0
        return self.current_block_cap

    def can_spend(self, estimated_tokens: int) -> bool:
        est = int(estimated_tokens)
        return (
            self.current_block_spent + est <= self.current_block_cap
            and self.cumulative_spent + est <= self.total_token_budget
        )

    def record_spend(self, prompt_tokens: int, completion_tokens: int) -> None:
        tot = int(prompt_tokens) + int(completion_tokens)
        self.current_block_spent += tot
        self.cumulative_spent += tot


def _execute_queries_with_prompt_fn(
    client: BudgetedLLMClient,
    queries: List[QueryItem],
    ds: SocialDataset,
    summaries: Dict[int, ClusterSummary],
    block_id: int,
    bucket: Any,
    prompt_fn,
    strategy_tag: str,
) -> List[LLMDecision]:
    """Execute queries under arbitrary prompt builder with identical real-token ledger billing."""
    decisions: List[LLMDecision] = []
    for q in queries:
        sys_msg, user_msg = prompt_fn(q, ds, summaries)
        est_tokens = int((len(sys_msg) + len(user_msg)) / 3.5) + 120
        if not bucket.can_spend(est_tokens):
            break

        req_payload = {
            "model": client.model,
            "temperature": client.temperature,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        }
        req_hash = canonical_request_hash(req_payload)
        cached = client.ledger.lookup_cache(req_hash)

        if cached is not None:
            parsed = cached["response"]
            pt, ct = int(cached["prompt_tokens"]), int(cached["completion_tokens"])
            client.ledger.log_call(
                request_id=q.query_id,
                block_id=block_id,
                strategy=strategy_tag,
                query_type=q.query_type,
                model=client.model,
                request_hash=req_hash,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=0.0,
                status="CACHED_REPLAY",
                response_payload=parsed,
                from_cache=True,
            )
        else:
            pt, ct, latency = 0, 0, 0.0
            try:
                raw_text, pt, ct, latency, finish_reason = client._call_http(req_payload["messages"])
                try:
                    parsed = _extract_json(raw_text)
                    status = "OK"
                except Exception as parse_exc:
                    parsed = {
                        "action": "NOOP",
                        "confidence": 0.0,
                        "finish_reason": finish_reason,
                        "raw_snippet": raw_text[:120],
                        "error": str(parse_exc),
                    }
                    status = f"PARSE_ERROR:{type(parse_exc).__name__}"
            except Exception as http_exc:
                parsed = {"action": "NOOP", "confidence": 0.0, "error": str(http_exc)}
                status = f"HTTP_ERROR:{type(http_exc).__name__}"

            client.ledger.log_call(
                request_id=q.query_id,
                block_id=block_id,
                strategy=strategy_tag,
                query_type=q.query_type,
                model=client.model,
                request_hash=req_hash,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=latency,
                status=status,
                response_payload=parsed,
                from_cache=False,
            )

        bucket.record_spend(pt, ct)

        action = "NOOP"
        conf = 1.0
        if "error" in parsed:
            action, conf = "NOOP", 0.0
        elif q.query_type == "merge_pair":
            if "same_event" in parsed:
                action = "MERGE" if bool(parsed["same_event"]) else "SEPARATE"
                conf = 1.0
            else:
                raw_act = str(parsed.get("action", "NOOP")).upper().strip()
                if raw_act in {"MERGE", "MERGED", "SAME"}:
                    action = "MERGE"
                elif raw_act in {"SEPARATE", "DIFFERENT"}:
                    action = "SEPARATE"
                try:
                    conf = float(parsed.get("confidence", 1.0))
                except (ValueError, TypeError):
                    conf = 1.0
        elif q.query_type == "seam":
            if "same_event" in parsed:
                action = "KEEP_MERGED" if bool(parsed["same_event"]) else "SPLIT"
                conf = 1.0
            else:
                raw_act = str(parsed.get("action", "NOOP")).upper().strip()
                if raw_act == "SPLIT":
                    action = "SPLIT"
                elif raw_act in {"KEEP_MERGED", "KEEP", "MERGE"}:
                    action = "KEEP_MERGED"
                try:
                    conf = float(parsed.get("confidence", 1.0))
                except (ValueError, TypeError):
                    conf = 1.0

        decisions.append(
            LLMDecision(
                query_id=q.query_id,
                query_type=q.query_type,
                action=action,
                target_cluster=q.seam_cluster_id if q.query_type == "seam" else None,
                confidence=conf,
                prompt_tokens=pt,
                completion_tokens=ct,
            )
        )
    return decisions


def _audit_decisions_cm(
    sub_ds: SocialDataset,
    summaries: Dict[int, ClusterSummary],
    used_queries: List[QueryItem],
    decisions: List[LLMDecision],
    conf_threshold: float,
    cm_m: Counter,
    cm_s: Counter,
) -> None:
    maj = {c: Counter(sub_ds.labels[s.members].tolist()).most_common(1)[0][0] for c, s in summaries.items()}
    qmap = {q.query_id: q for q in used_queries}
    for dec in decisions:
        q = qmap[dec.query_id]
        if q.query_type == "merge_pair":
            ts = maj[q.merge_c1] == maj[q.merge_c2]
            pred = dec.action == "MERGE" and dec.confidence >= conf_threshold
            cm_m[(ts, pred)] += 1
        elif q.query_type == "seam":
            m1 = Counter(sub_ds.labels[q.seam_sub1].tolist()).most_common(1)[0][0]
            m2 = Counter(sub_ds.labels[q.seam_sub2].tolist()).most_common(1)[0][0]
            td = m1 != m2
            pred = dec.action == "SPLIT" and dec.confidence >= conf_threshold
            cm_s[(td, pred)] += 1


def _prf(cm: Counter) -> Dict[str, Any]:
    tp, fp, tn, fn = cm[(True, True)], cm[(False, True)], cm[(False, False)], cm[(True, False)]
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": round(tp / max(1, tp + fp), 4),
        "recall": round(tp / max(1, tp + fn), 4),
        "specificity": round(tn / max(1, tn + fp), 4),
    }


def run_dataset_ablations(
    dataset: str,
    model_name: str,
    provider: str,
    tokens_per_block: int,
    conf_threshold: float,
    dry_run: bool,
) -> Dict[str, Any]:
    embed_model = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if dataset == "Event2018"
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    ds_full = load_socialed_npy(dataset, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
    eval_block_ids = list(range(6, len(blocks)))
    total_budget = tokens_per_block * len(eval_block_ids)

    if not dry_run:
        prov_url, prov_model, api_key = load_provider_key(provider)
        base_url = prov_url
        model_name = model_name or prov_model
    else:
        base_url, api_key = "dry-run", ""

    safe_model = model_name.replace("/", "__")
    shared_cache_file = ROOT / "results" / "llm_cache" / f"{dataset}_{safe_model}" / "responses.jsonl"
    out_dir = ROOT / "results" / "ablations" / f"{dataset}_heldout_qwen3_8b"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n==================== ABLATIONS: {dataset} Held-Out M_06..M_{eval_block_ids[-1]:02d} "
        f"({len(eval_block_ids)} blocks, B={total_budget:,}) ===================="
    )
    block_cache: Dict[int, Dict[str, Any]] = {}
    eval_total_posts = 0
    for bid in eval_block_ids:
        bname, bidx = blocks[bid]
        sub_ds = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        emb = load_or_extract_block_embeddings(dataset, bid, sub_ds.texts, model_name=embed_model)
        graph = build_message_graph(sub_ds, emb, knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.5)
        labels_init, _ = solve_2d_se(graph.adj, refine_moves=True)
        summaries = build_cluster_summaries(sub_ds, emb, labels_init)
        init_audit = full_partition_audit(sub_ds.labels, labels_init)
        block_cache[bid] = {
            "name": bname,
            "ds": sub_ds,
            "graph": graph,
            "labels_init": labels_init,
            "summaries": summaries,
            "init_audit": init_audit,
        }
        eval_total_posts += len(sub_ds)

    ablations = [
        "merge_only",
        "seam_only",
        "fiedler_only_seam",
        "no_extract_compare",
        "uniform_budget",
    ]

    def _get_pool(abl: str, sub_ds: SocialDataset, graph: MessageGraph, labels_init: np.ndarray) -> List[QueryItem]:
        if abl == "merge_only":
            return select_sug_queries(sub_ds, graph, labels_init, 120, merge_frac=1.0, seam_frac=0.0, bridge_frac=0.0)
        if abl == "seam_only":
            return select_sug_queries(sub_ds, graph, labels_init, 120, merge_frac=0.0, seam_frac=1.0, bridge_frac=0.0)
        if abl == "fiedler_only_seam":
            return _select_fiedler_meso_pool(sub_ds, graph, labels_init, 120, merge_frac=0.65, seam_frac=0.35)
        return select_sug_queries(sub_ds, graph, labels_init, 120, merge_frac=0.65, seam_frac=0.35, bridge_frac=0.0)

    if dry_run:
        probe = TokenLedger(out_dir / "_dry_probe", shared_cache_file=shared_cache_file)
        for abl in ablations:
            bucket: Any = (
                UniformFixedBucket(total_budget, len(eval_block_ids))
                if abl == "uniform_budget"
                else CumulativeTokenBucket(total_budget, len(eval_block_ids))
            )
            pfn = _build_action_first_prompt if abl == "no_extract_compare" else build_query_prompt
            tot_q = tot_c = tot_l = tot_tok = 0
            for bid in eval_block_ids:
                bc = block_cache[bid]
                pool = _get_pool(abl, bc["ds"], bc["graph"], bc["labels_init"])
                mean_u = float(np.mean([q.score for q in pool])) if pool else 1.0
                bucket.start_block(len(bc["ds"]) / eval_total_posts, mean_u)
                for q in pool:
                    sm, um = pfn(q, bc["ds"], bc["summaries"])
                    est = int((len(sm) + len(um)) / 3.5) + 120
                    if not bucket.can_spend(est):
                        break
                    rh = canonical_request_hash(
                        {
                            "model": model_name,
                            "temperature": 0.0,
                            "messages": [{"role": "system", "content": sm}, {"role": "user", "content": um}],
                        }
                    )
                    hit = probe.lookup_cache(rh)
                    if hit is not None:
                        pt, ct = int(hit["prompt_tokens"]), int(hit["completion_tokens"])
                        tot_c += 1
                        bucket.record_spend(pt, ct)
                        tot_tok += pt + ct
                    else:
                        tot_l += 1
                        bucket.record_spend(est - 60, 60)
                        tot_tok += est
                    tot_q += 1
            print(
                f"  [DRY-RUN] {abl:20s}: total_q={tot_q:3d} (cached={tot_c:3d}, live={tot_l:3d}), est_tok={tot_tok:6d}/{total_budget}"
            )
        return {}

    ref_path = ROOT / "results" / "streaming" / f"{dataset}_heldout_qwen3_8b" / "streaming_report.json"
    ref_rep = json.loads(ref_path.read_text(encoding="utf-8"))

    out_report: Dict[str, Any] = {
        "dataset": dataset,
        "model": model_name,
        "eval_blocks": eval_block_ids,
        "budget": total_budget,
        "init_eval_mean": ref_rep["init_eval_mean"],
        "reference": {
            "sug_meso_direct": ref_rep["mean_comparison"]["sug_meso_only_direct"],
            "sug_meso_repair": ref_rep["mean_comparison"]["sug_meso_only_repair"],
            "sug_router_repair": ref_rep["mean_comparison"]["sug_router_repair"],
        },
        "ablations": {},
    }

    for abl in ablations:
        t0 = time.time()
        led_dir = out_dir / f"ledger_{abl}"
        if (led_dir / "journal.jsonl").exists():
            (led_dir / "journal.jsonl").unlink()
        ledger = TokenLedger(led_dir, shared_cache_file=shared_cache_file)
        bucket = (
            UniformFixedBucket(total_budget, len(eval_block_ids))
            if abl == "uniform_budget"
            else CumulativeTokenBucket(total_budget, len(eval_block_ids))
        )
        client = BudgetedLLMClient(model=model_name, ledger=ledger, base_url=base_url, api_key=api_key)
        pfn = _build_action_first_prompt if abl == "no_extract_compare" else build_query_prompt

        rows: List[Dict[str, float]] = []
        cm_m, cm_s = Counter(), Counter()
        block_details: Dict[str, Any] = {}

        for bid in eval_block_ids:
            bc = block_cache[bid]
            sub_ds, graph, labels_init, summaries = bc["ds"], bc["graph"], bc["labels_init"], bc["summaries"]
            pool = _get_pool(abl, sub_ds, graph, labels_init)
            mean_u = float(np.mean([q.score for q in pool])) if pool else 1.0
            cap_tok = bucket.start_block(len(sub_ds) / eval_total_posts, mean_u)

            decisions = _execute_queries_with_prompt_fn(
                client, pool, sub_ds, summaries, bid, bucket, pfn, abl
            )
            dec_ids = {d.query_id for d in decisions}
            used_queries = [q for q in pool if q.query_id in dec_ids]
            _audit_decisions_cm(sub_ds, summaries, used_queries, decisions, conf_threshold, cm_m, cm_s)

            lab_dir, qmask = apply_direct_assignment(
                labels_init, used_queries, decisions, conf_threshold=conf_threshold
            )
            aud = full_partition_audit(sub_ds.labels, lab_dir, y_before=labels_init, queried_mask=qmask)
            prop = aud["propagation_audit"]
            et = aud["error_transition"]
            btok = sum(d.prompt_tokens + d.completion_tokens for d in decisions)
            rows.append(
                {
                    "queries": float(len(decisions)),
                    "tokens": float(btok),
                    "ami": aud["ami"],
                    "delta_ami": aud["delta_ami"],
                    "nmi": aud["nmi"],
                    "ari": aud["ari"],
                    "macro_b3": aud["bcubed_full"]["macro_b3_f1"],
                    "delta_macro_b3": aud["delta_macro_b3_f1"],
                    "micro_b3": aud["bcubed_full"]["micro_b3_f1"],
                    "delta_micro_b3": aud["delta_micro_b3_f1"],
                    "unq_delta_micro_b3": prop["unqueried_delta_micro_b3_f1"],
                    "unq_delta_macro_b3": prop["unqueried_delta_macro_b3_f1"],
                    "unq_net_posts": float(prop["unqueried_net_improved_posts"]),
                    "net_fm_red": float(et["net_false_merge_reduction"]),
                    "net_fs_red": float(et["net_false_split_reduction"]),
                    "tail_b3": aud["stratified"]["tail"]["macro_b3_f1"],
                    "win_ami": 1.0 if aud["delta_ami"] > 1e-6 else 0.0,
                    "win_macb3": 1.0 if aud["delta_macro_b3_f1"] > 1e-6 else 0.0,
                }
            )
            block_details[f"M_{bid:02d}"] = {
                "n_queries": len(decisions),
                "tokens": btok,
                "cap_tokens": cap_tok,
                "delta_ami": aud["delta_ami"],
                "delta_macro_b3": aud["delta_macro_b3_f1"],
                "unq_net_posts": prop["unqueried_net_improved_posts"],
                "net_fm_red": et["net_false_merge_reduction"],
                "net_fs_red": et["net_false_split_reduction"],
            }

        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()}
        m["total_tokens_all_blocks"] = float(sum(r["tokens"] for r in rows))
        m["blocks_improved_ami"] = int(sum(r["win_ami"] for r in rows))
        m["blocks_improved_macb3"] = int(sum(r["win_macb3"] for r in rows))
        m["n_eval_blocks"] = len(eval_block_ids)

        lsum = ledger.summary()
        st_cnt = Counter()
        for line in (led_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                st_cnt[json.loads(line)["status"]] += 1

        out_report["ablations"][abl] = {
            "mean": m,
            "ledger_summary": lsum,
            "statuses": dict(st_cnt),
            "confusion_merge": _prf(cm_m),
            "confusion_seam": _prf(cm_s),
            "elapsed_sec": round(time.time() - t0, 2),
            "blocks": block_details,
        }
        print(
            f"  [{abl:18s}] tok={m['total_tokens_all_blocks']:6.0f} ({time.time()-t0:5.1f}s) | "
            f"AMI={m['ami']:.4f} ({m['delta_ami']:+.4f}) ARI={m['ari']:.4f} "
            f"MacB3={m['macro_b3']:.4f} ({m['delta_macro_b3']:+.4f}) Tail={m['tail_b3']:.4f} "
            f"UnqMic={m['unq_delta_micro_b3']:+.4f} NetFM={m['net_fm_red']:+7.0f} NetFS={m['net_fs_red']:+7.0f} "
            f"Win={m['blocks_improved_ami']}/{len(eval_block_ids)} | "
            f"M_prec={_prf(cm_m)['precision']:.3f} S_prec={_prf(cm_s)['precision']:.3f}"
        )

    out_json = out_dir / "ablation_report.json"
    out_json.write_text(json.dumps(out_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved ablation JSON: {out_json}")
    return out_report


def compile_ablation_outputs(r12: Dict[str, Any], r18: Dict[str, Any]) -> None:
    """Write Markdown summary, LaTeX ablation table, CSV, and bilingual bar figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_spec = [
        ("sug_meso_direct", "SUG-Meso (Full, Direct)", True),
        ("sug_meso_repair", "SUG-Meso (Branch 2 Repair)", True),
        ("sug_router_repair", "SUG-Router (+Bridge/Seed, Repair)", True),
        ("merge_only", "w/o SPLIT_SEAM (Merge-Only)", False),
        ("seam_only", "w/o MERGE_PAIR (Seam-Only)", False),
        ("fiedler_only_seam", "w/o Minority-Pocket (Fiedler Seam)", False),
        ("no_extract_compare", "w/o Extract-Then-Compare (Action-First)", False),
        ("uniform_budget", "w/o Adaptive Bucket (Uniform 10k/blk)", False),
    ]

    def _get_m(rep: Dict[str, Any], key: str, is_ref: bool) -> Dict[str, Any]:
        return rep["reference"][key] if is_ref else rep["ablations"][key]["mean"]

    md = [
        "# Track A（SUG-SED）双语保留测试集核心消融实验报告（`Qwen/Qwen3-8B`）",
        "",
        "- **评测范围**：与主实验完全一致的全部 27 个未见保留流式块（`Event2012 M_06..M_21` 16 块 + `Event2018 M_06..M_16` 11 块），同 `10k` 真实 Token/块严格累计预算，同 `Qwen/Qwen3-8B`（`enable_thinking=False`）。",
        "- **消融可视化**：[ablation_bilingual_qwen3_8b.png](../figures/ablation_bilingual_qwen3_8b.png) · **LaTeX 表**：[ablation_table_qwen3_8b.tex](./ablation_table_qwen3_8b.tex)",
        "",
    ]

    csv_rows: List[Dict[str, Any]] = []
    for ds_key, ds_label, rep in [
        ("Event2012", "Event2012（英语 · M_06..M_21，16 块）", r12),
        ("Event2018", "Event2018（法语 · M_06..M_16，11 块）", r18),
    ]:
        im = rep["init_eval_mean"]
        md.extend([
            f"## {ds_label}（初始 2D-SE: AMI={im['ami']:.4f}, ARI={im['ari']:.4f}, 宏 B³={im['macro_b3']:.4f}, 长尾 B³={im['tail_b3']:.4f}）",
            "",
            r"| 变体 | 实耗 Token | AMI ($\Delta$) | ARI | 宏 $B^3$ ($\Delta$) | 长尾 $B^3$ | 未查询 $\Delta$微 $B^3$ | 净减 FM / 净减 FS | 胜率 (AMI / 宏 $B^3$) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ])
        for k, name, is_ref in rows_spec:
            m = _get_m(rep, k, is_ref)
            md.append(
                f"| {name} | {m['total_tokens_all_blocks']:,.0f} | "
                f"{m['ami']:.4f} ({m['delta_ami']:+.4f}) | {m['ari']:.4f} | "
                f"{m['macro_b3']:.4f} ({m['delta_macro_b3']:+.4f}) | {m['tail_b3']:.4f} | "
                f"{m['unq_delta_micro_b3']:+.4f} | {m['net_fm_red']:+,.0f} / {m['net_fs_red']:+,.0f} | "
                f"{m['blocks_improved_ami']}/{m['n_eval_blocks']} · {m['blocks_improved_macb3']}/{m['n_eval_blocks']} |"
            )
            csv_rows.append(
                {
                    "dataset": ds_key,
                    "variant_key": k,
                    "variant_name": name,
                    "total_tokens": int(m["total_tokens_all_blocks"]),
                    "queries_per_block": round(m["queries"], 2),
                    "ami": round(m["ami"], 6),
                    "delta_ami": round(m["delta_ami"], 6),
                    "ari": round(m["ari"], 6),
                    "macro_b3": round(m["macro_b3"], 6),
                    "delta_macro_b3": round(m["delta_macro_b3"], 6),
                    "tail_b3": round(m["tail_b3"], 6),
                    "unq_delta_micro_b3": round(m["unq_delta_micro_b3"], 6),
                    "unq_net_posts": round(m["unq_net_posts"], 2),
                    "net_fm_red": round(m["net_fm_red"], 2),
                    "net_fs_red": round(m["net_fs_red"], 2),
                    "win_ami": int(m["blocks_improved_ami"]),
                    "win_macb3": int(m["blocks_improved_macb3"]),
                    "n_eval_blocks": int(m["n_eval_blocks"]),
                }
            )
        md.extend(["", "### 逐动作判决混淆矩阵与状态审计（消融变体）", ""])
        for abl_k in ["merge_only", "seam_only", "fiedler_only_seam", "no_extract_compare", "uniform_budget"]:
            ab = rep["ablations"][abl_k]
            md.append(
                f"- `{abl_k}` | statuses={ab['statuses']} | `MERGE`={ab['confusion_merge']} | `SEAM`={ab['confusion_seam']}"
            )
        md.extend(["", "---", ""])

    md_path = ROOT / "results" / "ablations" / "Ablation_Qwen3_8B_Report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    csv_path = ROOT / "results" / "ablations" / "ablation_bilingual_qwen3_8b.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    tex = [
        "% Auto-generated by trackA_sug_sed/run_ablations_qwen3.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Single-factor ablation study on unseen held-out streaming blocks of Event2012 ($M_{06}$--$M_{21}$, 16 blocks) and Event2018 ($M_{06}$--$M_{16}$, 11 blocks) under identical $10\\text{k}$ real tokens/block budgets using \\texttt{Qwen/Qwen3-8B}. Metrics shown as $\\times 100$.}",
        "\\label{tab:ablation}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l|cccc|cccc}",
        "\\toprule",
        " & \\multicolumn{4}{c|}{\\textbf{Event2012 Held-Out ($M_{06}$--$M_{21}$)}} & \\multicolumn{4}{c}{\\textbf{Event2018 Held-Out ($M_{06}$--$M_{16}$)}} \\\\",
        "\\textbf{Variant} & AMI ($\\Delta$) & Mac-$B^3$ ($\\Delta$) & Tail-$B^3$ & Unq $\\Delta$Mic & AMI ($\\Delta$) & Mac-$B^3$ ($\\Delta$) & Tail-$B^3$ & Unq $\\Delta$Mic \\\\",
        "\\midrule",
    ]
    for k, name, is_ref in rows_spec:
        m12 = _get_m(r12, k, is_ref)
        m18 = _get_m(r18, k, is_ref)
        tex_name = name.replace("_", "\\_")
        tex.append(
            f"{tex_name} & {m12['ami']*100:.2f} ({m12['delta_ami']*100:+.2f}) & "
            f"{m12['macro_b3']*100:.2f} ({m12['delta_macro_b3']*100:+.2f}) & "
            f"{m12['tail_b3']*100:.2f} & {m12['unq_delta_micro_b3']*100:+.2f} & "
            f"{m18['ami']*100:.2f} ({m18['delta_ami']*100:+.2f}) & "
            f"{m18['macro_b3']*100:.2f} ({m18['delta_macro_b3']*100:+.2f}) & "
            f"{m18['tail_b3']*100:.2f} & {m18['unq_delta_micro_b3']*100:+.2f} \\\\"
        )
    tex.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table*}"])
    tex_path = ROOT / "results" / "ablations" / "ablation_table_qwen3_8b.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")

    plot_keys = [
        ("sug_meso_direct", "SUG-Meso\n(Full)", True),
        ("sug_meso_repair", "Branch 2\n(Repair)", True),
        ("merge_only", "w/o SEAM\n(MergeOnly)", False),
        ("seam_only", "w/o MERGE\n(SeamOnly)", False),
        ("fiedler_only_seam", "w/o Pocket\n(Fiedler)", False),
        ("no_extract_compare", "w/o EtC\n(ActFirst)", False),
        ("uniform_budget", "w/o Bucket\n(Uniform)", False),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.9), dpi=220)
    for ax, (ds_title, rep) in zip(
        axes,
        [("Event2012 Held-Out (16 Blocks)", r12), ("Event2018 Held-Out (11 Blocks)", r18)],
    ):
        xs = np.arange(len(plot_keys))
        w = 0.36
        d_ami = [_get_m(rep, k, is_ref)["delta_ami"] * 100.0 for k, _, is_ref in plot_keys]
        d_mac = [_get_m(rep, k, is_ref)["delta_macro_b3"] * 100.0 for k, _, is_ref in plot_keys]
        b1 = ax.bar(xs - w / 2, d_ami, width=w, label="ΔAMI (pp)", color="#1f77b4", alpha=0.9)
        b2 = ax.bar(xs + w / 2, d_mac, width=w, label="ΔEvent Macro-B³ (pp)", color="#d62728", alpha=0.9)
        ax.axhline(0.0, color="#333333", linewidth=0.9)
        all_vals = d_ami + d_mac + [0.0]
        vmin, vmax = min(all_vals), max(all_vals)
        span = max(1.0, vmax - vmin)
        ax.set_ylim(vmin - 0.16 * span, vmax + 0.24 * span)
        for bars in (b1, b2):
            for rect in bars:
                h = rect.get_height()
                va = "bottom" if h >= 0 else "top"
                off = 0.025 * span if h >= 0 else -0.025 * span
                ax.text(rect.get_x() + rect.get_width() / 2, h + off, f"{h:+.2f}", ha="center", va=va, fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl, _ in plot_keys], fontsize=8.5)
        ax.set_ylabel("Gain over Init 2D-SE (percentage points)", fontsize=9.5)
        ax.set_title(ds_title, fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Single-Factor Component Ablations on Unseen Held-Out Streams (Qwen/Qwen3-8B, 10k tok/blk)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = ROOT / "results" / "figures" / "ablation_bilingual_qwen3_8b.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Done] Ablation Markdown : {md_path}")
    print(f"[Done] Ablation CSV      : {csv_path}")
    print(f"[Done] Ablation LaTeX    : {tex_path}")
    print(f"[Done] Ablation Figure   : {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="Event2012,Event2018")
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--tokens-per-block", type=int, default=10000)
    parser.add_argument("--conf-threshold", type=float, default=0.6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replot-only", action="store_true")
    args = parser.parse_args()

    if args.replot_only:
        r12 = json.loads((ROOT / "results" / "ablations" / "Event2012_heldout_qwen3_8b" / "ablation_report.json").read_text(encoding="utf-8"))
        r18 = json.loads((ROOT / "results" / "ablations" / "Event2018_heldout_qwen3_8b" / "ablation_report.json").read_text(encoding="utf-8"))
        compile_ablation_outputs(r12, r18)
        return

    ds_list = [x.strip() for x in args.datasets.split(",") if x.strip()]
    reports: Dict[str, Any] = {}
    for ds in ds_list:
        reports[ds] = run_dataset_ablations(
            ds,
            model_name=args.model,
            provider=args.provider,
            tokens_per_block=args.tokens_per_block,
            conf_threshold=args.conf_threshold,
            dry_run=args.dry_run,
        )
    if not args.dry_run and "Event2012" in reports and "Event2018" in reports:
        compile_ablation_outputs(reports["Event2012"], reports["Event2018"])


if __name__ == "__main__":
    main()
