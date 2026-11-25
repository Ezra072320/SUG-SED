"""IP&M Stage 2B: Entity/Hashtag Anonymization Anti-Memorization Stress Test.

Tests on ALL 27 held-out blocks (Event2012 M_06..M_21, Event2018 M_06..M_16) with
Qwen/Qwen3-8B under strictly causal CausalUniformBucket(10000, allow_carryover=False)
and uncontaminated all_shown U_t masking across 3 anonymized conditions:
  1. anon_pseudo_action  : Per-query consistent pseudonymization (action_first)
  2. anon_blind_action   : Strict blind generic redaction (action_first)
  3. anon_pseudo_extract : Per-query consistent pseudonymization (extract_compare)
                           + pretraining reconstruction audit on generated event_a/b.

All outputs are written strictly under results/ipm_experiments/stage2b_anonymization/
with isolated LLM cache results/llm_cache/{dataset}_Qwen__Qwen3-8B_anon/responses.jsonl.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import build_message_graph
from core_shared.key_loader import load_provider_key
from core_shared.metrics_audit import full_partition_audit
from core_shared.partition_cache import load_or_solve_2d_se
from core_shared.token_ledger import CausalUniformBucket, TokenLedger, canonical_request_hash
from trackA_sug_sed.anonymizer import (
    AnonymizationAuditStore,
    check_summary_reconstruction_leakage,
    make_anonymized_prompt_fn,
)
from trackA_sug_sed.high_leverage_router import (
    QueryItem,
    _interleave_query_queues,
    build_cluster_summaries,
    detect_merge_pair_queries,
    detect_seam_queries,
)
from trackA_sug_sed.llm_client import BudgetedLLMClient, parse_llm_response_to_decision
from trackA_sug_sed.run_gateA_oracle_diagnostic import load_or_extract_block_embeddings
from trackA_sug_sed.run_ipm_stage2a_qwen3 import (
    _cm_stats,
    _discover_all_caches,
    _holm_bonferroni,
    _summarize_arrays,
    _update_merge_seam_cm,
    _wilcoxon_and_effects,
)
from trackA_sug_sed.topology_surgery import LLMDecision, apply_direct_assignment


CONDITIONS: List[Dict[str, str]] = [
    {"name": "anon_pseudo_action", "prompt_mode": "action_first", "anon_mode": "pseudonym", "unmasked_ref": "sug_meso_causal_fixed_action"},
    {"name": "anon_blind_action", "prompt_mode": "action_first", "anon_mode": "blind", "unmasked_ref": "sug_meso_causal_fixed_action"},
    {"name": "anon_pseudo_extract", "prompt_mode": "extract_compare", "anon_mode": "pseudonym", "unmasked_ref": "sug_meso_causal_fixed_extract"},
]


def _load_unmasked_stage2a_actions(ds_name: str, variant_name: str, qmap_by_bid: Dict[int, Dict[str, QueryItem]]) -> Dict[Tuple[int, str], str]:
    """Load executed per-query actions from Stage 2A journal + cache (read-only) for agreement check."""
    led_dir = ROOT / "results" / "ipm_experiments" / "stage2a_qwen3" / ds_name / f"ledger_{variant_name}"
    jpath = led_dir / "journal.jsonl"
    out: Dict[Tuple[int, str], str] = {}
    if not jpath.exists():
        return out
    shared, extras = _discover_all_caches(ds_name)
    ref_led = TokenLedger(led_dir, extra_cache_dirs=extras, shared_cache_file=shared)
    with jpath.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") not in {"OK", "CACHED_REPLAY"}:
                continue
            bid = int(rec["block_id"])
            qid = str(rec["request_id"])
            q = qmap_by_bid.get(bid, {}).get(qid)
            if q is None:
                continue
            hit = ref_led.lookup_cache(str(rec.get("request_hash", "")))
            if hit is None:
                continue
            dec = parse_llm_response_to_decision(
                q,
                hit.get("response", {}),
                prompt_tokens=int(rec.get("prompt_tokens", 0)),
                completion_tokens=int(rec.get("completion_tokens", 0)),
            )
            out[(bid, qid)] = dec.action
    return out


def _audit_journal_payloads(
    led_dir: Path,
    ledger: TokenLedger,
    ds_name: str,
    block_cache: Dict[int, Dict[str, Any]],
    audit_store: AnonymizationAuditStore,
    prompt_mode: str,
) -> Dict[str, Any]:
    """Audit executed extract_compare event_a/b summaries for pretraining entity reconstruction."""
    jpath = led_dir / "journal.jsonl"
    statuses: Counter = Counter()
    total_extract_checked = 0
    leaked_queries = 0
    leaked_tokens_total = 0
    leaked_samples: List[Dict[str, Any]] = []
    if not jpath.exists():
        return {
            "errors": 0,
            "statuses": {},
            "reconstruction_checked": 0,
            "reconstruction_leaked_queries": 0,
            "reconstruction_rate": 0.0,
            "leaked_samples": [],
        }

    with jpath.open("r", encoding="utf-8") as jf:
        for line in jf:
            if not line.strip():
                continue
            rec = json.loads(line)
            st = rec.get("status", "UNKNOWN")
            statuses[st] += 1
            if prompt_mode == "extract_compare" and st in {"OK", "CACHED_REPLAY"}:
                bid = int(rec["block_id"])
                qid = str(rec["request_id"])
                hit = ledger.lookup_cache(str(rec.get("request_hash", "")))
                payload = hit.get("response", {}) if hit is not None else {}
                if isinstance(payload, dict) and ("event_a" in payload or "event_b" in payload):
                    ev_a = str(payload.get("event_a", ""))
                    ev_b = str(payload.get("event_b", ""))
                    ds_block_name = block_cache[bid]["ds"].name
                    arec = audit_store.records.get((ds_block_name, qid))
                    if arec is not None and arec.get("masked_originals"):
                        total_extract_checked += 1
                        hits = check_summary_reconstruction_leakage(ev_a, ev_b, arec["masked_originals"])
                        if hits:
                            leaked_queries += 1
                            leaked_tokens_total += len(hits)
                            if len(leaked_samples) < 5:
                                leaked_samples.append({
                                    "block_id": bid,
                                    "query_id": qid,
                                    "leaked_terms": hits,
                                    "event_a": ev_a[:120],
                                    "event_b": ev_b[:120],
                                })

    n_err = sum(v for k, v in statuses.items() if k not in {"OK", "CACHED_REPLAY"})
    return {
        "errors": int(n_err),
        "statuses": dict(statuses),
        "reconstruction_checked": int(total_extract_checked),
        "reconstruction_leaked_queries": int(leaked_queries),
        "reconstruction_rate": (leaked_queries / total_extract_checked) if total_extract_checked > 0 else 0.0,
        "leaked_samples": leaked_samples,
    }


def _run_condition_dry(
    cond: Dict[str, str],
    ds_name: str,
    held_ids: List[int],
    block_cache: Dict[int, Dict[str, Any]],
    probe_ledger: TokenLedger,
    model_name: str,
    seen_hashes: Dict[str, Tuple[int, int]],
) -> Dict[str, Any]:
    audit_store = AnonymizationAuditStore(max_examples=6)
    pfn = make_anonymized_prompt_fn(cond["prompt_mode"], cond["anon_mode"], audit_store=audit_store)
    tot_calls = tot_cached = tot_live = tot_tok = 0
    executed_keys: List[Tuple[str, str]] = []
    for bid in held_ids:
        bc = block_cache[bid]
        ds, summs, pool = bc["ds"], bc["summaries"], bc["sug_meso_pool"]
        mean_u = float(np.mean([q.score for q in pool])) if pool else 1.0
        bkt = CausalUniformBucket(tokens_per_block=10000, allow_carryover=False)
        bkt.start_block(mean_uncertainty=mean_u)
        for q in pool:
            sm, um = pfn(q, ds, summs)
            est = int((len(sm) + len(um)) / 3.5) + 120
            if not bkt.can_spend(est):
                break
            executed_keys.append((ds.name, q.query_id))
            rh = canonical_request_hash(
                {
                    "model": model_name,
                    "temperature": 0.0,
                    "messages": [{"role": "system", "content": sm}, {"role": "user", "content": um}],
                }
            )
            hit = probe_ledger.lookup_cache(rh)
            tot_calls += 1
            if hit is not None:
                pt, ct = int(hit["prompt_tokens"]), int(hit["completion_tokens"])
                tot_cached += 1
                bkt.record_spend(pt, ct)
                tot_tok += pt + ct
            elif rh in seen_hashes:
                pt, ct = seen_hashes[rh]
                tot_cached += 1
                bkt.record_spend(pt, ct)
                tot_tok += pt + ct
            else:
                est_pt = int((len(sm) + len(um)) / 3.5)
                est_ct = 40
                seen_hashes[rh] = (est_pt, est_ct)
                tot_live += 1
                bkt.record_spend(est_pt, est_ct)
                tot_tok += est_pt + est_ct

    mask_sum = audit_store.summarize_for_dataset(ds_name, executed_keys=executed_keys)
    return {
        "total_calls": tot_calls,
        "cached_or_dedup": tot_cached,
        "unique_live_needed": tot_live,
        "total_tokens": tot_tok,
        "mask_audit": mask_sum,
        "examples": list(audit_store.examples.values())[:3],
    }


def _run_condition_live(
    cond: Dict[str, str],
    ds_name: str,
    held_ids: List[int],
    block_cache: Dict[int, Dict[str, Any]],
    out_dir: Path,
    shared_cache_file: Path,
    model_name: str,
    base_url: str,
    api_key: str,
    workers: int,
    unmasked_actions: Dict[Tuple[int, str], str],
    conf_threshold: float = 0.6,
) -> Dict[str, Any]:
    name = cond["name"]
    audit_store = AnonymizationAuditStore(max_examples=8)
    pfn = make_anonymized_prompt_fn(cond["prompt_mode"], cond["anon_mode"], audit_store=audit_store)

    led_dir = out_dir / ds_name / f"ledger_{name}"
    if (led_dir / "journal.jsonl").exists():
        (led_dir / "journal.jsonl").unlink()
    ledger = TokenLedger(
        led_dir,
        price_per_1m_prompt_cny=0.5,
        price_per_1m_completion_cny=2.0,
        shared_cache_file=shared_cache_file,
    )
    client = BudgetedLLMClient(model=model_name, ledger=ledger, base_url=base_url, api_key=api_key)

    def _query_one_block(bid: int, bkt: Any) -> Tuple[int, int, List[LLMDecision]]:
        bc = block_cache[bid]
        ds, summs, pool = bc["ds"], bc["summaries"], bc["sug_meso_pool"]
        mean_u = float(np.mean([q.score for q in pool])) if pool else 1.0
        cap = bkt.start_block(mean_uncertainty=mean_u)
        decisions = client.execute_queries(
            pool, ds, summs, block_id=bid, bucket=bkt, prompt_fn=pfn, strategy_tag=name
        )
        return bid, cap, decisions

    raw_decisions: Dict[int, Tuple[int, List[LLMDecision]]] = {}
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_query_one_block, bid, CausalUniformBucket(tokens_per_block=10000, allow_carryover=False)): bid
                for bid in held_ids
            }
            for fut in as_completed(futs):
                bid_r, cap_r, decs_r = fut.result()
                raw_decisions[bid_r] = (cap_r, decs_r)
    else:
        for bid in held_ids:
            bkt = CausalUniformBucket(tokens_per_block=10000, allow_carryover=False)
            bid_r, cap_r, decs_r = _query_one_block(bid, bkt)
            raw_decisions[bid_r] = (cap_r, decs_r)

    eval_total_posts = sum(len(block_cache[b]["ds"]) for b in held_ids)
    cm_m, cm_s = Counter(), Counter()
    arrs: Dict[str, List[float]] = defaultdict(list)
    executed_keys: List[Tuple[str, str]] = []
    total_calls = total_shown = total_unq_dir = 0
    agree_all = [0, 0]
    agree_merge = [0, 0]
    agree_seam = [0, 0]

    for bid in held_ids:
        bc = block_cache[bid]
        ds, li, summs, pool = bc["ds"], bc["labels_init"], bc["summaries"], bc["sug_meso_pool"]
        _, decisions = raw_decisions[bid]
        dec_ids = {d.query_id for d in decisions}
        used_queries = [q for q in pool if q.query_id in dec_ids]
        for q in used_queries:
            executed_keys.append((ds.name, q.query_id))
        for d in decisions:
            if (bid, d.query_id) in unmasked_actions:
                ref_act = unmasked_actions[(bid, d.query_id)]
                agree_all[1] += 1
                if d.action == ref_act:
                    agree_all[0] += 1
                if d.query_type == "merge_pair":
                    agree_merge[1] += 1
                    if d.action == ref_act:
                        agree_merge[0] += 1
                elif d.query_type == "seam":
                    agree_seam[1] += 1
                    if d.action == ref_act:
                        agree_seam[0] += 1

        btok = sum(d.prompt_tokens + d.completion_tokens for d in decisions)
        _update_merge_seam_cm(ds, summs, used_queries, decisions, conf_threshold, cm_m, cm_s)
        lab_dir, qmask_dir = apply_direct_assignment(
            li, used_queries, decisions, conf_threshold=conf_threshold, summaries=summs, mask_mode="all_shown"
        )
        aud = full_partition_audit(ds.labels, lab_dir, y_before=li, queried_mask=qmask_dir)
        prop = aud["propagation_audit"]

        total_calls += len(decisions)
        total_shown += int(qmask_dir.sum())
        total_unq_dir += int(prop["unqueried_net_improved_posts"])

        arrs["queries"].append(float(len(decisions)))
        arrs["tokens"].append(float(btok))
        arrs["ami"].append(float(aud["ami"]))
        arrs["delta_ami"].append(float(aud["delta_ami"]))
        arrs["nmi"].append(float(aud["nmi"]))
        arrs["ari"].append(float(aud["ari"]))
        arrs["macro_b3"].append(float(aud["bcubed_full"]["macro_b3_f1"]))
        arrs["delta_macro_b3"].append(float(aud["delta_macro_b3_f1"]))
        arrs["micro_b3"].append(float(aud["bcubed_full"]["micro_b3_f1"]))
        arrs["delta_micro_b3"].append(float(aud["delta_micro_b3_f1"]))
        arrs["tail_b3"].append(float(aud["stratified"]["tail"]["macro_b3_f1"]))
        arrs["unq_delta_micro_b3"].append(float(prop["unqueried_delta_micro_b3_f1"]))
        arrs["unq_delta_macro_b3"].append(float(prop["unqueried_delta_macro_b3_f1"]))
        arrs["unq_net_posts"].append(float(prop["unqueried_net_improved_posts"]))

    j_audit = _audit_journal_payloads(led_dir, ledger, ds_name, block_cache, audit_store, cond["prompt_mode"])
    lsum = ledger.summary()
    lsum["errors"] = j_audit["errors"]
    lsum["statuses"] = j_audit["statuses"]

    m_dir = _summarize_arrays(arrs)
    mask_sum = audit_store.summarize_for_dataset(ds_name, executed_keys=executed_keys)
    agreement = {
        "overall_agree": agree_all[0],
        "overall_total": agree_all[1],
        "overall_rate": (agree_all[0] / agree_all[1]) if agree_all[1] > 0 else None,
        "merge_agree": agree_merge[0],
        "merge_total": agree_merge[1],
        "merge_rate": (agree_merge[0] / agree_merge[1]) if agree_merge[1] > 0 else None,
        "seam_agree": agree_seam[0],
        "seam_total": agree_seam[1],
        "seam_rate": (agree_seam[0] / agree_seam[1]) if agree_seam[1] > 0 else None,
    }
    print(
        f"  [{ds_name:9s} | {name:20s}] calls={total_calls:3d} ({100*total_shown/max(1,eval_total_posts):.2f}% shown) "
        f"live={lsum['total_calls']-lsum['cached_calls']:3d} err={lsum['errors']} | "
        f"AMI={m_dir['ami']['mean']:.4f} ({m_dir['delta_ami']['mean']:+.4f}) "
        f"MacB3={m_dir['macro_b3']['mean']:.4f} ({m_dir['delta_macro_b3']['mean']:+.4f}) "
        f"UnqNet={m_dir['unq_net_posts']['mean']:+.1f} | "
        f"Mask(Ent={100*mask_sum['entity_mask_rate']:.1f}%,Tag={100*mask_sum['hashtag_mask_rate']:.1f}%) "
        f"Agree={100*(agreement['overall_rate'] or 0.0):.1f}%"
    )
    return {
        "name": name,
        "prompt_mode": cond["prompt_mode"],
        "anon_mode": cond["anon_mode"],
        "unmasked_ref": cond["unmasked_ref"],
        "block_ids": held_ids,
        "eval_total_posts": eval_total_posts,
        "ledger": lsum,
        "counts": {
            "total_calls": total_calls,
            "calls_pct": 100.0 * total_calls / max(1, eval_total_posts),
            "all_shown_posts": total_shown,
            "all_shown_pct": 100.0 * total_shown / max(1, eval_total_posts),
            "unq_net_total_direct": total_unq_dir,
        },
        "confusion_merge": _cm_stats(cm_m),
        "confusion_seam": _cm_stats(cm_s),
        "mask_audit": mask_sum,
        "reconstruction_audit": {
            "checked": j_audit["reconstruction_checked"],
            "leaked_queries": j_audit["reconstruction_leaked_queries"],
            "reconstruction_rate": j_audit["reconstruction_rate"],
            "leaked_samples": j_audit["leaked_samples"],
        },
        "decision_agreement": agreement,
        "examples": list(audit_store.examples.values())[:3],
        "direct": m_dir,
    }


def _compute_stage2b_stats(
    ds_name: str,
    cond_results: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
    stage2a_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    comparators: Dict[str, Any] = {}
    if stage1_json and ds_name in stage1_json:
        s1_stats = stage1_json[ds_name]["D5_stats"]["summary_stats"]
        for sk in ["init_2dse", "heur_meso_0tok", "cequel_transitive"]:
            if sk in s1_stats:
                comparators[sk] = s1_stats[sk]
    if stage2a_json and ds_name in stage2a_json:
        s2_w = stage2a_json[ds_name]["within_block"]
        for sk in [
            "random_mean3",
            "random_meso_action_mean3",
            "random_meso_extract_mean3",
            "sug_meso_causal_fixed_action",
            "sug_meso_causal_fixed_extract",
        ]:
            if sk in s2_w:
                comparators[sk] = s2_w[sk]["direct"]

    retentions: Dict[str, Any] = {}
    tests: Dict[str, Any] = {}
    init_blk = comparators.get("init_2dse")
    for cname, cres in cond_results.items():
        d_anon = cres["direct"]
        ref_key = cres["unmasked_ref"]
        d_orig = comparators.get(ref_key)
        if init_blk is not None and d_orig is not None:
            ret_row: Dict[str, Any] = {}
            for met in ["ami", "macro_b3", "micro_b3", "tail_b3"]:
                o_gain = float(d_orig[met]["mean"]) - float(init_blk[met]["mean"])
                a_gain = float(d_anon[met]["mean"]) - float(init_blk[met]["mean"])
                ret_row[met] = {
                    "init_mean": float(init_blk[met]["mean"]),
                    "orig_mean": float(d_orig[met]["mean"]),
                    "anon_mean": float(d_anon[met]["mean"]),
                    "orig_gain": o_gain,
                    "anon_gain": a_gain,
                    "retention_ratio": (a_gain / o_gain) if abs(o_gain) > 1e-12 else None,
                    "delta_vs_unmasked": float(d_anon[met]["mean"]) - float(d_orig[met]["mean"]),
                }
            # Also track UnqNet retention vs unmasked
            o_unq = float(d_orig["unq_net_posts"]["mean"])
            a_unq = float(d_anon["unq_net_posts"]["mean"])
            ret_row["unq_net_posts"] = {
                "orig_mean": o_unq,
                "anon_mean": a_unq,
                "retention_ratio": (a_unq / o_unq) if abs(o_unq) > 1e-12 else None,
            }
            retentions[cname] = ret_row

        tests[cname] = {}
        comps = list(comparators.keys())
        for met in ["ami", "macro_b3", "micro_b3", "tail_b3"]:
            elig = [c for c in comps if met in comparators[c]]
            pvals = []
            rows = {}
            for c in elig:
                r = _wilcoxon_and_effects(d_anon[met]["per_block"], comparators[c][met]["per_block"])
                rows[c] = r
                pvals.append(r["p_value"])
            for c, ph in zip(elig, _holm_bonferroni(pvals)):
                rows[c]["p_holm"] = float(ph)
            tests[cname][met] = rows

    return {"comparators": {k: {m: comparators[k][m]["mean"] for m in ["ami", "macro_b3", "tail_b3"] if m in comparators[k]} for k in comparators}, "retentions": retentions, "tests": tests}


def _write_summary_md(report: Dict[str, Any], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# IP&M Stage 2B: Entity/Hashtag Anonymization Anti-Memorization Report\n")
    lines.append(f"- Generated: `{report['generated_at']}` | Model: `{report['model']}`\n")
    lines.append(f"- Ledger Totals across Stage 2B: `{report['ledger_totals']}`\n")

    for ds_name in ["Event2012", "Event2018"]:
        dr = report[ds_name]
        lines.append(f"\n---\n\n## {ds_name}\n")
        lines.append("### 1. Leakage & Masking Coverage Audit (Executed Queries)\n")
        lines.append("| Condition | Queries | Entity Mask | Hashtag Mask | Year Mask | URL Mask | PH Density | Action Agree vs Unmasked | Extr. Reconstr. Rate |\n")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |\n")
        for cname, cv in dr["conditions"].items():
            ma = cv["mask_audit"]
            ag = cv["decision_agreement"]
            rc = cv["reconstruction_audit"]
            ag_str = (
                f"{100*ag['overall_rate']:.1f}% (M={100*(ag['merge_rate'] or 0):.1f}%, S={100*(ag['seam_rate'] or 0):.1f}%)"
                if ag["overall_rate"] is not None
                else "N/A"
            )
            rc_str = (
                f"{100*rc['reconstruction_rate']:.2f}% ({rc['leaked_queries']}/{rc['checked']})"
                if rc["checked"] > 0
                else "N/A"
            )
            lines.append(
                f"| `{cname}` | {ma['n_queries']} | "
                f"{ma['entity_masked']}/{ma['entity_total']} ({100*ma['entity_mask_rate']:.2f}%) | "
                f"{ma['hashtag_masked']}/{ma['hashtag_total']} ({100*ma['hashtag_mask_rate']:.2f}%) | "
                f"{ma['year_masked']}/{ma['year_total']} ({100*ma['year_mask_rate']:.2f}%) | "
                f"{ma['url_masked']}/{ma['url_total']} ({100*ma['url_mask_rate']:.2f}%) | "
                f"{100*ma['placeholder_density']:.1f}% | {ag_str} | {rc_str} |\n"
            )

        lines.append("\n### 2. Clustering Performance & Gain Retention (`CausalUniformFixed 10k/blk`, `all_shown` $U_t$)\n")
        lines.append("| Variant | Prompt / Mask | AMI | Mac-B³ | Tail-B³ | UnqNet/blk | AMI Ret. | Mac Ret. | Merge P/R/S | Seam P/R/S | Err |\n")
        lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: |\n")
        comp_means = dr["statistics"]["comparators"]
        for ref_k in ["init_2dse", "random_mean3", "random_meso_action_mean3", "sug_meso_causal_fixed_action", "sug_meso_causal_fixed_extract"]:
            if ref_k in comp_means:
                cm = comp_means[ref_k]
                lines.append(
                    f"| `{ref_k}` (Ref) | unmasked | {cm.get('ami', 0):.4f} | {cm.get('macro_b3', 0):.4f} | {cm.get('tail_b3', 0):.4f} | - | - | - | - | - | 0 |\n"
                )
        rets = dr["statistics"]["retentions"]
        for cname, cv in dr["conditions"].items():
            d = cv["direct"]
            cm_m, cm_s = cv["confusion_merge"], cv["confusion_seam"]
            rr = rets.get(cname, {})
            r_ami = f"{100*rr['ami']['retention_ratio']:.1f}%" if rr.get("ami", {}).get("retention_ratio") is not None else "N/A"
            r_mac = f"{100*rr['macro_b3']['retention_ratio']:.1f}%" if rr.get("macro_b3", {}).get("retention_ratio") is not None else "N/A"
            lines.append(
                f"| `{cname}` | {cv['prompt_mode']} / {cv['anon_mode']} | "
                f"{d['ami']['mean']:.4f} | {d['macro_b3']['mean']:.4f} | {d['tail_b3']['mean']:.4f} | "
                f"{d['unq_net_posts']['mean']:+.1f} | {r_ami} | {r_mac} | "
                f"{cm_m['precision']:.2f}/{cm_m['recall']:.2f}/{cm_m['specificity']:.2f} | "
                f"{cm_s['precision']:.2f}/{cm_s['recall']:.2f}/{cm_s['specificity']:.2f} | {cv['ledger']['errors']} |\n"
            )

        lines.append("\n### 3. Paired Wilcoxon Tests (Anonymized vs. Baselines & Unmasked Reference)\n")
        for cname, tdict in dr["statistics"]["tests"].items():
            lines.append(f"\n#### `{cname}`\n")
            for comp in [
                "init_2dse",
                "random_mean3",
                "random_meso_action_mean3",
                "random_meso_extract_mean3",
                "sug_meso_causal_fixed_action",
                "sug_meso_causal_fixed_extract",
            ]:
                if comp in tdict["ami"]:
                    pa = tdict["ami"][comp]
                    pm = tdict["macro_b3"][comp]
                    lines.append(
                        f"- vs `{comp}`: ΔAMI={pa['mean_diff']:+.4f} (p={pa['p_value']:.2e}, H={pa['p_holm']:.2e}, dz={pa['cohens_dz']:+.2f}) | "
                        f"ΔMac={pm['mean_diff']:+.4f} (p={pm['p_value']:.2e}, H={pm['p_holm']:.2e}, dz={pm['cohens_dz']:+.2f})\n"
                    )

        lines.append("\n### 4. Sample Before / After Anonymized Prompts\n")
        for cname in ["anon_pseudo_action", "anon_blind_action"]:
            exs = dr["conditions"][cname]["examples"][:2]
            for ex in exs:
                lines.append(f"- **[{cname} | {ex['dataset']} | {ex['query_id']}]**\n")
                lines.append(f"  - Raw: `{ex['raw_user_msg'].replace(chr(10), ' ')[:260]}`\n")
                lines.append(f"  - Anon: `{ex['anon_user_msg'].replace(chr(10), ' ')[:260]}`\n")

    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="IP&M Stage 2B Anonymization Anti-Memorization Suite")
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "results" / "ipm_experiments" / "stage2b_anonymization"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = ROOT / "results" / "ipm_audit" / "stage1_diagnostics.json"
    stage2a_path = ROOT / "results" / "ipm_experiments" / "stage2a_qwen3" / "stage2a_report.json"
    stage1_json = json.loads(stage1_path.read_text(encoding="utf-8")) if stage1_path.exists() else None
    stage2a_json = json.loads(stage2a_path.read_text(encoding="utf-8")) if stage2a_path.exists() else None

    if args.dry_run:
        api_key = ""
        base_url = "dry-run"
        model_name = args.model
    else:
        prov_url, prov_model, api_key = load_provider_key(args.provider)
        base_url = prov_url
        model_name = args.model or prov_model

    t_start = time.time()
    all_report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "provider": args.provider,
    }
    grand = Counter()

    for ds_name in ["Event2012", "Event2018"]:
        embed_model = (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            if ds_name == "Event2018"
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        ds_full = load_socialed_npy(ds_name, sort_chronologically=True)
        blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
        held_ids = list(range(6, len(blocks)))

        t_pre = time.time()
        block_cache: Dict[int, Dict[str, Any]] = {}
        qmap_by_bid: Dict[int, Dict[str, QueryItem]] = {}
        for bid in held_ids:
            bname, bidx = blocks[bid]
            sub_ds = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
            emb = load_or_extract_block_embeddings(ds_name, bid, sub_ds.texts, model_name=embed_model)
            graph = build_message_graph(sub_ds, emb, knn_k=15, knn_sim_threshold=0.35, attr_alpha=0.5)
            labels_init, _ = load_or_solve_2d_se(ds_name, bid, graph.adj, 15, 0.35, 0.5, True)
            summaries = build_cluster_summaries(sub_ds, emb, labels_init)
            merges = detect_merge_pair_queries(sub_ds, graph, labels_init, summaries)
            seams = detect_seam_queries(sub_ds, graph, labels_init, summaries)
            pool = _interleave_query_queues([("merge", merges, 0.65), ("seam", seams, 0.35)], 120)
            block_cache[bid] = {
                "name": bname,
                "ds": sub_ds,
                "graph": graph,
                "labels_init": labels_init,
                "summaries": summaries,
                "sug_meso_pool": pool,
            }
            qmap_by_bid[bid] = {q.query_id: q for q in pool}
        print(f"[Preload {ds_name}] {len(held_ids)} held-out blocks ready in {time.time()-t_pre:.1f}s")

        shared_cache = ROOT / "results" / "llm_cache" / f"{ds_name}_Qwen__Qwen3-8B_anon" / "responses.jsonl"

        if args.dry_run:
            probe = TokenLedger(out_dir / "_dry_probe", shared_cache_file=shared_cache)
            seen_hashes: Dict[str, Tuple[int, int]] = {}
            print(f"\n=== DRY-RUN ANONYMIZATION AUDIT: {ds_name} ({len(held_ids)} held-out blocks) ===")
            for cond in CONDITIONS:
                dres = _run_condition_dry(cond, ds_name, held_ids, block_cache, probe, model_name, seen_hashes)
                ma = dres["mask_audit"]
                est_cny = dres["unique_live_needed"] * 560 * 0.6 / 1_000_000.0
                grand["live_needed"] += dres["unique_live_needed"]
                grand["tot_calls"] += dres["total_calls"]
                grand["est_milli_cny"] += int(round(est_cny * 1000.0))
                print(
                    f"  [{cond['name']:19s}] calls={dres['total_calls']} (live={dres['unique_live_needed']}, ~{est_cny:.3f} CNY) | "
                    f"EntMask={ma['entity_masked']}/{ma['entity_total']} ({100*ma['entity_mask_rate']:.2f}%) | "
                    f"TagMask={ma['hashtag_masked']}/{ma['hashtag_total']} ({100*ma['hashtag_mask_rate']:.2f}%) | "
                    f"YearMask={ma['year_masked']}/{ma['year_total']} ({100*ma['year_mask_rate']:.2f}%) | "
                    f"PH_density={100*ma['placeholder_density']:.1f}%"
                )
                for ex in dres["examples"][:2]:
                    print(f"    EX [{ex['dataset']} {ex['query_id']}] RAW : {ex['raw_user_msg'].replace(chr(10), ' ')[:175]}")
                    print(f"    EX [{ex['dataset']} {ex['query_id']}] ANON: {ex['anon_user_msg'].replace(chr(10), ' ')[:175]}")
            continue

        print(f"\n==================== STAGE 2B ANONYMIZATION LIVE: {ds_name} ({model_name}) ====================")
        ref_actions = {
            "sug_meso_causal_fixed_action": _load_unmasked_stage2a_actions(ds_name, "sug_meso_causal_fixed_action", qmap_by_bid),
            "sug_meso_causal_fixed_extract": _load_unmasked_stage2a_actions(ds_name, "sug_meso_causal_fixed_extract", qmap_by_bid),
        }
        cond_results: Dict[str, Any] = {}
        for cond in CONDITIONS:
            cres = _run_condition_live(
                cond,
                ds_name,
                held_ids,
                block_cache,
                out_dir,
                shared_cache,
                model_name,
                base_url,
                api_key,
                workers=args.workers,
                unmasked_actions=ref_actions[cond["unmasked_ref"]],
            )
            cond_results[cond["name"]] = cres
            grand["calls"] += cres["ledger"]["total_calls"]
            grand["live"] += cres["ledger"]["total_calls"] - cres["ledger"]["cached_calls"]
            grand["tokens"] += cres["ledger"]["total_tokens"]
            grand["errors"] += cres["ledger"]["errors"]

        all_report[ds_name] = {
            "conditions": cond_results,
            "statistics": _compute_stage2b_stats(ds_name, cond_results, stage1_json, stage2a_json),
        }

    if args.dry_run:
        print(
            f"\n[DRY-RUN TOTAL] calls={grand['tot_calls']}, unique_live={grand['live_needed']}, "
            f"est_cost~{grand['est_milli_cny']/1000.0:.3f} CNY"
        )
        return

    j_errors = 0
    for jp in sorted(out_dir.rglob("journal.jsonl")):
        with jp.open("r", encoding="utf-8") as jf:
            for line in jf:
                if line.strip() and json.loads(line).get("status", "UNKNOWN") not in {"OK", "CACHED_REPLAY"}:
                    j_errors += 1
    grand["all_journal_errors"] = j_errors
    all_report["ledger_totals"] = dict(grand)
    all_report["elapsed_sec"] = round(time.time() - t_start, 2)

    json_path = out_dir / "anonymization_report.json"
    md_path = out_dir / "anonymization_summary.md"
    json_path.write_text(json.dumps(all_report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_md(all_report, md_path)
    print(f"\n[DONE] Stage 2B Anonymization finished in {all_report['elapsed_sec']:.1f}s | errors={grand['errors']} all_journal_errors={j_errors}")
    print(f"  JSON: {json_path}")
    print(f"  MD  : {md_path}")


if __name__ == "__main__":
    main()
