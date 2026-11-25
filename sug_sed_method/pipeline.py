"""Standalone End-to-End Pipeline for SUG-SED v2 (Pure-Star-Meso + SUG-CrossStar).

Supports:
  1. `--mode synthetic`: offline synthetic 4-block stream (Graph -> Pure 2D-SE ->
     AutoMerge + GuardedUF -> Clean Seam Split + Star-Merge -> SUG-CrossStar).
  2. `--mode dataset --dry-run-oracle`: real SocialED blocks without spending API tokens.
  3. `--mode dataset --provider siliconflow`: full live budgeted LLM execution with
     CausalUniformBucket and TokenLedger.
"""

from __future__ import annotations

import os

for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PKG_PARENT = Path(__file__).resolve().parent.parent
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

from sug_sed_method.core.data_loader import SocialDataset, load_socialed_npy, split_daily_blocks
from sug_sed_method.core.graph_builder import build_message_graph
from sug_sed_method.core.key_loader import load_provider_key
from sug_sed_method.core.metrics_audit import full_partition_audit
from sug_sed_method.core.partition_cache import load_or_solve_2d_se
from sug_sed_method.core.se_solver import solve_2d_se
from sug_sed_method.core.token_ledger import CausalUniformBucket, TokenLedger
from sug_sed_method.method.anonymizer import AnonymizationAuditStore, QueryAnonymizer
from sug_sed_method.method.cross_day import evaluate_cross_day_stream
from sug_sed_method.method.embed_cache import load_or_extract_block_embeddings
from sug_sed_method.method.llm_client import BudgetedLLMClient
from sug_sed_method.method.star_meso import run_pure_star_meso

EMBED_MODELS: Dict[str, str] = {
    "Event2012": "sentence-transformers/all-MiniLM-L6-v2",
    "Event2012_100": "sentence-transformers/all-MiniLM-L6-v2",
    "Event2018": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


def _make_synthetic_block(
    block_id: int,
    centers: Dict[str, np.ndarray],
    seed: int = 7,
) -> Tuple[SocialDataset, np.ndarray]:
    """Build a synthetic block exercising AutoMerge, Clean Seam Split, Star-Merge, and Cross-Day."""
    rng = np.random.default_rng(seed + block_id)
    dim = next(iter(centers.values())).shape[0]
    embs, labels, texts, entities, hashtags, lemmas, tss = [], [], [], [], [], [], []
    day_str = f"2026-01-{block_id + 1:02d}"
    base_ts = 1e9 + block_id * 86400.0

    def _add_posts(
        ev_label: int,
        ctr: np.ndarray,
        count: int,
        noise: float,
        ents: List[str],
        tags: List[str],
        topic: str,
        t_start: float,
        t_step: float = 60.0,
    ) -> None:
        for k in range(count):
            v = ctr + noise * rng.normal(size=dim).astype(np.float32)
            v /= max(float(np.linalg.norm(v)), 1e-12)
            embs.append(v)
            labels.append(ev_label)
            texts.append(f"{topic} {' '.join(ents)} #{tags[0]} day{block_id} item {k}")
            entities.append(list(ents))
            hashtags.append(list(tags))
            lemmas.append([topic.lower(), "update", "report"])
            tss.append(base_ts + t_start + k * t_step)

    # Event 0: two high-sim fragments (cos ~ 0.88) -> triggers Stage-2 AutoMerge (>=0.85)
    _add_posts(0, centers["0a"], 9, 0.03, ["city0a"], ["tag0a"], "Airport strike", 0.0)
    _add_posts(0, centers["0b"], 9, 0.03, ["city0b"], ["tag0b"], "Terminal walkout", 1000.0)
    # Event 1: two ambiguous fragments (cos ~ 0.66) -> triggers Stage-4 Star-Merge ([0.55, 0.85))
    _add_posts(1, centers["1a"], 9, 0.03, ["city1a"], ["tag1a"], "River flood", 2000.0)
    _add_posts(1, centers["1b"], 9, 0.03, ["city1b"], ["tag1b"], "Levee breach", 3000.0)
    # Events 2 & 3: 14-post + 3-post events with overlapping timestamps & shared hub attributes -> merged by 2D-SE, split by Stage-3 Seam
    _add_posts(2, centers["2"], 14, 0.04, ["hub23", "venue23", "arena23", "loc23"], ["tag23", "live23"], "Stadium concert", 4000.0, 60.0)
    _add_posts(3, centers["3"], 3, 0.07, ["hub23", "venue23", "arena23", "loc23"], ["tag23", "live23"], "Subway outage", 4030.0, 180.0)
    # Event 4: well-separated baseline event
    _add_posts(4, centers["4"], 10, 0.04, ["city4"], ["tag4"], "Election debate", 6000.0)

    n = len(texts)
    ds = SocialDataset(
        name=f"synthetic_M{block_id:02d}",
        tweet_ids=np.arange(n, dtype=np.int64) + block_id * 1000,
        texts=texts,
        labels=np.array(labels, dtype=np.int64),
        tokens_lemma=lemmas,
        entities=entities,
        hashtags=hashtags,
        user_ids=[f"u{i % 5}" for i in range(n)],
        mentioned_users=[[] for _ in range(n)],
        timestamps=np.array(tss, dtype=np.float64),
        dates=[day_str] * n,
    )
    return ds, np.vstack(embs)


def _unit_orth(rng: np.random.Generator, base: np.ndarray, others: List[np.ndarray]) -> np.ndarray:
    v = rng.normal(size=base.shape[0]).astype(np.float32)
    for o in [base] + others:
        v -= float(np.dot(v, o)) * o
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return v


def run_synthetic_demo() -> None:
    """Run offline 4-block verification covering all 5 SUG-SED v2 stages."""
    rng = np.random.default_rng(7)
    dim = 32
    b0 = rng.normal(size=dim).astype(np.float32)
    b0 /= np.linalg.norm(b0)
    o0 = _unit_orth(rng, b0, [])
    b1 = _unit_orth(rng, b0, [o0])
    o1 = _unit_orth(rng, b0, [o0, b1])
    b2 = _unit_orth(rng, b0, [o0, b1, o1])
    o2 = _unit_orth(rng, b0, [o0, b1, o1, b2])
    b4 = _unit_orth(rng, b0, [o0, b1, o1, b2, o2])

    centers = {
        "0a": b0,
        "0b": 0.88 * b0 + np.sqrt(1.0 - 0.88**2) * o0,
        "1a": b1,
        "1b": 0.66 * b1 + np.sqrt(1.0 - 0.66**2) * o1,
        "2": b2,
        "3": 0.62 * b2 + np.sqrt(1.0 - 0.62**2) * o2,
        "4": b4,
    }
    for k in centers:
        centers[k] = (centers[k] / np.linalg.norm(centers[k])).astype(np.float32)

    stream = []
    block_cl: Dict[int, List[Tuple[int, int]]] = {}
    amis = []
    print("=== [SUG-SED v2 Standalone Pipeline] Synthetic 4-Block Verification ===")
    for bid in range(4):
        ds, emb = _make_synthetic_block(bid, centers)
        graph = build_message_graph(
            ds, emb, knn_k=8, knn_sim_threshold=0.50, attr_alpha=0.5, compute_non_entity_tfidf=False
        )
        lab0, _ = solve_2d_se(graph.adj, refine_moves=True)
        aud0 = full_partition_audit(ds.labels, lab0)
        pred, init_out, shown, stats = run_pure_star_meso(
            "synthetic",
            block_id=bid,
            ds=ds,
            embeddings=emb,
            client=None,
            dry_run_oracle=True,
            graph=graph,
            labels_init=lab0,
        )
        aud = full_partition_audit(ds.labels, pred, y_before=init_out, queried_mask=shown)
        ami = aud["ami"]
        amis.append(ami)
        assert ami >= 0.85, f"Synthetic block M_{bid:02d} AMI too low: {ami:.4f}"
        assert ami >= aud0["ami"] - 1e-6, f"Pipeline degraded base AMI: {aud0['ami']:.4f} -> {ami:.4f}"
        assert stats["auto_merge_ops"] >= 1, f"Stage-2 AutoMerge not exercised: {stats}"
        assert stats["seam_calls"] >= 1 and stats["seam_splits"] >= 1, f"Stage-3 Seam not exercised: {stats}"
        assert stats["star_calls"] >= 1 and stats["star_merges"] >= 1, f"Stage-4 Star-Merge not exercised: {stats}"
        assert stats["parse_errors"] == 0
        print(
            f"  Block M_{bid:02d}: Base AMI={aud0['ami']*100:5.2f} -> "
            f"Pure-Star-Meso AMI={ami*100:5.2f} MacB3={aud['bcubed_full']['macro_b3_f1']*100:5.2f} "
            f"Auto={stats['auto_merge_ops']} Seam={stats['seam_splits']}/{stats['seam_calls']} "
            f"Star={stats['star_merges']}/{stats['star_calls']}"
        )
        stream.append((bid, ds, emb, pred))
        block_cl[bid] = stats.get("intra_cannot_link_pairs", [])

    cd = evaluate_cross_day_stream(
        stream,
        mode="oracle_band",
        eval_start_block=2,
        tau_lo=0.65,
        tau_hi=0.90,
        tau_fb=0.90,
        min_cross_size=3,
        block_cannot_links=block_cl,
    )
    print(
        f"  Cross-Day (eval M_02..M_03): G-AMI={cd['g_ami']:.2f} "
        f"G-MacB3={cd['g_mac_b3']:.2f} G-MicB3={cd['g_mic_b3']:.2f} G-TailB3={cd['g_tail_b3']:.2f} "
        f"(mean within-block AMI={np.mean(amis)*100:.2f})"
    )
    assert cd["g_ami"] >= 85.0, f"Synthetic Cross-Day G-AMI too low: {cd}"
    print("SYNTHETIC STANDALONE PIPELINE PASSED cleanly.")


def run_dataset_pipeline(args: argparse.Namespace) -> None:
    """Run the 5-stage SUG-SED v2 pipeline on selected SocialED blocks."""
    ds_name = args.dataset
    block_ids = [int(x.strip()) for x in args.blocks.split(",") if x.strip()]
    if not block_ids:
        raise ValueError("At least one block id must be provided via --blocks.")

    embed_model = args.embed_model or EMBED_MODELS.get(ds_name, "sentence-transformers/all-MiniLM-L6-v2")
    ds_full = load_socialed_npy(ds_name, sort_chronologically=True)
    daily_blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)

    client: Optional[BudgetedLLMClient] = None
    if not args.dry_run_oracle:
        base_url, default_model, api_key = load_provider_key(args.provider)
        model_name = args.model or default_model
        log_dir = args.log_dir / ds_name
        shared_cache = args.log_dir / f"{args.provider}_{model_name.replace('/', '_')}_shared_responses.jsonl"
        ledger = TokenLedger(log_dir=log_dir, shared_cache_file=shared_cache)
        client = BudgetedLLMClient(
            model=model_name,
            ledger=ledger,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            max_tokens=384,
            timeout_sec=45.0,
        )

    print(
        f"=== [SUG-SED v2 Standalone] {ds_name} blocks={block_ids} "
        f"mode={'dry_run_oracle' if args.dry_run_oracle else args.provider} ==="
    )

    use_purity_base = not args.no_purity_base
    knn_tau = 0.50 if use_purity_base else 0.35

    block_outputs: Dict[int, Tuple[SocialDataset, np.ndarray, np.ndarray]] = {}
    block_cl: Dict[int, List[Tuple[int, int]]] = {}
    for bid in block_ids:
        _bname, bidx = daily_blocks[bid]
        sd = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        emb = load_or_extract_block_embeddings(ds_name, bid, sd.texts, model_name=embed_model)
        graph = build_message_graph(
            sd,
            emb,
            knn_k=15,
            knn_sim_threshold=knn_tau,
            attr_alpha=0.5,
            compute_non_entity_tfidf=False,
        )
        lab0, se_sec = load_or_solve_2d_se(
            ds_name,
            bid,
            graph.adj,
            knn_k=15,
            knn_sim_threshold=knn_tau,
            attr_alpha=0.5,
            refine_moves=True,
            embed_model=embed_model,
        )

        bk = (
            CausalUniformBucket(tokens_per_block=args.budget_per_block, allow_carryover=False)
            if not args.dry_run_oracle
            else None
        )
        if bk is not None:
            bk.start_block()

        anonymizer = QueryAnonymizer(args.anonymize) if args.anonymize != "none" else None
        audit_store = AnonymizationAuditStore() if anonymizer is not None else None

        pred, init_out, shown, stats = run_pure_star_meso(
            ds_name,
            bid,
            sd,
            emb,
            client=client,
            bucket=bk,
            use_purity_base=use_purity_base,
            dry_run_oracle=args.dry_run_oracle,
            graph=graph,
            labels_init=lab0,
            anonymizer=anonymizer,
            audit_store=audit_store,
            embed_model=embed_model,
        )
        aud0 = full_partition_audit(sd.labels, init_out)
        aud = full_partition_audit(sd.labels, pred, y_before=init_out, queried_mask=shown)
        pa = aud.get("propagation_audit", {})
        print(
            f"  M_{bid:02d} (n={len(sd)}, se={se_sec:.2f}s) | "
            f"Base AMI={aud0['ami']*100:5.2f} -> Ours AMI={aud['ami']*100:5.2f} "
            f"MacB3={aud['bcubed_full']['macro_b3_f1']*100:5.2f} "
            f"MicB3={aud['bcubed_full']['micro_b3_f1']*100:5.2f} "
            f"TailB3={aud['stratified']['tail']['macro_b3_f1']*100:5.2f} | "
            f"UnqNet={pa.get('unqueried_net_improved_posts', 0):+d} "
            f"Calls={stats['total_calls']} Tok={stats['total_tokens']} Err={stats['parse_errors']}"
        )
        block_outputs[bid] = (sd, emb, pred)
        block_cl[bid] = stats.get("intra_cannot_link_pairs", [])

    if args.cross_day:
        min_eval = min(block_ids)
        # Align with protocol_v2.json: warmup uses M_00..M_05 (all blocks prior to min_eval up to 6)
        warmup_ids = [w for w in range(min(6, min_eval))]
        stream: List[Tuple[int, SocialDataset, np.ndarray, np.ndarray]] = []
        for wid in warmup_ids:
            _bname, widx = daily_blocks[wid]
            wsd = ds_full.subset(widx, name_suffix=f"_M{wid:02d}")
            wemb = load_or_extract_block_embeddings(ds_name, wid, wsd.texts, model_name=embed_model)
            wg = build_message_graph(
                wsd,
                wemb,
                knn_k=15,
                knn_sim_threshold=knn_tau,
                attr_alpha=0.5,
                compute_non_entity_tfidf=False,
            )
            wl, _ = load_or_solve_2d_se(
                ds_name,
                wid,
                wg.adj,
                knn_k=15,
                knn_sim_threshold=knn_tau,
                attr_alpha=0.5,
                refine_moves=True,
                embed_model=embed_model,
            )
            stream.append((wid, wsd, wemb, wl))
        eval_start = min_eval if warmup_ids else (sorted(block_ids)[1] if len(block_ids) >= 2 else min_eval)
        for bid in sorted(block_ids):
            sd, emb, pred = block_outputs[bid]
            stream.append((bid, sd, emb, pred))

        cd_mode = "oracle_band" if args.dry_run_oracle else "sug_cross_star"
        cd = evaluate_cross_day_stream(
            stream,
            mode=cd_mode,
            eval_start_block=eval_start,
            tau_lo=0.65,
            tau_hi=0.90,
            tau_fb=0.90,
            min_cross_size=3,
            client=client,
            budget_per_block=args.cross_budget,
            block_cannot_links=block_cl,
        )
        print(
            f"  [Cross-Day {cd_mode}, eval>={eval_start}] "
            f"G-AMI={cd['g_ami']:.2f} G-MacB3={cd['g_mac_b3']:.2f} "
            f"G-MicB3={cd['g_mic_b3']:.2f} G-TailB3={cd['g_tail_b3']:.2f} "
            f"Q={cd['stats']['llm_queried']} Tok={cd['stats']['cross_tokens']} Err={cd['stats']['parse_errors']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="SUG-SED v2 Standalone Method Pipeline")
    parser.add_argument("--mode", choices=["synthetic", "dataset"], default="synthetic")
    parser.add_argument("--dataset", default="Event2012", choices=["Event2012", "Event2012_100", "Event2018"])
    parser.add_argument("--blocks", type=str, default="2,3", help="Comma-separated block indices (e.g. 2,3)")
    parser.add_argument("--embed-model", type=str, default="")
    parser.add_argument("--no-purity-base", action="store_true", help="Use tau=0.35 base instead of purity-first 0.50")
    parser.add_argument("--dry-run-oracle", action="store_true", help="Use oracle simulation with zero API calls")
    parser.add_argument("--provider", type=str, default="siliconflow")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--budget-per-block", type=int, default=10000)
    parser.add_argument("--cross-budget", type=int, default=6000)
    parser.add_argument("--cross-day", action="store_true", help="Run Stage-5 SUG-CrossStar across blocks")
    parser.add_argument("--anonymize", choices=["none", "pseudonym", "blind"], default="none")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Directory for TokenLedger logs when calling live LLM APIs",
    )
    args = parser.parse_args()

    if args.mode == "synthetic":
        run_synthetic_demo()
    else:
        run_dataset_pipeline(args)


if __name__ == "__main__":
    main()
