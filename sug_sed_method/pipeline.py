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
    base_centers: np.ndarray,
    n_per_ev: int = 12,
    seed: int = 7,
) -> Tuple[SocialDataset, np.ndarray]:
    rng = np.random.default_rng(seed + block_id)
    n_ev, dim = base_centers.shape
    embs, labels, texts, entities, hashtags, lemmas = [], [], [], [], [], []
    day_str = f"2026-01-{block_id + 1:02d}"
    for e in range(n_ev):
        for k in range(n_per_ev):
            v = base_centers[e] + 0.08 * rng.normal(size=dim).astype(np.float32)
            v /= max(float(np.linalg.norm(v)), 1e-12)
            embs.append(v)
            labels.append(e)
            texts.append(f"Breaking incident City{e} summit #tag{e} day{block_id} update {k}")
            entities.append([f"city{e}"])
            hashtags.append([f"tag{e}"])
            lemmas.append(["breaking", "incident", "summit", "update"])
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
        timestamps=np.arange(n, dtype=np.float64) * 60.0 + (1e9 + block_id * 86400.0),
        dates=[day_str] * n,
    )
    return ds, np.vstack(embs)


def run_synthetic_demo() -> None:
    """Run offline 4-block verification covering all 5 SUG-SED v2 stages."""
    rng = np.random.default_rng(7)
    n_ev, dim = 4, 32
    centers = rng.normal(size=(n_ev, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    c1 = centers[1] - float(np.dot(centers[1], centers[0])) * centers[0]
    c1 /= max(float(np.linalg.norm(c1)), 1e-9)
    centers[1] = 0.52 * centers[0] + np.sqrt(1.0 - 0.52**2) * c1
    centers[1] /= np.linalg.norm(centers[1])

    stream = []
    amis = []
    print("=== [SUG-SED v2 Standalone Pipeline] Synthetic 4-Block Verification ===")
    for bid in range(4):
        ds, emb = _make_synthetic_block(bid, centers)
        graph = build_message_graph(ds, emb, knn_k=6, knn_sim_threshold=0.50, attr_alpha=0.5)
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
        assert stats["parse_errors"] == 0
        print(
            f"  Block M_{bid:02d}: Base AMI={aud0['ami']*100:5.2f} -> "
            f"Pure-Star-Meso AMI={ami*100:5.2f} MacB3={aud['bcubed_full']['macro_b3_f1']*100:5.2f} "
            f"Auto={stats['auto_merge_ops']} StarCalls={stats['star_calls']} SeamCalls={stats['seam_calls']}"
        )
        stream.append((bid, ds, emb, pred))

    cd = evaluate_cross_day_stream(
        stream,
        mode="oracle_band",
        eval_start_block=2,
        tau_lo=0.65,
        tau_hi=0.90,
        tau_fb=0.90,
        min_cross_size=3,
    )
    print(
        f"  Cross-Day (eval M_02..M_03): G-AMI={cd['g_ami']:.2f} "
        f"G-MacB3={cd['g_mac_b3']:.2f} G-MicB3={cd['g_mic_b3']:.2f} G-TailB3={cd['g_tail_b3']:.2f} "
        f"(mean within-block AMI={np.mean(amis)*100:.2f})"
    )
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

    block_outputs: Dict[int, Tuple[SocialDataset, np.ndarray, np.ndarray]] = {}
    for bid in block_ids:
        _bname, bidx = daily_blocks[bid]
        sd = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        emb = load_or_extract_block_embeddings(ds_name, bid, sd.texts, model_name=embed_model)
        graph = build_message_graph(sd, emb, knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5)
        lab0, se_sec = load_or_solve_2d_se(
            ds_name,
            bid,
            graph.adj,
            knn_k=15,
            knn_sim_threshold=0.50,
            attr_alpha=0.5,
            refine_moves=True,
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
            dry_run_oracle=args.dry_run_oracle,
            graph=graph,
            labels_init=lab0,
            anonymizer=anonymizer,
            audit_store=audit_store,
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

    if args.cross_day:
        min_eval = min(block_ids)
        warmup_ids = [w for w in (0, 1) if w < min_eval]
        stream: List[Tuple[int, SocialDataset, np.ndarray, np.ndarray]] = []
        for wid in warmup_ids:
            _bname, widx = daily_blocks[wid]
            wsd = ds_full.subset(widx, name_suffix=f"_M{wid:02d}")
            wemb = load_or_extract_block_embeddings(ds_name, wid, wsd.texts, model_name=embed_model)
            wg = build_message_graph(wsd, wemb, knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5)
            wl, _ = load_or_solve_2d_se(
                ds_name, wid, wg.adj, knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5, refine_moves=True
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
