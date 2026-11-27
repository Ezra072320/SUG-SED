"""Validation benchmark on Event2012 M_01..M_05 (strictly ARI, NMI, AMI).

Compares FRAG-SED against strictly aligned implementations across 3 baseline categories:
  Group 1 — 0-LLM Original (0 Tokens):
    1. SBERT*    (Reimers & Gurevych, EMNLP 2019): preprocessed text + KMeans(Oracle-K)
    2. HISEvent  (Cao et al., AAAI 2024 — SELGroup/HISEvent): raw unified attributes + 1D/2D-SE
  Group 2 — Unbounded LLM-SED Original (Original Protocol, Token Cost Reported):
    3. RagSEDE   (Liu et al., WWW 2026 — SELGroup/RagSEDE)
    4. LSED      (Qiu et al., ACL 2025 — 2025.acl-long.233): 3-layer HGCN Poincare MLP (w=infty)
  Group 3 — Budgeted LLM Baselines (20k Tokens / Block):
    5. COBRAS    (Van Craenendonck et al., IDA 2018 — ML-KULeuven/cobras, Binary LLM oracle)
    6. Cequel    (Wang et al., CIKM 2025 — HKBU-LAGAS/Cequel): WCSC (primary) & WCKMeans
  Group 4 — Ours:
    7. FRAG-SED  (20k Budget/Blk & Unbounded)
"""

from __future__ import annotations

import os

for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import SocialDataset, load_socialed_npy, split_daily_blocks
from core_shared.graph_builder import build_message_graph
from core_shared.key_loader import load_provider_key
from core_shared.partition_cache import load_or_solve_2d_se
from core_shared.token_ledger import CausalUniformBucket, TokenLedger
from trackA_sug_sed.baselines_official import (
    build_lsed_official_prompt,
    preprocess_for_dataset,
    preprocess_social_sentence,
    run_cequel_official,
    run_cobras_official,
    run_hisevent_official,
    run_lsed_official,
    run_ragsede_official,
    run_sbert_kmeans_official,
    train_lsed_hyperbolic_encoder,
)
from trackA_sug_sed.embed_cache import load_or_extract_block_embeddings
from trackA_sug_sed.frag_sed import run_frag_sed
from trackA_sug_sed.llm_client import BudgetedLLMClient

VAL_BLOCKS = [1, 2, 3, 4, 5]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BUDGET_20K = 20000
OUT_DIR = ROOT / "results" / "val_m01_m05_benchmark"


def _dataset_encoders(dataset_name: str) -> Dict[str, str]:
    """Return dataset-specific encoders matching official baselines and FRAG-SED paper spec."""
    if "2018" in dataset_name:
        return {
            "official_prep": "sentence-transformers/distiluse-base-multilingual-cased-v1",
            "raw_main": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "lsed_summary": "sentence-transformers/all-MiniLM-L6-v2",
        }
    return {
        "official_prep": "sentence-transformers/all-MiniLM-L6-v2",
        "raw_main": "sentence-transformers/all-MiniLM-L6-v2",
        "lsed_summary": "sentence-transformers/all-MiniLM-L6-v2",
    }


def _eval_3metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "ari": round(float(adjusted_rand_score(y_true, y_pred) * 100.0), 2),
        "nmi": round(float(normalized_mutual_info_score(y_true, y_pred) * 100.0), 2),
        "ami": round(float(adjusted_mutual_info_score(y_true, y_pred) * 100.0), 2),
    }


def _aggregate_method(
    per_block_metrics: Dict[str, Dict[str, Any]],
    val_blocks: Sequence[int],
) -> Dict[str, Any]:
    aris = [per_block_metrics[f"M_{b:02d}"]["ari"] for b in val_blocks]
    nmis = [per_block_metrics[f"M_{b:02d}"]["nmi"] for b in val_blocks]
    amis = [per_block_metrics[f"M_{b:02d}"]["ami"] for b in val_blocks]
    tot_calls = sum(int(per_block_metrics[f"M_{b:02d}"].get("calls", 0)) for b in val_blocks)
    tot_tok = sum(int(per_block_metrics[f"M_{b:02d}"].get("tokens", 0)) for b in val_blocks)
    tot_err = sum(int(per_block_metrics[f"M_{b:02d}"].get("parse_errors", 0)) for b in val_blocks)
    return {
        "mean": {
            "ari": round(float(np.mean(aris)), 2),
            "nmi": round(float(np.mean(nmis)), 2),
            "ami": round(float(np.mean(amis)), 2),
        },
        "total_calls": tot_calls,
        "total_tokens": tot_tok,
        "total_parse_errors": tot_err,
        "per_block": per_block_metrics,
    }


def _prepare_context(dataset_name: str, val_blocks: Sequence[int], include_m0: bool = True) -> Dict[str, Any]:
    ds_full = load_socialed_npy(dataset_name, sort_chronologically=True)
    blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)
    encs = _dataset_encoders(dataset_name)

    ds_map: Dict[int, SocialDataset] = {}
    emb_prep_map: Dict[int, np.ndarray] = {}
    emb_raw_f32_map: Dict[int, np.ndarray] = {}
    init_2dse_map: Dict[int, np.ndarray] = {}

    block_ids = ([0] if include_m0 else []) + [b for b in val_blocks if not (include_m0 and b == 0)]
    for bid in block_ids:
        _bname, bidx = blocks[bid]
        sd = ds_full.subset(bidx, name_suffix=f"_M{bid:02d}")
        prep_texts = [preprocess_for_dataset(t, dataset_name) for t in sd.texts]
        em_prep = load_or_extract_block_embeddings(
            f"{dataset_name}_prep",
            bid,
            prep_texts,
            model_name=encs["official_prep"],
            storage_dtype="float32",
            allow_legacy_cache=False,
            max_length=256,
        )
        em_raw_f32 = load_or_extract_block_embeddings(
            f"{dataset_name}_raw",
            bid,
            sd.texts,
            model_name=encs["raw_main"],
            storage_dtype="float32",
            allow_legacy_cache=False,
            max_length=256,
        )
        ds_map[bid] = sd
        emb_prep_map[bid] = em_prep
        emb_raw_f32_map[bid] = em_raw_f32
        if bid in val_blocks:
            g = build_message_graph(
                sd, em_raw_f32, knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5, compute_non_entity_tfidf=False
            )
            lbl, _ = load_or_solve_2d_se(
                dataset_name=f"{dataset_name}_raw_f32_ml256",
                block_id=bid,
                adj=g.adj,
                knn_k=15,
                knn_sim_threshold=0.50,
                attr_alpha=0.5,
                refine_moves=True,
                embed_model=encs["raw_main"],
            )
            init_2dse_map[bid] = lbl

    return {
        "dataset_name": dataset_name,
        "encoders": encs,
        "ds": ds_map,
        "emb": emb_raw_f32_map,
        "emb_prep": emb_prep_map,
        "emb_raw_f32": emb_raw_f32_map,
        "init_2dse": init_2dse_map,
    }


def export_colab_prompts(ctx: Dict[str, Any], val_blocks: Sequence[int], colab_in_dir: Path) -> Dict[str, int]:
    """Export unbounded LSED LLM tweet-rewrite prompts for Colab batch inference."""
    colab_in_dir.mkdir(parents=True, exist_ok=True)
    b_lo, b_hi = min(val_blocks), max(val_blocks)
    lsed_path = colab_in_dir / f"lsed_M{b_lo:02d}_M{b_hi:02d}_prompts.jsonl"
    lsed_m0_path = colab_in_dir / "lsed_M00_prompts.jsonl"

    l_cnt = 0
    with lsed_path.open("w", encoding="utf-8") as lf:
        for bid in val_blocks:
            sd = ctx["ds"][bid]
            for i in range(len(sd)):
                sys_m, usr_m = build_lsed_official_prompt(sd.texts[i])
                rec = {
                    "request_id": f"lsed_M{bid:02d}_n{i}",
                    "block_id": bid,
                    "sys_msg": sys_m,
                    "user_msg": usr_m,
                    "max_tokens": 256,
                }
                lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                l_cnt += 1

    l0_cnt = 0
    sd0 = ctx["ds"][0]
    with lsed_m0_path.open("w", encoding="utf-8") as l0f:
        for i in range(len(sd0)):
            sys_m, usr_m = build_lsed_official_prompt(sd0.texts[i])
            rec = {
                "request_id": f"lsed_M00_n{i}",
                "block_id": 0,
                "sys_msg": sys_m,
                "user_msg": usr_m,
                "max_tokens": 256,
            }
            l0f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            l0_cnt += 1

    return {"lsed_prompts": l_cnt, "lsed_m0_prompts": l0_cnt}


def _load_colab_responses(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("request_id")
                if rid:
                    out[str(rid)] = obj
            except json.JSONDecodeError:
                continue
    return out


def _make_client(
    provider: str, ledger_name: str, out_dir: Path, model_override: Optional[str], pace_sec: float
) -> BudgetedLLMClient:
    base_url, def_model, api_key = load_provider_key(provider)
    model_name = model_override or def_model
    ledger_dir = out_dir / "ledgers" / ledger_name
    shared_cache = out_dir / "ledgers" / "shared_cache.jsonl"
    ledger = TokenLedger(ledger_dir, shared_cache_file=shared_cache)
    return BudgetedLLMClient(
        model=model_name,
        ledger=ledger,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
        max_tokens=384,
        timeout_sec=60.0,
        use_json_mode=True,
        enable_thinking=False,
        min_pace_sec=pace_sec,
    )


def _render_report(
    results: Dict[str, Any],
    out_md: Path,
    val_blocks: Sequence[int],
    dataset_name: str = "Event2012",
    encoders: Optional[Dict[str, str]] = None,
) -> None:
    method_order = [
        ("sbert_kmeans", "0-LLM Original", "SBERT* (KMeans, Oracle-K)", "0 Tokens"),
        ("hisevent_official", "0-LLM Original", "HISEvent Official (AAAI'24)", "0 Tokens"),
        ("ragsede_unbounded", "Unbounded Original", "RagSEDE Official (WWW'26)", "Unbounded (Cost Reported)"),
        ("lsed_unbounded_oracleK", "Unbounded Original", "LSED Official HGCN (ACL'25, Oracle-K, w=infty)", "Unbounded (All N)"),
        ("lsed_unbounded_estK", "Unbounded Original", "LSED Official HGCN (ACL'25, Est-K, w=infty)", "Unbounded (All N)"),
        ("cobras_20k", "Budgeted LLM (20k)", "COBRAS Official (IDA'18, Binary LLM)", "20k Budget/Blk"),
        ("cequel_20k_wcsc_oracleK", "Budgeted LLM (20k)", "Cequel Official WCSC (CIKM'25, Oracle-K)", "20k Budget/Blk"),
        ("cequel_20k_wcsc_estK", "Budgeted LLM (20k)", "Cequel Official WCSC (CIKM'25, Est-K)", "20k Budget/Blk"),
        ("cequel_20k_wckmeans_oracleK", "Budgeted LLM (20k)", "Cequel WCKMeans (CIKM'25, Oracle-K)", "20k Budget/Blk"),
        ("frag_sed_20k", "Ours (FRAG-SED)", "**FRAG-SED (Ours)**", "20k Budget/Blk"),
        ("frag_sed_unbounded", "Ours (FRAG-SED)", "**FRAG-SED (Ours)**", "Unbounded"),
    ]

    b_lo, b_hi = min(val_blocks), max(val_blocks)
    span_str = f"M_{b_lo:02d}..M_{b_hi:02d}"
    blk_cols = [f"M_{b:02d} (ARI/NMI/AMI)" for b in val_blocks]
    blk_align = ["---:"] * len(val_blocks)

    encs = encoders or _dataset_encoders(dataset_name)
    prep_short = encs["official_prep"].split("/")[-1]
    raw_short = encs["raw_main"].split("/")[-1]
    lsed_short = encs["lsed_summary"].split("/")[-1]
    enc_banner = (
        f"{raw_short} / {prep_short}" if prep_short != raw_short else raw_short
    )

    lines = [
        f"# {dataset_name} Benchmark (`{span_str}`, `Qwen/Qwen3-8B`, `{enc_banner}`)",
        "",
        f"> Metrics strictly reported as **ARI / NMI / AMI (%)** per block (`{span_str}`) and {len(val_blocks)}-block average (`Mean`).",
        "",
        "| Group | Method | Setting | " + " | ".join(blk_cols) + " | **Mean (ARI / NMI / AMI)** | Calls / Tokens |",
        "|---|---|---|" + "|".join(blk_align) + "|---:|---:|",
    ]

    for key, group, label, setting in method_order:
        if key not in results:
            pend = ["*Pending Run*"] + ["-"] * (len(val_blocks) - 1)
            lines.append(f"| {group} | {label} | {setting} | " + " | ".join(pend) + " | *Pending* | - |")
            continue
        entry = results[key]
        pb = entry["per_block"]
        cells = []
        for b in val_blocks:
            bm = pb[f"M_{b:02d}"]
            cells.append(f"{bm['ari']:.2f} / {bm['nmi']:.2f} / {bm['ami']:.2f}")
        mn = entry["mean"]
        mean_cell = f"**{mn['ari']:.2f} / {mn['nmi']:.2f} / {mn['ami']:.2f}**"
        cost_cell = f"{entry['total_calls']} / {entry['total_tokens']:,}"
        lines.append(f"| {group} | {label} | {setting} | " + " | ".join(cells) + f" | {mean_cell} | {cost_cell} |")

    lines.append("")
    lines.append("## Per-Metric Expanded Summary (Mean ARI / NMI / AMI)")
    lines.append("")
    lines.append("| Group | Method | Setting | ARI | NMI | AMI | Total Calls | Total Tokens | Parse Errors |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for key, group, label, setting in method_order:
        if key not in results:
            continue
        entry = results[key]
        mn = entry["mean"]
        lines.append(
            f"| {group} | {label} | {setting} | {mn['ari']:.2f} | {mn['nmi']:.2f} | {mn['ami']:.2f} | "
            f"{entry['total_calls']} | {entry['total_tokens']:,} | {entry['total_parse_errors']} |"
        )

    held_blocks = [b for b in val_blocks if b >= 6]
    if held_blocks and any(b < 6 for b in val_blocks):
        h_lo, h_hi = min(held_blocks), max(held_blocks)
        lines.extend([
            "",
            f"## Held-Out Test Split Summary (`M_{h_lo:02d}..M_{h_hi:02d}`, {len(held_blocks)} Blocks)",
            "",
            "| Group | Method | Setting | ARI | NMI | AMI | Calls | Tokens | Parse Errors |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for key, group, label, setting in method_order:
            if key not in results:
                continue
            pb = results[key]["per_block"]
            if not all(f"M_{b:02d}" in pb for b in held_blocks):
                continue
            h_ari = float(np.mean([pb[f"M_{b:02d}"]["ari"] for b in held_blocks]))
            h_nmi = float(np.mean([pb[f"M_{b:02d}"]["nmi"] for b in held_blocks]))
            h_ami = float(np.mean([pb[f"M_{b:02d}"]["ami"] for b in held_blocks]))
            h_calls = sum(int(pb[f"M_{b:02d}"].get("calls", 0)) for b in held_blocks)
            h_tok = sum(int(pb[f"M_{b:02d}"].get("tokens", 0)) for b in held_blocks)
            h_err = sum(int(pb[f"M_{b:02d}"].get("parse_errors", 0)) for b in held_blocks)
            lines.append(
                f"| {group} | {label} | {setting} | {h_ari:.2f} | {h_nmi:.2f} | {h_ami:.2f} | "
                f"{h_calls} | {h_tok:,} | {h_err} |"
            )

    lsed_m0_cost = (results.get("lsed_unbounded_oracleK") or {}).get("m0_summary_cost")
    if lsed_m0_cost:
        lines.extend([
            "",
            f"> Note: LSED `M_00` training-set LLM summarization cost: "
            f"{lsed_m0_cost.get('calls', 0):,} calls / {lsed_m0_cost.get('tokens', 0):,} tokens "
            f"(separate from test-time `{span_str}` token totals above).",
        ])

    sim_thresh_str = "0.30" if "2018" in dataset_name else "0.40"
    n_sub_str = "300" if "2018" in dataset_name else "400"
    lines.extend([
        "",
        "## Protocol Alignment & Documented Unavoidable Exceptions",
        "",
        "1. **Shared LLM Backbone (`Qwen/Qwen3-8B`)**: RagSEDE (Qwen2/2.5/Llama3), Cequel (GPT-3.5-Turbo/Llama3-8B), and LSED (Llama3.1-8B/Qwen2.5-7B/Gemma2-9B) originally used different LLMs; all LLM baselines use unified `Qwen/Qwen3-8B` (deterministic `temperature=0.0` for `FRAG-SED`/`RagSEDE`/`COBRAS`/`LSED`, and official `temp=1.0, n=5` majority voting for `Cequel`, with `enable_thinking=False`) for fair comparison.",
        f"2. **3-Category Benchmark Design ({dataset_name})**: Keeps 0-LLM original (`SBERT*`, `HISEvent Official` with language-specific preprocessing, `n_sub={n_sub_str}`, `{prep_short}` encoder & `eps=0.0`), unbounded original LLM-SED with exact token-cost accounting (`RagSEDE Official`, `LSED Official w=infty`), budgeted active-query LLM baselines at `20k tokens/block` (`COBRAS Official`, `Cequel Official` on `{raw_short}`), and `FRAG-SED` (`20k Budget/Blk` & `Unbounded` on `{raw_short}`).",
        f"3. **RagSEDE Infrastructure**: Uses official `max_retries=10`, `similarity_threshold={sim_thresh_str}`, and `use_json_mode=False` (non-forced JSON output parsed from ```json blocks or raw JSON); external RAGFlow v0.17.2 server (`BAAI/bge-small-zh-v1.5`) is replaced by a functionally equivalent in-memory hybrid index (`{prep_short}` dense cosine + IDF-weighted keyword similarity excluding prompt boilerplate, retrieved by `Enc('INPUT: ' + a_k[:200])`) with `temp=0.2` retry jitter before falling back to `'Confusion'`.",
        "4. **COBRAS LLM Oracle Adaptation**: Original IDA'18 COBRAS assumes a noise-free ground-truth oracle; under the budgeted setting, queries use a concise binary `Yes/No` `Qwen/Qwen3-8B` oracle (`max_tokens=8`) capped cleanly by the `20k tokens/block` causal budget.",
        "5. **Cequel Solver & Sampling**: Uses official 5-sample majority voting (`n=5, temp=1.0, max_tokens=1, max_vote_retries=5`) via `call_text_n`, official `WCSC` (`Q` in `float64` per paper Eq. 4, sparse `eigsh` for `n > 2500`, primary on tweets, `[0.5, 1.5]` Q weights) and `WCKMeans` (`init='random'`, `ML in [0.01, 0.1]`, `CL in [0, 0.01]`) under the `20k tokens/block` causal budget.",
        f"6. **LSED Open-Set Window `w=infty`**: Paper Fig. 6 open-set setting (`w=infty`: train 3-layer HGCN Poincare MLP once on LLM-summarized `M_00`, cluster LLM-summarized `{span_str}` via KMeans on `{lsed_short}` embeddings, numeric `tan_reg=0.02` preventing 100% Poincare boundary saturation) replaces supervised `w=1` updates; both `M_00` (`lsed_M00_prompts.jsonl`) and `{span_str}` (`lsed_M{b_lo:02d}_M{b_hi:02d}_prompts.jsonl`) summaries are required to avoid train/test distribution shift.",
        "7. **`Oracle-K` vs `Est-K`**: Cequel and LSED originally assume known `K*`; unsupervised `Est-K` uses `HISEvent Official`'s predicted cluster count (`n_pred_clusters`) alongside `Oracle-K`.",
    ])

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", type=str, default="Event2012")
    parser.add_argument("--provider", type=str, default="siliconflow")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--pace-sec", type=float, default=0.15)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--methods",
        type=str,
        default="sbert_kmeans,hisevent_official,ragsede_unbounded,lsed_unbounded,cobras_20k,cequel_20k,frag_sed_20k,frag_sed_unbounded",
    )
    parser.add_argument("--use-colab-outputs", type=str, default=None)
    parser.add_argument("--blocks", type=str, default=",".join(str(b) for b in VAL_BLOCKS))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--skip-export-colab", action="store_true")
    parser.add_argument("--allow-online-lsed", action="store_true")
    parser.add_argument("--allow-unsummarized-lsed-m0", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset
    val_blocks = [int(x.strip()) for x in args.blocks.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    include_m0 = not (args.skip_export_colab and "lsed_unbounded" not in methods)

    ctx = _prepare_context(dataset_name, val_blocks, include_m0=include_m0)
    encs = ctx["encoders"]
    colab_in = out_dir / "colab_inputs"

    b_lo, b_hi = min(val_blocks), max(val_blocks)
    print(f"=== {dataset_name} Blocks (M_{b_lo:02d}..M_{b_hi:02d}{', +M_00' if include_m0 else ''}) ===")
    for bid in sorted(ctx["ds"].keys()):
        sd = ctx["ds"][bid]
        print(f"  M_{bid:02d}: N={len(sd)}, K*={len(set(sd.labels))}")
    if not args.skip_export_colab and 0 in ctx["ds"]:
        exp_counts = export_colab_prompts(ctx, val_blocks, colab_in)
        print(f"Exported Colab LSED prompts: {exp_counts} -> {colab_in}")

    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    res_json_path = out_dir / "val_benchmark_results.json"
    res_md_path = out_dir / "val_benchmark_report.md"
    if res_json_path.exists():
        results: Dict[str, Any] = json.loads(res_json_path.read_text(encoding="utf-8"))
    else:
        results = {}

    def _save() -> None:
        res_json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        _render_report(results, res_md_path, val_blocks, dataset_name=dataset_name, encoders=encs)

    pool_workers = max(1, min(int(args.max_workers), len(val_blocks)))

    def _run_blocks_parallel(fn_block):
        pb_out: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=pool_workers) as ex:
            futs = {ex.submit(fn_block, bid): bid for bid in val_blocks}
            for fut, bid in futs.items():
                pb_out[f"M_{bid:02d}"] = fut.result()
        return pb_out

    # Group 1A: SBERT* Literature-Standard KMeans(Oracle-K)
    if "sbert_kmeans" in methods:
        pb_sbert_km: Dict[str, Any] = {}
        for bid in val_blocks:
            k_or = len(set(ctx["ds"][bid].labels))
            lbl, st = run_sbert_kmeans_official(ctx["emb_prep"][bid], k_oracle=k_or, seed=42)
            m3 = _eval_3metrics(ctx["ds"][bid].labels, lbl)
            m3.update({"calls": 0, "tokens": 0, "parse_errors": 0, **st})
            pb_sbert_km[f"M_{bid:02d}"] = m3
        results["sbert_kmeans"] = _aggregate_method(pb_sbert_km, val_blocks)
        _save()
        print(f"[SBERT* (KMeans, Oracle-K)] Mean ARI/NMI/AMI = {results['sbert_kmeans']['mean']}", flush=True)

    # Group 1B: HISEvent Official (also supplies unsupervised Est-K for Cequel & LSED)
    hisevent_k_est: Dict[int, int] = {}
    need_hisevent_k = any(m in methods for m in ("cequel_20k", "lsed_unbounded"))
    if "hisevent_official" in methods:
        pb_his: Dict[str, Any] = {}
        for bid in val_blocks:
            lbl, st = run_hisevent_official(
                ctx["ds"][bid],
                embeddings=ctx["emb_prep"][bid],
                n_sub=None,
                max_num_neighbors=200,
                dataset_name=dataset_name,
                use_raw_attributes=True,
            )
            hisevent_k_est[bid] = int(st.get("n_pred_clusters", len(set(lbl))))
            m3 = _eval_3metrics(ctx["ds"][bid].labels, lbl)
            m3.update({"calls": 0, "tokens": 0, "parse_errors": 0, **st})
            pb_his[f"M_{bid:02d}"] = m3
            print(f"  [HISEvent Official M_{bid:02d}] {m3}", flush=True)
        results["hisevent_official"] = _aggregate_method(pb_his, val_blocks)
        _save()
        print(f"[HISEvent Official] Mean = {results['hisevent_official']['mean']}", flush=True)
    elif need_hisevent_k:
        cached_his = (results.get("hisevent_official") or {}).get("per_block", {})
        if all(f"M_{bid:02d}" in cached_his and "n_pred_clusters" in cached_his[f"M_{bid:02d}"] for bid in val_blocks):
            for bid in val_blocks:
                hisevent_k_est[bid] = int(cached_his[f"M_{bid:02d}"]["n_pred_clusters"])
        else:
            pb_his = {}
            for bid in val_blocks:
                lbl, st = run_hisevent_official(
                    ctx["ds"][bid],
                    embeddings=ctx["emb_prep"][bid],
                    n_sub=None,
                    max_num_neighbors=200,
                    dataset_name=dataset_name,
                    use_raw_attributes=True,
                )
                hisevent_k_est[bid] = int(st.get("n_pred_clusters", len(set(lbl))))
                m3 = _eval_3metrics(ctx["ds"][bid].labels, lbl)
                m3.update({"calls": 0, "tokens": 0, "parse_errors": 0, **st})
                pb_his[f"M_{bid:02d}"] = m3
            results["hisevent_official"] = _aggregate_method(pb_his, val_blocks)
            _save()

    # Group 2A: RagSEDE Official Unbounded (Original Protocol, Token Cost Reported)
    if "ragsede_unbounded" in methods:
        client = _make_client(args.provider, "ragsede_unbounded", out_dir, args.model, args.pace_sec)

        def _do_ragsede(bid: int) -> Dict[str, Any]:
            lbl, _shown, st = run_ragsede_official(
                dataset_name,
                bid,
                ctx["ds"][bid],
                client,
                bucket=None,
                embed_model=encs["official_prep"],
                embeddings_prep=ctx["emb_prep"][bid],
                strategy_tag="ragsede_unbounded",
            )
            m3 = _eval_3metrics(ctx["ds"][bid].labels, lbl)
            m3.update({"calls": st["total_calls"], "tokens": st["total_tokens"], "parse_errors": st["parse_errors"], **st})
            print(f"  [RagSEDE Unbounded M_{bid:02d}] {m3['ari']}/{m3['nmi']}/{m3['ami']} calls={st['total_calls']} tok={st['total_tokens']}", flush=True)
            return m3

        results["ragsede_unbounded"] = _aggregate_method(_run_blocks_parallel(_do_ragsede), val_blocks)
        _save()
        print(f"[RagSEDE Unbounded] Mean = {results['ragsede_unbounded']['mean']}", flush=True)

    # Group 2B: LSED Official Unbounded (HGCN 3-layer Poincare MLP, w=infty, Token Cost Reported)
    colab_dir = Path(args.use_colab_outputs) if args.use_colab_outputs else None
    lsed_pre = None
    if colab_dir is not None:
        cand_resp = colab_dir / f"lsed_M{b_lo:02d}_M{b_hi:02d}_responses.jsonl"
        fallback_resp = colab_dir / "lsed_M01_M05_responses.jsonl"
        lsed_pre = _load_colab_responses(cand_resp if cand_resp.exists() else fallback_resp)
    lsed_m0_pre = _load_colab_responses(colab_dir / "lsed_M00_responses.jsonl") if colab_dir else None

    if "lsed_unbounded" in methods:
        if not lsed_pre and not args.allow_online_lsed:
            print(f"[SKIP lsed_unbounded] Requires --use-colab-outputs with lsed_M{b_lo:02d}_M{b_hi:02d}_responses.jsonl (or pass --allow-online-lsed)")
        elif not lsed_m0_pre and not args.allow_unsummarized_lsed_m0:
            print(
                "[SKIP lsed_unbounded] Official LSED requires summarized M_00 (lsed_M00_responses.jsonl) "
                "to avoid train/test distribution shift (or pass --allow-unsummarized-lsed-m0 to override)"
            )
        else:
            ckpt_path = out_dir / "lsed_m00_hgcn_poincare.pt"
            hyp_model, hyp_dev = train_lsed_hyperbolic_encoder(
                ctx["ds"][0],
                ctx["emb_raw_f32"][0],
                ckpt_path,
                precomputed_m0_responses=lsed_m0_pre if (lsed_m0_pre and len(lsed_m0_pre) > 0) else None,
                embed_model=encs["lsed_summary"],
            )
            client = None if (lsed_pre and len(lsed_pre) > 0) else _make_client(args.provider, "lsed_unbounded", out_dir, args.model, args.pace_sec)

            def _do_lsed(bid: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                k_or = len(set(ctx["ds"][bid].labels))
                k_es = hisevent_k_est[bid]
                lbl_o, lbl_e, _shown, st = run_lsed_official(
                    bid,
                    ctx["ds"][bid],
                    ctx["emb_raw_f32"][bid],
                    hyp_model,
                    hyp_dev,
                    k_oracle=k_or,
                    k_est=k_es,
                    embed_model=encs["lsed_summary"],
                    client=client,
                    bucket=None,
                    precomputed_responses=lsed_pre if (lsed_pre and len(lsed_pre) > 0) else None,
                    strategy_tag="lsed_unbounded",
                )
                mo = _eval_3metrics(ctx["ds"][bid].labels, lbl_o)
                mo.update({"calls": st["total_calls"], "tokens": st["total_tokens"], "parse_errors": st["parse_errors"], **st})
                me = _eval_3metrics(ctx["ds"][bid].labels, lbl_e)
                me.update({"calls": st["total_calls"], "tokens": st["total_tokens"], "parse_errors": st["parse_errors"], **st})
                print(
                    f"  [lsed_unbounded M_{bid:02d}] OracleK={mo['ari']}/{mo['nmi']}/{mo['ami']} "
                    f"EstK={me['ari']}/{me['nmi']}/{me['ami']} calls={st['total_calls']} tok={st['total_tokens']} "
                    f"||hyp||={st['mean_hyp_norm']}",
                    flush=True,
                )
                return mo, me

            pb_ok: Dict[str, Any] = {}
            pb_ek: Dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=pool_workers) as ex:
                futs = {ex.submit(_do_lsed, bid): bid for bid in val_blocks}
                for fut, bid in futs.items():
                    mo, me = fut.result()
                    pb_ok[f"M_{bid:02d}"] = mo
                    pb_ek[f"M_{bid:02d}"] = me
            agg_ok = _aggregate_method(pb_ok, val_blocks)
            agg_ek = _aggregate_method(pb_ek, val_blocks)
            if lsed_m0_pre:
                m0_tokens = sum(
                    int(v.get("prompt_tokens", 0)) + int(v.get("completion_tokens", 0))
                    for v in lsed_m0_pre.values()
                )
                m0_cost = {"calls": len(lsed_m0_pre), "tokens": m0_tokens}
                agg_ok["m0_summary_cost"] = m0_cost
                agg_ek["m0_summary_cost"] = m0_cost
            results["lsed_unbounded_oracleK"] = agg_ok
            results["lsed_unbounded_estK"] = agg_ek
            _save()

    # Group 3A: COBRAS 20k Budgeted (Binary LLM oracle)
    if "cobras_20k" in methods:
        client = _make_client(args.provider, "cobras_20k", out_dir, args.model, args.pace_sec)

        def _do_cobras(bid: int) -> Dict[str, Any]:
            bkt = CausalUniformBucket(tokens_per_block=BUDGET_20K, allow_carryover=False)
            bkt.start_block()
            lbl, _shown, st = run_cobras_official(
                bid,
                ctx["ds"][bid],
                ctx["emb_raw_f32"][bid],
                client,
                bucket=bkt,
                max_questions=10000,
                strategy_tag="cobras_20k",
            )
            m3 = _eval_3metrics(ctx["ds"][bid].labels, lbl)
            m3.update({"calls": st["total_calls"], "tokens": st["total_tokens"], "parse_errors": st["parse_errors"], **st})
            print(f"  [COBRAS 20k M_{bid:02d}] {m3['ari']}/{m3['nmi']}/{m3['ami']} calls={st['total_calls']} tok={st['total_tokens']}", flush=True)
            return m3

        results["cobras_20k"] = _aggregate_method(_run_blocks_parallel(_do_cobras), val_blocks)
        _save()
        print(f"[COBRAS 20k] Mean = {results['cobras_20k']['mean']}", flush=True)

    # Group 3B: Cequel 20k Budgeted (WCSC primary + WCKMeans)
    if "cequel_20k" in methods:
        client = _make_client(args.provider, "cequel_20k", out_dir, args.model, args.pace_sec)

        def _do_cequel(bid: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            bkt = CausalUniformBucket(tokens_per_block=BUDGET_20K, allow_carryover=False)
            bkt.start_block()
            k_or = len(set(ctx["ds"][bid].labels))
            k_es = hisevent_k_est[bid]
            lbl_o, lbl_e, _shown, st = run_cequel_official(
                bid,
                ctx["ds"][bid],
                ctx["emb_raw_f32"][bid],
                k_oracle=k_or,
                k_est=k_es,
                client=client,
                bucket=bkt,
                clustering_method="wcsc",
                include_wckmeans=True,
                strategy_tag="cequel_20k",
            )
            lbl_wk_o = st.pop("wckmeans_oracleK")
            _lbl_wk_e = st.pop("wckmeans_estK")
            common = {
                "calls": st["total_calls"],
                "tokens": st["total_tokens"],
                "parse_errors": st["parse_errors"],
                "n_triangles_total": st["n_triangles_total"],
                "n_ml": st["n_ml"],
                "n_cl": st["n_cl"],
                "prompt_tokens": st["prompt_tokens"],
                "completion_tokens": st["completion_tokens"],
                "shown_posts": st["shown_posts"],
                "llm_api_sec": st["llm_api_sec"],
                "total_wall_sec": st["total_wall_sec"],
            }
            mo = {**_eval_3metrics(ctx["ds"][bid].labels, lbl_o), **common}
            me = {**_eval_3metrics(ctx["ds"][bid].labels, lbl_e), **common}
            mwk = {**_eval_3metrics(ctx["ds"][bid].labels, lbl_wk_o), **common}
            print(
                f"  [cequel_20k M_{bid:02d}] WCSC(OracleK)={mo['ari']}/{mo['nmi']}/{mo['ami']} "
                f"WCKMeans(OracleK)={mwk['ari']}/{mwk['nmi']}/{mwk['ami']} "
                f"EstK={me['ari']}/{me['nmi']}/{me['ami']} calls={st['total_calls']} tok={st['total_tokens']}",
                flush=True,
            )
            return mo, me, mwk

        pb_ok = {}
        pb_ek = {}
        pb_wk: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=pool_workers) as ex:
            futs = {ex.submit(_do_cequel, bid): bid for bid in val_blocks}
            for fut, bid in futs.items():
                mo, me, mwk = fut.result()
                pb_ok[f"M_{bid:02d}"] = mo
                pb_ek[f"M_{bid:02d}"] = me
                pb_wk[f"M_{bid:02d}"] = mwk
        results["cequel_20k_wcsc_oracleK"] = _aggregate_method(pb_ok, val_blocks)
        results["cequel_20k_wcsc_estK"] = _aggregate_method(pb_ek, val_blocks)
        results["cequel_20k_wckmeans_oracleK"] = _aggregate_method(pb_wk, val_blocks)
        _save()

    # Group 4A: FRAG-SED 20k Budgeted
    if "frag_sed_20k" in methods:
        client = _make_client(args.provider, "frag_sed_20k", out_dir, args.model, args.pace_sec)

        def _do_frag_20k(bid: int) -> Dict[str, Any]:
            bkt = CausalUniformBucket(tokens_per_block=BUDGET_20K, allow_carryover=False)
            bkt.start_block()
            lbl, _shown, st = run_frag_sed(
                bid, ctx["ds"][bid], ctx["emb"][bid], ctx["init_2dse"][bid], client, bucket=bkt, strategy_tag="frag_sed_20k"
            )
            m3 = _eval_3metrics(ctx["ds"][bid].labels, lbl)
            m3.update({"calls": st["total_calls"], "tokens": st["total_tokens"], "parse_errors": st["parse_errors"], **st})
            print(f"  [FRAG-SED 20k M_{bid:02d}] {m3['ari']}/{m3['nmi']}/{m3['ami']} calls={st['total_calls']} tok={st['total_tokens']}", flush=True)
            return m3

        results["frag_sed_20k"] = _aggregate_method(_run_blocks_parallel(_do_frag_20k), val_blocks)
        _save()
        print(f"[FRAG-SED 20k] Mean = {results['frag_sed_20k']['mean']}", flush=True)

    # Group 4B: FRAG-SED Unbounded
    if "frag_sed_unbounded" in methods:
        client = _make_client(args.provider, "frag_sed_unbounded", out_dir, args.model, args.pace_sec)

        def _do_frag_unb(bid: int) -> Dict[str, Any]:
            lbl, _shown, st = run_frag_sed(
                bid, ctx["ds"][bid], ctx["emb"][bid], ctx["init_2dse"][bid], client, bucket=None, strategy_tag="frag_sed_unbounded"
            )
            m3 = _eval_3metrics(ctx["ds"][bid].labels, lbl)
            m3.update({"calls": st["total_calls"], "tokens": st["total_tokens"], "parse_errors": st["parse_errors"], **st})
            print(f"  [FRAG-SED Unbounded M_{bid:02d}] {m3['ari']}/{m3['nmi']}/{m3['ami']} calls={st['total_calls']} tok={st['total_tokens']}", flush=True)
            return m3

        results["frag_sed_unbounded"] = _aggregate_method(_run_blocks_parallel(_do_frag_unb), val_blocks)
        _save()
        print(f"[FRAG-SED Unbounded] Mean = {results['frag_sed_unbounded']['mean']}", flush=True)

    print(f"\nReport saved to {res_md_path}")


if __name__ == "__main__":
    main()
