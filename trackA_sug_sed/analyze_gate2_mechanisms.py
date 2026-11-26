"""Reproducible 0-API P1 Mechanistic Attribution & Boundary Calibration Audit for Gate 2.

Synthesizes controlled 3-factor attribution (structural star-pack prior vs. priority scheduling
vs. LLM semantic gain), 5-pillar paired Wilcoxon + operational FM/FS balance, EN/FR encoder
cosine calibration at AutoMerge >=0.85, GuardedUF transitive-snowballing insurance, and
Pred/True fragmentation + Cross-Day acceptance boundaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.data_loader import load_socialed_npy, split_daily_blocks

HELDOUT_BLOCKS: Dict[str, List[int]] = {
    "Event2012": list(range(6, 22)),
    "Event2018": list(range(6, 17)),
}
EMBED_MODELS: Dict[str, str] = {
    "Event2012": "sentence-transformers/all-MiniLM-L6-v2",
    "Event2018": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
OUT_DIR = ROOT / "results" / "heldout_gate2"
EMB_CACHE_DIR = ROOT / "cache" / "embeddings"
PART_CACHE_DIR = ROOT / "cache" / "partitions"

METRICS = ("ami", "macro_b3", "micro_b3", "tail_b3")

ABLATION_META: List[Tuple[str, str, str]] = [
    ("abl_no_star_packing", "w/o Star Packing (k=1 Pair)", "Core Universal Engine"),
    ("abl_no_seam_split", "w/o Seam Split", "Core Universal Engine"),
    ("abl_no_automerge", "w/o AutoMerge (>=0.85)", "Efficiency & Risk Guardrail"),
    ("abl_no_bayes_band", "w/o Two-Tier Band", "Efficiency & Risk Guardrail"),
    ("abl_no_guarded_uf", "w/o Guarded UF", "Efficiency & Risk Guardrail"),
]

COSINE_BINS: List[Tuple[float, float, str]] = [
    (0.90, 1.01, "[0.90, 1.00]"),
    (0.85, 0.90, "[0.85, 0.90)"),
    (0.85, 1.01, ">=0.85 (AutoMerge)"),
    (0.75, 0.85, "[0.75, 0.85)"),
    (0.65, 0.75, "[0.65, 0.75)"),
    (0.55, 0.65, "[0.55, 0.65)"),
    (0.45, 0.55, "[0.45, 0.55)"),
]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Gate 2 artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_block_keys(entry: Dict[str, Any]) -> List[str]:
    pb = entry.get("per_block", {})
    return sorted(pb.keys(), key=lambda k: int(k.split("_")[1]))


def _block_vec(entry: Dict[str, Any], field: str) -> np.ndarray:
    pb = entry.get("per_block", {})
    return np.array([float(pb[k][field]) for k in _ordered_block_keys(entry)], dtype=float)


def _sum_stat(entry: Dict[str, Any], stat_key: str) -> int:
    pb = entry.get("per_block", {})
    return int(sum(int(pb[k].get("stats", {}).get(stat_key, 0)) for k in _ordered_block_keys(entry)))


def _holm_bonferroni(p_vals: Sequence[float]) -> List[float]:
    m = len(p_vals)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_vals), key=lambda x: x[1])
    adj = [1.0] * m
    prev_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, float(p) * (m - rank))
        val = max(prev_max, val)
        prev_max = val
        adj[orig_idx] = val
    return adj


def _paired_wilcoxon_stats(vals_a: np.ndarray, vals_b: np.ndarray) -> Dict[str, Any]:
    diffs = np.asarray(vals_a, dtype=float) - np.asarray(vals_b, dtype=float)
    nz = diffs[np.abs(diffs) > 1e-9]
    if len(nz) == 0:
        p_val = 1.0
    else:
        no_ties = len(np.unique(np.round(np.abs(nz), 9))) == len(nz)
        method = "exact" if (len(nz) <= 50 and no_ties) else "auto"
        p_val = float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method=method).pvalue)
    return {
        "mean_diff": round(float(np.mean(diffs)), 2),
        "n_pairs": int(len(diffs)),
        "n_wins": int(np.sum(diffs > 1e-9)),
        "n_ties": int(np.sum(np.abs(diffs) <= 1e-9)),
        "n_losses": int(np.sum(diffs < -1e-9)),
        "p_raw": float(f"{p_val:.6g}"),
    }


def _scope_vectors(
    scope: str,
    entry_en_a: Dict[str, Any],
    entry_en_b: Dict[str, Any],
    entry_fr_a: Dict[str, Any],
    entry_fr_b: Dict[str, Any],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if scope == "Event2012":
        return _block_vec(entry_en_a, metric), _block_vec(entry_en_b, metric)
    if scope == "Event2018":
        return _block_vec(entry_fr_a, metric), _block_vec(entry_fr_b, metric)
    a = np.concatenate([_block_vec(entry_en_a, metric), _block_vec(entry_fr_a, metric)])
    b = np.concatenate([_block_vec(entry_en_b, metric), _block_vec(entry_fr_b, metric)])
    return a, b


def _summarize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    pb = entry.get("per_block", {})
    keys = _ordered_block_keys(entry)
    out: Dict[str, Any] = {
        "ami": round(float(entry.get("ami", np.mean([pb[k]["ami"] for k in keys]))), 2),
        "macro_b3": round(float(entry.get("macro_b3", np.mean([pb[k]["macro_b3"] for k in keys]))), 2),
        "micro_b3": round(float(entry.get("micro_b3", np.mean([pb[k]["micro_b3"] for k in keys]))), 2),
        "tail_b3": round(float(entry.get("tail_b3", np.mean([pb[k]["tail_b3"] for k in keys]))), 2),
        "unq_net_mean": entry.get("unq_net_mean", 0.0),
        "total_tokens": int(round(float(entry.get("total_tokens", 0)))),
        "total_calls": round(float(entry.get("total_calls", 0)), 2),
    }
    if keys and "false_merge_pairs" in pb[keys[0]]:
        fm = int(sum(int(pb[k]["false_merge_pairs"]) for k in keys))
        fs = int(sum(int(pb[k]["false_split_pairs"]) for k in keys))
        out["fm_sum"] = fm
        out["fs_sum"] = fs
        out["fs_fm_ratio"] = round(float(fs) / max(1, fm), 2)
    return out


def compute_factor_attribution(s0: Dict[str, Any], main: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose gains into structural star-pack prior, priority scheduling, and semantic LLM gain."""
    systems_order = [
        ("ctrl_random_mean3", "Random (Post Assign, 3-seed)", "main"),
        ("ctrl_random_meso_mean3", "Random-Meso (Star-Meso Rand-Order, 3-seed)", "main"),
        ("ctrl_meso_heur_pure", "Meso-Heur-Best (0-Tok Cosine Heuristic)", "s0"),
        ("our_pure_star_meso", "Full Pure-Star-Meso (Qwen3-8B)", "main"),
    ]
    systems: Dict[str, Any] = {}
    for key, label, src in systems_order:
        src_obj = main if src == "main" else s0
        en_e, fr_e = src_obj["Event2012"][key], src_obj["Event2018"][key]
        pooled_ami = round(float(np.mean(np.concatenate([_block_vec(en_e, "ami"), _block_vec(fr_e, "ami")]))), 2)
        pooled_mac = round(float(np.mean(np.concatenate([_block_vec(en_e, "macro_b3"), _block_vec(fr_e, "macro_b3")]))), 2)
        pooled_mic = round(float(np.mean(np.concatenate([_block_vec(en_e, "micro_b3"), _block_vec(fr_e, "micro_b3")]))), 2)
        pooled_tail = round(float(np.mean(np.concatenate([_block_vec(en_e, "tail_b3"), _block_vec(fr_e, "tail_b3")]))), 2)
        systems[key] = {
            "label": label,
            "Event2012": _summarize_entry(en_e),
            "Event2018": _summarize_entry(fr_e),
            "Pooled27": {"ami": pooled_ami, "macro_b3": pooled_mac, "micro_b3": pooled_mic, "tail_b3": pooled_tail},
        }

    transitions = [
        ("factor1_structural_prior", "Factor 1: Mesoscopic Star-Pack Architecture (Random -> Random-Meso)", "ctrl_random_meso_mean3", "main", "ctrl_random_mean3", "main", None),
        ("factor2_priority_scheduling", "Factor 2: Calibrated Priority Scheduling (Random-Meso -> Full)", "our_pure_star_meso", "main", "ctrl_random_meso_mean3", "main", "Random-Meso (3-seed)"),
        ("factor3_llm_over_heuristic", "Factor 3: LLM Semantic Gain over 0-Tok Cosine (Meso-Heur-Best -> Full)", "our_pure_star_meso", "main", "ctrl_meso_heur_pure", "s0", "Meso-Heur-Best (0-Tok)"),
        ("factor_total_over_random", "Total LLM Routing Gain over Naive Random (Random -> Full)", "our_pure_star_meso", "main", "ctrl_random_mean3", "main", None),
    ]
    wh = main.get("wilcoxon_holm", {})
    deltas: Dict[str, Any] = {}
    for tid, title, ka, sa, kb, sb, wh_key in transitions:
        obj_a = main if sa == "main" else s0
        obj_b = main if sb == "main" else s0
        by_scope: Dict[str, Any] = {}
        for scope in ("Event2012", "Event2018", "Pooled27"):
            m_out: Dict[str, Any] = {}
            for mk in METRICS:
                va, vb = _scope_vectors(scope, obj_a["Event2012"][ka], obj_b["Event2012"][kb], obj_a["Event2018"][ka], obj_b["Event2018"][kb], mk)
                st = _paired_wilcoxon_stats(va, vb)
                if wh_key and scope in wh and wh_key in wh[scope] and mk in wh[scope][wh_key]:
                    st["p_holm"] = wh[scope][wh_key][mk]["p_holm"]
                m_out[mk] = st
            by_scope[scope] = m_out
        deltas[tid] = {"title": title, "scopes": by_scope}

    return {
        "code_isolation": {
            "source": "trackA_sug_sed/star_meso.py (detect_clean_seams & run_pure_star_meso)",
            "shared_in_random_meso": [
                "Pure 2D-SE initialization (knn_k=15, tau=0.50, alpha=0.5)",
                "AutoMerge (>=0.85 cosine) + GuardedUnionFind (guard_sim=0.40)",
                "Clean Seam Candidate Pool (detect_clean_seams, only order shuffled)",
                "Star Packing (k_nb=4) + identical Star-Merge & Seam-Split prompts",
            ],
            "ablated_in_random_meso": [
                "Two-tier Bayes utility anchor ordering -> uniform random step selection (tier [0.45, 0.85))",
                "Seam score priority ordering -> random permutation",
            ],
        },
        "systems": systems,
        "transitions": deltas,
    }


def compute_five_pillar_ablation_audit(main: Dict[str, Any], abl: Dict[str, Any]) -> Dict[str, Any]:
    """Compute paired Wilcoxon stats, operational counts, and FM/FS structural balance for 5 pillars."""
    full_en = main["Event2012"]["our_pure_star_meso"]
    full_fr = main["Event2018"]["our_pure_star_meso"]
    stat_keys = ("total_calls", "auto_merge_ops", "seam_calls", "seam_splits", "star_calls", "star_edges_checked", "star_merges", "blocked_by_guard")

    def _ops_and_balance(en_e: Dict[str, Any], fr_e: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for ds, e in (("Event2012", en_e), ("Event2018", fr_e)):
            fm = int(_block_vec(e, "false_merge_pairs").sum())
            fs = int(_block_vec(e, "false_split_pairs").sum())
            out[ds] = {
                **{sk: _sum_stat(e, sk) for sk in stat_keys},
                "fm_sum": fm,
                "fs_sum": fs,
                "fs_fm_ratio": round(float(fs) / max(1, fm), 2),
                "ami": float(e["ami"]),
                "macro_b3": float(e["macro_b3"]),
                "micro_b3": float(e["micro_b3"]),
                "tail_b3": float(e["tail_b3"]),
                "unq_net_mean": float(e.get("unq_net_mean", 0.0)),
            }
        return out

    variants: Dict[str, Any] = {}
    for ak, label, role in ABLATION_META:
        ab_en = abl["Event2012"][ak]
        ab_fr = abl["Event2018"][ak]
        wilcox: Dict[str, Any] = {}
        for scope in ("Event2012", "Event2018", "Pooled27"):
            wilcox[scope] = {}
            for mk in METRICS:
                va, vb = _scope_vectors(scope, full_en, ab_en, full_fr, ab_fr, mk)
                wilcox[scope][mk] = _paired_wilcoxon_stats(va, vb)
        variants[ak] = {
            "label": label,
            "role": role,
            "full_minus_ablated_wilcoxon": wilcox,
            "ablated_ops": _ops_and_balance(ab_en, ab_fr),
        }

    # Holm-Bonferroni correction across the 5 pillar variants per (scope, metric)
    ab_keys = [ak for ak, _, _ in ABLATION_META]
    for scope in ("Event2012", "Event2018", "Pooled27"):
        for mk in METRICS:
            p_raws = [variants[ak]["full_minus_ablated_wilcoxon"][scope][mk]["p_raw"] for ak in ab_keys]
            p_holms = _holm_bonferroni(p_raws)
            for ak, ph in zip(ab_keys, p_holms):
                variants[ak]["full_minus_ablated_wilcoxon"][scope][mk]["p_holm"] = float(f"{ph:.6g}")

    return {"full_ops": _ops_and_balance(full_en, full_fr), "variants": variants}


def compute_encoder_cosine_calibration(abl: Dict[str, Any], main: Dict[str, Any]) -> Dict[str, Any]:
    """Audit EN vs FR frozen encoder inter-cluster cosine calibration and AutoMerge outliers."""
    calibration: Dict[str, Any] = {}
    for ds_name, block_ids in HELDOUT_BLOCKS.items():
        model_name = EMBED_MODELS[ds_name]
        model_slug = model_name.replace("/", "__")
        ds_full = load_socialed_npy(ds_name, sort_chronologically=True)
        daily_blocks = split_daily_blocks(ds_full, merge_initial_days=7, min_block_size=200)

        bin_counts: Dict[str, List[int]] = {lbl: [0, 0] for _, _, lbl in COSINE_BINS}
        am_same_posts = am_diff_posts = 0
        diff_sims_all: List[float] = []
        same_sims_all: List[float] = []

        for bid in block_ids:
            _, idxs = daily_blocks[bid]
            sub = ds_full.subset(idxs, name_suffix=f"_M{bid:02d}")
            from trackA_sug_sed.embed_cache import load_or_extract_block_embeddings

            emb = load_or_extract_block_embeddings(
                ds_name, bid, sub.texts, model_name=model_name, cache_dir=EMB_CACHE_DIR
            )
            if emb.shape[0] != len(sub):
                raise ValueError(f"Embedding cache shape mismatch for {ds_name} M_{bid:02d}: {emb.shape[0]} vs {len(sub)}")

            part_prefix = f"{ds_name}_M{bid:02d}_k15_tau0.500_a0.500_ref1"
            enc_part_matches = sorted(PART_CACHE_DIR.glob(f"{part_prefix}_enc_{model_slug}_g*.npz"))
            legacy_part = PART_CACHE_DIR / f"{part_prefix}.npz"
            if enc_part_matches:
                part_path = enc_part_matches[-1]
            elif legacy_part.exists():
                part_path = legacy_part
            else:
                any_matches = sorted(PART_CACHE_DIR.glob(f"{part_prefix}*.npz"))
                if not any_matches:
                    raise FileNotFoundError(f"Missing Pure 2D-SE partition cache for {part_prefix} in {PART_CACHE_DIR}")
                part_path = any_matches[-1]
            with np.load(part_path, allow_pickle=False) as z:
                labels = np.asarray(z["labels"], dtype=np.int64)
            if len(labels) != len(sub):
                raise ValueError(f"Partition cache length mismatch for {part_path.name}: {len(labels)} vs {len(sub)}")

            cids = np.unique(labels)
            protos, majs, sizes = [], [], []
            y = np.asarray(sub.labels)
            for c in cids:
                ii = np.where(labels == c)[0]
                if len(ii) < 2:
                    continue
                vec = emb[ii].mean(axis=0)
                vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
                protos.append(vec)
                vals, cnts = np.unique(y[ii], return_counts=True)
                majs.append(int(vals[int(np.argmax(cnts))]))
                sizes.append(int(len(ii)))

            mat = np.vstack(protos)
            sim_mat = mat @ mat.T
            iu, ju = np.triu_indices(len(protos), k=1)
            sims = sim_mat[iu, ju]
            same = np.array([majs[i] == majs[j] for i, j in zip(iu, ju)], dtype=bool)
            diff_sims_all.extend(sims[~same].tolist())
            same_sims_all.extend(sims[same].tolist())

            for lo, hi, lbl in COSINE_BINS:
                mask = (sims >= lo) & (sims < hi)
                bin_counts[lbl][0] += int(np.sum(same[mask]))
                bin_counts[lbl][1] += int(np.sum(mask))

            for i, j in zip(iu[sims >= 0.85], ju[sims >= 0.85]):
                prod = sizes[i] * sizes[j]
                if majs[i] == majs[j]:
                    am_same_posts += prod
                else:
                    am_diff_posts += prod

        diff_arr = np.asarray(diff_sims_all, dtype=float)
        same_arr = np.asarray(same_sims_all, dtype=float)

        # Block-level AutoMerge divergence
        pb_f = main[ds_name]["our_pure_star_meso"]["per_block"]
        pb_a = abl[ds_name]["abl_no_automerge"]["per_block"]
        outliers: List[Dict[str, Any]] = []
        tot_dfm = tot_dfs = 0
        for bk in _ordered_block_keys(main[ds_name]["our_pure_star_meso"]):
            dfm = int(pb_f[bk]["false_merge_pairs"]) - int(pb_a[bk]["false_merge_pairs"])
            dfs = int(pb_f[bk]["false_split_pairs"]) - int(pb_a[bk]["false_split_pairs"])
            tot_dfm += dfm
            tot_dfs += dfs
            if abs(pb_f[bk]["ami"] - pb_a[bk]["ami"]) > 1e-4 or abs(dfm) > 10000:
                outliers.append({
                    "block": bk,
                    "auto_ops": int(pb_f[bk]["stats"].get("auto_merge_ops", 0)),
                    "d_ami": round(float(pb_f[bk]["ami"]) - float(pb_a[bk]["ami"]), 2),
                    "d_mac": round(float(pb_f[bk]["macro_b3"]) - float(pb_a[bk]["macro_b3"]), 2),
                    "full_minus_noam_fm": dfm,
                    "full_minus_noam_fs": dfs,
                })

        calibration[ds_name] = {
            "embed_model": model_name,
            "cluster_filter": "size >= 2 (matching AutoMerge domain)",
            "diff_event_pair_cosine": {
                "mean": round(float(diff_arr.mean()), 4),
                "p50": round(float(np.percentile(diff_arr, 50)), 4),
                "p90": round(float(np.percentile(diff_arr, 90)), 4),
                "p95": round(float(np.percentile(diff_arr, 95)), 4),
                "p99": round(float(np.percentile(diff_arr, 99)), 4),
            },
            "same_event_pair_cosine": {
                "mean": round(float(same_arr.mean()), 4),
                "p50": round(float(np.percentile(same_arr, 50)), 4),
                "p90": round(float(np.percentile(same_arr, 90)), 4),
            },
            "automerge_size_weighted_precision_pct": round(100.0 * am_same_posts / max(1, am_same_posts + am_diff_posts), 2),
            "automerge_same_post_pairs": int(am_same_posts),
            "automerge_diff_post_pairs": int(am_diff_posts),
            "bins": [
                {
                    "bin": lbl,
                    "same_event": bin_counts[lbl][0],
                    "total_pairs": bin_counts[lbl][1],
                    "precision_pct": round(100.0 * bin_counts[lbl][0] / max(1, bin_counts[lbl][1]), 1),
                }
                for _, _, lbl in COSINE_BINS
            ],
            "automerge_ablation_delta_fm": int(tot_dfm),
            "automerge_ablation_delta_fs": int(tot_dfs),
            "automerge_divergent_blocks": outliers,
        }
    return calibration


def compute_guarded_uf_audit(main: Dict[str, Any], abl: Dict[str, Any]) -> Dict[str, Any]:
    """Audit GuardedUnionFind transitive-snowballing prevention across all 27 blocks."""
    out: Dict[str, Any] = {}
    for ds_name in ("Event2012", "Event2018"):
        pb_f = main[ds_name]["our_pure_star_meso"]["per_block"]
        pb_g = abl[ds_name]["abl_no_guarded_uf"]["per_block"]
        full_fm = int(_block_vec(main[ds_name]["our_pure_star_meso"], "false_merge_pairs").sum())
        nog_fm = int(_block_vec(abl[ds_name]["abl_no_guarded_uf"], "false_merge_pairs").sum())
        extra_fm = nog_fm - full_fm
        blocks_div: List[Dict[str, Any]] = []
        for bk in _ordered_block_keys(main[ds_name]["our_pure_star_meso"]):
            dfm = int(pb_g[bk]["false_merge_pairs"]) - int(pb_f[bk]["false_merge_pairs"])
            dfs = int(pb_g[bk]["false_split_pairs"]) - int(pb_f[bk]["false_split_pairs"])
            d_ami = round(float(pb_f[bk]["ami"]) - float(pb_g[bk]["ami"]), 2)
            bg = int(pb_f[bk]["stats"].get("blocked_by_guard", 0))
            if abs(d_ami) > 1e-4 or bg > 0 or dfm != 0:
                blocks_div.append({
                    "block": bk,
                    "blocked_by_guard_post_llm": bg,
                    "full_minus_noguard_ami": d_ami,
                    "full_minus_noguard_mac": round(float(pb_f[bk]["macro_b3"]) - float(pb_g[bk]["macro_b3"]), 2),
                    "noguard_extra_fm": dfm,
                    "noguard_extra_fs": dfs,
                })
        out[ds_name] = {
            "full_fm_sum": full_fm,
            "noguard_fm_sum": nog_fm,
            "prevented_fm_pairs": extra_fm,
            "prevented_fm_pct_of_full": round(100.0 * extra_fm / max(1, full_fm), 1),
            "post_llm_guard_blocks": _sum_stat(main[ds_name]["our_pure_star_meso"], "blocked_by_guard"),
            "divergent_blocks": blocks_div,
            "note": "GuardedUF also vetoes pairs pre-query in uf.can_merge() before counting post-LLM blocked_by_guard.",
        }
    return out


def compute_fragmentation_and_crossday_boundary(
    s0: Dict[str, Any],
    main: Dict[str, Any],
    cd: Dict[str, Any],
) -> Dict[str, Any]:
    """Quantify Pred/True cluster over-fragmentation vs Tail-B3 and Cross-Day acceptance."""
    frag: Dict[str, Any] = {}
    targets = [
        ("clusterllm_emnlp23", "ClusterLLM-NoFT (EMNLP'23)", main),
        ("sbert_emnlp19", "SBERT-Agglom (EMNLP'19)", s0),
        ("ctrl_pure_2dse_0tok", "Pure 2D-SE Base (0.50)", s0),
        ("ctrl_meso_heur_pure", "Meso-Heur-Best (Pure)", s0),
        ("ctrl_random_meso_s42", "Random-Meso (seed=42)", main),
        ("our_pure_star_meso", "Pure-Star-Meso (Qwen3-8B)", main),
    ]
    for ds_name in ("Event2012", "Event2018"):
        rows: List[Dict[str, Any]] = []
        for k, label, src in targets:
            e = src[ds_name][k]
            np_ = _block_vec(e, "n_pred_clusters")
            nt_ = _block_vec(e, "n_true_events")
            rows.append({
                "key": k,
                "label": label,
                "pred_clusters_sum": int(np_.sum()),
                "true_events_sum": int(nt_.sum()),
                "pred_over_true_ratio": round(float(np.mean(np_ / np.maximum(nt_, 1.0))), 2),
                "ami": float(e["ami"]),
                "micro_b3": float(e["micro_b3"]),
                "tail_b3": float(e["tail_b3"]),
            })
        frag[ds_name] = rows

    crossday: Dict[str, Any] = {}
    for ds_name in ("Event2012", "Event2018"):
        sug = cd[ds_name]["sug_cross_star"]
        orc = cd[ds_name]["oracle_band"]
        p95 = cd[ds_name]["proto_attr_0.95"]
        sq, sl = int(sug["stats"]["llm_queried"]), int(sug["stats"]["llm_linked"])
        oq, ol = int(orc["stats"]["llm_queried"]), int(orc["stats"]["llm_linked"])
        crossday[ds_name] = {
            "sug_cross_star_g_ami": float(sug["g_ami"]),
            "sug_cross_star_g_mac": float(sug["g_mac_b3"]),
            "proto_attr_0.95_g_ami": float(p95["g_ami"]),
            "oracle_band_g_ami": float(orc["g_ami"]),
            "sug_llm_linked": sl,
            "sug_llm_queried": sq,
            "sug_accept_rate_pct": round(100.0 * sl / max(1, sq), 1),
            "oracle_llm_linked": ol,
            "oracle_llm_queried": oq,
            "oracle_accept_rate_pct": round(100.0 * ol / max(1, oq), 1),
            "gap_to_oracle_ami": round(float(orc["g_ami"]) - float(sug["g_ami"]), 2),
        }

    return {"fragmentation_tail_b3": frag, "cross_day_acceptance": crossday}


def build_p1_report_markdown(payload: Dict[str, Any]) -> str:
    """Format P1 mechanistic diagnostics into a concise, citation-ready Markdown report."""
    lines: List[str] = [
        "# Track A Gate 2 P1 Mechanistic Attribution & Boundary Calibration Report\n",
        "> Zero LLM API calls · 100% Dev-Locked · Reproducible via `python trackA_sug_sed/analyze_gate2_mechanisms.py`\n",
    ]

    fa = payload["factor_attribution"]
    lines.append("## 1. Controlled 3-Factor Attribution (`Random` → `Random-Meso` → `Meso-Heur-Best` → `Full Pure-Star-Meso`)\n")
    lines.append(
        "- `ctrl_random_meso_*` runs `run_pure_star_meso(..., random_order=True)`: it inherits `Pure 2D-SE + AutoMerge (>=0.85) + GuardedUF (0.40) + Clean Seam Candidate Pool + Top-k Star Packing + Identical Prompts`, ablating **only** calibrated two-tier Bayes utility & seam score ordering.\n"
    )
    lines.append("| Decomposition Step | Scope | ΔAMI | ΔMac-$B^3$ | ΔMic-$B^3$ | ΔTail-$B^3$ | `p_raw / p_Holm` (`AMI`, `Mac-B³`, `Mic-B³`) |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for tid in ("factor1_structural_prior", "factor2_priority_scheduling", "factor3_llm_over_heuristic", "factor_total_over_random"):
        t = fa["transitions"][tid]
        for scope in ("Event2012", "Event2018", "Pooled27"):
            sc = t["scopes"][scope]
            p_str = ", ".join(
                f"{m.split('_')[0]}:{sc[m]['p_raw']}" + (f"/{sc[m]['p_holm']}" if "p_holm" in sc[m] else "")
                for m in ("ami", "macro_b3", "micro_b3")
            )
            lines.append(
                f"| {t['title']} | {scope} | {sc['ami']['mean_diff']:+.2f} | {sc['macro_b3']['mean_diff']:+.2f} | "
                f"{sc['micro_b3']['mean_diff']:+.2f} | {sc['tail_b3']['mean_diff']:+.2f} | {p_str} |"
            )
    lines.append("")

    pa = payload["five_pillar_ablation_audit"]
    lines.append("## 2. 5-Pillar Ablation Paired Wilcoxon (`Full - Ablated`) & Structural Balance\n")
    lines.append("| Pillar Variant | Role | Scope | ΔAMI (W/T/L, `p_raw / p_Holm`) | ΔMac-$B^3$ (W/T/L, `p_raw / p_Holm`) | ΔMic-$B^3$ (W/T/L, `p_raw / p_Holm`) | ΔTail-$B^3$ (W/T/L, `p_raw / p_Holm`) |")
    lines.append("|---|---|---|---|---|---|---|")
    for ak, lbl, _ in ABLATION_META:
        v = pa["variants"][ak]
        for scope in ("Event2012", "Event2018", "Pooled27"):
            w = v["full_minus_ablated_wilcoxon"][scope]
            cells = [
                f"{w[m]['mean_diff']:+.2f} ({w[m]['n_wins']}/{w[m]['n_ties']}/{w[m]['n_losses']}, {w[m]['p_raw']} / {w[m]['p_holm']})"
                for m in METRICS
            ]
            lines.append(f"| {lbl} | {v['role']} | {scope} | " + " | ".join(cells) + " |")
    lines.append("")

    ec = payload["encoder_cosine_calibration"]
    lines.append("## 3. EN vs FR Encoder Cosine Calibration & `AutoMerge (>=0.85)` Anisotropy\n")
    lines.append("| Dataset (`Embed Model`) | Diff-Event Cos `mean / p95 / p99` | Same-Event Cos `mean / p50` | `>=0.85` Pair Prec% (Size-Wtd%) | `[0.85,0.90)` Prec% | `[0.90,1.00]` Prec% | Ablation `Full-NoAM` ΔFM / ΔFS |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for ds_name in ("Event2012", "Event2018"):
        c = ec[ds_name]
        bmap = {b["bin"]: b for b in c["bins"]}
        dc, sc_ = c["diff_event_pair_cosine"], c["same_event_pair_cosine"]
        lines.append(
            f"| {ds_name} (`{c['embed_model']}`) | {dc['mean']:.3f} / {dc['p95']:.3f} / {dc['p99']:.3f} | "
            f"{sc_['mean']:.3f} / {sc_['p50']:.3f} | "
            f"{bmap['>=0.85 (AutoMerge)']['precision_pct']:.1f}% ({c['automerge_size_weighted_precision_pct']:.1f}%) | "
            f"{bmap['[0.85, 0.90)']['precision_pct']:.1f}% | {bmap['[0.90, 1.00]']['precision_pct']:.1f}% | "
            f"{c['automerge_ablation_delta_fm']:+,d} / {c['automerge_ablation_delta_fs']:+,d} |"
        )
    lines.append("")

    gu = payload["guarded_uf_audit"]
    lines.append("## 4. `GuardedUnionFind` Transitive-Snowballing Risk Insurance\n")
    for ds_name in ("Event2012", "Event2018"):
        g = gu[ds_name]
        lines.append(
            f"- **{ds_name}**: Prevents `{g['prevented_fm_pairs']:+,d}` false-merge pairs (`+{g['prevented_fm_pct_of_full']:.1f}%` vs Full; "
            f"`post_llm_guard_blocks={g['post_llm_guard_blocks']}` plus pre-query `can_merge()` filtering)."
        )
    lines.append("")

    fb = payload["fragmentation_and_crossday_boundary"]
    lines.append("## 5. `Pred/True` Granularity & Cross-Day Link Acceptance Boundary\n")
    for ds_name in ("Event2012", "Event2018"):
        cd_s = fb["cross_day_acceptance"][ds_name]
        lines.append(
            f"- **Cross-Day {ds_name}**: `sug_cross_star` G-AMI `{cd_s['sug_cross_star_g_ami']:.2f}` vs `proto_attr_0.95` `{cd_s['proto_attr_0.95_g_ami']:.2f}` "
            f"and `oracle_band` `{cd_s['oracle_band_g_ami']:.2f}` (gap `-{cd_s['gap_to_oracle_ami']:.2f}`); "
            f"LLM link accept rate `{cd_s['sug_llm_linked']}/{cd_s['sug_llm_queried']} = {cd_s['sug_accept_rate_pct']:.1f}%` vs Oracle `{cd_s['oracle_llm_linked']}/{cd_s['oracle_llm_queried']} = {cd_s['oracle_accept_rate_pct']:.1f}%`."
        )
    lines.append("")
    return "\n".join(lines)


def run_p1_analysis() -> Dict[str, Any]:
    """Execute full P1 mechanistic analysis and persist JSON + Markdown outputs."""
    s0 = _load_json(OUT_DIR / "stage_0tok.json")
    main = _load_json(OUT_DIR / "stage_qwen3_main.json")
    abl = _load_json(OUT_DIR / "stage_qwen3_abl.json")
    cd = _load_json(OUT_DIR / "stage_qwen3_crossday.json")

    payload: Dict[str, Any] = {
        "factor_attribution": compute_factor_attribution(s0, main),
        "five_pillar_ablation_audit": compute_five_pillar_ablation_audit(main, abl),
        "encoder_cosine_calibration": compute_encoder_cosine_calibration(abl, main),
        "guarded_uf_audit": compute_guarded_uf_audit(main, abl),
        "fragmentation_and_crossday_boundary": compute_fragmentation_and_crossday_boundary(s0, main, cd),
    }

    json_path = OUT_DIR / "stage_p1_mechanisms.json"
    md_path = OUT_DIR / "p1_mechanisms_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(build_p1_report_markdown(payload), encoding="utf-8")
    print(f"Saved P1 mechanistic JSON -> {json_path}")
    print(f"Saved P1 mechanistic report -> {md_path}")
    return payload


if __name__ == "__main__":
    run_p1_analysis()
