"""Generate publication-quality 3-panel diagnostic & evaluation figure for Track B Phase 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PHASE2_DIR = ROOT / "results" / "trackB_phase2"
FIG_DIR = ROOT / "results" / "figures"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 2 result JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_phase2(val_path: Path, hold_path: Path, out_path: Path) -> None:
    val = _load_json(val_path)
    hold = _load_json(hold_path)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.1), dpi=300)
    fig.subplots_adjust(wspace=0.27, top=0.80, bottom=0.19, left=0.048, right=0.985)

    # ------------------------------------------------------------------
    # Panel 1: Stage-1 Unsupervised 2D-SE Ablations (Val vs Holdout Macro-B3 & AMI)
    # ------------------------------------------------------------------
    ax1 = axes[0]
    s1_keys = [
        ("text_only_2dse", "Text-Only\n2D-SE"),
        ("naive_mv_2dse", "Naive\nMV"),
        ("cntd_raw_mv_2dse", "CNTD-\nRaw"),
        ("visual_idf_no_impute_mv_2dse", "Vis-IDF\nNo-Impute"),
        ("visual_idf_mv_2dse", "Vis-IDF\nMV-2DSE"),
    ]
    x1 = np.arange(len(s1_keys))
    w1 = 0.20

    vs1 = val["conditions"]["clean"]["stage1"]
    hs1 = hold["conditions"]["clean"]["stage1"]

    val_b3 = [vs1[k]["macro_b3"] * 100.0 for k, _ in s1_keys]
    hold_b3 = [hs1[k]["macro_b3"] * 100.0 for k, _ in s1_keys]
    val_ami = [vs1[k]["ami"] * 100.0 for k, _ in s1_keys]
    hold_ami = [hs1[k]["ami"] * 100.0 for k, _ in s1_keys]

    b1 = ax1.bar(x1 - 1.5 * w1, val_b3, width=w1, color="#2563eb", label="Val Macro-B3", zorder=3)
    b2 = ax1.bar(x1 - 0.5 * w1, hold_b3, width=w1, color="#1d4ed8", alpha=0.55, hatch="//", edgecolor="#1e3a8a", label="Holdout Macro-B3", zorder=3)
    b3 = ax1.bar(x1 + 0.5 * w1, val_ami, width=w1, color="#f59e0b", label="Val AMI", zorder=3)
    b4 = ax1.bar(x1 + 1.5 * w1, hold_ami, width=w1, color="#d97706", alpha=0.55, hatch="//", edgecolor="#78350f", label="Holdout AMI", zorder=3)

    all_p1 = val_b3 + hold_b3 + val_ami + hold_ami
    y1_lo = max(0.0, min(all_p1) - 8.0)
    y1_hi = min(105.0, max(all_p1) + 12.0)
    ax1.set_ylim(y1_lo, y1_hi)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for _, lbl in s1_keys], fontsize=8.5)
    ax1.set_ylabel("Score (x100)")
    ax1.set_title("Stage 1: MV-2DSE & §3.3 Soft-Imputation", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax1.legend(loc="upper left", ncol=2, frameon=False, fontsize=7.6, columnspacing=0.8)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.6, f"{h:.1f}", ha="center", va="bottom", fontsize=6.4, rotation=90)

    # ------------------------------------------------------------------
    # Panel 2: Stage-2 End-to-End Progression & Budget Curve on Holdout
    # ------------------------------------------------------------------
    ax2 = axes[1]
    hs1_vis_b3 = hs1["visual_idf_mv_2dse"]["macro_b3"] * 100.0
    hs1_txt_b3 = hs1["text_only_2dse"]["macro_b3"] * 100.0
    hs1_vis_mi = hs1["visual_idf_mv_2dse"]["micro_b3"] * 100.0
    curves = hold["conditions"]["clean"]["stage2_prefix_curves"]

    series_spec = [
        ("text_only_vlm", "+Text-Only VLM", "#64748b", "--", "s"),
        ("naive_vlm", "+Naive VLM", "#dc2626", "-.", "^"),
        ("evid_self_gate_inline", "+Evid-Gate (Inline)", "#16a34a", "-", "o"),
        ("evid_self_gate_clean_fallback", "+Evid-Gate (CleanFB)", "#2563eb", "-", "D"),
    ]
    all_p2: List[float] = [hs1_vis_b3, hs1_vis_mi]
    max_k = 180
    # Upper tracks: Micro-B3 (solid progress +3.5 to +4.2 pp); Lower tracks: Macro-B3
    for pol, lbl, col, ls, mk in series_spec:
        if pol not in curves:
            continue
        pts = curves[pol]
        ks = [0] + [int(p["k"]) for p in pts]
        mi_vals = [hs1_vis_mi] + [float(p["micro_b3"]) * 100.0 for p in pts]
        ma_vals = [hs1_vis_b3] + [float(p["macro_b3"]) * 100.0 for p in pts]
        all_p2.extend(mi_vals + ma_vals)
        max_k = max(max_k, ks[-1])
        ax2.plot(
            ks,
            mi_vals,
            label=f"{lbl} Micro ({mi_vals[-1]:.1f}) / Macro ({ma_vals[-1]:.1f})",
            color=col,
            linestyle=ls,
            marker=mk,
            linewidth=2.0,
            markersize=5.2,
            zorder=4,
        )
        ax2.plot(
            ks,
            ma_vals,
            color=col,
            linestyle=ls,
            marker=mk,
            linewidth=1.5,
            markersize=4.2,
            alpha=0.65,
            zorder=3,
        )

    ax2.axhline(hs1_vis_mi, color="#475569", linestyle=":", linewidth=1.1, alpha=0.75)
    ax2.axhline(hs1_vis_b3, color="#475569", linestyle=":", linewidth=1.1, alpha=0.75)
    ax2.text(4, hs1_vis_mi - 0.85, f"Stage-1 Micro-B³ ({hs1_vis_mi:.1f}) | Text-Only: 38.3", fontsize=7.6, color="#334155")
    ax2.text(4, hs1_vis_b3 + 0.55, f"Stage-1 Macro-B³ ({hs1_vis_b3:.1f}) | Text-Only: {hs1_txt_b3:.1f}", fontsize=7.6, color="#334155")

    y2_lo = min(all_p2) - 4.8
    y2_hi = max(all_p2) + 2.2
    ax2.set_ylim(y2_lo, y2_hi)
    ax2.set_xlim(-6, max_k + 12)
    ax2.set_xlabel("Stage-2 Seam Queries (K)")
    ax2.set_ylabel("Holdout B³ Score (x100)")
    ax2.set_title(f"Stage 2: NED Holdout ({hold['n_events']} ev, {hold['n_posts']:,} posts)", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#e2e8f0", fontsize=7.1)

    # ------------------------------------------------------------------
    # Panel 3: Clean vs. Stress-30% Competing-Image Robustness on Validation
    # ------------------------------------------------------------------
    ax3 = axes[2]
    vc_s1 = val["conditions"]["clean"]["stage1"]
    vc_s2 = val["conditions"]["clean"]["stage2_policies"]
    vs_s1 = val["conditions"]["stress30"]["stage1"]
    vs_s2 = val["conditions"]["stress30"]["stage2_policies"]

    rob_items = [
        ("Naive\nMV", vc_s1["naive_mv_2dse"]["macro_b3"] * 100.0, vs_s1["naive_mv_2dse"]["macro_b3"] * 100.0),
        ("Vis-IDF\n2D-SE", vc_s1["visual_idf_mv_2dse"]["macro_b3"] * 100.0, vs_s1["visual_idf_mv_2dse"]["macro_b3"] * 100.0),
        ("Text\nVLM", vc_s2["text_only_vlm"]["macro_b3"] * 100.0, vs_s2["text_only_vlm"]["macro_b3"] * 100.0),
        ("Naive\nVLM", vc_s2["naive_vlm"]["macro_b3"] * 100.0, vs_s2["naive_vlm"]["macro_b3"] * 100.0),
        ("Gate\nInline", vc_s2["evid_self_gate_inline"]["macro_b3"] * 100.0, vs_s2["evid_self_gate_inline"]["macro_b3"] * 100.0),
        ("Gate\nCleanFB", vc_s2["evid_self_gate_clean_fallback"]["macro_b3"] * 100.0, vs_s2["evid_self_gate_clean_fallback"]["macro_b3"] * 100.0),
    ]
    x3 = np.arange(len(rob_items))
    w3 = 0.34
    clean_vals = [r[1] for r in rob_items]
    stress_vals = [r[2] for r in rob_items]

    c_bars = ax3.bar(x3 - w3 / 2, clean_vals, width=w3, color="#2563eb", label="Clean Images", zorder=3)
    s_bars = ax3.bar(x3 + w3 / 2, stress_vals, width=w3, color="#ef4444", label="Stress-30% Competing Images", zorder=3)

    all_p3 = clean_vals + stress_vals
    y3_lo = max(0.0, min(all_p3) - 6.0)
    y3_hi = min(105.0, max(all_p3) + 8.5)
    ax3.set_ylim(y3_lo, y3_hi)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([r[0] for r in rob_items], fontsize=8.4)
    ax3.set_ylabel("Event Macro-B³ (x100)")
    ax3.set_title("Validation Robustness: Clean vs. Stress-30%", pad=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=8.0)

    for bars in [c_bars, s_bars]:
        for bar in bars:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, h + 0.35, f"{h:.1f}", ha="center", va="bottom", fontsize=7.0)

    fig.suptitle(
        "Track B (Evid-MSED) Phase 2 on NED: Unsupervised Visual-IDF 2D-SE & Qwen3-VL-8B Conflict Gating",
        fontsize=12.5,
        fontweight="bold",
        y=0.955,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] Phase 2 3-panel figure saved to {out_path}")


def plot_holdout_diag(hold_path: Path, diag_path: Path, out_path: Path) -> None:
    """Plot 3-panel Holdout Stress-30% end-to-end, 24-event strata B3, & action confusion precision."""
    hold = _load_json(hold_path)
    diag = _load_json(diag_path)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.1), dpi=300)
    fig.subplots_adjust(wspace=0.27, top=0.80, bottom=0.19, left=0.048, right=0.985)

    # ------------------------------------------------------------------
    # Panel 1: Holdout (24 ev, 11,955 posts) Clean vs Stress-30% @ K=60
    # ------------------------------------------------------------------
    ax1 = axes[0]
    hc_s1 = hold["conditions"]["clean"]["stage1"]
    hc_curves = hold["conditions"]["clean"]["stage2_prefix_curves"]
    hc_k60 = {pol: pts[0] for pol, pts in hc_curves.items()}  # k=60 is index 0
    hs_all = diag["holdout_stress30_end_to_end"]
    hs_s1 = hs_all["stage1"]
    hs_s2 = hs_all["stage2_policies_k60"]

    items_p1 = [
        ("Naive\nMV", hc_s1["naive_mv_2dse"]["macro_b3"] * 100.0, hs_s1["naive_mv_2dse"]["macro_b3"] * 100.0),
        ("Vis-IDF\n2D-SE", hc_s1["visual_idf_mv_2dse"]["macro_b3"] * 100.0, hs_s1["visual_idf_mv_2dse"]["macro_b3"] * 100.0),
        ("Text\nVLM", hc_k60["text_only_vlm"]["macro_b3"] * 100.0, hs_s2["text_only_vlm"]["macro_b3"] * 100.0),
        ("Naive\nVLM", hc_k60["naive_vlm"]["macro_b3"] * 100.0, hs_s2["naive_vlm"]["macro_b3"] * 100.0),
        ("Gate\nInline", hc_k60["evid_self_gate_inline"]["macro_b3"] * 100.0, hs_s2["evid_self_gate_inline"]["macro_b3"] * 100.0),
        ("Gate\nCleanFB", hc_k60["evid_self_gate_clean_fallback"]["macro_b3"] * 100.0, hs_s2["evid_self_gate_clean_fallback"]["macro_b3"] * 100.0),
    ]
    x1 = np.arange(len(items_p1))
    w1 = 0.34
    c1 = [r[1] for r in items_p1]
    s1 = [r[2] for r in items_p1]

    b1 = ax1.bar(x1 - w1 / 2, c1, width=w1, color="#2563eb", label="Clean Images", zorder=3)
    b2 = ax1.bar(x1 + w1 / 2, s1, width=w1, color="#ef4444", label="Stress-30% Competing Images", zorder=3)
    ax1.axhline(hc_s1["text_only_2dse"]["macro_b3"] * 100.0, color="#64748b", linestyle=":", linewidth=1.2, label="Text-Only 2D-SE (34.85)", zorder=2)

    all_p1 = c1 + s1 + [hc_s1["text_only_2dse"]["macro_b3"] * 100.0]
    ax1.set_ylim(max(0.0, min(all_p1) - 4.0), min(105.0, max(all_p1) + 5.8))
    ax1.set_xticks(x1)
    ax1.set_xticklabels([r[0] for r in items_p1], fontsize=8.3)
    ax1.set_ylabel("Event Macro-B³ (x100)")
    ax1.set_title("Holdout Clean vs. Stress-30% (K=60)", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.90, edgecolor="#e2e8f0", fontsize=7.5)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.25, f"{h:.1f}", ha="center", va="bottom", fontsize=7.0)

    # ------------------------------------------------------------------
    # Panel 2: Holdout Clean 24-Event Stratified B3-F1 Breakdown
    # ------------------------------------------------------------------
    ax2 = axes[1]
    st_clean = diag["stratified_b3"]["clean"]
    methods_p2 = [
        ("text_only_2dse", "Text 2D-SE", "#64748b"),
        ("visual_idf_mv_2dse", "Stage-1 Vis-IDF", "#2563eb"),
        ("k60_naive_vlm", "Naive VLM (K=60)", "#ef4444"),
        ("k60_evid_self_gate_inline", "Gate Inline (K=60)", "#16a34a"),
        ("k60_evid_self_gate_clean_fallback", "Gate CleanFB (K=60)", "#f59e0b"),
    ]
    strata_names = [
        ("small", "Small <200\n(8 events)"),
        ("medium", "Medium 200-599\n(7 events)"),
        ("head", "Head >=600\n(9 events)"),
    ]
    x2 = np.arange(len(strata_names))
    w2 = 0.155
    offsets = np.linspace(-2 * w2, 2 * w2, len(methods_p2))
    all_p2: List[float] = []
    for idx_m, (mkey, mlbl, mcol) in enumerate(methods_p2):
        vals = [st_clean[mkey]["strata"][sk]["macro_f1"] * 100.0 for sk, _ in strata_names]
        all_p2.extend(vals)
        bars = ax2.bar(x2 + offsets[idx_m], vals, width=w2, color=mcol, label=mlbl, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.45, f"{h:.1f}", ha="center", va="bottom", fontsize=6.6, rotation=90)

    ax2.set_ylim(max(0.0, min(all_p2) - 5.5), min(105.0, max(all_p2) + 9.0))
    ax2.set_xticks(x2)
    ax2.set_xticklabels([slbl for _, slbl in strata_names], fontsize=8.8)
    ax2.set_ylabel("Stratum Macro B³-F1 (x100)")
    ax2.set_title("Holdout Clean Stratified Breakdown", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.90, edgecolor="#e2e8f0", fontsize=7.4)

    # ------------------------------------------------------------------
    # Panel 3: Stage-2 Action Precision (Merge & Split @ K=60)
    # ------------------------------------------------------------------
    ax3 = axes[2]
    ac_clean = diag["action_confusion"]["clean_k60"]
    ac_stress = diag["action_confusion"]["stress30_k60"]
    pols_p3 = [
        ("text_only_vlm", "Text\nVLM"),
        ("naive_vlm", "Naive\nVLM"),
        ("evid_self_gate_inline", "Gate\nInline"),
        ("evid_self_gate_clean_fallback", "Gate\nCleanFB"),
    ]
    x3 = np.arange(len(pols_p3))
    w3 = 0.19
    m_c = [ac_clean[p]["merge_prec"] * 100.0 for p, _ in pols_p3]
    m_s = [ac_stress[p]["merge_prec"] * 100.0 for p, _ in pols_p3]
    sp_c = [ac_clean[p]["split_prec"] * 100.0 for p, _ in pols_p3]
    sp_s = [ac_stress[p]["split_prec"] * 100.0 for p, _ in pols_p3]

    bm1 = ax3.bar(x3 - 1.5 * w3, m_c, width=w3, color="#2563eb", label="Merge Prec (Clean)", zorder=3)
    bm2 = ax3.bar(x3 - 0.5 * w3, m_s, width=w3, color="#ef4444", label="Merge Prec (Stress)", zorder=3)
    bs1 = ax3.bar(x3 + 0.5 * w3, sp_c, width=w3, color="#16a34a", alpha=0.80, label="Split Prec (Clean)", zorder=3)
    bs2 = ax3.bar(x3 + 1.5 * w3, sp_s, width=w3, color="#f59e0b", alpha=0.85, label="Split Prec (Stress)", zorder=3)

    ax3.set_ylim(0.0, 122.0)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([lbl for _, lbl in pols_p3], fontsize=8.6)
    ax3.set_ylabel("Executed Action Precision (%)")
    ax3.set_title("Holdout Executed Action Precision (K=60)", pad=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax3.legend(loc="upper center", ncol=2, frameon=False, fontsize=7.5, columnspacing=0.8)

    for bars in [bm1, bm2, bs1, bs2]:
        for bar in bars:
            h = bar.get_height()
            lbl = "0%" if h <= 0.1 else f"{h:.0f}%"
            ax3.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, lbl, ha="center", va="bottom", fontsize=7.0)

    fig.suptitle(
        "Track B (Evid-MSED) Holdout Diagnosis: End-to-End Stress-30%, 24-Event Head/Tail Strata & Action Precision",
        fontsize=12.2,
        fontweight="bold",
        y=0.955,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] Holdout stress & strata figure saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Track B Phase 2 Validation & Holdout figures.")
    parser.add_argument("--val-json", type=Path, default=PHASE2_DIR / "ned_validation_results.json")
    parser.add_argument("--hold-json", type=Path, default=PHASE2_DIR / "ned_holdout_results.json")
    parser.add_argument("--diag-json", type=Path, default=PHASE2_DIR / "ned_holdout_stress_and_diag.json")
    parser.add_argument("--out", type=Path, default=FIG_DIR / "trackB_phase2_ned_val_holdout.png")
    parser.add_argument("--diag-out", type=Path, default=FIG_DIR / "trackB_phase2_holdout_stress_and_strata.png")
    parser.add_argument("--diag-only", action="store_true")
    args = parser.parse_args()
    if not args.diag_only:
        plot_phase2(args.val_json, args.hold_json, args.out)
    if args.diag_json.exists():
        plot_holdout_diag(args.hold_json, args.diag_json, args.diag_out)


if __name__ == "__main__":
    main()
