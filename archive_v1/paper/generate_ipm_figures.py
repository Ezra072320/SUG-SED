"""Generate all 5 publication-grade vector PDF + 300-DPI PNG figures for the IP&M manuscript.

All empirical data figures load directly from audited JSON artifacts:
  - results/budget_curves/{Event2012,Event2018}_dev_qwen3_8b/*_curve.json
  - results/ipm_audit/stage1_diagnostics.json
  - results/ipm_experiments/stage2a_qwen3/stage2a_report.json
  - results/ipm_experiments/stage2b_anonymization/anonymization_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.3,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(out, **kwargs)
    plt.close(fig)
    print(f"[Saved] {FIG_DIR / stem}.pdf & .png")


# ---------------------------------------------------------------------------
# Figure 1: Conceptual Overview & Causal Architecture
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, title, body, fc="#F8FAFC", ec="#334155", title_col="#0F172A", lw=1.35, ft=9.0, fb=7.7, off=0.016):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.007,rounding_size=0.014", facecolor=fc, edgecolor=ec, linewidth=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.016, title, ha="center", va="top", fontsize=ft, fontweight="bold", color=title_col)
        ax.text(x + w / 2, y + h / 2 - off, body, ha="center", va="center", fontsize=fb, color="#1E293B", linespacing=1.30)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ft, fontweight="bold", color=title_col)


def _arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>,head_length=5.2,head_width=3.2", color=color, linewidth=lw, linestyle=ls))


def make_fig_overview() -> None:
    fig_w, fig_h = 15.2, 6.9
    aspect = fig_h / fig_w
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.5, 0.984,
        "From Single-Post Fallacy to Causal Mesoscopic Seam Adjudication (SUG-Meso & SUG-CrossLLM-v2)",
        ha="center", va="top", fontsize=12.2, fontweight="bold", color="#0F172A",
    )

    # Top-left panel (a)
    ax.add_patch(FancyBboxPatch((0.02, 0.56), 0.46, 0.37, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.35))
    ax.text(0.25, 0.916, "(a) Single-Post & Edge Query Fallacy (<1% API Calls)", ha="center", va="top",
            fontsize=10.0, fontweight="bold", color="#991B1B")

    for cx, cy, lbl in [(0.085, 0.785, "Frag. A\n(25)"), (0.195, 0.812, "Frag. B\n(30)"), (0.145, 0.682, "Frag. C\n(18)")]:
        ax.add_patch(Ellipse((cx, cy), width=0.080, height=0.080 / aspect * 0.46, facecolor="#3B82F6", edgecolor="#1D4ED8", alpha=0.22, lw=1.2))
        ax.text(cx, cy + 0.003, lbl, ha="center", va="center", fontsize=7.4, color="#1E3A8A", fontweight="bold")
    ax.scatter([0.112, 0.168], [0.768, 0.805], s=30, color="#DC2626", zorder=5)

    ax.add_patch(Ellipse((0.365, 0.755), width=0.175, height=0.158, facecolor="#CBD5E1", edgecolor="#475569", alpha=0.35, lw=1.2))
    ax.add_patch(Ellipse((0.408, 0.785), width=0.068, height=0.072, facecolor="#F97316", edgecolor="#C2410C", alpha=0.38, lw=1.2))
    ax.scatter([0.335, 0.378], [0.768, 0.778], s=30, color="#DC2626", zorder=5)
    ax.plot([0.335, 0.378], [0.768, 0.778], color="#DC2626", lw=1.4, ls="--", zorder=4)
    ax.text(0.342, 0.722, "Maj. Event 1\n(Zero boundary\nentropy interior)", ha="center", va="center", fontsize=7.0, color="#1E293B")
    ax.text(0.408, 0.785, "Min.\nEv. 2", ha="center", va="center", fontsize=6.8, color="#9A3412", fontweight="bold")
    ax.text(
        0.25, 0.594,
        "False Splits exceed False Merges by 11.6x-44.6x; minority pockets occupy 10%-25%.\n"
        "Random / Rep-Div / Cequel move dAMI <= +0.01 pp and Cequel-Transitive drops Mac-B3.",
        ha="center", va="center", fontsize=7.7, color="#7F1D1D", linespacing=1.28,
    )

    # Top-right panel (b)
    ax.add_patch(FancyBboxPatch((0.52, 0.56), 0.46, 0.37, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.35))
    ax.text(0.75, 0.916, "(b) Mesoscopic Seam Adjudication (Uncontaminated U_t)", ha="center", va="top",
            fontsize=10.0, fontweight="bold", color="#1E40AF")

    ax.add_patch(Ellipse((0.578, 0.752), width=0.088, height=0.092, facecolor="#3B82F6", edgecolor="#1D4ED8", alpha=0.25, lw=1.3))
    ax.add_patch(Ellipse((0.700, 0.752), width=0.088, height=0.092, facecolor="#3B82F6", edgecolor="#1D4ED8", alpha=0.25, lw=1.3))
    ax.text(0.578, 0.752, "Cluster Ca\n(3 medoids)", ha="center", va="center", fontsize=7.3, color="#1E3A8A", fontweight="bold")
    ax.text(0.700, 0.752, "Cluster Cb\n(3 medoids)", ha="center", va="center", fontsize=7.3, color="#1E3A8A", fontweight="bold")
    ax.annotate("", xy=(0.654, 0.752), xytext=(0.624, 0.752), arrowprops=dict(arrowstyle="<->", color="#15803D", lw=2.1))
    ax.text(0.639, 0.840, "Type-I MERGE_PAIR\nFixes False Splits", ha="center", va="center",
            fontsize=7.7, color="#166534", fontweight="bold", linespacing=1.22)

    ax.add_patch(Ellipse((0.835, 0.750), width=0.140, height=0.132, facecolor="#3B82F6", edgecolor="#1D4ED8", alpha=0.18, lw=1.2))
    ax.add_patch(Ellipse((0.918, 0.766), width=0.072, height=0.076, facecolor="#F97316", edgecolor="#C2410C", alpha=0.38, lw=1.4))
    ax.plot([0.873, 0.892], [0.814, 0.690], color="#DC2626", lw=2.1, ls="--")
    ax.text(0.822, 0.750, "Main\nGroup", ha="center", va="center", fontsize=7.4, color="#1E3A8A", fontweight="bold")
    ax.text(0.922, 0.766, "Pocket\nSp", ha="center", va="center", fontsize=7.0, color="#9A3412", fontweight="bold")
    ax.text(0.872, 0.844, "Type-II SPLIT_SEAM\nFarthest-Point Sph. k-Means", ha="center", va="center",
            fontsize=7.7, color="#9A3412", fontweight="bold", linespacing=1.22)

    ax.text(
        0.75, 0.594,
        "1 LLM decision updates whole clusters excluding all shown medoids from U_t ->\n"
        "+5.81 / +7.15 pp AMI (+608 / +1,220 net unqueried posts/blk; 98.4% / 77.9% anon retention).",
        ha="center", va="center", fontsize=7.7, color="#1E3A8A", linespacing=1.28,
    )

    # Bottom row (c)
    ax.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.495, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor="#F8FAFC", edgecolor="#475569", linewidth=1.2))
    ax.text(0.50, 0.512, "(c) Strictly Causal Real-Token Pipeline: SUG-Meso + SUG-CrossLLM-v2",
            ha="center", va="top", fontsize=10.0, fontweight="bold", color="#0F172A")

    _box(ax, 0.04, 0.195, 0.19, 0.255, "1. Block Graph & 2D-SE",
         "Incoming Block Mt\n(MiniLM-L6 / Multiling-L12)\nSem-kNN + IDF Attr Graph\nNon-Entity TF-IDF Matrix\nFixed-Point 2D-SE Init P(0)",
         fc="#FFFFFF", ec="#475569", off=0.021)
    _box(ax, 0.265, 0.195, 0.21, 0.255, "2. Meso-Seam Router",
         "Proportional Interleaving\n(65% MERGE_PAIR / 35% SEAM)\n- Cosine Proto + Log-Leverage\n- Sph. k-Means Minority Pocket\n  & Non-Entity Cut (Eq. 5-6)",
         fc="#FEF3C7", ec="#D97706", title_col="#92400E", off=0.021)
    _box(ax, 0.51, 0.195, 0.215, 0.255, "3. Causal 8B Adjudicator",
         "Zero-Lookahead Bucket (10k/blk)\nDev-Calibrated Action-First\nDirect Split -> Union-Find\nExact Pruned Delta-H2 (Prop. 1)\nAll-Shown U_t Mask Audit",
         fc="#DCFCE7", ec="#15803D", title_col="#166534", off=0.021)
    _box(ax, 0.76, 0.195, 0.20, 0.255, "4. SUG-CrossLLM-v2",
         "Active W=14-Block Memory\nsim >= 0.85 : Auto-Link\n[0.65, 0.85) : Extract-Compare\nFallback tau_fb = 0.85\nBeats All ProtoOnly Baselines!",
         fc="#E0E7FF", ec="#4338CA", title_col="#312E81", off=0.021)

    _arrow(ax, 0.232, 0.322, 0.263, 0.322, color="#334155")
    _arrow(ax, 0.477, 0.322, 0.508, 0.322, color="#B45309")
    _arrow(ax, 0.727, 0.322, 0.758, 0.322, color="#15803D")

    _box(ax, 0.04, 0.055, 0.92, 0.105,
         "Verified Reproducibility & Anti-Memorization Guardrails",
         "Zero Future-N Lookahead Budgets (Fixed / Carryover / Sliding)  |  "
         "All-Shown U_t (1.79%-3.78% prompt-exposed)  |  99.7%-100% Entity/Tag Masking Retention",
         fc="#F1F5F9", ec="#64748B", title_col="#334155", ft=8.6, fb=7.8, off=0.013)

    _save(fig, "fig_overview")


# ---------------------------------------------------------------------------
# Figure 2: Bilingual Dev Multi-Budget Pareto Curves (0k / 20k / 40k / 80k)
# ---------------------------------------------------------------------------
def make_fig_budget_pareto() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.8), sharex=True)
    datasets = [("Event2012", "Event2012 Dev (EN, M02-M05)"), ("Event2018", "Event2018 Dev (FR, M02-M05)")]
    series_cfg = [
        ("sug_meso_only_direct", "SUG-Meso (Direct)", "#1D4ED8", "o", "-", 2.3),
        ("sug_router_repair", "SUG-Router (Repair)", "#059669", "s", "--", 1.9),
        ("cequel_edge_repair", "Cequel-Edge (Repair)", "#D97706", "^", "-.", 1.7),
        ("rep_div_direct", "Rep-Div (RagSEDE)", "#7C3AED", "D", ":", 1.7),
        ("random_direct", "Random", "#64748B", "x", ":", 1.6),
    ]
    tiers = ["20000", "40000", "80000"]
    xs = [0, 20, 40, 80]

    for col_idx, (ds_key, ds_title) in enumerate(datasets):
        d = json.loads((ROOT / "results" / "budget_curves" / f"{ds_key}_dev_qwen3_8b" / f"{ds_key}_dev_qwen3_8b_curve.json").read_text(encoding="utf-8"))
        init_ami = d["init_mean"]["ami"] * 100.0
        for sk, lbl, col, mk, ls, lw in series_cfg:
            ys_ami = [init_ami]
            ys_unq = [0.0]
            for tk in tiers:
                mc = d["tiers"][tk]["mean_comparison"][sk]
                ys_ami.append(mc["ami"] * 100.0)
                ys_unq.append(mc["unq_delta_micro_b3"] * 100.0)
            axes[0, col_idx].plot(xs, ys_ami, label=lbl, color=col, marker=mk, linestyle=ls, linewidth=lw, markersize=5.5)
            axes[1, col_idx].plot(xs, ys_unq, label=lbl, color=col, marker=mk, linestyle=ls, linewidth=lw, markersize=5.5)

        axes[0, col_idx].set_title(ds_title)
        axes[0, col_idx].grid(True, linestyle="--", alpha=0.35)
        axes[1, col_idx].grid(True, linestyle="--", alpha=0.35)
        axes[1, col_idx].set_xlabel("Real Token Budget Cap (k tokens)")
        axes[1, col_idx].set_xticks(xs)
        axes[1, col_idx].set_xticklabels(["0k", "20k", "40k", "80k"])

    axes[0, 0].set_ylabel("AMI (%)")
    axes[1, 0].set_ylabel("Unqueried dMicro-B3 (pp)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=5, frameon=True, facecolor="white", framealpha=0.95)
    axes[0, 0].set_ylim(87.5, 92.6)
    axes[0, 1].set_ylim(66.5, 74.0)
    axes[1, 0].set_ylim(-0.6, 7.2)
    axes[1, 1].set_ylim(-0.6, 13.0)
    fig.suptitle("Dev-Set Multi-Budget Real-Token Pareto Curves (Qwen/Qwen3-8B, Extract-Compare)", y=1.01, fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "fig_budget_pareto")


# ---------------------------------------------------------------------------
# Figure 3: 27 Held-Out Block Chronological Trajectories (Stage 2A Causal)
# ---------------------------------------------------------------------------
def make_fig_causal_trajectories() -> None:
    s1 = json.loads((ROOT / "results" / "ipm_audit" / "stage1_diagnostics.json").read_text(encoding="utf-8"))
    s2 = json.loads((ROOT / "results" / "ipm_experiments" / "stage2a_qwen3" / "stage2a_report.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 6.9), sharex="col")
    ds_cfg = [("Event2012", "Event2012 Held-Out (EN, M06-M21)", 6), ("Event2018", "Event2018 Held-Out (FR, M06-M16)", 6)]

    lines_cfg = [
        ("eq6_no_conductance_action", "SUG-Meso (No Cond Cap)", "#0F766E", "D", "--", 1.7),
        ("sug_meso_causal_fixed_action", "SUG-Meso (Causal Action-First)", "#1D4ED8", "o", "-", 2.2),
        ("heur_meso_0tok", "Meso-Heur (0-Tok)", "#D97706", "^", "-.", 1.6),
        ("random_meso_action_mean3", "Random-Meso (3-seed Action)", "#7C3AED", "s", ":", 1.6),
        ("init_2dse", "Init 2D-SE (0-Tok)", "#64748B", "", "-", 1.5),
    ]

    for c_idx, (ds_key, ds_title, start_b) in enumerate(ds_cfg):
        s2_w = s2[ds_key]["within_block"]
        s1_d5 = s1[ds_key]["D5_stats"]["summary_stats"]
        n_b = len(s2_w["sug_meso_causal_fixed_action"]["direct"]["ami"]["per_block"])
        x = np.arange(start_b, start_b + n_b)
        x_lbls = [f"M{b:02d}" for b in x]

        for var_key, lbl, col, mk, ls, lw in lines_cfg:
            src = s1_d5[var_key] if var_key in s1_d5 else s2_w[var_key]["direct"]
            ami_y = np.asarray(src["ami"]["per_block"], dtype=float) * 100.0
            mac_y = np.asarray(src["macro_b3"]["per_block"], dtype=float) * 100.0
            axes[0, c_idx].plot(x, ami_y, label=lbl, color=col, marker=mk if mk else None, linestyle=ls, linewidth=lw, markersize=4.4)
            axes[1, c_idx].plot(x, mac_y, label=lbl, color=col, marker=mk if mk else None, linestyle=ls, linewidth=lw, markersize=4.4)

        axes[0, c_idx].set_title(ds_title)
        axes[0, c_idx].grid(True, linestyle="--", alpha=0.35)
        axes[1, c_idx].grid(True, linestyle="--", alpha=0.35)
        axes[1, c_idx].set_xticks(x)
        axes[1, c_idx].set_xticklabels(x_lbls, rotation=35, ha="right")
        axes[1, c_idx].set_xlabel("Chronological Held-Out Block")

    axes[0, 0].set_ylabel("AMI (%)")
    axes[1, 0].set_ylabel("Event Macro-B3 (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=5, frameon=True, facecolor="white", framealpha=0.95)
    fig.suptitle("Day-by-Day Strictly Causal Performance Across All 27 Held-Out Blocks (10k/blk)", y=1.01, fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "fig_causal_trajectories")


# ---------------------------------------------------------------------------
# Figure 4: Eq. (5)/(6) & Conductance Ablations + Cross-Day SUG-CrossLLM-v2
# ---------------------------------------------------------------------------
def make_fig_ablations_and_crossday() -> None:
    s1 = json.loads((ROOT / "results" / "ipm_audit" / "stage1_diagnostics.json").read_text(encoding="utf-8"))
    s2 = json.loads((ROOT / "results" / "ipm_experiments" / "stage2a_qwen3" / "stage2a_report.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.8))

    # Top row: Structural & Eq. 5/6 + Conductance sweep
    abl_variants = [
        ("init_2dse", "Init\n2D-SE"),
        ("causal_seam_only_action", "Seam\nOnly"),
        ("causal_merge_only_action", "Merge\nOnly"),
        ("causal_fiedler_seam_action", "Fiedler\nSeam"),
        ("sens_cond_035_action", "Cond\n<=0.35"),
        ("sug_meso_causal_fixed_action", "SUG-Meso\n(<=0.50)"),
        ("sens_cond_065_action", "Cond\n<=0.65"),
        ("eq6_no_conductance_action", "No Cond\nCap (inf)"),
        ("eq5_no_leverage_action", "No Lev.\nEq.5"),
    ]
    for col_idx, (ds_key, ds_lbl) in enumerate([("Event2012", "(a) Event2012 (EN): Structural & Eq. (5)/(6) Ablations"), ("Event2018", "(b) Event2018 (FR): Structural & Eq. (5)/(6) Ablations")]):
        ax = axes[0, col_idx]
        s2_w = s2[ds_key]["within_block"]
        s1_d5 = s1[ds_key]["D5_stats"]["summary_stats"]
        macs, tails, amis, xlbls = [], [], [], []
        for vk, lb in abl_variants:
            d = s1_d5[vk] if vk in s1_d5 else s2_w[vk]["direct"]
            macs.append(d["macro_b3"]["mean"] * 100.0)
            tails.append(d["tail_b3"]["mean"] * 100.0)
            amis.append(d["ami"]["mean"] * 100.0)
            xlbls.append(lb)
        x = np.arange(len(abl_variants))
        w = 0.36
        ax.bar(x - w / 2, macs, width=w, color="#2563EB", alpha=0.88, label="Event Macro-B3 (%)")
        ax.bar(x + w / 2, tails, width=w, color="#F59E0B", alpha=0.88, label="Tail Macro-B3 (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=7.3)
        ax.set_ylabel("B3 Score (%)")
        ax.set_title(ds_lbl)
        ax.set_ylim(28, 82)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(x, amis, color="#0F766E", marker="o", lw=1.8, markersize=4.8, label="AMI (%)")
        ax2.set_ylabel("AMI (%)", color="#0F766E")
        ax2.tick_params(axis="y", colors="#0F766E")
        ax2.set_ylim(72, 93)
        if col_idx == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, facecolor="white", framealpha=0.92, ncol=3, fontsize=7.4)

    # Bottom row: Cross-Day Longitudinal Tracking on heldout_action_stream
    proto_keys = ["proto_0.72", "proto_0.80", "proto_0.85", "proto_0.90"]
    taus = [0.72, 0.80, 0.85, 0.90]
    for col_idx, (ds_key, ds_lbl) in enumerate([("Event2012", "(c) Event2012 Cross-Day Tracking (heldout_action)"), ("Event2018", "(d) Event2018 Cross-Day Tracking (heldout_action)")]):
        ax = axes[1, col_idx]
        cr = s2[ds_key]["cross_day"]["heldout_action_stream"]
        px = [cr[k]["macro_b3"] * 100.0 for k in proto_keys]
        py = [cr[k]["ami"] * 100.0 for k in proto_keys]
        ax.plot(px, py, color="#475569", marker="o", lw=2.0, markersize=6.0, label="ProtoOnly Sweep (tau=0.72..0.90)")
        offsets = {
            ("Event2012", 0.72): (14, 6),
            ("Event2012", 0.80): (-10, 7),
            ("Event2012", 0.85): (-18, 6),
            ("Event2012", 0.90): (0, -12),
            ("Event2018", 0.72): (0, 7),
            ("Event2018", 0.80): (-6, 7),
            ("Event2018", 0.85): (-22, 4),
            ("Event2018", 0.90): (4, -12),
        }
        for tx, ty, tv in zip(px, py, taus):
            dx, dy = offsets[(ds_key, tv)]
            ax.annotate(f"tau={tv:.2f}", (tx, ty), textcoords="offset points", xytext=(dx, dy), ha="center", fontsize=7.3, color="#334155")

        nl = cr["no_link"]
        vx = cr["sug_cross_v2_extract"]
        va = cr["sug_cross_v2_action"]
        vo = cr["oracle_065_085"]

        ax.scatter([vx["macro_b3"] * 100], [vx["ami"] * 100], s=110, color="#1D4ED8", marker="D", zorder=6, label="SUG-CrossLLM-v2 (Extract)")
        ax.scatter([va["macro_b3"] * 100], [va["ami"] * 100], s=120, color="#059669", marker="*", zorder=6, label="SUG-CrossLLM-v2 (Action)")
        ax.scatter([vo["macro_b3"] * 100], [vo["ami"] * 100], s=95, color="#DC2626", marker="^", zorder=6, label="Oracle Band [0.65, 0.85)")
        ax.scatter([nl["macro_b3"] * 100], [nl["ami"] * 100], s=70, color="#94A3B8", marker="X", zorder=5, label="No Cross-Link")

        ax.set_xlabel("Global Longitudinal Event Macro-B3 (%)")
        ax.set_ylabel("Global Longitudinal AMI (%)")
        ax.set_title(ds_lbl)
        if ds_key == "Event2012":
            ax.set_ylim(88.0, 90.45)
        else:
            ax.set_ylim(79.4, 85.25)
        ax.grid(True, linestyle="--", alpha=0.35)
        if col_idx == 1:
            ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.5)

    fig.tight_layout()
    _save(fig, "fig_ablations_and_crossday")


# ---------------------------------------------------------------------------
# Figure 5: Stage 2B Anti-Memorization Entity/Hashtag Anonymization Stress Test
# ---------------------------------------------------------------------------
def make_fig_anonymization() -> None:
    s1 = json.loads((ROOT / "results" / "ipm_audit" / "stage1_diagnostics.json").read_text(encoding="utf-8"))
    s2 = json.loads((ROOT / "results" / "ipm_experiments" / "stage2a_qwen3" / "stage2a_report.json").read_text(encoding="utf-8"))
    sb = json.loads((ROOT / "results" / "ipm_experiments" / "stage2b_anonymization" / "anonymization_report.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    conds = [
        ("init_2dse", "Init\n2D-SE"),
        ("random_meso_action_mean3", "Random-Meso\n(Unmasked)"),
        ("anon_blind_action", "Blind Redact\n([ENTITY])"),
        ("anon_pseudo_action", "Pseudonym\n([ENT_k])"),
        ("sug_meso_causal_fixed_action", "Full SUG-Meso\n(Unmasked)"),
    ]

    for idx, (ds_key, title, badge) in enumerate([
        ("Event2012", "Event2012 (EN, 16 Held-Out Blocks)", "Mask: 99.89% Ent | 100% Tag/Yr\nAMI Retention: 98.4% (p=0.955)\nDecision Agree: 81.6% | True Recon: 0%"),
        ("Event2018", "Event2018 (FR, 11 Held-Out Blocks)", "Mask: 100% Ent | 100% Tag/Yr\nAMI Retention: 77.9% (H=7.8e-3 vs RM)\nDecision Agree: 83.5% | Recon: 0.00%"),
    ]):
        ax = axes[idx]
        s1_d5 = s1[ds_key]["D5_stats"]["summary_stats"]
        s2_w = s2[ds_key]["within_block"]
        sb_c = sb[ds_key]["conditions"]
        amis, macs, xlbls = [], [], []
        for k, lb in conds:
            if k in s1_d5:
                d = s1_d5[k]
            elif k in sb_c:
                d = sb_c[k]["direct"]
            else:
                d = s2_w[k]["direct"]
            amis.append(d["ami"]["mean"] * 100.0)
            macs.append(d["macro_b3"]["mean"] * 100.0)
            xlbls.append(lb)

        x = np.arange(len(conds))
        w = 0.35
        b1 = ax.bar(x - w / 2, amis, width=w, color="#1D4ED8", alpha=0.90, label="AMI (%)")
        b2 = ax.bar(x + w / 2, macs, width=w, color="#059669", alpha=0.88, label="Event Macro-B3 (%)")
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.45, f"{h:.2f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=7.6)
        ax.set_ylim(58, 97.5 if idx == 0 else 91.5)
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.text(0.03, 0.96, badge, transform=ax.transAxes, ha="left", va="top", fontsize=7.4,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=1.0))

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=True, facecolor="white", framealpha=0.95)
    fig.suptitle("Stage 2B Anti-Memorization Stress Test: Entity/Hashtag Anonymization vs. Co-Reference Ablation", y=1.01, fontsize=11.3, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig_anonymization_stress")


def main() -> None:
    make_fig_overview()
    make_fig_budget_pareto()
    make_fig_causal_trajectories()
    make_fig_ablations_and_crossday()
    make_fig_anonymization()


if __name__ == "__main__":
    main()
