"""Generate publication-grade Figure 1 for Track B (Causal Visual Dilemma & Two-Stage Evid-MSED Architecture)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    fc: str = "#F8FAFC",
    ec: str = "#334155",
    title_col: str = "#0F172A",
    lw: float = 1.4,
    fontsize_t: float = 9.0,
    fontsize_b: float = 7.7,
    body_offset: float = 0.018,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    if body:
        ax.text(x + w / 2, y + h - 0.016, title, ha="center", va="top", fontsize=fontsize_t, fontweight="bold", color=title_col)
        ax.text(x + w / 2, y + h / 2 - body_offset, body, ha="center", va="center", fontsize=fontsize_b, color="#1E293B", linespacing=1.30)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fontsize_t, fontweight="bold", color=title_col)


def _arrow(ax, x1: float, y1: float, x2: float, y2: float, color: str = "#334155", lw: float = 1.7) -> None:
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>,head_length=5.5,head_width=3.5",
        color=color,
        linewidth=lw,
    )
    ax.add_patch(arr)


def main() -> None:
    fig = plt.figure(figsize=(15.0, 6.9), dpi=240)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.5,
        0.985,
        "Concurrent Competing-Event Visual Dilemma & Conflict-Gated Multimodal SED (Evid-MSED)",
        ha="center",
        va="top",
        fontsize=12.6,
        fontweight="bold",
        color="#0F172A",
    )

    # ================= TOP ROW: (a) FAILURE MODES vs (b) EVIDENCE-GATED PRINCIPLE =================
    p1 = FancyBboxPatch(
        (0.02, 0.56),
        0.46,
        0.38,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor="#FEF2F2",
        edgecolor="#DC2626",
        linewidth=1.4,
    )
    ax.add_patch(p1)
    ax.text(
        0.25,
        0.928,
        "(a) Two Failure Modes When 72h Chronological Shortcut Is Removed",
        ha="center",
        va="top",
        fontsize=9.8,
        fontweight="bold",
        color="#991B1B",
    )

    _box(
        ax,
        0.038,
        0.665,
        0.20,
        0.21,
        "Regime 1: Naive Fusion Collapse",
        "Competing Mismatch & Distractors:\n"
        "- Dev Orig: 27 (CLIP) / 7 (VLM) misleads\n"
        "- Dev Mismatch: 158 (CLIP) / 77 (VLM) flips\n"
        "- Stress-30%: Naive MV crashes -13.35 pp\n"
        "  (Val) & -8.11 pp (Holdout), < Text-Only!",
        fc="#FFFFFF",
        ec="#EF4444",
        title_col="#991B1B",
        fontsize_t=8.3,
        fontsize_b=7.3,
        body_offset=0.016,
    )
    _box(
        ax,
        0.262,
        0.665,
        0.20,
        0.21,
        "Regime 2: CNTD-Raw Over-Filtering",
        "Text-dispersion penalty kills rescues:\n"
        "- Same event w/ varied captions gets cut\n"
        "- Dev Rescues halved: +96 -> +59 (CLIP)\n"
        "  and 17 -> 7 (Qwen3-VL-8B)\n"
        "- Macro-B3 drops -4.80 pp vs. Vis-IDF",
        fc="#FFFFFF",
        ec="#EA580C",
        title_col="#9A3412",
        fontsize_t=8.3,
        fontsize_b=7.3,
        body_offset=0.016,
    )
    ax.text(
        0.25,
        0.602,
        "Real concurrent NED has +96/+17 genuine rescues alongside 158/77 mismatch flips.\n"
        "Static Visual-IDF fails on unique (vidf=1.0) distractors -> requires cross-modal gating.",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#7F1D1D",
        linespacing=1.28,
    )

    p2 = FancyBboxPatch(
        (0.52, 0.56),
        0.46,
        0.38,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor="#EFF6FF",
        edgecolor="#2563EB",
        linewidth=1.4,
    )
    ax.add_patch(p2)
    ax.text(
        0.75,
        0.928,
        "(b) Evid-MSED Resolution: Stage-1 Vis-IDF + Stage-2 Explicit Conflict Gate",
        ha="center",
        va="top",
        fontsize=9.8,
        fontweight="bold",
        color="#1E40AF",
    )

    _box(
        ax,
        0.538,
        0.665,
        0.20,
        0.21,
        "Stage 1: Vis-IDF + §3.3 Impute",
        "Zero-API Graph Evidence Filtering:\n"
        "- Downweight repeated memes (CLIP>=0.88)\n"
        "- Unified soft-imputation (no $\\delta_i\\delta_j$ bug):\n"
        "  5/5 Macro-B3 wins (+3.27 pp Stress-30%)\n"
        "- Holds +8.48 / +4.18 pp vs Naive Stress",
        fc="#FFFFFF",
        ec="#2563EB",
        title_col="#1E40AF",
        fontsize_t=8.3,
        fontsize_b=7.3,
        body_offset=0.016,
    )
    _box(
        ax,
        0.762,
        0.665,
        0.20,
        0.21,
        "Stage 2: Cross-Modal Conflict Gate",
        "Qwen3-VL-8B Explicit Self/Fallback Gate:\n"
        "- Gate = (vis_disc && !vis_conflict)\n"
        "- Dev Orig: 85.0% (+10 pp, 0 misleads)\n"
        "- Dev Mismatch: 77 -> 5 (Inline) / 0 (Comp)\n"
        "- Holdout K=60: 24.63 vs 12.77 Fix/Q",
        fc="#FFFFFF",
        ec="#16A34A",
        title_col="#166534",
        fontsize_t=8.3,
        fontsize_b=7.3,
        body_offset=0.016,
    )
    ax.text(
        0.75,
        0.602,
        "Clean Separation: Stage 1 removes structural hubs & fills missing images safely;\n"
        "Stage 2 catches unique semantic conflicts on high-leverage mesoscopic cluster seams.",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#1E3A8A",
        linespacing=1.28,
    )

    # ================= BOTTOM ROW: END-TO-END EVID-MSED PIPELINE =================
    p3 = FancyBboxPatch(
        (0.02, 0.03),
        0.96,
        0.49,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor="#F8FAFC",
        edgecolor="#475569",
        linewidth=1.2,
    )
    ax.add_patch(p3)
    ax.text(
        0.50,
        0.506,
        "(c) End-to-End Evid-MSED Pipeline Under Frozen Leakage-Free Protocol (protocol_trackB_v1.json)",
        ha="center",
        va="top",
        fontsize=9.9,
        fontweight="bold",
        color="#0F172A",
    )

    _box(
        ax,
        0.038,
        0.185,
        0.20,
        0.265,
        "1. Concurrent Multimodal Graph",
        "Event-Relative Time (No 72h)\n"
        "Dense Text (MiniLM) + Attr Graph\n"
        "Chunked O(NB) Visual-IDF Score\n"
        "§3.3 Soft Imputation (b_mean >= 0.55,\n"
        "no $\\delta_i\\delta_j$) -> Sparse Visual View $W^V$",
        fc="#FFFFFF",
        ec="#475569",
        title_col="#0F172A",
        fontsize_t=8.6,
        fontsize_b=7.4,
        body_offset=0.020,
    )
    _box(
        ax,
        0.268,
        0.185,
        0.21,
        0.265,
        "2. 2D-SE & Seam Router",
        "Exact Greedy + Refinement 2D-SE\n"
        "Interleaved Budgeted Seams:\n"
        "• 70% MM_MERGE_PAIR\n"
        "  (text_bridge + visual_rescue)\n"
        "• 30% MM_SPLIT_SEAM (cond<=0.5)",
        fc="#FEF3C7",
        ec="#D97706",
        title_col="#92400E",
        fontsize_t=8.6,
        fontsize_b=7.4,
        body_offset=0.020,
    )
    _box(
        ax,
        0.508,
        0.185,
        0.22,
        0.265,
        "3. Dual-Image VLM Seam Query",
        "Visual-IDF Weighted Rep. Images\n"
        "(384px JPEG) + Anonymized Medoids\n"
        "Qwen/Qwen3-VL-8B-Instruct\n"
        "Structured Extract-Then-Compare:\n"
        "event_a/b, vis_disc, vis_conflict",
        fc="#E0E7FF",
        ec="#4338CA",
        title_col="#312E81",
        fontsize_t=8.6,
        fontsize_b=7.4,
        body_offset=0.020,
    )
    _box(
        ax,
        0.758,
        0.185,
        0.204,
        0.265,
        "4. Gate & Partition Edit",
        "Accept Multimodal iff\n"
        "vis_disc=True AND vis_conflict=False\n"
        "Else Text Fallback (Inline / CleanFB)\n"
        "Direct Split-then-Union-Find\n"
        "Concave Peak: K=60 (180-query budget)",
        fc="#DCFCE7",
        ec="#15803D",
        title_col="#166534",
        fontsize_t=8.6,
        fontsize_b=7.4,
        body_offset=0.020,
    )

    _arrow(ax, 0.240, 0.317, 0.266, 0.317, color="#334155")
    _arrow(ax, 0.480, 0.317, 0.506, 0.317, color="#B45309")
    _arrow(ax, 0.730, 0.317, 0.756, 0.317, color="#4338CA")

    _box(
        ax,
        0.038,
        0.052,
        0.924,
        0.098,
        "Leakage-Free Split Isolation & Audited Real-Token Verification",
        "Dev (8 ev, Gate B) -> Validation (8 ev, Clean+Stress30) -> Frozen SHA-256 Protocol -> "
        "One-Shot Holdout (24 ev, 11,955 posts, 9,311/9,311 imgs, 360/360 OK @ 277.6k tok)",
        fc="#F1F5F9",
        ec="#64748B",
        title_col="#334155",
        fontsize_t=8.6,
        fontsize_b=7.7,
        body_offset=0.013,
    )

    out = Path(__file__).resolve().parent / "figures" / "fig_overview_evid_msed.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig1] Saved Track B overview figure to {out}")


if __name__ == "__main__":
    main()
