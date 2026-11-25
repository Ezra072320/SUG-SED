"""Generate publication-grade Figure 1 (Conceptual Motivation + SUG-Meso / SUG-CrossLLM Architecture)."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch


def _box(
    ax, x, y, w, h, title, body,
    fc="#F8FAFC", ec="#334155", title_col="#0F172A",
    lw=1.4, fontsize_t=9.3, fontsize_b=8.0, body_offset=0.015,
):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    if body:
        ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=fontsize_t, fontweight="bold", color=title_col)
        ax.text(x + w / 2, y + h / 2 - body_offset, body, ha="center", va="center", fontsize=fontsize_b, color="#1E293B", linespacing=1.32)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fontsize_t, fontweight="bold", color=title_col)


def _arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.7, ls="-"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5.5,head_width=3.5",
        color=color, linewidth=lw, linestyle=ls,
    )
    ax.add_patch(arr)


def main() -> None:
    fig_w, fig_h = 15.0, 6.8
    aspect = fig_h / fig_w
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=240)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.5, 0.982,
        "From Single-Post Fallacy to Mesoscopic Seam Adjudication (SUG-Meso & SUG-CrossLLM)",
        ha="center", va="top", fontsize=12.8, fontweight="bold", color="#0F172A",
    )

    # ================= TOP ROW: FAILURE DIAGNOSIS vs MESOSCOPIC SEAMS =================
    p1 = FancyBboxPatch((0.02, 0.56), 0.46, 0.37, boxstyle="round,pad=0.008,rounding_size=0.015",
                        facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.4)
    ax.add_patch(p1)
    ax.text(0.25, 0.918, "(a) Single-Post & Pairwise Query Fallacy (<1% Budget)", ha="center", va="top",
            fontsize=10.2, fontweight="bold", color="#991B1B")

    # Fragmented clusters (True event 1 split into 3 pieces)
    for cx, cy, lbl in [
        (0.085, 0.78, "Frag. A\n(25)"),
        (0.195, 0.81, "Frag. B\n(30)"),
        (0.145, 0.68, "Frag. C\n(18)"),
    ]:
        ax.add_patch(Ellipse((cx, cy), width=0.082, height=0.082 / aspect * 0.48,
                             facecolor="#3B82F6", edgecolor="#1D4ED8", alpha=0.22, lw=1.2))
        ax.text(cx, cy + 0.004, lbl, ha="center", va="center", fontsize=7.5, color="#1E3A8A", fontweight="bold")
    # Place queried single dots in corners so text stays untouched
    ax.scatter([0.112, 0.168], [0.768, 0.805], s=32, color="#DC2626", zorder=5)

    # Impure super-cluster C4 with buried minority pocket
    ax.add_patch(Ellipse((0.365, 0.755), width=0.175, height=0.16,
                         facecolor="#CBD5E1", edgecolor="#475569", alpha=0.35, lw=1.2))
    ax.add_patch(Ellipse((0.408, 0.785), width=0.068, height=0.072,
                         facecolor="#F97316", edgecolor="#C2410C", alpha=0.38, lw=1.2))
    ax.scatter([0.335, 0.378], [0.768, 0.778], s=32, color="#DC2626", zorder=5)
    ax.plot([0.335, 0.378], [0.768, 0.778], color="#DC2626", lw=1.5, ls="--", zorder=4)
    ax.text(0.342, 0.722, "Maj. Event 1\n(Zero boundary\nentropy interior)", ha="center", va="center", fontsize=7.1, color="#1E293B")
    ax.text(0.408, 0.785, "Min.\nEv. 2", ha="center", va="center", fontsize=6.9, color="#9A3412", fontweight="bold")

    ax.text(
        0.25, 0.595,
        "Rep-Div / Cequel-Edge update only 1-2 posts/call -> dAMI = -0.02 ~ +0.01 pp\n"
        "Boundary-entropy routers miss minor pockets buried inside impure super-clusters.",
        ha="center", va="center", fontsize=8.0, color="#7F1D1D", linespacing=1.28,
    )

    # Right Card: Mesoscopic Seams
    p2 = FancyBboxPatch((0.52, 0.56), 0.46, 0.37, boxstyle="round,pad=0.008,rounding_size=0.015",
                        facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.4)
    ax.add_patch(p2)
    ax.text(0.75, 0.918, "(b) Mesoscopic Seam Adjudication (Ours: 28-54 Unqueried Posts / Call)", ha="center", va="top",
            fontsize=10.2, fontweight="bold", color="#1E40AF")

    # Merge-pair side
    ax.add_patch(Ellipse((0.578, 0.755), width=0.088, height=0.095, facecolor="#3B82F6", edgecolor="#1D4ED8", alpha=0.25, lw=1.3))
    ax.add_patch(Ellipse((0.700, 0.755), width=0.088, height=0.095, facecolor="#3B82F6", edgecolor="#1D4ED8", alpha=0.25, lw=1.3))
    ax.text(0.578, 0.755, "Cluster Ca\n(3 medoids)", ha="center", va="center", fontsize=7.4, color="#1E3A8A", fontweight="bold")
    ax.text(0.700, 0.755, "Cluster Cb\n(3 medoids)", ha="center", va="center", fontsize=7.4, color="#1E3A8A", fontweight="bold")
    ax.annotate("", xy=(0.654, 0.755), xytext=(0.624, 0.755),
                arrowprops=dict(arrowstyle="<->", color="#15803D", lw=2.2))
    ax.text(0.639, 0.842, "Type-I MERGE_PAIR\nFixes False Splits (3-10x error)", ha="center", va="center",
            fontsize=7.8, color="#166534", fontweight="bold", linespacing=1.22)

    # Split-seam side (dashed cut cleanly in gap between Main Group & Pocket Sp)
    ax.add_patch(Ellipse((0.835, 0.752), width=0.140, height=0.135, facecolor="#3B82F6", edgecolor="#1D4ED8", alpha=0.18, lw=1.2))
    ax.add_patch(Ellipse((0.918, 0.768), width=0.072, height=0.078, facecolor="#F97316", edgecolor="#C2410C", alpha=0.38, lw=1.4))
    ax.plot([0.873, 0.892], [0.818, 0.692], color="#DC2626", lw=2.2, ls="--")
    ax.text(0.822, 0.752, "Main\nGroup", ha="center", va="center", fontsize=7.5, color="#1E3A8A", fontweight="bold")
    ax.text(0.922, 0.768, "Pocket\nSp", ha="center", va="center", fontsize=7.1, color="#9A3412", fontweight="bold")
    ax.text(0.872, 0.846, "Type-II SPLIT_SEAM\nSpherical KMeans++ + Bottleneck", ha="center", va="center",
            fontsize=7.8, color="#9A3412", fontweight="bold", linespacing=1.22)

    ax.text(
        0.75, 0.595,
        "1 LLM decision updates an entire cluster pair or minority pocket at once ->\n"
        "+3.44 ~ +4.16 pp AMI (+3.21 ~ +3.37 pp Mac-B3), winning 27/27 held-out blocks.",
        ha="center", va="center", fontsize=8.0, color="#1E3A8A", linespacing=1.28,
    )

    # ================= BOTTOM ROW: END-TO-END STREAMING PIPELINE =================
    p3 = FancyBboxPatch((0.02, 0.03), 0.96, 0.495, boxstyle="round,pad=0.008,rounding_size=0.015",
                        facecolor="#F8FAFC", edgecolor="#475569", linewidth=1.2)
    ax.add_patch(p3)
    ax.text(0.50, 0.512, "(c) End-to-End Streaming Architecture: SUG-Meso + Longitudinal SUG-CrossLLM",
            ha="center", va="top", fontsize=10.2, fontweight="bold", color="#0F172A")

    _box(
        ax, 0.04, 0.195, 0.19, 0.255,
        "1. Block Graph & 2D-SE",
        "Incoming Block Mt\n(MiniLM-L6 / Multiling-L12)\nSparse Sem-kNN + Attr Graph\nNon-Entity TF-IDF Matrix\nFixed-Point 2D-SE Init P(0)",
        fc="#FFFFFF", ec="#475569", title_col="#0F172A", body_offset=0.022,
    )
    _box(
        ax, 0.265, 0.195, 0.21, 0.255,
        "2. Meso-Seam Router",
        "Proportional Interleaving\n(65% MERGE_PAIR / 35% SEAM)\n• Cosine Proto + Log-Leverage\n• Spherical KMeans++ Pocket\n  cond <= 0.50, (1 - phi_nonent)",
        fc="#FEF3C7", ec="#D97706", title_col="#92400E", body_offset=0.022,
    )
    _box(
        ax, 0.51, 0.195, 0.215, 0.255,
        "3. Budgeted 8B Adjudicator",
        "Strict Real-Token Ledger\n(prompt + completion capped)\nQwen/Qwen3-8B (non-thinking)\nDirect Split -> Union-Find Merge\n[Ablation: Branch 2 Re-Cluster]",
        fc="#DCFCE7", ec="#15803D", title_col="#166534", body_offset=0.022,
    )
    _box(
        ax, 0.76, 0.195, 0.20, 0.255,
        "4. Longitudinal CrossLLM",
        "Active W=14-Block Memory\n(Proto + Medoids + Keywords)\nsim >= 0.80 : Auto-Link\n[0.55, 0.80) : CrossLLM Query\nRescues Cross-Day Mac-B3!",
        fc="#E0E7FF", ec="#4338CA", title_col="#312E81", body_offset=0.022,
    )

    _arrow(ax, 0.232, 0.322, 0.263, 0.322, color="#334155")
    _arrow(ax, 0.477, 0.322, 0.508, 0.322, color="#B45309")
    _arrow(ax, 0.727, 0.322, 0.758, 0.322, color="#15803D")

    _box(
        ax, 0.04, 0.055, 0.92, 0.105,
        "Strict Real-Token Accounting & Leakage-Free Evaluation Envelope",
        "Immutable JSONL Audit (0.50% - 0.92% queried posts; 3.15 - 4.75 tokens/post)  |  "
        "Dev (M_02..M_05) Frozen Before Unlocking 27 Unseen Held-Out Blocks (M_06+)",
        fc="#F1F5F9", ec="#64748B", title_col="#334155", fontsize_t=8.8, fontsize_b=8.0, body_offset=0.014,
    )

    out = Path(__file__).resolve().parent / "figures" / "fig_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig1] Saved polished {out}")


if __name__ == "__main__":
    main()
