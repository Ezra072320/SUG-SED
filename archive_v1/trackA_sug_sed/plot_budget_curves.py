"""Generate publication-quality Real-Token Pareto Curve figures from curve CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLE_MAP = {
    ("sug_meso_only", "direct"): dict(label="SUG-Meso (Direct)", color="#d62728", marker="o", linestyle="-", linewidth=2.4, markersize=7),
    ("sug_meso_only", "repair"): dict(label="SUG-Meso (Repair)", color="#ff7f0e", marker="s", linestyle="--", linewidth=1.8, markersize=6),
    ("sug_router", "repair"): dict(label="SUG-Router (Repair)", color="#1f77b4", marker="^", linestyle="-", linewidth=2.2, markersize=7),
    ("sug_router", "direct"): dict(label="SUG-Router (Direct)", color="#17becf", marker="v", linestyle=":", linewidth=1.8, markersize=6),
    ("cequel_edge", "repair"): dict(label="Cequel-Edge (Repair)", color="#2ca02c", marker="D", linestyle="-.", linewidth=1.8, markersize=5.5),
    ("rep_div", "direct"): dict(label="Rep-Div / RagSEDE", color="#9467bd", marker="P", linestyle="--", linewidth=1.6, markersize=5.5),
    ("random", "direct"): dict(label="Random", color="#7f7f7f", marker="x", linestyle=":", linewidth=1.5, markersize=5.5),
}


def _load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row: Dict[str, Any] = dict(r)
            for k in [
                "budget_cap", "total_tokens", "live_calls", "cached_calls",
            ]:
                row[k] = int(r[k])
            for k in [
                "queries_per_block", "ami", "delta_ami", "nmi", "ari",
                "macro_b3", "delta_macro_b3", "micro_b3", "delta_micro_b3",
                "unq_delta_micro_b3", "unq_delta_macro_b3", "unq_net_posts",
                "net_fm_red", "net_fs_red", "tail_b3",
            ]:
                row[k] = float(r[k])
            rows.append(row)
    return rows


def plot_single_dataset(csv_path: Path, out_png: Path, title_prefix: str) -> None:
    rows = _load_rows(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=220)

    panels = [
        ("ami", "Full-Partition AMI", "AMI"),
        ("macro_b3", "Full-Partition Event Macro-B³", "Event Macro-B³"),
        ("unq_delta_micro_b3", "Unqueried-Post Propagation Gain (ΔMicro-B³)", "ΔMicro-B³ (Unqueried Posts)"),
    ]

    for ax, (metric, ptitle, ylabel) in zip(axes, panels):
        for (strat, branch), style in STYLE_MAP.items():
            sub = [r for r in rows if r["strategy"] == strat and r["branch"] == branch]
            if not sub:
                continue
            sub.sort(key=lambda r: r["total_tokens"])
            xs = [r["total_tokens"] / 1000.0 for r in sub]
            ys = [r[metric] for r in sub]
            ax.plot(xs, ys, **style)
        ax.set_title(ptitle, fontsize=11, fontweight="bold")
        ax.set_xlabel("Real Cumulative Tokens (×1k, Prompt + Completion)", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9.2,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(f"{title_prefix} — Real-Token Pareto Curve (Qwen/Qwen3-8B, non-thinking)", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {out_png}")


def plot_combined(csv_2012: Path, csv_2018: Path, out_png: Path) -> None:
    rows_2012 = _load_rows(csv_2012)
    rows_2018 = _load_rows(csv_2018)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), dpi=220)

    datasets = [
        ("Event2012 (English, M_02..M_05)", rows_2012),
        ("Event2018 (French, M_02..M_05)", rows_2018),
    ]
    panels = [
        ("ami", "Full-Partition AMI", "AMI"),
        ("macro_b3", "Event Macro-B³", "Event Macro-B³"),
        ("unq_delta_micro_b3", "Unqueried-Post Gain (ΔMicro-B³)", "ΔMicro-B³ (Unqueried)"),
    ]

    for row_i, (ds_label, rows) in enumerate(datasets):
        for col_i, (metric, ptitle, ylabel) in enumerate(panels):
            ax = axes[row_i, col_i]
            for (strat, branch), style in STYLE_MAP.items():
                sub = [r for r in rows if r["strategy"] == strat and r["branch"] == branch]
                if not sub:
                    continue
                sub.sort(key=lambda r: r["total_tokens"])
                xs = [r["total_tokens"] / 1000.0 for r in sub]
                ys = [r[metric] for r in sub]
                ax.plot(xs, ys, **style)
            if row_i == 0:
                ax.set_title(ptitle, fontsize=11, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(f"{ds_label}\n{ylabel}", fontsize=9.5, fontweight="bold")
            else:
                ax.set_ylabel(ylabel, fontsize=9.5)
            if row_i == 1:
                ax.set_xlabel("Real Cumulative Tokens (×1k, Prompt + Completion)", fontsize=9.5)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Gate A Multi-Budget Real-Token Pareto Curves (Qwen/Qwen3-8B, Bilingual Event2012 & Event2018)", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved combined figure: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-2012", type=Path, default=None)
    parser.add_argument("--csv-2018", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    if args.csv_2012 and args.csv_2012.exists():
        plot_single_dataset(args.csv_2012, args.out_dir / "budget_curve_Event2012_qwen3_8b.png", "Event2012 (English Dev)")
    if args.csv_2018 and args.csv_2018.exists():
        plot_single_dataset(args.csv_2018, args.out_dir / "budget_curve_Event2018_qwen3_8b.png", "Event2018 (French Dev)")
    if args.csv_2012 and args.csv_2018 and args.csv_2012.exists() and args.csv_2018.exists():
        plot_combined(args.csv_2012, args.csv_2018, args.out_dir / "budget_curve_bilingual_qwen3_8b.png")


if __name__ == "__main__":
    main()
