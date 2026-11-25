"""Plot block-by-block streaming trajectories and longitudinal cross-block comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_per_block_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row: Dict[str, Any] = dict(r)
            row["block_id"] = int(r["block_id"])
            row["n_posts"] = int(r["n_posts"])
            for k in ["init_ami", "init_macro_b3", "ami", "delta_ami", "macro_b3", "delta_macro_b3", "unq_delta_micro_b3", "unq_net_posts"]:
                row[k] = float(r[k])
            rows.append(row)
    return rows


def plot_streaming_trajectories(csv_path: Path, report_json: Path, out_png: Path, title: str) -> None:
    rows = _load_per_block_csv(csv_path)
    report = json.loads(report_json.read_text(encoding="utf-8"))

    blocks = sorted(set(r["block_id"] for r in rows))
    x = np.arange(len(blocks))
    xlabels = [f"M{b:02d}" for b in blocks]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=220)

    # Extract init curve
    init_ami = []
    init_mac = []
    for b in blocks:
        r0 = next(r for r in rows if r["block_id"] == b)
        init_ami.append(r0["init_ami"])
        init_mac.append(r0["init_macro_b3"])

    curves = [
        ("sug_meso_only", "direct", "SUG-Meso (Direct)", "#d62728", "o", "-", 2.2),
        ("sug_router", "repair", "SUG-Router (Repair)", "#1f77b4", "^", "-", 2.0),
        ("cequel_edge", "repair", "Cequel-Edge (Repair)", "#2ca02c", "D", "-.", 1.5),
        ("rep_div", "direct", "Rep-Div / RagSEDE", "#9467bd", "P", "--", 1.4),
    ]

    # Panel 1: Per-block AMI
    ax = axes[0]
    ax.plot(x, init_ami, label="Init 2D-SE (0 tok)", color="#555555", linestyle=":", linewidth=1.6, marker=".", markersize=5)
    for strat, branch, lab, col, mk, ls, lw in curves:
        ys = [next(r["ami"] for r in rows if r["block_id"] == b and r["strategy"] == strat and r["branch"] == branch) for b in blocks]
        ax.plot(x, ys, label=lab, color=col, marker=mk, linestyle=ls, linewidth=lw, markersize=5)
    ax.set_title("Per-Block Streaming AMI", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
    ax.set_ylabel("AMI")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: Per-block Event Macro-B3
    ax = axes[1]
    ax.plot(x, init_mac, label="Init 2D-SE (0 tok)", color="#555555", linestyle=":", linewidth=1.6, marker=".", markersize=5)
    for strat, branch, lab, col, mk, ls, lw in curves:
        ys = [next(r["macro_b3"] for r in rows if r["block_id"] == b and r["strategy"] == strat and r["branch"] == branch) for b in blocks]
        ax.plot(x, ys, label=lab, color=col, marker=mk, linestyle=ls, linewidth=lw, markersize=5)
    ax.set_title("Per-Block Event Macro-B³", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
    ax.set_ylabel("Event Macro-B³")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: Longitudinal Cross-Block Global AMI & Macro-B3
    ax = axes[2]
    lt = report.get("longitudinal_tracking", {})
    bar_keys = [
        ("init_2dse__no_link", "Init\nNoLink"),
        ("init_2dse__proto_only", "Init\nProto"),
        ("sug_meso_only_direct__proto_only", "SUG-Meso\nProto"),
        ("sug_meso_only_direct__sug_cross_llm", "SUG-Meso\n+CrossLLM"),
    ]
    present = [(k, lbl) for k, lbl in bar_keys if k in lt]
    if present:
        bx = np.arange(len(present))
        w = 0.36
        g_ami = [lt[k]["global_ami"] for k, _ in present]
        g_mac = [lt[k]["global_macro_b3"] for k, _ in present]
        b1 = ax.bar(bx - w / 2, g_ami, width=w, label="Global Stream AMI", color="#1f77b4", alpha=0.88)
        b2 = ax.bar(bx + w / 2, g_mac, width=w, label="Global Stream Macro-B³", color="#d62728", alpha=0.88)
        for bars in (b1, b2):
            for rect in bars:
                h = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, h + 0.008, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(bx)
        ax.set_xticklabels([lbl for _, lbl in present], fontsize=8.5)
        ax.set_ylim(0, min(1.08, max(g_ami + g_mac) + 0.20))
        ax.set_title("Multi-Week Cross-Block Stream Tracking", fontsize=11, fontweight="bold")
        ax.set_ylabel("Stream-Wide Score")
        ax.legend(fontsize=8.2, frameon=False, loc="upper left", ncol=2)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.35, -0.05))
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved streaming figure: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Held-Out Streaming SED Evaluation (Qwen/Qwen3-8B)")
    args = parser.parse_args()
    plot_streaming_trajectories(args.csv, args.json, args.out, args.title)


if __name__ == "__main__":
    main()
