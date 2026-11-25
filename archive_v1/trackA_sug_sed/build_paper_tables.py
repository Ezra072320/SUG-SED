"""Compile publication-ready Markdown and LaTeX tables from audited Qwen3-8B results.

Reads:
  - results/streaming/Event2012_heldout_qwen3_8b/streaming_report.json
  - results/streaming/Event2018_heldout_qwen3_8b/streaming_report.json
  - results/budget_curves/Event2012_dev_qwen3_8b/Event2012_dev_qwen3_8b_curve.csv
  - results/budget_curves/Event2018_dev_qwen3_8b/Event2018_dev_qwen3_8b_curve.csv

Writes:
  - results/streaming/HeldOut_Bilingual_Qwen3_8B_Report.md
  - results/streaming/paper_tables_qwen3_8b.tex
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}"


def _fmt_dpct(v: float) -> str:
    return f"{v * 100:+.2f}"


def _load_report(dataset: str) -> Dict[str, Any]:
    p = ROOT / "results" / "streaming" / f"{dataset}_heldout_qwen3_8b" / "streaming_report.json"
    return json.loads(p.read_text(encoding="utf-8"))


def build_tables() -> None:
    r12 = _load_report("Event2012")
    r18 = _load_report("Event2018")

    order: List[Tuple[str, str]] = [
        ("random_direct", "Random (Direct)"),
        ("rep_div_direct", "Rep-Div / RagSEDE (Direct)"),
        ("cequel_edge_direct", "Cequel-Edge (Direct)"),
        ("cequel_edge_repair", "Cequel-Edge (Repair)"),
        ("sug_router_direct", "SUG-Router (Direct)"),
        ("sug_router_repair", "SUG-Router (Repair)"),
        ("sug_meso_only_direct", "SUG-Meso (Direct)"),
        ("sug_meso_only_repair", "SUG-Meso (Repair)"),
    ]

    md_lines: List[str] = [
        "# Track A（SUG-SED）双语保留测试集完整裁决报告（`Qwen/Qwen3-8B`）",
        "",
        "- **协议快照**：[protocol_v1.json](../../trackA_sug_sed/protocol_v1.json)（在触碰任何 `M_06+` 保留块前完成开发块校准与 SHA-256 冻结）",
        "- **评测模型**：`Qwen/Qwen3-8B`（轻量 8B 稠密开源模型，硅基流动官方 API，`enable_thinking=False`，`temperature=0.0`，`json_object`）",
        "- **数据划分（严格零泄露）**：",
        f"  - `Event2012`（英语）：`M_00..M_05` 仅作历史记忆预热（0 Token）；**保留测试块 `M_06..M_21`（{len(r12['eval_blocks'])} 块，{r12['longitudinal_tracking']['init_2dse__no_link']['n_total_posts']:,} 条推文）**；",
        f"  - `Event2018`（法语）：`M_00..M_05` 仅作历史记忆预热（0 Token）；**保留测试块 `M_06..M_16`（{len(r18['eval_blocks'])} 块，{r18['longitudinal_tracking']['init_2dse__no_link']['n_total_posts']:,} 条推文）**。",
        "- **预算口径**：全流程严格按真实输入+输出 Token（`prompt_tokens + completion_tokens`）累计封顶（块内 `10k` Token/块，跨块 `3k` Token/块），缓存命中调用均按实际 Token 计费。",
        "- **图表索引**：[Event2012 轨迹图](../figures/streaming_Event2012_heldout_qwen3_8b.png) · [Event2018 轨迹图](../figures/streaming_Event2018_heldout_qwen3_8b.png) · [双语多档帕累托图](../figures/budget_curve_bilingual_qwen3_8b.png)",
        "",
        "---",
        "",
    ]

    for ds_name, rep in [("Event2012（英语 · M_06..M_21，16 块）", r12), ("Event2018（法语 · M_06..M_16，11 块）", r18)]:
        im = rep["init_eval_mean"]
        md_lines.extend([
            f"## {ds_name}：逐块流式检测对照（初始 2D-SE: AMI={im['ami']:.4f}, NMI={im['nmi']:.4f}, ARI={im['ari']:.4f}, 宏 B³={im['macro_b3']:.4f}, 微 B³={im['micro_b3']:.4f}, 长尾 B³={im['tail_b3']:.4f}）",
            "",
            r"| 方法与分支 | 真实总 Token | 均查询/块 | AMI ($\Delta$) | ARI | 宏 $B^3$ ($\Delta$) | 长尾 $B^3$ | 未查询 $\Delta$微 $B^3$ | 未查询净改善帖/块 | 日块胜率 (AMI / 宏 $B^3$) | 净减 FS/块 |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
        ])
        mc = rep["mean_comparison"]
        for k, label in order:
            if k not in mc:
                continue
            m = mc[k]
            md_lines.append(
                f"| {label} | {m['total_tokens_all_blocks']:,.0f} | {m['queries']:.1f} | "
                f"{m['ami']:.4f} ({m['delta_ami']:+.4f}) | {m['ari']:.4f} | "
                f"{m['macro_b3']:.4f} ({m['delta_macro_b3']:+.4f}) | {m['tail_b3']:.4f} | "
                f"{m['unq_delta_micro_b3']:+.4f} | {m['unq_net_posts']:+.1f} | "
                f"{m['blocks_improved_ami']}/{m['n_eval_blocks']} · {m['blocks_improved_macb3']}/{m['n_eval_blocks']} | "
                f"{m['net_fs_red']:+,.0f} |"
            )
        md_lines.append("")

        md_lines.extend([
            f"### {ds_name}：多周跨块纵向事件追踪（全量保留流串联评测）",
            "",
            "| 分区来源与跨块链接模式 | 预测簇数 $K$ | Global AMI | Global NMI | Global 宏 $B^3$ | Global 微 $B^3$ | 自动链接 / LLM 查询 (是/否) | 跨块实耗 Token |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        lt_order = [
            "init_2dse__no_link",
            "init_2dse__proto_only",
            "rep_div_direct__proto_only",
            "cequel_edge_repair__proto_only",
            "sug_meso_only_direct__no_link",
            "sug_meso_only_direct__proto_only",
            "sug_meso_only_direct__sug_cross_llm",
            "sug_router_repair__proto_only",
            "sug_router_repair__sug_cross_llm",
        ]
        lt = rep["longitudinal_tracking"]
        for lk in lt_order:
            if lk not in lt:
                continue
            v = lt[lk]
            st = v["link_stats"]
            md_lines.append(
                f"| `{lk}` | {v['n_pred_events']} | {v['global_ami']:.4f} | {v['global_nmi']:.4f} | "
                f"{v['global_macro_b3']:.4f} | {v['global_micro_b3']:.4f} | "
                f"{st['auto_linked']} / {st['llm_queried']} ({st['llm_linked']}/{st['llm_rejected']}) | "
                f"{st['cross_tokens']:,} |"
            )
        md_lines.extend(["", "---", ""])

    # Build clean LaTeX Table 1 (Main Held-Out Bilingual Comparison) & Table 2 (Longitudinal Tracking)
    tex_lines = [
        "% Auto-generated by trackA_sug_sed/build_paper_tables.py from audited Qwen/Qwen3-8B held-out JSONs",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Held-out streaming SED results on unseen chronological blocks of Event2012 ($M_{06}$--$M_{21}$, 16 blocks, 32,695 tweets) and Event2018 ($M_{06}$--$M_{16}$, 11 blocks, 33,161 tweets) under identical strict real-token budgets ($10\\text{k}$ prompt+completion tokens/block) using \\texttt{Qwen/Qwen3-8B} (non-thinking). All values are percentages ($\\times 100$) except Q/blk and UnqNet.}",
        "\\label{tab:heldout_main}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l|ccccc|ccccc}",
        "\\toprule",
        " & \\multicolumn{5}{c|}{\\textbf{Event2012 Held-Out ($M_{06}$--$M_{21}$, 16 blocks)}} & \\multicolumn{5}{c}{\\textbf{Event2018 Held-Out ($M_{06}$--$M_{16}$, 11 blocks)}} \\\\",
        "\\textbf{Method (Branch)} & AMI ($\\Delta$) & Mac-$B^3$ ($\\Delta$) & Unq $\\Delta$Mic & UnqNet & Win & AMI ($\\Delta$) & Mac-$B^3$ ($\\Delta$) & Unq $\\Delta$Mic & UnqNet & Win \\\\",
        "\\midrule",
        f"Init 2D-SE (0 tok) & {_fmt_pct(r12['init_eval_mean']['ami'])} (---) & {_fmt_pct(r12['init_eval_mean']['macro_b3'])} (---) & --- & --- & --- & {_fmt_pct(r18['init_eval_mean']['ami'])} (---) & {_fmt_pct(r18['init_eval_mean']['macro_b3'])} (---) & --- & --- & --- \\\\",
        "\\midrule",
    ]
    for k, label in order:
        m12 = r12["mean_comparison"][k]
        m18 = r18["mean_comparison"][k]
        tex_lines.append(
            f"{label} & {_fmt_pct(m12['ami'])} ({_fmt_dpct(m12['delta_ami'])}) & "
            f"{_fmt_pct(m12['macro_b3'])} ({_fmt_dpct(m12['delta_macro_b3'])}) & "
            f"{_fmt_dpct(m12['unq_delta_micro_b3'])} & {m12['unq_net_posts']:+.1f} & "
            f"{m12['blocks_improved_ami']}/{m12['n_eval_blocks']} & "
            f"{_fmt_pct(m18['ami'])} ({_fmt_dpct(m18['delta_ami'])}) & "
            f"{_fmt_pct(m18['macro_b3'])} ({_fmt_dpct(m18['delta_macro_b3'])}) & "
            f"{_fmt_dpct(m18['unq_delta_micro_b3'])} & {m18['unq_net_posts']:+.1f} & "
            f"{m18['blocks_improved_ami']}/{m18['n_eval_blocks']} \\\\"
        )
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Multi-week longitudinal cross-block event tracking evaluated across all concatenated held-out posts (Event2012: 32,695 posts; Event2018: 33,161 posts).}",
        "\\label{tab:longitudinal}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l|ccc|ccc}",
        "\\toprule",
        " & \\multicolumn{3}{c|}{\\textbf{Event2012 ($M_{06}$--$M_{21}$)}} & \\multicolumn{3}{c}{\\textbf{Event2018 ($M_{06}$--$M_{16}$)}} \\\\",
        "\\textbf{Pipeline + Cross-Link} & G-AMI & G-Mac$B^3$ & G-Mic$B^3$ & G-AMI & G-Mac$B^3$ & G-Mic$B^3$ \\\\",
        "\\midrule",
    ])
    for lk, nice in [
        ("init_2dse__no_link", "Init 2D-SE + NoLink"),
        ("init_2dse__proto_only", "Init 2D-SE + ProtoOnly"),
        ("cequel_edge_repair__proto_only", "Cequel-Edge + ProtoOnly"),
        ("sug_meso_only_direct__proto_only", "SUG-Meso (Direct) + ProtoOnly"),
        ("sug_meso_only_direct__sug_cross_llm", "SUG-Meso (Direct) + CrossLLM"),
        ("sug_router_repair__sug_cross_llm", "SUG-Router (Repair) + CrossLLM"),
    ]:
        v12 = r12["longitudinal_tracking"][lk]
        v18 = r18["longitudinal_tracking"][lk]
        tex_lines.append(
            f"{nice} & {_fmt_pct(v12['global_ami'])} & {_fmt_pct(v12['global_macro_b3'])} & {_fmt_pct(v12['global_micro_b3'])} & "
            f"{_fmt_pct(v18['global_ami'])} & {_fmt_pct(v18['global_macro_b3'])} & {_fmt_pct(v18['global_micro_b3'])} \\\\"
        )
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
    ])

    md_out = ROOT / "results" / "streaming" / "HeldOut_Bilingual_Qwen3_8B_Report.md"
    tex_out = ROOT / "results" / "streaming" / "paper_tables_qwen3_8b.tex"
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    tex_out.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"[Tables] Wrote Markdown report : {md_out}")
    print(f"[Tables] Wrote LaTeX tables    : {tex_out}")


if __name__ == "__main__":
    build_tables()
