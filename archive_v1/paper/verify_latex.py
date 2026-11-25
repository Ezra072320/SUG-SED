r"""Static LaTeX + BibTeX + Elsevier IP&M + Audited Number Consistency Checker.

Verifies the Elsevier `elsarticle` manuscript in `paper/` without requiring a
local `pdflatex` binary:
1. No placeholder markers (`TODO`, `TBD`, `FIXME`, `XXX`).
2. No non-ASCII Unicode bytes in `.tex` files (for `pdfLaTeX` safety).
3. Balanced braces `{}` / environments `\begin{}..\end{}` / inline `$` delimiters (no `$$`).
4. No unescaped `_` in prose outside math mode and citation/ref/texttt/path commands.
5. No unescaped `&` outside tabular/array/align/cases environments.
6. All `\label{}` unique; all `\ref{}` and `\eqref{}` resolved.
7. All `\cite`/`\citep`/`\citet` keys present in `paper/references.bib`.
8. All `\includegraphics` targets exist in `paper/`.
9. Elsevier `\begin{highlights}` has 3..5 bullets and each bullet has <= 85 visible chars.
10. Dynamic numeric audit verifying verbatim consistency between `.tex` tables/prose and
    Stage 1 (`stage1_diagnostics.json`), Dev budget curves, Stage 2A (`stage2a_report.json`),
    and Stage 2B (`anonymization_report.json`).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PAPER_DIR = Path(__file__).resolve().parent
ROOT_DIR = PAPER_DIR.parent

S1_JSON = ROOT_DIR / "results" / "ipm_audit" / "stage1_diagnostics.json"
S2A_JSON = ROOT_DIR / "results" / "ipm_experiments" / "stage2a_qwen3" / "stage2a_report.json"
S2B_JSON = ROOT_DIR / "results" / "ipm_experiments" / "stage2b_anonymization" / "anonymization_report.json"
CURVE_E12_JSON = ROOT_DIR / "results" / "budget_curves" / "Event2012_dev_qwen3_8b" / "Event2012_dev_qwen3_8b_curve.json"
CURVE_E18_JSON = ROOT_DIR / "results" / "budget_curves" / "Event2018_dev_qwen3_8b" / "Event2018_dev_qwen3_8b_curve.json"

MATH_ENVS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "aligned",
    "alignedat",
    "gather",
    "gather*",
    "eqnarray",
    "eqnarray*",
    "multline",
    "multline*",
    "cases",
}

AMP_ALLOWED_ENVS = MATH_ENVS | {
    "tabular",
    "tabular*",
    "tabularx",
    "longtable",
    "array",
    "split",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
}

SKIP_ARG_COMMANDS = {
    "texttt",
    "label",
    "ref",
    "eqref",
    "cite",
    "citep",
    "citet",
    "citeauthor",
    "citeyear",
    "citealp",
    "includegraphics",
    "input",
    "url",
    "href",
    "bibliography",
    "bibliographystyle",
}


def strip_comments(text: str) -> str:
    """Strip unescaped `%` comments per line while preserving newlines for line numbering."""
    out_lines: List[str] = []
    for line in text.splitlines():
        buf: List[str] = []
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            if ch == "\\":
                buf.append(line[i : i + 2])
                i += 2
                continue
            if ch == "%":
                break
            buf.append(ch)
            i += 1
        out_lines.append("".join(buf))
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def strip_latex_for_highlight_len(item: str) -> str:
    """Convert an Elsevier highlight LaTeX item to approximate visible plain text."""
    s = strip_comments(item).strip()
    s = re.sub(r"\\&", "&", s)
    s = re.sub(r"\\%", "%", s)
    s = re.sub(r"\\_", "_", s)
    s = re.sub(r"\\times\s*", "x", s)
    s = re.sub(r"\\,", " ", s)
    s = re.sub(r"~", " ", s)
    s = re.sub(r"--", "-", s)
    # Unwrap common formatting macros while preserving argument text
    for _ in range(4):
        s_new = re.sub(r"\\(?:textbf|emph|textit|texttt|text|mathrm|mathit|operatorname)\{([^{}]*)\}", r"\1", s)
        if s_new == s:
            break
        s = s_new
    s = s.replace("$", "")
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_bib_keys(bib_path: Path) -> Set[str]:
    bib_text = bib_path.read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text))


def scan_tex_tokens(stripped: str, rel: str, errors: List[str]) -> Tuple[int, List[str], List[Tuple[str, int, str]]]:
    """Scan braces, `$`, unescaped `_`, and unescaped `&` accurately with full escape/env tracking."""
    brace_depth = 0
    in_inline_math = False
    in_bracket_math = False
    dollar_count = 0
    env_stack: List[str] = []
    line_no = 1
    i = 0
    n = len(stripped)

    while i < n:
        ch = stripped[i]
        if ch == "\n":
            line_no += 1
            i += 1
            continue

        # Handle backslash escape sequences
        if ch == "\\":
            if i + 1 < n:
                nxt = stripped[i + 1]
                if nxt == "[":
                    in_bracket_math = True
                    i += 2
                    continue
                if nxt == "]":
                    in_bracket_math = False
                    i += 2
                    continue
                if nxt in {"\\", "{", "}", "$", "_", "&", "%", "#", "~", "^"}:
                    i += 2
                    continue
            # Check command name
            m_cmd = re.match(r"\\([a-zA-Z]+)\*?", stripped[i:])
            if m_cmd:
                cmd = m_cmd.group(1)
                j = i + len(m_cmd.group(0))
                if cmd in {"begin", "end"}:
                    while j < n and stripped[j] in " \t":
                        j += 1
                    if j < n and stripped[j] == "{":
                        end_b = stripped.find("}", j + 1)
                        if end_b != -1:
                            env_name = stripped[j + 1 : end_b].strip()
                            if cmd == "begin":
                                env_stack.append(env_name)
                            else:
                                if not env_stack or env_stack[-1] != env_name:
                                    errors.append(
                                        f"[{rel}:{line_no}] Mismatched \\end{{{env_name}}} (stack={env_stack})"
                                    )
                                else:
                                    env_stack.pop()
                            i = end_b + 1
                            continue
                if cmd in SKIP_ARG_COMMANDS:
                    # Skip optional [...] args and up to 2 balanced {...} args
                    k = j
                    while k < n and stripped[k] in " \t\n":
                        if stripped[k] == "\n":
                            line_no += 1
                        k += 1
                    while k < n and stripped[k] == "[":
                        k += 1
                        while k < n and stripped[k] != "]":
                            if stripped[k] == "\n":
                                line_no += 1
                            elif stripped[k] == "\\":
                                k += 1
                            k += 1
                        if k < n:
                            k += 1
                        while k < n and stripped[k] in " \t":
                            k += 1
                    args_to_skip = 2 if cmd == "href" else 1
                    for _ in range(args_to_skip):
                        while k < n and stripped[k] in " \t":
                            k += 1
                        if k < n and stripped[k] == "{":
                            d = 1
                            k += 1
                            while k < n and d > 0:
                                c2 = stripped[k]
                                if c2 == "\n":
                                    line_no += 1
                                elif c2 == "\\":
                                    k += 2
                                    continue
                                elif c2 == "{":
                                    d += 1
                                elif c2 == "}":
                                    d -= 1
                                k += 1
                    i = k
                    continue
                i = j
                continue
            i += 1
            continue

        if ch == "{":
            brace_depth += 1
            i += 1
            continue
        if ch == "}":
            brace_depth -= 1
            if brace_depth < 0:
                errors.append(f"[{rel}:{line_no}] Unbalanced closing brace '}}' (depth < 0)")
                brace_depth = 0
            i += 1
            continue
        if ch == "$":
            if i + 1 < n and stripped[i + 1] == "$":
                errors.append(f"[{rel}:{line_no}] Double dollar '$$' found; use equation/\\[...\\]")
                i += 2
                continue
            dollar_count += 1
            in_inline_math = not in_inline_math
            i += 1
            continue

        in_math = in_inline_math or in_bracket_math or any(e in MATH_ENVS for e in env_stack)
        if ch == "_" and not in_math:
            errors.append(f"[{rel}:{line_no}] Unescaped '_' outside math/verbatim command")
        if ch == "&" and not any(e in AMP_ALLOWED_ENVS for e in env_stack):
            errors.append(f"[{rel}:{line_no}] Unescaped '&' outside tabular/align environment")

        i += 1

    if brace_depth != 0:
        errors.append(f"[{rel}] Unbalanced curly braces at EOF (net depth={brace_depth})")
    if dollar_count % 2 != 0:
        errors.append(f"[{rel}] Odd number of '$' delimiters ({dollar_count})")
    if env_stack:
        errors.append(f"[{rel}] Unclosed environments at EOF: {env_stack}")

    return dollar_count, env_stack, []


def _val(x) -> float:
    return float(x["mean"]) if isinstance(x, dict) and "mean" in x else float(x)


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals)


def run_numeric_audit(
    main_tex: str,
    intro_tex: str,
    exp_tex: str,
) -> Tuple[List[str], List[str]]:
    """Verify verbatim numeric consistency against audited Stage 1, 2A, 2B, and curve JSONs."""
    s1 = json.loads(S1_JSON.read_text(encoding="utf-8"))
    s2a = json.loads(S2A_JSON.read_text(encoding="utf-8"))
    s2b = json.loads(S2B_JSON.read_text(encoding="utf-8"))
    c12 = json.loads(CURVE_E12_JSON.read_text(encoding="utf-8"))
    c18 = json.loads(CURVE_E18_JSON.read_text(encoding="utf-8"))

    checks: List[Tuple[str, str, str]] = []  # (label, text_blob_name, expected_substr)

    # --- 1. Stage 1 D4 Pathology & Intro/Experiments ---
    e12_d4_held = s1["Event2012"]["D4_pathology"]["heldout"]
    e12_d4_dev = s1["Event2012"]["D4_pathology"]["dev"]
    checks.append(("S1 E12 held FS/FM ratio (1dp main)", "main", f"{e12_d4_held['overall_fs_fm_ratio']:.1f}"))
    checks.append(("S1 E12 held FS/FM ratio (1dp intro)", "intro", f"{e12_d4_held['overall_fs_fm_ratio']:.1f}"))
    checks.append(("S1 E12 dev FS/FM ratio", "intro", f"{e12_d4_dev['overall_fs_fm_ratio']:.1f}"))

    # --- 2. Stage 1 D3 Zero-LLM Baselines (Table tab:zero_llm_d3) ---
    for ds_tag, ds_name in [("E12", "Event2012"), ("E18", "Event2018")]:
        d3 = s1[ds_name]["D3_zero_llm_baselines"]
        for key in [
            "vanilla_2dse_no_refine",
            "semantic_only_2dse_alpha0",
            "dev_tuned_2dse",
            "louvain",
            "agglom_cosine",
            "heur_meso_full",
        ]:
            ami_str = f"{_val(d3[key]['ami']) * 100:.2f}"
            mac_str = f"{_val(d3[key]['macro_b3']) * 100:.2f}"
            checks.append((f"S1 D3 {ds_tag} {key} AMI", "exp", ami_str))
            checks.append((f"S1 D3 {ds_tag} {key} MacB3", "exp", mac_str))

    # --- 3. Dev 80k Budget Curve Anchors (text + Stage 2A Table) ---
    for ds_tag, curve_obj in [("E12", c12), ("E18", c18)]:
        blocks_80k = curve_obj["tiers"]["80000"]["strategies"]["sug_meso_only"]["blocks"]
        mean_ami_80k = _mean([b["branch1_direct"]["ami"] for b in blocks_80k.values()]) * 100.0
        checks.append((f"Dev 80k {ds_tag} sug_meso_only AMI", "exp", f"{mean_ami_80k:.2f}"))

    # --- 4. Stage 2A Dev Calibration & Main 27-Block Causal Suite ---
    for ds_tag, ds_name in [("E12", "Event2012"), ("E18", "Event2018")]:
        wb = s2a[ds_name]["within_block"]
        for key in [
            "dev_sug_meso_action",
            "dev_sug_meso_extract",
            "sug_meso_causal_fixed_action",
            "sug_meso_causal_fixed_extract",
            "sug_meso_causal_carryover_action",
            "sug_meso_causal_sliding_action",
            "random_mean3",
            "random_meso_action_mean3",
            "eq6_no_conductance_action",
            "eq5_no_leverage_action",
            "causal_merge_only_action",
            "causal_seam_only_action",
            "causal_fiedler_seam_action",
        ]:
            d = wb[key]["direct"]
            checks.append((f"S2A {ds_tag} {key} AMI", "exp", f"{_val(d['ami']) * 100:.2f}"))
            checks.append((f"S2A {ds_tag} {key} MacB3", "exp", f"{_val(d['macro_b3']) * 100:.2f}"))

        # Tail B3 and UnqNet on primary causal models + no_cond
        for key in ["sug_meso_causal_fixed_action", "eq6_no_conductance_action"]:
            d = wb[key]["direct"]
            checks.append((f"S2A {ds_tag} {key} TailB3", "exp", f"{_val(d['tail_b3']) * 100:.2f}"))
            checks.append((f"S2A {ds_tag} {key} UnqNet", "exp", f"{_val(d['unq_net_posts']):.1f}"))

        # Shown % on primary fixed action
        show_pct = s2a[ds_name]["within_block"]["sug_meso_causal_fixed_action"]["counts"]["all_shown_pct"]
        checks.append((f"S2A {ds_tag} fixed_action shown_pct", "exp", f"{show_pct:.2f}"))

    # Headline checks in Abstract (`main.tex`, 0-1 AMI scale and pp gain)
    e12_fix = s2a["Event2012"]["within_block"]["sug_meso_causal_fixed_action"]["direct"]
    e18_fix = s2a["Event2018"]["within_block"]["sug_meso_causal_fixed_action"]["direct"]
    checks.append(("Abstract E12 AMI", "main", f"{_val(e12_fix['ami']):.4f}"))
    checks.append(("Abstract E12 dAMI", "main", f"{_val(e12_fix['delta_ami']) * 100:.2f}"))
    checks.append(("Abstract E18 AMI", "main", f"{_val(e18_fix['ami']):.4f}"))
    checks.append(("Abstract E18 dAMI", "main", f"{_val(e18_fix['delta_ami']) * 100:.2f}"))

    # --- 5. Stage 2A Cross-Day SUG-CrossLLM-v2 (`heldout_action_stream`) ---
    for ds_tag, ds_name in [("E12", "Event2012"), ("E18", "Event2018")]:
        cd = s2a[ds_name]["cross_day"]["heldout_action_stream"]
        for key in ["no_link", "proto_0.80", "proto_0.85", "sug_cross_v2_extract", "sug_cross_v2_action", "oracle_065_085"]:
            checks.append((f"CrossDay {ds_tag} {key} G-AMI", "exp", f"{cd[key]['ami'] * 100:.2f}"))
            checks.append((f"CrossDay {ds_tag} {key} G-MicB3", "exp", f"{cd[key]['micro_b3'] * 100:.2f}"))

    # --- 6. Stage 2B Anonymization Stress Test ---
    for ds_tag, ds_name in [("E12", "Event2012"), ("E18", "Event2018")]:
        conds = s2b[ds_name]["conditions"]
        rets = s2b[ds_name]["statistics"]["retentions"]
        for ckey in ["anon_pseudo_action", "anon_blind_action", "anon_pseudo_extract"]:
            checks.append((f"S2B {ds_tag} {ckey} AMI", "exp", f"{_val(conds[ckey]['direct']['ami']) * 100:.2f}"))
            checks.append((f"S2B {ds_tag} {ckey} MacB3", "exp", f"{_val(conds[ckey]['direct']['macro_b3']) * 100:.2f}"))
            checks.append((f"S2B {ds_tag} {ckey} AMI ret", "exp", f"{rets[ckey]['ami']['retention_ratio'] * 100:.1f}"))
        pa = conds["anon_pseudo_action"]
        checks.append((f"S2B {ds_tag} pseudo_action mask rate", "exp", f"{pa['mask_audit']['entity_mask_rate'] * 100:.2f}"))
        checks.append((f"S2B {ds_tag} pseudo_action agree", "exp", f"{pa['decision_agreement']['overall_rate'] * 100:.1f}"))

    e12_ret = s2b["Event2012"]["statistics"]["retentions"]["anon_pseudo_action"]["ami"]["retention_ratio"] * 100
    e18_ret = s2b["Event2018"]["statistics"]["retentions"]["anon_pseudo_action"]["ami"]["retention_ratio"] * 100
    checks.append(("Abstract S2B E12 retention", "main", f"{e12_ret:.1f}"))
    checks.append(("Abstract S2B E18 retention", "main", f"{e18_ret:.1f}"))

    blobs = {"main": main_tex, "intro": intro_tex, "exp": exp_tex}
    failures: List[str] = []
    passed: List[str] = []
    for label, blob_key, substr in checks:
        if substr in blobs[blob_key]:
            passed.append(f"{label} ({substr})")
        else:
            failures.append(f"{label}: expected '{substr}' not found in {blob_key}")
    return passed, failures


def main() -> None:
    tex_files = [PAPER_DIR / "main.tex"] + sorted((PAPER_DIR / "sections").glob("*.tex"))
    bib_path = PAPER_DIR / "references.bib"
    bib_keys = parse_bib_keys(bib_path)

    errors: List[str] = []
    warnings: List[str] = []
    all_labels: Dict[str, str] = {}
    all_refs: List[Tuple[str, str]] = []
    all_cites: List[Tuple[str, str]] = []
    all_imgs: List[Tuple[str, str]] = []

    for tf in tex_files:
        rel = str(tf.relative_to(PAPER_DIR)).replace("\\", "/")
        raw = tf.read_text(encoding="utf-8")
        stripped = strip_comments(raw)

        # 1. Non-ASCII byte check (line-by-line on full file; comments warn, non-comment error)
        raw_lines = raw.splitlines()
        stripped_lines = stripped.splitlines()
        for ln_idx, (r_line, s_line) in enumerate(zip(raw_lines, stripped_lines), start=1):
            bad_s = sorted({f"U+{ord(c):04X}({c})" for c in s_line if ord(c) > 127})
            if bad_s:
                errors.append(f"[{rel}:{ln_idx}] Non-ASCII char(s) in LaTeX body: {bad_s}")
            elif any(ord(c) > 127 for c in r_line):
                bad_r = sorted({f"U+{ord(c):04X}" for c in r_line if ord(c) > 127})
                warnings.append(f"[{rel}:{ln_idx}] Non-ASCII char(s) inside % comment: {bad_r}")

        # 2. Placeholder check
        for m in re.finditer(r"\b(TODO|TBD|FIXME|XXX)\b", stripped):
            ln = stripped[: m.start()].count("\n") + 1
            errors.append(f"[{rel}:{ln}] Placeholder marker '{m.group(1)}'")

        # 3 & 4 & 5 & 6. Syntax scan (braces, envs, $, _, &)
        scan_tex_tokens(stripped, rel, errors)

        # 7. Labels, refs, cites, images
        for m in re.finditer(r"\\label\{([^}]+)\}", stripped):
            lbl = m.group(1).strip()
            if lbl in all_labels:
                errors.append(f"[{rel}] Duplicate \\label{{{lbl}}} (already in {all_labels[lbl]})")
            all_labels[lbl] = rel

        for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", stripped):
            for key in m.group(1).split(","):
                k = key.strip()
                if k:
                    all_refs.append((rel, k))

        for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\s*\[[^\]]*\])*\s*\{([^}]+)\}", stripped):
            for key in m.group(1).split(","):
                k = key.strip()
                if k:
                    all_cites.append((rel, k))

        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", stripped):
            all_imgs.append((rel, m.group(1).strip()))

    # Cross-reference resolution
    for rel, rf in all_refs:
        if rf not in all_labels:
            errors.append(f"[{rel}] Undefined \\ref{{{rf}}}")

    cited_keys: Set[str] = set()
    for rel, ck in all_cites:
        cited_keys.add(ck)
        if ck not in bib_keys:
            errors.append(f"[{rel}] Missing BibTeX key \\cite*{{{ck}}}")

    # Image existence check
    for rel, img in all_imgs:
        p = PAPER_DIR / img
        found = p.exists() or any((PAPER_DIR / f"{img}{ext}").exists() for ext in [".pdf", ".png", ".jpg"])
        if not found:
            errors.append(f"[{rel}] Missing graphic file: {img}")

    # 8. Elsevier Highlights check in main.tex
    main_tex = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    m_hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", main_tex, re.S)
    highlight_info: List[Tuple[int, str]] = []
    if not m_hl:
        errors.append("[main.tex] Missing \\begin{highlights}...\\end{highlights} block")
    else:
        raw_items = re.split(r"\\item\b", m_hl.group(1))[1:]
        if not (3 <= len(raw_items) <= 5):
            errors.append(f"[main.tex] Expected 3..5 highlight bullets, found {len(raw_items)}")
        for idx, it in enumerate(raw_items, start=1):
            vis = strip_latex_for_highlight_len(it)
            highlight_info.append((len(vis), vis))
            if len(vis) > 85:
                errors.append(f"[main.tex] Highlight #{idx} exceeds 85 visible chars ({len(vis)}): '{vis}'")

    # 9. Audited Numeric Audit
    intro_tex = (PAPER_DIR / "sections" / "01_intro.tex").read_text(encoding="utf-8")
    exp_tex = (PAPER_DIR / "sections" / "05_experiments.tex").read_text(encoding="utf-8")
    num_passed, num_failures = run_numeric_audit(main_tex, intro_tex, exp_tex)
    for nf in num_failures:
        errors.append(f"[NUMERIC_AUDIT] {nf}")

    # Summary Output
    print(
        f"Checked {len(tex_files)} .tex files | {len(bib_keys)} BibTeX entries ({len(cited_keys)} cited) | "
        f"{len(all_labels)} labels | {len(all_refs)} refs | {len(all_cites)} cites | "
        f"{len(all_imgs)} graphics | {len(num_passed)}/{len(num_passed) + len(num_failures)} numeric checks"
    )
    if highlight_info:
        print("Elsevier highlights lengths (limit <= 85 chars):")
        for idx, (hlen, htxt) in enumerate(highlight_info, start=1):
            print(f"  [{idx}] ({hlen:2d} chars) {htxt}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  WARN: {w}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ERR:  {e}")
        sys.exit(1)

    print(f"\nALL CHECKS PASSED cleanly (0 errors, {len(warnings)} warnings).")


if __name__ == "__main__":
    main()
