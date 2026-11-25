"""Static LaTeX + BibTeX + Figure + Audited Track B Number Consistency Checker."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = Path(__file__).resolve().parent


def _strip_comments(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        res = []
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            res.append(line[i])
            i += 1
        out_lines.append("".join(res))
    return "\n".join(out_lines)


def check_bib_keys(bib_path: Path) -> Set[str]:
    text = bib_path.read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text))


def main() -> int:
    errors: List[str] = []
    bib_keys = check_bib_keys(PAPER_DIR / "references.bib")
    if not bib_keys:
        errors.append("No BibTeX keys parsed from references.bib")

    tex_files = [PAPER_DIR / "main.tex"] + sorted((PAPER_DIR / "sections").glob("*.tex"))
    all_labels: Set[str] = set()
    all_refs: List[tuple] = []
    all_cites: List[tuple] = []

    verbatim_like_cmds = ("\\texttt{", "\\label{", "\\ref{", "\\eqref{", "\\cite{", "\\includegraphics", "\\input{")
    math_envs = {"equation", "equation*", "align", "align*", "gather", "gather*"}
    verbatim_envs = {"CCSXML", "verbatim", "lstlisting"}

    for tf in tex_files:
        raw = tf.read_text(encoding="utf-8")
        clean = _strip_comments(raw)
        rel = tf.relative_to(PAPER_DIR).as_posix()

        for pat in ["TODO", "TBD", "FIXME", "XXX"]:
            if re.search(rf"\b{pat}\b", clean):
                errors.append(f"[{rel}] Found placeholder token: {pat}")

        # Balanced braces ignoring \{ and \}
        depth = 0
        i = 0
        while i < len(clean):
            ch = clean[i]
            if ch == "\\" and i + 1 < len(clean) and clean[i + 1] in "{}$%&_#":
                i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    errors.append(f"[{rel}] Unmatched closing brace '}}' at char {i}")
                    break
            i += 1
        if depth != 0:
            errors.append(f"[{rel}] Unclosed braces remaining: depth={depth}")

        # Balanced environments
        env_stack: List[str] = []
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", clean):
            kind, ename = m.group(1), m.group(2)
            if kind == "begin":
                env_stack.append(ename)
            else:
                if not env_stack or env_stack[-1] != ename:
                    errors.append(f"[{rel}] Environment mismatch: \\end{{{ename}}} (stack={env_stack[-3:]})")
                    break
                env_stack.pop()
        if env_stack:
            errors.append(f"[{rel}] Unclosed environments: {env_stack}")

        # Balanced $ delimiters
        dollar_cnt = 0
        j = 0
        while j < len(clean):
            if clean[j] == "\\" and j + 1 < len(clean):
                j += 2
                continue
            if clean[j] == "$":
                if j + 1 < len(clean) and clean[j + 1] == "$":
                    errors.append(f"[{rel}] Found raw '$$' (prefer equation environment)")
                    j += 2
                    continue
                dollar_cnt += 1
            j += 1
        if dollar_cnt % 2 != 0:
            errors.append(f"[{rel}] Odd number of inline '$' math delimiters ({dollar_cnt})")

        for m in re.finditer(r"\\label\{([^}]+)\}", clean):
            all_labels.add(m.group(1).strip())
        for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", clean):
            all_refs.append((rel, m.group(1).strip()))
        for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", clean):
            for ck in m.group(1).split(","):
                all_cites.append((rel, ck.strip()))
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean):
            img_rel = m.group(1).strip()
            if not (PAPER_DIR / img_rel).exists():
                errors.append(f"[{rel}] Missing figure file: {img_rel}")

        # Check unescaped underscores outside math/verbatim commands/envs
        env_cur: List[str] = []
        in_math = False
        for ln_no, line in enumerate(clean.splitlines(), 1):
            m_b = re.search(r"\\begin\{([^}]+)\}", line)
            m_e = re.search(r"\\end\{([^}]+)\}", line)
            in_skip_env = any(e in math_envs or e in verbatim_envs for e in env_cur)
            if not in_skip_env and m_b is None and m_e is None:
                k = 0
                while k < len(line):
                    if line[k] == "\\":
                        matched_cmd = False
                        for cmd in verbatim_like_cmds:
                            if line.startswith(cmd, k):
                                end_b = line.find("}", k + len(cmd))
                                if end_b != -1:
                                    k = end_b + 1
                                    matched_cmd = True
                                    break
                        if matched_cmd:
                            continue
                        k += 2
                        continue
                    if line[k] == "$":
                        in_math = not in_math
                        k += 1
                        continue
                    if line[k] == "_" and not in_math:
                        errors.append(f"[{rel}:{ln_no}] Unescaped '_' outside math/command: {line.strip()[:90]}")
                        break
                    k += 1
            for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
                if m.group(1) == "begin":
                    env_cur.append(m.group(2))
                elif env_cur and env_cur[-1] == m.group(2):
                    env_cur.pop()

    for rel, rf in all_refs:
        if rf not in all_labels:
            errors.append(f"[{rel}] Undefined \\ref{{{rf}}}")
    for rel, ck in all_cites:
        if ck not in bib_keys:
            errors.append(f"[{rel}] Missing BibTeX key \\cite{{{ck}}}")

    # Spot-check audited numbers across ALL 5 Track B JSON files
    sec3 = (PAPER_DIR / "sections" / "03_causal_diagnosis.tex").read_text(encoding="utf-8")
    sec5 = (PAPER_DIR / "sections" / "05_experiments.tex").read_text(encoding="utf-8")

    t1 = json.loads((ROOT / "results" / "trackB_gateB" / "tier1_audit.json").read_text(encoding="utf-8"))
    t2 = json.loads((ROOT / "results" / "trackB_gateB" / "ned_dev_qwen3_vl_8b" / "tier2_vlm_audit.json").read_text(encoding="utf-8"))
    val = json.loads((ROOT / "results" / "trackB_phase2" / "ned_validation_results.json").read_text(encoding="utf-8"))
    hold = json.loads((ROOT / "results" / "trackB_phase2" / "ned_holdout_results.json").read_text(encoding="utf-8"))
    diag = json.loads((ROOT / "results" / "trackB_phase2" / "ned_holdout_stress_and_diag.json").read_text(encoding="utf-8"))

    t1_lam = t1["tier1a_paired_classification"]["lambda_0.75"]
    t1b = t1["tier1b_unsupervised_mv_2dse"]
    t2m = t2["vlm_comparison"]

    checks_sec3 = [
        ("Tier1A Naive Orig Acc", f"{t1_lam['naive_visual']['novel_non_dup']['orig_acc']:.4f}"),
        ("Tier1A CNTD-Raw Resc", str(t1_lam["cntd_raw"]["novel_non_dup"]["n_rescue"])),
        ("Tier1B Vis-IDF MacroB3", f"{t1b['visual_idf_mv_2dse']['macro_b3']:.4f}"),
        ("Tier2 SelfGate Orig Acc", f"{t2m['vlm_self_gate']['orig_acc']:.4f}"),
        ("Tier2 Naive MisFlips", str(t2m["vlm_naive_orig"]["n_mismatch_flip"])),
        ("Tier2 CompEvid MisAcc", f"{t2m['vlm_comp_evid_gate']['mismatch_acc']:.4f}"),
    ]

    vs1_c = val["conditions"]["clean"]["stage1"]
    vs1_s = val["conditions"]["stress30"]["stage1"]
    vs2_c = val["conditions"]["clean"]["stage2_policies"]
    vs2_s = val["conditions"]["stress30"]["stage2_policies"]
    hs1_c = hold["conditions"]["clean"]["stage1"]
    hs1_s = hold["conditions"]["stress30"]["stage1"]
    hs2_c = hold["conditions"]["clean"]["stage2_policies"]
    hs2_st = diag["holdout_stress30_end_to_end"]["stage2_policies_k60"]
    st_cl = diag["stratified_b3"]["clean"]

    checks_sec5 = [
        ("Val Clean Naive MacroB3", f"{vs1_c['naive_mv_2dse']['macro_b3']:.4f}"),
        ("Val Stress Vis-IDF MacroB3", f"{vs1_s['visual_idf_mv_2dse']['macro_b3']:.4f}"),
        ("Val Clean Inline AMI", f"{vs2_c['evid_self_gate_inline']['ami']:.4f}"),
        ("Val Stress CleanFB MacroB3", f"{vs2_s['evid_self_gate_clean_fallback']['macro_b3']:.4f}"),
        ("Holdout Clean Vis-IDF AMI", f"{hs1_c['visual_idf_mv_2dse']['ami']:.4f}"),
        ("Holdout Stress CNTD-Cond MacroB3", f"{hs1_s['cntd_cond_mv_2dse']['macro_b3']:.4f}"),
        ("Holdout Clean K180 Inline AMI", f"{hs2_c['evid_self_gate_inline']['ami']:.4f}"),
        ("Holdout Clean K180 CleanFB MicroB3", f"{hs2_c['evid_self_gate_clean_fallback']['micro_b3']:.4f}"),
        ("Holdout Stress K60 Inline MacroB3", f"{hs2_st['evid_self_gate_inline']['macro_b3']:.4f}"),
        ("Holdout Head CleanFB B3", f"{st_cl['k60_evid_self_gate_clean_fallback']['strata']['head']['macro_f1']:.4f}"),
    ]

    for desc, exp in checks_sec3:
        if exp not in sec3:
            errors.append(f"[03_causal_diagnosis.tex] Expected metric '{desc}={exp}' not found!")
    for desc, exp in checks_sec5:
        if exp not in sec5:
            errors.append(f"[05_experiments.tex] Expected metric '{desc}={exp}' not found!")

    n_checks = len(checks_sec3) + len(checks_sec5)
    print(
        f"Checked {len(tex_files)} .tex files | {len(bib_keys)} BibTeX entries | "
        f"{len(all_labels)} labels | {len(all_refs)} refs | {len(all_cites)} cites | {n_checks} numeric audit checks"
    )
    if errors:
        print("ERRORS:")
        for e in errors:
            print(" ", e)
        return 1
    print("ALL CHECKS PASSED cleanly (0 errors).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
