# Overleaf & Local LaTeX Compilation Guide (*Information Processing & Management* / Elsevier `elsarticle`)

This folder (`paper/`) contains the complete, self-contained **Elsevier `elsarticle`** manuscript targeting ***Information Processing & Management* (IP&M)**, backed by the audited **Stage 1 (`D1–D6`)**, **Stage 2A (`Qwen/Qwen3-8B` strictly causal suite + `SUG-CrossLLM-v2`)**, and **Stage 2B (27-block entity/hashtag/year anonymization stress test)** evaluation artifacts.

## Directory Structure

- `main.tex` — Master Elsevier `elsarticle` document (`preprint,12pt,authoryear` review draft; `highlights` environment, abstract, keywords, `elsarticle-harv` bibliography style, and clean optional camera-ready toggle)
- `references.bib` — 26 BibTeX entries (`authoryear` / `elsarticle-harv` compatible via `natbib`)
- `sections/`
  - `01_intro.tex` — Introduction, formal diagnosis of non-causal leaks, and four contributions
  - `02_related.tex` — Related work across SE/hyperbolic SED, hybrid/LLM-assisted clustering, causal streaming vs. memorization, and cross-day continuity
  - `03_problem_theory.tex` — Formal strictly causal problem statement, `all_shown` vs. legacy $U_t$ utility definitions, and structural-entropy information-gain theory (Proposition 1)
  - `04_method.tex` — SUG-SED methodology: deterministic seeded spherical $k$-means (`Meso-0`), causal boundary-knife & meso-merge controllers, Eq. (5)/(6) routing utilities, `SUG-CrossLLM-v2` `[0.65, 0.85)` ambiguity-band linker, and Algorithm 1
  - `05_experiments.tex` — Comprehensive evaluation answering RQ1–RQ6 (dev budget trajectories, held-out causal streaming vs. 0-LLM/random/non-causal baselines, component ablations & Wilcoxon/Holm statistics, cross-day `SUG-CrossLLM-v2`, Stage 2B anonymization anti-memorization test, and Stage 1 qualitative/root-cause audit)
  - `06_limitations_conclusion.tex` — Limitations, ethical considerations, and conclusion
  - `07_appendix.tex` — Reproducibility appendix: prompt templates, entity-type anonymization rules, cross-day protocol details, and Stage 1 leave-one-out root-cause cases
- `figures/` — Vector `.pdf` and high-DPI `.png` figures rendered strictly from audited JSONs:
  - `fig_overview.{pdf,png}`
  - `fig_budget_pareto.{pdf,png}`
  - `fig_causal_trajectories.{pdf,png}`
  - `fig_ablations_and_crossday.{pdf,png}`
  - `fig_anonymization_stress.{pdf,png}`
- `generate_ipm_figures.py` — Deterministic figure generator reading exclusively from audited JSON/summary files
- `verify_latex.py` — Static LaTeX, BibTeX, figure path, Elsevier `highlights` limit, and audited-number consistency verifier

## Option A: Compile on Overleaf (Recommended)

1. Compress the contents of `paper/` into a `.zip` archive (or upload `main.tex`, `references.bib`, `sections/*.tex`, and `figures/*.pdf` directly into a new Overleaf project preserving folder structure).
2. In Overleaf **Menu**:
   - Set **Compiler** to `pdfLaTeX`.
   - Set **TeX Live version** to `2023` or newer (`elsarticle` and `elsarticle-harv` are pre-installed).
   - Set **Main document** to `main.tex`.
3. Click **Recompile**.

> **Camera-Ready / Final Layout Toggle**: In `paper/main.tex` lines 4–5, switch from:
> ```latex
> \documentclass[preprint,12pt,authoryear]{elsarticle}
> % \documentclass[final,5p,times,twocolumn,authoryear]{elsarticle}
> ```
> to the commented two-column `5p` line when preparing the final accepted proofs.

## Option B: Compile Locally (TeX Live / MiKTeX)

From the `paper/` directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or with `latexmk`:

```bash
latexmk -pdf -bibtex main.tex
```

## Pre-Submission Verification & Figure Regeneration

From the repository root:

```bash
python paper/generate_ipm_figures.py
python paper/verify_latex.py
```

`verify_latex.py` validates balanced braces/environments/math delimiters, unescaped `&`/`_` and non-ASCII characters, unresolved `\ref`/`\eqref`/`\citep`/`\citet` citations, Elsevier `highlights` length (`<= 85` visible chars per bullet), missing figure files, and verbatim numeric consistency against all Stage 1, Stage 2A, Stage 2B, and Dev budget JSON logs.
