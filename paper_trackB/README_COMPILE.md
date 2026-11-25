# Overleaf & Local LaTeX Compilation Guide (`paper_trackB/`)

This folder contains the self-contained anonymous ACM `sigconf` manuscript for **Track B (`Evid-MSED`)**, strictly backed by audited \texttt{NED} Gate~B and Phase~2 Validation/Holdout JSON ledgers with `Qwen/Qwen3-VL-8B-Instruct`.

## Directory Manifest

- `main.tex` — Root ACM `sigconf` manuscript (`\documentclass[sigconf,anonymous,review]{acmart}`)
- `references.bib` — Complete, verified BibTeX database (20 citations)
- `sections/01_intro.tex` — Introduction, 72-hour chronological shortcut critique, and 4 verified contributions
- `sections/02_related.tex` — Related work (`KPGNN`, `HISEvent`, `RPLM-SED`, `HyperSED`, `LSED`, `LHG`, `RagSEDE`, `Cequel`, `NED`, `CrisisMMD`, `KQGNN/WEIBO2022`, `SeInEvent`, `CLIP`, `PID`, `Qwen3-VL`)
- `sections/03_causal_diagnosis.tex` — Time-shortcut-free concurrent alignment, Gate~B 3-condition counterfactual CLIP & `Qwen3-VL-8B` causal audit (`tab:gateb_summary`, `fig:gateb`)
- `sections/04_method.tex` — Stage-1 chunked `Visual-IDF MV-2DSE`, Proposition~1 (unified soft imputation eliminating $\delta_i\delta_j$), Stage-2 conflict-gated mesoscopic seam adjudication (`alg:evid_msed`)
- `sections/05_experiments.tex` — Validation & one-shot 24-event Holdout results under frozen `protocol_trackB_v1.json` (Clean + Stress-30\%, prefix budget trajectories, Head/Medium/Small stratification, action confusion matrix)
- `sections/06_limitations_conclusion.tex` — Candid limitations and conclusion
- `sections/07_appendix.tex` — Anonymized dual-image JSON schema and frozen hyperparameter summary (`tab:protocol_trackb`)
- `figures/` — All 4 publication figures (`fig_overview_evid_msed.png`, `gateB_ned_dev_qwen3_vl_8b.png`, `trackB_phase2_ned_val_holdout.png`, `trackB_phase2_holdout_stress_and_strata.png`)
- `generate_overview_fig.py` — Script to regenerate `figures/fig_overview_evid_msed.png`
- `verify_latex.py` — Static LaTeX syntax, citation, reference, figure-path, and JSON metric consistency checker

## Overleaf Compile

1. Zip the entire `paper_trackB/` directory (`main.tex`, `references.bib`, `sections/`, `figures/`).
2. Upload to [Overleaf](https://www.overleaf.com/) via *Upload Project*.
3. Set Compiler = **pdfLaTeX** (TeX Live >= **2023**) and click **Recompile**.

## Verification

Run `python paper_trackB/verify_latex.py` before compiling to confirm zero syntax errors, zero broken references/citations, and exact agreement with all 5 audited Track~B JSON result files.
