# FRAG-SED: Fragment-Level Selective LLM Verification for Budget-Constrained Streaming Social Event Detection

**FRAG-SED** (**Frag**ment-Level Selective LLM Verification for **S**ocial **E**vent **D**etection) is a training-free, backbone-agnostic 3-step open-set streaming Social Event Detection framework combining plug-and-play **2D Structural Entropy (2D-SE)** graph partitioning with a unified $k$-way selective LLM verification operator under strict zero-lookahead causal token budgets, alongside strictly aligned implementations of 6 reference SED and active-clustering baselines.

## Active 3-Category Benchmark Design

All baselines and **FRAG-SED** are evaluated on bilingual streaming benchmarks (**`Event2012` English**, `M_01..M_21`; **`Event2018` French**, `M_01..M_16`) covering Dev (`M_02..M_05`), Held-Out (`M_06+`, 27 blocks), and Full-Stream (`M_01+`, 37 blocks) splits using strict **ARI / NMI / AMI (%)** metrics and unified `Qwen/Qwen3-8B` (`enable_thinking=False`). Dataset-specific encoders follow official specs:
- **`Event2012`**: `all-MiniLM-L6-v2` (`max_length=256`, `float32`).
- **`Event2018`**: `distiluse-base-multilingual-cased-v1` (with Sentence-Transformers `2_Dense` `768->512` projection) + French accent-preserving cleaning for official `SBERT*`, `HISEvent Official` (`n_sub=300`), and `RagSEDE Official` (`sim_thresh=0.30`); `paraphrase-multilingual-MiniLM-L12-v2` (`float32`, `max_length=256`) for `FRAG-SED`, `COBRAS 20k`, and `Cequel 20k`; `all-MiniLM-L6-v2` for `LSED` English summaries.

| Group | Method | Setting | Token Budget |
|---|---|---|---|
| **1. 0-LLM Original** | `SBERT*` (EMNLP'19) | Language-specific text cleaning + `KMeans(Oracle-K)` | `0` |
| **1. 0-LLM Original** | `HISEvent Official` (AAAI'24) | Language-specific text cleaning + raw unified attributes + `eps=0.0` 1D/2D-SE (`n_sub=400/300`) | `0` |
| **2. Unbounded Original** | `RagSEDE Official` (WWW'26) | Agglom anchors (`0.40/0.30`) + MMR top-3 + in-memory hybrid Event KB (`max_retries=10`, non-JSON-mode) | Unbounded (Cost Reported) |
| **2. Unbounded Original** | `LSED Official` (ACL'25, `w=infty` Fig. 6) | Summarized `M_00` + all-`N` rewrites (`Please respond in English`) + OLE 2D TIME + 3-layer HGCN Poincare MLP (`Oracle-K` / `HISEvent Est-K`) | Unbounded (Cost Reported) |
| **3. Budgeted LLM** | `COBRAS Official` (IDA'18) | Top-down 2-means splitting + bottom-up merging + binary `Yes/No` LLM oracle | `20k Tokens / Block` |
| **3. Budgeted LLM** | `Cequel Official` (CIKM'25) | Triangle Explore-Consolidate + uncapped 5-option prompt + 5-vote (`temp=1.0, max_tokens=1`) + `WCSC` (`Q in float64`) / `WCKMeans` (`init='random'`) | `20k Tokens / Block` |
| **4. Ours** | `FRAG-SED` | Step 0 `2D-SE` -> Step 1 Split Verification (`<= rho*B`) -> Step 2 Utility-Guided `k`-Way Merge (`CleanConstraintDSU`) -> Step 3 Output | `20k Budget/Blk` & `Unbounded` |

## Completed Full-Stream (`M_01+`) & Held-Out (`M_06+`) Results Snapshot

Mean **ARI / NMI / AMI (%)** and `(Total Tokens)` across **Event2012 Held-Out (`M_06..M_21`, 16 blocks)**, **Event2018 Held-Out (`M_06..M_16`, 11 blocks)**, **Pooled 27 Held-Out (`M_06+`)**, and **Pooled 37 Full (`M_01+`)**:

| Group | Method | EN Held-Out (16) ARI/NMI/AMI | FR Held-Out (11) ARI/NMI/AMI | **Pooled 27 Held-Out (`M_06+`)** (Tokens) | **Pooled 37 Full (`M_01+`)** (Tokens) |
|---|---|---:|---:|---:|---:|
| **0-LLM** | `SBERT* (Oracle-K)` | 44.38 / 79.88 / 77.76 | 37.48 / 71.26 / 70.10 | 41.57 / 76.37 / 74.64 (0) | 41.43 / 74.90 / 73.34 (0) |
| **0-LLM** | `HISEvent Official` | 63.52 / 83.02 / 81.83 | 64.22 / **83.88 / 83.18** | 63.81 / 83.37 / 82.38 (0) | 61.67 / 81.85 / 80.93 (0) |
| **Unbounded** | `RagSEDE Official` | **80.21 / 90.36 / 89.43** | **68.55** / 78.99 / 77.85 | **75.46** / 85.73 / 84.71 (3,045,492) | **73.75 / 84.13 / 83.15** (4,432,774) |
| **Unbounded** | `LSED Official (w=infty)` | *Running on Colab* | *Running on Colab* | *53,423 Dev rows verified; full end-to-end deployed* | - |
| **Budget 20k** | `COBRAS Official` | 59.29 / 75.90 / 74.34 | 39.07 / 56.65 / 56.06 | 51.05 / 68.06 / 66.90 (538,051) | 50.26 / 67.34 / 66.29 (737,610) |
| **Budget 20k** | `Cequel WCSC (Oracle-K)` | 44.72 / 79.53 / 77.31 | 51.53 / 75.16 / 74.15 | 47.49 / 77.75 / 76.02 (537,168) | 48.32 / 76.65 / 75.10 (736,262) |
| **Budget 20k** | `Cequel WCSC (Est-K)` | 58.80 / 81.60 / 80.24 | 45.74 / 73.46 / 72.23 | 53.48 / 78.28 / 76.97 (537,168) | 54.25 / 77.67 / 76.49 (736,262) |
| **Budget 20k** | `Cequel WCKMeans (Oracle-K)` | 42.39 / 78.44 / 76.22 | 34.71 / 67.67 / 66.37 | 39.26 / 74.05 / 72.21 (537,168) | 38.26 / 72.16 / 70.47 (736,262) |
| **Ours** | **`FRAG-SED (20k)`** | 74.37 / 89.36 / 88.17 | 67.65 / 80.93 / 80.04 | 71.63 / **85.93 / 84.86** (**286,711**) | 69.35 / 83.96 / 82.93 (**419,243**) |
| **Ours** | **`FRAG-SED (Unbounded)`** | 74.37 / 89.36 / 88.18 | 67.65 / 80.93 / 80.05 | 71.64 / **85.93 / 84.87** (325,501) | 69.38 / 84.00 / 82.98 (496,166) |

See [Event2012 Full-Stream Report](results/full_event2012_m01_m21/val_benchmark_report.md) and [Event2018 Full-Stream Report](results/full_event2018_m01_m16/val_benchmark_report.md).

## Completed Dev (`M_02..M_05`) Results Snapshot

Mean **ARI / NMI / AMI (%)** and `(Calls / Tokens)` across 4 Dev blocks (`M_02..M_05`):

| Group | Method | Event2012 Mean (ARI / NMI / AMI) | EN Cost | Event2018 Mean (ARI / NMI / AMI) | FR Cost |
|---|---|---:|---:|---:|---:|
| **0-LLM Original** | `SBERT* (KMeans, Oracle-K)` | 62.55 / 85.78 / 84.50 | 0 / 0 | 34.17 / 66.81 / 66.00 | 0 / 0 |
| **0-LLM Original** | `HISEvent Official (AAAI'24)` | 67.53 / 86.18 / 85.33 | 0 / 0 | 55.87 / **77.28 / 76.70** | 0 / 0 |
| **Unbounded Original** | `RagSEDE Official (WWW'26)` | **89.70** / 92.87 / 92.23 | 639 / 251,203 | **61.02** / 74.05 / 73.02 | 1,049 / 595,592 |
| **Unbounded Original** | `LSED Official (w=infty)` | *Pending HGCN backfill* (`27,424` summaries verified) | - | *Pending HGCN backfill* (`25,999` summaries verified) | - |
| **Budgeted LLM (20k)** | `COBRAS Official (IDA'18)` | 73.08 / 85.98 / 85.06 | 752 / 79,845 | 40.72 / 59.15 / 58.53 | 499 / 79,699 |
| **Budgeted LLM (20k)** | `Cequel WCSC (Oracle-K / Est-K)` | 57.11/83.63/82.17 & 69.39/86.10/85.19 | 79 / 79,592 | 54.62/71.55/70.83 & 41.81/68.76/67.91 | 68 / 79,586 |
| **Budgeted LLM (20k)** | `Cequel WCKMeans (Oracle-K)` | 51.50 / 81.96 / 80.41 | 79 / 79,592 | 32.05 / 61.88 / 60.96 | 68 / 79,586 |
| **Ours (FRAG-SED)** | **`FRAG-SED (20k Budget/Blk)`** | 83.49 / **93.05 / 92.35** | 58 / 20,871 | 59.83 / 74.48 / 73.65 | 115 / 72,704 |
| **Ours (FRAG-SED)** | **`FRAG-SED (Unbounded)`** | 83.49 / **93.05 / 92.35** | 58 / 20,871 | 59.84 / 74.50 / 73.67 | 136 / 83,361 |

See [Event2012 Dev Report](results/dev_m02_m05_benchmark/val_benchmark_report.md) and [Event2018 Dev Report](results/dev_event2018_m02_m05_benchmark/val_benchmark_report.md).

## Repository Structure

```text
.
├── FRAG_SED_Paper_Design.md                 # Complete FRAG-SED paper design & experiment plan
├── README.md                                # Repository guide, locked 3-category setup & full/dev results
├── core_shared/                             # Shared data loader, graph builder, 1D/2D-SE solver, token ledger
├── trackA_sug_sed/
│   ├── frag_sed.py                          # FRAG-SED 3-step algorithm (Split Verification -> k-Way Merge)
│   ├── star_meso.py                         # Cluster summarization, TF-IDF & k-way star helpers
│   ├── baselines_official.py                # Strictly aligned implementations of the 6 baseline papers
│   ├── baselines_sed.py                     # Structural-entropy baseline utilities (HISEvent helpers)
│   ├── run_val_benchmark.py                 # Main Event2012/Event2018 full & dev runner (--max-workers, M_06+ summary)
│   ├── colab_batch_qwen3_8b.ipynb           # End-to-end Colab GPU runner for unbounded LSED (inference + HGCN + eval)
│   ├── embed_cache.py                       # Per-block embedding cache (Sentence-Transformers Dense head + ml256)
│   ├── llm_client.py                        # Causal-budgeted OpenAI-compatible client (incl. call_text_n 5-vote)
│   ├── anonymizer.py                        # Entity/privacy anonymization module (planned §8.2)
│   └── cross_day.py                         # Cross-day temporal alignment module (planned §8.1)
├── tests/
│   ├── test_frag_sed.py                     # Unit tests for FRAG-SED pipeline & causal budget enforcement (4/4)
│   └── test_baselines_official.py           # Strict alignment unit tests for all 6 official baselines (9/9)
├── cache/embeddings/                        # Cached per-block SBERT embeddings
├── results/
│   ├── full_event2012_m01_m21/              # Event2012 Full-Stream (M_01..M_21) & Held-Out (M_06..M_21) results
│   ├── full_event2018_m01_m16/              # Event2018 Full-Stream (M_01..M_16) & Held-Out (M_06..M_16) results
│   ├── dev_m02_m05_benchmark/               # Event2012 Dev (M_02..M_05) verified reports & TokenLedgers
│   ├── dev_event2018_m02_m05_benchmark/     # Event2018 Dev (M_02..M_05) verified reports & TokenLedgers
│   └── val_m01_m05_benchmark/               # Legacy M_01..M_05 reference reports
└── _deprecated_to_delete/                   # Quarantined legacy v2.0/v2.1 & pre-alignment files (see MANIFEST.md)
```

## Quick Start

1. **Run unit tests (offline)**:
   ```powershell
   python tests/test_frag_sed.py
   python tests/test_baselines_official.py
   ```
2. **Dry-run benchmark context & export unbounded LSED Colab prompts**:
   ```powershell
   python trackA_sug_sed/run_val_benchmark.py --dataset Event2012 --blocks 2,3,4,5 --out-dir results/dev_m02_m05_benchmark --dry-run
   python trackA_sug_sed/run_val_benchmark.py --dataset Event2018 --blocks 2,3,4,5 --out-dir results/dev_event2018_m02_m05_benchmark --dry-run
   ```
3. **Run full-stream benchmarks (0-LLM + budgeted LLM + FRAG-SED + unbounded RagSEDE)**:
   ```powershell
   python trackA_sug_sed/run_val_benchmark.py --dataset Event2012 --blocks 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21 --out-dir results/full_event2012_m01_m21 --methods sbert_kmeans,hisevent_official,ragsede_unbounded,cobras_20k,cequel_20k,frag_sed_20k,frag_sed_unbounded --max-workers 6 --pace-sec 0.08 --skip-export-colab
   python trackA_sug_sed/run_val_benchmark.py --dataset Event2018 --blocks 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --out-dir results/full_event2018_m01_m16 --methods sbert_kmeans,hisevent_official,ragsede_unbounded,cobras_20k,cequel_20k,frag_sed_20k,frag_sed_unbounded --max-workers 6 --pace-sec 0.08 --skip-export-colab
   ```
4. **End-to-end `LSED Official (w=infty)` on Colab GPU**:
   Run `trackA_sug_sed/colab_batch_qwen3_8b.ipynb` on Colab GPU (`event2012_full` & `event2018_full`) to perform LLM summarization + `all-MiniLM-L6-v2` encoding + 3-layer HGCN Poincare training + `Oracle-K` / `Est-K` evaluation (`lsed_full_results.json`).
