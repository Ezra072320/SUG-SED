# Event2018 Benchmark (`M_01..M_16`, `Qwen/Qwen3-8B`, `paraphrase-multilingual-MiniLM-L12-v2 / distiluse-base-multilingual-cased-v1`)

> Metrics strictly reported as **ARI / NMI / AMI (%)** per block (`M_01..M_16`) and 16-block average (`Mean`).

| Group | Method | Setting | M_01 (ARI/NMI/AMI) | M_02 (ARI/NMI/AMI) | M_03 (ARI/NMI/AMI) | M_04 (ARI/NMI/AMI) | M_05 (ARI/NMI/AMI) | M_06 (ARI/NMI/AMI) | M_07 (ARI/NMI/AMI) | M_08 (ARI/NMI/AMI) | M_09 (ARI/NMI/AMI) | M_10 (ARI/NMI/AMI) | M_11 (ARI/NMI/AMI) | M_12 (ARI/NMI/AMI) | M_13 (ARI/NMI/AMI) | M_14 (ARI/NMI/AMI) | M_15 (ARI/NMI/AMI) | M_16 (ARI/NMI/AMI) | **Mean (ARI / NMI / AMI)** | Calls / Tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-LLM Original | SBERT* (KMeans, Oracle-K) | 0 Tokens | 20.49 / 59.26 / 58.57 | 31.57 / 64.00 / 63.22 | 33.15 / 64.34 / 63.75 | 24.58 / 61.62 / 60.79 | 47.38 / 77.27 / 76.24 | 39.35 / 71.35 / 70.54 | 28.62 / 65.50 / 64.58 | 46.87 / 76.77 / 75.64 | 24.26 / 65.03 / 63.57 | 35.86 / 72.66 / 70.84 | 37.11 / 73.96 / 72.60 | 50.63 / 76.90 / 75.89 | 36.15 / 66.55 / 65.63 | 48.20 / 74.26 / 73.31 | 39.27 / 73.47 / 72.39 | 25.93 / 67.39 / 66.07 | **35.59 / 69.40 / 68.35** | 0 / 0 |
| 0-LLM Original | HISEvent Official (AAAI'24) | 0 Tokens | 56.16 / 77.46 / 77.19 | 64.74 / 78.41 / 77.95 | 48.94 / 75.36 / 74.92 | 47.00 / 73.32 / 72.71 | 62.80 / 82.05 / 81.22 | 56.83 / 82.46 / 81.76 | 54.08 / 80.68 / 80.11 | 82.88 / 90.63 / 90.23 | 51.63 / 77.06 / 76.33 | 56.90 / 82.77 / 81.51 | 52.25 / 82.39 / 81.32 | 87.18 / 89.18 / 88.67 | 77.53 / 88.04 / 87.69 | 80.50 / 90.25 / 89.84 | 69.99 / 85.10 / 84.47 | 36.66 / 74.11 / 73.00 | **61.63 / 81.83 / 81.18** | 0 / 0 |
| Unbounded Original | RagSEDE Official (WWW'26) | Unbounded (Cost Reported) | 76.39 / 78.32 / 77.67 | 81.08 / 81.23 / 80.65 | 48.02 / 69.94 / 68.90 | 58.60 / 67.49 / 66.32 | 56.40 / 77.53 / 76.19 | 90.44 / 87.30 / 86.83 | 46.96 / 72.05 / 70.80 | 76.42 / 84.66 / 83.64 | 42.04 / 69.17 / 67.71 | 69.62 / 78.27 / 76.68 | 73.46 / 80.63 / 79.42 | 87.68 / 89.73 / 89.18 | 51.36 / 68.99 / 67.62 | 58.54 / 72.46 / 70.79 | 84.54 / 84.58 / 83.91 | 73.01 / 81.02 / 79.79 | **67.16 / 77.71 / 76.63** | 4792 / 2,746,082 |
| Unbounded Original | LSED Official HGCN (ACL'25, Oracle-K, w=infty) | Unbounded (All N) | *Pending Run* | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | *Pending* | - |
| Unbounded Original | LSED Official HGCN (ACL'25, Est-K, w=infty) | Unbounded (All N) | *Pending Run* | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | *Pending* | - |
| Budgeted LLM (20k) | COBRAS Official (IDA'18, Binary LLM) | 20k Budget/Blk | 41.96 / 65.09 / 64.72 | 36.34 / 56.75 / 55.96 | 31.01 / 55.73 / 55.11 | 40.06 / 51.69 / 51.35 | 55.48 / 72.43 / 71.70 | 18.48 / 34.53 / 34.39 | 35.45 / 66.00 / 65.35 | 6.93 / 24.27 / 23.99 | 17.95 / 30.77 / 30.56 | 43.14 / 61.17 / 60.25 | 44.00 / 62.75 / 62.00 | 58.94 / 73.11 / 72.64 | 56.66 / 65.17 / 64.29 | 61.34 / 69.00 / 68.55 | 57.76 / 75.56 / 74.91 | 29.10 / 60.78 / 59.78 | **39.66 / 57.80 / 57.22** | 2005 / 318,584 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Oracle-K) | 20k Budget/Blk | 48.03 / 69.63 / 69.11 | 48.64 / 66.82 / 66.12 | 62.79 / 73.68 / 73.22 | 51.86 / 67.39 / 66.68 | 55.20 / 78.29 / 77.31 | 64.66 / 78.39 / 77.77 | 48.62 / 73.85 / 73.17 | 67.58 / 82.39 / 81.54 | 32.20 / 69.43 / 68.15 | 58.41 / 77.87 / 76.41 | 51.03 / 75.49 / 74.19 | 61.96 / 79.19 / 78.30 | 47.88 / 69.34 / 68.49 | 44.20 / 74.98 / 74.03 | 54.44 / 75.42 / 74.42 | 35.90 / 70.41 / 69.20 | **52.09 / 73.91 / 73.01** | 249 / 318,385 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Est-K) | 20k Budget/Blk | 89.10 / 84.34 / 84.15 | 44.94 / 65.56 / 64.80 | 37.76 / 67.47 / 66.89 | 34.67 / 65.89 / 65.05 | 49.88 / 76.12 / 74.90 | 38.79 / 70.23 / 68.90 | 33.42 / 69.01 / 68.03 | 78.44 / 85.77 / 85.14 | 64.08 / 77.34 / 76.56 | 43.69 / 74.64 / 72.66 | 46.19 / 74.39 / 72.73 | 45.81 / 75.31 / 74.06 | 35.64 / 64.47 / 63.26 | 38.23 / 71.94 / 70.66 | 40.50 / 72.43 / 71.17 | 38.35 / 72.50 / 71.31 | **47.47 / 72.96 / 71.89** | 249 / 318,385 |
| Budgeted LLM (20k) | Cequel WCKMeans (CIKM'25, Oracle-K) | 20k Budget/Blk | 18.53 / 56.92 / 56.19 | 28.71 / 56.96 / 56.04 | 27.81 / 57.55 / 56.86 | 26.58 / 60.02 / 59.17 | 45.11 / 73.00 / 71.78 | 33.36 / 67.33 / 66.41 | 31.91 / 65.32 / 64.40 | 40.69 / 70.46 / 69.03 | 20.35 / 62.89 / 61.32 | 33.97 / 70.73 / 68.81 | 25.56 / 68.18 / 66.54 | 54.86 / 74.61 / 73.51 | 28.43 / 59.13 / 58.01 | 38.87 / 67.03 / 65.82 | 43.33 / 72.50 / 71.38 | 30.44 / 66.18 / 64.79 | **33.03 / 65.55 / 64.38** | 249 / 318,385 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | 20k Budget/Blk | 54.70 / 73.35 / 72.78 | 52.69 / 69.69 / 68.66 | 44.49 / 69.37 / 68.60 | 67.36 / 73.41 / 72.63 | 74.79 / 85.46 / 84.71 | 80.32 / 83.36 / 82.80 | 57.48 / 79.42 / 78.74 | 68.59 / 84.79 / 83.91 | 74.03 / 79.30 / 78.39 | 82.95 / 85.59 / 84.63 | 65.31 / 83.13 / 82.24 | 77.28 / 85.65 / 84.94 | 70.11 / 74.28 / 73.32 | 66.56 / 79.58 / 78.74 | 68.32 / 83.40 / 82.71 | 33.24 / 71.72 / 70.01 | **64.89 / 78.84 / 77.99** | 438 / 270,005 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | Unbounded | 54.74 / 73.43 / 72.88 | 52.69 / 69.70 / 68.69 | 44.49 / 69.37 / 68.60 | 67.40 / 73.46 / 72.69 | 74.79 / 85.46 / 84.71 | 80.32 / 83.36 / 82.80 | 57.48 / 79.42 / 78.74 | 68.59 / 84.79 / 83.91 | 74.03 / 79.30 / 78.39 | 82.95 / 85.59 / 84.63 | 65.31 / 83.13 / 82.24 | 77.28 / 85.65 / 84.94 | 70.12 / 74.33 / 73.40 | 66.56 / 79.58 / 78.74 | 68.32 / 83.40 / 82.71 | 33.24 / 71.72 / 70.01 | **64.89 / 78.86 / 78.00** | 527 / 321,785 |

## Per-Metric Expanded Summary (Mean ARI / NMI / AMI)

| Group | Method | Setting | ARI | NMI | AMI | Total Calls | Total Tokens | Parse Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0-LLM Original | SBERT* (KMeans, Oracle-K) | 0 Tokens | 35.59 | 69.40 | 68.35 | 0 | 0 | 0 |
| 0-LLM Original | HISEvent Official (AAAI'24) | 0 Tokens | 61.63 | 81.83 | 81.18 | 0 | 0 | 0 |
| Unbounded Original | RagSEDE Official (WWW'26) | Unbounded (Cost Reported) | 67.16 | 77.71 | 76.63 | 4792 | 2,746,082 | 152 |
| Budgeted LLM (20k) | COBRAS Official (IDA'18, Binary LLM) | 20k Budget/Blk | 39.66 | 57.80 | 57.22 | 2005 | 318,584 | 3 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Oracle-K) | 20k Budget/Blk | 52.09 | 73.91 | 73.01 | 249 | 318,385 | 1 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Est-K) | 20k Budget/Blk | 47.47 | 72.96 | 71.89 | 249 | 318,385 | 1 |
| Budgeted LLM (20k) | Cequel WCKMeans (CIKM'25, Oracle-K) | 20k Budget/Blk | 33.03 | 65.55 | 64.38 | 249 | 318,385 | 1 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | 20k Budget/Blk | 64.89 | 78.84 | 77.99 | 438 | 270,005 | 0 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | Unbounded | 64.89 | 78.86 | 78.00 | 527 | 321,785 | 0 |

## Held-Out Test Split Summary (`M_06..M_16`, 11 Blocks)

| Group | Method | Setting | ARI | NMI | AMI | Calls | Tokens | Parse Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0-LLM Original | SBERT* (KMeans, Oracle-K) | 0 Tokens | 37.48 | 71.26 | 70.10 | 0 | 0 | 0 |
| 0-LLM Original | HISEvent Official (AAAI'24) | 0 Tokens | 64.22 | 83.88 | 83.18 | 0 | 0 | 0 |
| Unbounded Original | RagSEDE Official (WWW'26) | Unbounded (Cost Reported) | 68.55 | 78.99 | 77.85 | 3353 | 1,932,980 | 122 |
| Budgeted LLM (20k) | COBRAS Official (IDA'18, Binary LLM) | 20k Budget/Blk | 39.07 | 56.65 | 56.06 | 1385 | 218,876 | 3 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Oracle-K) | 20k Budget/Blk | 51.53 | 75.16 | 74.15 | 163 | 218,842 | 1 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Est-K) | 20k Budget/Blk | 45.74 | 73.46 | 72.23 | 163 | 218,842 | 1 |
| Budgeted LLM (20k) | Cequel WCKMeans (CIKM'25, Oracle-K) | 20k Budget/Blk | 34.71 | 67.67 | 66.37 | 163 | 218,842 | 1 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | 20k Budget/Blk | 67.65 | 80.93 | 80.04 | 294 | 177,902 | 0 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | Unbounded | 67.65 | 80.93 | 80.05 | 350 | 212,211 | 0 |

## Protocol Alignment & Documented Unavoidable Exceptions

1. **Shared LLM Backbone (`Qwen/Qwen3-8B`)**: RagSEDE (Qwen2/2.5/Llama3), Cequel (GPT-3.5-Turbo/Llama3-8B), and LSED (Llama3.1-8B/Qwen2.5-7B/Gemma2-9B) originally used different LLMs; all LLM baselines use unified `Qwen/Qwen3-8B` (deterministic `temperature=0.0` for `FRAG-SED`/`RagSEDE`/`COBRAS`/`LSED`, and official `temp=1.0, n=5` majority voting for `Cequel`, with `enable_thinking=False`) for fair comparison.
2. **3-Category Benchmark Design (Event2018)**: Keeps 0-LLM original (`SBERT*`, `HISEvent Official` with language-specific preprocessing, `n_sub=300`, `distiluse-base-multilingual-cased-v1` encoder & `eps=0.0`), unbounded original LLM-SED with exact token-cost accounting (`RagSEDE Official`, `LSED Official w=infty`), budgeted active-query LLM baselines at `20k tokens/block` (`COBRAS Official`, `Cequel Official` on `paraphrase-multilingual-MiniLM-L12-v2`), and `FRAG-SED` (`20k Budget/Blk` & `Unbounded` on `paraphrase-multilingual-MiniLM-L12-v2`).
3. **RagSEDE Infrastructure**: Uses official `max_retries=10`, `similarity_threshold=0.30`, and `use_json_mode=False` (non-forced JSON output parsed from ```json blocks or raw JSON); external RAGFlow v0.17.2 server (`BAAI/bge-small-zh-v1.5`) is replaced by a functionally equivalent in-memory hybrid index (`distiluse-base-multilingual-cased-v1` dense cosine + IDF-weighted keyword similarity excluding prompt boilerplate, retrieved by `Enc('INPUT: ' + a_k[:200])`) with `temp=0.2` retry jitter before falling back to `'Confusion'`.
4. **COBRAS LLM Oracle Adaptation**: Original IDA'18 COBRAS assumes a noise-free ground-truth oracle; under the budgeted setting, queries use a concise binary `Yes/No` `Qwen/Qwen3-8B` oracle (`max_tokens=8`) capped cleanly by the `20k tokens/block` causal budget.
5. **Cequel Solver & Sampling**: Uses official 5-sample majority voting (`n=5, temp=1.0, max_tokens=1, max_vote_retries=5`) via `call_text_n`, official `WCSC` (`Q` in `float64` per paper Eq. 4, sparse `eigsh` for `n > 2500`, primary on tweets, `[0.5, 1.5]` Q weights) and `WCKMeans` (`init='random'`, `ML in [0.01, 0.1]`, `CL in [0, 0.01]`) under the `20k tokens/block` causal budget.
6. **LSED Open-Set Window `w=infty`**: Paper Fig. 6 open-set setting (`w=infty`: train 3-layer HGCN Poincare MLP once on LLM-summarized `M_00`, cluster LLM-summarized `M_01..M_16` via KMeans on `all-MiniLM-L6-v2` embeddings, numeric `tan_reg=0.02` preventing 100% Poincare boundary saturation) replaces supervised `w=1` updates; both `M_00` (`lsed_M00_prompts.jsonl`) and `M_01..M_16` (`lsed_M01_M16_prompts.jsonl`) summaries are required to avoid train/test distribution shift.
7. **`Oracle-K` vs `Est-K`**: Cequel and LSED originally assume known `K*`; unsupervised `Est-K` uses `HISEvent Official`'s predicted cluster count (`n_pred_clusters`) alongside `Oracle-K`.
