# Event2018 Benchmark (`M_02..M_05`, `Qwen/Qwen3-8B`, `paraphrase-multilingual-MiniLM-L12-v2 / distiluse-base-multilingual-cased-v1`)

> Metrics strictly reported as **ARI / NMI / AMI (%)** per block (`M_02..M_05`) and 4-block average (`Mean`).

| Group | Method | Setting | M_02 (ARI/NMI/AMI) | M_03 (ARI/NMI/AMI) | M_04 (ARI/NMI/AMI) | M_05 (ARI/NMI/AMI) | **Mean (ARI / NMI / AMI)** | Calls / Tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0-LLM Original | SBERT* (KMeans, Oracle-K) | 0 Tokens | 31.57 / 64.00 / 63.22 | 33.15 / 64.34 / 63.75 | 24.58 / 61.62 / 60.79 | 47.38 / 77.27 / 76.24 | **34.17 / 66.81 / 66.00** | 0 / 0 |
| 0-LLM Original | HISEvent Official (AAAI'24) | 0 Tokens | 64.74 / 78.41 / 77.95 | 48.94 / 75.36 / 74.92 | 47.00 / 73.32 / 72.71 | 62.80 / 82.05 / 81.22 | **55.87 / 77.28 / 76.70** | 0 / 0 |
| Unbounded Original | RagSEDE Official (WWW'26) | Unbounded (Cost Reported) | 81.08 / 81.23 / 80.65 | 48.02 / 69.94 / 68.90 | 58.60 / 67.49 / 66.32 | 56.40 / 77.53 / 76.19 | **61.02 / 74.05 / 73.02** | 1049 / 595,592 |
| Unbounded Original | LSED Official HGCN (ACL'25, Oracle-K, w=infty) | Unbounded (All N) | *Pending Run* | - | - | - | *Pending* | - |
| Unbounded Original | LSED Official HGCN (ACL'25, Est-K, w=infty) | Unbounded (All N) | *Pending Run* | - | - | - | *Pending* | - |
| Budgeted LLM (20k) | COBRAS Official (IDA'18, Binary LLM) | 20k Budget/Blk | 36.34 / 56.75 / 55.96 | 31.01 / 55.73 / 55.11 | 40.06 / 51.69 / 51.35 | 55.48 / 72.43 / 71.70 | **40.72 / 59.15 / 58.53** | 499 / 79,699 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Oracle-K) | 20k Budget/Blk | 48.64 / 66.82 / 66.12 | 62.79 / 73.68 / 73.22 | 51.86 / 67.39 / 66.68 | 55.20 / 78.29 / 77.31 | **54.62 / 71.55 / 70.83** | 68 / 79,586 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Est-K) | 20k Budget/Blk | 44.94 / 65.56 / 64.80 | 37.76 / 67.47 / 66.89 | 34.67 / 65.89 / 65.05 | 49.88 / 76.12 / 74.90 | **41.81 / 68.76 / 67.91** | 68 / 79,586 |
| Budgeted LLM (20k) | Cequel WCKMeans (CIKM'25, Oracle-K) | 20k Budget/Blk | 28.71 / 56.96 / 56.04 | 27.81 / 57.55 / 56.86 | 26.58 / 60.02 / 59.17 | 45.11 / 73.00 / 71.78 | **32.05 / 61.88 / 60.96** | 68 / 79,586 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | 20k Budget/Blk | 52.69 / 69.69 / 68.66 | 44.49 / 69.37 / 68.60 | 67.36 / 73.41 / 72.63 | 74.79 / 85.46 / 84.71 | **59.83 / 74.48 / 73.65** | 115 / 72,704 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | Unbounded | 52.69 / 69.70 / 68.69 | 44.49 / 69.37 / 68.60 | 67.40 / 73.46 / 72.69 | 74.79 / 85.46 / 84.71 | **59.84 / 74.50 / 73.67** | 136 / 83,361 |

## Per-Metric Expanded Summary (Mean ARI / NMI / AMI)

| Group | Method | Setting | ARI | NMI | AMI | Total Calls | Total Tokens | Parse Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0-LLM Original | SBERT* (KMeans, Oracle-K) | 0 Tokens | 34.17 | 66.81 | 66.00 | 0 | 0 | 0 |
| 0-LLM Original | HISEvent Official (AAAI'24) | 0 Tokens | 55.87 | 77.28 | 76.70 | 0 | 0 | 0 |
| Unbounded Original | RagSEDE Official (WWW'26) | Unbounded (Cost Reported) | 61.02 | 74.05 | 73.02 | 1049 | 595,592 | 10 |
| Budgeted LLM (20k) | COBRAS Official (IDA'18, Binary LLM) | 20k Budget/Blk | 40.72 | 59.15 | 58.53 | 499 | 79,699 | 0 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Oracle-K) | 20k Budget/Blk | 54.62 | 71.55 | 70.83 | 68 | 79,586 | 0 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Est-K) | 20k Budget/Blk | 41.81 | 68.76 | 67.91 | 68 | 79,586 | 0 |
| Budgeted LLM (20k) | Cequel WCKMeans (CIKM'25, Oracle-K) | 20k Budget/Blk | 32.05 | 61.88 | 60.96 | 68 | 79,586 | 0 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | 20k Budget/Blk | 59.83 | 74.48 | 73.65 | 115 | 72,704 | 0 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | Unbounded | 59.84 | 74.50 | 73.67 | 136 | 83,361 | 0 |

## Protocol Alignment & Documented Unavoidable Exceptions

1. **Shared LLM Backbone (`Qwen/Qwen3-8B`)**: RagSEDE (Qwen2/2.5/Llama3), Cequel (GPT-3.5-Turbo/Llama3-8B), and LSED (Llama3.1-8B/Qwen2.5-7B/Gemma2-9B) originally used different LLMs; all LLM baselines use unified `Qwen/Qwen3-8B` (deterministic `temperature=0.0` for `FRAG-SED`/`RagSEDE`/`COBRAS`/`LSED`, and official `temp=1.0, n=5` majority voting for `Cequel`, with `enable_thinking=False`) for fair comparison.
2. **3-Category Benchmark Design (Event2018)**: Keeps 0-LLM original (`SBERT*`, `HISEvent Official` with language-specific preprocessing, `n_sub=300`, `distiluse-base-multilingual-cased-v1` encoder & `eps=0.0`), unbounded original LLM-SED with exact token-cost accounting (`RagSEDE Official`, `LSED Official w=infty`), budgeted active-query LLM baselines at `20k tokens/block` (`COBRAS Official`, `Cequel Official` on `paraphrase-multilingual-MiniLM-L12-v2`), and `FRAG-SED` (`20k Budget/Blk` & `Unbounded` on `paraphrase-multilingual-MiniLM-L12-v2`).
3. **RagSEDE Infrastructure**: Uses official `max_retries=10`, `similarity_threshold=0.30`, and `use_json_mode=False` (non-forced JSON output parsed from ```json blocks or raw JSON); external RAGFlow v0.17.2 server (`BAAI/bge-small-zh-v1.5`) is replaced by a functionally equivalent in-memory hybrid index (`distiluse-base-multilingual-cased-v1` dense cosine + IDF-weighted keyword similarity excluding prompt boilerplate, retrieved by `Enc('INPUT: ' + a_k[:200])`) with `temp=0.2` retry jitter before falling back to `'Confusion'`.
4. **COBRAS LLM Oracle Adaptation**: Original IDA'18 COBRAS assumes a noise-free ground-truth oracle; under the budgeted setting, queries use a concise binary `Yes/No` `Qwen/Qwen3-8B` oracle (`max_tokens=8`) capped cleanly by the `20k tokens/block` causal budget.
5. **Cequel Solver & Sampling**: Uses official 5-sample majority voting (`n=5, temp=1.0, max_tokens=1, max_vote_retries=5`) via `call_text_n`, official `WCSC` (`Q` in `float64` per paper Eq. 4, sparse `eigsh` for `n > 2500`, primary on tweets, `[0.5, 1.5]` Q weights) and `WCKMeans` (`init='random'`, `ML in [0.01, 0.1]`, `CL in [0, 0.01]`) under the `20k tokens/block` causal budget.
6. **LSED Open-Set Window `w=infty`**: Paper Fig. 6 open-set setting (`w=infty`: train 3-layer HGCN Poincare MLP once on LLM-summarized `M_00`, cluster LLM-summarized `M_02..M_05` via KMeans on `all-MiniLM-L6-v2` embeddings, numeric `tan_reg=0.02` preventing 100% Poincare boundary saturation) replaces supervised `w=1` updates; both `M_00` (`lsed_M00_prompts.jsonl`) and `M_02..M_05` (`lsed_M02_M05_prompts.jsonl`) summaries are required to avoid train/test distribution shift.
7. **`Oracle-K` vs `Est-K`**: Cequel and LSED originally assume known `K*`; unsupervised `Est-K` uses `HISEvent Official`'s predicted cluster count (`n_pred_clusters`) alongside `Oracle-K`.
