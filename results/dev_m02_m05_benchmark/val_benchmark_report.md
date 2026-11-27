# Event2012 Benchmark (`M_02..M_05`, `Qwen/Qwen3-8B`, `all-MiniLM-L6-v2`)

> Metrics strictly reported as **ARI / NMI / AMI (%)** per block (`M_02..M_05`) and 4-block average (`Mean`).

| Group | Method | Setting | M_02 (ARI/NMI/AMI) | M_03 (ARI/NMI/AMI) | M_04 (ARI/NMI/AMI) | M_05 (ARI/NMI/AMI) | **Mean (ARI / NMI / AMI)** | Calls / Tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0-LLM Original | SBERT* (KMeans, Oracle-K) | 0 Tokens | 73.36 / 86.86 / 85.68 | 65.71 / 87.57 / 86.50 | 38.32 / 81.26 / 79.29 | 72.82 / 87.42 / 86.52 | **62.55 / 85.78 / 84.50** | 0 / 0 |
| 0-LLM Original | HISEvent Official (AAAI'24) | 0 Tokens | 77.43 / 81.49 / 80.74 | 81.57 / 92.20 / 91.70 | 49.15 / 85.00 / 83.79 | 61.96 / 86.04 / 85.07 | **67.53 / 86.18 / 85.33** | 0 / 0 |
| Unbounded Original | RagSEDE Official (WWW'26) | Unbounded (Cost Reported) | 92.85 / 95.30 / 94.84 | 88.65 / 93.52 / 93.00 | 89.62 / 90.50 / 89.53 | 87.67 / 92.15 / 91.54 | **89.70 / 92.87 / 92.23** | 639 / 251,203 |
| Unbounded Original | LSED Official HGCN (ACL'25, Oracle-K, w=infty) | Unbounded (All N) | *Pending Run* | - | - | - | *Pending* | - |
| Unbounded Original | LSED Official HGCN (ACL'25, Est-K, w=infty) | Unbounded (All N) | *Pending Run* | - | - | - | *Pending* | - |
| Budgeted LLM (20k) | COBRAS Official (IDA'18, Binary LLM) | 20k Budget/Blk | 86.92 / 90.00 / 89.47 | 84.10 / 87.30 / 86.69 | 41.88 / 82.76 / 80.67 | 79.42 / 83.85 / 83.43 | **73.08 / 85.98 / 85.06** | 752 / 79,845 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Oracle-K) | 20k Budget/Blk | 61.73 / 81.79 / 80.14 | 59.96 / 86.33 / 85.16 | 42.32 / 82.06 / 80.16 | 64.44 / 84.33 / 83.21 | **57.11 / 83.63 / 82.17** | 79 / 79,592 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Est-K) | 20k Budget/Blk | 81.13 / 86.10 / 85.54 | 81.50 / 90.42 / 89.78 | 50.04 / 83.15 / 81.76 | 64.90 / 84.74 / 83.67 | **69.39 / 86.10 / 85.19** | 79 / 79,592 |
| Budgeted LLM (20k) | Cequel WCKMeans (CIKM'25, Oracle-K) | 20k Budget/Blk | 68.17 / 84.86 / 83.60 | 59.75 / 84.64 / 83.36 | 33.29 / 79.13 / 76.91 | 44.79 / 79.22 / 77.75 | **51.50 / 81.96 / 80.41** | 79 / 79,592 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | 20k Budget/Blk | 92.87 / 94.94 / 94.41 | 95.99 / 95.79 / 95.40 | 58.68 / 89.47 / 88.30 | 86.41 / 91.98 / 91.28 | **83.49 / 93.05 / 92.35** | 58 / 20,871 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | Unbounded | 92.87 / 94.94 / 94.41 | 95.99 / 95.79 / 95.40 | 58.68 / 89.47 / 88.30 | 86.41 / 91.98 / 91.28 | **83.49 / 93.05 / 92.35** | 58 / 20,871 |

## Per-Metric Expanded Summary (Mean ARI / NMI / AMI)

| Group | Method | Setting | ARI | NMI | AMI | Total Calls | Total Tokens | Parse Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0-LLM Original | SBERT* (KMeans, Oracle-K) | 0 Tokens | 62.55 | 85.78 | 84.50 | 0 | 0 | 0 |
| 0-LLM Original | HISEvent Official (AAAI'24) | 0 Tokens | 67.53 | 86.18 | 85.33 | 0 | 0 | 0 |
| Unbounded Original | RagSEDE Official (WWW'26) | Unbounded (Cost Reported) | 89.70 | 92.87 | 92.23 | 639 | 251,203 | 41 |
| Budgeted LLM (20k) | COBRAS Official (IDA'18, Binary LLM) | 20k Budget/Blk | 73.08 | 85.98 | 85.06 | 752 | 79,845 | 0 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Oracle-K) | 20k Budget/Blk | 57.11 | 83.63 | 82.17 | 79 | 79,592 | 0 |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Est-K) | 20k Budget/Blk | 69.39 | 86.10 | 85.19 | 79 | 79,592 | 0 |
| Budgeted LLM (20k) | Cequel WCKMeans (CIKM'25, Oracle-K) | 20k Budget/Blk | 51.50 | 81.96 | 80.41 | 79 | 79,592 | 0 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | 20k Budget/Blk | 83.49 | 93.05 | 92.35 | 58 | 20,871 | 0 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | Unbounded | 83.49 | 93.05 | 92.35 | 58 | 20,871 | 0 |

## Protocol Alignment & Documented Unavoidable Exceptions

1. **Shared LLM Backbone (`Qwen/Qwen3-8B`)**: RagSEDE (Qwen2/2.5/Llama3), Cequel (GPT-3.5-Turbo/Llama3-8B), and LSED (Llama3.1-8B/Qwen2.5-7B/Gemma2-9B) originally used different LLMs; all LLM baselines use unified `Qwen/Qwen3-8B` (deterministic `temperature=0.0` for `FRAG-SED`/`RagSEDE`/`COBRAS`/`LSED`, and official `temp=1.0, n=5` majority voting for `Cequel`, with `enable_thinking=False`) for fair comparison.
2. **3-Category Benchmark Design**: Keeps 0-LLM original (`SBERT*`, `HISEvent Official` with language-specific preprocessing & `eps=0.0`), unbounded original LLM-SED with exact token-cost accounting (`RagSEDE Official`, `LSED Official w=infty`), budgeted active-query LLM baselines at `20k tokens/block` (`COBRAS Official`, `Cequel Official`), and `FRAG-SED` (`20k Budget/Blk` & `Unbounded`).
3. **RagSEDE Infrastructure**: Uses official `max_retries=10` and `use_json_mode=False` (non-forced JSON output parsed from ```json blocks or raw JSON); external RAGFlow v0.17.2 server (`BAAI/bge-small-zh-v1.5`) is replaced by a functionally equivalent in-memory hybrid index (`all-MiniLM-L6-v2` dense cosine + IDF-weighted keyword similarity excluding prompt boilerplate, retrieved by `Enc('INPUT: ' + a_k[:200])`) with `temp=0.2` retry jitter before falling back to `'Confusion'`.
4. **COBRAS LLM Oracle Adaptation**: Original IDA'18 COBRAS assumes a noise-free ground-truth oracle; under the budgeted setting, queries use a concise binary `Yes/No` `Qwen/Qwen3-8B` oracle (`max_tokens=8`) capped cleanly by the `20k tokens/block` causal budget.
5. **Cequel Solver & Sampling**: Uses official 5-sample majority voting (`n=5, temp=1.0, max_tokens=1, max_vote_retries=5`) via `call_text_n`, official `WCSC` (`Q` in `float64` per paper Eq. 4, sparse `eigsh` for `n > 2500`, primary on tweets, `[0.5, 1.5]` Q weights) and `WCKMeans` (`init='random'`, `ML in [0.01, 0.1]`, `CL in [0, 0.01]`) under the `20k tokens/block` causal budget; Event2012 uncapped-tweet prompts target real-world events rather than general tweet topics.
6. **LSED Open-Set Window `w=infty`**: Paper Fig. 6 open-set setting (`w=infty`: train 3-layer HGCN Poincare MLP once on LLM-summarized `M_00`, cluster LLM-summarized `M_01..M_05` via KMeans, numeric `tan_reg=0.02` preventing 100% Poincare boundary saturation) replaces supervised `w=1` updates; both `M_00` (`lsed_M00_prompts.jsonl`) and `M_01..M_05` (`lsed_M01_M05_prompts.jsonl`) summaries are required to avoid train/test distribution shift.
7. **`Oracle-K` vs `Est-K`**: Cequel and LSED originally assume known `K*`; unsupervised `Est-K` uses `HISEvent Official`'s predicted cluster count (`n_pred_clusters`) alongside `Oracle-K`.
