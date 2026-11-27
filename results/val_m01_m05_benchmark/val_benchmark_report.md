# Event2012 Validation Benchmark (`M_01..M_05`, `Qwen/Qwen3-8B`, `all-MiniLM-L6-v2`)

> Metrics strictly reported as **ARI / NMI / AMI (%)** per block (`M_01..M_05`) and 5-block average (`Mean`).

| Group | Method | Setting | M_01 (ARI/NMI/AMI) | M_02 (ARI/NMI/AMI) | M_03 (ARI/NMI/AMI) | M_04 (ARI/NMI/AMI) | M_05 (ARI/NMI/AMI) | **Mean (ARI / NMI / AMI)** | Calls / Tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0-LLM Original | SBERT* (KMeans, Oracle-K) | 0 Tokens | 3.16 / 39.86 / 38.01 | 73.36 / 86.86 / 85.68 | 65.71 / 87.57 / 86.50 | 38.32 / 81.26 / 79.29 | 72.82 / 87.42 / 86.52 | **50.67 / 76.59 / 75.20** | 0 / 0 |
| 0-LLM Original | HISEvent Official (AAAI'24) | 0 Tokens | 9.31 / 46.23 / 44.96 | 77.43 / 81.49 / 80.74 | 81.57 / 92.20 / 91.70 | 49.15 / 85.00 / 83.79 | 61.96 / 86.04 / 85.07 | **55.88 / 78.19 / 77.25** | 0 / 0 |
| Unbounded Original | RagSEDE Official (WWW'26) | Unbounded (Cost Reported) | *Pending Run* | - | - | - | - | *Pending* | - |
| Unbounded Original | LSED Official HGCN (ACL'25, Oracle-K, w=infty) | Unbounded (All N) | *Pending Run* | - | - | - | - | *Pending* | - |
| Unbounded Original | LSED Official HGCN (ACL'25, Est-K, w=infty) | Unbounded (All N) | *Pending Run* | - | - | - | - | *Pending* | - |
| Budgeted LLM (20k) | COBRAS Official (IDA'18, Binary LLM) | 20k Budget/Blk | *Pending Run* | - | - | - | - | *Pending* | - |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Oracle-K) | 20k Budget/Blk | *Pending Run* | - | - | - | - | *Pending* | - |
| Budgeted LLM (20k) | Cequel Official WCSC (CIKM'25, Est-K) | 20k Budget/Blk | *Pending Run* | - | - | - | - | *Pending* | - |
| Budgeted LLM (20k) | Cequel WCKMeans (CIKM'25, Oracle-K) | 20k Budget/Blk | *Pending Run* | - | - | - | - | *Pending* | - |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | 20k Budget/Blk | 4.09 / 42.69 / 40.16 | 92.87 / 94.94 / 94.41 | 95.99 / 95.79 / 95.40 | 79.70 / 92.20 / 91.34 | 86.41 / 91.98 / 91.28 | **71.81 / 83.52 / 82.52** | 102 / 40,712 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | Unbounded | 6.71 / 45.05 / 42.74 | 92.87 / 94.94 / 94.41 | 95.99 / 95.79 / 95.40 | 79.70 / 92.20 / 91.34 | 86.41 / 91.98 / 91.28 | **72.34 / 83.99 / 83.03** | 142 / 57,101 |

## Per-Metric Expanded Summary (Mean ARI / NMI / AMI)

| Group | Method | Setting | ARI | NMI | AMI | Total Calls | Total Tokens | Parse Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0-LLM Original | SBERT* (KMeans, Oracle-K) | 0 Tokens | 50.67 | 76.59 | 75.20 | 0 | 0 | 0 |
| 0-LLM Original | HISEvent Official (AAAI'24) | 0 Tokens | 55.88 | 78.19 | 77.25 | 0 | 0 | 0 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | 20k Budget/Blk | 71.81 | 83.52 | 82.52 | 102 | 40,712 | 0 |
| Ours (FRAG-SED) | **FRAG-SED (Ours)** | Unbounded | 72.34 | 83.99 | 83.03 | 142 | 57,101 | 0 |

## Protocol Alignment & Documented Unavoidable Exceptions

1. **Shared LLM Backbone (`Qwen/Qwen3-8B`)**: RagSEDE (Qwen2/2.5/Llama3), Cequel (GPT-3.5-Turbo/Llama3-8B), and LSED (Llama3.1-8B/Qwen2.5-7B/Gemma2-9B) originally used different LLMs; all LLM baselines use unified `Qwen/Qwen3-8B` (deterministic `temperature=0.0` for `FRAG-SED`/`RagSEDE`/`COBRAS`/`LSED`, and official `temp=1.0, n=5` majority voting for `Cequel`, with `enable_thinking=False`) for fair comparison.
2. **3-Category Benchmark Design**: Keeps 0-LLM original (`SBERT*`, `HISEvent Official` with language-specific preprocessing & `eps=0.0`), unbounded original LLM-SED with exact token-cost accounting (`RagSEDE Official`, `LSED Official w=infty`), budgeted active-query LLM baselines at `20k tokens/block` (`COBRAS Official`, `Cequel Official`), and `FRAG-SED` (`20k Budget/Blk` & `Unbounded`).
3. **RagSEDE Infrastructure**: Uses official `max_retries=10` and `use_json_mode=False` (non-forced JSON output parsed from ```json blocks or raw JSON); external RAGFlow v0.17.2 server (`BAAI/bge-small-zh-v1.5`) is replaced by a functionally equivalent in-memory hybrid index (`all-MiniLM-L6-v2` dense cosine + IDF-weighted keyword similarity excluding prompt boilerplate, retrieved by `Enc('INPUT: ' + a_k[:200])`) with `temp=0.2` retry jitter before falling back to `'Confusion'`.
4. **COBRAS LLM Oracle Adaptation**: Original IDA'18 COBRAS assumes a noise-free ground-truth oracle; under the budgeted setting, queries use a concise binary `Yes/No` `Qwen/Qwen3-8B` oracle (`max_tokens=8`) capped cleanly by the `20k tokens/block` causal budget.
5. **Cequel Solver & Sampling**: Uses official 5-sample majority voting (`n=5, temp=1.0, max_tokens=1, max_vote_retries=5`) via `call_text_n`, official `WCSC` (`Q` in `float64` per paper Eq. 4, sparse `eigsh` for `n > 2500`, primary on tweets, `[0.5, 1.5]` Q weights) and `WCKMeans` (`init='random'`, `ML in [0.01, 0.1]`, `CL in [0, 0.01]`) under the `20k tokens/block` causal budget; Event2012 uncapped-tweet prompts target real-world events rather than general tweet topics.
6. **LSED Open-Set Window `w=infty`**: Paper Fig. 6 open-set setting (`w=infty`: train 3-layer HGCN Poincare MLP once on LLM-summarized `M_00`, cluster LLM-summarized `M_01..M_05` via KMeans, numeric `tan_reg=0.02` preventing 100% Poincare boundary saturation) replaces supervised `w=1` updates; both `M_00` (`lsed_M00_prompts.jsonl`) and `M_01..M_05` (`lsed_M01_M05_prompts.jsonl`) summaries are required to avoid train/test distribution shift.
7. **`Oracle-K` vs `Est-K`**: Cequel and LSED originally assume known `K*`; unsupervised `Est-K` uses `HISEvent Official`'s predicted cluster count (`n_pred_clusters`) alongside `Oracle-K`.
