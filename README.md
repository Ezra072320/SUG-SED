# FRAG-SED: Failure-Region Active Graph Refinement for Social Event Detection

Open-set streaming Social Event Detection (SED) framework combining **2D Structural Entropy (2D-SE)** graph partitioning with **budget-bounded active LLM query selection**, plus strictly aligned implementations of 6 reference SED/active-clustering baselines.

## Active 3-Category Benchmark Design

All baselines and **FRAG-SED** are evaluated on `Event2012 (M_01..M_05)` using strict **ARI / NMI / AMI (%)** metrics, shared `all-MiniLM-L6-v2` embeddings (`max_length=256`, `float32` for official baseline pipelines), and unified `Qwen/Qwen3-8B` (`temperature=0.0`, `enable_thinking=False`) for LLM baselines:

| Group | Method | Setting | Token Budget |
|---|---|---|---|
| **1. 0-LLM Original** | `SBERT*` (EMNLP'19) | Preprocessed text + `KMeans(Oracle-K)` | `0` |
| **1. 0-LLM Original** | `HISEvent Official` (AAAI'24) | Exact preprocessed text + raw unified attributes + 1D/2D-SE | `0` |
| **2. Unbounded Original** | `RagSEDE Official` (WWW'26) | Agglom anchors + MMR top-3 + in-memory hybrid Event KB | Unbounded (Cost Reported) |
| **2. Unbounded Original** | `LSED Official` (ACL'25, `w=infty`) | All-`N` LLM tweet rewrites + OLE 2D TIME + 3-layer HGCN Poincare MLP (`Oracle-K` / `HISEvent Est-K`) | Unbounded (Cost Reported) |
| **3. Budgeted LLM** | `COBRAS Official` (IDA'18) | Top-down 2-means splitting + bottom-up merging + binary LLM oracle | `20k Tokens / Block` |
| **3. Budgeted LLM** | `Cequel Official` (CIKM'25) | Triangle Explore-Consolidate + uncapped 5-option prompt + `WCSC` (primary) / `WCKMeans` (`Oracle-K` / `HISEvent Est-K`) | `20k Tokens / Block` |
| **4. Ours** | `FRAG-SED` | Selective Pairwise Ambiguous Nodes -> Multi-Cluster Star Anchor (Meso) -> Boundary Merge | `20k Budget/Blk` & `Unbounded` |

## Repository Structure

```text
.
├── FRAG_SED_Paper_Design.md                 # Complete FRAG-SED paper design & experiment plan
├── README.md                                # Repository guide and locked 3-category setup
├── core_shared/                             # Shared data loader, graph builder, 1D/2D-SE solver, token ledger
├── trackA_sug_sed/
│   ├── frag_sed.py                          # FRAG-SED 3-step algorithm (pairwise -> star meso -> boundary merge)
│   ├── star_meso.py                         # Multi-cluster star anchor helper
│   ├── baselines_official.py                # Strictly aligned implementations of the 6 baseline papers
│   ├── baselines_sed.py                     # Structural-entropy baseline utilities (HISEvent helpers)
│   ├── run_val_benchmark.py                 # Main M_01..M_05 benchmark & Colab LSED prompt exporter
│   ├── colab_batch_qwen3_8b.ipynb           # Colab GPU batch runner for unbounded LSED M_01..M_05 (+ M_00)
│   ├── embed_cache.py                       # Per-block embedding cache with max_length & dtype isolation
│   ├── llm_client.py                        # Causal-budgeted OpenAI-compatible client with JSONL caching
│   ├── anonymizer.py                        # Entity/privacy anonymization module (planned §8.1)
│   └── cross_day.py                         # Cross-day temporal alignment module (planned §8.2)
├── tests/
│   ├── test_frag_sed.py                     # Unit tests for FRAG-SED pipeline & causal budget enforcement
│   └── test_baselines_official.py           # Strict alignment unit tests for all 6 official baselines
├── cache/embeddings/                        # Cached per-block SBERT embeddings
├── results/val_m01_m05_benchmark/           # Verified JSON/Markdown benchmark reports & ledgers
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
   python trackA_sug_sed/run_val_benchmark.py --dry-run
   ```
3. **Run 0-LLM baselines (`SBERT*`, `HISEvent Official`)**:
   ```powershell
   python trackA_sug_sed/run_val_benchmark.py --methods sbert_kmeans,hisevent_official
   ```
4. **Run budgeted LLM baselines + FRAG-SED (requires API key config via `core_shared/key_loader.py`)**:
   ```powershell
   python trackA_sug_sed/run_val_benchmark.py --methods cobras_20k,cequel_20k,frag_sed_20k,frag_sed_unbounded
   ```
5. **Run unbounded original baselines (LSED via Colab outputs + RagSEDE online with cost reporting)**:
   ```powershell
   python trackA_sug_sed/run_val_benchmark.py --methods ragsede_unbounded,lsed_unbounded --use-colab-outputs <dir_with_lsed_responses>
   ```
