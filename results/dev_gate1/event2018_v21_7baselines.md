# Track A v2.1 Fairness Benchmark (7 Baselines, `M_02..M_05`, `Qwen/Qwen3-8B`)

## Event2018 Dev (`M_02..M_05`)

| Group | Method | Calls | Tok | Show% | Err | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet | CPU(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-LLM Ref | Pure 2D-SE Base (0.50, 0.5) | 0 | 0 | 0.00% | 0 | 68.68 | 71.24 | 59.69 | 54.26 | +0.0 | 19.70 |
| 0-LLM Ref | AutoMerge-Only (0-Tok) | 0 | 0 | 0.00% | 0 | 69.77 | 72.11 | 62.00 | 54.15 | +0.0 | 0.21 |
| 0-LLM Paper 1 | SBERT-Agglom (EMNLP'19) | 0 | 0 | 0.00% | 0 | 67.52 | 71.24 | 68.56 | 67.31 | +0.0 | 8.41 |
| 0-LLM Paper 2a | HiSEvent Default (AAAI'24) | 0 | 0 | 0.00% | 0 | 68.06 | 64.02 | 59.12 | 27.23 | +0.0 | 16.46 |
| 0-LLM Paper 2b | HiSEvent Dev-Tuned (AAAI'24) | 0 | 0 | 0.00% | 0 | 67.36 | 76.12 | 57.80 | 77.99 | +0.0 | 13.38 |
| 0-LLM Paper 3 | HyperSED (AAAI'25) | 0 | 0 | 0.00% | 0 | 66.83 | 65.69 | 57.79 | 42.29 | +0.0 | 7.92 |
| LLM Paper 4 | ClusterLLM-NoFT (EMNLP'23) | 122 | 39280 | 0.87% | 0 | 60.35 | 72.81 | 53.51 | 82.04 | -1517.5 | 12.45 |
| LLM Paper 5 | Few-Shot LLM-Post (TACL'24) | 57 | 39300 | 1.79% | 0 | 68.84 | 71.32 | 59.78 | 54.27 | +540.0 | 1.25 |
| LLM Paper 6 | LSED (ACL'25) | 265 | 39412 | 2.27% | 0 | 68.37 | 70.77 | 58.97 | 53.61 | +264.8 | 28.86 |
| LLM Paper 7 | RagSEDE (WWW'26) | 49 | 38341 | 2.52% | 0 | 68.72 | 71.24 | 59.70 | 54.25 | +83.0 | 1.69 |
| **OUR METHOD** | **Pure-Star-Meso v2.1** | 48 | 38318 | 1.94% | 0 | 78.14 | 78.37 | 78.64 | 66.92 | +1426.0 | 2.41 |
