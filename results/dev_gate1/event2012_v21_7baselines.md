# Track A v2.1 Fairness Benchmark (7 Baselines, `M_02..M_05`, `Qwen/Qwen3-8B`)

## Event2012 Dev (`M_02..M_05`)

| Group | Method | Calls | Tok | Show% | Err | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet | CPU(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-LLM Ref | Pure 2D-SE Base (0.50, 0.5) | 0 | 0 | 0.00% | 0 | 88.88 | 78.17 | 82.18 | 60.69 | +0.0 | 2.38 |
| 0-LLM Ref | AutoMerge-Only (0-Tok) | 0 | 0 | 0.00% | 0 | 91.19 | 79.28 | 87.18 | 60.60 | +0.0 | 0.09 |
| 0-LLM Paper 1 | SBERT-Agglom (EMNLP'19) | 0 | 0 | 0.00% | 0 | 90.51 | 77.65 | 87.74 | 64.53 | +0.0 | 2.92 |
| 0-LLM Paper 2a | HiSEvent Default (AAAI'24) | 0 | 0 | 0.00% | 0 | 88.53 | 69.17 | 82.10 | 30.25 | +0.0 | 3.50 |
| 0-LLM Paper 2b | HiSEvent Dev-Tuned (AAAI'24) | 0 | 0 | 0.00% | 0 | 88.88 | 78.17 | 82.18 | 60.69 | +0.0 | 0.61 |
| 0-LLM Paper 3 | HyperSED (AAAI'25) | 0 | 0 | 0.00% | 0 | 90.13 | 73.38 | 85.17 | 44.67 | +0.0 | 3.64 |
| LLM Paper 4 | ClusterLLM-NoFT (EMNLP'23) | 141 | 39076 | 1.94% | 0 | 86.28 | 74.68 | 81.33 | 67.80 | -726.8 | 4.12 |
| LLM Paper 5 | Few-Shot LLM-Post (TACL'24) | 75 | 38877 | 3.39% | 0 | 88.85 | 78.24 | 82.15 | 60.69 | +78.5 | 0.98 |
| LLM Paper 6 | LSED (ACL'25) | 313 | 39609 | 4.37% | 0 | 88.90 | 77.43 | 82.36 | 57.99 | -16.2 | 11.77 |
| LLM Paper 7 | RagSEDE (WWW'26) | 67 | 39063 | 5.23% | 0 | 88.85 | 78.17 | 82.12 | 60.69 | -35.0 | 1.55 |
| **OUR METHOD** | **Pure-Star-Meso v2.1** | 64 | 33104 | 3.49% | 0 | 93.13 | 81.00 | 90.16 | 63.73 | +634.8 | 1.41 |
